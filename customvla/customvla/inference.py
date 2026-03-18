"""
customvla/inference.py

VLAInference — High-level inference API for deploying a trained VLA model.

Usage
─────
    from customvla import VLAInference
    from customvla.arms import get_arm

    arm = get_arm("so100")
    runner = VLAInference.from_checkpoint("runs/vla_so100/checkpoint_best.pt", arm)

    # Single-step inference from numpy images + state
    action = runner.predict(
        images={"top": img_top_np, "wrist": img_wrist_np},
        state=joint_state_np,
        task="pick up the cup",
    )

    # Run closed-loop control on a live robot
    runner.run_loop(arm, task="pick up the cup", steps=200)
"""

from __future__ import annotations

import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from customvla.models.vla import VLAModel, TrainConfig
from customvla.models.encoders import SimpleTokenizer
from customvla.arms.base import BaseArm
from customvla.rl.safety import RLSafetyValidator


# Standard ImageNet normalisation (matches DINOv2 / torchvision pretrain)
_IMG_TRANSFORM = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class VLAInference:
    """
    Wraps a trained VLAModel for deployment.

    Handles:
      - Checkpoint loading
      - Image / state pre-processing
      - Action de-normalisation
      - RL safety validation before sending to robot
      - Chunk-based action buffering (execute predicted chunk, re-predict)
    """

    def __init__(
        self,
        model: VLAModel,
        cfg: TrainConfig,
        tokenizer: SimpleTokenizer,
        stats: dict,
        arm: Optional[BaseArm] = None,
        validator: Optional[RLSafetyValidator] = None,
        device: str = "cpu",
    ):
        self.model     = model.to(device).eval()
        self.cfg       = cfg
        self.tokenizer = tokenizer
        self.arm       = arm
        self.validator = validator
        self.device    = device

        # Normalisation arrays
        self._act_mean = np.array(stats["action_mean"], dtype=np.float32)
        self._act_std  = np.array(stats["action_std"],  dtype=np.float32)
        self._st_mean  = np.array(stats["state_mean"],  dtype=np.float32)
        self._st_std   = np.array(stats["state_std"],   dtype=np.float32)

        # Action chunk buffer (populated after each model call)
        self._chunk_buffer: List[np.ndarray] = []
        self._chunk_ptr: int = 0

    # ── Construction ──────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        arm: Optional[BaseArm] = None,
        device: Optional[str] = None,
        use_rl_validator: bool = True,
    ) -> "VLAInference":
        """
        Load a checkpoint saved by the training script and build an inference runner.

        Args:
            checkpoint_path : path to .pt checkpoint file
            arm             : arm instance (overrides arm_name in config if given)
            device          : "cuda" | "cpu" | None (auto)
            use_rl_validator: attach RLSafetyValidator if arm is provided
        """
        ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = TrainConfig(**ck["config"])

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load stats from checkpoint or adjacent stats.json
        stats = ck.get("stats")
        if stats is None:
            stats_path = Path(checkpoint_path).parent.parent / "cache" / "stats.json"
            if stats_path.exists():
                with open(stats_path) as f:
                    stats = json.load(f)
            else:
                raise FileNotFoundError(
                    "stats.json not found. Pass stats dict explicitly or ensure "
                    "cache/stats.json exists relative to checkpoint."
                )

        tokenizer = SimpleTokenizer(texts=[cfg.task_text])
        model = VLAModel(cfg, tokenizer)
        model.load_state_dict(ck["model"])
        model.eval()
        print(f"  Loaded VLAModel from {checkpoint_path}")
        print(f"  {model.summary()}")

        validator = None
        if use_rl_validator and arm is not None:
            validator = RLSafetyValidator(arm)
            print(f"  {validator.describe()}")

        return cls(
            model=model,
            cfg=cfg,
            tokenizer=tokenizer,
            stats=stats,
            arm=arm,
            validator=validator,
            device=device,
        )

    # ── Core predict ──────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        images: Dict[str, Union[np.ndarray, Image.Image]],
        state: np.ndarray,
        task: Optional[str] = None,
        return_chunk: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Run one forward pass and return the first predicted action.

        Args:
            images       : dict mapping camera name → numpy [H,W,3] uint8 or PIL.Image
            state        : current joint state [state_dim] in real (un-normalised) space
            task         : task description string (uses arm.TASK_TEXT if None)
            return_chunk : if True, also return the full predicted chunk [C, action_dim]

        Returns:
            action       : np.ndarray [action_dim] — first step, in real joint space
            (optional) chunk : np.ndarray [C, action_dim] — full chunk
        """
        task_text = task or self.cfg.task_text
        task_ids  = self.tokenizer.encode(task_text).unsqueeze(0).to(self.device)

        # Pre-process images in camera order
        img_tensors = []
        for cam in self.cfg.cameras:
            if cam not in images:
                raise KeyError(f"Camera '{cam}' missing from images dict. "
                               f"Expected cameras: {self.cfg.cameras}")
            raw = images[cam]
            if isinstance(raw, np.ndarray):
                raw = Image.fromarray(raw.astype(np.uint8))
            img_tensors.append(
                _IMG_TRANSFORM(raw).unsqueeze(0).to(self.device)
            )

        # Normalise state
        state_norm = (state - self._st_mean) / self._st_std
        state_t    = torch.from_numpy(state_norm).float().unsqueeze(0).to(self.device)

        # Forward
        pred_chunk, _ = self.model(
            *img_tensors,
            state=state_t,
            task_ids=task_ids,
        )  # [1, C, action_dim]

        # De-normalise
        chunk_np = pred_chunk.squeeze(0).cpu().numpy()        # [C, action_dim]
        chunk_real = chunk_np * self._act_std + self._act_mean

        # Validate first step
        first_action = chunk_real[0]
        if self.validator is not None:
            first_action, info = self.validator.validate_and_clip(
                first_action, verbose=True
            )
        elif self.arm is not None:
            first_action = self.arm.clip_to_limits(first_action)

        if return_chunk:
            return first_action, chunk_real
        return first_action

    # ── Chunk buffered prediction (more efficient: re-plan every C steps) ─

    @torch.no_grad()
    def predict_buffered(
        self,
        images: Dict[str, Union[np.ndarray, Image.Image]],
        state: np.ndarray,
        task: Optional[str] = None,
    ) -> np.ndarray:
        """
        Return actions from a buffered chunk.
        Re-runs the model only when the buffer is empty.
        This matches the chunk-prediction paradigm from ACT / π0.
        """
        if self._chunk_ptr >= len(self._chunk_buffer):
            _, chunk = self.predict(images, state, task, return_chunk=True)
            # Validate entire chunk
            if self.validator is not None:
                safe_chunk = []
                for step in chunk:
                    clipped, _ = self.validator.validate_and_clip(step)
                    safe_chunk.append(clipped)
                self._chunk_buffer = safe_chunk
            elif self.arm is not None:
                self._chunk_buffer = [self.arm.clip_to_limits(a) for a in chunk]
            else:
                self._chunk_buffer = list(chunk)
            self._chunk_ptr = 0

        action = self._chunk_buffer[self._chunk_ptr]
        self._chunk_ptr += 1
        return action

    def reset_buffer(self):
        """Force re-prediction on the next call to predict_buffered."""
        self._chunk_buffer = []
        self._chunk_ptr = 0

    # ── Closed-loop control ───────────────────────────────────────────────

    def run_loop(
        self,
        arm: Optional[BaseArm] = None,
        task: Optional[str] = None,
        steps: int = 200,
        hz: float = 10.0,
        camera_fn = None,
        verbose: bool = True,
    ):
        """
        Run the model in a closed-loop control loop.

        Args:
            arm       : robot arm (uses self.arm if None)
            task      : task description (uses cfg.task_text if None)
            steps     : number of control steps
            hz        : control frequency in Hz
            camera_fn : callable() → dict of {cam_name: np.ndarray}
                        If None, uses arm.get_state() only (no vision — for testing)
            verbose   : print per-step info
        """
        arm = arm or self.arm
        if arm is None:
            raise ValueError("run_loop requires an arm. Pass arm= or set self.arm.")

        task_text = task or self.cfg.task_text
        dt = 1.0 / hz
        self.reset_buffer()

        print(f"\n  [run_loop] Starting: task='{task_text[:60]}...' "
              f"steps={steps} hz={hz}")

        for step in range(steps):
            t0 = time.time()

            # Get camera images
            if camera_fn is not None:
                images = camera_fn()
            else:
                # No camera_fn: create blank images (for dry-run / testing)
                images = {
                    cam: np.zeros((224, 224, 3), dtype=np.uint8)
                    for cam in self.cfg.cameras
                }

            # Get current state
            state = arm.get_state()

            # Predict next action
            action = self.predict_buffered(images, state, task=task_text)

            # Send to robot
            arm.send_action(action)

            elapsed = time.time() - t0
            if verbose and step % 10 == 0:
                joints_str = ", ".join(f"{v:.3f}" for v in action)
                print(f"  step {step:4d}/{steps} | action=[{joints_str}] "
                      f"| t={elapsed*1000:.1f}ms")

            # Rate-limit
            sleep_t = dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        print(f"  [run_loop] Done after {steps} steps.")

    # ── Utilities ─────────────────────────────────────────────────────────

    def benchmark(
        self,
        n_iters: int = 100,
        warmup: int = 10,
    ) -> dict:
        """Measure inference latency in ms."""
        import time
        dummy_images = {
            cam: np.zeros((224, 224, 3), dtype=np.uint8)
            for cam in self.cfg.cameras
        }
        dummy_state = np.zeros(self.cfg.state_dim, dtype=np.float32)

        for _ in range(warmup):
            self.predict(dummy_images, dummy_state)

        times = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            self.predict(dummy_images, dummy_state)
            times.append((time.perf_counter() - t0) * 1000)

        times_arr = np.array(times)
        result = {
            "mean_ms":   float(np.mean(times_arr)),
            "p50_ms":    float(np.percentile(times_arr, 50)),
            "p95_ms":    float(np.percentile(times_arr, 95)),
            "p99_ms":    float(np.percentile(times_arr, 99)),
            "max_ms":    float(np.max(times_arr)),
            "device":    self.device,
            "n_iters":   n_iters,
        }
        print(f"  Inference benchmark ({self.device}):")
        for k, v in result.items():
            if k.endswith("_ms"):
                print(f"    {k:<12} {v:.2f} ms")
        return result
