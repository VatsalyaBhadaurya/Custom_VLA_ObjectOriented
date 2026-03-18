"""
scripts/run_inference.py

Command-line inference runner. Supports:
  - Webcam / live robot loop
  - Single image + state test
  - Inference benchmark

Usage
─────
# Run closed-loop inference (webcam + mock robot state):
    python scripts/run_inference.py \\
        --checkpoint runs/vla_so100/checkpoint_best.pt \\
        --arm so100 \\
        --task "pick up the cup" \\
        --steps 200 \\
        --hz 10

# Single inference test (no robot):
    python scripts/run_inference.py \\
        --checkpoint runs/vla_so100/checkpoint_best.pt \\
        --arm so100 \\
        --mode single

# Benchmark inference speed:
    python scripts/run_inference.py \\
        --checkpoint runs/vla_so100/checkpoint_best.pt \\
        --arm so100 \\
        --mode benchmark \\
        --n_iters 200
"""

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from customvla.arms.registry import get_arm
from customvla.arms import builtin  # register built-ins
from customvla.inference import VLAInference


def get_webcam_frames(cameras: list, cap_map: dict) -> dict:
    """Grab latest frames from OpenCV VideoCapture objects."""
    import cv2
    frames = {}
    for cam_name, cap in cap_map.items():
        ret, frame = cap.read()
        if ret:
            frames[cam_name] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frames[cam_name] = np.zeros((224, 224, 3), dtype=np.uint8)
    return frames


def run_webcam_loop(runner: VLAInference, arm, args):
    """Open webcam(s) and run closed-loop inference."""
    try:
        import cv2
    except ImportError:
        print("  OpenCV not installed. pip install opencv-python")
        sys.exit(1)

    cameras = runner.cfg.cameras
    print(f"  Opening cameras: {cameras}")

    cap_map = {}
    for i, cam_name in enumerate(cameras):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            print(f"  WARNING: Cannot open camera index {i} for '{cam_name}'. "
                  f"Using blank frames.")
        cap_map[cam_name] = cap

    def camera_fn():
        return get_webcam_frames(cameras, cap_map)

    try:
        runner.run_loop(
            arm   = arm,
            task  = args.task,
            steps = args.steps,
            hz    = args.hz,
            camera_fn = camera_fn,
            verbose   = True,
        )
    finally:
        for cap in cap_map.values():
            cap.release()


def run_single_test(runner: VLAInference, arm):
    """Single-step inference with dummy inputs."""
    print("\n  Single-step inference test (dummy inputs)")
    images = {
        cam: np.zeros((224, 224, 3), dtype=np.uint8)
        for cam in runner.cfg.cameras
    }
    state = np.zeros(runner.cfg.state_dim, dtype=np.float32)
    action, chunk = runner.predict(images, state, return_chunk=True)
    print(f"\n  Predicted action (step 0): {action}")
    print(f"  Full chunk shape: {chunk.shape}")
    print(f"  Chunk mean abs: {np.abs(chunk).mean():.4f}")
    print()

    # Validate against arm limits
    if arm is not None:
        result = arm.validate_action(action)
        status = "✓ SAFE" if result["valid"] else f"⚠ VIOLATIONS: {result['violations']}"
        print(f"  RL Limit check: {status}")
    print()


def main():
    p = argparse.ArgumentParser(description="CustomVLA Inference Runner")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    p.add_argument("--arm",        default="so100", help="Registered arm name")
    p.add_argument("--task",       default=None, help="Task description (overrides arm default)")
    p.add_argument("--mode",       default="webcam",
                   choices=["webcam", "single", "benchmark"],
                   help="Inference mode")
    p.add_argument("--steps",      type=int,   default=200, help="Steps for webcam loop")
    p.add_argument("--hz",         type=float, default=10.0, help="Control frequency")
    p.add_argument("--n_iters",    type=int,   default=100, help="Benchmark iterations")
    p.add_argument("--device",     default=None, choices=["cuda", "cpu"])
    p.add_argument("--no_rl",      action="store_true", help="Disable RL safety validator")
    args = p.parse_args()

    arm    = get_arm(args.arm)
    runner = VLAInference.from_checkpoint(
        checkpoint_path   = args.checkpoint,
        arm               = arm,
        device            = args.device,
        use_rl_validator  = not args.no_rl,
    )

    if args.mode == "single":
        run_single_test(runner, arm)

    elif args.mode == "benchmark":
        runner.benchmark(n_iters=args.n_iters)

    elif args.mode == "webcam":
        run_webcam_loop(runner, arm, args)


if __name__ == "__main__":
    main()
