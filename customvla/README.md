# CustomVLA — Modular Vision-Language-Action Package

A self-contained, arm-agnostic VLA package for robotic manipulation.
Inspired by GR-1, Groot-1.5, and π0. Drop-in support for new arms via a decorator.

---

## Architecture

```
Camera(s) ──→ VisionEncoder (DINOv2-S/14 frozen | ResNet-18) ──→ [B, d_vis]
Task text ──→ LanguageEncoder (BiGRU + masked mean-pool)      ──→ [B, d_lang]
Joint state → StateEncoder (3-layer MLP)                       ──→ [B, d_state]
                             │
                       FusionMoE (top-k experts, load-balance loss)
                             │
                       ActionHead (MLP) ──→ pred_chunk [B, C, ACTION_DIM]
                             │
                   RLSafetyValidator
                   ├── HardLimitChecker (rule-based hinge, 0 params)
                   └── PPOCritic        (learned V(s,a), differentiable penalty)
```

---

## Quick Start

```bash
pip install -e .
```

### 1. List available arms

```bash
python scripts/train_vla.py --list_arms
```

### 2. Prepare data + train (SO100)

```bash
python scripts/train_vla.py --arm so100 --prepare --train
```

### 3. Run inference

```bash
# Single test (dummy inputs, no robot needed)
python scripts/run_inference.py \
    --checkpoint runs/vla_so100/checkpoint_best.pt \
    --arm so100 --mode single

# Webcam + mock robot loop
python scripts/run_inference.py \
    --checkpoint runs/vla_so100/checkpoint_best.pt \
    --arm so100 --mode webcam --steps 200 --hz 10

# Benchmark latency
python scripts/run_inference.py \
    --checkpoint runs/vla_so100/checkpoint_best.pt \
    --arm so100 --mode benchmark --n_iters 200
```

---

## Defining a New Arm

### Interactive (recommended)

```bash
python scripts/define_arm.py
```

Follow the prompts. A ready-to-use `.py` file is generated in `customvla/arms/custom/`.

### Non-interactive

```bash
python scripts/define_arm.py \
    --name myrobot \
    --dof 7 \
    --joint_names "j1,j2,j3,j4,j5,j6,gripper" \
    --joint_limits "-3.14,3.14;-3.14,3.14;-3.14,3.14;-3.14,3.14;-3.14,3.14;-3.14,3.14;0,1" \
    --cameras "top,wrist" \
    --task "Pick and place the object on the table"
```

### Manual (copy this template)

```python
# customvla/arms/custom/myrobot.py
import numpy as np
from customvla.arms.base import BaseArm
from customvla.arms.registry import register_arm

@register_arm("myrobot")
class MyRobotArm(BaseArm):
    ACTION_DIM  = 7
    STATE_DIM   = 7
    JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]
    JOINT_LIMITS = [
        (-3.14, 3.14),  # j1
        (-3.14, 3.14),  # j2
        (-3.14, 3.14),  # j3
        (-3.14, 3.14),  # j4
        (-3.14, 3.14),  # j5
        (-3.14, 3.14),  # j6
        (0.0,   1.0),   # gripper
    ]
    CAMERAS   = ["top", "wrist"]
    TASK_TEXT = "Pick and place the object."

    def __init__(self, robot_ip: str = "192.168.1.100"):
        self._state = np.zeros(self.STATE_DIM, dtype=np.float32)

    def send_action(self, action_vector: np.ndarray) -> None:
        # TODO: your hardware call here
        self._state = self.clip_to_limits(action_vector)

    def get_state(self) -> np.ndarray:
        # TODO: your hardware state read here
        return self._state.copy()
```

Then train:

```bash
# Register your arm by importing it in the training script, then:
python scripts/train_vla.py --arm myrobot --train \
    --repo_id YourOrg/myrobot_demos_v1 \
    --data_dir ./data/myrobot
```

---

## Python API

```python
from customvla import VLAInference
from customvla.arms import get_arm
import numpy as np

# Load arm + model
arm    = get_arm("so100")
runner = VLAInference.from_checkpoint(
    "runs/vla_so100/checkpoint_best.pt",
    arm=arm,
    use_rl_validator=True,
)

# Single-step inference
images = {"top": top_frame_np, "wrist": wrist_frame_np}   # uint8 HxWx3
state  = arm.get_state()                                    # [6] float

action = runner.predict(images, state, task="pick up the cup")
arm.send_action(action)

# Chunk-buffered inference (re-plans every C steps)
action = runner.predict_buffered(images, state)

# Closed-loop loop
runner.run_loop(arm, task="pick up the cup", steps=300, hz=10)
```

---

## RL Safety Validator

The `RLSafetyValidator` has two components:

| Component | Type | Purpose |
|-----------|------|---------|
| `HardLimitChecker` | Rule-based | Differentiable hinge penalty on joint limits |
| `PPOCritic` | Learned V(s,a) | Soft penalty for low-quality trajectories |

During **training**, the validator contributes a penalty to the VLA loss:
```
total_loss = w_action * L1_action + w_moe * lb_loss + w_rl * rl_penalty
```

At **inference**, it hard-clips every action to `JOINT_LIMITS` before `send_action()`.

To fine-tune the critic with real rollout data:
```python
from customvla.rl import RLSafetyValidator
validator = RLSafetyValidator(arm)
# collect (state, action, return) tuples from real robot episodes
ppo_loss = validator.ppo_loss(states, actions, returns, old_values)
```

---

## File Structure

```
customvla/
├── customvla/
│   ├── __init__.py
│   ├── inference.py            ← VLAInference: load + predict + run_loop
│   ├── arms/
│   │   ├── base.py             ← BaseArm abstract class
│   │   ├── registry.py         ← @register_arm decorator + get_arm()
│   │   └── builtin.py          ← SO100, Franka, UR5, Generic7DOF
│   ├── models/
│   │   ├── encoders.py         ← VisionEncoder, LanguageEncoder, StateEncoder
│   │   ├── fusion.py           ← FusionMoE (top-k experts)
│   │   └── vla.py              ← VLAModel + TrainConfig + compute_loss
│   └── rl/
│       └── safety.py           ← RLSafetyValidator, HardLimitChecker, PPOCritic
├── scripts/
│   ├── train_vla.py            ← Main training script
│   ├── run_inference.py        ← Inference CLI
│   └── define_arm.py           ← Interactive arm scaffolding tool
├── configs/
│   └── so100.yaml              ← Default SO100 training config
├── setup.py
└── README.md
```

---

## Built-in Arms

| Name | DOF | Cameras | Interface |
|------|-----|---------|-----------|
| `so100` | 6 | top, wrist | ROS / serial |
| `franka` | 7 | wrist, external | libfranka / franka_ros |
| `ur5` | 6 | overhead, wrist | ur_rtde / ROS |
| `generic7dof` | 7 | camera_0 | template (no-op) |

---

## Training Config Reference

All fields can be overridden via CLI arguments:

| Field | Default | Description |
|-------|---------|-------------|
| `vision_backbone` | `dinov2` | `"dinov2"` or `"resnet18"` |
| `n_experts` | 4 | Number of MoE experts |
| `top_k` | 2 | Active experts per forward pass |
| `chunk_size` | 16 | Predicted action horizon (frames) |
| `w_action` | 1.0 | Weight for L1 action loss |
| `w_moe` | 0.01 | Weight for MoE load-balance loss |
| `w_rl` | 0.1 | Weight for RL safety penalty (0 = off) |
| `lr` | 1e-4 | AdamW learning rate |
| `epochs` | 100 | Training epochs |

---

## Requirements

```
torch >= 2.0
torchvision >= 0.15
numpy, pillow, tqdm, pandas, pyarrow
av              # video decoding
huggingface_hub # dataset download
ultralytics     # YOLOv8 (for object tokenizer)
opencv-python   # webcam / inference
```
