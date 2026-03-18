"""
customvla/models/vla.py

The full VLA model — arm-agnostic version.
Instantiate with a TrainConfig and a BaseArm (or arm name string).

Architecture
────────────
  N cameras → VisionEncoder(DINOv2-S/14 frozen | ResNet-18) → [B, d_vis]
  Task text  → LanguageEncoder(BiGRU)                         → [B, d_lang]
  Joint state → StateEncoder(MLP)                             → [B, d_state]
  cat all three → FusionMoE(top-k experts)                    → [B, d_model]
  ActionHead(MLP) → pred_chunk [B, chunk_size, ACTION_DIM]
  RLSafetyValidator (optional, post-process)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional, Union, List
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from customvla.models.encoders import VisionEncoder, LanguageEncoder, StateEncoder, SimpleTokenizer
from customvla.models.fusion import FusionMoE
from customvla.arms.base import BaseArm


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    """Full training configuration — all hyper-parameters in one place."""

    # ── Arm (set automatically from arm object in train script) ──────────
    arm_name:   str = "so100"
    action_dim: int = 6
    state_dim:  int = 6
    cameras:    List[str] = field(default_factory=lambda: ["top", "wrist"])
    task_text:  str = "Perform the task."

    # ── Paths ─────────────────────────────────────────────────────────────
    data_dir:   str = "./data"
    cache_dir:  str = "./cache/frames"
    output_dir: str = "./runs/vla"

    # ── Architecture ──────────────────────────────────────────────────────
    d_vis:           int = 256
    d_lang:          int = 128
    d_state:         int = 128
    d_model:         int = 256
    n_experts:       int = 4
    top_k:           int = 2
    chunk_size:      int = 16
    vision_backbone: str = "dinov2"   # "dinov2" | "resnet18"
    share_proj:      bool = True       # share projector across cameras

    # ── Training ──────────────────────────────────────────────────────────
    epochs:       int   = 100
    batch_size:   int   = 16
    lr:           float = 1e-4
    weight_decay: float = 1e-4
    grad_clip:    float = 1.0
    val_frac:     float = 0.10

    # ── Loss weights ──────────────────────────────────────────────────────
    w_action: float = 1.0
    w_moe:    float = 0.01
    w_rl:     float = 0.1    # weight for RL safety penalty (0 = disabled)

    # ── Misc ──────────────────────────────────────────────────────────────
    num_workers: int  = 2
    seed:        int  = 42
    save_every:  int  = 10
    log_every:   int  = 25
    img_size:    int  = 224
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    @classmethod
    def from_arm(cls, arm: BaseArm, **overrides) -> "TrainConfig":
        """Build a config pre-populated from an arm definition."""
        cfg = cls(
            arm_name   = getattr(arm, "_arm_registry_name", arm.__class__.__name__),
            action_dim = arm.ACTION_DIM,
            state_dim  = arm.STATE_DIM,
            cameras    = list(arm.CAMERAS),
            task_text  = arm.TASK_TEXT,
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    def save(self, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TrainConfig":
        with open(path) as f:
            d = json.load(f)
        return cls(**d)


# ─────────────────────────────────────────────────────────────────────────────
# VLA Model
# ─────────────────────────────────────────────────────────────────────────────

class VLAModel(nn.Module):
    """
    Vision-Language-Action model.

    Inputs (forward):
        *images     : N camera tensors, each [B, 3, H, W]
        state       : [B, state_dim]  — normalised joint state
        task_ids    : [B, max_lang]   — tokenised task text

    Outputs:
        pred_chunk  : [B, chunk_size, action_dim] — normalised action predictions
        lb_loss     : scalar                       — MoE load-balancing aux loss
    """

    def __init__(self, cfg: TrainConfig, tokenizer: SimpleTokenizer):
        super().__init__()
        self.cfg = cfg

        self.vision_enc = VisionEncoder(
            d_out=cfg.d_vis,
            backbone=cfg.vision_backbone,
            n_cameras=len(cfg.cameras),
            share_proj=cfg.share_proj,
        )
        self.lang_enc = LanguageEncoder(
            vocab_size=tokenizer.vocab_size,
            d_out=cfg.d_lang,
        )
        self.state_enc = StateEncoder(
            state_dim=cfg.state_dim,
            d_out=cfg.d_state,
        )
        self.moe = FusionMoE(
            d_vis=cfg.d_vis,
            d_lang=cfg.d_lang,
            d_state=cfg.d_state,
            d_model=cfg.d_model,
            n_experts=cfg.n_experts,
            top_k=cfg.top_k,
        )
        # Action head: d_model → chunk_size * action_dim
        self.action_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.chunk_size * cfg.action_dim),
        )

    def forward(
        self,
        *images: torch.Tensor,
        state: torch.Tensor,
        task_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vis   = self.vision_enc(*images)            # [B, d_vis]
        lang  = self.lang_enc(task_ids)             # [B, d_lang]
        st    = self.state_enc(state)               # [B, d_state]
        fused, lb_loss = self.moe(vis, lang, st)    # [B, d_model]
        flat  = self.action_head(fused)             # [B, C * action_dim]
        chunk = flat.view(-1, self.cfg.chunk_size, self.cfg.action_dim)
        return chunk, lb_loss

    @property
    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        cfg = self.cfg
        return (
            f"VLAModel | arm={cfg.arm_name} | DOF={cfg.action_dim} "
            f"| cameras={cfg.cameras} | backbone={cfg.vision_backbone}\n"
            f"  Experts={cfg.n_experts} top-{cfg.top_k} | chunk={cfg.chunk_size} "
            f"| d_model={cfg.d_model}\n"
            f"  Trainable params: {self.n_trainable:,}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def compute_loss(
    pred: torch.Tensor,       # [B, C, action_dim]
    gt: torch.Tensor,         # [B, C, action_dim]
    lb: torch.Tensor,         # scalar
    cfg: TrainConfig,
    rl_penalty: Optional[torch.Tensor] = None,  # scalar, from RLSafetyValidator
) -> Tuple[torch.Tensor, dict]:
    act_loss = F.l1_loss(pred, gt)
    total = cfg.w_action * act_loss + cfg.w_moe * lb
    info = {"act_loss": act_loss.item(), "lb_loss": lb.item(), "rl_penalty": 0.0}
    if rl_penalty is not None and cfg.w_rl > 0:
        total = total + cfg.w_rl * rl_penalty
        info["rl_penalty"] = rl_penalty.item()
    info["total_loss"] = total.item()
    return total, info
