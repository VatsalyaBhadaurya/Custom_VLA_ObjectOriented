"""
customvla/rl/safety.py

RL Safety Validator — PPO-style critic that:

  1. Scores an action chunk against joint limits (rule-based hard constraints)
  2. Learns a soft value function (V(s, a)) that estimates trajectory quality
  3. Provides a differentiable penalty term for the VLA training loss
  4. Can be used at inference time to gate / clip unsafe actions

Architecture:
  StateEncoder(MLP) + ActionEncoder(MLP)  → concat → PPO Critic head → V(s,a)
  Additionally: hard limit checking per-joint
  
Usage in training:
  validator = RLSafetyValidator(arm)
  # inside train loop:
  penalty = validator.compute_penalty(pred_chunk, state, arm)
  loss = vla_loss + cfg.w_rl * penalty

Usage at inference:
  safe_action, info = validator.validate_and_clip(raw_action, arm)
"""

from __future__ import annotations
from typing import Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from customvla.arms.base import BaseArm


# ─────────────────────────────────────────────────────────────────────────────
# Hard constraint checker (rule-based, no learning required)
# ─────────────────────────────────────────────────────────────────────────────

class HardLimitChecker:
    """
    Pure rule-based joint limit enforcement.
    No parameters — works entirely from arm.JOINT_LIMITS.
    """

    def __init__(self, arm: BaseArm):
        self.arm = arm
        self.lo, self.hi = arm.get_limit_arrays()
        self.lo_t = torch.from_numpy(self.lo)
        self.hi_t = torch.from_numpy(self.hi)

    def check_batch(
        self,
        actions: torch.Tensor,  # [B, action_dim] or [B, C, action_dim]
    ) -> Dict[str, object]:
        """
        Returns:
            violations_per_joint : [action_dim] int tensor — count of violations
            fraction_safe        : float — fraction of (batch * chunk) steps that are safe
            penalty              : scalar tensor — differentiable soft penalty
        """
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)   # [B, 1, D]

        lo = self.lo_t.to(actions.device)
        hi = self.hi_t.to(actions.device)

        below = (actions < lo).float()   # [B, C, D]
        above = (actions > hi).float()

        # Soft hinge penalty: sum of margin violations
        below_pen = F.relu(lo - actions).sum()
        above_pen = F.relu(actions - hi).sum()
        penalty   = (below_pen + above_pen) / (actions.numel() + 1e-8)

        violations_per_joint = (below + above).sum(dim=(0, 1)).long()  # [D]
        n_total = actions.shape[0] * actions.shape[1]
        n_safe  = ((below + above).sum(dim=-1) == 0).sum().item()

        return {
            "violations_per_joint": violations_per_joint,
            "fraction_safe": n_safe / max(n_total, 1),
            "penalty": penalty,
        }

    def clip(self, action: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Clip a single action vector to joint limits. Returns (clipped, was_safe)."""
        clipped = np.clip(action, self.lo, self.hi)
        was_safe = np.allclose(action, clipped)
        return clipped, was_safe


# ─────────────────────────────────────────────────────────────────────────────
# PPO Critic (learned value function)
# ─────────────────────────────────────────────────────────────────────────────

class PPOCritic(nn.Module):
    """
    Critic network V(state, action) for PPO-style training.

    Given the current state and a proposed first-step action,
    outputs a scalar value estimating trajectory quality.

    During VLA training: used to generate a soft penalty signal.
    Trained separately or jointly (set rl_lr in TrainConfig).
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        in_dim = state_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),     nn.Tanh(),
            nn.Linear(hidden, hidden),     nn.Tanh(),
            nn.Linear(hidden, hidden // 2), nn.Tanh(),
            nn.Linear(hidden // 2, 1),
        )
        # Orthogonal init — standard for PPO critics
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        # Last layer with small gain
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)

    def forward(
        self,
        state: torch.Tensor,   # [B, state_dim]
        action: torch.Tensor,  # [B, action_dim]  — first step of chunk
    ) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x).squeeze(-1)  # [B]


# ─────────────────────────────────────────────────────────────────────────────
# Full Safety Validator
# ─────────────────────────────────────────────────────────────────────────────

class RLSafetyValidator(nn.Module):
    """
    Combined hard + soft safety layer.

    Components:
      HardLimitChecker  — differentiable hinge penalty on joint limits
      PPOCritic         — learned value function for soft quality penalty

    The composite penalty (used in VLA training loss):
        penalty = hard_penalty + alpha * soft_penalty

    where soft_penalty = relu(-V(s,a))  (penalise low-value actions)
    and alpha scales the learned component relative to the rule-based one.

    Args:
        arm        : a BaseArm instance (provides limits and dims)
        hidden     : hidden size for PPO critic
        alpha      : weight between hard (1.0) and soft (alpha) penalties
    """

    def __init__(
        self,
        arm: BaseArm,
        hidden: int = 256,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.arm   = arm
        self.alpha = alpha
        self.hard  = HardLimitChecker(arm)
        self.critic = PPOCritic(arm.STATE_DIM, arm.ACTION_DIM, hidden)

    # ── Training-time API ─────────────────────────────────────────────────

    def compute_penalty(
        self,
        pred_chunk: torch.Tensor,  # [B, C, action_dim]  — normalised
        state: torch.Tensor,       # [B, state_dim]       — normalised
        action_mean: torch.Tensor, # for unnormalisation
        action_std: torch.Tensor,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns a scalar penalty tensor to add to VLA training loss.
        Gradient flows back through both hard hinge and PPO critic.
        """
        # Unnormalise for limit checking (limits are in real joint space)
        pred_real  = pred_chunk  * action_std + action_mean   # [B, C, D]
        state_real = state       * state_std  + state_mean    # [B, D]

        # Hard limit penalty
        hard_info = self.hard.check_batch(pred_real)
        hard_pen  = hard_info["penalty"]

        # Soft critic penalty — evaluate on first predicted step
        first_action = pred_real[:, 0, :]                     # [B, D]
        value = self.critic(state_real, first_action)         # [B]
        soft_pen = F.relu(-value).mean()                      # penalise V < 0

        return hard_pen + self.alpha * soft_pen

    def ppo_loss(
        self,
        states: torch.Tensor,    # [B, state_dim]
        actions: torch.Tensor,   # [B, action_dim]
        returns: torch.Tensor,   # [B]  — Monte Carlo or GAE returns
        old_values: torch.Tensor,# [B]  — from previous critic pass
        clip_eps: float = 0.2,
    ) -> torch.Tensor:
        """
        Standard PPO value loss (clipped). Call this when fine-tuning
        the critic with real rollout data from the robot.
        """
        values = self.critic(states, actions)                 # [B]
        v_clipped = old_values + torch.clamp(
            values - old_values, -clip_eps, clip_eps
        )
        loss_v1 = F.mse_loss(values, returns)
        loss_v2 = F.mse_loss(v_clipped, returns)
        return 0.5 * torch.max(loss_v1, loss_v2)

    # ── Inference-time API ────────────────────────────────────────────────

    @torch.no_grad()
    def validate_and_clip(
        self,
        action: np.ndarray,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Validate and hard-clip a single action vector (numpy).
        Use this in the inference loop before sending to the robot.

        Returns:
            safe_action  : np.ndarray — clipped to joint limits
            info         : dict with validation results
        """
        result  = self.arm.validate_action(action)
        clipped = result["clipped"]
        info = {
            "was_safe":   result["valid"],
            "violations": result["violations"],
            "clipped":    clipped,
        }
        if verbose and not result["valid"]:
            print(f"  [RL Validator] Joint limit violations: {result['violations']}")
        return clipped, info

    @torch.no_grad()
    def score_action(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> float:
        """
        Query the learned critic V(s, a). Higher = better.
        Useful for action selection / beam search at inference.
        """
        s_t = torch.from_numpy(state).float().unsqueeze(0)
        a_t = torch.from_numpy(action).float().unsqueeze(0)
        return self.critic(s_t, a_t).item()

    def describe(self) -> str:
        hard_params = 0
        soft_params = sum(p.numel() for p in self.critic.parameters())
        return (
            f"RLSafetyValidator | arm={self.arm.__class__.__name__}\n"
            f"  HardLimitChecker : rule-based, 0 params, {self.arm.ACTION_DIM} joints\n"
            f"  PPOCritic        : {soft_params:,} params\n"
            f"  alpha (hard:soft balance) = {self.alpha}"
        )
