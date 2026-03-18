"""
customvla/models/fusion.py

Mixture-of-Experts fusion layer.
Takes (vision, language, state) embeddings and fuses them
through top-K expert routing with load-balancing auxiliary loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Expert(nn.Module):
    """3-layer FFN with residual + LayerNorm."""

    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d * 2, d), nn.GELU(),
            nn.Linear(d, d),
        )
        self.ln = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x + self.net(x))


class FusionMoE(nn.Module):
    """
    Fuse (vision, language, state) → d_model via MoE routing.

    Args:
        d_vis      : vision embedding dimension
        d_lang     : language embedding dimension
        d_state    : state embedding dimension
        d_model    : internal + output dimension
        n_experts  : total number of experts
        top_k      : how many experts are active per token

    Forward:
        Returns (output [B, d_model], lb_loss scalar).
        lb_loss is the load-balancing auxiliary loss from the Switch Transformer.
        Add  cfg.w_moe * lb_loss  to your total loss during training.
    """

    def __init__(
        self,
        d_vis: int,
        d_lang: int,
        d_state: int,
        d_model: int = 256,
        n_experts: int = 4,
        top_k: int = 2,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.d_model = d_model

        self.fusion = nn.Sequential(
            nn.Linear(d_vis + d_lang + d_state, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        # Zero-init gate → uniform routing at the start of training
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        nn.init.zeros_(self.gate.weight)

        self.experts = nn.ModuleList([Expert(d_model) for _ in range(n_experts)])

    def forward(
        self,
        vis: torch.Tensor,
        lang: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = vis.shape[0]
        x = self.fusion(torch.cat([vis, lang, state], dim=-1))  # [B, d_model]

        logits   = self.gate(x)                                  # [B, n_experts]
        all_probs = F.softmax(logits, dim=-1)                    # [B, n_experts]

        top_vals, top_idx = logits.topk(self.top_k, dim=-1)     # [B, k]
        top_w = F.softmax(top_vals, dim=-1)                      # [B, k] renorm

        # Dispatch through top-k experts
        out = torch.zeros(B, self.d_model, device=x.device, dtype=x.dtype)
        for k in range(self.top_k):
            for e in range(self.n_experts):
                mask = (top_idx[:, k] == e)
                if not mask.any():
                    continue
                w = top_w[mask, k].unsqueeze(-1)
                out[mask] = out[mask] + w * self.experts[e](x[mask])

        # Load-balancing loss:  L = n_experts * Σ_i (f_i · P_i)
        counts = torch.zeros(self.n_experts, device=x.device, dtype=x.dtype)
        for k in range(self.top_k):
            counts.scatter_add_(
                0, top_idx[:, k],
                torch.ones(B, device=x.device, dtype=x.dtype),
            )
        f_i    = counts / (B * self.top_k)
        lb_loss = self.n_experts * (f_i * all_probs.mean(0)).sum()

        return out, lb_loss

    def expert_utilization(
        self,
        vis: torch.Tensor,
        lang: torch.Tensor,
        state: torch.Tensor,
    ) -> dict:
        """
        Diagnostic: returns per-expert routing fraction over a batch.
        Call after forward on a validation batch to monitor load balance.
        """
        with torch.no_grad():
            x = self.fusion(torch.cat([vis, lang, state], dim=-1))
            logits = self.gate(x)
            top_idx = logits.topk(self.top_k, dim=-1).indices
            B = vis.shape[0]
            counts = torch.zeros(self.n_experts)
            for k in range(self.top_k):
                for e in range(self.n_experts):
                    counts[e] += (top_idx[:, k] == e).sum().item()
            fracs = (counts / (B * self.top_k)).tolist()
        return {f"expert_{e}": round(fracs[e], 4) for e in range(self.n_experts)}
