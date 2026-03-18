"""
customvla/models/encoders.py

All three encoder modules extracted from train.py and made arm-agnostic.

VisionEncoder  — DINOv2-S/14 (frozen) or ResNet-18, supports N cameras
LanguageEncoder — Word embedding + BiGRU + masked mean-pool
StateEncoder    — 3-layer MLP with LayerNorm
"""

import re
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
from typing import List, Optional

_DINO_DIM   = 384
_RESNET_DIM = 512

# ─────────────────────────────────────────────────────────────────────────────
# Vision Encoder
# ─────────────────────────────────────────────────────────────────────────────

class VisionEncoder(nn.Module):
    """
    Encode images from one or more cameras into a single fused embedding.

    Args:
        d_out      : output embedding dimension
        backbone   : "dinov2" (frozen ViT-S/14) | "resnet18" (layer4 trainable)
        n_cameras  : number of camera views (default 2 for top+wrist)
        share_proj : if True, all cameras share one projector (default True)
    """

    def __init__(
        self,
        d_out: int = 256,
        backbone: str = "dinov2",
        n_cameras: int = 2,
        share_proj: bool = True,
    ):
        super().__init__()
        self.n_cameras = n_cameras
        self.backbone_type = backbone
        self._backbone_dim = self._load_backbone(backbone)

        if share_proj:
            self.proj = nn.Sequential(
                nn.Linear(self._backbone_dim, d_out),
                nn.LayerNorm(d_out),
                nn.GELU(),
            )
            self.projs = None
        else:
            self.proj = None
            self.projs = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self._backbone_dim, d_out),
                    nn.LayerNorm(d_out),
                    nn.GELU(),
                )
                for _ in range(n_cameras)
            ])

        # Fuse all camera embeddings → single d_out vector
        self.fuse = nn.Sequential(
            nn.Linear(d_out * n_cameras, d_out),
            nn.LayerNorm(d_out),
            nn.GELU(),
        )

    def _load_backbone(self, backbone: str) -> int:
        if backbone == "dinov2":
            try:
                model = torch.hub.load(
                    "facebookresearch/dinov2", "dinov2_vits14",
                    pretrained=True, verbose=False,
                )
                for p in model.parameters():
                    p.requires_grad_(False)
                self._backbone = model
                print("  VisionEncoder: DINOv2 ViT-S/14 frozen ✓")
                return _DINO_DIM
            except Exception as exc:
                print(f"  DINOv2 unavailable ({exc}), falling back to ResNet-18")

        model = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()
        for name, p in model.named_parameters():
            p.requires_grad_(name.startswith("layer4"))
        self._backbone = model
        self.backbone_type = "resnet18"
        print("  VisionEncoder: ResNet-18 (layer4 trainable)")
        return _RESNET_DIM

    def _encode_one(self, img: torch.Tensor, cam_idx: int = 0) -> torch.Tensor:
        """[B, 3, H, W] → [B, d_out]"""
        if self.backbone_type == "dinov2":
            with torch.no_grad():
                feat = self._backbone(img)
        else:
            feat = self._backbone(img)

        proj = self.proj if self.proj is not None else self.projs[cam_idx]
        return proj(feat)

    def forward(self, *images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            *images: one tensor per camera, each [B, 3, H, W]
        Returns:
            fused embedding [B, d_out]
        """
        assert len(images) == self.n_cameras, (
            f"Expected {self.n_cameras} camera tensors, got {len(images)}"
        )
        feats = [self._encode_one(img, i) for i, img in enumerate(images)]
        return self.fuse(torch.cat(feats, dim=-1))


# ─────────────────────────────────────────────────────────────────────────────
# Language Tokenizer + Encoder
# ─────────────────────────────────────────────────────────────────────────────

class SimpleTokenizer:
    """
    Whitespace tokenizer. Vocabulary is built from task descriptions.
    For multi-task use, pass all task texts at construction time.
    """
    PAD = 0
    UNK = 1
    MAX_LEN = 64

    def __init__(self, texts: Optional[List[str]] = None):
        texts = texts or []
        words: set = set()
        for t in texts:
            words.update(re.sub(r"[^\w\s]", " ", t.lower()).split())
        self._w2i = {w: i + 2 for i, w in enumerate(sorted(words))}
        self.vocab_size = len(self._w2i) + 2

    def encode(self, text: str) -> torch.Tensor:
        toks = re.sub(r"[^\w\s]", " ", text.lower()).split()
        ids = [self._w2i.get(t, self.UNK) for t in toks[: self.MAX_LEN]]
        ids += [self.PAD] * (self.MAX_LEN - len(ids))
        return torch.tensor(ids[: self.MAX_LEN], dtype=torch.long)


class LanguageEncoder(nn.Module):
    """
    Word embedding → BiGRU → masked mean-pool → linear projection.

    Input : token ids  [B, L]
    Output: embedding  [B, d_out]
    """

    def __init__(
        self,
        vocab_size: int,
        d_word: int = 64,
        d_gru: int = 128,
        d_out: int = 128,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_word, padding_idx=0)
        self.gru = nn.GRU(
            d_word, d_gru,
            num_layers=2, bidirectional=True,
            batch_first=True, dropout=0.1,
        )
        self.proj = nn.Sequential(
            nn.Linear(d_gru * 2, d_out),
            nn.LayerNorm(d_out),
            nn.GELU(),
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x    = self.emb(ids)                              # [B, L, d_word]
        mask = (ids != 0).float().unsqueeze(-1)           # [B, L, 1]
        out, _ = self.gru(x)                              # [B, L, 2*d_gru]
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.proj(pooled)                          # [B, d_out]


# ─────────────────────────────────────────────────────────────────────────────
# State Encoder
# ─────────────────────────────────────────────────────────────────────────────

class StateEncoder(nn.Module):
    """
    3-layer MLP to embed joint states.

    Input : [B, state_dim]
    Output: [B, d_out]
    """

    def __init__(self, state_dim: int, d_out: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),  nn.GELU(),
            nn.Linear(64, 128),        nn.GELU(),
            nn.Linear(128, d_out),
            nn.LayerNorm(d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
