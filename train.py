#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════════════
#  vla_train_so100.py  —  Full VLA Training Pipeline
#  Dataset : Tomas0413/so100_screw_lid_v0  (HuggingFace)
#
#  Architecture:
#    Image(top+wrist) → VisionEncoder(DINOv2-S/14 frozen | ResNet-18)  → vis  [B,256]
#    Task text        → LanguageEncoder(BiGRU)                          → lang [B,128]
#    Robot state(6D)  → StateEncoder(MLP)                               → st   [B,128]
#    cat([vis,lang,st]) → FusionMoE(4 experts, top-2)                  → [B,256]
#    → ActionHead(MLP)  → pred_chunk [B, chunk_size, 6]
#
#  Training:
#    Behaviour cloning · L1 action loss · load-balance aux loss
#    AdamW + cosine LR · mixed precision (CUDA) · gradient clipping
#
#  Usage (two phases):
#    # Phase 1 — download dataset + extract video frames to JPEG (run once):
#    python vla_train_so100.py --prepare
#
#    # Phase 2 — train:
#    python vla_train_so100.py --train
#
#    # Both at once:
#    python vla_train_so100.py --prepare --train
#
#  Requirements:
#    pip install torch torchvision av pandas pyarrow \
#                huggingface_hub pillow tqdm
#
#  Optional (for DINOv2 backbone):
#    pip install timm   # already pulled in by torch.hub
# ══════════════════════════════════════════════════════════════════════════════

import os, sys, json, csv, re, math, time, random, argparse, shutil
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

try:
    import av
    AV_OK = True
except ImportError:
    AV_OK = False
    print("WARNING: PyAV not found. Install: pip install av")

import torchvision.transforms as T
import torchvision.models as tvm
from huggingface_hub import snapshot_download

# ─────────────────────────────────────────────────────────────────────────────
# § 1  Dataset-specific constants  (from dataset card + sister SO100 datasets)
# ─────────────────────────────────────────────────────────────────────────────
REPO_ID    = "Tomas0413/so100_screw_lid_v0"
ACTION_DIM = 6          # 6-DoF: shoulder_pan, _lift, elbow_flex, wrist_flex, _roll, gripper
STATE_DIM  = 6          # same joint order as actions
CAMERAS    = ["top", "wrist"]
FPS        = 30
IMG_SIZE   = 224        # resize all frames to 224×224 (ViT / ImageNet standard)

JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex",   "wrist_roll",    "gripper",
]

# Single task description for the entire dataset
TASK_TEXT = (
    "Pick the plastic jar from the table, place it upright on the silicone puck, "
    "seat the lid on the jar to engage the threads, "
    "then carry the closed jar to the wooden goal block."
)

# ─────────────────────────────────────────────────────────────────────────────
# § 2  Training Config
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    # Paths
    data_dir:    str   = "./data/so100_screw_lid_v0"
    cache_dir:   str   = "./cache/so100_frames"
    output_dir:  str   = "./runs/vla_so100"

    # Architecture
    d_vis:       int   = 256   # vision embedding dim (per-camera projector output)
    d_lang:      int   = 128   # language embedding dim
    d_state:     int   = 128   # state embedding dim
    d_model:     int   = 256   # internal MoE/head dim
    n_experts:   int   = 4     # number of MoE experts
    top_k:       int   = 2     # MoE top-K routing
    chunk_size:  int   = 16    # action prediction horizon (frames)
    vision_backbone: str = "dinov2"  # "dinov2" or "resnet18"

    # Training
    epochs:      int   = 100
    batch_size:  int   = 16
    lr:          float = 1e-4
    weight_decay:float = 1e-4
    grad_clip:   float = 1.0
    val_frac:    float = 0.10   # fraction of episodes held out for validation

    # Loss weights
    w_action:    float = 1.0    # primary L1 action chunk loss
    w_moe:       float = 0.01   # auxiliary MoE load-balancing loss

    # Misc
    num_workers: int   = 2
    seed:        int   = 42
    save_every:  int   = 10     # save numbered checkpoint every N epochs
    log_every:   int   = 25     # print progress every N batches
    device:      str   = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )


# ─────────────────────────────────────────────────────────────────────────────
# § 3  Data Preparation
#       3a  download_dataset  — HF snapshot → local disk
#       3b  extract_frames    — MP4 → per-frame JPEG (run once)
#       3c  compute_stats     — action / state mean & std from parquet
# ─────────────────────────────────────────────────────────────────────────────

def download_dataset(cfg: TrainConfig) -> None:
    """Download the full dataset from HuggingFace Hub (skipped if already present)."""
    sentinel = Path(cfg.data_dir) / "meta" / "info.json"
    if sentinel.exists():
        print(f"  Dataset already at {cfg.data_dir} — skipping download.")
        return
    print(f"  Downloading {REPO_ID} to {cfg.data_dir} …")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=cfg.data_dir,
        ignore_patterns=["*.git*", "*.gitattributes"],
    )
    print("  Download complete.")


def _extract_video_to_jpegs(video_path: Path, out_dir: Path, size: int) -> int:
    """
    Sequentially decode an MP4 with PyAV and save each frame as JPEG.
    Returns the number of frames extracted.
    """
    if not AV_OK:
        raise RuntimeError("PyAV (av) is required for frame extraction. "
                           "pip install av")
    out_dir.mkdir(parents=True, exist_ok=True)
    resize = T.Resize((size, size),
                      interpolation=T.InterpolationMode.BILINEAR,
                      antialias=True)
    n = 0
    try:
        container = av.open(str(video_path))
        stream    = container.streams.video[0]
        stream.thread_type = "AUTO"          # use multi-threaded decode
        for frame in container.decode(stream):
            dest = out_dir / f"{n:06d}.jpg"
            if not dest.exists():
                img = resize(frame.to_image())   # PIL.Image, resized
                img.save(dest, quality=92)
            n += 1
        container.close()
    except Exception as exc:
        print(f"    WARNING: extraction error for {video_path}: {exc}")
    return n


def extract_frames(cfg: TrainConfig) -> None:
    """
    Walk every episode's MP4 files and extract frames to:
      {cache_dir}/episode_XXXXXX/{camera}/{frame_index:06d}.jpg

    A .done sentinel file prevents re-extraction on subsequent runs.
    """
    data_path  = Path(cfg.data_dir)
    cache_path = Path(cfg.cache_dir)
    pq_dir     = data_path / "data" / "chunk-000"
    episodes   = sorted(pq_dir.glob("episode_*.parquet"))

    print(f"  Extracting frames for {len(episodes)} episodes "
          f"({len(CAMERAS)} cameras each) …")

    for pf in tqdm(episodes, desc="Episodes"):
        ep_str  = pf.stem                            # episode_000000
        ep_dir  = cache_path / ep_str
        done    = ep_dir / ".done"
        if done.exists():
            continue

        for cam in CAMERAS:
            vid = (data_path / "videos" / "chunk-000"
                   / f"observation.images.{cam}" / f"{ep_str}.mp4")
            if not vid.exists():
                print(f"    MISSING: {vid}")
                continue
            n = _extract_video_to_jpegs(vid, ep_dir / cam, IMG_SIZE)

        ep_dir.mkdir(parents=True, exist_ok=True)
        done.touch()

    print("  Frame extraction complete.")


def _read_parquet_column(df: pd.DataFrame, col: str) -> np.ndarray:
    """
    Safely read a parquet column that may be stored as:
      • list of Python lists / np.arrays
      • list of bytes (unusual)
    Returns np.ndarray [N, D] float32.
    """
    raw = df[col].tolist()
    try:
        return np.stack([np.asarray(r, dtype=np.float32) for r in raw])
    except Exception as e:
        raise ValueError(f"Cannot parse column '{col}': {e}  "
                         f"(first element type: {type(raw[0])})")


def compute_normalization_stats(cfg: TrainConfig) -> Dict[str, Any]:
    """
    Compute per-joint mean & std across all episodes.
    Result saved to {cache_dir}/stats.json (cached across runs).
    """
    stats_path = Path(cfg.cache_dir) / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        print(f"  Loaded normalisation stats from {stats_path}")
        return stats

    print("  Computing normalisation statistics …")
    pq_dir = Path(cfg.data_dir) / "data" / "chunk-000"
    all_actions, all_states = [], []

    for pf in sorted(pq_dir.glob("episode_*.parquet")):
        df = pd.read_parquet(pf, columns=["action", "observation.state"])
        all_actions.append(_read_parquet_column(df, "action"))
        all_states.append(_read_parquet_column(df, "observation.state"))

    acts  = np.concatenate(all_actions, axis=0)   # [total_frames, 6]
    stats_val = np.concatenate(all_states,  axis=0)

    stats = {
        "action_mean": acts.mean(axis=0).tolist(),
        "action_std":  np.maximum(acts.std(axis=0), 1e-6).tolist(),
        "state_mean":  stats_val.mean(axis=0).tolist(),
        "state_std":   np.maximum(stats_val.std(axis=0), 1e-6).tolist(),
        "total_frames": int(len(acts)),
    }

    Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  Stats computed over {stats['total_frames']:,} frames → {stats_path}")
    _print_stats_table(stats)
    return stats


def _print_stats_table(stats: Dict) -> None:
    print(f"  {'Joint':<20} {'act_mean':>10} {'act_std':>10} "
          f"{'st_mean':>10} {'st_std':>10}")
    for i, name in enumerate(JOINT_NAMES):
        print(f"  {name:<20} "
              f"{stats['action_mean'][i]:>10.4f} "
              f"{stats['action_std'][i]:>10.4f} "
              f"{stats['state_mean'][i]:>10.4f} "
              f"{stats['state_std'][i]:>10.4f}")


def prepare(cfg: TrainConfig) -> Dict:
    """Full data-preparation pipeline (idempotent)."""
    Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)
    download_dataset(cfg)
    extract_frames(cfg)
    stats = compute_normalization_stats(cfg)
    print("  ✓ Data preparation complete.\n")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# § 4  PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

# Standard ImageNet normalisation (matches DINOv2 / torchvision pretrain)
_IMG_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


class So100Dataset(Dataset):
    """
    One item = one timestep t in a demonstration episode:
      img_top    [3, 224, 224]   overhead RGB frame (normalised)
      img_wrist  [3, 224, 224]   wrist RGB frame (normalised)
      state      [6]             normalised joint state at t
      act_chunk  [chunk_size, 6] normalised action sequence [t … t+C-1]
                                 (last action repeated at episode boundaries)
      task_ids   [max_lang_len]  tokenised task description (same for all items)
    """

    def __init__(self, cfg: TrainConfig, episode_indices: List[int],
                 stats: Dict, tokenizer: "SimpleTokenizer"):
        self.cfg        = cfg
        self.cache_path = Path(cfg.cache_dir)
        self.data_path  = Path(cfg.data_dir)
        self.chunk_size = cfg.chunk_size
        self.tok        = tokenizer

        # Normalisation constants (numpy, broadcast-friendly)
        self._act_mean = np.asarray(stats["action_mean"], np.float32)
        self._act_std  = np.asarray(stats["action_std"],  np.float32)
        self._st_mean  = np.asarray(stats["state_mean"],  np.float32)
        self._st_std   = np.asarray(stats["state_std"],   np.float32)

        # Build flat index: list of (ep_str, frame_idx, state_t, chunk)
        self._samples: List[Tuple] = []
        self._build_index(episode_indices)

        # Pre-encode task text — identical for every sample
        self._task_ids: torch.Tensor = tokenizer.encode(TASK_TEXT)

    # ── index building ────────────────────────────────────────────────────
    def _build_index(self, episode_indices: List[int]) -> None:
        pq_dir = self.data_path / "data" / "chunk-000"
        skipped = 0
        for ep_idx in sorted(episode_indices):
            ep_str = f"episode_{ep_idx:06d}"
            pf     = pq_dir / f"{ep_str}.parquet"
            if not pf.exists():
                skipped += 1
                continue

            # Check frame cache is ready
            ep_cache = self.cache_path / ep_str
            if not (ep_cache / ".done").exists():
                skipped += 1
                continue

            df  = pd.read_parquet(
                pf, columns=["frame_index", "action", "observation.state"]
            )
            N   = len(df)
            fis = df["frame_index"].values.astype(int)          # [N] within-episode
            acts  = _read_parquet_column(df, "action")           # [N, 6]
            sts   = _read_parquet_column(df, "observation.state")  # [N, 6]

            for i in range(N):
                # Action chunk [i … i+chunk_size), pad at boundary
                end   = min(i + self.chunk_size, N)
                chunk = acts[i:end].copy()
                if len(chunk) < self.chunk_size:
                    pad   = np.tile(acts[-1:], (self.chunk_size - len(chunk), 1))
                    chunk = np.concatenate([chunk, pad], axis=0)   # [C, 6]

                self._samples.append((ep_str, int(fis[i]), sts[i], chunk))

        if skipped:
            print(f"  WARNING: {skipped} episodes skipped (missing parquet or cache)")

    # ── torch Dataset interface ───────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        ep_str, fi, state, chunk = self._samples[idx]

        img_top   = self._load_jpg(ep_str, "top",   fi)
        img_wrist = self._load_jpg(ep_str, "wrist", fi)

        # Normalise state + actions
        state_n = (state - self._st_mean) / self._st_std           # [6]
        chunk_n = (chunk - self._act_mean) / self._act_std          # [C, 6]

        return (
            img_top,                                   # [3, H, W]
            img_wrist,                                 # [3, H, W]
            torch.from_numpy(state_n.copy()),          # [6]
            torch.from_numpy(chunk_n.copy()),          # [chunk_size, 6]
            self._task_ids.clone(),                    # [max_lang_len]
        )

    def _load_jpg(self, ep_str: str, cam: str, frame_idx: int) -> torch.Tensor:
        path = self.cache_path / ep_str / cam / f"{frame_idx:06d}.jpg"
        try:
            img = Image.open(path).convert("RGB")
            return _IMG_TRANSFORM(img)
        except Exception:
            # Zero frame as graceful fallback (broken/missing JPEG)
            return torch.zeros(3, IMG_SIZE, IMG_SIZE)


# ─────────────────────────────────────────────────────────────────────────────
# § 5  Language — Tokenizer + BiGRU Encoder
# ─────────────────────────────────────────────────────────────────────────────

class SimpleTokenizer:
    """
    Whitespace tokenizer.  Vocabulary is built from TASK_TEXT only since this
    is a single-task dataset.  Extend _CORE_WORDS for multi-task use.
    """
    PAD     = 0
    UNK     = 1
    MAX_LEN = 40

    def __init__(self):
        words = set(re.sub(r"[^\w\s]", " ", TASK_TEXT.lower()).split())
        self._w2i   = {w: i + 2 for i, w in enumerate(sorted(words))}
        self.vocab_size = len(self._w2i) + 2

    def encode(self, text: str) -> torch.Tensor:
        toks = re.sub(r"[^\w\s]", " ", text.lower()).split()
        ids  = [self._w2i.get(t, self.UNK) for t in toks[:self.MAX_LEN]]
        ids += [self.PAD] * (self.MAX_LEN - len(ids))
        return torch.tensor(ids[:self.MAX_LEN], dtype=torch.long)


class LanguageEncoder(nn.Module):
    """
    Word-embedding → BiGRU → masked mean-pool → linear projection.
    Input  : token ids [B, L]
    Output : lang embedding [B, d_out]
    """
    def __init__(self, vocab_size: int, d_word: int = 64,
                 d_gru: int = 128, d_out: int = 128):
        super().__init__()
        self.emb  = nn.Embedding(vocab_size, d_word, padding_idx=0)
        self.gru  = nn.GRU(d_word, d_gru, num_layers=2,
                           bidirectional=True, batch_first=True,
                           dropout=0.1)
        self.proj = nn.Sequential(
            nn.Linear(d_gru * 2, d_out),
            nn.LayerNorm(d_out),
            nn.GELU(),
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x      = self.emb(ids)                                  # [B, L, d_word]
        mask   = (ids != 0).float().unsqueeze(-1)               # [B, L, 1]
        out, _ = self.gru(x)                                    # [B, L, 2*d_gru]
        # Masked mean-pooling (ignore PAD positions)
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1) # [B, 2*d_gru]
        return self.proj(pooled)                                 # [B, d_out]


# ─────────────────────────────────────────────────────────────────────────────
# § 6  Vision Encoder
#        DINOv2 ViT-S/14 (frozen) — preferred, 384 → d_vis
#        ResNet-18 (last block trainable) — fallback, 512 → d_vis
#        Two cameras → separate projectors → cross-camera fusion linear
# ─────────────────────────────────────────────────────────────────────────────
_DINO_DIM   = 384
_RESNET_DIM = 512


class VisionEncoder(nn.Module):
    def __init__(self, d_out: int = 256, backbone: str = "dinov2"):
        super().__init__()
        self.backbone_type = backbone
        self._backbone_dim  = self._load_backbone(backbone)

        # One projector shared across both cameras (weight sharing is a
        # reasonable inductive bias when both cameras see the same workspace)
        self.proj = nn.Sequential(
            nn.Linear(self._backbone_dim, d_out),
            nn.LayerNorm(d_out),
            nn.GELU(),
        )

        # Fuse top + wrist embeddings → single d_out vector
        self.fuse = nn.Sequential(
            nn.Linear(d_out * 2, d_out),
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
                    p.requires_grad_(False)   # backbone is fully frozen
                self._backbone = model
                print("  VisionEncoder: DINOv2 ViT-S/14 — frozen backbone ✓")
                return _DINO_DIM
            except Exception as exc:
                print(f"  DINOv2 unavailable ({exc}), falling back to ResNet-18")

        # ResNet-18 fallback: fine-tune only layer4 + fc projection
        model = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()
        for name, p in model.named_parameters():
            p.requires_grad_(name.startswith("layer4") or name == "fc")
        self._backbone = model
        self.backbone_type = "resnet18"
        print("  VisionEncoder: ResNet-18 — layer4 fine-tunable")
        return _RESNET_DIM

    def _encode_one(self, img: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] → [B, d_out]"""
        if self.backbone_type == "dinov2":
            with torch.no_grad():
                feat = self._backbone(img)       # [B, 384]
        else:
            feat = self._backbone(img)           # [B, 512]
        return self.proj(feat)                   # [B, d_out]

    def forward(self, img_top: torch.Tensor,
                img_wrist: torch.Tensor) -> torch.Tensor:
        v_top   = self._encode_one(img_top)
        v_wrist = self._encode_one(img_wrist)
        return self.fuse(torch.cat([v_top, v_wrist], dim=-1))   # [B, d_out]


# ─────────────────────────────────────────────────────────────────────────────
# § 7  State Encoder  (MLP, always trainable)
# ─────────────────────────────────────────────────────────────────────────────

class StateEncoder(nn.Module):
    def __init__(self, in_dim: int = STATE_DIM, d_out: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),   nn.GELU(),
            nn.Linear(64, 128),      nn.GELU(),
            nn.Linear(128, d_out),
            nn.LayerNorm(d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)    # [B, 6] → [B, d_out]


# ─────────────────────────────────────────────────────────────────────────────
# § 8  Mixture of Experts
# ─────────────────────────────────────────────────────────────────────────────

class Expert(nn.Module):
    """3-layer FFN with residual connection and dropout."""
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d * 2),  nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d * 2, d),  nn.GELU(),
            nn.Linear(d, d),
        )
        self.ln = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x + self.net(x))


class FusionMoE(nn.Module):
    """
    Fuse (vision, language, state) → d_model via linear projection,
    then route through top-K experts, returning a weighted combination.

    Also returns a scalar auxiliary load-balancing loss (importance × load).

    Input dims: d_vis + d_lang + d_state
    Output    : [B, d_model], lb_loss scalar
    """

    def __init__(self, d_vis: int, d_lang: int, d_state: int,
                 d_model: int = 256, n_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k     = top_k
        self.d_model   = d_model

        self.fusion = nn.Sequential(
            nn.Linear(d_vis + d_lang + d_state, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Router: zero-initialised → uniform start distribution
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        nn.init.zeros_(self.gate.weight)

        self.experts = nn.ModuleList([Expert(d_model) for _ in range(n_experts)])

    def forward(self, vis: torch.Tensor, lang: torch.Tensor,
                state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = vis.shape[0]
        x = self.fusion(torch.cat([vis, lang, state], dim=-1))   # [B, d_model]

        logits    = self.gate(x)                                  # [B, n_experts]
        all_probs = F.softmax(logits, dim=-1)                     # [B, n_experts]

        top_vals, top_idx = logits.topk(self.top_k, dim=-1)      # [B, k]
        top_w             = F.softmax(top_vals, dim=-1)           # [B, k] renorm

        # Weighted dispatch
        out = torch.zeros(B, self.d_model, device=x.device, dtype=x.dtype)
        for k in range(self.top_k):
            for e in range(self.n_experts):
                mask = (top_idx[:, k] == e)
                if not mask.any():
                    continue
                w = top_w[mask, k].unsqueeze(-1)    # [m, 1]
                out[mask] = out[mask] + w * self.experts[e](x[mask])

        # Load-balancing auxiliary loss
        # L_aux = n * Σ_i (f_i · P_i)
        # f_i = fraction of tokens routed to expert i (over top-k selections)
        # P_i = mean soft probability assigned to expert i
        counts = torch.zeros(self.n_experts, device=x.device, dtype=x.dtype)
        for k in range(self.top_k):
            counts.scatter_add_(0, top_idx[:, k],
                                torch.ones(B, device=x.device, dtype=x.dtype))
        f_i     = counts / (B * self.top_k)
        lb_loss = self.n_experts * (f_i * all_probs.mean(0)).sum()

        return out, lb_loss


# ─────────────────────────────────────────────────────────────────────────────
# § 9  Full VLA Model
# ─────────────────────────────────────────────────────────────────────────────

class VLAModel(nn.Module):
    """
    Vision-Language-Action model for SO100 manipulation.

    forward() inputs:
        img_top   [B, 3, 224, 224]
        img_wrist [B, 3, 224, 224]
        state     [B, 6]           normalised joint state
        task_ids  [B, max_lang]    task token ids

    forward() outputs:
        pred_chunk [B, chunk_size, 6]   normalised predicted actions
        lb_loss    scalar               MoE load-balancing aux loss
    """

    def __init__(self, cfg: TrainConfig, tokenizer: SimpleTokenizer):
        super().__init__()
        self.cfg = cfg

        self.vision_enc = VisionEncoder(d_out=cfg.d_vis,
                                         backbone=cfg.vision_backbone)
        self.lang_enc   = LanguageEncoder(tokenizer.vocab_size,
                                           d_out=cfg.d_lang)
        self.state_enc  = StateEncoder(STATE_DIM, d_out=cfg.d_state)
        self.moe        = FusionMoE(cfg.d_vis, cfg.d_lang, cfg.d_state,
                                    cfg.d_model, cfg.n_experts, cfg.top_k)

        # Action prediction head: [B, d_model] → [B, chunk_size * 6]
        self.action_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.chunk_size * ACTION_DIM),
        )

    def forward(
        self,
        img_top:   torch.Tensor,
        img_wrist: torch.Tensor,
        state:     torch.Tensor,
        task_ids:  torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vis   = self.vision_enc(img_top, img_wrist)   # [B, d_vis]
        lang  = self.lang_enc(task_ids)               # [B, d_lang]
        st    = self.state_enc(state)                 # [B, d_state]

        fused, lb_loss = self.moe(vis, lang, st)      # [B, d_model]

        flat  = self.action_head(fused)               # [B, C*6]
        chunk = flat.view(-1, self.cfg.chunk_size, ACTION_DIM)  # [B, C, 6]
        return chunk, lb_loss

    @property
    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# § 10  DataLoaders
# ─────────────────────────────────────────────────────────────────────────────

def make_dataloaders(
    cfg: TrainConfig, stats: Dict, tokenizer: SimpleTokenizer
) -> Tuple[DataLoader, DataLoader]:

    pq_dir   = Path(cfg.data_dir) / "data" / "chunk-000"
    all_eps  = sorted(
        int(f.stem.split("_")[1]) for f in pq_dir.glob("episode_*.parquet")
    )

    random.seed(cfg.seed)
    random.shuffle(all_eps)
    n_val   = max(1, int(len(all_eps) * cfg.val_frac))
    val_eps = all_eps[:n_val]
    tr_eps  = all_eps[n_val:]

    print(f"  Episodes — train: {len(tr_eps)}, val: {len(val_eps)}")

    tr_ds  = So100Dataset(cfg, tr_eps,  stats, tokenizer)
    val_ds = So100Dataset(cfg, val_eps, stats, tokenizer)

    print(f"  Samples  — train: {len(tr_ds):,}, val: {len(val_ds):,}")

    loader_kw = dict(
        batch_size  = cfg.batch_size,
        num_workers = cfg.num_workers,
        pin_memory  = (cfg.device == "cuda"),
        persistent_workers = (cfg.num_workers > 0),
    )
    tr_dl  = DataLoader(tr_ds,  shuffle=True,  **loader_kw)
    val_dl = DataLoader(val_ds, shuffle=False, **loader_kw)
    return tr_dl, val_dl


# ─────────────────────────────────────────────────────────────────────────────
# § 11  Loss
# ─────────────────────────────────────────────────────────────────────────────

def compute_loss(
    pred: torch.Tensor,    # [B, C, 6]
    gt:   torch.Tensor,    # [B, C, 6]
    lb:   torch.Tensor,    # scalar
    cfg:  TrainConfig,
) -> Tuple[torch.Tensor, float, float]:
    act_loss = F.l1_loss(pred, gt)
    total    = cfg.w_action * act_loss + cfg.w_moe * lb
    return total, act_loss.item(), lb.item()


# ─────────────────────────────────────────────────────────────────────────────
# § 12  Train / Val loops
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model:     VLAModel,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler:    Optional[torch.amp.GradScaler],
    cfg:       TrainConfig,
    epoch:     int,
) -> Dict[str, float]:

    model.train()
    device = cfg.device
    total_loss = act_sum = lb_sum = 0.0
    n = 0

    for bi, (img_top, img_wrist, state, act_chunk, task_ids) in enumerate(loader):
        img_top   = img_top.to(device,   non_blocking=True)
        img_wrist = img_wrist.to(device, non_blocking=True)
        state     = state.to(device,     non_blocking=True)
        act_chunk = act_chunk.to(device, non_blocking=True)
        task_ids  = task_ids.to(device,  non_blocking=True)

        use_amp = (device == "cuda")
        with torch.amp.autocast(device_type=device, enabled=use_amp):
            pred, lb = model(img_top, img_wrist, state, task_ids)
            loss, al, ll = compute_loss(pred, act_chunk, lb, cfg)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        total_loss += loss.item()
        act_sum    += al
        lb_sum     += ll
        n          += 1

        if (bi + 1) % cfg.log_every == 0:
            print(f"    Ep {epoch:03d} [{bi+1:4d}/{len(loader)}]  "
                  f"loss={total_loss/n:.4f}  "
                  f"act={act_sum/n:.4f}  "
                  f"lb={lb_sum/n:.5f}")

    return {
        "train_loss": total_loss / max(n, 1),
        "train_act":  act_sum    / max(n, 1),
        "train_lb":   lb_sum     / max(n, 1),
    }


@torch.no_grad()
def val_epoch(
    model:  VLAModel,
    loader: DataLoader,
    cfg:    TrainConfig,
) -> Dict[str, float]:

    model.eval()
    device = cfg.device
    total_loss = act_sum = 0.0
    n = 0

    for img_top, img_wrist, state, act_chunk, task_ids in loader:
        img_top   = img_top.to(device,   non_blocking=True)
        img_wrist = img_wrist.to(device, non_blocking=True)
        state     = state.to(device,     non_blocking=True)
        act_chunk = act_chunk.to(device, non_blocking=True)
        task_ids  = task_ids.to(device,  non_blocking=True)

        pred, lb = model(img_top, img_wrist, state, task_ids)
        loss, al, _ = compute_loss(pred, act_chunk, lb, cfg)

        total_loss += loss.item()
        act_sum    += al
        n          += 1

    return {
        "val_loss": total_loss / max(n, 1),
        "val_act":  act_sum    / max(n, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# § 13  Checkpoint + CSV Logger
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model:     VLAModel,
    optimizer: torch.optim.Optimizer,
    epoch:     int,
    metrics:   Dict,
    cfg:       TrainConfig,
    tag:       str = "latest",
) -> None:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics":   metrics,
        "config":    asdict(cfg),
    }, out / f"checkpoint_{tag}.pt")


def load_checkpoint(
    path:      Path,
    model:     VLAModel,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[int, Dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    if optimizer and "optimizer" in ck:
        optimizer.load_state_dict(ck["optimizer"])
    print(f"  Resumed from epoch {ck.get('epoch', 0)}")
    return ck.get("epoch", 0), ck.get("metrics", {})


class CsvLogger:
    """Append rows to a CSV file, creating header on first write."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file   = open(path, "a", newline="")
        self._writer = None
        self._has_header = (path.stat().st_size > 0) if path.exists() else False

    def log(self, row: Dict) -> None:
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()))
            if not self._has_header:
                self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


# ─────────────────────────────────────────────────────────────────────────────
# § 14  Training orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: TrainConfig, stats: Dict, tokenizer: SimpleTokenizer) -> VLAModel:
    # ── reproducibility ───────────────────────────────────────────────────
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.device == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    device = cfg.device

    print(f"\n{'='*70}")
    print(f"  VLA Training — SO100 Screw-Lid")
    print(f"  Device   : {device}")
    print(f"  Backbone : {cfg.vision_backbone}")
    print(f"  Experts  : {cfg.n_experts}  (top-{cfg.top_k})")
    print(f"  ChunkSize: {cfg.chunk_size}  |  ActionDim: {ACTION_DIM}")
    print(f"  Epochs   : {cfg.epochs}  |  BS: {cfg.batch_size}  |  LR: {cfg.lr}")
    print(f"{'='*70}\n")

    # ── build model ───────────────────────────────────────────────────────
    model = VLAModel(cfg, tokenizer).to(device)
    print(f"  Trainable parameters: {model.n_trainable:,}\n")

    # ── optimiser + LR schedule ───────────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr,
                                   weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.05
    )
    scaler = torch.amp.GradScaler() if device == "cuda" else None

    # ── data ──────────────────────────────────────────────────────────────
    tr_dl, val_dl = make_dataloaders(cfg, stats, tokenizer)

    # ── logging ───────────────────────────────────────────────────────────
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger  = CsvLogger(out_dir / "train_log.csv")

    # ── optional resume ───────────────────────────────────────────────────
    latest_ck = out_dir / "checkpoint_latest.pt"
    start_ep  = 0
    best_val  = float("inf")
    if latest_ck.exists():
        start_ep, prev_metrics = load_checkpoint(latest_ck, model, optimizer)
        best_val = prev_metrics.get("val_loss", float("inf"))

    # ── save cfg snapshot ─────────────────────────────────────────────────
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # ── training loop ─────────────────────────────────────────────────────
    t0 = time.time()
    for epoch in range(start_ep + 1, cfg.epochs + 1):
        ep_t = time.time()

        tr_met  = train_epoch(model, tr_dl, optimizer, scaler, cfg, epoch)
        val_met = val_epoch(model, val_dl, cfg)
        scheduler.step()

        elapsed = time.time() - ep_t
        lr_now  = scheduler.get_last_lr()[0]

        row = {
            "epoch":      epoch,
            "lr":         round(lr_now, 8),
            "time_s":     round(elapsed, 2),
            **tr_met,
            **val_met,
        }
        logger.log(row)

        is_best = val_met["val_loss"] < best_val
        if is_best:
            best_val = val_met["val_loss"]
            save_checkpoint(model, optimizer, epoch, row, cfg, tag="best")

        save_checkpoint(model, optimizer, epoch, row, cfg, tag="latest")
        if epoch % cfg.save_every == 0:
            save_checkpoint(model, optimizer, epoch, row, cfg,
                            tag=f"epoch_{epoch:04d}")

        star = " ★" if is_best else ""
        print(f"  Ep {epoch:3d}/{cfg.epochs}  "
              f"train={tr_met['train_loss']:.4f}  "
              f"val={val_met['val_loss']:.4f}{star}  "
              f"act={val_met['val_act']:.4f}  "
              f"lr={lr_now:.2e}  "
              f"t={elapsed:.1f}s")

    total = time.time() - t0
    logger.close()

    print(f"\n  Training complete — {total/60:.1f} min")
    print(f"  Best val loss : {best_val:.4f}")
    print(f"  Checkpoints   : {out_dir}/")
    print(f"  Log           : {out_dir}/train_log.csv\n")

    # ── final summary table ───────────────────────────────────────────────
    _print_expert_routing(model, val_dl, cfg, device)

    return model


# ─────────────────────────────────────────────────────────────────────────────
# § 15  Post-training: expert routing summary
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _print_expert_routing(
    model:  VLAModel,
    loader: DataLoader,
    cfg:    TrainConfig,
    device: str,
    n_batches: int = 20,
) -> None:
    """Print how often each MoE expert was selected in the first N val batches."""
    model.eval()
    counts = torch.zeros(cfg.n_experts, device=device)

    for i, (img_top, img_wrist, state, _, task_ids) in enumerate(loader):
        if i >= n_batches:
            break
        img_top   = img_top.to(device)
        img_wrist = img_wrist.to(device)
        state     = state.to(device)
        task_ids  = task_ids.to(device)

        vis  = model.vision_enc(img_top, img_wrist)
        lang = model.lang_enc(task_ids)
        st   = model.state_enc(state)
        x    = model.moe.fusion(torch.cat([vis, lang, st], dim=-1))
        logits = model.moe.gate(x)
        _, idx = logits.topk(cfg.top_k, dim=-1)
        for k in range(cfg.top_k):
            counts.scatter_add_(0, idx[:, k],
                                torch.ones(idx.shape[0], device=device))

    total = counts.sum().item()
    print("  ── Expert routing (val set sample) ───────────────────────────")
    for e in range(cfg.n_experts):
        pct = 100 * counts[e].item() / max(total, 1)
        bar = "█" * int(pct / 3) + "░" * (34 - int(pct / 3))
        print(f"  Expert {e}: {bar}  {pct:.1f}%")
    print("  ─────────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# § 16  Inference helper
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer(
    model:     VLAModel,
    img_top:   torch.Tensor,    # [3, 224, 224] normalised
    img_wrist: torch.Tensor,    # [3, 224, 224] normalised
    state:     torch.Tensor,    # [6] normalised
    tokenizer: SimpleTokenizer,
    stats:     Dict,
    device:    str = "cpu",
) -> np.ndarray:
    """
    Single-step inference.  Returns action chunk [chunk_size, 6] in
    original (unnormalised) joint-space.
    """
    model.eval()
    task_ids = tokenizer.encode(TASK_TEXT).unsqueeze(0).to(device)

    pred_n, _ = model(
        img_top.unsqueeze(0).to(device),
        img_wrist.unsqueeze(0).to(device),
        state.unsqueeze(0).to(device),
        task_ids,
    )
    chunk_n = pred_n.squeeze(0).cpu().numpy()     # [C, 6]

    act_mean = np.asarray(stats["action_mean"], np.float32)
    act_std  = np.asarray(stats["action_std"],  np.float32)
    return chunk_n * act_std + act_mean            # unnormalise


# ─────────────────────────────────────────────────────────────────────────────
# § 17  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="VLA Training Pipeline — SO100 Screw-Lid",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode flags
    ap.add_argument("--prepare",    action="store_true",
                    help="Download dataset + extract video frames (run once)")
    ap.add_argument("--train",      action="store_true",
                    help="Train the VLA model")

    # Paths
    ap.add_argument("--data_dir",   default="./data/so100_screw_lid_v0")
    ap.add_argument("--cache_dir",  default="./cache/so100_frames")
    ap.add_argument("--output_dir", default="./runs/vla_so100")

    # Architecture
    ap.add_argument("--vision",     default="dinov2",
                    choices=["dinov2", "resnet18"],
                    help="Vision backbone")
    ap.add_argument("--n_experts",  type=int, default=4,
                    help="Number of MoE experts")
    ap.add_argument("--top_k",      type=int, default=2,
                    help="MoE top-K routing")
    ap.add_argument("--chunk_size", type=int, default=16,
                    help="Action prediction horizon (timesteps)")
    ap.add_argument("--d_model",    type=int, default=256,
                    help="Internal model dimension")

    # Training
    ap.add_argument("--epochs",     type=int,   default=100)
    ap.add_argument("--batch_size", type=int,   default=16)
    ap.add_argument("--lr",         type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_frac",   type=float, default=0.10)
    ap.add_argument("--num_workers",type=int,   default=2)
    ap.add_argument("--device",     default=None,
                    help="cuda / cpu (auto-detected if omitted)")

    args = ap.parse_args()

    if not args.prepare and not args.train:
        ap.print_help()
        print("\n  Quick start:")
        print("    python vla_train_so100.py --prepare          # first run")
        print("    python vla_train_so100.py --train            # training")
        print("    python vla_train_so100.py --prepare --train  # both\n")
        return

    # ── build config ──────────────────────────────────────────────────────
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig(
        data_dir         = args.data_dir,
        cache_dir        = args.cache_dir,
        output_dir       = args.output_dir,
        vision_backbone  = args.vision,
        n_experts        = args.n_experts,
        top_k            = args.top_k,
        chunk_size       = args.chunk_size,
        d_model          = args.d_model,
        epochs           = args.epochs,
        batch_size       = args.batch_size,
        lr               = args.lr,
        weight_decay     = args.weight_decay,
        val_frac         = args.val_frac,
        num_workers      = args.num_workers,
        device           = device,
    )

    tokenizer = SimpleTokenizer()
    print(f"  Task vocab size: {tokenizer.vocab_size}")

    # ── Phase 1: prepare ──────────────────────────────────────────────────
    if args.prepare:
        print("\n[Phase 1 — Data Preparation]")
        stats = prepare(cfg)
    else:
        stats_path = Path(cfg.cache_dir) / "stats.json"
        if not stats_path.exists():
            print(f"\n  ERROR: {stats_path} not found.")
            print("  Run  python vla_train_so100.py --prepare  first.")
            sys.exit(1)
        with open(stats_path) as f:
            stats = json.load(f)
        print(f"  Loaded stats from {stats_path}  "
              f"({stats.get('total_frames', '?')} frames)")

    # ── Phase 2: train ────────────────────────────────────────────────────
    if args.train:
        print("\n[Phase 2 — Training]")
        model = train(cfg, stats, tokenizer)

        # Quick inference smoke-test
        print("  Smoke-test: running one inference step …")
        dummy_img   = torch.zeros(3, IMG_SIZE, IMG_SIZE)
        dummy_state = torch.zeros(STATE_DIM)
        chunk = infer(model, dummy_img, dummy_img, dummy_state,
                      tokenizer, stats, device=device)
        print(f"  Output chunk shape: {chunk.shape}  "
              f"(first action: {chunk[0].round(4)})")
        print("  ✓ Done.\n")


if __name__ == "__main__":
    main()