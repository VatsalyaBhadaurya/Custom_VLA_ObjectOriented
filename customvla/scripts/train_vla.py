"""
scripts/train_vla.py

Training script for CustomVLA — arm-agnostic, with RL safety validator.

Usage
─────
# Prepare data (download + extract frames + compute stats):
    python scripts/train_vla.py --arm so100 --prepare

# Train:
    python scripts/train_vla.py --arm so100 --train

# Both:
    python scripts/train_vla.py --arm so100 --prepare --train

# Custom arm with config overrides:
    python scripts/train_vla.py --arm franka --train \\
        --epochs 200 --batch_size 8 --lr 5e-5 \\
        --data_dir /mnt/data/franka_demos \\
        --repo_id YourOrg/franka_demos_v1

# List available arms:
    python scripts/train_vla.py --list_arms
"""

import os, sys, json, csv, random, time, argparse
from pathlib import Path
from dataclasses import asdict
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

import torchvision.transforms as T

# ── make customvla importable from project root ───────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from customvla.arms.registry import list_arms, get_arm
from customvla.arms import builtin  # trigger registration
from customvla.arms.base import BaseArm
from customvla.models.vla import VLAModel, TrainConfig, compute_loss
from customvla.models.encoders import SimpleTokenizer
from customvla.rl.safety import RLSafetyValidator

try:
    import av
    AV_OK = True
except ImportError:
    AV_OK = False
    print("WARNING: PyAV not found. Install: pip install av")

try:
    from huggingface_hub import snapshot_download
    HF_OK = True
except ImportError:
    HF_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Image transform
# ─────────────────────────────────────────────────────────────────────────────

_IMG_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def download_dataset(cfg: TrainConfig, repo_id: str) -> None:
    sentinel = Path(cfg.data_dir) / "meta" / "info.json"
    if sentinel.exists():
        print(f"  Dataset already at {cfg.data_dir} — skipping download.")
        return
    if not HF_OK:
        raise ImportError("huggingface_hub not installed. pip install huggingface_hub")
    print(f"  Downloading {repo_id} → {cfg.data_dir} …")
    snapshot_download(repo_id=repo_id, repo_type="dataset",
                      local_dir=cfg.data_dir,
                      ignore_patterns=["*.git*", "*.gitattributes"])
    print("  Download complete.")


def extract_frames(cfg: TrainConfig) -> None:
    if not AV_OK:
        raise RuntimeError("PyAV required for frame extraction. pip install av")
    data_path  = Path(cfg.data_dir)
    cache_path = Path(cfg.cache_dir)
    pq_dir     = data_path / "data" / "chunk-000"
    episodes   = sorted(pq_dir.glob("episode_*.parquet"))
    print(f"  Extracting frames for {len(episodes)} episodes …")
    resize = T.Resize((cfg.img_size, cfg.img_size),
                      interpolation=T.InterpolationMode.BILINEAR, antialias=True)
    for pf in tqdm(episodes, desc="Episodes"):
        ep_str = pf.stem
        ep_dir = cache_path / ep_str
        if (ep_dir / ".done").exists():
            continue
        for cam in cfg.cameras:
            vid = (data_path / "videos" / "chunk-000"
                   / f"observation.images.{cam}" / f"{ep_str}.mp4")
            if not vid.exists():
                continue
            out_dir = ep_dir / cam
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                container = av.open(str(vid))
                stream    = container.streams.video[0]
                stream.thread_type = "AUTO"
                for n, frame in enumerate(container.decode(stream)):
                    dest = out_dir / f"{n:06d}.jpg"
                    if not dest.exists():
                        resize(frame.to_image()).save(dest, quality=92)
                container.close()
            except Exception as exc:
                print(f"  WARNING: {vid}: {exc}")
        ep_dir.mkdir(parents=True, exist_ok=True)
        (ep_dir / ".done").touch()
    print("  Frame extraction complete.")


def _read_col(df: pd.DataFrame, col: str) -> np.ndarray:
    raw = df[col].tolist()
    return np.stack([np.asarray(r, dtype=np.float32) for r in raw])


def compute_stats(cfg: TrainConfig) -> dict:
    stats_path = Path(cfg.cache_dir) / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            return json.load(f)
    print("  Computing normalisation statistics …")
    pq_dir = Path(cfg.data_dir) / "data" / "chunk-000"
    all_acts, all_sts = [], []
    for pf in sorted(pq_dir.glob("episode_*.parquet")):
        df = pd.read_parquet(pf, columns=["action", "observation.state"])
        all_acts.append(_read_col(df, "action"))
        all_sts.append(_read_col(df, "observation.state"))
    acts = np.concatenate(all_acts, axis=0)
    sts  = np.concatenate(all_sts,  axis=0)
    stats = {
        "action_mean": acts.mean(axis=0).tolist(),
        "action_std":  np.maximum(acts.std(axis=0), 1e-6).tolist(),
        "state_mean":  sts.mean(axis=0).tolist(),
        "state_std":   np.maximum(sts.std(axis=0), 1e-6).tolist(),
        "total_frames": int(len(acts)),
    }
    Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats over {stats['total_frames']:,} frames → {stats_path}")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class VLADataset(Dataset):
    def __init__(self, cfg: TrainConfig, episode_indices, stats, tokenizer):
        self.cfg        = cfg
        self.cache_path = Path(cfg.cache_dir)
        self.data_path  = Path(cfg.data_dir)
        self.tokenizer  = tokenizer
        self._act_mean  = np.array(stats["action_mean"], np.float32)
        self._act_std   = np.array(stats["action_std"],  np.float32)
        self._st_mean   = np.array(stats["state_mean"],  np.float32)
        self._st_std    = np.array(stats["state_std"],   np.float32)
        self._task_ids  = tokenizer.encode(cfg.task_text)
        self._samples   = []
        self._build_index(episode_indices)

    def _build_index(self, episode_indices):
        pq_dir  = self.data_path / "data" / "chunk-000"
        skipped = 0
        for ep_idx in sorted(episode_indices):
            ep_str = f"episode_{ep_idx:06d}"
            pf     = pq_dir / f"{ep_str}.parquet"
            if not pf.exists():
                skipped += 1; continue
            if not (self.cache_path / ep_str / ".done").exists():
                skipped += 1; continue
            df   = pd.read_parquet(pf, columns=["frame_index", "action", "observation.state"])
            N    = len(df)
            fis  = df["frame_index"].values.astype(int)
            acts = _read_col(df, "action")
            sts  = _read_col(df, "observation.state")
            for i in range(N):
                end   = min(i + self.cfg.chunk_size, N)
                chunk = acts[i:end].copy()
                if len(chunk) < self.cfg.chunk_size:
                    pad   = np.tile(acts[-1:], (self.cfg.chunk_size - len(chunk), 1))
                    chunk = np.concatenate([chunk, pad], axis=0)
                self._samples.append((ep_str, int(fis[i]), sts[i], chunk))
        if skipped:
            print(f"  WARNING: {skipped} episodes skipped")

    def __len__(self): return len(self._samples)

    def __getitem__(self, idx):
        ep_str, fi, state, chunk = self._samples[idx]
        imgs = []
        for cam in self.cfg.cameras:
            imgs.append(self._load_jpg(ep_str, cam, fi))
        state_n = (state - self._st_mean) / self._st_std
        chunk_n = (chunk - self._act_mean) / self._act_std
        return (
            *imgs,
            torch.from_numpy(state_n.copy()),
            torch.from_numpy(chunk_n.copy()),
            self._task_ids.clone(),
        )

    def _load_jpg(self, ep_str, cam, fi):
        path = self.cache_path / ep_str / cam / f"{fi:06d}.jpg"
        try:
            return _IMG_TRANSFORM(Image.open(path).convert("RGB"))
        except Exception:
            return torch.zeros(3, self.cfg.img_size, self.cfg.img_size)


# ─────────────────────────────────────────────────────────────────────────────
# DataLoaders
# ─────────────────────────────────────────────────────────────────────────────

def make_dataloaders(cfg: TrainConfig, stats: dict, tokenizer: SimpleTokenizer):
    pq_dir  = Path(cfg.data_dir) / "data" / "chunk-000"
    all_eps = sorted(int(f.stem.split("_")[1]) for f in pq_dir.glob("episode_*.parquet"))
    random.seed(cfg.seed)
    random.shuffle(all_eps)
    n_val   = max(1, int(len(all_eps) * cfg.val_frac))
    val_eps = all_eps[:n_val]
    tr_eps  = all_eps[n_val:]
    print(f"  Episodes — train: {len(tr_eps)}, val: {len(val_eps)}")
    tr_ds  = VLADataset(cfg, tr_eps,  stats, tokenizer)
    val_ds = VLADataset(cfg, val_eps, stats, tokenizer)
    print(f"  Samples  — train: {len(tr_ds):,}, val: {len(val_ds):,}")
    kw = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers,
              pin_memory=(cfg.device == "cuda"),
              persistent_workers=(cfg.num_workers > 0))
    return DataLoader(tr_ds, shuffle=True, **kw), DataLoader(val_ds, shuffle=False, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# Train / Val loops
# ─────────────────────────────────────────────────────────────────────────────

def _unpack_batch(batch, n_cams, device):
    """Unpack variable-length batch tuple based on number of cameras."""
    imgs     = [b.to(device, non_blocking=True) for b in batch[:n_cams]]
    state    = batch[n_cams].to(device, non_blocking=True)
    act_chunk = batch[n_cams + 1].to(device, non_blocking=True)
    task_ids  = batch[n_cams + 2].to(device, non_blocking=True)
    return imgs, state, act_chunk, task_ids


def train_epoch(model, validator, loader, optimizer, val_opt, scaler, cfg, epoch, stats):
    model.train()
    device   = cfg.device
    n_cams   = len(cfg.cameras)
    act_mean = torch.tensor(stats["action_mean"]).float().to(device)
    act_std  = torch.tensor(stats["action_std"]).float().to(device)
    st_mean  = torch.tensor(stats["state_mean"]).float().to(device)
    st_std   = torch.tensor(stats["state_std"]).float().to(device)

    totals = {"total_loss": 0, "act_loss": 0, "lb_loss": 0, "rl_penalty": 0}
    n = 0

    for bi, batch in enumerate(loader):
        imgs, state, act_chunk, task_ids = _unpack_batch(batch, n_cams, device)
        use_amp = (device == "cuda")

        with torch.amp.autocast(device_type=device, enabled=use_amp):
            pred, lb = model(*imgs, state=state, task_ids=task_ids)

            rl_penalty = None
            if validator is not None and cfg.w_rl > 0:
                rl_penalty = validator.compute_penalty(
                    pred, state,
                    act_mean, act_std, st_mean, st_std,
                )

            loss, info = compute_loss(pred, act_chunk, lb, cfg, rl_penalty)

        optimizer.zero_grad(set_to_none=True)
        if val_opt is not None:
            val_opt.zero_grad(set_to_none=True)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            if val_opt is not None:
                scaler.step(val_opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            if val_opt is not None:
                val_opt.step()

        for k in totals:
            totals[k] += info.get(k, 0)
        n += 1

        if (bi + 1) % cfg.log_every == 0:
            print(f"  Ep {epoch:03d} [{bi+1:4d}/{len(loader)}] "
                  f"loss={totals['total_loss']/n:.4f} "
                  f"act={totals['act_loss']/n:.4f} "
                  f"rl={totals['rl_penalty']/n:.5f}")

    return {f"train_{k}": v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def val_epoch(model, loader, cfg):
    model.eval()
    device, n_cams = cfg.device, len(cfg.cameras)
    total_loss = act_sum = 0.0
    n = 0
    for batch in loader:
        imgs, state, act_chunk, task_ids = _unpack_batch(batch, n_cams, device)
        pred, lb = model(*imgs, state=state, task_ids=task_ids)
        loss, info = compute_loss(pred, act_chunk, lb, cfg)
        total_loss += info["total_loss"]
        act_sum    += info["act_loss"]
        n += 1
    return {"val_loss": total_loss / max(n, 1), "val_act": act_sum / max(n, 1)}


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model, validator, optimizer, epoch, metrics, cfg, stats, tag="latest"):
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ck = {
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics":   metrics,
        "config":    asdict(cfg),
        "stats":     stats,
    }
    if validator is not None:
        ck["validator"] = validator.state_dict()
    torch.save(ck, out / f"checkpoint_{tag}.pt")


class CsvLogger:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file   = open(path, "a", newline="")
        self._writer = None
        self._has_hdr = path.exists() and path.stat().st_size > 0

    def log(self, row):
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()))
            if not self._has_hdr:
                self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()

    def close(self): self._file.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_arg_parser():
    p = argparse.ArgumentParser(description="CustomVLA Training Script")
    p.add_argument("--arm",         default="so100", help="Registered arm name")
    p.add_argument("--repo_id",     default=None,    help="HuggingFace dataset repo ID")
    p.add_argument("--data_dir",    default=None)
    p.add_argument("--cache_dir",   default=None)
    p.add_argument("--output_dir",  default=None)
    p.add_argument("--epochs",      type=int,   default=None)
    p.add_argument("--batch_size",  type=int,   default=None)
    p.add_argument("--lr",          type=float, default=None)
    p.add_argument("--chunk_size",  type=int,   default=None)
    p.add_argument("--n_experts",   type=int,   default=None)
    p.add_argument("--backbone",    default=None, choices=["dinov2", "resnet18"])
    p.add_argument("--w_rl",        type=float, default=None,
                   help="RL safety penalty weight (0 = disabled)")
    p.add_argument("--prepare",     action="store_true", help="Download + extract frames")
    p.add_argument("--train",       action="store_true", help="Run training")
    p.add_argument("--list_arms",   action="store_true", help="List registered arms")
    return p


def main():
    p    = build_arg_parser()
    args = p.parse_args()

    if args.list_arms:
        list_arms(); return

    if not args.prepare and not args.train:
        p.print_help(); return

    # ── Load arm ─────────────────────────────────────────────────────────
    arm = get_arm(args.arm)
    print(f"\n{arm.describe()}\n")

    # ── Build config from arm + overrides ─────────────────────────────────
    overrides = {}
    if args.data_dir:   overrides["data_dir"]   = args.data_dir
    if args.cache_dir:  overrides["cache_dir"]  = args.cache_dir
    if args.output_dir: overrides["output_dir"] = args.output_dir
    if args.epochs:     overrides["epochs"]     = args.epochs
    if args.batch_size: overrides["batch_size"] = args.batch_size
    if args.lr:         overrides["lr"]         = args.lr
    if args.chunk_size: overrides["chunk_size"] = args.chunk_size
    if args.n_experts:  overrides["n_experts"]  = args.n_experts
    if args.backbone:   overrides["vision_backbone"] = args.backbone
    if args.w_rl is not None: overrides["w_rl"] = args.w_rl

    # Default paths derived from arm name
    arm_tag = getattr(arm, "_arm_registry_name", arm.__class__.__name__.lower())
    overrides.setdefault("data_dir",   f"./data/{arm_tag}")
    overrides.setdefault("cache_dir",  f"./cache/{arm_tag}_frames")
    overrides.setdefault("output_dir", f"./runs/vla_{arm_tag}")

    cfg = TrainConfig.from_arm(arm, **overrides)
    repo_id = args.repo_id or f"<your-org>/{arm_tag}_demos_v0"

    # ── Prepare ───────────────────────────────────────────────────────────
    if args.prepare:
        Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)
        download_dataset(cfg, repo_id)
        extract_frames(cfg)

    stats = compute_stats(cfg)

    if not args.train:
        return

    # ── Training ──────────────────────────────────────────────────────────
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = cfg.device

    tokenizer = SimpleTokenizer(texts=[cfg.task_text])
    model     = VLAModel(cfg, tokenizer).to(device)
    print(f"\n{model.summary()}\n")

    # RL Safety Validator
    validator = None
    val_opt   = None
    if cfg.w_rl > 0:
        validator = RLSafetyValidator(arm).to(device)
        print(validator.describe())
        val_opt = torch.optim.Adam(validator.critic.parameters(), lr=cfg.lr * 0.1)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.05,
    )
    scaler = torch.amp.GradScaler() if device == "cuda" else None

    tr_dl, val_dl = make_dataloaders(cfg, stats, tokenizer)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(out_dir / "config.json")

    logger = CsvLogger(out_dir / "train_log.csv")
    best_val = float("inf")
    latest_ck = out_dir / "checkpoint_latest.pt"
    start_ep = 0
    if latest_ck.exists():
        ck = torch.load(latest_ck, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_ep  = ck.get("epoch", 0)
        best_val  = ck.get("metrics", {}).get("val_loss", float("inf"))
        if validator is not None and "validator" in ck:
            validator.load_state_dict(ck["validator"])
        print(f"  Resumed from epoch {start_ep}")

    print(f"\n{'='*65}")
    print(f"  CustomVLA Training — arm={arm_tag}")
    print(f"  Device: {device}  Backbone: {cfg.vision_backbone}")
    print(f"  Experts: {cfg.n_experts} (top-{cfg.top_k})  RL: w={cfg.w_rl}")
    print(f"  Epochs: {cfg.epochs}  BS: {cfg.batch_size}  LR: {cfg.lr}")
    print(f"{'='*65}\n")

    t0 = time.time()
    for epoch in range(start_ep + 1, cfg.epochs + 1):
        ep_t = time.time()
        tr_met  = train_epoch(model, validator, tr_dl, optimizer, val_opt, scaler, cfg, epoch, stats)
        val_met = val_epoch(model, val_dl, cfg)
        scheduler.step()

        lr_now  = scheduler.get_last_lr()[0]
        elapsed = time.time() - ep_t
        is_best = val_met["val_loss"] < best_val

        row = {"epoch": epoch, "lr": round(lr_now, 8), "time_s": round(elapsed, 2),
               **tr_met, **val_met}
        logger.log(row)

        if is_best:
            best_val = val_met["val_loss"]
            save_checkpoint(model, validator, optimizer, epoch, row, cfg, stats, tag="best")

        save_checkpoint(model, validator, optimizer, epoch, row, cfg, stats, tag="latest")
        if epoch % cfg.save_every == 0:
            save_checkpoint(model, validator, optimizer, epoch, row, cfg, stats,
                            tag=f"epoch_{epoch:04d}")

        star = " ★" if is_best else ""
        print(f"  Ep {epoch:3d}/{cfg.epochs} "
              f"train={tr_met['train_total_loss']:.4f} "
              f"val={val_met['val_loss']:.4f}{star} "
              f"act={val_met['val_act']:.4f} "
              f"lr={lr_now:.2e} t={elapsed:.1f}s")

    logger.close()
    total = time.time() - t0
    print(f"\n  Training complete — {total/60:.1f} min")
    print(f"  Best val loss : {best_val:.4f}")
    print(f"  Checkpoints   : {out_dir}/")


if __name__ == "__main__":
    main()
