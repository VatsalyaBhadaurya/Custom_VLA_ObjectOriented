"""
tests/test_smoke.py

Smoke tests — no dataset required, runs fully on CPU with random tensors.

Run with:
    python tests/test_smoke.py
    # or
    python -m pytest tests/test_smoke.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Arm registry
# ─────────────────────────────────────────────────────────────────────────────

def test_arm_registry():
    from customvla.arms import get_arm, list_arms, builtin
    for name in ["so100", "franka", "ur5", "generic7dof"]:
        arm = get_arm(name)
        assert arm.ACTION_DIM == arm.STATE_DIM
        assert len(arm.JOINT_NAMES) == arm.ACTION_DIM
        assert len(arm.JOINT_LIMITS) == arm.ACTION_DIM
        assert len(arm.CAMERAS) >= 1
        lo, hi = arm.get_limit_arrays()
        assert (hi > lo).all()
        print(f"  ✓ arm '{name}' — DOF={arm.ACTION_DIM} cameras={arm.CAMERAS}")


def test_arm_validate():
    from customvla.arms import get_arm
    arm = get_arm("so100")
    lo, hi = arm.get_limit_arrays()
    good_action = (lo + hi) / 2
    result = arm.validate_action(good_action)
    assert result["valid"], f"Mid-range action should be valid: {result}"
    bad_action = hi + 1.0
    result2 = arm.validate_action(bad_action)
    assert not result2["valid"]
    print("  ✓ arm validate_action / clip_to_limits")


# ─────────────────────────────────────────────────────────────────────────────
# Encoders
# ─────────────────────────────────────────────────────────────────────────────

def test_state_encoder():
    from customvla.models.encoders import StateEncoder
    enc = StateEncoder(state_dim=6, d_out=128)
    x   = torch.randn(4, 6)
    out = enc(x)
    assert out.shape == (4, 128)
    print("  ✓ StateEncoder")


def test_language_encoder():
    from customvla.models.encoders import LanguageEncoder, SimpleTokenizer
    tok = SimpleTokenizer(texts=["pick up the cup from the table"])
    enc = LanguageEncoder(tok.vocab_size, d_out=128)
    ids = torch.stack([tok.encode("pick up the cup")] * 4)
    out = enc(ids)
    assert out.shape == (4, 128)
    print("  ✓ LanguageEncoder")


def test_vision_encoder_resnet():
    from customvla.models.encoders import VisionEncoder
    enc = VisionEncoder(d_out=256, backbone="resnet18", n_cameras=2)
    img = torch.randn(2, 3, 224, 224)
    out = enc(img, img)
    assert out.shape == (2, 256)
    print("  ✓ VisionEncoder (resnet18, 2 cameras)")


def test_vision_encoder_single_cam():
    from customvla.models.encoders import VisionEncoder
    enc = VisionEncoder(d_out=128, backbone="resnet18", n_cameras=1)
    img = torch.randn(3, 3, 224, 224)
    out = enc(img)
    assert out.shape == (3, 128)
    print("  ✓ VisionEncoder (resnet18, 1 camera)")


# ─────────────────────────────────────────────────────────────────────────────
# FusionMoE
# ─────────────────────────────────────────────────────────────────────────────

def test_fusion_moe():
    from customvla.models.fusion import FusionMoE
    moe = FusionMoE(d_vis=256, d_lang=128, d_state=128, d_model=256,
                    n_experts=4, top_k=2)
    vis   = torch.randn(8, 256)
    lang  = torch.randn(8, 128)
    state = torch.randn(8, 128)
    out, lb = moe(vis, lang, state)
    assert out.shape == (8, 256)
    assert lb.ndim == 0
    util = moe.expert_utilization(vis, lang, state)
    assert abs(sum(util.values()) - 1.0) < 1e-3, f"Routing fracs don't sum to 1: {util}"
    print(f"  ✓ FusionMoE — lb_loss={lb.item():.4f}  routing={util}")


# ─────────────────────────────────────────────────────────────────────────────
# Full VLA model
# ─────────────────────────────────────────────────────────────────────────────

def test_vla_model_so100():
    from customvla.arms import get_arm
    from customvla.models.vla import VLAModel, TrainConfig, compute_loss
    from customvla.models.encoders import SimpleTokenizer
    arm = get_arm("so100")
    cfg = TrainConfig.from_arm(arm, vision_backbone="resnet18",
                               epochs=1, batch_size=2)
    tok   = SimpleTokenizer(texts=[cfg.task_text])
    model = VLAModel(cfg, tok)

    B = 2
    img  = torch.randn(B, 3, 224, 224)
    st   = torch.randn(B, cfg.state_dim)
    ids  = torch.stack([tok.encode(cfg.task_text)] * B)
    pred, lb = model(img, img, state=st, task_ids=ids)
    assert pred.shape == (B, cfg.chunk_size, cfg.action_dim)

    gt   = torch.randn_like(pred)
    loss, info = compute_loss(pred, gt, lb, cfg)
    assert loss.item() > 0
    print(f"  ✓ VLAModel SO100 — pred={pred.shape}  loss={loss.item():.4f}  "
          f"params={model.n_trainable:,}")


def test_vla_model_franka():
    from customvla.arms import get_arm
    from customvla.models.vla import VLAModel, TrainConfig
    from customvla.models.encoders import SimpleTokenizer
    arm = get_arm("franka")
    cfg = TrainConfig.from_arm(arm, vision_backbone="resnet18",
                               n_cameras=2, chunk_size=8)
    tok   = SimpleTokenizer(texts=[cfg.task_text])
    model = VLAModel(cfg, tok)
    B = 3
    img  = torch.randn(B, 3, 224, 224)
    st   = torch.randn(B, cfg.state_dim)
    ids  = torch.stack([tok.encode(cfg.task_text)] * B)
    pred, lb = model(img, img, state=st, task_ids=ids)
    assert pred.shape == (B, cfg.chunk_size, cfg.action_dim)
    print(f"  ✓ VLAModel Franka — pred={pred.shape}  params={model.n_trainable:,}")


# ─────────────────────────────────────────────────────────────────────────────
# RL Safety Validator
# ─────────────────────────────────────────────────────────────────────────────

def test_hard_limit_checker():
    from customvla.arms import get_arm
    from customvla.rl.safety import HardLimitChecker
    arm     = get_arm("so100")
    checker = HardLimitChecker(arm)
    lo, hi  = arm.get_limit_arrays()
    safe_actions = torch.tensor((lo + hi) / 2).unsqueeze(0).unsqueeze(0)
    info = checker.check_batch(safe_actions)
    assert info["fraction_safe"] == 1.0
    assert info["penalty"].item() >= 0

    bad_actions = torch.tensor(hi + 1.0).unsqueeze(0).unsqueeze(0)
    info2 = checker.check_batch(bad_actions)
    assert info2["penalty"].item() > 0
    print(f"  ✓ HardLimitChecker — safe_pen={info['penalty'].item():.4f}  "
          f"bad_pen={info2['penalty'].item():.4f}")


def test_rl_safety_validator():
    from customvla.arms import get_arm
    from customvla.rl.safety import RLSafetyValidator
    arm = get_arm("so100")
    val = RLSafetyValidator(arm)

    # compute_penalty
    B, C, D = 4, 16, arm.ACTION_DIM
    pred_chunk = torch.randn(B, C, D)
    state      = torch.randn(B, arm.STATE_DIM)
    am = torch.zeros(D); as_ = torch.ones(D)
    sm = torch.zeros(arm.STATE_DIM); ss = torch.ones(arm.STATE_DIM)
    penalty = val.compute_penalty(pred_chunk, state, am, as_, sm, ss)
    assert penalty.ndim == 0

    # validate_and_clip
    lo, hi = arm.get_limit_arrays()
    bad_action = hi + 0.5
    clipped, info = val.validate_and_clip(bad_action)
    assert not info["was_safe"]
    assert np.all(clipped <= hi + 1e-6)

    # score_action
    score = val.score_action(np.zeros(arm.STATE_DIM), np.zeros(arm.ACTION_DIM))
    assert isinstance(score, float)
    print(f"  ✓ RLSafetyValidator — penalty={penalty.item():.4f}  score={score:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Inference API
# ─────────────────────────────────────────────────────────────────────────────

def test_inference_api():
    """Test VLAInference without a real checkpoint (build model manually)."""
    import json, tempfile
    from customvla.arms import get_arm
    from customvla.models.vla import VLAModel, TrainConfig
    from customvla.models.encoders import SimpleTokenizer
    from customvla.rl.safety import RLSafetyValidator
    from customvla.inference import VLAInference

    arm = get_arm("so100")
    cfg = TrainConfig.from_arm(arm, vision_backbone="resnet18")
    tok = SimpleTokenizer(texts=[cfg.task_text])
    model = VLAModel(cfg, tok)
    validator = RLSafetyValidator(arm)

    stats = {
        "action_mean": [0.0] * cfg.action_dim,
        "action_std":  [1.0] * cfg.action_dim,
        "state_mean":  [0.0] * cfg.state_dim,
        "state_std":   [1.0] * cfg.state_dim,
    }

    runner = VLAInference(
        model=model, cfg=cfg, tokenizer=tok,
        stats=stats, arm=arm, validator=validator,
    )

    images = {cam: np.zeros((224, 224, 3), dtype=np.uint8) for cam in cfg.cameras}
    state  = np.zeros(cfg.state_dim, dtype=np.float32)
    action, chunk = runner.predict(images, state, return_chunk=True)
    assert action.shape == (cfg.action_dim,)
    assert chunk.shape  == (cfg.chunk_size, cfg.action_dim)

    # Buffered
    action2 = runner.predict_buffered(images, state)
    assert action2.shape == (cfg.action_dim,)

    print(f"  ✓ VLAInference.predict — action={action.shape}  chunk={chunk.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# TrainConfig serialization
# ─────────────────────────────────────────────────────────────────────────────

def test_config_serialization():
    import tempfile, json
    from customvla.arms import get_arm
    from customvla.models.vla import TrainConfig
    arm = get_arm("so100")
    cfg = TrainConfig.from_arm(arm, epochs=42, lr=3e-4)
    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/config.json"
        cfg.save(path)
        cfg2 = TrainConfig.load(path)
    assert cfg2.epochs == 42
    assert abs(cfg2.lr - 3e-4) < 1e-10
    assert cfg2.arm_name == "so100"
    print("  ✓ TrainConfig save / load")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_arm_registry,
        test_arm_validate,
        test_state_encoder,
        test_language_encoder,
        test_vision_encoder_resnet,
        test_vision_encoder_single_cam,
        test_fusion_moe,
        test_vla_model_so100,
        test_vla_model_franka,
        test_hard_limit_checker,
        test_rl_safety_validator,
        test_inference_api,
        test_config_serialization,
    ]
    print(f"\n  CustomVLA Smoke Tests ({len(tests)} tests)\n")
    passed = 0
    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ {fn.__name__}: {e}")
            import traceback; traceback.print_exc()
    print(f"\n  Passed: {passed}/{len(tests)}\n")
