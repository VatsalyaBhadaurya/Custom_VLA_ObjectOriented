"""
CustomVLA — A modular Vision-Language-Action package for robotic manipulation.

Architecture:
  Camera(s) → DINOv2 Vision Encoder → ]
  Language Command → BiGRU Encoder   → FusionMoE → ActionHead (MLP) → Joint Actions
  Robot State → State MLP Encoder    → ]
                                             ↑
                                    RL Safety Validator (PPO critic)

Inspired by GR-1, Groot-1.5, and π0. Designed to be arm-agnostic.
"""

__version__ = "0.1.0"
__author__ = "VatsalyaBhadaurya"

from customvla.models.vla import VLAModel
from customvla.models.encoders import VisionEncoder, LanguageEncoder, StateEncoder
from customvla.models.fusion import FusionMoE
from customvla.arms.base import BaseArm
from customvla.arms.registry import ArmRegistry, register_arm, get_arm
from customvla.rl.safety import RLSafetyValidator
from customvla.inference import VLAInference

__all__ = [
    "VLAModel",
    "VisionEncoder",
    "LanguageEncoder",
    "StateEncoder",
    "FusionMoE",
    "BaseArm",
    "ArmRegistry",
    "register_arm",
    "get_arm",
    "RLSafetyValidator",
    "VLAInference",
]
