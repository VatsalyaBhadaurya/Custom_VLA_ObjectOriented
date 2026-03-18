"""
customvla/arms/base.py

Abstract base class for robot arm definitions.
All arms registered in the package must inherit from BaseArm.

To define a new arm:

    from customvla.arms.base import BaseArm
    from customvla.arms.registry import register_arm

    @register_arm("my_arm")
    class MyArm(BaseArm):
        ACTION_DIM = 7
        STATE_DIM  = 7
        JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]
        JOINT_LIMITS = [(-3.14, 3.14)] * 6 + [(0.0, 1.0)]
        CAMERAS = ["front"]
        TASK_TEXT = "Default task description for this arm."

        def send_action(self, action_vector):
            # send to ROS / serial / MoveIt / sim
            ...

        def get_state(self):
            # return current joint positions as list[float]
            ...
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict
import numpy as np


class BaseArm(ABC):
    """
    Abstract definition of a robot arm for use with CustomVLA.

    Class-level attributes (override in subclass):
        ACTION_DIM   (int)  : Number of action dimensions (e.g. 6 for SO100, 7 for Franka)
        STATE_DIM    (int)  : Number of state dimensions (usually same as ACTION_DIM)
        JOINT_NAMES  (list) : Human-readable joint names (len == ACTION_DIM)
        JOINT_LIMITS (list) : List of (min, max) tuples per joint — used by RL validator
        CAMERAS      (list) : Camera keys expected by the dataset / live pipeline
        TASK_TEXT    (str)  : Default task description fed to the language encoder
    """

    # ── Must be overridden ────────────────────────────────────────────────
    ACTION_DIM:   int = NotImplemented
    STATE_DIM:    int = NotImplemented
    JOINT_NAMES:  List[str] = NotImplemented
    JOINT_LIMITS: List[Tuple[float, float]] = NotImplemented
    CAMERAS:      List[str] = NotImplemented
    TASK_TEXT:    str = "Perform the task."

    # ── Optional overrides ────────────────────────────────────────────────
    FPS: int = 30
    IMG_SIZE: int = 224

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        required = ["ACTION_DIM", "STATE_DIM", "JOINT_NAMES", "JOINT_LIMITS", "CAMERAS"]
        for attr in required:
            if getattr(cls, attr, NotImplemented) is NotImplemented:
                raise TypeError(
                    f"Class '{cls.__name__}' must define class attribute '{attr}'."
                )
        if len(cls.JOINT_NAMES) != cls.ACTION_DIM:
            raise ValueError(
                f"'{cls.__name__}': len(JOINT_NAMES)={len(cls.JOINT_NAMES)} "
                f"!= ACTION_DIM={cls.ACTION_DIM}"
            )
        if len(cls.JOINT_LIMITS) != cls.ACTION_DIM:
            raise ValueError(
                f"'{cls.__name__}': len(JOINT_LIMITS)={len(cls.JOINT_LIMITS)} "
                f"!= ACTION_DIM={cls.ACTION_DIM}"
            )

    # ── Abstract interface ────────────────────────────────────────────────

    @abstractmethod
    def send_action(self, action_vector: np.ndarray) -> None:
        """
        Send a joint action to the physical / simulated robot.

        Args:
            action_vector: np.ndarray of shape (ACTION_DIM,)
                           Values are in real (un-normalised) joint space.
        """

    @abstractmethod
    def get_state(self) -> np.ndarray:
        """
        Read current joint state from the robot.

        Returns:
            np.ndarray of shape (STATE_DIM,) in real joint space.
        """

    # ── Concrete helpers (may override) ──────────────────────────────────

    def clip_to_limits(self, action: np.ndarray) -> np.ndarray:
        """Hard-clip action to JOINT_LIMITS before sending."""
        lo = np.array([lim[0] for lim in self.JOINT_LIMITS], dtype=np.float32)
        hi = np.array([lim[1] for lim in self.JOINT_LIMITS], dtype=np.float32)
        return np.clip(action, lo, hi)

    def validate_action(self, action: np.ndarray) -> Dict[str, object]:
        """
        Check whether every joint is within limits.

        Returns a dict with:
            valid   (bool)
            violations (list of joint names that are out of range)
            clipped (np.ndarray)  — safe version
        """
        clipped = self.clip_to_limits(action)
        violations = [
            self.JOINT_NAMES[i]
            for i in range(self.ACTION_DIM)
            if not (self.JOINT_LIMITS[i][0] <= action[i] <= self.JOINT_LIMITS[i][1])
        ]
        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "clipped": clipped,
        }

    def get_limit_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lo, hi) arrays for RL validator."""
        lo = np.array([lim[0] for lim in self.JOINT_LIMITS], dtype=np.float32)
        hi = np.array([lim[1] for lim in self.JOINT_LIMITS], dtype=np.float32)
        return lo, hi

    def describe(self) -> str:
        lines = [
            f"Arm       : {self.__class__.__name__}",
            f"DOF       : {self.ACTION_DIM}",
            f"Cameras   : {self.CAMERAS}",
            f"Task      : {self.TASK_TEXT[:80]}{'...' if len(self.TASK_TEXT) > 80 else ''}",
            "Joints:",
        ]
        for name, (lo, hi) in zip(self.JOINT_NAMES, self.JOINT_LIMITS):
            lines.append(f"  {name:<20} [{lo:+.3f}, {hi:+.3f}]")
        return "\n".join(lines)
