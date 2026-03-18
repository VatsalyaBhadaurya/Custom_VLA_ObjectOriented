"""
customvla/arms/builtin.py

Built-in arm definitions shipped with the package.
These serve as reference implementations and can be sub-classed.

Included:
    SO100Arm     — 6-DOF SO100 tabletop arm (matches train.py dataset)
    FrankaArm    — 7-DOF Franka Emika Panda
    UR5Arm       — 6-DOF Universal Robots UR5
    Generic7DOFArm — Generic 7-DOF placeholder (no real robot connection)
"""

import numpy as np
from customvla.arms.base import BaseArm
from customvla.arms.registry import register_arm


# ─────────────────────────────────────────────────────────────────────────────
# SO100  (matches Tomas0413/so100_screw_lid_v0 — the training dataset)
# ─────────────────────────────────────────────────────────────────────────────

@register_arm("so100")
class SO100Arm(BaseArm):
    """
    6-DOF SO100 tabletop manipulator.
    Matches the architecture in train.py (ACTION_DIM=6, two cameras).
    Connect via ROS topic or direct serial.
    """
    ACTION_DIM  = 6
    STATE_DIM   = 6
    JOINT_NAMES = [
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper",
    ]
    # Approx limits in radians; gripper is 0→1 (open→closed)
    JOINT_LIMITS = [
        (-3.14, 3.14),   # shoulder_pan
        (-1.57, 1.57),   # shoulder_lift
        (-2.50, 2.50),   # elbow_flex
        (-1.57, 1.57),   # wrist_flex
        (-3.14, 3.14),   # wrist_roll
        (0.00,  1.00),   # gripper
    ]
    CAMERAS   = ["top", "wrist"]
    TASK_TEXT = (
        "Pick the plastic jar from the table, place it upright on the silicone puck, "
        "seat the lid on the jar to engage the threads, "
        "then carry the closed jar to the wooden goal block."
    )

    def __init__(self, ros_topic: str = "/so100/joint_commands"):
        self._ros_topic = ros_topic
        self._ros_pub = None   # lazy-init in send_action
        self._state = np.zeros(self.STATE_DIM, dtype=np.float32)

    def send_action(self, action_vector: np.ndarray) -> None:
        """
        Send joint command.
        Override this to connect to your ROS publisher / serial port.
        """
        safe = self.clip_to_limits(action_vector)
        # ── Stub: replace with actual robot interface ──
        # import rospy
        # from sensor_msgs.msg import JointState
        # msg = JointState(); msg.position = safe.tolist()
        # self._ros_pub.publish(msg)
        self._state = safe.copy()  # mock feedback

    def get_state(self) -> np.ndarray:
        """Read current joint positions."""
        # ── Stub: replace with actual robot state subscriber ──
        return self._state.copy()


# ─────────────────────────────────────────────────────────────────────────────
# Franka Emika Panda  (7-DOF)
# ─────────────────────────────────────────────────────────────────────────────

@register_arm("franka")
class FrankaArm(BaseArm):
    """
    7-DOF Franka Emika Panda.
    Typical interface: franka_ros / libfranka.
    """
    ACTION_DIM  = 7
    STATE_DIM   = 7
    JOINT_NAMES = ["panda_j1", "panda_j2", "panda_j3", "panda_j4",
                   "panda_j5", "panda_j6", "panda_j7"]
    JOINT_LIMITS = [
        (-2.897, 2.897),
        (-1.763, 1.763),
        (-2.897, 2.897),
        (-3.072, -0.069),
        (-2.897, 2.897),
        (-0.018, 3.753),
        (-2.897, 2.897),
    ]
    CAMERAS   = ["wrist", "external"]
    TASK_TEXT = "Manipulate the object on the table with the Franka arm."

    def __init__(self, robot_ip: str = "172.16.0.2"):
        self._robot_ip = robot_ip
        self._state = np.zeros(self.STATE_DIM, dtype=np.float32)

    def send_action(self, action_vector: np.ndarray) -> None:
        safe = self.clip_to_limits(action_vector)
        # ── Stub: replace with libfranka / franka_ros call ──
        # robot.move(JointMotion(safe.tolist()))
        self._state = safe.copy()

    def get_state(self) -> np.ndarray:
        # ── Stub: replace with robot.read_once().q ──
        return self._state.copy()


# ─────────────────────────────────────────────────────────────────────────────
# Universal Robots UR5  (6-DOF)
# ─────────────────────────────────────────────────────────────────────────────

@register_arm("ur5")
class UR5Arm(BaseArm):
    """
    6-DOF Universal Robots UR5.
    Interface via ur_rtde or ROS ur_robot_driver.
    """
    ACTION_DIM  = 6
    STATE_DIM   = 6
    JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow",
                   "wrist_1", "wrist_2", "wrist_3"]
    JOINT_LIMITS = [
        (-6.283, 6.283),
        (-6.283, 6.283),
        (-3.141, 3.141),
        (-6.283, 6.283),
        (-6.283, 6.283),
        (-6.283, 6.283),
    ]
    CAMERAS   = ["overhead", "wrist"]
    TASK_TEXT = "Perform pick-and-place manipulation with the UR5 arm."

    def __init__(self, robot_ip: str = "192.168.1.100"):
        self._robot_ip = robot_ip
        self._state = np.zeros(self.STATE_DIM, dtype=np.float32)
        # import rtde_control, rtde_receive  # uncomment for real robot

    def send_action(self, action_vector: np.ndarray) -> None:
        safe = self.clip_to_limits(action_vector)
        # ── Stub: replace with rtde_control.moveJ(safe.tolist()) ──
        self._state = safe.copy()

    def get_state(self) -> np.ndarray:
        # ── Stub: replace with rtde_receive.getActualQ() ──
        return self._state.copy()


# ─────────────────────────────────────────────────────────────────────────────
# Generic 7-DOF placeholder (good starting point for custom arms)
# ─────────────────────────────────────────────────────────────────────────────

@register_arm("generic7dof")
class Generic7DOFArm(BaseArm):
    """
    Generic 7-DOF arm placeholder.
    Use this as a template when adding a new custom arm.
    Copy this class, rename it, set your limits, and implement
    send_action() / get_state() for your hardware.
    """
    ACTION_DIM  = 7
    STATE_DIM   = 7
    JOINT_NAMES = [f"joint_{i}" for i in range(1, 8)]
    JOINT_LIMITS = [(-3.14, 3.14)] * 6 + [(0.0, 1.0)]
    CAMERAS   = ["camera_0"]
    TASK_TEXT = "Perform the manipulation task."

    def __init__(self):
        self._state = np.zeros(self.STATE_DIM, dtype=np.float32)

    def send_action(self, action_vector: np.ndarray) -> None:
        self._state = self.clip_to_limits(action_vector)

    def get_state(self) -> np.ndarray:
        return self._state.copy()
