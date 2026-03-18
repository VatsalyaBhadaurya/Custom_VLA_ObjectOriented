"""customvla/arms/__init__.py — imports trigger arm registration."""
from customvla.arms.base import BaseArm
from customvla.arms.registry import ArmRegistry, register_arm, get_arm, list_arms
from customvla.arms import builtin  # side-effect: registers SO100, Franka, UR5, Generic7DOF
