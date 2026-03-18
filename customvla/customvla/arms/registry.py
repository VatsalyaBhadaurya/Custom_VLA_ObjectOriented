"""
customvla/arms/registry.py

Global registry for arm definitions.
Use @register_arm("name") to add a new arm class.
Use get_arm("name") to instantiate it.

Example
-------
    from customvla.arms.registry import register_arm, get_arm

    @register_arm("so100")
    class SO100Arm(BaseArm):
        ...

    arm = get_arm("so100")
    print(arm.describe())
"""

from typing import Dict, Type, Optional
from customvla.arms.base import BaseArm

_REGISTRY: Dict[str, Type[BaseArm]] = {}


def register_arm(name: str):
    """
    Decorator to register an arm class under a given string key.

    Usage:
        @register_arm("franka")
        class FrankaArm(BaseArm):
            ...
    """
    def decorator(cls: Type[BaseArm]):
        if not issubclass(cls, BaseArm):
            raise TypeError(f"@register_arm: '{cls.__name__}' must subclass BaseArm")
        if name in _REGISTRY:
            raise ValueError(
                f"Arm '{name}' is already registered by '{_REGISTRY[name].__name__}'. "
                f"Use a unique name."
            )
        _REGISTRY[name] = cls
        cls._arm_registry_name = name
        return cls
    return decorator


def get_arm(name: str, **kwargs) -> BaseArm:
    """
    Instantiate a registered arm by name.

    Args:
        name: the key used in @register_arm(...)
        **kwargs: forwarded to the arm's __init__

    Returns:
        An instance of the arm class.

    Raises:
        KeyError if the name is not registered.
    """
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise KeyError(
            f"Arm '{name}' not found in registry. "
            f"Available arms: {available}"
        )
    return _REGISTRY[name](**kwargs)


def list_arms():
    """Print all registered arm names and their key specs."""
    if not _REGISTRY:
        print("No arms registered yet.")
        return
    print(f"{'Name':<20} {'DOF':>5} {'Cameras':<25} Class")
    print("-" * 70)
    for name, cls in sorted(_REGISTRY.items()):
        cams = ", ".join(cls.CAMERAS)
        print(f"{name:<20} {cls.ACTION_DIM:>5} {cams:<25} {cls.__name__}")


class ArmRegistry:
    """Namespace wrapper for the global registry (for IDE discoverability)."""
    register = staticmethod(register_arm)
    get = staticmethod(get_arm)
    list = staticmethod(list_arms)
    _registry = _REGISTRY
