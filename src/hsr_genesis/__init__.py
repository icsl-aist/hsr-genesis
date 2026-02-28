"""HSR-specific integrations for the Genesis ecosystem.

License: BSD 3-Clause, compatible with the original ROS packages.
See `hsr_genesis/LICENSE.txt` for details.
"""

__all__ = [
    "__version__",
    "Vector2",
    "RobotFunction2Request",
    "RobotFunction2Response",
    "RobotFunction2Parameter",
    "RobotFunction2",
    "BiGoldenSectionLineSearch",
    "HookeAndJeevesMethod",
    "RobotOptimizer",
    "RobotOptimizerGPU",
    "URDFSensorManager",
    "apply_runtime_patches",
    "apply_raycast_filter_patch",
    "set_raycast_ignore_list",
    "update_raycast_ignore_list",
    "clear_raycast_ignore_list",
]

__version__ = "0.1.0"

from .analytic_ik import (
    Vector2,
    RobotFunction2Request,
    RobotFunction2Response,
    RobotFunction2Parameter,
    RobotFunction2,
    BiGoldenSectionLineSearch,
    HookeAndJeevesMethod,
    RobotOptimizer,
    RobotOptimizerGPU,
)

from .sensor_manager import URDFSensorManager


def __getattr__(name: str):
    if name in ("HSRBURDF", "HSRRigidEntity"):
        from .hsr_rigid_entity import HSRBURDF, HSRRigidEntity

        return HSRBURDF if name == "HSRBURDF" else HSRRigidEntity
    if name == "apply_runtime_patches":
        from .genesis_patches import apply_runtime_patches

        return apply_runtime_patches
    if name == "apply_raycast_filter_patch":
        from .raycast_filter_patch import apply_raycast_filter_patch

        return apply_raycast_filter_patch
    if name == "set_raycast_ignore_list":
        from .raycast_filter_patch import set_raycast_ignore_list

        return set_raycast_ignore_list
    if name == "update_raycast_ignore_list":
        from .raycast_filter_patch import update_raycast_ignore_list

        return update_raycast_ignore_list
    if name == "clear_raycast_ignore_list":
        from .raycast_filter_patch import clear_raycast_ignore_list

        return clear_raycast_ignore_list
    raise AttributeError(name)


def __dir__():
    return sorted(__all__ + ["HSRBURDF", "HSRRigidEntity"])
