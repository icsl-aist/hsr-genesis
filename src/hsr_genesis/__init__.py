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
    raise AttributeError(name)


def __dir__():
    return sorted(__all__ + ["HSRBURDF", "HSRRigidEntity"])
