"""Import-safe dataclass for target marker sphere options.

This module has NO dependency on Genesis, so it can be imported and tested
without the Genesis runtime environment.
"""

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class TargetMarkerOptions:
    """Encodes creation options for the red target marker sphere.

    Fields match gs.morphs.Sphere keyword arguments exactly.
    The default ``fixed=True`` makes the marker kinematic so it stays
    at its placed position (does not fall under gravity).

    Import-safe: no Genesis import required.
    """

    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 0.04
    collision: bool = False
    contype: int = 0
    conaffinity: int = 0
    fixed: bool = True  # marker is kinematic; won't fall under gravity

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict suitable for ``gs.morphs.Sphere(**opts.to_dict())``."""
        return asdict(self)
