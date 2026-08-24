"""Camera crane path interpolation for promo video.

Given a set of keyframes (pos, lookat, time), interpolates the camera
pose along the path with smoothstep easing for ease-in/ease-out.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CraneKeyframe:
    """A single keyframe in the camera crane path.

    Attributes:
        pos: Camera position (3,) in env0-relative coordinates.
        lookat: Camera look-at point (3,) in env0-relative coordinates.
        time: Normalized time of this keyframe in [0, 1].
    """
    pos: tuple[float, float, float]
    lookat: tuple[float, float, float]
    time: float


def smoothstep(t: float) -> float:
    """Hermite smoothstep: 3t^2 - 2t^3. Maps [0,1] -> [0,1] with zero derivative at endpoints."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation between two 3D points."""
    return a + (b - a) * t


def crane_path(t: float, keyframes: list[CraneKeyframe]) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate camera pose at normalized time t in [0, 1].

    The raw time t is first passed through smoothstep for ease-in/ease-out,
    then linearly interpolated between the surrounding keyframes.

    Args:
        t: Normalized time in [0, 1].
        keyframes: Sorted list of CraneKeyframes with times spanning [0, 1].

    Returns:
        (pos, lookat) as np.ndarray shape (3,).
    """
    t_eased = smoothstep(t)

    # Find the surrounding keyframes
    if t_eased <= keyframes[0].time:
        return np.array(keyframes[0].pos), np.array(keyframes[0].lookat)
    if t_eased >= keyframes[-1].time:
        return np.array(keyframes[-1].pos), np.array(keyframes[-1].lookat)

    # Find surrounding pair
    for i in range(len(keyframes) - 1):
        kf_a = keyframes[i]
        kf_b = keyframes[i + 1]
        if kf_a.time <= t_eased <= kf_b.time:
            local_t = (t_eased - kf_a.time) / (kf_b.time - kf_a.time)
            pos = _lerp(np.array(kf_a.pos), np.array(kf_b.pos), local_t)
            lookat = _lerp(np.array(kf_a.lookat), np.array(kf_b.lookat), local_t)
            return pos, lookat

    # Should not reach here
    return np.array(keyframes[-1].pos), np.array(keyframes[-1].lookat)
