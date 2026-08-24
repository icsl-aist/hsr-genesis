"""Tests for camera crane path interpolation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "rl"))

import numpy as np
import pytest
from camera_crane import CraneKeyframe, crane_path, smoothstep


def test_smoothstep_endpoints():
    """Smoothstep is 0 at t=0 and 1 at t=1."""
    assert smoothstep(0.0) == 0.0
    assert smoothstep(1.0) == 1.0


def test_smoothstep_midpoint():
    """Smoothstep at t=0.5 is 0.5 (symmetric)."""
    assert abs(smoothstep(0.5) - 0.5) < 1e-10


def test_smoothstep_monotonic():
    """Smoothstep is monotonically increasing."""
    ts = np.linspace(0, 1, 100)
    vals = [smoothstep(t) for t in ts]
    assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))


def test_crane_path_at_start():
    """At t=0, crane_path returns the first keyframe's pos and lookat."""
    keyframes = [
        CraneKeyframe(pos=(4.5, -4.0, 2.5), lookat=(1.5, 1.5, 0.5), time=0.0),
        CraneKeyframe(pos=(25.0, 25.0, 50.0), lookat=(25.0, 25.0, 0.0), time=0.5),
        CraneKeyframe(pos=(46.5, 46.5, 100.0), lookat=(46.5, 46.5, 0.0), time=1.0),
    ]
    pos, lookat = crane_path(0.0, keyframes)
    np.testing.assert_allclose(pos, [4.5, -4.0, 2.5])
    np.testing.assert_allclose(lookat, [1.5, 1.5, 0.5])


def test_crane_path_at_end():
    """At t=1, crane_path returns the last keyframe's pos and lookat."""
    keyframes = [
        CraneKeyframe(pos=(4.5, -4.0, 2.5), lookat=(1.5, 1.5, 0.5), time=0.0),
        CraneKeyframe(pos=(25.0, 25.0, 50.0), lookat=(25.0, 25.0, 0.0), time=0.5),
        CraneKeyframe(pos=(46.5, 46.5, 100.0), lookat=(46.5, 46.5, 0.0), time=1.0),
    ]
    pos, lookat = crane_path(1.0, keyframes)
    np.testing.assert_allclose(pos, [46.5, 46.5, 100.0])
    np.testing.assert_allclose(lookat, [46.5, 46.5, 0.0])


def test_crane_path_between_keyframes():
    """At t=0.5 (smoothstep-eased), crane_path interpolates between keyframes."""
    keyframes = [
        CraneKeyframe(pos=(0.0, 0.0, 0.0), lookat=(0.0, 0.0, 0.0), time=0.0),
        CraneKeyframe(pos=(10.0, 10.0, 10.0), lookat=(5.0, 5.0, 5.0), time=1.0),
    ]
    # smoothstep(0.5) = 0.5, so midpoint should be exactly halfway
    pos, lookat = crane_path(0.5, keyframes)
    np.testing.assert_allclose(pos, [5.0, 5.0, 5.0])
    np.testing.assert_allclose(lookat, [2.5, 2.5, 2.5])


def test_crane_path_three_keyframes_mid():
    """With 3 keyframes, t=0.5 hits the middle keyframe exactly."""
    keyframes = [
        CraneKeyframe(pos=(0.0, 0.0, 0.0), lookat=(0.0, 0.0, 0.0), time=0.0),
        CraneKeyframe(pos=(10.0, 10.0, 10.0), lookat=(5.0, 5.0, 5.0), time=0.5),
        CraneKeyframe(pos=(20.0, 20.0, 20.0), lookat=(10.0, 10.0, 10.0), time=1.0),
    ]
    pos, lookat = crane_path(0.5, keyframes)
    np.testing.assert_allclose(pos, [10.0, 10.0, 10.0])
    np.testing.assert_allclose(lookat, [5.0, 5.0, 5.0])
