"""Tests for TargetMarkerOptions — no Genesis runtime required."""

from target_marker_options import TargetMarkerOptions


def test_default_fixed_is_true() -> None:
    """The default marker should be fixed=True (kinematic, no gravity)."""
    opts = TargetMarkerOptions()
    assert opts.fixed is True


def test_custom_fixed_is_true() -> None:
    """Explicit fixed=True should round-trip through to_dict."""
    opts = TargetMarkerOptions(fixed=True, pos=(0.7, -0.25, 0.55))
    d = opts.to_dict()
    assert d["pos"] == (0.7, -0.25, 0.55)
    assert d["fixed"] is True


def test_to_dict_contains_fixed() -> None:
    """to_dict() must include the fixed key."""
    opts = TargetMarkerOptions()
    d = opts.to_dict()
    assert "fixed" in d
    assert d["fixed"] is True


def test_default_collision_false() -> None:
    """Default collision should be False — marker is not a physical obstacle."""
    opts = TargetMarkerOptions()
    assert opts.collision is False
    assert opts.contype == 0
    assert opts.conaffinity == 0
