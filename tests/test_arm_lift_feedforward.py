"""Test arm_lift feed-forward gravity compensation helper.

The feed-forward force is applied to arm_lift_joint during whole-body
trajectory execution to counter arm sag under gravity.

Run with:
    PYTHONPATH=src .venv/bin/python -m pytest tests/test_arm_lift_feedforward.py -v -s
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import genesis as gs

if not getattr(gs, "_initialized", False):
    gs.init(backend=gs.cpu, precision="32", logging_level="warning")

from hsr_genesis.hsr_rigid_entity import HSRRigidEntity  # noqa: E402


class TestArmLiftFeedforwardForce:
    """Verify the static helper that provides the gravity-compensation force."""

    def test_returns_positive_float(self):
        """_arm_lift_gravity_compensation_force returns a positive float
        in a physically plausible range (30–100 N)."""
        # This will fail until we add the static method to HSRRigidEntity.
        force = HSRRigidEntity._arm_lift_gravity_compensation_force()
        assert isinstance(force, float), f"Expected float, got {type(force)}"
        assert force > 0.0, f"Expected positive force, got {force}"
        assert 30.0 <= force <= 100.0, (
            f"Expected force in [30, 100] N range, got {force}"
        )
