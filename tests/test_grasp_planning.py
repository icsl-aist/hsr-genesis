"""Pure-tensor tests for hsr_genesis.grasp_planning helpers and phase math."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hsr_genesis.grasp_planning import (  # noqa: E402
    IKPlan,
    IKPlanner,
    quat_from_palm_and_jaw,
    quat_rotate_batch,
    quat_z_to_vec_batch,
    yaw_to_quat_wxyz_batch,
)


def test_quat_z_to_vec_down() -> None:
    q = quat_z_to_vec_batch(torch.tensor([[0.0, 0.0, -1.0]]))
    # +Z rotated to -Z = 180° about X → w=0, x=±1
    assert abs(q[0, 0].item()) < 1e-6
    assert abs(q[0, 1].item()) > 0.99


def test_quat_z_to_vec_identity_and_perp() -> None:
    # +Z → +Z is identity
    q = quat_z_to_vec_batch(torch.tensor([[0.0, 0.0, 1.0]]))
    assert torch.allclose(q, torch.tensor([[1.0, 0.0, 0.0, 0.0]]), atol=1e-6)
    # +Z → +X: rotating palm forward
    q = quat_z_to_vec_batch(torch.tensor([[1.0, 0.0, 0.0]]))
    rotated = quat_rotate_batch(q, torch.tensor([[0.0, 0.0, 1.0]]))
    assert torch.allclose(rotated, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-5)


def test_quat_from_palm_and_jaw_orthonormal() -> None:
    q = quat_from_palm_and_jaw(
        palm_direction=torch.tensor([[0.0, 0.0, 1.0]]),
        jaw_axis=torch.tensor([[0.0, 1.0, 0.0]]),
    )
    assert torch.allclose(q.norm(dim=-1), torch.ones(1), atol=1e-6)
    # local +Z maps to palm direction
    z_world = quat_rotate_batch(q, torch.tensor([[0.0, 0.0, 1.0]]))
    assert torch.allclose(z_world, torch.tensor([[0.0, 0.0, 1.0]]), atol=1e-5)


def test_yaw_to_quat_roundtrip() -> None:
    yaws = torch.tensor([0.0, torch.pi / 2, -torch.pi / 2])
    q = yaw_to_quat_wxyz_batch(yaws)
    assert torch.allclose(q.norm(dim=-1), torch.ones(3), atol=1e-6)


def test_phase_boundaries() -> None:
    saved = (
        IKPlanner.approach_steps, IKPlanner.descend_steps,
        IKPlanner.grasp_steps, IKPlanner.lift_steps,
    )
    try:
        IKPlanner.configure_phase_steps(approach=10, descend=20, grasp=30, lift=40)
        assert IKPlanner.descend_start() == 10
        assert IKPlanner.grasp_start() == 30
        assert IKPlanner.lift_start() == 60
        assert IKPlanner.get_phase(5) == 0
        assert IKPlanner.get_phase(15) == 1
        assert IKPlanner.get_phase(55) == 2
        assert IKPlanner.get_phase(61) == 3
        assert IKPlanner.max_steps() == 100
    finally:
        IKPlanner.configure_phase_steps(*saved)


def test_get_phase_targets_grasp_holds_descend() -> None:
    z = torch.zeros(2, 5)
    b = torch.zeros(2, 3)
    plan = IKPlan(
        approach_arm=z + 1, approach_base=b + 1,
        descend_arm=z + 2, descend_base=b + 2,
        lift_arm=z + 3, lift_base=b + 3,
        obj_pos=b.clone(), obj_init_z=torch.zeros(2),
        ik_success=torch.zeros(2, 3, dtype=torch.bool),
        pre_grasp_height=0.1, grasp_offset_z=0.02,
        gripper_effort=3.0, grasp_hold_steps=300,
    )
    arm, base = IKPlanner.get_phase_targets(plan, 2)
    assert torch.equal(arm, plan.descend_arm)
    assert torch.equal(base, plan.descend_base)
