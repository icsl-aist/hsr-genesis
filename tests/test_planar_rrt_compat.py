"""Small native scenes exercise the planar planner without the HSR asset or IK."""
from types import SimpleNamespace

import genesis as gs
import pytest
import torch

from hsr_genesis.hsr_rigid_entity import HSRRigidEntity


@pytest.mark.parametrize("planner", ["RRT", "RRTConnect"])
def test_native_planar_rrt_keeps_endpoints_and_valid_quaternions(planner):
    scene = gs.Scene(show_viewer=False)
    box = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1)))
    scene.build()
    # Use the small native FREE entity with the exact HSR planar adapter.
    box._morph = SimpleNamespace(hsr_base_mode="planar")
    start = box.get_qpos().clone()
    goal = start.clone()
    goal[0] = 0.1
    goal[3] = torch.cos(goal.new_tensor(0.1))
    goal[6] = torch.sin(goal.new_tensor(0.1))
    lower, upper = (limit.clone() for limit in box.get_dofs_limit())
    try:
        path, valid = HSRRigidEntity.plan_path_planar(
            box, goal, qpos_start=start, planner=planner, num_waypoints=12,
            max_nodes=1000, timeout=10, smooth_path=True, ignore_collision=False,
            base_xy_range=0.5, base_yaw_range=0.4, return_valid_mask=True,
        )
        assert valid
        torch.testing.assert_close(path[0], start)
        torch.testing.assert_close(path[-1], goal, atol=0.005, rtol=0)
        torch.testing.assert_close(path[:, 3:7].norm(dim=-1), torch.ones_like(path[:, 3]))
        assert torch.count_nonzero(path[:, [2, 4, 5]]) == 0
        assert (path[:, :2].abs() <= 0.5).all()
        assert (2 * torch.atan2(path[:, 6], path[:, 3])).abs().max() <= 0.4
        torch.testing.assert_close(box.get_qpos(), start)
        restored_lower, restored_upper = box.get_dofs_limit()
        torch.testing.assert_close(restored_lower, lower)
        torch.testing.assert_close(restored_upper, upper)
        assert not hasattr(box, "_hsr_planar_sampling_limits")
    finally:
        scene.destroy()


def test_planar_limits_map_dofs_to_qpos_and_restrict_yaw():
    from hsr_genesis.genesis_patches import apply_planar_rrt_patch
    from genesis.utils.path_planning import RRTConnect

    apply_planar_rrt_patch()
    values = lambda x: torch.tensor(x, dtype=gs.tc_float, device=gs.device)
    lower = values([-float("inf")] * 6 + [-0.3])
    upper = values([float("inf")] * 6 + [0.7])
    entity = SimpleNamespace(
        _morph=SimpleNamespace(hsr_base_mode="planar"),
        _solver=SimpleNamespace(_options=SimpleNamespace(batch_dofs_info=False)),
        _hsr_planar_sampling_limits=(0.5, 0.4),
        n_qs=8, n_dofs=7,
        joints=[
            SimpleNamespace(type=gs.JOINT_TYPE.FREE, qs_idx_local=list(range(7))),
            SimpleNamespace(type=gs.JOINT_TYPE.REVOLUTE, qs_idx_local=[7], dofs_idx_local=[6]),
        ],
        get_dofs_limit=lambda **kwargs: (lower, upper),
    )
    planner = RRTConnect(entity)
    goal = values([[0, 0, 0, 1, 0, 0, 0, 0.2], [0, 0, 0, 0.7, 0, 0, 0.7, 0.2]])
    lo, hi, invalid = planner._get_dofs_limit(goal, torch.tensor([0, 1], device=gs.device))
    assert lo.shape == hi.shape == (2, 8)
    torch.testing.assert_close(lo[:, 7], values([-0.3, -0.3]))
    torch.testing.assert_close(hi[:, 7], values([0.7, 0.7]))
    assert invalid.tolist() == [False, True]
