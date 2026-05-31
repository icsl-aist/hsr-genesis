from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import genesis as gs


_HSR = None


def _check_gpu():
    try:
        import quadrants as ti

        arch = ti.cfg.arch
    except Exception:
        return False
    gpu_arches = [ti.cuda, ti.vulkan]
    if hasattr(ti, "opengl"):
        gpu_arches.append(ti.opengl)
    if hasattr(ti, "metal"):
        gpu_arches.append(ti.metal)
    return arch in tuple(gpu_arches)


def _ensure_hsr():
    global _HSR
    if _HSR is not None:
        return _HSR

    if not getattr(gs, "_initialized", False):
        gs.init(backend=gs.gpu, logging_level="warning")

    if not _check_gpu():
        pytest.skip("planar RRT requires a GPU-capable Taichi backend (CUDA/Vulkan/OpenGL/Metal)")

    from hsr_genesis.hsr_rigid_entity import HSRBURDF

    URDF_PATH = Path(__file__).resolve().parents[1] / "data" / "urdf" / "hsrb4s.urdf"
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        rigid_options=gs.options.RigidOptions(use_gjk_collision=True),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    hsr = scene.add_entity(
        HSRBURDF(
            file=str(URDF_PATH),
            fixed=False,
            recompute_inertia=False,
            links_to_keep=["hand_palm_link"],
            robot="hsrb",
            base_mode="planar",
            end_effector_frame="hand_palm_link",
            use_base_controller=False,
            optimizer="gpu",
        ),
    )
    scene.build()
    _HSR = hsr
    return _HSR


def _free_joint_start(hsr):
    for joint in hsr.joints:
        if joint.type == gs.JOINT_TYPE.FREE:
            idxs = joint.qs_idx_local
            return min(idxs) if idxs else 0
    return 0


class TestPlanarRRT:
    def test_planar_rrt_plans_path(self):
        hsr = _ensure_hsr()
        end_effector = hsr.get_link("hand_palm_link")

        start_qpos = hsr.get_qpos().clone()

        goal_pos = torch.tensor([0.6, 0.0, 0.4], dtype=gs.tc_float)
        goal_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=gs.tc_float)

        goal_qpos = hsr.inverse_kinematics(
            link=end_effector,
            pos=goal_pos,
            quat=goal_quat,
            max_samples=100,
            max_solver_iters=80,
        )
        if goal_qpos.ndim == 2:
            goal_qpos = goal_qpos.squeeze(0)

        result = hsr.plan_path_planar(
            qpos_goal=goal_qpos,
            qpos_start=start_qpos,
            planner="RRTConnect",
            num_waypoints=50,
            smooth_path=True,
            resolution=0.05,
            max_nodes=2000,
            timeout=10.0,
        )

        if isinstance(result, tuple):
            path = result[0]
        else:
            path = result

        assert path is not None
        assert path.ndim >= 2
        assert path.shape[0] > 1

    def test_planar_rrt_quaternion_normalized(self):
        hsr = _ensure_hsr()
        end_effector = hsr.get_link("hand_palm_link")

        start_qpos = hsr.get_qpos().clone()

        goal_pos = torch.tensor([0.5, 0.2, 0.3], dtype=gs.tc_float)
        goal_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=gs.tc_float)

        goal_qpos = hsr.inverse_kinematics(
            link=end_effector,
            pos=goal_pos,
            quat=goal_quat,
            max_samples=100,
            max_solver_iters=80,
        )
        if goal_qpos.ndim == 2:
            goal_qpos = goal_qpos.squeeze(0)

        result = hsr.plan_path_planar(
            qpos_goal=goal_qpos,
            qpos_start=start_qpos,
            planner="RRTConnect",
            num_waypoints=50,
            smooth_path=True,
            resolution=0.05,
            max_nodes=2000,
            timeout=10.0,
        )

        if isinstance(result, tuple):
            path = result[0]
        else:
            path = result

        free_start = _free_joint_start(hsr)
        qw = path[..., free_start + 3]
        qx = path[..., free_start + 4]
        qy = path[..., free_start + 5]
        qz = path[..., free_start + 6]
        norm_sq = qw * qw + qx * qx + qy * qy + qz * qz
        assert torch.allclose(norm_sq, torch.ones_like(norm_sq), atol=1e-6), (
            "Quaternion components must be unit-norm after planning"
        )

    def test_planar_rrt_keeps_planar_constraints(self):
        hsr = _ensure_hsr()
        end_effector = hsr.get_link("hand_palm_link")

        start_qpos = hsr.get_qpos().clone()

        goal_pos = torch.tensor([0.5, -0.2, 0.4], dtype=gs.tc_float)
        goal_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=gs.tc_float)

        goal_qpos = hsr.inverse_kinematics(
            link=end_effector,
            pos=goal_pos,
            quat=goal_quat,
            max_samples=100,
            max_solver_iters=80,
        )
        if goal_qpos.ndim == 2:
            goal_qpos = goal_qpos.squeeze(0)

        result = hsr.plan_path_planar(
            qpos_goal=goal_qpos,
            qpos_start=start_qpos,
            planner="RRTConnect",
            num_waypoints=50,
            smooth_path=True,
            resolution=0.05,
            max_nodes=2000,
            timeout=10.0,
        )

        if isinstance(result, tuple):
            path = result[0]
        else:
            path = result

        free_start = _free_joint_start(hsr)

        z_vals = path[..., free_start + 2]
        assert torch.allclose(z_vals, torch.zeros_like(z_vals), atol=1e-6), (
            f"Base z must remain 0, got range [{z_vals.min().item():.6f}, {z_vals.max().item():.6f}]"
        )

        qx_vals = path[..., free_start + 4]
        qy_vals = path[..., free_start + 5]
        assert torch.allclose(qx_vals, torch.zeros_like(qx_vals), atol=1e-6), (
            f"qx must be 0, got range [{qx_vals.min().item():.6f}, {qx_vals.max().item():.6f}]"
        )
        assert torch.allclose(qy_vals, torch.zeros_like(qy_vals), atol=1e-6), (
            f"qy must be 0, got range [{qy_vals.min().item():.6f}, {qy_vals.max().item():.6f}]"
        )

    def test_planar_rrt_restores_q_limit(self):
        hsr = _ensure_hsr()
        end_effector = hsr.get_link("hand_palm_link")

        original_lower = np.array(hsr.q_limit[0], copy=True)
        original_upper = np.array(hsr.q_limit[1], copy=True)

        start_qpos = hsr.get_qpos().clone()
        goal_pos = torch.tensor([0.5, 0.0, 0.3], dtype=gs.tc_float)
        goal_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=gs.tc_float)
        goal_qpos = hsr.inverse_kinematics(
            link=end_effector,
            pos=goal_pos,
            quat=goal_quat,
            max_samples=100,
            max_solver_iters=80,
        )
        if goal_qpos.ndim == 2:
            goal_qpos = goal_qpos.squeeze(0)

        hsr.plan_path_planar(
            qpos_goal=goal_qpos,
            qpos_start=start_qpos,
            planner="RRTConnect",
            num_waypoints=50,
            smooth_path=True,
            resolution=0.05,
            max_nodes=2000,
            timeout=10.0,
        )

        restored_lower = np.array(hsr.q_limit[0])
        restored_upper = np.array(hsr.q_limit[1])

        assert np.allclose(restored_lower, original_lower), "q_limit lower bounds not restored"
        assert np.allclose(restored_upper, original_upper), "q_limit upper bounds not restored"
