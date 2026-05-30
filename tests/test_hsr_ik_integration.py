import math
from pathlib import Path

import pytest
import torch

import genesis as gs


_HSR = None


def _check_gpu():
    """Return True if the IK batch solver's GPU requirement is met."""
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
        pytest.skip("IK batch solver requires a GPU-capable Taichi backend (CUDA/Vulkan/OpenGL/Metal)")

    from hsr_genesis.hsr_rigid_entity import HSRBURDF

    URDF_PATH = Path(__file__).resolve().parents[1] / "data" / "urdf" / "hsrb4s.urdf"
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
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
        ),
    )
    scene.build()
    _HSR = hsr
    return _HSR


def _mat4_from_pos_quat_wxyz(pos, quat):
    w, x, y, z = quat
    n = w * w + x * x + y * y + z * z
    eps = 1e-12
    n_safe = max(n, eps)
    s = 2.0 / n_safe
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    xx, xy, xz = x * x * s, x * y * s, x * z * s
    yy, yz, zz = y * y * s, y * z * s, z * z * s
    R = [[1.0 - (yy + zz), xy - wz, xz + wy],
         [xy + wz, 1.0 - (xx + zz), yz - wx],
         [xz - wy, yz + wx, 1.0 - (xx + yy)]]
    T = torch.eye(4, dtype=gs.tc_float)
    T[:3, :3] = torch.tensor(R, dtype=gs.tc_float)
    T[:3, 3] = torch.tensor(pos, dtype=gs.tc_float)
    return T


class TestHSRIKIntegration:
    """Entity-level integration tests for end_effector_offset and init_qpos.

    These tests exercise the full HSRBURDF.inverse_kinematics() path, which
    delegates to the GPU-based batch IK solver.  They require a GPU-capable
    Taichi backend (CUDA / Vulkan / OpenGL / Metal).
    """

    def test_end_effector_offset_reaches_target(self):
        hsr = _ensure_hsr()
        end_effector = hsr.get_link("hand_palm_link")

        offset_z = 0.04
        hsr.end_effector_offset = [0.0, 0.0, offset_z]

        target_pos = torch.tensor([0.45, 0.0, 0.06], dtype=gs.tc_float)
        target_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=gs.tc_float)

        qpos = hsr.inverse_kinematics(
            link=end_effector,
            pos=target_pos,
            quat=target_quat,
            max_samples=50,
            max_solver_iters=30,
            respect_joint_limit=False,
        )

        assert qpos is not None
        assert not torch.isnan(qpos).any()

        # FK the solution to get the EE pose
        end_link = hsr.get_link("hand_palm_link")
        fk_pos, fk_quat = hsr.forward_kinematics(
            qpos, links_idx_local=[end_link.idx_local]
        )

        # Recompute: offset point world = EE_pos + R_EE * local_point
        # local_point = [0, 0, offset_z]
        w, x, y, z = fk_quat[0]
        n = w * w + x * x + y * y + z * z
        eps = 1e-12
        n_safe = max(n.item(), eps)
        s = 2.0 / n_safe
        wx, wy, wz = w * x * s, w * y * s, w * z * s
        xx, xy, xz = x * x * s, x * y * s, x * z * s
        yy, yz, zz = y * y * s, y * z * s, z * z * s
        R = torch.tensor([[1.0 - (yy + zz), xy - wz, xz + wy],
                          [xy + wz, 1.0 - (xx + zz), yz - wx],
                          [xz - wy, yz + wx, 1.0 - (xx + yy)]],
                         dtype=gs.tc_float)

        local_point_t = torch.tensor([0.0, 0.0, offset_z], dtype=gs.tc_float)
        offset_world = fk_pos[0] + R @ local_point_t

        assert torch.allclose(offset_world, target_pos, atol=5e-3), (
            f"Offset point {offset_world.tolist()} should match target {target_pos.tolist()}"
        )

    def test_ik_with_init_qpos_produces_valid_solution(self):
        hsr = _ensure_hsr()
        # Clear any offset from previous tests so we test init_qpos in isolation
        hsr.end_effector_offset = None
        end_effector = hsr.get_link("hand_palm_link")

        target_pos = torch.tensor([0.45, 0.0, 0.06], dtype=gs.tc_float)
        target_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=gs.tc_float)

        # Create a custom init_qpos with a retracted arm configuration
        arm_dofs = hsr._ensure_arm_qs_idx()
        init_qpos = hsr.get_qpos().clone()
        for idx in arm_dofs:
            init_qpos[idx] = 0.0

        qpos = hsr.inverse_kinematics(
            link=end_effector,
            pos=target_pos,
            quat=target_quat,
            init_qpos=init_qpos,
            max_samples=50,
            max_solver_iters=30,
            respect_joint_limit=False,
        )

        assert qpos is not None
        assert not torch.isnan(qpos).any()

        # Verify the solution FK's back to the target
        end_link = hsr.get_link("hand_palm_link")
        fk_pos, fk_quat = hsr.forward_kinematics(
            qpos, links_idx_local=[end_link.idx_local]
        )

        assert torch.allclose(fk_pos[0], target_pos, atol=1e-2), (
            f"FK position {fk_pos[0].tolist()} should match target {target_pos.tolist()}"
        )
