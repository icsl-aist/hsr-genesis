import math
from pathlib import Path

import pytest
import torch

import genesis as gs

LINK_NAMES = [
    "base_footprint",
    "arm_lift_link",
    "arm_flex_link",
    "arm_roll_link",
    "wrist_flex_link",
    "hand_palm_link",
    "torso_lift_link",
]

_HSR = None
_LINK_INDICES = None


def _ensure_hsr():
    global _HSR, _LINK_INDICES
    if _HSR is not None:
        return _HSR, _LINK_INDICES

    if not getattr(gs, "_initialized", False):
        gs.init(backend=gs.cpu, logging_level="warning")

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
            links_to_keep=LINK_NAMES,
            robot="hsrb",
            base_mode="planar",
            end_effector_frame="hand_palm_link",
            use_base_controller=False,
        ),
    )
    scene.build()
    name_to_local = {link.name: i for i, link in enumerate(hsr.links)}
    link_indices = [name_to_local[n] for n in LINK_NAMES]
    _HSR = hsr
    _LINK_INDICES = link_indices
    return _HSR, _LINK_INDICES


def _build_qpos(hsr, arm_angles, torso_lift=0.0, base_xyyaw=None):
    qpos = torch.zeros(hsr.n_qs, dtype=gs.tc_float)
    arm_qs_idx = hsr._ensure_arm_qs_idx()
    torso_qs_idx = hsr._ensure_torso_qs_idx()
    base_qs_idx = hsr._ensure_base_qs_idx()

    if len(base_qs_idx) >= 7:
        if base_xyyaw is not None:
            bx, by, byaw = base_xyyaw
            hc = math.cos(byaw * 0.5)
            hs = math.sin(byaw * 0.5)
            qpos[base_qs_idx[0]] = bx
            qpos[base_qs_idx[1]] = by
            qpos[base_qs_idx[2]] = 0.0
            qpos[base_qs_idx[3]] = hc
            qpos[base_qs_idx[4]] = 0.0
            qpos[base_qs_idx[5]] = 0.0
            qpos[base_qs_idx[6]] = hs
        else:
            qpos[base_qs_idx[3]] = 1.0

    if torso_qs_idx is not None:
        qpos[torso_qs_idx] = torso_lift
    for i, val in enumerate(arm_angles):
        qpos[arm_qs_idx[i]] = val
    return qpos


def _quat_to_rpy(w, x, y, z):
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


class TestForwardKinematics:
    def test_home_pose(self):
        hsr, link_indices = _ensure_hsr()
        qpos = _build_qpos(hsr, [0.0, 0.0, 0.0, 0.0, 0.0])
        pos, quat = hsr.forward_kinematics(qpos, links_idx_local=link_indices)

        assert pos[0, 0].item() == pytest.approx(0.0, abs=1e-4)
        assert pos[0, 1].item() == pytest.approx(0.0, abs=1e-4)
        assert pos[0, 2].item() == pytest.approx(0.0, abs=1e-4)

        assert pos[1, 2].item() == pytest.approx(0.34, abs=1e-2)

        assert pos[5, 2].item() > 0.8

    def test_base_translation_only(self):
        hsr, link_indices = _ensure_hsr()
        qpos = _build_qpos(hsr, [0.0, 0.0, 0.0, 0.0, 0.0])

        pos, _ = hsr.forward_kinematics(
            qpos, links_idx_local=link_indices, base_xyyaw=(1.0, -0.5, 0.0)
        )

        assert pos[0, 0].item() == pytest.approx(1.0, abs=1e-4)
        assert pos[0, 1].item() == pytest.approx(-0.5, abs=1e-4)

    def test_base_rotation_only(self):
        hsr, link_indices = _ensure_hsr()
        qpos = _build_qpos(hsr, [0.0, 0.0, 0.0, 0.0, 0.0])

        pos, quat = hsr.forward_kinematics(
            qpos, links_idx_local=link_indices, base_xyyaw=(0.0, 0.0, math.pi / 2)
        )

        _, _, y = _quat_to_rpy(
            quat[0, 0].item(), quat[0, 1].item(), quat[0, 2].item(), quat[0, 3].item()
        )
        assert y == pytest.approx(math.pi / 2, abs=1e-4)

        assert pos[2, 0].item() == pytest.approx(-0.078, abs=1e-2)
        assert pos[2, 1].item() == pytest.approx(0.141, abs=1e-2)

    def test_arm_flex_lowers_ee_z(self):
        hsr, link_indices = _ensure_hsr()

        qpos_s = _build_qpos(hsr, [0.3, 0.0, 0.0, 0.0, 0.0])
        qpos_f = _build_qpos(hsr, [0.3, -1.5, 0.0, 0.0, 0.0])
        pos_s, _ = hsr.forward_kinematics(qpos_s, links_idx_local=link_indices)
        pos_f, _ = hsr.forward_kinematics(qpos_f, links_idx_local=link_indices)

        assert pos_f[5, 2].item() < pos_s[5, 2].item()

    def test_torso_lift_raises_torso_link(self):
        hsr, link_indices = _ensure_hsr()

        qpos_low = _build_qpos(hsr, [0.0, 0.0, 0.0, 0.0, 0.0], torso_lift=0.0)
        qpos_high = _build_qpos(hsr, [0.0, 0.0, 0.0, 0.0, 0.0], torso_lift=0.2)
        pos_low, _ = hsr.forward_kinematics(qpos_low, links_idx_local=link_indices)
        pos_high, _ = hsr.forward_kinematics(qpos_high, links_idx_local=link_indices)

        dz = pos_high[6, 2].item() - pos_low[6, 2].item()
        assert dz == pytest.approx(0.2, abs=1e-2)

    def test_base_xyyaw_with_arm_config(self):
        hsr, link_indices = _ensure_hsr()
        qpos = _build_qpos(hsr, [0.2, -0.5, 0.3, -0.2, 0.1])

        pos, quat = hsr.forward_kinematics(
            qpos, links_idx_local=link_indices, base_xyyaw=(0.1, 0.2, 0.5)
        )

        assert torch.isfinite(pos).all()
        assert torch.isfinite(quat).all()
        assert pos[0, 0].item() == pytest.approx(0.1, abs=1e-4)
        assert pos[0, 1].item() == pytest.approx(0.2, abs=1e-4)

    def test_deterministic(self):
        hsr, link_indices = _ensure_hsr()
        qpos = _build_qpos(hsr, [0.1, -0.4, 0.2, -0.1, 0.05])

        p1, q1 = hsr.forward_kinematics(qpos, links_idx_local=link_indices)
        p2, q2 = hsr.forward_kinematics(qpos, links_idx_local=link_indices)

        assert torch.equal(p1, p2)
        assert torch.equal(q1, q2)

    def test_fk_restores_solver_state(self):
        hsr, _ = _ensure_hsr()

        qpos_before = hsr.get_qpos().clone()
        qpos = _build_qpos(hsr, [0.3, -0.8, 0.5, -0.4, 0.2])

        hsr.forward_kinematics(qpos, links_idx_local=range(len(hsr.links)))

        qpos_after = hsr.get_qpos()
        assert torch.equal(qpos_before, qpos_after)

    def test_all_links_returned_when_no_links_idx(self):
        hsr, _ = _ensure_hsr()
        qpos = _build_qpos(hsr, [0.0, 0.0, 0.0, 0.0, 0.0])

        pos, quat = hsr.forward_kinematics(qpos)

        assert pos.shape[0] == hsr.n_links
        assert quat.shape[0] == hsr.n_links
        assert pos.shape[1] == 3
        assert quat.shape[1] == 4

    def test_forward_kinematics_base_planar_without_override(self):
        hsr, link_indices = _ensure_hsr()

        qpos = _build_qpos(hsr, [0.0, 0.0, 0.0, 0.0, 0.0])
        base_qs_idx = hsr._ensure_base_qs_idx()
        if len(base_qs_idx) >= 7:
            qpos[base_qs_idx[0]] = 2.0
            qpos[base_qs_idx[1]] = 1.0
            hc = math.cos(0.75 * 0.5)
            hs = math.sin(0.75 * 0.5)
            qpos[base_qs_idx[3]] = hc
            qpos[base_qs_idx[4]] = 0.0
            qpos[base_qs_idx[5]] = 0.0
            qpos[base_qs_idx[6]] = hs

        pos, quat = hsr.forward_kinematics(qpos, links_idx_local=link_indices)

        assert pos[0, 0].item() == pytest.approx(2.0, abs=1e-4)
        assert pos[0, 1].item() == pytest.approx(1.0, abs=1e-4)
        r, p, y = _quat_to_rpy(
            quat[0, 0].item(), quat[0, 1].item(), quat[0, 2].item(), quat[0, 3].item()
        )
        assert y == pytest.approx(0.75, abs=1e-4)

    def test_ee_pose_via_all_links(self):
        hsr, link_indices = _ensure_hsr()
        qpos = _build_qpos(hsr, [0.15, -0.4, 0.6, -0.3, 0.1])

        pos, quat = hsr.forward_kinematics(qpos, links_idx_local=link_indices)

        pos_all, quat_all = hsr.forward_kinematics(qpos)
        ee_idx = link_indices[-1]
        assert torch.equal(pos[-1], pos_all[ee_idx])
        assert torch.equal(quat[-1], quat_all[ee_idx])
