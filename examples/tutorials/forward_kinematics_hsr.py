"""
Forward kinematics (FK) for the HSR robot using RigidEntity.forward_kinematics.

Demonstrates the overridden forward_kinematics on HSRRigidEntity with
built-in base (x, y, yaw) support.
"""

import sys
import math
from pathlib import Path

import numpy as np
import torch

import genesis as gs

sys.path.insert(0, str(Path(__file__).resolve().parent))

ARM_JOINT_NAMES = [
    "arm_lift_joint",
    "arm_flex_joint",
    "arm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
]

LINK_NAMES = [
    "base_footprint",
    "arm_lift_link",
    "arm_flex_link",
    "arm_roll_link",
    "wrist_flex_link",
    "hand_palm_link",
]

URDF_PATH = Path(__file__).resolve().parents[2] / "data" / "urdf" / "hsrb4s.urdf"


def _quat_to_rpy(w: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _mat4_from_pos_quat(
    pos: torch.Tensor, quat_wxyz: torch.Tensor
) -> np.ndarray:
    q = quat_wxyz.detach().cpu().numpy()
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ]
    )
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos.detach().cpu().numpy().flatten()
    return T


def _build_qpos(
    hsr,
    arm_angles: list[float],
    torso_lift: float,
) -> torch.Tensor:
    qpos = torch.zeros(hsr.n_qs, dtype=gs.tc_float)

    arm_qs_idx = hsr._ensure_arm_qs_idx()
    torso_qs_idx = hsr._ensure_torso_qs_idx()

    if torso_qs_idx is not None:
        qpos[torso_qs_idx] = torso_lift

    for i, val in enumerate(arm_angles):
        qpos[arm_qs_idx[i]] = val

    return qpos


def _fk_for_config(
    hsr,
    arm_angles: list[float],
    link_indices: list[int],
    *,
    torso_lift: float = 0.0,
    base_xyyaw: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    print()
    print("=" * 72)
    parts = [f"{n}: {v:+.3f}" for n, v in zip(ARM_JOINT_NAMES, arm_angles)]
    print("Arm joints:  " + "  |  ".join(parts))
    print(f"Torso lift: {torso_lift:+.3f} m  |  Base x,y,yaw: {base_xyyaw}")
    print("-" * 72)

    qpos = _build_qpos(hsr, arm_angles, torso_lift)

    links_pos, links_quat = hsr.forward_kinematics(
        qpos, links_idx_local=link_indices, base_xyyaw=base_xyyaw
    )

    print(f"{'Link':<24} {'Position (m)':<30} {'Orientation RPY (rad)'}")
    print("-" * 72)
    for i, name in enumerate(LINK_NAMES):
        pos = links_pos[i].cpu().numpy()
        q = links_quat[i].cpu().numpy()
        r, p, y = _quat_to_rpy(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        print(
            f"  {name:<22}"
            f" [{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]   "
            f" [{r:+.4f}, {p:+.4f}, {y:+.4f}]"
        )

    T = _mat4_from_pos_quat(links_pos[-1], links_quat[-1])
    print()
    print("End-effector (hand_palm_link) world transform:")
    for row in T:
        print(f"  [{row[0]:+.4f}, {row[1]:+.4f}, {row[2]:+.4f}, {row[3]:+.4f}]")


def main() -> None:
    gs.init(backend=gs.cpu)

    from hsr_genesis.hsr_rigid_entity import HSRBURDF

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

    print("HSR Forward Kinematics Demo\n")

    configs = [
        ([0.0, 0.0, 0.0, 0.0, 0.0], 0.0, (0.0, 0.0, 0.0), "home"),
        ([0.25, -0.8, 0.0, -0.4, 0.0], 0.15, (0.0, 0.0, 0.0), "arm lifted forward"),
        ([0.35, -1.2, 0.5, 0.8, 0.0], 0.2, (0.0, 0.0, 0.0), "arm high wrist down"),
        ([0.15, -0.3, 1.5, -0.6, 0.3], 0.0, (0.5, 0.0, 0.3), "side reach + base turn"),
    ]

    for arm, torso, base, label in configs:
        print(f"[{label}]")
        _fk_for_config(hsr, arm, link_indices, torso_lift=torso, base_xyyaw=base)

    print()
    print("=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()
