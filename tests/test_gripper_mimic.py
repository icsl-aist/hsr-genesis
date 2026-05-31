"""Test that Genesis native URDF <mimic> constraints work for the gripper hand.

Commands only hand_motor_joint and checks that all mimic joints follow
the expected multiplier+offset relationship WITHOUT using the manual
mirroring code in gripper_controller.py — this tests the native Genesis
equality-constraint solver path only.

After verifying baseline, the manual mirror code is removed and this test
is re-run to confirm Genesis native mimic alone is sufficient.

Run with the repo venv:
    PYTHONPATH=src .venv/bin/python -m pytest tests/test_gripper_mimic.py -v -s
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator

import pytest
import torch

import genesis as gs

if not getattr(gs, "_initialized", False):
    gs.init(backend=gs.cpu, precision="32", logging_level="warning")

from hsr_genesis.hsr_rigid_entity import HSRBURDF

URDF_PATH = Path(__file__).resolve().parents[1] / "data" / "urdf" / "hsrb4s.urdf"

DT = 0.01
SUBSTEPS = 4
SETTLE_STEPS = 50

MIMIC_TOLERANCE_RAD = math.radians(0.2)

MOTOR_TEST_POSITIONS = [0.0, 0.3, 0.5, 0.8]

MIMIC_RELATIONS = [
    ("hand_l_proximal_joint", "hand_motor_joint", 1.0, 0.0),
    ("hand_r_proximal_joint", "hand_motor_joint", 1.0, 0.0),
    ("hand_l_distal_joint", "hand_motor_joint", -1.0, -0.087),
    ("hand_r_distal_joint", "hand_motor_joint", -1.0, -0.087),
    ("hand_l_mimic_distal_joint", "hand_l_spring_proximal_joint", -1.0, 0.0),
    ("hand_r_mimic_distal_joint", "hand_r_spring_proximal_joint", -1.0, 0.0),
]

PASSIVE_JOINTS = [
    "hand_l_spring_proximal_joint",
    "hand_r_spring_proximal_joint",
]

JOINT_LIMITS_ARM = {
    "arm_lift_joint":   (0.0,   0.69),
    "arm_flex_joint":   (-2.62, 0.0),
    "arm_roll_joint":   (-2.09, 3.84),
    "wrist_flex_joint": (-1.92, 1.22),
    "wrist_roll_joint": (-1.92, 3.67),
}


def _build_scene(show_viewer: bool = False) -> tuple[gs.Scene, HSRBURDF]:
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, 0.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    robot: HSRBURDF = scene.add_entity(
        HSRBURDF(
            file=str(URDF_PATH),
            robot="hsrb",
            base_mode="planar",
            end_effector_frame="hand_palm_link",
            use_base_controller=True,
            base_control_mode="controller",
            fixed=False,
            recompute_inertia=False,
            pos=(0.0, 0.0, 0.05),
        ),
    )
    scene.build()
    return scene, robot


def _arm_neutral() -> torch.Tensor:
    return torch.tensor(
        [(lo + hi) / 2.0 for lo, hi in JOINT_LIMITS_ARM.values()],
        device=gs.device,
        dtype=gs.tc_float,
    )


def _get_dof_idx(entity, joint_name: str) -> int:
    dofs = entity.get_joint(joint_name).dofs_idx_local
    if isinstance(dofs, (list, tuple)):
        return int(dofs[0])
    return int(dofs)


@pytest.fixture(scope="module")
def scene_and_robot(request) -> Iterator[tuple[gs.Scene, HSRBURDF]]:
    show = request.config.getoption("--visualize", default=False)
    scene, robot = _build_scene(show_viewer=show)
    robot._hsr_apply_default_gains()

    arm_dofs = robot._hsr_arm_dofs_idx_local
    neutral = _arm_neutral()
    robot.set_dofs_position(neutral, dofs_idx_local=arm_dofs)
    for _ in range(100):
        scene.step()

    yield scene, robot


def test_native_mimic_follows_motor(scene_and_robot):
    """Command ONLY hand_motor_joint, verify all mimic joints follow.

    This bypasses the manual mirror code in gripper_controller.py entirely
    by calling control_dofs_position directly on the motor DOF.

    The spring proximal joints are NOT expected to stay at 0 — they are
    loaded by gravity acting on the distal finger links (see the parallel
    linkage geometry).  The test only checks that the *mimic* relationships
    hold exactly between parent and child joints.
    """
    scene, robot = scene_and_robot
    motor_idx = _get_dof_idx(robot, "hand_motor_joint")

    all_joints = {"hand_motor_joint"}
    for child, parent, *_ in MIMIC_RELATIONS:
        all_joints.add(child)
        all_joints.add(parent)
    for j in PASSIVE_JOINTS:
        all_joints.add(j)

    dof_indices = {name: _get_dof_idx(robot, name) for name in all_joints}

    for motor_cmd in MOTOR_TEST_POSITIONS:
        motor_t = torch.tensor([motor_cmd], device=gs.device, dtype=gs.tc_float)
        robot.control_dofs_position(
            motor_t, dofs_idx_local=[motor_idx], envs_idx=[0]
        )

        for _ in range(SETTLE_STEPS):
            scene.step()

        positions = {}
        for name, idx in dof_indices.items():
            pos = robot.get_dofs_position(
                dofs_idx_local=[idx], envs_idx=[0]
            )
            positions[name] = float(pos.reshape(-1)[0].item())

        print(
            f"\n  motor_cmd={motor_cmd:.3f}  actual={positions['hand_motor_joint']:.4f}"
        )

        for child, parent, mult, offset in MIMIC_RELATIONS:
            expected = mult * positions[parent] + offset
            error = abs(positions[child] - expected)
            print(
                f"    {child:35s} = {positions[child]:.4f}  "
                f"(expected {expected:.4f}, err={error:.4f})"
            )
            assert error <= MIMIC_TOLERANCE_RAD, (
                f"motor_cmd={motor_cmd:.3f}: {child}={positions[child]:.4f}, "
                f"expected {expected:.4f} (from {parent}={positions[parent]:.4f} "
                f"× {mult} + {offset}), error={error:.4f}"
            )

        for j in PASSIVE_JOINTS:
            print(f"    {j:35s} = {positions[j]:.4f}  (passive, gravity-loaded)")

        print(
            f"    {'hand_l_mimic_distal_joint':35s} = {positions['hand_l_mimic_distal_joint']:.4f}  "
            f"(spring mimic, err={abs(positions['hand_l_mimic_distal_joint'] - (-1.0 * positions['hand_l_spring_proximal_joint'] + 0.0)):.4f})"
        )
        print(
            f"    {'hand_r_mimic_distal_joint':35s} = {positions['hand_r_mimic_distal_joint']:.4f}  "
            f"(spring mimic, err={abs(positions['hand_r_mimic_distal_joint'] - (-1.0 * positions['hand_r_spring_proximal_joint'] + 0.0)):.4f})"
        )
