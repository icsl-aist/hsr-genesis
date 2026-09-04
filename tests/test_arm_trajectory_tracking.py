"""Whole-body arm trajectory tracking under gravity."""

from __future__ import annotations

from pathlib import Path

import genesis as gs
import torch

from hsr_genesis.analytic_ik import JOINT_ORDER
from hsr_genesis.hsr_rigid_entity import HSRBURDF, JointTrajectory


URDF_PATH = Path(__file__).resolve().parents[1] / "data" / "urdf" / "hsrb4s.urdf"
DT = 0.01
SUBSTEPS = 4

# Representative palm-down banana approach and descend IK solutions.
APPROACH_ARM = torch.tensor([0.1834, -1.9292, 0.0, -1.2124, 0.0719])
DESCEND_ARM = torch.tensor([0.0975, -2.0053, 0.0, -1.1362, 0.0609])
JOINT_TOLERANCE = torch.tensor([0.001, 0.005, 0.005, 0.005, 0.005])


def test_whole_body_trajectory_rejects_steady_gravity_disturbance() -> None:
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS),
        show_viewer=False,
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

    arm_dofs = robot._hsr_arm_dofs_idx_local
    robot.set_dofs_position(APPROACH_ARM.to(gs.device), dofs_idx_local=arm_dofs)
    robot.set_whole_body_trajectory_batched(
        arm_trajectory=JointTrajectory(
            positions=DESCEND_ARM.to(gs.device).unsqueeze(0),
            time_from_start=torch.tensor([5.0], device=gs.device),
            joint_names=list(JOINT_ORDER),
        ),
        base_trajectory=None,
        envs_idx=[0],
    )

    # Five-second move plus one-second endpoint hold. A longer hold must not be
    # required to eliminate a constant disturbance.
    for _ in range(600):
        robot.step_whole_body_trajectory_batched(DT, envs_idx=[0])
        scene.step()

    actual = robot.get_dofs_position(dofs_idx_local=arm_dofs, envs_idx=[0]).reshape(-1)
    error = (DESCEND_ARM.to(gs.device) - actual).abs()
    assert torch.all(error <= JOINT_TOLERANCE.to(gs.device)), (
        f"target={DESCEND_ARM.tolist()} actual={actual.tolist()} error={error.tolist()}"
    )
