from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

import genesis as gs

sys.path.insert(0, str(Path(__file__).resolve().parent))


URDF_PATH = Path(__file__).resolve().parents[2] / "data" / "urdf" / "hsrb4s.urdf"


def _quat_wxyz_to_yaw(quat: torch.Tensor | np.ndarray) -> float:
    if isinstance(quat, torch.Tensor):
        quat_val = quat.detach().cpu().numpy()
    else:
        quat_val = np.asarray(quat, dtype=np.float64)
    w, x, y, z = quat_val[:4]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _arm_dofs_idx_local(entity) -> list[int]:
    from hsr_genesis.analytic_ik import JOINT_ORDER

    dofs: list[int] = []
    for name in JOINT_ORDER:
        joint_dofs = entity.get_joint(name).dofs_idx_local
        if isinstance(joint_dofs, (list, tuple)):
            dofs.extend(int(idx) for idx in joint_dofs)
        else:
            dofs.append(int(joint_dofs))
    return dofs


def _qpos_to_arm_dofs(entity, qpos: torch.Tensor, arm_dofs_idx_local: list[int]) -> torch.Tensor:
    saved_qpos = entity.get_qpos().clone()
    try:
        entity.set_qpos(qpos, zero_velocity=False)
        dofs = entity.get_dofs_position()
    finally:
        entity.set_qpos(saved_qpos, zero_velocity=False)
    if dofs.ndim == 1:
        dofs = dofs.unsqueeze(0)
    return dofs[:, arm_dofs_idx_local]


def arm_traj_names() -> list[str]:
    from hsr_genesis.analytic_ik import JOINT_ORDER

    return list(JOINT_ORDER)


def main() -> None:
    gs.init(backend=gs.gpu)
    from hsr_genesis.hsr_rigid_entity import HSRBURDF, JointTrajectory
    from hsr_genesis.base_controller import Trajectory

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(dt=0.02, substeps=10),
        rigid_options=gs.options.RigidOptions(
            iterations=100,
            ls_iterations=100,
            noslip_iterations=5,
            noslip_tolerance=1e-7,
            constraint_timeconst=0.005,
            use_gjk_collision=True,
        ),
        show_viewer=True,
    )

    scene.add_entity(gs.morphs.Plane(), visualize_contact=True)

    cube_pos = np.array([0.45, 0.0, 0.02], dtype=np.float32)
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=tuple(cube_pos.tolist()),
        ),
        visualize_contact=True,
    )

    hsr = scene.add_entity(
        HSRBURDF(
            file=URDF_PATH,
            fixed=False,
            recompute_inertia=False,
            links_to_keep=["hand_palm_link"],
            robot="hsrb",
            base_mode="planar",
            end_effector_frame="hand_palm_link",
            use_base_controller=True,
            base_control_mode="controller",
            optimizer="gpu",
        ),
        visualize_contact=True,
    )

    scene.build()

    end_effector = hsr.get_link("hand_palm_link")
    hsr.end_effector_offset = [0.0, 0.0, 0.09]
    hand_quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    qpos = hsr.inverse_kinematics(
        link=end_effector,
        pos=cube_pos,
        quat=hand_quat,
        max_samples=200,
        max_solver_iters=150,
        max_step_size=0.7,
        respect_joint_limit=False,
    )

    arm_dofs_idx_local = _arm_dofs_idx_local(hsr)
    arm_dofs = _qpos_to_arm_dofs(hsr, qpos, arm_dofs_idx_local)

    target_pos = (float(qpos[0]), float(qpos[1]), float(qpos[2]))
    target_yaw = _quat_wxyz_to_yaw(qpos[3:7])

    dt = float(scene.sim_options.dt)
    duration = 3.0

    base_traj = Trajectory(
        positions=torch.tensor([[target_pos[0], target_pos[1], target_yaw]], device=gs.device, dtype=gs.tc_float),
        time_from_start=torch.tensor([duration], device=gs.device, dtype=gs.tc_float),
    )
    arm_traj = JointTrajectory(
        positions=arm_dofs,
        time_from_start=torch.tensor([duration], device=gs.device, dtype=gs.tc_float),
        joint_names=arm_traj_names(),
    )

    hsr.set_whole_body_trajectory_batched(
        arm_trajectory=arm_traj,
        base_trajectory=base_traj,
        envs_idx=[0],
        start_time=None,
    )

    # --- Approach: move arm to pre-grasp pose with hand open ---
    motor_dofs = hsr.get_joint("hand_motor_joint").dofs_idx_local
    motor_idx = int(motor_dofs[0]) if isinstance(motor_dofs, (list, tuple)) else int(motor_dofs)
    hand_open = torch.tensor([[1.0]], device=gs.device, dtype=gs.tc_float)

    max_steps = int(duration / dt) + 50
    for step in range(max_steps):
        hsr.step_whole_body_trajectory_batched(dt, envs_idx=[0])
        if step == 0:
            hsr.control_dofs_position(hand_open, dofs_idx_local=[motor_idx])
        scene.step()

    # --- Close gripper using apply-force action for torque-controlled grasp ---
    gripper = hsr.get_gripper_batched()
    effort = torch.tensor([3.0], device=gs.device, dtype=gs.tc_float)
    active = torch.tensor([True], device=gs.device, dtype=torch.bool)
    gripper.set_apply_force_goal(effort=effort, active_mask=active, envs_idx=[0])

    for _ in range(300):
        gripper.step_apply_force(dt, envs_idx=[0])
        hsr.step_whole_body_trajectory_batched(dt, envs_idx=[0])
        scene.step()

    # --- Lift: raise grasped object ---
    current_qpos = hsr.get_qpos().clone()
    lift_height = 0.25
    lift_pos = np.array([cube_pos[0], cube_pos[1], lift_height], dtype=np.float32)

    lift_qpos = hsr.inverse_kinematics(
        link=end_effector,
        pos=lift_pos,
        quat=hand_quat,
        init_qpos=current_qpos,
        max_samples=200,
        max_solver_iters=150,
        max_step_size=0.7,
        respect_joint_limit=False,
    )

    lift_arm_dofs = _qpos_to_arm_dofs(hsr, lift_qpos, arm_dofs_idx_local)

    lift_duration = 2.0
    lift_arm_traj = JointTrajectory(
        positions=lift_arm_dofs,
        time_from_start=torch.tensor([lift_duration], device=gs.device, dtype=gs.tc_float),
        joint_names=arm_traj_names(),
    )

    hsr.set_whole_body_trajectory_batched(
        arm_trajectory=lift_arm_traj,
        base_trajectory=None,
        envs_idx=[0],
        start_time=None,
    )

    lift_steps = int(lift_duration / dt) + 50
    for step in range(lift_steps):
        gripper.step_apply_force(dt, envs_idx=[0])
        hsr.step_whole_body_trajectory_batched(dt, envs_idx=[0])
        scene.step()

    for _ in range(100):
        gripper.step_apply_force(dt, envs_idx=[0])
        hsr.step_whole_body_trajectory_batched(dt, envs_idx=[0])
        scene.step()


if __name__ == "__main__":
    main()
