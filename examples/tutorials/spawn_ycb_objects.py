"""Spawn YCB object models (from the tmc_wrs_gazebo submodule) around the HSR
and pick a random object.

This example demonstrates the SDF -> URDF converter in
``hsr_genesis.sdf_parser``:
Gazebo SDF models are converted on the fly and added to a Genesis scene at
random poses around the robot.  After letting the objects settle on the ground
plane, a random target is chosen and the robot drives + reaches + grasps +
lifts it using the whole-body trajectory controller and the apply-force
gripper.

Run:
    PYTHONPATH=src .venv/bin/python examples/tutorials/spawn_ycb_objects.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

import genesis as gs

sys.path.insert(0, str(Path(__file__).resolve().parent))

parser = argparse.ArgumentParser()
parser.add_argument(
    "--steps", type=int, default=0,
    help="Number of sim steps (0 = run forever)",
)
parser.add_argument(
    "--no-viewer", action="store_true",
    help="Disable the viewer window",
)
parser.add_argument(
    "--settle-steps", type=int, default=200,
    help="Steps to let objects settle before picking",
)
parser.add_argument(
    "--grasp-effort", type=float, default=3.0,
    help="Gripper closing force (N)",
)
parser.add_argument(
    "--seed", type=int, default=42,
    help="Random seed for object placement and target selection",
)
args = parser.parse_args()

URDF_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "urdf" / "hsrb4s.urdf"
)
MODELS_DIR = (
    Path(__file__).resolve().parents[2]
    / "data" / "tmc_wrs_gazebo" / "tmc_wrs_gazebo_worlds" / "models"
)

# A small, visually varied subset of the YCB object set.
YCB_MODELS = [
    "ycb_013_apple",
    "ycb_011_banana",
    "ycb_017_orange",
    "ycb_005_tomato_soup_can",
    "ycb_010_potted_meat_can",
    "ycb_077_rubiks_cube",
    "ycb_056_tennis_ball",
    "ycb_055_baseball",
    "ycb_061_foam_brick",
]

# Grasp approach offsets (meters) relative to the object center.
PRE_GRASP_HEIGHT = 0.15   # hover above the object before descending
GRASP_OFFSET_Z = 0.02     # final grasp height above object center
LIFT_HEIGHT = 0.30        # lift height after grasping


# ---------------------------------------------------------------------------
# Helpers (mirrors examples/tutorials/IK_grasp_hsr.py)
# ---------------------------------------------------------------------------

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


def _qpos_to_arm_dofs(
    entity, qpos: torch.Tensor, arm_dofs_idx_local: list[int],
) -> torch.Tensor:
    saved_qpos = entity.get_qpos().clone()
    try:
        entity.set_qpos(qpos, zero_velocity=False)
        dofs = entity.get_dofs_position()
    finally:
        entity.set_qpos(saved_qpos, zero_velocity=False)
    if dofs.ndim == 1:
        dofs = dofs.unsqueeze(0)
    return dofs[:, arm_dofs_idx_local]


def _arm_traj_names() -> list[str]:
    from hsr_genesis.analytic_ik import JOINT_ORDER

    return list(JOINT_ORDER)


def _entity_pos(ent) -> np.ndarray:
    """Get the 3D position of an entity as a numpy array (handles batched)."""
    pos = ent.get_pos()
    if isinstance(pos, torch.Tensor):
        pos = pos.detach().cpu().numpy()
    if pos.ndim > 1:
        pos = pos[0]
    return np.asarray(pos[:3], dtype=np.float32)


def _run_trajectory(
    hsr, scene, *,
    arm_traj, base_traj, dt, duration,
    motor_idx=None, hand_cmd=None,
):
    """Execute a whole-body trajectory for ``duration`` seconds.

    If ``motor_idx`` / ``hand_cmd`` are provided, the hand motor is controlled
    with ``hand_cmd`` position on the first step (used to open the hand during
    approach).
    """
    hsr.set_whole_body_trajectory_batched(
        arm_trajectory=arm_traj,
        base_trajectory=base_traj,
        envs_idx=[0],
        start_time=None,
    )
    n_steps = int(duration / dt) + 50
    for step in range(n_steps):
        hsr.step_whole_body_trajectory_batched(dt, envs_idx=[0])
        if step == 0 and motor_idx is not None and hand_cmd is not None:
            hsr.control_dofs_position(hand_cmd, dofs_idx_local=[motor_idx])
        scene.step()


def _run_gripper_hold(hsr, scene, gripper, *, dt, n_steps):
    """Hold gripper closing force for ``n_steps`` while keeping arm still."""
    for _ in range(n_steps):
        gripper.step_apply_force(dt, envs_idx=[0])
        hsr.step_whole_body_trajectory_batched(dt, envs_idx=[0])
        scene.step()


# ---------------------------------------------------------------------------
# Scene setup & spawning
# ---------------------------------------------------------------------------

def _init_genesis() -> None:
    """Initialize Genesis, falling back to CPU if GPU is unavailable."""
    try:
        gs.init(backend=gs.gpu)
    except RuntimeError as exc:  # pragma: no cover - demo fallback
        print(
            f"[Genesis] GPU backend unavailable ({exc});"
            " falling back to CPU."
        )
        gs.init(backend=gs.cpu)


def _build_scene():
    """Create the Genesis scene with viewer/vis/sim options."""
    return gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, -2.0, 2.0),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
            max_FPS=60,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
            world_frame_size=1.0,
            show_link_frame=False,
            plane_reflection=True,
            ambient_light=(0.3, 0.3, 0.3),
        ),
        sim_options=gs.options.SimOptions(dt=0.02, substeps=10),
        rigid_options=gs.options.RigidOptions(
            use_gjk_collision=True,
        ),
        show_viewer=not args.no_viewer,
    )


def _spawn_ycb_objects(scene, rng):
    """Spawn YCB models at random poses around the origin.

    Returns a list of ``(name, entity)`` tuples.
    """
    from hsr_genesis.sdf_parser import load_sdf_model

    objects = []
    for name in YCB_MODELS:
        model_dir = MODELS_DIR / name
        if not model_dir.exists():
            print(f"[skip] {name} not found (submodule not initialized?)")
            continue

        theta = rng.uniform(0.0, 2.0 * math.pi)
        radius = rng.uniform(0.6, 1.6)
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        z = rng.uniform(0.05, 0.25)
        yaw = rng.uniform(0.0, 2.0 * math.pi)

        robot = load_sdf_model(model_dir)
        entity = scene.add_entity(
            gs.morphs.URDF(
                file=robot,
                pos=(x, y, z),
                euler=(0.0, 0.0, math.degrees(yaw)),
                fixed=False,
            ),
        )
        objects.append((name, entity))
        print(
            f"[spawn] {name:30s} at "
            f"({x:+.2f}, {y:+.2f}, {z:+.2f}) yaw={yaw:+.2f}"
        )
    return objects


# ---------------------------------------------------------------------------
# Pick sequence
# ---------------------------------------------------------------------------

def _pick_object(
    hsr, scene, *, target_name, target_entity, target_pos, dt,
):
    """Run the full approach-grasp-lift sequence on the target object.

    Returns the final object position after lifting.
    """
    from hsr_genesis.hsr_rigid_entity import JointTrajectory
    from hsr_genesis.base_controller import Trajectory
    end_effector = hsr.get_link("hand_palm_link")
    hsr.end_effector_offset = [0.0, 0.0, 0.09]
    hand_quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    arm_dofs_idx_local = _arm_dofs_idx_local(hsr)
    motor_dofs = hsr.get_joint("hand_motor_joint").dofs_idx_local
    motor_idx = (
        int(motor_dofs[0])
        if isinstance(motor_dofs, (list, tuple))
        else int(motor_dofs)
    )
    hand_open = torch.tensor([[1.0]], device=gs.device, dtype=gs.tc_float)
    gripper = hsr.get_gripper_batched()

    def ik_and_trajectory(goal_pos, *, duration, base=True, init_qpos=None):
        """Run IK to ``goal_pos`` and build arm/base trajectories."""
        qpos = hsr.inverse_kinematics(
            link=end_effector,
            pos=goal_pos,
            quat=hand_quat,
            init_qpos=init_qpos,
            max_samples=200,
            max_solver_iters=150,
            max_step_size=0.7,
            respect_joint_limit=False,
        )
        arm_dofs = _qpos_to_arm_dofs(hsr, qpos, arm_dofs_idx_local)
        target_x = float(qpos[0])
        target_y = float(qpos[1])
        target_yaw = _quat_wxyz_to_yaw(qpos[3:7])

        base_traj = None
        if base:
            base_traj = Trajectory(
                positions=torch.tensor(
                    [[target_x, target_y, target_yaw]],
                    device=gs.device, dtype=gs.tc_float,
                ),
                time_from_start=torch.tensor(
                    [duration], device=gs.device, dtype=gs.tc_float,
                ),
            )
        arm_traj = JointTrajectory(
            positions=arm_dofs,
            time_from_start=torch.tensor(
                [duration], device=gs.device, dtype=gs.tc_float,
            ),
            joint_names=_arm_traj_names(),
        )
        return arm_traj, base_traj, qpos

    # --- Phase 3: approach (pre-grasp hover above the object) ---
    pre_grasp_pos = target_pos.copy()
    pre_grasp_pos[2] += PRE_GRASP_HEIGHT
    print(
        f"[approach] pre-grasp pos=({pre_grasp_pos[0]:+.2f},"
        f" {pre_grasp_pos[1]:+.2f}, {pre_grasp_pos[2]:+.2f})"
    )
    approach_duration = 4.0
    arm_traj, base_traj, _ = ik_and_trajectory(
        pre_grasp_pos, duration=approach_duration, base=True,
    )
    _run_trajectory(
        hsr, scene,
        arm_traj=arm_traj, base_traj=base_traj,
        dt=dt, duration=approach_duration,
        motor_idx=motor_idx, hand_cmd=hand_open,
    )

    # --- Phase 4: descend to grasp pose ---
    grasp_pos = target_pos.copy()
    grasp_pos[2] += GRASP_OFFSET_Z
    print(
        f"[descend] grasp pos=({grasp_pos[0]:+.2f},"
        f" {grasp_pos[1]:+.2f}, {grasp_pos[2]:+.2f})"
    )
    descend_duration = 2.0
    current_qpos = hsr.get_qpos().clone()
    arm_traj, base_traj, _ = ik_and_trajectory(
        grasp_pos, duration=descend_duration,
        base=False, init_qpos=current_qpos,
    )
    _run_trajectory(
        hsr, scene,
        arm_traj=arm_traj, base_traj=base_traj,
        dt=dt, duration=descend_duration,
    )

    # --- Phase 5: close gripper ---
    print(f"[grasp] closing gripper (effort={args.grasp_effort} N)...")
    effort = torch.tensor(
        [args.grasp_effort], device=gs.device, dtype=gs.tc_float,
    )
    active = torch.tensor([True], device=gs.device, dtype=torch.bool)
    gripper.set_apply_force_goal(
        effort=effort, active_mask=active, envs_idx=[0],
    )
    _run_gripper_hold(hsr, scene, gripper, dt=dt, n_steps=300)

    # --- Phase 6: lift ---
    lift_pos = target_pos.copy()
    lift_pos[2] = LIFT_HEIGHT
    print(
        f"[lift] lift pos=({lift_pos[0]:+.2f},"
        f" {lift_pos[1]:+.2f}, {lift_pos[2]:+.2f})"
    )
    lift_duration = 2.0
    current_qpos = hsr.get_qpos().clone()
    arm_traj, base_traj, _ = ik_and_trajectory(
        lift_pos, duration=lift_duration,
        base=False, init_qpos=current_qpos,
    )
    hsr.set_whole_body_trajectory_batched(
        arm_trajectory=arm_traj,
        base_trajectory=None,
        envs_idx=[0],
        start_time=None,
    )
    lift_steps = int(lift_duration / dt) + 50
    _run_gripper_hold(hsr, scene, gripper, dt=dt, n_steps=lift_steps)

    # Hold and report.
    _run_gripper_hold(hsr, scene, gripper, dt=dt, n_steps=100)
    final_obj_pos = _entity_pos(target_entity)
    print(
        f"\n[done] {target_name} final pos="
        f"({final_obj_pos[0]:+.2f}, {final_obj_pos[1]:+.2f},"
        f" {final_obj_pos[2]:+.2f})"
    )
    if final_obj_pos[2] > target_pos[2] + 0.05:
        delta = final_obj_pos[2] - target_pos[2]
        print(f"[result] SUCCESS: object lifted by {delta:.2f} m")
    else:
        delta = final_obj_pos[2] - target_pos[2]
        print(f"[result] object barely moved (delta_z={delta:+.2f} m)")

    return final_obj_pos, gripper


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _init_genesis()

    from hsr_genesis.hsr_rigid_entity import HSRBURDF, JointTrajectory
    from hsr_genesis.base_controller import Trajectory

    scene = _build_scene()

    # Ground plane.
    scene.add_entity(gs.morphs.Plane())

    # HSR robot.
    hsr = scene.add_entity(
        HSRBURDF(
            file=str(URDF_PATH),
            fixed=False,
            recompute_inertia=True,
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

    # Spawn YCB objects at random positions around the robot.
    rng = np.random.default_rng(seed=args.seed)
    objects = _spawn_ycb_objects(scene, rng)

    scene.build()

    print(f"\nSpawned {len(objects)} YCB objects around the HSR.")

    dt = float(scene.sim_options.dt)

    # --- Phase 1: let objects settle on the ground ---
    # Hold the robot joints at their initial positions while objects fall.
    # Set a hold trajectory (single waypoint at the current arm pose) so
    # that step_whole_body_trajectory_batched keeps the arm, head, and
    # base in place.  Without this the arm droops and the head flops due
    # to gravity during the settle period.
    arm_dofs_idx_local = _arm_dofs_idx_local(hsr)
    initial_arm_pos = hsr.get_dofs_position(
        dofs_idx_local=arm_dofs_idx_local,
    )
    if initial_arm_pos.ndim == 1:
        initial_arm_pos = initial_arm_pos.unsqueeze(0)
    initial_base_pos = hsr.get_pos()
    if initial_base_pos.ndim == 1:
        initial_base_xy = initial_base_pos[:2].reshape(1, 2)
    else:
        initial_base_xy = initial_base_pos[:, :2]
    # Base yaw from quaternion
    initial_qpos = hsr.get_qpos()
    if initial_qpos.ndim == 1:
        initial_yaw = _quat_wxyz_to_yaw(initial_qpos[3:7])
    else:
        initial_yaw = _quat_wxyz_to_yaw(initial_qpos[0, 3:7])

    hold_arm_traj = JointTrajectory(
        positions=initial_arm_pos,
        time_from_start=torch.tensor(
            [args.settle_steps * dt], device=gs.device, dtype=gs.tc_float,
        ),
        joint_names=_arm_traj_names(),
    )
    hold_base_traj = Trajectory(
        positions=torch.tensor(
            [[float(initial_base_xy[0, 0]), float(initial_base_xy[0, 1]),
              initial_yaw]],
            device=gs.device, dtype=gs.tc_float,
        ),
        time_from_start=torch.tensor(
            [args.settle_steps * dt], device=gs.device, dtype=gs.tc_float,
        ),
    )
    hsr.set_whole_body_trajectory_batched(
        arm_trajectory=hold_arm_traj,
        base_trajectory=hold_base_traj,
        envs_idx=[0],
        start_time=None,
    )

    print(
        f"[settle] running {args.settle_steps} steps"
        " to let objects settle..."
    )
    for _ in range(args.settle_steps):
        hsr.step_whole_body_trajectory_batched(dt, envs_idx=[0])
        scene.step()
    print("[settle] object positions after settling:")
    for name, ent in objects:
        pos = _entity_pos(ent)
        print(
            f"         {name:30s} pos="
            f"({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})"
        )

    # --- Phase 2: pick a random target object ---
    target_idx = int(rng.integers(len(objects)))
    target_name, target_entity = objects[target_idx]
    target_pos = _entity_pos(target_entity)
    print(
        f"\n[pick] target: {target_name} at "
        f"({target_pos[0]:+.2f}, {target_pos[1]:+.2f}, {target_pos[2]:+.2f})"
    )

    # --- Phases 3-6: approach, descend, grasp, lift ---
    _, gripper = _pick_object(
        hsr, scene,
        target_name=target_name,
        target_entity=target_entity,
        target_pos=target_pos,
        dt=dt,
    )

    # --- Phase 7: keep simulating (viewer loop) ---
    print("\nContinuing simulation. Close the viewer window to exit.")
    n_steps = 0
    while True:
        gripper.step_apply_force(dt, envs_idx=[0])
        hsr.step_whole_body_trajectory_batched(dt, envs_idx=[0])
        scene.step()
        n_steps += 1
        if n_steps % 200 == 0:
            pos = _entity_pos(target_entity)
            print(
                f"[{n_steps:5d}] {target_name:30s} pos="
                f"({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})"
            )
        if args.steps > 0 and n_steps >= args.steps:
            break


if __name__ == "__main__":
    main()
