"""Simplified HSR control API for tutorial notebooks.

This module wraps the Genesis + hsr_genesis APIs into a small set of
novice-friendly functions inspired by the ``utils.py`` from
https://github.com/hsr-project/notebooks.

Typical usage in a notebook::

    from hsr_genesis.tutorial_utils import *

    init_sim()
    move_base_vel(0.1, 0, 0)
    run(2.0)
    show_video()

    move_hand(1.0)          # open hand
    run(1.0)
    show_video()

All angles passed to public functions are in **degrees** (matching the
reference notebook convention) unless stated otherwise.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

import genesis as gs

# ---------------------------------------------------------------------------
# Module-level state (singleton, mirrors the reference utils.py pattern)
# ---------------------------------------------------------------------------


class _State:
    """Holds all simulation state so tutorial functions don't need globals."""

    scene = None
    hsr = None
    cam = None
    dt: float = 0.02
    frames: list = []
    base_vel_cmd: tuple[float, float, float] | None = None  # (vx, vy, vw_rad)
    gripper = None
    gripper_active: bool = False
    motor_idx: int | None = None
    arm_dofs_idx: list[int] = []
    end_effector = None
    ft_sensor = None
    urdf_path: str | None = None
    head_idx: int | None = None
    built: bool = False


_state = _State()


# Named arm poses (joint order: arm_lift, arm_flex, arm_roll, wrist_flex, wrist_roll)
ARM_NEUTRAL = [0.0, -0.5, 0.0, -0.3, 0.0]
ARM_INIT = [0.0, 0.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clear_sim_state() -> None:
    """Reset all simulation state to defaults.

    Shared internal path used by both ``init_sim`` (when rebuilding) and
    ``reset_sim`` (explicit tear-down).
    """
    _state.scene = None
    _state.hsr = None
    _state.cam = None
    _state.dt = 0.02
    _state.frames.clear()
    _state.base_vel_cmd = None
    _state.gripper = None
    _state.gripper_active = False
    _state.motor_idx = None
    _state.arm_dofs_idx.clear()
    _state.end_effector = None
    _state.head_idx = None
    _state.built = False


def _find_urdf() -> Path:
    """Locate hsrb4s.urdf relative to the package, repo, or Colab clone."""
    import hsr_genesis

    pkg_dir = Path(hsr_genesis.__file__).resolve().parent
    candidates = [
        pkg_dir.parent.parent / "data" / "urdf" / "hsrb4s.urdf",  # repo root (editable install)
        pkg_dir / "data" / "urdf" / "hsrb4s.urdf",
        Path("/content/hsr-genesis/data/urdf/hsrb4s.urdf"),  # Colab clone
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "hsrb4s.urdf not found. Install hsr-genesis with `pip install -e .` "
        "or clone the repo to /content/hsr-genesis on Colab."
    )


def _quat_wxyz_to_yaw(quat) -> float:
    if isinstance(quat, torch.Tensor):
        q = quat.detach().cpu().numpy()
    else:
        q = np.asarray(quat, dtype=np.float64)
    w, x, y, z = q[:4]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _euler_deg_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles in degrees to WXYZ quaternion (Genesis convention)."""
    r = math.radians(roll)
    p = math.radians(pitch)
    y = math.radians(yaw)
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y_ = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y_, z], dtype=np.float32)


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


def _qpos_to_arm_dofs(entity, qpos, arm_dofs_idx: list[int]) -> torch.Tensor:
    saved = entity.get_qpos().clone()
    try:
        entity.set_qpos(qpos, zero_velocity=False)
        dofs = entity.get_dofs_position()
    finally:
        entity.set_qpos(saved, zero_velocity=False)
    if dofs.ndim == 1:
        dofs = dofs.unsqueeze(0)
    return dofs[:, arm_dofs_idx]


def _arm_traj_names() -> list[str]:
    from hsr_genesis.analytic_ik import JOINT_ORDER

    return list(JOINT_ORDER)


def _step_once(render: bool = True) -> None:
    """Advance the simulation by one dt, handling velocity / gripper / housekeeping."""
    # Raw velocity is already a robot-body command: +X forward, +Y left,
    # +yaw counter-clockwise.  Do not rotate it by the world pose; only the
    # trajectory follower performs a world-to-body conversion.  Refreshing the
    # stored command here also keeps it inside the controller's 0.5 s timeout.
    if _state.base_vel_cmd is not None:
        from hsr_genesis.base_controller import CartSpace

        ctrl = _state.hsr.get_base_controller()
        cmd = CartSpace()
        cmd.dot_x, cmd.dot_y, cmd.dot_r = _state.base_vel_cmd
        ctrl.update_velocity_command(cmd, envs_idx=[0])

    # Gripper force control (if active).
    if _state.gripper_active and _state.gripper is not None:
        _state.gripper.step_apply_force(_state.dt, envs_idx=[0])

    # Whole-body step (handles housekeeping: collision, friction, gains, head hold).
    _state.hsr.step_whole_body_trajectory_batched(_state.dt, envs_idx=[0])

    # Physics step.
    _state.scene.step()

    if render and _state.cam is not None:
        _state.frames.append(_state.cam.render()[0])


# ---------------------------------------------------------------------------
# Colab setup — re-exported from ``colab_bootstrap`` (standalone module with
# zero heavy deps) for backward-compatibility with existing notebooks that
# import ``setup_colab`` from ``tutorial_utils`` AFTER package installation.
# ---------------------------------------------------------------------------

from hsr_genesis.colab_bootstrap import _log, setup_colab  # noqa: F401 — re-exported


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def init_sim(
    dt: float = 0.02,
    cam_res: tuple[int, int] = (640, 480),
    cam_pos: tuple[float, float, float] = (3.0, -1.0, 1.5),
    cam_lookat: tuple[float, float, float] = (0.0, 0.0, 0.5),
    cam_fov: float = 30,
    substeps: int = 20,
) -> None:
    """Initialize Genesis and create a headless HSR scene with an offscreen camera.

    The scene is created but **not yet built** — :func:`spawn_box` and other
    spawn functions can add entities before the first :func:`run` / :func:`step`
    call.  The scene is built lazily on the first step (see :func:`_maybe_build`).

    Calling it a second time rebuilds a fresh scene (same as :func:`reset_sim`).
    """
    if _state.scene is not None:
        _clear_sim_state()

    # Genesis init (idempotent guard).
    if not getattr(gs, "_initialized", False):
        gs.init(backend=gs.gpu)

    # Colab EGL ICD config (fallback if setup_colab() was not called).
    icd_path = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    if os.path.exists("/content") and not os.path.exists(icd_path):
        os.makedirs(os.path.dirname(icd_path), exist_ok=True)
        with open(icd_path, "w") as f:
            f.write(
                '{\n    "file_format_version" : "1.0.0",\n'
                '    "ICD" : {\n        "library_path" : "libEGL_nvidia.so.0"\n    }\n}\n'
            )

    from hsr_genesis.hsr_rigid_entity import HSRBURDF  # noqa: E402

    urdf_path = _find_urdf()
    _state.urdf_path = str(urdf_path)
    _state.dt = dt

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=cam_pos,
            camera_lookat=cam_lookat,
            camera_fov=cam_fov,
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(dt=dt, substeps=substeps),
        rigid_options=gs.options.RigidOptions(use_gjk_collision=True),
        show_viewer=False,
        show_FPS=False,
    )

    scene.add_entity(gs.morphs.Plane(), visualize_contact=True)

    hsr = scene.add_entity(
        HSRBURDF(
            file=str(urdf_path),
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

    cam = scene.add_camera(
        res=cam_res,
        pos=cam_pos,
        lookat=cam_lookat,
        fov=cam_fov,
        GUI=False,
    )

    # Store entities but defer scene.build() so spawn_* can add more entities.
    _state.scene = scene
    _state.hsr = hsr
    _state.cam = cam
    _state.frames = []
    _state.base_vel_cmd = None
    _state.gripper_active = False
    _state.built = False

    print(f"Simulation created (not yet built). dt={dt}s, camera res={cam_res}, URDF={urdf_path}")
    print("Call run() or step() to build the scene and start simulating.")


def _maybe_build() -> None:
    """Build the scene if it hasn't been built yet (lazy build).

    This is called on the first ``run()`` / ``step()`` so that ``spawn_*``
    functions can add entities before the scene is built.
    """
    if _state.built or _state.scene is None:
        return
    hsr = _state.hsr
    _state.scene.build()

    _state.arm_dofs_idx = _arm_dofs_idx_local(hsr)
    _state.end_effector = hsr.get_link("hand_palm_link")
    hsr.end_effector_offset = [0.0, 0.0, 0.09]

    # Hand motor DOF for gripper position control.
    motor_dofs = hsr.get_joint("hand_motor_joint").dofs_idx_local
    _state.motor_idx = int(motor_dofs[0]) if isinstance(motor_dofs, (list, tuple)) else int(motor_dofs)

    # Head tilt DOF.
    head_dofs = hsr.get_joint("head_tilt_joint").dofs_idx_local
    _state.head_idx = int(head_dofs[0]) if isinstance(head_dofs, (list, tuple)) else int(head_dofs)

    # Gripper controller (lazy-initialized via get_gripper_batched).
    _state.gripper = hsr.get_gripper_batched()

    _state.built = True
    print(f"Scene built. dt={_state.dt}s, URDF={_state.urdf_path}")


def reset_sim(**kwargs) -> None:
    """Tear down the current simulation and reinitialize."""
    _clear_sim_state()
    init_sim(**kwargs)


def get_robot():
    """Return the HSR rigid entity."""
    return _state.hsr


def get_scene():
    """Return the Genesis scene."""
    return _state.scene


def get_camera():
    """Return the offscreen camera."""
    return _state.cam


# ---------------------------------------------------------------------------
# Stepping & rendering
# ---------------------------------------------------------------------------


def step(n: int = 1, render: bool = True) -> None:
    """Step the simulation ``n`` times, optionally capturing camera frames."""
    _maybe_build()
    for _ in range(n):
        _step_once(render=render)


def run(seconds: float = 1.0, render: bool = True) -> None:
    """Run the simulation for ``seconds`` seconds, capturing frames.

    Use :func:`show_video` afterwards to display the captured animation.
    """
    _maybe_build()
    n_steps = int(seconds / _state.dt)
    for _ in range(n_steps):
        _step_once(render=render)


def show_video(fps: int | None = None) -> None:
    """Display captured frames as an inline video (requires mediapy)."""
    if not _state.frames:
        print("No frames captured. Call run() or step() with render=True first.")
        return
    import mediapy as media

    if fps is None:
        fps = int(round(1.0 / _state.dt))
    media.show_video(_state.frames, fps=fps)


def show_frame() -> None:
    """Display the latest captured frame as a still image."""
    if not _state.frames:
        print("No frames captured. Call run() or step() with render=True first.")
        return
    import mediapy as media

    media.show_image(_state.frames[-1])


def clear_frames() -> None:
    """Clear the frame buffer (e.g. to start a fresh video segment)."""
    _state.frames = []


def save_video(path: str, fps: int | None = None) -> None:
    """Save captured frames to an mp4 file."""
    if not _state.frames:
        print("No frames captured.")
        return
    import mediapy as media

    if fps is None:
        fps = int(round(1.0 / _state.dt))
    media.write_video(path, _state.frames, fps=fps)
    print(f"Saved video to {path}")


# ---------------------------------------------------------------------------
# Base control
# ---------------------------------------------------------------------------


def move_base_vel(vx: float, vy: float, vw: float) -> None:
    """Set the base velocity command.

    Args:
        vx: Forward velocity [m/s] (positive = forward).
        vy: Lateral velocity [m/s] (positive = left).
        vw: Rotational velocity [deg/s] (positive = counter-clockwise).

    The command persists until you call this again with different values
    or call :func:`stop_base`.
    """
    vw_rad = math.radians(vw)
    _state.base_vel_cmd = (float(vx), float(vy), float(vw_rad))


def stop_base() -> None:
    """Stop the base by zeroing the velocity command."""
    _state.base_vel_cmd = (0.0, 0.0, 0.0)


def move_base_goal(x: float, y: float, theta: float, duration: float = 3.0) -> float:
    """Command the base to move to a goal position (x, y, theta).

    Args:
        x: Target x position [m].
        y: Target y position [m].
        theta: Target heading [deg].
        duration: Time to reach the goal [s].

    Returns:
        The duration (so you can pass it to :func:`run`).
    """
    from hsr_genesis.base_controller import Trajectory

    _maybe_build()
    # Goal poses are world/odom-frame, unlike move_base_vel's body-frame twist.
    # OmniBaseTrajectoryControl owns the world-to-body conversion.
    # Cancel any raw velocity command so both producers cannot race to update
    # the same inner HSRBBaseController command buffer.
    _state.base_vel_cmd = None

    yaw_rad = math.radians(theta)
    base_traj = Trajectory(
        positions=torch.tensor([[x, y, yaw_rad]], device=gs.device, dtype=gs.tc_float),
        time_from_start=torch.tensor([duration], device=gs.device, dtype=gs.tc_float),
    )
    _state.hsr.set_base_trajectory_batched(base_traj, envs_idx=[0], start_time=None)
    return duration


def get_base_pos() -> tuple[float, float, float]:
    """Return the current base position as (x, y, yaw_deg)."""
    _maybe_build()
    pos = _state.hsr.get_pos()
    quat = _state.hsr.get_quat()
    if isinstance(pos, torch.Tensor):
        pos = pos.detach().cpu().numpy()
    if isinstance(quat, torch.Tensor):
        quat = quat.detach().cpu().numpy()
    if pos.ndim > 1:
        pos = pos[0]
    if quat.ndim > 1:
        quat = quat[0]
    yaw = _quat_wxyz_to_yaw(quat)
    return float(pos[0]), float(pos[1]), math.degrees(yaw)


# ---------------------------------------------------------------------------
# Arm control
# ---------------------------------------------------------------------------


def _set_arm_trajectory(arm_angles: list[float], duration: float) -> float:
    """Internal: set an arm-only trajectory (no base motion)."""
    from hsr_genesis.hsr_rigid_entity import JointTrajectory

    _maybe_build()
    positions = torch.tensor([arm_angles], device=gs.device, dtype=gs.tc_float)
    arm_traj = JointTrajectory(
        positions=positions,
        time_from_start=torch.tensor([duration], device=gs.device, dtype=gs.tc_float),
        joint_names=_arm_traj_names(),
    )
    _state.hsr.set_whole_body_trajectory_batched(
        arm_trajectory=arm_traj,
        base_trajectory=None,
        envs_idx=[0],
        start_time=None,
    )
    return duration


def move_arm_neutral(duration: float = 2.0) -> float:
    """Move the arm to a neutral (ready) pose. Returns duration."""
    return _set_arm_trajectory(ARM_NEUTRAL, duration)


def move_arm_init(duration: float = 2.0) -> float:
    """Move the arm to the initial (home) pose. Returns duration."""
    return _set_arm_trajectory(ARM_INIT, duration)


def move_arm_joints(angles: list[float], duration: float = 2.0) -> float:
    """Move arm joints to specified angles [rad].

    Args:
        angles: List of 5 joint angles in radians, in order:
            [arm_lift, arm_flex, arm_roll, wrist_flex, wrist_roll].
        duration: Time to reach the target [s].

    Returns:
        The duration.
    """
    if len(angles) != 5:
        raise ValueError("angles must have 5 elements [arm_lift, arm_flex, arm_roll, wrist_flex, wrist_roll]")
    return _set_arm_trajectory(list(angles), duration)


def move_wholebody_ik(
    x: float,
    y: float,
    z: float,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    duration: float = 3.0,
) -> float:
    """Move the end-effector to a target pose using whole-body IK.

    The IK solver optimizes both base position and arm joints to reach
    the target.

    Args:
        x, y, z: Target end-effector position [m].
        roll, pitch, yaw: Target end-effector orientation [deg].
        duration: Time to reach the target [s].

    Returns:
        The duration.
    """
    from hsr_genesis.hsr_rigid_entity import JointTrajectory
    from hsr_genesis.base_controller import Trajectory

    _maybe_build()
    # Cancel velocity command.
    _state.base_vel_cmd = None

    hand_quat = _euler_deg_to_quat_wxyz(roll, pitch, yaw)
    target_pos = np.array([x, y, z], dtype=np.float32)

    qpos = _state.hsr.inverse_kinematics(
        link=_state.end_effector,
        pos=target_pos,
        quat=hand_quat,
        max_samples=200,
        max_solver_iters=150,
        max_step_size=0.7,
        respect_joint_limit=False,
    )

    arm_dofs = _qpos_to_arm_dofs(_state.hsr, qpos, _state.arm_dofs_idx)
    target_x, target_y = float(qpos[0]), float(qpos[1])
    target_yaw = _quat_wxyz_to_yaw(qpos[3:7])

    base_traj = Trajectory(
        positions=torch.tensor([[target_x, target_y, target_yaw]], device=gs.device, dtype=gs.tc_float),
        time_from_start=torch.tensor([duration], device=gs.device, dtype=gs.tc_float),
    )
    arm_traj = JointTrajectory(
        positions=arm_dofs,
        time_from_start=torch.tensor([duration], device=gs.device, dtype=gs.tc_float),
        joint_names=_arm_traj_names(),
    )

    _state.hsr.set_whole_body_trajectory_batched(
        arm_trajectory=arm_traj,
        base_trajectory=base_traj,
        envs_idx=[0],
        start_time=None,
    )
    return duration


def get_hand_pos() -> tuple[float, float, float]:
    """Return the current end-effector (hand_palm_link) position as (x, y, z)."""
    _maybe_build()
    pos = _state.end_effector.get_pos()
    if isinstance(pos, torch.Tensor):
        pos = pos.detach().cpu().numpy()
    if pos.ndim > 1:
        pos = pos[0]
    return float(pos[0]), float(pos[1]), float(pos[2])


def forward_kinematics(
    arm_angles: list[float],
    torso_lift: float = 0.0,
    base_xyyaw: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, np.ndarray]:
    """Compute forward kinematics for a given joint configuration.

    Args:
        arm_angles: 5 arm joint angles [rad].
        torso_lift: Torso lift height [m].
        base_xyyaw: Base (x, y, yaw_rad) in world frame.

    Returns:
        Dict mapping link name -> 4x4 homogeneous transform (numpy).
    """
    link_names = [
        "base_footprint",
        "arm_lift_link",
        "arm_flex_link",
        "arm_roll_link",
        "wrist_flex_link",
        "hand_palm_link",
    ]
    _maybe_build()
    name_to_local = {link.name: i for i, link in enumerate(_state.hsr.links)}
    link_indices = [name_to_local[n] for n in link_names if n in name_to_local]

    qpos = torch.zeros(_state.hsr.n_qs, dtype=gs.tc_float)
    arm_qs_idx = _state.hsr._ensure_arm_qs_idx()
    torso_qs_idx = _state.hsr._ensure_torso_qs_idx()
    # The torso_lift_joint is a mimic joint driven by arm_lift_joint
    # (torso = multiplier * arm_lift).  Setting the torso qpos directly has no
    # effect because Genesis computes the torso position from arm_lift.  So we
    # fold the requested torso_lift into the arm_lift value.
    arm_angles = list(arm_angles)
    if torso_qs_idx is not None and torso_lift != 0.0:
        multiplier = _state.hsr._hsr_torso_mimic_multiplier
        if multiplier is not None and float(multiplier) != 0.0:
            arm_angles[0] = torso_lift / float(multiplier)
        else:
            qpos[torso_qs_idx] = torso_lift
    for i, val in enumerate(arm_angles):
        qpos[arm_qs_idx[i]] = val

    links_pos, links_quat = _state.hsr.forward_kinematics(
        qpos, links_idx_local=link_indices, base_xyyaw=base_xyyaw
    )

    result = {}
    for i, name in enumerate(link_names):
        if name not in name_to_local:
            continue
        p = links_pos[i].cpu().numpy()
        q = links_quat[i].cpu().numpy()
        w, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        xx, yy, zz = qx * qx, qy * qy, qz * qz
        xy, xz, yz = qx * qy, qx * qz, qy * qz
        wx, wy, wz = w * qx, w * qy, w * qz
        R = np.array(
            [
                [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
                [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
                [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
            ]
        )
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        result[name] = T
    return result


# ---------------------------------------------------------------------------
# Gripper / hand control
# ---------------------------------------------------------------------------


def move_hand(v: float) -> None:
    """Set the hand opening position.

    Args:
        v: 0.0 = closed, 1.0 = open.
    """
    _maybe_build()
    hand_cmd = torch.tensor([[float(v)]], device=gs.device, dtype=gs.tc_float)
    _state.hsr.control_dofs_position(hand_cmd, dofs_idx_local=[_state.motor_idx])
    # Deactivate force-based grasping if it was active.
    _state.gripper_active = False


def grasp_object(effort: float = 3.0) -> None:
    """Start torque-controlled grasping with the given effort [N].

    The grasp remains active during subsequent :func:`run` / :func:`step` calls
    until :func:`move_hand` or :func:`release_object` is called.
    """
    _maybe_build()
    eff = torch.tensor([effort], device=gs.device, dtype=gs.tc_float)
    active = torch.tensor([True], device=gs.device, dtype=torch.bool)
    _state.gripper.set_apply_force_goal(effort=eff, active_mask=active, envs_idx=[0])
    _state.gripper_active = True


def release_object() -> None:
    """Release the currently grasped object (open the hand)."""
    _state.gripper_active = False
    move_hand(1.0)


# ---------------------------------------------------------------------------
# Head control
# ---------------------------------------------------------------------------


def move_head_tilt(v: float) -> None:
    """Set the head tilt angle [rad].

    Negative = look down, positive = look up.
    """
    _maybe_build()
    head_cmd = torch.tensor([[float(v)]], device=gs.device, dtype=gs.tc_float)
    _state.hsr.control_dofs_position(head_cmd, dofs_idx_local=[_state.head_idx])


# ---------------------------------------------------------------------------
# Object spawning
# ---------------------------------------------------------------------------


def _check_not_built() -> None:
    """Raise clear RuntimeError if spawn_* is called before init_sim() or after build."""
    if _state.scene is None:
        raise RuntimeError(
            "Cannot spawn entities before init_sim() is called. "
            "Call init_sim() first to create a simulation scene."
        )
    if _state.built:
        raise RuntimeError(
            "Cannot spawn entities after the scene is built. "
            "Call spawn_box / spawn_sphere / spawn_cylinder *before* "
            "the first run() or step() call."
        )


def spawn_box(
    pos: tuple[float, float, float],
    size: tuple[float, float, float] = (0.04, 0.04, 0.04),
    color: tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0),
    name: str | None = None,
):
    """Spawn a box at the given world position. Returns the entity."""
    _check_not_built()
    entity = _state.scene.add_entity(
        gs.morphs.Box(size=size, pos=pos),
        surface=gs.surfaces.Default(color=color),
        visualize_contact=True,
    )
    if name:
        print(f"Spawned box '{name}' at {pos}, size={size}")
    return entity


def spawn_sphere(
    pos: tuple[float, float, float],
    radius: float = 0.03,
    color: tuple[float, float, float, float] = (0.2, 0.8, 0.2, 1.0),
    name: str | None = None,
):
    """Spawn a sphere at the given world position. Returns the entity."""
    _check_not_built()
    entity = _state.scene.add_entity(
        gs.morphs.Sphere(pos=pos, radius=radius),
        surface=gs.surfaces.Default(color=color),
        visualize_contact=True,
    )
    if name:
        print(f"Spawned sphere '{name}' at {pos}, radius={radius}")
    return entity


def spawn_cylinder(
    pos: tuple[float, float, float],
    radius: float = 0.05,
    height: float = 0.2,
    color: tuple[float, float, float, float] = (0.2, 0.2, 0.8, 1.0),
    name: str | None = None,
):
    """Spawn a cylinder at the given world position. Returns the entity."""
    _check_not_built()
    entity = _state.scene.add_entity(
        gs.morphs.Cylinder(pos=pos, radius=radius, height=height),
        surface=gs.surfaces.Default(color=color),
        visualize_contact=True,
    )
    if name:
        print(f"Spawned cylinder '{name}' at {pos}, radius={radius}, height={height}")
    return entity


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles [deg] to quaternion [x, y, z, w] (ROS convention)."""
    q_wxyz = _euler_deg_to_quat_wxyz(roll, pitch, yaw)
    # Return as xyzw (ROS convention) to match the reference utils.py.
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float32)


def get_object_pos(entity) -> tuple[float, float, float]:
    """Return the world position of any entity as (x, y, z)."""
    pos = entity.get_pos()
    if isinstance(pos, torch.Tensor):
        pos = pos.detach().cpu().numpy()
    if pos.ndim > 1:
        pos = pos[0]
    return float(pos[0]), float(pos[1]), float(pos[2])
