from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

import genesis as gs

sys.path.insert(0, str(Path(__file__).resolve().parent))
from target_marker_options import TargetMarkerOptions

DEBUG_DIAGNOSTICS = os.environ.get("HSR_RRT_DEBUG", "") not in ("", "0", "false", "False")

URDF_PATH = Path(__file__).resolve().parents[2] / "data" / "urdf" / "hsrb4s.urdf"


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


def _quat_wxyz_to_yaw(quat: torch.Tensor | np.ndarray) -> float:
    if isinstance(quat, torch.Tensor):
        quat_val = quat.detach().cpu().numpy()
    else:
        quat_val = np.asarray(quat, dtype=np.float64)
    w, x, y, z = quat_val[:4]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _qpos_to_ee_pos(entity, qpos, ee_link, goal_pos_np) -> np.ndarray:
    """Temporarily set qpos, read EE position, then restore.

    Returns the FK-computed EE position as a numpy array and prints error.
    """
    saved_qpos = entity.get_qpos().clone()
    try:
        entity.set_qpos(qpos, zero_velocity=False)
        ee_pos = ee_link.get_pos().detach().cpu().numpy().ravel()
    finally:
        entity.set_qpos(saved_qpos, zero_velocity=False)
    error = float(np.linalg.norm(ee_pos - goal_pos_np[:3]))
    print(f"  FK-predicted EE position: ({ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f})")
    print(f"  Target position:          ({goal_pos_np[0]:.4f}, {goal_pos_np[1]:.4f}, {goal_pos_np[2]:.4f})")
    print(f"  FK vs target error:       {error:.4f} m")
    return ee_pos


def _build_velocity_limited_time_from_start(
    arm_positions: torch.Tensor,
    dt: float,
    segment_end_wps: list[int],
    per_segment_duration: float,
    velocity_limits: dict[int, float],
    descend_velocity_limits: dict[int, float] | None = None,
) -> torch.Tensor:
    """Build monotonic time_from_start that respects per-joint velocity limits.

    Each trajectory segment (as defined by *segment_end_wps*) is allocated at
    least *per_segment_duration* seconds with uniform intra-segment spacing.
    For every joint listed in *velocity_limits*, any interval where the joint's
    position change would exceed its speed limit is stretched so the speed
    stays within bounds.  The per-interval minimum time is the maximum over
    all joint-limited times and the uniform minimum.

    Parameters
    ----------
    arm_positions : (N, n_arm_dofs) tensor of arm joint positions.
        Columns correspond to the arm DOFs (see :data:`JOINT_ORDER`).
    dt : float
        Simulation timestep (assigned to the first waypoint).
    segment_end_wps : list of int
        0-based indices of the last waypoint of each trajectory segment.
    per_segment_duration : float
        Minimum time span (seconds) per segment if no velocity-limiting is
        needed.
    velocity_limits : dict[int, float]
        Maps joint index (column in *arm_positions*) to max speed (same limit
        for both ascending and descending motion).
    descend_velocity_limits : dict[int, float] | None
        Maps joint index to a more restrictive descending speed limit.
        Overrides *velocity_limits* for descending motion of those joints.
        Only joints listed here get asymmetric limits; all other joints use
        the symmetric limit from *velocity_limits* for both directions.

    Returns
    -------
    torch.Tensor
        Monotonic ``time_from_start`` of shape ``(N,)``, starting at *dt*.
    """
    n_wp = arm_positions.shape[0]
    device = arm_positions.device
    dtype = arm_positions.dtype

    # --- Phase 1: minimum dt per interval from per-segment uniform spacing ---
    uniform_dt = torch.zeros(n_wp - 1, device=device, dtype=dtype)
    prev_end = -1
    for seg_idx, seg_end in enumerate(segment_end_wps):
        n_wp_seg = seg_end - prev_end
        n_intervals = max(n_wp_seg - 1, 1)
        seg_uniform_dt = per_segment_duration / n_intervals

        # Intervals between waypoints WITHIN this segment
        first_wp_idx = prev_end + 1  # first waypoint of this segment
        for idx in range(first_wp_idx, seg_end):
            uniform_dt[idx] = seg_uniform_dt

        # Interval BETWEEN this segment and the next (boundary)
        # uses the next segment's uniform spacing
        if seg_idx < len(segment_end_wps) - 1:
            next_seg_end = segment_end_wps[seg_idx + 1]
            next_n_wp_seg = next_seg_end - seg_end
            next_n_intervals = max(next_n_wp_seg - 1, 1)
            uniform_dt[seg_end] = per_segment_duration / next_n_intervals

        prev_end = seg_end

    # If no velocity limits, return uniform-only timing
    if not velocity_limits:
        time_from_start = torch.zeros(n_wp, device=device, dtype=dtype)
        time_from_start[0] = dt
        time_from_start[1:] = torch.cumsum(uniform_dt, dim=0) + dt
        return time_from_start

    # --- Phase 2: per-interval min dt per joint from velocity limits ---
    # Start with uniform spacing, then take the max over joint-limited dt
    min_dt_per_interval = uniform_dt.clone()

    for joint_idx, speed_limit in velocity_limits.items():
        if speed_limit <= 0.0:
            continue
        joint_vals: torch.Tensor = arm_positions[:, joint_idx]
        changes = joint_vals[1:] - joint_vals[:-1]
        abs_deltas = torch.abs(changes)

        joint_min_dt = torch.zeros(n_wp - 1, device=device, dtype=dtype)

        # Ascending intervals (change >= 0)
        ascend_mask = changes >= 0
        if ascend_mask.any():
            joint_min_dt[ascend_mask] = abs_deltas[ascend_mask] / speed_limit

        # Descending intervals (change < 0)
        descend_mask = changes < 0
        if descend_mask.any():
            desc_limit = speed_limit
            if descend_velocity_limits is not None and joint_idx in descend_velocity_limits:
                dv = descend_velocity_limits[joint_idx]
                if dv > 0.0:
                    desc_limit = dv
            joint_min_dt[descend_mask] = abs_deltas[descend_mask] / desc_limit

        min_dt_per_interval = torch.maximum(min_dt_per_interval, joint_min_dt)

    # --- Phase 3: cumulative time ---
    time_from_start = torch.zeros(n_wp, device=device, dtype=dtype)
    time_from_start[0] = dt
    time_from_start[1:] = torch.cumsum(min_dt_per_interval, dim=0) + dt

    return time_from_start


def _sample_trajectory_at(
    positions: torch.Tensor,
    time_from_start: torch.Tensor,
    t: float,
) -> torch.Tensor:
    """Linearly interpolate a 1-d trajectory at time *t*.

    Parameters
    ----------
    positions : (N,) tensor of position values (e.g. a single joint).
    time_from_start : (N,) monotonic time stamps.
    t : float
        Query time.

    Returns
    -------
    torch.Tensor
        Interpolated position (scalar, 0-d tensor).  Clamped to the
        trajectory's time range.
    """
    if t <= float(time_from_start[0].item()):
        return positions[0].clone()
    if t >= float(time_from_start[-1].item()):
        return positions[-1].clone()
    # Binary search for the interval
    idx = torch.searchsorted(time_from_start, t, right=True) - 1
    idx = int(idx.item())
    t0 = float(time_from_start[idx].item())
    t1 = float(time_from_start[idx + 1].item())
    p0 = positions[idx]
    p1 = positions[idx + 1]
    if t1 - t0 < 1e-12:
        return p0.clone()
    alpha = (t - t0) / (t1 - t0)
    return p0 + alpha * (p1 - p0)


def _trajectory_execution_steps(execution_duration: float, dt: float) -> int:
    """Return the number of execution steps needed so the controller's
    first-step-relative-time-at-zero sampling reaches the final waypoint.

    The whole-body controller samples relative time 0 on the first step, so
    after N steps the last sampled time is (N-1)*dt.  To cover *duration*
    we need (N-1)*dt >= duration, i.e. N >= duration/dt + 1.
    """
    return int(math.ceil(execution_duration / dt)) + 1


def main() -> None:
    gs.init(backend=gs.gpu)
    from hsr_genesis.hsr_rigid_entity import HSRBURDF, JointTrajectory
    from hsr_genesis.base_controller import Trajectory
    from hsr_genesis.analytic_ik import JOINT_ORDER

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
            max_FPS=60,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
            world_frame_size=1.0,
            show_link_frame=True,
        ),
        rigid_options=gs.options.RigidOptions(
            use_gjk_collision=True,
        ),
        show_viewer=True,
    )

    scene.add_entity(gs.morphs.Plane(), visualize_contact=True)

    scene.add_entity(
        gs.morphs.Box(
            pos=(0.7, 0.0, 0.15),
            size=(0.25, 0.25, 0.3),
            fixed=True,
            collision=True,
        ),
        surface=gs.surfaces.Default(color=(0.2, 0.8, 0.2, 1.0)),
        visualize_contact=True,
    )

    scene.add_entity(
        gs.morphs.Box(
            pos=(0.55, 0.35, 0.1),
            size=(0.15, 0.15, 0.2),
            fixed=True,
            collision=True,
        ),
        surface=gs.surfaces.Default(color=(0.2, 0.2, 0.8, 1.0)),
        visualize_contact=True,
    )

    scene.add_entity(
        gs.morphs.Cylinder(
            pos=(0.5, -0.4, 0.25),
            radius=0.08,
            height=0.5,
            fixed=True,
            collision=True,
        ),
        surface=gs.surfaces.Default(color=(0.8, 0.6, 0.2, 1.0)),
        visualize_contact=True,
    )

    hsr = scene.add_entity(
        HSRBURDF(
            file=str(URDF_PATH),
            fixed=False,
            recompute_inertia=False,
            links_to_keep=[
                "hand_palm_link",
                "hand_l_proximal_link",
                "hand_r_proximal_link",
                "hand_l_distal_link",
                "hand_r_distal_link",
                "hand_l_finger_tip_frame",
                "hand_r_finger_tip_frame",
            ],
            robot="hsrb",
            base_mode="planar",
            end_effector_frame="hand_palm_link",
            use_base_controller=True,
            base_control_mode="controller",
            optimizer="gpu",
        ),
        visualize_contact=True,
    )

    # Define three target positions with shared goal quaternion
    # Red:   farther from the central object
    # Green: same height as object (z=0.15), XY-offset behind/right to avoid overlap
    # Yellow: right-side exit
    targets = [
        np.array([0.45, -0.45, 0.55], dtype=np.float32),
        np.array([0.90,  0.18, 0.15], dtype=np.float32),  # low height; descending arm lift needs slow speed limit
        np.array([0.55,  0.35, 0.55], dtype=np.float32),
    ]
    goal_quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    marker_opts = TargetMarkerOptions(pos=(0.0, 0.0, 0.0), radius=0.04)
    marker_colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)]
    markers = []
    for color in marker_colors:
        marker = scene.add_entity(
            gs.morphs.Sphere(**marker_opts.to_dict()),
            surface=gs.surfaces.Default(color=color),
        )
        markers.append(marker)

    scene.build()

    dt = float(scene.sim_options.dt)
    end_effector = hsr.get_link("hand_palm_link")
    arm_dofs_idx = _arm_dofs_idx_local(hsr)

    start_qpos = hsr.get_qpos().clone()
    if DEBUG_DIAGNOSTICS:
        print(f"Start qpos (first 14): {start_qpos[:14]}")
    print(f"n_qs: {hsr.n_qs}")

    # --- IK and path planning for each target ---
    planning_start_qpos_list: list[torch.Tensor] = []
    planning_goal_qpos_list: list[torch.Tensor] = []
    all_paths = []
    fk_errors: list[tuple[np.ndarray, float]] = []
    prev_solved_qpos = None

    for i, goal_pos in enumerate(targets):
        # Set initial qpos for IK: previous solved qpos, or start with base seed
        if i == 0:
            goal_init_qpos = hsr.get_qpos().clone()
            goal_init_qpos[0] = 0.25
            segment_start_qpos = start_qpos
        else:
            goal_init_qpos = prev_solved_qpos.clone()
            segment_start_qpos = prev_solved_qpos.clone()

        planning_start_qpos_list.append(segment_start_qpos.clone())

        goal_qpos = hsr.inverse_kinematics(
            link=end_effector,
            pos=goal_pos,
            quat=goal_quat,
            init_qpos=goal_init_qpos,
            max_samples=200,
            max_solver_iters=150,
        )
        if goal_qpos.ndim == 2:
            goal_qpos = goal_qpos.squeeze(0)

        print(f"\n=== Target {i+1}: ({goal_pos[0]:.2f}, {goal_pos[1]:.2f}, {goal_pos[2]:.2f}) ===")
        if DEBUG_DIAGNOSTICS:
            print(f"Goal qpos (first 14): {goal_qpos[:14]}")

        # FK diagnostic
        print("\nFK diagnostic at goal_qpos:")
        ee_pos_fk = _qpos_to_ee_pos(hsr, goal_qpos, end_effector, goal_pos)
        fk_error = float(np.linalg.norm(ee_pos_fk - np.asarray(goal_pos, dtype=np.float64)[:3]))
        fk_errors.append((goal_pos, fk_error))

        # Place marker at target
        markers[i].set_pos(goal_pos, zero_velocity=True)

        # Plan path from current segment start to goal qpos
        print(f"\nPlanning path for target {i+1} with RRTConnect (planar base)...")
        result = hsr.plan_path_planar(
            qpos_goal=goal_qpos,
            qpos_start=segment_start_qpos,
            planner="RRTConnect",
            num_waypoints=200,
            smooth_path=True,
            resolution=0.05,
            max_nodes=5000,
            timeout=15.0,
            base_xy_range=2.0,
            base_yaw_range=6.2832,
        )

        if isinstance(result, tuple):
            path, is_invalid = result
            if is_invalid is True or (isinstance(is_invalid, torch.Tensor) and is_invalid.item()):
                print(f"RRTConnect planning failed for target {i+1}!")
                return
        else:
            path = result

        print(f"  Path planned: {path.shape[0]} waypoints")
        if path.ndim == 2:
            path = path.unsqueeze(1)

        # Verify planned path end matches goal qpos; append goal if not
        path_last_qpos = path[-1, 0]
        if not torch.allclose(path_last_qpos, goal_qpos, atol=1e-4):
            diff_norm = (path_last_qpos - goal_qpos).norm().item()
            if DEBUG_DIAGNOSTICS:
                print(f"  WARNING: Path end differs from goal qpos (norm diff={diff_norm:.6f}). Appending goal qpos.")
            goal_wp = goal_qpos.unsqueeze(0).unsqueeze(1)  # shape (1, 1, n_qs)
            path = torch.cat([path, goal_wp], dim=0)
            if DEBUG_DIAGNOSTICS:
                print(f"  Path now has {path.shape[0]} waypoints")

        all_paths.append(path)
        planning_goal_qpos_list.append(goal_qpos.clone())
        prev_solved_qpos = goal_qpos.clone()

    # --- Concatenate segment paths ---
    segment_end_wps: list[int] = []
    concatenated_parts = [all_paths[0]]
    segment_end_wps.append(all_paths[0].shape[0] - 1)
    for path in all_paths[1:]:
        skip = 1  # skip duplicate first waypoint
        concatenated_parts.append(path[skip:])
        segment_end_wps.append(segment_end_wps[-1] + path.shape[0] - skip)

    full_path = torch.cat(concatenated_parts, dim=0)
    n_wp = full_path.shape[0]
    print(f"\nConcatenated path: {n_wp} waypoints")
    print(f"Segment end waypoint indices (0-based): {segment_end_wps}")

    if DEBUG_DIAGNOSTICS:
        # --- IK goal_qpos vs concatenated path end diagnostic ---
        print(f"\n=== IK goal_qpos vs concatenated path end diagnostic ===")
        for i in range(len(targets)):
            goal_qpos = planning_goal_qpos_list[i]
            wp_idx = segment_end_wps[i]
            path_end_qpos = full_path[wp_idx, 0]

            goal_np = goal_qpos.detach().cpu().numpy().ravel()
            path_end_np = path_end_qpos.detach().cpu().numpy().ravel()

            diff_np = goal_np - path_end_np

            dx = float(diff_np[0])
            dy = float(diff_np[1])

            goal_yaw = _quat_wxyz_to_yaw(goal_np[3:7])
            path_yaw = _quat_wxyz_to_yaw(path_end_np[3:7])
            dyaw = goal_yaw - path_yaw

            goal_arm = _qpos_to_arm_dofs(hsr, goal_qpos, arm_dofs_idx)
            path_arm = _qpos_to_arm_dofs(hsr, path_end_qpos, arm_dofs_idx)
            d_arm_lift = float((goal_arm[0, 0] - path_arm[0, 0]).item())
            d_arm_flex = float((goal_arm[0, 1] - path_arm[0, 1]).item())

            full_norm = float(np.linalg.norm(diff_np))

            color_names = ["red", "green", "yellow"]
            print(f"  Target {i} ({color_names[i]}):")
            print(f"    base_x diff (IK - path_end):   {dx:.6f}")
            print(f"    base_y diff (IK - path_end):   {dy:.6f}")
            print(f"    base_yaw diff (IK - path_end): {dyaw:.6f}")
            print(f"    arm_lift diff (IK - path_end): {d_arm_lift:.6f}")
            print(f"    arm_flex diff (IK - path_end): {d_arm_flex:.6f}")
            print(f"    full qpos norm:                {full_norm:.6f}")
            if full_norm > 1e-6:
                print(f"    *** WARNING: IK goal differs from concatenated path end!")

            # FK EE error for both states
            target_pos = targets[i]
            target_np = np.asarray(target_pos, dtype=np.float64)
            print(f"\n    FK EE error for IK goal_qpos vs target [{i}]:")
            _qpos_to_ee_pos(hsr, goal_qpos, end_effector, target_np)
            print(f"    FK EE error for path end qpos vs target [{i}]:")
            _qpos_to_ee_pos(hsr, path_end_qpos, end_effector, target_np)

    if DEBUG_DIAGNOSTICS:
        # --- Planning chain diagnostics ---
        print(f"\n=== Planning chain: segment_start vs previous goal_qpos ===")
        for seg_i in range(1, len(planning_start_qpos_list)):
            s_qpos = planning_start_qpos_list[seg_i]
            g_qpos = planning_goal_qpos_list[seg_i - 1]
            diff = s_qpos - g_qpos
            diff_np = diff.detach().cpu().numpy().ravel()
            s_np = s_qpos.detach().cpu().numpy().ravel()
            g_np = g_qpos.detach().cpu().numpy().ravel()

            dx = float(diff_np[0])
            dy = float(diff_np[1])
            syaw = _quat_wxyz_to_yaw(s_np[3:7])
            gyaw = _quat_wxyz_to_yaw(g_np[3:7])
            dyaw = syaw - gyaw

            s_arm = _qpos_to_arm_dofs(hsr, s_qpos, arm_dofs_idx)
            g_arm = _qpos_to_arm_dofs(hsr, g_qpos, arm_dofs_idx)
            d_arm_lift = float((s_arm[0, 0] - g_arm[0, 0]).item())
            d_arm_flex = float((s_arm[0, 1] - g_arm[0, 1]).item())

            full_norm = float(np.linalg.norm(diff_np))

            print(f"  Segment {seg_i} start vs Segment {seg_i-1} goal:")
            print(f"    base_x diff:     {dx:.6f}")
            print(f"    base_y diff:     {dy:.6f}")
            print(f"    base_yaw diff:   {dyaw:.6f}")
            print(f"    arm_lift diff:   {d_arm_lift:.6f}")
            print(f"    arm_flex diff:   {d_arm_flex:.6f}")
            print(f"    full qpos norm:  {full_norm:.6f}")
            if full_norm > 1e-6:
                print(f"    *** WARNING: Non-zero diff — planning chain may not be continuous!")

    # --- Build trajectories from concatenated path ---
    n_arm_dofs = len(arm_dofs_idx)

    arm_positions = torch.zeros((n_wp, n_arm_dofs), device=gs.device, dtype=gs.tc_float)
    base_positions = torch.zeros((n_wp, 3), device=gs.device, dtype=gs.tc_float)

    per_segment_duration = 4.0
    total_duration = per_segment_duration * len(targets)

    for i in range(n_wp):
        qpos = full_path[i, 0]
        arm_dofs = _qpos_to_arm_dofs(hsr, qpos, arm_dofs_idx)
        arm_positions[i] = arm_dofs[0]
        base_positions[i, 0] = qpos[0]
        base_positions[i, 1] = qpos[1]
        base_positions[i, 2] = _quat_wxyz_to_yaw(qpos[3:7])

    velocity_limits: dict[int, float] = {
        0: 0.18,  # arm_lift_joint: 0.18 m/s (URDF limit is 0.2; torso mimic 0.5x)
        1: 1.2,   # arm_flex_joint: 1.2 rad/s (URDF limit)
    }
    descend_velocity_limits: dict[int, float] = {
        0: 0.09,  # arm_lift_joint descending: 0.09 m/s (torso mimic 0.5x → ~0.045 m/s)
    }
    time_from_start = _build_velocity_limited_time_from_start(
        arm_positions, dt, segment_end_wps,
        per_segment_duration, velocity_limits,
        descend_velocity_limits=descend_velocity_limits,
    )

    # Diagnostic: report lift-limiting impact
    max_lift_speed: float = 0.0
    max_flex_speed: float = 0.0
    n_wp_s = time_from_start.shape[0]
    for k in range(n_wp_s - 1):
        d_lift = float(abs(arm_positions[k + 1, 0].item() - arm_positions[k, 0].item()))
        d_flex = float(abs(arm_positions[k + 1, 1].item() - arm_positions[k, 1].item()))
        d_t = float(time_from_start[k + 1].item() - time_from_start[k].item())
        if d_t > 1e-12:
            lift_speed = d_lift / d_t
            flex_speed = d_flex / d_t
            if lift_speed > max_lift_speed:
                max_lift_speed = lift_speed
            if flex_speed > max_flex_speed:
                max_flex_speed = flex_speed
    total_duration_actual = float(time_from_start[-1].item() - time_from_start[0].item())
    print(f"\nLift- and flex-limited trajectory timing:")
    print(f"  Requested total duration (uniform): {total_duration:.3f}s")
    print(f"  Actual total duration:               {total_duration_actual:.3f}s")
    lift_limit = velocity_limits.get(0, 0.0)
    lift_descend_limit = descend_velocity_limits.get(0, lift_limit) if descend_velocity_limits else lift_limit
    flex_limit = velocity_limits.get(1, 0.0)
    print(f"  Max arm_lift speed:                  {max_lift_speed:.4f} m/s (ascend limit: {lift_limit} m/s, descend limit: {lift_descend_limit} m/s)")
    print(f"  Max arm_flex speed:                  {max_flex_speed:.4f} rad/s (limit: {flex_limit} rad/s)")
    print(f"  Per-segment duration:                {per_segment_duration}s")
    # Per-segment max speeds
    print(f"  Per-segment max speeds:")
    prev_end = -1
    for seg_idx, seg_end in enumerate(segment_end_wps):
        seg_max_lift = 0.0
        seg_max_flex = 0.0
        for k in range(prev_end + 1, seg_end):
            d_lift = float(abs(arm_positions[k + 1, 0].item() - arm_positions[k, 0].item()))
            d_flex = float(abs(arm_positions[k + 1, 1].item() - arm_positions[k, 1].item()))
            d_t = float(time_from_start[k + 1].item() - time_from_start[k].item())
            if d_t > 1e-12:
                lift_speed = d_lift / d_t
                flex_speed = d_flex / d_t
                if lift_speed > seg_max_lift:
                    seg_max_lift = lift_speed
                if flex_speed > seg_max_flex:
                    seg_max_flex = flex_speed
        print(f"    Segment {seg_idx}: max arm_lift = {seg_max_lift:.4f} m/s, max arm_flex = {seg_max_flex:.4f} rad/s")
        prev_end = seg_end

    arm_traj = JointTrajectory(
        positions=arm_positions,
        time_from_start=time_from_start,
        joint_names=list(JOINT_ORDER),
    )

    base_traj = Trajectory(
        positions=base_positions,
        time_from_start=time_from_start,
    )

    hsr.set_whole_body_trajectory_batched(
        arm_trajectory=arm_traj,
        base_trajectory=base_traj,
        envs_idx=[0],
        start_time=0.0,
    )

    # --- Execute via whole-body controller ---
    settle_duration = 2.0
    traj_duration = total_duration_actual
    total_steps = _trajectory_execution_steps(traj_duration + settle_duration, dt)
    print(f"Executing path: {traj_duration:.1f}s trajectory (lift-limited) + {settle_duration:.1f}s settle ({total_steps} steps)...")

    # Pre-compute segment end times (in controller-internal time)
    segment_end_times = [float(time_from_start[wp_idx].item()) for wp_idx in segment_end_wps]

    # Per-target EE error tracking: will record at segment end steps
    ee_errors_at_segment_end: list[tuple[int, float, np.ndarray, float]] = []

    # Segment-2 collision detection window
    # "Second segment" = segment index 1, from target 0 end to target 1 end + padding
    seg2_start_time = segment_end_times[0]  # end of target 0 = start of segment 1 execution
    seg2_end_time = segment_end_times[1]    # end of target 1
    seg2_window_end_time = seg2_end_time + 0.5  # small padding
    hand_finger_links = {
        "hand_palm_link",
        "hand_l_proximal_link",
        "hand_r_proximal_link",
        "hand_l_distal_link",
        "hand_r_distal_link",
        "hand_l_finger_tip_frame",
        "hand_r_finger_tip_frame",
    }
    seg2_collision_events: list[tuple[float, str]] = []
    seg2_collision_count = 0

    # Segment-2 tracking error diagnostics: sample desired vs actual for
    # arm_lift_joint and torso_lift_joint.
    arm_lift_positions = arm_traj.positions[:, 0]  # (N,) arm_lift_joint trajectory
    arm_lift_time = arm_traj.time_from_start
    # torso_lift_joint is a mimic of arm_lift_joint with ratio ~0.5
    torso_lift_positions = arm_lift_positions * 0.5  # approximate desired torso
    seg2_max_arm_lift_error = 0.0
    seg2_max_torso_lift_error = 0.0

    # Per-segment-end tracking state
    recorded_segments = set()

    step_result = None
    for step in range(total_steps):
        elapsed = (step + 1) * dt  # controller internal time
        step_result = hsr.step_whole_body_trajectory_batched(dt, envs_idx=[0])
        scene.step()

        # --- Per-target EE error at segment end times ---
        for seg_idx, seg_end_time in enumerate(segment_end_times):
            if seg_idx in recorded_segments:
                continue
            if elapsed >= seg_end_time - dt * 0.5:
                recorded_segments.add(seg_idx)
                ee_pos = end_effector.get_pos().detach().cpu().numpy().ravel()
                target_pos = targets[seg_idx]
                target_np = np.asarray(target_pos, dtype=np.float64).ravel()
                error = float(np.linalg.norm(ee_pos - target_np[:3]))
                color_names = ["red", "green", "yellow"]
                print(f"\n  Target {seg_idx} ({color_names[seg_idx]}) at segment end:")
                print(f"    Elapsed time:      {elapsed:.3f}s")
                print(f"    EE position:       ({ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f})")
                print(f"    Target position:   ({target_np[0]:.4f}, {target_np[1]:.4f}, {target_np[2]:.4f})")
                print(f"    EE error:          {error:.4f} m")
                # --- Actual vs planned qpos at segment boundary ---
                goal_wp_idx = segment_end_wps[seg_idx]
                planned_goal_qpos = full_path[goal_wp_idx, 0]
                actual_qpos = hsr.get_qpos().clone()
                actual_np = actual_qpos.detach().cpu().numpy().ravel()
                planned_np = planned_goal_qpos.detach().cpu().numpy().ravel()
                qpos_diff_np = actual_np - planned_np
                # Base
                ax, ay = actual_np[0], actual_np[1]
                px, py = planned_np[0], planned_np[1]
                ayaw = _quat_wxyz_to_yaw(actual_np[3:7])
                pyaw = _quat_wxyz_to_yaw(planned_np[3:7])
                # Arm dofs (actual via hardware, planned via FK)
                actual_arm = hsr.get_dofs_position(
                    dofs_idx_local=hsr._hsr_arm_dofs_idx_local, envs_idx=[0]
                )
                actual_arm_lift = float(actual_arm[0].item())
                actual_arm_flex = float(actual_arm[1].item())
                planned_arm = _qpos_to_arm_dofs(hsr, planned_goal_qpos, arm_dofs_idx)
                planned_arm_lift = float(planned_arm[0, 0].item())
                planned_arm_flex = float(planned_arm[0, 1].item())
                full_norm = float(np.linalg.norm(qpos_diff_np))
                if DEBUG_DIAGNOSTICS:
                    print(f"    Actual vs Planned qpos:")
                    print(f"      base_x:       actual={ax:.4f}, planned={px:.4f}, diff={ax-px:.6f}")
                    print(f"      base_y:       actual={ay:.4f}, planned={py:.4f}, diff={ay-py:.6f}")
                    print(f"      base_yaw:     actual={ayaw:.4f}, planned={pyaw:.4f}, diff={ayaw-pyaw:.6f}")
                    print(f"      arm_lift:     actual={actual_arm_lift:.4f}, planned={planned_arm_lift:.4f}, diff={actual_arm_lift-planned_arm_lift:.6f}")
                    print(f"      arm_flex:     actual={actual_arm_flex:.4f}, planned={planned_arm_flex:.4f}, diff={actual_arm_flex-planned_arm_flex:.6f}")
                    print(f"      full qpos norm: {full_norm:.6f}")
                ee_errors_at_segment_end.append((seg_idx, elapsed, ee_pos.copy(), error))
                break

        # --- Segment-2 collision detection + tracking diagnostics ---
        if seg2_start_time <= elapsed <= seg2_window_end_time:
            # Tracking error diagnostics
            desired_arm_lift = float(
                _sample_trajectory_at(
                    arm_lift_positions, arm_lift_time, elapsed
                ).item()
            )
            desired_torso_lift = desired_arm_lift * 0.5
            actual_arm_dofs = hsr.get_dofs_position(
                dofs_idx_local=hsr._hsr_arm_dofs_idx_local, envs_idx=[0]
            )
            actual_arm_lift = float(actual_arm_dofs[0].item())
            actual_torso_dof_idx_local = hsr._ensure_torso_dof_idx()
            if actual_torso_dof_idx_local is not None:
                actual_torso_lift = float(
                    hsr.get_dofs_position(
                        dofs_idx_local=[actual_torso_dof_idx_local], envs_idx=[0]
                    )[0].item()
                )
            else:
                actual_torso_lift = 0.0
            err_arm = abs(actual_arm_lift - desired_arm_lift)
            err_torso = abs(actual_torso_lift - desired_torso_lift)
            seg2_max_arm_lift_error = max(seg2_max_arm_lift_error, err_arm)
            seg2_max_torso_lift_error = max(seg2_max_torso_lift_error, err_torso)

            # Collision detection
            contact_info = hsr._hsr_check_collisions()
            fc = contact_info.get("floor_collisions", [])
            for link in fc:
                if link in hand_finger_links:
                    if DEBUG_DIAGNOSTICS and seg2_collision_count < 5:
                        print(f"  [Collision during segment 2] t={elapsed:.3f}s: link={link}")
                    seg2_collision_events.append((elapsed, link))
                    seg2_collision_count += 1

    # Report segment-2 tracking error diagnostics
    print(f"\n  Segment-2 tracking error diagnostics (max during execution):")
    print(f"    Max arm_lift_joint tracking error:   {seg2_max_arm_lift_error:.4f} m")
    print(f"    Max torso_lift_joint tracking error:  {seg2_max_torso_lift_error:.4f} m")
    if seg2_max_arm_lift_error > 0.02 or seg2_max_torso_lift_error > 0.01:
        print(f"    WARNING: Tracking errors may indicate lift/torso synchronization issues!")

    # Report segment-2 collision summary
    if seg2_collision_events:
        print(f"\n  Segment-2 hand/finger collision summary:")
        print(f"    Total events: {seg2_collision_count}")
        unique_links = sorted(set(link for _, link in seg2_collision_events))
        print(f"    Links involved: {unique_links}")
    else:
        print(f"\n  Segment-2 hand/finger collisions: none detected")

    # --- Collision diagnostic (regression check) ---
    contact_info = hsr._hsr_check_collisions()
    self_collisions = contact_info.get("self_collisions", [])
    floor_collisions = contact_info.get("floor_collisions", [])
    # Wheel-floor contacts are expected for a rolling robot.
    wheel_links = {
        "base_r_drive_wheel_link", "base_l_drive_wheel_link",
        "base_r_passive_wheel_z_link", "base_l_passive_wheel_z_link",
    }
    unexpected_contacts = [
        link for link in floor_collisions if link not in wheel_links
    ]
    print(f"\nCollision diagnostics after execution:")
    print(f"  Self-collisions: {self_collisions}")
    print(f"  Unexpected robot contacts: {unexpected_contacts}")
    print(f"  (Expected wheel-floor contacts omitted: {len(floor_collisions) - len(unexpected_contacts)} link(s))")
    if self_collisions or unexpected_contacts:
        print(f"  WARNING: {'Self-collision' if self_collisions else 'Unexpected contact'} detected!")
    else:
        print("  No unintended contacts — path execution is collision-free.")

    # --- End-effector reach diagnostic (against final target) ---
    ee_pos = end_effector.get_pos().detach().cpu().numpy().ravel()
    final_target = targets[-1]
    final_target_np = np.asarray(final_target, dtype=np.float64).ravel()
    final_error = float(np.linalg.norm(ee_pos - final_target_np[:3]))
    print(f"\nEnd-effector reach diagnostic:")
    print(f"  Final EE position: ({ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f})")
    print(f"  Final target position (target {len(targets)}): ({final_target_np[0]:.4f}, {final_target_np[1]:.4f}, {final_target_np[2]:.4f})")
    print(f"  Euclidean error:   {final_error:.4f} m")
    if step_result is not None:
        active = step_result["active"]
        active_str = ", ".join(str(a.item()) for a in active)
        elapsed = total_steps * dt
        print(f"  Trajectory command active/holding: [{active_str}]")
        print(f"  Elapsed simulation time: {elapsed:.3f}s ({total_steps} steps × {dt:.6f}s)")

    # --- Planned FK error summary for all targets ---
    print(f"\nPlanned FK errors (all targets):")
    for i, (pos, err) in enumerate(fk_errors):
        print(f"  Target {i+1}: pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), "
              f"FK error={err:.4f} m")

    # --- Segment end diagnostics ---
    print(f"\nSegment end waypoints / times:")
    for i, wp_idx in enumerate(segment_end_wps):
        t = time_from_start[wp_idx].item()
        print(f"  Segment {i+1}: waypoint {wp_idx}, time={t:.3f}s")

    print("\nPath execution complete!")
    print(
        f"Base trajectory range: "
        f"x=[{base_positions[:, 0].min().item():.3f}, {base_positions[:, 0].max().item():.3f}], "
        f"y=[{base_positions[:, 1].min().item():.3f}, {base_positions[:, 1].max().item():.3f}], "
        f"yaw=[{base_positions[:, 2].min().item():.3f}, {base_positions[:, 2].max().item():.3f}]"
    )
    final_pos = hsr.get_pos()
    final_quat = hsr.get_quat()
    final_yaw = _quat_wxyz_to_yaw(final_quat)
    print(f"Final base pose: x={final_pos[0].item():.3f}, y={final_pos[1].item():.3f}, yaw={final_yaw:.3f}")


if __name__ == "__main__":
    main()
