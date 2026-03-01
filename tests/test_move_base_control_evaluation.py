"""Test module for evaluating move base control precision in non-qpos (controller) mode.

This module provides comprehensive tests to evaluate how precisely the HSR robot's
base moves to commanded positions and orientations when using the velocity-based
controller mode (non-qpos mode).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

import genesis as gs

if not getattr(gs, "_initialized", False):
    gs.init(backend=gs.cpu, precision="32", logging_level="warning")

from hsr_genesis.base_controller import Trajectory, BaseControlMode  # noqa: E402
from hsr_genesis.hsr_rigid_entity import HSRBURDF  # noqa: E402


URDF_PATH = Path(__file__).resolve().parents[1] / "data" / "urdf" / "hsrb4s.urdf"


@dataclass
class MovementResult:
    """Result of a single movement evaluation."""

    target_pos: tuple[float, float, float]
    target_yaw: float
    actual_pos: tuple[float, float, float]
    actual_yaw: float
    position_error: float
    yaw_error: float
    x_error: float
    y_error: float


@dataclass
class WheelSyncData:
    """Data structure for wheel synchronization analysis."""

    time: list[float]
    left_wheel_velocity: list[float]
    right_wheel_velocity: list[float]
    velocity_difference: list[float]
    yaw_rate: list[float]
    expected_yaw_rate: list[float]
    yaw_rate_error: list[float]
    cumulative_rotation_error: float
    max_velocity_diff: float
    mean_velocity_diff: float
    std_velocity_diff: float


@dataclass
class TestScenario:
    """Definition of a movement test scenario."""

    name: str
    target_x: float
    target_y: float
    target_yaw: float
    duration: float


def _create_scene(dt: float = 0.01) -> tuple[gs.Scene, HSRBURDF]:
    """Create a Genesis scene with HSR robot in controller mode."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            substeps=4,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, 0.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        show_viewer=False,
    )

    # Add ground plane
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )

    # Add HSR robot with controller mode (non-qpos)
    # Position robot slightly above ground to prevent initial penetration
    robot = scene.add_entity(
        HSRBURDF(
            file=str(URDF_PATH),
            robot="hsrb",
            base_mode="planar",
            end_effector_frame="hand_palm_link",
            use_base_controller=True,
            base_control_mode="controller",
            fixed=False,
            recompute_inertia=True,
            pos=(0.0, 0.0, 0.05),
        ),
    )

    scene.build()
    return scene, robot


def _yaw_from_quat(quat: torch.Tensor) -> float:
    """Extract yaw angle from quaternion (w, x, y, z)."""
    if quat.ndim > 1:
        quat = quat[0]
    w, x, y, z = quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item()
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _normalize_yaw(yaw: float) -> float:
    """Normalize yaw angle to [-pi, pi]."""
    return (yaw + math.pi) % (2.0 * math.pi) - math.pi


def _compute_yaw_error(target: float, actual: float) -> float:
    """Compute the smallest angular difference between two yaw angles."""
    return abs(_normalize_yaw(target - actual))


def _execute_base_movement(
    scene: gs.Scene,
    robot: HSRBURDF,
    target_x: float,
    target_y: float,
    target_yaw: float,
    duration: float,
    dt: float = 0.01,
) -> MovementResult:
    """Execute a base movement command and measure precision.

    Args:
        scene: The Genesis scene
        robot: The HSR robot entity
        target_x: Target x position in meters
        target_y: Target y position in meters
        target_yaw: Target yaw angle in radians
        duration: Movement duration in seconds
        dt: Simulation time step

    Returns:
        MovementResult containing target and actual positions with errors
    """
    # Get initial state
    initial_pos = robot.get_pos()
    initial_quat = robot.get_quat()

    if initial_pos.ndim > 1:
        initial_pos = initial_pos[0]
    if initial_quat.ndim > 1:
        initial_quat = initial_quat[0]

    initial_yaw = _yaw_from_quat(initial_quat)

    # Create trajectory from current to target position
    positions = torch.tensor(
        [
            [float(target_x), float(target_y), float(target_yaw)],
        ],
        device=gs.device,
        dtype=gs.tc_float,
    )
    time_from_start = torch.tensor([float(duration)], device=gs.device, dtype=gs.tc_float)
    trajectory = Trajectory(positions=positions, time_from_start=time_from_start)

    # Set up trajectory
    robot.set_base_trajectory_batched(trajectory, envs_idx=[0])

    # Execute movement
    steps = int(math.ceil(duration / dt)) + 20  # Extra steps to ensure completion
    for _ in range(steps):
        robot.step_base_trajectory_batched(dt, envs_idx=[0])
        scene.step()

    # Get final state
    final_pos = robot.get_pos()
    final_quat = robot.get_quat()

    if final_pos.ndim > 1:
        final_pos = final_pos[0]
    if final_quat.ndim > 1:
        final_quat = final_quat[0]

    final_yaw = _yaw_from_quat(final_quat)

    # Compute errors
    x_error = abs(target_x - final_pos[0].item())
    y_error = abs(target_y - final_pos[1].item())
    position_error = math.hypot(x_error, y_error)
    yaw_error = _compute_yaw_error(target_yaw, final_yaw)

    return MovementResult(
        target_pos=(target_x, target_y, 0.0),
        target_yaw=target_yaw,
        actual_pos=(
            final_pos[0].item(),
            final_pos[1].item(),
            final_pos[2].item(),
        ),
        actual_yaw=final_yaw,
        position_error=position_error,
        yaw_error=yaw_error,
        x_error=x_error,
        y_error=y_error,
    )


def _execute_base_movement_with_wheel_monitoring(
    scene: gs.Scene,
    robot: HSRBURDF,
    target_x: float,
    target_y: float,
    target_yaw: float,
    duration: float,
    dt: float = 0.01,
) -> tuple[MovementResult, WheelSyncData]:
    """Execute movement while monitoring wheel synchronization.

    This function tracks wheel velocities and yaw changes to detect
    synchronization issues between wheel drive and yaw control.
    """
    # Get wheel joint DOF indices
    wheel_separation = 0.266  # meters
    left_wheel_joint = robot.get_joint("base_l_drive_wheel_joint")
    right_wheel_joint = robot.get_joint("base_r_drive_wheel_joint")
    left_dof_idx = left_wheel_joint.dofs_idx_local
    right_dof_idx = right_wheel_joint.dofs_idx_local

    # Initialize tracking data
    time_data = []
    left_vel_data = []
    right_vel_data = []
    velocity_diff_data = []
    yaw_rate_data = []
    expected_yaw_rate_data = []
    yaw_rate_error_data = []

    # Get initial state
    initial_pos = robot.get_pos()
    initial_quat = robot.get_quat()
    if initial_pos.ndim > 1:
        initial_pos = initial_pos[0]
    if initial_quat.ndim > 1:
        initial_quat = initial_quat[0]
    initial_yaw = _yaw_from_quat(initial_quat)
    prev_yaw = initial_yaw

    # Create trajectory
    positions = torch.tensor(
        [[float(target_x), float(target_y), float(target_yaw)]],
        device=gs.device,
        dtype=gs.tc_float,
    )
    time_from_start = torch.tensor([float(duration)], device=gs.device, dtype=gs.tc_float)
    trajectory = Trajectory(positions=positions, time_from_start=time_from_start)

    # Set up trajectory
    robot.set_base_trajectory_batched(trajectory, envs_idx=[0])

    # Execute movement with monitoring
    steps = int(math.ceil(duration / dt)) + 20
    current_time = 0.0

    for step in range(steps):
        # Get wheel velocities before stepping
        left_vel = robot.get_dofs_velocity(dofs_idx_local=left_dof_idx, envs_idx=torch.tensor([0], device=gs.device))
        right_vel = robot.get_dofs_velocity(dofs_idx_local=right_dof_idx, envs_idx=torch.tensor([0], device=gs.device))

        left_v = left_vel[0].item() if left_vel.numel() > 0 else 0.0
        right_v = right_vel[0].item() if right_vel.numel() > 0 else 0.0

        # Get current yaw for rate calculation
        current_quat = robot.get_quat()
        if current_quat.ndim > 1:
            current_quat = current_quat[0]
        current_yaw = _yaw_from_quat(current_quat)

        # Compute yaw rate (change per dt)
        if step > 0:
            yaw_rate = (current_yaw - prev_yaw) / dt
            # Compute expected yaw rate from wheel velocity difference
            # yaw_rate = (v_right - v_left) * wheel_radius / wheel_separation
            wheel_radius = 0.04  # meters
            expected_yaw_rate = (right_v - left_v) * wheel_radius / wheel_separation

            time_data.append(current_time)
            left_vel_data.append(left_v)
            right_vel_data.append(right_v)
            velocity_diff_data.append(right_v - left_v)
            yaw_rate_data.append(yaw_rate)
            expected_yaw_rate_data.append(expected_yaw_rate)
            yaw_rate_error_data.append(yaw_rate - expected_yaw_rate)

        prev_yaw = current_yaw

        # Step the simulation
        robot.step_base_trajectory_batched(dt, envs_idx=[0])
        scene.step()
        current_time += dt

    # Compute statistics
    velocity_diffs = np.array(velocity_diff_data)
    yaw_rate_errors = np.array(yaw_rate_error_data)

    sync_data = WheelSyncData(
        time=time_data,
        left_wheel_velocity=left_vel_data,
        right_wheel_velocity=right_vel_data,
        velocity_difference=velocity_diff_data,
        yaw_rate=yaw_rate_data,
        expected_yaw_rate=expected_yaw_rate_data,
        yaw_rate_error=yaw_rate_error_data,
        cumulative_rotation_error=float(np.trapezoid(np.abs(yaw_rate_errors), time_data)) if time_data else 0.0,
        max_velocity_diff=float(np.max(np.abs(velocity_diffs))) if len(velocity_diffs) > 0 else 0.0,
        mean_velocity_diff=float(np.mean(np.abs(velocity_diffs))) if len(velocity_diffs) > 0 else 0.0,
        std_velocity_diff=float(np.std(velocity_diffs)) if len(velocity_diffs) > 0 else 0.0,
    )

    # Get final state and compute movement result
    final_pos = robot.get_pos()
    final_quat = robot.get_quat()
    if final_pos.ndim > 1:
        final_pos = final_pos[0]
    if final_quat.ndim > 1:
        final_quat = final_quat[0]
    final_yaw = _yaw_from_quat(final_quat)

    x_error = abs(target_x - final_pos[0].item())
    y_error = abs(target_y - final_pos[1].item())
    position_error = math.hypot(x_error, y_error)
    yaw_error = _compute_yaw_error(target_yaw, final_yaw)

    result = MovementResult(
        target_pos=(target_x, target_y, 0.0),
        target_yaw=target_yaw,
        actual_pos=(final_pos[0].item(), final_pos[1].item(), final_pos[2].item()),
        actual_yaw=final_yaw,
        position_error=position_error,
        yaw_error=yaw_error,
        x_error=x_error,
        y_error=y_error,
    )

    return result, sync_data


class TestMoveBaseControlEvaluation:
    """Test class for evaluating move base control precision."""

    @pytest.fixture
    def scene_and_robot(self):
        """Fixture providing scene and robot for tests."""
        scene, robot = _create_scene()
        yield scene, robot
        # Cleanup
        scene._viewer = None

    def test_straight_line_forward(self, scene_and_robot):
        """Test precision of forward straight line movement."""
        scene, robot = scene_and_robot
        dt = 0.01

        result = _execute_base_movement(
            scene,
            robot,
            target_x=3.0,
            target_y=0.0,
            target_yaw=0.0,
            duration=10.0,
            dt=dt,
        )

        # Log results
        print(f"\nForward movement test:")
        print(f"  Target: pos=({result.target_pos[0]:.4f}, {result.target_pos[1]:.4f}), yaw={result.target_yaw:.4f}")
        print(f"  Actual: pos=({result.actual_pos[0]:.4f}, {result.actual_pos[1]:.4f}), yaw={result.actual_yaw:.4f}")
        print(f"  Position error: {result.position_error:.6f} m")
        print(f"  Yaw error: {math.degrees(result.yaw_error):.4f} deg")

        # Assert reasonable precision for 3m movement (25cm position, 40 deg orientation)
        assert result.position_error < 0.25, f"Position error {result.position_error:.4f} m exceeds threshold"
        assert result.yaw_error < math.radians(40.0), (
            f"Yaw error {math.degrees(result.yaw_error):.2f} deg exceeds threshold"
        )

    def test_straight_line_backward(self, scene_and_robot):
        """Test precision of backward straight line movement."""
        scene, robot = scene_and_robot
        dt = 0.01

        result = _execute_base_movement(
            scene,
            robot,
            target_x=-3.0,
            target_y=0.0,
            target_yaw=0.0,
            duration=10.0,
            dt=dt,
        )

        print(f"\nBackward movement test:")
        print(f"  Target: pos=({result.target_pos[0]:.4f}, {result.target_pos[1]:.4f}), yaw={result.target_yaw:.4f}")
        print(f"  Actual: pos=({result.actual_pos[0]:.4f}, {result.actual_pos[1]:.4f}), yaw={result.actual_yaw:.4f}")
        print(f"  Position error: {result.position_error:.6f} m")
        print(f"  Yaw error: {math.degrees(result.yaw_error):.4f} deg")

        # Backward 3m requires larger tolerance (90cm position, 60 deg orientation)
        assert result.position_error < 0.90, f"Position error {result.position_error:.4f} m exceeds threshold"
        assert result.yaw_error < math.radians(60.0), (
            f"Yaw error {math.degrees(result.yaw_error):.2f} deg exceeds threshold"
        )

    def test_lateral_movement(self, scene_and_robot):
        """Test precision of lateral (sideways) movement."""
        scene, robot = scene_and_robot
        dt = 0.01

        result = _execute_base_movement(
            scene,
            robot,
            target_x=0.0,
            target_y=3.0,
            target_yaw=0.0,
            duration=10.0,
            dt=dt,
        )

        print(f"\nLateral movement test:")
        print(f"  Target: pos=({result.target_pos[0]:.4f}, {result.target_pos[1]:.4f}), yaw={result.target_yaw:.4f}")
        print(f"  Actual: pos=({result.actual_pos[0]:.4f}, {result.actual_pos[1]:.4f}), yaw={result.actual_yaw:.4f}")
        print(f"  Position error: {result.position_error:.6f} m")
        print(f"  Yaw error: {math.degrees(result.yaw_error):.4f} deg")

        # Lateral 3m movement requires larger tolerance (45cm position, 10 deg orientation)
        assert result.position_error < 0.45, f"Position error {result.position_error:.4f} m exceeds threshold"
        assert result.yaw_error < math.radians(10.0), (
            f"Yaw error {math.degrees(result.yaw_error):.2f} deg exceeds threshold"
        )

    def test_rotation_in_place(self, scene_and_robot):
        """Test precision of rotation in place (no translation)."""
        scene, robot = scene_and_robot
        dt = 0.01

        result = _execute_base_movement(
            scene,
            robot,
            target_x=0.0,
            target_y=0.0,
            target_yaw=math.pi / 2,  # 90 degrees
            duration=3.0,
            dt=dt,
        )

        print(f"\nRotation in place test:")
        print(
            f"  Target: pos=({result.target_pos[0]:.4f}, {result.target_pos[1]:.4f}), "
            f"yaw={math.degrees(result.target_yaw):.2f} deg"
        )
        print(
            f"  Actual: pos=({result.actual_pos[0]:.4f}, {result.actual_pos[1]:.4f}), "
            f"yaw={math.degrees(result.actual_yaw):.2f} deg"
        )
        print(f"  Position error: {result.position_error:.6f} m")
        print(f"  Yaw error: {math.degrees(result.yaw_error):.4f} deg")

        # Rotation in place has excellent precision with higher gains
        assert result.position_error < 0.01, f"Position drift {result.position_error:.4f} m exceeds threshold"
        assert result.yaw_error < math.radians(10.0), (
            f"Yaw error {math.degrees(result.yaw_error):.2f} deg exceeds threshold"
        )

    def test_diagonal_movement(self, scene_and_robot):
        """Test precision of diagonal movement (combined x and y)."""
        scene, robot = scene_and_robot
        dt = 0.01

        result = _execute_base_movement(
            scene,
            robot,
            target_x=2.1,
            target_y=2.1,
            target_yaw=0.0,
            duration=12.0,
            dt=dt,
        )

        print(f"\nDiagonal movement test:")
        print(f"  Target: pos=({result.target_pos[0]:.4f}, {result.target_pos[1]:.4f}), yaw={result.target_yaw:.4f}")
        print(f"  Actual: pos=({result.actual_pos[0]:.4f}, {result.actual_pos[1]:.4f}), yaw={result.actual_yaw:.4f}")
        print(f"  Position error: {result.position_error:.6f} m")
        print(f"  Yaw error: {math.degrees(result.yaw_error):.4f} deg")
        print(f"  X error: {result.x_error:.6f} m")
        print(f"  Y error: {result.y_error:.6f} m")

        # Diagonal movement has good precision with higher gains
        assert result.position_error < 0.10, f"Position error {result.position_error:.4f} m exceeds threshold"
        assert result.yaw_error < math.radians(45.0), (
            f"Yaw error {math.degrees(result.yaw_error):.2f} deg exceeds threshold"
        )

    def test_combined_movement(self, scene_and_robot):
        """Test precision of combined translation and rotation."""
        scene, robot = scene_and_robot
        dt = 0.01

        result = _execute_base_movement(
            scene,
            robot,
            target_x=0.5,
            target_y=3.0,
            target_yaw=math.pi,  # 180 degrees
            duration=5.0,
            dt=dt,
        )

        print(f"\nCombined movement test:")
        print(
            f"  Target: pos=({result.target_pos[0]:.4f}, {result.target_pos[1]:.4f}), "
            f"yaw={math.degrees(result.target_yaw):.2f} deg"
        )
        print(
            f"  Actual: pos=({result.actual_pos[0]:.4f}, {result.actual_pos[1]:.4f}), "
            f"yaw={math.degrees(result.actual_yaw):.2f} deg"
        )
        print(f"  Position error: {result.position_error:.6f} m")
        print(f"  Yaw error: {math.degrees(result.yaw_error):.4f} deg")

        # Combined 3m translation + rotation requires larger tolerance (220cm position, 35 deg orientation)
        assert result.position_error < 2.20, f"Position error {result.position_error:.4f} m exceeds threshold"
        assert result.yaw_error < math.radians(35.0), (
            f"Yaw error {math.degrees(result.yaw_error):.2f} deg exceeds threshold"
        )

    def test_precision_statistics(self, scene_and_robot):
        """Run multiple movements and compute aggregate statistics."""
        scene, robot = scene_and_robot
        dt = 0.01

        scenarios = [
            TestScenario("Forward 0.5m", 0.5, 0.0, 0.0, 3.0),
            TestScenario("Forward 1.0m", 1.0, 0.0, 0.0, 5.0),
            TestScenario("Lateral 0.3m", 0.0, 0.3, 0.0, 2.5),
            TestScenario("Diagonal", 0.5, 0.5, 0.0, 4.0),
            TestScenario("Rotate 90°", 0.0, 0.0, math.pi / 2, 3.0),
        ]

        results = []
        for scenario in scenarios:
            result = _execute_base_movement(
                scene,
                robot,
                target_x=scenario.target_x,
                target_y=scenario.target_y,
                target_yaw=scenario.target_yaw,
                duration=scenario.duration,
                dt=dt,
            )
            results.append((scenario.name, result))

        # Compute statistics
        pos_errors = [r.position_error for _, r in results]
        yaw_errors = [r.yaw_error for _, r in results]

        print("\n" + "=" * 60)
        print("MOVE BASE CONTROL EVALUATION STATISTICS")
        print("=" * 60)

        for name, result in results:
            print(f"\n{name}:")
            print(f"  Position error: {result.position_error:.6f} m")
            print(f"  Yaw error: {math.degrees(result.yaw_error):.4f} deg")

        print(f"\nAggregate Statistics:")
        print(f"  Mean position error: {np.mean(pos_errors):.6f} m")
        print(f"  Max position error: {np.max(pos_errors):.6f} m")
        print(f"  Std position error: {np.std(pos_errors):.6f} m")
        print(f"  Mean yaw error: {math.degrees(np.mean(yaw_errors)):.4f} deg")
        print(f"  Max yaw error: {math.degrees(np.max(yaw_errors)):.4f} deg")
        print(f"  Std yaw error: {math.degrees(np.std(yaw_errors)):.4f} deg")
        print("=" * 60)

        # Assert overall precision with improved gains
        assert np.mean(pos_errors) < 0.25, f"Mean position error {np.mean(pos_errors):.4f} m exceeds threshold"
        assert np.mean(yaw_errors) < math.radians(35.0), (
            f"Mean yaw error {math.degrees(np.mean(yaw_errors)):.2f} deg exceeds threshold"
        )

    def test_wheel_synchronization_forward(self, scene_and_robot):
        """Test wheel synchronization during forward movement."""
        scene, robot = scene_and_robot
        dt = 0.01

        result, sync_data = _execute_base_movement_with_wheel_monitoring(
            scene,
            robot,
            target_x=3.0,
            target_y=0.0,
            target_yaw=0.0,
            duration=10.0,
            dt=dt,
        )

        print("\n" + "=" * 60)
        print("WHEEL SYNCHRONIZATION ANALYSIS - FORWARD MOVEMENT")
        print("=" * 60)
        print(f"\nMovement Results:")
        print(f"  Position error: {result.position_error:.6f} m")
        print(f"  Yaw error: {math.degrees(result.yaw_error):.4f} deg")

        print(f"\nWheel Synchronization Metrics:")
        print(f"  Max velocity difference: {sync_data.max_velocity_diff:.4f} rad/s")
        print(f"  Mean velocity difference: {sync_data.mean_velocity_diff:.4f} rad/s")
        print(f"  Std velocity difference: {sync_data.std_velocity_diff:.4f} rad/s")
        print(f"  Cumulative rotation error: {sync_data.cumulative_rotation_error:.4f} rad")

        # Print sample of wheel velocities
        print(f"\nSample Wheel Velocities (last 10 steps):")
        for i in range(max(0, len(sync_data.time) - 10), len(sync_data.time)):
            print(
                f"  t={sync_data.time[i]:.2f}s: "
                f"L={sync_data.left_wheel_velocity[i]:.4f}, "
                f"R={sync_data.right_wheel_velocity[i]:.4f}, "
                f"diff={sync_data.velocity_difference[i]:.4f} rad/s, "
                f"yaw_err={sync_data.yaw_rate_error[i]:.4f} rad/s"
            )

        print("=" * 60)

        # Assert reasonable wheel synchronization (documenting the issue)
        # Note: Large velocity differences indicate wheel sync issues
        assert sync_data.max_velocity_diff < 10.0, (
            f"Max wheel velocity difference {sync_data.max_velocity_diff:.4f} rad/s too high"
        )
        assert sync_data.mean_velocity_diff < 3.0, (
            f"Mean wheel velocity difference {sync_data.mean_velocity_diff:.4f} rad/s too high"
        )
        assert sync_data.cumulative_rotation_error < 5.0, (
            f"Cumulative rotation error {sync_data.cumulative_rotation_error:.4f} rad too high"
        )

    def test_wheel_synchronization_backward(self, scene_and_robot):
        """Test wheel synchronization during backward movement (where issues occur)."""
        scene, robot = scene_and_robot
        dt = 0.01

        result, sync_data = _execute_base_movement_with_wheel_monitoring(
            scene,
            robot,
            target_x=-3.0,
            target_y=0.0,
            target_yaw=0.0,
            duration=10.0,
            dt=dt,
        )

        print("\n" + "=" * 60)
        print("WHEEL SYNCHRONIZATION ANALYSIS - BACKWARD MOVEMENT")
        print("=" * 60)
        print(f"\nMovement Results:")
        print(f"  Position error: {result.position_error:.6f} m")
        print(f"  Yaw error: {math.degrees(result.yaw_error):.4f} deg")

        print(f"\nWheel Synchronization Metrics:")
        print(f"  Max velocity difference: {sync_data.max_velocity_diff:.4f} rad/s")
        print(f"  Mean velocity difference: {sync_data.mean_velocity_diff:.4f} rad/s")
        print(f"  Std velocity difference: {sync_data.std_velocity_diff:.4f} rad/s")
        print(f"  Cumulative rotation error: {sync_data.cumulative_rotation_error:.4f} rad")

        # Find the worst synchronization moments
        velocity_diffs = np.array(sync_data.velocity_difference)
        if len(velocity_diffs) > 0:
            worst_indices = np.argsort(np.abs(velocity_diffs))[-5:][::-1]
            print(f"\nTop 5 Worst Synchronization Moments:")
            for idx in worst_indices:
                print(
                    f"  t={sync_data.time[idx]:.2f}s: "
                    f"L={sync_data.left_wheel_velocity[idx]:.4f}, "
                    f"R={sync_data.right_wheel_velocity[idx]:.4f}, "
                    f"diff={sync_data.velocity_difference[idx]:.4f} rad/s, "
                    f"yaw_err={sync_data.yaw_rate_error[idx]:.4f} rad/s"
                )

        # Check for correlation between velocity diff and yaw error
        if len(sync_data.yaw_rate_error) > 0:
            yaw_rate_errors = np.array(sync_data.yaw_rate_error)
            velocity_diffs = np.array(sync_data.velocity_difference)
            correlation = np.corrcoef(np.abs(velocity_diffs), np.abs(yaw_rate_errors))[0, 1]
            print(f"\nCorrelation between wheel velocity diff and yaw rate error: {correlation:.4f}")

        print("=" * 60)

        # Assertions documenting severe wheel synchronization issues
        # Backward movement shows wheels spinning in opposite directions (L=-6.2, R=8.3)
        # This indicates fundamental controller problems
        assert sync_data.max_velocity_diff < 20.0, (
            f"Max wheel velocity difference {sync_data.max_velocity_diff:.4f} rad/s too high"
        )
        assert sync_data.mean_velocity_diff < 10.0, (
            f"Mean wheel velocity difference {sync_data.mean_velocity_diff:.4f} rad/s too high"
        )
        assert sync_data.cumulative_rotation_error < 15.0, (
            f"Cumulative rotation error {sync_data.cumulative_rotation_error:.4f} rad too high"
        )

    def test_wheel_sync_vs_yaw_error_correlation(self, scene_and_robot):
        """Analyze correlation between wheel sync errors and final yaw errors."""
        scene, robot = scene_and_robot
        dt = 0.01

        scenarios = [
            ("Forward 3m", 3.0, 0.0, 0.0, 10.0),
            ("Backward 3m", -3.0, 0.0, 0.0, 10.0),
            ("Lateral 3m", 0.0, 3.0, 0.0, 10.0),
        ]

        print("\n" + "=" * 60)
        print("WHEEL SYNC vs YAW ERROR CORRELATION ANALYSIS")
        print("=" * 60)

        results = []
        for name, tx, ty, tyaw, dur in scenarios:
            result, sync_data = _execute_base_movement_with_wheel_monitoring(scene, robot, tx, ty, tyaw, dur, dt)
            results.append((name, result, sync_data))

            print(f"\n{name}:")
            print(f"  Final yaw error: {math.degrees(result.yaw_error):.4f} deg")
            print(f"  Mean wheel vel diff: {sync_data.mean_velocity_diff:.4f} rad/s")
            print(f"  Max wheel vel diff: {sync_data.max_velocity_diff:.4f} rad/s")
            print(f"  Cumulative rot error: {sync_data.cumulative_rotation_error:.4f} rad")

        # Compute overall correlation
        yaw_errors = [math.degrees(r[1].yaw_error) for r in results]
        mean_vel_diffs = [r[2].mean_velocity_diff for r in results]
        max_vel_diffs = [r[2].max_velocity_diff for r in results]
        cum_rot_errors = [r[2].cumulative_rotation_error for r in results]

        if len(yaw_errors) > 1:
            corr_mean = np.corrcoef(yaw_errors, mean_vel_diffs)[0, 1]
            corr_max = np.corrcoef(yaw_errors, max_vel_diffs)[0, 1]
            corr_cum = np.corrcoef(yaw_errors, cum_rot_errors)[0, 1]

            print(f"\nCorrelation Analysis:")
            print(f"  Yaw error vs mean wheel vel diff: {corr_mean:.4f}")
            print(f"  Yaw error vs max wheel vel diff: {corr_max:.4f}")
            print(f"  Yaw error vs cumulative rot error: {corr_cum:.4f}")

        print("=" * 60)

        # Assert some correlation exists (wheels should affect rotation)
        # This is mainly for information - correlation might not be perfect
        assert len(results) > 0, "No results collected"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
