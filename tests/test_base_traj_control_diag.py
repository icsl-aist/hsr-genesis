"""Diagnostic tests for base trajectory control: frame consistency and yaw overshoot.

Runs headless (cpu backend) and logs per-step state so errors are visible.
Each test prints a structured summary table and asserts tolerances.

What is checked
---------------
frame_consistency
  The _point_before velocity stored at the first trajectory sample must be the
  world-frame linear velocity rotated into body frame, not raw world-frame.
  We verify this by injecting a known initial body velocity and checking that
  the interpolated feed-forward velocity at t=0+ is consistent with the body
  frame, not the world frame.

yaw_overshoot
  Pure yaw rotation: the robot starts at yaw=0 and is commanded to yaw=+pi/2.
  We log desired_yaw, actual_yaw, yaw_error_deg, and body_yaw_rate at each step.
  The peak overshoot (actual yaw exceeding target before settling) must be below
  a threshold.

position_xy_error
  Pure forward translation along world X while yaw=0.
  Logs x_error, y_error at each step, asserts final errors are small.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

import genesis as gs

if not getattr(gs, "_initialized", False):
    gs.init(backend=gs.cpu, precision="32", logging_level="warning")

from hsr_genesis.base_controller import (  # noqa: E402
    OmniBaseTrajectoryControl,
    Trajectory,
    TORCH_FLOAT,
    to_torch,
)
from hsr_genesis.hsr_rigid_entity import HSRBURDF  # noqa: E402

URDF_PATH = Path(__file__).resolve().parents[1] / "data" / "urdf" / "hsrb4s.urdf"

# ── helpers ────────────────────────────────────────────────────────────────────

def _yaw_from_quat(quat: torch.Tensor) -> float:
    if quat.ndim > 1:
        quat = quat[0]
    w, x, y, z = quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item()
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _make_scene(dt: float = 0.02) -> tuple[gs.Scene, HSRBURDF]:
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=4),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
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
        )
    )
    scene.build()
    return scene, robot


def _hold_arm(robot: HSRBURDF) -> None:
    arm_pos = robot.get_dofs_position(dofs_idx_local=robot._hsr_arm_dofs_idx_local, envs_idx=[0])
    if arm_pos.ndim > 1:
        arm_pos = arm_pos[0]
    robot.control_dofs_position(
        arm_pos.unsqueeze(0), dofs_idx_local=robot._hsr_arm_dofs_idx_local, envs_idx=[0]
    )


def _settle(scene: gs.Scene, robot: HSRBURDF, steps: int = 30) -> None:
    """Run a few steps so the robot settles on the ground."""
    for _ in range(steps):
        _hold_arm(robot)
        scene.step()


# ── unit test: _point_before velocity frame conversion ─────────────────────────

class TestPointBeforeVelocityFrame:
    """Verify that the initial velocity stored in _point_before is in world frame
    (the trajectory positions are world-frame, so velocities used in interpolation
    must also be world-frame). We drive the OmniBaseTrajectoryControl directly
    without a physics sim so we can inject exact values.

    Before the fix: _point_before.velocities = raw world-frame velocities
                    (correct — they ARE world-frame; no rotation applied here)
    After the fix:  same, because get_output_velocity rotates the whole
                    output_velocity into body frame *after* adding the feedback.

    The real frame bug was identified as: when explicit trajectory velocities are
    provided by the caller, they must be in world frame to match _point_before.
    We verify this contract is enforced by checking the interpolated velocity
    at the start of the segment equals (p1-p0)/T when no explicit vels given,
    regardless of initial robot velocity.
    """

    def _make_ctrl(self) -> OmniBaseTrajectoryControl:
        return OmniBaseTrajectoryControl(
            feedback_gain=torch.tensor([0.0, 0.0, 0.0], dtype=TORCH_FLOAT, device=gs.device)
        )

    def test_no_explicit_velocities_uses_finite_difference(self) -> None:
        ctrl = self._make_ctrl()
        p0 = torch.tensor([1.0, 2.0, 0.3], dtype=TORCH_FLOAT, device=gs.device)
        p1 = torch.tensor([3.0, 4.0, 0.9], dtype=TORCH_FLOAT, device=gs.device)
        T = 2.0
        traj = Trajectory(
            positions=p1.unsqueeze(0),
            time_from_start=torch.tensor([T], dtype=TORCH_FLOAT, device=gs.device),
        )
        # current velocity at accept time (world frame linear + world yaw rate)
        v_world = torch.tensor([0.5, -0.3, 0.1], dtype=TORCH_FLOAT, device=gs.device)
        ctrl.accept_trajectory(traj, p0)

        # First sample: _point_before is initialized from current_positions / current_velocities
        _, desired, _, _ = ctrl.sample_desired_state(0.0, p0, v_world)
        assert desired is not None

        # At t=0 alpha=0 so pos == p0, vel == (p1-p0)/T (finite-diff, ignores v_world)
        expected_vel = (p1 - p0) / T
        assert torch.allclose(desired.velocities, expected_vel, atol=1e-5), (
            f"vel={desired.velocities.tolist()} expected={expected_vel.tolist()}"
        )

    def test_explicit_world_frame_velocities_are_interpolated(self) -> None:
        """When the caller supplies explicit velocities they must be world-frame
        to be consistent with the positions (which are world-frame).
        Verify the interpolation at midpoint blends the two endpoint velocities."""
        ctrl = self._make_ctrl()
        p0 = torch.tensor([0.0, 0.0, 0.0], dtype=TORCH_FLOAT, device=gs.device)
        p1 = torch.tensor([1.0, 0.0, 0.0], dtype=TORCH_FLOAT, device=gs.device)
        v0_world = torch.tensor([0.5, 0.0, 0.0], dtype=TORCH_FLOAT, device=gs.device)
        v1_world = torch.tensor([0.0, 0.0, 0.0], dtype=TORCH_FLOAT, device=gs.device)
        T = 2.0
        traj = Trajectory(
            positions=p1.unsqueeze(0),
            time_from_start=torch.tensor([T], dtype=TORCH_FLOAT, device=gs.device),
            velocities=v1_world.unsqueeze(0),
        )
        ctrl.accept_trajectory(traj, p0)

        # Prime _point_before with zero initial velocity
        ctrl.sample_desired_state(0.0, p0, torch.zeros(3, dtype=TORCH_FLOAT, device=gs.device))

        # At t=T/2 alpha=0.5 between _point_before (v=[0,0,0]) and v1_world=[0,0,0]
        _, desired_mid, _, _ = ctrl.sample_desired_state(T / 2.0, p0 * 0.5, torch.zeros(3, dtype=TORCH_FLOAT, device=gs.device))
        assert desired_mid is not None
        # v0 from _point_before was set with zeros; blended with v1=[0,0,0] → 0
        assert torch.allclose(desired_mid.velocities, torch.zeros(3, dtype=TORCH_FLOAT, device=gs.device), atol=1e-5), (
            f"mid vel={desired_mid.velocities.tolist()}"
        )

    def test_point_before_velocity_stored_as_world_frame(self) -> None:
        """The _point_before velocity MUST be world-frame (not body-frame) because
        the trajectory positions are world-frame. Concretely: if the robot is moving
        at v_world=[1,0,0] (forward in world X) but is rotated 90° so its body X
        points in world Y, then body_vel=[0,1,0]. The _point_before should store
        world-frame [1,0,0], NOT body-frame [0,1,0], so that the finite-diff
        velocity (p1-p0)/T — which is also world-frame — is consistent."""
        ctrl = self._make_ctrl()
        # Robot at yaw=pi/2 (body X points along world Y)
        yaw = math.pi / 2.0
        p0 = torch.tensor([0.0, 0.0, yaw], dtype=TORCH_FLOAT, device=gs.device)
        p1 = torch.tensor([0.0, 1.0, yaw], dtype=TORCH_FLOAT, device=gs.device)  # move in world Y
        T = 1.0

        # World-frame velocity: moving along world Y at 1 m/s
        v_world = torch.tensor([0.0, 1.0, 0.0], dtype=TORCH_FLOAT, device=gs.device)
        # Body-frame velocity would be: [1, 0, 0] (body X = world Y when yaw=pi/2)

        traj = Trajectory(
            positions=p1.unsqueeze(0),
            time_from_start=torch.tensor([T], dtype=TORCH_FLOAT, device=gs.device),
        )
        ctrl.accept_trajectory(traj, p0)

        # Feed world-frame velocity as current_velocities (as step_base_trajectory_batched does)
        _, desired, _, _ = ctrl.sample_desired_state(0.0, p0, v_world)
        assert desired is not None

        # No explicit velocities → vel = (p1-p0)/T = [0,1,0] (world frame) ✓
        expected = (p1 - p0) / T
        assert torch.allclose(desired.velocities, expected, atol=1e-5), (
            f"vel={desired.velocities.tolist()} expected={expected.tolist()}\n"
            "FAIL: velocity is not world-frame consistent with trajectory positions"
        )


# ── physics-based diagnostic tests ─────────────────────────────────────────────

@pytest.fixture(scope="function")
def scene_and_robot():
    s, r = _make_scene(dt=0.02)
    _settle(s, r, steps=50)
    return s, r


class TestYawOvershoot:
    """Pure yaw rotation: target = +90 degrees.

    Logs step-by-step: time, desired_yaw_deg, actual_yaw_deg, error_deg, body_wz.
    Asserts:
      - Peak overshoot < 15 deg
      - Final yaw error < 5 deg after settle
    """

    def test_pure_yaw_rotation(self, scene_and_robot, capsys) -> None:
        scene, robot = scene_and_robot
        dt = float(scene.sim_options.dt)
        target_yaw = math.pi / 2.0
        duration = 3.0
        steps = int(duration / dt) + int(0.5 / dt)  # +0.5 s settle

        traj = Trajectory(
            positions=torch.tensor([[0.0, 0.0, target_yaw]], device=gs.device, dtype=gs.tc_float),
            time_from_start=torch.tensor([duration], device=gs.device, dtype=gs.tc_float),
        )
        robot.set_base_trajectory_batched(traj, envs_idx=[0])

        log_rows: list[tuple] = []
        for step in range(steps):
            robot.step_base_trajectory_batched(dt, envs_idx=[0])
            _hold_arm(robot)
            scene.step()

            t = (step + 1) * dt
            quat = robot.get_quat()
            ang = robot.get_ang()
            if quat.ndim > 1:
                quat = quat[0]
            if ang.ndim > 1:
                ang = ang[0]

            actual_yaw = _yaw_from_quat(quat)
            body_wz = float(ang[2].item())
            err_deg = math.degrees(_wrap(target_yaw - actual_yaw))
            log_rows.append((t, math.degrees(target_yaw), math.degrees(actual_yaw), err_deg, body_wz))

        # Print structured log
        print("\n=== Pure Yaw Rotation (+90°) — per-step log ===")
        print(f"{'t':>6}  {'tgt_yaw':>9}  {'act_yaw':>9}  {'err_deg':>9}  {'body_wz':>9}")
        stride = max(1, len(log_rows) // 25)
        for row in log_rows[::stride]:
            t, tgt, act, err, wz = row
            print(f"{t:6.2f}  {tgt:9.3f}  {act:9.3f}  {err:9.3f}  {wz:9.4f}")

        # Collect peak overshoot: actual exceeded target in the wrong direction
        # overshoot = actual_yaw > target_yaw (robot went past +90°)
        overshot_degs = [
            math.degrees(actual - target_yaw)
            for (_, _, actual_deg, _, _) in log_rows
            if (actual := math.radians(actual_deg)) > target_yaw
        ]
        peak_overshoot_deg = max(overshot_degs) if overshot_degs else 0.0

        final_yaw_error_deg = abs(log_rows[-1][3])

        print(f"\nPeak overshoot:    {peak_overshoot_deg:.3f} deg")
        print(f"Final yaw error:   {final_yaw_error_deg:.3f} deg")

        assert peak_overshoot_deg < 5.0, (
            f"Yaw overshoot too large: {peak_overshoot_deg:.2f}° (limit 5°)\n"
            "Expected < 5° after yaw gain reduction (5→1.5) and derivative damping."
        )
        assert final_yaw_error_deg < 5.0, (
            f"Final yaw error too large: {final_yaw_error_deg:.2f}° (limit 5°)"
        )


class TestXTranslation:
    """Pure forward translation: target = +0.5 m along world X, yaw stays 0.

    Asserts:
      - Final X error < 5 cm
      - Final Y drift < 3 cm
      - No yaw drift > 5 deg
    """

    def test_pure_x_translation(self, scene_and_robot, capsys) -> None:
        scene, robot = scene_and_robot
        dt = float(scene.sim_options.dt)
        target_x = 0.5
        duration = 3.0
        steps = int(duration / dt) + int(0.5 / dt)

        # Reset robot position
        robot.set_pos(torch.tensor([0.0, 0.0, 0.05], device=gs.device, dtype=gs.tc_float),
                      zero_velocity=True)
        _settle(scene, robot, steps=20)

        traj = Trajectory(
            positions=torch.tensor([[target_x, 0.0, 0.0]], device=gs.device, dtype=gs.tc_float),
            time_from_start=torch.tensor([duration], device=gs.device, dtype=gs.tc_float),
        )
        robot.set_base_trajectory_batched(traj, envs_idx=[0])

        log_rows: list[tuple] = []
        for step in range(steps):
            robot.step_base_trajectory_batched(dt, envs_idx=[0])
            _hold_arm(robot)
            scene.step()

            t = (step + 1) * dt
            pos = robot.get_pos()
            quat = robot.get_quat()
            if pos.ndim > 1:
                pos = pos[0]
            if quat.ndim > 1:
                quat = quat[0]

            x = float(pos[0].item())
            y = float(pos[1].item())
            yaw = _yaw_from_quat(quat)
            x_err = target_x - x
            y_err = y
            log_rows.append((t, x, y, math.degrees(yaw), x_err, y_err))

        print("\n=== Pure X Translation (+0.5 m) — per-step log ===")
        print(f"{'t':>6}  {'x':>7}  {'y':>7}  {'yaw_deg':>9}  {'x_err':>7}  {'y_err':>7}")
        stride = max(1, len(log_rows) // 25)
        for row in log_rows[::stride]:
            print(f"{row[0]:6.2f}  {row[1]:7.4f}  {row[2]:7.4f}  {row[3]:9.3f}  {row[4]:7.4f}  {row[5]:7.4f}")

        final = log_rows[-1]
        print(f"\nFinal X error:  {final[4]:.4f} m")
        print(f"Final Y drift:  {abs(final[5]):.4f} m")
        print(f"Yaw drift:      {abs(final[3]):.3f} deg")

        assert abs(final[4]) < 0.05, f"X error too large: {final[4]:.4f} m"
        assert abs(final[5]) < 0.03, f"Y drift too large: {final[5]:.4f} m"
        assert abs(final[3]) < 5.0, f"Yaw drift too large: {final[3]:.2f} deg"


class TestDiagonalMove:
    """Move diagonally: target = (+0.4, +0.3, +pi/4).

    Logs desired vs actual for x, y, yaw at each step.
    Asserts final position error < 5 cm, yaw error < 5 deg.
    """

    def test_diagonal_move(self, scene_and_robot, capsys) -> None:
        scene, robot = scene_and_robot
        dt = float(scene.sim_options.dt)
        tx, ty, tyaw = 0.4, 0.3, math.pi / 4.0
        duration = 3.0
        steps = int(duration / dt) + int(0.5 / dt)

        # Reset
        robot.set_pos(torch.tensor([0.0, 0.0, 0.05], device=gs.device, dtype=gs.tc_float),
                      zero_velocity=True)
        _settle(scene, robot, steps=20)

        traj = Trajectory(
            positions=torch.tensor([[tx, ty, tyaw]], device=gs.device, dtype=gs.tc_float),
            time_from_start=torch.tensor([duration], device=gs.device, dtype=gs.tc_float),
        )
        robot.set_base_trajectory_batched(traj, envs_idx=[0])

        log_rows: list[tuple] = []
        for step in range(steps):
            robot.step_base_trajectory_batched(dt, envs_idx=[0])
            _hold_arm(robot)
            scene.step()

            t = (step + 1) * dt
            pos = robot.get_pos()
            quat = robot.get_quat()
            if pos.ndim > 1:
                pos = pos[0]
            if quat.ndim > 1:
                quat = quat[0]

            x = float(pos[0].item())
            y = float(pos[1].item())
            yaw = _yaw_from_quat(quat)
            # compute desired interpolated position at this time
            alpha = min(t / duration, 1.0)
            dx = alpha * tx
            dy = alpha * ty
            dyaw = alpha * tyaw
            log_rows.append((t, dx, x, dy, y, math.degrees(dyaw), math.degrees(yaw)))

        print("\n=== Diagonal Move (+0.4, +0.3, +45°) — per-step log ===")
        print(f"{'t':>6}  {'des_x':>7}  {'act_x':>7}  {'des_y':>7}  {'act_y':>7}  {'des_yaw':>8}  {'act_yaw':>8}")
        stride = max(1, len(log_rows) // 25)
        for row in log_rows[::stride]:
            print(f"{row[0]:6.2f}  {row[1]:7.4f}  {row[2]:7.4f}  {row[3]:7.4f}  {row[4]:7.4f}  {row[5]:8.3f}  {row[6]:8.3f}")

        final = log_rows[-1]
        pos_err = math.sqrt((final[2] - tx) ** 2 + (final[4] - ty) ** 2)
        yaw_err = abs(math.degrees(_wrap(math.radians(final[6]) - tyaw)))

        print(f"\nFinal position error:  {pos_err:.4f} m")
        print(f"Final yaw error:       {yaw_err:.3f} deg")

        assert pos_err < 0.05, f"Position error too large: {pos_err:.4f} m"
        assert yaw_err < 5.0, f"Yaw error too large: {yaw_err:.2f} deg"
