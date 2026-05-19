"""Head joint gain / tracking test.

For each head DOF, command a target angle and simulate for a fixed settle
time, then measure the steady-state error and the peak overshoot.

Run with the repo venv:
    PYTHONPATH=src .venv/bin/python -m pytest tests/test_head_joint_tracking.py -v -s

Pass --visualize to open the Genesis viewer window.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pytest
import torch

import genesis as gs

if not getattr(gs, "_initialized", False):
    gs.init(backend=gs.cpu, precision="32", logging_level="warning")

from hsr_genesis.hsr_rigid_entity import HSRBURDF  # noqa: E402


# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

URDF_PATH = Path(__file__).resolve().parents[1] / "data" / "urdf" / "hsrb4s.urdf"

DT = 0.01          # simulation timestep (s)
SUBSTEPS = 4       # physics substeps per timestep
SETTLE_TIME = 3.0  # seconds of sim time to wait for each target
STEADY_WINDOW = 0.5  # last N seconds used to measure steady-state error

# Tolerances for pass/fail assertions
STEADY_STATE_TOL_RAD = math.radians(2.0)   # 2 deg steady-state error

# Joint definitions: name → (lower, upper) from URDF
HEAD_JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "head_pan_joint":  (-3.84, 1.75),
    "head_tilt_joint": (-1.57, 0.52),
}

HEAD_JOINT_ORDER = ["head_pan_joint", "head_tilt_joint"]

# Test targets: (fraction-of-range from lower) for each joint.
# Using 25 %, 50 %, and 75 % of range.
TARGET_FRACTIONS = [0.25, 0.50, 0.75]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class JointTrackingResult:
    joint: str
    target: float
    final: float
    steady_error: float       # mean |error| over STEADY_WINDOW
    peak_overshoot: float     # maximum excursion past the target
    history_t: list[float] = field(default_factory=list)
    history_pos: list[float] = field(default_factory=list)
    history_err: list[float] = field(default_factory=list)

    def tol(self) -> float:
        return STEADY_STATE_TOL_RAD

    def passed(self) -> bool:
        return self.steady_error <= self.tol()


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------

def _build_scene(show_viewer: bool = False) -> tuple[gs.Scene, HSRBURDF]:
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            substeps=SUBSTEPS,
        ),
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


def _dof_idx_for_joint(robot: HSRBURDF, joint_name: str) -> int:
    dofs = robot.get_joint(joint_name).dofs_idx_local
    if isinstance(dofs, (list, tuple)):
        return int(dofs[0])
    return int(dofs)


def _get_dof_pos(robot: HSRBURDF, dof_idx: int) -> float:
    pos = robot.get_dofs_position(dofs_idx_local=[dof_idx], envs_idx=[0])
    return float(pos.reshape(-1)[0].item())


# ---------------------------------------------------------------------------
# Core measurement routine
# ---------------------------------------------------------------------------

def _track_single_head_joint(
    scene: gs.Scene,
    robot: HSRBURDF,
    joint_name: str,
    target: float,
) -> JointTrackingResult:
    """Command ``target`` to one head DOF, hold the other at mid-range,
    simulate for SETTLE_TIME, return tracking metrics.

    Each call resets both head joints to a neutral (mid-range) position so that
    tests are independent.
    """

    head_dofs = robot._hsr_head_dofs_idx_local

    # Build a neutral baseline at mid-range for both head joints
    neutral = torch.tensor(
        [(lo + hi) / 2.0 for lo, hi in HEAD_JOINT_LIMITS.values()],
        device=gs.device,
        dtype=gs.tc_float,
    )
    # Teleport head to neutral and zero velocity.
    robot.set_dofs_position(neutral, dofs_idx_local=head_dofs)

    dof_idx = _dof_idx_for_joint(robot, joint_name)
    # Find the position of this dof within the head_dofs list
    head_order_idx = head_dofs.index(dof_idx)

    # Build desired: neutral for all head joints, target for the joint under test
    desired = neutral.clone()
    desired[head_order_idx] = target

    steps = int(math.ceil(SETTLE_TIME / DT))
    steady_start_step = steps - int(math.ceil(STEADY_WINDOW / DT))

    history_t: list[float] = []
    history_pos: list[float] = []
    history_err: list[float] = []
    peak_overshoot: float = 0.0

    for i in range(steps):
        robot.control_dofs_position(
            desired,
            dofs_idx_local=head_dofs,
            envs_idx=[0],
        )
        scene.step()

        t = (i + 1) * DT
        pos = _get_dof_pos(robot, dof_idx)
        err = abs(pos - target)
        # Overshoot: joint went past the target (signed)
        signed_err = pos - target
        # Peak overshoot magnitude (excursion beyond target in either direction)
        if abs(signed_err) > peak_overshoot:
            peak_overshoot = abs(signed_err)

        history_t.append(t)
        history_pos.append(pos)
        history_err.append(err)

    # Steady-state: mean error over the last STEADY_WINDOW seconds
    steady_errors = history_err[steady_start_step:]
    steady_error = float(sum(steady_errors) / len(steady_errors)) if steady_errors else float("nan")
    final_pos = history_pos[-1]

    return JointTrackingResult(
        joint=joint_name,
        target=target,
        final=final_pos,
        steady_error=steady_error,
        peak_overshoot=peak_overshoot,
        history_t=history_t,
        history_pos=history_pos,
        history_err=history_err,
    )


# ---------------------------------------------------------------------------
# Parameterised test cases
# ---------------------------------------------------------------------------

def _test_cases() -> Iterator[tuple[str, float]]:
    for joint_name, (lo, hi) in HEAD_JOINT_LIMITS.items():
        for frac in TARGET_FRACTIONS:
            target = lo + frac * (hi - lo)
            yield joint_name, round(target, 4)


@pytest.fixture(scope="module")
def scene_and_robot(request) -> Iterator[tuple[gs.Scene, HSRBURDF]]:
    show = request.config.getoption("--visualize", default=False)
    scene, robot = _build_scene(show_viewer=show)
    # Ensure the tuned PD gains are applied (normally done lazily in
    # step_base_trajectory_batched, but we use control_dofs_position directly).
    robot._hsr_apply_default_gains()
    yield scene, robot


@pytest.mark.parametrize("joint_name,target", list(_test_cases()))
def test_head_joint_tracks_target(
    scene_and_robot: tuple[gs.Scene, HSRBURDF],
    joint_name: str,
    target: float,
) -> None:
    """Command each head joint to a target angle and verify the
    controller converges within STEADY_STATE_TOL_RAD in steady state."""
    scene, robot = scene_and_robot

    result = _track_single_head_joint(scene, robot, joint_name, target)

    print(
        f"\n  {joint_name}  target={target:.4f} rad"
        f"  final={result.final:.4f} rad"
        f"  steady_err={result.steady_error:.4f} rad"
        f"  peak_overshoot={result.peak_overshoot:.4f} rad"
        f"  {'PASS' if result.passed() else 'FAIL'}"
    )

    assert result.passed(), (
        f"{joint_name}: steady-state error {result.steady_error:.4f} rad "
        f"> tolerance {result.tol():.4f} rad "
        f"(target={target:.4f}, final={result.final:.4f})"
    )


# ---------------------------------------------------------------------------
# Back-and-forth sweep
# ---------------------------------------------------------------------------

@dataclass
class SweepLegResult:
    """Result for one leg of the back-and-forth sweep."""
    direction: str          # "forward" or "backward"
    start: float
    target: float
    final: float
    steady_error: float
    history_t: list[float] = field(default_factory=list)
    history_pos: list[float] = field(default_factory=list)


def _sweep_joint_back_and_forth(
    scene: gs.Scene,
    robot: HSRBURDF,
    joint_name: str,
    *,
    n_cycles: int = 2,
) -> list[SweepLegResult]:
    """Move joint from lower-25% to upper-25% and back, n_cycles times.

    Returns one SweepLegResult per leg (2 * n_cycles legs total).
    Each leg starts with a teleport to the previous settled position so we
    isolate the motion from the previous leg's residual velocity.
    """
    lo, hi = HEAD_JOINT_LIMITS[joint_name]
    pos_a = lo + 0.25 * (hi - lo)   # lower quarter
    pos_b = hi - 0.25 * (hi - lo)   # upper quarter

    head_dofs = robot._hsr_head_dofs_idx_local
    dof_idx = _dof_idx_for_joint(robot, joint_name)
    head_order_idx = head_dofs.index(dof_idx)

    # Start: teleport both head joints to neutral, tested joint to pos_a
    neutral = torch.tensor(
        [(lo_ + hi_) / 2.0 for lo_, hi_ in HEAD_JOINT_LIMITS.values()],
        device=gs.device,
        dtype=gs.tc_float,
    )
    start_pos = neutral.clone()
    start_pos[head_order_idx] = pos_a
    robot.set_dofs_position(start_pos, dofs_idx_local=head_dofs)

    results: list[SweepLegResult] = []
    steps = int(math.ceil(SETTLE_TIME / DT))
    steady_start = steps - int(math.ceil(STEADY_WINDOW / DT))

    current_start = pos_a
    targets = [pos_b, pos_a] * n_cycles

    for leg_idx, target in enumerate(targets):
        direction = "forward" if target > current_start else "backward"
        desired = neutral.clone()
        desired[head_order_idx] = target

        history_t: list[float] = []
        history_pos: list[float] = []
        steady_errors: list[float] = []

        for i in range(steps):
            robot.control_dofs_position(desired, dofs_idx_local=head_dofs, envs_idx=[0])
            scene.step()
            t = (leg_idx * SETTLE_TIME) + (i + 1) * DT
            pos = _get_dof_pos(robot, dof_idx)
            history_t.append(t)
            history_pos.append(pos)
            if i >= steady_start:
                steady_errors.append(abs(pos - target))

        steady_error = float(sum(steady_errors) / len(steady_errors)) if steady_errors else float("nan")
        final_pos = history_pos[-1]

        results.append(SweepLegResult(
            direction=direction,
            start=current_start,
            target=target,
            final=final_pos,
            steady_error=steady_error,
            history_t=history_t,
            history_pos=history_pos,
        ))

        # Next leg starts from the settled position (teleport to clean state)
        start_pos = neutral.clone()
        start_pos[head_order_idx] = final_pos
        robot.set_dofs_position(start_pos, dofs_idx_local=head_dofs)
        current_start = final_pos

    return results


@pytest.mark.parametrize("joint_name", list(HEAD_JOINT_LIMITS.keys()))
def test_head_joint_back_and_forth(
    scene_and_robot: tuple[gs.Scene, HSRBURDF],
    joint_name: str,
) -> None:
    """Sweep each head joint back and forth and verify steady-state error in
    both directions stays within tolerance."""
    scene, robot = scene_and_robot
    tol = STEADY_STATE_TOL_RAD

    legs = _sweep_joint_back_and_forth(scene, robot, joint_name, n_cycles=2)

    print(f"\n  {joint_name} back-and-forth:")
    failures: list[str] = []
    for leg in legs:
        ok = leg.steady_error <= tol
        mark = "PASS" if ok else "FAIL"
        print(
            f"    {leg.direction:>9}  {leg.start:+.4f} -> {leg.target:+.4f}"
            f"  final={leg.final:+.4f}  err={leg.steady_error:.4f} rad  {mark}"
        )
        if not ok:
            failures.append(
                f"{leg.direction} {leg.start:+.4f}->{leg.target:+.4f}: "
                f"err={leg.steady_error:.4f} > tol={tol:.4f} rad"
            )

    assert not failures, f"{joint_name} back-and-forth failures:\n" + "\n".join(failures)


# ---------------------------------------------------------------------------
# Standalone summary (run directly with python, not pytest)
# ---------------------------------------------------------------------------

def _run_standalone() -> None:
    import sys

    show = "--visualize" in sys.argv
    print(f"Building scene (show_viewer={show}) ...")
    scene, robot = _build_scene(show_viewer=show)

    # Force the lazy gain application, then read the actual gains
    robot._hsr_apply_default_gains()
    kp = robot.get_dofs_kp(dofs_idx_local=robot._hsr_head_dofs_idx_local)
    kv = robot.get_dofs_kv(dofs_idx_local=robot._hsr_head_dofs_idx_local)
    print("\nCurrent head gains:")
    for i, name in enumerate(HEAD_JOINT_ORDER):
        print(f"  {name:<22}  kp={float(kp.reshape(-1)[i].item()):8.2f}  kv={float(kv.reshape(-1)[i].item()):8.3f}")

    print(f"\nRunning joint tracking test (settle={SETTLE_TIME}s, dt={DT}s) ...")
    print(f"{'Joint':<22} {'Target':>10} {'Final':>10} {'SteadyErr':>12} {'PeakOver':>12} {'Result':>8}")
    print("-" * 80)

    all_pass = True
    for joint_name, target in _test_cases():
        result = _track_single_head_joint(scene, robot, joint_name, target)
        status = "PASS" if result.passed() else "FAIL"
        if not result.passed():
            all_pass = False
        print(
            f"{joint_name:<22} {target:>10.4f} {result.final:>10.4f}"
            f" {result.steady_error:>12.4f} {result.peak_overshoot:>12.4f}"
            f"  {status}"
        )

    print()
    print("Overall static targets:", "ALL PASS" if all_pass else "SOME FAILED")

    print(f"\n{'='*80}")
    print(f"Back-and-forth sweep (settle={SETTLE_TIME}s per leg, {2*2} legs per joint)")
    print(f"{'='*80}")
    sweep_all_pass = True
    for joint_name in HEAD_JOINT_LIMITS:
        legs = _sweep_joint_back_and_forth(scene, robot, joint_name, n_cycles=2)
        print(f"\n  {joint_name}:")
        for leg in legs:
            ok = leg.steady_error <= STEADY_STATE_TOL_RAD
            if not ok:
                sweep_all_pass = False
            mark = "PASS" if ok else "FAIL"
            print(
                f"    {leg.direction:>9}  {leg.start:+.4f} -> {leg.target:+.4f}"
                f"  final={leg.final:+.4f}  err={leg.steady_error:.4f} rad  {mark}"
            )
    print()
    print("Overall sweep:", "ALL PASS" if sweep_all_pass else "SOME FAILED")


if __name__ == "__main__":
    _run_standalone()
