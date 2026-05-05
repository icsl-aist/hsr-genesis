"""Arm joint gain / tracking test.

For each arm DOF, command a target angle and simulate for a fixed settle
time, then measure the steady-state error and the peak overshoot.

Run with the repo venv:
    PYTHONPATH=src .venv/bin/python -m pytest tests/test_arm_joint_tracking.py -v -s

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
from hsr_genesis.analytic_ik import JOINT_ORDER  # noqa: E402


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
# arm_lift has a ~75 N gravity load and a 300 N effort cap; with kp=10000 the
# irreducible gravity-induced offset is grav/kp ≈ 0.0075 m.  Use 10 mm.
STEADY_STATE_TOL_M = 0.010                 # 10 mm for prismatic (arm_lift)

# Joint definitions: name → (lower, upper) from URDF
JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "arm_lift_joint":   (0.0,   0.69),
    "arm_flex_joint":   (-2.62, 0.0),
    "arm_roll_joint":   (-2.09, 3.84),
    "wrist_flex_joint": (-1.92, 1.22),
    "wrist_roll_joint": (-1.92, 3.67),
}

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

    @property
    def is_revolute(self) -> bool:
        lo, hi = JOINT_LIMITS[self.joint]
        return not math.isclose(lo, 0.0) or math.isclose(hi - lo, 0.69)

    def tol(self) -> float:
        lo, hi = JOINT_LIMITS[self.joint]
        # arm_lift is prismatic (meters), everything else radians
        if math.isclose(hi - lo, 0.69, rel_tol=0.01) and lo >= 0.0:
            return STEADY_STATE_TOL_M
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

def _track_single_joint(
    scene: gs.Scene,
    robot: HSRBURDF,
    joint_name: str,
    target: float,
) -> JointTrackingResult:
    """Command ``target`` to one DOF, hold all others at the midpoint of their
    range, simulate for SETTLE_TIME, return tracking metrics.

    Each call resets all arm joints to a neutral (mid-range) position so that
    tests are independent and arm_lift_joint is not stuck at its lower limit.
    """

    arm_dofs = robot._hsr_arm_dofs_idx_local

    # Build a neutral baseline at mid-range for all arm joints
    neutral = torch.tensor(
        [(lo + hi) / 2.0 for lo, hi in JOINT_LIMITS.values()],
        device=gs.device,
        dtype=gs.tc_float,
    )
    # Teleport arm to neutral and zero velocity.
    # envs_idx is omitted — set_dofs_position raises if passed for non-batched scenes.
    robot.set_dofs_position(neutral, dofs_idx_local=arm_dofs)

    dof_idx = _dof_idx_for_joint(robot, joint_name)
    # Find the position of this dof within the arm_dofs list
    arm_order_idx = arm_dofs.index(dof_idx)

    # Build desired: neutral for all joints, target for the joint under test
    desired = neutral.clone()
    desired[arm_order_idx] = target

    steps = int(math.ceil(SETTLE_TIME / DT))
    steady_start_step = steps - int(math.ceil(STEADY_WINDOW / DT))

    history_t: list[float] = []
    history_pos: list[float] = []
    history_err: list[float] = []
    peak_overshoot: float = 0.0

    for i in range(steps):
        robot.control_dofs_position(
            desired,
            dofs_idx_local=arm_dofs,
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
    for joint_name, (lo, hi) in JOINT_LIMITS.items():
        for frac in TARGET_FRACTIONS:
            target = lo + frac * (hi - lo)
            yield joint_name, round(target, 4)


@pytest.fixture(scope="module")
def scene_and_robot(request) -> Iterator[tuple[gs.Scene, HSRBURDF]]:
    show = request.config.getoption("--visualize", default=False)
    scene, robot = _build_scene(show_viewer=show)
    yield scene, robot


@pytest.mark.parametrize("joint_name,target", list(_test_cases()))
def test_arm_joint_tracks_target(
    scene_and_robot: tuple[gs.Scene, HSRBURDF],
    joint_name: str,
    target: float,
) -> None:
    """Command each joint to a target angle/position and verify the
    controller converges within STEADY_STATE_TOL_RAD / STEADY_STATE_TOL_M
    in steady state."""
    scene, robot = scene_and_robot

    result = _track_single_joint(scene, robot, joint_name, target)

    unit = "m" if joint_name == "arm_lift_joint" else "rad"
    print(
        f"\n  {joint_name}  target={target:.4f} {unit}"
        f"  final={result.final:.4f} {unit}"
        f"  steady_err={result.steady_error:.4f} {unit}"
        f"  peak_overshoot={result.peak_overshoot:.4f} {unit}"
        f"  {'PASS' if result.passed() else 'FAIL'}"
    )

    assert result.passed(), (
        f"{joint_name}: steady-state error {result.steady_error:.4f} {unit} "
        f"> tolerance {result.tol():.4f} {unit} "
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

    @property
    def unit(self) -> str:
        return "m" if abs(self.target - self.start) < 1.0 else "rad"


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
    lo, hi = JOINT_LIMITS[joint_name]
    pos_a = lo + 0.25 * (hi - lo)   # lower quarter
    pos_b = hi - 0.25 * (hi - lo)   # upper quarter

    arm_dofs = robot._hsr_arm_dofs_idx_local
    dof_idx = _dof_idx_for_joint(robot, joint_name)
    arm_order_idx = arm_dofs.index(dof_idx)

    # Start: teleport all joints to neutral, tested joint to pos_a
    neutral = torch.tensor(
        [(lo_ + hi_) / 2.0 for lo_, hi_ in JOINT_LIMITS.values()],
        device=gs.device,
        dtype=gs.tc_float,
    )
    start_pos = neutral.clone()
    start_pos[arm_order_idx] = pos_a
    robot.set_dofs_position(start_pos, dofs_idx_local=arm_dofs)

    results: list[SweepLegResult] = []
    steps = int(math.ceil(SETTLE_TIME / DT))
    steady_start = steps - int(math.ceil(STEADY_WINDOW / DT))

    current_start = pos_a
    targets = ([pos_b, pos_a] * n_cycles)

    for leg_idx, target in enumerate(targets):
        direction = "forward" if target > current_start else "backward"
        desired = neutral.clone()
        desired[arm_order_idx] = target

        history_t: list[float] = []
        history_pos: list[float] = []
        steady_errors: list[float] = []

        for i in range(steps):
            robot.control_dofs_position(desired, dofs_idx_local=arm_dofs, envs_idx=[0])
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
        start_pos[arm_order_idx] = final_pos
        robot.set_dofs_position(start_pos, dofs_idx_local=arm_dofs)
        current_start = final_pos

    return results


@pytest.mark.parametrize("joint_name", list(JOINT_LIMITS.keys()))
def test_arm_joint_back_and_forth(
    scene_and_robot: tuple[gs.Scene, HSRBURDF],
    joint_name: str,
) -> None:
    """Sweep each joint back and forth and verify steady-state error in both
    directions stays within tolerance."""
    scene, robot = scene_and_robot
    tol = STEADY_STATE_TOL_M if joint_name == "arm_lift_joint" else STEADY_STATE_TOL_RAD
    unit = "m" if joint_name == "arm_lift_joint" else "rad"

    legs = _sweep_joint_back_and_forth(scene, robot, joint_name, n_cycles=2)

    print(f"\n  {joint_name} back-and-forth:")
    failures: list[str] = []
    for leg in legs:
        ok = leg.steady_error <= tol
        mark = "PASS" if ok else "FAIL"
        print(
            f"    {leg.direction:>9}  {leg.start:+.4f} -> {leg.target:+.4f}"
            f"  final={leg.final:+.4f}  err={leg.steady_error:.4f} {unit}  {mark}"
        )
        if not ok:
            failures.append(
                f"{leg.direction} {leg.start:+.4f}->{leg.target:+.4f}: "
                f"err={leg.steady_error:.4f} > tol={tol:.4f} {unit}"
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
    kp = robot.get_dofs_kp(dofs_idx_local=robot._hsr_arm_dofs_idx_local)
    kv = robot.get_dofs_kv(dofs_idx_local=robot._hsr_arm_dofs_idx_local)
    print("\nCurrent arm gains:")
    for i, name in enumerate(JOINT_ORDER):
        print(f"  {name:<22}  kp={float(kp.reshape(-1)[i].item()):8.2f}  kv={float(kv.reshape(-1)[i].item()):8.3f}")

    print(f"\nRunning joint tracking test (settle={SETTLE_TIME}s, dt={DT}s) ...")
    print(f"{'Joint':<22} {'Target':>10} {'Final':>10} {'SteadyErr':>12} {'PeakOver':>12} {'Result':>8}")
    print("-" * 80)

    all_pass = True
    for joint_name, target in _test_cases():
        result = _track_single_joint(scene, robot, joint_name, target)
        unit = "m" if joint_name == "arm_lift_joint" else "rad"
        status = "PASS" if result.passed() else "FAIL"
        if not result.passed():
            all_pass = False
        print(
            f"{joint_name:<22} {target:>10.4f} {result.final:>10.4f}"
            f" {result.steady_error:>12.4f} {result.peak_overshoot:>12.4f}"
            f"  {status}  ({unit})"
        )

    print()
    print("Overall static targets:", "ALL PASS" if all_pass else "SOME FAILED")

    print(f"\n{'='*80}")
    print(f"Back-and-forth sweep (settle={SETTLE_TIME}s per leg, {2*2} legs per joint)")
    print(f"{'='*80}")
    tol_rad = STEADY_STATE_TOL_RAD
    tol_m = STEADY_STATE_TOL_M
    sweep_all_pass = True
    for joint_name in JOINT_LIMITS:
        unit = "m" if joint_name == "arm_lift_joint" else "rad"
        tol = tol_m if joint_name == "arm_lift_joint" else tol_rad
        legs = _sweep_joint_back_and_forth(scene, robot, joint_name, n_cycles=2)
        print(f"\n  {joint_name}:")
        for leg in legs:
            ok = leg.steady_error <= tol
            if not ok:
                sweep_all_pass = False
            mark = "PASS" if ok else "FAIL"
            print(
                f"    {leg.direction:>9}  {leg.start:+.4f} -> {leg.target:+.4f}"
                f"  final={leg.final:+.4f}  err={leg.steady_error:.4f} {unit}  {mark}"
            )
    print()
    print("Overall sweep:", "ALL PASS" if sweep_all_pass else "SOME FAILED")


if __name__ == "__main__":
    _run_standalone()
