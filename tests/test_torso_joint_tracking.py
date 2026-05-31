"""Torso lift joint gain / tracking test.

Commands the torso_lift_joint to target positions with and without
feed-forward gravity compensation, measures steady-state error and
oscillation.

Run with the repo venv:
    PYTHONPATH=src .venv/bin/python -m pytest tests/test_torso_joint_tracking.py -v -s

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


URDF_PATH = Path(__file__).resolve().parents[1] / "data" / "urdf" / "hsrb4s.urdf"

DT = 0.01
SUBSTEPS = 4
SETTLE_TIME = 3.0
STEADY_WINDOW = 0.5

STEADY_STATE_TOL_M = 0.010

TORSO_LOWER = 0.0
TORSO_UPPER = 0.345

TARGET_FRACTIONS = [0.25, 0.50, 0.75]

# Estimated gravity load on the torso lift joint (~75 N arm weight × 0.5 mimic ratio)
FEED_FORWARD_N = 40.0


@dataclass
class TorsoTrackingResult:
    target: float
    final: float
    steady_error: float
    peak_overshoot: float
    has_feedforward: bool
    history_t: list[float] = field(default_factory=list)
    history_pos: list[float] = field(default_factory=list)
    history_err: list[float] = field(default_factory=list)

    def passed(self) -> bool:
        return self.steady_error <= STEADY_STATE_TOL_M


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


JOINT_LIMITS_ARM: dict[str, tuple[float, float]] = {
    "arm_lift_joint":   (0.0,   0.69),
    "arm_flex_joint":   (-2.62, 0.0),
    "arm_roll_joint":   (-2.09, 3.84),
    "wrist_flex_joint": (-1.92, 1.22),
    "wrist_roll_joint": (-1.92, 3.67),
}


def _arm_neutral() -> torch.Tensor:
    return torch.tensor(
        [(lo + hi) / 2.0 for lo, hi in JOINT_LIMITS_ARM.values()],
        device=gs.device,
        dtype=gs.tc_float,
    )


def _track_torso_joint(
    scene: gs.Scene,
    robot: HSRBURDF,
    target: float,
    *,
    feedforward: bool = False,
) -> TorsoTrackingResult:
    """Command ``target`` to the torso DOF, hold arm at neutral, simulate
    for SETTLE_TIME, return tracking metrics.

    The torso_lift_joint is a mimic of arm_lift_joint (torso = arm_lift * 0.5).
    Therefore the arm_lift is set to 2 * target so the PD targets are consistent
    with the mimic constraint.

    When *feedforward* is True a constant feed-forward force is applied to
    the torso DOF to compensate for gravity.
    """
    arm_dofs = robot._hsr_arm_dofs_idx_local
    neutral = _arm_neutral()
    robot.set_dofs_position(neutral, dofs_idx_local=arm_dofs)

    torso_dof_idx = robot._ensure_torso_dof_idx()
    if torso_dof_idx is None:
        pytest.skip("Entity has no torso_lift_joint")

    # The torso mimics arm_lift with multiplier 0.5:
    #   torso = arm_lift * MULTIPLIER  (offset = 0)
    # To hold torso at target, set arm_lift at 2 * target.
    mimic_multiplier = float(robot._hsr_torso_mimic_multiplier)
    arm_lift_target = target / mimic_multiplier
    desired_arm = neutral.unsqueeze(0)
    arm_lift_idx = robot._hsr_arm_lift_order_idx
    desired_arm[0, arm_lift_idx] = arm_lift_target

    desired_torso = torch.tensor([[target]], device=gs.device, dtype=gs.tc_float)

    if feedforward:
        ff_force = torch.tensor([FEED_FORWARD_N], device=gs.device, dtype=gs.tc_float)

    steps = int(math.ceil(SETTLE_TIME / DT))
    steady_start_step = steps - int(math.ceil(STEADY_WINDOW / DT))

    history_t: list[float] = []
    history_pos: list[float] = []
    history_err: list[float] = []
    peak_overshoot: float = 0.0

    for i in range(steps):
        if feedforward:
            robot.control_dofs_force(ff_force, dofs_idx_local=[torso_dof_idx], envs_idx=[0])
        robot.control_dofs_position(desired_arm, dofs_idx_local=arm_dofs, envs_idx=[0])
        robot.control_dofs_position(desired_torso, dofs_idx_local=[torso_dof_idx], envs_idx=[0])
        scene.step()

        t = (i + 1) * DT
        pos = robot.get_dofs_position(dofs_idx_local=[torso_dof_idx], envs_idx=[0])
        pos_val = float(pos.reshape(-1)[0].item())
        err = abs(pos_val - target)

        signed_err = pos_val - target
        if abs(signed_err) > peak_overshoot:
            peak_overshoot = abs(signed_err)

        history_t.append(t)
        history_pos.append(pos_val)
        history_err.append(err)

    steady_errors = history_err[steady_start_step:]
    steady_error = float(sum(steady_errors) / len(steady_errors)) if steady_errors else float("nan")
    final_pos = history_pos[-1]

    return TorsoTrackingResult(
        target=target,
        final=final_pos,
        steady_error=steady_error,
        peak_overshoot=peak_overshoot,
        has_feedforward=feedforward,
        history_t=history_t,
        history_pos=history_pos,
        history_err=history_err,
    )


def _test_cases() -> Iterator[tuple[float, bool]]:
    for frac in TARGET_FRACTIONS:
        target = TORSO_LOWER + frac * (TORSO_UPPER - TORSO_LOWER)
        yield round(target, 4), False
        yield round(target, 4), True


@pytest.fixture(scope="module")
def scene_and_robot(request) -> Iterator[tuple[gs.Scene, HSRBURDF]]:
    show = request.config.getoption("--visualize", default=False)
    scene, robot = _build_scene(show_viewer=show)
    robot._hsr_apply_default_gains()
    yield scene, robot


@pytest.mark.parametrize("target,feedforward", list(_test_cases()))
def test_torso_joint_tracks_target(
    scene_and_robot: tuple[gs.Scene, HSRBURDF],
    target: float,
    feedforward: bool,
) -> None:
    scene, robot = scene_and_robot
    result = _track_torso_joint(scene, robot, target, feedforward=feedforward)

    label = "FF" if feedforward else "PD"
    print(
        f"\n  torso_lift_joint [{label}]  target={target:.4f} m"
        f"  final={result.final:.4f} m"
        f"  steady_err={result.steady_error:.4f} m"
        f"  peak_overshoot={result.peak_overshoot:.4f} m"
        f"  {'PASS' if result.passed() else 'FAIL'}"
    )

    assert result.passed(), (
        f"torso_lift_joint [{label}]: steady-state error {result.steady_error:.4f} m "
        f"> tolerance {STEADY_STATE_TOL_M:.4f} m "
        f"(target={target:.4f}, final={result.final:.4f})"
    )
