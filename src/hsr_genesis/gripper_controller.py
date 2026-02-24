"""Genesis-side HSR hand controllers.

This module ports the ROS/ros_control gripper actions used on
HSR (apply force and grasp) into pure Python so that they can be driven
inside Genesis.  The goal is API parity with the ROS actions while
keeping the code lightweight enough to integrate in training scripts.

Two public helpers are provided:
  * :class:`HSRBGripperApplyForceAction` implements the PID based force
    regulator exposed as ``tmc_control_msgs/GripperApplyEffort``.
  * :class:`HSRBGripperGraspAction` implements the grasp (torque) action
    exposed on the same message.

Both actions operate on top of a ``hardware_interface`` object that must
provide a tiny subset of the Genesis entity API.  The
:class:`HSRBGenesisGripperInterface` class offers an implementation for a
standard HSR URDF loaded into Genesis.  Tests can pass their own fake
interfaces to simulate hardware behaviour deterministically.

License: Portions ported from hsrb_controllers/hsrb_gripper_controller
are under BSD-compatible terms. This package is released under the
BSD 3-Clause License (see `hsr_genesis/LICENSE.txt`).
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch
import yaml

import genesis as gs

from .base_controller import to_torch


def _first_dof_index(dofs: int | Sequence[int]) -> int:
    if isinstance(dofs, (list, tuple)):
        return int(dofs[0]) if dofs else 0
    return int(dofs)


class ActionState(enum.IntEnum):
    """Simple state machine for the pseudo actions."""

    IDLE = enum.auto()
    RUNNING = enum.auto()
    SUCCEEDED = enum.auto()
    ABORTED = enum.auto()
    CANCELED = enum.auto()


@dataclass
class ActionStatus:
    """Container returned by ``step`` to report progress."""

    state: ActionState
    result: Optional[object] = None


@dataclass
class ApplyForceGoal:
    effort: float
    do_control_stop: bool = False


@dataclass
class ApplyForceResult:
    stalled: bool
    effort: float


@dataclass
class GraspGoal:
    effort: float


@dataclass
class GraspResult:
    stalled: bool
    effort: float


@dataclass
class HSRBGripperApplyForceConfig:
    goal_tolerance: float = 0.1
    stall_velocity_threshold: float = 0.05
    stall_timeout: float = 2.0
    force_control_pgain: float = 0.1
    force_control_igain: float = 0.15
    force_control_dgain: float = 0.4
    force_ierr_max: float = 0.15
    force_lpf_coeff: float = 0.8
    force_calib_data_path: Optional[str] = None


@dataclass
class HSRBGripperGraspConfig:
    torque_goal_tolerance: float = 1.0


class HrhGripperApplyForceCalculator:
    """Python port of ``HrhGripperApplyForceCalculator`` from ROS."""

    def __init__(self, calibration_file_path: Optional[str] = None):
        self.hand_spring_coeff = 1.0
        self.arm_length = 1.0
        self.hand_left_force_calib_data: list[list[float]] | None = None
        self.hand_right_force_calib_data: list[list[float]] | None = None
        if calibration_file_path:
            self._load_force_calibration_data(calibration_file_path)

    def _load_force_calibration_data(self, path: str) -> None:
        try:
            node = yaml.safe_load(open(path, "r", encoding="utf-8"))
        except OSError:
            return
        if not isinstance(node, dict):
            return
        left = node.get("hand_left_force", [])
        right = node.get("hand_right_force", [])
        if all(isinstance(entry, Sequence) and len(entry) == 2 for entry in left):
            self.hand_left_force_calib_data = [list(map(float, entry)) for entry in left]
        if all(isinstance(entry, Sequence) and len(entry) == 2 for entry in right):
            self.hand_right_force_calib_data = [list(map(float, entry)) for entry in right]
        self.hand_spring_coeff = float(
            node.get("hand_spring", self.hand_spring_coeff)
        )
        self.arm_length = float(
            node.get("arm_length", self.arm_length)
        )

    def get_current_force(
        self,
        hand_motor_pos: float,
        left_spring_proximal_joint_pos: float,
        right_spring_proximal_joint_pos: float,
    ) -> float:
        left_force = self._calculate_force(
            hand_motor_pos,
            left_spring_proximal_joint_pos,
            self.hand_left_force_calib_data,
        )
        right_force = self._calculate_force(
            hand_motor_pos,
            right_spring_proximal_joint_pos,
            self.hand_right_force_calib_data,
        )
        return 0.5 * (left_force + right_force)

    def _calculate_force(
        self,
        hand_motor_pos: float,
        spring_proximal_joint_pos: float,
        calib_points: Optional[list[list[float]]],
    ) -> float:
        spring_force = spring_proximal_joint_pos * self.hand_spring_coeff / self.arm_length
        if not calib_points:
            return spring_force
        if hand_motor_pos < calib_points[0][0]:
            return max(0.0, spring_force - calib_points[0][1])
        for p0, p1 in zip(calib_points, calib_points[1:]):
            if p0[0] <= hand_motor_pos < p1[0]:
                return max(0.0, spring_force - self._calculate_internal_force(hand_motor_pos, p0, p1))
        return 0.0

    @staticmethod
    def _calculate_internal_force(hand_motor_pos: float, calib_p0: Sequence[float], calib_p1: Sequence[float]) -> float:
        denom = calib_p1[0] - calib_p0[0]
        if abs(denom) <= torch.finfo(torch.float64).eps:
            return 0.5 * (calib_p1[1] + calib_p0[1])
        slope = (calib_p1[1] - calib_p0[1]) / denom
        return slope * (hand_motor_pos - calib_p0[0]) + calib_p0[1]


class HSRBGenesisGripperInterface:
    """Thin adapter around a Genesis entity exposing the joints we need."""

    def __init__(
        self,
        entity,
        *,
        motor_joint: str = "hand_motor_joint",
        left_spring_joint: str = "hand_l_spring_proximal_joint",
        right_spring_joint: str = "hand_r_spring_proximal_joint",
        envs_idx: Optional[int] = None,
    ) -> None:
        self.entity = entity
        self.envs_idx = envs_idx
        self.motor_idx = _first_dof_index(entity.get_joint(motor_joint).dofs_idx_local)
        self.left_spring_idx = _first_dof_index(
            entity.get_joint(left_spring_joint).dofs_idx_local
        )
        self.right_spring_idx = _first_dof_index(
            entity.get_joint(right_spring_joint).dofs_idx_local
        )
        self._commanded_grasp_torque = 0.0
        self._grasping_flag = False
        self._measured_torque = 0.0
        self._joint_name_to_idx = {
            motor_joint: self.motor_idx,
            left_spring_joint: self.left_spring_idx,
            right_spring_joint: self.right_spring_idx,
        }
        self._mimic_children = self._default_mimic_children()

    # ------------------------------------------------------------------
    # Sensors

    def get_motor_position(self) -> float:
        return float(
            to_torch(
                self.entity.get_dofs_position(
                    dofs_idx_local=[self.motor_idx],
                    envs_idx=self.envs_idx,
                )
            )[0].item()
        )

    def get_motor_velocity(self) -> float:
        return float(
            to_torch(
                self.entity.get_dofs_velocity(
                    dofs_idx_local=[self.motor_idx],
                    envs_idx=self.envs_idx,
                )
            )[0].item()
        )

    def get_left_spring_position(self) -> float:
        return float(
            to_torch(
                self.entity.get_dofs_position(
                    dofs_idx_local=[self.left_spring_idx],
                    envs_idx=self.envs_idx,
                )
            )[0].item()
        )

    def get_right_spring_position(self) -> float:
        return float(
            to_torch(
                self.entity.get_dofs_position(
                    dofs_idx_local=[self.right_spring_idx],
                    envs_idx=self.envs_idx,
                )
            )[0].item()
        )

    def get_current_grasping_flag(self) -> bool:
        return self._grasping_flag

    def get_current_torque(self) -> float:
        return self._measured_torque

    # ------------------------------------------------------------------
    # Actuation hooks

    def command_motor_position(self, position: float) -> None:
        self._set_joint_position(self.motor_idx, position)
        for child in self._mimic_children.get("hand_motor_joint", []):
            idx = self._joint_name_to_idx.get(child.joint)
            if idx is None:
                try:
                    idx = _first_dof_index(self.entity.get_joint(child.joint).dofs_idx_local)
                    self._joint_name_to_idx[child.joint] = idx
                except Exception:  # pragma: no cover - joint missing in entity
                    continue
            commanded = child.multiplier * position + child.offset
            self._set_joint_position(idx, commanded)

    def set_grasp_command(self, start_grasping: bool, effort: float) -> None:
        self._commanded_grasp_torque = effort
        if start_grasping:
            # On the physical robot the same register is used for command
            # and feedback.  We mimic this behaviour here.
            self._grasping_flag = True

    # ------------------------------------------------------------------
    # Simulation aides (tests / demos)

    def set_simulated_grasping_flag(self, value: bool) -> None:
        self._grasping_flag = bool(value)

    def set_measured_torque(self, value: float) -> None:
        self._measured_torque = float(value)

    # ------------------------------------------------------------------
    # URDF mimic helpers

    @dataclass(frozen=True)
    class _MimicJoint:
        joint: str
        multiplier: float = 1.0
        offset: float = 0.0

    def _default_mimic_children(self):
        return {
            "hand_motor_joint": [
                HSRBGenesisGripperInterface._MimicJoint("hand_l_proximal_joint", 1.0, 0.0),
                HSRBGenesisGripperInterface._MimicJoint("hand_r_proximal_joint", 1.0, 0.0),
                HSRBGenesisGripperInterface._MimicJoint("hand_l_distal_joint", -1.0, -0.087),
                HSRBGenesisGripperInterface._MimicJoint("hand_r_distal_joint", -1.0, -0.087),
            ],
            "hand_l_spring_proximal_joint": [
                HSRBGenesisGripperInterface._MimicJoint("hand_l_mimic_distal_joint", -1.0, 0.0),
            ],
            "hand_r_spring_proximal_joint": [
                HSRBGenesisGripperInterface._MimicJoint("hand_r_mimic_distal_joint", -1.0, 0.0),
            ],
        }

    def _set_joint_position(self, idx: int, position: float) -> None:
        self.entity.control_dofs_position(
            torch.tensor([position], device=gs.device, dtype=gs.tc_float),
            dofs_idx_local=[idx],
            envs_idx=self.envs_idx,
        )


class HSRBGripperApplyForceAction:
    """Genesis implementation of the apply_force action."""

    def __init__(
        self,
        hardware_interface,
        *,
        config: Optional[HSRBGripperApplyForceConfig] = None,
        calculator: Optional[HrhGripperApplyForceCalculator] = None,
    ) -> None:
        self.hw = hardware_interface
        self.config = config or HSRBGripperApplyForceConfig()
        self.calculator = calculator or HrhGripperApplyForceCalculator(
            self.config.force_calib_data_path
        )
        self._state = ActionState.IDLE
        self._goal: Optional[ApplyForceGoal] = None
        self._current_force_lpf = 0.0
        self._force_ierr = 0.0
        self._last_force_sample = 0.0
        self._time = 0.0
        self._last_movement_time = 0.0
        self._maintained_position = self.hw.get_motor_position()
        self._result: Optional[ApplyForceResult] = None

    def set_goal(self, goal: ApplyForceGoal) -> None:
        self._goal = goal
        self._state = ActionState.RUNNING
        self._result = None
        self._force_ierr = 0.0
        self._current_force_lpf = 0.0
        self._last_force_sample = 0.0
        self._time = 0.0
        self._last_movement_time = 0.0

    def cancel(self) -> None:
        self._goal = None
        self._state = ActionState.CANCELED
        self._result = None

    def step(self, dt: float) -> ActionStatus:
        dt = float(dt)
        self._time += dt
        active = self._state == ActionState.RUNNING and self._goal is not None

        if not active and self._goal and self._goal.do_control_stop:
            # Control stop requested and the goal already finished.
            return ActionStatus(self._state, self._result)

        if active or (self._goal and not self._goal.do_control_stop):
            commanded = self._compute_command_position()
            self.hw.command_motor_position(commanded)
            self._maintained_position = commanded

        if active:
            self._check_for_success()

        return ActionStatus(self._state, self._result)

    # ------------------------------------------------------------------
    # Internal helpers

    def _compute_command_position(self) -> float:
        ref_force = self._goal.effort if self._goal else 0.0
        motor_pos = self.hw.get_motor_position()
        left_spring = self.hw.get_left_spring_position()
        right_spring = self.hw.get_right_spring_position()
        current_force = self.calculator.get_current_force(
            motor_pos,
            left_spring,
            right_spring,
        )
        coeff = self.config.force_lpf_coeff
        self._current_force_lpf = (1.0 - coeff) * current_force + coeff * self._last_force_sample
        self._last_force_sample = self._current_force_lpf

        error = self._current_force_lpf - ref_force
        self._force_ierr += error
        self._force_ierr = max(-self.config.force_ierr_max, min(self.config.force_ierr_max, self._force_ierr))

        delta = (
            self.config.force_control_pgain * error
            + self.config.force_control_igain * self._force_ierr
            + self.config.force_control_dgain * (self._current_force_lpf - current_force)
        )
        return motor_pos + delta

    def _check_for_success(self) -> None:
        velocity = abs(self.hw.get_motor_velocity())
        if velocity > self.config.stall_velocity_threshold:
            self._last_movement_time = self._time
            return
        if (self._time - self._last_movement_time) <= self.config.stall_timeout:
            return

        measured_force = self._current_force_lpf
        result = ApplyForceResult(stalled=True, effort=measured_force)
        if abs(self._goal.effort - measured_force) < self.config.goal_tolerance:
            self._state = ActionState.SUCCEEDED
        else:
            self._state = ActionState.ABORTED
        self._result = result
        self._goal = ApplyForceGoal(self._goal.effort, self._goal.do_control_stop)


class HSRBGripperGraspAction:
    """Genesis implementation of the grasp action."""

    def __init__(
        self,
        hardware_interface,
        *,
        config: Optional[HSRBGripperGraspConfig] = None,
    ) -> None:
        self.hw = hardware_interface
        self.config = config or HSRBGripperGraspConfig()
        self._state = ActionState.IDLE
        self._goal: Optional[GraspGoal] = None
        self._is_sent_start_grasping = False
        self._result: Optional[GraspResult] = None

    def set_goal(self, goal: GraspGoal) -> None:
        self._goal = goal
        self._state = ActionState.RUNNING
        self._is_sent_start_grasping = False
        self._result = None

    def cancel(self) -> None:
        self._goal = None
        self._state = ActionState.CANCELED
        self._is_sent_start_grasping = False
        self._result = None

    def step(self) -> ActionStatus:
        if self._state != ActionState.RUNNING or self._goal is None:
            return ActionStatus(self._state, self._result)
        grasping_flag = self.hw.get_current_grasping_flag()
        if grasping_flag:
            start_flag = False
            self._is_sent_start_grasping = True
        else:
            start_flag = not self._is_sent_start_grasping
        self.hw.set_grasp_command(start_flag, self._goal.effort)
        self._check_for_success()
        return ActionStatus(self._state, self._result)

    def _check_for_success(self) -> None:
        grasping_flag = self.hw.get_current_grasping_flag()
        if not (self._is_sent_start_grasping and not grasping_flag and self._goal):
            return
        result = GraspResult(stalled=True, effort=self.hw.get_current_torque())
        if abs(self._goal.effort - result.effort) < self.config.torque_goal_tolerance:
            self._state = ActionState.SUCCEEDED
        else:
            self._state = ActionState.ABORTED
        self._result = result
        self._goal = None


class HSRBGripperController:
    """Facade bundling apply-force and grasp behaviours together."""

    def __init__(
        self,
        hardware_interface,
        *,
        apply_force_config: Optional[HSRBGripperApplyForceConfig] = None,
        grasp_config: Optional[HSRBGripperGraspConfig] = None,
        calculator: Optional[HrhGripperApplyForceCalculator] = None,
    ) -> None:
        self.apply_force_action = HSRBGripperApplyForceAction(
            hardware_interface,
            config=apply_force_config,
            calculator=calculator,
        )
        self.grasp_action = HSRBGripperGraspAction(hardware_interface, config=grasp_config)

    def set_apply_force_goal(self, goal: ApplyForceGoal) -> None:
        self.apply_force_action.set_goal(goal)

    def step_apply_force(self, dt: float) -> ActionStatus:
        return self.apply_force_action.step(dt)

    def cancel_apply_force(self) -> None:
        self.apply_force_action.cancel()

    def set_grasp_goal(self, goal: GraspGoal) -> None:
        self.grasp_action.set_goal(goal)

    def step_grasp(self) -> ActionStatus:
        return self.grasp_action.step()

    def cancel_grasp(self) -> None:
        self.grasp_action.cancel()


class HSRBGenesisGripperInterfaceBatch:
    def __init__(
        self,
        entity,
        *,
        motor_joint: str = "hand_motor_joint",
        left_spring_joint: str = "hand_l_spring_proximal_joint",
        right_spring_joint: str = "hand_r_spring_proximal_joint",
    ) -> None:
        self.entity = entity
        self.motor_idx = _first_dof_index(entity.get_joint(motor_joint).dofs_idx_local)
        self.left_spring_idx = _first_dof_index(
            entity.get_joint(left_spring_joint).dofs_idx_local
        )
        self.right_spring_idx = _first_dof_index(
            entity.get_joint(right_spring_joint).dofs_idx_local
        )

        self._mimic_children = HSRBGenesisGripperInterface(
            entity
        )._default_mimic_children()
        self._mimic_joint_names: list[str] = []
        self._mimic_multipliers: list[float] = []
        self._mimic_offsets: list[float] = []
        for child in self._mimic_children.get(
            motor_joint,
            [],
        ):
            self._mimic_joint_names.append(child.joint)
            self._mimic_multipliers.append(float(child.multiplier))
            self._mimic_offsets.append(float(child.offset))

        dof_indices: list[int] = [self.motor_idx]
        valid_multipliers: list[float] = []
        valid_offsets: list[float] = []
        valid_mimic_joint_names: list[str] = []
        for name, mult, offset in zip(
            self._mimic_joint_names,
            self._mimic_multipliers,
            self._mimic_offsets,
        ):
            try:
                idx = _first_dof_index(entity.get_joint(name).dofs_idx_local)
            except Exception:
                continue
            dof_indices.append(idx)
            valid_multipliers.append(mult)
            valid_offsets.append(offset)
            valid_mimic_joint_names.append(name)

        self._command_dofs_idx_local = dof_indices
        self._mimic_joint_names = valid_mimic_joint_names
        self._mimic_multipliers_t = torch.tensor(
            valid_multipliers,
            device=gs.device,
            dtype=gs.tc_float,
        )
        self._mimic_offsets_t = torch.tensor(
            valid_offsets,
            device=gs.device,
            dtype=gs.tc_float,
        )

    def get_motor_position(
        self,
        *,
        envs_idx: torch.Tensor,
    ) -> torch.Tensor:
        return to_torch(
            self.entity.get_dofs_position(
                dofs_idx_local=[self.motor_idx],
                envs_idx=envs_idx,
            )
        ).reshape(-1).to(
            device=gs.device,
            dtype=gs.tc_float,
        )

    def get_motor_velocity(
        self,
        *,
        envs_idx: torch.Tensor,
    ) -> torch.Tensor:
        return to_torch(
            self.entity.get_dofs_velocity(
                dofs_idx_local=[self.motor_idx],
                envs_idx=envs_idx,
            )
        ).reshape(-1).to(
            device=gs.device,
            dtype=gs.tc_float,
        )

    def get_left_spring_position(
        self,
        *,
        envs_idx: torch.Tensor,
    ) -> torch.Tensor:
        out = to_torch(
            self.entity.get_dofs_position(
                dofs_idx_local=[self.left_spring_idx],
                envs_idx=envs_idx,
            )
        )
        out = out.reshape(-1)
        return out.to(
            device=gs.device,
            dtype=gs.tc_float,
        )

    def get_right_spring_position(
        self,
        *,
        envs_idx: torch.Tensor,
    ) -> torch.Tensor:
        out = to_torch(
            self.entity.get_dofs_position(
                dofs_idx_local=[self.right_spring_idx],
                envs_idx=envs_idx,
            )
        )
        out = out.reshape(-1)
        return out.to(
            device=gs.device,
            dtype=gs.tc_float,
        )

    def command_motor_position(
        self,
        position: torch.Tensor,
        *,
        envs_idx: torch.Tensor,
    ) -> None:
        position = (
            to_torch(position)
            .reshape(-1)
            .to(
                device=gs.device,
                dtype=gs.tc_float,
            )
        )
        if self._mimic_multipliers_t.numel() == 0:
            cmd = position.reshape(-1, 1)
        else:
            mimic = (
                position.unsqueeze(1)
                * self._mimic_multipliers_t.unsqueeze(0)
                + self._mimic_offsets_t.unsqueeze(0)
            )
            cmd = torch.cat([position.reshape(-1, 1), mimic], dim=1)

        self.entity.control_dofs_position(
            cmd,
            dofs_idx_local=self._command_dofs_idx_local,
            envs_idx=envs_idx,
        )


class HrhGripperApplyForceCalculatorBatch:
    def __init__(self, calibration_file_path: Optional[str] = None):
        self.hand_spring_coeff = 1.0
        self.arm_length = 1.0
        self._left_xy: torch.Tensor | None = None
        self._right_xy: torch.Tensor | None = None
        if calibration_file_path:
            self._load_force_calibration_data(calibration_file_path)

    def _load_force_calibration_data(self, path: str) -> None:
        try:
            node = yaml.safe_load(open(path, "r", encoding="utf-8"))
        except OSError:
            return
        if not isinstance(node, dict):
            return
        left = node.get("hand_left_force", [])
        right = node.get("hand_right_force", [])
        if left and all(
            isinstance(entry, Sequence) and len(entry) == 2 for entry in left
        ):
            self._left_xy = torch.tensor(left, device=gs.device, dtype=torch.float64)
        if right and all(
            isinstance(entry, Sequence) and len(entry) == 2 for entry in right
        ):
            self._right_xy = torch.tensor(right, device=gs.device, dtype=torch.float64)
        self.hand_spring_coeff = float(node.get("hand_spring", self.hand_spring_coeff))
        self.arm_length = float(node.get("arm_length", self.arm_length))

    def get_current_force(
        self,
        hand_motor_pos: torch.Tensor,
        left_spring_proximal_joint_pos: torch.Tensor,
        right_spring_proximal_joint_pos: torch.Tensor,
    ) -> torch.Tensor:
        left_force = self._calculate_force(
            hand_motor_pos,
            left_spring_proximal_joint_pos,
            self._left_xy,
        )
        right_force = self._calculate_force(
            hand_motor_pos,
            right_spring_proximal_joint_pos,
            self._right_xy,
        )
        return 0.5 * (left_force + right_force)

    def _calculate_force(
        self,
        hand_motor_pos: torch.Tensor,
        spring_proximal_joint_pos: torch.Tensor,
        xy: torch.Tensor | None,
    ) -> torch.Tensor:
        spring_force = (
            spring_proximal_joint_pos
            * float(self.hand_spring_coeff)
            / float(self.arm_length)
        )
        if xy is None or xy.numel() == 0:
            return spring_force

        x = xy[:, 0]
        y = xy[:, 1]
        motor = hand_motor_pos.to(dtype=torch.float64, device=gs.device)
        spring_force64 = spring_force.to(dtype=torch.float64, device=gs.device)

        idx = torch.bucketize(motor, x, right=False) - 1
        idx = torch.clamp(idx, 0, max(0, int(x.numel()) - 2))

        x0 = x[idx]
        x1 = x[idx + 1]
        y0 = y[idx]
        y1 = y[idx + 1]
        denom = x1 - x0
        denom = torch.where(denom.abs() <= torch.finfo(torch.float64).eps, torch.ones_like(denom), denom)
        t = (motor - x0) / denom
        internal = y0 + t * (y1 - y0)

        below = motor < x[0]
        internal = torch.where(below, y[0].expand_as(internal), internal)
        above = motor >= x[-1]
        internal = torch.where(above, torch.zeros_like(internal), internal)

        out = spring_force64 - internal
        out = torch.clamp(out, min=0.0)
        return out.to(dtype=hand_motor_pos.dtype, device=gs.device)


class HSRBGripperApplyForceActionBatch:
    def __init__(
        self,
        hardware_interface: HSRBGenesisGripperInterfaceBatch,
        *,
        n_envs: int,
        config: Optional[HSRBGripperApplyForceConfig] = None,
        calculator: Optional[HrhGripperApplyForceCalculatorBatch] = None,
    ) -> None:
        self.hw = hardware_interface
        self.config = config or HSRBGripperApplyForceConfig()
        self.calculator = calculator or HrhGripperApplyForceCalculatorBatch(
            self.config.force_calib_data_path
        )

        self._n_envs = int(n_envs)
        self._time = torch.tensor(0.0, device=gs.device, dtype=gs.tc_float)

        self._state = torch.full(
            (self._n_envs,),
            int(ActionState.IDLE),
            device=gs.device,
            dtype=torch.int64,
        )
        self._has_goal = torch.zeros(
            (self._n_envs,),
            device=gs.device,
            dtype=torch.bool,
        )
        self._goal_effort = torch.zeros(
            (self._n_envs,),
            device=gs.device,
            dtype=gs.tc_float,
        )
        self._goal_do_control_stop = torch.zeros(
            (self._n_envs,),
            device=gs.device,
            dtype=torch.bool,
        )
        self._active_mask = torch.zeros(
            (self._n_envs,),
            device=gs.device,
            dtype=torch.bool,
        )
        self._current_force_lpf = torch.zeros(
            (self._n_envs,),
            device=gs.device,
            dtype=gs.tc_float,
        )
        self._last_force_sample = torch.zeros(
            (self._n_envs,),
            device=gs.device,
            dtype=gs.tc_float,
        )
        self._force_ierr = torch.zeros(
            (self._n_envs,),
            device=gs.device,
            dtype=gs.tc_float,
        )
        self._last_movement_time = torch.zeros(
            (self._n_envs,),
            device=gs.device,
            dtype=gs.tc_float,
        )
        self._maintained_position = torch.zeros(
            (self._n_envs,),
            device=gs.device,
            dtype=gs.tc_float,
        )
        self._result_stalled = torch.zeros(
            (self._n_envs,),
            device=gs.device,
            dtype=torch.bool,
        )
        self._result_effort = torch.zeros(
            (self._n_envs,),
            device=gs.device,
            dtype=gs.tc_float,
        )

    def set_goal(
        self,
        *,
        effort: torch.Tensor,
        active_mask: torch.Tensor,
        do_control_stop: bool = False,
        env_ids: torch.Tensor,
    ) -> None:
        env_ids = (
            to_torch(env_ids)
            .to(device=gs.device, dtype=torch.int64)
            .reshape(-1)
        )
        if env_ids.numel() == 0:
            return
        effort = (
            to_torch(effort)
            .to(device=gs.device, dtype=gs.tc_float)
            .reshape(-1)
        )
        active_mask = (
            to_torch(active_mask)
            .to(device=gs.device, dtype=torch.bool)
            .reshape(-1)
        )

        self._has_goal[env_ids] = True
        self._goal_effort[env_ids] = effort
        self._goal_do_control_stop[env_ids] = bool(do_control_stop)
        self._active_mask[env_ids] = active_mask
        self._state[env_ids] = int(ActionState.RUNNING)
        self._force_ierr[env_ids] = 0.0
        self._current_force_lpf[env_ids] = 0.0
        self._last_force_sample[env_ids] = 0.0
        self._last_movement_time[env_ids] = self._time
        motor_pos = self.hw.get_motor_position(
            envs_idx=env_ids,
        )
        self._maintained_position[env_ids] = motor_pos

    def cancel(self, *, env_ids: torch.Tensor) -> None:
        env_ids = (
            to_torch(env_ids)
            .to(device=gs.device, dtype=torch.int64)
            .reshape(-1)
        )
        if env_ids.numel() == 0:
            return
        self._has_goal[env_ids] = False
        self._active_mask[env_ids] = False
        self._state[env_ids] = int(ActionState.CANCELED)

    def step(self, dt: float, *, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        env_ids = (
            to_torch(env_ids)
            .to(device=gs.device, dtype=torch.int64)
            .reshape(-1)
        )
        if env_ids.numel() == 0:
            return {}
        dt_t = torch.tensor(float(dt), device=gs.device, dtype=gs.tc_float)
        self._time = self._time + dt_t

        state = self._state[env_ids]
        has_goal = self._has_goal[env_ids]
        active_mask = self._active_mask[env_ids]
        goal_stop = self._goal_do_control_stop[env_ids]

        running = (state == int(ActionState.RUNNING)) & has_goal
        should_command = (running & active_mask) | (has_goal & (~goal_stop))

        motor_pos = self.hw.get_motor_position(envs_idx=env_ids)
        motor_vel = self.hw.get_motor_velocity(envs_idx=env_ids)
        left_spring = self.hw.get_left_spring_position(envs_idx=env_ids)
        right_spring = self.hw.get_right_spring_position(envs_idx=env_ids)

        ref_force = self._goal_effort[env_ids]
        current_force = self.calculator.get_current_force(
            motor_pos,
            left_spring,
            right_spring,
        )

        coeff = torch.tensor(
            float(self.config.force_lpf_coeff), device=gs.device, dtype=gs.tc_float
        )
        last_sample = self._last_force_sample[env_ids]
        current_lpf = (1.0 - coeff) * current_force + coeff * last_sample
        self._current_force_lpf[env_ids] = torch.where(
            should_command,
            current_lpf,
            self._current_force_lpf[env_ids],
        )
        self._last_force_sample[env_ids] = torch.where(
            should_command,
            current_lpf,
            last_sample,
        )

        error = current_lpf - ref_force
        ierr = self._force_ierr[env_ids] + error
        ierr = torch.clamp(
            ierr,
            min=-float(self.config.force_ierr_max),
            max=float(self.config.force_ierr_max),
        )
        self._force_ierr[env_ids] = torch.where(
            should_command,
            ierr,
            self._force_ierr[env_ids],
        )

        delta = (
            float(self.config.force_control_pgain) * error
            + float(self.config.force_control_igain) * ierr
            + float(self.config.force_control_dgain) * (current_lpf - current_force)
        )
        commanded = motor_pos + delta
        maintained = self._maintained_position[env_ids]
        commanded = torch.where(should_command, commanded, maintained)

        self.hw.command_motor_position(commanded, envs_idx=env_ids)
        self._maintained_position[env_ids] = commanded

        moving = motor_vel.abs() > float(self.config.stall_velocity_threshold)
        last_move = self._last_movement_time[env_ids]
        last_move = torch.where(
            moving,
            self._time.expand_as(last_move),
            last_move,
        )
        self._last_movement_time[env_ids] = last_move

        stalled = (self._time - last_move) > float(self.config.stall_timeout)
        done_mask = running & stalled
        measured_force = self._current_force_lpf[env_ids]
        success = (ref_force - measured_force).abs() < float(
            self.config.goal_tolerance
        )
        success = success & done_mask
        fail = (~success) & done_mask

        state_next = self._state[env_ids]
        state_next = torch.where(
            success,
            int(ActionState.SUCCEEDED),
            state_next,
        )
        state_next = torch.where(
            fail,
            int(ActionState.ABORTED),
            state_next,
        )
        self._state[env_ids] = state_next

        self._result_stalled[env_ids] = torch.where(
            done_mask,
            torch.ones_like(done_mask),
            self._result_stalled[env_ids],
        )
        self._result_effort[env_ids] = torch.where(
            done_mask,
            measured_force,
            self._result_effort[env_ids],
        )

        return {
            "state": self._state[env_ids],
            "has_goal": self._has_goal[env_ids],
            "active_mask": self._active_mask[env_ids],
            "measured_force": self._current_force_lpf[env_ids],
            "result_stalled": self._result_stalled[env_ids],
            "result_effort": self._result_effort[env_ids],
        }


class HSRBGripperControllerBatch:
    def __init__(
        self,
        entity,
        *,
        n_envs: int,
        apply_force_config: Optional[HSRBGripperApplyForceConfig] = None,
        calculator: Optional[HrhGripperApplyForceCalculatorBatch] = None,
    ) -> None:
        self.hw = HSRBGenesisGripperInterfaceBatch(entity)
        self.apply_force_action = HSRBGripperApplyForceActionBatch(
            self.hw,
            n_envs=int(n_envs),
            config=apply_force_config,
            calculator=calculator,
        )

    def set_apply_force_goal(
        self,
        *,
        effort: torch.Tensor,
        active_mask: torch.Tensor,
        envs_idx: torch.Tensor | Sequence[int],
        do_control_stop: bool = False,
    ) -> None:
        env_ids = (
            to_torch(envs_idx)
            .to(device=gs.device, dtype=torch.int64)
            .reshape(-1)
        )
        self.apply_force_action.set_goal(
            effort=effort,
            active_mask=active_mask,
            do_control_stop=do_control_stop,
            env_ids=env_ids,
        )

    def step_apply_force(
        self,
        dt: float,
        *,
        envs_idx: torch.Tensor | Sequence[int],
    ) -> dict[str, torch.Tensor]:
        env_ids = (
            to_torch(envs_idx)
            .to(device=gs.device, dtype=torch.int64)
            .reshape(-1)
        )
        return self.apply_force_action.step(float(dt), env_ids=env_ids)
