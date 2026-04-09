"""HSR base vehicle controller utilities and kinematics kernels.

Porting status:
- Source: hsrb_base_controllers (ROS/hsrb_base_controllers).
- Scope: ported from the HSRB base controller package with API-aligned data
  structures and kinematic helpers for Genesis integration.

License: Portions ported from hsrb_base_controllers are under the
BSD-compatible terms. This package is released under the
BSD 3-Clause License (see `hsr_genesis/LICENSE.txt`).
"""

import math
from dataclasses import dataclass
from typing import Sequence

try:
    import gstaichi as ti
except Exception:
    import quadrants as ti
import torch


def to_torch(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor
    return torch.as_tensor(tensor, device=gs.device if "gs" in globals() else None)


try:
    import genesis as gs
except Exception:
    gs = None

if "gs" in globals() and getattr(gs, "_initialized", False) and getattr(gs, "ti_float", None) is not None:
    TI_FLOAT = gs.ti_float
    TORCH_FLOAT = gs.tc_float
else:
    TI_FLOAT = ti.f32
    TORCH_FLOAT = torch.float32


@ti.kernel
def _vehicle_inverse_kernel(
    n: ti.i32,
    cmd: ti.types.ndarray(dtype=TI_FLOAT, ndim=2),
    steer_angle: ti.types.ndarray(dtype=TI_FLOAT, ndim=1),
    wheel_radius: TI_FLOAT,
    wheel_separation: TI_FLOAT,
    wheel_offset: TI_FLOAT,
    yaw_velocity_limit: TI_FLOAT,
    wheel_velocity_limit: TI_FLOAT,
    out_jcmd: ti.types.ndarray(dtype=TI_FLOAT, ndim=2),
):
    for i in range(n):
        dot_x = cmd[i, 0]
        dot_y = cmd[i, 1]
        dot_r = cmd[i, 2]

        cos_s = ti.cos(steer_angle[i])
        sin_s = ti.sin(steer_angle[i])

        inv_wr = 1.0 / wheel_radius
        inv_wo = 1.0 / wheel_offset
        half_ws_inv_wr_wo = wheel_separation / 2.0 * inv_wr * inv_wo

        vel_r = (cos_s * inv_wr - sin_s * half_ws_inv_wr_wo) * dot_x
        vel_r += (sin_s * inv_wr + cos_s * half_ws_inv_wr_wo) * dot_y

        vel_l = (cos_s * inv_wr + sin_s * half_ws_inv_wr_wo) * dot_x
        vel_l += (sin_s * inv_wr - cos_s * half_ws_inv_wr_wo) * dot_y

        vel_steer = (-sin_s * inv_wo * dot_x + cos_s * inv_wo * dot_y) - dot_r

        abs_steer = ti.abs(vel_steer)
        if abs_steer > yaw_velocity_limit:
            ratio = abs_steer / yaw_velocity_limit
            vel_steer /= ratio
            vel_r /= ratio
            vel_l /= ratio

        max_wheel = ti.max(ti.abs(vel_r), ti.abs(vel_l))
        if max_wheel > wheel_velocity_limit:
            ratio = max_wheel / wheel_velocity_limit
            vel_steer /= ratio
            vel_r /= ratio
            vel_l /= ratio

        out_jcmd[i, 0] = vel_r
        out_jcmd[i, 1] = vel_l
        out_jcmd[i, 2] = vel_steer


class JointSpace:
    def __init__(self) -> None:
        self.vel_wheel_l = 0.0
        self.vel_wheel_r = 0.0
        self.vel_steer = 0.0


class CartSpace:
    def __init__(self) -> None:
        self.dot_x = 0.0
        self.dot_y = 0.0
        self.dot_r = 0.0


class BaseControlMode:
    CONTROLLER = "controller"
    QPOS = "qpos"

    @classmethod
    def normalize(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("base_control_mode must be a string")
        value = value.strip().lower()
        if value in (cls.CONTROLLER, cls.QPOS):
            return value
        raise ValueError(f"Unknown base_control_mode: {value}")


class IIRFilter:
    def __init__(
        self,
        a: Sequence[float] | None = None,
        b: Sequence[float] | None = None,
    ) -> None:
        self.a = torch.tensor(list(a) if a else [1.0], device=gs.device, dtype=TORCH_FLOAT)
        self.b = torch.tensor(list(b) if b else [1.0], device=gs.device, dtype=TORCH_FLOAT)
        self.reset(0.0)

    def reset(self, value: float) -> None:
        self.x = torch.full((self.b.numel(),), float(value), device=gs.device, dtype=TORCH_FLOAT)
        self.y = torch.full((self.a.numel(),), float(value), device=gs.device, dtype=TORCH_FLOAT)

    def update(self, x_new: float) -> float:
        return float(
            _iir_update_kernel(
                int(self.a.numel()),
                int(self.b.numel()),
                self.a,
                self.b,
                self.x,
                self.y,
                float(x_new),
            )
        )


class IIRFilterBatch:
    def __init__(
        self,
        a: Sequence[float] | None,
        b: Sequence[float] | None,
        n_envs: int,
    ) -> None:
        self.a = torch.tensor(list(a) if a else [1.0], device=gs.device, dtype=TORCH_FLOAT)
        self.b = torch.tensor(list(b) if b else [1.0], device=gs.device, dtype=TORCH_FLOAT)
        self._n_envs = int(n_envs)
        self.reset(0.0)

    def reset(self, value: float) -> None:
        self.x = torch.full((self._n_envs, self.b.numel()), float(value), device=gs.device, dtype=TORCH_FLOAT)
        self.y = torch.full((self._n_envs, self.a.numel()), float(value), device=gs.device, dtype=TORCH_FLOAT)

    def update_batch(self, x_new: torch.Tensor) -> torch.Tensor:
        x_new = to_torch(x_new).reshape(self._n_envs).to(dtype=TORCH_FLOAT, device=gs.device)
        out = torch.zeros((self._n_envs,), device=gs.device, dtype=TORCH_FLOAT)
        _iir_update_batch_kernel(
            int(self._n_envs),
            int(self.a.numel()),
            int(self.b.numel()),
            self.a,
            self.b,
            self.x,
            self.y,
            x_new,
            out,
        )
        return out


@ti.kernel
def _iir_update_kernel(
    len_a: ti.i32,
    len_b: ti.i32,
    a: ti.types.ndarray(dtype=TI_FLOAT, ndim=1),
    b: ti.types.ndarray(dtype=TI_FLOAT, ndim=1),
    x: ti.types.ndarray(dtype=TI_FLOAT, ndim=1),
    y: ti.types.ndarray(dtype=TI_FLOAT, ndim=1),
    x_new: TI_FLOAT,
) -> TI_FLOAT:
    out = ti.cast(0.0, TI_FLOAT)
    for i in range(1, len_a):
        idx = len_a - i
        y[idx] = y[idx - 1]
        out -= a[idx] * y[idx]
    for i in range(1, len_b):
        idx = len_b - i
        x[idx] = x[idx - 1]
        out += b[idx] * x[idx]
    out += b[0] * x_new
    x[0] = x_new
    y[0] = out
    return out


@ti.kernel
def _iir_update_batch_kernel(
    n: ti.i32,
    len_a: ti.i32,
    len_b: ti.i32,
    a: ti.types.ndarray(dtype=TI_FLOAT, ndim=1),
    b: ti.types.ndarray(dtype=TI_FLOAT, ndim=1),
    x: ti.types.ndarray(dtype=TI_FLOAT, ndim=2),
    y: ti.types.ndarray(dtype=TI_FLOAT, ndim=2),
    x_new: ti.types.ndarray(dtype=TI_FLOAT, ndim=1),
    out: ti.types.ndarray(dtype=TI_FLOAT, ndim=1),
):
    for e in range(n):
        acc = ti.cast(0.0, TI_FLOAT)
        for i in range(1, len_a):
            idx = len_a - i
            y[e, idx] = y[e, idx - 1]
            acc -= a[idx] * y[e, idx]
        for i in range(1, len_b):
            idx = len_b - i
            x[e, idx] = x[e, idx - 1]
            acc += b[idx] * x[e, idx]
        acc += b[0] * x_new[e]
        x[e, 0] = x_new[e]
        y[e, 0] = acc
        out[e] = acc


@dataclass(frozen=True)
class HSRBBaseControllersConfig:
    wheel_drive_joints: tuple[str, ...] = (
        "base_r_drive_wheel_joint",
        "base_l_drive_wheel_joint",
    )
    wheel_passive_joints: tuple[str, ...] = (
        "base_r_passive_wheel_x_frame_joint",
        "base_l_passive_wheel_x_frame_joint",
        "base_r_passive_wheel_y_frame_joint",
        "base_l_passive_wheel_y_frame_joint",
        "base_r_passive_wheel_z_joint",
        "base_l_passive_wheel_z_joint",
    )
    steer_joint: str = "base_roll_joint"

    wheel_separation: float = 0.266
    wheel_radius: float = 0.04
    wheel_offset: float = 0.11

    command_timeout: float = 0.5
    yaw_velocity_limit: float = 1.8
    wheel_velocity_limit: float = 8.5

    wheel_command_velocity_filter_a: tuple[float, ...] = ()
    wheel_command_velocity_filter_b: tuple[float, ...] = ()
    steer_command_velocity_filter_a: tuple[float, ...] = ()
    steer_command_velocity_filter_b: tuple[float, ...] = ()

    kp_wheel: float = 100.0
    kv_wheel: float = 62.460087776184096
    wheel_force_limit: float = 87.0

    kp_steer: float = 50.0
    kv_steer: float = 6.324555320336759
    steer_force_limit: float = 50.0
    base_control_mode: str = BaseControlMode.CONTROLLER


class HSRBBaseController:
    def __init__(
        self,
        entity,
        *,
        config: HSRBBaseControllersConfig | None = None,
    ) -> None:
        self.entity = entity
        self.config = config or HSRBBaseControllersConfig()

        self.wheel_drive_dofs_idx_local = []
        for joint_name in self.config.wheel_drive_joints:
            dofs = self.entity.get_joint(joint_name).dofs_idx_local
            if isinstance(dofs, (list, tuple)):
                self.wheel_drive_dofs_idx_local.extend(int(idx) for idx in dofs)
            else:
                self.wheel_drive_dofs_idx_local.append(int(dofs))
        self.wheel_passive_dofs_idx_local = []
        for joint_name in self.config.wheel_passive_joints:
            dofs = self.entity.get_joint(joint_name).dofs_idx_local
            if isinstance(dofs, (list, tuple)):
                self.wheel_passive_dofs_idx_local.extend(int(idx) for idx in dofs)
            else:
                self.wheel_passive_dofs_idx_local.append(int(dofs))
        steer_dofs = self.entity.get_joint(self.config.steer_joint).dofs_idx_local
        if isinstance(steer_dofs, (list, tuple)):
            self.steer_dof_idx_local = int(steer_dofs[0]) if steer_dofs else 0
        else:
            self.steer_dof_idx_local = int(steer_dofs)

        self._time = 0.0
        self._wheel_filter_batch_r = None
        self._wheel_filter_batch_l = None
        self._steer_filter_batch = None

        self._cmd_batch = None
        self._last_cmd_time_batch = None
        self._desired_steer_pos_batch = None
        self._initialized_desired_steer_pos_batch = None
        self._initialize_joints()

    def _initialize_joints(self) -> None:
        drive = self.wheel_drive_dofs_idx_local
        self.entity.set_dofs_kp(
            kp=torch.tensor([self.config.kp_wheel] * len(drive), device=gs.device, dtype=TORCH_FLOAT),
            dofs_idx_local=drive,
        )
        self.entity.set_dofs_kv(
            kv=torch.tensor([self.config.kv_wheel] * len(drive), device=gs.device, dtype=TORCH_FLOAT),
            dofs_idx_local=drive,
        )
        self.entity.set_dofs_force_range(
            lower=torch.tensor([-self.config.wheel_force_limit] * len(drive), device=gs.device, dtype=TORCH_FLOAT),
            upper=torch.tensor([self.config.wheel_force_limit] * len(drive), device=gs.device, dtype=TORCH_FLOAT),
            dofs_idx_local=drive,
        )

        passive = self.wheel_passive_dofs_idx_local
        if passive:
            self.entity.set_dofs_kp(
                kp=torch.zeros((len(passive),), device=gs.device, dtype=TORCH_FLOAT),
                dofs_idx_local=passive,
            )
            self.entity.set_dofs_kv(
                kv=torch.zeros((len(passive),), device=gs.device, dtype=TORCH_FLOAT),
                dofs_idx_local=passive,
            )
            self.entity.set_dofs_force_range(
                lower=torch.full((len(passive),), -float("inf"), device=gs.device, dtype=TORCH_FLOAT),
                upper=torch.full((len(passive),), float("inf"), device=gs.device, dtype=TORCH_FLOAT),
                dofs_idx_local=passive,
            )

        steer = [self.steer_dof_idx_local]
        self.entity.set_dofs_kp(
            kp=torch.tensor([self.config.kp_steer], device=gs.device, dtype=TORCH_FLOAT),
            dofs_idx_local=steer,
        )
        self.entity.set_dofs_kv(
            kv=torch.tensor([self.config.kv_steer], device=gs.device, dtype=TORCH_FLOAT),
            dofs_idx_local=steer,
        )
        self.entity.set_dofs_force_range(
            lower=torch.tensor([-self.config.steer_force_limit], device=gs.device, dtype=TORCH_FLOAT),
            upper=torch.tensor([self.config.steer_force_limit], device=gs.device, dtype=TORCH_FLOAT),
            dofs_idx_local=steer,
        )

    def _ensure_batch_state(self, n_envs: int) -> None:
        n_envs = int(n_envs)
        if self._cmd_batch is None or self._cmd_batch.shape[0] < n_envs:
            old_cmd = self._cmd_batch
            old_last = self._last_cmd_time_batch
            old_desired = self._desired_steer_pos_batch
            old_init = self._initialized_desired_steer_pos_batch
            old_n = 0 if old_cmd is None else int(old_cmd.shape[0])

            self._cmd_batch = torch.zeros((n_envs, 3), device=gs.device, dtype=TORCH_FLOAT)
            self._last_cmd_time_batch = torch.full((n_envs,), -float("inf"), device=gs.device, dtype=TORCH_FLOAT)
            self._desired_steer_pos_batch = torch.zeros((n_envs,), device=gs.device, dtype=TORCH_FLOAT)
            self._initialized_desired_steer_pos_batch = torch.zeros((n_envs,), device=gs.device, dtype=torch.bool)

            if old_n:
                self._cmd_batch[:old_n] = old_cmd
                self._last_cmd_time_batch[:old_n] = old_last
                self._desired_steer_pos_batch[:old_n] = old_desired
                self._initialized_desired_steer_pos_batch[:old_n] = old_init
            if self.config.wheel_command_velocity_filter_a or self.config.wheel_command_velocity_filter_b:
                self._wheel_filter_batch_r = IIRFilterBatch(
                    self.config.wheel_command_velocity_filter_a,
                    self.config.wheel_command_velocity_filter_b,
                    n_envs,
                )
                self._wheel_filter_batch_l = IIRFilterBatch(
                    self.config.wheel_command_velocity_filter_a,
                    self.config.wheel_command_velocity_filter_b,
                    n_envs,
                )
            if self.config.steer_command_velocity_filter_a or self.config.steer_command_velocity_filter_b:
                self._steer_filter_batch = IIRFilterBatch(
                    self.config.steer_command_velocity_filter_a,
                    self.config.steer_command_velocity_filter_b,
                    n_envs,
                )

    def update_velocity_command(self, cmd: CartSpace, *, envs_idx=None) -> None:
        if envs_idx is None:
            envs_idx_arr = torch.tensor([0], device=gs.device, dtype=torch.int64)
        else:
            envs_idx_arr = torch.as_tensor(envs_idx, device=gs.device, dtype=torch.int64).reshape(-1)
        if envs_idx_arr.numel() == 0:
            return
        self._ensure_batch_state(int(envs_idx_arr.max().item() + 1))
        assert self._cmd_batch is not None
        assert self._last_cmd_time_batch is not None

        self._cmd_batch[envs_idx_arr, 0] = float(cmd.dot_x)
        self._cmd_batch[envs_idx_arr, 1] = float(cmd.dot_y)
        self._cmd_batch[envs_idx_arr, 2] = float(cmd.dot_r)
        self._last_cmd_time_batch[envs_idx_arr] = self._time

    def update_velocity_command_batch(
        self,
        cmds: torch.Tensor,
        *,
        envs_idx: Sequence[int],
    ) -> None:
        envs_idx_arr = torch.as_tensor(envs_idx, device=gs.device, dtype=torch.int64).reshape(-1)
        if envs_idx_arr.numel() == 0:
            return
        self._ensure_batch_state(int(envs_idx_arr.max().item() + 1))
        assert self._cmd_batch is not None
        assert self._last_cmd_time_batch is not None

        cmds = to_torch(cmds).to(device=gs.device, dtype=TORCH_FLOAT).reshape(envs_idx_arr.numel(), 3)
        self._cmd_batch[envs_idx_arr] = cmds
        self._last_cmd_time_batch[envs_idx_arr] = self._time

    def step(self, dt: float, *, envs_idx=None) -> None:
        if envs_idx is None:
            self.step_batch(float(dt), envs_idx=[0])
        else:
            self.step_batch(float(dt), envs_idx=envs_idx)

    def step_batch(self, dt: float, *, envs_idx: Sequence[int]) -> None:
        dt = float(dt)
        self._time += dt
        envs_idx_arr = torch.as_tensor(envs_idx, device=gs.device, dtype=torch.int64).reshape(-1)
        if envs_idx_arr.numel() == 0:
            return
        self._ensure_batch_state(int(envs_idx_arr.max().item() + 1))
        assert self._cmd_batch is not None
        assert self._last_cmd_time_batch is not None
        assert self._desired_steer_pos_batch is not None
        assert self._initialized_desired_steer_pos_batch is not None

        steer = to_torch(
            self.entity.get_dofs_position(
                dofs_idx_local=[self.steer_dof_idx_local],
                envs_idx=envs_idx_arr,
            )
        )
        steer = steer.to(device=gs.device, dtype=TORCH_FLOAT).reshape(-1)

        init_mask = ~self._initialized_desired_steer_pos_batch[envs_idx_arr]
        if bool(torch.any(init_mask).item()):
            self._desired_steer_pos_batch[envs_idx_arr[init_mask]] = steer[init_mask]
            self._initialized_desired_steer_pos_batch[envs_idx_arr[init_mask]] = True

        active = (self._time - self._last_cmd_time_batch[envs_idx_arr]) <= self.config.command_timeout
        cmd = torch.zeros((envs_idx_arr.numel(), 3), device=gs.device, dtype=TORCH_FLOAT)
        cmd[active] = self._cmd_batch[envs_idx_arr[active]]

        out = torch.zeros((envs_idx_arr.numel(), 3), device=gs.device, dtype=TORCH_FLOAT)
        _vehicle_inverse_kernel(
            int(envs_idx_arr.numel()),
            cmd,
            steer,
            float(self.config.wheel_radius),
            float(self.config.wheel_separation),
            float(self.config.wheel_offset),
            float(self.config.yaw_velocity_limit),
            float(self.config.wheel_velocity_limit),
            out,
        )

        if self._wheel_filter_batch_r is not None:
            out[:, 0] = self._wheel_filter_batch_r.update_batch(out[:, 0])
            out[:, 1] = self._wheel_filter_batch_l.update_batch(out[:, 1])
        if self._steer_filter_batch is not None:
            out[:, 2] = self._steer_filter_batch.update_batch(out[:, 2])

        self.entity.control_dofs_velocity(
            out[:, :2],
            dofs_idx_local=self.wheel_drive_dofs_idx_local,
            envs_idx=envs_idx_arr,
        )

        self._desired_steer_pos_batch[envs_idx_arr] += out[:, 2] * dt
        self.entity.control_dofs_position(
            self._desired_steer_pos_batch[envs_idx_arr].reshape(-1, 1),
            dofs_idx_local=[self.steer_dof_idx_local],
            envs_idx=envs_idx_arr,
        )


@dataclass(frozen=True)
class Trajectory:
    positions: torch.Tensor  # (T, 3)
    time_from_start: torch.Tensor  # (T,)
    velocities: torch.Tensor | None = None  # (T, 3) or None
    accelerations: torch.Tensor | None = None  # (T, 3) or None
    joint_names: Sequence[str] | None = None


@dataclass(frozen=True)
class DesiredState:
    positions: torch.Tensor  # (3,)
    velocities: torch.Tensor  # (3,)
    accelerations: torch.Tensor  # (3,)


class OmniBaseTrajectoryControl:
    def __init__(
        self,
        coordinate_names: Sequence[str] = ("odom_x", "odom_y", "odom_t"),
        feedback_gain: torch.Tensor | None = None,
        stop_velocity_threshold: float = 0.001,
        stop_time_margin: float = 0.2,
    ) -> None:
        self.coordinate_names = list(coordinate_names)
        if len(self.coordinate_names) != 3:
            raise ValueError("coordinate_names must have length 3")

        self.stop_velocity_threshold = float(stop_velocity_threshold)
        self.stop_time_margin = float(stop_time_margin)

        if feedback_gain is None:
            feedback_gain = torch.tensor([1.0, 1.0, 1.0], device=gs.device, dtype=TORCH_FLOAT)
        self.feedback_gain = to_torch(feedback_gain).to(device=gs.device, dtype=TORCH_FLOAT).reshape(3)

        self._trajectory: Trajectory | None = None
        self._trajectory_start_time: float | None = None
        self._sampled_already = False
        self._point_before: DesiredState | None = None

    @staticmethod
    def _wrap_to_pi(angle: torch.Tensor | float) -> torch.Tensor | float:
        if isinstance(angle, torch.Tensor):
            return (angle + math.pi) % (2.0 * math.pi) - math.pi
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    @classmethod
    def _shortest_angular_distance(cls, from_angle: float, to_angle: float) -> float:
        return float(cls._wrap_to_pi(to_angle - from_angle))

    @staticmethod
    def _make_permutation_vector(names1: Sequence[str], names2: Sequence[str]) -> list[int]:
        if len(names1) != len(names2):
            return []
        perm = []
        for name in names1:
            try:
                perm.append(list(names2).index(name))
            except ValueError:
                return []
        return perm

    def validate_trajectory(self, traj: Trajectory) -> bool:
        positions = to_torch(traj.positions).to(device=gs.device, dtype=TORCH_FLOAT)
        time_from_start = to_torch(traj.time_from_start).to(device=gs.device, dtype=TORCH_FLOAT)

        if positions.ndim != 2 or positions.shape[1] != 3:
            return False
        if time_from_start.ndim != 1 or time_from_start.shape[0] != positions.shape[0]:
            return False
        if traj.velocities is not None:
            velocities = to_torch(traj.velocities).to(device=gs.device, dtype=TORCH_FLOAT)
            if velocities.shape != positions.shape:
                return False
        if traj.accelerations is not None:
            accelerations = to_torch(traj.accelerations).to(device=gs.device, dtype=TORCH_FLOAT)
            if accelerations.shape != positions.shape:
                return False

        if time_from_start.numel() == 0:
            return False
        if not torch.all(time_from_start[1:] > time_from_start[:-1]):
            return False

        if traj.joint_names is not None:
            if len(traj.joint_names) != 3:
                return False
            if self._make_permutation_vector(self.coordinate_names, traj.joint_names) == []:
                return False

        return True

    def accept_trajectory(
        self,
        traj: Trajectory,
        base_positions: torch.Tensor,
        *,
        start_time: float | None = None,
    ) -> None:
        if not self.validate_trajectory(traj):
            raise ValueError("invalid trajectory")

        positions = to_torch(traj.positions).to(device=gs.device, dtype=TORCH_FLOAT)
        time_from_start = to_torch(traj.time_from_start).to(device=gs.device, dtype=TORCH_FLOAT)
        velocities = (
            None if traj.velocities is None else to_torch(traj.velocities).to(device=gs.device, dtype=TORCH_FLOAT)
        )
        accelerations = (
            None if traj.accelerations is None else to_torch(traj.accelerations).to(device=gs.device, dtype=TORCH_FLOAT)
        )

        if traj.joint_names is not None:
            perm = self._make_permutation_vector(self.coordinate_names, traj.joint_names)
            if not perm:
                raise ValueError("trajectory joint_names mismatch")
            positions = positions[:, perm]
            if velocities is not None:
                velocities = velocities[:, perm]
            if accelerations is not None:
                accelerations = accelerations[:, perm]

        base_positions = to_torch(base_positions).to(device=gs.device, dtype=TORCH_FLOAT).reshape(3)
        prev = float(base_positions[2].item())
        yaws = positions[:, 2].clone()
        for i in range(yaws.shape[0]):
            diff = self._shortest_angular_distance(prev, float(yaws[i].item()))
            prev = prev + diff
            yaws[i] = prev
        positions = positions.clone()
        positions[:, 2] = yaws

        self._trajectory = Trajectory(
            positions=positions,
            time_from_start=time_from_start,
            velocities=velocities,
            accelerations=accelerations,
            joint_names=self.coordinate_names,
        )
        self._trajectory_start_time = start_time
        self._sampled_already = False
        self._point_before = None

    def reset_current_trajectory(self) -> None:
        self._trajectory = None
        self._trajectory_start_time = None
        self._sampled_already = False
        self._point_before = None

    def update_active_trajectory(self) -> bool:
        return self._trajectory is not None and self._trajectory.positions.numel() > 0

    def _ensure_start_time(self, time: float) -> float:
        if self._trajectory_start_time is None:
            self._trajectory_start_time = float(time)
        return self._trajectory_start_time

    def sample_desired_state(
        self,
        time: float,
        current_positions: torch.Tensor,
        current_velocities: torch.Tensor,
    ) -> tuple[bool, DesiredState | None, bool, float]:
        if self._trajectory is None:
            return False, None, False, 0.0

        start_time = self._ensure_start_time(time)
        t = float(time - start_time)

        cur_pos = to_torch(current_positions).to(device=gs.device, dtype=TORCH_FLOAT).reshape(3)
        cur_vel = to_torch(current_velocities).to(device=gs.device, dtype=TORCH_FLOAT).reshape(3)

        traj = self._trajectory
        times = traj.time_from_start
        positions = traj.positions
        velocities = traj.velocities
        accelerations = traj.accelerations

        if not self._sampled_already:
            self._point_before = DesiredState(
                positions=cur_pos.clone(),
                velocities=cur_vel.clone(),
                accelerations=torch.zeros_like(cur_pos),
            )
            self._sampled_already = True

        if t <= float(times[0].item()):
            t0 = 0.0
            t1 = float(times[0].item())
            p0 = self._point_before.positions
            p1 = positions[0]
            v0 = self._point_before.velocities
            v1 = velocities[0] if velocities is not None else None
            a0 = self._point_before.accelerations
            a1 = accelerations[0] if accelerations is not None else None
            before_last = True
        elif t >= float(times[-1].item()):
            p1 = positions[-1]
            v1 = velocities[-1] if velocities is not None else None
            a1 = accelerations[-1] if accelerations is not None else None
            desired = DesiredState(
                positions=p1.clone(),
                velocities=v1.clone() if v1 is not None else torch.zeros_like(p1),
                accelerations=a1.clone() if a1 is not None else torch.zeros_like(p1),
            )
            return True, desired, False, t - float(times[-1].item())
        else:
            idx = int(torch.searchsorted(times, torch.tensor(t, device=times.device)).item())
            t0 = float(times[idx - 1].item())
            t1 = float(times[idx].item())
            p0 = positions[idx - 1]
            p1 = positions[idx]
            v0 = velocities[idx - 1] if velocities is not None else None
            v1 = velocities[idx] if velocities is not None else None
            a0 = accelerations[idx - 1] if accelerations is not None else None
            a1 = accelerations[idx] if accelerations is not None else None
            before_last = True

        dt = max(t1 - t0, 1.0e-9)
        alpha = (t - t0) / dt
        alpha_t = torch.tensor(alpha, dtype=p0.dtype, device=p0.device)

        pos = (1.0 - alpha_t) * p0 + alpha_t * p1
        if v0 is None or v1 is None:
            vel = (p1 - p0) / dt
        else:
            vel = (1.0 - alpha_t) * v0 + alpha_t * v1
        if a0 is None or a1 is None:
            acc = torch.zeros_like(pos)
        else:
            acc = (1.0 - alpha_t) * a0 + alpha_t * a1

        desired = DesiredState(positions=pos, velocities=vel, accelerations=acc)
        time_from_point = t - t0
        return True, desired, before_last, time_from_point

    def get_output_velocity(
        self, actual_positions: torch.Tensor, desired_state: DesiredState, dt: float = 0.01, current_velocities: torch.Tensor | None = None
    ) -> torch.Tensor:
        actual_positions = to_torch(actual_positions).to(device=gs.device, dtype=TORCH_FLOAT).reshape(3)

        error_pos = desired_state.positions - actual_positions
        error_pos = error_pos.clone()
        error_pos[2] = self._wrap_to_pi(error_pos[2])

        output_velocity = desired_state.velocities + self.feedback_gain * error_pos

        # Runge-Kutta 2nd order integration in heading frame (from hsr.py fix)
        # Use midpoint angle for better accuracy
        yaw = actual_positions[2]
        if current_velocities is not None:
            current_velocities = to_torch(current_velocities).to(device=gs.device, dtype=TORCH_FLOAT).reshape(3)
            diff_r = current_velocities[2] * dt
            yaw = yaw + 0.5 * diff_r

        c = torch.cos(yaw)
        s = torch.sin(yaw)
        rot = torch.stack(
            [
                torch.stack([c, s, torch.zeros_like(c)], dim=0),
                torch.stack([-s, c, torch.zeros_like(c)], dim=0),
                torch.tensor([0.0, 0.0, 1.0], device=gs.device, dtype=TORCH_FLOAT),
            ],
            dim=0,
        )
        return rot @ output_velocity

    def terminate_control_if_stopped(self, time: float, current_velocities: torch.Tensor) -> bool:
        if self._trajectory is None or self._trajectory_start_time is None:
            return False

        times = self._trajectory.time_from_start
        last_time = float(times[-1].item())
        if time - self._trajectory_start_time <= last_time + self.stop_time_margin:
            return False

        cur_vel = to_torch(current_velocities).to(device=gs.device, dtype=TORCH_FLOAT).reshape(3)
        if torch.linalg.norm(cur_vel) >= self.stop_velocity_threshold:
            return False

        self.reset_current_trajectory()
        return True

    def step(
        self,
        time: float,
        current_positions: torch.Tensor,
        current_velocities: torch.Tensor,
    ) -> tuple[bool, torch.Tensor | None, DesiredState | None]:
        if not self.update_active_trajectory():
            return False, None, None

        ok, desired, _before_last, _time_from_point = self.sample_desired_state(
            time, current_positions, current_velocities
        )
        if not ok or desired is None:
            return True, None, None

        output_velocity = self.get_output_velocity(current_positions, desired)
        return True, output_velocity, desired
