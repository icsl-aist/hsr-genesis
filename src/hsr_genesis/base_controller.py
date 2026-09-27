"""HSR base vehicle controller utilities and kinematics kernels.

Control architecture
--------------------
The physical ``controller`` mode has two layers:

1. ``OmniBaseTrajectoryControl`` accepts world/odom-frame trajectories,
   computes feed-forward + P feedback + velocity damping, and rotates the
   resulting planar velocity into the robot body frame.
2. ``HSRBBaseController`` accepts that body-frame twist (or a raw ``CartSpace``
   command), maps it to right-wheel, left-wheel, and steering-joint rates, then
   sends wheel-velocity and integrated steering-position targets to Genesis.

The mechanism is not a conventional differential drive.  Both driven wheels
are mounted on ``base_roll_link``, which can yaw relative to ``base_link``.
The wheels are 0.11 m behind that steering pivot; two front casters are
passive.  Consequently, wheel-speed difference gives the absolute yaw rate of
the wheel module, while chassis yaw is the module yaw rate minus the internal
steering-joint rate.

Frame invariant: trajectory positions and velocities and Genesis root feedback
are world-frame.  The inner controller input is body-frame
``[forward, left, counter-clockwise yaw]``.  Raw ``CartSpace`` callers must
already provide body-frame velocities; they do not pass through a frame
rotation.

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
    # Body command: dot_x is forward [m/s], dot_y is left [m/s], and
    # dot_r is counter-clockwise chassis yaw rate [rad/s].
    #
    # Let s be steer_angle and rotate the requested translation into the
    # steerable wheel-module frame:
    #
    #   u =  cos(s) * dot_x + sin(s) * dot_y   (along-wheel velocity)
    #   v = -sin(s) * dot_x + cos(s) * dot_y   (cross-wheel velocity)
    #
    # With wheel radius R, separation B, and the drive axle D metres behind
    # the steering pivot, the equations below are:
    #
    #   wheel_r = u/R + B*v/(2*R*D)
    #   wheel_l = u/R - B*v/(2*R*D)
    #   steer   = v/D - dot_r
    #
    # Therefore the wheel module's absolute yaw rate is
    # R/B * (wheel_r-wheel_l) = dot_r+steer.  In particular, pure chassis yaw
    # commands zero wheel rates and the opposite steering rate.  Do not replace
    # this with ordinary differential-drive kinematics.
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


def _forward_kinematics_batch(
    joint_rates: torch.Tensor,
    steer_angles: torch.Tensor,
    wheel_radius: float,
    wheel_separation: float,
    wheel_offset: float,
) -> torch.Tensor:
    """Convert joint rates [right, left, steer] to body velocity [vx, vy, yaw].

    Mirrors ``TwinCasterDrive::ConvertForward`` from hsrb_base_controllers 3.0.0.
    The Jacobian third row is ``[R/B, -R/B, -1]``, so chassis yaw is
    ``R/B*(right - left) - steer``.
    """
    r = float(wheel_radius)
    w = float(wheel_separation)
    d = float(wheel_offset)
    cos_s = torch.cos(steer_angles)
    sin_s = torch.sin(steer_angles)
    right = joint_rates[:, 0]
    left = joint_rates[:, 1]
    steer = joint_rates[:, 2]
    j11 = r * cos_s * 0.5 - r * d * sin_s / w
    j12 = r * cos_s * 0.5 + r * d * sin_s / w
    j21 = r * sin_s * 0.5 + r * d * cos_s / w
    j22 = r * sin_s * 0.5 - r * d * cos_s / w
    base_x = j11 * right + j12 * left
    base_y = j21 * right + j22 * left
    base_yaw = (r / w) * (right - left) - steer
    return torch.stack([base_x, base_y, base_yaw], dim=1)


def _inverse_kinematics_batch(
    base_vel: torch.Tensor,
    steer_angles: torch.Tensor,
    wheel_radius: float,
    wheel_separation: float,
    wheel_offset: float,
) -> torch.Tensor:
    """Convert body velocity [vx, vy, yaw] to joint rates [right, left, steer].

    Torch equivalent of the Taichi ``_vehicle_inverse_kernel`` without speed
    limiting.  Mirrors ``TwinCasterDrive::ConvertInverse`` from 3.0.0.
    """
    r = float(wheel_radius)
    w = float(wheel_separation)
    d = float(wheel_offset)
    cos_s = torch.cos(steer_angles)
    sin_s = torch.sin(steer_angles)
    dot_x = base_vel[:, 0]
    dot_y = base_vel[:, 1]
    dot_r = base_vel[:, 2]
    u = cos_s * dot_x + sin_s * dot_y
    v = -sin_s * dot_x + cos_s * dot_y
    inv_r = 1.0 / r
    inv_d = 1.0 / d
    half_w_inv_r_inv_d = w * 0.5 * inv_r * inv_d
    vel_r = u * inv_r + v * half_w_inv_r_inv_d
    vel_l = u * inv_r - v * half_w_inv_r_inv_d
    vel_steer = v * inv_d - dot_r
    return torch.stack([vel_r, vel_l, vel_steer], dim=1)


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
    # Geometry must match hsrb4s.urdf: the right/left drive wheels have radius
    # 0.04 m and centres at (-0.11, -0.133)/(-0.11, +0.133) relative to the
    # base_roll_link steering pivot.  Hence separation B=0.266 and rearward
    # pivot-to-axle distance D=0.11.

    wheel_separation: float = 0.266
    wheel_radius: float = 0.04
    wheel_offset: float = 0.11

    command_timeout: float = 0.5
    # Public HSR-B speed limits (hsrb_bringup/config/controllers.yaml, Jazzy).
    yaw_velocity_limit: float = 1.8
    wheel_velocity_limit: float = 8.5
    # C++ defaults are 1.0e10, effectively disabling acceleration shaping.
    yaw_acceleration_limit: float = 1.0e10
    wheel_acceleration_limit: float = 1.0e10
    # HSR-B default: velocity-mode steering (use_base_roll_velocity: true).
    use_base_roll_velocity: bool = True

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
    """Inner body-twist controller for the steerable dual-wheel mechanism.

    Input commands are robot-body velocities ``[forward, left, CCW yaw]``.
    ``step_batch`` reads the measured steering angle for inverse kinematics,
    applies uniform speed scaling, optional IIR filters, and a C++-ordered
    acceleration-feasibility search, then velocity-controls the drive wheels.
    Steering defaults to velocity mode (HSR-B ``use_base_roll_velocity``);
    position mode integrates the rate into a target.  Use
    ``set_base_roll_velocity_mode`` for a live controller-wide transition.
    World-frame trajectory handling belongs to ``OmniBaseTrajectoryControl``
    and must not be duplicated here.
    """

    def __init__(
        self,
        entity,
        *,
        config: HSRBBaseControllersConfig | None = None,
    ) -> None:
        self.entity = entity
        self.config = config or HSRBBaseControllersConfig()
        self._validate_config()
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

        self._use_base_roll_velocity = self.config.use_base_roll_velocity
        self._pending_mode_transition: bool | None = None
        self._previous_joint_output = None
        self._previous_base_output = None
        self._initialize_joints()

    def _validate_config(self) -> None:
        """Validate that all limits are finite and strictly positive."""
        for name in (
            "yaw_velocity_limit",
            "wheel_velocity_limit",
            "yaw_acceleration_limit",
            "wheel_acceleration_limit",
        ):
            value = getattr(self.config, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {value}")

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

        # The inner base controller exclusively owns base_roll_joint gains and limits.
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
            old_prev_joint = self._previous_joint_output
            old_prev_base = self._previous_base_output
            old_init = self._initialized_desired_steer_pos_batch
            old_n = 0 if old_cmd is None else int(old_cmd.shape[0])
            self._cmd_batch = torch.zeros((n_envs, 3), device=gs.device, dtype=TORCH_FLOAT)
            self._last_cmd_time_batch = torch.full((n_envs,), -float("inf"), device=gs.device, dtype=TORCH_FLOAT)
            self._desired_steer_pos_batch = torch.zeros((n_envs,), device=gs.device, dtype=TORCH_FLOAT)
            self._previous_joint_output = torch.zeros((n_envs, 3), device=gs.device, dtype=TORCH_FLOAT)
            self._previous_base_output = torch.zeros((n_envs, 3), device=gs.device, dtype=TORCH_FLOAT)
            self._initialized_desired_steer_pos_batch = torch.zeros((n_envs,), device=gs.device, dtype=torch.bool)
            if old_n:
                self._cmd_batch[:old_n] = old_cmd
                self._last_cmd_time_batch[:old_n] = old_last
                self._desired_steer_pos_batch[:old_n] = old_desired
                self._previous_joint_output[:old_n] = old_prev_joint
                self._previous_base_output[:old_n] = old_prev_base
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

    # Acceleration-feasibility search (hsrb_base_controllers 3.0.0 port)
    # ------------------------------------------------------------------

    _TERNARY_ITERATIONS = 36

    @staticmethod
    def _ternary_search(
        cost_fn,
        n: int,
        prefer_right: bool,
    ) -> torch.Tensor:
        """Batched ternary search over ratio in [0, 1].

        ``prefer_right=True`` uses strict ``<`` (MinRight: prefer target side).
        ``prefer_right=False`` uses ``<=`` (MinLeft: prefer braking side).
        Returns ``(n,)`` tensor of optimal ratios.
        """
        low = torch.zeros(n, device=gs.device, dtype=TORCH_FLOAT)
        high = torch.ones(n, device=gs.device, dtype=TORCH_FLOAT)
        for _ in range(HSRBBaseController._TERNARY_ITERATIONS):
            mid1 = (2.0 * low + high) / 3.0
            mid2 = (low + 2.0 * high) / 3.0
            cost1 = cost_fn(mid1)
            cost2 = cost_fn(mid2)
            if prefer_right:
                move_high = cost1 < cost2
            else:
                move_high = cost1 <= cost2
            high = torch.where(move_high, mid2, high)
            low = torch.where(move_high, low, mid1)
        return (low + high) / 2.0

    @staticmethod
    def _calc_velocity_min_max(
        prev: torch.Tensor,
        vel_limit: torch.Tensor,
        acc_limit: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-joint feasible box: max(prev - acc*dt, -vel_limit), min(prev + acc*dt, vel_limit)."""
        vel_max = torch.minimum(prev + acc_limit * dt, vel_limit)
        vel_min = torch.maximum(prev - acc_limit * dt, -vel_limit)
        return vel_min, vel_max

    @staticmethod
    def _joint_command_distance(
        joint_vel: torch.Tensor,
        q_min: torch.Tensor,
        q_max: torch.Tensor,
    ) -> torch.Tensor:
        """Sum of out-of-box distances per environment.  ``(n,)`` tensor."""
        below = torch.clamp(q_min - joint_vel, min=0.0)
        above = torch.clamp(joint_vel - q_max, min=0.0)
        return (below + above).sum(dim=1)

    @staticmethod
    def _is_joint_command_valid(
        joint_vel: torch.Tensor,
        q_min: torch.Tensor,
        q_max: torch.Tensor,
    ) -> torch.Tensor:
        """Boolean mask: True where every joint is inside the feasible box."""
        return ((joint_vel >= q_min) & (joint_vel <= q_max)).all(dim=1)

    def _apply_acceleration_limits_batch(
        self,
        desired_joint: torch.Tensor,
        steer_angles: torch.Tensor,
        dt: float,
        envs_idx_arr: torch.Tensor,
    ) -> torch.Tensor:
        """C++-ordered acceleration-feasibility search (5 paths).

        Returns the final joint command ``(N, 3)`` [right, left, steer].
        """
        n = desired_joint.shape[0]
        R = float(self.config.wheel_radius)
        W = float(self.config.wheel_separation)
        D = float(self.config.wheel_offset)

        prev_joint = self._previous_joint_output[envs_idx_arr]  # (n, 3)
        prev_base = self._previous_base_output[envs_idx_arr]    # (n, 3)

        vel_limit = torch.tensor(
            [self.config.wheel_velocity_limit, self.config.wheel_velocity_limit,
             self.config.yaw_velocity_limit],
            device=gs.device, dtype=TORCH_FLOAT,
        ).unsqueeze(0)
        acc_limit = torch.tensor(
            [self.config.wheel_acceleration_limit, self.config.wheel_acceleration_limit,
             self.config.yaw_acceleration_limit],
            device=gs.device, dtype=TORCH_FLOAT,
        ).unsqueeze(0)

        q_min, q_max = self._calc_velocity_min_max(prev_joint, vel_limit, acc_limit, dt)

        # Early exit: desired is already feasible.
        feasible = self._is_joint_command_valid(desired_joint, q_min, q_max)
        if bool(feasible.all().item()):
            return desired_joint

        # Convert desired joint vector to base velocity for base-space searches.
        base_desired = _forward_kinematics_batch(desired_joint, steer_angles, R, W, D)

        # Path 1: Approach desired base velocity (MinRight).
        def interp_cost(ratio):
            base_cand = base_desired * ratio.unsqueeze(1) + prev_base * (1.0 - ratio.unsqueeze(1))
            joint_cand = _inverse_kinematics_batch(base_cand, steer_angles, R, W, D)
            return self._joint_command_distance(joint_cand, q_min, q_max)

        ratio1 = self._ternary_search(interp_cost, n, prefer_right=True)
        dist1 = interp_cost(ratio1)
        success1 = dist1 == 0.0
        if bool(success1.all().item()):
            base_cand = base_desired * ratio1.unsqueeze(1) + prev_base * (1.0 - ratio1.unsqueeze(1))
            return _inverse_kinematics_batch(base_cand, steer_angles, R, W, D)

        # Path 2: Brake previous base velocity (MinLeft).
        def brake_base_cost(ratio):
            base_cand = prev_base * ratio.unsqueeze(1)
            joint_cand = _inverse_kinematics_batch(base_cand, steer_angles, R, W, D)
            return self._joint_command_distance(joint_cand, q_min, q_max)

        ratio2 = self._ternary_search(brake_base_cost, n, prefer_right=False)
        dist2 = brake_base_cost(ratio2)
        success2 = (dist2 == 0.0) & ~success1
        if bool(success2.any().item()):
            base_cand = prev_base * ratio2.unsqueeze(1)
            joint_cand = _inverse_kinematics_batch(base_cand, steer_angles, R, W, D)
            result = torch.where(success2.unsqueeze(1), joint_cand, desired_joint)
            if bool(success2.all().item()):
                return result
        else:
            result = desired_joint

        # Path 3: Hold previous base velocity.
        joint_hold = _inverse_kinematics_batch(prev_base, steer_angles, R, W, D)
        valid3 = self._is_joint_command_valid(joint_hold, q_min, q_max) & ~success1 & ~success2
        if bool(valid3.any().item()):
            result = torch.where(valid3.unsqueeze(1), joint_hold, result)
            if bool(valid3.all().item()):
                return result

        # Path 4: Brake previous joint velocity (MinLeft).
        def brake_joint_cost(ratio):
            joint_cand = prev_joint * ratio.unsqueeze(1)
            return self._joint_command_distance(joint_cand, q_min, q_max)

        ratio4 = self._ternary_search(brake_joint_cost, n, prefer_right=False)
        dist4 = brake_joint_cost(ratio4)
        success4 = (dist4 == 0.0) & ~success1 & ~success2 & ~valid3
        if bool(success4.any().item()):
            joint_cand = prev_joint * ratio4.unsqueeze(1)
            result = torch.where(success4.unsqueeze(1), joint_cand, result)

        # Path 5: Fallback — use speed-limited filtered desired command.
        return result

    # ------------------------------------------------------------------
    # Steering actuation modes
    # ------------------------------------------------------------------

    def set_base_roll_velocity_mode(self, enabled: bool) -> None:
        """Transition between velocity and position steering mode.

        Applied at the next ``step_batch`` tick:
        - velocity → position: reseed position accumulator from measured position.
        - position → velocity: discard position accumulator and send velocity.
        Preserves ``previous_joint_output`` and ``previous_base_output``.
        """
        self._pending_mode_transition = bool(enabled)

    def _apply_pending_mode_transition(self, steer: torch.Tensor) -> None:
        """Apply a pending mode transition at the start of step_batch."""
        if self._pending_mode_transition is None:
            return
        new_mode = self._pending_mode_transition
        self._pending_mode_transition = None
        if new_mode == self._use_base_roll_velocity:
            # Already in the requested mode, no transition needed.
            return
        self._use_base_roll_velocity = new_mode
        if not new_mode:
            # velocity → position: reseed from measured position.
            self._desired_steer_pos_batch[: steer.shape[0]] = steer
            self._initialized_desired_steer_pos_batch[: steer.shape[0]] = True
        # position → velocity: nothing to do; velocity command is sent directly.

    def step_batch(self, dt: float, *, envs_idx: Sequence[int]) -> None:
        dt = float(dt)
        if dt <= 0.0:
            return
        self._time += dt
        envs_idx_arr = torch.as_tensor(envs_idx, device=gs.device, dtype=torch.int64).reshape(-1)
        if envs_idx_arr.numel() == 0:
            return
        self._ensure_batch_state(int(envs_idx_arr.max().item() + 1))
        assert self._cmd_batch is not None
        assert self._last_cmd_time_batch is not None
        assert self._desired_steer_pos_batch is not None
        assert self._initialized_desired_steer_pos_batch is not None
        assert self._previous_joint_output is not None
        assert self._previous_base_output is not None

        # Read measured steering angle for inverse kinematics.
        steer = to_torch(
            self.entity.get_dofs_position(
                dofs_idx_local=[self.steer_dof_idx_local],
                envs_idx=envs_idx_arr,
            )
        )
        steer = steer.to(device=gs.device, dtype=TORCH_FLOAT).reshape(-1)

        # Apply pending mode transition before any actuation.
        self._apply_pending_mode_transition(steer)

        # Position-mode seeding: initialize desired steering position from
        # measured position on the first position-mode command.
        if not self._use_base_roll_velocity:
            init_mask = ~self._initialized_desired_steer_pos_batch[envs_idx_arr]
            if bool(torch.any(init_mask).item()):
                self._desired_steer_pos_batch[envs_idx_arr[init_mask]] = steer[init_mask]
                self._initialized_desired_steer_pos_batch[envs_idx_arr[init_mask]] = True

        # Timeout: zero stale commands before inverse kinematics.
        active = (self._time - self._last_cmd_time_batch[envs_idx_arr]) <= self.config.command_timeout
        cmd = torch.zeros((envs_idx_arr.numel(), 3), device=gs.device, dtype=TORCH_FLOAT)
        cmd[active] = self._cmd_batch[envs_idx_arr[active]]

        # Reject non-finite commands before actuator calls.
        if not bool(torch.isfinite(cmd).all().item()):
            cmd = torch.nan_to_num(cmd, nan=0.0, posinf=0.0, neginf=0.0)

        # Inverse kinematics with uniform speed scaling (Taichi kernel).
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

        # Optional IIR command filters (after speed limiting, before accel).
        if self._wheel_filter_batch_r is not None:
            out[:, 0] = self._wheel_filter_batch_r.update_batch(out[:, 0])
            out[:, 1] = self._wheel_filter_batch_l.update_batch(out[:, 1])
        if self._steer_filter_batch is not None:
            out[:, 2] = self._steer_filter_batch.update_batch(out[:, 2])

        # Acceleration-feasibility search (C++ 5-path ternary search).
        out = self._apply_acceleration_limits_batch(out, steer, dt, envs_idx_arr)

        # Drive-wheel velocity commands.
        self.entity.control_dofs_velocity(
            out[:, :2],
            dofs_idx_local=self.wheel_drive_dofs_idx_local,
            envs_idx=envs_idx_arr,
        )

        # Steering actuation: velocity mode (default) or position mode.
        if self._use_base_roll_velocity:
            self.entity.control_dofs_velocity(
                out[:, 2:3],
                dofs_idx_local=[self.steer_dof_idx_local],
                envs_idx=envs_idx_arr,
            )
        else:
            self._desired_steer_pos_batch[envs_idx_arr] += out[:, 2] * dt
            self.entity.control_dofs_position(
                self._desired_steer_pos_batch[envs_idx_arr].reshape(-1, 1),
                dofs_idx_local=[self.steer_dof_idx_local],
                envs_idx=envs_idx_arr,
            )

        # Store previous joint and base output for next tick's acceleration search.
        self._previous_joint_output[envs_idx_arr] = out
        R = float(self.config.wheel_radius)
        W = float(self.config.wheel_separation)
        D = float(self.config.wheel_offset)
        self._previous_base_output[envs_idx_arr] = _forward_kinematics_batch(
            out, steer, R, W, D,
        )



@dataclass(frozen=True)
class Trajectory:
    """Base trajectory waypoints — world-frame data only; no controller tuning.

    All fields use world/odom-frame coordinates:
      positions[:, 0]  – world X [m]
      positions[:, 1]  – world Y [m]
      positions[:, 2]  – world yaw [rad]

    This dataclass carries trajectory geometry only.  Outer-loop feedback and
    damping gains are owned by ``OmniBaseTrajectoryControl.TUNING`` and cannot
    be supplied or overridden through a trajectory request.

    Frame contract for optional velocities
    ----------------------------------------
    When ``velocities`` is provided it **must** be expressed in the world/odom
    frame (same frame as ``positions``), i.e. [ẋ_world, ẏ_world, ω_z].

    Rationale: ``OmniBaseTrajectoryControl`` keeps the internal ``_point_before``
    state in world frame (seeded from ``get_vel`` / ``get_ang`` which are world-
    frame Genesis outputs). The interpolated feed-forward velocity is therefore
    also world-frame. ``get_output_velocity_batch`` rotates the combined
    feed-forward + P-error term into body frame as its final step, so callers
    never need to supply body-frame velocities here.

    If ``velocities`` is None the controller falls back to finite-difference
    (``(p[i+1] - p[i]) / dt``), which is automatically world-frame because the
    positions are world-frame.

    ``accelerations`` are validated and interpolated into ``DesiredState`` for
    API compatibility, but the current velocity-output controller does not use
    them.
    """

    positions: torch.Tensor  # (T, 3) – world frame [x, y, yaw]
    time_from_start: torch.Tensor  # (T,)
    velocities: torch.Tensor | None = None  # (T, 3) – world frame, see docstring
    accelerations: torch.Tensor | None = None  # (T, 3) or None
    joint_names: Sequence[str] | None = None


@dataclass(frozen=True)
class DesiredState:
    """Desired trajectory state — all quantities in the world/odom frame.

    positions    – [x_world, y_world, yaw]
    velocities   – [ẋ_world, ẏ_world, ω_z]   (world frame)
    accelerations – [ẍ_world, ÿ_world, α_z]  (world frame)
    """

    positions: torch.Tensor  # (3,) world frame
    velocities: torch.Tensor  # (3,) world frame
    accelerations: torch.Tensor  # (3,) world frame


@dataclass(frozen=True)
class _TrajectoryControlTuning:
    feedback_gain: tuple[float, float, float]
    yaw_derivative_gain: float
    xy_derivative_gain: float


class OmniBaseTrajectoryControl:
    """Trajectory following controller for the HSR omnidirectional base.

    Frame convention (important)
    ----------------------------
    All positions and velocities handled by this class are in the **world/odom
    frame**: [x_world, y_world, yaw].

    ``sample_desired_state`` receives ``current_velocities`` as
    [ẋ_world, ẏ_world, ω_z_world] (from Genesis ``get_vel`` / ``get_ang``).
    The initial ``_point_before`` velocity seed is therefore world-frame.
    Trajectory waypoint velocities (``Trajectory.velocities``) must also be
    world-frame for interpolation to be consistent — see ``Trajectory`` docstring.

    ``get_output_velocity_batch`` converts the combined world-frame output
    velocity into the robot body frame (via R(−yaw)) before returning it to the
    base joint controller.  Callers must not pre-rotate velocities into body
    frame.

    Tuning
    ------
    Outer-loop gains are controller-owned and immutable (``TUNING``); they
    cannot be supplied or overridden by a trajectory request.  The scalar
    ``get_output_velocity`` delegates to the shared batched calculation
    ``get_output_velocity_batch`` so both production paths use identical math.

    Validation and time boundaries
    ------------------------------
    ``validate_trajectory`` rejects non-finite positions, timestamps,
    velocities, and accelerations, and rejects a negative first timestamp.
    Strict timestamp increase and existing shape/name validation are preserved.

    Before a delayed start (``t < 0``) the pose captured at acceptance is held
    with zero feed-forward velocity.  A zero-time first waypoint is reached
    immediately at ``t == 0``.  At and after the final timestamp the final
    position is held with **zero** feed-forward velocity — explicit final
    velocity is ignored.

    Lifecycle
    ---------
    An accepted trajectory remains active until ``reset_current_trajectory``
    or replacement via ``accept_trajectory``.  Sampling past the horizon does
    not release it; the controller keeps issuing feedback-only commands toward
    the final pose.
    """

    TUNING = _TrajectoryControlTuning(
        feedback_gain=(1.0, 1.0, 1.5),
        yaw_derivative_gain=0.3,
        xy_derivative_gain=0.1,
    )

    def __init__(
        self,
        coordinate_names: Sequence[str] = ("odom_x", "odom_y", "odom_t"),
    ) -> None:
        self.coordinate_names = list(coordinate_names)
        if len(self.coordinate_names) != 3:
            raise ValueError("coordinate_names must have length 3")

        self._feedback_gain = torch.tensor(
            self.TUNING.feedback_gain,
            device=gs.device,
            dtype=TORCH_FLOAT,
        )

        self._trajectory: Trajectory | None = None
        self._trajectory_start_time: float | None = None
        self._sampled_already = False
        self._point_before: DesiredState | None = None

        self._accepted_start_position: torch.Tensor | None = None

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

        if not torch.isfinite(positions).all() or not torch.isfinite(time_from_start).all():
            return False
        if traj.velocities is not None and not torch.isfinite(velocities).all():
            return False
        if traj.accelerations is not None and not torch.isfinite(accelerations).all():
            return False
        if time_from_start.numel() == 0:
            return False
        if float(time_from_start[0].item()) < 0.0:
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
        self._accepted_start_position = base_positions.clone()
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
        self._accepted_start_position = None

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
        """Sample the desired state at *time* for the accepted trajectory.

        Time boundary contract
        ----------------------
        * Before the trajectory's time zero (``t < 0``), the pose captured at
          acceptance is held with zero feed-forward velocity.
        * A first waypoint at ``time_from_start[0] == 0`` is reached
          immediately at ``t == 0``.
        * At and after the final timestamp the final position is held with
          **zero** feed-forward velocity (explicit final-velocity is ignored).

        The controller retains an accepted trajectory until ``reset_current_trajectory``
        or replacement via ``accept_trajectory``; sampling past the horizon does
        not release it.
        """
        if self._trajectory is None:
            return False, None, False, 0.0

        start_time = self._ensure_start_time(time)
        t = float(time - start_time)

        # Pre-start hold: before time zero, hold the pose captured at
        # acceptance with zero feed-forward velocity and acceleration.
        if t < 0.0:
            hold_pos = self._accepted_start_position
            desired = DesiredState(
                positions=hold_pos.clone(),
                velocities=torch.zeros_like(hold_pos),
                accelerations=torch.zeros_like(hold_pos),
            )
            return True, desired, True, t

        cur_pos = to_torch(current_positions).to(device=gs.device, dtype=TORCH_FLOAT).reshape(3)
        cur_vel = to_torch(current_velocities).to(device=gs.device, dtype=TORCH_FLOAT).reshape(3)

        traj = self._trajectory
        times = traj.time_from_start
        positions = traj.positions
        velocities = traj.velocities
        accelerations = traj.accelerations

        if not self._sampled_already:
            # Seed the pre-trajectory state with the current world-frame
            # position and velocity.  cur_vel is [ẋ_world, ẏ_world, ω_z_world]
            # as supplied by step_base_trajectory_batched (Genesis get_vel /
            # get_ang outputs are world-frame).  Storing world-frame velocity
            # here keeps the interpolation frame consistent with Trajectory
            # waypoints and the finite-difference fall-back (p1-p0)/dt which
            # is inherently world-frame.
            self._point_before = DesiredState(
                positions=cur_pos.clone(),
                velocities=cur_vel.clone(),  # world-frame [ẋ, ẏ, ω_z]
                accelerations=torch.zeros_like(cur_pos),
            )
            self._sampled_already = True

        # Zero-time first waypoint: reached immediately at t == 0.  For a
        # single waypoint the desired velocity is zero; for multiple waypoints
        # use velocities[0] when present, otherwise the finite-difference to
        # the next waypoint.
        if float(times[0].item()) == 0.0 and t == 0.0:
            p1 = positions[0]
            if positions.shape[0] == 1:
                vel = torch.zeros_like(p1)
            elif velocities is not None:
                vel = velocities[0].clone()
            else:
                vel = (positions[1] - positions[0]) / (
                    float(times[1].item()) - float(times[0].item())
                )
            acc = accelerations[0].clone() if accelerations is not None else torch.zeros_like(p1)
            desired = DesiredState(
                positions=p1.clone(),
                velocities=vel,
                accelerations=acc,
            )
            return True, desired, True, 0.0

        if t >= float(times[-1].item()):
            # Post-horizon hold: final position with zero feed-forward
            # velocity.  Explicit final-velocity is intentionally ignored;
            # acceleration retains its existing field behavior.
            p1 = positions[-1]
            a1 = accelerations[-1] if accelerations is not None else None
            desired = DesiredState(
                positions=p1.clone(),
                velocities=torch.zeros_like(p1),
                accelerations=a1.clone() if a1 is not None else torch.zeros_like(p1),
            )
            return True, desired, False, t - float(times[-1].item())
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

    def get_output_velocity_batch(
        self,
        actual_positions: torch.Tensor,
        desired_positions: torch.Tensor,
        desired_velocities: torch.Tensor,
        *,
        dt: float,
        current_velocities: torch.Tensor,
    ) -> torch.Tensor:
        """Compute body-frame velocity commands for a batch of robots.

        All inputs are world/odom frame ``[x, y, yaw]`` / ``[ẋ, ẏ, ω_z]``.
        Returns ``(N, 3)`` body-frame velocities ``[forward, left, yaw_rate]``.

        This is the single source of truth for the outer-loop calculation; the
        scalar ``get_output_velocity`` delegates here with a one-row tensor.
        """
        actual = to_torch(actual_positions).to(device=gs.device, dtype=TORCH_FLOAT).reshape(-1, 3)
        desired_pos = to_torch(desired_positions).to(device=gs.device, dtype=TORCH_FLOAT).reshape(-1, 3)
        desired_vel = to_torch(desired_velocities).to(device=gs.device, dtype=TORCH_FLOAT).reshape(-1, 3)
        current_vel = to_torch(current_velocities).to(device=gs.device, dtype=TORCH_FLOAT).reshape(-1, 3)

        error = desired_pos - actual
        error = error.clone()
        error[:, 2] = self._wrap_to_pi(error[:, 2])
        output_world = desired_vel + self._feedback_gain.unsqueeze(0) * error
        output_world = output_world.clone()
        output_world[:, :2] -= self.TUNING.xy_derivative_gain * current_vel[:, :2]
        output_world[:, 2] -= self.TUNING.yaw_derivative_gain * current_vel[:, 2]

        yaw_mid = actual[:, 2] + 0.5 * current_vel[:, 2] * float(dt)
        c = torch.cos(yaw_mid)
        s = torch.sin(yaw_mid)
        return torch.stack(
            (
                c * output_world[:, 0] + s * output_world[:, 1],
                -s * output_world[:, 0] + c * output_world[:, 1],
                output_world[:, 2],
            ),
            dim=1,
        )

    def get_output_velocity(
        self,
        actual_positions: torch.Tensor,
        desired_state: DesiredState,
        dt: float = 0.01,
        current_velocities: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the body-frame velocity command for the base joint controller.

        Inputs are all in world/odom frame:
          actual_positions  – [x_world, y_world, yaw]
          desired_state     – world-frame (see DesiredState docstring)
          current_velocities – [ẋ_world, ẏ_world, ω_z_world]  (optional, for RK2)

        Delegates to ``get_output_velocity_batch`` with a one-row tensor so the
        scalar and vector production paths share one calculation.

        Returns body-frame velocity [dot_x_body, dot_y_body, dot_r].
        """
        actual = to_torch(actual_positions).to(device=gs.device, dtype=TORCH_FLOAT).reshape(3)
        if current_velocities is None:
            current_velocities = torch.zeros_like(actual)
        return self.get_output_velocity_batch(
            actual.unsqueeze(0),
            desired_state.positions.unsqueeze(0),
            desired_state.velocities.unsqueeze(0),
            dt=dt,
            current_velocities=current_velocities.unsqueeze(0),
        )[0]


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
