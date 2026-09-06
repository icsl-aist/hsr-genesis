import math

import quadrants as ti
import torch


import genesis as gs

if not getattr(gs, "_initialized", False):
    gs.init(backend=gs.cpu, precision="32", logging_level="warning")

from hsr_genesis.base_controller import (  # noqa: E402
    CartSpace,
    HSRBBaseController,
    HSRBBaseControllersConfig,
    IIRFilter,
)

try:
    ti.init(arch=ti.cpu)
except Exception:
    pass


class _FakeJoint:
    def __init__(self, name: str, dofs_idx_local: int) -> None:
        self.name = name
        self.dofs_idx_local = dofs_idx_local


class _FakeEntity:
    def __init__(self, joint_names: list[str], n_envs: int = 1) -> None:
        self._joints: dict[str, _FakeJoint] = {
            name: _FakeJoint(name, i) for i, name in enumerate(joint_names)
        }
        self._pos = torch.zeros((int(n_envs), len(joint_names)), dtype=torch.float32)

        self.last_kp = None
        self.last_kv = None
        self.last_force_range = None

        self.last_velocity_cmd = None
        self.last_velocity_dofs = None
        self.last_steer_velocity_cmd = None
        self.last_steer_velocity_dofs = None

        self.last_position_cmd = None
        self.last_position_dofs = None
        self.last_position_envs = None

    def get_joint(self, name: str) -> _FakeJoint:
        return self._joints[name]

    def set_dofs_kp(self, kp, dofs_idx_local=None, envs_idx=None):
        self.last_kp = (torch.as_tensor(kp, dtype=torch.float32), list(dofs_idx_local or []))

    def set_dofs_kv(self, kv, dofs_idx_local=None, envs_idx=None):
        self.last_kv = (torch.as_tensor(kv, dtype=torch.float32), list(dofs_idx_local or []))

    def set_dofs_force_range(
        self,
        lower,
        upper,
        dofs_idx_local=None,
        envs_idx=None,
    ):
        self.last_force_range = (
            torch.as_tensor(lower, dtype=torch.float32),
            torch.as_tensor(upper, dtype=torch.float32),
            list(dofs_idx_local or []),
        )

    def control_dofs_velocity(
        self,
        velocity,
        dofs_idx_local=None,
        envs_idx=None,
    ):
        vel_t = torch.as_tensor(velocity, dtype=torch.float32)
        dofs = list(dofs_idx_local or [])
        if len(dofs) == 2:
            self.last_velocity_cmd = vel_t
            self.last_velocity_dofs = dofs
            self.last_velocity_envs = None if envs_idx is None else torch.as_tensor(envs_idx, dtype=torch.int64)
        else:
            self.last_steer_velocity_cmd = vel_t
            self.last_steer_velocity_dofs = dofs

    def control_dofs_position(
        self,
        position,
        dofs_idx_local=None,
        envs_idx=None,
    ):
        self.last_position_cmd = torch.as_tensor(position, dtype=torch.float32)
        self.last_position_dofs = list(dofs_idx_local or [])
        self.last_position_envs = None if envs_idx is None else torch.as_tensor(envs_idx, dtype=torch.int64)
        if dofs_idx_local is not None:
            if envs_idx is None:
                for v, idx in zip(self.last_position_cmd.tolist(), dofs_idx_local):
                    self._pos[0, int(idx)] = float(v)
            else:
                envs_idx_arr = torch.as_tensor(envs_idx, dtype=torch.int64).reshape(-1)
                values = self.last_position_cmd
                if values.ndim == 1:
                    values = values.reshape(-1, 1)
                for row, env in enumerate(envs_idx_arr):
                    for col, idx in enumerate(dofs_idx_local):
                        self._pos[int(env), int(idx)] = float(values[row, col].item())

    def get_dofs_position(self, dofs_idx_local=None, envs_idx=None):
        if dofs_idx_local is None:
            if envs_idx is None:
                return self._pos[0].clone()
            return self._pos[torch.as_tensor(envs_idx, dtype=torch.int64)].clone()
        resolved = []
        for key in dofs_idx_local:
            if isinstance(key, str):
                resolved.append(self.get_joint(key).dofs_idx_local)
            else:
                resolved.append(int(key))
        if envs_idx is None:
            return torch.stack([self._pos[0, i] for i in resolved], dim=0)
        envs_idx_arr = torch.as_tensor(envs_idx, dtype=torch.int64).reshape(-1)
        out = torch.zeros((envs_idx_arr.numel(), len(resolved)), dtype=torch.float32)
        for row, env in enumerate(envs_idx_arr):
            out[row, :] = torch.stack([self._pos[int(env), i] for i in resolved], dim=0)
        return out


def test_iir_filter_default_matches_cpp():
    filt = IIRFilter()
    assert math.isclose(filt.update(1.0), 1.0, rel_tol=0.0, abs_tol=0.0)
    assert math.isclose(filt.update(2.0), 2.0, rel_tol=0.0, abs_tol=0.0)
    assert math.isclose(filt.update(3.0), 3.0, rel_tol=0.0, abs_tol=0.0)


def test_iir_filter_normal_matches_cpp():
    eps = 1.0e-5
    filt = IIRFilter(a=[1.0, 0.1, 0.9], b=[0.2, 0.8])
    assert abs(filt.update(1.0) - 0.2) <= eps
    assert abs(filt.update(2.0) - 1.18) <= eps
    assert abs(filt.update(3.0) - 1.902) <= eps
    assert abs(filt.update(4.0) - 1.9478) <= eps


def test_iir_filter_reset_matches_cpp():
    eps = 1.0e-5
    filt = IIRFilter(a=[1.0, 0.1, 0.9], b=[0.2, 0.8])
    filt.reset(1.0)
    assert abs(filt.update(1.0) - 0.0) <= eps
    assert abs(filt.update(2.0) - 0.3) <= eps
    assert abs(filt.update(3.0) - 2.17) <= eps
    assert abs(filt.update(4.0) - 2.713) <= eps


def _make_controller(
    *,
    timeout: float = 0.1,
    n_envs: int = 1,
) -> tuple[HSRBBaseController, _FakeEntity]:
    cfg = HSRBBaseControllersConfig(command_timeout=timeout)

    joint_names = (
        list(cfg.wheel_drive_joints)
        + list(cfg.wheel_passive_joints)
        + [cfg.steer_joint]
    )
    entity = _FakeEntity(joint_names, n_envs=n_envs)

    ctrl = HSRBBaseController(entity, config=cfg)
    return ctrl, entity


def test_controller_timeout_zeros_command():
    ctrl, entity = _make_controller(timeout=0.1)

    cmd = CartSpace()
    cmd.dot_x = 0.1
    cmd.dot_y = 0.0
    cmd.dot_r = 0.0

    ctrl.update_velocity_command(cmd)
    ctrl.step(0.01)
    assert entity.last_velocity_cmd is not None
    assert torch.linalg.norm(entity.last_velocity_cmd) > 0.0

    for _ in range(20):
        ctrl.step(0.01)

    assert entity.last_velocity_cmd is not None
    assert torch.allclose(
        entity.last_velocity_cmd,
        torch.zeros_like(entity.last_velocity_cmd),
    )


def test_controller_yaw_limit_saturates_steer_rate():
    cfg = HSRBBaseControllersConfig(
        command_timeout=10.0,
        yaw_velocity_limit=2.5,
    )
    joint_names = (
        list(cfg.wheel_drive_joints)
        + list(cfg.wheel_passive_joints)
        + [cfg.steer_joint]
    )
    entity = _FakeEntity(joint_names)
    ctrl = HSRBBaseController(entity, config=cfg)

    cmd = CartSpace()
    cmd.dot_x = 0.0
    cmd.dot_y = 0.0
    cmd.dot_r = 10.0

    ctrl.update_velocity_command(cmd)
    ctrl.step(0.1)

    # Velocity mode: steering rate sent via control_dofs_velocity for steer DOF.
    assert entity.last_steer_velocity_cmd is not None
    steer_vel = float(entity.last_steer_velocity_cmd.reshape(-1)[0])
    assert math.isclose(
        steer_vel,
        -2.5,
        rel_tol=0.0,
        abs_tol=1.0e-6,
    )


def test_controller_wheel_limit_saturates_wheel_rate():
    cfg = HSRBBaseControllersConfig(
        command_timeout=10.0,
        wheel_velocity_limit=12.0,
    )
    joint_names = (
        list(cfg.wheel_drive_joints)
        + list(cfg.wheel_passive_joints)
        + [cfg.steer_joint]
    )
    entity = _FakeEntity(joint_names)
    ctrl = HSRBBaseController(entity, config=cfg)

    cmd = CartSpace()
    cmd.dot_x = 1.0
    cmd.dot_y = 0.0
    cmd.dot_r = 0.0

    ctrl.update_velocity_command(cmd)
    ctrl.step(0.01)

    assert entity.last_velocity_cmd is not None
    assert torch.allclose(
        entity.last_velocity_cmd,
        torch.tensor([12.0, 12.0], dtype=torch.float32),
    )


def test_controller_timeout_batch_zeros_command():
    ctrl, entity = _make_controller(timeout=0.05, n_envs=3)
    envs_idx = [0, 2]

    cmds = torch.tensor(
        [
            [0.1, 0.0, 0.0],
            [0.2, 0.1, 0.0],
        ],
        dtype=torch.float32,
    )
    ctrl.update_velocity_command_batch(cmds, envs_idx=envs_idx)
    ctrl.step_batch(0.01, envs_idx=envs_idx)
    assert entity.last_velocity_cmd is not None
    assert torch.linalg.norm(entity.last_velocity_cmd) > 0.0

    for _ in range(10):
        ctrl.step_batch(0.01, envs_idx=envs_idx)

    assert entity.last_velocity_cmd is not None
    assert torch.allclose(
        entity.last_velocity_cmd,
        torch.zeros_like(entity.last_velocity_cmd),
    )

def test_controller_yaw_limit_batch_saturates_steer_rate():
    cfg = HSRBBaseControllersConfig(
        command_timeout=10.0,
        yaw_velocity_limit=2.5,
    )
    joint_names = (
        list(cfg.wheel_drive_joints)
        + list(cfg.wheel_passive_joints)
        + [cfg.steer_joint]
    )
    entity = _FakeEntity(joint_names, n_envs=2)
    ctrl = HSRBBaseController(entity, config=cfg)

    envs_idx = [0, 1]
    cmds = torch.tensor(
        [
            [0.0, 0.0, 10.0],
            [0.0, 0.0, 10.0],
        ],
        dtype=torch.float32,
    )
    ctrl.update_velocity_command_batch(cmds, envs_idx=envs_idx)
    ctrl.step_batch(0.1, envs_idx=envs_idx)

    # Velocity mode: steering rate sent via control_dofs_velocity for steer DOF.
    assert entity.last_steer_velocity_cmd is not None
    steer_rates = entity.last_steer_velocity_cmd.reshape(-1)
    assert torch.allclose(steer_rates, torch.tensor([-2.5, -2.5], dtype=torch.float32), atol=1.0e-6)


# ---------------------------------------------------------------------------
# Upstream HSR-B parity tests (hsrb_base_controllers 3.0.0)
# ---------------------------------------------------------------------------

def test_config_rejects_invalid_limits():
    """Invalid config limits raise ValueError at construction."""
    for kwargs in [
        {"yaw_velocity_limit": 0.0},
        {"wheel_velocity_limit": -1.0},
        {"yaw_acceleration_limit": float("nan")},
        {"wheel_acceleration_limit": float("inf")},
    ]:
        cfg = HSRBBaseControllersConfig(**kwargs)
        joint_names = (
            list(cfg.wheel_drive_joints)
            + list(cfg.wheel_passive_joints)
            + [cfg.steer_joint]
        )
        entity = _FakeEntity(joint_names)
        try:
            HSRBBaseController(entity, config=cfg)
            assert False, f"Should have raised for {kwargs}"
        except ValueError:
            pass


def test_config_defaults_match_public_hsrb():
    """Default config matches public HSR-B Jazzy configuration."""
    cfg = HSRBBaseControllersConfig()
    assert cfg.yaw_velocity_limit == 1.8
    assert cfg.wheel_velocity_limit == 8.5
    assert cfg.yaw_acceleration_limit == 1.0e10
    assert cfg.wheel_acceleration_limit == 1.0e10
    assert cfg.use_base_roll_velocity is True


def test_velocity_mode_is_default():
    """Velocity mode is the HSR-B default; steering uses control_dofs_velocity."""
    ctrl, entity = _make_controller(timeout=10.0)
    assert ctrl._use_base_roll_velocity is True

    cmd = CartSpace()
    cmd.dot_r = 0.5
    ctrl.update_velocity_command(cmd)
    ctrl.step(0.01)

    assert entity.last_steer_velocity_cmd is not None
    assert entity.last_position_cmd is None


def test_position_mode_integrates_steering():
    """Position mode integrates steering rate into a position target."""
    cfg = HSRBBaseControllersConfig(
        command_timeout=10.0,
        use_base_roll_velocity=False,
    )
    joint_names = (
        list(cfg.wheel_drive_joints)
        + list(cfg.wheel_passive_joints)
        + [cfg.steer_joint]
    )
    entity = _FakeEntity(joint_names)
    ctrl = HSRBBaseController(entity, config=cfg)

    cmd = CartSpace()
    cmd.dot_r = 1.0
    ctrl.update_velocity_command(cmd)
    ctrl.step(0.1)

    assert entity.last_position_cmd is not None
    # Steering rate for pure yaw=1.0 at steer=0: steer = -1.0 (before speed limit).
    # After 0.1s integration: pos = -1.0 * 0.1 = -0.1
    steer_pos = float(entity.last_position_cmd.reshape(-1)[0])
    assert abs(steer_pos - (-0.1)) < 1.0e-5


def test_live_transition_velocity_to_position():
    """velocity → position transition reseeds from measured position."""
    ctrl, entity = _make_controller(timeout=10.0)

    # Step once in velocity mode.
    cmd = CartSpace()
    cmd.dot_x = 0.1
    ctrl.update_velocity_command(cmd)
    ctrl.step(0.01)
    assert entity.last_steer_velocity_cmd is not None

    # Transition to position mode.
    ctrl.set_base_roll_velocity_mode(False)
    ctrl.step(0.01)

    # After transition, position mode is active.
    assert ctrl._use_base_roll_velocity is False
    assert entity.last_position_cmd is not None


def test_live_transition_position_to_velocity():
    """position → velocity transition discards position accumulator."""
    cfg = HSRBBaseControllersConfig(
        command_timeout=10.0,
        use_base_roll_velocity=False,
    )
    joint_names = (
        list(cfg.wheel_drive_joints)
        + list(cfg.wheel_passive_joints)
        + [cfg.steer_joint]
    )
    entity = _FakeEntity(joint_names)
    ctrl = HSRBBaseController(entity, config=cfg)

    # Step once in position mode.
    cmd = CartSpace()
    cmd.dot_r = 0.5
    ctrl.update_velocity_command(cmd)
    ctrl.step(0.01)
    assert entity.last_position_cmd is not None

    # Transition to velocity mode.
    ctrl.set_base_roll_velocity_mode(True)
    ctrl.step(0.01)

    assert ctrl._use_base_roll_velocity is True
    assert entity.last_steer_velocity_cmd is not None


def test_dt_zero_is_noop():
    """dt <= 0 does not update time, commands, or limiter state."""
    ctrl, entity = _make_controller(timeout=10.0)
    time_before = ctrl._time

    cmd = CartSpace()
    cmd.dot_x = 0.1
    ctrl.update_velocity_command(cmd)
    ctrl.step(0.0)

    assert ctrl._time == time_before
    # No actuator calls made.
    assert entity.last_velocity_cmd is None or entity.last_velocity_dofs is None


def test_acceleration_defaults_produce_no_ramp():
    """1.0e10 acceleration defaults: solver early-exits, no observable ramp."""
    ctrl, entity = _make_controller(timeout=10.0)

    cmd = CartSpace()
    cmd.dot_x = 0.3
    ctrl.update_velocity_command(cmd)
    ctrl.step(0.01)
    first_vel = entity.last_velocity_cmd.clone()
    ctrl.step(0.01)
    second_vel = entity.last_velocity_cmd.clone()

    # With effectively-disabled acceleration, both steps produce the same command.
    assert torch.allclose(first_vel, second_vel, atol=1.0e-6)


def test_pure_yaw_wheels_zero():
    """Pure yaw command: wheels remain zero, steering opposes yaw."""
    ctrl, entity = _make_controller(timeout=10.0, n_envs=1)
    # Use a small yaw rate within limits.
    cmd = CartSpace()
    cmd.dot_r = 0.5
    ctrl.update_velocity_command(cmd)
    ctrl.step(0.01)

    assert entity.last_velocity_cmd is not None
    wheel_vel = entity.last_velocity_cmd.reshape(-1)
    assert torch.allclose(wheel_vel[:2], torch.zeros(2, dtype=torch.float32), atol=1.0e-5)

    assert entity.last_steer_velocity_cmd is not None
    steer_vel = float(entity.last_steer_velocity_cmd.reshape(-1)[0])
    assert math.isclose(steer_vel, -0.5, abs_tol=1.0e-5)


def test_speed_saturation_uses_common_scale():
    """Speed limiting applies one common scale to all three joint rates."""
    cfg = HSRBBaseControllersConfig(
        command_timeout=10.0,
        yaw_velocity_limit=1.8,
        wheel_velocity_limit=8.5,
    )
    joint_names = (
        list(cfg.wheel_drive_joints)
        + list(cfg.wheel_passive_joints)
        + [cfg.steer_joint]
    )
    entity = _FakeEntity(joint_names)
    ctrl = HSRBBaseController(entity, config=cfg)

    # Command a pure forward velocity that exceeds wheel limit.
    # wheel_rate = vx / R = 1.0 / 0.04 = 25.0 rad/s > 8.5
    cmd = CartSpace()
    cmd.dot_x = 1.0
    ctrl.update_velocity_command(cmd)
    ctrl.step(0.01)

    assert entity.last_velocity_cmd is not None
    wheel_vels = entity.last_velocity_cmd.reshape(-1)[:2]
    # Both wheels should be at the limit (common scale).
    assert torch.allclose(wheel_vels, torch.tensor([8.5, 8.5], dtype=torch.float32), atol=1.0e-4)


def test_forward_inverse_kinematics_roundtrip():
    """Forward then inverse kinematics is identity for arbitrary steer angles."""
    import genesis as gs
    from hsr_genesis.base_controller import _forward_kinematics_batch, _inverse_kinematics_batch

    R, W, D = 0.04, 0.266, 0.11
    steer = torch.tensor([0.0, 0.5, -0.3, 1.2, -2.0], device=gs.device, dtype=torch.float32)
    base_vel = torch.tensor([
        [0.3, 0.0, 0.0],
        [0.0, 0.2, 0.1],
        [0.3, 0.1, -0.05],
        [-0.2, -0.1, 0.3],
        [0.15, -0.15, -0.2],
    ], device=gs.device, dtype=torch.float32)

    joint_rates = _inverse_kinematics_batch(base_vel, steer, R, W, D)
    recovered = _forward_kinematics_batch(joint_rates, steer, R, W, D)
    assert torch.allclose(recovered, base_vel, atol=1.0e-5)


def test_acceleration_envelope_with_finite_limits():
    """With finite acceleration limits, consecutive commands stay within acc*dt."""
    cfg = HSRBBaseControllersConfig(
        command_timeout=10.0,
        yaw_acceleration_limit=5.0,
        wheel_acceleration_limit=50.0,
        yaw_velocity_limit=1.8,
        wheel_velocity_limit=8.5,
    )
    joint_names = (
        list(cfg.wheel_drive_joints)
        + list(cfg.wheel_passive_joints)
        + [cfg.steer_joint]
    )
    entity = _FakeEntity(joint_names)
    ctrl = HSRBBaseController(entity, config=cfg)

    dt = 0.01
    # Step 1: zero command (prev is zero).
    ctrl.step(dt)
    prev_wheels = entity.last_velocity_cmd.reshape(-1)[:2].clone() if entity.last_velocity_cmd is not None else torch.zeros(2)
    prev_steer = entity.last_steer_velocity_cmd.reshape(-1)[0].clone() if entity.last_steer_velocity_cmd is not None else torch.tensor(0.0)

    # Step 2: large forward command.
    cmd = CartSpace()
    cmd.dot_x = 1.0
    ctrl.update_velocity_command(cmd)
    ctrl.step(dt)

    curr_wheels = entity.last_velocity_cmd.reshape(-1)[:2]
    curr_steer = entity.last_steer_velocity_cmd.reshape(-1)[0]

    wheel_delta = (curr_wheels - prev_wheels).abs().max().item()
    steer_delta = (curr_steer - prev_steer).abs().item()

    # acc_limit * dt = 50.0 * 0.01 = 0.5 for wheels, 5.0 * 0.01 = 0.05 for steer.
    assert wheel_delta <= 0.5 + 1.0e-5, f"Wheel delta {wheel_delta} exceeds acc*dt"
    assert steer_delta <= 0.05 + 1.0e-5, f"Steer delta {steer_delta} exceeds acc*dt"
