import math

import gstaichi as ti
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
    def __init__(self, name: str, dof_idx_local: int) -> None:
        self.name = name
        self.dof_idx_local = dof_idx_local


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
        self.last_velocity_envs = None

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
        self.last_velocity_cmd = torch.as_tensor(velocity, dtype=torch.float32)
        self.last_velocity_dofs = list(dofs_idx_local or [])
        self.last_velocity_envs = None if envs_idx is None else torch.as_tensor(envs_idx, dtype=torch.int64)

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
                resolved.append(self.get_joint(key).dof_idx_local)
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
        yaw_velocity_limit=1.8,
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

    assert entity.last_position_cmd is not None
    steer_target = float(entity.last_position_cmd[0])
    inferred_steer_vel = steer_target / 0.1
    assert math.isclose(
        inferred_steer_vel,
        -1.8,
        rel_tol=0.0,
        abs_tol=1.0e-6,
    )


def test_controller_wheel_limit_saturates_wheel_rate():
    cfg = HSRBBaseControllersConfig(
        command_timeout=10.0,
        wheel_velocity_limit=8.5,
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
        torch.tensor([8.5, 8.5], dtype=torch.float32),
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
        yaw_velocity_limit=1.8,
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

    assert entity.last_position_cmd is not None
    steer_targets = entity.last_position_cmd.reshape(-1)
    inferred = steer_targets / 0.1
    assert torch.allclose(inferred, torch.tensor([-1.8, -1.8], dtype=torch.float32), atol=1.0e-6)
