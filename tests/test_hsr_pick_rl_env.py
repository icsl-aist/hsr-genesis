from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "rl"))

import hsr_pick_rl_env as rl_env  # noqa: E402


def test_batched_vec_env_close_delegates_to_env():
    closed = False

    class DummyEnv:
        n_envs = 2
        observation_space = None
        action_space = None

        def close(self):
            nonlocal closed
            closed = True

    env = rl_env.BatchedGenesisVecEnv.__new__(rl_env.BatchedGenesisVecEnv)
    env.env = DummyEnv()

    rl_env.BatchedGenesisVecEnv.close(env)

    assert closed is True


def test_reset_without_ik_guidance_skips_planner(monkeypatch):
    monkeypatch.setattr(rl_env.gs, "device", "cpu", raising=False)
    monkeypatch.setattr(rl_env.gs, "tc_float", torch.float32, raising=False)

    planner_called = False
    hand_commands: list[torch.Tensor] = []
    reset_calls: list[int] = []

    env = rl_env.HSRPickRLEnv.__new__(rl_env.HSRPickRLEnv)
    env.use_ik_guidance = False
    env.n_envs = 2
    env.settle_steps = 7
    env.envs_all = torch.arange(2)
    env.curriculum = SimpleNamespace(policy_weight=0.3)
    env._success_step = torch.full((2,), -1, dtype=torch.long)
    env._planner = SimpleNamespace(
        plan=lambda _pick_env: (_ for _ in ()).throw(AssertionError("planner should not run"))
    )
    env._pick_env = SimpleNamespace(
        reset=lambda settle_steps: reset_calls.append(settle_steps),
        hand_open=torch.tensor([[1.0], [1.0]], dtype=torch.float32),
        motor_idx=9,
        hsr=SimpleNamespace(
            control_dofs_position=lambda value, dofs_idx_local, envs_idx: hand_commands.append(value.clone())
        ),
    )
    env.get_obs = lambda: np.zeros((2, rl_env.OBS_DIM), dtype=np.float32)

    obs, info = rl_env.HSRPickRLEnv.reset(env)

    assert planner_called is False
    assert reset_calls == [7]
    assert obs.shape == (2, rl_env.OBS_DIM)
    assert np.array_equal(info["ik_success"], np.zeros(2, dtype=bool))
    assert len(hand_commands) == 1


def test_direct_targets_without_ik_guidance_follow_current_state(monkeypatch):
    monkeypatch.setattr(rl_env.gs, "device", "cpu", raising=False)
    monkeypatch.setattr(rl_env.gs, "tc_float", torch.float32, raising=False)

    env = rl_env.HSRPickRLEnv.__new__(rl_env.HSRPickRLEnv)
    env.n_envs = 2
    env.envs_all = torch.arange(2)
    env._pick_env = SimpleNamespace(
        arm_dofs_idx=[0, 1, 2, 3, 4],
        hsr=SimpleNamespace(
            get_dofs_position=lambda dofs_idx_local, envs_idx: torch.tensor(
                [[0.0, 0.1, 0.2, 0.3, 0.4], [0.5, 0.4, 0.3, 0.2, 0.1]],
                dtype=torch.float32,
            ),
            get_pos=lambda envs_idx: torch.tensor(
                [[1.0, 2.0, 0.0], [-1.0, -2.0, 0.0]], dtype=torch.float32
            ),
            get_quat=lambda envs_idx: torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32
            ),
        ),
    )

    action = torch.tensor(
        [
            [1.0, -1.0, 0.5, 0.0, 0.25, 1.0, -1.0, 0.5, 0.0],
            [-0.5, 0.5, -1.0, 1.0, 0.0, -0.5, 0.5, -1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    arm_target, base_target = env._direct_targets_from_action(action)

    expected_arm = torch.tensor(
        [[0.1, 0.0, 0.25, 0.3, 0.425], [0.45, 0.45, 0.2, 0.3, 0.1]],
        dtype=torch.float32,
    )
    expected_base = torch.tensor(
        [[1.05, 1.95, 0.025], [-1.025, -1.975, -0.05]],
        dtype=torch.float32,
    )

    assert torch.allclose(arm_target, expected_arm)
    assert torch.allclose(base_target, expected_base)
