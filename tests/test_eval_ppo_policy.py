from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "rl"))

import eval_ppo_policy as eval_mod  # noqa: E402


def test_evaluate_object_closes_vec_env(monkeypatch):
    closed = False

    class FakeEnv:
        pass

    class FakeVecEnv:
        def __init__(self, env):
            self.env = env

        def reset(self):
            return np.zeros((2, 32), dtype=np.float32)

        def step(self, action):
            infos = [{"success": True}, {"success": False}]
            return np.zeros((2, 32), dtype=np.float32), np.zeros(2, dtype=np.float32), np.array([True, True]), infos

        def close(self):
            nonlocal closed
            closed = True

    class FakeModel:
        def predict(self, obs, deterministic=True):
            return np.zeros((2, 9), dtype=np.float32), None

    monkeypatch.setattr(eval_mod, "HSRPickRLEnv", lambda **kwargs: FakeEnv())
    monkeypatch.setattr(eval_mod, "BatchedGenesisVecEnv", FakeVecEnv)

    rates = eval_mod.evaluate_object(
        FakeModel(),
        "ycb_061_foam_brick",
        n_envs=2,
        trials=1,
        settle_steps=30,
        seed=0,
        curriculum=SimpleNamespace(),
        use_ik_guidance=True,
    )

    assert rates == [0.5]
    assert closed is True
