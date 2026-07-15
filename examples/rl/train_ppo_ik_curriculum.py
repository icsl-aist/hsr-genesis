"""PPO training for HSR grasp with optional IK guidance.

Trains either a residual policy on top of IK pick trajectories or a direct
policy baseline without IK guidance using SB3 PPO.

Usage:
    PYTHONPATH=src .venv/bin/python examples/rl/train_ppo_ik_curriculum.py \\
        --envs 64 --total-steps 500000 --object ycb_061_foam_brick

    # Quick test
    PYTHONPATH=src .venv/bin/python examples/rl/train_ppo_ik_curriculum.py \\
        --envs 8 --total-steps 10000 --object ycb_061_foam_brick
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import genesis as gs

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from hsr_pick_rl_env import HSRPickRLEnv, BatchedGenesisVecEnv
from curriculum import CurriculumManager, EVAL_INTERVAL


class CurriculumCallback(BaseCallback):
    """SB3 callback that runs eval rounds and updates the curriculum."""

    def __init__(self, curriculum: CurriculumManager, eval_episodes: int = 5, verbose: int = 1):
        super().__init__(verbose)
        self.curriculum = curriculum
        self.eval_episodes = eval_episodes

    def _on_step(self) -> bool:
        total_steps = self.num_timesteps
        if self.curriculum.should_eval(total_steps):
            self._run_eval()
        return True

    def _run_eval(self):
        """Run eval episodes and update curriculum."""
        env = self.model.get_env()
        success_rates = []
        for _ in range(self.eval_episodes):
            obs = env.reset()
            done = False
            steps = 0
            infos = []
            while not done and steps < 800:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, dones, infos = env.step(action)
                done = bool(np.all(dones))
                steps += 1
            # Collect success from all envs at episode end
            for info in infos:
                success_rates.append(float(info.get("success", False)))

        if success_rates:
            mean_success = float(np.mean(success_rates))
        else:
            mean_success = 0.0

        self.curriculum.update(mean_success, self.num_timesteps)
        if self.verbose:
            print(f"[eval] step={self.num_timesteps} success={mean_success:.3f} "
                  f"stage={self.curriculum.stage} pw={self.curriculum.policy_weight:.1f}")


def train(
    *,
    n_envs: int = 64,
    total_steps: int = 500_000,
    object_name: str = "ycb_061_foam_brick",
    settle_steps: int = 30,
    seed: int = 0,
    output_dir: str = "results/ppo_ik_curriculum",
    learning_rate: float = 3e-4,
    n_steps: int = 1024,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    use_ik_guidance: bool = True,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        gs.init(backend=gs.gpu)
    except RuntimeError as exc:
        print(f"[Genesis] GPU unavailable ({exc}); falling back to CPU.")
        gs.init(backend=gs.cpu)

    curriculum = CurriculumManager()

    env = HSRPickRLEnv(
        n_envs=n_envs,
        object_name=object_name,
        seed=seed,
        settle_steps=settle_steps,
        curriculum=curriculum,
        use_ik_guidance=use_ik_guidance,
    )
    vec_env = BatchedGenesisVecEnv(env)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=None,
        policy_kwargs={
            "net_arch": [256, 256],
        },
        seed=seed,
        device="auto",
    )

    callback = CurriculumCallback(curriculum, eval_episodes=3, verbose=1) if use_ik_guidance else None

    print(f"[train] Starting PPO training: {total_steps} steps, {n_envs} envs, object={object_name}")
    if use_ik_guidance:
        print(f"[train] Mode: IK-guided curriculum")
        print(f"[train] Curriculum: 3 stages, policy_weight = [0.0, 0.3, 0.7]")
    else:
        print(f"[train] Mode: direct PPO baseline (no IK guidance)")
    t0 = time.time()
    model.learn(total_timesteps=total_steps, callback=callback)
    dt = time.time() - t0
    print(f"[train] Training complete in {dt:.1f}s")

    run_name = "ppo_ik_curriculum" if use_ik_guidance else "ppo_no_ik_guidance"
    model_path = output_path / run_name
    model.save(str(model_path))
    curriculum.save(str(output_path / "curriculum_state.json"))
    with open(output_path / "run_config.json", "w") as f:
        json.dump({
            "use_ik_guidance": use_ik_guidance,
            "object_name": object_name,
            "n_envs": n_envs,
            "total_steps": total_steps,
        }, f, indent=2)
    print(f"[train] Model saved to {model_path}.zip")
    print(f"[train] Curriculum saved to {output_path / 'curriculum_state.json'}")
    print(f"[train] Run config saved to {output_path / 'run_config.json'}")
    print(f"[train] Final stage: {curriculum.stage}, policy_weight: {curriculum.policy_weight:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO residual policy with IK curriculum")
    parser.add_argument("--envs", type=int, default=64, help="Number of parallel Genesis envs")
    parser.add_argument("--total-steps", type=int, default=500_000, help="Total training steps")
    parser.add_argument("--object", type=str, default="ycb_061_foam_brick",
                        help="YCB object name")
    parser.add_argument("--settle-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="results/ppo_ik_curriculum")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--no-ik-guidance", action="store_true",
                        help="Train direct PPO baseline without IK reference trajectories")
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir == "results/ppo_ik_curriculum" and args.no_ik_guidance:
        output_dir = "results/ppo_no_ik_guidance"

    train(
        n_envs=args.envs,
        total_steps=args.total_steps,
        object_name=args.object,
        settle_steps=args.settle_steps,
        seed=args.seed,
        output_dir=output_dir,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        use_ik_guidance=not args.no_ik_guidance,
    )


if __name__ == "__main__":
    main()
