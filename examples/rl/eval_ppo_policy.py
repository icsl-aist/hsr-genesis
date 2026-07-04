"""Evaluate trained PPO residual policy on YCB objects.

Loads a trained PPO model and evaluates it with the curriculum at the
final stage (policy_weight from saved curriculum state).

Usage:
    PYTHONPATH=src .venv/bin/python examples/rl/eval_ppo_policy.py \\
        --model results/ppo_ik_curriculum/ppo_ik_curriculum.zip \\
        --curriculum results/ppo_ik_curriculum/curriculum_state.json \\
        --envs 32 --trials 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import genesis as gs

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stable_baselines3 import PPO

from hsr_pick_rl_env import HSRPickRLEnv, BatchedGenesisVecEnv
from curriculum import CurriculumManager
from grasp_params import OBJECT_NAMES


def evaluate_object(
    model: PPO,
    object_name: str,
    n_envs: int,
    trials: int,
    settle_steps: int,
    seed: int,
    curriculum: CurriculumManager,
) -> list[float]:
    """Evaluate the model on a single object. Returns list of per-trial success rates."""
    env = HSRPickRLEnv(
        n_envs=n_envs,
        object_name=object_name,
        seed=seed,
        settle_steps=settle_steps,
        curriculum=curriculum,
    )
    vec_env = BatchedGenesisVecEnv(env)

    rates = []
    for trial in range(trials):
        obs = vec_env.reset()
        done = False
        steps = 0
        while not done and steps < 800:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = vec_env.step(action)
            done = bool(np.all(dones))
            steps += 1
        # Success rate across all envs in this trial
        trial_success = sum(1 for info in infos if info.get("success", False)) / len(infos)
        rates.append(trial_success)
        print(f"  {object_name} trial {trial}: {trial_success:.2%} ({steps} steps)")
    return rates


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained PPO policy")
    parser.add_argument("--model", type=str, required=True, help="Path to PPO .zip model")
    parser.add_argument("--curriculum", type=str, default=None,
                        help="Path to curriculum_state.json")
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--settle-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--object", type=str, default=None,
                        help="Specific object or 'all' (default: all)")
    args = parser.parse_args()

    try:
        gs.init(backend=gs.gpu)
    except RuntimeError as exc:
        print(f"[Genesis] GPU unavailable ({exc}); falling back to CPU.")
        gs.init(backend=gs.cpu)

    # Load curriculum
    if args.curriculum and Path(args.curriculum).exists():
        curriculum = CurriculumManager.load(args.curriculum)
        print(f"Loaded curriculum: stage={curriculum.stage}, pw={curriculum.policy_weight:.1f}")
    else:
        curriculum = CurriculumManager()
        curriculum.stage_idx = 2
        print("No curriculum file, using stage 2 (policy_weight=0.7)")

    # Load model
    model = PPO.load(args.model, device="auto")
    print(f"Loaded model: {args.model}")

    # Determine objects to eval
    if args.object and args.object != "all":
        objects = [args.object]
    else:
        objects = OBJECT_NAMES

    print(f"\nEvaluating {len(objects)} objects x {args.trials} trials x {args.envs} envs\n")

    all_rates = []
    for obj_name in objects:
        rates = evaluate_object(
            model, obj_name, args.envs, args.trials,
            args.settle_steps, args.seed, curriculum,
        )
        mean_rate = float(np.mean(rates))
        all_rates.append(mean_rate)
        print(f"  {obj_name} mean: {mean_rate:.2%}\n")

    overall = float(np.mean(all_rates))
    print(f"{'='*60}")
    print(f"Overall mean success rate: {overall:.2%}")
    for obj_name, rate in zip(objects, all_rates):
        print(f"  {obj_name}: {rate:.2%}")


if __name__ == "__main__":
    main()
