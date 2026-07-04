"""Evaluate learned grasp parameters on all YCB objects.

Loads a checkpoint from train_grasp_cmaes.py and evaluates the grasp
params across all objects with fixed seeds for reproducibility.

Usage:
    PYTHONPATH=src .venv/bin/python examples/rl/eval_grasp_params.py \\
        --checkpoint results/grasp_cmaes/grasp_cmaes_best.json --envs 64
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import genesis as gs

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from grasp_params import OBJECT_NAMES, params_from_dict
from ycb_pick_ik_parallel import HSRPickEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate learned grasp params")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint JSON")
    parser.add_argument("--envs", type=int, default=64, help="Envs per object")
    parser.add_argument("--trials", type=int, default=3, help="Trials per object")
    parser.add_argument("--settle-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    try:
        gs.init(backend=gs.gpu)
    except RuntimeError as exc:
        print(f"[Genesis] GPU unavailable ({exc}); falling back to CPU.")
        gs.init(backend=gs.cpu)

    with open(args.checkpoint) as f:
        ckpt = json.load(f)
    param_dict = ckpt["params"]
    param_matrix = params_from_dict(param_dict).to(device=gs.device, dtype=gs.tc_float)

    print(f"Loaded checkpoint: gen={ckpt['generation']}, fitness={ckpt['fitness']:.3f}")
    print(f"Evaluating {len(OBJECT_NAMES)} objects x {args.trials} trials x {args.envs} envs\n")

    all_rates = []
    for obj_idx, obj_name in enumerate(OBJECT_NAMES):
        params = param_matrix[obj_idx].unsqueeze(0).expand(args.envs, 4).clone()
        env = HSRPickEnv(
            n_envs=args.envs,
            object_name=obj_name,
            show_viewer=False,
            seed=args.seed + obj_idx,
            disable_visualizer=True,
            grasp_params=params,
        )
        rates = []
        for trial in range(args.trials):
            result = env.run_pick_pipeline(settle_steps=args.settle_steps)
            rates.append(result["success_rate"])
            print(f"  {obj_name} trial {trial}: {result['success_rate']:.2%}")
        mean_rate = float(np.mean(rates))
        all_rates.append(mean_rate)
        print(f"  {obj_name} mean: {mean_rate:.2%}\n")

    overall = float(np.mean(all_rates))
    print(f"{'='*60}")
    print(f"Overall mean success rate: {overall:.2%}")
    for obj_name, rate in zip(OBJECT_NAMES, all_rates):
        print(f"  {obj_name}: {rate:.2%}")


if __name__ == "__main__":
    main()
