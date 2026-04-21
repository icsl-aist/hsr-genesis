#!/usr/bin/env python3
# Copyright (C) 2026 Toyota Motor Corporation
import argparse
from pathlib import Path
import pickle

import genesis as gs
gs.init(backend=gs.gpu, precision="32", performance_mode=True)

from rsl_rl.runners import OnPolicyRunner

from env import Environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="grasp")
    parser.add_argument("--stage", type=str, default="rl")
    parser.add_argument("-v", "--vis", action="store_true", default=True)
    parser.add_argument("-B", "--num_envs", type=int, default=16)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--max_iterations", type=int, default=300)
    args = parser.parse_args()

    rl_train_cfg = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.05,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0003,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "relu",
            "actor_hidden_dims": [256, 256, 128],
            "critic_hidden_dims": [256, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": args.exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": args.max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 128,
        "save_interval": 10,
        "empirical_normalization": None,
        "seed": 1,
    }

    log_dir = Path("logs") / f"{args.exp_name + '_' + args.stage}"
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(log_dir / "cfgs.pkl", "wb") as f:
        pickle.dump((rl_train_cfg,), f)

    env = Environment(n_envs=args.num_envs, dt=args.dt, show_viewer=args.vis)
    if args.stage == "bc":
        pass
    else:
        runner = OnPolicyRunner(env, rl_train_cfg, log_dir, device=gs.device)
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
