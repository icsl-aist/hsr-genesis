import argparse
import re
import pickle
from importlib import metadata
from pathlib import Path

import torch

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs
gs.init(backend=gs.gpu, precision="32", performance_mode=True)

from env import Environment


def load_rl_policy(env, train_cfg, log_dir):
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    checkpoint_files = [f for f in log_dir.iterdir() if re.match(r"model_\d+\.pt", f.name)]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")

    try:
        last_ckpt = max(checkpoint_files, key=lambda f: int(re.search(r"model_(\d+)\.pt", f.name).group(1)))
    except (ValueError, AttributeError) as e:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}") from e
    runner.load(last_ckpt)
    print(f"Loaded RL checkpoint from {last_ckpt}")

    return runner.get_inference_policy(device=gs.device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="grasp")
    parser.add_argument("--stage", type=str, default="rl")
    parser.add_argument("-B", "--num_envs", type=int, default=16)
    parser.add_argument("--dt", type=float, default=0.01)
    args = parser.parse_args()

    log_dir = Path("logs") / f"{args.exp_name + '_' + args.stage}"
    rl_train_cfg, = pickle.load(open(log_dir / "cfgs.pkl", "rb"))

    env = Environment(n_envs=args.num_envs, dt=args.dt, show_viewer=True)
    policy = load_rl_policy(env, rl_train_cfg, log_dir)

    obs, _ = env.reset()

    max_sim_step = 20000

    with torch.no_grad():
        for _ in range(max_sim_step):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)


if __name__ == "__main__":
    main()
