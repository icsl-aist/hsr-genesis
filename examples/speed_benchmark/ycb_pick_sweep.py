"""Benchmark HSR IK pick pipeline simulation performance across env counts.

Sweeps the number of parallel environments and measures the wall-clock
performance of the full IK pick pipeline (settle → approach → descend →
grasp → lift) from ``examples/rl/ycb_pick_ik_parallel.py``.

For each env count it reports:
  - total sim steps
  - wall time (s)
  - steps/sec (simulation rate)
  - envs·steps/sec (parallel throughput)
  - episode wall time (s)
  - success rate (sanity check)

Run
---
    PYTHONPATH=src .venv/bin/python examples/speed_benchmark/ycb_pick_sweep.py

    # custom sweep
    PYTHONPATH=src .venv/bin/python examples/speed_benchmark/ycb_pick_sweep.py \
        --envs 1 4 16 64 256 --trials 3

    # save results to CSV
    PYTHONPATH=src .venv/bin/python examples/speed_benchmark/ycb_pick_sweep.py --csv results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

import genesis as gs

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples" / "rl"))


def sync_if_cuda() -> None:
    if gs.device.type == "cuda":
        torch.cuda.synchronize()


def benchmark_env_count(
    n_envs: int,
    *,
    object_name: str,
    settle_steps: int,
    trials: int,
    seed: int,
) -> list[dict]:
    """Run the IK pick pipeline ``trials`` times for ``n_envs`` envs.

    Returns a list of per-trial result dicts.
    """
    from ycb_pick_ik_parallel import HSRPickEnv

    results = []
    for trial in range(trials):
        # Rebuild scene for each trial to get clean state.
        env = HSRPickEnv(
            n_envs=n_envs,
            object_name=object_name,
            show_viewer=False,
            seed=seed + trial,
            disable_visualizer=True,
        )

        # Warmup: a few sim steps to prime kernels / JIT.
        for _ in range(5):
            env.scene.step()
        sync_if_cuda()

        t0 = time.perf_counter()
        summary = env.run_pick_pipeline(settle_steps=settle_steps)
        sync_if_cuda()
        wall = time.perf_counter() - t0

        total_steps = int(env.total_steps.max().item()) if n_envs > 0 else 0
        steps_per_sec = total_steps / wall if wall > 0 else 0.0
        env_steps_per_sec = (total_steps * n_envs) / wall if wall > 0 else 0.0

        result = {
            "n_envs": n_envs,
            "trial": trial,
            "total_steps": total_steps,
            "wall_s": wall,
            "steps_per_sec": steps_per_sec,
            "env_steps_per_sec": env_steps_per_sec,
            "episode_wall_s": wall,
            "success_rate": summary["success_rate"],
            "n_success": summary["n_success"],
            "avg_time_to_success": summary["avg_time_to_success"],
        }
        results.append(result)
        print(
            f"  [envs={n_envs:>4d} trial={trial}] "
            f"wall={wall:.2f}s steps={total_steps} "
            f"steps/s={steps_per_sec:.1f} "
            f"envs·steps/s={env_steps_per_sec:.0f} "
            f"success={summary['success_rate']:.0%}"
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark HSR IK pick pipeline across env counts."
    )
    parser.add_argument(
        "--envs", type=int, nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256],
        help="List of env counts to sweep",
    )
    parser.add_argument("--object", type=str, default="ycb_061_foam_brick")
    parser.add_argument("--settle-steps", type=int, default=50)
    parser.add_argument("--trials", type=int, default=2,
                        help="Number of repeated trials per env count")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--csv", type=str, default=None,
                        help="Save results to a CSV file")
    args = parser.parse_args()

    backend = gs.cpu if args.backend == "cpu" else gs.gpu
    gs.init(backend=backend)

    print(f"[benchmark] object={args.object} backend={args.backend}")
    print(f"[benchmark] envs={args.envs} trials={args.trials} "
          f"settle_steps={args.settle_steps}")
    print()

    all_results = []
    for n_envs in args.envs:
        print(f"--- n_envs={n_envs} ---")
        results = benchmark_env_count(
            n_envs,
            object_name=args.object,
            settle_steps=args.settle_steps,
            trials=args.trials,
            seed=args.seed,
        )
        all_results.extend(results)
        print()

    # Summary table.
    print("=" * 80)
    print(f"{'envs':>6s} {'trials':>7s} {'wall_s':>8s} {'steps/s':>10s} "
          f"{'envs·steps/s':>14s} {'success':>8s}")
    print("-" * 80)
    for n_envs in args.envs:
        trials = [r for r in all_results if r["n_envs"] == n_envs]
        if not trials:
            continue
        avg_wall = sum(r["wall_s"] for r in trials) / len(trials)
        avg_sps = sum(r["steps_per_sec"] for r in trials) / len(trials)
        avg_esps = sum(r["env_steps_per_sec"] for r in trials) / len(trials)
        avg_succ = sum(r["success_rate"] for r in trials) / len(trials)
        print(
            f"{n_envs:>6d} {len(trials):>7d} {avg_wall:>8.2f} "
            f"{avg_sps:>10.1f} {avg_esps:>14.0f} {avg_succ:>7.0%}"
        )
    print("=" * 80)

    if args.csv:
        path = Path(args.csv)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        print(f"[benchmark] results saved to {path}")


if __name__ == "__main__":
    main()
