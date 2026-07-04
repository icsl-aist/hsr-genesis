"""CMA-ES grasp parameter optimization for the IK pick pipeline.

Uses EvoTorch's GPU-accelerated CMA-ES to optimize 4 grasp parameters
per YCB object (28D total), maximizing pick success rate.

Usage:
    PYTHONPATH=src .venv/bin/python examples/rl/train_grasp_cmaes.py \\
        --popsize 256 --generations 50 --settle-steps 30

    # Quick test
    PYTHONPATH=src .venv/bin/python examples/rl/train_grasp_cmaes.py \\
        --popsize 8 --generations 3 --settle-steps 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

import genesis as gs

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evotorch import Problem
from evotorch.algorithms import CMAES

from grasp_params import (
    OBJECT_NAMES,
    PARAM_BOUNDS,
    PARAM_DEFAULTS,
    SOLUTION_LENGTH,
    N_OBJECTS,
    N_PARAMS,
    denormalize,
    params_to_dict,
    default_params,
)
from ycb_pick_ik_parallel import HSRPickEnv


class GraspProblem(Problem):
    """EvoTorch Problem that evaluates grasp params via IK pick simulation.

    Each candidate is a 28D vector (4 params x 7 objects).
    Fitness = mean binary success across all 7 objects.
    """

    def __init__(self, *, popsize: int, settle_steps: int, seed: int = 0):
        super().__init__(
            objective_sense="max",
            solution_length=SOLUTION_LENGTH,
            initial_bounds=(0.0, 1.0),  # normalized search space
            device="cpu",
        )
        self.popsize = popsize
        self.settle_steps = settle_steps
        self.seed = seed
        self._envs: dict[str, HSRPickEnv] = {}
        self._gen_count = 0

    def _get_env(self, object_name: str) -> HSRPickEnv:
        if object_name not in self._envs:
            self._envs[object_name] = HSRPickEnv(
                n_envs=self.popsize,
                object_name=object_name,
                show_viewer=False,
                seed=self.seed,
                disable_visualizer=True,
            )
        return self._envs[object_name]

    def _evaluate_batch(self, solutions) -> None:
        n = solutions.values.shape[0]
        raw = solutions.values.clone()  # (n, 28) in [0,1] normalized space

        # Denormalize: [0,1] → actual param ranges.
        # Each object shares the same 4-param bounds, repeated 7 times.
        lo = PARAM_BOUNDS[:, 0].repeat(N_OBJECTS).to(raw.device)  # (28,)
        hi = PARAM_BOUNDS[:, 1].repeat(N_OBJECTS).to(raw.device)  # (28,)
        scaled = raw.clamp(0.0, 1.0) * (hi - lo) + lo  # (n, 28)
        clipped = scaled.reshape(n, N_OBJECTS, N_PARAMS)
        # Round grasp_hold_steps (index 3) to int.
        clipped[..., 3] = torch.round(clipped[..., 3])

        success_per_obj = torch.zeros(n, N_OBJECTS, dtype=torch.float32)

        for obj_idx, obj_name in enumerate(OBJECT_NAMES):
            env = self._get_env(obj_name)
            params_for_obj = clipped[:, obj_idx, :].to(device=gs.device, dtype=gs.tc_float)
            env.grasp_params = params_for_obj
            result = env.run_pick_pipeline(settle_steps=self.settle_steps)
            success_per_obj[:, obj_idx] = result["success_per_env"]

        fitness = success_per_obj.mean(dim=1)  # (n,)
        solutions.set_evals(fitness)

        self._gen_count += 1
        best_idx = int(fitness.argmax())
        per_obj_means = [float(success_per_obj[:, i].mean()) for i in range(N_OBJECTS)]
        print(
            f"  [eval] n={n} best={float(fitness[best_idx]):.3f} "
            f"mean={float(fitness.mean()):.3f} "
            f"per_obj={[f'{v:.2f}' for v in per_obj_means]}"
        )


def train(
    *,
    popsize: int = 256,
    generations: int = 50,
    settle_steps: int = 30,
    seed: int = 0,
    output_dir: str = "results/grasp_cmaes",
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "training_log.csv"
    best_path = output_path / "grasp_cmaes_best.json"

    # Write CSV header.
    with open(csv_path, "w") as f:
        f.write("generation,best_fitness,mean_fitness,min_fitness," + ",".join(OBJECT_NAMES) + "\n")

    problem = GraspProblem(popsize=popsize, settle_steps=settle_steps, seed=seed)

    # Initialize CMA-ES center at the normalized default params.
    # Defaults: [0.15, 0.02, 3.0, 300] → normalized to [0,1] per param range.
    lo_all = PARAM_BOUNDS[:, 0].repeat(N_OBJECTS)
    hi_all = PARAM_BOUNDS[:, 1].repeat(N_OBJECTS)
    defaults_norm = (PARAM_DEFAULTS.repeat(N_OBJECTS) - lo_all) / (hi_all - lo_all)

    cmaes = CMAES(
        problem=problem,
        stdev_init=0.2,
        popsize=popsize,
        center_init=defaults_norm,
    )

    best_fitness = -1.0
    best_params = None

    for gen in range(generations):
        t0 = time.time()
        cmaes.step()
        pop = cmaes.population
        evals = pop.evals
        best_f = float(evals.max())
        mean_f = float(evals.mean())
        min_f = float(evals.min())
        dt = time.time() - t0

        # Track best.
        if best_f > best_fitness:
            best_fitness = best_f
            best_idx = int(evals.argmax())
            best_raw = pop.values[best_idx].clone().cpu()
            # Denormalize from [0,1] to actual param ranges.
            lo = PARAM_BOUNDS[:, 0].repeat(N_OBJECTS)
            hi = PARAM_BOUNDS[:, 1].repeat(N_OBJECTS)
            scaled = best_raw.clamp(0.0, 1.0) * (hi - lo) + lo
            best_matrix = scaled.reshape(N_OBJECTS, N_PARAMS)
            best_matrix[:, 3] = torch.round(best_matrix[:, 3])
            best_params = params_to_dict(best_matrix)

        print(
            f"[gen {gen:>3d}] best={best_f:.3f} mean={mean_f:.3f} min={min_f:.3f} "
            f"({dt:.1f}s) overall_best={best_fitness:.3f}"
        )

        # Log to CSV.
        with open(csv_path, "a") as f:
            f.write(f"{gen},{best_f:.6f},{mean_f:.6f},{min_f:.6f}\n")

        # Checkpoint best.
        if best_params is not None:
            with open(best_path, "w") as f:
                json.dump(
                    {
                        "generation": gen,
                        "fitness": best_fitness,
                        "params": best_params,
                    },
                    f,
                    indent=2,
                )

    print(f"\nTraining complete. Best fitness: {best_fitness:.3f}")
    print(f"Best params saved to: {best_path}")
    print(f"Training log: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CMA-ES grasp parameter optimization")
    parser.add_argument("--popsize", type=int, default=256, help="CMA-ES population size")
    parser.add_argument("--generations", type=int, default=50, help="Number of generations")
    parser.add_argument("--settle-steps", type=int, default=30, help="Object settle steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="results/grasp_cmaes")
    args = parser.parse_args()

    try:
        gs.init(backend=gs.gpu)
    except RuntimeError as exc:
        print(f"[Genesis] GPU unavailable ({exc}); falling back to CPU.")
        gs.init(backend=gs.cpu)

    train(
        popsize=args.popsize,
        generations=args.generations,
        settle_steps=args.settle_steps,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
