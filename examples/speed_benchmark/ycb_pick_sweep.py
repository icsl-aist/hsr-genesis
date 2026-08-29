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
  - GPU memory usage (mean / max / peak MiB) via ``nvidia-smi``
  - GPU compute-unit (SM) utilization (mean / max %) via ``nvidia-smi``

Run
---
    # default: auto-scale N (doubling from 1) until the GPU OOMs -- the max N
    # is capped only by GPU memory, not by any hardcoded ceiling.
    PYTHONPATH=src .venv/bin/python examples/speed_benchmark/ycb_pick_sweep.py --csv results.csv

    # explicit sweep
    PYTHONPATH=src .venv/bin/python examples/speed_benchmark/ycb_pick_sweep.py \
        --envs 1 4 16 64 256 --trials 3

    # finer auto-scaling (1.5x steps instead of doubling)
    PYTHONPATH=src .venv/bin/python examples/speed_benchmark/ycb_pick_sweep.py \
        --auto-start 1 --auto-factor 1.5 --csv results.csv

    # save results to CSV
    PYTHONPATH=src .venv/bin/python examples/speed_benchmark/ycb_pick_sweep.py --csv results.csv

    # capture a detailed Nsight Systems trace (one .nsys-rep for the whole
    # sweep, with NVTX ranges labeling each N condition and trial)
    PYTHONPATH=src .venv/bin/python examples/speed_benchmark/ycb_pick_sweep.py \
        --envs 1 32 256 --nsys-out bench.nsys-rep --csv bench.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import subprocess
import sys
import threading
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


# ---------------------------------------------------------------------------
# GPU metrics sampling via nvidia-smi (NVML developer tool)
# ---------------------------------------------------------------------------

# Fields queried from `nvidia-smi --query-gpu`.  ``utilization.gpu`` is the
# percentage of time the GPU's compute units (SMs) were active over the
# sampling window -- i.e. compute-unit usage.  ``utilization.memory`` is the
# memory-controller utilization.  ``memory.used`` is the current framebuffer
# usage in MiB.
_NVIDIA_SMI_QUERY = (
    "memory.used,memory.total,utilization.gpu,utilization.memory,power.draw"
)


class GpuSampler:
    """Sample GPU utilization/memory in a background thread via ``nvidia-smi``.

    Polls ``nvidia-smi --query-gpu`` at a fixed interval and records per-sample
    metrics.  Call :meth:`start` before the measured region and :meth:`stop`
    after, then read :meth:`summary`.  This uses the NVIDIA management library
    through the ``nvidia-smi`` developer CLI, so it works without any extra
    Python dependencies.
    """

    def __init__(self, interval: float = 0.2, gpu_index: int = 0) -> None:
        self.interval = interval
        self.gpu_index = gpu_index
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[dict[str, float]] = []

    def start(self) -> None:
        self._stop.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        cmd = [
            "nvidia-smi",
            "--query-gpu=" + _NVIDIA_SMI_QUERY,
            "--format=csv,noheader,nounits",
            "-i", str(self.gpu_index),
        ]
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                out = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=5,
                ).stdout.strip()
                if out:
                    parts = [p.strip() for p in out.split(",")]
                    if len(parts) >= 5:
                        self.samples.append({
                            "t": time.perf_counter(),
                            "mem_used_mib": float(parts[0]),
                            "mem_total_mib": float(parts[1]),
                            "gpu_util_pct": float(parts[2]),
                            "mem_util_pct": float(parts[3]),
                            "power_w": float(parts[4]) if parts[4] != "[N/A]" else 0.0,
                        })
            except (ValueError, subprocess.SubprocessError):
                pass
            # Sleep for the remainder of the interval (subtract poll time).
            elapsed = time.perf_counter() - t0
            self._stop.wait(max(0.0, self.interval - elapsed))

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def summary(self) -> dict[str, float]:
        """Return aggregate GPU metrics over the sampled region."""
        if not self.samples:
            return {
                "gpu_mem_mean_mib": 0.0, "gpu_mem_max_mib": 0.0,
                "gpu_util_mean_pct": 0.0, "gpu_util_max_pct": 0.0,
                "gpu_mem_util_mean_pct": 0.0, "gpu_mem_util_max_pct": 0.0,
                "gpu_power_mean_w": 0.0, "gpu_samples": 0,
            }
        mem = [s["mem_used_mib"] for s in self.samples]
        util = [s["gpu_util_pct"] for s in self.samples]
        mem_util = [s["mem_util_pct"] for s in self.samples]
        pwr = [s["power_w"] for s in self.samples]
        return {
            "gpu_mem_mean_mib": statistics.fmean(mem),
            "gpu_mem_max_mib": float(max(mem)),
            "gpu_util_mean_pct": statistics.fmean(util),
            "gpu_util_max_pct": float(max(util)),
            "gpu_mem_util_mean_pct": statistics.fmean(mem_util),
            "gpu_mem_util_max_pct": float(max(mem_util)),
            "gpu_power_mean_w": statistics.fmean(pwr),
            "gpu_samples": len(self.samples),
        }


# ---------------------------------------------------------------------------
# NVTX range helpers (for Nsight Systems timeline labeling)
# ---------------------------------------------------------------------------

def nvtx_push(name: str) -> None:
    """Push an NVTX range so it shows up in ``nsys`` traces."""
    try:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(name)
    except Exception:
        pass


def nvtx_pop() -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()
    except Exception:
        pass


def _is_oom_error(exc: BaseException) -> bool:
    """Heuristic: did this error come from running out of GPU memory?"""
    msg = str(exc).lower()
    needles = (
        "out of memory", "out-of-memory", "oom",
        "memory allocation", "alloc failed", "no enough memory",
        "device allocation", "not enough memory",
    )
    return any(n in msg for n in needles)


def _is_max_n_error(exc: BaseException) -> bool:
    """Heuristic: did this error indicate N is too large for the GPU/engine?

    Catches both CUDA OOM and Genesis's internal ``Jacobian shape ... is too
    large`` exception (which fires before OOM on GPUs with large VRAM).
    """
    if _is_oom_error(exc):
        return True
    msg = str(exc).lower()
    return "too large" in msg or "shape" in msg and "n_envs" in msg


def benchmark_env_count(
    n_envs: int,
    *,
    object_name: str,
    settle_steps: int,
    trials: int,
    seed: int,
    gpu_sample_interval: float = 0.2,
) -> list[dict]:
    """Run the IK pick pipeline ``trials`` times for ``n_envs`` envs.

    Returns a list of per-trial result dicts.  Each dict includes aggregate
    GPU metrics (memory usage + compute-unit utilization) sampled via
    ``nvidia-smi`` during the measured region of that trial.

    Raises ``RuntimeError`` (with a max-N-tagged message) if the scene cannot
    be built or stepped at this ``n_envs`` because the GPU ran out of memory
    or the engine's internal Jacobian size limit was exceeded.
    The caller is expected to catch this to stop the sweep at the max N.
    """
    from ycb_pick_ik_parallel import HSRPickEnv

    results = []
    for trial in range(trials):
        nvtx_push(f"N={n_envs} trial={trial}")

        # Rebuild scene for each trial to get clean state.  This is where the
        # bulk of GPU memory is allocated, so OOM / size-limit errors usually
        # surface here.
        try:
            env = HSRPickEnv(
                n_envs=n_envs,
                object_name=object_name,
                show_viewer=False,
                seed=seed + trial,
                disable_visualizer=True,
            )
        except BaseException as exc:
            if _is_max_n_error(exc):
                raise RuntimeError(
                    f"Max-N exceeded at N={n_envs} during scene build: {exc}"
                ) from exc
            raise

        # Warmup: a few sim steps to prime kernels / JIT.  Kernel JIT and the
        # first few allocations can also OOM here.
        try:
            for _ in range(5):
                env.scene.step()
            sync_if_cuda()
        except BaseException as exc:
            if _is_max_n_error(exc):
                raise RuntimeError(
                    f"Max-N exceeded at N={n_envs} during warmup: {exc}"
                ) from exc
            raise

        # Start GPU sampling right before the timed region so the metrics
        # reflect the actual pick-pipeline workload for this N condition.
        sampler = GpuSampler(interval=gpu_sample_interval)
        sampler.start()

        try:
            t0 = time.perf_counter()
            summary = env.run_pick_pipeline(settle_steps=settle_steps)
            sync_if_cuda()
            wall = time.perf_counter() - t0
        except BaseException as exc:
            sampler.stop()
            if _is_max_n_error(exc):
                raise RuntimeError(
                    f"Max-N exceeded at N={n_envs} during pipeline: {exc}"
                ) from exc
            raise

        sampler.stop()
        gpu = sampler.summary()

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
            "gpu_mem_mean_mib": gpu["gpu_mem_mean_mib"],
            "gpu_mem_max_mib": gpu["gpu_mem_max_mib"],
            "gpu_util_mean_pct": gpu["gpu_util_mean_pct"],
            "gpu_util_max_pct": gpu["gpu_util_max_pct"],
            "gpu_mem_util_mean_pct": gpu["gpu_mem_util_mean_pct"],
            "gpu_mem_util_max_pct": gpu["gpu_mem_util_max_pct"],
            "gpu_power_mean_w": gpu["gpu_power_mean_w"],
            "gpu_samples": gpu["gpu_samples"],
        }
        results.append(result)
        print(
            f"  [envs={n_envs:>4d} trial={trial}] "
            f"wall={wall:.2f}s steps={total_steps} "
            f"steps/s={steps_per_sec:.1f} "
            f"envs·steps/s={env_steps_per_sec:.0f} "
            f"success={summary['success_rate']:.0%} "
            f"gpu_mem={gpu['gpu_mem_mean_mib']:.0f}/{gpu['gpu_mem_max_mib']:.0f}MiB "
            f"sm={gpu['gpu_util_mean_pct']:.0f}/{gpu['gpu_util_max_pct']:.0f}%"
        )
        nvtx_pop()
    return results


def _auto_n_sequence(start: int, factor: float, cap: int):
    """Yield N values for auto-scaling mode: start, start*factor, ... up to cap.

    Each value is rounded to the nearest integer.  The sequence terminates once
    a value exceeds ``cap`` (a safety bound; OOM normally stops the sweep first).
    """
    n = start
    while n <= cap:
        yield int(round(n))
        n = n * factor
        # Guard against factor <= 1.0 stalling at the same value.
        if int(round(n)) <= int(round(n / factor)):
            n = int(round(n / factor)) + 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark HSR IK pick pipeline across env counts."
    )
    parser.add_argument(
        "--envs", type=int, nargs="+", default=None,
        help="Explicit list of env counts to sweep. If omitted, the sweep "
             "auto-scales N (doubling from --auto-start) until the GPU runs "
             "out of memory -- i.e. the max N is capped only by OOM.",
    )
    parser.add_argument(
        "--auto-start", type=int, default=1,
        help="Starting N for auto-scaling mode (default 1).",
    )
    parser.add_argument(
        "--auto-factor", type=float, default=2.0,
        help="Multiply N by this each step in auto-scaling mode (default 2.0, "
             "i.e. doubling). Use 1.5 for a finer sweep.",
    )
    parser.add_argument(
        "--auto-cap", type=int, default=100_000_000,
        help="Safety hard cap on N in auto-scaling mode (default 100M). The "
             "sweep normally stops at OOM long before this; it only prevents "
             "an infinite loop if the GPU never OOMs.",
    )
    parser.add_argument("--object", type=str, default="ycb_061_foam_brick")
    parser.add_argument("--settle-steps", type=int, default=50)
    parser.add_argument("--trials", type=int, default=2,
                        help="Number of repeated trials per env count")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--csv", type=str, default=None,
                        help="Save results to a CSV file")
    parser.add_argument(
        "--gpu-sample-interval", type=float, default=0.2,
        help="Seconds between nvidia-smi GPU metric samples (default 0.2)",
    )
    parser.add_argument(
        "--nsys-out", type=str, default=None,
        help="If set, re-exec the whole sweep under `nsys profile`, writing a "
             "Nsight Systems trace to this path (.nsys-rep). NVTX ranges label "
             "each N condition and trial in the timeline.",
    )
    parser.add_argument(
        "--no-stop-on-oom", action="store_true",
        help="Continue the sweep past an OOM condition instead of stopping at "
             "the max supported N. By default the sweep stops at the first N "
             "that exhausts GPU memory and reports it.",
    )
    args = parser.parse_args()

    # --- Nsight Systems re-exec -------------------------------------------------
    # When --nsys-out is requested and we are not already inside an nsys
    # profiling session, relaunch this script under `nsys profile`.  The child
    # process inherits all other args (minus --nsys-out) and runs normally,
    # emitting NVTX ranges that segment the trace per N / per trial.
    nsys_child_env = "HSR_BENCH_NSYS_CHILD"
    if args.nsys_out and nsys_child_env not in os.environ:
        nsys_cmd = [
            "nsys", "profile",
            "-o", args.nsys_out,
            "-f", "true",          # overwrite existing report
            "--trace=cuda,nvtx,osrt,cudnn",
            "-s", "none",          # skip the slow sampling post-processing
            "--",
            sys.executable, __file__,
        ]
        # Forward every CLI arg except --nsys-out (and its value).
        forwarded = []
        skip_next = False
        for tok in sys.argv[1:]:
            if skip_next:
                skip_next = False
                continue
            if tok == "--nsys-out":
                skip_next = True
                continue
            if tok.startswith("--nsys-out="):
                continue
            forwarded.append(tok)
        nsys_cmd += forwarded
        print(f"[benchmark] launching under nsys: {' '.join(nsys_cmd)}")
        env = dict(os.environ)
        env[nsys_child_env] = "1"
        rc = subprocess.call(nsys_cmd, env=env)
        sys.exit(rc)

    backend = gs.cpu if args.backend == "cpu" else gs.gpu
    gs.init(backend=backend)

    # Resolve the N sequence: explicit list, or auto-scaling (doubling until
    # OOM) when --envs is not given.  In auto mode the max N is capped only by
    # GPU memory exhaustion, not by any hardcoded ceiling.
    auto_mode = args.envs is None
    if auto_mode:
        n_iter = _auto_n_sequence(
            args.auto_start, args.auto_factor, args.auto_cap,
        )
        print(f"[benchmark] object={args.object} backend={args.backend}")
        print(f"[benchmark] AUTO mode: start={args.auto_start} "
              f"factor={args.auto_factor} cap={args.auto_cap} "
              f"(max N capped only by OOM)")
        print(f"[benchmark] trials={args.trials} settle_steps={args.settle_steps}")
    else:
        n_iter = iter(args.envs)
        print(f"[benchmark] object={args.object} backend={args.backend}")
        print(f"[benchmark] envs={args.envs} trials={args.trials} "
              f"settle_steps={args.settle_steps}")
    print(f"[benchmark] gpu_sample_interval={args.gpu_sample_interval}s")
    print()

    all_results = []
    max_supported_n: int | None = None
    oom_error: str | None = None
    swept_n: list[int] = []   # N values actually attempted (for summary table)
    for n_envs in n_iter:
        print(f"--- n_envs={n_envs} ---")
        swept_n.append(n_envs)
        nvtx_push(f"N={n_envs}")
        try:
            results = benchmark_env_count(
                n_envs,
                object_name=args.object,
                settle_steps=args.settle_steps,
                trials=args.trials,
                seed=args.seed,
                gpu_sample_interval=args.gpu_sample_interval,
            )
        except RuntimeError as exc:
            # Max-N exceeded (OOM or engine size limit) raised by
            # benchmark_env_count with a "Max-N exceeded at N=..." message.
            nvtx_pop()
            if not _is_max_n_error(exc) and not str(exc).startswith("Max-N") \
                    and not str(exc).startswith("OOM"):
                raise
            oom_error = str(exc)
            print(f"  [ERROR] {oom_error}")
            if args.no_stop_on_oom:
                print(f"  (--no-stop-on-oom: continuing sweep)")
                continue
            # The largest N that succeeded is the max supported N.
            successful = [r["n_envs"] for r in all_results]
            max_supported_n = max(successful) if successful else 0
            print(
                f"  [max-N] GPU/engine limit reached at N={n_envs}. "
                f"Max supported N on this GPU: {max_supported_n}."
            )
            break
        else:
            nvtx_pop()
            all_results.extend(results)
            print()

    if oom_error is not None:
        print()
        print("=" * 80)
        print(f"[max-N] {oom_error}")
        if max_supported_n is not None:
            print(f"[max-N] Maximum supported N on this GPU: {max_supported_n}")
        print("=" * 80)
        print()

    # Summary table (over N values that produced results).
    print("=" * 104)
    print(f"{'envs':>6s} {'trials':>7s} {'wall_s':>8s} {'steps/s':>10s} "
          f"{'envs·steps/s':>14s} {'success':>8s} "
          f"{'mem(MiB)':>16s} {'SM(%)':>14s}")
    print(f"{'':>6s} {'':>7s} {'':>8s} {'':>10s} {'':>14s} {'':>8s} "
          f"{'mean/max':>16s} {'mean/max':>14s}")
    print("-" * 104)
    for n_envs in swept_n:
        trials = [r for r in all_results if r["n_envs"] == n_envs]
        if not trials:
            continue
        avg_wall = sum(r["wall_s"] for r in trials) / len(trials)
        avg_sps = sum(r["steps_per_sec"] for r in trials) / len(trials)
        avg_esps = sum(r["env_steps_per_sec"] for r in trials) / len(trials)
        avg_succ = sum(r["success_rate"] for r in trials) / len(trials)
        avg_mem = sum(r["gpu_mem_mean_mib"] for r in trials) / len(trials)
        max_mem = max(r["gpu_mem_max_mib"] for r in trials)
        avg_sm = sum(r["gpu_util_mean_pct"] for r in trials) / len(trials)
        max_sm = max(r["gpu_util_max_pct"] for r in trials)
        print(
            f"{n_envs:>6d} {len(trials):>7d} {avg_wall:>8.2f} "
            f"{avg_sps:>10.1f} {avg_esps:>14.0f} {avg_succ:>7.0%} "
            f"{avg_mem:>7.0f}/{max_mem:<7.0f} "
            f"{avg_sm:>5.0f}/{max_sm:<5.0f}"
        )
    print("=" * 104)

    if args.csv:
        path = Path(args.csv)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        print(f"[benchmark] results saved to {path}")


if __name__ == "__main__":
    main()
