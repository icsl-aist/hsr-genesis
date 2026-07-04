"""Run the YCB pick benchmark on RunPod across different GPU types.

Creates a RunPod pod for each GPU type, clones the repo, installs
dependencies, runs the env-count sweep benchmark, collects the CSV results,
and tears down the pod.  Aggregates all results into a combined report.

Prerequisites
-------------
- ``runpodctl`` installed and configured (``runpodctl doctor``).
- SSH key registered with RunPod (``runpodctl ssh list-keys``).
- Sufficient RunPod balance for the selected GPUs.

Usage
-----
    # benchmark on RTX 4090 and A100 with a small sweep
    PYTHONPATH=src .venv/bin/python examples/speed_benchmark/runpod_gpu_sweep.py \
        --gpus "NVIDIA GeForce RTX 4090" "NVIDIA A100-SXM4-80GB" \
        --envs 1 8 32 128 --trials 2

    # dry-run (print what would be done, don't create pods)
    PYTHONPATH=src .venv/bin/python examples/speed_benchmark/runpod_gpu_sweep.py --dry-run

    # keep pods alive after benchmark (for debugging)
    PYTHONPATH=src .venv/bin/python examples/speed_benchmark/runpod_gpu_sweep.py --keep-pods
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# runpodctl helpers
# ---------------------------------------------------------------------------

def run_cmd(cmd: list[str], *, timeout: int = 120, check: bool = True) -> str:
    """Run a command and return stdout. Raises on failure if check=True."""
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
    )
    if check and result.returncode != 0:
        msg = f"Command failed: {' '.join(cmd)}\nstdout: {result.stdout}\nstderr: {result.stderr}"
        raise RuntimeError(msg)
    return result.stdout.strip()


def pod_create(gpu_id: str, *, name: str, template_id: str, container_disk_gb: int = 40) -> dict:
    """Create a RunPod pod and return its info dict."""
    out = run_cmd([
        "runpodctl", "pod", "create",
        "--template-id", template_id,
        "--gpu-id", gpu_id,
        "--name", name,
        "--container-disk-in-gb", str(container_disk_gb),
        "--ports", "22/tcp",
    ])
    data = json.loads(out)
    return data


def pod_get(pod_id: str) -> dict:
    """Get pod details."""
    out = run_cmd(["runpodctl", "pod", "get", pod_id])
    return json.loads(out)


def pod_delete(pod_id: str) -> None:
    """Delete a pod."""
    run_cmd(["runpodctl", "pod", "delete", pod_id], check=False)


def wait_for_pod_ready(pod_id: str, *, timeout: int = 300) -> dict:
    """Wait until pod is RUNNING and SSH info is available.

    RunPod's ``pod get`` output doesn't have a ``status`` field; instead we
    check for the presence of ``ssh`` info (IP + port) which indicates the
    pod is up and accessible.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = pod_get(pod_id)
        ssh_info = info.get("ssh") if isinstance(info, dict) else None
        if ssh_info and ssh_info.get("ip") and ssh_info.get("port"):
            return info
        time.sleep(5)
    raise TimeoutError(f"Pod {pod_id} SSH info not available within {timeout}s")


def get_ssh_info(pod_id: str) -> tuple[str, int, str]:
    """Return (host, port, key_path) for SSH access to the pod.

    Uses ``pod get`` which includes the SSH connection details and the
    runpodctl-managed key path.
    """
    info = pod_get(pod_id)
    ssh_info = info.get("ssh", {})
    host = ssh_info.get("ip", "localhost")
    port = int(ssh_info.get("port", 22))
    # The runpodctl-managed key is the one registered with the pod.
    key = ssh_info.get("ssh_key", {}).get("path")
    if not key:
        key = str(Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key")
    return host, port, key


# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------

def ssh_run(host: str, port: int, key: str, cmd: str, *, timeout: int = 600) -> tuple[int, str, str]:
    """Run a command on the remote pod via SSH. Returns (exit_code, stdout, stderr)."""
    ssh_cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=30", "-o", "ServerAliveInterval=60",
        "-p", str(port), "-i", key,
        f"root@{host}",
        cmd,
    ]
    result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def scp_from(host: str, port: int, key: str, remote_path: str, local_path: str) -> None:
    """Copy a file from the remote pod."""
    scp_cmd = [
        "scp", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "-P", str(port), "-i", key,
        f"root@{host}:{remote_path}", str(local_path),
    ]
    result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"scp failed: {result.stderr}")


def scp_to(host: str, port: int, key: str, local_path: str, remote_path: str) -> None:
    """Copy a file to the remote pod."""
    scp_cmd = [
        "scp", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "-P", str(port), "-i", key,
        str(local_path), f"root@{host}:{remote_path}",
    ]
    result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"scp to failed: {result.stderr}")


# ---------------------------------------------------------------------------
# Remote benchmark
# ---------------------------------------------------------------------------

REMOTE_REPO_DIR = "/workspace/hsr_genesis"
REMOTE_CSV = "/workspace/bench_results.csv"

SETUP_SCRIPT = f"""set -ex
cd /workspace
if [ ! -d hsr_genesis ]; then
  git clone --recursive https://github.com/icsl-aist/hsr-genesis.git hsr_genesis
else
  cd hsr_genesis && git pull --recurse-submodules
fi
cd {REMOTE_REPO_DIR}
python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .
echo "Setup complete."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
"""

BENCHMARK_SCRIPT_TEMPLATE = f"""set -ex
cd {REMOTE_REPO_DIR}
PYTHONPATH=src .venv/bin/python examples/speed_benchmark/ycb_pick_sweep.py \
    --envs {{envs}} --trials {{trials}} --settle-steps {{settle_steps}} \
    --csv {REMOTE_CSV}
echo "BENCHMARK_DONE"
cat {REMOTE_CSV}
"""


def run_gpu_benchmark(
    gpu_id: str,
    *,
    envs: list[int],
    trials: int,
    settle_steps: int,
    template_id: str,
    keep_pod: bool,
    dry_run: bool,
) -> list[dict]:
    """Create a pod with the given GPU, run the benchmark, collect results."""
    gpu_short = gpu_id.replace("NVIDIA ", "").replace(" ", "_").replace("/", "_")
    pod_name = f"hsr-bench-{gpu_short}"

    print(f"\n{'='*70}")
    print(f"[GPU] {gpu_id}")
    print(f"{'='*70}")

    if dry_run:
        print(f"  [dry-run] would create pod: {pod_name}")
        print(f"  [dry-run] envs={envs} trials={trials}")
        return []

    # --- Create pod ---
    print(f"  Creating pod '{pod_name}' with {gpu_id}...")
    pod_info = pod_create(gpu_id, name=pod_name, template_id=template_id)
    pod_id = None
    if isinstance(pod_info, dict):
        pod_id = pod_info.get("id")
    elif isinstance(pod_info, list) and pod_info:
        pod_id = pod_info[0].get("id")
    if not pod_id:
        raise RuntimeError(f"Could not extract pod ID from: {pod_info}")
    print(f"  Pod ID: {pod_id}")

    try:
        # --- Wait for pod to be ready ---
        print(f"  Waiting for pod to be RUNNING...")
        wait_for_pod_ready(pod_id, timeout=300)
        print(f"  Pod is RUNNING.")

        # --- Get SSH info ---
        host, port, key = get_ssh_info(pod_id)
        print(f"  SSH: root@{host}:{port}")

        # --- Wait for SSH to be accessible ---
        print(f"  Waiting for SSH...")
        for attempt in range(12):
            rc, out, err = ssh_run(host, port, key, "echo READY", timeout=30)
            if rc == 0 and "READY" in out:
                print(f"  SSH connected.")
                break
            time.sleep(5)
        else:
            raise RuntimeError("SSH not accessible after 60s")

        # --- Setup: clone repo, install deps ---
        print(f"  Running setup (clone + install)...")
        rc, out, err = ssh_run(host, port, key, SETUP_SCRIPT, timeout=2400)
        print(f"  Remote output (tail):")
        for line in out.split("\n")[-15:]:
            print(f"    {line}")
        if rc != 0:
            print(f"  STDERR (tail):")
            for line in err.split("\n")[-10:]:
                print(f"    {line}")
            raise RuntimeError(f"Setup failed (exit {rc})")

        # --- Copy local benchmark + RL example files to the pod ---
        # These files may not be on GitHub yet, so we scp them directly.
        local_root = Path(__file__).resolve().parents[2]
        files_to_copy = [
            (local_root / "examples" / "speed_benchmark" / "ycb_pick_sweep.py",
             f"{REMOTE_REPO_DIR}/examples/speed_benchmark/ycb_pick_sweep.py"),
            (local_root / "examples" / "rl" / "ycb_pick_ik_parallel.py",
             f"{REMOTE_REPO_DIR}/examples/rl/ycb_pick_ik_parallel.py"),
        ]
        # Ensure remote directories exist before scp.
        remote_dirs = sorted(set(
            str(Path(rp).parent) for _, rp in files_to_copy
        ))
        ssh_run(host, port, key, f"mkdir -p {' '.join(remote_dirs)}", timeout=30)
        print(f"  Copying benchmark files to pod...")
        for local_path, remote_path in files_to_copy:
            scp_to(host, port, key, str(local_path), remote_path)
        print(f"  Files copied.")

        # --- Run benchmark ---
        print(f"  Running benchmark...")
        envs_str = " ".join(str(e) for e in envs)
        bench_cmd = BENCHMARK_SCRIPT_TEMPLATE.format(
            envs=envs_str, trials=trials, settle_steps=settle_steps,
        )
        rc, out, err = ssh_run(host, port, key, bench_cmd, timeout=1800)
        print(f"  Remote output (tail):")
        for line in out.split("\n")[-30:]:
            print(f"    {line}")
        if rc != 0:
            print(f"  STDERR (tail):")
            for line in err.split("\n")[-10:]:
                print(f"    {line}")
            raise RuntimeError(f"Benchmark failed (exit {rc})")

        # --- Collect results ---
        local_csv = Path(f"/tmp/bench_{gpu_short}.csv")
        scp_from(host, port, key, REMOTE_CSV, str(local_csv))
        print(f"  Results saved to {local_csv}")

        results = []
        with local_csv.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["gpu_id"] = gpu_id
                row["gpu_short"] = gpu_short
                row["pod_id"] = pod_id
                results.append(row)
        return results

    finally:
        if not keep_pod:
            print(f"  Deleting pod {pod_id}...")
            pod_delete(pod_id)
            print(f"  Pod deleted.")
        else:
            print(f"  Pod {pod_id} kept alive (--keep-pods).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run YCB pick benchmark on RunPod across GPU types."
    )
    parser.add_argument(
        "--gpus", type=str, nargs="+",
        default=["NVIDIA GeForce RTX 4090", "NVIDIA A100-SXM4-80GB"],
        help="RunPod GPU IDs to benchmark (see: runpodctl gpu list)",
    )
    parser.add_argument(
        "--envs", type=int, nargs="+",
        default=[1, 8, 32, 128],
        help="Env counts to sweep",
    )
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--settle-steps", type=int, default=30)
    parser.add_argument(
        "--template-id", type=str, default="runpod-torch-v280",
        help="RunPod template ID (PyTorch image). Must have CUDA >= 12.4",
    )
    parser.add_argument("--keep-pods", action="store_true",
                        help="Keep pods alive after benchmark (for debugging)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without creating pods")
    parser.add_argument("--csv", type=str, default="runpod_gpu_benchmark.csv",
                        help="Output CSV file for aggregated results")
    args = parser.parse_args()

    print(f"[runpod-bench] GPUs={args.gpus}")
    print(f"[runpod-bench] envs={args.envs} trials={args.trials} "
          f"settle_steps={args.settle_steps}")
    print(f"[runpod-bench] template={args.template_id}")

    all_results = []
    for gpu_id in args.gpus:
        try:
            results = run_gpu_benchmark(
                gpu_id,
                envs=args.envs,
                trials=args.trials,
                settle_steps=args.settle_steps,
                template_id=args.template_id,
                keep_pod=args.keep_pods,
                dry_run=args.dry_run,
            )
            all_results.extend(results)
        except Exception as exc:
            print(f"  [ERROR] GPU {gpu_id} failed: {exc}")
            continue

    if not all_results:
        print("\n[runpod-bench] No results collected.")
        return

    # --- Save aggregated CSV ---
    csv_path = Path(args.csv)
    fieldnames = list(all_results[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n[runpod-bench] Aggregated results saved to {csv_path}")

    # --- Summary table ---
    print(f"\n{'='*90}")
    print(f"{'GPU':30s} {'envs':>6s} {'wall_s':>8s} {'steps/s':>10s} "
          f"{'envs·steps/s':>14s} {'success':>8s}")
    print(f"{'-'*90}")
    for gpu_id in args.gpus:
        gpu_results = [r for r in all_results if r["gpu_id"] == gpu_id]
        if not gpu_results:
            continue
        gpu_short = gpu_results[0]["gpu_short"]
        for n_envs in args.envs:
            trials = [r for r in gpu_results if int(r["n_envs"]) == n_envs]
            if not trials:
                continue
            avg_wall = sum(float(r["wall_s"]) for r in trials) / len(trials)
            avg_sps = sum(float(r["steps_per_sec"]) for r in trials) / len(trials)
            avg_esps = sum(float(r["env_steps_per_sec"]) for r in trials) / len(trials)
            avg_succ = sum(float(r["success_rate"]) for r in trials) / len(trials)
            print(
                f"{gpu_short:30s} {n_envs:>6d} {avg_wall:>8.2f} "
                f"{avg_sps:>10.1f} {avg_esps:>14.0f} {avg_succ:>7.0%}"
            )
        print()
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
