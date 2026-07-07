"""Builds examples/tutorials/8_parallel_simulation_colab.ipynb.

Re-run after editing to regenerate the notebook JSON.
"""
from __future__ import annotations

import nbformat as nbf
from pathlib import Path

NB_PATH = Path(__file__).parent / "8_parallel_simulation_colab.ipynb"


def _md(src: str) -> dict:
    return nbf.v4.new_markdown_cell(src)


def _code(src: str) -> dict:
    return nbf.v4.new_code_cell(src)


def build_notebook() -> nbf.notebooknode.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata.language_info = {"name": "python"}
    nb.cells.extend(_collect_cells())
    return nb


def _collect_cells() -> list:
    cells: list = []
    # ------------------------------------------------------------------
    # Section 0 — Badge + Title + Objectives + Setup (Tasks 2, 3)
    # ------------------------------------------------------------------
    cells.append(_md(_BADGE_TITLE))
    cells.append(_md(_OBJECTIVES))
    cells.append(_md(_SETUP_HEADING))
    cells.append(_code(_SETUP_CODE))
    cells.append(_code(_GPU_INIT))

    # ------------------------------------------------------------------
    # Section 1 — Recap single-env idiom (Task 4)
    # ------------------------------------------------------------------
    cells.append(_md(_RECAP_HEADING))
    cells.append(_code(_RECAP_SINGLE_ENV))

    # ------------------------------------------------------------------
    # Section 2 — Why batched (markdown) + correspondence table (Task 5)
    # ------------------------------------------------------------------
    cells.append(_md(_WHY_BATCHED))
    cells.append(_md(_DIAGRAM))
    cells.append(_md(_CORRESPONDENCE_TABLE))

    # ------------------------------------------------------------------
    # Section 3 — Build parallel scene + envs_idx (Task 6)
    # ------------------------------------------------------------------
    cells.append(_md(_BUILD_SCENE_HEADING))
    cells.append(_code(_BUILD_SCENE_CODE))
    cells.append(_md(_ENTITY_GRAPH_NOTE))
    cells.append(_code(_ENVS_IDX_CODE))

    # ------------------------------------------------------------------
    # Section 4 — Batched IK reach demo (Task 7)
    # ------------------------------------------------------------------
    cells.append(_md(_BATCHED_IK_HEADING))
    cells.append(_code(_DEFINE_TARGETS_CODE))
    cells.append(_code(_BATCHED_IK_SOLVE))
    cells.append(_code(_BATCHED_STEP_RENDER))

    # ------------------------------------------------------------------
    # Section 5 — _batched controller layer (Task 8)
    # ------------------------------------------------------------------
    cells.append(_md(_BATCHED_CONTROLLERS_HEADING))
    cells.append(_md(_BATCHED_CONTROLLERS_NOTE))
    cells.append(_code(_WHOLE_BODY_TRAJECTORY_DEMO))
    cells.append(_code(_GRIPPER_BATCHED_DEMO))

    # ------------------------------------------------------------------
    # Section 6 — Benchmark (Task 9)
    # ------------------------------------------------------------------
    cells.append(_md(_BENCHMARK_HEADING))
    cells.append(_code(_BENCHMARK_CODE))
    cells.append(_code(_BATCHED_RESULT_FRAME))

    # ------------------------------------------------------------------
    # Section 7 — Forward-pointer + recap (Task 10)
    # ------------------------------------------------------------------
    cells.append(_md(_FORWARD_POINTER))
    cells.append(_md(_RECAP_BULLETS))
    return cells


# ---------------------------------------------------------------------------
# Cell contents are defined in subsequent tasks. Skeleton placeholders for
# the first task; later tasks replace these with real content.
# ---------------------------------------------------------------------------

_BADGE_TITLE = (
    '<a href="https://colab.research.google.com/github/icsl-aist/hsr-genesis/blob/main/examples/tutorials/8_parallel_simulation_colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>\n\n'
    "# 8. 並列シミュレーション / Parallel Simulation on GPU\n"
)

_OBJECTIVES = """## Overview / 概要

This notebook bridges tutorials 1–7 (single-env `tutorial_utils.step()` control) with notebooks 9–10 (CMA-ES and PPO, which use GPU batched parallel sim). After completing it you will be able to:

- Build N parallel environments with `scene.build(n_envs=N)` — one scene, N parallel copies of every entity
- Use `envs_idx` (a `torch.Tensor` of env indices on `gs.device`) to drive all envs in a single Python call
- Identify the one-to-one correspondence between the single-env API you already know (`move_arm_*`, `move_base_*`, `grasp_object`) and the batched `*_batched` controller methods
- Measure the speedup of one batched `scene.step()` call vs N separate single-env calls

このノートブックは、チュートリアル 1–7(単一環境の `tutorial_utils.step()` 制御)と、チュートリアル 9–10(CMA-ES・PPO、GPU 並列シミュレーション使用)の橋渡しをします。完了すると以下ができるようになります:

- `scene.build(n_envs=N)` で N 個の並列環境を構築(1 シーン、N 個の並列コピー)
- `envs_idx`(`gs.device` 上の env インデックスの `torch.Tensor`)で全環境を 1 回の Python 呼び出しで制御
- 既知の単一環境 API(`move_arm_*`, `move_base_*`, `grasp_object`)とバッチ化 `_batched` メソッドの対応関係
- バッチ化 `scene.step()` 1 回が N 回の単一環境呼び出しに比べてどれくらい速いかを計測
"""

_SETUP_HEADING = """## 1. Setup / セットアップ

The Colab bootstrap below mirrors tutorials 1–7. The only change from prior notebooks: starting in Section 4 we will **not** call `tutorial_utils.step(n)` — we call `scene.step()` directly so we can drive multiple envs in one Python call.
"""
_SETUP_CODE = """# Open this notebook on Colab for full GPU support.
# Run this cell first.
import sys
if "google.colab" in sys.modules:
    import urllib.request
    exec(urllib.request.urlopen(
        "https://raw.githubusercontent.com/icsl-aist/hsr-genesis/main/examples/tutorials/colab_setup.py"
    ).read(), globals())
    setup_colab()

import numpy as np
import torch
import genesis as gs
from hsr_genesis.hsr_rigid_entity import HSRBURDF
from hsr_genesis import tutorial_utils  # for the recap cell — what tutorials 1–7 taught you
"""

_GPU_INIT = """# Initialize Genesis on GPU if available; fall back to CPU.
gs.init(backend=gs.gpu)
device = gs.device
print(f"Genesis device: {device}")

# Quick sanity: ensure torch sees the same device.
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
"""
_RECAP_HEADING = """## 2. Recap — the single-env API you already know

Tutorials 1–7 taught you to drive one robot through `tutorial_utils`:

| What you called | What it does internally |
| --- | --- |
| `init_sim()` | Builds a `gs.Scene` with **one** HSR entity, calls `scene.build()` |
| `step(n=1)` or `run(seconds)` | Calls `scene.step()` n times in a Python loop |
| `move_arm_neutral()`, `move_arm_joints(j)`, `move_wholebody_ik(...)` | Sets one trajectory on the HSR's single controller, steps in a loop |
| `move_base_vel(v)`, `move_base_goal(p)` | Same — base controller of the single HSR |
| `grasp_object(o)`, `move_hand(p)` | Single-env gripper controller |

Every call above has an implicit `envs_idx=0`. The cell below reproduces a tiny end-to-end example: build the scene, reach one target with IK, step.
"""

_RECAP_SINGLE_ENV = """# --- Recap: 1 HSR, 1 target, 1 scene.step() per Python iteration ---
tutorial_utils.init_sim(n_envs=1)   # single env (default)

# Re-create the HSR handle the way tutorial_utils does internally (so we can
# call set_qpos / inverse_kinematics directly in later sections).
hsr_single = tutorial_utils._state.hsr   # internal attribute; safe to expose in tutorial

ik_link = hsr_single.get_link("hand_palm_link")
target_pos_single = torch.tensor([0.6, 0.0, 0.9], device=device)
target_quat_single = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)  # identity wxyz

# Single-env IK call — no envs_idx argument; returns (dof,) qpos tensor.
qpos_single = hsr_single.inverse_kinematics(
    link=ik_link,
    pos=target_pos_single,
    quat=target_quat_single,
)
hsr_single.set_qpos(qpos_single)

# Step the sim long enough to settle.
for _ in range(60):
    tutorial_utils.step(render=False)
print(f"Single-env arm reached target: qpos shape = {qpos_single.shape}")
"""
_WHY_BATCHED = """## 3. Why not just loop N times?

A natural thought: "If 8 single-env `scene.step()` calls work, can I just call them in a Python loop?" Three reasons the GPU batched API wins:

**1. Per-call kernel-launch overhead.** Every `scene.step()` enqueues a batch of CUDA kernels and waits for them to finish before returning control to Python. The CPU↔GPU synchronization dominates wall time for small scenes. With N=8 and ~5 ms of physics per step, 70%+ of the step time is overhead. One batched `scene.step()` pays the kernel-launch + sync cost once.

**2. Per-env tensors live on the GPU.** The single-env style pulls qpos/user state back to Python every step (a `.cpu()` round trip). In RL training you read `qpos` thousands of times per second. Batched tensors stay on `gs.device`; the only host transfers happen at the boundary of a feature (e.g. logging).

**3. One physics kernel beats N.** Genesis' batched rigid-body solver sees all N envs as a single contiguous block and works them in one parallel kernel rather than N serial ones. This is the actual hardware utilization win — the GPU sees more parallelism to fill its SMs.
"""

_DIAGRAM = """```
Single-env, N=8 (Python loop):              Batched, N=8 (one call):

  for i in range(8):                         scene.build(n_envs=8)
      scene.step()                           envs_idx = torch.arange(8)
                                              hsr.set_qpos(q, envs_idx=envs_idx)
  ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐     scene.step()
  │k1│ │k1│ │k1│ │k1│ │k1│ │k1│ │k1│ │k1│   ┌──────────────────────────────┐
  └──┘ └──┘ └──┘ └──┘ └──┘ └──┘ └──┘ └──┘   │     one kernel, 8 envs     │
   ↑    ↑    ↑    ↑    ↑    ↑    ↑    ↑     └──────────────────────────────┘
   8 sync points with Python                  1 sync point with Python
```
"""

_CORRESPONDENCE_TABLE = """## 4. The one-to-one correspondence

Every single-env call you learned in tutorials 1–7 maps to one batched call. The mental switch is a single parameter: instead of `envs_idx=0` (or implied), pass `envs_idx=envs_all` where `envs_all = torch.arange(N, device=gs.device)`.

| Single-env (tutorials 1–7) | Batched (Section 5 onward) |
| --- | --- |
| `init_sim()` builds `scene.build(n_envs=1)` | `scene.build(n_envs=N, env_spacing=(3.0,3.0))` |
| `tutorial_utils.step(n)` loops `scene.step()` n times | `for _ in range(n): scene.step()` (same call, advances **all** envs) |
| `move_arm_neutral()` / `move_arm_joints(j)` | `hsr.set_whole_body_trajectory_batched(...)` + `hsr.step_whole_body_trajectory_batched(dt, envs_idx=envs_all)` |
| `move_base_vel(v)` / `move_base_goal(p)` | `hsr.set_base_trajectory_batched(traj, envs_idx=envs_all)` + `hsr.step_base_trajectory_batched(dt, envs_idx=envs_all)` |
| `grasp_object(o)` / `move_hand(p)` | `hsr.step_gripper_batched(dt, envs_idx=envs_all)` |
| `hsr.get_qpos()` → `(dof,)` tensor | `hsr.get_qpos(envs_idx=envs_all)` → `(N, dof)` tensor |
| `hsr.set_qpos(q)` | `hsr.set_qpos(q, envs_idx=envs_all)` where `q` is `(N, dof)` |
| `hsr.inverse_kinematics(link, pos=p, quat=q)` | same call, but `pos` is `(N,3)`, `quat` is `(N,4)`, `envs_idx=envs_all` |

> **Rule of thumb:** Every batched tensor lives on `gs.device`. Never call `.item()` / `.cpu()` inside the hot loop — move to host once, at the boundary of a feature.
"""
_BUILD_SCENE_HEADING = """## 5. Building N parallel environments

The same `gs.Scene(...)` you used for one env. The only batched-ism happens at `scene.build(n_envs=N)`: Genesis clones the entity graph N times internally. After `build()`, your single Python handle to `hsr` refers to **all N copies** — every method call touches every environment unless you index it down with `envs_idx`.
"""

_BUILD_SCENE_CODE = """N = 8  # try 16, 32, 64 on Colab Pro

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.02),
    rigid_options=gs.options.RigidOptions(use_gjk_collision=True),
    show_viewer=False,
)
_ = scene.add_entity(gs.morphs.Plane())
hsr = scene.add_entity(
    # Use tutorial_utils to find the URDF path (works on Colab and locally)
    HSRBURDF(file=str(tutorial_utils._find_urdf()), robot="hsrb", base_mode="planar")
)
scene.build(n_envs=N, env_spacing=(3.0, 3.0))
print(f"Built scene with n_envs={N}; hsr entity = {type(hsr).__name__}")
"""

_ENTITY_GRAPH_NOTE = """**Key mental model:** There is one `hsr` Python object. Genesis holds N parallel copies of the HSR internally; `hsr.get_qpos(envs_idx=envs_all)` returns a `(N, dof)` tensor that covers all of them in one kernel call. The same applies to every batched method — `set_qpos`, `inverse_kinematics`, `step_whole_body_trajectory_batched`, etc."""

_ENVS_IDX_CODE = """# The single mental switch: a torch tensor of env indices on gs.device.
envs_all = torch.arange(N, device=device, dtype=gs.tc_int)
print(f"envs_all = {envs_all}   (shape={tuple(envs_all.shape)}, device={envs_all.device})")

# Single-env access still works — pass an int (e.g. envs_idx=0) for one env.
# Batched access — pass the full tensor (or any subset of env indices).
"""
_BATCHED_IK_HEADING = """## 6. Batched IK reach — N distinct targets, one solve

In tutorial 3 you called `move_wholebody_ik(...)` for **one** target. Here we extend to N: define a `(N,3)` tensor of distinct target positions and a `(N,4)` tensor of target orientations, then call `inverse_kinematics` exactly the same way — only the inputs are batched tensors and `envs_idx` is a tensor instead of an int.
"""

_DEFINE_TARGETS_CODE = """ik_link = hsr.get_link("hand_palm_link")

# One target per env: arrange them on a circle around each env's origin so that
# visually you'll see N arms each reaching a different point in their own copy.
import math
angles = torch.linspace(0.0, 2 * math.pi, N + 1, device=device)[:N]
offsets = torch.stack([
    0.55 + 0.1 * torch.cos(angles),   # x
    0.1 * torch.sin(angles),          # y
    torch.full((N,), 0.9, device=device),  # z
], dim=-1)  # shape (N, 3)

# Repeat the identity quaternion N times.
target_quat_batched = torch.tensor(
    [[1.0, 0.0, 0.0, 0.0]] * N, device=device, dtype=gs.tc_float
)  # shape (N, 4)

print(f"target_pos shape={tuple(offsets.shape)}, target_quat shape={tuple(target_quat_batched.shape)}")
"""

_BATCHED_IK_SOLVE = """# The SAME call as in the recap cell — envs_idx is now a tensor and inputs are batched.
qpos_batched = hsr.inverse_kinematics(
    link=ik_link,
    pos=offsets,
    quat=target_quat_batched,
    envs_idx=envs_all,
)
print(f"qpos_batched shape={tuple(qpos_batched.shape)}  (expected (N, dof))")

# Set all N envs at once.
hsr.set_qpos(qpos_batched, envs_idx=envs_all)
print("All N arms commanded to IK solutions.")
"""

_BATCHED_STEP_RENDER = """# ONE scene.step() advances every env. 60 steps to let arms settle visually.
for _ in range(60):
    scene.step()

# Verify the per-env hand position matches the requested targets.
hand_pos = hsr.get_link("hand_palm_link").get_pos(envs_idx=envs_all)  # (N, 3)
err = (hand_pos - offsets).norm(dim=-1)
print(f"per-env hand position error (m): mean={err.mean().item():.4f}  max={err.max().item():.4f}")
"""
_BATCHED_CONTROLLERS_HEADING = """## 7. The `_batched` controller layer

Tutorials 3–4 taught you `move_arm_*`, `move_base_*`, and `grasp_object`. In RL scripts we usually don't drive qpos directly — we use the controller layer that mirrors what `move_arm_*` did for you, but in batched form. The correspondence is:

| Tutorial 3–4 call | Batched equivalent |
| --- | --- |
| `move_arm_neutral()`, `move_arm_joints(j)` | `hsr.set_whole_body_trajectory_batched(...)` + `hsr.step_whole_body_trajectory_batched(dt, envs_idx=envs_all)` |
| `move_base_vel(v)`, `move_base_goal(p)` | `hsr.set_base_trajectory_batched(traj, envs_idx=envs_all)` + `hsr.step_base_trajectory_batched(dt, envs_idx=envs_all)` |
| `grasp_object(o)`, `move_hand(p)` | `hsr.step_gripper_batched(dt, envs_idx=envs_all)` |

`JointTrajectory` and `Trajectory` are the same types tutorials 3–4 used internally; you can pass either one trajectory (broadcast to all envs) or a list of N trajectories (per-env).
"""

_BATCHED_CONTROLLERS_NOTE = """**Why show this here?** Notebooks 9 (CMA-ES) and 10 (PPO) keep the controller running across thousands of steps. Knowing this layer exists — and that it's the same call you made in tutorials 3–4 with `envs_idx` added — matters more than memorizing any single signature. The next cell runs one batched whole-body trajectory step."""

_WHOLE_BODY_TRAJECTORY_DEMO = """# Build a tiny batched whole-body trajectory and step it.
# JointTrajectory / Trajectory are tutorial 3 types — same ones used internally by move_arm_*.
from hsr_genesis.hsr_rigid_entity import JointTrajectory

# Build a one-waypoint arm trajectory: every env returns to its "init" pose.
arm_dof = len(hsr._hsr_arm_dofs_idx_local)
traj = JointTrajectory(
    time_from_start=torch.tensor([0.0, 1.0], device=device),
    positions=torch.zeros(2, arm_dof, device=device),  # (2 time steps, arm_dof)
)

hsr.reset_whole_body_trajectory_batched(envs_idx=envs_all)
hsr.set_whole_body_trajectory_batched(
    arm_trajectory=traj,
    base_trajectory=None,           # no base motion in this demo
    envs_idx=envs_all,
)

# Step until the trajectory reports inactive across all envs.
for step_i in range(100):
    state = hsr.step_whole_body_trajectory_batched(dt=0.02, envs_idx=envs_all)
    scene.step()
    if not state["active"].any():
        print(f"whole-body trajectory settled after {step_i + 1} steps")
        break
print(f"final state keys = {list(state.keys())}, active = {state['active'].tolist()}")
"""

_GRIPPER_BATCHED_DEMO = """# Batched gripper — corresponds to grasp_object / move_hand from tutorial 4.
gripper_batch = hsr.get_gripper_batched()   # HSRBGripperControllerBatch (lazy-init, n_envs=N)
# Start a grasp-force goal on all envs (same API as gripper_controller.step_apply_force
# from tutorial 4 — only the input is per-env now).
state = hsr.step_gripper_batched(dt=0.02, envs_idx=envs_all)
print(f"gripper batched state keys = {list(state.keys())}")
scene.step()
"""
_BENCHMARK_HEADING = "# TODO: Task 9"
_BENCHMARK_CODE = "# TODO: Task 9"
_BATCHED_RESULT_FRAME = "# TODO: Task 9"
_FORWARD_POINTER = "# TODO: Task 10"
_RECAP_BULLETS = "# TODO: Task 10"


def main() -> None:
    nb = build_notebook()
    with NB_PATH.open("w", encoding="utf-8") as fp:
        nbf.write(nb, fp)
    print(f"Wrote {NB_PATH} with {len(nb.cells)} cells")


if __name__ == "__main__":
    main()
