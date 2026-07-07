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
_RECAP_HEADING = "# TODO: Task 4"
_RECAP_SINGLE_ENV = "# TODO: Task 4"
_WHY_BATCHED = "# TODO: Task 5"
_DIAGRAM = "# TODO: Task 5"
_CORRESPONDENCE_TABLE = "# TODO: Task 5"
_BUILD_SCENE_HEADING = "# TODO: Task 6"
_BUILD_SCENE_CODE = "# TODO: Task 6"
_ENTITY_GRAPH_NOTE = "# TODO: Task 6"
_ENVS_IDX_CODE = "# TODO: Task 6"
_BATCHED_IK_HEADING = "# TODO: Task 7"
_DEFINE_TARGETS_CODE = "# TODO: Task 7"
_BATCHED_IK_SOLVE = "# TODO: Task 7"
_BATCHED_STEP_RENDER = "# TODO: Task 7"
_BATCHED_CONTROLLERS_HEADING = "# TODO: Task 8"
_BATCHED_CONTROLLERS_NOTE = "# TODO: Task 8"
_WHOLE_BODY_TRAJECTORY_DEMO = "# TODO: Task 8"
_GRIPPER_BATCHED_DEMO = "# TODO: Task 8"
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
