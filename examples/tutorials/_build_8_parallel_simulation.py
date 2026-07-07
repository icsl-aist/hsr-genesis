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

_BADGE_TITLE = "# TODO: Task 2"
_OBJECTIVES = "# TODO: Task 2"
_SETUP_HEADING = "# TODO: Task 2"
_SETUP_CODE = "# TODO: Task 3"
_GPU_INIT = "# TODO: Task 3"
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
