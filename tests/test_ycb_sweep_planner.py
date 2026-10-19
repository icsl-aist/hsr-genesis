from __future__ import annotations

import math
import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hsr_genesis.grasp_planning import YcbSweepPlanner  # noqa: E402


def test_banana_yaw_penalty_aligns_long_axis_across_fingers() -> None:
    yaws = torch.tensor([-math.pi / 2, 0.0, math.pi / 2], dtype=torch.float32)

    penalty = YcbSweepPlanner._object_yaw_penalty("ycb_011_banana", yaws)

    assert torch.allclose(penalty, torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
