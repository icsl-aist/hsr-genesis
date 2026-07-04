"""Grasp parameter definitions for CMA-ES optimization.

28D search vector = 4 params × 7 YCB objects, concatenated per object.
"""
from __future__ import annotations

import torch

PARAM_NAMES = ["pre_grasp_height", "grasp_offset_z", "gripper_effort", "grasp_hold_steps"]
N_PARAMS = len(PARAM_NAMES)

OBJECT_NAMES = [
    "ycb_061_foam_brick",
    "ycb_013_apple",
    "ycb_011_banana",
    "ycb_017_orange",
    "ycb_056_tennis_ball",
    "ycb_055_baseball",
    "ycb_077_rubiks_cube",
]
N_OBJECTS = len(OBJECT_NAMES)
SOLUTION_LENGTH = N_PARAMS * N_OBJECTS  # 28

# (lower, upper) bounds for each param.
PARAM_BOUNDS = torch.tensor(
    [
        [0.05, 0.30],   # pre_grasp_height (m)
        [-0.02, 0.08],  # grasp_offset_z (m)
        [1.0, 8.0],     # gripper_effort (N)
        [100, 500],     # grasp_hold_steps (int)
    ],
    dtype=torch.float32,
)

# Default values (current hardcoded constants from ycb_pick_ik_parallel.py).
PARAM_DEFAULTS = torch.tensor(
    [0.15, 0.02, 3.0, 300],
    dtype=torch.float32,
)


def denormalize(raw_vector: torch.Tensor) -> torch.Tensor:
    """Clip a 28D (or 4D) raw vector to parameter bounds.

    Args:
        raw_vector: (28,) or (N, 28) or (4,) or (N, 4) tensor.

    Returns:
        Clipped tensor of same shape, within bounds.
        grasp_hold_steps is rounded to int.
    """
    if raw_vector.ndim == 1:
        if raw_vector.shape[0] == SOLUTION_LENGTH:
            matrix = raw_vector.reshape(N_OBJECTS, N_PARAMS)
        elif raw_vector.shape[0] == N_PARAMS:
            matrix = raw_vector.unsqueeze(0)
        else:
            raise ValueError(f"Expected vector of length {SOLUTION_LENGTH} or {N_PARAMS}, got {raw_vector.shape[0]}")
        single = True
    else:
        matrix = raw_vector
        single = False

    lo = PARAM_BOUNDS[:, 0]
    hi = PARAM_BOUNDS[:, 1]
    matrix = matrix.clamp(min=lo, max=hi)
    # Round grasp_hold_steps (index 3) to int.
    matrix[..., 3] = torch.round(matrix[..., 3])
    if single and matrix.shape[0] == 1:
        return matrix.squeeze(0)
    return matrix


def params_to_dict(param_matrix: torch.Tensor) -> dict[str, dict[str, float]]:
    """Convert (7, 4) param matrix to {object_name: {param_name: value}}."""
    if param_matrix.ndim == 1:
        param_matrix = param_matrix.reshape(N_OBJECTS, N_PARAMS)
    result = {}
    for i, obj in enumerate(OBJECT_NAMES):
        result[obj] = {}
        for j, name in enumerate(PARAM_NAMES):
            val = float(param_matrix[i, j])
            if name == "grasp_hold_steps":
                val = int(val)
            result[obj][name] = val
    return result


def params_from_dict(d: dict[str, dict[str, float]]) -> torch.Tensor:
    """Convert {object_name: {param_name: value}} to (7, 4) param matrix."""
    matrix = torch.zeros(N_OBJECTS, N_PARAMS, dtype=torch.float32)
    for i, obj in enumerate(OBJECT_NAMES):
        for j, name in enumerate(PARAM_NAMES):
            matrix[i, j] = float(d[obj][name])
    return matrix


def default_params() -> torch.Tensor:
    """Return (7, 4) matrix with default params for all objects."""
    return PARAM_DEFAULTS.unsqueeze(0).expand(N_OBJECTS, N_PARAMS).clone()
