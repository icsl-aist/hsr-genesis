"""Taichi analytic IK port for HSR-B/HSR-C.

This is a full Python/Taichi port of hsrb_analytic_ik.

License: Portions ported from hsrb_manipulation/hsrb_analytic_ik
are under BSD-compatible terms. This package is released under the
BSD 3-Clause License (see `hsr_genesis/LICENSE.txt`).
"""

import enum
import math
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import gstaichi as ti
import torch
try:
    import taichi.math as tm
except ModuleNotFoundError:
    tm = ti.math

try:
    import genesis as gs
    from genesis.utils.misc import ti_to_torch
except Exception:
    gs = None
    ti_to_torch = None

if "gs" in globals() and getattr(gs, "_initialized", False) and getattr(gs, "ti_float", None) is not None:
    TI_FLOAT = gs.ti_float
else:
    TI_FLOAT = ti.f32


class OptResult(enum.Enum):
    SUCCESS = enum.auto()
    MAX_ITERATION = enum.auto()
    FAIL = enum.auto()


class IKResult(enum.Enum):
    SUCCESS = enum.auto()
    CONVERGE = enum.auto()
    MAX_ITERATION = enum.auto()
    FAIL = enum.auto()


@dataclass
class JointState:
    name: List[str]
    position: torch.Tensor
    velocity: torch.Tensor | None = None
    effort: torch.Tensor | None = None


@dataclass
class IKRequest:
    frame_name: str
    frame_to_end: torch.Tensor
    ref_origin_to_end: torch.Tensor
    origin_to_base: torch.Tensor
    initial_angle: JointState
    use_joints: List[str]
    weight: torch.Tensor
    linear_base_movements: List[torch.Tensor]
    rotational_base_movements: List[torch.Tensor]


@dataclass
class IKResponse:
    solution_angle: JointState
    origin_to_base: torch.Tensor
    origin_to_end: torch.Tensor


@dataclass
class BasePositionRange:
    center: torch.Tensor
    radius_min: float = -1.0
    radius_max: float = -1.0


def _torch_device() -> torch.device:
    if "gs" in globals() and getattr(gs, "_initialized", False) and getattr(gs, "device", None) is not None:
        return gs.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _torch_dtype() -> torch.dtype:
    if "gs" in globals() and getattr(gs, "_initialized", False) and getattr(gs, "tc_float", None) is not None:
        return gs.tc_float
    return torch.float32


def _is_torch_tensor(value) -> bool:
    return isinstance(value, torch.Tensor)


def _as_torch(value) -> torch.Tensor:
    if _is_torch_tensor(value):
        return value
    return torch.as_tensor(value, device=_torch_device(), dtype=_torch_dtype())


def _mat4_from_field(field) -> torch.Tensor:
    mat = ti_to_torch(field, copy=True)
    if mat.dim() == 3 and mat.shape[0] == 1:
        return mat[0]
    return mat


@dataclass
class Vector2:
    v1: float = 0.0
    v2: float = 0.0

    def set(self, a1: float, a2: float) -> None:
        self.v1 = a1
        self.v2 = a2

    def zero(self) -> None:
        self.v1 = 0.0
        self.v2 = 0.0

    def clone(self) -> "Vector2":
        return Vector2(self.v1, self.v2)

    def norm(self) -> float:
        return math.sqrt(self.v1 * self.v1 + self.v2 * self.v2)

    def norm2(self) -> float:
        return self.v1 * self.v1 + self.v2 * self.v2

    def normalize(self) -> None:
        n = self.norm()
        if n == 0.0:
            return
        self.v1 /= n
        self.v2 /= n

    @staticmethod
    def diff_norm(x: "Vector2", y: "Vector2") -> float:
        dx = x.v1 - y.v1
        dy = x.v2 - y.v2
        return math.sqrt(dx * dx + dy * dy)

    def __add__(self, other: "Vector2") -> "Vector2":
        return Vector2(self.v1 + other.v1, self.v2 + other.v2)

    def __sub__(self, other: "Vector2") -> "Vector2":
        return Vector2(self.v1 - other.v1, self.v2 - other.v2)

    def __mul__(self, scalar: float) -> "Vector2":
        return Vector2(self.v1 * scalar, self.v2 * scalar)

    __rmul__ = __mul__

    def __truediv__(self, scalar: float) -> "Vector2":
        inv = 1.0 / scalar
        return Vector2(self.v1 * inv, self.v2 * inv)


class GoldenSectionLineSearch:
    def __init__(self, max_iteration: int = 100, epsilon: float = 1e-4):
        self.max_iteration = max_iteration
        self.epsilon = epsilon
        self.result = OptResult.FAIL
        self.iteration = 0
        self.solution = 0.0
        self.value = 0.0

    def search(self, func: Callable[[float], float], a: float, b: float) -> OptResult:
        self.result = OptResult.FAIL
        self.iteration = 0
        self.solution = 0.0
        self.value = 0.0

        a_k = float(a)
        b_k = float(b)
        f_a = func(a_k)
        f_b = func(b_k)
        alpha = 0.6180339887498948482

        s = a_k + (1.0 - alpha) * (b_k - a_k)
        t = a_k + alpha * (b_k - a_k)
        f_s = func(s)
        f_t = func(t)

        for k in range(1, self.max_iteration + 1):
            left = (f_a <= f_s and f_a <= f_t and f_a <= f_b) or (
                f_s <= f_a and f_s <= f_t and f_s <= f_b
            )

            if left:
                b_k = t
                f_b = f_t
                t = s
                f_t = f_s
                s = a_k + (1.0 - alpha) * (b_k - a_k)
                f_s = func(s)
            else:
                a_k = s
                f_a = f_s
                s = t
                f_s = f_t
                t = a_k + alpha * (b_k - a_k)
                f_t = func(t)

            if (b_k - a_k) <= self.epsilon:
                self.result = OptResult.SUCCESS
                self.iteration = k
                if f_a <= f_b:
                    self.solution = a_k
                    self.value = f_a
                else:
                    self.solution = b_k
                    self.value = f_b
                return self.result

        self.result = OptResult.MAX_ITERATION
        self.iteration = self.max_iteration
        if f_a <= f_b:
            self.solution = a_k
            self.value = f_a
        else:
            self.solution = b_k
            self.value = f_b
        return self.result


class UniGoldenSectionLineSearch:
    def __init__(self, max_iteration: int = 100, epsilon: float = 1e-4):
        self.max_iteration = max_iteration
        self.epsilon = epsilon
        self.result = OptResult.FAIL
        self.iteration = 0
        self.solution = 0.0
        self.value = 0.0

    def search(self, func: Callable[[float], float], step: float) -> OptResult:
        self.result = OptResult.FAIL
        self.iteration = 0
        self.solution = 0.0
        self.value = 0.0

        search = GoldenSectionLineSearch(self.max_iteration, self.epsilon)
        a = 0.0
        b = float(step)

        result = search.search(func, a, b)
        self.iteration = search.iteration
        self.solution = search.solution
        self.value = search.value

        if result == OptResult.SUCCESS:
            self.result = OptResult.SUCCESS
        elif result == OptResult.MAX_ITERATION:
            self.result = OptResult.MAX_ITERATION
            return self.result

        if result == OptResult.FAIL:
            for _k in range(2, self.max_iteration + 1):
                b *= 0.5
                result = search.search(func, a, b)
                if result == OptResult.SUCCESS:
                    self.result = OptResult.SUCCESS
                    self.iteration = search.iteration
                    self.solution = search.solution
                    self.value = search.value
                    return self.result
            self.result = OptResult.MAX_ITERATION
            return self.result

        for _k in range(2, self.max_iteration + 1):
            b *= 2.0
            result = search.search(func, a, b)
            if result == OptResult.SUCCESS:
                comp_solution = search.solution
                comp_value = search.value
                if abs(self.solution - comp_solution) <= self.epsilon:
                    if comp_value < self.value:
                        self.iteration = search.iteration
                        self.solution = comp_solution
                        self.value = comp_value
                    return self.result
                if comp_value < self.value:
                    self.iteration = search.iteration
                    self.solution = comp_solution
                    self.value = comp_value
                    continue
                return self.result
            return self.result

        return self.result


@ti.dataclass
class RobotFunction2Request:
    R11: TI_FLOAT
    R12: TI_FLOAT
    R13: TI_FLOAT
    px: TI_FLOAT
    R21: TI_FLOAT
    R22: TI_FLOAT
    R23: TI_FLOAT
    py: TI_FLOAT
    R31: TI_FLOAT
    R32: TI_FLOAT
    R33: TI_FLOAT
    pz: TI_FLOAT
    w0: TI_FLOAT
    w1: TI_FLOAT
    w2: TI_FLOAT
    w3: TI_FLOAT
    w4: TI_FLOAT
    w5: TI_FLOAT
    w6: TI_FLOAT
    w7: TI_FLOAT
    r0: TI_FLOAT
    r1: TI_FLOAT
    r2: TI_FLOAT
    r3: TI_FLOAT
    r4: TI_FLOAT
    r5: TI_FLOAT
    r6: TI_FLOAT
    r7: TI_FLOAT


@ti.dataclass
class RobotFunction2Response:
    t0: TI_FLOAT
    t1: TI_FLOAT
    t2: TI_FLOAT
    t3: TI_FLOAT
    t4: TI_FLOAT
    t5: TI_FLOAT
    t6: TI_FLOAT
    t7: TI_FLOAT


@ti.dataclass
class RobotFunction2Parameter:
    L3: TI_FLOAT
    L41: TI_FLOAT
    L42: TI_FLOAT
    L51: TI_FLOAT
    L52: TI_FLOAT
    L81: TI_FLOAT
    L82: TI_FLOAT
    t3_min: TI_FLOAT
    t3_max: TI_FLOAT
    t4_min: TI_FLOAT
    t4_max: TI_FLOAT
    t5_min: TI_FLOAT
    t5_max: TI_FLOAT
    t6_min: TI_FLOAT
    t6_max: TI_FLOAT
    t7_min: TI_FLOAT
    t7_max: TI_FLOAT


class PenaltyType(enum.Enum):
    NONE = enum.auto()
    BIG_DISCONTINUOUS = enum.auto()
    BIG_PROPORTIONAL = enum.auto()


MAX_GRID_RESOLUTION = 52
MAX_GRID_POINTS = MAX_GRID_RESOLUTION * MAX_GRID_RESOLUTION
MAX_BASE_YAW_SOLUTIONS = 16
MAX_OPTIMIZER_ITER = 500
GOLDEN_ALPHA = 0.6180339887498948482


def _taichi_gpu_available() -> bool:
    try:
        arch = ti.cfg.arch
    except Exception:
        return False
    gpu_arches = [ti.cuda, ti.vulkan]
    if hasattr(ti, "opengl"):
        gpu_arches.append(ti.opengl)
    if hasattr(ti, "metal"):
        gpu_arches.append(ti.metal)
    return arch in tuple(gpu_arches)


@ti.data_oriented
class AnalyticIKWorkspace:
    def __init__(self):
        self.candidate_grid = ti.Vector.field(
            2,
            dtype=TI_FLOAT,
            shape=MAX_GRID_POINTS,
        )
        self.candidate_count = ti.field(dtype=ti.i32, shape=())

        self.base_range_center_field = ti.Vector.field(
            2,
            dtype=TI_FLOAT,
            shape=(),
        )
        self.base_range_radius_min_field = ti.field(dtype=TI_FLOAT, shape=())
        self.base_range_radius_max_field = ti.field(dtype=TI_FLOAT, shape=())

        self.closest_solution_index_field = ti.field(dtype=ti.i32, shape=())

        self.origin_to_base_field = ti.Matrix.field(
            4,
            4,
            dtype=TI_FLOAT,
            shape=(),
        )
        self.origin_to_end_field = ti.Matrix.field(
            4,
            4,
            dtype=TI_FLOAT,
            shape=(),
        )

        self.base_yaw_solution_count = ti.field(dtype=ti.i32, shape=())
        self.base_yaw_solutions = ti.Vector.field(
            8,
            dtype=TI_FLOAT,
            shape=MAX_BASE_YAW_SOLUTIONS,
        )


@ti.kernel
def _solve_base_yaw_ik_batch_kernel(
    param: RobotFunction2Parameter,
    ref_origin_to_end: ti.types.ndarray(dtype=TI_FLOAT, ndim=3),
    theta0: ti.types.ndarray(dtype=TI_FLOAT, ndim=1),
    theta1: ti.types.ndarray(dtype=TI_FLOAT, ndim=1),
    base_yaw_solution_count: ti.template(),
    base_yaw_solutions: ti.template(),
):
    for e in range(ref_origin_to_end.shape[0]):
        base_yaw_solution_count[e] = 0
        T78_inv = ti.Matrix(
            [
                [-1.0, 0.0, 0.0, param.L81],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -param.L82],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        UB8o = ti.Matrix(
            [
                [
                    ref_origin_to_end[e, 0, 0],
                    ref_origin_to_end[e, 0, 1],
                    ref_origin_to_end[e, 0, 2],
                    ref_origin_to_end[e, 0, 3],
                ],
                [
                    ref_origin_to_end[e, 1, 0],
                    ref_origin_to_end[e, 1, 1],
                    ref_origin_to_end[e, 1, 2],
                    ref_origin_to_end[e, 1, 3],
                ],
                [
                    ref_origin_to_end[e, 2, 0],
                    ref_origin_to_end[e, 2, 1],
                    ref_origin_to_end[e, 2, 2],
                    ref_origin_to_end[e, 2, 3],
                ],
                [
                    ref_origin_to_end[e, 3, 0],
                    ref_origin_to_end[e, 3, 1],
                    ref_origin_to_end[e, 3, 2],
                    ref_origin_to_end[e, 3, 3],
                ],
            ]
        )
        UB7o = UB8o @ T78_inv

        xwo = UB7o[0, 3]
        ywo = UB7o[1, 3]
        zwo = UB7o[2, 3]

        A2o = -theta0[e] + xwo
        B2o = theta1[e] - ywo
        C2o = -param.L42
        D2o = tm.sqrt(A2o * A2o + B2o * B2o)

        trig2 = _ti_trigonometric_composition_formula(A2o, B2o, C2o, D2o)
        if trig2[0] >= 0.5:

            eps_theta = 1e-3
            A4o = -param.L52
            B4o = param.L51
            D4o = tm.sqrt(A4o * A4o + B4o * B4o)

            for idx2 in ti.static(range(2)):
                theta2 = trig2[1] if idx2 == 0 else trig2[2]
                C4o = (
                    -tm.cos(theta2) * (theta0[e] - xwo)
                    - tm.sin(theta2) * (theta1[e] - ywo)
                    - param.L41
                )
                trig4 = _ti_trigonometric_composition_formula(A4o, B4o, C4o, D4o)
                if trig4[0] >= 0.5:
                    for idx4 in ti.static(range(2)):
                        theta4 = trig4[1] if idx4 == 0 else trig4[2]
                        judge4 = _ti_judge_theta_pi_pi(
                            param.t4_max,
                            param.t4_min,
                            eps_theta,
                            theta4,
                        )
                        if judge4[0] >= 0.5:
                            theta4_checked = judge4[1]
                            theta3 = (
                                zwo
                                - param.L3
                                - param.L52 * tm.cos(theta4_checked)
                                - param.L51 * tm.sin(theta4_checked)
                            )
                            judge3 = _ti_judge_theta_pi_pi(
                                param.t3_max,
                                param.t3_min,
                                eps_theta,
                                theta3,
                            )
                            if judge3[0] >= 0.5:
                                theta3_checked = judge3[1]

                                TBm1 = ti.Matrix(
                                    [
                                        [0.0, 0.0, 1.0, 0.0],
                                        [1.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0],
                                    ]
                                )
                                Tm10 = ti.Matrix(
                                    [
                                        [0.0, -1.0, 0.0, 0.0],
                                        [1.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, theta0[e]],
                                        [0.0, 0.0, 0.0, 1.0],
                                    ]
                                )
                                T01 = ti.Matrix(
                                    [
                                        [0.0, -1.0, 0.0, 0.0],
                                        [0.0, 0.0, -1.0, -theta1[e]],
                                        [1.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0],
                                    ]
                                )
                                T12 = ti.Matrix(
                                    [
                                        [-tm.sin(theta2), -tm.cos(theta2), 0.0, 0.0],
                                        [0.0, 0.0, -1.0, 0.0],
                                        [tm.cos(theta2), -tm.sin(theta2), 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0],
                                    ]
                                )
                                T23 = ti.Matrix(
                                    [
                                        [0.0, 1.0, 0.0, 0.0],
                                        [-1.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, theta3_checked + param.L3],
                                        [0.0, 0.0, 0.0, 1.0],
                                    ]
                                )
                                T34 = ti.Matrix(
                                    [
                                        [tm.cos(theta4_checked), -tm.sin(theta4_checked), 0.0, param.L41],
                                        [0.0, 0.0, -1.0, param.L42],
                                        [tm.sin(theta4_checked), tm.cos(theta4_checked), 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0],
                                    ]
                                )

                                UB4o = TBm1 @ Tm10 @ T01 @ T12 @ T23 @ T34
                                U47o = _ti_transform_inv(UB4o) @ UB7o

                                theta61 = tm.atan2(
                                    -tm.sqrt(
                                        U47o[1, 0] * U47o[1, 0]
                                        + U47o[1, 1] * U47o[1, 1]
                                    ),
                                    U47o[1, 2],
                                )
                                theta62 = tm.atan2(
                                    tm.sqrt(
                                        U47o[1, 0] * U47o[1, 0]
                                        + U47o[1, 1] * U47o[1, 1]
                                    ),
                                    U47o[1, 2],
                                )

                                for idx6 in ti.static(range(2)):
                                    theta6 = theta61 if idx6 == 0 else theta62
                                    if ti.abs(theta6) < 1e-9:
                                        theta57a = tm.atan2(-U47o[0, 1], U47o[0, 0])
                                        theta57b = tm.atan2(-U47o[2, 0], -U47o[2, 1])
                                        if ti.abs(theta57a - theta57b) < 1e-6:
                                            theta5 = theta57a * 0.5
                                            theta7 = theta57a * 0.5
                                            idx = ti.atomic_add(
                                                base_yaw_solution_count[e],
                                                1,
                                            )
                                            if idx < ti.static(MAX_BASE_YAW_SOLUTIONS):
                                                base_yaw_solutions[e, idx] = ti.Vector(
                                                    [
                                                        theta0[e],
                                                        theta1[e],
                                                        theta2,
                                                        theta3_checked,
                                                        theta4_checked,
                                                        theta5,
                                                        theta6,
                                                        theta7,
                                                    ]
                                                )
                                    else:
                                        judge6 = _ti_judge_theta_pi_pi(
                                            param.t6_max,
                                            param.t6_min,
                                            eps_theta,
                                            theta6,
                                        )
                                        if judge6[0] >= 0.5:
                                            theta6_checked = judge6[1]
                                            sign6 = ti.select(
                                                theta6_checked >= 0.0,
                                                1.0,
                                                -1.0,
                                            )
                                            theta7 = tm.atan2(
                                                -U47o[1, 1] * sign6,
                                                U47o[1, 0] * sign6,
                                            )
                                            judge7 = _ti_judge_theta_pi_pi(
                                                param.t7_max,
                                                param.t7_min,
                                                eps_theta,
                                                theta7,
                                            )
                                            if judge7[0] >= 0.5:
                                                theta7_checked = judge7[1]
                                                theta5 = tm.atan2(
                                                    U47o[2, 2] * sign6,
                                                    -U47o[0, 2] * sign6,
                                                )
                                                judge5 = _ti_judge_theta_pi_pi(
                                                    param.t5_max,
                                                    param.t5_min,
                                                    eps_theta,
                                                    theta5,
                                                )
                                                if judge5[0] >= 0.5:
                                                    theta5_checked = judge5[1]
                                                    idx = ti.atomic_add(
                                                        base_yaw_solution_count[e],
                                                        1,
                                                    )
                                                    if idx < ti.static(MAX_BASE_YAW_SOLUTIONS):
                                                        base_yaw_solutions[e, idx] = ti.Vector(
                                                            [
                                                                theta0[e],
                                                                theta1[e],
                                                                theta2,
                                                                theta3_checked,
                                                                theta4_checked,
                                                                theta5_checked,
                                                                theta6_checked,
                                                                theta7_checked,
                                                            ]
                                                        )


@ti.func
def _grid_value(lower: TI_FLOAT, step: TI_FLOAT, idx: ti.i32) -> TI_FLOAT:
    return lower + step * ti.cast(idx, TI_FLOAT)


@ti.kernel
def _generate_candidate_grid(
    lower: TI_FLOAT,
    upper: TI_FLOAT,
    t2_lower: TI_FLOAT,
    t2_upper: TI_FLOAT,
    grid: ti.i32,
    candidate_grid: ti.template(),
    candidate_count: ti.template(),
):
    dim = ti.max(3, grid + 2)
    candidate_count[None] = 0
    t2_step = (t2_upper - t2_lower) / ti.cast(dim, TI_FLOAT)
    t4_step = (upper - lower) / ti.cast(dim, TI_FLOAT)

    for i, j in ti.ndrange((1, dim - 1), (1, dim - 1)):
        idx = candidate_count[None]
        if idx < ti.static(MAX_GRID_POINTS):
            x = _grid_value(t2_lower, t2_step, i)
            y = _grid_value(lower, t4_step, j)
            candidate_grid[idx] = ti.Vector([x, y])
            candidate_count[None] = idx + 1


def _sample_candidate_grid(
    lower: float,
    upper: float,
    t2_lower: float,
    t2_upper: float,
    grid: int,
    workspace: AnalyticIKWorkspace,
) -> torch.Tensor:
    grid = max(1, min(grid, MAX_GRID_RESOLUTION - 2))
    _generate_candidate_grid(
        lower,
        upper,
        t2_lower,
        t2_upper,
        grid,
        workspace.candidate_grid,
        workspace.candidate_count,
    )
    count = int(ti_to_torch(workspace.candidate_count, copy=True).item())
    return ti_to_torch(workspace.candidate_grid, copy=True)[:count]


@ti.func
def _ti_penalty_term(
    lower: TI_FLOAT,
    upper: TI_FLOAT,
    value: TI_FLOAT,
    weight: TI_FLOAT,
) -> TI_FLOAT:
    return weight * (ti.max(0.0, lower - value) + ti.max(0.0, value - upper))


@ti.func
def _ti_penalty_grade(
    req: RobotFunction2Request,
    param: RobotFunction2Parameter,
    resp: RobotFunction2Response,
    weighted: ti.i32,
) -> TI_FLOAT:
    w3 = ti.select(weighted == 1, req.w3, 1.0)
    w4 = ti.select(weighted == 1, req.w4, 1.0)
    w5 = ti.select(weighted == 1, req.w5, 1.0)
    w6 = ti.select(weighted == 1, req.w6, 1.0)
    w7 = ti.select(weighted == 1, req.w7, 1.0)

    term3 = _ti_penalty_term(param.t3_min, param.t3_max, resp.t3, w3)
    term4 = _ti_penalty_term(param.t4_min, param.t4_max, resp.t4, w4)
    term5 = _ti_penalty_term(param.t5_min, param.t5_max, resp.t5, w5)
    term6 = _ti_penalty_term(param.t6_min, param.t6_max, resp.t6, w6)
    term7 = _ti_penalty_term(param.t7_min, param.t7_max, resp.t7, w7)
    return term3 + term4 + term5 + term6 + term7


@ti.func
def _ti_is_feasible(
    param: RobotFunction2Parameter,
    resp: RobotFunction2Response,
) -> ti.i32:
    cond = (
        (resp.t3 >= param.t3_min)
        and (resp.t3 <= param.t3_max)
        and (resp.t4 >= param.t4_min)
        and (resp.t4 <= param.t4_max)
        and (resp.t5 >= param.t5_min)
        and (resp.t5 <= param.t5_max)
        and (resp.t6 >= param.t6_min)
        and (resp.t6 <= param.t6_max)
        and (resp.t7 >= param.t7_min)
        and (resp.t7 <= param.t7_max)
    )
    return ti.cast(cond, ti.i32)


@ti.func
def _ti_weighted_squared_error(
    req: RobotFunction2Request,
    resp: RobotFunction2Response,
) -> TI_FLOAT:
    d0 = req.w0 * (req.r0 - resp.t0)
    d1 = req.w1 * (req.r1 - resp.t1)
    d2 = req.w2 * (req.r2 - resp.t2)
    d3 = req.w3 * (req.r3 - resp.t3)
    d4 = req.w4 * (req.r4 - resp.t4)
    d5 = req.w5 * (req.r5 - resp.t5)
    d6 = req.w6 * (req.r6 - resp.t6)
    d7 = req.w7 * (req.r7 - resp.t7)
    return (
        d0 * d0
        + d1 * d1
        + d2 * d2
        + d3 * d3
        + d4 * d4
        + d5 * d5
        + d6 * d6
        + d7 * d7
    )


@ti.func
def _ti_normalize_angle(angle: TI_FLOAT, lower: TI_FLOAT) -> TI_FLOAT:
    two_pi = 2.0 * tm.pi
    offset = ti.ceil((lower - angle) / two_pi)
    return angle + offset * two_pi


@ti.func
def _ti_compute_chain(
    t6_val: TI_FLOAT,
    S2: TI_FLOAT,
    C2: TI_FLOAT,
    S4: TI_FLOAT,
    C4: TI_FLOAT,
    req: RobotFunction2Request,
    param: RobotFunction2Parameter,
) -> ti.types.vector(2, TI_FLOAT):
    inv = ti.select(tm.sin(t6_val) > 0.0, 1.0, -1.0)
    a1 = (req.R23 * C2 - req.R13 * S2) * inv
    b1 = -(req.R33 * S4 + req.R23 * S2 * C4 + req.R13 * C2 * C4) * inv
    t5 = -tm.atan2(a1, b1)
    t5 = _ti_normalize_angle(t5, param.t5_min)

    a2 = (-req.R22 * S2 * S4 - req.R12 * C2 * S4 + req.R32 * C4) * inv
    b2 = -(-req.R21 * S2 * S4 - req.R11 * C2 * S4 + req.R31 * C4) * inv
    t7 = tm.atan2(a2, b2)
    t7 = _ti_normalize_angle(t7, param.t7_min)
    return ti.Vector([t5, t7])


@ti.func
def _ti_handle_t6_singular(
    resp: RobotFunction2Response,
    S2: TI_FLOAT,
    C2: TI_FLOAT,
    req: RobotFunction2Request,
    param: RobotFunction2Parameter,
):
    a = tm.atan2(req.R11 * S2 - req.R21 * C2, req.R12 * S2 - req.R22 * C2)
    numerator = req.w7 * req.w7 * (a - req.r7) + req.r5 * req.w5 * req.w5
    denominator = req.w7 * req.w7 + req.w5 * req.w5
    t5 = numerator / denominator
    t5 = ti.min(ti.max(t5, param.t5_min), param.t5_max)
    resp.t5 = t5
    resp.t7 = -t5 + a
    resp.t6 = 0.0
    zero = ti.Vector([ti.cast(0.0, TI_FLOAT), ti.cast(0.0, TI_FLOAT)])
    return resp, zero, zero, zero, ti.cast(1, ti.i32)


@ti.func
def _ti_partial_t6(
    S2: TI_FLOAT,
    C2: TI_FLOAT,
    S4: TI_FLOAT,
    C4: TI_FLOAT,
    req: RobotFunction2Request,
    use_plus: ti.i32,
) -> ti.types.vector(2, TI_FLOAT):
    a = req.R23 * S2 * S4 + req.R13 * C2 * S4 - req.R33 * C4
    a_sq = a * a
    b = req.R13 * S2 * S4 - req.R23 * C2 * S4
    c = -(req.R33 * S4 + req.R23 * S2 * C4 + req.R13 * C2 * C4)
    eps = ti.cast(1e-12, TI_FLOAT)
    one = ti.cast(1.0, TI_FLOAT)
    denom = tm.sqrt(tm.max(eps, one - a_sq))
    d6_2 = b / denom
    d6_4 = c / denom
    flip = ti.select(use_plus == 1, ti.cast(-1.0, TI_FLOAT), ti.cast(1.0, TI_FLOAT))
    return ti.Vector([d6_2 * flip, d6_4 * flip])


@ti.func
def _ti_partial_t5(
    S2: TI_FLOAT,
    C2: TI_FLOAT,
    S4: TI_FLOAT,
    C4: TI_FLOAT,
    req: RobotFunction2Request,
) -> ti.types.vector(2, TI_FLOAT):
    a1 = req.R23 * S2 + req.R13 * C2
    P1 = a1 * req.R33 * S4 + (req.R23 * req.R23 + req.R13 * req.R13) * C4
    Q1 = (
        2.0 * req.R13 * req.R23 * C2 * S2
        + (req.R13 * req.R13 - req.R23 * req.R23) * C2 * C2
        - req.R33 * req.R33
        + req.R23 * req.R23
    )
    R1 = -2.0 * a1 * req.R33 * C4 * S4 - (
        req.R23 * req.R23 + req.R13 * req.R13
    )
    d5_2 = P1 / (Q1 * S4 * S4 + R1)

    b1 = req.R13 * S2 - req.R23 * C2
    P2 = a1 * b1 * S4 + (req.R23 * req.R33 * C2 - req.R13 * req.R33 * S2) * C4
    Q2 = req.R33 * req.R33 * S4 * S4 + 2.0 * a1 * req.R33 * C4 * S4
    R2 = a1 * a1 * C4 * C4 + b1 * b1
    d5_4 = -P2 / (Q2 + R2)
    return ti.Vector([d5_2, d5_4])


@ti.func
def _ti_partial_t7(
    S2: TI_FLOAT,
    C2: TI_FLOAT,
    S4: TI_FLOAT,
    C4: TI_FLOAT,
    req: RobotFunction2Request,
) -> ti.types.vector(2, TI_FLOAT):
    a2 = req.R12 * req.R31 - req.R11 * req.R32
    b2 = req.R21 * req.R32 - req.R22 * req.R31
    c2 = 2.0 * (req.R12 * req.R22 + req.R11 * req.R21) * C2 * S2

    Z = -2.0 * (
        (req.R22 * req.R32 + req.R21 * req.R31) * S2
        + (req.R12 * req.R32 + req.R11 * req.R31) * C2
    ) * C4 * S4

    P3 = (
        (req.R11 * req.R22 - req.R12 * req.R21) * S4 * S4
        + (a2 * S2 + b2 * C2) * C4 * S4
    )
    Q3 = (
        c2
        + (-(req.R22 * req.R22 + req.R21 * req.R21) + (req.R12 * req.R12 + req.R11 * req.R11))
        * C2
        * C2
        + req.R22 * req.R22
        + req.R21 * req.R21
    ) * S4 * S4
    U3 = Z + (req.R32 * req.R32 + req.R31 * req.R31) * C4 * C4
    d7_2 = -P3 / (Q3 + U3)

    P4 = b2 * S2 - a2 * C2
    Q4 = (
        (req.R22 * req.R22 + req.R21 * req.R21) * S2 * S2
        + c2
        + (req.R12 * req.R12 + req.R11 * req.R11) * C2 * C2
        - (req.R32 * req.R32 + req.R31 * req.R31)
    ) * S4 * S4
    U4 = Z + req.R32 * req.R32 + req.R31 * req.R31
    d7_4 = -P4 / (Q4 + U4)
    return ti.Vector([d7_2, d7_4])


@ti.func
def _ti_calculate_theta(
    t2: TI_FLOAT,
    t4: TI_FLOAT,
    req: RobotFunction2Request,
    param: RobotFunction2Parameter,
    resp: RobotFunction2Response,
):
    resp.t2 = t2
    resp.t4 = t4
    S2 = tm.sin(t2)
    C2 = tm.cos(t2)
    S4 = tm.sin(t4)
    C4 = tm.cos(t4)

    resp.t0 = (
        -req.R13 * param.L82
        + req.R11 * param.L81
        + param.L52 * C2 * S4
        - param.L51 * C2 * C4
        + param.L42 * S2
        - param.L41 * C2
        + req.px
    )
    resp.t1 = (
        -req.R23 * param.L82
        + req.R21 * param.L81
        + param.L52 * S2 * S4
        - param.L51 * S2 * C4
        - param.L42 * C2
        - param.L41 * S2
        + req.py
    )
    resp.t3 = (
        -req.R33 * param.L82
        + req.R31 * param.L81
        - param.L52 * C4
        - param.L51 * S4
        - param.L3
        + req.pz
    )

    b = -req.R23 * S2 * S4 - req.R13 * C2 * S4 + req.R33 * C4
    a = tm.sqrt(tm.max(0.0, 1.0 - b * b))
    s_plus = tm.atan2(a, b)
    s_minus = -s_plus

    partial_t5 = _ti_partial_t5(S2, C2, S4, C4, req)
    partial_t7 = _ti_partial_t7(S2, C2, S4, C4, req)

    # Default outputs (will be overwritten below)
    resp_out = resp
    partial_t6 = ti.Vector([ti.cast(0.0, TI_FLOAT), ti.cast(0.0, TI_FLOAT)])
    use_plus = ti.cast(1, ti.i32)

    # Singular branch (t6 ~= 0)
    singular = ti.abs(tm.sin(s_plus)) < 1e-9
    if singular:
        resp_out, partial_t5, partial_t6, partial_t7, use_plus = _ti_handle_t6_singular(
            resp,
            S2,
            C2,
            req,
            param,
        )
    else:
        pair_plus = _ti_compute_chain(s_plus, S2, C2, S4, C4, req, param)
        pair_minus = _ti_compute_chain(s_minus, S2, C2, S4, C4, req, param)

        resp_plus = resp
        resp_plus.t5 = pair_plus[0]
        resp_plus.t6 = s_plus
        resp_plus.t7 = pair_plus[1]

        resp_minus = resp
        resp_minus.t5 = pair_minus[0]
        resp_minus.t6 = s_minus
        resp_minus.t7 = pair_minus[1]

        plus_value = _ti_weighted_squared_error(req, resp_plus)
        minus_value = _ti_weighted_squared_error(req, resp_minus)

        use_plus = ti.cast(0, ti.i32)
        resp_out = resp_minus
        if plus_value <= minus_value:
            use_plus = ti.cast(1, ti.i32)
            resp_out = resp_plus

        partial_t6 = _ti_partial_t6(S2, C2, S4, C4, req, use_plus)

    return resp_out, partial_t5, partial_t6, partial_t7, use_plus


@ti.func
def _ti_base_gradient(
    req: RobotFunction2Request,
    param: RobotFunction2Parameter,
    resp: RobotFunction2Response,
    partial_t5: ti.types.vector(2, TI_FLOAT),
    partial_t6: ti.types.vector(2, TI_FLOAT),
    partial_t7: ti.types.vector(2, TI_FLOAT),
) -> ti.types.vector(2, TI_FLOAT):
    S2 = tm.sin(resp.t2)
    C2 = tm.cos(resp.t2)
    S4 = tm.sin(resp.t4)
    C4 = tm.cos(resp.t4)

    d0_2 = (
        -param.L52 * S2 * S4
        + param.L51 * S2 * C4
        + param.L42 * C2
        + param.L41 * S2
    )
    d0_4 = param.L52 * C2 * C4 + param.L51 * C2 * S4

    d1_2 = (
        param.L52 * C2 * S4
        - param.L51 * C2 * C4
        + param.L42 * S2
        - param.L41 * C2
    )
    d1_4 = param.L52 * S2 * C4 + param.L51 * S2 * S4

    d2_2 = 1.0
    d2_4 = 0.0

    d3_2 = 0.0
    d3_4 = param.L52 * S4 - param.L51 * C4

    d4_2 = 0.0
    d4_4 = 1.0

    d5_2 = partial_t5[0]
    d5_4 = partial_t5[1]
    d6_2 = partial_t6[0]
    d6_4 = partial_t6[1]
    d7_2 = partial_t7[0]
    d7_4 = partial_t7[1]

    coeffs = ti.Vector(
        [
            req.w0 * req.w0 * (req.r0 - resp.t0),
            req.w1 * req.w1 * (req.r1 - resp.t1),
            req.w2 * req.w2 * (req.r2 - resp.t2),
            req.w3 * req.w3 * (req.r3 - resp.t3),
            req.w4 * req.w4 * (req.r4 - resp.t4),
            req.w5 * req.w5 * (req.r5 - resp.t5),
            req.w6 * req.w6 * (req.r6 - resp.t6),
            req.w7 * req.w7 * (req.r7 - resp.t7),
        ]
    )

    g2 = (
        d0_2 * coeffs[0]
        + d1_2 * coeffs[1]
        + d2_2 * coeffs[2]
        + d3_2 * coeffs[3]
        + d4_2 * coeffs[4]
        + d5_2 * coeffs[5]
        + d6_2 * coeffs[6]
        + d7_2 * coeffs[7]
    )
    g4 = (
        d0_4 * coeffs[0]
        + d1_4 * coeffs[1]
        + d2_4 * coeffs[2]
        + d3_4 * coeffs[3]
        + d4_4 * coeffs[4]
        + d5_4 * coeffs[5]
        + d6_4 * coeffs[6]
        + d7_4 * coeffs[7]
    )

    return ti.Vector([-2.0 * g2, -2.0 * g4])


@ti.func
def _ti_penalty_gradient_vec(
    req: RobotFunction2Request,
    param: RobotFunction2Parameter,
    resp: RobotFunction2Response,
    partial_t5: ti.types.vector(2, TI_FLOAT),
    partial_t6: ti.types.vector(2, TI_FLOAT),
    partial_t7: ti.types.vector(2, TI_FLOAT),
) -> ti.types.vector(2, TI_FLOAT):
    d3 = ti.Vector(
        [
            0.0,
            param.L52 * tm.sin(resp.t4) - param.L51 * tm.cos(resp.t4),
        ]
    )
    d4 = ti.Vector([0.0, 1.0])
    d5 = partial_t5
    d6 = partial_t6
    d7 = partial_t7

    h = ti.Vector([0.0, 0.0])

    def accumulate(val, lower, upper, weight, dvec):
        if val > upper:
            h[0] += weight * dvec[0]
            h[1] += weight * dvec[1]
        elif val < lower:
            h[0] -= weight * dvec[0]
            h[1] -= weight * dvec[1]

    accumulate(resp.t3, param.t3_min, param.t3_max, req.w3, d3)
    accumulate(resp.t4, param.t4_min, param.t4_max, req.w4, d4)
    accumulate(resp.t5, param.t5_min, param.t5_max, req.w5, d5)
    accumulate(resp.t6, param.t6_min, param.t6_max, req.w6, d6)
    accumulate(resp.t7, param.t7_min, param.t7_max, req.w7, d7)

    return h


@ti.data_oriented
class RobotFunction2:
    def __init__(
        self,
        request: RobotFunction2Request,
        parameter: RobotFunction2Parameter,
    ):
        self.request = request
        self.parameter = parameter
        self.penalty_type = PenaltyType.BIG_PROPORTIONAL
        self.penalty_coeff = 1000.0
        self._response = RobotFunction2Response(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._t6_use_plus = True

        self._response_field = RobotFunction2Response.field(shape=())
        self._request_field = RobotFunction2Request.field(shape=())
        self._parameter_field = RobotFunction2Parameter.field(shape=())
        self._partial_t5_field = ti.Vector.field(2, dtype=TI_FLOAT, shape=())
        self._partial_t6_field = ti.Vector.field(2, dtype=TI_FLOAT, shape=())
        self._partial_t7_field = ti.Vector.field(2, dtype=TI_FLOAT, shape=())
        self._t6_use_plus_field = ti.field(dtype=ti.i32, shape=())
        self._gradient_base_field = ti.Vector.field(2, dtype=TI_FLOAT, shape=())
        self._penalty_gradient_field = ti.Vector.field(2, dtype=TI_FLOAT, shape=())

        self.update_inputs(request, parameter)

    def update_inputs(
        self,
        request: RobotFunction2Request,
        parameter: RobotFunction2Parameter,
    ) -> None:
        self.request = request
        self.parameter = parameter
        self._request_field[None] = request
        self._parameter_field[None] = parameter

    @property
    def response(self) -> RobotFunction2Response:
        return self._response

    @property
    def t6_use_plus(self) -> bool:
        return self._t6_use_plus

    def set_penalty_coeff(self, coeff: float) -> None:
        self.penalty_coeff = coeff

    def set_penalty_type(self, penalty_type: PenaltyType) -> None:
        self.penalty_type = penalty_type

    def is_feasible(self, x: Vector2) -> bool:
        _, _, feasible = self.evaluate_candidate(x, weighted_penalty=False)
        return feasible

    def is_feasible_from_members(self) -> bool:
        p = self.parameter
        r = self._response
        return (
            p.t3_min <= r.t3 <= p.t3_max
            and p.t4_min <= r.t4 <= p.t4_max
            and p.t5_min <= r.t5 <= p.t5_max
            and p.t6_min <= r.t6 <= p.t6_max
            and p.t7_min <= r.t7 <= p.t7_max
        )

    def force_feasible(self) -> None:
        p = self.parameter
        r = self._response
        if r.t4 < p.t4_min:
            self._calculate_theta(Vector2(r.t2, p.t4_min))
        elif r.t4 > p.t4_max:
            self._calculate_theta(Vector2(r.t2, p.t4_max))

        r.t3 = min(max(r.t3, p.t3_min), p.t3_max)
        r.t5 = min(max(r.t5, p.t5_min), p.t5_max)
        r.t6 = min(max(r.t6, p.t6_min), p.t6_max)
        r.t7 = min(max(r.t7, p.t7_min), p.t7_max)
        self._response = r

    def value(self, x: Vector2) -> float:
        value, penalty, feasible = self.evaluate_candidate(x)
        if self.penalty_type == PenaltyType.BIG_DISCONTINUOUS:
            if feasible:
                return value
            return self.penalty_coeff + penalty
        if self.penalty_type == PenaltyType.BIG_PROPORTIONAL:
            return value + self.penalty_coeff * penalty
        return value

    def evaluate_candidate(
        self, x: Vector2, *, weighted_penalty: bool = True
    ) -> Tuple[float, float, bool]:
        self._calculate_theta(x)
        vec = self._evaluate_state_kernel(1 if weighted_penalty else 0)
        feasible = bool(vec[0])
        penalty = float(vec[1])
        value = float(vec[2])
        return value, penalty, feasible

    def outer_grade(self) -> float:
        return self._penalty_grade(weighted=False)

    def gradient(self, x: Vector2) -> Vector2:
        self._calculate_theta(x)
        self._compute_gradient_kernel()
        base = self._gradient_base_field[None]
        penalty_vec = self._penalty_gradient_field[None]
        grad = Vector2(float(base[0]), float(base[1]))

        if self.penalty_type == PenaltyType.NONE:
            return grad

        penalty = Vector2(float(penalty_vec[0]), float(penalty_vec[1]))
        if self.penalty_type == PenaltyType.BIG_DISCONTINUOUS:
            return penalty
        if self.penalty_type == PenaltyType.BIG_PROPORTIONAL:
            return grad + penalty * self.penalty_coeff
        return grad

    def theta4_boundary(self) -> Tuple[float, float, bool]:
        p = self.parameter
        r = self.request
        A = p.L52
        B = p.L51
        C0 = -r.R33 * p.L82 + r.R31 * p.L81 - p.L3 + r.pz
        C1 = C0 - p.t3_max
        C2 = C0 - p.t3_min
        AA_BB = A * A + B * B
        sqrt_AA_BB = math.sqrt(AA_BB)
        Cmax = sqrt_AA_BB
        Cmin = -sqrt_AA_BB
        X_Cmax = A / sqrt_AA_BB
        X_Cmin = -A / sqrt_AA_BB

        def x_from_c(C: float) -> float:
            AA_BB_CC = max(0.0, AA_BB - C * C)
            return (A * C + B * math.sqrt(AA_BB_CC)) / AA_BB

        try:
            Xmin, Xmax = self._determine_x_bounds(
                C1,
                C2,
                Cmin,
                Cmax,
                A,
                x_from_c,
                X_Cmin,
                X_Cmax,
            )
        except ValueError:
            return 0.0, 0.0, False

        Xmin = max(-1.0, min(1.0, Xmin))
        Xmax = max(-1.0, min(1.0, Xmax))
        lower = max(-math.acos(Xmin), p.t4_min)
        upper = min(-math.acos(Xmax), p.t4_max)
        if lower > upper:
            lower, upper = upper, lower
        return lower, upper, True

    def _determine_x_bounds(
        self,
        C1: float,
        C2: float,
        Cmin: float,
        Cmax: float,
        A: float,
        x_from_c: Callable[[float], float],
        X_Cmin: float,
        X_Cmax: float,
    ) -> Tuple[float, float]:
        if C1 < Cmin:
            if C2 < Cmin:
                raise ValueError("No feasible theta4")
            if C2 <= A:
                return X_Cmin, x_from_c(C2)
            return X_Cmin, 1.0
        if C1 <= A:
            if C2 <= A:
                return x_from_c(C1), x_from_c(C2)
            if C2 <= Cmax:
                return min(x_from_c(C1), x_from_c(C2)), 1.0
            return min(x_from_c(C1), X_Cmax), 1.0
        if C1 <= Cmax:
            if C2 <= Cmax:
                return x_from_c(C2), x_from_c(C1)
            return X_Cmax, x_from_c(C1)
        raise ValueError("No feasible theta4")

    def _calculate_theta(self, x: Vector2) -> None:
        self._calculate_theta_kernel(x.v1, x.v2)
        self._response = self._response_field[None]
        self._t6_use_plus = bool(self._t6_use_plus_field[None])

    def _value_without_penalty(self) -> float:
        state = self._evaluate_state_kernel(0)
        return float(state[2])

    def _penalty_grade(self, *, weighted: bool) -> float:
        state = self._evaluate_state_kernel(1 if weighted else 0)
        return float(state[1])

    @ti.kernel
    def _calculate_theta_kernel(self, t2: TI_FLOAT, t4: TI_FLOAT):
        resp = RobotFunction2Response(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        req = self._request_field[None]
        param = self._parameter_field[None]
        resp, partial_t5, partial_t6, partial_t7, use_plus = _ti_calculate_theta(
            t2,
            t4,
            req,
            param,
            resp,
        )
        self._response_field[None] = resp
        self._partial_t5_field[None] = partial_t5
        self._partial_t6_field[None] = partial_t6
        self._partial_t7_field[None] = partial_t7
        self._t6_use_plus_field[None] = use_plus

    @ti.kernel
    def _evaluate_state_kernel(
        self, weighted_penalty: ti.i32
    ) -> ti.types.vector(3, TI_FLOAT):
        resp = self._response_field[None]
        req = self._request_field[None]
        param = self._parameter_field[None]
        feasible = ti.cast(_ti_is_feasible(param, resp), TI_FLOAT)
        penalty = _ti_penalty_grade(
            req,
            param,
            resp,
            weighted_penalty,
        )
        objective = _ti_weighted_squared_error(req, resp)
        return ti.Vector([feasible, penalty, objective])

    @ti.kernel
    def _compute_gradient_kernel(self):
        resp = self._response_field[None]
        req = self._request_field[None]
        param = self._parameter_field[None]
        base = _ti_base_gradient(
            req,
            param,
            resp,
            self._partial_t5_field[None],
            self._partial_t6_field[None],
            self._partial_t7_field[None],
        )
        penalty = _ti_penalty_gradient_vec(
            req,
            param,
            resp,
            self._partial_t5_field[None],
            self._partial_t6_field[None],
            self._partial_t7_field[None],
        )
        self._gradient_base_field[None] = base
        self._penalty_gradient_field[None] = penalty


class BiGoldenSectionLineSearch:
    def __init__(self, max_iteration: int = 100, epsilon: float = 1e-4):
        self.max_iteration = max_iteration
        self.epsilon = epsilon
        self.result = OptResult.FAIL
        self.iteration = 0
        self.solution = 0.0
        self.value = 0.0

    def search(self, func: Callable[[float], float], step: float) -> OptResult:
        self.result = OptResult.FAIL
        self.iteration = 0
        self.solution = 0.0
        self.value = 0.0

        # Positive direction search (0, +inf)
        pos_search = UniGoldenSectionLineSearch(self.max_iteration, self.epsilon)
        pos_result = pos_search.search(func, step)

        # Record if succeeded / max-iteration (C++ records both, ignores FAIL)
        if pos_result != OptResult.FAIL:
            self.result = pos_result
            self.iteration = pos_search.iteration
            self.solution = pos_search.solution
            self.value = pos_search.value

        # Negative direction: search on reversed function g(u)=f(-u) over (0,+inf)
        def reversed_func(u: float) -> float:
            return func(-u)

        neg_search = UniGoldenSectionLineSearch(self.max_iteration, self.epsilon)
        neg_result = neg_search.search(reversed_func, step)

        # If negative fails, keep positive result (could be FAIL)
        if neg_result == OptResult.FAIL:
            return self.result

        # Choose negative if positive failed or negative is better
        if self.result == OptResult.FAIL or neg_search.value < self.value:
            self.result = neg_result
            self.iteration = neg_search.iteration
            self.solution = -neg_search.solution
            self.value = neg_search.value

        return self.result


class HookeAndJeevesMethod:
    def __init__(self, max_iteration: int = 100, epsilon: float = 1e-4):
        self.max_iteration = max_iteration
        self.epsilon = epsilon
        self.result = OptResult.FAIL
        self.iteration = 0
        self.solution = Vector2(0.0, 0.0)

    def search(
        self,
        func: RobotFunction2,
        line_search: BiGoldenSectionLineSearch,
        x0: Vector2,
        step: float,
    ) -> OptResult:
        self.result = OptResult.FAIL
        self.iteration = 0
        self.solution = x0.clone()

        x = x0.clone()
        z_prev = x0.clone()

        for k in range(1, self.max_iteration + 1):
            y = self._line_search(
                func,
                line_search,
                x,
                Vector2(1.0, 0.0),
                step,
            )
            z = self._line_search(
                func,
                line_search,
                y,
                Vector2(0.0, 1.0),
                step,
            )

            d = z - z_prev
            if d.norm() <= self.epsilon:
                self.result = OptResult.SUCCESS
                self.iteration = k
                self.solution = (
                    z if func.value(z) <= func.value(z_prev) else z_prev
                )
                return self.result

            d.normalize()
            x_next = self._line_search(func, line_search, z, d, step)

            if Vector2.diff_norm(x, x_next) <= self.epsilon:
                self.result = OptResult.SUCCESS
                self.iteration = k
                self.solution = x_next if func.value(x_next) <= func.value(x) else x
                return self.result

            z_prev = z
            x = x_next

        self.result = OptResult.MAX_ITERATION
        self.iteration = self.max_iteration
        self.solution = x
        return self.result

    def _line_search(
        self,
        func: RobotFunction2,
        line_search: BiGoldenSectionLineSearch,
        origin: Vector2,
        direction: Vector2,
        step: float,
    ) -> Vector2:
        direction = direction.clone()
        if direction.norm() == 0.0:
            return origin.clone()
        direction.normalize()

        def wrapped(alpha: float) -> float:
            candidate = origin + direction * alpha
            return func.value(candidate)

        line_search.search(wrapped, step)
        alpha = line_search.solution
        return origin + direction * alpha


class RobotOptimizer:
    MAX_ITER = 500

    @staticmethod
    def optimize(
        func: RobotFunction2,
        workspace: AnalyticIKWorkspace,
        init: Vector2 | None = None,
    ) -> OptResult:
        method = HookeAndJeevesMethod(RobotOptimizer.MAX_ITER, 1e-3)
        line_search = BiGoldenSectionLineSearch(RobotOptimizer.MAX_ITER, 1e-4)
        if init is None:
            init = RobotOptimizer._find_initial_point(func, workspace)
        func.set_penalty_coeff(1e7)
        return method.search(func, line_search, init, 1.0)

    @staticmethod
    def _find_initial_point(
        func: RobotFunction2,
        workspace: AnalyticIKWorkspace,
    ) -> Vector2:
        result = Vector2(0.0, -1.0)
        lower, upper, feasible = func.theta4_boundary()
        if not feasible:
            return result

        best_feasible = None
        best_feasible_value = float("inf")
        best_infeasible = None
        best_infeasible_value = float("inf")

        t2_lower, t2_upper = -math.pi, math.pi
        for grid in range(10, 51, 10):
            candidates = _sample_candidate_grid(
                lower,
                upper,
                t2_lower,
                t2_upper,
                grid,
                workspace,
            )
            for t2, t4 in candidates:
                x = Vector2(float(t2), float(t4))
                value = func.value(x)
                if func.is_feasible_from_members():
                    if value < best_feasible_value:
                        best_feasible_value = value
                        best_feasible = x
                else:
                    if value < best_infeasible_value:
                        best_infeasible_value = value
                        best_infeasible = x
            if best_feasible is not None:
                return best_feasible
        return best_infeasible or result


@ti.func
def _ti_value_at(
    req: RobotFunction2Request,
    param: RobotFunction2Parameter,
    t2: TI_FLOAT,
    t4: TI_FLOAT,
    penalty_coeff: TI_FLOAT,
) -> TI_FLOAT:
    resp = RobotFunction2Response(
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    resp, _, _, _, _ = _ti_calculate_theta(t2, t4, req, param, resp)
    penalty = _ti_penalty_grade(req, param, resp, ti.cast(1, ti.i32))
    objective = _ti_weighted_squared_error(req, resp)
    return objective + penalty_coeff * penalty


@ti.func
def _ti_line_value(
    req: RobotFunction2Request,
    param: RobotFunction2Parameter,
    origin: ti.types.vector(2, TI_FLOAT),
    direction: ti.types.vector(2, TI_FLOAT),
    alpha: TI_FLOAT,
    penalty_coeff: TI_FLOAT,
) -> TI_FLOAT:
    x = origin + direction * alpha
    return _ti_value_at(req, param, x[0], x[1], penalty_coeff)


@ti.func
def _ti_golden_section_search(
    req: RobotFunction2Request,
    param: RobotFunction2Parameter,
    origin: ti.types.vector(2, TI_FLOAT),
    direction: ti.types.vector(2, TI_FLOAT),
    a_in: TI_FLOAT,
    b_in: TI_FLOAT,
    epsilon: TI_FLOAT,
    penalty_coeff: TI_FLOAT,
    max_iter: ti.i32,
) -> ti.types.vector(2, TI_FLOAT):
    a_k = ti.cast(a_in, TI_FLOAT)
    b_k = ti.cast(b_in, TI_FLOAT)
    f_a = _ti_line_value(req, param, origin, direction, a_k, penalty_coeff)
    f_b = _ti_line_value(req, param, origin, direction, b_k, penalty_coeff)
    alpha = ti.cast(GOLDEN_ALPHA, TI_FLOAT)

    one = ti.cast(1.0, TI_FLOAT)
    s = a_k + (one - alpha) * (b_k - a_k)
    t = a_k + alpha * (b_k - a_k)
    f_s = _ti_line_value(req, param, origin, direction, s, penalty_coeff)
    f_t = _ti_line_value(req, param, origin, direction, t, penalty_coeff)

    done = ti.cast(0, ti.i32)
    best_pos = ti.cast(0.0, TI_FLOAT)
    best_val = ti.cast(0.0, TI_FLOAT)
    for _k in range(max_iter):
        if done == 0:
            left = (f_a <= f_s and f_a <= f_t and f_a <= f_b) or (
                f_s <= f_a and f_s <= f_t and f_s <= f_b
            )
            if left:
                b_k = t
                f_b = f_t
                t = s
                f_t = f_s
                s = a_k + (one - alpha) * (b_k - a_k)
                f_s = _ti_line_value(req, param, origin, direction, s, penalty_coeff)
            else:
                a_k = s
                f_a = f_s
                s = t
                f_s = f_t
                t = a_k + alpha * (b_k - a_k)
                f_t = _ti_line_value(req, param, origin, direction, t, penalty_coeff)

            if (b_k - a_k) <= epsilon:
                done = ti.cast(1, ti.i32)
                use_a = f_a <= f_b
                best_pos = ti.select(use_a, a_k, b_k)
                best_val = ti.select(use_a, f_a, f_b)

    if done == 0:
        use_a = f_a <= f_b
        best_pos = ti.select(use_a, a_k, b_k)
        best_val = ti.select(use_a, f_a, f_b)
    return ti.Vector([best_pos, best_val])


@ti.func
def _ti_line_search(
    req: RobotFunction2Request,
    param: RobotFunction2Parameter,
    origin: ti.types.vector(2, TI_FLOAT),
    direction: ti.types.vector(2, TI_FLOAT),
    step: TI_FLOAT,
    epsilon: TI_FLOAT,
    penalty_coeff: TI_FLOAT,
    max_iter: ti.i32,
) -> ti.types.vector(2, TI_FLOAT):
    d = direction
    norm = ti.sqrt(d[0] * d[0] + d[1] * d[1])
    res = origin
    if norm > 0.0:
        d = d / norm
        pos = _ti_golden_section_search(
            req,
            param,
            origin,
            d,
            0.0,
            step,
            epsilon,
            penalty_coeff,
            max_iter,
        )
        neg = _ti_golden_section_search(
            req,
            param,
            origin,
            -d,
            0.0,
            step,
            epsilon,
            penalty_coeff,
            max_iter,
        )
        use_neg = neg[1] < pos[1]
        res = ti.select(use_neg, origin - d * neg[0], origin + d * pos[0])
    return res


@ti.kernel
def _hooke_jeeves_kernel(
    n_envs: ti.i32,
    req_field: ti.template(),
    param_field: ti.template(),
    init_field: ti.template(),
    epsilon: TI_FLOAT,
    line_epsilon: TI_FLOAT,
    step: TI_FLOAT,
    penalty_coeff: TI_FLOAT,
    max_iter: ti.i32,
    out_solution: ti.template(),
    out_result: ti.template(),
    out_iteration: ti.template(),
):
    for e in range(n_envs):
        req = req_field[e]
        param = param_field[None]
        x = init_field[e]
        z_prev = x
        result = ti.cast(0, ti.i32)
        iteration = ti.cast(0, ti.i32)
        solution = x
        done = ti.cast(0, ti.i32)

        for k in range(max_iter):
            if done == 0:
                y = _ti_line_search(
                    req,
                    param,
                    x,
                    ti.Vector([1.0, 0.0]),
                    step,
                    line_epsilon,
                    penalty_coeff,
                    max_iter,
                )
                z = _ti_line_search(
                    req,
                    param,
                    y,
                    ti.Vector([0.0, 1.0]),
                    step,
                    line_epsilon,
                    penalty_coeff,
                    max_iter,
                )

                d = z - z_prev
                d_norm = ti.sqrt(d[0] * d[0] + d[1] * d[1])
                if d_norm <= epsilon:
                        result = ti.cast(1, ti.i32)
                        iteration = ti.cast(k + 1, ti.i32)
                        f_z = _ti_value_at(req, param, z[0], z[1], penalty_coeff)
                        f_prev = _ti_value_at(req, param, z_prev[0], z_prev[1], penalty_coeff)
                        solution = ti.select(f_z <= f_prev, z, z_prev)
                        done = ti.cast(1, ti.i32)
                else:
                    d = d / d_norm
                    x_next = _ti_line_search(
                        req,
                        param,
                        z,
                        d,
                        step,
                        line_epsilon,
                        penalty_coeff,
                        max_iter,
                    )
                    dx = x[0] - x_next[0]
                    dy = x[1] - x_next[1]
                    if ti.sqrt(dx * dx + dy * dy) <= epsilon:
                        result = ti.cast(1, ti.i32)
                        iteration = ti.cast(k + 1, ti.i32)
                        f_next = _ti_value_at(req, param, x_next[0], x_next[1], penalty_coeff)
                        f_x = _ti_value_at(req, param, x[0], x[1], penalty_coeff)
                        solution = ti.select(f_next <= f_x, x_next, x)
                        done = ti.cast(1, ti.i32)
                    else:
                        z_prev = z
                        x = x_next

        if result == 0:
            result = ti.cast(2, ti.i32)
            iteration = ti.cast(max_iter, ti.i32)
            solution = x

        out_solution[e] = solution
        out_result[e] = result
        out_iteration[e] = iteration


@ti.data_oriented
class RobotOptimizerGPU:
    RESULT_FAIL = 0
    RESULT_SUCCESS = 1
    RESULT_MAX_ITER = 2

    def __init__(
        self,
        max_iteration: int = MAX_OPTIMIZER_ITER,
        epsilon: float = 1e-3,
        line_epsilon: float = 1e-4,
    ):
        self.max_iteration = int(max_iteration)
        self.epsilon = float(epsilon)
        self.line_epsilon = float(line_epsilon)
        self._capacity = 0
        self._init_field = None
        self._solution_field = None
        self._result_field = None
        self._iteration_field = None
        self._single_req_field = RobotFunction2Request.field(shape=(1,))
        self._single_param_field = RobotFunction2Parameter.field(shape=())

    def _ensure_capacity(self, n_envs: int) -> None:
        n_envs = int(n_envs)
        if n_envs <= self._capacity:
            return
        self._capacity = n_envs
        self._init_field = ti.Vector.field(2, dtype=TI_FLOAT, shape=(n_envs,))
        self._solution_field = ti.Vector.field(2, dtype=TI_FLOAT, shape=(n_envs,))
        self._result_field = ti.field(dtype=ti.i32, shape=(n_envs,))
        self._iteration_field = ti.field(dtype=ti.i32, shape=(n_envs,))

    def optimize_single(
        self,
        request: RobotFunction2Request,
        parameter: RobotFunction2Parameter,
        init: Vector2,
        *,
        step: float = 1.0,
        penalty_coeff: float = 1e7,
    ) -> Tuple[OptResult, Vector2]:
        self._ensure_capacity(1)
        self._single_req_field[0] = request
        self._single_param_field[None] = parameter
        init_t = torch.tensor(
            [[float(init.v1), float(init.v2)]],
            device=_torch_device(),
            dtype=_torch_dtype(),
        )
        assert self._init_field is not None
        self._init_field.from_torch(init_t)
        assert self._solution_field is not None
        assert self._result_field is not None
        assert self._iteration_field is not None
        _hooke_jeeves_kernel(
            1,
            self._single_req_field,
            self._single_param_field,
            self._init_field,
            float(self.epsilon),
            float(self.line_epsilon),
            float(step),
            float(penalty_coeff),
            int(self.max_iteration),
            self._solution_field,
            self._result_field,
            self._iteration_field,
        )
        solution = ti_to_torch(self._solution_field, copy=True)[0]
        result = int(ti_to_torch(self._result_field, copy=True).item())
        if result == self.RESULT_SUCCESS:
            status = OptResult.SUCCESS
        elif result == self.RESULT_MAX_ITER:
            status = OptResult.MAX_ITERATION
        else:
            status = OptResult.FAIL
        return status, Vector2(float(solution[0].item()), float(solution[1].item()))

    def optimize_batch(
        self,
        req_field,
        param_field,
        init: Sequence[Vector2],
        *,
        step: float = 1.0,
        penalty_coeff: float = 1e7,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_envs = len(init)
        self._ensure_capacity(n_envs)
        init_t = torch.zeros((n_envs, 2), device=_torch_device(), dtype=_torch_dtype())
        for i, item in enumerate(init):
            init_t[i, 0] = float(item.v1)
            init_t[i, 1] = float(item.v2)
        assert self._init_field is not None
        self._init_field.from_torch(init_t)
        assert self._solution_field is not None
        assert self._result_field is not None
        assert self._iteration_field is not None
        _hooke_jeeves_kernel(
            int(n_envs),
            req_field,
            param_field,
            self._init_field,
            float(self.epsilon),
            float(self.line_epsilon),
            float(step),
            float(penalty_coeff),
            int(self.max_iteration),
            self._solution_field,
            self._result_field,
            self._iteration_field,
        )
        return ti_to_torch(self._solution_field, copy=True), ti_to_torch(self._result_field, copy=True)

    def optimize_batch_tensors(
        self,
        req_field,
        param_field,
        init_t: torch.Tensor,
        *,
        step: float = 1.0,
        penalty_coeff: float = 1e7,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_envs = int(init_t.shape[0])
        self._ensure_capacity(n_envs)
        assert self._init_field is not None
        assert self._solution_field is not None
        assert self._result_field is not None
        assert self._iteration_field is not None
        self._init_field.from_torch(init_t)
        _hooke_jeeves_kernel(
            int(n_envs),
            req_field,
            param_field,
            self._init_field,
            float(self.epsilon),
            float(self.line_epsilon),
            float(step),
            float(penalty_coeff),
            int(self.max_iteration),
            self._solution_field,
            self._result_field,
            self._iteration_field,
        )
        return ti_to_torch(self._solution_field, copy=True), ti_to_torch(self._result_field, copy=True)


@ti.kernel
def _batch_eval_initial_candidates_kernel(
    n_envs: ti.i32,
    n_candidates: ti.i32,
    valid_env: ti.types.ndarray(),
    t2: ti.types.ndarray(),
    t4: ti.types.ndarray(),
    req_field: ti.template(),
    param_field: ti.template(),
    penalty_coeff: TI_FLOAT,
    out_value: ti.types.ndarray(),
    out_feasible: ti.types.ndarray(),
):
    for e, k in ti.ndrange(n_envs, n_candidates):
        if valid_env[e] == 0:
            out_value[e, k] = 1e30
            out_feasible[e, k] = 0
        else:
            req = req_field[e]
            param = param_field[None]
            resp = RobotFunction2Response(
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
            resp, _, _, _, _ = _ti_calculate_theta(
                ti.cast(t2[e, k], TI_FLOAT),
                ti.cast(t4[e, k], TI_FLOAT),
                req,
                param,
                resp,
            )
            feasible = _ti_is_feasible(param, resp)
            penalty = _ti_penalty_grade(req, param, resp, ti.cast(1, ti.i32))
            objective = _ti_weighted_squared_error(req, resp)
            out_value[e, k] = objective + penalty_coeff * penalty
            out_feasible[e, k] = feasible


@ti.kernel
def _batch_eval_initial_candidates_shared_grid_kernel(
    n_envs: ti.i32,
    n_candidates: ti.i32,
    valid_env: ti.types.ndarray(),
    t2: ti.types.ndarray(),
    t4: ti.types.ndarray(),
    req_field: ti.template(),
    param_field: ti.template(),
    penalty_coeff: TI_FLOAT,
    out_value: ti.types.ndarray(),
    out_feasible: ti.types.ndarray(),
):
    for e, k in ti.ndrange(n_envs, n_candidates):
        if valid_env[e] == 0:
            out_value[e, k] = 1e30
            out_feasible[e, k] = 0
        else:
            req = req_field[e]
            param = param_field[None]
            resp = RobotFunction2Response(
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
            resp, _, _, _, _ = _ti_calculate_theta(
                ti.cast(t2[k], TI_FLOAT),
                ti.cast(t4[k], TI_FLOAT),
                req,
                param,
                resp,
            )
            feasible = _ti_is_feasible(param, resp)
            penalty = _ti_penalty_grade(req, param, resp, ti.cast(1, ti.i32))
            objective = _ti_weighted_squared_error(req, resp)
            out_value[e, k] = objective + penalty_coeff * penalty
            out_feasible[e, k] = feasible


@ti.func
def _ti_x_from_c(A: TI_FLOAT, B: TI_FLOAT, AA_BB: TI_FLOAT, C: TI_FLOAT) -> TI_FLOAT:
    AA_BB_CC = ti.max(0.0, AA_BB - C * C)
    return (A * C + B * ti.sqrt(AA_BB_CC)) / AA_BB


@ti.func
def _ti_determine_x_bounds(
    C1: TI_FLOAT,
    C2: TI_FLOAT,
    Cmin: TI_FLOAT,
    Cmax: TI_FLOAT,
    A: TI_FLOAT,
    B: TI_FLOAT,
    X_Cmin: TI_FLOAT,
    X_Cmax: TI_FLOAT,
) -> ti.types.vector(3, TI_FLOAT):
    feasible = 1.0
    Xmin = 0.0
    Xmax = 0.0
    if C1 < Cmin:
        if C2 < Cmin:
            feasible = 0.0
        elif C2 <= A:
            Xmin = X_Cmin
            Xmax = _ti_x_from_c(A, B, A * A + B * B, C2)
        else:
            Xmin = X_Cmin
            Xmax = 1.0
    elif C1 <= A:
        if C2 <= A:
            Xmin = _ti_x_from_c(A, B, A * A + B * B, C1)
            Xmax = _ti_x_from_c(A, B, A * A + B * B, C2)
        elif C2 <= Cmax:
            Xmin = ti.min(
                _ti_x_from_c(A, B, A * A + B * B, C1),
                _ti_x_from_c(A, B, A * A + B * B, C2),
            )
            Xmax = 1.0
        else:
            Xmin = ti.min(
                _ti_x_from_c(A, B, A * A + B * B, C1),
                X_Cmax,
            )
            Xmax = 1.0
    elif C1 <= Cmax:
        if C2 <= Cmax:
            Xmin = _ti_x_from_c(A, B, A * A + B * B, C2)
            Xmax = _ti_x_from_c(A, B, A * A + B * B, C1)
        else:
            Xmin = X_Cmax
            Xmax = _ti_x_from_c(A, B, A * A + B * B, C1)
    else:
        feasible = 0.0
    return ti.Vector([Xmin, Xmax, feasible], dt=TI_FLOAT)


@ti.kernel
def _batch_build_req_field_kernel(
    n_envs: ti.i32,
    ref_origin_to_end: ti.types.ndarray(dtype=TI_FLOAT, ndim=3),
    origin_to_base: ti.types.ndarray(dtype=TI_FLOAT, ndim=3),
    weight: ti.types.ndarray(dtype=TI_FLOAT, ndim=1),
    init_angles: ti.types.ndarray(dtype=TI_FLOAT, ndim=2),
    req_field: ti.template(),
):
    for e in range(n_envs):
        R11 = ref_origin_to_end[e, 0, 0]
        R12 = ref_origin_to_end[e, 0, 1]
        R13 = ref_origin_to_end[e, 0, 2]
        px = ref_origin_to_end[e, 0, 3]
        R21 = ref_origin_to_end[e, 1, 0]
        R22 = ref_origin_to_end[e, 1, 1]
        R23 = ref_origin_to_end[e, 1, 2]
        py = ref_origin_to_end[e, 1, 3]
        R31 = ref_origin_to_end[e, 2, 0]
        R32 = ref_origin_to_end[e, 2, 1]
        R33 = ref_origin_to_end[e, 2, 2]
        pz = ref_origin_to_end[e, 2, 3]
        base_yaw = tm.atan2(origin_to_base[e, 1, 0], origin_to_base[e, 0, 0])
        req_field[e] = RobotFunction2Request(
            R11=R11,
            R12=R12,
            R13=R13,
            px=px,
            R21=R21,
            R22=R22,
            R23=R23,
            py=py,
            R31=R31,
            R32=R32,
            R33=R33,
            pz=pz,
            w0=weight[5],
            w1=weight[6],
            w2=weight[7],
            w3=weight[0],
            w4=weight[1],
            w5=weight[2],
            w6=weight[3],
            w7=weight[4],
            r0=origin_to_base[e, 0, 3],
            r1=origin_to_base[e, 1, 3],
            r2=base_yaw,
            r3=init_angles[e, 0],
            r4=init_angles[e, 1],
            r5=init_angles[e, 2],
            r6=init_angles[e, 3],
            r7=init_angles[e, 4],
        )


@ti.kernel
def _batch_theta4_boundary_kernel(
    n_envs: ti.i32,
    req_field: ti.template(),
    param_field: ti.template(),
    lower_arr: ti.types.ndarray(dtype=TI_FLOAT, ndim=1),
    upper_arr: ti.types.ndarray(dtype=TI_FLOAT, ndim=1),
    valid_env: ti.types.ndarray(dtype=ti.i32, ndim=1),
):
    for e in range(n_envs):
        req = req_field[e]
        param = param_field[None]
        A = param.L52
        B = param.L51
        C0 = -req.R33 * param.L82 + req.R31 * param.L81 - param.L3 + req.pz
        C1 = C0 - param.t3_max
        C2 = C0 - param.t3_min
        AA_BB = A * A + B * B
        sqrt_AA_BB = ti.sqrt(AA_BB)
        Cmax = sqrt_AA_BB
        Cmin = -sqrt_AA_BB
        X_Cmax = A / sqrt_AA_BB
        X_Cmin = -A / sqrt_AA_BB
        bounds = _ti_determine_x_bounds(C1, C2, Cmin, Cmax, A, B, X_Cmin, X_Cmax)
        Xmin = bounds[0]
        Xmax = bounds[1]
        feasible = bounds[2]
        if feasible < 0.5:
            lower_arr[e] = 0.0
            upper_arr[e] = 0.0
            valid_env[e] = 0
        else:
            Xmin = ti.max(-1.0, ti.min(1.0, Xmin))
            Xmax = ti.max(-1.0, ti.min(1.0, Xmax))
            lower = ti.max(-tm.acos(Xmin), param.t4_min)
            upper = ti.min(-tm.acos(Xmax), param.t4_max)
            if lower > upper:
                tmp = lower
                lower = upper
                upper = tmp
            lower_arr[e] = lower
            upper_arr[e] = upper
            valid_env[e] = 1


@ti.kernel
def _batch_finalize_kernel(
    n_envs: ti.i32,
    req_field: ti.template(),
    param_field: ti.template(),
    solutions: ti.types.ndarray(dtype=TI_FLOAT, ndim=2),
    opt_results: ti.types.ndarray(dtype=ti.i32, ndim=1),
    result_out: ti.template(),
    sol_out: ti.template(),
    o2b_out: ti.template(),
    o2e_out: ti.template(),
):
    for e in range(n_envs):
        if opt_results[e] == 0:
            result_out[e] = 0
            for j in ti.static(range(5)):
                sol_out[e, j] = 0.0
            for i in ti.static(range(4)):
                for j in ti.static(range(4)):
                    o2b_out[e, i, j] = 0.0
                    o2e_out[e, i, j] = 0.0
            continue

        req = req_field[e]
        param = param_field[None]
        resp = RobotFunction2Response(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        resp, _, _, _, _ = _ti_calculate_theta(
            ti.cast(solutions[e, 0], TI_FLOAT),
            ti.cast(solutions[e, 1], TI_FLOAT),
            req,
            param,
            resp,
        )
        penalty = _ti_penalty_grade(req, param, resp, ti.cast(0, ti.i32))
        if penalty > 1e-6:
            result_out[e] = 0
        else:
            result_out[e] = 1

        sol_out[e, 0] = resp.t3
        sol_out[e, 1] = resp.t4
        sol_out[e, 2] = resp.t5
        sol_out[e, 3] = resp.t6
        sol_out[e, 4] = resp.t7

        T0 = _ti_translation(resp.t0, 0.0, 0.0)
        T1 = _ti_translation(0.0, resp.t1, 0.0)
        T2 = _ti_rot_z(resp.t2)
        T3 = _ti_translation(0.0, 0.0, resp.t3 + param.L3)
        T4 = _ti_translation(param.L41, param.L42, 0.0) @ _ti_rot_y(-resp.t4)
        T5 = _ti_translation(param.L51, 0.0, param.L52) @ _ti_rot_z(resp.t5)
        T6 = _ti_rot_y(-resp.t6)
        T7 = _ti_rot_z(resp.t7)

        origin_to_base = T0 @ T1 @ T2
        origin_to_end = origin_to_base @ T3 @ T4 @ T5 @ T6 @ T7
        for i in ti.static(range(4)):
            for j in ti.static(range(4)):
                o2b_out[e, i, j] = origin_to_base[i, j]
                o2e_out[e, i, j] = origin_to_end[i, j]


@ti.kernel
def _batch_select_base_yaw_kernel(
    n_envs: ti.i32,
    origin_to_base: ti.types.ndarray(dtype=TI_FLOAT, ndim=3),
    init_angles: ti.types.ndarray(dtype=TI_FLOAT, ndim=2),
    weight: ti.types.ndarray(dtype=TI_FLOAT, ndim=1),
    solution_count: ti.template(),
    solutions: ti.template(),
    param: RobotFunction2Parameter,
    result_out: ti.template(),
    sol_out: ti.template(),
    o2b_out: ti.template(),
    o2e_out: ti.template(),
):
    for e in range(n_envs):
        count = solution_count[e]
        if count <= 0:
            result_out[e] = 0
            for j in ti.static(range(5)):
                sol_out[e, j] = 0.0
            for i in ti.static(range(4)):
                for j in ti.static(range(4)):
                    o2b_out[e, i, j] = 0.0
                    o2e_out[e, i, j] = 0.0
            continue

        current_base = ti.Vector(
            [
                origin_to_base[e, 0, 3],
                origin_to_base[e, 1, 3],
                tm.atan2(origin_to_base[e, 1, 0], origin_to_base[e, 0, 0]),
            ]
        )
        min_distance = 1e18
        min_index = 0
        for i in range(count):
            distance = 0.0
            for j in ti.static(range(5)):
                diff = ti.abs(init_angles[e, j] - solutions[e, i][j + 3])
                distance += diff * weight[j]
            for k in ti.static(range(3)):
                diff = ti.abs(current_base[k] - solutions[e, i][k])
                distance += diff * weight[5 + k]
            if distance < min_distance:
                min_distance = distance
                min_index = i

        res = solutions[e, min_index]
        result_out[e] = 1
        sol_out[e, 0] = res[3]
        sol_out[e, 1] = res[4]
        sol_out[e, 2] = res[5]
        sol_out[e, 3] = res[6]
        sol_out[e, 4] = res[7]

        T0 = _ti_translation(res[0], 0.0, 0.0)
        T1 = _ti_translation(0.0, res[1], 0.0)
        T2 = _ti_rot_z(res[2])
        T3 = _ti_translation(0.0, 0.0, res[3] + param.L3)
        T4 = _ti_translation(param.L41, param.L42, 0.0) @ _ti_rot_y(-res[4])
        T5 = _ti_translation(param.L51, 0.0, param.L52) @ _ti_rot_z(res[5])
        T6 = _ti_rot_y(-res[6])
        T7 = _ti_rot_z(res[7])

        origin_to_base_mat = T0 @ T1 @ T2
        origin_to_end_mat = origin_to_base_mat @ T3 @ T4 @ T5 @ T6 @ T7
        for i in ti.static(range(4)):
            for j in ti.static(range(4)):
                o2b_out[e, i, j] = origin_to_base_mat[i, j]
                o2e_out[e, i, j] = origin_to_end_mat[i, j]


def _sample_candidate_grid_torch(
    *,
    lower: float,
    upper: float,
    t2_lower: float,
    t2_upper: float,
    grid: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    grid = max(1, min(int(grid), MAX_GRID_RESOLUTION - 2))
    dim = max(3, grid + 2)
    t2_step = (float(t2_upper) - float(t2_lower)) / float(dim)
    t4_step = (float(upper) - float(lower)) / float(dim)

    idx = torch.arange(1, dim - 1, device=device, dtype=dtype)
    t2_vals = float(t2_lower) + idx * t2_step
    t4_vals = float(lower) + idx * t4_step
    grid_t2, grid_t4 = torch.meshgrid(t2_vals, t4_vals, indexing="ij")
    return grid_t2.reshape(-1), grid_t4.reshape(-1)


JOINT_ORDER = [
    "arm_lift_joint",
    "arm_flex_joint",
    "arm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
]


def _unit_vector(axis: int) -> torch.Tensor:
    vec = torch.zeros(3, device=_torch_device(), dtype=_torch_dtype())
    vec[axis] = 1.0
    return vec


def suit_base_movement(request: IKRequest) -> bool:
    if (
        len(request.linear_base_movements) != 2
        or len(request.rotational_base_movements) != 1
    ):
        return False
    return (
        torch.allclose(_as_torch(request.linear_base_movements[0]), _unit_vector(0))
        and torch.allclose(_as_torch(request.linear_base_movements[1]), _unit_vector(1))
        and torch.allclose(_as_torch(request.rotational_base_movements[0]), _unit_vector(2))
    )


def suit_base_rotation_z(request: IKRequest) -> bool:
    if (
        len(request.linear_base_movements) != 0
        or len(request.rotational_base_movements) != 1
    ):
        return False
    return torch.allclose(_as_torch(request.rotational_base_movements[0]), _unit_vector(2))


def suit_use_joint(request: IKRequest) -> bool:
    return all(joint in request.use_joints for joint in JOINT_ORDER)


def suit_frame(request: IKRequest) -> bool:
    return request.frame_name in (
        "hand_palm_link",
        "hand_palm_joint",
    )


def map_joint_and_id(use_joints: Sequence[str]) -> dict:
    mapping = {}
    for joint in JOINT_ORDER:
        if joint not in use_joints:
            raise ValueError(f"Missing joint {joint}")
        mapping[joint] = use_joints.index(joint)
    return mapping


def extract_joint_position(
    joint_state: JointState,
    joint_name: str,
    default_pos: float,
) -> float:
    if len(joint_state.name) != len(joint_state.position):
        return default_pos
    for idx, name in enumerate(joint_state.name):
        if name == joint_name:
            return float(joint_state.position[idx])
    return default_pos


def theta_representation_change(theta: float) -> float:
    if theta > math.pi:
        return -2.0 * math.pi + theta
    if theta < -math.pi:
        return theta + 2.0 * math.pi
    return theta


def trigonometric_composition_formula(
    A2o: float,
    B2o: float,
    C2o: float,
    D2o: float,
) -> Tuple[bool, float, float]:
    if abs(C2o) > D2o:
        return False, 0.0, 0.0
    alpha2o = math.atan2(B2o, A2o)
    AS1 = math.asin(C2o / D2o)
    AS2 = math.pi - AS1 if AS1 >= 0 else -math.pi - AS1
    theta21 = theta_representation_change(AS1 - alpha2o)
    theta22 = theta_representation_change(AS2 - alpha2o)
    return True, theta21, theta22


def theta_within_limit(
    max_in: float,
    min_in: float,
    epsilon: float,
    theta: float,
) -> float:
    if theta > max_in and theta <= max_in + epsilon:
        return max_in
    if theta < min_in and theta >= min_in - epsilon:
        return min_in
    return theta


def judge_theta_pi_pi(
    max_in: float,
    min_in: float,
    epsilon: float,
    theta: float,
) -> Tuple[bool, float]:
    max_limit = max_in + epsilon
    min_limit = min_in - epsilon
    if min_limit <= theta <= max_limit:
        return True, theta_within_limit(max_in, min_in, epsilon, theta)

    if max_limit > math.pi:
        if -math.pi <= theta <= -2.0 * math.pi + max_limit:
            theta = theta + 2.0 * math.pi
            return True, theta_within_limit(max_in, min_in, epsilon, theta)
    elif min_limit < -math.pi:
        if 2.0 * math.pi + min_limit <= theta <= math.pi:
            theta = theta - 2.0 * math.pi
            return True, theta_within_limit(max_in, min_in, epsilon, theta)
    return False, theta


def _translation(x: float, y: float, z: float) -> torch.Tensor:
    mat = torch.eye(4, device=_torch_device(), dtype=_torch_dtype())
    mat[:3, 3] = torch.tensor([x, y, z], device=mat.device, dtype=mat.dtype)
    return mat


def _rot_z(theta: float) -> torch.Tensor:
    c = math.cos(theta)
    s = math.sin(theta)
    return torch.tensor(
        [
            [c, -s, 0.0, 0.0],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=_torch_device(),
        dtype=_torch_dtype(),
    )


def _rot_y(theta: float) -> torch.Tensor:
    c = math.cos(theta)
    s = math.sin(theta)
    return torch.tensor(
        [
            [c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=_torch_device(),
        dtype=_torch_dtype(),
    )


@ti.func
def _ti_translation(x: TI_FLOAT, y: TI_FLOAT, z: TI_FLOAT):
    return ti.Matrix(
        [
            [1.0, 0.0, 0.0, x],
            [0.0, 1.0, 0.0, y],
            [0.0, 0.0, 1.0, z],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


@ti.func
def _ti_rot_z(theta: TI_FLOAT):
    c = tm.cos(theta)
    s = tm.sin(theta)
    return ti.Matrix(
        [
            [c, -s, 0.0, 0.0],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


@ti.func
def _ti_rot_y(theta: TI_FLOAT):
    c = tm.cos(theta)
    s = tm.sin(theta)
    return ti.Matrix(
        [
            [c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


@ti.kernel
def _fk_from_solution_kernel(
    param: RobotFunction2Parameter,
    t0: TI_FLOAT,
    t1: TI_FLOAT,
    t2: TI_FLOAT,
    t3: TI_FLOAT,
    t4: TI_FLOAT,
    t5: TI_FLOAT,
    t6: TI_FLOAT,
    t7: TI_FLOAT,
    origin_to_base_field: ti.template(),
    origin_to_end_field: ti.template(),
):
    T0 = _ti_translation(t0, 0.0, 0.0)
    T1 = _ti_translation(0.0, t1, 0.0)
    T2 = _ti_rot_z(t2)
    T3 = _ti_translation(0.0, 0.0, t3 + param.L3)
    T4 = _ti_translation(param.L41, param.L42, 0.0) @ _ti_rot_y(-t4)
    T5 = _ti_translation(param.L51, 0.0, param.L52) @ _ti_rot_z(t5)
    T6 = _ti_rot_y(-t6)
    T7 = _ti_rot_z(t7)
    T8 = _ti_translation(param.L81, 0.0, param.L82) @ _ti_rot_z(tm.pi)

    origin_to_base = T0 @ T1 @ T2
    origin_to_end = origin_to_base @ T3 @ T4 @ T5 @ T6 @ T7 @ T8

    origin_to_base_field[None] = origin_to_base
    origin_to_end_field[None] = origin_to_end


@ti.kernel
def _base_position_range_kernel(
    param: RobotFunction2Parameter,
    origin_to_hand: ti.types.ndarray(dtype=TI_FLOAT, ndim=2),
    base_range_center_field: ti.template(),
    base_range_radius_min_field: ti.template(),
    base_range_radius_max_field: ti.template(),
):
    zwo = (
        origin_to_hand[2, 0] * param.L81
        - origin_to_hand[2, 2] * param.L82
        + origin_to_hand[2, 3]
    )
    invalid = (
        (zwo > param.t3_max + param.L3 + param.L52)
        or (
            zwo
            < param.t3_min
            + param.L3
            + param.L52 * tm.cos(param.t4_min)
            + param.L51 * tm.sin(param.t4_min)
        )
    )
    if invalid:
        base_range_center_field[None] = ti.Vector([0.0, 0.0])
        base_range_radius_min_field[None] = -1.0
        base_range_radius_max_field[None] = -1.0
    else:
        radius_min = ti.cast(0.0, TI_FLOAT)
        radius_max = ti.cast(0.0, TI_FLOAT)

        center_x = (
            origin_to_hand[0, 0] * param.L81
            - origin_to_hand[0, 2] * param.L82
            + origin_to_hand[0, 3]
        )
        center_y = (
            origin_to_hand[1, 0] * param.L81
            - origin_to_hand[1, 2] * param.L82
            + origin_to_hand[1, 3]
        )
        l5_length = tm.sqrt(param.L51 * param.L51 + param.L52 * param.L52)
        alpha_t4 = tm.atan2(param.L52, param.L51)

        if zwo < param.t3_min + param.L3:
            t4 = (
                tm.asin((zwo - param.t3_min - param.L3) / l5_length)
                - alpha_t4
            )
            S4 = tm.sin(t4)
            C4 = tm.cos(t4)
            radius_max = tm.sqrt(
                (param.L52 * S4 - param.L51 * C4 - param.L41) ** 2.0
                + param.L42 * param.L42
            )
        elif zwo > param.t3_max + param.L3:
            t4 = (
                tm.asin((zwo - param.t3_max - param.L3) / l5_length)
                - alpha_t4
            )
            S4 = tm.sin(t4)
            C4 = tm.cos(t4)
            radius_max = tm.sqrt(
                (param.L52 * S4 - param.L51 * C4 - param.L41) ** 2.0
                + param.L42 * param.L42
            )
        else:
            S4 = tm.sin(-alpha_t4)
            C4 = tm.cos(-alpha_t4)
            radius_max = tm.sqrt(
                (param.L52 * S4 - param.L51 * C4 - param.L41) ** 2.0
                + param.L42 * param.L42
            )

        t4_min_rev = -param.t4_min - alpha_t4
        if zwo >= param.t3_min + param.L3 + param.L52:
            radius_min = tm.sqrt(
                (-param.L51 - param.L41) ** 2.0 + param.L42 * param.L42
            )
        elif zwo > (
            param.t3_min
            + param.L3
            + param.L52 * tm.cos(t4_min_rev)
            + param.L51 * tm.sin(t4_min_rev)
        ):
            t4 = (
                tm.asin((zwo - param.t3_min - param.L3) / l5_length)
                - alpha_t4
            )
            S4 = tm.sin(t4)
            C4 = tm.cos(t4)
            radius_min = tm.sqrt(
                (param.L52 * S4 - param.L51 * C4 - param.L41) ** 2.0
                + param.L42 * param.L42
            )
        else:
            S4 = tm.sin(param.t4_min)
            C4 = tm.cos(param.t4_min)
            radius_min = tm.sqrt(
                (param.L52 * S4 - param.L51 * C4 - param.L41) ** 2.0
                + param.L42 * param.L42
            )

        base_range_center_field[None] = ti.Vector([center_x, center_y])
        base_range_radius_min_field[None] = radius_min
        base_range_radius_max_field[None] = radius_max


@ti.kernel
def _select_closest_solution_kernel(
    current_arm: ti.types.ndarray(dtype=TI_FLOAT, ndim=1),
    current_base: ti.types.ndarray(dtype=TI_FLOAT, ndim=1),
    response_arm: ti.types.ndarray(dtype=TI_FLOAT, ndim=2),
    response_base: ti.types.ndarray(dtype=TI_FLOAT, ndim=2),
    weight: ti.types.ndarray(dtype=TI_FLOAT, ndim=1),
    joint_count: ti.i32,
    response_count: ti.i32,
    closest_solution_index_field: ti.template(),
):
    min_distance = 1e18
    min_index = -1
    for i in range(response_count):
        distance = 0.0
        for j in range(joint_count):
            diff = ti.abs(current_arm[j] - response_arm[i, j])
            distance += diff * weight[j]
        for k in range(3):
            diff = ti.abs(current_base[k] - response_base[i, k])
            distance += diff * weight[joint_count + k]
        if distance < min_distance:
            min_distance = distance
            min_index = i
    closest_solution_index_field[None] = min_index


@ti.func
def _ti_theta_representation_change(theta: TI_FLOAT) -> TI_FLOAT:
    out = theta
    if theta > tm.pi:
        out = theta - 2.0 * tm.pi
    if theta < -tm.pi:
        out = theta + 2.0 * tm.pi
    return out


@ti.func
def _ti_trigonometric_composition_formula(
    A: TI_FLOAT,
    B: TI_FLOAT,
    C: TI_FLOAT,
    D: TI_FLOAT,
) -> ti.types.vector(3, TI_FLOAT):
    res = ti.Vector([0.0, 0.0, 0.0])
    if ti.abs(C) <= D:
        alpha = tm.atan2(B, A)
        as1 = tm.asin(C / D)
        as2 = tm.pi - as1 if as1 >= 0.0 else -tm.pi - as1
        theta21 = _ti_theta_representation_change(as1 - alpha)
        theta22 = _ti_theta_representation_change(as2 - alpha)
        res = ti.Vector([1.0, theta21, theta22])
    return res


@ti.func
def _ti_theta_within_limit(
    max_in: TI_FLOAT,
    min_in: TI_FLOAT,
    epsilon: TI_FLOAT,
    theta: TI_FLOAT,
) -> TI_FLOAT:
    out = theta
    if theta > max_in and theta <= max_in + epsilon:
        out = max_in
    if theta < min_in and theta >= min_in - epsilon:
        out = min_in
    return out


@ti.func
def _ti_judge_theta_pi_pi(
    max_in: TI_FLOAT,
    min_in: TI_FLOAT,
    epsilon: TI_FLOAT,
    theta: TI_FLOAT,
) -> ti.types.vector(2, TI_FLOAT):
    max_limit = max_in + epsilon
    min_limit = min_in - epsilon
    ok = 0.0
    value = theta
    if min_limit <= theta <= max_limit:
        ok = 1.0
        value = _ti_theta_within_limit(max_in, min_in, epsilon, theta)
    else:
        if max_limit > tm.pi:
            if -tm.pi <= theta <= -2.0 * tm.pi + max_limit:
                ok = 1.0
                theta = theta + 2.0 * tm.pi
                value = _ti_theta_within_limit(max_in, min_in, epsilon, theta)
        else:
            if min_limit < -tm.pi:
                if 2.0 * tm.pi + min_limit <= theta <= tm.pi:
                    ok = 1.0
                    theta = theta - 2.0 * tm.pi
                    value = _ti_theta_within_limit(
                        max_in,
                        min_in,
                        epsilon,
                        theta,
                    )
    return ti.Vector([ok, value])


@ti.func
def _ti_transform_inv(mat):
    r00, r01, r02 = mat[0, 0], mat[0, 1], mat[0, 2]
    r10, r11, r12 = mat[1, 0], mat[1, 1], mat[1, 2]
    r20, r21, r22 = mat[2, 0], mat[2, 1], mat[2, 2]
    tx, ty, tz = mat[0, 3], mat[1, 3], mat[2, 3]
    inv = ti.Matrix(
        [
            [r00, r10, r20, 0.0],
            [r01, r11, r21, 0.0],
            [r02, r12, r22, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    inv[0, 3] = -(r00 * tx + r10 * ty + r20 * tz)
    inv[1, 3] = -(r01 * tx + r11 * ty + r21 * tz)
    inv[2, 3] = -(r02 * tx + r12 * ty + r22 * tz)
    return inv


@ti.kernel
def _solve_base_yaw_ik_kernel(
    param: RobotFunction2Parameter,
    ref_origin_to_end: ti.types.ndarray(dtype=TI_FLOAT, ndim=2),
    theta0: TI_FLOAT,
    theta1: TI_FLOAT,
    base_yaw_solution_count: ti.template(),
    base_yaw_solutions: ti.template(),
):
    base_yaw_solution_count[None] = 0
    T78_inv = ti.Matrix(
        [
            [-1.0, 0.0, 0.0, param.L81],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -param.L82],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    UB8o = ti.Matrix(
        [
            [
                ref_origin_to_end[0, 0],
                ref_origin_to_end[0, 1],
                ref_origin_to_end[0, 2],
                ref_origin_to_end[0, 3],
            ],
            [
                ref_origin_to_end[1, 0],
                ref_origin_to_end[1, 1],
                ref_origin_to_end[1, 2],
                ref_origin_to_end[1, 3],
            ],
            [
                ref_origin_to_end[2, 0],
                ref_origin_to_end[2, 1],
                ref_origin_to_end[2, 2],
                ref_origin_to_end[2, 3],
            ],
            [
                ref_origin_to_end[3, 0],
                ref_origin_to_end[3, 1],
                ref_origin_to_end[3, 2],
                ref_origin_to_end[3, 3],
            ],
        ]
    )
    UB7o = UB8o @ T78_inv

    xwo = UB7o[0, 3]
    ywo = UB7o[1, 3]
    zwo = UB7o[2, 3]

    A2o = -theta0 + xwo
    B2o = theta1 - ywo
    C2o = -param.L42
    D2o = tm.sqrt(A2o * A2o + B2o * B2o)

    trig2 = _ti_trigonometric_composition_formula(A2o, B2o, C2o, D2o)
    if trig2[0] >= 0.5:

        eps_theta = 1e-3
        A4o = -param.L52
        B4o = param.L51
        D4o = tm.sqrt(A4o * A4o + B4o * B4o)

        for idx2 in ti.static(range(2)):
            theta2 = trig2[1] if idx2 == 0 else trig2[2]
            C4o = (
                -tm.cos(theta2) * (theta0 - xwo)
                - tm.sin(theta2) * (theta1 - ywo)
                - param.L41
            )
            trig4 = _ti_trigonometric_composition_formula(A4o, B4o, C4o, D4o)
            if trig4[0] >= 0.5:
                for idx4 in ti.static(range(2)):
                    theta4 = trig4[1] if idx4 == 0 else trig4[2]
                    judge4 = _ti_judge_theta_pi_pi(
                        param.t4_max,
                        param.t4_min,
                        eps_theta,
                        theta4,
                    )
                    if judge4[0] >= 0.5:
                        theta4_checked = judge4[1]
                        theta3 = (
                            zwo
                            - param.L3
                            - param.L52 * tm.cos(theta4_checked)
                            - param.L51 * tm.sin(theta4_checked)
                        )
                        judge3 = _ti_judge_theta_pi_pi(
                            param.t3_max,
                            param.t3_min,
                            eps_theta,
                            theta3,
                        )
                        if judge3[0] >= 0.5:
                            theta3_checked = judge3[1]

                            TBm1 = ti.Matrix(
                                [
                                    [0.0, 0.0, 1.0, 0.0],
                                    [1.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0],
                                ]
                            )
                            Tm10 = ti.Matrix(
                                [
                                    [0.0, -1.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, theta0],
                                    [0.0, 0.0, 0.0, 1.0],
                                ]
                            )
                            T01 = ti.Matrix(
                                [
                                    [0.0, -1.0, 0.0, 0.0],
                                    [0.0, 0.0, -1.0, -theta1],
                                    [1.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0],
                                ]
                            )
                            T12 = ti.Matrix(
                                [
                                    [-tm.sin(theta2), -tm.cos(theta2), 0.0, 0.0],
                                    [0.0, 0.0, -1.0, 0.0],
                                    [tm.cos(theta2), -tm.sin(theta2), 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0],
                                ]
                            )
                            T23 = ti.Matrix(
                                [
                                    [0.0, 1.0, 0.0, 0.0],
                                    [-1.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, theta3_checked + param.L3],
                                    [0.0, 0.0, 0.0, 1.0],
                                ]
                            )
                            T34 = ti.Matrix(
                                [
                                    [tm.cos(theta4_checked), -tm.sin(theta4_checked), 0.0, param.L41],
                                    [0.0, 0.0, -1.0, param.L42],
                                    [tm.sin(theta4_checked), tm.cos(theta4_checked), 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0],
                                ]
                            )

                            UB4o = TBm1 @ Tm10 @ T01 @ T12 @ T23 @ T34
                            U47o = _ti_transform_inv(UB4o) @ UB7o

                            theta61 = tm.atan2(
                                -tm.sqrt(
                                    U47o[1, 0] * U47o[1, 0]
                                    + U47o[1, 1] * U47o[1, 1]
                                ),
                                U47o[1, 2],
                            )
                            theta62 = tm.atan2(
                                tm.sqrt(
                                    U47o[1, 0] * U47o[1, 0]
                                    + U47o[1, 1] * U47o[1, 1]
                                ),
                                U47o[1, 2],
                            )

                            for idx6 in ti.static(range(2)):
                                theta6 = theta61 if idx6 == 0 else theta62
                                if ti.abs(theta6) < 1e-9:
                                    theta57a = tm.atan2(-U47o[0, 1], U47o[0, 0])
                                    theta57b = tm.atan2(-U47o[2, 0], -U47o[2, 1])
                                    if ti.abs(theta57a - theta57b) < 1e-6:
                                        theta5 = theta57a * 0.5
                                        theta7 = theta57a * 0.5
                                        idx = ti.atomic_add(
                                            base_yaw_solution_count[None],
                                            1,
                                        )
                                        if idx < ti.static(MAX_BASE_YAW_SOLUTIONS):
                                            base_yaw_solutions[idx] = ti.Vector(
                                                [
                                                    theta0,
                                                    theta1,
                                                    theta2,
                                                    theta3_checked,
                                                    theta4_checked,
                                                    theta5,
                                                    theta6,
                                                    theta7,
                                                ]
                                            )
                                else:
                                    judge6 = _ti_judge_theta_pi_pi(
                                        param.t6_max,
                                        param.t6_min,
                                        eps_theta,
                                        theta6,
                                    )
                                    if judge6[0] >= 0.5:
                                        theta6_checked = judge6[1]
                                        sign6 = ti.select(
                                            theta6_checked >= 0.0,
                                            1.0,
                                            -1.0,
                                        )
                                        theta7 = tm.atan2(
                                            -U47o[1, 1] * sign6,
                                            U47o[1, 0] * sign6,
                                        )
                                        judge7 = _ti_judge_theta_pi_pi(
                                            param.t7_max,
                                            param.t7_min,
                                            eps_theta,
                                            theta7,
                                        )
                                        if judge7[0] >= 0.5:
                                            theta7_checked = judge7[1]
                                            theta5 = tm.atan2(
                                                U47o[2, 2] * sign6,
                                                -U47o[0, 2] * sign6,
                                            )
                                            judge5 = _ti_judge_theta_pi_pi(
                                                param.t5_max,
                                                param.t5_min,
                                                eps_theta,
                                                theta5,
                                            )
                                            if judge5[0] >= 0.5:
                                                theta5_checked = judge5[1]
                                                idx = ti.atomic_add(
                                                    base_yaw_solution_count[None],
                                                    1,
                                                )
                                                if idx < ti.static(MAX_BASE_YAW_SOLUTIONS):
                                                    base_yaw_solutions[idx] = ti.Vector(
                                                        [
                                                            theta0,
                                                            theta1,
                                                            theta2,
                                                            theta3_checked,
                                                            theta4_checked,
                                                            theta5_checked,
                                                            theta6_checked,
                                                            theta7_checked,
                                                        ]
                                                    )


def base_position_range(
    robot_param: RobotFunction2Parameter,
    origin_to_hand: torch.Tensor,
    workspace: AnalyticIKWorkspace,
) -> BasePositionRange:
    if not _is_torch_tensor(origin_to_hand):
        origin_to_hand = torch.as_tensor(origin_to_hand, device=_torch_device(), dtype=_torch_dtype())
    _base_position_range_kernel(
        robot_param,
        origin_to_hand,
        workspace.base_range_center_field,
        workspace.base_range_radius_min_field,
        workspace.base_range_radius_max_field,
    )
    center = ti_to_torch(workspace.base_range_center_field, copy=True)[()]
    radius_min = float(ti_to_torch(workspace.base_range_radius_min_field, copy=True).item())
    radius_max = float(ti_to_torch(workspace.base_range_radius_max_field, copy=True).item())
    return BasePositionRange(center=center, radius_min=radius_min, radius_max=radius_max)


def _build_solution_angle(request: IKRequest, *, to_torch: bool) -> JointState:
    position = torch.zeros(
        len(request.use_joints),
        device=_torch_device(),
        dtype=_torch_dtype(),
    )
    solution_angle = JointState(
        name=list(request.use_joints),
        position=position,
    )
    for i, joint_name in enumerate(request.use_joints):
        solution_angle.position[i] = extract_joint_position(
            request.initial_angle,
            joint_name,
            0.0,
        )
    return solution_angle


def _finalize_ik_result(
    request: IKRequest,
    function_param: RobotFunction2Parameter,
    workspace: AnalyticIKWorkspace,
    func: RobotFunction2,
    joint_map: dict,
    *,
    to_torch: bool,
) -> Tuple[IKResult, JointState, torch.Tensor, torch.Tensor]:
    solution_angle = _build_solution_angle(request, to_torch=to_torch)
    origin_to_base = torch.eye(4, device=_torch_device(), dtype=_torch_dtype())
    origin_to_end = torch.eye(4, device=_torch_device(), dtype=_torch_dtype())

    outer_grade = func.outer_grade()
    if outer_grade > 1e-6:
        return IKResult.FAIL, solution_angle, origin_to_base, origin_to_end
    if outer_grade > 0.0:
        func.force_feasible()

    res = func.response
    solution_angle.position[joint_map["arm_lift_joint"]] = res.t3
    solution_angle.position[joint_map["arm_flex_joint"]] = res.t4
    solution_angle.position[joint_map["arm_roll_joint"]] = res.t5
    solution_angle.position[joint_map["wrist_flex_joint"]] = res.t6
    solution_angle.position[joint_map["wrist_roll_joint"]] = res.t7

    _fk_from_solution_kernel(
        function_param,
        res.t0,
        res.t1,
        res.t2,
        res.t3,
        res.t4,
        res.t5,
        res.t6,
        res.t7,
        workspace.origin_to_base_field,
        workspace.origin_to_end_field,
    )
    origin_to_base = _mat4_from_field(workspace.origin_to_base_field)
    origin_to_end = _mat4_from_field(workspace.origin_to_end_field)

    return IKResult.SUCCESS, solution_angle, origin_to_base, origin_to_end


def solve_ik(
    request: IKRequest,
    function_param: RobotFunction2Parameter,
    workspace: AnalyticIKWorkspace,
    func: "RobotFunction2 | None" = None,
    retry_workspace: AnalyticIKWorkspace | None = None,
    init: "Vector2 | None" = None,
    optimizer: str = "cpu",
    gpu_optimizer: "RobotOptimizerGPU | None" = None,
) -> Tuple[IKResult, JointState, torch.Tensor, torch.Tensor]:
    use_torch = True
    joint_map = map_joint_and_id(request.use_joints)

    origin_to_base = _as_torch(request.origin_to_base)
    ref_origin_to_end = _as_torch(request.ref_origin_to_end)
    base_yaw = math.atan2(
        float(origin_to_base[1, 0].item()),
        float(origin_to_base[0, 0].item()),
    )
    init_lift = extract_joint_position(
        request.initial_angle,
        "arm_lift_joint",
        0.0,
    )
    init_flex = extract_joint_position(
        request.initial_angle,
        "arm_flex_joint",
        0.0,
    )
    init_roll = extract_joint_position(
        request.initial_angle,
        "arm_roll_joint",
        0.0,
    )
    init_wrist_flex = extract_joint_position(
        request.initial_angle,
        "wrist_flex_joint",
        0.0,
    )
    init_wrist_roll = extract_joint_position(
        request.initial_angle,
        "wrist_roll_joint",
        0.0,
    )

    weight = request.weight
    function_req = RobotFunction2Request(
        R11=float(ref_origin_to_end[0, 0]),
        R12=float(ref_origin_to_end[0, 1]),
        R13=float(ref_origin_to_end[0, 2]),
        px=float(ref_origin_to_end[0, 3]),
        R21=float(ref_origin_to_end[1, 0]),
        R22=float(ref_origin_to_end[1, 1]),
        R23=float(ref_origin_to_end[1, 2]),
        py=float(ref_origin_to_end[1, 3]),
        R31=float(ref_origin_to_end[2, 0]),
        R32=float(ref_origin_to_end[2, 1]),
        R33=float(ref_origin_to_end[2, 2]),
        pz=float(ref_origin_to_end[2, 3]),
        w0=float(weight[len(request.use_joints)]),
        w1=float(weight[len(request.use_joints) + 1]),
        w2=float(weight[len(request.use_joints) + 2]),
        w3=float(weight[joint_map["arm_lift_joint"]]),
        w4=float(weight[joint_map["arm_flex_joint"]]),
        w5=float(weight[joint_map["arm_roll_joint"]]),
        w6=float(weight[joint_map["wrist_flex_joint"]]),
        w7=float(weight[joint_map["wrist_roll_joint"]]),
        r0=float(origin_to_base[0, 3]),
        r1=float(origin_to_base[1, 3]),
        r2=float(base_yaw),
        r3=float(init_lift),
        r4=float(init_flex),
        r5=float(init_roll),
        r6=float(init_wrist_flex),
        r7=float(init_wrist_roll),
    )

    if func is None:
        func = RobotFunction2(function_req, function_param)
    else:
        func.update_inputs(function_req, function_param)

    # Ensure per-call behavior matches non-cached usage.
    func.set_penalty_coeff(1000.0)
    optimizer_mode = str(optimizer).replace("-", "_")
    gpu_only = optimizer_mode == "gpu_only"
    use_gpu = optimizer_mode == "gpu" or gpu_only or (optimizer_mode == "auto" and _taichi_gpu_available())
    if use_gpu:
        if init is None:
            init = RobotOptimizer._find_initial_point(func, workspace)
        func.set_penalty_coeff(1e7)
        if gpu_optimizer is None:
            gpu_optimizer = RobotOptimizerGPU()
        status, solution = gpu_optimizer.optimize_single(
            function_req,
            function_param,
            init,
        )
        if status == OptResult.FAIL:
            if gpu_only:
                solution_angle = _build_solution_angle(request, to_torch=use_torch)
                origin_to_base = torch.eye(4, device=_torch_device(), dtype=_torch_dtype())
                origin_to_end = torch.eye(4, device=_torch_device(), dtype=_torch_dtype())
                return IKResult.FAIL, solution_angle, origin_to_base, origin_to_end
            use_gpu = False
        else:
            func._calculate_theta(solution)
            return _finalize_ik_result(
                request,
                function_param,
                workspace,
                func,
                joint_map,
                to_torch=use_torch,
            )

    result = RobotOptimizer.optimize(
        func,
        workspace,
        init=init,
    )
    if result == OptResult.FAIL:
        if retry_workspace is None:
            retry_workspace = AnalyticIKWorkspace()
        func.update_inputs(function_req, function_param)
        result = RobotOptimizer.optimize(
            func,
            retry_workspace,
            init=init,
        )
        if result == OptResult.FAIL:
            solution_angle = _build_solution_angle(request, to_torch=use_torch)
            origin_to_base = torch.eye(4, device=_torch_device(), dtype=_torch_dtype())
            origin_to_end = torch.eye(4, device=_torch_device(), dtype=_torch_dtype())
            return IKResult.FAIL, solution_angle, origin_to_base, origin_to_end

    return _finalize_ik_result(
        request,
        function_param,
        workspace,
        func,
        joint_map,
        to_torch=use_torch,
    )


def solve_base_yaw_ik(
    request: IKRequest,
    function_param: RobotFunction2Parameter,
    workspace: AnalyticIKWorkspace,
) -> Tuple[IKResult, List[IKResponse]]:
    joint_map = map_joint_and_id(request.use_joints)

    origin_to_base = request.origin_to_base
    ref_origin_to_end = request.ref_origin_to_end
    if not _is_torch_tensor(origin_to_base):
        origin_to_base = torch.as_tensor(origin_to_base, device=_torch_device(), dtype=_torch_dtype())
    if not _is_torch_tensor(ref_origin_to_end):
        ref_origin_to_end = torch.as_tensor(ref_origin_to_end, device=_torch_device(), dtype=_torch_dtype())
    theta0 = float(origin_to_base[0, 3].item())
    theta1 = float(origin_to_base[1, 3].item())
    _solve_base_yaw_ik_kernel(
        function_param,
        ref_origin_to_end,
        theta0,
        theta1,
        workspace.base_yaw_solution_count,
        workspace.base_yaw_solutions,
    )

    solution_count = int(ti_to_torch(workspace.base_yaw_solution_count, copy=True).item())
    if solution_count <= 0:
        return IKResult.FAIL, []

    raw_solutions = ti_to_torch(workspace.base_yaw_solutions, copy=True)[:solution_count]
    responses: List[IKResponse] = []
    for joint_positions in raw_solutions:
        t0, t1, t2, t3, t4, t5, t6, t7 = (joint_positions[i] for i in range(8))
        solution_angle = _build_solution_angle(request, to_torch=True)
        solution_angle.position[joint_map["arm_lift_joint"]] = t3
        solution_angle.position[joint_map["arm_flex_joint"]] = t4
        solution_angle.position[joint_map["arm_roll_joint"]] = t5
        solution_angle.position[joint_map["wrist_flex_joint"]] = t6
        solution_angle.position[joint_map["wrist_roll_joint"]] = t7

        _fk_from_solution_kernel(
            function_param,
            float(t0),
            float(t1),
            float(t2),
            float(t3),
            float(t4),
            float(t5),
            float(t6),
            float(t7),
            workspace.origin_to_base_field,
            workspace.origin_to_end_field,
        )
        origin_to_base = _mat4_from_field(workspace.origin_to_base_field)
        origin_to_end = _mat4_from_field(workspace.origin_to_end_field)
        responses.append(
            IKResponse(
                solution_angle=solution_angle,
                origin_to_base=origin_to_base,
                origin_to_end=origin_to_end,
            )
        )

    return IKResult.SUCCESS, responses


def select_closest_solution(
    request: IKRequest,
    responses: Sequence[IKResponse],
    workspace: AnalyticIKWorkspace | None = None,
) -> int:
    if not responses:
        return -1

    if workspace is None:
        workspace = AnalyticIKWorkspace()

    joint_count = len(request.use_joints)
    response_count = len(responses)
    device = _torch_device()
    dtype = _torch_dtype()
    current_arm_positions = torch.zeros(joint_count, device=device, dtype=dtype)
    for i, joint in enumerate(request.use_joints):
        current_arm_positions[i] = extract_joint_position(
            request.initial_angle,
            joint,
            0.0,
        )
    origin_to_base = request.origin_to_base
    if not _is_torch_tensor(origin_to_base):
        origin_to_base = torch.as_tensor(origin_to_base, device=device, dtype=dtype)
    current_base_positions = torch.tensor(
        [
            origin_to_base[0, 3],
            origin_to_base[1, 3],
            torch.atan2(origin_to_base[1, 0], origin_to_base[0, 0]),
        ],
        device=device,
        dtype=dtype,
    )
    response_arm_positions = torch.zeros((response_count, joint_count), device=device, dtype=dtype)
    response_base_positions = torch.zeros((response_count, 3), device=device, dtype=dtype)
    for i, response in enumerate(responses):
        if not _is_torch_tensor(response.solution_angle.position):
            response_arm_positions[i, :] = torch.as_tensor(
                response.solution_angle.position,
                device=device,
                dtype=dtype,
            )
        else:
            response_arm_positions[i, :] = response.solution_angle.position
        base = response.origin_to_base
        if not _is_torch_tensor(base):
            base = torch.as_tensor(base, device=response_arm_positions.device, dtype=response_arm_positions.dtype)
        response_base_positions[i, :] = torch.stack(
            [
                base[0, 3],
                base[1, 3],
                torch.atan2(base[1, 0], base[0, 0]),
            ],
            dim=0,
        )

    _select_closest_solution_kernel(
        current_arm_positions,
        current_base_positions,
        response_arm_positions,
        response_base_positions,
        torch.as_tensor(request.weight, device=device, dtype=dtype),
        joint_count,
        response_count,
        workspace.closest_solution_index_field,
    )
    return int(ti_to_torch(workspace.closest_solution_index_field, copy=True).item())


@ti.data_oriented
class AnalyticIK2:
    def __init__(self, *, optimizer: str = "auto"):
        self._optimizer_mode = optimizer
        self._gpu_optimizer = RobotOptimizerGPU()
        self._hsrb_param = self._make_hsrb_param()
        self._hsrc_param = self._make_hsrc_param()
        self._workspace = AnalyticIKWorkspace()
        self._retry_workspace = AnalyticIKWorkspace()
        self._select_workspace = AnalyticIKWorkspace()

        dummy_req = RobotFunction2Request(
            R11=1.0,
            R12=0.0,
            R13=0.0,
            px=0.0,
            R21=0.0,
            R22=1.0,
            R23=0.0,
            py=0.0,
            R31=0.0,
            R32=0.0,
            R33=1.0,
            pz=0.0,
            w0=1.0,
            w1=1.0,
            w2=1.0,
            w3=1.0,
            w4=1.0,
            w5=1.0,
            w6=1.0,
            w7=1.0,
            r0=0.0,
            r1=0.0,
            r2=0.0,
            r3=0.0,
            r4=0.0,
            r5=0.0,
            r6=0.0,
            r7=0.0,
        )
        self._func_hsrb = RobotFunction2(dummy_req, self._hsrb_param)
        self._func_hsrc = RobotFunction2(dummy_req, self._hsrc_param)

        self._batch_capacity = 0
        self._batch_req_field = None
        self._batch_value = None
        self._batch_feasible = None
        self._batch_param_field = RobotFunction2Parameter.field(shape=())
        self._batch_result_field = None
        self._batch_solution_field = None
        self._batch_o2b_field = None
        self._batch_o2e_field = None
        self._batch_base_yaw_solution_count = None
        self._batch_base_yaw_solutions = None

    @staticmethod
    def _make_hsrb_param() -> RobotFunction2Parameter:
        return RobotFunction2Parameter(
            L3=0.340,
            L41=0.141,
            L42=0.078,
            L51=0.005,
            L52=0.345,
            L81=0.012,
            L82=0.1405,
            t3_min=0.0,
            t3_max=0.69,
            t4_min=-2.62,
            t4_max=0.0,
            t5_min=-1.92,
            t5_max=3.67,
            t6_min=-1.92,
            t6_max=1.22,
            t7_min=-1.92,
            t7_max=3.67,
        )

    @staticmethod
    def _make_hsrc_param() -> RobotFunction2Parameter:
        return RobotFunction2Parameter(
            L3=0.350,
            L41=0.141,
            L42=0.0785,
            L51=0.005,
            L52=0.345,
            L81=0.0,
            L82=0.155,
            t3_min=0.0,
            t3_max=0.69,
            t4_min=-2.62,
            t4_max=0.0,
            t5_min=-1.92,
            t5_max=3.67,
            t6_min=-1.74,
            t6_max=1.22,
            t7_min=-1.92,
            t7_max=3.67,
        )

    def hsrb_param(self) -> RobotFunction2Parameter:
        return self._hsrb_param

    def hsrc_param(self) -> RobotFunction2Parameter:
        return self._hsrc_param

    def get_hsrb_base_position_range(self, origin_to_hand: torch.Tensor) -> BasePositionRange:
        return base_position_range(
            self._hsrb_param,
            origin_to_hand,
            self._workspace,
        )

    def get_hsrc_base_position_range(self, origin_to_hand: torch.Tensor) -> BasePositionRange:
        return base_position_range(
            self._hsrc_param,
            origin_to_hand,
            self._workspace,
        )

    def solve_ik(self, request: IKRequest) -> Tuple[IKResult, JointState, torch.Tensor, torch.Tensor]:
        return solve_ik(
            request,
            self._hsrb_param,
            self._workspace,
            func=self._func_hsrb,
            retry_workspace=self._retry_workspace,
            optimizer=self._optimizer_mode,
            gpu_optimizer=self._gpu_optimizer,
        )

    def solve_hsrc_ik(self, request: IKRequest) -> Tuple[IKResult, JointState, torch.Tensor, torch.Tensor]:
        return solve_ik(
            request,
            self._hsrc_param,
            self._workspace,
            func=self._func_hsrc,
            retry_workspace=self._retry_workspace,
            optimizer=self._optimizer_mode,
            gpu_optimizer=self._gpu_optimizer,
        )

    def _ensure_batch_capacity(self, n_envs: int) -> None:
        n_envs = int(n_envs)
        if n_envs <= self._batch_capacity:
            return
        self._batch_capacity = n_envs
        self._batch_req_field = RobotFunction2Request.field(shape=(n_envs,))
        self._batch_result_field = ti.field(dtype=ti.i32, shape=(n_envs,))
        self._batch_solution_field = ti.field(dtype=TI_FLOAT, shape=(n_envs, 5))
        self._batch_o2b_field = ti.field(dtype=TI_FLOAT, shape=(n_envs, 4, 4))
        self._batch_o2e_field = ti.field(dtype=TI_FLOAT, shape=(n_envs, 4, 4))
        self._batch_base_yaw_solution_count = ti.field(dtype=ti.i32, shape=(n_envs,))
        self._batch_base_yaw_solutions = ti.Vector.field(
            8,
            dtype=TI_FLOAT,
            shape=(n_envs, MAX_BASE_YAW_SOLUTIONS),
        )

    def _build_function_req(self, request: IKRequest) -> RobotFunction2Request:
        joint_map = map_joint_and_id(request.use_joints)
        origin_to_base = _as_torch(request.origin_to_base)
        ref_origin_to_end = _as_torch(request.ref_origin_to_end)
        weight = _as_torch(request.weight)
        base_yaw = math.atan2(
            float(origin_to_base[1, 0].item()),
            float(origin_to_base[0, 0].item()),
        )
        init_lift = extract_joint_position(request.initial_angle, "arm_lift_joint", 0.0)
        init_flex = extract_joint_position(request.initial_angle, "arm_flex_joint", 0.0)
        init_roll = extract_joint_position(request.initial_angle, "arm_roll_joint", 0.0)
        init_wrist_flex = extract_joint_position(
            request.initial_angle,
            "wrist_flex_joint",
            0.0,
        )
        init_wrist_roll = extract_joint_position(
            request.initial_angle,
            "wrist_roll_joint",
            0.0,
        )

        return RobotFunction2Request(
            R11=float(ref_origin_to_end[0, 0]),
            R12=float(ref_origin_to_end[0, 1]),
            R13=float(ref_origin_to_end[0, 2]),
            px=float(ref_origin_to_end[0, 3]),
            R21=float(ref_origin_to_end[1, 0]),
            R22=float(ref_origin_to_end[1, 1]),
            R23=float(ref_origin_to_end[1, 2]),
            py=float(ref_origin_to_end[1, 3]),
            R31=float(ref_origin_to_end[2, 0]),
            R32=float(ref_origin_to_end[2, 1]),
            R33=float(ref_origin_to_end[2, 2]),
            pz=float(ref_origin_to_end[2, 3]),
            w0=float(weight[len(request.use_joints)]),
            w1=float(weight[len(request.use_joints) + 1]),
            w2=float(weight[len(request.use_joints) + 2]),
            w3=float(weight[joint_map["arm_lift_joint"]]),
            w4=float(weight[joint_map["arm_flex_joint"]]),
            w5=float(weight[joint_map["arm_roll_joint"]]),
            w6=float(weight[joint_map["wrist_flex_joint"]]),
            w7=float(weight[joint_map["wrist_roll_joint"]]),
            r0=float(origin_to_base[0, 3]),
            r1=float(origin_to_base[1, 3]),
            r2=float(base_yaw),
            r3=float(init_lift),
            r4=float(init_flex),
            r5=float(init_roll),
            r6=float(init_wrist_flex),
            r7=float(init_wrist_roll),
        )

    def _find_initial_points_batch(
        self,
        requests: Sequence[IKRequest],
        *,
        function_param: RobotFunction2Parameter,
        func: RobotFunction2,
    ) -> list[Vector2]:
        n_envs = len(requests)
        self._ensure_batch_capacity(n_envs)
        assert self._batch_req_field is not None
        device = _torch_device()
        dtype = _torch_dtype()

        # Update batch request field.
        for i, req in enumerate(requests):
            self._batch_req_field[i] = self._build_function_req(req)

        self._batch_param_field[None] = function_param

        # theta4 bounds are computed per env using the existing exact method.
        lower_arr = torch.zeros((n_envs,), device=device, dtype=dtype)
        upper_arr = torch.zeros((n_envs,), device=device, dtype=dtype)
        valid_env = torch.zeros((n_envs,), device=device, dtype=torch.int32)
        for i, req in enumerate(requests):
            func.update_inputs(self._batch_req_field[i], function_param)
            func.set_penalty_coeff(1000.0)
            lo, hi, feasible = func.theta4_boundary()
            if feasible:
                lower_arr[i] = float(lo)
                upper_arr[i] = float(hi)
                valid_env[i] = 1

        t2_lower, t2_upper = -math.pi, math.pi

        init_out: list[Vector2] = [Vector2(0.0, -1.0) for _ in range(n_envs)]
        done = torch.zeros((n_envs,), device=device, dtype=torch.bool)
        best_infeasible_val = torch.full((n_envs,), float("inf"), device=device, dtype=dtype)
        best_infeasible_xy = torch.zeros((n_envs, 2), device=device, dtype=dtype)

        for grid in range(10, 51, 10):
            # Build candidate arrays with a common K (max across envs).
            t2_list = []
            t4_list = []
            k_list = []
            k_max = 0
            for i in range(n_envs):
                if int(valid_env[i].item()) == 0:
                    k_list.append(0)
                    t2_list.append(torch.zeros((0,), device=device, dtype=dtype))
                    t4_list.append(torch.zeros((0,), device=device, dtype=dtype))
                    continue
                t2_i, t4_i = _sample_candidate_grid_torch(
                    lower=float(lower_arr[i].item()),
                    upper=float(upper_arr[i].item()),
                    t2_lower=t2_lower,
                    t2_upper=t2_upper,
                    grid=grid,
                    device=device,
                    dtype=dtype,
                )
                k = int(t2_i.shape[0])
                k_list.append(k)
                k_max = max(k_max, k)
                t2_list.append(t2_i)
                t4_list.append(t4_i)

            if k_max == 0:
                continue

            t2 = torch.zeros((n_envs, k_max), device=device, dtype=dtype)
            t4 = torch.zeros((n_envs, k_max), device=device, dtype=dtype)
            for i in range(n_envs):
                k = k_list[i]
                if k:
                    t2[i, :k] = t2_list[i]
                    t4[i, :k] = t4_list[i]

            out_value = torch.empty((n_envs, k_max), device=device, dtype=dtype)
            out_feasible = torch.empty((n_envs, k_max), device=device, dtype=torch.int32)
            _batch_eval_initial_candidates_kernel(
                int(n_envs),
                int(k_max),
                valid_env,
                t2,
                t4,
                self._batch_req_field,
                self._batch_param_field,
                1000.0,
                out_value,
                out_feasible,
            )

            for i in range(n_envs):
                k = k_list[i]
                if k == 0 or int(valid_env[i].item()) == 0:
                    continue

                vals = out_value[i, :k]
                feas = out_feasible[i, :k].bool()

                # Track best infeasible overall.
                j = int(torch.argmin(vals).item())
                if float(vals[j].item()) < float(best_infeasible_val[i].item()):
                    best_infeasible_val[i] = vals[j]
                    best_infeasible_xy[i, 0] = t2[i, j]
                    best_infeasible_xy[i, 1] = t4[i, j]

                if bool(done[i].item()):
                    continue

                if bool(torch.any(feas).item()):
                    masked = torch.where(feas, vals, torch.full_like(vals, float("inf")))
                    j2 = int(torch.argmin(masked).item())
                    init_out[i] = Vector2(float(t2[i, j2].item()), float(t4[i, j2].item()))
                    done[i] = True

            if bool(torch.all(done | (valid_env == 0)).item()):
                break

        for i in range(n_envs):
            if int(valid_env[i].item()) == 0:
                continue
            if (not bool(done[i].item())) and torch.isfinite(best_infeasible_val[i]):
                init_out[i] = Vector2(
                    float(best_infeasible_xy[i, 0].item()),
                    float(best_infeasible_xy[i, 1].item()),
                )
        return init_out

    def _find_initial_points_batch_from_fields(
        self,
        n_envs: int,
        *,
        function_param: RobotFunction2Parameter,
        func: RobotFunction2,
        lower_arr: torch.Tensor,
        upper_arr: torch.Tensor,
        valid_env: torch.Tensor,
    ) -> list[Vector2]:
        n_envs = int(n_envs)
        self._ensure_batch_capacity(n_envs)
        assert self._batch_req_field is not None
        device = _torch_device()
        dtype = _torch_dtype()

        self._batch_param_field[None] = function_param

        t2_lower, t2_upper = -math.pi, math.pi

        init_out: list[Vector2] = [Vector2(0.0, -1.0) for _ in range(n_envs)]
        done = torch.zeros((n_envs,), device=device, dtype=torch.bool)
        best_infeasible_val = torch.full((n_envs,), float("inf"), device=device, dtype=dtype)
        best_infeasible_xy = torch.zeros((n_envs, 2), device=device, dtype=dtype)

        for grid in range(10, 51, 10):
            t2_list = []
            t4_list = []
            k_list = []
            k_max = 0
            for i in range(n_envs):
                if int(valid_env[i].item()) == 0:
                    k_list.append(0)
                    t2_list.append(torch.zeros((0,), device=device, dtype=dtype))
                    t4_list.append(torch.zeros((0,), device=device, dtype=dtype))
                    continue
                t2_i, t4_i = _sample_candidate_grid_torch(
                    lower=float(lower_arr[i].item()),
                    upper=float(upper_arr[i].item()),
                    t2_lower=t2_lower,
                    t2_upper=t2_upper,
                    grid=grid,
                    device=device,
                    dtype=dtype,
                )
                k = int(t2_i.shape[0])
                k_list.append(k)
                k_max = max(k_max, k)
                t2_list.append(t2_i)
                t4_list.append(t4_i)

            if k_max == 0:
                continue

            t2 = torch.zeros((n_envs, k_max), device=device, dtype=dtype)
            t4 = torch.zeros((n_envs, k_max), device=device, dtype=dtype)
            for i in range(n_envs):
                k = k_list[i]
                if k:
                    t2[i, :k] = t2_list[i]
                    t4[i, :k] = t4_list[i]

            out_value = torch.empty((n_envs, k_max), device=device, dtype=dtype)
            out_feasible = torch.empty((n_envs, k_max), device=device, dtype=torch.int32)
            _batch_eval_initial_candidates_kernel(
                int(n_envs),
                int(k_max),
                valid_env,
                t2,
                t4,
                self._batch_req_field,
                self._batch_param_field,
                float(1000.0),
                out_value,
                out_feasible,
            )

            # Pick best candidate for each env.
            for i in range(n_envs):
                if bool(done[i].item()) or int(valid_env[i].item()) == 0:
                    continue
                values = out_value[i]
                feasible = out_feasible[i]
                feasible_mask = feasible == 1
                if bool(feasible_mask.any().item()):
                    best_idx = int(values.masked_fill(~feasible_mask, float("inf")).argmin().item())
                    init_out[i] = Vector2(
                        float(t2[i, best_idx].item()),
                        float(t4[i, best_idx].item()),
                    )
                    done[i] = True
                    continue
                best_idx = int(values.argmin().item())
                if float(values[best_idx].item()) < float(best_infeasible_val[i].item()):
                    best_infeasible_val[i] = values[best_idx]
                    best_infeasible_xy[i, 0] = t2[i, best_idx]
                    best_infeasible_xy[i, 1] = t4[i, best_idx]

        # Fill remaining with best infeasible candidates.
        for i in range(n_envs):
            if not bool(done[i].item()):
                init_out[i] = Vector2(
                    float(best_infeasible_xy[i, 0].item()),
                    float(best_infeasible_xy[i, 1].item()),
                )
        return init_out

    def solve_ik_batch_tensors(
        self,
        *,
        ref_origin_to_end: torch.Tensor,
        origin_to_base: torch.Tensor,
        init_angles: torch.Tensor,
        weight: torch.Tensor,
        robot: str = "hsrb",
        to_torch: bool | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        n_envs = int(ref_origin_to_end.shape[0])
        self._ensure_batch_capacity(n_envs)
        assert self._batch_req_field is not None
        assert self._batch_result_field is not None
        assert self._batch_solution_field is not None
        assert self._batch_o2b_field is not None
        assert self._batch_o2e_field is not None
        assert self._batch_base_yaw_solution_count is not None
        assert self._batch_base_yaw_solutions is not None

        if to_torch is None:
            to_torch = True

        _batch_build_req_field_kernel(
            int(n_envs),
            ref_origin_to_end,
            origin_to_base,
            weight,
            init_angles,
            self._batch_req_field,
        )
        function_param = self._hsrb_param if robot == "hsrb" else self._hsrc_param
        self._batch_param_field[None] = function_param

        valid_env = torch.ones((n_envs,), device=_torch_device(), dtype=torch.int32)
        t2_lower, t2_upper = -math.pi, math.pi
        t4_lower, t4_upper = function_param.t4_min, function_param.t4_max
        grid = 20
        idx = torch.arange(1, grid - 1, device=_torch_device(), dtype=_torch_dtype())
        t2_step = (float(t2_upper) - float(t2_lower)) / float(grid)
        t4_step = (float(t4_upper) - float(t4_lower)) / float(grid)
        t2_vals = float(t2_lower) + idx * t2_step
        t4_vals = float(t4_lower) + idx * t4_step
        grid_t2, grid_t4 = torch.meshgrid(t2_vals, t4_vals, indexing="ij")
        t2 = grid_t2.reshape(-1)
        t4 = grid_t4.reshape(-1)
        k_max = int(t2.shape[0])

        out_value = torch.empty((n_envs, k_max), device=_torch_device(), dtype=_torch_dtype())
        out_feasible = torch.empty((n_envs, k_max), device=_torch_device(), dtype=torch.int32)
        _batch_eval_initial_candidates_shared_grid_kernel(
            int(n_envs),
            int(k_max),
            valid_env,
            t2,
            t4,
            self._batch_req_field,
            self._batch_param_field,
            float(1000.0),
            out_value,
            out_feasible,
        )

        feasible_mask = out_feasible == 1
        feasible_any = feasible_mask.any(dim=1)
        values_feasible = out_value.masked_fill(~feasible_mask, float("inf"))
        idx_feasible = values_feasible.argmin(dim=1)
        idx_any = out_value.argmin(dim=1)
        idx_best = torch.where(feasible_any, idx_feasible, idx_any)
        init_t = torch.stack([t2[idx_best], t4[idx_best]], dim=1)

        optimizer_mode = str(self._optimizer_mode).replace("-", "_")
        use_gpu = optimizer_mode == "gpu" or optimizer_mode == "gpu_only" or (
            optimizer_mode == "auto" and _taichi_gpu_available()
        )
        if not use_gpu:
            raise RuntimeError("solve_ik_batch_tensors requires GPU optimizer.")

        solutions, opt_results = self._gpu_optimizer.optimize_batch_tensors(
            self._batch_req_field,
            self._batch_param_field,
            init_t,
        )
        if getattr(self, "_debug", False):
            opt_results_t = torch.as_tensor(opt_results)
            fail = int((opt_results_t == RobotOptimizerGPU.RESULT_FAIL).sum().item())
            total = int(opt_results_t.numel())
            print(f"[IK DEBUG] batch gpu result: fails={fail}/{total}")

        _batch_finalize_kernel(
            int(n_envs),
            self._batch_req_field,
            self._batch_param_field,
            solutions,
            opt_results.to(torch.int32),
            self._batch_result_field,
            self._batch_solution_field,
            self._batch_o2b_field,
            self._batch_o2e_field,
        )

        results = ti_to_torch(self._batch_result_field, copy=True)
        sol = ti_to_torch(self._batch_solution_field, copy=True)
        o2b = ti_to_torch(self._batch_o2b_field, copy=True)
        o2e = ti_to_torch(self._batch_o2e_field, copy=True)
        return results, sol, o2b, o2e

    def solve_base_yaw_ik_batch_tensors(
        self,
        *,
        ref_origin_to_end: torch.Tensor,
        origin_to_base: torch.Tensor,
        init_angles: torch.Tensor,
        weight: torch.Tensor,
        robot: str = "hsrb",
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        n_envs = int(ref_origin_to_end.shape[0])
        self._ensure_batch_capacity(n_envs)
        assert self._batch_req_field is not None
        assert self._batch_result_field is not None
        assert self._batch_solution_field is not None
        assert self._batch_o2b_field is not None
        assert self._batch_o2e_field is not None

        function_param = self._hsrb_param if robot == "hsrb" else self._hsrc_param
        self._batch_param_field[None] = function_param

        theta0 = origin_to_base[:, 0, 3]
        theta1 = origin_to_base[:, 1, 3]

        _solve_base_yaw_ik_batch_kernel(
            function_param,
            ref_origin_to_end,
            theta0,
            theta1,
            self._batch_base_yaw_solution_count,
            self._batch_base_yaw_solutions,
        )

        _batch_select_base_yaw_kernel(
            int(n_envs),
            origin_to_base,
            init_angles,
            weight,
            self._batch_base_yaw_solution_count,
            self._batch_base_yaw_solutions,
            function_param,
            self._batch_result_field,
            self._batch_solution_field,
            self._batch_o2b_field,
            self._batch_o2e_field,
        )

        results = ti_to_torch(self._batch_result_field, copy=True)
        sol = ti_to_torch(self._batch_solution_field, copy=True)
        o2b = ti_to_torch(self._batch_o2b_field, copy=True)
        o2e = ti_to_torch(self._batch_o2e_field, copy=True)
        return results, sol, o2b, o2e

    def solve_ik_batch(
        self,
        requests: Sequence[IKRequest],
        *,
        to_torch: bool | None = None,
    ) -> tuple[
        list[IKResult],
        list[JointState],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        if to_torch is None:
            to_torch = any(
                _is_torch_tensor(req.ref_origin_to_end) or _is_torch_tensor(req.origin_to_base)
                for req in requests
            )
        init = self._find_initial_points_batch(
            requests,
            function_param=self._hsrb_param,
            func=self._func_hsrb,
        )
        optimizer_mode = str(self._optimizer_mode).replace("-", "_")
        gpu_only = optimizer_mode == "gpu_only"
        use_gpu = optimizer_mode == "gpu" or gpu_only or (
            optimizer_mode == "auto" and _taichi_gpu_available()
        )
        results: list[IKResult] = []
        sol: list[JointState] = []
        o2b: list[torch.Tensor] = []
        o2e: list[torch.Tensor] = []
        if use_gpu:
            solutions, opt_results = self._gpu_optimizer.optimize_batch(
                self._batch_req_field,
                self._batch_param_field,
                init,
            )
            if getattr(self, "_debug", False):
                opt_results_t = torch.as_tensor(opt_results)
                fail = int((opt_results_t == RobotOptimizerGPU.RESULT_FAIL).sum().item())
                total = int(opt_results_t.numel())
                print(f"[IK DEBUG] batch gpu result: fails={fail}/{total}")
            for i, req in enumerate(requests):
                if int(opt_results[i]) == RobotOptimizerGPU.RESULT_FAIL:
                    if gpu_only:
                        r = IKResult.FAIL
                        a = _build_solution_angle(req, to_torch=to_torch)
                        b = torch.eye(4, device=_torch_device(), dtype=_torch_dtype())
                        e = torch.eye(4, device=_torch_device(), dtype=_torch_dtype())
                    else:
                        r, a, b, e = solve_ik(
                            req,
                            self._hsrb_param,
                            self._workspace,
                            func=self._func_hsrb,
                            retry_workspace=self._retry_workspace,
                            init=init[i],
                            optimizer="cpu",
                        )
                else:
                    func_req = self._batch_req_field[i]
                    self._func_hsrb.update_inputs(func_req, self._hsrb_param)
                    self._func_hsrb.set_penalty_coeff(1e7)
                    solution = Vector2(float(solutions[i, 0]), float(solutions[i, 1]))
                    self._func_hsrb._calculate_theta(solution)
                    r, a, b, e = _finalize_ik_result(
                        req,
                        self._hsrb_param,
                        self._workspace,
                        self._func_hsrb,
                        map_joint_and_id(req.use_joints),
                        to_torch=to_torch,
                    )
                results.append(r)
                sol.append(a)
                o2b.append(b)
                o2e.append(e)
            return results, sol, o2b, o2e

        for req, init_i in zip(requests, init):
            r, a, b, e = solve_ik(
                req,
                self._hsrb_param,
                self._workspace,
                func=self._func_hsrb,
                retry_workspace=self._retry_workspace,
                init=init_i,
                optimizer="cpu",
            )
            results.append(r)
            sol.append(a)
            o2b.append(b)
            o2e.append(e)
        return results, sol, o2b, o2e

    def solve_hsrc_ik_batch(
        self,
        requests: Sequence[IKRequest],
        *,
        to_torch: bool | None = None,
    ) -> tuple[
        list[IKResult],
        list[JointState],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        if to_torch is None:
            to_torch = any(
                _is_torch_tensor(req.ref_origin_to_end) or _is_torch_tensor(req.origin_to_base)
                for req in requests
            )
        init = self._find_initial_points_batch(
            requests,
            function_param=self._hsrc_param,
            func=self._func_hsrc,
        )
        optimizer_mode = str(self._optimizer_mode).replace("-", "_")
        gpu_only = optimizer_mode == "gpu_only"
        use_gpu = optimizer_mode == "gpu" or gpu_only or (
            optimizer_mode == "auto" and _taichi_gpu_available()
        )
        results: list[IKResult] = []
        sol: list[JointState] = []
        o2b: list[torch.Tensor] = []
        o2e: list[torch.Tensor] = []
        if use_gpu:
            solutions, opt_results = self._gpu_optimizer.optimize_batch(
                self._batch_req_field,
                self._batch_param_field,
                init,
            )
            if getattr(self, "_debug", False):
                opt_results_t = torch.as_tensor(opt_results)
                fail = int((opt_results_t == RobotOptimizerGPU.RESULT_FAIL).sum().item())
                total = int(opt_results_t.numel())
                print(f"[IK DEBUG] batch gpu result: fails={fail}/{total}")
            for i, req in enumerate(requests):
                if int(opt_results[i]) == RobotOptimizerGPU.RESULT_FAIL:
                    if gpu_only:
                        r = IKResult.FAIL
                        a = _build_solution_angle(req, to_torch=to_torch)
                        b = torch.eye(4, device=_torch_device(), dtype=_torch_dtype())
                        e = torch.eye(4, device=_torch_device(), dtype=_torch_dtype())
                    else:
                        r, a, b, e = solve_ik(
                            req,
                            self._hsrc_param,
                            self._workspace,
                            func=self._func_hsrc,
                            retry_workspace=self._retry_workspace,
                            init=init[i],
                            optimizer="cpu",
                        )
                else:
                    func_req = self._batch_req_field[i]
                    self._func_hsrc.update_inputs(func_req, self._hsrc_param)
                    self._func_hsrc.set_penalty_coeff(1e7)
                    solution = Vector2(float(solutions[i, 0]), float(solutions[i, 1]))
                    self._func_hsrc._calculate_theta(solution)
                    r, a, b, e = _finalize_ik_result(
                        req,
                        self._hsrc_param,
                        self._workspace,
                        self._func_hsrc,
                        map_joint_and_id(req.use_joints),
                        to_torch=to_torch,
                    )
                results.append(r)
                sol.append(a)
                o2b.append(b)
                o2e.append(e)
            return results, sol, o2b, o2e

        for req, init_i in zip(requests, init):
            r, a, b, e = solve_ik(
                req,
                self._hsrc_param,
                self._workspace,
                func=self._func_hsrc,
                retry_workspace=self._retry_workspace,
                init=init_i,
                optimizer="cpu",
            )
            results.append(r)
            sol.append(a)
            o2b.append(b)
            o2e.append(e)
        return results, sol, o2b, o2e

    def solve_base_yaw_ik(self, request: IKRequest) -> Tuple[IKResult, List[IKResponse]]:
        return solve_base_yaw_ik(request, self._hsrb_param, self._workspace)

    def solve_hsrc_base_yaw_ik(self, request: IKRequest) -> Tuple[IKResult, List[IKResponse]]:
        return solve_base_yaw_ik(request, self._hsrc_param, self._workspace)

    def select_closest_solution(
        self,
        request: IKRequest,
        responses: Sequence[IKResponse],
    ) -> int:
        return select_closest_solution(
            request,
            responses,
            self._select_workspace,
        )
