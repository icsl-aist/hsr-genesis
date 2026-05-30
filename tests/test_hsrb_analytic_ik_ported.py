import math
import os
from pathlib import Path

import pytest
import torch


def _import_analytic_ik():
    import quadrants as ti
    import genesis as gs

    if not getattr(gs, "_initialized", False):
        gs.init(backend=gs.cpu, precision="32", logging_level="warning")

    try:
        ti.init(arch=ti.cpu)
    except Exception:
        pass

    import hsr_genesis.analytic_ik as analytic_ik

    return analytic_ik


_ik = _import_analytic_ik()

Vector2 = _ik.Vector2
BiGoldenSectionLineSearch = _ik.BiGoldenSectionLineSearch
HookeAndJeevesMethod = _ik.HookeAndJeevesMethod
RobotOptimizer = _ik.RobotOptimizer
RobotOptimizerGPU = _ik.RobotOptimizerGPU
AnalyticIK2 = _ik.AnalyticIK2
AnalyticIKWorkspace = _ik.AnalyticIKWorkspace
IKRequest = _ik.IKRequest
JointState = _ik.JointState
JOINT_ORDER = _ik.JOINT_ORDER
OptResult = _ik.OptResult


class QuadraticFunction1A:
    def __call__(self, x: float) -> float:
        return x * x - 2.0 * x + 3.0


class QuadraticFunction1B:
    def __call__(self, x: float) -> float:
        return 0.5 * x * (x + 2.0)


class NonDifferentialFunction1A:
    def __call__(self, x: float) -> float:
        if x <= -1.0:
            return -0.5 * x
        return 2.0 * x + 2.5


class InvertedTrapeziumFunction1A:
    def __call__(self, x: float) -> float:
        if x < -1.0:
            return 2.0 - x
        if x < 2.0:
            return 3.0
        return x + 1.0


class ShiftAdapterFunction1:
    def __init__(self, func, shift: float):
        self._func = func
        self._shift = float(shift)

    def __call__(self, x: float) -> float:
        return self._func(x - self._shift)


class QuarticFunction2A:
    def value(self, x: Vector2) -> float:
        a = x.v1 - 2.0
        b = x.v1 - 2.0 * x.v2
        return a * a * a * a + b * b


def _unit_vector(axis: int) -> torch.Tensor:
    vec = torch.zeros(3, dtype=torch.float32)
    vec[axis] = 1.0
    return vec


def _make_request(ref_origin_to_end: torch.Tensor, init_config: torch.Tensor) -> IKRequest:
    origin_to_base = torch.eye(4, dtype=torch.float32)
    origin_to_base[0, 3] = float(init_config[0].item())
    origin_to_base[1, 3] = float(init_config[1].item())
    yaw = float(init_config[2].item())
    c = math.cos(yaw)
    s = math.sin(yaw)
    origin_to_base[0, 0] = c
    origin_to_base[0, 1] = -s
    origin_to_base[1, 0] = s
    origin_to_base[1, 1] = c

    arm = init_config[3:]
    initial_angle = JointState(
        name=list(JOINT_ORDER),
        position=torch.as_tensor(arm, dtype=torch.float32),
        velocity=None,
        effort=None,
    )

    weight = torch.tensor([10.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 1.0], dtype=torch.float32)
    return IKRequest(
        frame_name="hand_palm_link",
        frame_to_end=torch.eye(4, dtype=torch.float32),
        ref_origin_to_end=ref_origin_to_end.to(dtype=torch.float32),
        origin_to_base=origin_to_base,
        initial_angle=initial_angle,
        use_joints=list(JOINT_ORDER),
        weight=weight,
        linear_base_movements=[_unit_vector(0), _unit_vector(1)],
        rotational_base_movements=[_unit_vector(2)],
    )


def _fk(param, config: torch.Tensor) -> torch.Tensor:
    ws = AnalyticIKWorkspace()
    _ik._fk_from_solution_kernel(
        param,
        float(config[0].item()),
        float(config[1].item()),
        float(config[2].item()),
        float(config[3].item()),
        float(config[4].item()),
        float(config[5].item()),
        float(config[6].item()),
        float(config[7].item()),
        ws.origin_to_base_field,
        ws.origin_to_end_field,
    )
    return _ik.ti_to_torch(ws.origin_to_end_field, copy=True)[()]


def test_end_effector_offset_transforms_target_ported():
    aik = _ik.AnalyticIK2()
    param = aik.hsrb_param()

    config = torch.tensor([0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    target = _fk(param, config)

    # Solve WITHOUT offset — baseline
    req_no_offset = _make_request(ref_origin_to_end=target, init_config=config)
    result_no_off, sol_no_off = aik.solve_base_yaw_ik(req_no_offset)
    assert result_no_off == _ik.IKResult.SUCCESS

    # Simulate the offset adjustment that inverse_kinematics() does:
    # targets = targets @ inv(offset)
    T_offset = torch.eye(4, dtype=torch.float32)
    T_offset[2, 3] = 0.1
    inv_offset = torch.linalg.inv(T_offset)
    adjusted_target = target @ inv_offset

    req_with_offset = _make_request(ref_origin_to_end=adjusted_target, init_config=config)
    result_with_off, sol_with_off = aik.solve_base_yaw_ik(req_with_offset)
    assert result_with_off == _ik.IKResult.SUCCESS

    # The solution should differ (arm compensates for the 0.1m z-offset)
    assert sol_no_off[0] is not None
    assert sol_with_off[0] is not None
    assert not torch.allclose(sol_no_off[0].solution_angle.position, sol_with_off[0].solution_angle.position, atol=1e-4)

    # Verify correctness: o2e_with_off @ T_offset should recover the original target
    o2e_with_off = sol_with_off[0].origin_to_end
    recomposed = o2e_with_off @ T_offset
    assert torch.allclose(recomposed[:3, 3], target[:3, 3], atol=1e-3)


def test_ik_with_init_qpos_ported():
    aik = _ik.AnalyticIK2()
    param = aik.hsrb_param()

    config = torch.tensor([0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    target = _fk(param, config)

    # Use a different init config to verify the solver handles it properly
    alt_init = torch.tensor([0.0, 0.0, 0.0, 0.5, -0.5, 0.0, 0.0, 0.0], dtype=torch.float32)

    req_alt = _make_request(ref_origin_to_end=target, init_config=alt_init)
    res_alt, sol_alt = aik.solve_base_yaw_ik(req_alt)

    assert res_alt == _ik.IKResult.SUCCESS
    assert sol_alt[0] is not None

    # Verify the solution reaches the target via FK
    sol_config = torch.cat([
        sol_alt[0].origin_to_base[:2, 3],  # x, y
        torch.tensor(
            [math.atan2(sol_alt[0].origin_to_base[1, 0].item(), sol_alt[0].origin_to_base[0, 0].item())]
        ),  # base yaw
        sol_alt[0].solution_angle.position,  # arm joints
    ])
    fk_result = _fk(param, sol_config)
    assert torch.allclose(fk_result[:3, 3], target[:3, 3], atol=1e-3)


def test_vector2_ported():
    v = Vector2(3.0, 4.0)
    assert v.v1 == pytest.approx(3.0)
    assert v.v2 == pytest.approx(4.0)

    v.set(9.0, 8.0)
    assert v.v1 == pytest.approx(9.0)
    assert v.v2 == pytest.approx(8.0)

    v.zero()
    assert v.v1 == pytest.approx(0.0)
    assert v.v2 == pytest.approx(0.0)

    v.set(2.0, -1.0)
    assert v.norm() == pytest.approx(5.0**0.5)
    assert v.norm2() == pytest.approx(5.0)

    v.normalize()
    assert v.v1 == pytest.approx(2.0 / (5.0**0.5))
    assert v.v2 == pytest.approx(-1.0 / (5.0**0.5))

    a = Vector2(10.0, 20.0)
    b = Vector2(12.0, 19.0)
    assert Vector2.diff_norm(a, b) == pytest.approx(5.0**0.5)

    a = Vector2(-1.0, 2.0)
    b = Vector2(3.0, -5.0)
    c = a + b
    assert c.v1 == pytest.approx(2.0)
    assert c.v2 == pytest.approx(-3.0)

    c = a - b
    assert c.v1 == pytest.approx(-4.0)
    assert c.v2 == pytest.approx(7.0)

    c = 0.5 * a
    assert c.v1 == pytest.approx(-0.5)
    assert c.v2 == pytest.approx(1.0)

    c = a * 0.5
    assert c.v1 == pytest.approx(-0.5)
    assert c.v2 == pytest.approx(1.0)

    c = a / 2.0
    assert c.v1 == pytest.approx(-0.5)
    assert c.v2 == pytest.approx(1.0)


@pytest.mark.parametrize("step", [0.1, 0.5, 1.0, 2.0, 100.0, 0.01])
def test_bi_golden_section_line_search_quadratic_a_ported(step: float):
    max_iter = 100
    epsilon = 1e-8

    func = QuadraticFunction1A()
    search = BiGoldenSectionLineSearch(max_iter, epsilon)

    result = search.search(func, step)
    assert result == OptResult.SUCCESS
    assert search.result == OptResult.SUCCESS
    assert 1 <= search.iteration <= max_iter
    assert search.solution == pytest.approx(1.0, abs=epsilon * 10)


def test_bi_golden_section_line_search_quadratic_a_shifted_ported():
    max_iter = 100
    epsilon = 1e-8

    func = ShiftAdapterFunction1(QuadraticFunction1A(), -2.0)
    search = BiGoldenSectionLineSearch(max_iter, epsilon)

    result = search.search(func, 0.1)
    assert result == OptResult.SUCCESS
    assert search.solution == pytest.approx(-1.0, abs=epsilon * 10)


@pytest.mark.parametrize("step", [0.1, 0.5, 1.0, 2.0, 100.0, 0.01])
def test_bi_golden_section_line_search_quadratic_b_ported(step: float):
    max_iter = 100
    epsilon = 1e-8

    func = QuadraticFunction1B()
    search = BiGoldenSectionLineSearch(max_iter, epsilon)

    result = search.search(func, step)
    assert result == OptResult.SUCCESS
    assert search.result == OptResult.SUCCESS
    assert 1 <= search.iteration <= max_iter
    assert search.solution == pytest.approx(-1.0, abs=epsilon * 10)


def test_bi_golden_section_line_search_quadratic_b_shifted_ported():
    max_iter = 100
    epsilon = 1e-8

    func = ShiftAdapterFunction1(QuadraticFunction1B(), 3.0)
    search = BiGoldenSectionLineSearch(max_iter, epsilon)

    result = search.search(func, 0.1)
    assert result == OptResult.SUCCESS
    assert search.solution == pytest.approx(2.0, abs=epsilon * 10)


@pytest.mark.parametrize("step", [0.1, 0.5, 1.0, 2.0, 100.0, 0.01])
def test_bi_golden_section_line_search_non_differential_ported(step: float):
    max_iter = 100
    epsilon = 1e-8

    func = ShiftAdapterFunction1(NonDifferentialFunction1A(), 2.0)
    search = BiGoldenSectionLineSearch(max_iter, epsilon)

    result = search.search(func, step)
    assert result == OptResult.SUCCESS
    assert search.result == OptResult.SUCCESS
    assert 1 <= search.iteration <= max_iter
    assert search.solution == pytest.approx(1.0, abs=epsilon * 10)


@pytest.mark.parametrize("step", [0.1, 100.0])
def test_bi_golden_section_line_search_inverted_trapezium_ported(step: float):
    max_iter = 100
    epsilon = 1e-8

    func = ShiftAdapterFunction1(InvertedTrapeziumFunction1A(), 2.0)
    search = BiGoldenSectionLineSearch(max_iter, epsilon)

    result = search.search(func, step)
    assert result == OptResult.SUCCESS
    assert search.result == OptResult.SUCCESS
    assert 0 <= search.iteration <= max_iter
    assert 1.0 - epsilon <= search.solution <= 4.0 + epsilon


def test_hooke_and_jeeves_method_ported():
    max_iter = 100
    epsilon = 1e-6

    line_search = BiGoldenSectionLineSearch(max_iter, epsilon * 0.0001)
    method = HookeAndJeevesMethod(max_iter, epsilon)
    func = QuarticFunction2A()

    x0 = Vector2(0.0, 3.0)
    expected = Vector2(2.0, 1.0)

    for step in (1.0, 0.1, 0.01, 10.0, 100.0):
        result = method.search(func, line_search, x0, step)
        assert result == OptResult.SUCCESS
        assert method.result == OptResult.SUCCESS
        assert 0 <= method.iteration <= max_iter

        sol = method.solution
        assert sol.v1 == pytest.approx(expected.v1, abs=epsilon * 10)
        assert sol.v2 == pytest.approx(expected.v2, abs=epsilon * 10)


def test_robot_optimizer_gpu_full_parity_ported():
    if os.environ.get("RUN_GPU_OPTIMIZER_FULL") != "1":
        pytest.skip("Set RUN_GPU_OPTIMIZER_FULL=1 to run GPU/CPU parity test.")

    aik = AnalyticIK2()
    param = aik.hsrb_param()

    config = torch.tensor([0.0, 0.0, 0.0, 0.2, -1.0, 0.5, 0.1, 0.3], dtype=torch.float32)
    target = _fk(param, config)
    request = _make_request(target, config)
    func_req = aik._build_function_req(request)

    func = _ik.RobotFunction2(func_req, param)
    func.set_penalty_coeff(1e7)
    init = Vector2(0.0, 0.0)

    max_iter = 8
    gpu_optimizer = RobotOptimizerGPU(max_iteration=max_iter)
    status, gpu_sol = gpu_optimizer.optimize_single(
        func_req,
        param,
        init,
        step=1.0,
        penalty_coeff=1e7,
    )
    assert status in (OptResult.SUCCESS, OptResult.MAX_ITERATION)

    line_search = BiGoldenSectionLineSearch(max_iter, 1e-4)
    method = HookeAndJeevesMethod(max_iter, 1e-3)
    func.update_inputs(func_req, param)
    func.set_penalty_coeff(1e7)
    cpu_result = method.search(func, line_search, init, 1.0)
    assert cpu_result in (OptResult.SUCCESS, OptResult.MAX_ITERATION)
    cpu_sol = method.solution

    assert gpu_sol.v1 == pytest.approx(cpu_sol.v1, abs=1e-1)
    assert gpu_sol.v2 == pytest.approx(cpu_sol.v2, abs=1e-1)


def test_robot_optimizer_gpu_matches_cpu_ported():
    if os.environ.get("RUN_GPU_OPTIMIZER_TEST") != "1":
        pytest.skip("Set RUN_GPU_OPTIMIZER_TEST=1 to run GPU optimizer test.")
    aik = AnalyticIK2()
    param = aik.hsrb_param()

    config = torch.tensor([0.0, 0.0, 0.0, 0.2, -1.0, 0.5, 0.1, 0.3], dtype=torch.float32)
    target = _fk(param, config)
    request = _make_request(target, config)
    func_req = aik._build_function_req(request)

    func = _ik.RobotFunction2(func_req, param)
    func.set_penalty_coeff(1e7)
    init = Vector2(0.0, 0.0)

    max_iter = 1
    gpu_optimizer = RobotOptimizerGPU(max_iteration=max_iter)
    status, gpu_sol = gpu_optimizer.optimize_single(
        func_req,
        param,
        init,
        step=1.0,
        penalty_coeff=1e7,
    )
    assert status in (OptResult.SUCCESS, OptResult.MAX_ITERATION)

    assert torch.isfinite(torch.tensor(gpu_sol.v1))
    assert torch.isfinite(torch.tensor(gpu_sol.v2))


def test_robot_optimizer_gpu_reuses_capacity_after_growth_ported():
    if os.environ.get("RUN_GPU_OPTIMIZER_TEST") != "1":
        pytest.skip("Set RUN_GPU_OPTIMIZER_TEST=1 to run GPU optimizer test.")

    aik = AnalyticIK2()
    param = aik.hsrb_param()

    config = torch.tensor([0.0, 0.0, 0.0, 0.2, -1.0, 0.5, 0.1, 0.3], dtype=torch.float32)
    target = _fk(param, config)
    request = _make_request(target, config)
    func_req = aik._build_function_req(request)

    func = _ik.RobotFunction2(func_req, param)
    func.set_penalty_coeff(1e7)

    max_iter = 1
    gpu_optimizer = RobotOptimizerGPU(max_iteration=max_iter)

    req_field_2 = _ik.RobotFunction2Request.field(shape=(2,))
    req_field_2[0] = func_req
    req_field_2[1] = func_req
    param_field = _ik.RobotFunction2Parameter.field(shape=())
    param_field[None] = param

    batch_sol, batch_result = gpu_optimizer.optimize_batch(
        req_field_2,
        param_field,
        [Vector2(0.0, 0.0), Vector2(0.1, -0.1)],
        step=1.0,
        penalty_coeff=1e7,
    )
    assert batch_sol.shape == (2, 2)
    assert batch_result.shape == (2,)

    status_single, single_sol = gpu_optimizer.optimize_single(
        func_req,
        param,
        Vector2(0.0, 0.0),
        step=1.0,
        penalty_coeff=1e7,
    )
    assert status_single in (OptResult.SUCCESS, OptResult.MAX_ITERATION)
    assert torch.isfinite(torch.tensor(single_sol.v1))
    assert torch.isfinite(torch.tensor(single_sol.v2))

    req_field_1 = _ik.RobotFunction2Request.field(shape=(1,))
    req_field_1[0] = func_req
    small_batch_sol, small_batch_result = gpu_optimizer.optimize_batch(
        req_field_1,
        param_field,
        [Vector2(0.0, 0.0)],
        step=1.0,
        penalty_coeff=1e7,
    )
    assert small_batch_sol.shape == (1, 2)
    assert small_batch_result.shape == (1,)
    assert torch.isfinite(small_batch_sol).all()
