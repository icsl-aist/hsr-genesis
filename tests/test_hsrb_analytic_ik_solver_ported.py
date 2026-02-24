import math
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="session")
def backend():
    return None


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

AnalyticIK2 = _ik.AnalyticIK2
IKRequest = _ik.IKRequest
JointState = _ik.JointState
IKResult = _ik.IKResult
JOINT_ORDER = _ik.JOINT_ORDER


def _unit_vector(axis: int) -> torch.Tensor:
    vec = torch.zeros(3, dtype=torch.float32)
    vec[axis] = 1.0
    return vec


def _make_request(*, ref_origin_to_end: torch.Tensor, init_config: torch.Tensor, base_mode: str) -> IKRequest:
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

    # Match C++ weights: [x, y, yaw, lift, flex, roll, wrist_flex, wrist_roll]
    # analytic_ik expects [lift, flex, roll, wrist_flex, wrist_roll, x, y, yaw]
    weight = torch.tensor([10.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 1.0], dtype=torch.float32)

    if base_mode == "planar":
        linear = [_unit_vector(0), _unit_vector(1)]
        rotational = [_unit_vector(2)]
    elif base_mode == "rotation_z":
        linear = []
        rotational = [_unit_vector(2)]
    else:
        raise ValueError(f"Unknown base_mode: {base_mode}")

    return IKRequest(
        frame_name="hand_palm_link",
        frame_to_end=torch.eye(4, dtype=torch.float32),
        ref_origin_to_end=ref_origin_to_end.to(dtype=torch.float32),
        origin_to_base=origin_to_base,
        initial_angle=initial_angle,
        use_joints=list(JOINT_ORDER),
        weight=weight,
        linear_base_movements=linear,
        rotational_base_movements=rotational,
    )


def _near_affine(ref: torch.Tensor, cur: torch.Tensor) -> bool:
    trans_thr = 1e-3
    rot_thr = 2.0 * math.pi * 1e-2

    trans_diff = float(torch.linalg.norm(ref[:3, 3] - cur[:3, 3]).item())
    r = ref[:3, :3] @ cur[:3, :3].T
    angle = float(math.acos(max(-1.0, min(1.0, (float(torch.trace(r).item()) - 1.0) * 0.5))))
    return (trans_diff < trans_thr) and (abs(angle) < rot_thr)


def _load_data_file(path: Path) -> list[torch.Tensor]:
    data: list[torch.Tensor] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values = [float(x) for x in line.split()]
            if len(values) != 8:
                continue
            data.append(torch.tensor(values, dtype=torch.float32))
    return data


def _fk(param, config: torch.Tensor) -> torch.Tensor:
    # Uses analytic_ik's internal FK kernel to match its IK target representation.
    ws = _ik.AnalyticIKWorkspace()
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


def _arm_within_limits(param, arm: torch.Tensor) -> bool:
    t3, t4, t5, t6, t7 = (float(x.item()) for x in arm)
    eps = 1e-6
    return (
        (param.t3_min - eps) <= t3 <= (param.t3_max + eps)
        and (param.t4_min - eps) <= t4 <= (param.t4_max + eps)
        and (param.t5_min - eps) <= t5 <= (param.t5_max + eps)
        and (param.t6_min - eps) <= t6 <= (param.t6_max + eps)
        and (param.t7_min - eps) <= t7 <= (param.t7_max + eps)
    )


@pytest.mark.parametrize(
    "robot,base_mode,ref_config_file,init_config_file",
    [
        ("hsrb", "planar", "random_config10.dat", "init_config.dat"),
        ("hsrb", "planar", "limit_config.dat", "small_init_config.dat"),
        ("hsrb", "planar", "singular_config.dat", "small_init_config.dat"),
        ("hsrc", "planar", "random_config10.dat", "init_config.dat"),
        ("hsrc", "planar", "limit_config_501.dat", "small_init_config.dat"),
        ("hsrc", "planar", "singular_config.dat", "small_init_config.dat"),
        ("hsrb", "rotation_z", "random_config10.dat", "init_config.dat"),
        ("hsrb", "rotation_z", "limit_config.dat", "small_init_config.dat"),
        ("hsrb", "rotation_z", "singular_config.dat", "small_init_config.dat"),
        ("hsrc", "rotation_z", "random_config10.dat", "init_config.dat"),
        ("hsrc", "rotation_z", "limit_config_501.dat", "small_init_config.dat"),
        ("hsrc", "rotation_z", "singular_config.dat", "small_init_config.dat"),
    ],
)
def test_ik_solver_ported_solvable(robot: str, base_mode: str, ref_config_file: str, init_config_file: str):
    cfg_dir = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "hsrb_analytic_ik"
        / "joint_configs"
    )
    ref_configs = _load_data_file(cfg_dir / ref_config_file)
    init_configs = _load_data_file(cfg_dir / init_config_file)

    aik = AnalyticIK2()
    param = aik.hsrb_param() if robot == "hsrb" else aik.hsrc_param()

    # Mirror C++ masking behavior for base rotation-z solver: x,y are not solved.
    mask_xy = base_mode == "rotation_z"

    # Keep runtime reasonable: mimic C++ test driver sampling.
    max_ref = 1 if "random" in ref_config_file else 5
    max_init = 3
    for ref in ref_configs[:max_ref]:
        if not _arm_within_limits(param, ref[3:]):
            continue
        for init in init_configs[:max_init]:
            if not _arm_within_limits(param, init[3:]):
                continue

            masked_ref = ref.clone()
            if mask_xy:
                masked_ref[0] = init[0]
                masked_ref[1] = init[1]

            target = _fk(param, masked_ref)
            request = _make_request(ref_origin_to_end=target, init_config=init, base_mode=base_mode)

            if base_mode == "planar":
                if robot == "hsrb":
                    result, sol_angle, _origin_to_base, origin_to_end = aik.solve_ik(request)
                else:
                    result, sol_angle, _origin_to_base, origin_to_end = aik.solve_hsrc_ik(request)
                assert result == IKResult.SUCCESS
                assert _near_affine(target, origin_to_end)
                assert _arm_within_limits(param, sol_angle.position)
            else:
                if robot == "hsrb":
                    result, responses = aik.solve_base_yaw_ik(request)
                else:
                    result, responses = aik.solve_hsrc_base_yaw_ik(request)
                assert result == IKResult.SUCCESS
                assert responses
                idx = _ik.select_closest_solution(request, responses)
                assert 0 <= idx < len(responses)
                picked = responses[idx]
                assert _near_affine(target, picked.origin_to_end)
                assert _arm_within_limits(param, picked.solution_angle.position)


@pytest.mark.parametrize("robot,base_mode", [("hsrb", "planar"), ("hsrc", "planar"), ("hsrb", "rotation_z"), ("hsrc", "rotation_z")])
def test_ik_solver_ported_unsolve(robot: str, base_mode: str):
    cfg_dir = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "hsrb_analytic_ik"
        / "joint_configs"
    )
    init_configs = _load_data_file(cfg_dir / "small_init_config.dat")

    aik = AnalyticIK2()
    param = aik.hsrb_param() if robot == "hsrb" else aik.hsrc_param()

    ref = torch.eye(4, dtype=torch.float32)
    ref[2, 3] = 2.0

    for init in init_configs:
        request = _make_request(ref_origin_to_end=ref, init_config=init, base_mode=base_mode)
        if base_mode == "planar":
            if robot == "hsrb":
                result, *_rest = aik.solve_ik(request)
            else:
                result, *_rest = aik.solve_hsrc_ik(request)
            assert result == IKResult.FAIL
        else:
            if robot == "hsrb":
                result, responses = aik.solve_base_yaw_ik(request)
            else:
                result, responses = aik.solve_hsrc_base_yaw_ik(request)
            assert result == IKResult.FAIL
            assert responses == []


def _quat_to_mat4(qx: float, qy: float, qz: float, qw: float) -> torch.Tensor:
    x, y, z, w = qx, qy, qz, qw
    n = x * x + y * y + z * z + w * w
    if n == 0.0:
        return torch.eye(4, dtype=torch.float32)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s

    m = torch.eye(4, dtype=torch.float32)
    m[0, 0] = 1.0 - (yy + zz)
    m[0, 1] = xy - wz
    m[0, 2] = xz + wy
    m[1, 0] = xy + wz
    m[1, 1] = 1.0 - (xx + zz)
    m[1, 2] = yz - wx
    m[2, 0] = xz - wy
    m[2, 1] = yz + wx
    m[2, 2] = 1.0 - (xx + yy)
    return m


def test_hsrb_base_yaw_multi_solution_ported():
    aik = AnalyticIK2()
    param = aik.hsrb_param()

    # Pose used in C++: Translation(0.6,0.07,0.6) * Quaterniond(w=0,x=0.707,y=0,z=0.707)
    target = _quat_to_mat4(0.707, 0.0, 0.707, 0.0)
    target[0, 3] = 0.6
    target[1, 3] = 0.07
    target[2, 3] = 0.6

    init = torch.zeros(8, dtype=torch.float32)
    request = _make_request(ref_origin_to_end=target, init_config=init, base_mode="rotation_z")

    result, responses = aik.solve_base_yaw_ik(request)
    assert result == IKResult.SUCCESS
    assert len(responses) == 4
    for resp in responses:
        assert _near_affine(target, resp.origin_to_end)
        assert _arm_within_limits(param, resp.solution_angle.position)


@pytest.mark.parametrize(
    "robot,func,center_x,center_y,expected",
    [
        (
            "hsrb",
            "get_hsrb_base_position_range",
            -0.14,
            0.0,
            [
                (1.4, None, None, "invalid"),
                (0.0, None, None, "invalid"),
                (0.1, 0.40, None, "check_center"),
                (0.35, 0.49, None, "check_center"),
                (0.95, 0.49, None, "no_center"),
                (1.2, 0.44, None, "no_center"),
                (0.2, None, 0.32, "no_center"),
                (0.55, None, 0.41, "no_center"),
                (0.7, None, 0.16, "no_center"),
            ],
        ),
        (
            "hsrc",
            "get_hsrc_base_position_range",
            -0.15,
            0.0,
            [
                (1.4, None, None, "invalid"),
                (0.0, None, None, "invalid"),
                (0.1, 0.38, None, "check_center"),
                (0.35, 0.49, None, "check_center"),
                (0.95, 0.49, None, "no_center"),
                (1.2, 0.45, None, "no_center"),
                (0.2, None, 0.32, "no_center"),
                (0.55, None, 0.43, "no_center"),
                (0.7, None, 0.16, "no_center"),
            ],
        ),
    ],
)
def test_base_position_range_ported(robot: str, func: str, center_x: float, center_y: float, expected):
    aik = AnalyticIK2()
    get_range = getattr(aik, func)
    # C++ uses Eigen::Quaterniond(w=0,x=0.707,y=0,z=0.707)
    rot = _quat_to_mat4(0.707, 0.0, 0.707, 0.0)

    eps = 1.0e-2
    for z, exp_rmax, exp_rmin, mode in expected:
        origin_to_hand = rot.clone()
        origin_to_hand[2, 3] = float(z)
        r = get_range(origin_to_hand)
        if mode == "invalid":
            assert r.radius_min < 0.0
            continue
        if mode == "check_center":
            assert r.center[0] == pytest.approx(center_x, abs=eps)
            assert r.center[1] == pytest.approx(center_y, abs=eps)
            assert r.radius_max == pytest.approx(exp_rmax, abs=eps)
            assert r.radius_min < r.radius_max
            continue
        if exp_rmax is not None:
            assert r.radius_max == pytest.approx(exp_rmax, abs=eps)
        if exp_rmin is not None:
            assert r.radius_min == pytest.approx(exp_rmin, abs=eps)
        assert r.radius_min < r.radius_max
