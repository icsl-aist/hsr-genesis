import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class JointSpec:
    name: str
    joint_type: str
    axis: tuple[float, float, float]
    child: str
    lower: float | None
    upper: float | None
    effort: float | None
    velocity: float | None
    mimic: bool


@dataclass(frozen=True)
class GainResult:
    name: str
    mode: str
    kp: float
    kv: float
    note: str | None = None


def _parse_xyz(text: str | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if not text:
        return default
    parts = [p for p in text.replace(",", " ").split() if p]
    if len(parts) != 3:
        return default
    return float(parts[0]), float(parts[1]), float(parts[2])


def _parse_limits(limit: ET.Element | None) -> tuple[float | None, float | None, float | None, float | None]:
    if limit is None:
        return None, None, None, None
    lower = limit.attrib.get("lower")
    upper = limit.attrib.get("upper")
    effort = limit.attrib.get("effort")
    velocity = limit.attrib.get("velocity")
    return (
        float(lower) if lower is not None else None,
        float(upper) if upper is not None else None,
        float(effort) if effort is not None else None,
        float(velocity) if velocity is not None else None,
    )


def _first_dof_index(dofs: int | list[int] | tuple[int, ...]) -> int:
    if isinstance(dofs, (list, tuple)):
        return int(dofs[0]) if dofs else 0
    return int(dofs)


def _parse_joints(urdf_path: Path) -> list[JointSpec]:
    root = ET.parse(urdf_path).getroot()
    joints: list[JointSpec] = []
    for joint in root.findall("joint"):
        joint_type = joint.attrib.get("type", "")
        name = joint.attrib.get("name", "")
        child_el = joint.find("child")
        child = child_el.attrib.get("link", "") if child_el is not None else ""
        axis_el = joint.find("axis")
        axis = _parse_xyz(axis_el.attrib.get("xyz") if axis_el is not None else None, (1.0, 0.0, 0.0))
        limit = joint.find("limit")
        lower, upper, effort, velocity = _parse_limits(limit)
        mimic = joint.find("mimic") is not None
        if not name:
            continue
        joints.append(
            JointSpec(
                name=name,
                joint_type=joint_type,
                axis=axis,
                child=child or "",
                lower=lower,
                upper=upper,
                effort=effort,
                velocity=velocity,
                mimic=mimic,
            )
        )
    return joints


def _pick_step(lower: float | None, upper: float | None, default_step: float) -> float:
    if lower is None or upper is None:
        return float(default_step)
    span = max(upper - lower, 0.0)
    if span <= 1e-6:
        return float(default_step)
    return max(min(0.25 * span, default_step), 0.05 * span)


def _clamp(value: float, lower: float | None, upper: float | None) -> float:
    if lower is not None:
        value = max(lower, value)
    if upper is not None:
        value = min(upper, value)
    return value


def _pick_velocity(limit: float | None, default_velocity: float) -> float:
    if limit is None or limit <= 0.0:
        return float(default_velocity)
    return max(min(0.5 * limit, default_velocity), 0.2 * limit)


def _cap_kp_by_effort(kp: float, effort: float | None, step: float) -> tuple[float, str | None]:
    if effort is None or effort <= 0.0:
        return kp, None
    max_kp = effort / max(step, 1e-4)
    if kp > max_kp:
        return max_kp, "kp_capped_by_effort"
    return kp, None


def _cap_kv_by_effort(kv: float, effort: float | None, target_velocity: float) -> tuple[float, str | None]:
    if effort is None or effort <= 0.0:
        return kv, None
    max_kv = effort / max(abs(target_velocity), 1e-4)
    if kv > max_kv:
        return max_kv, "kv_capped_by_effort"
    return kv, None


def _format_float(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:8.1f}"
    if abs(value) >= 10:
        return f"{value:8.3f}"
    return f"{value:8.5f}"


def _scale_gain(value: float, factor: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value * factor, max_value))


def _split_names(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _infer_wheel_drive(joints: list[JointSpec]) -> set[str]:
    names = {joint.name for joint in joints}
    hsr_defaults = {
        "base_r_drive_wheel_joint",
        "base_l_drive_wheel_joint",
    }
    if hsr_defaults.issubset(names):
        return hsr_defaults
    inferred = {name for name in names if "drive_wheel" in name or "wheel_drive" in name}
    if inferred:
        return inferred
    return {name for name in names if name.endswith("_wheel_joint")}


def _infer_passive(joints: list[JointSpec]) -> set[str]:
    names = {joint.name for joint in joints}
    hsr_defaults = {
        "base_r_passive_wheel_x_frame_joint",
        "base_l_passive_wheel_x_frame_joint",
        "base_r_passive_wheel_y_frame_joint",
        "base_l_passive_wheel_y_frame_joint",
        "base_r_passive_wheel_z_joint",
        "base_l_passive_wheel_z_joint",
    }
    if hsr_defaults.issubset(names):
        return hsr_defaults
    return {name for name in names if "passive" in name}


def _infer_skip(joints: list[JointSpec]) -> set[str]:
    names = {joint.name for joint in joints}
    skip = {name for name in names if "ft_sensor" in name}
    skip.update({name for name in names if "spring" in name})
    return skip


def main() -> None:
    parser = argparse.ArgumentParser(description="Autotune PD gains per joint using URDF inertia data.")
    parser.add_argument(
        "--urdf",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data" / "urdf" / "hsrb4s.urdf"),
        help="Path to the robot URDF.",
    )
    parser.add_argument("--wn", type=float, default=12.0, help="Target natural frequency for position joints.")
    parser.add_argument(
        "--wheel-wn",
        type=float,
        default=20.0,
        help="Target natural frequency for wheel velocity joints.",
    )
    parser.add_argument("--zeta", type=float, default=1.0, help="Damping ratio.")
    parser.add_argument(
        "--step",
        type=float,
        default=0.2,
        help="Reference position step size (rad/m) for effort capping.",
    )
    parser.add_argument(
        "--vel",
        type=float,
        default=2.0,
        help="Reference velocity step size for effort capping.",
    )
    parser.add_argument(
        "--wheel-drive",
        type=str,
        default="",
        help="Comma-separated wheel drive joint names (velocity control).",
    )
    parser.add_argument(
        "--passive",
        type=str,
        default="",
        help="Comma-separated passive joint names (excluded from tuning).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tuned_gains.json",
        help="Write tuned gains to this JSON file.",
    )
    parser.add_argument(
        "--no-sim-tune",
        action="store_false",
        dest="sim_tune",
        help="Disable simulator-based tuning.",
    )
    parser.set_defaults(sim_tune=True)
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Skip simulator validation of tuned gains.",
    )
    parser.add_argument(
        "--test-steps",
        type=int,
        default=600,
        help="Number of simulation steps for each test.",
    )
    parser.add_argument(
        "--pos-tol",
        type=float,
        default=0.01,
        help="Position error tolerance for position joints.",
    )
    parser.add_argument(
        "--vel-tol",
        type=float,
        default=0.1,
        help="Velocity error tolerance for velocity joints.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the tuned gains to a Genesis scene.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="cpu",
        choices=("cpu", "gpu"),
        help="Genesis backend used when --apply is set.",
    )
    args = parser.parse_args()

    urdf_path = Path(args.urdf)
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    joints = _parse_joints(urdf_path)

    wheel_drive = set(_split_names(args.wheel_drive))
    passive = set(_split_names(args.passive))
    if not wheel_drive:
        wheel_drive = _infer_wheel_drive(joints)
    if not passive:
        passive = _infer_passive(joints)
    skip_joints = _infer_skip(joints)

    results: list[GainResult] = []

    print("Tuned gains (kp, kv):")
    print("{:<35s} {:<9s} {:>10s} {:>10s} {:<20s}".format("joint", "mode", "kp", "kv", "note"))
    for res in results:
        print(
            "{:<35s} {:<9s} {:>10s} {:>10s} {:<20s}".format(
                res.name,
                res.mode,
                _format_float(res.kp),
                _format_float(res.kv),
                res.note or "",
            )
        )

    if args.no_test and not args.apply and not args.sim_tune:
        return

    try:
        import genesis as gs
    except Exception as exc:
        raise RuntimeError("Genesis is required for simulator tuning/testing/apply.") from exc

    if not getattr(gs, "_initialized", False):
        backend = gs.cpu if args.backend == "cpu" else gs.gpu
        gs.init(backend=backend, precision="32", logging_level="warning")

    from hsr_genesis.hsr_rigid_entity import HSRBURDF

    dt = 0.01
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    hsr = scene.add_entity(
        HSRBURDF(
            file=str(urdf_path),
            fixed=False,
            recompute_inertia=True,
            robot="hsrb",
            base_mode="planar",
            use_base_controller=False,
            optimizer="cpu",
        )
    )
    scene.build(n_envs=1)

    joint_by_name = {joint.name: joint for joint in joints}

    def _step_scene(steps: int) -> None:
        for _ in range(steps):
            scene.step()

    def _simulate_position(idx: int, target: float, steps: int, lower: float | None, upper: float | None) -> float:
        target = _clamp(target, lower, upper)
        hsr.control_dofs_position([target], dofs_idx_local=[idx])
        _step_scene(steps)
        return float(hsr.get_dofs_position(dofs_idx_local=[idx])[0])

    def _simulate_velocity(idx: int, target: float, steps: int) -> float:
        hsr.control_dofs_velocity([target], dofs_idx_local=[idx])
        _step_scene(steps)
        return float(hsr.get_dofs_velocity(dofs_idx_local=[idx])[0])

    tuned_step_by_joint: dict[str, float] = {}
    tuned_vel_by_joint: dict[str, float] = {}

    if args.sim_tune:
        print("\nSimulator tuning:")
        for joint in joints:
            if joint.joint_type == "fixed" or joint.mimic or joint.name in passive or joint.name in skip_joints:
                continue
            idx = _first_dof_index(hsr.get_joint(joint.name).dofs_idx_local)
            if joint.name in wheel_drive:
                target = _pick_velocity(joint.velocity, float(args.vel))
                kv = max(0.5, 0.1 * abs(target))
                note = None
                for _ in range(5):
                    for _ in range(6):
                        hsr.set_dofs_kp([0.0], dofs_idx_local=[idx])
                        hsr.set_dofs_kv([kv], dofs_idx_local=[idx])
                        actual = _simulate_velocity(idx, target, max(int(args.test_steps), 50))
                        err = abs(actual - target)
                        if err <= float(args.vel_tol):
                            break
                        scale = max(min(err / max(float(args.vel_tol), 1e-4), 5.0), 1.2)
                        kv = _scale_gain(kv, scale, 0.01, 1e6)
                    if err <= float(args.vel_tol):
                        break
                    target *= 0.5
                    note = "target_scaled"
                tuned_vel_by_joint[joint.name] = target
                results.append(GainResult(joint.name, "velocity", 0.0, kv, note))
            else:
                step = _pick_step(joint.lower, joint.upper, float(args.step))
                kp = 10.0
                note = None
                for _ in range(5):
                    start = float(hsr.get_dofs_position(dofs_idx_local=[idx])[0])
                    target = _clamp(start + step, joint.lower, joint.upper)
                    for _ in range(6):
                        kv = 2.0 * float(args.zeta) * math.sqrt(max(kp, 1e-6))
                        hsr.set_dofs_kp([kp], dofs_idx_local=[idx])
                        hsr.set_dofs_kv([kv], dofs_idx_local=[idx])
                        actual = _simulate_position(idx, target, max(int(args.test_steps), 50), joint.lower, joint.upper)
                        err = abs(actual - target)
                        if err <= float(args.pos_tol):
                            break
                        scale = max(min(err / max(float(args.pos_tol), 1e-4), 5.0), 1.2)
                        kp = _scale_gain(kp, scale, 0.01, 1e6)
                    if err <= float(args.pos_tol):
                        break
                    step *= 0.5
                    note = "step_scaled"
                tuned_step_by_joint[joint.name] = step
                results.append(GainResult(joint.name, "position", kp, kv, note))
    else:
        for joint in joints:
            if joint.joint_type == "fixed" or joint.mimic or joint.name in passive or joint.name in skip_joints:
                continue
            if joint.name in wheel_drive:
                target = _pick_velocity(joint.velocity, float(args.vel))
                kv = max(0.5, 0.1 * abs(target))
                kv, note = _cap_kv_by_effort(kv, joint.effort, target)
                results.append(GainResult(joint.name, "velocity", 0.0, kv, note))
            else:
                step = _pick_step(joint.lower, joint.upper, float(args.step))
                kp = 10.0
                kp, note = _cap_kp_by_effort(kp, joint.effort, step)
                kv = 2.0 * float(args.zeta) * math.sqrt(max(kp, 1e-6))
                results.append(GainResult(joint.name, "position", kp, kv, note))

    if wheel_drive:
        wheel_gains = [res for res in results if res.name in wheel_drive and res.mode == "velocity"]
        if wheel_gains:
            shared_kv = max(res.kv for res in wheel_gains)
            for idx, res in enumerate(results):
                if res.name in wheel_drive and res.mode == "velocity":
                    results[idx] = GainResult(res.name, res.mode, res.kp, shared_kv, res.note)

    print("Tuned gains (kp, kv):")
    print("{:<35s} {:<9s} {:>10s} {:>10s} {:<20s}".format("joint", "mode", "kp", "kv", "note"))
    for res in results:
        print(
            "{:<35s} {:<9s} {:>10s} {:>10s} {:<20s}".format(
                res.name,
                res.mode,
                _format_float(res.kp),
                _format_float(res.kv),
                res.note or "",
            )
        )

    output_path = Path(args.output)
    payload = {
        "urdf": str(urdf_path),
        "wn": float(args.wn),
        "wheel_wn": float(args.wheel_wn),
        "zeta": float(args.zeta),
        "step": float(args.step),
        "vel": float(args.vel),
        "gains": [
            {
                "name": res.name,
                "mode": res.mode,
                "kp": res.kp,
                "kv": res.kv,
                "note": res.note,
            }
            for res in results
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {len(results)} gains to {output_path}")

    if args.apply:
        dof_indices: list[int] = []
        kps: list[float] = []
        kvs: list[float] = []
        for res in results:
            dof_indices.append(_first_dof_index(hsr.get_joint(res.name).dofs_idx_local))
            kps.append(float(res.kp))
            kvs.append(float(res.kv))
        if dof_indices:
            hsr.set_dofs_kp(kps, dofs_idx_local=dof_indices)
            hsr.set_dofs_kv(kvs, dofs_idx_local=dof_indices)
            print(f"Applied gains to {len(dof_indices)} joints in the scene")

    if args.no_test:
        return

    test_steps = max(int(args.test_steps), 1)
    pos_tol = float(args.pos_tol)
    vel_tol = float(args.vel_tol)

    print("\nSimulator validation:")
    print("{:<35s} {:<9s} {:>10s} {:>10s} {:<10s}".format("joint", "mode", "target", "final", "status"))

    for res in results:
        if res.name in skip_joints:
            continue
        idx = _first_dof_index(hsr.get_joint(res.name).dofs_idx_local)
        if res.mode == "velocity":
            joint = joint_by_name.get(res.name)
            if res.name in tuned_vel_by_joint:
                target = tuned_vel_by_joint[res.name]
            else:
                target = _pick_velocity(joint.velocity if joint else None, float(args.vel))
            actual = _simulate_velocity(idx, target, test_steps)
            status = "ok" if abs(actual - target) <= vel_tol else "fail"
            print(
                "{:<35s} {:<9s} {:>10s} {:>10s} {:<10s}".format(
                    res.name, res.mode, _format_float(target), _format_float(actual), status
                )
            )
        else:
            joint = joint_by_name.get(res.name)
            if res.name in tuned_step_by_joint:
                step = tuned_step_by_joint[res.name]
            else:
                step = _pick_step(joint.lower if joint else None, joint.upper if joint else None, float(args.step))
            start = float(hsr.get_dofs_position(dofs_idx_local=[idx])[0])
            target = _clamp(start + step, joint.lower if joint else None, joint.upper if joint else None)
            actual = _simulate_position(
                idx,
                target,
                test_steps,
                joint.lower if joint else None,
                joint.upper if joint else None,
            )
            status = "ok" if abs(actual - target) <= pos_tol else "fail"
            print(
                "{:<35s} {:<9s} {:>10s} {:>10s} {:<10s}".format(
                    res.name, res.mode, _format_float(target), _format_float(actual), status
                )
            )


if __name__ == "__main__":
    main()
