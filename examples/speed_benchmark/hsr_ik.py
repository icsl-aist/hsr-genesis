import argparse
import math
import sys
import time
from pathlib import Path

import genesis as gs
import torch

ROOT = Path(__file__).resolve().parents[2]
TUTORIALS_DIR = ROOT / "examples" / "tutorials"
URDF_PATH_DEFAULT = Path(__file__).resolve().parents[2] / "data" / "urdf" / "hsrb4s.urdf"

sys.path.insert(0, str(TUTORIALS_DIR))


def quat_wxyz_from_rpy_rad(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    half = 0.5
    cr = torch.cos(roll * half)
    sr = torch.sin(roll * half)
    cp = torch.cos(pitch * half)
    sp = torch.sin(pitch * half)
    cy = torch.cos(yaw * half)
    sy = torch.sin(yaw * half)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y, z], dim=-1)


def sample_random_targets(
    *,
    n_envs: int,
    base_xy: torch.Tensor,
    rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = base_xy.device
    dtype = base_xy.dtype
    pi = torch.tensor(math.pi, device=device, dtype=dtype)

    theta = torch.rand((n_envs,), generator=rng, device=device, dtype=dtype) * (2.0 * pi)
    r = 1.2 * torch.sqrt(torch.rand((n_envs,), generator=rng, device=device, dtype=dtype))
    x = base_xy[:, 0] + r * torch.cos(theta)
    y = base_xy[:, 1] + r * torch.sin(theta)
    z = 0.1 + 0.9 * torch.rand((n_envs,), generator=rng, device=device, dtype=dtype)

    roll = (torch.rand((n_envs,), generator=rng, device=device, dtype=dtype) * 2.0 - 1.0) * pi
    pitch = (torch.rand((n_envs,), generator=rng, device=device, dtype=dtype) - 0.5) * pi
    yaw = (torch.rand((n_envs,), generator=rng, device=device, dtype=dtype) * 2.0 - 1.0) * pi

    pos = torch.stack([x, y, z], dim=-1)
    quat = quat_wxyz_from_rpy_rad(roll, pitch, yaw)
    return pos, quat


def sync_if_cuda() -> None:
    if gs.device.type == "cuda":
        torch.cuda.synchronize()


def build_scene(
    urdf_path: Path,
    n_envs: int,
    *,
    disable_visualizer: bool,
    skip_invweight: bool,
) -> tuple[gs.Scene, "HSRBURDF", torch.Tensor, torch.Generator]:
    from hsr_genesis.hsr_rigid_entity import HSRBURDF  # noqa: E402

    scene = gs.Scene(
        show_viewer=False,
        rigid_options=gs.options.RigidOptions(
            dt=0.02,
            enable_collision=False,
            enable_self_collision=False,
        ),
    )
    scene.add_entity(gs.morphs.Plane())
    hsr = scene.add_entity(
        HSRBURDF(
            file=str(urdf_path),
            fixed=False,
            recompute_inertia=True,
            links_to_keep=["hand_palm_link"],
            robot="hsrb",
            base_mode="planar",
            end_effector_frame="hand_palm_link",
            use_base_controller=False,
            optimizer="gpu",
        )
    )
    if disable_visualizer:
        scene._visualizer.build = lambda: None
    if skip_invweight:
        scene._sim.rigid_solver._init_invweight_and_meaninertia = lambda *args, **kwargs: None
    scene.build(n_envs=n_envs, env_spacing=(3.0, 3.0))

    envs_idx = torch.arange(n_envs, device=gs.device, dtype=gs.tc_int)
    rng = torch.Generator(device=gs.device)
    return scene, hsr, envs_idx, rng


def run_single(
    urdf_path: Path,
    iters: int,
    warmup: int,
    *,
    disable_visualizer: bool,
    skip_invweight: bool,
) -> None:
    scene, hsr, envs_idx, rng = build_scene(
        urdf_path,
        n_envs=1,
        disable_visualizer=disable_visualizer,
        skip_invweight=skip_invweight,
    )
    base_pos = hsr.get_pos(envs_idx=envs_idx)
    if base_pos.ndim == 1:
        base_xy = base_pos[:2].reshape(1, 2)
    else:
        base_xy = base_pos[:, :2]

    for _ in range(warmup):
        pos, quat = sample_random_targets(n_envs=1, base_xy=base_xy, rng=rng)
        hsr.inverse_kinematics(
            link=hsr.get_link("hand_palm_link"),
            pos=pos[0],
            quat=quat[0],
            envs_idx=envs_idx,
        )

    sync_if_cuda()
    t0 = time.perf_counter()
    for _ in range(iters):
        pos, quat = sample_random_targets(n_envs=1, base_xy=base_xy, rng=rng)
        hsr.inverse_kinematics(
            link=hsr.get_link("hand_palm_link"),
            pos=pos[0],
            quat=quat[0],
            envs_idx=envs_idx,
        )
    sync_if_cuda()
    dt = time.perf_counter() - t0

    avg_ms = (dt / max(1, iters)) * 1000.0
    print(f"[single] iters={iters} avg={avg_ms:.3f} ms per call")


def run_batch(
    urdf_path: Path,
    n_envs: int,
    iters: int,
    warmup: int,
    *,
    disable_visualizer: bool,
    skip_invweight: bool,
) -> None:
    scene, hsr, envs_idx, rng = build_scene(
        urdf_path,
        n_envs=n_envs,
        disable_visualizer=disable_visualizer,
        skip_invweight=skip_invweight,
    )
    base_pos = hsr.get_pos(envs_idx=envs_idx)
    if base_pos.ndim == 1:
        base_xy = base_pos[:2].reshape(1, 2)
    else:
        base_xy = base_pos[:, :2]

    for _ in range(warmup):
        pos, quat = sample_random_targets(n_envs=n_envs, base_xy=base_xy, rng=rng)
        hsr.inverse_kinematics(
            link=hsr.get_link("hand_palm_link"),
            pos=pos,
            quat=quat,
            envs_idx=envs_idx,
        )

    sync_if_cuda()
    t0 = time.perf_counter()
    for _ in range(iters):
        pos, quat = sample_random_targets(n_envs=n_envs, base_xy=base_xy, rng=rng)
        hsr.inverse_kinematics(
            link=hsr.get_link("hand_palm_link"),
            pos=pos,
            quat=quat,
            envs_idx=envs_idx,
        )
    sync_if_cuda()
    dt = time.perf_counter() - t0

    avg_ms = (dt / max(1, iters)) * 1000.0
    per_env_ms = avg_ms / max(1, n_envs)
    print(
        f"[batch] envs={n_envs} iters={iters} avg={avg_ms:.3f} ms per call "
        f"({per_env_ms:.6f} ms per env)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark HSR IK using HSRRigidEntity.")
    parser.add_argument("--urdf", type=str, default=str(URDF_PATH_DEFAULT))
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batch-envs", type=int, default=256)
    parser.add_argument("--mode", choices=("single", "batch", "both"), default="both")
    parser.add_argument("--disable-visualizer", action="store_true")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--skip-invweight", action="store_true")
    args = parser.parse_args()

    backend = gs.cpu if args.backend == "cpu" else gs.gpu
    gs.init(backend=backend, performance_mode=True)

    urdf_path = Path(args.urdf)
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    if args.mode in ("single", "both"):
        run_single(
            urdf_path,
            iters=args.iters,
            warmup=args.warmup,
            disable_visualizer=args.disable_visualizer,
            skip_invweight=args.skip_invweight,
        )

    if args.mode in ("batch", "both"):
        run_batch(
            urdf_path,
            n_envs=args.batch_envs,
            iters=args.iters,
            warmup=args.warmup,
            disable_visualizer=args.disable_visualizer,
            skip_invweight=args.skip_invweight,
        )


if __name__ == "__main__":
    main()
