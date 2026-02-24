import sys
import math
from pathlib import Path
import argparse
import xml.etree.ElementTree as ET

import genesis as gs
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--depth-res", type=str, default="160x120")
args = parser.parse_args()

IS_DEBUG = True
n_envs = 1


def _parse_res(value: str) -> tuple[int, int] | None:
    value = str(value).strip()
    if not value:
        return None
    if "x" not in value:
        raise ValueError("--depth-res must be formatted like '320x240'")
    w_str, h_str = value.split("x", 1)
    return int(w_str), int(h_str)


def _sensor_reference_links_from_urdf(urdf_path: str) -> list[str]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    refs: list[str] = []
    for gazebo in root.findall("gazebo"):
        ref = gazebo.attrib.get("reference")
        if ref and ref not in refs:
            refs.append(ref)
    return refs


depth_res_override = _parse_res(args.depth_res)

dt = 0.02

try:
    gs.init(backend=gs.gpu)
except RuntimeError as exc:  # pragma: no cover - demo fallback
    print(f"[Genesis] GPU backend unavailable ({exc}); falling back to CPU.")
    gs.init(backend=gs.cpu)

# Import after Genesis init; some genesis modules assert initialization at import time.
from hsr_genesis.sensor_manager import URDFSensorManager  # noqa: E402
from hsr_genesis.hsr_rigid_entity import HSRBURDF  # noqa: E402

URDF_PATH = Path(__file__).resolve().parents[2] / "data" / "urdf" / "hsrb4s.urdf"

scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30,
        max_FPS=60,
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=True,
        world_frame_size=1.0,
        show_link_frame=True,
        show_cameras=True,
        plane_reflection=True,
        ambient_light=(0.1, 0.1, 0.1),
    ),
    sim_options=gs.options.SimOptions(
        dt=dt,
    ),
    rigid_options=gs.options.RigidOptions(
        use_gjk_collision=True,
    ),
    show_viewer=IS_DEBUG,
)

scene.add_entity(
    gs.morphs.Plane(),
)

if n_envs == 1:
    scene.add_entity(
        gs.morphs.Box(
            pos=(0.9, 0.0, 0.15),
            size=(0.2, 0.2, 0.3),
            fixed=True,
            collision=True,
        ),
        surface=gs.surfaces.Default(color=(0.2, 0.8, 0.2, 1.0)),
    )

    scene.add_entity(
        gs.morphs.Box(
            pos=(0.6, 0.6, 0.1),
            size=(0.35, 0.15, 0.2),
            fixed=True,
            collision=True,
        ),
        surface=gs.surfaces.Default(color=(0.2, 0.2, 0.8, 1.0)),
    )

    scene.add_entity(
        gs.morphs.Cylinder(
            pos=(1.0, -0.5, 0.25),
            radius=0.1,
            height=0.5,
            fixed=True,
            collision=True,
        ),
        surface=gs.surfaces.Default(color=(0.8, 0.6, 0.2, 1.0)),
    )

hsr = scene.add_entity(
    HSRBURDF(
        file=str(URDF_PATH),
        fixed=False,
        recompute_inertia=True,
        links_to_keep=_sensor_reference_links_from_urdf(URDF_PATH),
        robot="hsrb",
        base_mode="planar",
        end_effector_frame="hand_palm_link",
        use_base_controller=True,
        optimizer="gpu",
    )
)

rng = torch.Generator(device=gs.device)

sensors: dict[str, object] = {}
if n_envs == 1:
    sensors = URDFSensorManager(scene=scene, entity=hsr).create_from_urdf(
        URDF_PATH,
        create_lidar=True,
        create_cameras=IS_DEBUG,
        create_depth_cameras=IS_DEBUG,
        create_imu=True,
        create_force_torque=True,
        camera_backend="rasterizer",
        depth_res_override=depth_res_override,
        draw_debug=IS_DEBUG,
    )

    for name, sensor in sensors.items():
        if type(sensor).__name__ == "ForceTorqueSensor":
            try:
                print(f"[ForceTorque] {name}:", sensor.read())
            except Exception:
                pass

    for name, sensor in sensors.items():
        if type(sensor).__name__ == "IMUSensor":
            try:
                print(f"[IMU] {name}:", sensor.read())
            except Exception:
                pass

ik_target_marker = scene.add_entity(
    gs.morphs.Sphere(
        pos=(0.0, 0.0, 0.0),
        radius=0.04,
        collision=False,
        contype=0,
        conaffinity=0,
        fixed=False,
    ),
    surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0)),
)

scene.build(n_envs=n_envs, env_spacing=(3.0, 3.0))

envs_idx_all_torch = torch.arange(n_envs, device=gs.device, dtype=gs.tc_int)

initial_base_pos = hsr.get_pos()
if initial_base_pos.ndim == 1:
    initial_base_xy = initial_base_pos[:2].reshape(1, 2)
else:
    initial_base_xy = initial_base_pos[:, :2]


def quat_wxyz_from_rpy_rad(
    roll: torch.Tensor,
    pitch: torch.Tensor,
    yaw: torch.Tensor,
) -> torch.Tensor:
    half = torch.tensor(0.5, device=roll.device, dtype=roll.dtype)
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


def sample_random_ik_targets(
    *,
    base_xy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = base_xy.device
    dtype = base_xy.dtype
    pi = torch.tensor(math.pi, device=device, dtype=dtype)

    theta = torch.rand((n_envs,), generator=rng, device=device, dtype=dtype) * (2.0 * pi)
    r = 1.5 * torch.sqrt(torch.rand((n_envs,), generator=rng, device=device, dtype=dtype))
    x = base_xy[:, 0] + r * torch.cos(theta)
    y = base_xy[:, 1] + r * torch.sin(theta)
    z = 0.05 + (1.0 - 0.05) * torch.rand((n_envs,), generator=rng, device=device, dtype=dtype)

    roll = (torch.rand((n_envs,), generator=rng, device=device, dtype=dtype) * 2.0 - 1.0) * pi
    pitch = (torch.rand((n_envs,), generator=rng, device=device, dtype=dtype) - 0.5) * pi
    yaw = (torch.rand((n_envs,), generator=rng, device=device, dtype=dtype) * 2.0 - 1.0) * pi

    pos = torch.stack([x, y, z], dim=-1)
    quat = quat_wxyz_from_rpy_rad(roll, pitch, yaw)
    return pos, quat

steps = int(args.steps)
step_count = 0

next_target_time = 0.0
target_period = 1.1
step_count_per_target = max(1, int(math.ceil(float(target_period) / float(dt))))

current_target_pos: torch.Tensor | None = None
sim_time = [0.0]

while True:
    def _step():
        scene.step()
        sim_time[0] += dt
        if ik_target_marker is not None and current_target_pos is not None:
            ik_target_marker.set_pos(
                current_target_pos,
                envs_idx=envs_idx_all_torch,
                zero_velocity=True,
                relative=False,
            )

    if sim_time[0] >= next_target_time:
        next_target_time = sim_time[0] + target_period
        target_pos, target_quat_wxyz = sample_random_ik_targets(
            base_xy=initial_base_xy,
        )
        current_target_pos = target_pos
        qpos = hsr.inverse_kinematics(
            link=hsr.get_link("hand_palm_link"),
            pos=target_pos,
            quat=target_quat_wxyz,
            envs_idx=envs_idx_all_torch,
        )
        if qpos.ndim == 1:
            qpos = qpos.unsqueeze(0)
        qpos_cur = hsr.get_qpos(envs_idx=envs_idx_all_torch)
        if qpos_cur.ndim == 1:
            qpos_cur = qpos_cur.unsqueeze(0)
        for i in range(1, step_count_per_target + 1):
            alpha = float(i) / float(step_count_per_target)
            qpos_i = torch.lerp(qpos_cur, qpos, alpha)
            hsr.set_qpos(qpos_i, envs_idx=envs_idx_all_torch)
            _step()
    else:
        _step()

    step_count += 1
    if steps > 0 and step_count >= steps:
        break
