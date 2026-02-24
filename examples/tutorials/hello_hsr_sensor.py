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
    show_viewer=True,
)

scene.add_entity(
    gs.morphs.Plane(),
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

scene.build(n_envs=1, env_spacing=(3.0, 3.0))

sensors = URDFSensorManager(scene=scene, entity=hsr).create_from_urdf(
    URDF_PATH,
    create_lidar=True,
    create_cameras=True,
    create_depth_cameras=True,
    create_imu=True,
    create_force_torque=True,
    camera_backend="rasterizer",
    depth_res_override=depth_res_override,
    draw_debug=True,
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

steps = int(args.steps)
step_count = 0

while True:
    scene.step()
    step_count += 1
    if steps > 0 and step_count >= steps:
        break
