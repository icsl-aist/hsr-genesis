import sys
import math
from pathlib import Path
import argparse
import xml.etree.ElementTree as ET

import numpy as np
import genesis as gs
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=0)
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


def _quat_wxyz_to_yaw(quat) -> float:
    if isinstance(quat, torch.Tensor):
        q = quat.detach().cpu().numpy()
    else:
        q = np.asarray(quat, dtype=np.float64)
    w, x, y, z = q[:4]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _arm_dofs_idx_local(entity) -> list[int]:
    dofs: list[int] = []
    for name in JOINT_ORDER:
        joint_dofs = entity.get_joint(name).dofs_idx_local
        if isinstance(joint_dofs, (list, tuple)):
            dofs.extend(int(idx) for idx in joint_dofs)
        else:
            dofs.append(int(joint_dofs))
    return dofs


def _qpos_to_arm_dofs(entity, qpos: torch.Tensor, arm_dofs_idx: list[int]) -> torch.Tensor:
    saved_qpos = entity.get_qpos().clone()
    try:
        entity.set_qpos(qpos, zero_velocity=False)
        dofs = entity.get_dofs_position()
    finally:
        entity.set_qpos(saved_qpos, zero_velocity=False)
    if dofs.ndim == 1:
        dofs = dofs.unsqueeze(0)
    return dofs[:, arm_dofs_idx]


depth_res_override = _parse_res(args.depth_res)

dt = 0.02

try:
    gs.init(backend=gs.gpu)
except RuntimeError as exc:  # pragma: no cover - demo fallback
    print(f"[Genesis] GPU backend unavailable ({exc}); falling back to CPU.")
    gs.init(backend=gs.cpu)

# Import after Genesis init; some genesis modules assert initialization at import time.
from hsr_genesis.sensor_manager import URDFSensorManager  # noqa: E402
from hsr_genesis.hsr_rigid_entity import HSRBURDF, JointTrajectory  # noqa: E402
from hsr_genesis.base_controller import Trajectory  # noqa: E402
from hsr_genesis.analytic_ik import JOINT_ORDER  # noqa: E402

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
    visualize_contact=True,
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
        visualize_contact=True,
    )

    scene.add_entity(
        gs.morphs.Box(
            pos=(0.6, 0.6, 0.1),
            size=(0.35, 0.15, 0.2),
            fixed=True,
            collision=True,
        ),
        surface=gs.surfaces.Default(color=(0.2, 0.2, 0.8, 1.0)),
        visualize_contact=True,
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
        visualize_contact=True,
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
        base_control_mode="controller",
        optimizer="gpu",
    ),
    visualize_contact=True,
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

arm_dofs_idx = _arm_dofs_idx_local(hsr)

# Hold head at its initial position via PD control.
head_dofs_idx = hsr._hsr_head_dofs_idx_local
head_hold_pos = hsr.get_dofs_position(dofs_idx_local=head_dofs_idx)
if head_hold_pos.ndim == 1:
    head_hold_pos = head_hold_pos.unsqueeze(0)


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

# Convergence parameters: move to the next target when the hand is within
# this distance of the goal, or after a maximum timeout.
REACH_THRESHOLD = 0.05  # meters
MAX_TARGET_TIME = 15.0  # seconds
TRAJ_DURATION = 3.0  # trajectory interpolation duration

end_effector_link = hsr.get_link("hand_palm_link")

current_target_pos: torch.Tensor | None = None
current_target_ee_pos: torch.Tensor | None = None
sim_time = [0.0]
trajectory_active = False
target_start_time = 0.0
need_new_target = True

while True:
    # Hold head via PD control every step.
    hsr.control_dofs_position(head_hold_pos, dofs_idx_local=head_dofs_idx)

    if need_new_target:
        need_new_target = False
        target_start_time = sim_time[0]
        target_pos, target_quat_wxyz = sample_random_ik_targets(
            base_xy=initial_base_xy,
        )
        current_target_pos = target_pos
        current_target_ee_pos = target_pos.clone()
        qpos = hsr.inverse_kinematics(
            link=end_effector_link,
            pos=target_pos,
            quat=target_quat_wxyz,
            envs_idx=envs_idx_all_torch,
        )
        if qpos.ndim == 1:
            qpos = qpos.unsqueeze(0)

        # Extract arm DOF positions from the IK solution.
        arm_dofs = _qpos_to_arm_dofs(hsr, qpos[0], arm_dofs_idx)

        # Extract base pose (x, y, yaw) from the IK solution.
        target_x = float(qpos[0, 0])
        target_y = float(qpos[0, 1])
        target_yaw = _quat_wxyz_to_yaw(qpos[0, 3:7])

        # Build arm trajectory (single waypoint).
        arm_traj = JointTrajectory(
            positions=arm_dofs,
            time_from_start=torch.tensor(
                [TRAJ_DURATION], device=gs.device, dtype=gs.tc_float
            ),
            joint_names=list(JOINT_ORDER),
        )

        # Build base trajectory (single waypoint).
        base_traj = Trajectory(
            positions=torch.tensor(
                [[target_x, target_y, target_yaw]],
                device=gs.device,
                dtype=gs.tc_float,
            ),
            time_from_start=torch.tensor(
                [TRAJ_DURATION], device=gs.device, dtype=gs.tc_float
            ),
        )

        hsr.set_whole_body_trajectory_batched(
            arm_trajectory=arm_traj,
            base_trajectory=base_traj,
            envs_idx=[0],
            start_time=None,
        )
        trajectory_active = True

    # Step the whole-body trajectory controller (arm PD + base controller).
    if trajectory_active:
        hsr.step_whole_body_trajectory_batched(dt, envs_idx=[0])

    scene.step()
    sim_time[0] += dt
    if ik_target_marker is not None and current_target_pos is not None:
        ik_target_marker.set_pos(
            current_target_pos,
            envs_idx=envs_idx_all_torch,
            zero_velocity=True,
            relative=False,
        )

    # Check if the hand has reached the target or the timeout has elapsed.
    if trajectory_active and current_target_ee_pos is not None:
        ee_pos = end_effector_link.get_pos()
        if ee_pos.ndim == 2:
            ee_pos = ee_pos[0]
        goal = current_target_ee_pos
        if goal.ndim == 2:
            goal = goal[0]
        dist = torch.norm(ee_pos - goal).item()
        elapsed = sim_time[0] - target_start_time
        if dist < REACH_THRESHOLD or elapsed >= MAX_TARGET_TIME:
            need_new_target = True

    step_count += 1
    if steps > 0 and step_count >= steps:
        break
