"""HSR-specific rigid entity helpers and high-level control wrappers.

License: Integrates components ported from hsrb_manipulation and
hsrb_controllers under BSD-compatible terms. This package is released
under the BSD 3-Clause License (see `hsr_genesis/LICENSE.txt`).
"""

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Sequence

import genesis as gs
import torch

from genesis.engine.entities.rigid_entity import RigidEntity

from .genesis_patches import apply_entity_cls_override_patch, apply_runtime_patches
from .analytic_ik import AnalyticIK2, IKRequest, IKResult, JointState, JOINT_ORDER
from .gripper_controller import HSRBGripperControllerBatch
from .base_controller import (
    BaseControlMode,
    HSRBBaseController,
    OmniBaseTrajectoryControl,
    Trajectory,
    to_torch,
)

apply_entity_cls_override_patch()


@dataclass(frozen=True)
class JointTrajectory:
    positions: torch.Tensor  # (T, N)
    time_from_start: torch.Tensor  # (T,)
    velocities: torch.Tensor | None = None  # (T, N) or None
    accelerations: torch.Tensor | None = None  # (T, N) or None
    joint_names: Sequence[str] | None = None


@dataclass
class _ArmTrajectoryState:
    traj: JointTrajectory | None = None
    start_time: float | None = None
    sampled_already: bool = False
    point_before_pos: torch.Tensor | None = None
    point_before_vel: torch.Tensor | None = None
    done: bool = False


def _read_torso_mimic_params(urdf_path: str | None) -> tuple[float, float] | None:
    if not urdf_path:
        return None
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        for joint in root.findall("joint"):
            if joint.attrib.get("name") != "torso_lift_joint":
                continue
            mimic = joint.find("mimic")
            if mimic is None:
                return None
            multiplier = float(mimic.attrib.get("multiplier", 1.0))
            offset = float(mimic.attrib.get("offset", 0.0))
            return multiplier, offset
    except Exception:
        return None
    return None


def _mat4_from_pos_quat_wxyz_torch(
    pos: torch.Tensor,
    quat_wxyz: torch.Tensor,
) -> torch.Tensor:
    w, x, y, z = (quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3])
    n = w * w + x * x + y * y + z * z
    eps = torch.tensor(1e-12, device=pos.device, dtype=pos.dtype)
    n_safe = torch.where(n <= eps, torch.ones_like(n), n)
    s = 2.0 / n_safe
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    xx, xy, xz = x * x * s, x * y * s, x * z * s
    yy, yz, zz = y * y * s, y * z * s, z * z * s
    R = torch.stack(
        [
            torch.stack([1.0 - (yy + zz), xy - wz, xz + wy]),
            torch.stack([xy + wz, 1.0 - (xx + zz), yz - wx]),
            torch.stack([xz - wy, yz + wx, 1.0 - (xx + yy)]),
        ],
        dim=0,
    )
    R = torch.where((n <= eps).reshape(1, 1), torch.eye(3, device=pos.device, dtype=pos.dtype), R)
    T = torch.eye(4, device=pos.device, dtype=pos.dtype)
    T[:3, :3] = R
    T[:3, 3] = pos[:3]
    return T


def _mat4_from_pos_quat_wxyz_batch(
    pos: torch.Tensor,
    quat_wxyz: torch.Tensor,
) -> torch.Tensor:
    w, x, y, z = (quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3])
    n = w * w + x * x + y * y + z * z
    eps = torch.tensor(1e-12, device=pos.device, dtype=pos.dtype)
    n_safe = torch.where(n <= eps, torch.ones_like(n), n)
    s = 2.0 / n_safe
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    xx, xy, xz = x * x * s, x * y * s, x * z * s
    yy, yz, zz = y * y * s, y * z * s, z * z * s
    R = torch.stack(
        [
            torch.stack([1.0 - (yy + zz), xy - wz, xz + wy], dim=-1),
            torch.stack([xy + wz, 1.0 - (xx + zz), yz - wx], dim=-1),
            torch.stack([xz - wy, yz + wx, 1.0 - (xx + yy)], dim=-1),
        ],
        dim=-2,
    )
    eye = torch.eye(3, device=pos.device, dtype=pos.dtype).unsqueeze(0)
    R = torch.where((n <= eps).reshape(-1, 1, 1), eye, R)

    T = torch.eye(4, device=pos.device, dtype=pos.dtype).unsqueeze(0).repeat(pos.shape[0], 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = pos[:, :3]
    return T


def _quat_wxyz_from_mat3_torch(R: torch.Tensor) -> torch.Tensor:
    m00 = R[0, 0]
    m11 = R[1, 1]
    m22 = R[2, 2]
    trace = m00 + m11 + m22
    eps = torch.tensor(1e-12, device=R.device, dtype=R.dtype)

    s0 = torch.sqrt(torch.clamp(trace + 1.0, min=0.0)) * 2.0
    s0_safe = torch.where(s0.abs() < eps, torch.ones_like(s0), s0)
    w0 = 0.25 * s0
    x0 = (R[2, 1] - R[1, 2]) / s0_safe
    y0 = (R[0, 2] - R[2, 0]) / s0_safe
    z0 = (R[1, 0] - R[0, 1]) / s0_safe

    s1 = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=0.0)) * 2.0
    s1_safe = torch.where(s1.abs() < eps, torch.ones_like(s1), s1)
    w1 = (R[2, 1] - R[1, 2]) / s1_safe
    x1 = 0.25 * s1
    y1 = (R[0, 1] + R[1, 0]) / s1_safe
    z1 = (R[0, 2] + R[2, 0]) / s1_safe

    s2 = torch.sqrt(torch.clamp(1.0 + m11 - m00 - m22, min=0.0)) * 2.0
    s2_safe = torch.where(s2.abs() < eps, torch.ones_like(s2), s2)
    w2 = (R[0, 2] - R[2, 0]) / s2_safe
    x2 = (R[0, 1] + R[1, 0]) / s2_safe
    y2 = 0.25 * s2
    z2 = (R[1, 2] + R[2, 1]) / s2_safe

    s3 = torch.sqrt(torch.clamp(1.0 + m22 - m00 - m11, min=0.0)) * 2.0
    s3_safe = torch.where(s3.abs() < eps, torch.ones_like(s3), s3)
    w3 = (R[1, 0] - R[0, 1]) / s3_safe
    x3 = (R[0, 2] + R[2, 0]) / s3_safe
    y3 = (R[1, 2] + R[2, 1]) / s3_safe
    z3 = 0.25 * s3

    use0 = trace > 0.0
    use1 = (~use0) & (m00 > m11) & (m00 > m22)
    use2 = (~use0) & (~use1) & (m11 > m22)
    use3 = (~use0) & (~use1) & (~use2)

    w = torch.where(use0, w0, torch.where(use1, w1, torch.where(use2, w2, w3)))
    x = torch.where(use0, x0, torch.where(use1, x1, torch.where(use2, x2, x3)))
    y = torch.where(use0, y0, torch.where(use1, y1, torch.where(use2, y2, y3)))
    z = torch.where(use0, z0, torch.where(use1, z1, torch.where(use2, z2, z3)))

    q = torch.stack([w, x, y, z], dim=0)
    return q / torch.norm(q).clamp(min=1e-12)


def _quat_wxyz_from_mat3_torch_batch(R: torch.Tensor) -> torch.Tensor:
    m00 = R[:, 0, 0]
    m11 = R[:, 1, 1]
    m22 = R[:, 2, 2]
    trace = m00 + m11 + m22
    eps = torch.tensor(1e-12, device=R.device, dtype=R.dtype)

    s0 = torch.sqrt(torch.clamp(trace + 1.0, min=0.0)) * 2.0
    s0_safe = torch.where(s0.abs() < eps, torch.ones_like(s0), s0)
    w0 = 0.25 * s0
    x0 = (R[:, 2, 1] - R[:, 1, 2]) / s0_safe
    y0 = (R[:, 0, 2] - R[:, 2, 0]) / s0_safe
    z0 = (R[:, 1, 0] - R[:, 0, 1]) / s0_safe

    s1 = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=0.0)) * 2.0
    s1_safe = torch.where(s1.abs() < eps, torch.ones_like(s1), s1)
    w1 = (R[:, 2, 1] - R[:, 1, 2]) / s1_safe
    x1 = 0.25 * s1
    y1 = (R[:, 0, 1] + R[:, 1, 0]) / s1_safe
    z1 = (R[:, 0, 2] + R[:, 2, 0]) / s1_safe

    s2 = torch.sqrt(torch.clamp(1.0 + m11 - m00 - m22, min=0.0)) * 2.0
    s2_safe = torch.where(s2.abs() < eps, torch.ones_like(s2), s2)
    w2 = (R[:, 0, 2] - R[:, 2, 0]) / s2_safe
    x2 = (R[:, 0, 1] + R[:, 1, 0]) / s2_safe
    y2 = 0.25 * s2
    z2 = (R[:, 1, 2] + R[:, 2, 1]) / s2_safe

    s3 = torch.sqrt(torch.clamp(1.0 + m22 - m00 - m11, min=0.0)) * 2.0
    s3_safe = torch.where(s3.abs() < eps, torch.ones_like(s3), s3)
    w3 = (R[:, 1, 0] - R[:, 0, 1]) / s3_safe
    x3 = (R[:, 0, 2] + R[:, 2, 0]) / s3_safe
    y3 = (R[:, 1, 2] + R[:, 2, 1]) / s3_safe
    z3 = 0.25 * s3

    use0 = trace > 0.0
    use1 = (~use0) & (m00 > m11) & (m00 > m22)
    use2 = (~use0) & (~use1) & (m11 > m22)
    use3 = (~use0) & (~use1) & (~use2)

    w = torch.where(use0, w0, torch.where(use1, w1, torch.where(use2, w2, w3)))
    x = torch.where(use0, x0, torch.where(use1, x1, torch.where(use2, x2, x3)))
    y = torch.where(use0, y0, torch.where(use1, y1, torch.where(use2, y2, y3)))
    z = torch.where(use0, z0, torch.where(use1, z1, torch.where(use2, z2, z3)))

    q = torch.stack([w, x, y, z], dim=-1)
    norms = torch.norm(q, dim=-1, keepdim=True).clamp(min=1e-12)
    return q / norms


def _yaw_from_quat_wxyz_batch(quat_wxyz: torch.Tensor) -> torch.Tensor:
    if quat_wxyz.ndim == 1:
        quat_wxyz = quat_wxyz.unsqueeze(0)
    w, x, y, z = (quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3])
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def _quat_wxyz_from_yaw_batch(yaw: torch.Tensor, *, device, dtype) -> torch.Tensor:
    if yaw.ndim == 0:
        yaw = yaw.unsqueeze(0)
    half = 0.5 * yaw
    w = torch.cos(half)
    z = torch.sin(half)
    zeros = torch.zeros_like(w)
    return torch.stack([w, zeros, zeros, z], dim=-1).to(device=device, dtype=dtype)


def _pose_error_torch(target: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    err_pos = target[:3, 3] - current[:3, 3]
    R_err = target[:3, :3] @ current[:3, :3].transpose(0, 1)
    trace = torch.trace(R_err)
    angle = torch.acos(torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0))
    eps = torch.tensor(1e-12, device=target.device, dtype=target.dtype)
    denom = 2.0 * torch.sin(angle)
    denom = torch.where(denom.abs() < eps, torch.ones_like(denom), denom)
    axis = torch.stack(
        [
            (R_err[2, 1] - R_err[1, 2]) / denom,
            (R_err[0, 2] - R_err[2, 0]) / denom,
            (R_err[1, 0] - R_err[0, 1]) / denom,
        ],
        dim=0,
    )
    err_rot = axis * angle
    err_rot = torch.where(
        (torch.abs(angle) < eps).reshape(1), torch.zeros(3, device=target.device, dtype=target.dtype), err_rot
    )
    return torch.cat([err_pos, err_rot], dim=0)


def _pose_error_torch_batch(target: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    err_pos = target[:, :3, 3] - current[:, :3, 3]
    R_err = target[:, :3, :3] @ current[:, :3, :3].transpose(1, 2)
    trace = R_err[:, 0, 0] + R_err[:, 1, 1] + R_err[:, 2, 2]
    angle = torch.acos(torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0))
    eps = torch.tensor(1e-12, device=target.device, dtype=target.dtype)
    denom = 2.0 * torch.sin(angle)
    denom = torch.where(denom.abs() < eps, torch.ones_like(denom), denom)
    axis = torch.stack(
        [
            (R_err[:, 2, 1] - R_err[:, 1, 2]) / denom,
            (R_err[:, 0, 2] - R_err[:, 2, 0]) / denom,
            (R_err[:, 1, 0] - R_err[:, 0, 1]) / denom,
        ],
        dim=-1,
    )
    err_rot = axis * angle.unsqueeze(-1)
    err_rot = torch.where(
        (torch.abs(angle) < eps).reshape(-1, 1),
        torch.zeros_like(err_rot),
        err_rot,
    )
    return torch.cat([err_pos, err_rot], dim=-1)


class HSRRigidEntity(RigidEntity):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        apply_runtime_patches()
        morph = kwargs.get("morph", None)
        self._hsr_robot = getattr(morph, "hsr_robot", "hsrb")
        self._hsr_base_mode = getattr(morph, "hsr_base_mode", "planar")
        self._hsr_end_effector_frame = getattr(morph, "hsr_end_effector_frame", "hand_palm_link")
        self._hsr_optimizer = getattr(morph, "hsr_optimizer", "auto")
        self._hsr_use_base_controller = bool(getattr(morph, "hsr_use_base_controller", False))
        requested_mode = getattr(morph, "hsr_base_control_mode", None)
        if requested_mode is None:
            requested_mode = BaseControlMode.CONTROLLER if self._hsr_use_base_controller else BaseControlMode.QPOS
        self._hsr_base_control_mode = BaseControlMode.normalize(requested_mode)
        self._hsr_use_base_yaw_ik = bool(getattr(morph, "hsr_use_base_yaw_ik", False))
        self._hsr_ik = AnalyticIK2(optimizer=self._hsr_optimizer)
        self._hsr_param = self._hsr_ik.hsrb_param() if self._hsr_robot == "hsrb" else self._hsr_ik.hsrc_param()
        self._hsr_weight = torch.tensor(
            [10.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 1.0],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self._hsr_arm_lift_order_idx = JOINT_ORDER.index("arm_lift_joint")
        self._hsr_frame_to_end = torch.eye(4, device=gs.device, dtype=gs.tc_float)
        if self._hsr_base_mode == "planar":
            self._hsr_linear_base = [
                torch.tensor([1.0, 0.0, 0.0], device=gs.device, dtype=gs.tc_float),
                torch.tensor([0.0, 1.0, 0.0], device=gs.device, dtype=gs.tc_float),
            ]
            self._hsr_rotational_base = [torch.tensor([0.0, 0.0, 1.0], device=gs.device, dtype=gs.tc_float)]
        elif self._hsr_base_mode == "rotation_z":
            self._hsr_linear_base = []
            self._hsr_rotational_base = [torch.tensor([0.0, 0.0, 1.0], device=gs.device, dtype=gs.tc_float)]
        else:
            raise ValueError(f"Unknown base_mode: {self._hsr_base_mode}")
        self._hsr_use_joints = list(JOINT_ORDER)
        self._hsr_arm_qs_idx_local: list[int] | None = None
        self._hsr_base_qs_idx_local: list[int] | None = None
        self._hsr_torso_qs_idx_local: int | None = None
        self._hsr_torso_mimic_multiplier: torch.Tensor | None = None
        self._hsr_torso_mimic_offset: torch.Tensor | None = None
        mimic_params = _read_torso_mimic_params(getattr(morph, "file", None))
        if mimic_params is not None:
            mult, offset = mimic_params
            self._hsr_torso_mimic_multiplier = torch.tensor(mult, device=gs.device, dtype=gs.tc_float)
            self._hsr_torso_mimic_offset = torch.tensor(offset, device=gs.device, dtype=gs.tc_float)
        self._hsr_arm_dofs_idx_local = []
        for name in self._hsr_use_joints:
            dofs = self.get_joint(name).dofs_idx_local
            if isinstance(dofs, (list, tuple)):
                self._hsr_arm_dofs_idx_local.extend(int(idx) for idx in dofs)
            else:
                self._hsr_arm_dofs_idx_local.append(int(dofs))
        self._hsr_default_gains_applied = False

        self._hsr_gripper_batch: HSRBGripperControllerBatch | None = None
        self._hsr_base_controller: HSRBBaseController | None = None
        self._hsr_base_traj_ctrls: list[OmniBaseTrajectoryControl] | None = None
        self._hsr_base_traj_time: torch.Tensor | None = None
        self._hsr_arm_traj_states: list[_ArmTrajectoryState] | None = None
        self._hsr_whole_body_time: torch.Tensor | None = None
        self._hsr_torso_dof_idx_local: int | None = None
        self._hsr_collision_disable_applied = False
        self._hsr_passive_wheel_friction_applied = False
        self._hsr_high_friction_applied = False
        self._hsr_head_hold_applied = False
        self._hsr_debug_log_counter = 0
        self._hsr_debug_log_every = 120
        self._hsr_head_dofs_idx_local = []
        for name in ("head_pan_joint", "head_tilt_joint"):
            try:
                dofs = self.get_joint(name).dofs_idx_local
                if isinstance(dofs, (list, tuple)):
                    self._hsr_head_dofs_idx_local.extend(int(idx) for idx in dofs)
                else:
                    self._hsr_head_dofs_idx_local.append(int(dofs))
            except Exception:
                continue
        # Gains are applied lazily after build.

    def get_gripper_batched(self) -> HSRBGripperControllerBatch:
        if self._hsr_gripper_batch is None:
            n_envs = int(getattr(self._solver, "n_envs", 1) or 1)
            self._hsr_gripper_batch = HSRBGripperControllerBatch(self, n_envs=n_envs)
        return self._hsr_gripper_batch

    def set_gripper_apply_force_goal_batched(
        self,
        *,
        effort: torch.Tensor,
        active_mask: torch.Tensor,
        envs_idx,
        do_control_stop: bool = False,
    ) -> None:
        self.get_gripper_batched().set_apply_force_goal(
            effort=effort,
            active_mask=active_mask,
            envs_idx=envs_idx,
            do_control_stop=bool(do_control_stop),
        )

    def step_gripper_batched(
        self,
        dt: float,
        *,
        envs_idx,
    ) -> dict[str, torch.Tensor]:
        return self.get_gripper_batched().step_apply_force(
            float(dt),
            envs_idx=envs_idx,
        )

    def get_base_controller(self) -> HSRBBaseController:
        if self._hsr_base_control_mode != BaseControlMode.CONTROLLER:
            raise RuntimeError("Base controller is disabled (base_control_mode != 'controller').")
        if not getattr(self, "_hsr_use_base_controller", False):
            raise RuntimeError("Base controller is disabled for this entity.")
        if self._hsr_base_controller is None:
            self._hsr_base_controller = HSRBBaseController(self)
        return self._hsr_base_controller

    def _hsr_disable_collision_between_links(self, link_names_a, link_names_b) -> None:
        if self._scene is None or self._scene.sim is None:
            return
        solver = self._scene.sim.rigid_solver
        if solver is None or solver.collider is None:
            return
        link_names_a = set(link_names_a)
        link_names_b = set(link_names_b)
        geoms_a = [geom.idx for geom in self.geoms if geom.link.name in link_names_a]
        geoms_b = [geom.idx for geom in self.geoms if geom.link.name in link_names_b]
        if not geoms_a or not geoms_b:
            return
        collision_pair_idx = solver.collider._collider_info.collision_pair_idx.to_numpy()
        for i_ga in geoms_a:
            for i_gb in geoms_b:
                collision_pair_idx[i_ga, i_gb] = -1
                collision_pair_idx[i_gb, i_ga] = -1
        solver.collider._collider_info.collision_pair_idx.from_numpy(collision_pair_idx)

    def _hsr_apply_default_collision_disable(self) -> None:
        if self._hsr_collision_disable_applied:
            return
        if self._scene is None or self._scene.sim is None:
            return
        solver = self._scene.sim.rigid_solver
        if solver is None or solver.collider is None:
            return
        self._hsr_disable_collision_between_links(["base_f_bumper_link"], ["base_b_bumper_link"])
        for link_name in ["base_link", "base_footprint", "base_f_bumper_link", "base_b_bumper_link"]:
            self._hsr_disable_collision_between_links(["base_l_drive_wheel_link"], [link_name])
            self._hsr_disable_collision_between_links(["base_r_drive_wheel_link"], [link_name])
            self._hsr_disable_collision_between_links(["base_l_passive_wheel_z_link"], [link_name])
            self._hsr_disable_collision_between_links(["base_r_passive_wheel_z_link"], [link_name])
            self._hsr_disable_collision_between_links(["base_l_passive_wheel_y_frame"], [link_name])
            self._hsr_disable_collision_between_links(["base_r_passive_wheel_y_frame"], [link_name])
        self._hsr_collision_disable_applied = True

    def _hsr_check_collisions(self, envs_idx=None) -> dict:
        """Check and return collision events between body links, floor, and other objects.
        
        Returns a dictionary with:
        - 'self_collisions': list of (link_a, link_b) tuples for self-collisions
        - 'floor_collisions': list of link names colliding with floor
        - 'object_collisions': list of (link, object) tuples for collisions with other objects
        """
        if self._scene is None or self._scene.sim is None:
            return {'self_collisions': [], 'floor_collisions': [], 'object_collisions': []}
        solver = self._scene.sim.rigid_solver
        if solver is None or solver.collider is None:
            return {'self_collisions': [], 'floor_collisions': [], 'object_collisions': []}
        
        try:
            contacts = solver.collider.get_contacts(as_tensor=True, to_torch=True)
        except Exception:
            return {'self_collisions': [], 'floor_collisions': [], 'object_collisions': []}
        
        if contacts is None or len(contacts) == 0:
            return {'self_collisions': [], 'floor_collisions': [], 'object_collisions': []}
        
        # Extract collision information
        link_a = contacts.get("link_a", None)
        link_b = contacts.get("link_b", None)
        
        if link_a is None or link_b is None:
            return {'self_collisions': [], 'floor_collisions': [], 'object_collisions': []}
        
        # Get link indices for this entity
        link_indices = {link.name: link.idx for link in self.links}
        
        # Wheel link names
        wheel_links = {
            "base_l_drive_wheel_link",
            "base_r_drive_wheel_link",
            "base_l_passive_wheel_z_link",
            "base_r_passive_wheel_z_link",
        }
        
        # Categorize collisions
        self_collisions = []
        floor_collisions = []
        object_collisions = []
        
        # Convert to numpy for easier processing
        link_a_np = link_a.cpu().numpy()
        link_b_np = link_b.cpu().numpy()
        
        for i in range(len(link_a_np)):
            idx_a = int(link_a_np[i])
            idx_b = int(link_b_np[i])
            
            # Check if both links belong to this entity
            a_in_entity = idx_a in link_indices.values()
            b_in_entity = idx_b in link_indices.values()
            
            if a_in_entity and b_in_entity:
                # Self-collision
                name_a = None
                name_b = None
                for name, idx in link_indices.items():
                    if idx == idx_a:
                        name_a = name
                    if idx == idx_b:
                        name_b = name
                
                if name_a and name_b:
                    self_collisions.append((name_a, name_b))
            
            elif a_in_entity or b_in_entity:
                # Collision with floor or other object
                entity_link_idx = idx_a if a_in_entity else idx_b
                entity_link_name = None
                for name, idx in link_indices.items():
                    if idx == entity_link_idx:
                        entity_link_name = name
                        break
                
                if entity_link_name:
                    # Assume collision with floor if one link is not in entity
                    floor_collisions.append(entity_link_name)
        
        return {
            'self_collisions': self_collisions,
            'floor_collisions': floor_collisions,
            'object_collisions': object_collisions,
        }

    def _hsr_apply_passive_wheel_friction(self) -> None:
        if self._hsr_passive_wheel_friction_applied:
            return
        if self._scene is None or self._scene.sim is None:
            return
        # Genesis blends contact friction as max(friction_a, friction_b), so setting
        # only the caster-wheel link friction to a low value is insufficient when the
        # floor has the default friction of 1.0 (Genesis never reads <gazebo> mu tags
        # from URDF).  We therefore also lower the friction on every floor-plane entity
        # so that the combined contact friction reflects the intended near-frictionless
        # caster behaviour.
        caster_friction = 0.01
        for name in ("base_r_passive_wheel_z_link", "base_l_passive_wheel_z_link"):
            try:
                link = self.get_link(name)
            except Exception:
                continue
            link.set_friction(caster_friction)
        # Lower the friction of any Plane (floor) entity in the scene so the
        # max-blended contact friction stays small for the caster wheels.
        try:
            for entity in self._scene.entities:
                morph = getattr(entity, "_morph", None)
                if morph is not None and isinstance(morph, gs.morphs.Plane):
                    entity.set_friction(caster_friction)
        except Exception:
            pass
        # Make all caster joints free-spinning: zero out PD gains and reduce
        # damping so the caster swivels and rolls freely.
        caster_joint_names = (
            "base_r_passive_wheel_x_frame_joint",
            "base_r_passive_wheel_y_frame_joint",
            "base_r_passive_wheel_z_joint",
            "base_l_passive_wheel_x_frame_joint",
            "base_l_passive_wheel_y_frame_joint",
            "base_l_passive_wheel_z_joint",
        )
        caster_dof_indices: list[int] = []
        for jname in caster_joint_names:
            try:
                jnt = self.get_joint(jname)
            except Exception:
                continue
            dofs = jnt.dofs_idx_local
            if isinstance(dofs, (list, tuple)):
                caster_dof_indices.extend(int(d) for d in dofs)
            else:
                caster_dof_indices.append(int(dofs))
        if caster_dof_indices:
            zeros = torch.zeros(len(caster_dof_indices), device=gs.device, dtype=gs.tc_float)
            small_damping = torch.full_like(zeros, 0.01)
            self.set_dofs_kp(zeros, dofs_idx_local=caster_dof_indices)
            self.set_dofs_kv(zeros, dofs_idx_local=caster_dof_indices)
            self.set_dofs_damping(small_damping, dofs_idx_local=caster_dof_indices)
            self.set_dofs_frictionloss(zeros, dofs_idx_local=caster_dof_indices)
        self._hsr_passive_wheel_friction_applied = True

    def _hsr_apply_high_friction_links(self) -> None:
        if self._hsr_high_friction_applied:
            return
        if self._scene is None or self._scene.sim is None:
            return
        high_friction = 1.0
        for name in (
            "base_r_drive_wheel_link",
            "base_l_drive_wheel_link",
            "hand_l_finger_tip_frame",
            "hand_r_finger_tip_frame",
            "hand_l_distal_link",
            "hand_r_distal_link",
        ):
            try:
                link = self.get_link(name)
            except Exception:
                continue
            link.set_friction(high_friction)
        self._hsr_high_friction_applied = True

    def _hsr_apply_default_gains(self) -> None:
        if self._hsr_default_gains_applied:
            return
        if self._scene is None or self._scene.sim is None:
            return
        tuned_kp = {
            # arm_lift: kp=10000 -> gravity error ~0.007 m (limited by 300 N effort cap)
            "arm_lift_joint": 10000.0,
            # arm_flex: kp=300 -> gravity error ~0.025 rad < 2 deg (grav load ~7.6 Nm)
            "arm_flex_joint": 300.0,
            # arm_roll / wrist: grav loads are small; raised from 10 for better tracking
            "arm_roll_joint": 40.0,
            "wrist_flex_joint": 60.0,
            "wrist_roll_joint": 40.0,
            "head_pan_joint": 10.0,
            "head_tilt_joint": 10.0,
            "hand_motor_joint": 10.0,
        }
        tuned_kv = {
            # kv = 2 * sqrt(kp)  (critically damped)
            "arm_lift_joint": 200.0,
            "arm_flex_joint": 34.641,
            "arm_roll_joint": 12.649,
            "wrist_flex_joint": 15.492,
            "wrist_roll_joint": 12.649,
            "head_pan_joint": 6.324555320336759,
            "head_tilt_joint": 6.324555320336759,
            "hand_motor_joint": 6.324555320336759,
        }
        if self._hsr_arm_dofs_idx_local:
            arm_kp = torch.tensor(
                [tuned_kp.get(name, 4500.0) for name in self._hsr_use_joints],
                device=gs.device,
                dtype=gs.tc_float,
            )
            arm_kv = torch.tensor(
                [tuned_kv.get(name, 450.0) for name in self._hsr_use_joints],
                device=gs.device,
                dtype=gs.tc_float,
            )
            self.set_dofs_kp(arm_kp, dofs_idx_local=self._hsr_arm_dofs_idx_local)
            self.set_dofs_kv(arm_kv, dofs_idx_local=self._hsr_arm_dofs_idx_local)
        if self._hsr_head_dofs_idx_local:
            head_names = []
            for name in ("head_pan_joint", "head_tilt_joint"):
                try:
                    self.get_joint(name)
                except Exception:
                    continue
                head_names.append(name)
            head_kp = torch.tensor(
                [tuned_kp.get(name, 4500.0) for name in head_names],
                device=gs.device,
                dtype=gs.tc_float,
            )
            head_kv = torch.tensor(
                [tuned_kv.get(name, 450.0) for name in head_names],
                device=gs.device,
                dtype=gs.tc_float,
            )
            self.set_dofs_kp(head_kp, dofs_idx_local=self._hsr_head_dofs_idx_local)
            self.set_dofs_kv(head_kv, dofs_idx_local=self._hsr_head_dofs_idx_local)
        try:
            hand_joint = self.get_joint("hand_motor_joint")
        except Exception:
            hand_joint = None
        if hand_joint is not None:
            hand_dofs = hand_joint.dofs_idx_local
            if isinstance(hand_dofs, (list, tuple)):
                hand_idx = int(hand_dofs[0]) if hand_dofs else None
            else:
                hand_idx = int(hand_dofs)
            if hand_idx is None:
                self._hsr_default_gains_applied = True
                return
            self.set_dofs_kp(
                torch.tensor([tuned_kp["hand_motor_joint"]], device=gs.device, dtype=gs.tc_float),
                dofs_idx_local=[hand_idx],
            )
            self.set_dofs_kv(
                torch.tensor([tuned_kv["hand_motor_joint"]], device=gs.device, dtype=gs.tc_float),
                dofs_idx_local=[hand_idx],
            )
        self._hsr_default_gains_applied = True

    def _hsr_apply_head_hold(self) -> None:
        if self._hsr_head_hold_applied:
            return
        if not self._hsr_head_dofs_idx_local:
            self._hsr_head_hold_applied = True
            return
        if self._scene is None or self._scene.sim is None:
            return
        head_pos = self.get_dofs_position(dofs_idx_local=self._hsr_head_dofs_idx_local, envs_idx=[0])
        if isinstance(head_pos, torch.Tensor) and head_pos.ndim > 1:
            head_pos = head_pos[0]
        head_pos = to_torch(head_pos).reshape(1, -1)
        self.control_dofs_position(head_pos, dofs_idx_local=self._hsr_head_dofs_idx_local, envs_idx=[0])
        self._hsr_head_hold_applied = True

    def _ensure_base_traj_state(self, n_envs: int) -> None:
        n_envs = int(n_envs)
        if self._hsr_base_traj_ctrls is None:
            self._hsr_base_traj_ctrls = []
        if len(self._hsr_base_traj_ctrls) < n_envs:
            for _ in range(len(self._hsr_base_traj_ctrls), n_envs):
                # Higher feedback gain for yaw (index 2) to improve yaw control
                feedback_gain = torch.tensor([1.0, 1.0, 5.0], device=gs.device, dtype=gs.tc_float)
                self._hsr_base_traj_ctrls.append(OmniBaseTrajectoryControl(feedback_gain=feedback_gain))
        if self._hsr_base_traj_time is None or self._hsr_base_traj_time.numel() < n_envs:
            old = self._hsr_base_traj_time
            self._hsr_base_traj_time = torch.zeros((n_envs,), device=gs.device, dtype=gs.tc_float)
            if old is not None and old.numel() > 0:
                self._hsr_base_traj_time[: old.numel()] = old

    def _ensure_whole_body_state(self, n_envs: int) -> None:
        n_envs = int(n_envs)
        if self._hsr_arm_traj_states is None:
            self._hsr_arm_traj_states = []
        if len(self._hsr_arm_traj_states) < n_envs:
            for _ in range(len(self._hsr_arm_traj_states), n_envs):
                self._hsr_arm_traj_states.append(_ArmTrajectoryState())
        if self._hsr_whole_body_time is None or self._hsr_whole_body_time.numel() < n_envs:
            old = self._hsr_whole_body_time
            self._hsr_whole_body_time = torch.zeros((n_envs,), device=gs.device, dtype=gs.tc_float)
            if old is not None and old.numel() > 0:
                self._hsr_whole_body_time[: old.numel()] = old

    @staticmethod
    def _make_permutation_vector(names1: Sequence[str], names2: Sequence[str]) -> list[int]:
        if len(names1) != len(names2):
            return []
        perm = []
        for name in names1:
            try:
                perm.append(list(names2).index(name))
            except ValueError:
                return []
        return perm

    @staticmethod
    def _sample_linear_trajectory(
        t: float,
        times: torch.Tensor,
        positions: torch.Tensor,
        velocities: torch.Tensor | None,
        accelerations: torch.Tensor | None,
        point_before_pos: torch.Tensor,
        point_before_vel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, float]:
        if t <= float(times[0].item()):
            t0 = 0.0
            t1 = float(times[0].item())
            p0 = point_before_pos
            p1 = positions[0]
            v0 = point_before_vel
            v1 = velocities[0] if velocities is not None else None
            a0 = torch.zeros_like(p0)
            a1 = accelerations[0] if accelerations is not None else None
            before_last = True
        elif t >= float(times[-1].item()):
            p1 = positions[-1]
            v1 = velocities[-1] if velocities is not None else None
            a1 = accelerations[-1] if accelerations is not None else None
            pos = p1.clone()
            vel = v1.clone() if v1 is not None else torch.zeros_like(p1)
            acc = a1.clone() if a1 is not None else torch.zeros_like(p1)
            return pos, vel, acc, False, t - float(times[-1].item())
        else:
            idx = int(torch.searchsorted(times, torch.tensor(t, device=times.device)).item())
            t0 = float(times[idx - 1].item())
            t1 = float(times[idx].item())
            p0 = positions[idx - 1]
            p1 = positions[idx]
            v0 = velocities[idx - 1] if velocities is not None else None
            v1 = velocities[idx] if velocities is not None else None
            a0 = accelerations[idx - 1] if accelerations is not None else None
            a1 = accelerations[idx] if accelerations is not None else None
            before_last = True

        dt = max(t1 - t0, 1.0e-9)
        alpha = (t - t0) / dt
        alpha_t = torch.tensor(alpha, dtype=p0.dtype, device=p0.device)
        pos = (1.0 - alpha_t) * p0 + alpha_t * p1
        if v0 is None or v1 is None:
            vel = (p1 - p0) / dt
        else:
            vel = (1.0 - alpha_t) * v0 + alpha_t * v1
        if a0 is None or a1 is None:
            acc = torch.zeros_like(pos)
        else:
            acc = (1.0 - alpha_t) * a0 + alpha_t * a1
        time_from_point = t - t0
        return pos, vel, acc, before_last, time_from_point

    def set_base_trajectory_batched(
        self,
        trajectory: Trajectory | Sequence[Trajectory],
        *,
        envs_idx,
        start_time: float | Sequence[float] | None = None,
    ) -> None:
        if self._solver.n_envs > 0:
            envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        envs_idx_arr = torch.as_tensor(envs_idx, device=gs.device, dtype=gs.tc_int).reshape(-1)
        if envs_idx_arr.numel() == 0:
            return

        self._ensure_base_traj_state(int(envs_idx_arr.max().item() + 1))
        assert self._hsr_base_traj_ctrls is not None
        assert self._hsr_base_traj_time is not None

        pos = self.get_pos(envs_idx=envs_idx_arr)
        quat = self.get_quat(envs_idx=envs_idx_arr)
        if pos.ndim == 1:
            pos = pos.unsqueeze(0)
        if quat.ndim == 1:
            quat = quat.unsqueeze(0)
        yaw = _yaw_from_quat_wxyz_batch(quat)
        base_positions = torch.stack([pos[:, 0], pos[:, 1], yaw], dim=-1)

        if isinstance(trajectory, Trajectory):
            trajectories = [trajectory] * int(envs_idx_arr.numel())
        else:
            trajectories = list(trajectory)
            if len(trajectories) != int(envs_idx_arr.numel()):
                raise ValueError("trajectory list length must match envs_idx length")

        if start_time is None:
            start_times = [None] * int(envs_idx_arr.numel())
        elif isinstance(start_time, Sequence):
            if len(start_time) != int(envs_idx_arr.numel()):
                raise ValueError("start_time length must match envs_idx length")
            start_times = list(start_time)
        else:
            start_times = [float(start_time)] * int(envs_idx_arr.numel())

        for i, env in enumerate(envs_idx_arr.tolist()):
            ctrl = self._hsr_base_traj_ctrls[int(env)]
            t0 = start_times[i]
            if t0 is None:
                t0 = float(self._hsr_base_traj_time[int(env)].item())
            ctrl.accept_trajectory(trajectories[i], base_positions[i], start_time=t0)

    def reset_base_trajectory_batched(self, *, envs_idx) -> None:
        if self._solver.n_envs > 0:
            envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        envs_idx_arr = torch.as_tensor(envs_idx, device=gs.device, dtype=gs.tc_int).reshape(-1)
        if envs_idx_arr.numel() == 0:
            return
        self._ensure_base_traj_state(int(envs_idx_arr.max().item() + 1))
        assert self._hsr_base_traj_ctrls is not None
        for env in envs_idx_arr.tolist():
            self._hsr_base_traj_ctrls[int(env)].reset_current_trajectory()

    def step_base_trajectory_batched(
        self,
        dt: float,
        *,
        envs_idx,
    ) -> dict[str, torch.Tensor]:
        self._hsr_apply_default_collision_disable()
        self._hsr_apply_passive_wheel_friction()
        self._hsr_apply_high_friction_links()
        self._hsr_apply_default_gains()
        self._hsr_apply_head_hold()
        if self._solver.n_envs > 0:
            envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        envs_idx_arr = torch.as_tensor(envs_idx, device=gs.device, dtype=gs.tc_int).reshape(-1)
        if envs_idx_arr.numel() == 0:
            return {"active": torch.zeros((0,), device=gs.device, dtype=torch.bool)}

        self._ensure_base_traj_state(int(envs_idx_arr.max().item() + 1))
        assert self._hsr_base_traj_ctrls is not None
        assert self._hsr_base_traj_time is not None

        self._hsr_base_traj_time[envs_idx_arr] += dt

        pos = self.get_pos(envs_idx=envs_idx_arr)
        quat = self.get_quat(envs_idx=envs_idx_arr)
        vel = self.get_vel(envs_idx=envs_idx_arr)
        ang = self.get_ang(envs_idx=envs_idx_arr)

        if pos.ndim == 1:
            pos = pos.unsqueeze(0)
        if quat.ndim == 1:
            quat = quat.unsqueeze(0)
        if vel.ndim == 1:
            vel = vel.unsqueeze(0)
        if ang.ndim == 1:
            ang = ang.unsqueeze(0)

        yaw = _yaw_from_quat_wxyz_batch(quat)
        current_positions = torch.stack([pos[:, 0], pos[:, 1], yaw], dim=-1)
        current_velocities = torch.stack([vel[:, 0], vel[:, 1], ang[:, 2]], dim=-1)

        out = torch.zeros((envs_idx_arr.numel(), 3), device=gs.device, dtype=gs.tc_float)
        active = torch.zeros((envs_idx_arr.numel(),), device=gs.device, dtype=torch.bool)
        desired_pos = torch.zeros((envs_idx_arr.numel(), 3), device=gs.device, dtype=gs.tc_float)

        for i, env in enumerate(envs_idx_arr.tolist()):
            ctrl = self._hsr_base_traj_ctrls[int(env)]
            time_now = float(self._hsr_base_traj_time[int(env)].item())
            if not ctrl.update_active_trajectory():
                continue
            ok, desired, _before_last, _time_from_point = ctrl.sample_desired_state(
                time_now,
                current_positions[i],
                current_velocities[i],
            )
            if not ok or desired is None:
                continue
            if self._hsr_base_control_mode == BaseControlMode.QPOS:
                desired_pos[i] = desired.positions
                out[i] = desired.positions
            else:
                out[i] = ctrl.get_output_velocity(current_positions[i], desired, dt=dt, current_velocities=current_velocities[i])
            active[i] = True
            ctrl.terminate_control_if_stopped(time_now, current_velocities[i])

        if self._hsr_base_control_mode == BaseControlMode.QPOS:
            active_envs = envs_idx_arr[active]
            if active_envs.numel() > 0:
                base_qs_idx_local = self._ensure_base_qs_idx()
                if len(base_qs_idx_local) < 7:
                    raise RuntimeError("Base qpos indices not available for direct position control.")
                pos = self.get_pos(envs_idx=active_envs)
                if pos.ndim == 1:
                    pos = pos.unsqueeze(0)
                z = pos[:, 2]
                yaw = desired_pos[active][:, 2]
                half_yaw = 0.5 * yaw
                quat = torch.stack(
                    [torch.cos(half_yaw), torch.zeros_like(half_yaw), torch.zeros_like(half_yaw), torch.sin(half_yaw)],
                    dim=-1,
                )
                base_pos = torch.stack(
                    [desired_pos[active][:, 0], desired_pos[active][:, 1], z],
                    dim=-1,
                )
                # Explicitly set position and yaw to avoid qpos ordering ambiguity.
                if self._solver.n_envs == 0:
                    pos_arg = base_pos[0] if base_pos.ndim == 2 else base_pos
                    quat_arg = quat[0] if quat.ndim == 2 else quat
                    self.set_pos(pos_arg, envs_idx=None, zero_velocity=False)
                    self.set_quat(quat_arg, envs_idx=None, zero_velocity=False)
                else:
                    self.set_pos(base_pos, envs_idx=active_envs, zero_velocity=False)
                    self.set_quat(quat, envs_idx=active_envs, zero_velocity=False)
        else:
            base_controller = self.get_base_controller()
            base_controller.update_velocity_command_batch(out, envs_idx=envs_idx_arr.tolist())
            base_controller.step_batch(dt, envs_idx=envs_idx_arr.tolist())

        return {"active": active, "command": out}

    def set_whole_body_trajectory_batched(
        self,
        *,
        arm_trajectory: JointTrajectory | Sequence[JointTrajectory],
        base_trajectory: Trajectory | Sequence[Trajectory] | None,
        envs_idx,
        start_time: float | Sequence[float] | None = None,
    ) -> None:
        if self._solver.n_envs > 0:
            envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        envs_idx_arr = torch.as_tensor(envs_idx, device=gs.device, dtype=gs.tc_int).reshape(-1)
        if envs_idx_arr.numel() == 0:
            return

        self._ensure_whole_body_state(int(envs_idx_arr.max().item() + 1))
        assert self._hsr_arm_traj_states is not None

        if base_trajectory is not None:
            self.set_base_trajectory_batched(base_trajectory, envs_idx=envs_idx_arr.tolist(), start_time=start_time)

        if isinstance(arm_trajectory, JointTrajectory):
            arm_trajs = [arm_trajectory] * int(envs_idx_arr.numel())
        else:
            arm_trajs = list(arm_trajectory)
            if len(arm_trajs) != int(envs_idx_arr.numel()):
                raise ValueError("arm_trajectory list length must match envs_idx length")

        if start_time is None:
            start_times = [None] * int(envs_idx_arr.numel())
        elif isinstance(start_time, Sequence):
            if len(start_time) != int(envs_idx_arr.numel()):
                raise ValueError("start_time length must match envs_idx length")
            start_times = list(start_time)
        else:
            start_times = [float(start_time)] * int(envs_idx_arr.numel())

        for i, env in enumerate(envs_idx_arr.tolist()):
            traj = arm_trajs[i]
            positions = to_torch(traj.positions).to(device=gs.device, dtype=gs.tc_float)
            time_from_start = to_torch(traj.time_from_start).to(device=gs.device, dtype=gs.tc_float)
            velocities = (
                None if traj.velocities is None else to_torch(traj.velocities).to(device=gs.device, dtype=gs.tc_float)
            )
            accelerations = (
                None
                if traj.accelerations is None
                else to_torch(traj.accelerations).to(device=gs.device, dtype=gs.tc_float)
            )

            if positions.ndim != 2 or time_from_start.ndim != 1:
                raise ValueError("arm_trajectory positions must be (T, N) and time_from_start (T,)")
            if positions.shape[0] != time_from_start.shape[0]:
                raise ValueError("arm_trajectory positions and time_from_start length mismatch")

            joint_names = traj.joint_names
            if joint_names is not None:
                perm = self._make_permutation_vector(JOINT_ORDER, joint_names)
                if not perm:
                    raise ValueError("arm_trajectory joint_names mismatch")
                positions = positions[:, perm]
                if velocities is not None:
                    velocities = velocities[:, perm]
                if accelerations is not None:
                    accelerations = accelerations[:, perm]
            if positions.shape[1] != len(self._hsr_arm_dofs_idx_local):
                raise ValueError("arm_trajectory joint dimension mismatch")

            state = self._hsr_arm_traj_states[int(env)]
            state.traj = JointTrajectory(
                positions=positions,
                time_from_start=time_from_start,
                velocities=velocities,
                accelerations=accelerations,
                joint_names=list(JOINT_ORDER),
            )
            state.start_time = start_times[i]
            state.sampled_already = False
            state.point_before_pos = None
            state.point_before_vel = None
            state.done = False

    def reset_whole_body_trajectory_batched(self, *, envs_idx) -> None:
        if self._solver.n_envs > 0:
            envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        envs_idx_arr = torch.as_tensor(envs_idx, device=gs.device, dtype=gs.tc_int).reshape(-1)
        if envs_idx_arr.numel() == 0:
            return

        self._ensure_whole_body_state(int(envs_idx_arr.max().item() + 1))
        assert self._hsr_arm_traj_states is not None
        for env in envs_idx_arr.tolist():
            state = self._hsr_arm_traj_states[int(env)]
            state.traj = None
            state.start_time = None
            state.sampled_already = False
            state.point_before_pos = None
            state.point_before_vel = None
            state.done = False
        self.reset_base_trajectory_batched(envs_idx=envs_idx_arr.tolist())

    def step_whole_body_trajectory_batched(
        self,
        dt: float,
        *,
        envs_idx,
    ) -> dict[str, torch.Tensor]:
        self._hsr_apply_default_collision_disable()
        self._hsr_apply_passive_wheel_friction()
        self._hsr_apply_high_friction_links()
        self._hsr_apply_default_gains()
        self._hsr_apply_head_hold()
        if self._solver.n_envs > 0:
            envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        envs_idx_arr = torch.as_tensor(envs_idx, device=gs.device, dtype=gs.tc_int).reshape(-1)
        if envs_idx_arr.numel() == 0:
            return {
                "active": torch.zeros((0,), device=gs.device, dtype=torch.bool),
                "base": torch.zeros((0, 3), device=gs.device, dtype=gs.tc_float),
                "arm": torch.zeros((0, len(self._hsr_arm_dofs_idx_local)), device=gs.device, dtype=gs.tc_float),
            }

        self._ensure_whole_body_state(int(envs_idx_arr.max().item() + 1))
        assert self._hsr_arm_traj_states is not None
        assert self._hsr_whole_body_time is not None

        self._hsr_whole_body_time[envs_idx_arr] += dt

        arm_pos = self.get_dofs_position(dofs_idx_local=self._hsr_arm_dofs_idx_local, envs_idx=envs_idx_arr)
        arm_vel = self.get_dofs_velocity(dofs_idx_local=self._hsr_arm_dofs_idx_local, envs_idx=envs_idx_arr)
        if arm_pos.ndim == 1:
            arm_pos = arm_pos.unsqueeze(0)
        if arm_vel.ndim == 1:
            arm_vel = arm_vel.unsqueeze(0)

        desired_arm = torch.zeros_like(arm_pos)
        active = torch.zeros((envs_idx_arr.numel(),), device=gs.device, dtype=torch.bool)

        for i, env in enumerate(envs_idx_arr.tolist()):
            state = self._hsr_arm_traj_states[int(env)]
            if state.traj is None:
                continue

            # When trajectory is done, keep commanding the final position
            # so the arm holds steady while the base may still be moving.
            if state.done:
                desired_arm[i] = state.traj.positions[-1]
                active[i] = True
                continue

            time_now = float(self._hsr_whole_body_time[int(env)].item())
            if state.start_time is None:
                state.start_time = time_now
            t = time_now - state.start_time

            if not state.sampled_already:
                state.point_before_pos = arm_pos[i].clone()
                state.point_before_vel = arm_vel[i].clone()
                state.sampled_already = True

            pos, _vel, _acc, _before_last, _time_from_point = self._sample_linear_trajectory(
                t,
                state.traj.time_from_start,
                state.traj.positions,
                state.traj.velocities,
                state.traj.accelerations,
                state.point_before_pos,
                state.point_before_vel,
            )
            desired_arm[i] = pos
            active[i] = True

            if t >= float(state.traj.time_from_start[-1].item()):
                state.done = True

        if torch.any(active):
            active_envs = envs_idx_arr[active].tolist()
            self.control_dofs_position(
                desired_arm[active],
                dofs_idx_local=self._hsr_arm_dofs_idx_local,
                envs_idx=active_envs,
            )

            torso_idx = self._ensure_torso_dof_idx()
            if (
                torso_idx is not None
                and self._hsr_torso_mimic_multiplier is not None
                and self._hsr_torso_mimic_offset is not None
            ):
                arm_lift = desired_arm[active][:, self._hsr_arm_lift_order_idx]
                torso_pos = arm_lift * self._hsr_torso_mimic_multiplier + self._hsr_torso_mimic_offset
                self.control_dofs_position(
                    torso_pos.reshape(-1, 1),
                    dofs_idx_local=[torso_idx],
                    envs_idx=active_envs,
                )
                self._hsr_debug_log_counter += 1
                if self._hsr_debug_log_every > 0 and self._hsr_debug_log_counter % self._hsr_debug_log_every == 0:
                    env0 = active_envs[0] if active_envs else -1
                    arm0 = float(arm_lift[0].item()) if arm_lift.numel() else 0.0
                    torso0 = float(torso_pos[0].item()) if torso_pos.numel() else 0.0
                    print(
                        f"[hsr] torso_mimic env={env0} arm_lift={arm0:.4f} torso={torso0:.4f}",
                        flush=True,
                    )

        base_status = self.step_base_trajectory_batched(dt, envs_idx=envs_idx_arr.tolist())

        return {
            "active": active,
            "base": base_status.get(
                "command", torch.zeros((envs_idx_arr.numel(), 3), device=gs.device, dtype=gs.tc_float)
            ),
            "arm": desired_arm,
        }

    def _current_base_origin_to_base(self, *, envs_idx=None) -> torch.Tensor:
        pos = self.get_pos(envs_idx=envs_idx)
        quat = self.get_quat(envs_idx=envs_idx)
        if pos.ndim >= 2:
            pos = pos[0]
        if quat.ndim >= 2:
            quat = quat[0]
        w, x, y, z = (quat[0], quat[1], quat[2], quat[3])
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        c = torch.cos(yaw)
        s = torch.sin(yaw)
        T = torch.eye(4, device=pos.device, dtype=pos.dtype)
        T[0, 0] = c
        T[0, 1] = -s
        T[1, 0] = s
        T[1, 1] = c
        T[0, 3] = pos[0]
        T[1, 3] = pos[1]
        T[2, 3] = pos[2]
        return T

    def _current_arm_joint_state(self, *, envs_idx=None) -> JointState:
        dofs = self.get_dofs_position(dofs_idx_local=self._hsr_arm_dofs_idx_local, envs_idx=envs_idx)
        if dofs.ndim >= 2:
            dofs = dofs[0]
        dofs = dofs.reshape(-1)
        return JointState(name=list(self._hsr_use_joints), position=dofs)

    def _current_arm_joint_state_batch(self, *, envs_idx=None) -> torch.Tensor:
        dofs = self.get_dofs_position(dofs_idx_local=self._hsr_arm_dofs_idx_local, envs_idx=envs_idx)
        if dofs.ndim == 1:
            dofs = dofs.unsqueeze(0)
        return dofs.reshape(dofs.shape[0], -1)

    def _current_base_origin_to_base_batch(self, *, envs_idx=None) -> torch.Tensor:
        pos = self.get_pos(envs_idx=envs_idx)
        quat = self.get_quat(envs_idx=envs_idx)
        if pos.ndim == 1:
            pos = pos.unsqueeze(0)
        if quat.ndim == 1:
            quat = quat.unsqueeze(0)
        w, x, y, z = (quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3])
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        c = torch.cos(yaw)
        s = torch.sin(yaw)
        T = torch.zeros((pos.shape[0], 4, 4), device=pos.device, dtype=pos.dtype)
        T[:, 0, 0] = c
        T[:, 0, 1] = -s
        T[:, 1, 0] = s
        T[:, 1, 1] = c
        T[:, 2, 2] = 1.0
        T[:, 3, 3] = 1.0
        T[:, 0, 3] = pos[:, 0]
        T[:, 1, 3] = pos[:, 1]
        T[:, 2, 3] = pos[:, 2]
        return T

    def _build_request(self, *, target_origin_to_end: torch.Tensor, envs_idx=None) -> IKRequest:
        origin_to_base = self._current_base_origin_to_base(envs_idx=envs_idx)
        initial_angle = self._current_arm_joint_state(envs_idx=envs_idx)
        target_origin_to_end = (
            target_origin_to_end
            if torch.is_tensor(target_origin_to_end)
            else torch.as_tensor(target_origin_to_end, device=gs.device, dtype=gs.tc_float)
        )
        return IKRequest(
            frame_name=self._hsr_end_effector_frame,
            frame_to_end=self._hsr_frame_to_end,
            ref_origin_to_end=target_origin_to_end,
            origin_to_base=origin_to_base,
            initial_angle=initial_angle,
            use_joints=list(self._hsr_use_joints),
            weight=self._hsr_weight,
            linear_base_movements=self._hsr_linear_base,
            rotational_base_movements=self._hsr_rotational_base,
        )

    def _ensure_arm_qs_idx(self) -> list[int]:
        if self._hsr_arm_qs_idx_local is None:
            idx = []
            for name in JOINT_ORDER:
                joint = self.get_joint(name)
                qs_idx = joint.qs_idx_local
                if not qs_idx:
                    raise RuntimeError(f"Joint {name} has no q indices")
                idx.append(int(qs_idx[0]))
            self._hsr_arm_qs_idx_local = idx
        return self._hsr_arm_qs_idx_local

    def _ensure_base_qs_idx(self) -> list[int]:
        if self._hsr_base_qs_idx_local is None:
            try:
                joint = self.get_joint("root_joint")
            except Exception:
                self._hsr_base_qs_idx_local = []
                return self._hsr_base_qs_idx_local
            qs_idx = joint.qs_idx_local
            self._hsr_base_qs_idx_local = [int(i) for i in qs_idx] if qs_idx else []
        return self._hsr_base_qs_idx_local

    def _ensure_torso_qs_idx(self) -> int | None:
        if self._hsr_torso_qs_idx_local is None:
            try:
                joint = self.get_joint("torso_lift_joint")
            except Exception:
                self._hsr_torso_qs_idx_local = None
                return None
            qs_idx = joint.qs_idx_local
            self._hsr_torso_qs_idx_local = int(qs_idx[0]) if qs_idx else None
        return self._hsr_torso_qs_idx_local

    def _ensure_torso_dof_idx(self) -> int | None:
        if self._hsr_torso_dof_idx_local is None:
            try:
                joint = self.get_joint("torso_lift_joint")
            except Exception:
                self._hsr_torso_dof_idx_local = None
                return None
            dofs = joint.dofs_idx_local
            if isinstance(dofs, (list, tuple)):
                self._hsr_torso_dof_idx_local = int(dofs[0]) if dofs else None
            else:
                self._hsr_torso_dof_idx_local = int(dofs)
        return self._hsr_torso_dof_idx_local

    @gs.assert_built
    def inverse_kinematics(
        self,
        link,
        pos=None,
        quat=None,
        init_qpos=None,
        respect_joint_limit=True,
        max_samples=50,
        max_solver_iters=20,
        damping=0.01,
        pos_tol=5e-4,
        rot_tol=5e-3,
        pos_mask=[True, True, True],
        rot_mask=[True, True, True],
        max_step_size=0.5,
        dofs_idx_local=None,
        return_error=False,
        envs_idx=None,
    ):
        if pos is None and quat is None:
            gs.raise_exception("Either pos or quat must be provided for IK.")

        if self._solver.n_envs > 0:
            envs_idx = self._scene._sanitize_envs_idx(envs_idx)

        qs_idx_local = self._ensure_arm_qs_idx()

        n_envs = len(envs_idx) if envs_idx is not None else 1
        device = gs.device
        dtype = gs.tc_float
        if pos is None:
            pos_arr = torch.zeros((n_envs, 3), device=device, dtype=dtype)
        else:
            pos_arr = torch.as_tensor(pos, device=device, dtype=dtype)
        if quat is None:
            quat_arr = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype).repeat(pos_arr.shape[0], 1)
        else:
            quat_arr = torch.as_tensor(quat, device=device, dtype=dtype)

        if pos_arr.ndim == 1:
            pos_arr = pos_arr.reshape(1, 3)
        if quat_arr.ndim == 1:
            quat_arr = quat_arr.reshape(1, 4)
        if pos_arr.shape[0] != n_envs or quat_arr.shape[0] != n_envs:
            gs.raise_exception("First dimension of `pos` and `quat` must match envs_idx length.")

        targets = _mat4_from_pos_quat_wxyz_batch(pos_arr, quat_arr)

        if envs_idx is None:
            envs_idx = [0]

        use_base_yaw = self._hsr_use_base_yaw_ik or self._hsr_base_mode == "rotation_z"
        use_base_yaw_effective = use_base_yaw
        if use_base_yaw and n_envs == 1:
            requests = [
                self._build_request(target_origin_to_end=targets[i], envs_idx=e) for i, e in enumerate(envs_idx)
            ]
            results = []
            sol = []
            o2b = []
            o2e = []
            fallback_indices = []
            for req_i, req in enumerate(requests):
                if self._hsr_robot == "hsrb":
                    r, responses = self._hsr_ik.solve_base_yaw_ik(req)
                else:
                    r, responses = self._hsr_ik.solve_hsrc_base_yaw_ik(req)
                if r != IKResult.SUCCESS or not responses:
                    fallback_indices.append(req_i)
                    results.append(IKResult.FAIL)
                    sol.append(None)
                    o2b.append(None)
                    o2e.append(None)
                    continue
                idx = self._hsr_ik.select_closest_solution(req, responses)
                if idx < 0:
                    fallback_indices.append(req_i)
                    results.append(IKResult.FAIL)
                    sol.append(None)
                    o2b.append(None)
                    o2e.append(None)
                    continue
                chosen = responses[idx]
                results.append(IKResult.SUCCESS)
                sol.append(chosen.solution_angle)
                o2b.append(chosen.origin_to_base)
                o2e.append(chosen.origin_to_end)
            # Fallback: retry failed environments with the optimizer-based solver.
            # The analytic base-yaw solver only searches over yaw + arm joints
            # with the base x,y fixed, so it can fail when the target is outside
            # the arm's workspace.  The optimizer can also move the base.
            if fallback_indices:
                fb_targets = torch.stack([targets[i] for i in fallback_indices], dim=0)
                fb_o2b_in = torch.stack(
                    [requests[i].origin_to_base for i in fallback_indices], dim=0
                )
                fb_init = torch.stack(
                    [requests[i].initial_angle.position for i in fallback_indices], dim=0
                )
                fb_res, fb_sol, fb_o2b_out, fb_o2e = self._hsr_ik.solve_ik_batch_tensors(
                    ref_origin_to_end=fb_targets,
                    origin_to_base=fb_o2b_in,
                    init_angles=fb_init,
                    weight=self._hsr_weight,
                    robot=self._hsr_robot,
                    to_torch=True,
                )
                for fb_j, orig_i in enumerate(fallback_indices):
                    fb_ok = bool(fb_res[fb_j].item() > 0) if torch.is_tensor(fb_res) else fb_res[fb_j] == IKResult.SUCCESS
                    if fb_ok:
                        results[orig_i] = IKResult.SUCCESS
                        sol[orig_i] = JointState(
                            name=list(self._hsr_use_joints),
                            position=fb_sol[fb_j],
                        )
                        o2b[orig_i] = fb_o2b_out[fb_j]
                        o2e[orig_i] = fb_o2e[fb_j]
            # Fill remaining failures with zero placeholders
            for i in range(len(results)):
                if sol[i] is None:
                    sol[i] = JointState(
                        name=list(self._hsr_use_joints),
                        position=torch.zeros(
                            len(self._hsr_use_joints),
                            device=gs.device,
                            dtype=gs.tc_float,
                        ),
                    )
                    o2b[i] = torch.eye(4, device=gs.device, dtype=gs.tc_float)
                    o2e[i] = torch.eye(4, device=gs.device, dtype=gs.tc_float)
        else:
            origin_to_base = self._current_base_origin_to_base_batch(envs_idx=envs_idx)
            init_angles = self._current_arm_joint_state_batch(envs_idx=envs_idx)
            if use_base_yaw:
                results, sol, o2b, o2e = self._hsr_ik.solve_base_yaw_ik_batch_tensors(
                    ref_origin_to_end=targets,
                    origin_to_base=origin_to_base,
                    init_angles=init_angles,
                    weight=self._hsr_weight,
                    robot=self._hsr_robot,
                )
                # Fallback: retry failed environments with the optimizer-based solver
                if torch.is_tensor(results):
                    fail_mask = results <= 0
                else:
                    fail_mask = torch.tensor(
                        [r != IKResult.SUCCESS for r in results],
                        device=gs.device,
                        dtype=torch.bool,
                    )
                if fail_mask.any():
                    fail_idx = fail_mask.nonzero(as_tuple=True)[0]
                    fb_targets = targets[fail_idx]
                    fb_o2b_in = origin_to_base[fail_idx]
                    fb_init = init_angles[fail_idx]
                    fb_res, fb_sol, fb_o2b_out, fb_o2e = self._hsr_ik.solve_ik_batch_tensors(
                        ref_origin_to_end=fb_targets,
                        origin_to_base=fb_o2b_in,
                        init_angles=fb_init,
                        weight=self._hsr_weight,
                        robot=self._hsr_robot,
                        to_torch=True,
                    )
                    if torch.is_tensor(results):
                        results[fail_idx] = fb_res
                        sol[fail_idx] = fb_sol
                        o2b[fail_idx] = fb_o2b_out
                        o2e[fail_idx] = fb_o2e
                    else:
                        for fb_j, orig_i in enumerate(fail_idx.tolist()):
                            fb_ok = bool(fb_res[fb_j].item() > 0) if torch.is_tensor(fb_res) else fb_res[fb_j] == IKResult.SUCCESS
                            if fb_ok:
                                results[orig_i] = IKResult.SUCCESS
                                sol[orig_i] = JointState(
                                    name=list(self._hsr_use_joints),
                                    position=fb_sol[fb_j],
                                )
                                o2b[orig_i] = fb_o2b_out[fb_j]
                                o2e[orig_i] = fb_o2e[fb_j]
            else:
                results, sol, o2b, o2e = self._hsr_ik.solve_ik_batch_tensors(
                    ref_origin_to_end=targets,
                    origin_to_base=origin_to_base,
                    init_angles=init_angles,
                    weight=self._hsr_weight,
                    robot=self._hsr_robot,
                    to_torch=True,
                )
            use_base_yaw_effective = False

        qpos = self.get_qpos(envs_idx=envs_idx)
        if not torch.is_tensor(qpos):
            qpos = torch.as_tensor(qpos, device=gs.device, dtype=gs.tc_float)
        if qpos.ndim == 1:
            qpos = qpos.unsqueeze(0)

        base_qs_idx_local = self._ensure_base_qs_idx()
        torso_qs_idx_local = self._ensure_torso_qs_idx()
        cur_pos_all = None
        if base_qs_idx_local and len(base_qs_idx_local) >= 7:
            cur_pos_all = self.get_pos(envs_idx=envs_idx)
            if not torch.is_tensor(cur_pos_all):
                cur_pos_all = torch.as_tensor(cur_pos_all, device=qpos.device, dtype=qpos.dtype)
            if cur_pos_all.ndim == 1:
                cur_pos_all = cur_pos_all.reshape(1, 3)

        if use_base_yaw_effective:
            for i, (res, s, b, e) in enumerate(zip(results, sol, o2b, o2e)):
                if res == IKResult.SUCCESS:
                    for q_idx, val in zip(qs_idx_local, s.position):
                        qpos[i, q_idx] = (
                            val
                            if torch.is_tensor(val)
                            else torch.tensor(float(val), device=qpos.device, dtype=qpos.dtype)
                        )
                    if (
                        torso_qs_idx_local is not None
                        and self._hsr_torso_mimic_multiplier is not None
                        and self._hsr_torso_mimic_offset is not None
                    ):
                        arm_lift = s.position[self._hsr_arm_lift_order_idx]
                        if not torch.is_tensor(arm_lift):
                            arm_lift = torch.tensor(float(arm_lift), device=qpos.device, dtype=qpos.dtype)
                        qpos[i, torso_qs_idx_local] = (
                            arm_lift * self._hsr_torso_mimic_multiplier + self._hsr_torso_mimic_offset
                        )
                    if base_qs_idx_local:
                        base = b
                        if not torch.is_tensor(base):
                            base = torch.as_tensor(base, device=qpos.device, dtype=qpos.dtype)
                        pos = base[:3, 3]
                        quat = _quat_wxyz_from_mat3_torch(base[:3, :3])
                        if len(base_qs_idx_local) >= 7:
                            # Preserve current base height to avoid drifting off the ground.
                            cur_pos = cur_pos_all
                            pos = pos.clone()
                            pos[2] = cur_pos[i, 2]
                            qpos[i, base_qs_idx_local[0:3]] = pos
                            qpos[i, base_qs_idx_local[3:7]] = quat
        else:
            success_mask = results > 0
            if success_mask.any():
                qpos[:, qs_idx_local] = torch.where(
                    success_mask.unsqueeze(1),
                    sol.to(qpos.dtype),
                    qpos[:, qs_idx_local],
                )
                if (
                    torso_qs_idx_local is not None
                    and self._hsr_torso_mimic_multiplier is not None
                    and self._hsr_torso_mimic_offset is not None
                ):
                    arm_lift = sol[:, self._hsr_arm_lift_order_idx]
                    torso_val = arm_lift * self._hsr_torso_mimic_multiplier + self._hsr_torso_mimic_offset
                    qpos[:, torso_qs_idx_local] = torch.where(
                        success_mask,
                        torso_val,
                        qpos[:, torso_qs_idx_local],
                    )
                if base_qs_idx_local and len(base_qs_idx_local) >= 7:
                    pos = o2b[:, :3, 3]
                    quat = _quat_wxyz_from_mat3_torch_batch(o2b[:, :3, :3])
                    pos = pos.clone()
                    pos[:, 2] = cur_pos_all[:, 2]
                    qpos[:, base_qs_idx_local[0:3]] = torch.where(
                        success_mask.unsqueeze(1),
                        pos,
                        qpos[:, base_qs_idx_local[0:3]],
                    )
                    qpos[:, base_qs_idx_local[3:7]] = torch.where(
                        success_mask.unsqueeze(1),
                        quat,
                        qpos[:, base_qs_idx_local[3:7]],
                    )

        if qpos.shape[0] == 1:
            qpos = qpos[0]

        if return_error:
            if use_base_yaw_effective:
                success_mask = torch.tensor([res == IKResult.SUCCESS for res in results], device=qpos.device)
            else:
                success_mask = results.to(qpos.device) > 0
            if qpos.ndim > 1:
                if success_mask.any():
                    if use_base_yaw:
                        current_end = torch.stack(
                            [
                                e if torch.is_tensor(e) else torch.as_tensor(e, device=qpos.device, dtype=qpos.dtype)
                                for e in o2e
                            ],
                            dim=0,
                        )
                    else:
                        current_end = o2e.to(device=qpos.device, dtype=qpos.dtype)
                    error_pose = _pose_error_torch_batch(targets, current_end)
                    error_pose = torch.where(
                        success_mask.reshape(-1, 1),
                        error_pose,
                        torch.full_like(error_pose, float("inf")),
                    )
                else:
                    error_pose = torch.full((len(results), 6), float("inf"), device=qpos.device, dtype=qpos.dtype)
            else:
                if bool(success_mask[0].item()):
                    if use_base_yaw:
                        cur_end = (
                            o2e[0]
                            if torch.is_tensor(o2e[0])
                            else torch.as_tensor(o2e[0], device=qpos.device, dtype=qpos.dtype)
                        )
                    else:
                        cur_end = o2e[0].to(device=qpos.device, dtype=qpos.dtype)
                    error_pose = _pose_error_torch(targets[0], cur_end)
                else:
                    error_pose = torch.full((6,), float("inf"), device=qpos.device, dtype=qpos.dtype)
            return qpos, error_pose
        return qpos


class HSRBURDF(gs.morphs.URDF):
    def __init__(
        self,
        *,
        file: str,
        robot: str = "hsrb",
        base_mode: str = "planar",
        end_effector_frame: str = "hand_palm_link",
        use_base_controller: bool = True,
        base_control_mode: str = BaseControlMode.CONTROLLER,
        optimizer: str = "auto",
        use_base_yaw_ik: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(file=file, **kwargs)
        object.__setattr__(self, "entity_cls", HSRRigidEntity)
        object.__setattr__(self, "hsr_robot", robot)
        object.__setattr__(self, "hsr_base_mode", base_mode)
        object.__setattr__(self, "hsr_end_effector_frame", end_effector_frame)
        object.__setattr__(self, "hsr_use_base_controller", bool(use_base_controller))
        object.__setattr__(self, "hsr_base_control_mode", base_control_mode)
        object.__setattr__(self, "hsr_optimizer", optimizer)
        object.__setattr__(self, "hsr_use_base_yaw_ik", bool(use_base_yaw_ik))
