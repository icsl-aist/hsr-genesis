#!/usr/bin/env python3
# Copyright (C) 2026 Toyota Motor Corporation
from pathlib import Path
from typing import Optional

import xml.etree.ElementTree as ET

import genesis as gs

from hsr_genesis.base_controller import _vehicle_inverse_kernel
from hsr_genesis.gripper_controller import HSRBGenesisGripperInterface
from hsr_genesis.hsr_rigid_entity import (
    _yaw_from_quat_wxyz_batch,
    HSRBURDF,
)

import torch


# hsr-genesis/examples/tutorials/hello_hsr_sensor.py
def _sensor_reference_links_from_urdf(urdf_path: str) -> list[str]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    refs: list[str] = []
    for gazebo in root.findall("gazebo"):
        ref = gazebo.attrib.get("reference")
        if ref and ref not in refs:
            refs.append(ref)
    return refs


class JointWithMimic:
    def __init__(self, entity: HSRBURDF, joint_name: str, mimic_list: list[HSRBGenesisGripperInterface._MimicJoint]):
        self._entity = entity
        self._qs_idx_local_joint = entity.get_joint(joint_name).qs_idx_local
        self._qs_idx_local_mimics = [entity.get_joint(mimic.joint).qs_idx_local for mimic in mimic_list]
        self._multipliers = [mimic.multiplier for mimic in mimic_list]
        self._offsets = [mimic.offset for mimic in mimic_list]

    def set_position(self, pos: torch.Tensor, envs_idx: torch.Tensor):
        self._entity.set_qpos(pos, qs_idx_local=self._qs_idx_local_joint, envs_idx=envs_idx)
        for qs_idx_local_mimic, multiplier, offset in zip(self._qs_idx_local_mimics, self._multipliers, self._offsets):
            mimic_pos = pos * multiplier + offset
            self._entity.set_qpos(mimic_pos, qs_idx_local=qs_idx_local_mimic, envs_idx=envs_idx)


class HsrWrapper:
    def __init__(self, scene: gs.Scene, urdf_path: Optional[str] = None):
        if urdf_path is None:
            URDF_PATH = Path(__file__).resolve().parents[2] / "data" / "urdf" / "hsrb4s.urdf"
            urdf_path = str(URDF_PATH)

        self._hsr = scene.add_entity(
            HSRBURDF(
                file=urdf_path,
                fixed=False,
                recompute_inertia=True,
                links_to_keep=_sensor_reference_links_from_urdf(urdf_path) + ["hand_palm_link"],
                robot="hsrb",
                base_mode="planar",
                end_effector_frame="hand_palm_link",
                use_base_controller=True,
                optimizer="gpu",
            )
        )

    def reset(self, envs_idx: torch.Tensor):
        self._arm_vmax = torch.tensor([0.15, 1.0, 1.5, 1.5, 1.5], device=gs.device)
        self._base_vmax = torch.tensor([0.15, 0.15, 0.5], device=gs.device)

        self._hsr._ensure_torso_dof_idx()

        self._hsr._hsr_apply_default_collision_disable()
        self._hsr._hsr_apply_passive_wheel_friction()
        self._hsr._hsr_apply_high_friction_links()
        self._hsr._hsr_apply_default_gains()

        self.set_arm_positions(torch.zeros((len(envs_idx), self.arm_dofs_num), device=gs.device), envs_idx=envs_idx)
        self.control_head_positions(torch.zeros((len(envs_idx), self.head_dofs_num), device=gs.device),
                                    envs_idx=envs_idx)

        self.set_base_positions(torch.zeros((len(envs_idx), 3), device=gs.device), envs_idx=envs_idx)

        self._gripper = self._hsr.get_gripper_batched()
        # TODO(Takeshita) Gripperクラスに移す
        self._hand_palm_link = self._hsr.get_link("hand_palm_link")
        self._left_finger_tip_frame = self._hsr.get_link("hand_l_finger_tip_frame")
        self._right_finger_tip_frame = self._hsr.get_link("hand_r_finger_tip_frame")

        self.control_gripper_position(torch.full((len(envs_idx),), 1.0, device=gs.device), envs_idx=envs_idx)
        self._hand_motor_joint = JointWithMimic(
            self._hsr, "hand_motor_joint", self._gripper.hw._mimic_children["hand_motor_joint"])
        self._left_spring_joint = JointWithMimic(
            self._hsr, "hand_l_spring_proximal_joint", self._gripper.hw._mimic_children["hand_l_spring_proximal_joint"])
        self._right_spring_joint = JointWithMimic(
            self._hsr, "hand_r_spring_proximal_joint", self._gripper.hw._mimic_children["hand_r_spring_proximal_joint"])
        self.set_gripper_motor_position(torch.full((len(envs_idx),), 1.0, device=gs.device), envs_idx=envs_idx)
        self.set_gripper_spring_position(torch.full((len(envs_idx),), 0.0, device=gs.device),
                                         torch.full((len(envs_idx),), 0.0, device=gs.device), envs_idx=envs_idx)

    def control_arm_positions(self, pos: torch.Tensor, envs_idx: torch.Tensor):
        self._hsr.control_dofs_position(pos,
                                        dofs_idx_local=self._hsr._hsr_arm_dofs_idx_local,
                                        envs_idx=envs_idx)

        arm_lift = pos[:, self._hsr._hsr_arm_lift_order_idx]
        torso_pos = arm_lift * self._hsr._hsr_torso_mimic_multiplier + self._hsr._hsr_torso_mimic_offset
        self._hsr.control_dofs_position(torso_pos.reshape(-1, 1),
                                        dofs_idx_local=[self._hsr._hsr_torso_dof_idx_local],
                                        envs_idx=envs_idx)

    def set_arm_positions(self, pos: torch.Tensor, envs_idx: torch.Tensor):
        self._hsr.set_qpos(pos, qs_idx_local=self._hsr._ensure_arm_qs_idx(), envs_idx=envs_idx)

        arm_lift = pos[:, self._hsr._hsr_arm_lift_order_idx]
        torso_pos = arm_lift * self._hsr._hsr_torso_mimic_multiplier + self._hsr._hsr_torso_mimic_offset
        self._hsr.set_qpos(torso_pos.reshape(-1, 1), qs_idx_local=[self._hsr._ensure_torso_qs_idx()], envs_idx=envs_idx)

    def control_gripper_position(self, pos: torch.Tensor, envs_idx: torch.Tensor):
        self._gripper.hw.command_motor_position(position=pos, envs_idx=envs_idx)

    def set_gripper_motor_position(self, pos: torch.Tensor, envs_idx: torch.Tensor):
        self._hand_motor_joint.set_position(pos, envs_idx=envs_idx)

    def set_gripper_spring_position(self, left_pos: torch.Tensor, right_pos: torch.Tensor, envs_idx: torch.Tensor):
        self._left_spring_joint.set_position(left_pos, envs_idx=envs_idx)
        self._right_spring_joint.set_position(right_pos, envs_idx=envs_idx)

    def control_head_positions(self, pos: torch.Tensor, envs_idx: torch.Tensor):
        self._hsr.control_dofs_position(pos,
                                        dofs_idx_local=self._hsr._hsr_head_dofs_idx_local,
                                        envs_idx=envs_idx)

    def control_base_positions(self, pos: torch.Tensor, envs_idx: torch.Tensor):
        self.control_base_velocities(pos - self.base_positions, envs_idx=envs_idx)

    def control_base_velocities(self, vel: torch.Tensor, envs_idx: torch.Tensor):
        control = self._hsr.get_base_controller()
        steer_pos = self._hsr.get_dofs_position(envs_idx=envs_idx, dofs_idx_local=control.steer_dof_idx_local)

        out = torch.zeros((envs_idx.numel(), 3), device=gs.device)
        _vehicle_inverse_kernel(
            int(envs_idx.numel()),
            vel,
            steer_pos.squeeze(-1),
            float(control.config.wheel_radius),
            float(control.config.wheel_separation),
            float(control.config.wheel_offset),
            float(control.config.yaw_velocity_limit),
            float(control.config.wheel_velocity_limit),
            out,
        )
        dofs_idx_local = control.wheel_drive_dofs_idx_local + [control.steer_dof_idx_local]
        self._hsr.control_dofs_velocity(out,
                                        dofs_idx_local=dofs_idx_local,
                                        envs_idx=envs_idx)

    def set_base_positions(self, pos: torch.Tensor, envs_idx: torch.Tensor):
        pos_local = torch.zeros((len(envs_idx), 3), device=gs.device)
        pos_local[:, :2] = pos[:, :2]

        quat = torch.zeros((len(envs_idx), 4), device=gs.device)
        yaw = pos[:, 2]
        half_yaw = 0.5 * yaw
        quat[:, 0] = torch.cos(half_yaw)
        quat[:, 3] = torch.sin(half_yaw)

        self._hsr.set_pos(pos_local, envs_idx=envs_idx)
        self._hsr.set_quat(quat, envs_idx=envs_idx)

    @property
    def arm_vmax(self) -> torch.Tensor:
        return self._arm_vmax

    @property
    def base_vmax(self) -> torch.Tensor:
        return self._base_vmax

    @property
    def arm_dofs_num(self) -> int:
        return len(self._hsr._hsr_arm_dofs_idx_local)

    @property
    def head_dofs_num(self) -> int:
        # pan and tilt
        return len(self._hsr._hsr_head_dofs_idx_local)

    @property
    def arm_positions(self) -> torch.Tensor:
        return self._hsr.get_dofs_position(dofs_idx_local=self._hsr._hsr_arm_dofs_idx_local)

    @property
    def gripper_positions(self) -> torch.Tensor:
        motor_pos = self._gripper.hw.get_motor_position()
        left_spring_pos = self._gripper.hw.get_left_spring_position()
        right_spring_pos = self._gripper.hw.get_right_spring_position()
        return torch.stack([motor_pos, left_spring_pos, right_spring_pos], dim=-1)

    @property
    def base_positions(self) -> torch.Tensor:
        """Positions x, y and yaw"""
        pos = self._hsr.get_pos()
        quat = self._hsr.get_quat()
        yaw = _yaw_from_quat_wxyz_batch(quat)
        base_positions = torch.stack([pos[:, 0], pos[:, 1], yaw], dim=-1)
        return base_positions

    def inverse_kinematics(self, pos: torch.Tensor, quat: torch.Tensor, envs_idx: torch.Tensor):
        result = self._hsr.inverse_kinematics(link=self._hand_palm_link,
                                              pos=pos,
                                              quat=quat,
                                              envs_idx=envs_idx)
        return result[:, self._hsr._hsr_arm_qs_idx_local], result[:, self._hsr._hsr_base_qs_idx_local]

    @property
    def hand_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos = self._hsr.get_links_pos(self._hand_palm_link.idx_local)
        quat = self._hsr.get_links_quat(self._hand_palm_link.idx_local)
        return pos.squeeze(1), quat.squeeze(1)

    @property
    def left_finger_tip_pose(self):
        pos = self._hsr.get_links_pos(self._left_finger_tip_frame.idx_local)
        quat = self._hsr.get_links_quat(self._left_finger_tip_frame.idx_local)
        return pos.squeeze(1), quat.squeeze(1)

    @property
    def right_finger_tip_pose(self):
        pos = self._hsr.get_links_pos(self._right_finger_tip_frame.idx_local)
        quat = self._hsr.get_links_quat(self._right_finger_tip_frame.idx_local)
        return pos.squeeze(1), quat.squeeze(1)
