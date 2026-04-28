#!/usr/bin/env python3
# Copyright (C) 2026 Toyota Motor Corporation
import math

import genesis as gs
from genesis.utils.geom import transform_quat_by_quat

import torch

from hsr_wrapper import HsrWrapper
from hsr_genesis.hsr_rigid_entity import _yaw_from_quat_wxyz_batch


_TIMEOUT_STEPS = 2000
_BOX_SIZE = [0.08, 0.04, 0.08]


def _compute_time(current: torch.Tensor, target: torch.Tensor, v_max: torch.Tensor) -> torch.Tensor:
    delta = target - current
    dist = torch.abs(delta)
    t_axis = dist / v_max
    return torch.max(t_axis, dim=-1).values


def _compute_command(start: torch.Tensor,
                     goal: torch.Tensor,
                     total_time: torch.Tensor,
                     current_time: torch.Tensor) -> torch.Tensor:
    ratio = torch.clamp(current_time / total_time, max=1.0)
    ratio = ratio.unsqueeze(-1)
    command = start + (goal - start) * ratio
    return command


def _rotate_to_local_frame(vec: torch.Tensor, quat_wxyz: torch.Tensor) -> torch.Tensor:
    w = quat_wxyz[:, 0]
    x = quat_wxyz[:, 1]
    y = quat_wxyz[:, 2]
    z = quat_wxyz[:, 3]
    vx, vy, vz = vec[:, 0], vec[:, 1], vec[:, 2]
    # 回転行列の転置 (= 逆回転) を適用
    local_x = (1 - 2 * (y * y + z * z)) * vx + (2 * (x * y + w * z)) * vy + (2 * (x * z - w * y)) * vz
    local_y = (2 * (x * y - w * z)) * vx + (1 - 2 * (x * x + z * z)) * vy + (2 * (y * z + w * x)) * vz
    local_z = (2 * (x * z + w * y)) * vx + (2 * (y * z - w * x)) * vy + (1 - 2 * (x * x + y * y)) * vz
    return torch.stack([local_x, local_y, local_z], dim=-1)


class Environment:
    def __init__(self, n_envs: int, dt: float, show_viewer: bool):
        self.num_envs = n_envs
        self.num_obs = 17
        self.num_privileged_obs = 0
        self.num_actions = 1
        self.device = gs.device
        self.max_episode_length = _TIMEOUT_STEPS

        self._scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=dt, substeps=2),
            rigid_options=gs.options.RigidOptions(
                dt=dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(1.0 / dt),
                camera_pos=(2.0, 0.0, 4.0),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            # renderer=gs.options.renderers.BatchRenderer(use_rasterizer=True),
            show_viewer=show_viewer,
        )

        self._scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        self._object = self._scene.add_entity(
            gs.morphs.Box(
                size=_BOX_SIZE,
                fixed=False,
                collision=True,
                batch_fixed_verts=True,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=(1.0, 0.0, 0.0),
                ),
            ),
        )

        self._hsr = HsrWrapper(self._scene)
        self._scene.build(n_envs=self.num_envs, env_spacing=(2.0, 2.0))

        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self._pick_done_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=torch.bool)
        self._grasp_done_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=torch.bool)

        self._target_pos = torch.zeros((self.num_envs, 3), device=gs.device)
        self._target_quat = torch.zeros((self.num_envs, 4), device=gs.device)
        self._arm_goal = torch.zeros((self.num_envs, self._hsr.arm_dofs_num), device=gs.device)
        self._base_goal = torch.zeros((self.num_envs, 3), device=gs.device)
        self._arm_start = torch.zeros((self.num_envs, self._hsr.arm_dofs_num), device=gs.device)
        self._base_start = torch.zeros((self.num_envs, 3), device=gs.device)
        self._last_arm_command = torch.zeros((self.num_envs, self._hsr.arm_dofs_num), device=gs.device)
        self._last_base_command = torch.zeros((self.num_envs, 3), device=gs.device)
        self._motion_base_episode = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self._movement_time = torch.zeros((self.num_envs,), device=gs.device)

        self._gripper_action_current = torch.zeros((self.num_envs,), device=gs.device, dtype=torch.bool)
        self._gripper_action_previous = torch.zeros((self.num_envs,), device=gs.device, dtype=torch.bool)
        self._has_grasped = torch.zeros((self.num_envs,), device=gs.device, dtype=torch.bool)

        self._dt = dt

        self.reset()

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        if len(envs_idx) == 0:
            return
        self.episode_length_buf[envs_idx] = 0
        self._pick_done_buf[envs_idx] = False
        self._grasp_done_buf[envs_idx] = False

        self._hsr.reset(envs_idx)

        num_reset = len(envs_idx)
        random_x = (torch.rand(num_reset, device=gs.device) - 0.5) * 0.3 + 0.6
        # ちょっと全体の動きが悪いので，簡単めな場所に設置する
        random_y = (torch.rand(num_reset, device=gs.device) - 0.5) * 0.1
        random_z = torch.ones(num_reset, device=gs.device) * 0.025
        random_pos = torch.stack([random_x, random_y, random_z], dim=-1)

        q_downward = torch.tensor([0.0, 1.0, 0.0, 0.0], device=gs.device).repeat(num_reset, 1)
        random_yaw = (torch.rand(num_reset, device=gs.device) * 2 * math.pi - math.pi) * 0.1
        q_yaw = torch.stack(
            [
                torch.cos(random_yaw / 2),
                torch.zeros(num_reset, device=gs.device),
                torch.zeros(num_reset, device=gs.device),
                torch.sin(random_yaw / 2),
            ],
            dim=-1,
        )
        goal_yaw = transform_quat_by_quat(q_yaw, q_downward)

        self._object.set_pos(random_pos, envs_idx=envs_idx)
        self._object.set_quat(goal_yaw, envs_idx=envs_idx)

        self._target_pos[envs_idx] = torch.zeros((num_reset, 3), device=gs.device)
        self._target_quat[envs_idx] = torch.zeros((num_reset, 4), device=gs.device)
        self._arm_goal[envs_idx] = torch.zeros((num_reset, self._hsr.arm_dofs_num), device=gs.device)
        self._base_goal[envs_idx] = torch.zeros((num_reset, 3), device=gs.device)
        self._arm_start[envs_idx] = torch.zeros((num_reset, self._hsr.arm_dofs_num), device=gs.device)
        self._base_start[envs_idx] = torch.zeros((num_reset, 3), device=gs.device)
        self._last_arm_command[envs_idx] = torch.zeros((num_reset, self._hsr.arm_dofs_num), device=gs.device)
        self._last_base_command[envs_idx] = torch.zeros((num_reset, 3), device=gs.device)
        self._motion_base_episode[envs_idx] = torch.zeros((num_reset,), device=gs.device, dtype=gs.tc_int)
        self._movement_time[envs_idx] = torch.zeros((num_reset,), device=gs.device)

        self._gripper_action_current[envs_idx] = False
        self._gripper_action_previous[envs_idx] = False
        self._has_grasped[envs_idx] = False

    def reset(self) -> tuple[torch.Tensor, dict]:
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))

        obs, extras = self.get_observations()
        return obs, extras

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        envs_idx = torch.arange(self.num_envs, device=gs.device)

        recalculate_idx = envs_idx[~self._pick_done_buf & (self.episode_length_buf % 40 == 0)]
        if len(recalculate_idx) > 0:
            object_pos = self._object.get_pos()
            object_pos[:, 2] += 0.09
            object_quat = self._object.get_quat()
            target_quat = torch.zeros_like(object_quat)
            target_quat[:, 1] = object_quat[:, 1]
            target_quat[:, 2] = object_quat[:, 2]

            arm_goal, base_goal = self._hsr.inverse_kinematics(
                object_pos[recalculate_idx], target_quat[recalculate_idx], recalculate_idx)
            base_goal_xyt = torch.zeros((base_goal.shape[0], 3), device=base_goal.device)
            base_goal_xyt[:, :2] = base_goal[:, :2]
            base_goal_xyt[:, 2] = _yaw_from_quat_wxyz_batch(base_goal[:, 3:7])

            arm_time = _compute_time(self._hsr.arm_positions[recalculate_idx], arm_goal,
                                     v_max=self._hsr.arm_vmax)
            base_time = _compute_time(self._hsr.base_positions[recalculate_idx], base_goal_xyt,
                                      v_max=self._hsr.base_vmax)

            # TODO(Takeshita) 別クラスに切り出す
            self._target_pos[recalculate_idx] = object_pos[recalculate_idx]
            self._target_quat[recalculate_idx] = target_quat[recalculate_idx]

            self._arm_goal[recalculate_idx] = arm_goal
            self._base_goal[recalculate_idx] = base_goal_xyt

            # 動きをなめらかにするためには，commandの連続性が大事
            self._arm_start[recalculate_idx] = self._last_arm_command[recalculate_idx]
            self._base_start[recalculate_idx] = self._last_base_command[recalculate_idx]

            self._motion_base_episode[recalculate_idx] = self.episode_length_buf[recalculate_idx].clone()
            self._movement_time[recalculate_idx] = torch.max(arm_time, base_time)

        self.episode_length_buf += 1

        # アクションは開閉のみ
        self._gripper_action_previous = self._gripper_action_current.clone()

        a = actions.squeeze(-1)  # (num_envs,)
        close_idx = self._gripper_action_previous
        if close_idx.any():
            self._gripper_action_current[close_idx] = ~(a[close_idx] < -0.5)
        open_idx = ~self._gripper_action_previous
        if open_idx.any():
            self._gripper_action_current[open_idx] = (a[open_idx] > 0.5)

        CLOSE_POS = -1.0
        OPEN_POS = 1.0
        gripper_target = torch.where(
            self._gripper_action_current,
            torch.tensor(CLOSE_POS, device=gs.device),
            torch.tensor(OPEN_POS, device=gs.device)
        )
        self._hsr.control_gripper_position(gripper_target, envs_idx=envs_idx)

        head_pos = torch.zeros((len(envs_idx), self._hsr.head_dofs_num), device=gs.device)
        self._hsr.control_head_positions(head_pos, envs_idx=envs_idx)

        time_since_motion_start = (self.episode_length_buf - self._motion_base_episode) * self._dt

        if not self._pick_done_buf.all():
            hand_pos, hand_quat = self._hsr.hand_pose
            pos_distance = torch.norm(hand_pos - self._target_pos, dim=-1)

            timeout = time_since_motion_start > self._movement_time + self._dt

            quat_distance = 1.0 - torch.abs(torch.sum(hand_quat * self._target_quat, dim=-1))
            done_idx = envs_idx[((pos_distance < 0.02) & (quat_distance < 0.01) & (~self._pick_done_buf)) | timeout]
            self._pick_done_buf[done_idx] = True

            self._arm_goal[done_idx] = torch.zeros((len(done_idx), self._hsr.arm_dofs_num), device=gs.device)
            self._base_goal[done_idx] = torch.zeros((len(done_idx), 3), device=gs.device)

            arm_time = _compute_time(self._hsr.arm_positions[done_idx],
                                     self._arm_goal[done_idx],
                                     v_max=self._hsr.arm_vmax)
            base_time = _compute_time(self._hsr.base_positions[done_idx],
                                      self._base_goal[done_idx],
                                      v_max=self._hsr.base_vmax)
            self._movement_time[done_idx] = torch.max(arm_time, base_time)
            self._arm_start[done_idx] = self._last_arm_command[done_idx]
            self._base_start[done_idx] = self._last_base_command[done_idx]
            self._motion_base_episode[done_idx] = self.episode_length_buf[done_idx].clone()

        time_since_motion_start = (self.episode_length_buf - self._motion_base_episode) * self._dt
        arm_command = _compute_command(self._arm_start, self._arm_goal, self._movement_time, time_since_motion_start)
        base_command = _compute_command(self._base_start, self._base_goal, self._movement_time, time_since_motion_start)
        self._hsr.control_arm_positions(arm_command, envs_idx=envs_idx)
        self._hsr.control_base_positions(base_command, envs_idx=envs_idx)

        self._last_arm_command = arm_command
        self._last_base_command = base_command

        self._scene.step()

        is_done_idx = self.is_episode_complete()
        if len(is_done_idx) > 0:
            self.reset_idx(is_done_idx)

        is_done_list = torch.zeros(self.num_envs, device=gs.device, dtype=torch.bool)
        is_done_list[is_done_idx] = True

        obs, extras = self.get_observations()
        reward = self._calculate_reward()
        return obs, reward, is_done_list, extras

    def is_episode_complete(self) -> torch.Tensor:
        time_from_motion_start = (self.episode_length_buf - self._motion_base_episode) * self._dt
        move_done = time_from_motion_start > self._movement_time

        timeout = self.episode_length_buf >= _TIMEOUT_STEPS

        envs_idx = torch.arange(self.num_envs, device=gs.device)
        return envs_idx[(self._pick_done_buf & move_done) | timeout]

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        hand_pos, hand_quat = self._hsr.hand_pose
        obj_pos = self._object.get_pos()
        obj_quat = self._object.get_quat()
        gripper_pos = self._hsr.gripper_positions
        obs_components = [
            gripper_pos,
            hand_pos - obj_pos,
            hand_quat,
            obj_pos,
            obj_quat,
        ]
        obs_tensor = torch.cat(obs_components, dim=-1)
        extras = {"observations": {"critic": obs_tensor}}
        return obs_tensor, extras

    def get_privileged_observations(self) -> None:
        return None

    def _calculate_reward(self) -> torch.Tensor:
        obj_pos = self._object.get_pos()
        hand_pos, hand_quat = self._hsr.hand_pose
        l_finger_pos, l_finger_quat = self._hsr.left_finger_tip_pose
        r_finger_pos, r_finger_quat = self._hsr.right_finger_tip_pose

        hand_to_obj = _rotate_to_local_frame(obj_pos - hand_pos, hand_quat)
        l_finger_to_obj = _rotate_to_local_frame(obj_pos - l_finger_pos, l_finger_quat)
        r_finger_to_obj = _rotate_to_local_frame(obj_pos - r_finger_pos, r_finger_quat)

        is_near = ((torch.abs(hand_to_obj[:, 0]) <= 0.03) & (torch.abs(hand_to_obj[:, 1]) <= 0.03)
                   & (torch.abs(hand_to_obj[:, 2]) > 0.0) & (torch.abs(hand_to_obj[:, 2]) < 0.12)
                   & (torch.abs(l_finger_to_obj[:, 2]) > 0.01)
                   & (torch.abs(r_finger_to_obj[:, 2]) > 0.01))

        close_near_reward = (is_near & self._gripper_action_current).float()

        close_far_penalty = (self._gripper_action_current & ~is_near).float()

        lift_height = obj_pos[:, 2]
        lift_reward = (lift_height > 0.05).float()

        gripper_change_penalty = (self._gripper_action_current != self._gripper_action_previous).float()

        return 2.0 * lift_reward \
            + 1.0 * close_near_reward \
            - 0.004 * close_far_penalty \
            - 0.2 * gripper_change_penalty
