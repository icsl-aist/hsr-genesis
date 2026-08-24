"""Batched Gymnasium env for PPO training on HSR pick.

Wraps N Genesis parallel envs as a single Gymnasium env with batched
obs/action. In IK-guided mode, the policy outputs 9D residual corrections
(delta arm 5D + delta base 3D + gripper effort 1D) added to IK-computed
targets. In direct-policy mode, the same action space drives arm/base step
targets without an IK reference trajectory.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

import genesis as gs

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn

from ycb_pick_ik_parallel import HSRPickEnv, GRIPPER_EFFORT, HAND_QUAT, LIFT_THRESHOLD
from ik_planner import IKPlanner, IKPlan
from curriculum import CurriculumManager

# Observation dimensions
OBS_DIM = 32
ACTION_DIM = 9

# Residual action scales
ARM_RESIDUAL_SCALE = 0.1    # rad
BASE_RESIDUAL_SCALE = 0.05  # m for xy, rad for yaw
GRIPPER_EFFORT_SCALE = 4.0  # N
GRIPPER_EFFORT_OFFSET = 2.0  # N (so output maps to [0, 8] after tanh)


class HSRPickRLEnv(gym.Env):
    """Batched RL env: N Genesis parallel envs as one Gymnasium env.

    Observation (32D per env):
        - Object pos (robot frame) [3]
        - Object yaw (robot frame) [1]
        - EE pos [3]
        - Arm joints [5]
        - Base xy-yaw [3]
        - Gripper motor pos [1]
        - Curriculum policy_weight [1]
        - Step progress [1]
        - Previous action [9]
        - Phase one-hot [5]

    Action (9D per env, tanh-squashed):
        - delta_arm [5] * 0.1 rad
        - delta_base [3] * 0.05 m / 0.1 rad
        - gripper_effort [1] * 4.0 + 2.0 -> [0, 8] N
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        n_envs: int = 64,
        object_name: str = "ycb_061_foam_brick",
        seed: int = 0,
        settle_steps: int = 30,
        curriculum: CurriculumManager | None = None,
        use_ik_guidance: bool = True,
        vis_options_overrides: dict | None = None,
        camera_config: dict | None = None,
        obj_radius_range: tuple[float, float] | None = None,
    ) -> None:
        super().__init__()
        self.n_envs = n_envs
        self.settle_steps = settle_steps
        self.curriculum = curriculum or CurriculumManager()
        self.use_ik_guidance = use_ik_guidance

        # Build the underlying HSRPickEnv (no viewer for training)
        self._pick_env = HSRPickEnv(
            n_envs=n_envs,
            object_name=object_name,
            show_viewer=False,
            seed=seed,
            disable_visualizer=True,
            vis_options_overrides=vis_options_overrides,
            camera_config=camera_config,
            obj_radius_range=obj_radius_range,
        )
        self.dt = self._pick_env.dt
        self.envs_all = self._pick_env.envs_all

        # IK planner (uses CMA-ES optimized params if available)
        self._planner = IKPlanner(object_name=object_name) if use_ik_guidance else None

        # Gymnasium spaces (batched: shape = (n_envs, dim))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n_envs, OBS_DIM), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(n_envs, ACTION_DIM), dtype=np.float32,
        )

        # Episode state
        self._plan: IKPlan | None = None
        self._step = 0
        self._prev_action = torch.zeros(n_envs, ACTION_DIM, device=gs.device, dtype=gs.tc_float)
        self._success = torch.zeros(n_envs, device=gs.device, dtype=torch.bool)
        self._success_reward_given = torch.zeros(n_envs, device=gs.device, dtype=torch.bool)

        # GPU obs cache: keep GPU tensor reference before .cpu().numpy()
        # so internal callers can access without a CPU round-trip.
        self._last_obs_gpu: torch.Tensor | None = None
        # Pinned memory buffer for async GPU→CPU obs transfer.
        self._use_pinned = str(gs.device) != "cpu"
        if self._use_pinned:
            self._obs_buf = torch.zeros(
                n_envs, OBS_DIM, device="cpu", dtype=gs.tc_float, pin_memory=True,
            )

    @property
    def camera(self):
        """The offscreen camera, or None if no camera_config was provided."""
        return self._pick_env.camera

    def _yaw_from_quat(self, quat: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return torch.atan2(siny_cosp, cosy_cosp)

    def _policy_weight(self) -> float:
        return self.curriculum.policy_weight if self.use_ik_guidance else 1.0

    def get_obs(self) -> np.ndarray:
        """Build 32D observation for all envs. Returns (n_envs, 32) numpy."""
        env = self._pick_env

        # Object pos in robot frame
        obj_pos = env._obj_pos()  # (N, 3) world
        base_pos = env.hsr.get_pos(envs_idx=self.envs_all)
        base_quat = env.hsr.get_quat(envs_idx=self.envs_all)
        if base_pos.ndim == 1:
            base_pos = base_pos.unsqueeze(0)
            base_quat = base_quat.unsqueeze(0)
        base_yaw = self._yaw_from_quat(base_quat)
        # Transform obj_pos to robot frame (subtract base, rotate by -yaw)
        dx = obj_pos[:, 0] - base_pos[:, 0]
        dy = obj_pos[:, 1] - base_pos[:, 1]
        cos_yaw = torch.cos(-base_yaw)
        sin_yaw = torch.sin(-base_yaw)
        obj_x_robot = dx * cos_yaw - dy * sin_yaw
        obj_y_robot = dx * sin_yaw + dy * cos_yaw
        obj_z_robot = obj_pos[:, 2] - base_pos[:, 2]
        obj_pos_robot = torch.stack([obj_x_robot, obj_y_robot, obj_z_robot], dim=-1)

        # Object yaw relative to base
        obj_quat = env.obj.get_quat(envs_idx=self.envs_all)
        if obj_quat.ndim == 1:
            obj_quat = obj_quat.unsqueeze(0)
        obj_yaw = self._yaw_from_quat(obj_quat)
        obj_yaw_rel = obj_yaw - base_yaw

        # EE pos
        ee_pos = env.ee_link.get_pos(envs_idx=self.envs_all)
        if ee_pos.ndim == 1:
            ee_pos = ee_pos.unsqueeze(0)

        # Arm joints
        arm_pos = env.hsr.get_dofs_position(
            dofs_idx_local=env.arm_dofs_idx, envs_idx=self.envs_all,
        )
        if arm_pos.ndim == 1:
            arm_pos = arm_pos.unsqueeze(0)

        # Base xy-yaw
        base_xy_yaw = torch.stack([base_pos[:, 0], base_pos[:, 1], base_yaw], dim=-1)

        # Gripper motor position
        motor_pos = env.hsr.get_dofs_position(
            dofs_idx_local=[env.motor_idx], envs_idx=self.envs_all,
        )
        if motor_pos.ndim == 1:
            motor_pos = motor_pos.unsqueeze(0)
        motor_pos = motor_pos.squeeze(-1) if motor_pos.shape[-1] == 1 else motor_pos[:, 0]

        # Curriculum blend
        pw = torch.full((self.n_envs,), self._policy_weight(),
                        device=gs.device, dtype=gs.tc_float)

        # Step progress
        progress = torch.full((self.n_envs,), self._step / IKPlanner.max_steps(),
                              device=gs.device, dtype=gs.tc_float)

        # Phase one-hot
        phase = IKPlanner.get_phase(self._step)
        phase_onehot = torch.zeros(self.n_envs, 5, device=gs.device, dtype=gs.tc_float)
        phase_onehot[:, phase] = 1.0

        # Concatenate all
        obs = torch.cat([
            obj_pos_robot,           # 3
            obj_yaw_rel.unsqueeze(-1),  # 1
            ee_pos,                  # 3
            arm_pos,                 # 5
            base_xy_yaw,             # 3
            motor_pos.unsqueeze(-1), # 1
            pw.unsqueeze(-1),        # 1
            progress.unsqueeze(-1),  # 1
            self._prev_action,       # 9
            phase_onehot,            # 5
        ], dim=-1)  # Total: 32

        self._last_obs_gpu = obs  # cache GPU tensor for internal access
        if self._use_pinned:
            # Async copy into pinned buffer; synchronize before reading.
            self._obs_buf.copy_(obs, non_blocking=True)
            torch.cuda.synchronize()
            return self._obs_buf.numpy()
        return obs.detach().cpu().numpy()

    def reset(self, *, seed=None, options=None):
        """Reset all envs: new object placement and optional IK plan."""
        super().reset(seed=seed)
        env = self._pick_env

        # Reset underlying env (random object placement + settle)
        env.reset(settle_steps=self.settle_steps)

        # Compute IK plan for this episode when guidance is enabled.
        if self.use_ik_guidance:
            assert self._planner is not None
            self._plan = self._planner.plan(env)
            ik_success = self._plan.ik_success.cpu().numpy()
        else:
            self._plan = None
            ik_success = np.zeros(self.n_envs, dtype=bool)

        # Reset episode state
        self._step = 0
        self._prev_action = torch.zeros(self.n_envs, ACTION_DIM, device=gs.device, dtype=gs.tc_float)
        self._success = torch.zeros(self.n_envs, device=gs.device, dtype=torch.bool)
        self._success_reward_given = torch.zeros(self.n_envs, device=gs.device, dtype=torch.bool)

        # Set initial trajectory (approach phase) and open hand
        self._set_phase_trajectory(0)
        env.hsr.control_dofs_position(
            env.hand_open, dofs_idx_local=[env.motor_idx],
            envs_idx=self.envs_all,
        )

        obs = self.get_obs()
        info = {"ik_success": ik_success}
        return obs, info

    def retarget(self, settle_steps: int | None = None) -> np.ndarray:
        """Move object to a new random position without resetting the robot.

        Opens the gripper (drops current object), places the object at a new
        random pose relative to the robot's *current* base position, settles
        briefly while holding the robot at its current pose, then recomputes
        the IK plan and resets the episode state.
        The robot keeps its current base/arm position — it will drive to the
        new target continuously.

        Returns the new observation.
        """
        env = self._pick_env
        if settle_steps is None:
            settle_steps = self.settle_steps

        # Open gripper to release the current object
        env.hsr.control_dofs_position(
            env.hand_open, dofs_idx_local=[env.motor_idx],
            envs_idx=self.envs_all,
        )

        # Place object at new random position, offset by robot's current base xy
        # so the object is always in front of the robot regardless of where it drove
        base_pos = env.hsr.get_pos(envs_idx=env.envs_all)  # (n_envs, 3)
        pos, quat = env._random_object_pose()
        pos[:, 0] += base_pos[:, 0]
        pos[:, 1] += base_pos[:, 1]
        env.obj.set_pos(pos, envs_idx=env.envs_all, zero_velocity=True, relative=False)
        env.obj.set_quat(quat, envs_idx=env.envs_all, zero_velocity=True, relative=False)

        # Settle object while holding robot at current pose
        env._settle(settle_steps)
        env.obj_init_z = env._obj_pos()[:, 2]

        # Recompute IK plan for the new object position, starting from current pose
        if self.use_ik_guidance:
            assert self._planner is not None
            cur_qpos = env.hsr.get_qpos(envs_idx=env.envs_all)
            if cur_qpos.ndim == 1:
                cur_qpos = cur_qpos.unsqueeze(0)
            self._plan = self._planner.plan(env, init_qpos=cur_qpos)

        # Reset episode state (but keep robot pose)
        self._step = 0
        self._prev_action = torch.zeros(self.n_envs, ACTION_DIM, device=gs.device, dtype=gs.tc_float)
        self._success = torch.zeros(self.n_envs, device=gs.device, dtype=torch.bool)
        self._success_reward_given = torch.zeros(self.n_envs, device=gs.device, dtype=torch.bool)

        # Set approach trajectory for the new target
        self._set_phase_trajectory(0)
        env.hsr.control_dofs_position(
            env.hand_open, dofs_idx_local=[env.motor_idx],
            envs_idx=self.envs_all,
        )

        obs = self.get_obs()
        return obs

    def _set_phase_trajectory(self, phase: int):
        """Set the whole-body trajectory for the current phase, using IK targets."""
        if not self.use_ik_guidance or self._plan is None:
            return

        env = self._pick_env
        arm_target, base_target = IKPlanner.get_phase_targets(self._plan, phase)

        from hsr_genesis.hsr_rigid_entity import JointTrajectory
        from hsr_genesis.base_controller import Trajectory
        from hsr_genesis.analytic_ik import JOINT_ORDER

        # Phase duration
        if phase == 0:
            duration = IKPlanner.approach_steps * self.dt
        elif phase == 1:
            duration = IKPlanner.descend_steps * self.dt
        elif phase == 2:
            duration = IKPlanner.grasp_steps * self.dt
        else:
            duration = IKPlanner.lift_steps * self.dt

        t = torch.tensor([duration], device=gs.device, dtype=gs.tc_float)
        # API-bound: set_whole_body_trajectory_batched expects list[JointTrajectory] per env.
        arm_trajs = [
            JointTrajectory(
                positions=arm_target[i].unsqueeze(0),
                time_from_start=t,
                joint_names=list(JOINT_ORDER),
            )
            for i in range(self.n_envs)
        ]
        base_trajs = [
            Trajectory(
                positions=base_target[i].unsqueeze(0),
                time_from_start=t,
            )
            for i in range(self.n_envs)
        ]
        env.hsr.set_whole_body_trajectory_batched(
            arm_trajectory=arm_trajs,
            base_trajectory=base_trajs,
            envs_idx=self.envs_all,
            start_time=None,
        )

    def _direct_targets_from_action(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert direct-policy action into next arm/base targets."""
        env = self._pick_env

        arm_pos = env.hsr.get_dofs_position(
            dofs_idx_local=env.arm_dofs_idx, envs_idx=self.envs_all,
        )
        if arm_pos.ndim == 1:
            arm_pos = arm_pos.unsqueeze(0)

        base_pos = env.hsr.get_pos(envs_idx=self.envs_all)
        base_quat = env.hsr.get_quat(envs_idx=self.envs_all)
        if base_pos.ndim == 1:
            base_pos = base_pos.unsqueeze(0)
            base_quat = base_quat.unsqueeze(0)
        base_yaw = self._yaw_from_quat(base_quat)
        base_xy_yaw = torch.stack([base_pos[:, 0], base_pos[:, 1], base_yaw], dim=-1)

        arm_target = (arm_pos + action[:, :5] * ARM_RESIDUAL_SCALE).clamp(-math.pi, math.pi)
        base_target = base_xy_yaw + action[:, 5:8] * BASE_RESIDUAL_SCALE
        return arm_target, base_target

    def _set_direct_action_trajectory(self, action: torch.Tensor):
        """Drive arm and base directly from policy action without IK guidance."""
        env = self._pick_env
        arm_target, base_target = self._direct_targets_from_action(action)

        from hsr_genesis.hsr_rigid_entity import JointTrajectory
        from hsr_genesis.base_controller import Trajectory
        from hsr_genesis.analytic_ik import JOINT_ORDER

        t = torch.tensor([self.dt], device=gs.device, dtype=gs.tc_float)
        # API-bound: set_whole_body_trajectory_batched expects list[JointTrajectory] per env.
        arm_trajs = [
            JointTrajectory(
                positions=arm_target[i].unsqueeze(0),
                time_from_start=t,
                joint_names=list(JOINT_ORDER),
            )
            for i in range(self.n_envs)
        ]
        base_trajs = [
            Trajectory(
                positions=base_target[i].unsqueeze(0),
                time_from_start=t,
            )
            for i in range(self.n_envs)
        ]
        env.hsr.set_whole_body_trajectory_batched(
            arm_trajectory=arm_trajs,
            base_trajectory=base_trajs,
            envs_idx=self.envs_all,
            start_time=None,
        )

    def _apply_residual(self, action: torch.Tensor):
        """Apply policy residual by overriding arm position target after trajectory step.

        The trajectory controller runs first (smooth IK motion), then we override
        the arm target with IK_target + residual * policy_weight.
        """
        env = self._pick_env
        pw = self._policy_weight()
        if pw == 0.0:
            return  # No residual at stage 0
        assert self._plan is not None

        phase = IKPlanner.get_phase(self._step)
        arm_target, _base_target = IKPlanner.get_phase_targets(self._plan, phase)

        # Decompose action — only arm residual is applied per-step
        # Base residual is applied at phase start (base moves slowly)
        delta_arm = action[:, :5] * ARM_RESIDUAL_SCALE * pw

        # Apply residual to arm target
        corrected_arm = arm_target + delta_arm
        corrected_arm = corrected_arm.clamp(-math.pi, math.pi)

        # Override arm position target directly (after trajectory controller)
        env.hsr.control_dofs_position(
            corrected_arm,
            dofs_idx_local=env.arm_dofs_idx,
            envs_idx=self.envs_all,
        )

    def _handle_phase_transition(self, old_phase: int, new_phase: int):
        """Handle phase transition: set new trajectory, open gripper on approach."""
        env = self._pick_env
        if new_phase != old_phase:
            self._set_phase_trajectory(new_phase)
            # Open hand at start of approach
            if new_phase == 0:
                env.hsr.control_dofs_position(
                    env.hand_open, dofs_idx_local=[env.motor_idx],
                    envs_idx=self.envs_all,
                )
            # Set gripper force goal at start of grasp phase
            if new_phase == 2:
                pw = self._policy_weight()
                cmaes_effort = self._plan.gripper_effort if self._plan is not None else GRIPPER_EFFORT
                if pw == 0.0:
                    effort = torch.full((self.n_envs,), cmaes_effort, device=gs.device, dtype=gs.tc_float)
                else:
                    policy_effort = self._prev_action[:, 8] * GRIPPER_EFFORT_SCALE + GRIPPER_EFFORT_OFFSET
                    default_effort = torch.full((self.n_envs,), cmaes_effort, device=gs.device, dtype=gs.tc_float)
                    effort = default_effort * (1.0 - pw) + policy_effort * pw
                effort = effort.clamp(0.0, 8.0)
                active = torch.ones(self.n_envs, device=gs.device, dtype=torch.bool)
                env.gripper.set_apply_force_goal(
                    effort=effort, active_mask=active, envs_idx=self.envs_all,
                )

    def _check_success(self):
        """Check if object lifted above threshold."""
        env = self._pick_env
        obj_z = env._obj_pos()[:, 2]
        obj_init_z = self._plan.obj_init_z if self._plan is not None else env.obj_init_z
        success = obj_z > (obj_init_z + LIFT_THRESHOLD)
        self._success = self._success | success

    def step(self, action: np.ndarray | torch.Tensor):
        """Take one sim step with the given residual action.

        Args:
            action: (n_envs, 9) numpy array or torch.Tensor in [-1, 1].
                    torch.Tensor kept on GPU when already on gs.device.

        Returns:
            obs (n_envs, 32), reward (n_envs,), terminated (n_envs,), truncated (n_envs,), info
        """
        env = self._pick_env
        if isinstance(action, torch.Tensor):
            action_t = action.to(device=gs.device, dtype=gs.tc_float, non_blocking=True)
        else:
            action_t = torch.tensor(action, device=gs.device, dtype=gs.tc_float)
        self._prev_action = action_t.clone()

        # Check for phase transition
        old_phase = IKPlanner.get_phase(self._step)
        new_phase = IKPlanner.get_phase(self._step + 1)
        if new_phase != old_phase:
            self._handle_phase_transition(old_phase, new_phase)

        # Step simulation — use new_phase for control (the phase we're entering)
        phase = new_phase

        # Apply gripper force during grasp AND lift phases (object must be held)
        if phase in (2, 3):
            env.gripper.step_apply_force(self.dt, envs_idx=self.envs_all)

        if not self.use_ik_guidance:
            # In non-IK mode, policy directly sets the next arm/base target each step.
            self._set_direct_action_trajectory(action_t)

        # Step trajectory controller (smooth IK motion for arm + base)
        env.hsr.step_whole_body_trajectory_batched(self.dt, envs_idx=self.envs_all)

        # Apply residual correction AFTER trajectory controller (override arm target)
        if self.use_ik_guidance and old_phase == new_phase:  # Don't override on transition steps
            self._apply_residual(action_t)

        env.scene.step()

        self._step += 1
        env.total_steps += 1

        # Check success every 10 steps
        if self._step % 10 == 0:
            self._check_success()

        # Reward: +1.0 on first success, 0 otherwise
        new_success = self._success & ~self._success_reward_given
        reward = new_success.float()
        self._success_reward_given = self._success_reward_given | new_success

        # Episode termination
        terminated = self._success.clone()  # terminate on success
        truncated = torch.full((self.n_envs,), self._step >= IKPlanner.max_steps(),
                               device=gs.device, dtype=torch.bool)

        obs = self.get_obs()
        info = {
            "success": self._success.cpu().numpy(),
            "step": self._step,
        }
        return obs, reward.cpu().numpy(), terminated.cpu().numpy(), truncated.cpu().numpy(), info

    def get_success_rate(self) -> float:
        """Return current episode success rate (for curriculum eval)."""
        return float(self._success.float().mean())

    def close(self) -> None:
        pick_env = getattr(self, "_pick_env", None)
        scene = getattr(pick_env, "scene", None)
        if scene is not None:
            del scene
        self._pick_env = None
        if getattr(gs, "_initialized", False):
            gs.destroy()


class BatchedGenesisVecEnv(VecEnv):
    """Custom VecEnv that wraps HSRPickRLEnv's batched interface.

    SB3's DummyVecEnv loops over individual envs, but our Genesis envs share
    one GPU context and must step together. This VecEnv presents N envs to
    SB3 while internally making one batched step call.
    """

    def __init__(self, env: HSRPickRLEnv) -> None:
        self.env = env
        super().__init__(
            num_envs=env.n_envs,
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(OBS_DIM,), dtype=np.float32,
            ),
            action_space=spaces.Box(
                low=-1.0, high=1.0,
                shape=(ACTION_DIM,), dtype=np.float32,
            ),
        )
        self._actions: np.ndarray | None = None
        self.reset_infos = [{} for _ in range(env.n_envs)]

    def reset(self) -> VecEnvObs:
        obs, info = self.env.reset()
        # Store per-env reset info
        ik_success = info.get("ik_success", None)
        for i in range(self.num_envs):
            self.reset_infos[i] = {}
            if ik_success is not None:
                self.reset_infos[i]["ik_success"] = ik_success[i]
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions.copy()

    def step_wait(self) -> VecEnvStepReturn:
        assert self._actions is not None
        obs, rewards, terminated, truncated, info = self.env.step(self._actions)
        # Combine terminated and truncated into dones for SB3
        dones = np.logical_or(terminated, truncated).astype(bool)
        # Build per-env info list (SB3 expects list of dicts)
        infos = []
        success_arr = info.get("success", np.zeros(self.num_envs, dtype=bool))
        for i in range(self.num_envs):
            env_info = {
                "success": bool(success_arr[i]),
                "step": int(info["step"]),
            }
            # Signal terminal observation for done envs
            if dones[i]:
                env_info["terminal_observation"] = obs[i]
            infos.append(env_info)
        # SB3 expects rewards as (n_envs,) float array
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
        return obs, rewards, dones, infos

    def close(self) -> None:
        self.env.close()

    def get_attr(self, attr_name: str, indices=None):
        if indices is None:
            indices = range(self.num_envs)
        results = []
        for idx in indices:
            if hasattr(self.env, attr_name):
                val = getattr(self.env, attr_name)
                if hasattr(val, "__getitem__") and not isinstance(val, (str, dict)):
                    try:
                        results.append(val[idx])
                    except (IndexError, TypeError):
                        results.append(val)
                else:
                    results.append(val)
            else:
                results.append(None)
        return results

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        pass

    def env_method(self, method_name: str, *args, indices=None, **kwargs):
        if hasattr(self.env, method_name):
            return getattr(self.env, method_name)(*args, **kwargs)
        return None

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def get_images(self):
        return None
