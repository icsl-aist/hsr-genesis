"""Parallel IK grasping example: HSR picks a random YCB object in many envs.

This mirrors the structure of upstream Genesis parallel RL examples (many
parallel environments, batched rollouts, evaluation metrics) but the
"policy" is the analytic IK + whole-body trajectory controller used
throughout the episode — approach → descend → close gripper → lift —
exactly like the single-env scripted pick in
``examples/tutorials/spawn_ycb_objects.py``, here parallelized across
``--envs`` environments.

Task
----
Each parallel env spawns the HSR with a random YCB object placed at a
random pose within arm reach.  The IK pipeline runs synchronously across
all envs (each env follows its own per-env IK trajectory).  An episode is
**successful** when the object is lifted above its initial height by a
threshold; the evaluation reports the success rate and the average
time-to-pick across parallel envs.

Run
---
    PYTHONPATH=src .venv/bin/python examples/rl/ycb_pick_ik_parallel.py --envs 64

    # quick smoke test (few envs, no viewer)
    PYTHONPATH=src .venv/bin/python examples/rl/ycb_pick_ik_parallel.py --envs 8 --no-viewer
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

import genesis as gs

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hsr_genesis.grasp_planning import GraspSolver, SweepGraspPlanner

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

URDF_PATH = Path(__file__).resolve().parents[2] / "data" / "urdf" / "hsrb4s.urdf"
MODELS_DIR = (
    Path(__file__).resolve().parents[2]
    / "data" / "tmc_wrs_gazebo" / "tmc_wrs_gazebo_worlds" / "models"
)

# A graspable, visually varied YCB subset.
YCB_MODELS = [
    "ycb_061_foam_brick",
    "ycb_013_apple",
    "ycb_011_banana",
    "ycb_017_orange",
    "ycb_056_tennis_ball",
    "ycb_055_baseball",
    "ycb_077_rubiks_cube",
]

# Grasp approach geometry (meters), relative to the object center.
PRE_GRASP_HEIGHT = 0.15     # hover above the object before descending
GRASP_OFFSET_Z = 0.02       # final grasp height above object center
LIFT_HEIGHT = 0.30          # lift height after grasping

# Object placement region (meters, base frame).  Kept in the forward
# hemisphere within arm reach so the analytic base-yaw IK (which keeps
# base x,y fixed) reliably finds a solution.
OBJ_RADIUS_MIN = 0.32
OBJ_RADIUS_MAX = 0.46
OBJ_ANG_MIN = -math.pi / 3.0   # -60 deg
OBJ_ANG_MAX = math.pi / 3.0    # +60 deg
OBJ_Z = 0.05

GRIPPER_EFFORT = 3.0        # N applied when closing
LIFT_THRESHOLD = 0.05       # success: object raised this much above start

# Phase durations (seconds).
APPROACH_DURATION = 3.0
DESCEND_DURATION = 1.5
GRASP_HOLD_STEPS = 200      # ~4 s at dt=0.02
LIFT_DURATION = 1.5
LIFT_HOLD_STEPS = 50

HAND_QUAT = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # palm-down


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _yaw_from_quat_wxyz_batch(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def _quat_conjugate_batch(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Conjugate of batched WXYZ quaternions (inverse for unit quats)."""
    return torch.stack([
        quat_wxyz[:, 0], -quat_wxyz[:, 1], -quat_wxyz[:, 2], -quat_wxyz[:, 3],
    ], dim=-1)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class HSRPickEnv:
    """Parallel HSR pick environment driven by the IK pipeline.

    A single YCB object type is replicated across all parallel envs (Genesis
    parallel scenes share one morph per env).  Each env gets a different
    random object pose.  The IK pipeline runs synchronously across envs.
    """

    def __init__(
        self,
        *,
        n_envs: int,
        object_name: str,
        show_viewer: bool = False,
        seed: int = 0,
        disable_visualizer: bool = False,
        grasp_params: torch.Tensor | None = None,
        vis_options_overrides: dict | None = None,
        camera_config: dict | None = None,
        obj_radius_range: tuple[float, float] | None = None,
    ) -> None:
        self.n_envs = int(n_envs)
        self.dt = 0.02
        self.rng = torch.Generator(device=gs.device).manual_seed(int(seed))
        self._obj_radius_range = obj_radius_range  # override default placement radius

        vis_options_kwargs = dict(
            show_world_frame=True,
            world_frame_size=1.0,
            show_link_frame=False,
            plane_reflection=True,
            ambient_light=(0.3, 0.3, 0.3),
        )
        if vis_options_overrides:
            vis_options_kwargs.update(vis_options_overrides)

        scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.5, -2.0, 2.0),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                max_FPS=60,
            ),
            vis_options=gs.options.VisOptions(**vis_options_kwargs),
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=4),
            rigid_options=gs.options.RigidOptions(use_gjk_collision=True),
            show_viewer=show_viewer,
            show_FPS=False,
        )
        scene.add_entity(gs.morphs.Plane())

        from hsr_genesis.hsr_rigid_entity import HSRBURDF
        self.hsr = scene.add_entity(
            HSRBURDF(
                file=str(URDF_PATH),
                fixed=False,
                recompute_inertia=True,
                links_to_keep=["hand_palm_link"],
                robot="hsrb",
                base_mode="planar",
                end_effector_frame="hand_palm_link",
                use_base_controller=True,
                base_control_mode="controller",
                optimizer="gpu",
                use_base_yaw_ik=True,
            ),
            visualize_contact=False,
        )

        # Maps env idx -> index into self.objects for the "active" object in
        # that env.  Single-object envs are all active on object 0; the
        # multi-object subclass overrides this (inside _build_objects) with
        # a round-robin split, so this default must be set *before*
        # _build_objects runs.
        self.env_object_idx = torch.zeros(
            self.n_envs, device=gs.device, dtype=torch.long,
        )
        self.object_names = [object_name]
        self.objects = self._build_objects(scene, self.object_names)
        self.obj = self.objects[0]

        # Optional camera for offscreen rendering (must be added before build).
        if camera_config is not None:
            self.camera = scene.add_camera(
                res=camera_config.get("res", (1920, 1080)),
                pos=camera_config.get("pos", (2, -2, 1.5)),
                lookat=camera_config.get("lookat", (0, 0, 0.5)),
                fov=camera_config.get("fov", 60),
                GUI=False,
                far=camera_config.get("far", 200.0),
                debug=camera_config.get("debug", False),
            )
        else:
            self.camera = None

        if disable_visualizer and camera_config is None:
            scene._visualizer.build = lambda: None
        scene.build(n_envs=self.n_envs, env_spacing=(3.0, 3.0))
        self.scene = scene

        # Apply tuned PD gains (arm_lift gravity comp, etc.).
        self.hsr._hsr_apply_default_gains()

        self.envs_all = torch.arange(self.n_envs, device=gs.device, dtype=gs.tc_int)

        # End-effector setup.
        self.ee_link = self.hsr.get_link("hand_palm_link")
        # IK targets the hand_palm_link origin directly (no end_effector_offset).
        # The 9cm gripper length is accounted for in pre_grasp_height and grasp_offset_z.
        self.hsr.end_effector_offset = None
        self.gripper = self.hsr.get_gripper_batched()

        # Joint index bookkeeping.
        self.arm_qs_idx = self.hsr._ensure_arm_qs_idx()
        self.base_qs_idx = self.hsr._ensure_base_qs_idx()  # [x,y,z, qw,qx,qy,qz]
        self.arm_dofs_idx = self.hsr._hsr_arm_dofs_idx_local

        # Hand motor dof (for opening the hand during approach).
        motor_dofs = self.hsr.get_joint("hand_motor_joint").dofs_idx_local
        self.motor_idx = (
            int(motor_dofs[0]) if isinstance(motor_dofs, (list, tuple)) else int(motor_dofs)
        )
        self.hand_open = torch.tensor(
            [[1.0]] * self.n_envs, device=gs.device, dtype=gs.tc_float,
        )

        # Per-env episode state.
        self.obj_init_z = torch.zeros(self.n_envs, device=gs.device, dtype=gs.tc_float)
        self.steps_to_success = torch.full(
            (self.n_envs,), -1, device=gs.device, dtype=torch.long,
        )
        self.total_steps = torch.zeros(self.n_envs, device=gs.device, dtype=torch.long)

        # Per-env grasp params (n_envs, 4) or None for module defaults.
        # Columns: [pre_grasp_height, grasp_offset_z, gripper_effort, grasp_hold_steps]
        self.grasp_params = grasp_params

        # Batched IK solver shared by planners and env methods.
        self.solver = GraspSolver(
            self.hsr, self.ee_link, scene, self.envs_all,
            self.arm_dofs_idx, self.n_envs, dt=self.dt,
        )
        # Default grasp-target provider surface: the object itself is the
        # graspable part (no joint geometry); HSRArtvipPickEnv overrides
        # these fields in _resolve_artvip_metadata().
        self._graspable_local_offset = None
        self._graspable_local_offset_3d = None
        self._graspable_jaw_axis_local = None
        self._joint_axis_local = None
        self._joint_origin_local = None
        self._target_joint_type = "revolute"
        self._target_joint_limits = (0.0, 1.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rand(self, shape) -> torch.Tensor:
        return torch.rand(shape, generator=self.rng, device=gs.device, dtype=gs.tc_float)

    def _build_objects(self, scene, object_names: list[str]) -> list:
        """Add one URDF entity per name.  Override to customize placement."""
        from hsr_genesis.sdf_parser import load_sdf_model

        entities = []
        for name in object_names:
            model_dir = MODELS_DIR / name
            if not model_dir.exists():
                raise FileNotFoundError(f"YCB model not found: {model_dir}")
            obj_urdf = load_sdf_model(model_dir)
            entities.append(
                scene.add_entity(
                    gs.morphs.URDF(file=obj_urdf, pos=(0.5, 0.0, OBJ_Z), fixed=False),
                )
            )
        return entities

    def _obj_pos(self) -> torch.Tensor:
        if len(self.objects) == 1:
            return self.obj.get_pos(envs_idx=self.envs_all)
        # Multi-object scene: gather each env's "active" object position.
        pos = torch.zeros(self.n_envs, 3, device=gs.device, dtype=gs.tc_float)
        for k, obj_entity in enumerate(self.objects):
            mask = self.env_object_idx == k
            if mask.any():
                pos[mask] = obj_entity.get_pos(envs_idx=self.envs_all)[mask]
        return pos

    def _random_object_pose(self):
        """Sample a random in-reach object pose per env: (pos, quat)."""
        theta = OBJ_ANG_MIN + (OBJ_ANG_MAX - OBJ_ANG_MIN) * self._rand((self.n_envs,))
        if self._obj_radius_range is not None:
            r_min, r_max = self._obj_radius_range
        else:
            r_min, r_max = OBJ_RADIUS_MIN, OBJ_RADIUS_MAX
        radius = r_min + (r_max - r_min) * self._rand((self.n_envs,))
        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)
        z = torch.full((self.n_envs,), OBJ_Z, device=gs.device, dtype=gs.tc_float)
        pos = torch.stack([x, y, z], dim=-1)
        yaw = (self._rand((self.n_envs,)) * 2.0 - 1.0) * math.pi
        half = yaw * 0.5
        quat = torch.stack(
            [torch.cos(half), torch.zeros_like(half),
             torch.zeros_like(half), torch.sin(half)],
            dim=-1,
        )
        return pos, quat

    def _settle(self, n_steps: int) -> None:
        """Hold the robot at its current pose while objects settle."""
        from hsr_genesis.hsr_rigid_entity import JointTrajectory
        from hsr_genesis.base_controller import Trajectory
        from hsr_genesis.analytic_ik import JOINT_ORDER

        arm_pos = self.hsr.get_dofs_position(
            dofs_idx_local=self.arm_dofs_idx, envs_idx=self.envs_all,
        )
        if arm_pos.ndim == 1:
            arm_pos = arm_pos.unsqueeze(0)
        base_pos = self.hsr.get_pos(envs_idx=self.envs_all)
        base_quat = self.hsr.get_quat(envs_idx=self.envs_all)
        if base_pos.ndim == 1:
            base_pos = base_pos.unsqueeze(0)
            base_quat = base_quat.unsqueeze(0)
        yaw = _yaw_from_quat_wxyz_batch(base_quat)
        base_xy_yaw = torch.stack(
            [base_pos[:, 0], base_pos[:, 1], yaw], dim=-1,
        )

        hold_t = torch.tensor(
            [n_steps * self.dt], device=gs.device, dtype=gs.tc_float,
        )
        names = list(JOINT_ORDER)
        # API-bound: set_whole_body_trajectory_batched expects list[JointTrajectory] per env.
        arm_trajs = [
            JointTrajectory(
                positions=arm_pos[i].unsqueeze(0),
                time_from_start=hold_t,
                joint_names=names,
            )
            for i in range(self.n_envs)
        ]
        base_trajs = [
            Trajectory(positions=base_xy_yaw[i].unsqueeze(0), time_from_start=hold_t)
            for i in range(self.n_envs)
        ]
        self.hsr.set_whole_body_trajectory_batched(
            arm_trajectory=arm_trajs,
            base_trajectory=base_trajs,
            envs_idx=self.envs_all,
            start_time=None,
        )
        for _ in range(n_steps):
            self.hsr.step_whole_body_trajectory_batched(self.dt, envs_idx=self.envs_all)
            self.scene.step()

    # ------------------------------------------------------------------
    # Reset: place objects at random poses and settle.
    # ------------------------------------------------------------------

    def reset(self, settle_steps: int = 200) -> None:
        # Random object pose per env, in front of the robot within arm reach.
        pos, quat = self._random_object_pose()
        self.obj.set_pos(pos, envs_idx=self.envs_all, zero_velocity=True, relative=False)
        self.obj.set_quat(quat, envs_idx=self.envs_all, zero_velocity=True, relative=False)

        self.steps_to_success[:] = -1
        self.total_steps[:] = 0

        self._settle(settle_steps)
        # Record settled object z as the success baseline.
        self.obj_init_z = self._obj_pos()[:, 2]

    # ------------------------------------------------------------------
    # GraspTargetProvider implementation (default: point target).
    # HSRArtvipPickEnv overrides with joint/handle geometry.
    # ------------------------------------------------------------------

    def obj_pos(self) -> torch.Tensor:
        return self._obj_pos()

    def obj_quat(self) -> torch.Tensor:
        quat = self.obj.get_quat(envs_idx=self.envs_all)
        return quat.unsqueeze(0) if quat.ndim == 1 else quat

    def part_pos(self, obj_pos: torch.Tensor, obj_quat: torch.Tensor) -> torch.Tensor:
        """Non-articulated object: grasp target is the object origin, unless
        a local offset was resolved (subclass)."""
        if self._graspable_local_offset_3d is not None:
            return obj_pos + self._quat_rotate_batch(obj_quat, self._graspable_local_offset_3d)
        return obj_pos.clone()

    def surface_normal(self, part_pos: torch.Tensor, obj_quat: torch.Tensor) -> torch.Tensor | None:
        return None

    def jaw_axis(self, obj_quat: torch.Tensor) -> torch.Tensor | None:
        if self._graspable_jaw_axis_local is None:
            return None
        local_axis = self._graspable_jaw_axis_local.unsqueeze(0).expand(self.n_envs, -1)
        return self._quat_rotate_batch(obj_quat, local_axis)

    def approach_dirs(
        self, part_pos: torch.Tensor, surface_normal: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        """8 evenly spaced XY directions → base ring around the object.

        Gives the sweep a full 360° of stand points (45° apart); the
        planner places the base at ``part_xy - dir*arm_reach`` for each
        and rejects colliding or unreachable candidates.
        """
        N = self.n_envs
        out = []
        for k in range(8):
            a = k * math.pi / 4.0
            d = torch.zeros(N, 3, device=gs.device, dtype=gs.tc_float)
            d[:, 0] = math.cos(a)
            d[:, 1] = math.sin(a)
            out.append(d)
        return out

    def yaw_candidates(self, obj_pos: torch.Tensor) -> list[torch.Tensor]:
        """Full-circle yaw sweep (30° increments) for unconstrained objects."""
        base = torch.zeros(self.n_envs, device=gs.device, dtype=gs.tc_float)
        deltas = torch.linspace(0, 2 * math.pi, 13, device=gs.device, dtype=gs.tc_float)[:-1]
        return [(base + d) for d in deltas]

    def set_obj_pose(self, quat: torch.Tensor) -> None:
        self.obj.set_quat(
            quat, envs_idx=self.envs_all, zero_velocity=True, relative=False,
        )
        self._settle(20)

    def terminal_target(self, grasp_pos: torch.Tensor, descend_pos: torch.Tensor) -> torch.Tensor:
        """Lift the object straight up (non-articulated terminal phase)."""
        lift = self._obj_pos().clone()
        lift[:, 2] = 0.30  # LIFT_HEIGHT
        return lift

    # ------------------------------------------------------------------
    # IK helpers (batched).
    # ------------------------------------------------------------------

    def _ik(self, goal_pos: torch.Tensor, *, init_qpos=None,
            hand_quat: torch.Tensor | None = None):
        """Batched analytic IK; delegates to :class:`GraspSolver`."""
        return self.solver.solve(goal_pos, init_qpos=init_qpos, hand_quat=hand_quat)

    def _calibrated_ik(self, goal_pos: torch.Tensor, *, init_qpos=None,
                       hand_quat: torch.Tensor | None = None,
                       max_iters: int = 3, respect_joint_limit: bool = True,
                       max_step_size: float = 0.1):
        """IK with sim-FK feedback calibration; delegates to GraspSolver."""
        return self.solver.calibrated(
            goal_pos, init_qpos=init_qpos, hand_quat=hand_quat,
            max_iters=max_iters, respect_joint_limit=respect_joint_limit,
            max_step_size=max_step_size,
        )

    def _jacobian_ik(self, goal_pos, *, init_qpos=None,
                     hand_quat=None, max_iters=50, max_step_size=0.1,
                     pos_tol=1e-3):
        """Position-only Jacobian IK (base fixed); delegates to GraspSolver."""
        return self.solver.jacobian(
            goal_pos, init_qpos=init_qpos, hand_quat=hand_quat,
            max_iters=max_iters, max_step_size=max_step_size, pos_tol=pos_tol,
        )

    def _qpos_to_arm_and_base(self, qpos: torch.Tensor):
        """Extract per-env arm joint angles (n_envs, 5) and base [x,y,yaw]."""
        return self.solver.qpos_to_arm_and_base(qpos)

    def _set_whole_body(self, arm_positions, base_xy_yaw, duration):
        """Build per-env arm + base trajectories and load them."""
        from hsr_genesis.hsr_rigid_entity import JointTrajectory
        from hsr_genesis.base_controller import Trajectory
        from hsr_genesis.analytic_ik import JOINT_ORDER

        t = torch.tensor([duration], device=gs.device, dtype=gs.tc_float)
        # API-bound: set_whole_body_trajectory_batched expects list[JointTrajectory] per env.
        arm_trajs = [
            JointTrajectory(
                positions=arm_positions[i].unsqueeze(0),
                time_from_start=t,
                joint_names=list(JOINT_ORDER),
            )
            for i in range(self.n_envs)
        ]
        base_trajs = [
            Trajectory(
                positions=base_xy_yaw[i].unsqueeze(0),
                time_from_start=t,
            )
            for i in range(self.n_envs)
        ]
        self.hsr.set_whole_body_trajectory_batched(
            arm_trajectory=arm_trajs,
            base_trajectory=base_trajs,
            envs_idx=self.envs_all,
            start_time=None,
        )

    def _set_arm_only(self, arm_positions, duration):
        """Build per-env arm-only trajectories (base held) and load them."""
        from hsr_genesis.hsr_rigid_entity import JointTrajectory
        from hsr_genesis.analytic_ik import JOINT_ORDER

        t = torch.tensor([duration], device=gs.device, dtype=gs.tc_float)
        # API-bound: set_whole_body_trajectory_batched expects list[JointTrajectory] per env.
        arm_trajs = [
            JointTrajectory(
                positions=arm_positions[i].unsqueeze(0),
                time_from_start=t,
                joint_names=list(JOINT_ORDER),
            )
            for i in range(self.n_envs)
        ]
        self.hsr.set_whole_body_trajectory_batched(
            arm_trajectory=arm_trajs,
            base_trajectory=None,
            envs_idx=self.envs_all,
            start_time=None,
        )

    def _run_phase(self, duration: float, *, open_hand_first: bool = False) -> None:
        n_steps = int(duration / self.dt) + 30
        for step in range(n_steps):
            self.hsr.step_whole_body_trajectory_batched(self.dt, envs_idx=self.envs_all)
            if step == 0 and open_hand_first:
                self.hsr.control_dofs_position(
                    self.hand_open, dofs_idx_local=[self.motor_idx],
                    envs_idx=self.envs_all,
                )
            self.scene.step()
            self.total_steps += 1
            if step % 10 == 0:
                self._check_success()

    def _run_gripper_hold(self, n_steps: int) -> None:
        for step in range(n_steps):
            self.gripper.step_apply_force(self.dt, envs_idx=self.envs_all)
            self.hsr.step_whole_body_trajectory_batched(self.dt, envs_idx=self.envs_all)
            self.scene.step()
            self.total_steps += 1
            if step % 10 == 0:
                self._check_success()

    def _check_success(self) -> None:
        obj_z = self._obj_pos()[:, 2]
        success = obj_z > (self.obj_init_z + LIFT_THRESHOLD)
        first_success = success & (self.steps_to_success < 0)
        if first_success.any():
            self.steps_to_success = torch.where(
                first_success, self.total_steps.clone(), self.steps_to_success,
            )

    # ------------------------------------------------------------------
    # Full IK pick pipeline (synchronous across envs).
    # ------------------------------------------------------------------

    def run_pick_pipeline(self, settle_steps: int = 200, debug: bool = False) -> dict:
        self.reset(settle_steps=settle_steps)
        obj_pos = self._obj_pos()
        if debug:
            print(f"  [debug] obj_pos after settle: {obj_pos.tolist()}")
            print(f"  [debug] obj_init_z: {self.obj_init_z.tolist()}")

        # Resolve per-env grasp params (or fall back to module defaults).
        gp = self.grasp_params
        if gp is not None and gp.shape[0] == 1:
            gp = gp.expand(self.n_envs, -1).clone()
        if gp is None:
            pre_grasp_h = PRE_GRASP_HEIGHT
            grasp_offset_z = GRASP_OFFSET_Z
            effort = torch.full(
                (self.n_envs,), GRIPPER_EFFORT,
                device=gs.device, dtype=gs.tc_float,
            )
            grasp_hold_steps = GRASP_HOLD_STEPS
        else:
            pre_grasp_h = gp[:, 0]
            grasp_offset_z = gp[:, 1]
            effort = gp[:, 2].to(dtype=gs.tc_float)
            grasp_hold_steps = int(gp[:, 3].max().item())

        # --- Phase 1: approach (pre-grasp hover above the object) ---
        pre_grasp = obj_pos.clone()
        pre_grasp[:, 2] += pre_grasp_h
        qpos, ik_ok = self._ik(pre_grasp)
        arm, base = self._qpos_to_arm_and_base(qpos)
        if debug:
            print(f"  [debug] approach IK success: {ik_ok.tolist()}")
            print(f"  [debug] approach base targets: {base.tolist()}")
        self._set_whole_body(arm, base, APPROACH_DURATION)
        self._run_phase(APPROACH_DURATION, open_hand_first=True)
        if debug:
            ee = self.ee_link.get_pos(envs_idx=self.envs_all)
            print(f"  [debug] after approach ee: {ee.tolist()}")
            print(f"  [debug] after approach obj: {self._obj_pos().tolist()}")

        # --- Phase 2: descend to grasp pose ---
        grasp = obj_pos.clone()
        grasp[:, 2] += grasp_offset_z
        cur_qpos = self.hsr.get_qpos(envs_idx=self.envs_all)
        if cur_qpos.ndim == 1:
            cur_qpos = cur_qpos.unsqueeze(0)
        qpos, ik_ok = self._ik(grasp, init_qpos=cur_qpos)
        arm, _base = self._qpos_to_arm_and_base(qpos)
        if debug:
            print(f"  [debug] descend IK success: {ik_ok.tolist()}")
        self._set_arm_only(arm, DESCEND_DURATION)
        self._run_phase(DESCEND_DURATION)
        if debug:
            ee = self.ee_link.get_pos(envs_idx=self.envs_all)
            print(f"  [debug] after descend ee: {ee.tolist()}")
            print(f"  [debug] after descend obj: {self._obj_pos().tolist()}")

        # --- Phase 3: close gripper ---
        active = torch.ones(self.n_envs, device=gs.device, dtype=torch.bool)
        self.gripper.set_apply_force_goal(
            effort=effort, active_mask=active, envs_idx=self.envs_all,
        )
        self._run_gripper_hold(grasp_hold_steps)
        if debug:
            print(f"  [debug] after grasp obj: {self._obj_pos().tolist()}")

        # --- Phase 4: lift ---
        lift = obj_pos.clone()
        lift[:, 2] = LIFT_HEIGHT
        cur_qpos = self.hsr.get_qpos(envs_idx=self.envs_all)
        if cur_qpos.ndim == 1:
            cur_qpos = cur_qpos.unsqueeze(0)
        qpos, ik_ok = self._ik(lift, init_qpos=cur_qpos)
        arm, _base = self._qpos_to_arm_and_base(qpos)
        if debug:
            print(f"  [debug] lift IK success: {ik_ok.tolist()}")
        self._set_arm_only(arm, LIFT_DURATION)
        self._run_gripper_hold(int(LIFT_DURATION / self.dt) + 30)
        # Hold a bit more to confirm the lift.
        self._run_gripper_hold(LIFT_HOLD_STEPS)
        # Final success check to catch the last few steps.
        self._check_success()
        if debug:
            print(f"  [debug] after lift obj: {self._obj_pos().tolist()}")
            print(f"  [debug] steps_to_success: {self.steps_to_success.tolist()}")

        return self.get_eval_summary()

    def get_eval_summary(self) -> dict:
        succeeded = self.steps_to_success >= 0
        n_succ = int(succeeded.sum().item())
        if n_succ > 0:
            avg_steps = float(self.steps_to_success[succeeded].to(torch.float32).mean().item())
            avg_time = avg_steps * self.dt
        else:
            avg_steps = float("nan")
            avg_time = float("nan")
        return {
            "success_per_env": succeeded.float(),
            "success_rate": n_succ / self.n_envs,
            "n_success": n_succ,
            "n_envs": self.n_envs,
            "avg_steps_to_success": avg_steps,
            "avg_time_to_success": avg_time,
        }


class HSRMultiObjectPickEnv(HSRPickEnv):
    """Fused-scene variant: spawns every object once and gives each env a
    single "active" object (round-robin across env indices), parking the
    other objects far from the robot workspace.  This evaluates all objects
    in one batched ``run_pick_pipeline()`` call instead of rebuilding a
    scene and re-running the pipeline once per object.
    """

    # Grid spacing (meters) between parked objects, chosen to stay well
    # outside OBJ_RADIUS_MAX (arm reach) and to keep parked objects from
    # touching each other.
    _PARK_SPACING = 1.0
    _PARK_ORIGIN = (3.0, 3.0)

    def __init__(
        self,
        *,
        n_envs: int,
        object_names: list[str] | None = None,
        show_viewer: bool = False,
        seed: int = 0,
        disable_visualizer: bool = False,
        grasp_params: torch.Tensor | None = None,
    ) -> None:
        names = list(object_names) if object_names else list(YCB_MODELS)
        self._multi_object_names = names
        super().__init__(
            n_envs=n_envs,
            object_name=names[0],  # unused: _build_objects is overridden
            show_viewer=show_viewer,
            seed=seed,
            disable_visualizer=disable_visualizer,
            grasp_params=grasp_params,
        )

    def _build_objects(self, scene, object_names: list[str]) -> list:
        # Ignore the single-name list from the base __init__ and spawn all
        # requested objects instead.
        entities = super()._build_objects(scene, self._multi_object_names)

        n_objects = len(entities)
        self.env_object_idx = torch.arange(
            self.n_envs, device=gs.device, dtype=torch.long,
        ) % n_objects
        self.object_names = list(self._multi_object_names)

        park_xy = torch.tensor(
            [
                [
                    self._PARK_ORIGIN[0] + self._PARK_SPACING * k,
                    self._PARK_ORIGIN[1],
                ]
                for k in range(n_objects)
            ],
            device=gs.device, dtype=gs.tc_float,
        )
        park_z = torch.full((n_objects, 1), OBJ_Z, device=gs.device, dtype=gs.tc_float)
        self.park_pos = torch.cat([park_xy, park_z], dim=-1)  # (n_objects, 3)
        self.park_quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=gs.device, dtype=gs.tc_float,
        )
        return entities

    def reset(self, settle_steps: int = 200) -> None:
        active_pos, active_quat = self._random_object_pose()
        for k, obj_entity in enumerate(self.objects):
            mask = self.env_object_idx == k
            pos_k = self.park_pos[k].unsqueeze(0).expand(self.n_envs, 3).clone()
            quat_k = self.park_quat.unsqueeze(0).expand(self.n_envs, 4).clone()
            pos_k[mask] = active_pos[mask]
            quat_k[mask] = active_quat[mask]
            obj_entity.set_pos(pos_k, envs_idx=self.envs_all, zero_velocity=True, relative=False)
            obj_entity.set_quat(quat_k, envs_idx=self.envs_all, zero_velocity=True, relative=False)

        self.steps_to_success[:] = -1
        self.total_steps[:] = 0

        self._settle(settle_steps)
        self.obj_init_z = self._obj_pos()[:, 2]

    def get_per_object_summary(self) -> dict:
        """Break down ``get_eval_summary()`` results by assigned object."""
        succeeded = self.steps_to_success >= 0
        summary = {}
        for k, name in enumerate(self.object_names):
            mask = self.env_object_idx == k
            n = int(mask.sum().item())
            n_succ = int((succeeded & mask).sum().item())
            if n_succ > 0:
                steps = self.steps_to_success[mask & succeeded].to(torch.float32)
                avg_steps = float(steps.mean().item())
                avg_time = avg_steps * self.dt
            else:
                avg_steps = float("nan")
                avg_time = float("nan")
            summary[name] = {
                "success_per_env": succeeded[mask].float(),
                "success_rate": (n_succ / n) if n > 0 else float("nan"),
                "n_success": n_succ,
                "n_envs": n,
                "avg_steps_to_success": avg_steps,
                "avg_time_to_success": avg_time,
            }
        return summary


class HSRArtvipPickEnv(HSRPickEnv):
    """HSR pick env with an ArtVIP articulated USD object instead of a YCB URDF.

    This stresses the physics solver with complex collision geometry (convexified
    meshes) and articulation joints, providing a "complex object" case for
    throughput benchmarking.  The ArtVIP object is downloaded on-demand via
    :mod:`hsr_genesis.artvip_loader` and loaded as a free-body articulated
    entity (``fixed=False``) so the pick pipeline can interact with it.

    The success rate will typically be lower than with simple YCB objects
    (large articulated objects are harder to lift), but the throughput
    metrics (steps/sec, envs·steps/sec, GPU memory) are the primary output.

    After scene build, this class also resolves ArtVIP metadata (joint info,
    part semantics, control script) for use by articulation-aware RL envs:
      - Target joint selection (prefer control-script named joints, then
        joints connected to graspable parts, then largest-range movable joint).
      - Joint geometry in object-root local frame (axis + origin) for
        computing world-space articulation directions at runtime.
      - Graspable link handle for approach-shaping rewards.
    """

    def __init__(
        self,
        *,
        n_envs: int,
        artvip_category: str,
        artvip_object: str,
        show_viewer: bool = False,
        seed: int = 0,
        disable_visualizer: bool = False,
        grasp_params: torch.Tensor | None = None,
        vis_options_overrides: dict | None = None,
        camera_config: dict | None = None,
        obj_radius_range: tuple[float, float] | None = None,
        cache_dir: str | Path | None = None,
        decimate_face_num: int = 500,
        merge_meshes: bool = False,
        obj_z: float | None = None,
    ) -> None:
        self._artvip_category = artvip_category
        self._artvip_object = artvip_object
        self._artvip_cache_dir = cache_dir
        self._decimate_face_num = decimate_face_num
        self._merge_meshes = merge_meshes
        self._obj_z_override = obj_z
        self._usd_path: Path | None = None
        # ArtVIP objects are much larger than YCB bricks; place them further
        # from the robot to reduce hand-object collision pairs during the
        # approach and descend phases.
        if obj_radius_range is None:
            obj_radius_range = (0.50, 0.65)
        label = f"artvip:{artvip_category}/{artvip_object}"
        super().__init__(
            n_envs=n_envs,
            object_name=label,
            show_viewer=show_viewer,
            seed=seed,
            disable_visualizer=disable_visualizer,
            grasp_params=grasp_params,
            vis_options_overrides=vis_options_overrides,
            camera_config=camera_config,
            obj_radius_range=obj_radius_range,
        )
        # Resolve ArtVIP metadata after scene is built and entity is ready.
        self._resolve_artvip_metadata()

    def _build_objects(self, scene, object_names: list[str]) -> list:
        from hsr_genesis.artvip_loader import download_artvip_object, merge_fixed_meshes

        usd_path = download_artvip_object(
            self._artvip_category,
            self._artvip_object,
            cache_dir=self._artvip_cache_dir,
        )
        if self._merge_meshes:
            usd_path = merge_fixed_meshes(usd_path)
        self._usd_path = usd_path
        obj_z_val = self._obj_z_override if self._obj_z_override is not None else OBJ_Z
        # When the object is raised above the default ground position, fix it
        # so it doesn't fall back down during settling.
        obj_fixed = self._obj_z_override is not None and self._obj_z_override > OBJ_Z + 0.01
        entity = scene.add_entity(
            gs.morphs.USD(
                file=str(usd_path),
                pos=(0.6, 0.0, obj_z_val),
                fixed=obj_fixed,
                decimate=True,
                decimate_face_num=self._decimate_face_num,
                convexify=True,
                # ArtVIP USDs contain fixed links; per-env set_pos requires
                # batching the fixed-vertex geometry across envs.
                batch_fixed_verts=True,
            ),
        )
        return [entity]

    # ------------------------------------------------------------------
    # ArtVIP metadata resolution
    # ------------------------------------------------------------------

    def _resolve_artvip_metadata(self) -> None:
        """Parse ArtVIP USD metadata and resolve target joint + graspable link.

        After scene.build(), the ArtVIP entity's joints and links are
        available. This method:
          1. Parses joint info and part info from the USD.
          2. Selects a target joint (prefer control script, then graspable
             part joints, then largest-range movable joint).
          3. Finds the target joint's dof index in the Genesis entity.
          4. Resolves the graspable link handle for approach shaping.
          5. Computes joint geometry (axis + origin) in object-root local
             frame for runtime world-space articulation direction.
          6. Stores articulation threshold from control script or default.
        """
        from hsr_genesis.artvip_loader import (
            parse_artvip_joint_info,
            parse_artvip_part_info,
            parse_artvip_control_script,
        )

        if self._usd_path is None:
            return

        usd_path = self._usd_path

        # Parse joint info from USD.
        joint_info = parse_artvip_joint_info(usd_path)

        # Parse part info from USD.
        part_info = parse_artvip_part_info(usd_path)

        # Parse control script if available.
        # ArtVIP control scripts are named inconsistently (e.g. trashcan_control.py
        # vs trash_can_control.py), so glob for *_control.py in the resource dir.
        ctrl_meta: dict = {}
        resource_dir = Path(usd_path).parent / "resource"
        ctrl_files = list(resource_dir.glob("*_control.py")) if resource_dir.is_dir() else []
        if ctrl_files:
            ctrl_meta = parse_artvip_control_script(ctrl_files[0])

        # Resolve target joint.
        target_joint = self._resolve_target_joint(joint_info, part_info, ctrl_meta)
        self._target_joint_name = target_joint.name
        self._target_joint_type = target_joint.joint_type
        self._target_joint_limits = target_joint.limits
        self._target_joint_range = abs(target_joint.limits[1] - target_joint.limits[0])
        # Find the target joint in the Genesis entity by name.
        # Genesis may rename USD joints, so try exact match first,
        # then substring match, then fallback to first joint with dofs.
        self._target_dof_idx: list[int] | None = None
        genesis_joint_names = [j.name for j in self.obj.joints]
        print(f"[artvip] Genesis entity joints: {genesis_joint_names}")
        print(f"[artvip] Target USD joint name: {target_joint.name}")

        # Try exact name match.
        for joint in self.obj.joints:
            if joint.name == target_joint.name:
                dofs = joint.dofs_idx_local
                if isinstance(dofs, (list, tuple)):
                    self._target_dof_idx = [int(d) for d in dofs]
                else:
                    self._target_dof_idx = [int(dofs)]
                break

        # Try substring match (USD joint type prefix may be stripped/renamed).
        if self._target_dof_idx is None:
            # Extract the meaningful part of the USD joint name (after the type prefix).
            usd_name_lower = target_joint.name.lower()
            for joint in self.obj.joints:
                gname_lower = joint.name.lower()
                if (usd_name_lower in gname_lower or gname_lower in usd_name_lower
                        or target_joint.name in joint.name
                        or joint.name in target_joint.name):
                    dofs = joint.dofs_idx_local
                    if isinstance(dofs, (list, tuple)):
                        self._target_dof_idx = [int(d) for d in dofs]
                    else:
                        self._target_dof_idx = [int(dofs)]
                    print(f"[artvip] Matched by substring: USD='{target_joint.name}' → Genesis='{joint.name}'")
                    break

        # Fallback: first joint with non-None dofs_idx_local.
        if self._target_dof_idx is None:
            for joint in self.obj.joints:
                dofs = joint.dofs_idx_local
                if dofs is not None:
                    if isinstance(dofs, (list, tuple)):
                        self._target_dof_idx = [int(d) for d in dofs]
                    else:
                        self._target_dof_idx = [int(dofs)]
                    print(f"[artvip] Fallback: using first joint with dofs: '{joint.name}'")
                    break

        # Resolve graspable interaction geometry and its containing physics link.
        # Nested handles are not Genesis links, so preserve their semantic
        # bounds center as the target while resolving the parent drawer/door
        # link for runtime motion and contact state.
        self._graspable_link_name: str | None = None
        self._graspable_link = None
        self._graspable_local_offset: torch.Tensor | None = None
        self._graspable_local_offset_3d: torch.Tensor | None = None
        self._graspable_jaw_axis_local: torch.Tensor | None = None
        genesis_link_names = [lk.name for lk in self.obj.links]

        def _resolve_link_by_short_name(short_name: str):
            for lk in self.obj.links:
                if lk.name == short_name or lk.name.endswith("/" + short_name):
                    return lk
            return None

        selected_part = None
        button_prim_name = ctrl_meta.get("button_prim_name")
        if button_prim_name:
            self._graspable_link_name = button_prim_name
            self._graspable_link = _resolve_link_by_short_name(button_prim_name)
            print(f"[artvip] Button-coupled mechanism: grasp target = {button_prim_name}")
        else:
            graspable_parts = part_info.get_graspable_parts() if part_info else []
            target_bodies = {target_joint.body0, target_joint.body1}
            connected = [p for p in graspable_parts if p.parent_link in target_bodies]
            selected_part = (connected or graspable_parts or [None])[0]
            if selected_part is not None:
                short_name = selected_part.parent_link.split("/")[-1]
                self._graspable_link_name = short_name
                self._graspable_link = _resolve_link_by_short_name(short_name)
                # Jaw separation follows the thinnest handle dimension.
                # The palm-normal component is projected out by the planner.
                extent = torch.tensor(
                    selected_part.root_local_extent,
                    device=gs.device,
                    dtype=gs.tc_float,
                )
                jaw_idx = int(torch.argmin(extent).item())
                jaw_axis = torch.zeros(3, device=gs.device, dtype=gs.tc_float)
                jaw_axis[jaw_idx] = 1.0
                self._graspable_jaw_axis_local = jaw_axis
                print(f"[artvip] Grasp interaction part: {selected_part.prim_path}")

        # Articulation threshold from control script or default 0.7.
        self._articulation_threshold = float(ctrl_meta.get("joint_threshold", 0.7))

        # Resolve joint geometry in object-root local frame.
        self._resolve_joint_geometry(target_joint, usd_path)

        # Compute the graspable offset from the Genesis link origin at
        # runtime.  This matches what ``get_graspable_part_pos()`` returns
        # (the link origin, not the handle mesh bounding-box center, which
        # can differ by the handle's local offset within the link).
        if self._graspable_link is not None and self._graspable_local_offset_3d is None:
            obj_pos = self.obj.get_pos(envs_idx=self.envs_all)
            obj_quat = self.obj.get_quat(envs_idx=self.envs_all)
            if obj_pos.ndim == 1:
                obj_pos = obj_pos.unsqueeze(0)
                obj_quat = obj_quat.unsqueeze(0)
            link_pos = self._graspable_link.get_pos(envs_idx=self.envs_all)
            if link_pos.ndim == 1:
                link_pos = link_pos.unsqueeze(0)
            world_offset = link_pos - obj_pos
            local_offset = self._quat_rotate_batch(
                _quat_conjugate_batch(obj_quat), world_offset,
            )
            self._graspable_local_offset = local_offset[:, :2].clone()
            self._graspable_local_offset_3d = local_offset.clone()
        if self._graspable_local_offset_3d is not None:
            print(f"[artvip] Graspable part local offset (xy): "
                  f"{self._graspable_local_offset[0].cpu().tolist()}")
            print(f"[artvip] Graspable part local offset (xyz): "
                  f"{self._graspable_local_offset_3d[0].cpu().tolist()}")

    def _random_object_pose(self):
        """Sample object pose with yaw oriented so the graspable part faces the robot.

        Overrides the parent's random-yaw placement.  Computes yaw so that
        the graspable part's local offset from the object root points toward
        the robot after placement.  The object center is placed farther from
        the robot by the part offset magnitude, so the *graspable part* (not
        the object center) lands at the target radius.
        """
        theta = OBJ_ANG_MIN + (OBJ_ANG_MAX - OBJ_ANG_MIN) * self._rand((self.n_envs,))
        if self._obj_radius_range is not None:
            r_min, r_max = self._obj_radius_range
        else:
            r_min, r_max = OBJ_RADIUS_MIN, OBJ_RADIUS_MAX
        part_radius = r_min + (r_max - r_min) * self._rand((self.n_envs,))

        # Compute yaw so the graspable part faces the robot.
        if self._graspable_local_offset is not None:
            offset = self._graspable_local_offset[0]  # (2,) — same for all envs
            part_offset_mag = offset.norm()
            local_angle = torch.atan2(offset[1], offset[0])
            # Direction from object to robot (robot at origin).
            # Place object center beyond the part: center_dist = part_radius + offset_mag
            center_radius = part_radius + part_offset_mag
            x = center_radius * torch.cos(theta)
            y = center_radius * torch.sin(theta)
            target_angle = torch.atan2(-y, -x)
            yaw = target_angle - local_angle
        else:
            # Fallback: no offset, use radius directly with random yaw.
            x = part_radius * torch.cos(theta)
            y = part_radius * torch.sin(theta)
            yaw = (self._rand((self.n_envs,)) * 2.0 - 1.0) * math.pi

        z_val = self._obj_z_override if self._obj_z_override is not None else OBJ_Z
        z = torch.full((self.n_envs,), z_val, device=gs.device, dtype=gs.tc_float)
        pos = torch.stack([x, y, z], dim=-1)
        half = yaw * 0.5
        quat = torch.stack(
            [torch.cos(half), torch.zeros_like(half),
             torch.zeros_like(half), torch.sin(half)],
            dim=-1,
        )
        return pos, quat

    def _resolve_target_joint(self, joint_info, part_info, ctrl_meta):
        """Select the target joint to articulate."""
        # Prefer joints named in the control script.
        ctrl_joint_names = ctrl_meta.get("joint_names", [])
        movable_by_name = {j.name: j for j in joint_info.movable_joints}

        for name in ctrl_joint_names:
            if name in movable_by_name:
                j = movable_by_name[name]
                if j.limits[1] > j.limits[0]:
                    return j

        # Find movable joints connected to graspable parts.
        graspable_parts = part_info.get_graspable_parts() if part_info else []
        graspable_links = {p.parent_link for p in graspable_parts}

        candidates = []
        for j in joint_info.movable_joints:
            if j.limits[1] <= j.limits[0]:
                continue
            if j.body1 in graspable_links or j.body0 in graspable_links:
                candidates.append(j)

        if candidates:
            return max(candidates, key=lambda j: abs(j.limits[1] - j.limits[0]))

        # Fallback: any movable joint with non-zero range.
        movable = [j for j in joint_info.movable_joints if j.limits[1] > j.limits[0]]
        if movable:
            return max(movable, key=lambda j: abs(j.limits[1] - j.limits[0]))

        raise ValueError(
            f"No movable joint with non-zero range found in {self._artvip_object}"
        )

    def _resolve_joint_geometry(self, target_joint, usd_path) -> None:
        """Compute joint axis and origin in object-root local frame.

        The USD joint prim's world transform gives us the joint axis and a
        point on the axis in USD world coordinates. We transform these into
        the object root's local frame so that at runtime we can rotate them
        into Genesis world coordinates using the entity's current pose.

        Stores:
          self._joint_axis_local  : (3,) torch tensor, object-root local
          self._joint_origin_local: (3,) torch tensor, object-root local
        """
        from pxr import Usd, UsdGeom, Gf

        try:
            stage = Usd.Stage.Open(str(usd_path))
        except Exception:
            self._joint_axis_local = None
            self._joint_origin_local = None
            return

        root = stage.GetDefaultPrim()
        if not root or not root.IsValid():
            root = stage.GetPseudoRoot()

        # Root world transform (USD world → root local).
        root_xform = UsdGeom.Xformable(root)
        root_world = root_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        root_inv = root_world.GetInverse()

        # Joint prim world transform.
        joint_prim = stage.GetPrimAtPath(target_joint.prim_path)
        if not joint_prim or not joint_prim.IsValid():
            self._joint_axis_local = None
            self._joint_origin_local = None
            return

        joint_xform = UsdGeom.Xformable(joint_prim)
        joint_world = joint_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        # Joint axis in USD world frame (direction → TransformDir).
        axis_usd = np.array(target_joint.axis, dtype=np.float64)
        axis_world = joint_world.TransformDir(Gf.Vec3d(*axis_usd))

        # Joint origin in USD world frame (point → TransformAffine).
        origin_world = joint_world.ExtractTranslation()

        # Transform to root local frame using inverse root matrix.
        axis_local = root_inv.TransformDir(axis_world)
        origin_local = root_inv.TransformAffine(origin_world)

        self._joint_axis_local = torch.tensor(
            [axis_local[0], axis_local[1], axis_local[2]],
            device=gs.device, dtype=gs.tc_float,
        )
        self._joint_origin_local = torch.tensor(
            [origin_local[0], origin_local[1], origin_local[2]],
            device=gs.device, dtype=gs.tc_float,
        )

    # ------------------------------------------------------------------
    # Runtime articulation queries
    # ------------------------------------------------------------------

    def get_target_joint_pos(self) -> torch.Tensor:
        """Read the target joint position for all envs. Returns (n_envs, 1)."""
        if self._target_dof_idx is None:
            return torch.zeros(self.n_envs, 1, device=gs.device, dtype=gs.tc_float)
        pos = self.obj.get_dofs_position(
            dofs_idx_local=self._target_dof_idx,
            envs_idx=self.envs_all,
        )
        if pos.ndim == 1:
            pos = pos.unsqueeze(-1)
        return pos

    def get_graspable_part_pos(self) -> torch.Tensor:
        """Get the graspable part world position for all envs. Returns (n_envs, 3)."""
        if self._graspable_link is not None:
            pos = self._graspable_link.get_pos(envs_idx=self.envs_all)
        else:
            pos = self._obj_pos()
        if pos.ndim == 1:
            pos = pos.unsqueeze(0)
        return pos

    def _quat_rotate_batch(self, quat_wxyz: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """Rotate batched vectors by batched WXYZ quaternions."""
        from hsr_genesis.grasp_planning import quat_rotate_batch
        return quat_rotate_batch(quat_wxyz, vec)

    def get_joint_axis_world(self) -> torch.Tensor:
        """Transform the stored joint axis to world frame using current object pose.

        Returns (n_envs, 3) world-frame axis vectors.
        """
        if self._joint_axis_local is None:
            return torch.zeros(self.n_envs, 3, device=gs.device, dtype=gs.tc_float)
        obj_quat = self.obj.get_quat(envs_idx=self.envs_all)
        if obj_quat.ndim == 1:
            obj_quat = obj_quat.unsqueeze(0)
        return self._quat_rotate_batch(obj_quat, self._joint_axis_local)

    def get_joint_origin_world(self) -> torch.Tensor:
        """Transform the stored joint origin to world frame using current object pose.

        Returns (n_envs, 3) world-frame positions on the joint axis.
        """
        if self._joint_origin_local is None:
            return self._obj_pos()
        obj_pos = self.obj.get_pos(envs_idx=self.envs_all)
        obj_quat = self.obj.get_quat(envs_idx=self.envs_all)
        if obj_pos.ndim == 1:
            obj_pos = obj_pos.unsqueeze(0)
        if obj_quat.ndim == 1:
            obj_quat = obj_quat.unsqueeze(0)
        rotated = self._quat_rotate_batch(obj_quat, self._joint_origin_local)
        return obj_pos + rotated

    def get_articulation_direction(self, grasp_pos: torch.Tensor) -> torch.Tensor:
        """Compute the world-space direction to pull the graspable part to open the joint.

        For **prismatic** joints: the direction is the joint axis (in world),
        oriented toward the open limit (the limit further from the current
        joint position).

        For **revolute** joints: the direction is the tangent to the
        articulation arc at the grasp point, computed as
        ``cross(axis_world, r)`` where ``r = grasp_pos - axis_origin_world``.
        The tangent is oriented toward the open limit.

        Args:
            grasp_pos: (n_envs, 3) world-frame grasp point (e.g. graspable
                part position).

        Returns:
            (n_envs, 3) normalized world-frame articulation directions.
        """
        axis_world = self.get_joint_axis_world()  # (N, 3)
        origin_world = self.get_joint_origin_world()  # (N, 3)

        if self._target_joint_type == "prismatic":
            direction = axis_world
        else:
            # Revolute: tangent = cross(axis, r), r = grasp - origin.
            r = grasp_pos - origin_world  # (N, 3)
            direction = torch.cross(axis_world, r, dim=-1)

        # Normalize.
        norm = direction.norm(dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=1e-6)
        direction = direction / norm

        # Orient toward open limit.
        # The "open" direction is from the current joint position toward the
        # limit that's further away.  At reset the joint is typically at one
        # limit (closed), so we move toward the other.
        qpos = self.get_target_joint_pos()  # (N, 1)
        lower, upper = self._target_joint_limits
        # If closer to lower, open toward upper; vice versa.
        open_sign = torch.where(
            (qpos - lower).abs() < (qpos - upper).abs(),
            torch.ones_like(qpos),   # move toward upper → positive
            -torch.ones_like(qpos),  # move toward lower → negative
        )
        direction = direction * open_sign

        return direction

    def get_surface_normal(self, grasp_pos: torch.Tensor) -> torch.Tensor:
        """Compute the surface normal of the graspable part in world frame.

        For **revolute** joints (hinged door/lid): the surface normal is
        ``cross(axis_world, r)`` where ``r = grasp_pos - joint_origin``.
        This gives a vector perpendicular to the door surface, pointing
        outward from the hinge side.

        For **prismatic** joints (drawer): the surface normal is the joint
        axis direction — the drawer face is perpendicular to the slide
        direction.

        Args:
            grasp_pos: (N, 3) world-frame grasp point.

        Returns:
            (N, 3) normalized surface normal vectors (horizontal component
            only, z zeroed — the robot approaches horizontally).
        """
        axis_world = self.get_joint_axis_world()  # (N, 3)
        origin_world = self.get_joint_origin_world()  # (N, 3)

        if self._target_joint_type == "prismatic":
            normal = axis_world
        else:
            # Revolute: normal = cross(axis, r), perpendicular to door surface.
            r = grasp_pos - origin_world  # (N, 3)
            normal = torch.cross(axis_world, r, dim=-1)

        # Zero out z component — robot approaches horizontally.
        normal[:, 2] = 0.0
        norm = normal.norm(dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=1e-6)
        return normal / norm

    # ------------------------------------------------------------------
    # GraspTargetProvider overrides (articulated object).
    # ------------------------------------------------------------------

    def part_pos(self, obj_pos: torch.Tensor, obj_quat: torch.Tensor) -> torch.Tensor:
        """graspable-part world position.  Prefers the live link pose (links
        articulate over time); falls back to obj_pos + rotated local offset."""
        if self._graspable_link is not None:
            pos = self._graspable_link.get_pos(envs_idx=self.envs_all)
            return pos.unsqueeze(0) if pos.ndim == 1 else pos
        if self._graspable_local_offset_3d is not None:
            return obj_pos + self._quat_rotate_batch(
                obj_quat, self._graspable_local_offset_3d,
            )
        return obj_pos.clone()

    def surface_normal(self, part_pos: torch.Tensor, obj_quat: torch.Tensor) -> torch.Tensor | None:
        """Surface normal from stored local joint geometry (analytic)."""
        if self._joint_axis_local is None or self._joint_origin_local is None:
            return None
        N = part_pos.shape[0]
        axis_local = self._joint_axis_local.unsqueeze(0).expand(N, -1)
        origin_local = self._joint_origin_local.unsqueeze(0).expand(N, -1)

        axis_world = self._quat_rotate_batch(obj_quat, axis_local)
        origin_world = self._obj_pos() + self._quat_rotate_batch(obj_quat, origin_local)

        if self._target_joint_type == "prismatic":
            normal = axis_world
        else:
            r = part_pos - origin_world
            normal = torch.cross(axis_world, r, dim=-1)

        normal[:, 2] = 0.0
        norm = normal.norm(dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=1e-6)
        normal = normal / norm
        return SweepGraspPlanner.orient_normal_toward_robot(normal, part_pos)

    def approach_dirs(
        self, part_pos: torch.Tensor, surface_normal: torch.Tensor | None,
    ) -> list[torch.Tensor] | None:
        """4 directions relative to the handle's surface normal.

        Front (+normal), back (-normal, usually inside the object —
        rejected by the collision check), and both perpendiculars.
        Returns None when no normal is available.
        """
        if surface_normal is None:
            return None
        n = surface_normal.clone()
        n[:, 2] = 0.0
        n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        nx, ny = n[:, 0], n[:, 1]
        z = torch.zeros_like(nx)
        return [
            torch.stack([nx, ny, z], dim=-1),
            torch.stack([-nx, -ny, z], dim=-1),
            torch.stack([-ny, nx, z], dim=-1),
            torch.stack([ny, -nx, z], dim=-1),
        ]

    def yaw_candidates(self, obj_pos: torch.Tensor) -> list[torch.Tensor]:
        """±60° sweep around the yaw facing the graspable part toward the
        robot (when the local offset is known), else full circle."""
        N = self.n_envs
        device = gs.device
        if self._graspable_local_offset is not None:
            offset = self._graspable_local_offset[0]  # (2,)
            local_angle = torch.atan2(offset[1], offset[0])
            x, y = obj_pos[0, 0], obj_pos[0, 1]
            target_angle = torch.atan2(-y, -x)
            base_yaw = target_angle - local_angle
            base_yaw = base_yaw.unsqueeze(0).expand(N) if base_yaw.ndim == 0 else base_yaw
            deltas = torch.linspace(
                -math.pi / 3, math.pi / 3, 7, device=device, dtype=gs.tc_float,
            )
            return [(base_yaw + d).expand(N) for d in deltas]
        return super().yaw_candidates(obj_pos)

    def terminal_target(self, grasp_pos: torch.Tensor, descend_pos: torch.Tensor) -> torch.Tensor:
        """Articulation IK target: pull along the joint's articulation
        direction by 50% of its physical range."""
        direction = self.get_articulation_direction(grasp_pos)  # (N, 3)
        lower, upper = self._target_joint_limits
        joint_range = abs(upper - lower)
        n = grasp_pos.shape[0]
        if self._target_joint_type == "prismatic":
            displacement = torch.full(
                (n,), joint_range * 0.5,
                device=grasp_pos.device, dtype=grasp_pos.dtype,
            )
        else:
            # Revolute: arc length = radius * angle.
            origin = self.get_joint_origin_world()
            radius = (grasp_pos - origin).norm(dim=-1)
            displacement = (radius * joint_range * 0.5).clamp(min=0.05, max=0.30)
        return grasp_pos + direction * displacement.unsqueeze(-1)




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--object", type=str, default="random",
                        help="YCB model name or 'random'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--settle-steps", type=int, default=200)
    parser.add_argument("--trials", type=int, default=1,
                        help="Number of independent pick trials to evaluate")
    parser.add_argument("--debug", action="store_true",
                        help="Print per-phase EE/object positions")
    args = parser.parse_args()

    try:
        gs.init(backend=gs.gpu)
    except RuntimeError as exc:
        print(f"[Genesis] GPU unavailable ({exc}); falling back to CPU.")
        gs.init(backend=gs.cpu)

    object_name = args.object
    if object_name == "random":
        object_name = YCB_MODELS[np.random.default_rng(args.seed).integers(len(YCB_MODELS))]
    print(f"[setup] object={object_name} envs={args.envs} trials={args.trials}")

    env = HSRPickEnv(
        n_envs=args.envs,
        object_name=object_name,
        show_viewer=not args.no_viewer,
        seed=args.seed,
    )

    all_rates = []
    all_times = []
    for trial in range(args.trials):
        t0 = time.time()
        summary = env.run_pick_pipeline(settle_steps=args.settle_steps, debug=args.debug)
        dt = time.time() - t0
        all_rates.append(summary["success_rate"])
        if not math.isnan(summary["avg_time_to_success"]):
            all_times.append(summary["avg_time_to_success"])
        print(
            f"[trial {trial}] success_rate={summary['success_rate']:.2%} "
            f"n={summary['n_success']}/{summary['n_envs']} "
            f"avg_steps={summary['avg_steps_to_success']:.1f} "
            f"avg_time={summary['avg_time_to_success']:.3f}s "
            f"({dt:.1f}s wall)"
        )

    if args.trials > 1:
        print(
            f"\n[summary] mean_success_rate={np.mean(all_rates):.2%} "
            f"(over {args.trials} trials)"
        )
        if all_times:
            print(f"[summary] mean_time_to_success={np.mean(all_times):.3f}s")


if __name__ == "__main__":
    main()
