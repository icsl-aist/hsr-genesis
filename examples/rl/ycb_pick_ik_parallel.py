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
APPROACH_DURATION = 4.0
DESCEND_DURATION = 2.0
GRASP_HOLD_STEPS = 300      # ~6 s at dt=0.02
LIFT_DURATION = 2.0
LIFT_HOLD_STEPS = 100

HAND_QUAT = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # palm-down


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _yaw_from_quat_wxyz_batch(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


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
    ) -> None:
        self.n_envs = int(n_envs)
        self.dt = 0.02
        self.rng = torch.Generator(device=gs.device).manual_seed(int(seed))

        scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.5, -2.0, 2.0),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                max_FPS=60,
            ),
            vis_options=gs.options.VisOptions(
                show_world_frame=True,
                world_frame_size=1.0,
                show_link_frame=False,
                plane_reflection=True,
                ambient_light=(0.3, 0.3, 0.3),
            ),
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=4),
            rigid_options=gs.options.RigidOptions(use_gjk_collision=True),
            show_viewer=show_viewer,
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

        from hsr_genesis.sdf_parser import load_sdf_model
        model_dir = MODELS_DIR / object_name
        if not model_dir.exists():
            raise FileNotFoundError(f"YCB model not found: {model_dir}")
        obj_urdf = load_sdf_model(model_dir)
        self.obj = scene.add_entity(
            gs.morphs.URDF(file=obj_urdf, pos=(0.5, 0.0, OBJ_Z), fixed=False),
        )

        if disable_visualizer:
            scene._visualizer.build = lambda: None
        scene.build(n_envs=self.n_envs, env_spacing=(3.0, 3.0))
        self.scene = scene

        # Apply tuned PD gains (arm_lift gravity comp, etc.).
        self.hsr._hsr_apply_default_gains()

        self.envs_all = torch.arange(self.n_envs, device=gs.device, dtype=gs.tc_int)

        # End-effector setup.
        self.ee_link = self.hsr.get_link("hand_palm_link")
        self.hsr.end_effector_offset = [0.0, 0.0, 0.09]
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rand(self, shape) -> torch.Tensor:
        return torch.rand(shape, generator=self.rng, device=gs.device, dtype=gs.tc_float)

    def _obj_pos(self) -> torch.Tensor:
        return self.obj.get_pos(envs_idx=self.envs_all)

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
        theta = OBJ_ANG_MIN + (OBJ_ANG_MAX - OBJ_ANG_MIN) * self._rand((self.n_envs,))
        radius = OBJ_RADIUS_MIN + (OBJ_RADIUS_MAX - OBJ_RADIUS_MIN) * self._rand((self.n_envs,))
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
        self.obj.set_pos(pos, envs_idx=self.envs_all, zero_velocity=True, relative=False)
        self.obj.set_quat(quat, envs_idx=self.envs_all, zero_velocity=True, relative=False)

        self.steps_to_success[:] = -1
        self.total_steps[:] = 0

        self._settle(settle_steps)
        # Record settled object z as the success baseline.
        self.obj_init_z = self._obj_pos()[:, 2]

    # ------------------------------------------------------------------
    # IK helpers (batched).
    # ------------------------------------------------------------------

    def _ik(self, goal_pos: torch.Tensor, *, init_qpos=None):
        """IK to ``goal_pos`` (n_envs, 3) using batched analytic solver.

        ``inverse_kinematics`` uses ``torch.as_tensor`` internally (which
        accepts non-contiguous memory), so a single batched call replaces
        the per-env loop.  Returns (qpos (n_envs, n_qs), success_mask
        (n_envs,)).
        """
        quat = torch.tensor(
            HAND_QUAT, device=gs.device, dtype=gs.tc_float,
        ).expand(self.n_envs, -1).contiguous()

        qpos, error = self.hsr.inverse_kinematics(
            link=self.ee_link,
            pos=goal_pos.contiguous(),
            quat=quat,
            init_qpos=init_qpos.contiguous() if init_qpos is not None else None,
            max_samples=200,
            max_solver_iters=150,
            max_step_size=0.7,
            respect_joint_limit=False,
            envs_idx=self.envs_all,
            return_error=True,
        )
        if error.ndim == 1:
            error = error.unsqueeze(0)
        if qpos.ndim == 1:
            qpos = qpos.unsqueeze(0)
        pos_err = error[:, :3].norm(dim=1)
        success = torch.isfinite(pos_err) & (pos_err < 0.02)
        return qpos, success

    def _qpos_to_arm_and_base(self, qpos: torch.Tensor):
        """Extract per-env arm joint angles (n_envs, 5) and base [x,y,yaw].

        IK fills ``qpos`` at the generalized-coordinate (qs) indices, which
        differ from dof indices because the root free joint has 7 qs but 6
        dofs.  For revolute arm joints the q value equals the dof position,
        so we extract from ``arm_qs_idx``.
        """
        arm = qpos[:, self.arm_qs_idx]
        base_pos = qpos[:, self.base_qs_idx[0:3]]
        base_quat = qpos[:, self.base_qs_idx[3:7]]
        yaw = _yaw_from_quat_wxyz_batch(base_quat)
        base_xy_yaw = torch.stack([base_pos[:, 0], base_pos[:, 1], yaw], dim=-1)
        return arm, base_xy_yaw

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
        n_steps = int(duration / self.dt) + 50
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
        self._run_gripper_hold(int(LIFT_DURATION / self.dt) + 50)
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
