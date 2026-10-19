"""General grasp planning: phase targets, batched IK solving, and IK sweeps.

Moved out of ``examples/rl/`` into the library so planners, solvers, and the
visualization helpers are reusable outside the RL example scripts.

Components:

- ``IKPlan`` / ``IKPlanner`` — precomputed per-phase arm+base targets for the
  pick pipeline (approach → descend → grasp → lift/terminal).  ``IKPlanner``
  is the baseline planner (single IK per phase); ``SweepGraspPlanner`` adds
  the yaw + approach-direction sweep.
- ``GraspSolver`` — owns the batched-IK plumbing (analytic IK, sim-FK
  calibration, position-only Jacobian fallback, base-drift rejection).  A
  solver is bound to a robot/ee/scene and is shared by envs and planners.
- ``GraspTargetProvider`` — protocol describing what a sweep planner needs
  from the world (object pose, graspable-part geometry, yaw candidates,
  terminal-phase target).  Envs implement it.
- ``SweepDiagnostics`` — recorded per ``plan()`` call so
  ``plan_visualization.draw_sweep_debug`` can render the swept candidates.

Numerics are preserved verbatim from the original examples:
pos_err threshold 0.02 m, base-drift tolerance 0.05 m, IK scan with
``max_samples=200``, ``max_solver_iters=150``.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch

import genesis as gs

logger = logging.getLogger(__name__)


# Phase durations in steps (640 total at dt=0.02 = 12.8s)
# Matches optimized CMA-ES pipeline timing: approach 3s+30, descend 1.5s+30, grasp 200, lift 1.5s+30+50
APPROACH_STEPS = 180   # 3.6s (3.0s + 30 settle)
DESCEND_STEPS = 105    # 2.1s (1.5s + 30 settle)
GRASP_STEPS = 200      # 4.0s (gripper close + hold)
LIFT_STEPS = 155       # 3.1s (lift motion + settle + success check)

# Phase boundaries (start step for each phase)
APPROACH_START = 0
DESCEND_START = APPROACH_STEPS      # 180
GRASP_START = DESCEND_START + DESCEND_STEPS  # 285
LIFT_START = GRASP_START + GRASP_STEPS       # 485
MAX_STEPS = LIFT_START + LIFT_STEPS          # 640

# Fixed grasp params for IK planning (can be overridden by CMA-ES results)
PRE_GRASP_HEIGHT = 0.15
GRASP_OFFSET_Z = 0.02
LIFT_HEIGHT = 0.30
DEFAULT_GRIPPER_EFFORT = 3.0
DEFAULT_GRASP_HOLD_STEPS = 300

# Palm-down hand orientation (WXYZ) used as the default IK orientation.
HAND_QUAT = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

# Shared IK tolerances (kept identical to the original example code).
POS_ERR_THRESHOLD = 0.02      # m — IK "success" position-error cutoff
BASE_DRIFT_TOL = 0.05         # m — reject solutions that move the base x/y
IK_SCAN_MAX_SAMPLES = 200
IK_SCAN_MAX_SOLVER_ITERS = 150

# CMA-ES optimized params per object (loaded from results if available)
_CMAES_PARAMS: dict[str, dict] | None = None


def load_cmaes_params(path: str = "results/grasp_small/grasp_cmaes_best.json") -> dict[str, dict] | None:
    """Load CMA-ES optimized grasp params from JSON file."""
    global _CMAES_PARAMS
    if _CMAES_PARAMS is not None:
        return _CMAES_PARAMS
    p = Path(path)
    if p.exists():
        with open(p) as f:
            data = json.load(f)
        _CMAES_PARAMS = data.get("params", None)
    return _CMAES_PARAMS


def get_object_params(object_name: str) -> dict:
    """Get grasp params for an object, preferring CMA-ES optimized values."""
    params = load_cmaes_params()
    if params and object_name in params:
        return params[object_name]
    return {
        "pre_grasp_height": PRE_GRASP_HEIGHT,
        "grasp_offset_z": GRASP_OFFSET_Z,
        "gripper_effort": DEFAULT_GRIPPER_EFFORT,
        "grasp_hold_steps": DEFAULT_GRASP_HOLD_STEPS,
    }


# ---------------------------------------------------------------------------
# Batched WXYZ quaternion helpers (pure tensor ops)
# ---------------------------------------------------------------------------

def quat_rotate_batch(quat_wxyz: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate batched vectors by batched WXYZ quaternions.

    Args:
        quat_wxyz: (N, 4) quaternions in [w, x, y, z] format.
        vec: (3,) or (N, 3) vectors to rotate. If (3,), broadcast.

    Returns:
        (N, 3) rotated vectors.
    """
    if vec.ndim == 1:
        vec = vec.unsqueeze(0).expand(quat_wxyz.shape[0], -1)
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    vx, vy, vz = vec[:, 0], vec[:, 1], vec[:, 2]
    # Step 1: q ⊗ v (treat v as pure quaternion [0, vx, vy, vz]).
    tw = -x * vx - y * vy - z * vz
    tx = w * vx + y * vz - z * vy
    ty = w * vy + z * vx - x * vz
    tz = w * vz + x * vy - y * vx
    # Step 2: (q ⊗ v) ⊗ q*  where q* = [w, -x, -y, -z].
    rx = -tw * x + tx * w - ty * z + tz * y
    ry = -tw * y + ty * w + tx * z - tz * x
    rz = -tw * z + tz * w - tx * y + ty * x
    return torch.stack([rx, ry, rz], dim=-1)


def quat_z_to_vec_batch(target: torch.Tensor) -> torch.Tensor:
    """Quaternion (WXYZ) rotating world +Z to ``target`` direction.

    The HSR hand's palm faces +Z in the identity orientation.
    ``HAND_QUAT = [0,1,0,0]`` rotates +Z to -Z (palm-down).  This function
    computes the quaternion that rotates +Z to any target direction, which
    directly gives the hand orientation for that palm direction.

    Args:
        target: (N, 3) desired palm direction (need not be normalized).

    Returns:
        (N, 4) WXYZ quaternions.
    """
    t = target / target.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    z = torch.zeros_like(t)
    z[:, 2] = 1.0
    # axis = cross(z, t), angle = acos(dot(z, t))
    axis = torch.cross(z, t, dim=-1)  # (N, 3)
    dot = (z * t).sum(dim=-1).clamp(-1.0, 1.0)  # (N,)
    angle = torch.acos(dot)  # (N,)
    # Handle near-parallel (axis ≈ 0): fall back to 180° around X.
    axis_norm = axis.norm(dim=-1, keepdim=True)  # (N, 1)
    parallel = axis_norm.squeeze(-1) < 1e-6  # (N,)
    axis = axis / axis_norm.clamp(min=1e-8)
    # For anti-parallel (dot ≈ -1), use X axis.
    axis[parallel & (dot < 0)] = torch.tensor(
        [1.0, 0.0, 0.0], device=t.device, dtype=t.dtype,
    )
    half = angle * 0.5
    w = torch.cos(half)  # (N,)
    s = torch.sin(half)  # (N,)
    return torch.stack([
        w, axis[:, 0] * s, axis[:, 1] * s, axis[:, 2] * s,
    ], dim=-1)  # (N, 4)


def quat_from_palm_and_jaw(
    palm_direction: torch.Tensor, jaw_axis: torch.Tensor,
) -> torch.Tensor:
    """Build WXYZ quaternion with local +Z=palm and local +Y=jaws."""
    z_axis = palm_direction / palm_direction.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    y_axis = jaw_axis - (jaw_axis * z_axis).sum(dim=-1, keepdim=True) * z_axis
    y_axis = y_axis / y_axis.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    x_axis = torch.cross(y_axis, z_axis, dim=-1)
    matrix = torch.stack([x_axis, y_axis, z_axis], dim=-1)

    trace = matrix[:, 0, 0] + matrix[:, 1, 1] + matrix[:, 2, 2]
    w = torch.sqrt(torch.clamp(1.0 + trace, min=1e-8)) * 0.5
    denom = 4.0 * w.clamp(min=1e-8)
    x = (matrix[:, 2, 1] - matrix[:, 1, 2]) / denom
    y = (matrix[:, 0, 2] - matrix[:, 2, 0]) / denom
    z = (matrix[:, 1, 0] - matrix[:, 0, 1]) / denom
    return torch.stack([w, x, y, z], dim=-1)


def yaw_to_quat_wxyz_batch(yaw: torch.Tensor) -> torch.Tensor:
    """(N,) yaw angles → (N, 4) WXYZ quaternions about world +Z."""
    half = yaw * 0.5
    return torch.stack([
        torch.cos(half), torch.zeros_like(half),
        torch.zeros_like(half), torch.sin(half),
    ], dim=-1)


# ---------------------------------------------------------------------------
# Plan dataclass + baseline planner
# ---------------------------------------------------------------------------

@dataclass
class IKPlan:
    """Precomputed IK targets for one episode."""
    # Per-env IK targets for each phase: (n_envs, 5) arm, (n_envs, 3) base
    approach_arm: torch.Tensor
    approach_base: torch.Tensor
    descend_arm: torch.Tensor
    descend_base: torch.Tensor   # same as approach_base (base held during descend)
    lift_arm: torch.Tensor
    lift_base: torch.Tensor      # same as approach_base (base held during lift)
    # Episode metadata
    obj_pos: torch.Tensor        # (n_envs, 3) object position at episode start
    obj_init_z: torch.Tensor     # (n_envs,) object initial z for success check
    ik_success: torch.Tensor     # (n_envs, 3) success for each phase IK
    # Grasp params used for this episode
    pre_grasp_height: float
    grasp_offset_z: float
    gripper_effort: float
    grasp_hold_steps: int


class IKPlanner:
    """Plans IK targets for all 4 phases of the pick pipeline.

    The baseline planner computes a single IK solution per phase via the
    env's solver (``env.solver.solve`` / ``env._ik``).  Subclasses like
    ``SweepGraspPlanner`` override ``plan()`` with a candidate sweep.

    ``plan(env, init_qpos=None)`` takes the pick env directly — the env both
    owns the solver and satisfies :class:`GraspTargetProvider`.
    """

    # Configurable phase durations (steps). Override via configure_phase_steps()
    # before creating episodes. Defaults match the module-level constants.
    approach_steps: int = APPROACH_STEPS
    descend_steps: int = DESCEND_STEPS
    grasp_steps: int = GRASP_STEPS
    lift_steps: int = LIFT_STEPS

    @classmethod
    def configure_phase_steps(
        cls,
        approach: int | None = None,
        descend: int | None = None,
        grasp: int | None = None,
        lift: int | None = None,
    ) -> None:
        """Override phase durations (in simulation steps)."""
        if approach is not None:
            cls.approach_steps = approach
        if descend is not None:
            cls.descend_steps = descend
        if grasp is not None:
            cls.grasp_steps = grasp
        if lift is not None:
            cls.lift_steps = lift

    @classmethod
    def descend_start(cls) -> int:
        return cls.approach_steps

    @classmethod
    def grasp_start(cls) -> int:
        return cls.approach_steps + cls.descend_steps

    @classmethod
    def lift_start(cls) -> int:
        return cls.approach_steps + cls.descend_steps + cls.grasp_steps

    @classmethod
    def max_steps(cls) -> int:
        return cls.approach_steps + cls.descend_steps + cls.grasp_steps + cls.lift_steps

    def __init__(self, object_name: str = "ycb_061_foam_brick") -> None:
        self.object_name = object_name
        self.params = get_object_params(object_name)

    def plan(self, env, *, init_qpos: torch.Tensor | None = None) -> IKPlan:
        """Compute IK targets for approach, descend, and lift phases.

        Args:
            env: pick env instance with objects already settled (provides
                ``_obj_pos``, ``_ik``, ``_qpos_to_arm_and_base``, ``hsr``,
                ``envs_all`` — see ``HSRPickEnv``).
            init_qpos: Optional initial configuration for the approach IK.

        Returns:
            IKPlan with per-env targets for all phases.
        """
        obj_pos = env._obj_pos()

        # Phase 1: approach — hover above object
        pre_grasp = obj_pos.clone()
        pre_grasp[:, 2] += self.params["pre_grasp_height"]
        qpos_approach, ok_approach = env._ik(pre_grasp, init_qpos=init_qpos)
        approach_arm, approach_base = env._qpos_to_arm_and_base(qpos_approach)

        # Phase 2: descend — go to grasp pose (base held)
        grasp = obj_pos.clone()
        grasp[:, 2] += self.params["grasp_offset_z"]
        cur_qpos = env.hsr.get_qpos(envs_idx=env.envs_all)
        if cur_qpos.ndim == 1:
            cur_qpos = cur_qpos.unsqueeze(0)
        qpos_descend, ok_descend = env._ik(grasp, init_qpos=cur_qpos)
        descend_arm, _descend_base = env._qpos_to_arm_and_base(qpos_descend)
        descend_base = approach_base.clone()  # base held during descend

        # Phase 3: grasp — no IK needed, gripper closes at current position

        # Phase 4: lift — raise object straight up. The arm IK target is
        # directly above the object; base held during lift.
        lift = obj_pos.clone()
        lift[:, 2] = LIFT_HEIGHT
        cur_qpos2 = env.hsr.get_qpos(envs_idx=env.envs_all)
        if cur_qpos2.ndim == 1:
            cur_qpos2 = cur_qpos2.unsqueeze(0)
        qpos_lift, ok_lift = env._ik(lift, init_qpos=cur_qpos2)
        lift_arm, lift_base = env._qpos_to_arm_and_base(qpos_lift)

        ik_success = torch.stack([ok_approach, ok_descend, ok_lift], dim=1)

        return IKPlan(
            approach_arm=approach_arm,
            approach_base=approach_base,
            descend_arm=descend_arm,
            descend_base=descend_base,
            lift_arm=lift_arm,
            lift_base=lift_base,
            obj_pos=obj_pos,
            obj_init_z=obj_pos[:, 2].clone(),
            ik_success=ik_success,
            pre_grasp_height=self.params["pre_grasp_height"],
            grasp_offset_z=self.params["grasp_offset_z"],
            gripper_effort=self.params["gripper_effort"],
            grasp_hold_steps=int(self.params["grasp_hold_steps"]),
        )

    @staticmethod
    def get_phase(step: int) -> int:
        """Return phase index (0=approach, 1=descend, 2=grasp, 3=lift) for a given step."""
        if step < IKPlanner.descend_start():
            return 0  # approach
        elif step < IKPlanner.grasp_start():
            return 1  # descend
        elif step < IKPlanner.lift_start():
            return 2  # grasp
        else:
            return 3  # lift

    @staticmethod
    def get_phase_targets(plan: IKPlan, phase: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get (arm_target, base_target) for a given phase."""
        if phase == 0:
            return plan.approach_arm, plan.approach_base
        elif phase == 1:
            return plan.descend_arm, plan.descend_base
        elif phase == 2:
            # Grasp: hold current arm position (use descend arm as static target)
            return plan.descend_arm, plan.descend_base
        else:
            return plan.lift_arm, plan.lift_base


def _pos_error_norm(error: torch.Tensor) -> torch.Tensor:
    """(n_envs, >=3) batched IK error → (n_envs,) position-error norm."""
    if error.ndim == 1:
        error = error.unsqueeze(0)
    return error[:, :3].norm(dim=1)


def _ensure_batched(t: torch.Tensor) -> torch.Tensor:
    """Promote (D,) → (1, D); pass through (N, D)."""
    return t.unsqueeze(0) if t.ndim == 1 else t



# ---------------------------------------------------------------------------
# GraspSolver: batched IK plumbing shared by envs and planners
# ---------------------------------------------------------------------------

class GraspSolver:
    """Batched IK solving for a whole-body HSR robot in a parallel scene.

    Owns the IK plumbing that used to live on the pick env so planners and
    visualization can use it without depending on env internals:

    - ``solve`` — batched analytic IK (whole-body; base yaw free, x/y kept).
    - ``calibrated`` — analytic IK + simulation-FK feedback calibration,
      with base-drift rejection (rejects optimizer-fallback solutions that
      relocate the base).
    - ``jacobian`` — position-only Genesis IK restricted to arm dofs
      (base fixed).  Used when the analytic solver's optimizer fallback
      relocates the base to a wrong position.
    - ``ik_scan`` — raw IK used inside the candidate sweep (respects joint
      limits; ``return_error=True``).
    - ``qpos_to_arm_and_base`` / ``base_drift_ok`` — qpos decomposition and
      the drift predicate shared by the scan and calibration loops.
    """

    def __init__(
        self,
        hsr,
        ee_link,
        scene,
        envs_all: torch.Tensor,
        arm_dofs_idx,
        n_envs: int | None = None,
        dt: float = 0.02,
    ) -> None:
        self._hsr = hsr
        self._ee_link = ee_link
        self._scene = scene
        self._envs_all = envs_all
        self._arm_dofs_idx = arm_dofs_idx
        self._n_envs = int(n_envs) if n_envs is not None else int(envs_all.shape[0])
        self._dt = dt

    # -- internal ----------------------------------------------------------

    def _default_quat(self, hand_quat: torch.Tensor | None) -> torch.Tensor:
        if hand_quat is None:
            return torch.tensor(
                HAND_QUAT, device=gs.device, dtype=gs.tc_float,
            ).expand(self._n_envs, -1).contiguous()
        quat = hand_quat.to(device=gs.device, dtype=gs.tc_float)
        if quat.ndim == 1:
            quat = quat.unsqueeze(0).expand(self._n_envs, -1).contiguous()
        return quat

    def _unpack_error(self, error: torch.Tensor) -> torch.Tensor:
        """(n_envs, >=3) error → (n_envs,) position error."""
        return _pos_error_norm(error)

    _ensure_batched = staticmethod(_ensure_batched)

    def _success_mask(self, pos_err: torch.Tensor) -> torch.Tensor:
        return torch.isfinite(pos_err) & (pos_err < POS_ERR_THRESHOLD)

    # -- solving -----------------------------------------------------------

    def ik_scan(
        self,
        pos: torch.Tensor,
        quat: torch.Tensor,
        *,
        init_qpos: torch.Tensor | None = None,
        max_step_size: float = 0.1,
        respect_joint_limit: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw batched IK for the candidate sweep.

        Returns (qpos (n_envs, n_qs), pos_err (n_envs,)) — callers combine
        ``pos_err`` with :meth:`base_drift_ok` to decide reachability.
        """
        qpos, error = self._hsr.inverse_kinematics(
            link=self._ee_link,
            pos=pos.contiguous(),
            quat=quat.contiguous(),
            init_qpos=init_qpos.contiguous() if init_qpos is not None else None,
            max_samples=IK_SCAN_MAX_SAMPLES,
            max_solver_iters=IK_SCAN_MAX_SOLVER_ITERS,
            max_step_size=max_step_size,
            respect_joint_limit=respect_joint_limit,
            envs_idx=self._envs_all,
            return_error=True,
        )
        return self._ensure_batched(qpos), self._unpack_error(error)

    def solve(
        self,
        goal_pos: torch.Tensor,
        *,
        init_qpos: torch.Tensor | None = None,
        hand_quat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched analytic IK to ``goal_pos``. Returns (qpos, success_mask).

        Whole-body IK: the solver may adjust base yaw but keeps x/y fixed.
        """
        quat = self._default_quat(hand_quat)
        qpos, error = self._hsr.inverse_kinematics(
            link=self._ee_link,
            pos=goal_pos.contiguous(),
            quat=quat,
            init_qpos=init_qpos.contiguous() if init_qpos is not None else None,
            max_samples=IK_SCAN_MAX_SAMPLES,
            max_solver_iters=IK_SCAN_MAX_SOLVER_ITERS,
            max_step_size=0.7,
            respect_joint_limit=False,
            envs_idx=self._envs_all,
            return_error=True,
        )
        pos_err = self._unpack_error(error)
        return self._ensure_batched(qpos), self._success_mask(pos_err)

    def collision_free(
        self,
        qpos: torch.Tensor,
        *,
        envs_idx: torch.Tensor | list[int] | None = None,
    ) -> torch.Tensor:
        """Check whether each robot qpos is collision-free.

        Runs a real-time collision query (``RigidSolver.detect_collision``,
        which does not rely on ``scene.step``) after temporarily posing the
        robot at each candidate, then restores the live qpos.  Batched
        detection isn't supported, so checks run serially.

        A pair is a violation when it involves a robot geom and:
          - a non-floor entity (object, furniture, wall), or
          - another robot geom not already contact-filtered by the URDF.
        Floor contact is expected — the robot rests on it.

        Args:
            qpos: (K, n_qs) candidate joint configurations.
            envs_idx: env each row is checked in; defaults to
                ``arange(K)`` — row k checked in env k (one candidate
                per env, the sweep's common case).

        Returns:
            Bool tensor (K,) — True if no violating contact.
        """
        qpos = self._ensure_batched(qpos.to(gs.device))
        K = qpos.shape[0]
        ok = torch.ones(K, dtype=torch.bool, device=gs.device)

        hsr = self._hsr
        scene = self._scene
        solver = scene.sim.rigid_solver
        # Map geom idx -> entity idx; find the floor (fixed, 0 dofs).
        geom_ent = torch.full(
            (solver.n_geoms,), -1, device=gs.device, dtype=torch.int64,
        )
        floor_geoms = torch.zeros(
            (solver.n_geoms,), dtype=torch.bool, device=gs.device,
        )
        for ent in scene.entities:
            if not hasattr(ent, "geom_start"):
                continue
            a, b = int(ent.geom_start), int(ent.geom_end)
            geom_ent[a:b] = int(ent.idx)
            if getattr(ent, "n_dofs", 1) == 0:
                floor_geoms[a:b] = True

        # Save live qpos for every env we'll touch; restored after.
        if envs_idx is None:
            envs = list(range(K))
        else:
            envs = [int(i) for i in torch.as_tensor(envs_idx).tolist()]
        unique_envs = sorted(set(envs))
        cur = hsr.get_qpos(envs_idx=unique_envs)
        if cur.ndim == 1:
            cur = cur.unsqueeze(0)
        saved = {ei: cur[i].clone() for i, ei in enumerate(unique_envs)}

        try:
            for k in range(K):
                ei = envs[k]
                hsr.set_qpos(qpos[k], envs_idx=[ei])
                pairs = solver.detect_collision(ei)
                if pairs.shape[0] == 0:
                    continue
                pa = torch.from_numpy(pairs[:, 0]).to(gs.device)
                pb = torch.from_numpy(pairs[:, 1]).to(gs.device)
                ent_a = geom_ent[pa]
                ent_b = geom_ent[pb]
                robot_a = ent_a == hsr.idx
                robot_b = ent_b == hsr.idx
                floor_a = floor_geoms[pa]
                floor_b = floor_geoms[pb]
                viol = (
                    (robot_a & ~floor_b & (ent_b != hsr.idx))
                    | (robot_b & ~floor_a & (ent_a != hsr.idx))
                    | (robot_a & robot_b)
                )
                ok[k] = not bool(viol.any().item())
        finally:
            for ei, q0 in saved.items():
                hsr.set_qpos(q0, envs_idx=[ei])
        return ok

    def calibrated(
        self,
        goal_pos: torch.Tensor,
        *,
        init_qpos: torch.Tensor | None = None,
        hand_quat: torch.Tensor | None = None,
        max_iters: int = 3,
        respect_joint_limit: bool = True,
        max_step_size: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """IK with simulation-FK feedback calibration.

        The analytic IK model's FK doesn't perfectly match the Genesis
        simulation.  This refines the solution by setting the robot to the
        IK qpos, measuring the actual EE, and re-running IK with a corrected
        target.  Base-drift rejections keep the previous qpos.
        Returns (qpos, success_mask).
        """
        quat = self._default_quat(hand_quat)

        qpos = init_qpos
        adjusted_pos = goal_pos.clone()
        base_qs = self._hsr._ensure_base_qs_idx()
        pre_base_xy = init_qpos[:, [base_qs[0], base_qs[1]]] if init_qpos is not None else None
        qpos_prev = init_qpos
        for _ in range(max_iters):
            qpos, error = self._hsr.inverse_kinematics(
                link=self._ee_link,
                pos=adjusted_pos.contiguous(),
                quat=quat,
                init_qpos=qpos.contiguous() if qpos is not None else None,
                max_samples=IK_SCAN_MAX_SAMPLES,
                max_solver_iters=IK_SCAN_MAX_SOLVER_ITERS,
                max_step_size=max_step_size,
                respect_joint_limit=respect_joint_limit,
                envs_idx=self._envs_all,
                return_error=True,
            )
            error = self._ensure_batched(error)
            qpos = self._ensure_batched(qpos)
            # Reject solutions where the optimizer fallback moved the base.
            if pre_base_xy is not None:
                drift_ok = self.base_drift_ok(qpos, pre_base_xy)
                if not drift_ok.all():
                    qpos = torch.where(drift_ok.unsqueeze(-1), qpos, qpos_prev)
            # Calibrate: measure actual EE and adjust target.
            self._hsr.set_qpos(qpos, envs_idx=self._envs_all)
            self._scene.step()
            actual_ee = self._ee_link.get_pos(envs_idx=self._envs_all)
            actual_ee = self._ensure_batched(actual_ee)
            offset = goal_pos - actual_ee
            if offset.norm(dim=-1).max().item() < 0.01:
                break
            adjusted_pos = goal_pos + offset
            qpos_prev = qpos
        pos_err = self._unpack_error(error)
        return qpos, self._success_mask(pos_err)

    def jacobian(
        self,
        goal_pos: torch.Tensor,
        *,
        init_qpos: torch.Tensor | None = None,
        hand_quat: torch.Tensor | None = None,
        max_iters: int = 50,
        max_step_size: float = 0.1,
        pos_tol: float = 1e-3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Jacobian-based IK that keeps the base fixed.

        Uses the Genesis built-in IK with ``dofs_idx_local`` restricted to
        arm joints, so the optimizer cannot move the base.  Position-only:
        the 5-DOF arm cannot satisfy position + full orientation; orientation
        is approximated by base yaw and the arm's natural pose.

        Returns (qpos, success_mask).
        """
        from genesis.engine.entities.rigid_entity import RigidEntity

        # hand_quat is accepted for API symmetry; position-only IK ignores it.
        del hand_quat

        if init_qpos is None:
            init_qpos = self._hsr.get_qpos(envs_idx=self._envs_all)
        init_qpos = self._ensure_batched(init_qpos)

        qpos, error = RigidEntity.inverse_kinematics(
            self._hsr,
            link=self._ee_link,
            pos=goal_pos.contiguous(),
            quat=None,
            init_qpos=init_qpos.contiguous(),
            dofs_idx_local=self._arm_dofs_idx,
            max_samples=max_iters,
            max_solver_iters=max_iters,
            max_step_size=max_step_size,
            pos_tol=pos_tol,
            respect_joint_limit=True,
            envs_idx=self._envs_all,
            return_error=True,
        )
        pos_err = self._unpack_error(error)
        return self._ensure_batched(qpos), self._success_mask(pos_err)

    # -- qpos helpers --------------------------------------------------------

    def base_drift_ok(
        self,
        qpos: torch.Tensor,
        expected_xy: torch.Tensor,
        tol: float = BASE_DRIFT_TOL,
    ) -> torch.Tensor:
        """(n_envs,) mask: base x/y within ``tol`` of ``expected_xy``."""
        base_qs = self._hsr._ensure_base_qs_idx()
        sol_xy = qpos[:, [base_qs[0], base_qs[1]]]
        return (sol_xy - expected_xy).norm(dim=-1) < tol

    def qpos_to_arm_and_base(
        self, qpos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract (n_envs, 5) arm joint angles and (n_envs, 3) base [x,y,yaw].

        IK fills ``qpos`` at the generalized-coordinate (qs) indices.
        """
        arm_qs = self._hsr._ensure_arm_qs_idx()
        base_qs = self._hsr._ensure_base_qs_idx()
        arm = qpos[:, arm_qs]
        bx = qpos[:, base_qs[0]]
        by = qpos[:, base_qs[1]]
        quat = qpos[:, base_qs[3:7]]  # (N, 4) WXYZ
        yaw = _yaw_from_quat_wxyz_batch(quat)
        base = torch.stack([bx, by, yaw], dim=-1)
        return arm, base

    def settle(self, n_steps: int, hold_arm: bool = True) -> None:
        """Step the scene holding the robot at its current pose.

        ``hold_arm=True`` matches the pick env's ``_settle`` (loads a static
        whole-body trajectory); ``False`` just steps the scene.
        """
        if not hold_arm:
            for _ in range(n_steps):
                self._scene.step()
            return
        from hsr_genesis.hsr_rigid_entity import JointTrajectory
        from hsr_genesis.base_controller import Trajectory
        from hsr_genesis.analytic_ik import JOINT_ORDER

        arm_pos = self._hsr.get_dofs_position(
            dofs_idx_local=self._arm_dofs_idx, envs_idx=self._envs_all,
        )
        arm_pos = self._ensure_batched(arm_pos)
        base_pos = self._ensure_batched(self._hsr.get_pos(envs_idx=self._envs_all))
        base_quat = self._ensure_batched(self._hsr.get_quat(envs_idx=self._envs_all))
        yaw = _yaw_from_quat_wxyz_batch(base_quat)
        base_xy_yaw = torch.stack(
            [base_pos[:, 0], base_pos[:, 1], yaw], dim=-1,
        )
        hold_t = torch.tensor(
            [n_steps * self._dt], device=gs.device, dtype=gs.tc_float,
        )
        names = list(JOINT_ORDER)
        # API-bound: set_whole_body_trajectory_batched expects list[JointTrajectory] per env.
        arm_trajs = [
            JointTrajectory(
                positions=arm_pos[i].unsqueeze(0),
                time_from_start=hold_t,
                joint_names=names,
            )
            for i in range(self._n_envs)
        ]
        base_trajs = [
            Trajectory(positions=base_xy_yaw[i].unsqueeze(0), time_from_start=hold_t)
            for i in range(self._n_envs)
        ]
        self._hsr.set_whole_body_trajectory_batched(
            arm_trajectory=arm_trajs,
            base_trajectory=base_trajs,
            envs_idx=self._envs_all,
            start_time=None,
        )
        for _ in range(n_steps):
            self._hsr.step_whole_body_trajectory_batched(self._dt, envs_idx=self._envs_all)
            self._scene.step()


def _yaw_from_quat_wxyz_batch(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


# ---------------------------------------------------------------------------
# GraspTargetProvider + sweep diagnostics
# ---------------------------------------------------------------------------

@runtime_checkable
class GraspTargetProvider(Protocol):
    """What a sweep planner needs from the world.
    """

    n_envs: int

    def obj_pos(self) -> torch.Tensor:
        """(n_envs, 3) object root positions."""
        ...

    def obj_quat(self) -> torch.Tensor:
        """(n_envs, 4) object root quaternions (WXYZ)."""
        ...

    def part_pos(
        self, obj_pos: torch.Tensor, obj_quat: torch.Tensor,
    ) -> torch.Tensor:
        """(n_envs, 3) world position of the graspable part for a pose."""
        ...

    def surface_normal(
        self, part_pos: torch.Tensor, obj_quat: torch.Tensor,
    ) -> torch.Tensor | None:
        """(n_envs, 3) world surface normal at the grasp point, or None."""
        ...

    def jaw_axis(self, obj_quat: torch.Tensor) -> torch.Tensor | None:
        """(n_envs, 3) world jaw axis for nested handles, or None."""
        ...

    def yaw_candidates(self, obj_pos: torch.Tensor) -> list[torch.Tensor]:
        """Candidate object yaw angles: list of (n_envs,) tensors."""
        ...

    def approach_dirs(
        self, part_pos: torch.Tensor, surface_normal: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        """Candidate XY approach directions: list of (n_envs, 3) tensors.

        Each direction d places the robot base at ``part_xy - d*reach``
        and the pre-grasp EE at ``part_pos + d*pre_grasp_height``.  To
        sweep the base around the object, return several directions.
        """
        ...

    def set_obj_pose(self, quat: torch.Tensor) -> None:
        """Apply the winning object orientation (and settle the scene)."""
        ...

    def terminal_target(
        self, grasp_pos: torch.Tensor, descend_pos: torch.Tensor,
    ) -> torch.Tensor:
        """IK target for the terminal phase (lift or articulation)."""
        ...


@dataclass
class SweepDiagnostics:
    """Sweep results recorded by :meth:`SweepGraspPlanner.plan`.

    Populated every call so visualization can render the swept candidates.
    All per-candidate tensors are stacked over yaw × approach direction.
    """
    candidate_pos: torch.Tensor        # (n_cand, N, 3) pre-grasp targets
    candidate_quat: torch.Tensor       # (n_cand, N, 4) hand orientations
    candidate_dirs: torch.Tensor       # (n_cand, N, 3) approach directions
    candidate_scores: torch.Tensor     # (n_cand, N)
    candidate_reachable: torch.Tensor  # (n_cand, N) bool — pos_err<thr & feasible
    candidate_feasible: torch.Tensor   # (n_cand, N) bool — coll_free & base_ok
    candidate_qpos: torch.Tensor       # (n_cand, N, n_qs) solved IK qpos per candidate
    part_pos: torch.Tensor             # (N, 3) winning part position
    best_index: torch.Tensor           # (N,) flat index into n_cand
    yaws: list[torch.Tensor]           # swept yaw candidates


# ---------------------------------------------------------------------------
# Sweep planner (yaw × approach-direction candidate search)
# ---------------------------------------------------------------------------

class SweepGraspPlanner(IKPlanner):
    """IK planner that sweeps object yaw and hand approach direction.

    For each yaw candidate (from the target provider), part pose and surface
    normal are computed analytically from stored local geometry, candidate
    pre-grasp poses are generated, and batched IK runs per candidate.  The
    (yaw, candidate) with lowest score per env wins; the object is then
    rotated to the winning yaw.

    Approach IK is refined with analytic IK + sim-FK calibration first
    (handles orientation via base yaw); if calibration moves the base
    (optimizer fallback), position-only Jacobian IK keeps the base fixed.

    ``plan()`` takes the pick env directly (it satisfies
    :class:`GraspTargetProvider` and owns a ``solver``); the ``solver`` /
    ``target`` kwargs allow calling with an explicit solver/provider pair.
    """

    def __init__(
        self,
        target: GraspTargetProvider | None = None,
        target_joint_type: str = "revolute",
        target_joint_limits: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        # Skip CMA-ES param loading — articulation uses fixed params.
        self.object_name = "artvip"
        self.params = {
            "pre_grasp_height": 0.24,  # 0.15m tip + 0.09m gripper = link-origin distance
            "grasp_offset_z": 0.09,    # gripper tip at handle = 0.09m link origin from handle
            "gripper_effort": 3.0,
            "grasp_hold_steps": 300,
        }
        self._target = target
        self._target_joint_type = target_joint_type
        self._target_joint_limits = target_joint_limits
        # Approach-direction pitches (rad above horizontal) swept per
        # radial direction — 0=side, ~45°=slanted-down.
        self.approach_pitches: tuple[float, ...] = (
            0.0, math.pi / 6.0, math.pi / 4.0, math.pi / 3.0,
        )
        self.last_sweep: SweepDiagnostics | None = None

    # -- scoring / orientation helpers -------------------------------------

    @staticmethod
    def orient_normal_toward_robot(
        normal: torch.Tensor, part_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Orient an unsigned surface normal from the part toward the robot."""
        to_robot = -part_pos.clone()
        to_robot[:, 2] = 0.0
        flip = (normal * to_robot).sum(dim=-1, keepdim=True) < 0.0
        return torch.where(flip, -normal, normal)

    @staticmethod
    def candidate_score(
        pos_err: torch.Tensor,
        approach_dir: torch.Tensor,
        surface_normal: torch.Tensor | None,
    ) -> torch.Tensor:
        """Rank reachable IK solutions by surface-normal alignment, then error."""
        if surface_normal is None:
            return pos_err
        alignment = (approach_dir * surface_normal).sum(dim=-1)
        return (1.0 - alignment) + pos_err

    def _generate_candidates(
        self,
        part_pos: torch.Tensor,                     # (N, 3)
        surface_normal: torch.Tensor | None,        # (N, 3) or None
        approach_dirs: list[torch.Tensor] | None,   # provider dirs (N,3)
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Generate candidate (pre_grasp_pos, hand_quat, approach_dir) triples."""
        N = part_pos.shape[0]
        d = self.params["pre_grasp_height"]
        device, dtype = part_pos.device, part_pos.dtype
        cands: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        # 1. Palm-down from above (always include as fallback).
        pos_above = part_pos.clone()
        pos_above[:, 2] += d
        quat_down = torch.zeros(N, 4, device=device, dtype=dtype)
        quat_down[:, 1] = 1.0  # [0, 1, 0, 0] = 180° around X → palm -Z
        dir_down = torch.zeros(N, 3, device=device, dtype=dtype)
        dir_down[:, 2] = 1.0  # approach from above (+Z direction)
        cands.append((pos_above, quat_down, dir_down))

        if approach_dirs is None:
            if surface_normal is None:
                return cands
            nx = surface_normal[:, 0]
            ny = surface_normal[:, 1]
            h = torch.sqrt(nx * nx + ny * ny + 1e-12)
            nxn = nx / h
            nyn = ny / h
            approach_dirs = [
                torch.stack([nxn, nyn, torch.zeros_like(nxn)], dim=-1),
                torch.stack([-nxn, -nyn, torch.zeros_like(nxn)], dim=-1),
                torch.stack([-nyn, nxn, torch.zeros_like(nxn)], dim=-1),
                torch.stack([nyn, -nxn, torch.zeros_like(nxn)], dim=-1),
            ]

        for approach_dir in approach_dirs:
            # For each horizontal direction, also emit slanted candidates
            # pitched downward by each entry in ``approach_pitches``:
            # 0° approaches from the side, 45° from above-side, etc.
            for pitch in self.approach_pitches:
                if pitch == 0.0:
                    tilted = approach_dir
                else:
                    # Rotate dir toward +Z around the horizontal axis
                    # perpendicular to it (right-hand rule) so the
                    # pre-grasp point rises above the object.
                    z = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
                    axis = torch.cross(
                        approach_dir, z.expand_as(approach_dir), dim=-1,
                    )
                    axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    c, s = math.cos(pitch), math.sin(pitch)
                    tilted = (
                        approach_dir * c
                        + torch.cross(axis, approach_dir, dim=-1) * s
                        + axis * (axis * approach_dir).sum(-1, keepdim=True) * (1.0 - c)
                    )
                    tilted = tilted / tilted.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                pos = part_pos + tilted * d
                quat = quat_z_to_vec_batch(-tilted)
                cands.append((pos, quat, tilted))
        return cands

    # -- the sweep ---------------------------------------------------------

    def plan(
        self,
        env_or_solver,
        target: GraspTargetProvider | None = None,
        *,
        init_qpos: torch.Tensor | None = None,
    ) -> IKPlan:
        """Compute IK targets for approach, descend, and terminal phases.

        Args:
            env_or_solver: either the pick env (satisfies the provider
                protocol and owns ``.solver``) or a :class:`GraspSolver`.
            target: provider; required when ``env_or_solver`` is a solver,
                otherwise defaults to ``env_or_solver`` itself or the
                ``target`` passed to ``__init__``.
            init_qpos: optional initial configuration for the approach IK.
        """
        if isinstance(env_or_solver, GraspSolver):
            solver = env_or_solver
            tgt = target if target is not None else self._target
            if tgt is None:
                raise ValueError("target provider required when passing a GraspSolver")
        else:
            env = env_or_solver
            solver = env.solver
            tgt = target if target is not None else (self._target or env)

        N = tgt.n_envs
        device = gs.device

        obj_pos = tgt.obj_pos()
        if obj_pos.ndim == 1:
            obj_pos = obj_pos.unsqueeze(0)

        # ------------------------------------------------------------------
        # Yaw sweep: try multiple object orientations.
        # ------------------------------------------------------------------
        yaws = tgt.yaw_candidates(obj_pos)

        best_score = torch.full((N,), float("inf"), device=device)
        best_err = torch.full((N,), float("inf"), device=device)
        best_qpos = torch.zeros(N, solver._hsr.n_qs, device=device)
        best_hand_quat = torch.zeros(N, 4, device=device)
        best_approach_dir = torch.zeros(N, 3, device=device)
        best_yaw = torch.zeros(N, device=device)
        best_part_pos = torch.zeros(N, 3, device=device)

        # Diagnostics: record every candidate for visualization.
        diag_pos: list[torch.Tensor] = []
        diag_quat: list[torch.Tensor] = []
        diag_dirs: list[torch.Tensor] = []
        diag_scores: list[torch.Tensor] = []
        diag_reach: list[torch.Tensor] = []
        diag_feas: list[torch.Tensor] = []
        diag_qpos: list[torch.Tensor] = []

        base_qs = solver._hsr._ensure_base_qs_idx()

        for yaw in yaws:
            obj_quat = yaw_to_quat_wxyz_batch(yaw)

            part_pos = tgt.part_pos(obj_pos, obj_quat)
            surface_normal = tgt.surface_normal(part_pos, obj_quat)
            dirs_hook = getattr(tgt, "approach_dirs", None)
            dirs = (
                dirs_hook(part_pos, surface_normal)
                if callable(dirs_hook) else None
            )

            # Generate approach candidates for this yaw. Nested-handle
            # metadata fixes wrist roll so jaws close across the thin axis.
            candidates = self._generate_candidates(part_pos, surface_normal, dirs)
            jaw = tgt.jaw_axis(obj_quat)
            if jaw is not None:
                candidates = [
                    (
                        pos,
                        quat if direction[:, 2].abs().max() > 0.5 else
                        quat_from_palm_and_jaw(-direction, jaw),
                        direction,
                    )
                    for pos, quat, direction in candidates
                ]

            for pos, quat, approach_dir in candidates:
                # Candidates radiate from the object center along each
                # approach direction; whole-body IK places the base —
                # no manual placement or drift constraint.
                qpos, pos_err = solver.ik_scan(pos, quat, init_qpos=init_qpos)

                # Collision check is diagnostic only (feasible/reachable);
                # it never gates selection.
                coll_ok = solver.collision_free(qpos)

                score = self.candidate_score(pos_err, approach_dir, surface_normal)
                reachable = (
                    torch.isfinite(pos_err)
                    & (pos_err < POS_ERR_THRESHOLD)
                    & coll_ok
                )
                improve = reachable & (score < best_score)
                best_score = torch.where(improve, score, best_score)
                best_err = torch.where(improve, pos_err, best_err)
                best_qpos = torch.where(
                    improve.unsqueeze(-1), qpos, best_qpos,
                )
                best_hand_quat = torch.where(
                    improve.unsqueeze(-1), quat, best_hand_quat,
                )
                best_approach_dir = torch.where(
                    improve.unsqueeze(-1), approach_dir, best_approach_dir,
                )
                best_yaw = torch.where(improve, yaw, best_yaw)
                best_part_pos = torch.where(
                    improve.unsqueeze(-1), part_pos, best_part_pos,
                )

                diag_pos.append(pos)
                diag_quat.append(quat)
                diag_dirs.append(approach_dir)
                diag_scores.append(score)
                diag_reach.append(reachable)
                diag_feas.append(coll_ok)
                diag_qpos.append(qpos)

        ok_approach = torch.isfinite(best_err) & (best_err < POS_ERR_THRESHOLD)

        if diag_pos:
            flat_best = torch.argmin(
                torch.stack(diag_scores), dim=0,
            )
            self.last_sweep = SweepDiagnostics(
                candidate_pos=torch.stack(diag_pos),
                candidate_quat=torch.stack(diag_quat),
                candidate_dirs=torch.stack(diag_dirs),
                candidate_scores=torch.stack(diag_scores),
                candidate_reachable=torch.stack(diag_reach),
                candidate_feasible=torch.stack(diag_feas),
                candidate_qpos=torch.stack(diag_qpos),
                part_pos=best_part_pos,
                best_index=flat_best,
                yaws=yaws,
            )
        else:
            self.last_sweep = None

        # ------------------------------------------------------------------
        # Refine arm joints: analytic IK + calibration first (handles
        # orientation via base yaw); fall back to position-only Jacobian IK
        # if calibration moves the base.
        # ------------------------------------------------------------------
        pre_grasp_dist = self.params["pre_grasp_height"]
        target_ee = best_part_pos + best_approach_dir * pre_grasp_dist
        quat_cal = best_hand_quat.to(device=device, dtype=gs.tc_float)
        if quat_cal.ndim == 1:
            quat_cal = quat_cal.unsqueeze(0).expand(N, -1).contiguous()

        cal_qpos = best_qpos.clone()
        cal_ok = torch.zeros(N, dtype=torch.bool, device=device)
        ik_offset = torch.zeros(N, 3, device=device)
        for _calib in range(3):
            solver._hsr.set_qpos(cal_qpos, envs_idx=solver._envs_all)
            solver._scene.step()
            actual_ee = solver._ee_link.get_pos(envs_idx=solver._envs_all)
            if actual_ee.ndim == 1:
                actual_ee = actual_ee.unsqueeze(0)
            new_offset = target_ee - actual_ee
            if (new_offset - ik_offset).norm(dim=-1).max().item() < 0.005:
                ik_offset = new_offset
                break
            ik_offset = new_offset
            adjusted_target = target_ee + ik_offset
            try_qpos, error = solver._hsr.inverse_kinematics(
                link=solver._ee_link,
                pos=adjusted_target.contiguous(),
                quat=quat_cal,
                init_qpos=cal_qpos.contiguous(),
                max_samples=IK_SCAN_MAX_SAMPLES,
                max_solver_iters=IK_SCAN_MAX_SOLVER_ITERS,
                max_step_size=0.1,
                respect_joint_limit=True,
                envs_idx=solver._envs_all,
                return_error=True,
            )
            error = GraspSolver._ensure_batched(error)
            try_qpos = GraspSolver._ensure_batched(try_qpos)
            # Reject if the optimizer moved the base.
            pre_base_xy = cal_qpos[:, [base_qs[0], base_qs[1]]]
            drift_ok = solver.base_drift_ok(try_qpos, pre_base_xy)
            cal_qpos = torch.where(drift_ok.unsqueeze(-1), try_qpos, cal_qpos)
            cal_err = error[:, :3].norm(dim=1)
            cal_ok = torch.isfinite(cal_err) & (cal_err < POS_ERR_THRESHOLD) & drift_ok
            if cal_ok.all():
                break

        if cal_ok.all():
            best_qpos = cal_qpos
            ok_approach = cal_ok
        else:
            # Attempt 2: position-only Jacobian IK (base fixed from scan).
            best_qpos, ok_approach = solver.jacobian(
                target_ee, init_qpos=best_qpos,
                hand_quat=best_hand_quat, max_iters=100,
            )

        # ------------------------------------------------------------------
        # Set the winning object pose and settle.
        # ------------------------------------------------------------------
        win_quat = yaw_to_quat_wxyz_batch(best_yaw)
        tgt.set_obj_pose(win_quat)

        # Recompute part position from stored local geometry after settle.
        obj_pos = tgt.obj_pos()
        if obj_pos.ndim == 1:
            obj_pos = obj_pos.unsqueeze(0)
        obj_quat = tgt.obj_quat()
        if obj_quat.ndim == 1:
            obj_quat = obj_quat.unsqueeze(0)
        part_pos = tgt.part_pos(obj_pos, obj_quat)

        approach_arm, approach_base = solver.qpos_to_arm_and_base(best_qpos)

        # ------------------------------------------------------------------
        # Phase 2: descend — approach the part along the best approach dir.
        # ------------------------------------------------------------------
        pre_grasp_dist = self.params["pre_grasp_height"]
        grasp_offset = self.params["grasp_offset_z"]
        offsets = [grasp_offset, pre_grasp_dist * 0.5, pre_grasp_dist]
        qpos_descend = best_qpos  # fallback if no offset passes threshold
        ok_descend = torch.zeros(N, dtype=torch.bool, device=device)
        descend_link_pos = part_pos + best_approach_dir * pre_grasp_dist  # fallback
        for off in offsets:
            grasp = part_pos + best_approach_dir * off
            qpos_try, ok_try = solver.calibrated(
                grasp, init_qpos=best_qpos, hand_quat=best_hand_quat,
                max_iters=3, respect_joint_limit=False,
            )
            if not ok_try.all():
                qpos_try, ok_try = solver.jacobian(
                    grasp, init_qpos=best_qpos, hand_quat=best_hand_quat,
                    max_iters=100,
                )
            if ok_try.all():
                qpos_descend = qpos_try
                ok_descend = ok_try
                descend_link_pos = grasp.clone()
                break
            if ok_try.any() and not ok_descend.any():
                qpos_descend = qpos_try
                descend_link_pos = grasp.clone()
        descend_arm, descend_base = solver.qpos_to_arm_and_base(qpos_descend)
        # Keep the base at the approach position during descend.
        descend_base = approach_base.clone()

        # ------------------------------------------------------------------
        # Terminal phase: lift or articulation, per the provider.
        # ------------------------------------------------------------------
        terminal_pos = tgt.terminal_target(part_pos, descend_link_pos)
        qpos_terminal, ok_terminal = solver.calibrated(
            terminal_pos, init_qpos=qpos_descend, hand_quat=best_hand_quat,
            max_iters=3, respect_joint_limit=False,
        )
        if not ok_terminal.all():
            qpos_terminal, ok_terminal = solver.jacobian(
                terminal_pos, init_qpos=qpos_descend, hand_quat=best_hand_quat,
                max_iters=100,
            )
        terminal_arm, terminal_base = solver.qpos_to_arm_and_base(qpos_terminal)
        terminal_base = approach_base.clone()

        ik_success = torch.stack([ok_approach, ok_descend, ok_terminal], dim=1)

        return IKPlan(
            approach_arm=approach_arm,
            approach_base=approach_base,
            descend_arm=descend_arm,
            descend_base=descend_base,
            lift_arm=terminal_arm,
            lift_base=terminal_base,
            obj_pos=obj_pos,
            obj_init_z=obj_pos[:, 2].clone(),
            ik_success=ik_success,
            pre_grasp_height=self.params["pre_grasp_height"],
            grasp_offset_z=self.params["grasp_offset_z"],
            gripper_effort=self.params["gripper_effort"],
            grasp_hold_steps=int(self.params["grasp_hold_steps"]),
        )


class YcbSweepPlanner(SweepGraspPlanner):
    """Yaw + approach-direction sweep planner for simple (non-articulated) objects.

    For objects with no joint geometry the grasp target is the object
    center.  The sweep still matters: object yaw affects gripper alignment
    with the object's long axis, and the approach-direction sweep picks the
    most reachable pre-grasp pose.
    """

    def __init__(self, object_name: str = "ycb_011_banana") -> None:
        super().__init__()
        self.object_name = object_name
        # The banana's successful floor-grasp band is below its body origin:
        # grasp_offset_z pushes the pre-grasp target down toward the table.
        self.params = get_object_params(object_name)

    @staticmethod
    def _object_yaw_penalty(object_name: str, yaw: torch.Tensor) -> torch.Tensor:
        """Return a dimensionless grasp-alignment penalty for object yaw."""
        if object_name == "ycb_011_banana":
            # The banana mesh's long axis is local Y.  Palm-down fingers close
            # across world Y, so rotate the long axis onto world X.
            return torch.abs(torch.cos(yaw))
        return torch.zeros_like(yaw)

    def plan(
        self,
        env_or_solver,
        target: GraspTargetProvider | None = None,
        *,
        init_qpos: torch.Tensor | None = None,
    ) -> IKPlan:
        if isinstance(env_or_solver, GraspSolver):
            solver = env_or_solver
            tgt = target if target is not None else self._target
            if tgt is None:
                raise ValueError("target provider required when passing a GraspSolver")
        else:
            env = env_or_solver
            solver = env.solver
            tgt = target if target is not None else (self._target or env)

        N = tgt.n_envs
        device = gs.device

        obj_pos = tgt.obj_pos()
        if obj_pos.ndim == 1:
            obj_pos = obj_pos.unsqueeze(0)

        # ------------------------------------------------------------------
        # Yaw sweep.  part_pos = obj_pos regardless of yaw, but approach
        # direction candidates differ and yaw affects gripper alignment with
        # the object's long axis.
        # ------------------------------------------------------------------
        n_yaws = 7
        yaw_spacing = math.pi / 6.0  # 30° spacing → ±90°
        base_yaw = torch.zeros(N, device=device, dtype=gs.tc_float)
        yaws = []
        for i in range(n_yaws):
            offset = torch.full(
                (N,), (i - n_yaws // 2) * yaw_spacing,
                device=device, dtype=gs.tc_float,
            )
            yaws.append(base_yaw + offset)

        best_score = torch.full((N,), float("inf"), device=device)
        best_err = torch.full((N,), float("inf"), device=device)
        best_qpos = torch.zeros(N, solver._hsr.n_qs, device=device)
        best_hand_quat = torch.zeros(N, 4, device=device)
        best_approach_dir = torch.zeros(N, 3, device=device)
        best_yaw = torch.zeros(N, device=device)

        diag_pos: list[torch.Tensor] = []
        diag_quat: list[torch.Tensor] = []
        diag_dirs: list[torch.Tensor] = []
        diag_scores: list[torch.Tensor] = []
        diag_reach: list[torch.Tensor] = []
        diag_feas: list[torch.Tensor] = []
        diag_qpos: list[torch.Tensor] = []

        for yaw in yaws:
            part_pos = obj_pos.clone()

            # No joint geometry → use approach direction toward the robot
            # (from object to robot base; base at origin).
            to_robot = -obj_pos.clone()
            to_robot[:, 2] = 0.0
            norm = to_robot.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            surface_normal = to_robot / norm  # (N, 3) horizontal

            dirs_hook = getattr(tgt, "approach_dirs", None)
            dirs = (
                dirs_hook(part_pos, surface_normal)
                if callable(dirs_hook) else None
            )
            candidates = self._generate_candidates(part_pos, surface_normal, dirs)

            for pos, quat, approach_dir in candidates:
                # Candidates radiate from the object centre along each
                # approach direction; whole-body IK places the base —
                # no manual placement or drift constraint.
                qpos, error = solver._hsr.inverse_kinematics(
                    link=solver._ee_link,
                    pos=pos.contiguous(),
                    quat=quat.contiguous(),
                    init_qpos=(init_qpos.contiguous() if init_qpos is not None else None),
                    max_samples=IK_SCAN_MAX_SAMPLES,
                    max_solver_iters=IK_SCAN_MAX_SOLVER_ITERS,
                    max_step_size=0.7,
                    respect_joint_limit=False,
                    envs_idx=solver._envs_all,
                    return_error=True,
                )
                pos_err = _pos_error_norm(error)
                qpos = GraspSolver._ensure_batched(qpos)

                # Collision check is diagnostic only (feasible/reachable);
                # it never gates selection.
                coll_ok = solver.collision_free(qpos)

                score = pos_err + 1e-4 * self._object_yaw_penalty(
                    self.object_name, yaw,
                )
                reachable = (
                    torch.isfinite(pos_err)
                    & (pos_err < POS_ERR_THRESHOLD)
                    & coll_ok
                )
                # Selection unconditional on pos_err/collision (parity
                # with the original sweep: descend recovers via
                # whole-body IK, and a plan must exist even if every
                # candidate fails a gate).  Gates are diagnostics only.
                improve = score < best_score
                best_score = torch.where(improve, score, best_score)
                best_err = torch.where(improve, pos_err, best_err)
                best_qpos = torch.where(
                    improve.unsqueeze(-1), qpos, best_qpos,
                )
                best_hand_quat = torch.where(
                    improve.unsqueeze(-1), quat, best_hand_quat,
                )
                best_approach_dir = torch.where(
                    improve.unsqueeze(-1), approach_dir, best_approach_dir,
                )
                best_yaw = torch.where(improve, yaw, best_yaw)

                diag_pos.append(pos)
                diag_quat.append(quat)
                diag_dirs.append(approach_dir)
                diag_scores.append(score)
                diag_reach.append(reachable)
                diag_feas.append(coll_ok)
                diag_qpos.append(qpos.clone())

        ok_approach = torch.isfinite(best_err) & (best_err < POS_ERR_THRESHOLD)

        if diag_pos:
            self.last_sweep = SweepDiagnostics(
                candidate_pos=torch.stack(diag_pos),
                candidate_quat=torch.stack(diag_quat),
                candidate_dirs=torch.stack(diag_dirs),
                candidate_scores=torch.stack(diag_scores),
                candidate_reachable=torch.stack(diag_reach),
                candidate_feasible=torch.stack(diag_feas),
                candidate_qpos=torch.stack(diag_qpos),
                part_pos=obj_pos.clone(),
                best_index=torch.argmin(torch.stack(diag_scores), dim=0),
                yaws=yaws,
            )
        else:
            self.last_sweep = None

        # ------------------------------------------------------------------
        # Set the winning object yaw.
        # ------------------------------------------------------------------
        win_quat = yaw_to_quat_wxyz_batch(best_yaw)
        tgt.set_obj_pose(win_quat)

        obj_pos = tgt.obj_pos()
        if obj_pos.ndim == 1:
            obj_pos = obj_pos.unsqueeze(0)

        approach_arm, approach_base = solver.qpos_to_arm_and_base(best_qpos)

        # ------------------------------------------------------------------
        # Phase 2: descend — whole-body IK (base + arm move together).  The
        # base drives forward while the arm descends, letting the gripper
        # reach low objects on the ground.  The IK solution's base position
        # is used (NOT held at approach).
        # ------------------------------------------------------------------
        grasp = obj_pos.clone()
        grasp[:, 2] += self.params["grasp_offset_z"]
        cur_qpos = solver._hsr.get_qpos(envs_idx=solver._envs_all)
        if cur_qpos.ndim == 1:
            cur_qpos = cur_qpos.unsqueeze(0)
        qpos_descend, ok_descend = solver.solve(
            grasp, init_qpos=cur_qpos, hand_quat=best_hand_quat,
        )
        descend_arm, descend_base = solver.qpos_to_arm_and_base(qpos_descend)

        # Phase 3: grasp — no IK needed (gripper closes at current position).

        # ------------------------------------------------------------------
        # Phase 4: lift — whole-body IK (base + arm move together).
        # ------------------------------------------------------------------
        lift = obj_pos.clone()
        lift[:, 2] = LIFT_HEIGHT
        cur_qpos2 = solver._hsr.get_qpos(envs_idx=solver._envs_all)
        if cur_qpos2.ndim == 1:
            cur_qpos2 = cur_qpos2.unsqueeze(0)
        qpos_lift, ok_lift = solver.solve(
            lift, init_qpos=cur_qpos2, hand_quat=best_hand_quat,
        )
        lift_arm, lift_base = solver.qpos_to_arm_and_base(qpos_lift)

        ik_success = torch.stack([ok_approach, ok_descend, ok_lift], dim=1)

        return IKPlan(
            approach_arm=approach_arm,
            approach_base=approach_base,
            descend_arm=descend_arm,
            descend_base=descend_base,
            lift_arm=lift_arm,
            lift_base=lift_base,
            obj_pos=obj_pos,
            obj_init_z=obj_pos[:, 2].clone(),
            ik_success=ik_success,
            pre_grasp_height=self.params["pre_grasp_height"],
            grasp_offset_z=self.params["grasp_offset_z"],
            gripper_effort=self.params["gripper_effort"],
            grasp_hold_steps=int(self.params["grasp_hold_steps"]),
        )

