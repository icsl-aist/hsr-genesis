"""IK planner: precomputes IK targets for all pick phases at episode reset.

Computes 3 IK solutions per env (approach, descend, lift) once per episode.
The RL policy applies residual corrections on top of these fixed targets.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

import genesis as gs


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
    """Plans IK targets for all 4 phases of the pick pipeline."""

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
            env: HSRPickEnv instance with objects already settled.
            init_qpos: Optional initial configuration for the approach IK.
                When provided (e.g. from retarget), the IK solver starts
                from the robot's current pose instead of the default.

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

        # Phase 4: lift — raise object (base held)
        lift = obj_pos.clone()
        lift[:, 2] = LIFT_HEIGHT
        cur_qpos2 = env.hsr.get_qpos(envs_idx=env.envs_all)
        if cur_qpos2.ndim == 1:
            cur_qpos2 = cur_qpos2.unsqueeze(0)
        qpos_lift, ok_lift = env._ik(lift, init_qpos=cur_qpos2)
        lift_arm, _lift_base = env._qpos_to_arm_and_base(qpos_lift)
        lift_base = approach_base.clone()  # base held during lift

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
