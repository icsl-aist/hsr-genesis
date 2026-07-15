"""Curriculum manager for IK-to-policy blend training.

3 stages: pure IK → 70/30 blend → 30/70 blend.
Advances globally based on consecutive eval success thresholds.
"""
from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class CurriculumStage:
    ik_weight: float
    policy_weight: float
    advance_threshold: float  # mean success rate needed to advance (0 = always advance)
    consecutive_needed: int   # consecutive evals above threshold
    warmup_steps: int         # fixed warmup before this stage can advance (0 = no warmup)


# Stage 0: pure IK warmup (policy observes but doesn't act, 1 eval round)
# Stage 1: 70% IK + 30% policy
# Stage 2: 30% IK + 70% policy (terminal)
STAGES = [
    CurriculumStage(ik_weight=1.0, policy_weight=0.0, advance_threshold=0.0, consecutive_needed=1, warmup_steps=5_000),
    CurriculumStage(ik_weight=0.7, policy_weight=0.3, advance_threshold=0.50, consecutive_needed=2, warmup_steps=0),
    CurriculumStage(ik_weight=0.3, policy_weight=0.7, advance_threshold=0.0, consecutive_needed=999, warmup_steps=0),
]

EVAL_INTERVAL = 5_000  # sim steps between eval rounds


class CurriculumManager:
    """Tracks global curriculum stage across all envs."""

    def __init__(self) -> None:
        self.stage_idx = 0
        self.consecutive_successes = 0
        self.total_steps = 0
        self.eval_history: list[dict] = []

    @property
    def stage(self) -> int:
        return self.stage_idx

    @property
    def policy_weight(self) -> float:
        return STAGES[self.stage_idx].policy_weight

    @property
    def ik_weight(self) -> float:
        return STAGES[self.stage_idx].ik_weight

    def should_eval(self, step: int) -> bool:
        """Check if an eval round should happen at this step."""
        return step > 0 and step % EVAL_INTERVAL == 0

    def update(self, success_rate: float, step: int) -> bool:
        """Update curriculum after an eval round. Returns True if stage advanced."""
        stage = STAGES[self.stage_idx]
        advanced = False

        # Check warmup period
        if step < stage.warmup_steps:
            self.eval_history.append({
                "step": step,
                "stage": self.stage_idx,
                "success_rate": success_rate,
                "advanced": False,
            })
            return False

        # Check success threshold (0.0 means always advance)
        if stage.advance_threshold == 0.0 or success_rate >= stage.advance_threshold:
            self.consecutive_successes += 1
        else:
            self.consecutive_successes = 0

        if self.consecutive_successes >= stage.consecutive_needed and self.stage_idx < len(STAGES) - 1:
            self.stage_idx += 1
            self.consecutive_successes = 0
            advanced = True
            print(f"[curriculum] Advanced to stage {self.stage_idx} "
                  f"(policy_weight={self.policy_weight:.1f})")

        self.eval_history.append({
            "step": step,
            "stage": self.stage_idx,
            "success_rate": success_rate,
            "advanced": advanced,
        })
        return advanced

    def state_dict(self) -> dict:
        return {
            "stage_idx": self.stage_idx,
            "consecutive_successes": self.consecutive_successes,
            "total_steps": self.total_steps,
            "eval_history": self.eval_history,
        }

    def load_state_dict(self, state: dict) -> None:
        self.stage_idx = state.get("stage_idx", 0)
        self.consecutive_successes = state.get("consecutive_successes", 0)
        self.total_steps = state.get("total_steps", 0)
        self.eval_history = state.get("eval_history", [])

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.state_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CurriculumManager":
        mgr = cls()
        with open(path) as f:
            mgr.load_state_dict(json.load(f))
        return mgr
