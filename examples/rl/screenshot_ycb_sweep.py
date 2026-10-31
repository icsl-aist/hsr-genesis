"""Screenshot the yaw-sweep IK pipeline on a YCB banana.

Sweeps object yaw and hand approach direction (palm-down + 4 horizontal),
picks the best reachable pre-grasp pose per env, then runs the standard
pick pipeline (approach, descend, grasp, lift).  Screenshots each phase
and reports whether the object was lifted.

Usage:
    PYTHONPATH=src .venv/bin/python examples/rl/screenshot_ycb_sweep.py \\
        --output-dir screenshots_banana_sweep
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import genesis as gs

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ycb_pick_ik_parallel import LIFT_THRESHOLD
from hsr_pick_rl_env import HSRPickRLEnv, BatchedGenesisVecEnv, ACTION_DIM
from hsr_genesis.grasp_planning import IKPlanner, YcbSweepPlanner
from curriculum import CurriculumManager

# ---------------------------------------------------------------------------
# Screenshot driver
# ---------------------------------------------------------------------------

def _ensure_genesis_initialized():
    if not getattr(gs, "_initialized", False):
        gs.init(backend=gs.cuda, precision="32", logging_level="info")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screenshot yaw-sweep IK pipeline on YCB banana",
    )
    parser.add_argument("--object", default="ycb_011_banana",
                        help="YCB object name. Default: ycb_011_banana")
    parser.add_argument("--output-dir", default="screenshots_banana_sweep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--obj-radius-min", type=float, default=0.35)
    parser.add_argument("--obj-radius-max", type=float, default=0.45)
    parser.add_argument("--obj-z", type=float, default=0.05,
                        help="Object spawn height (m). Raise for flat objects.")
    args = parser.parse_args()


    _ensure_genesis_initialized()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # The force-controlled fingers need the standard 300-step grasp phase to
    # close fully around the low-profile banana before lift begins.
    IKPlanner.configure_phase_steps(
        approach=180, descend=250, grasp=300, lift=155,
    )
    print(f"[screenshot] Phase steps: approach={IKPlanner.approach_steps}, "
          f"descend={IKPlanner.descend_steps}, grasp={IKPlanner.grasp_steps}, "
          f"lift={IKPlanner.lift_steps}, total={IKPlanner.max_steps()}")

    # Build env with camera and lighting.
    env = HSRPickRLEnv(
        n_envs=1,
        object_name=args.object,
        seed=args.seed,
        settle_steps=30,
        curriculum=CurriculumManager(),
        use_ik_guidance=True,
        vis_options_overrides={
            "split_envs": False,
            "lights": [gs.options.vis.DirectionalLight(
                dir=(0.5, 0.5, -1), color=(1, 1, 1), intensity=3.0,
            )],
        },
        camera_config={
            "res": (1920, 1080),
            "pos": (2, -2, 1.5),
            "lookat": (0, 0, 0.5),
            "fov": 60,
            "far": 200.0,
            "debug": True,
        },
        obj_radius_range=(args.obj_radius_min, args.obj_radius_max),
        gripper_effort_override=5.0,
        terminate_on_success=False,
    )

    # Replace the default IK planner with the sweep planner.
    env._planner = YcbSweepPlanner(object_name=args.object)

    vec_env = BatchedGenesisVecEnv(env)
    camera = env.camera
    assert camera is not None, "Camera not created"

    # Get env0 offset for camera framing.
    env0_offset = np.asarray(env._pick_env.scene.envs_offset[0])
    print(f"[screenshot] env0 offset: {env0_offset.tolist()}")

    # Reset to start episode.
    obs = vec_env.reset()
    pick_env = env._pick_env
    hsr = pick_env.hsr

    # Point head down toward the object.
    head_pan_target = torch.tensor([0.5], device=gs.device, dtype=gs.tc_float)
    head_tilt_target = torch.tensor([-0.8], device=gs.device, dtype=gs.tc_float)
    hsr.control_dofs_position(
        head_pan_target, dofs_idx_local=[13], envs_idx=pick_env.envs_all,
    )
    hsr.control_dofs_position(
        head_tilt_target, dofs_idx_local=[17], envs_idx=pick_env.envs_all,
    )

    # Print IK plan info.
    plan = env._plan
    if plan is not None:
        ik_ok = plan.ik_success.cpu().numpy()
        print(f"[screenshot] IK success: approach={ik_ok[0,0]}, "
              f"descend={ik_ok[0,1]}, lift={ik_ok[0,2]}")
        print(f"[screenshot] Object pos: {plan.obj_pos.cpu().numpy()[0]}")
    else:
        print("[screenshot] No IK plan")

    # Camera positions: look at env0's robot from a 3/4 angle.
    cam_pos = env0_offset + np.array([1.8, -1.5, 1.3])
    cam_lookat = env0_offset + np.array([0.0, 0.0, 0.5])

    zero_action = np.zeros((1, ACTION_DIM), dtype=np.float32)

    # Phase boundaries (step counts).
    approach_end = IKPlanner.approach_steps
    descend_end = approach_end + IKPlanner.descend_steps
    grasp_end = descend_end + IKPlanner.grasp_steps
    lift_end = grasp_end + IKPlanner.lift_steps

    print(f"[screenshot] Phase boundaries: approach={approach_end}, "
          f"descend={descend_end}, grasp={grasp_end}, lift={lift_end}")

    obj_init_z = plan.obj_init_z.cpu().numpy()[0] if plan is not None else 0.05

    from hsr_genesis.plan_visualization import (
        run_phases_with_captures, draw_sweep_debug, draw_sweep_ghosts,
    )

    # Debug-draw swept candidates + chosen grasp in the viewer.
    if env._planner is not None and getattr(env._planner, "last_sweep", None):
        draw_sweep_debug(env._pick_env.scene, env._planner.last_sweep, env_idx=0)
        draw_sweep_ghosts(env._pick_env.scene, env._planner.last_sweep,
                          env._pick_env.hsr, env_idx=0, max_ghosts=8)

    def _log(step_no, name):
        ee = pick_env.ee_link.get_pos(envs_idx=pick_env.envs_all)
        obj = pick_env._obj_pos()
        if ee.ndim == 1: ee = ee.unsqueeze(0)
        if obj.ndim == 1: obj = obj.unsqueeze(0)
        if name == "lift":
            obj_z = obj[0, 2].item()
            lifted = obj_z - obj_init_z
            print(f"  [step {step_no}] Lift: obj_z={obj_z:.4f}, "
                  f"init_z={obj_init_z:.4f}, lifted={lifted:.4f}")
            print(f"  [step {step_no}] Success (lifted > {LIFT_THRESHOLD}): "
                  f"{lifted > LIFT_THRESHOLD}")
        else:
            print(f"  [step {step_no}] {name}: EE={ee.cpu().tolist()[0]}, "
                  f"obj={obj.cpu().tolist()[0]}, "
                  f"dist={(ee-obj).norm().item():.4f}")

    captured = run_phases_with_captures(
        vec_env, camera, output_dir,
        phase_boundaries={
            "approach": approach_end, "descend": descend_end,
            "grasp": grasp_end, "lift": lift_end,
        },
        zero_action=zero_action,
        cam_pos=cam_pos, cam_lookat=cam_lookat,
        head_targets=(head_pan_target, head_tilt_target),
        env=env,
        extra_steps=10,
        per_phase_fn=_log,
    )
    screenshots_taken = captured

    # Final summary.
    obj = pick_env._obj_pos()
    if obj.ndim == 1: obj = obj.unsqueeze(0)
    final_z = obj[0, 2].item()
    total_lift = final_z - obj_init_z
    print(f"\n[screenshot] Final object z: {final_z:.4f}")
    print(f"[screenshot] Total lift: {total_lift:.4f}m")
    print(f"[screenshot] Success: {total_lift > LIFT_THRESHOLD}")
    print(f"\n[screenshot] Done! {len(screenshots_taken)}/4 screenshots saved to {output_dir}/")

    vec_env.close()


if __name__ == "__main__":
    main()
