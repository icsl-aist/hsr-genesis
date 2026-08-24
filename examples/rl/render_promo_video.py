"""Render a promotion video: HSR fleet reveal crane shot.

Loads a trained PPO policy, builds a 1024-env scene, and records a
continuous crane shot. The camera starts tight on a single robot,
holds until it completes a successful grasp, then cranes up to reveal
the full 32x32 grid of 1024 robots.

Usage:
    # Full render (1024 envs, 30s video)
    PYTHONPATH=src .venv/bin/python examples/rl/render_promo_video.py

    # Quick smoke test (16 envs, 1s video)
    PYTHONPATH=src .venv/bin/python examples/rl/render_promo_video.py \\
        --envs 16 --frames 30 --output results/promo_smoke.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import genesis as gs

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stable_baselines3 import PPO

from hsr_pick_rl_env import HSRPickRLEnv, BatchedGenesisVecEnv
from curriculum import CurriculumManager
from camera_crane import CraneKeyframe, crane_path
from ik_planner import IKPlanner


def _ensure_genesis_initialized() -> None:
    if getattr(gs, "_initialized", False):
        return
    try:
        gs.init(backend=gs.gpu)
    except RuntimeError as exc:
        print(f"[Genesis] GPU unavailable ({exc}); falling back to CPU.")
        gs.init(backend=gs.cpu)


def _build_crane_keyframes(env0_offset: np.ndarray) -> list[CraneKeyframe]:
    """Build the crane path keyframes in env0-relative coordinates.

    For 1024 envs at 3m spacing, the grid is centered at world origin.
    env0 is at (-46.5, -46.5, 0). The camera starts tight on env0's
    single robot, then cranes up to reveal the full grid.

    Camera positions are env0-relative: world (wx, wy, wz) -> relative
    (wx - env0_offset[0], wy - env0_offset[1], wz).
    """
    offset_x, offset_y = env0_offset[0], env0_offset[1]

    # Start: tight close-up on single robot (env0) at world (-46.5, -46.5).
    # Camera ~2.5m away, chest height, looking at the arm/grasp area.
    start_pos_world = (-44.0, -48.0, 1.2)
    start_lookat_world = (-46.5, -46.5, 0.8)

    # Mid: rising up, moving toward grid center
    mid_pos_world = (0.0, -5.0, 50.0)
    mid_lookat_world = (0.0, 0.0, 0.0)

    # End: top-down at grid center, full fleet visible
    end_pos_world = (0.0, 0.0, 100.0)
    end_lookat_world = (0.0, 0.0, 0.0)

    def to_rel(world_pos):
        return (world_pos[0] - offset_x, world_pos[1] - offset_y, world_pos[2])

    return [
        CraneKeyframe(pos=to_rel(start_pos_world), lookat=to_rel(start_lookat_world), time=0.0),
        CraneKeyframe(pos=to_rel(mid_pos_world), lookat=to_rel(mid_lookat_world), time=0.5),
        CraneKeyframe(pos=to_rel(end_pos_world), lookat=to_rel(end_lookat_world), time=1.0),
    ]


def render_promo(
    *,
    model_path: str,
    n_envs: int = 1024,
    total_frames: int = 900,
    fps: int = 30,
    output_path: str = "results/promo_video.mp4",
    object_name: str = "ycb_013_apple",
    seed: int = 0,
    settle_steps: int = 30,
    max_trial_frames: int = 600,
    linger_frames: int = 15,
    num_closeup_trials: int = 2,
    approach_steps: int = 60,
    descend_steps: int = 45,
    grasp_steps: int = 150,
    obj_radius_range: tuple[float, float] | None = None,
) -> None:
    """Render the promo video.

    The video has two phases:
    1. Close-up: Camera stays tight on env0's single robot. Runs multiple
       grasp trials with shortened approach/descend phases so the robot
       gets to the grasp quickly. Resets for a new random object position
       between trials.
    2. Crane: Camera cranes from the close-up to a top-down fleet reveal.
       Grasps continue during the crane with auto-resets on done.
    """
    _ensure_genesis_initialized()

    # Shorten approach/descend phases so the robot gets to the grasp faster.
    # Default is approach=180 (3.6s), descend=105 (2.1s). We use 60 (1.2s)
    # and 45 (0.9s) — the arm moves more quickly to the pre-grasp position.
    IKPlanner.configure_phase_steps(approach=approach_steps, descend=descend_steps, grasp=grasp_steps)
    print(f"[promo] Phase steps: approach={IKPlanner.approach_steps}, "
          f"descend={IKPlanner.descend_steps}, grasp={IKPlanner.grasp_steps}, "
          f"lift={IKPlanner.lift_steps}, total={IKPlanner.max_steps()}")

    # Load curriculum at final stage (stage 2, policy_weight=0.7)
    curriculum = CurriculumManager()
    curriculum.stage_idx = 2

    # Load model
    model = PPO.load(model_path, device="auto")
    print(f"[promo] Loaded model: {model_path}")

    # Load run config to determine IK guidance
    config_path = Path(model_path).with_name("run_config.json")
    use_ik_guidance = True
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        use_ik_guidance = bool(config.get("use_ik_guidance", True))
    print(f"[promo] use_ik_guidance={use_ik_guidance}")

    # Build env with camera and vis_options overrides.
    # env_separate_rigid=False: camera sees all envs in one frame at world positions.
    # far=200: needed for high-altitude wide shots (default 20m clips distant robots).
    env = HSRPickRLEnv(
        n_envs=n_envs,
        object_name=object_name,
        seed=seed,
        settle_steps=settle_steps,
        curriculum=curriculum,
        use_ik_guidance=use_ik_guidance,
        vis_options_overrides={
            "env_separate_rigid": False,
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
        },
        obj_radius_range=obj_radius_range,
    )
    vec_env = BatchedGenesisVecEnv(env)
    camera = env.camera
    assert camera is not None, "Camera was not created — check camera_config"

    # Build crane keyframes using the scene's env0 offset
    env0_offset = np.asarray(env._pick_env.scene.envs_offset[0])
    keyframes = _build_crane_keyframes(env0_offset)
    print(f"[promo] env0 offset: {env0_offset.tolist()}")
    print(f"[promo] Crane path: {len(keyframes)} keyframes")

    # Reset env to start the first episode
    obs = vec_env.reset()

    # Point head down toward the object/grasp area for all envs.
    # head_pan_joint (dof 13): pan toward arm side (~0.5 rad)
    # head_tilt_joint (dof 17): tilt down to look at object (~0.8 rad)
    # Applied every frame because step_whole_body_trajectory_batched may
    # override head control targets.
    pick_env = env._pick_env
    hsr = pick_env.hsr
    head_pan_target = 0.5    # rad — pan toward arm side
    head_tilt_target = -0.8  # rad — tilt DOWN toward object (negative = down)
    head_dofs = [13, 17]
    head_targets = torch.tensor(
        [head_pan_target, head_tilt_target],
        device=gs.device, dtype=gs.tc_float,
    )

    def _set_head_pose():
        hsr.control_dofs_position(
            head_targets, dofs_idx_local=head_dofs, envs_idx=pick_env.envs_all,
        )

    def _step_and_render(cam_pos, cam_lookat):
        """Step policy, set head, set camera, render. Returns obs, dones, infos."""
        nonlocal obs
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)
        _set_head_pose()
        camera.set_pose(pos=cam_pos.tolist(), lookat=cam_lookat.tolist())
        camera.render()
        return obs, dones, infos

    # Start recording
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    camera.start_recording()
    print(f"[promo] Recording (max {total_frames} frames at {fps} fps)...")
    print(f"[promo] Phase 1: {num_closeup_trials} close-up grasp trials with random resets")

    t0 = time.time()
    frames_recorded = 0
    closeup_pos, closeup_lookat = crane_path(0.0, keyframes)
    success_count = 0

    # --- Phase 1: Close-up grasp trials ---
    # All trials except the last use fixed close-up camera.
    # The last trial starts at close-up and cranes out while the robot
    # continues grasping — blending into the fleet reveal.
    crane_start_frame = None  # set when crane begins (last trial)

    for trial in range(num_closeup_trials):
        if frames_recorded >= total_frames:
            break
        is_last_trial = (trial == num_closeup_trials - 1)
        trial_success = False
        print(f"[promo]   Trial {trial+1}/{num_closeup_trials}:"
              f"{' (with crane)' if is_last_trial else ''}")

        # For the last trial, crane starts here and runs for the remaining frames
        if is_last_trial:
            crane_start_frame = frames_recorded
            crane_total = total_frames - frames_recorded
            print(f"[promo]     Crane starts now, {crane_total} frames remaining")

        for trial_idx in range(max_trial_frames):
            if frames_recorded >= total_frames:
                break

            # Last trial: interpolate camera along crane path
            if is_last_trial and crane_start_frame is not None:
                t_norm = (frames_recorded - crane_start_frame) / max(crane_total - 1, 1)
                cam_pos, cam_lookat = crane_path(t_norm, keyframes)
            else:
                cam_pos, cam_lookat = closeup_pos, closeup_lookat

            obs, dones, infos = _step_and_render(cam_pos, cam_lookat)
            # Reset all envs when most are done, so grasps keep going
            if is_last_trial and dones.sum() > n_envs * 0.5:
                obs = vec_env.reset()
                _set_head_pose()
            frames_recorded += 1

            if infos[0].get("success", False):
                trial_success = True
                success_count += 1
                print(f"[promo]     env0 success at frame {trial_idx}!")
                # Brief linger (camera keeps craning if last trial)
                for linger_idx in range(linger_frames):
                    if frames_recorded >= total_frames:
                        break
                    if is_last_trial and crane_start_frame is not None:
                        t_norm = (frames_recorded - crane_start_frame) / max(crane_total - 1, 1)
                        cam_pos, cam_lookat = crane_path(t_norm, keyframes)
                    else:
                        cam_pos, cam_lookat = closeup_pos, closeup_lookat
                    obs, dones, infos = _step_and_render(cam_pos, cam_lookat)
                    if is_last_trial and dones.sum() > n_envs * 0.5:
                        obs = vec_env.reset()
                        _set_head_pose()
                    frames_recorded += 1
                if is_last_trial:
                    break
                else:
                    break

            if trial_idx % 60 == 0:
                elapsed = time.time() - t0
                print(f"    trial {trial+1} frame {trial_idx} ({elapsed:.1f}s)")

        if not trial_success:
            print(f"[promo]     No success in trial {trial+1}")

        # Reset all envs for a new random object position (not on last trial)
        if not is_last_trial and frames_recorded < total_frames:
            obs = vec_env.reset()
            _set_head_pose()
            print(f"[promo]     Reset for next trial (total frames: {frames_recorded})")

    print(f"[promo] Phase 1 done: {success_count}/{num_closeup_trials} successes, "
          f"{frames_recorded} frames used")

    # --- Phase 2: Continue crane if not started yet, or fill remaining frames ---
    crane_frames = total_frames - frames_recorded
    if crane_frames < 60:
        crane_frames = 60  # ensure at least 2s of crane
    if crane_start_frame is not None:
        # Crane already ran during last trial; just fill remaining frames
        print(f"[promo] Phase 2: Continuing crane for {crane_frames} remaining frames")
    else:
        print(f"[promo] Phase 2: Craning over {crane_frames} frames")

    for frame_idx in range(crane_frames):
        t_norm = (frames_recorded - (crane_start_frame or 0)) / max(
            (total_frames - (crane_start_frame or 0)) - 1, 1)
        t_norm = min(t_norm, 1.0)
        cam_pos, cam_lookat = crane_path(t_norm, keyframes)
        obs, dones, infos = _step_and_render(cam_pos, cam_lookat)
        # Reset all envs when most are done, so grasps keep going during crane
        if dones.sum() > n_envs * 0.5:
            obs = vec_env.reset()
            _set_head_pose()
        frames_recorded += 1

        if frame_idx % 100 == 0:
            elapsed = time.time() - t0
            n_done = dones.sum()
            print(f"  crane frame {frame_idx}/{crane_frames} "
                  f"({elapsed:.1f}s, {n_done} done)")

    # Stop recording and save mp4
    camera.stop_recording(save_to_filename=str(output), fps=fps)
    elapsed = time.time() - t0
    print(f"[promo] Done in {elapsed:.1f}s. {frames_recorded} frames. Saved to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render HSR fleet reveal promo video")
    parser.add_argument("--model", type=str,
                        default="results/ppo_apple/ppo_ik_curriculum.zip",
                        help="Path to trained PPO .zip model")
    parser.add_argument("--envs", type=int, default=1024,
                        help="Number of parallel envs")
    parser.add_argument("--frames", type=int, default=900,
                        help="Total frames to capture")
    parser.add_argument("--fps", type=int, default=30,
                        help="Output video fps")
    parser.add_argument("--output", type=str, default="results/promo_video.mp4",
                        help="Output mp4 path")
    parser.add_argument("--object", type=str, default="ycb_013_apple",
                        help="YCB object name")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--settle-steps", type=int, default=30)
    parser.add_argument("--max-hold-frames", type=int, default=600,
                        help="Max frames per close-up trial waiting for grasp success")
    parser.add_argument("--linger-frames", type=int, default=15,
                        help="Frames to linger after success before reset/crane")
    parser.add_argument("--num-closeup-trials", type=int, default=2,
                        help="Number of close-up grasp trials before craning")
    parser.add_argument("--approach-steps", type=int, default=60,
                        help="IK approach phase duration in sim steps (default 180=3.6s)")
    parser.add_argument("--descend-steps", type=int, default=45,
                        help="IK descend phase duration in sim steps (default 105=2.1s)")
    parser.add_argument("--grasp-steps", type=int, default=150,
                        help="IK grasp phase duration in sim steps (default 200=4.0s)")
    parser.add_argument("--obj-radius-min", type=float, default=None,
                        help="Min object placement radius (m). Default: 0.32")
    parser.add_argument("--obj-radius-max", type=float, default=None,
                        help="Max object placement radius (m). Default: 0.46")
    args = parser.parse_args()

    obj_radius_range = None
    if args.obj_radius_min is not None or args.obj_radius_max is not None:
        r_min = args.obj_radius_min if args.obj_radius_min is not None else 0.32
        r_max = args.obj_radius_max if args.obj_radius_max is not None else 0.46
        obj_radius_range = (r_min, r_max)

    render_promo(
        model_path=args.model,
        n_envs=args.envs,
        total_frames=args.frames,
        fps=args.fps,
        output_path=args.output,
        object_name=args.object,
        seed=args.seed,
        settle_steps=args.settle_steps,
        max_trial_frames=args.max_hold_frames,
        linger_frames=args.linger_frames,
        num_closeup_trials=args.num_closeup_trials,
        approach_steps=args.approach_steps,
        descend_steps=args.descend_steps,
        grasp_steps=args.grasp_steps,
        obj_radius_range=obj_radius_range,
    )


if __name__ == "__main__":
    main()
