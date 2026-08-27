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
import subprocess
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

    # Mid: low-altitude fly-through the fleet at z=2.0, traveling diagonally
    # from the corner toward grid center. Camera is just above robot height
    # so individual robot motions are clearly visible as it sweeps past them.
    mid_pos_world = (0.0, -5.0, 2.0)
    mid_lookat_world = (5.0, 5.0, 0.5)

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


def _overlay_text(
    input_path: str,
    output_path: str,
    text: str,
    font_size: int = 36,
    fade_in_seconds: float = 1.0,
) -> None:
    """Overlay text at bottom-center of video using ffmpeg drawtext.

    Text fades in over fade_in_seconds and stays for the rest of the video.
    A subtle dark shadow improves readability on bright backgrounds.
    """
    font_file = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    # Escape colons and special chars for ffmpeg drawtext
    escaped = text.replace(":", "\\:").replace("'", "\\'")
    drawtext = (
        f"drawtext="
        f"fontfile={font_file}:"
        f"text='{escaped}':"
        f"fontsize={font_size}:"
        f"fontcolor=white:"
        f"borderw=2:bordercolor=black@0.6:"
        f"x=(w-text_w)/2:"
        f"y=h-text_h-30:"
        f"alpha='if(lt(t,{fade_in_seconds}),t/{fade_in_seconds},1)'"
    )
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", drawtext,
        "-c:a", "copy",
        output_path,
    ]
    print(f"[promo] Overlaying text: '{text}'")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[promo] WARNING: ffmpeg overlay failed: {result.stderr[-500:]}")
        # Fall back to original
        import shutil
        shutil.copy(input_path, output_path)
    else:
        print(f"[promo] Overlay saved to {output_path}")


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
    lift_steps: int = 180,
    obj_radius_range: tuple[float, float] | None = None,
    overlay_text: str | None = None,
    apple_preroll_frames: int = 60,
    gripper_effort: float = 8.0,
) -> None:
    """Render the promo video.

    The video has three phases:
    0. Apple close-up: Camera starts tight on the apple object so the
       viewer recognizes the task, then pulls back to reveal the robot.
    1. Close-up: Camera stays tight on env0's single robot. Runs multiple
       grasp trials with shortened approach/descend phases so the robot
       gets to the grasp quickly. Retargets for a new random object
       position between trials.
    2. Crane: Camera cranes from the close-up to a top-down fleet reveal.
       Grasps continue during the crane with auto-retargets on done.
    """
    _ensure_genesis_initialized()

    # Shorten approach/descend phases so the robot gets to the grasp faster.
    # Default is approach=180 (3.6s), descend=105 (2.1s). We use 60 (1.2s)
    # and 45 (0.9s) — the arm moves more quickly to the pre-grasp position.
    IKPlanner.configure_phase_steps(approach=approach_steps, descend=descend_steps, grasp=grasp_steps, lift=lift_steps)
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
        gripper_effort_override=gripper_effort,
        terminate_delay_steps=60,
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

    # --- Warmup: step simulation to an interesting state before recording ---
    # The first frame is used as a thumbnail in some systems, so we want it
    # to show the fleet mid-action (robots driving/grasping) rather than the
    # initial pose. Step the sim without recording until robots are in motion.
    warmup_steps = IKPlanner.lift_start() + 30  # mid-lift — robots holding objects up
    print(f"[promo] Warmup: stepping {warmup_steps} frames (no recording)...")
    for _ in range(warmup_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)
        _set_head_pose()
        # Retarget done envs so the fleet keeps moving
        if dones.sum() > 0:
            obs = env.retarget()
            _set_head_pose()

    # Retarget all envs after warmup so closeup trials start fresh —
    # otherwise env0 is already mid-lift and terminates immediately.
    obs = env.retarget()
    _set_head_pose()
    print(f"[promo] Retarget after warmup (fresh episodes for recording)")

    # Start recording — first frame will be the fleet mid-action thumbnail
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    camera.start_recording()
    print(f"[promo] Recording (max {total_frames} frames at {fps} fps)...")

    t0 = time.time()
    frames_recorded = 0
    closeup_pos, closeup_lookat = crane_path(0.0, keyframes)
    success_count = 0

    # --- Phase 0: Apple close-up pre-roll ---
    # Start with a tight shot of the apple on the table so the viewer
    # recognizes the task, then pull back to reveal the robot.
    if apple_preroll_frames > 0:
        print(f"[promo] Phase 0: Apple close-up pre-roll ({apple_preroll_frames} frames)")
        # Get env0 apple world position
        obj_pos_world = pick_env._obj_pos()[0].cpu().numpy()  # (3,)
        env0_xy = env0_offset[:2]

        # Apple close-up: camera 0.3m from apple, at table height, looking at apple
        apple_cam_pos = np.array([
            obj_pos_world[0] + 0.3,
            obj_pos_world[1] - 0.2,
            obj_pos_world[2] + 0.15,
        ])
        apple_cam_lookat = obj_pos_world.copy()
        apple_cam_lookat[2] += 0.02  # look at apple center

        # Robot close-up (current crane start)
        robot_cam_pos = closeup_pos.copy()
        robot_cam_lookat = closeup_lookat.copy()

        for pre_idx in range(apple_preroll_frames):
            if frames_recorded >= total_frames:
                break
            # First half: hold on apple. Second half: pull back to robot.
            if pre_idx < apple_preroll_frames // 2:
                cam_pos = apple_cam_pos
                cam_lookat = apple_cam_lookat
            else:
                t_pull = (pre_idx - apple_preroll_frames // 2) / max(
                    apple_preroll_frames - apple_preroll_frames // 2 - 1, 1)
                t_pull = t_pull * t_pull * (3 - 2 * t_pull)  # smoothstep
                cam_pos = apple_cam_pos + (robot_cam_pos - apple_cam_pos) * t_pull
                cam_lookat = apple_cam_lookat + (robot_cam_lookat - apple_cam_lookat) * t_pull

            # Step sim but don't apply policy yet (robot holds during pre-roll)
            camera.set_pose(pos=cam_pos.tolist(), lookat=cam_lookat.tolist())
            camera.render()
            frames_recorded += 1

        print(f"[promo]   Pre-roll done ({frames_recorded} frames)")

    print(f"[promo] Phase 1: {num_closeup_trials} close-up grasp trials with random resets")

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
            # During the last trial, don't retarget — let env0 finish and
            # linger. The crane phase will handle retargeting.
            frames_recorded += 1

            # Detect success but don't break — wait for full episode (dones)
            # so the viewer sees the complete lift+move motion.
            if infos[0].get("success", False) and not trial_success:
                trial_success = True
                success_count += 1
                print(f"[promo]     env0 success at frame {trial_idx}!")

            # Break when env0's episode is complete (dones[0] = True at max_steps)
            if dones[0]:
                if not trial_success:
                    print(f"[promo]     env0 episode ended at frame {trial_idx}")
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
                    # No retarget during last trial's linger
                    frames_recorded += 1
                break

            if trial_idx % 60 == 0:
                elapsed = time.time() - t0
                print(f"    trial {trial+1} frame {trial_idx} ({elapsed:.1f}s)")

        if not trial_success:
            print(f"[promo]     No success in trial {trial+1}")

        # Retarget for a new random object position (robot stays in place)
        if not is_last_trial and frames_recorded < total_frames:
            obs = env.retarget()
            _set_head_pose()
            print(f"[promo]     Retarget for next trial (total frames: {frames_recorded})")

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
        # Retarget all envs when any are done (new object, robot stays)
        if dones.sum() > 0:
            obs = env.retarget()
            _set_head_pose()
        frames_recorded += 1

        if frame_idx % 100 == 0:
            elapsed = time.time() - t0
            n_done = dones.sum()
            n_success = sum(1 for info in infos if info.get("success", False))
            print(f"  crane frame {frame_idx}/{crane_frames} "
                  f"({elapsed:.1f}s, {n_done} done, {n_success} success)")

    # Stop recording and save mp4
    camera.stop_recording(save_to_filename=str(output), fps=fps)
    elapsed = time.time() - t0
    print(f"[promo] Done in {elapsed:.1f}s. {frames_recorded} frames. Saved to {output}")

    # Overlay text (e.g. repo URL) via ffmpeg post-processing
    if overlay_text:
        tmp_path = str(output.parent / (output.stem + "_raw" + output.suffix))
        import shutil
        shutil.move(str(output), tmp_path)
        _overlay_text(tmp_path, str(output), overlay_text)
        Path(tmp_path).unlink(missing_ok=True)


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
    parser.add_argument("--lift-steps", type=int, default=180,
                        help="IK lift phase duration in sim steps (default 155=3.1s)")
    parser.add_argument("--gripper-effort", type=float, default=8.0,
                        help="Gripper effort in N (default 8.0 for mobile lift)")
    parser.add_argument("--obj-radius-min", type=float, default=None,
                        help="Min object placement radius (m). Default: 0.32")
    parser.add_argument("--obj-radius-max", type=float, default=None,
                        help="Max object placement radius (m). Default: 0.46")
    parser.add_argument("--overlay-text", type=str, default=None,
                        help="Text to overlay at bottom of video (e.g. repo URL)")
    parser.add_argument("--apple-preroll-frames", type=int, default=60,
                        help="Frames to show apple close-up before robot view (0=disable)")
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
        lift_steps=args.lift_steps,
        obj_radius_range=obj_radius_range,
        overlay_text=args.overlay_text,
        apple_preroll_frames=args.apple_preroll_frames,
        gripper_effort=args.gripper_effort,
    )


if __name__ == "__main__":
    main()
