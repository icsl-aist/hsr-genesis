"""Render a promotion video: HSR fleet reveal crane shot.

Loads a trained PPO policy, builds a 1024-env scene, and records a
continuous crane shot from a close-up of 4 robots to a top-down reveal
of the full 32x32 grid.

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
    env0 is at (-46.5, -46.5, 0). The 2x2 corner (envs 0,1,32,33) is
    centered at world (-45, -45, 0).

    Camera positions are env0-relative: world (wx, wy, wz) -> relative
    (wx - env0_offset[0], wy - env0_offset[1], wz).
    """
    offset_x, offset_y = env0_offset[0], env0_offset[1]

    # Start: low heroic close-up of the 2x2 corner
    start_pos_world = (-43.0, -49.0, 2.5)
    start_lookat_world = (-45.0, -45.0, 0.5)

    # Mid: approaching grid center, rising
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
) -> None:
    """Render the promo video."""
    _ensure_genesis_initialized()

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

    # Start recording
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    camera.start_recording()
    print(f"[promo] Recording {total_frames} frames at {fps} fps...")

    t0 = time.time()
    for frame_idx in range(total_frames):
        # Step the policy
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)

        # Reset if all envs are done
        if np.all(dones):
            obs = vec_env.reset()

        # Interpolate camera pose along crane path with smoothstep easing
        t_norm = frame_idx / max(total_frames - 1, 1)
        pos, lookat = crane_path(t_norm, keyframes)
        camera.set_pose(pos=pos.tolist(), lookat=lookat.tolist())

        # Render to capture the frame for recording
        camera.render()

        if frame_idx % 100 == 0:
            elapsed = time.time() - t0
            print(f"  frame {frame_idx}/{total_frames} ({elapsed:.1f}s)")

    # Stop recording and save mp4
    camera.stop_recording(save_to_filename=str(output), fps=fps)
    elapsed = time.time() - t0
    print(f"[promo] Done in {elapsed:.1f}s. Saved to {output}")


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
    args = parser.parse_args()

    render_promo(
        model_path=args.model,
        n_envs=args.envs,
        total_frames=args.frames,
        fps=args.fps,
        output_path=args.output,
        object_name=args.object,
        seed=args.seed,
        settle_steps=args.settle_steps,
    )


if __name__ == "__main__":
    main()
