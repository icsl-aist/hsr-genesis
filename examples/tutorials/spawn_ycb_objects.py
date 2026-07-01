"""Spawn YCB object models (from the tmc_wrs_gazebo submodule) around the HSR.

This example demonstrates the SDF -> URDF converter in ``hsr_genesis.sdf_parser``:
Gazebo SDF models are converted on the fly and added to a Genesis scene at
random poses around the robot, then the simulation runs so the objects settle
on the ground plane.

Run:
    PYTHONPATH=src .venv/bin/python examples/tutorials/spawn_ycb_objects.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

import genesis as gs

sys.path.insert(0, str(Path(__file__).resolve().parent))

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=0, help="Number of sim steps (0 = run forever)")
parser.add_argument("--no-viewer", action="store_true", help="Disable the viewer window")
args = parser.parse_args()

URDF_PATH = Path(__file__).resolve().parents[2] / "data" / "urdf" / "hsrb4s.urdf"
MODELS_DIR = (
    Path(__file__).resolve().parents[2]
    / "data" / "tmc_wrs_gazebo" / "tmc_wrs_gazebo_worlds" / "models"
)

# A small, visually varied subset of the YCB object set.
YCB_MODELS = [
    "ycb_013_apple",
    "ycb_011_banana",
    "ycb_017_orange",
    "ycb_005_tomato_soup_can",
    "ycb_010_potted_meat_can",
    "ycb_077_rubiks_cube",
    "ycb_056_tennis_ball",
    "ycb_055_baseball",
    "ycb_061_foam_brick",
    "ycb_029_plate",
]


def main() -> None:
    try:
        gs.init(backend=gs.gpu)
    except RuntimeError as exc:  # pragma: no cover - demo fallback
        print(f"[Genesis] GPU backend unavailable ({exc}); falling back to CPU.")
        gs.init(backend=gs.cpu)

    from hsr_genesis.hsr_rigid_entity import HSRBURDF
    from hsr_genesis.sdf_parser import load_sdf_model

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
        sim_options=gs.options.SimOptions(dt=0.02, substeps=10),
        rigid_options=gs.options.RigidOptions(
            use_gjk_collision=True,
        ),
        show_viewer=not args.no_viewer,
    )

    # Ground plane.
    scene.add_entity(gs.morphs.Plane())

    # HSR robot.
    hsr = scene.add_entity(
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
        ),
        visualize_contact=True,
    )

    # Spawn YCB objects at random positions around the robot.
    rng = np.random.default_rng(seed=42)
    objects = []
    for name in YCB_MODELS:
        model_dir = MODELS_DIR / name
        if not model_dir.exists():
            print(f"[skip] {name} not found (submodule not initialized?)")
            continue

        # Sample a pose in an annulus around the robot base.
        theta = rng.uniform(0.0, 2.0 * math.pi)
        radius = rng.uniform(0.6, 1.6)
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        z = rng.uniform(0.05, 0.25)
        yaw = rng.uniform(0.0, 2.0 * math.pi)

        robot = load_sdf_model(model_dir)
        entity = scene.add_entity(
            gs.morphs.URDF(
                file=robot,
                pos=(x, y, z),
                euler=(0.0, 0.0, math.degrees(yaw)),
                fixed=False,
            ),
        )
        objects.append((name, entity))
        print(f"[spawn] {name:30s} at ({x:+.2f}, {y:+.2f}, {z:+.2f}) yaw={yaw:+.2f}")

    scene.build()

    print(f"\nSpawned {len(objects)} YCB objects around the HSR.")
    if args.no_viewer and args.steps > 0:
        print(f"Running {args.steps} steps headless.")
    else:
        print("Running simulation. Close the viewer window to exit.")
    print()

    # Hold the robot in place and let the objects settle.
    n_steps = 0
    while True:
        scene.step()
        n_steps += 1
        if n_steps % 100 == 0:
            # Report object positions for a sanity check.
            for name, ent in objects:
                pos = ent.get_pos()
                if pos.ndim > 1:
                    pos = pos[0]
                print(f"[{n_steps:4d}] {name:30s} pos=({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})")
        if args.steps > 0 and n_steps >= args.steps:
            break


if __name__ == "__main__":
    main()
