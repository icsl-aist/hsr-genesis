"""Import the WRS2020 Gazebo arena (``tmc_wrs_gazebo`` submodule) into Genesis.

The arena ships as a Gazebo *world* file,
``data/tmc_wrs_gazebo/tmc_wrs_gazebo_worlds/worlds/wrs2020.world.xacro``, which
``<include>``s the ``wrc_*`` furniture models plus two ``person_standing``
models at fixed poses.  ``hsr_genesis.sdf_world`` expands the xacro world,
resolves the ``model://`` URIs against the sibling ``models/`` directory, and
spawns every model into a Genesis scene; the world's ground plane
(``wrc_ground_plane``, SDF ``<plane>`` geometry) becomes ``gs.morphs.Plane``.
The HSR robot is then placed at the arena start pose used by the upstream
launch file (``wrs_practice0.launch``: ``-x -2.1 -y 1.2 -Y -1.57``).

Run:
    PYTHONPATH=src .venv/bin/python examples/tutorials/spawn_wrc_world.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import genesis as gs

sys.path.insert(0, str(Path(__file__).resolve().parent))

parser = argparse.ArgumentParser()
parser.add_argument(
    "--steps", type=int, default=0,
    help="Number of sim steps (0 = run forever)",
)
parser.add_argument(
    "--no-viewer", action="store_true",
    help="Disable the viewer window",
)
parser.add_argument(
    "--world", type=str, default=None,
    help="World file to import (default: the wrs2020 xacro world)",
)
parser.add_argument(
    "--no-robot", action="store_true",
    help="Import only the arena, without the HSR robot",
)
parser.add_argument(
    "--xacro-arg", action="append", default=[], metavar="NAME:=VALUE",
    help="xacro argument for the world (repeatable), e.g. "
         "--xacro-arg fast_physics:=false",
)
args = parser.parse_args()

XACRO_ARGS = {}
for item in args.xacro_arg:
    name, sep, value = item.partition(":=")
    if not sep:
        parser.error(f"--xacro-arg expects NAME:=VALUE, got {item!r}")
    XACRO_ARGS[name.strip()] = value.strip()

ROOT = Path(__file__).resolve().parents[2]
WORLD_PATH = (
    Path(args.world) if args.world
    else ROOT / "data" / "tmc_wrs_gazebo" / "tmc_wrs_gazebo_worlds"
    / "worlds" / "wrs2020.world.xacro"
)
URDF_PATH = ROOT / "data" / "urdf" / "hsrb4s.urdf"

# Upstream wrs_practice0.launch robot_pos: -x -2.1 -y 1.2 -z 0 -Y -1.57
ROBOT_POS = (-2.1, 1.2, 0.0)
ROBOT_YAW_DEG = -90.0


def _init_genesis() -> None:
    try:
        gs.init(backend=gs.gpu)
    except RuntimeError as exc:  # pragma: no cover - demo fallback
        print(f"[Genesis] GPU backend unavailable ({exc}); falling back to CPU.")
        gs.init(backend=gs.cpu)


def main() -> None:
    _init_genesis()

    from hsr_genesis.hsr_rigid_entity import HSRBURDF
    from hsr_genesis.sdf_world import parse_sdf_world, spawn_sdf_world

    world = parse_sdf_world(WORLD_PATH, xacro_args=XACRO_ARGS or None)
    print(
        f"[world] {world.source.name}: {len(world.models)} models,"
        f" gravity={world.gravity}, max_step_size={world.max_step_size}"
    )
    # Fast physics is on by default in the xacro (``fast_physics`` default is
    # true -> max_step_size 0.003).  With ``fast_physics:=false`` (the upstream
    # launch default) the <physics> block stays empty; fall back to Genesis'
    # own default step size.
    dt = float(world.max_step_size) if world.max_step_size else 0.02
    sim_kwargs = {"dt": dt, "substeps": 10}
    if world.gravity is not None:
        sim_kwargs["gravity"] = tuple(float(v) for v in world.gravity)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, -1.5, 1.8),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=45,
            max_FPS=60,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
            world_frame_size=1.0,
            show_link_frame=False,
            plane_reflection=True,
            ambient_light=(0.3, 0.3, 0.3),
        ),
        sim_options=gs.options.SimOptions(**sim_kwargs),
        rigid_options=gs.options.RigidOptions(use_gjk_collision=True),
        show_viewer=not args.no_viewer,
    )

    entities = spawn_sdf_world(scene, WORLD_PATH, world=world)
    for model in world.models:
        kind = "plane" if model.is_plane else "urdf"
        print(
            f"[arena] {model.name:24s} {kind:5s} static={model.static}"
            f" pose={tuple(round(float(v), 2) for v in model.pose[:3, 3])}"
        )

    if not args.no_robot:
        scene.add_entity(
            HSRBURDF(
                file=str(URDF_PATH),
                pos=ROBOT_POS,
                euler=(0.0, 0.0, ROBOT_YAW_DEG),
                fixed=False,
                recompute_inertia=True,
                links_to_keep=["hand_palm_link"],
                robot="hsrb",
                base_mode="planar",
                end_effector_frame="hand_palm_link",
                use_base_controller=True,
                optimizer="gpu",
            ),
            name="hsr",
        )
        print(
            f"[robot] hsrb4s at ({ROBOT_POS[0]:+.2f}, {ROBOT_POS[1]:+.2f},"
            f" {ROBOT_POS[2]:+.2f}) yaw={ROBOT_YAW_DEG:+.1f} deg"
        )

    scene.build()
    print(f"\nImported {len(entities)} world models (dt={dt}).")


    n_steps = 0
    while True:
        scene.step()
        n_steps += 1
        if n_steps % 200 == 0:
            print(f"[{n_steps:5d}] sim running")
        if args.steps > 0 and n_steps >= args.steps:
            break


if __name__ == "__main__":
    main()
