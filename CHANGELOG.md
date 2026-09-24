# Changelog

All notable changes to this package will be documented in this file.

## [Unreleased]

### Added
- `hsr_genesis.sdf_world`: import Gazebo SDF worlds (incl. `.world.xacro`) into Genesis.
  - `parse_sdf_world`: xacro expansion, `model://` URI resolution, world poses/static flags, gravity and physics step size.
  - `spawn_sdf_world`: spawns every `<include>`d model into a `gs.Scene` (before `build()`); SDF `<plane>` models (e.g. `wrc_ground_plane`) become `gs.morphs.Plane`.
- Example `examples/tutorials/spawn_wrc_world.py`: loads the WRS2020 arena (`wrs2020.world.xacro`) plus the HSR robot at the upstream start pose.
- Tests `tests/test_sdf_world.py`.

### Changed
- `sdf_parser`: extracted `_resolve_sdf_file` from `load_sdf_model` (shared with `sdf_world`); module scope note now points at `sdf_world` for world-level files.

## [0.1.0] - 2026-02-24

### Initial Import
- Imported HSR-specific IK, base control, gripper control, and sensor utilities.
- Added GPU-enabled Taichi/Torch paths for IK and base control.
- Added data assets under `data/` (URDF, meshes via submodule, and IK test configs).
- Added tutorial examples:
  - `hello_hsr_parallel.py` (parallel IK demo)
  - `hello_hsr_sensor.py` (sensor setup demo)
- Added BSD 3-Clause license compatible with original ROS packages.
- Added README with quick-start, module descriptions, GPU notes, and example commands.

Author: Yosuke Matsusaka <yosuke.matsusaka@gmail.com>

