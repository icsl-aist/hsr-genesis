# Changelog

All notable changes to this package will be documented in this file.

## [Unreleased]

### Changed
- Base trajectories now hold their final pose until `reset_base_trajectory_batched` or replacement; removed the automatic-stop constructor options `stop_velocity_threshold` and `stop_time_margin`.
- Outer trajectory feedback and damping now use one controller-owned production tuning value instead of per-instance constructor overrides.
- Single-waypoint batched base control now uses shortest-path yaw and routes named or derivative-bearing trajectories through the canonical controller.
- Steering gains now have one controller-owned source and no longer depend on initialization order.

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

