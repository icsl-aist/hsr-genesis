# Changelog

All notable changes to this package will be documented in this file.

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

