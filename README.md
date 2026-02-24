# hsr-genesis

HSR-specific integrations for the Genesis ecosystem.

This repository provides HSR tools (IK, base control, gripper control, sensors)
as a standalone Python package that lives alongside the main Genesis codebase.

Repository: `https://github.com/icsl-aist/hsr-genesis.git`

License: BSD 3-Clause (compatible with the original ROS packages).

## About Genesis and GPU Acceleration

Genesis is a physics simulator that can run on the GPU for fast, large-scale
simulation. Internally it uses the Taichi compiler to JIT-compile numerical
kernels and run them efficiently on supported GPUs. This allows higher
throughput for simulation, IK, and control workloads compared to CPU-only runs.
The HSR modules here also support PyTorch tensors for inputs/outputs, so you
can integrate them into Torch-based pipelines.

## Quick Start

Clone the repository (includes required submodules):

```bash
git clone https://github.com/icsl-aist/hsr-genesis.git
git submodule update --init --recursive
```

## Modules (What’s Inside)

- `hsr_genesis.hsr_rigid_entity`: HSR-specific rigid entity helpers that connect IK, base control, and gripper control.
- `hsr_genesis.analytic_ik`: Analytic IK solver for HSR-B/HSR-C (ported from `hsrb_analytic_ik`).
- `hsr_genesis.base_controller`: Base controller utilities and kinematics kernels (ported from `hsrb_base_controllers`).
- `hsr_genesis.gripper_controller`: Gripper control actions and interfaces (apply-force, grasp), ported from `hsrb_gripper_controller`.
- `hsr_genesis.sensor_manager`: URDF-driven sensor attachment helpers for HSR.

### GPU-accelerated modules

The following modules include Taichi/Torch kernels and can run on GPU when
Genesis is initialized with a GPU backend:

- `hsr_genesis.analytic_ik`
- `hsr_genesis.base_controller`

## Data (Required Assets)

Required assets live under `hsr_genesis/data`:

- `data/hsrb_analytic_ik/joint_configs`: Reference joint configurations used by IK tests.
- `data/urdf/hsrb4s.urdf`: Main HSR URDF file.
- `data/urdf/hsrb_meshes` and `data/urdf/hsrb_description`: Mesh and description assets referenced by the URDF.

## Development

Install the package in editable mode:

```bash
pip install -e hsr_genesis
```

## Run Hello HSR (venv)

This example opens a viewer window and loads the HSR robot.
From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e hsr_genesis
PYTHONPATH=hsr_genesis/src python hsr_genesis/examples/tutorials/hello_hsr_parallel.py

Sensor demo:

```bash
PYTHONPATH=hsr_genesis/src python hsr_genesis/examples/tutorials/hello_hsr_sensor.py
```
```

If you see a viewer window, the example is running correctly.
