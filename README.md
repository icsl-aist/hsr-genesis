# hsr-genesis

HSR-specific integrations for the Genesis ecosystem.

This repository provides HSR tools (IK, base control, gripper control, sensors)
as a standalone Python package that lives alongside the main Genesis codebase.

Repository: `https://github.com/icsl-aist/hsr-genesis.git`

License: BSD 3-Clause (compatible with the original ROS packages).

## Docker (Recommended)

The Docker environment provides a reproducible setup with CUDA 12.4,
necessary for the `batch_renderer` camera backend (the prebuilt
`gs-madrona` wheel requires CUDA 12.x for NVVM JIT linking).

### Prerequisites

- Docker with the NVIDIA Container Toolkit (`nvidia-container-runtime`)
- NVIDIA driver ≥ 550 (tested with 595)

### Quick start

```bash
# Run all tests (headless, xvfb auto-started)
./scripts/docker-run.sh

# Run specific tests
./scripts/docker-run.sh -- python -m pytest tests/test_camera_lighting.py -v

# Run a user script (headless)
./scripts/docker-run.sh -- python examples/tutorials/hello_hsr_sensor.py

# Interactive shell
./scripts/docker-run.sh -- bash
```

### Viewer (windowed GUI)

To see the Genesis viewer window on your host desktop:

```bash
xhost +local:docker
./scripts/docker-run.sh --viewer -- examples/tutorials/hello_hsr_parallel.py
```

The `--viewer` flag forwards your X11 socket and sets `--network host`
so the OpenGL window appears on your host desktop.  Always run
`xhost +local:docker` first to allow the container to connect.

### How it works

| Component | What it provides |
|-----------|------------------|
| `nvidia/cuda:12.4.1-runtime` | CUDA 12.4 runtime libraries |
| `libnvidia-gl-550` | NVIDIA Vulkan ICD manifest (needed by batch renderer) |
| `NVIDIA_DRIVER_CAPABILITIES=all` | Mounts host graphics/Vulkan driver libraries |
| `xvfb-run` | Virtual framebuffer for headless rendering |
| `libx11-dev libxrender-dev …` | X11 libraries for the Genesis viewer |

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
cd hsr-genesis
git submodule update --init --recursive
```

## Modules (What’s Inside)

- `hsr_genesis.hsr_rigid_entity`: HSR-specific rigid entity helpers that connect IK, base control, and gripper control.
- `hsr_genesis.analytic_ik`: Analytic IK solver for HSR-B/HSR-C (ported from `hsrb_analytic_ik`).
- `hsr_genesis.base_controller`: Base controller utilities and kinematics kernels (ported from `hsrb_base_controllers`).
- `hsr_genesis.gripper_controller`: Gripper control actions and interfaces (apply-force, grasp), ported from `hsrb_gripper_controller`.
- `hsr_genesis.sensor_manager`: URDF-driven sensor attachment helpers for HSR.

## HSR Rigid Entity Options

`HSRBURDF` is a thin wrapper around `gs.morphs.URDF` that wires in HSR-specific behavior by setting attributes consumed by `HSRRigidEntity`. You can pass the options below when constructing `HSRBURDF`.

- `file`: URDF path (required by `gs.morphs.URDF`).
- `robot`: Robot variant, `"hsrb"` (default) or `"hsrc"`. Selects IK parameters.
- `end_effector_frame`: Link name used as the IK end-effector frame. Default is `"hand_palm_link"`.
- `base_mode`: Base kinematics model. `"planar"` (default) uses x/y + yaw. `"rotation_z"` enables yaw-only rotation.
- `use_base_yaw_ik`: If `True`, include base yaw in IK solving for end-effector alignment.
- `use_base_controller`: Enable the base controller behavior. Default is `True`.
- `base_control_mode`: `"controller"` (default) uses the base controller, `"qpos"` drives the base by directly setting robot positions. Note: `"qpos"` is fast and precise, but it does not simulate real-robot base control error.
- `optimizer`: IK optimizer selection, `"auto"` (default) or a specific backend recognized by Genesis.

Minimal example:

```python
import genesis as gs
from hsr_genesis.hsr_rigid_entity import HSRBURDF

hsr = HSRBURDF(
    file="data/urdf/hsrb4s.urdf",
    robot="hsrb",
    base_mode="planar",
    base_control_mode="controller",
    use_base_controller=True,
)
```

### GPU-accelerated modules

The following modules include Taichi/Torch kernels and can run on GPU when
Genesis is initialized with a GPU backend:

- `hsr_genesis.analytic_ik`
- `hsr_genesis.base_controller`

## Data (Required Assets)

Required assets live under `hsr_genesis/data`:

- `data/hsrb_analytic_ik/joint_configs`: Reference joint configurations used by IK tests.
- `data/urdf/hsrb4s.urdf`: Main HSR URDF file.
- `data/urdf/hsrb_meshes`: Mesh assets referenced by the URDF.

## Run Hello HSR (venv)

This example opens a viewer window and loads the HSR robot.
From the repo root:

```bash
cd hsr-genesis
python -m venv .venv
source .venv/bin/activate
pip install -e .
PYTHONPATH=src python examples/tutorials/hello_hsr_parallel.py
```

Sensor demo (whole-body PD control + base controller + URDF sensors):

```bash
PYTHONPATH=src python examples/tutorials/hello_hsr_sensor.py
```

If you see a viewer window, the example is running correctly.

## Performance Tips

### Disable visualization for maximum throughput

Running the Genesis viewer has a significant overhead. For training, data
collection, or any headless workload, disable the viewer:

```python
gs.init(backend=gs.cuda)

scene = gs.Scene(
    show_viewer=False,   # disables the interactive viewer
)
```

Disabling the viewer typically gives a **large speedup** (often 5–10× or more
depending on the scene) because Genesis no longer needs to synchronize
simulation state with the GUI or render frames.

### Increase parallelism to saturate the GPU

Genesis supports batched simulation: multiple independent environments run
simultaneously on the same GPU. Increasing the number of parallel environments
(`n_envs`) amortizes kernel-launch overhead and keeps the GPU fully utilized.

```python
scene.build(n_envs=512)   # tune to your GPU VRAM
```

Practical guidance:

- Start with a power-of-two value (e.g. 64, 128, 256, 512) and increase until
  VRAM is nearly full or throughput stops scaling.
- Monitor VRAM usage with `nvidia-smi` and back off if you see OOM errors.
- Very large batch sizes (≥ 1024) can saturate memory bandwidth instead of
  compute; profile with `nvitop` or `nsys` to find the sweet spot.
- Combining `show_viewer=False` with a high `n_envs` is the recommended setup
  for RL training and large-scale data collection.
