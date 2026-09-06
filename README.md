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

## Demos

### GPU parallel simulation (1024 envs)

![promo_video](https://github.com/icsl-aist/hsr-genesis/releases/download/gif-assets/promo_video.gif)

### Sensor demo (debug visualization)

![hello_hsr_sensor](https://github.com/icsl-aist/hsr-genesis/releases/download/gif-assets/hello_hsr_sensor.gif)

### IK grasp

![IK_grasp_hsr](https://github.com/icsl-aist/hsr-genesis/releases/download/gif-assets/IK_grasp_hsr.gif)

### RRT path planning

![rrt_path_planning_hsr](https://github.com/icsl-aist/hsr-genesis/releases/download/gif-assets/rrt_path_planning_hsr.gif)

## Docker

The Docker environment provides a reproducible setup with CUDA 12.4,
necessary for the `batch_renderer` camera backend (the prebuilt
`gs-madrona` wheel requires CUDA 12.x for NVVM JIT linking).

**Use Docker if you need `camera_backend="batch_renderer"`.**  If you
only use the default rasterizer backend (or run headless physics without
cameras), a native install with a modern CUDA toolkit may give better
simulation performance — newer CUDA versions include optimised cuBLAS
and cuDNN kernels that accelerate Genesis internals.

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
scene.build(n_envs=1024)   # tune to your GPU VRAM
```

#### Measured throughput

Benchmarked on the YCB grasp pipeline (approach → descend → grasp → lift)
with `dt=0.02` s, 4 substeps, `show_viewer=False`. Realtime factor is
steps/s relative to wall-clock (1.0× = realtime):

| N envs | RTX 5060 Ti (16 GB) | | | A100-SXM4 (80 GB) | | |
|--------|---------------------|------|------|--------------------|------|------|
|        | steps/s | RT factor | envs·steps/s | steps/s | RT factor | envs·steps/s |
| 1      | 32.4    | 0.65×     | 32           | 24.6    | 0.49×     | 25            |
| 8      | 28.8    | 0.58×     | 230          | 21.0    | 0.42×     | 168           |
| 32     | 27.4    | 0.55×     | 878          | 19.9    | 0.40×     | 636           |
| 128    | 25.8    | 0.52×     | 3,308        | 18.9    | 0.38×     | 2,416         |
| 256    | 24.8    | 0.50×     | 6,338        | 17.8    | 0.36×     | 4,546         |
| 512    | 22.9    | 0.46×     | 11,702       | 16.7    | 0.33×     | 8,555         |
| 1024   | 19.8    | 0.40×     | 20,224       | 15.6    | 0.31×     | 15,972        |
| 4096   | 12.1    | 0.24×     | 49,605       | 10.3    | 0.21×     | 42,161        |

The 5060 Ti is competitive with the A100 at low-to-mid N (higher clock
speeds win when the GPU is underutilized), reaching **20,224 envs·steps/s**
at 1024 envs — a **632×** throughput improvement over a single environment.
At 4096 envs it peaks at **49,605 envs·steps/s** (1,550× over N=1) before
hitting the 16 GB VRAM ceiling. The A100 scales further (max N=16,384,
90,050 envs·steps/s) but is engine-limited, not VRAM-limited.

Even at N=1 the 5060 Ti runs at **0.65× realtime** — fast enough for
interactive development. At 1024 envs it still maintains 0.40× realtime
per environment, meaning 1024 independent simulations complete in the
time 410 sequential ones would.

#### IK performance

| Configuration | Time per call | Per-env |
|---------------|---------------|---------|
| Single env    | 5.78 ms       | 5.78 ms |
| 256-env batch | 7.71 ms       | 0.030 ms |

The 256-env batch achieves **0.030 ms/env** — a **192×** reduction in
per-environment IK latency, enabling the IK planner to pre-compute
approach/descend/lift trajectories for all environments at episode reset.

Practical guidance:

- Start with a power-of-two value (e.g. 64, 128, 256, 512) and increase until
  VRAM is nearly full or throughput stops scaling.
- Monitor VRAM usage with `nvidia-smi` and back off if you see OOM errors.
- Very large batch sizes (≥ 1024) can saturate memory bandwidth instead of
  compute; profile with `nvitop` or `nsys` to find the sweet spot.
- Combining `show_viewer=False` with a high `n_envs` is the recommended setup
  for RL training and large-scale data collection.

## Citation

If you use this work in your research, please cite the following paper:

```bibtex
@inproceedings{matsusaka2026hsr_genesis,
  author    = {Yosuke Matsusaka and Keisuke Takeshita and Ryuichi Sakakibara and Takashi Yamamoto},
  title     = {Development and Evaluation of a Massively Parallel Physics Simulator with GPU-Accelerated Inverse Kinematics for Mobile Manipulators},
  booktitle = {Proceedings of the Robotics Society of Japan Annual Conference (RSJ)},
  year      = {2026},
}
```
