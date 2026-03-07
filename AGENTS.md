# AGENTS

## Use the local virtualenv

- Prefer the repository virtual environment at `.venv` for Python commands.
- Use `.venv/bin/python` instead of the system `python`.
- Use `.venv/bin/pip` instead of the system `pip`.
- Run commands from the repository root (the directory containing this `AGENTS.md`).

## Setup

If `.venv` does not exist yet:

```bash
python -m venv .venv
.venv/bin/pip install -e .
```

If imports from `src/` are needed without an editable install, use:

```bash
PYTHONPATH=src .venv/bin/python <command>
```

## Common commands

Run an example:

```bash
PYTHONPATH=src .venv/bin/python examples/tutorials/hello_hsr_sensor.py
```

Run a module check:

```bash
.venv/bin/python -m py_compile src/hsr_genesis/sensor_manager.py
```

Run tests:

```bash
PYTHONPATH=src .venv/bin/python -m pytest
```

## Notes

- Keep the Genesis version pinned to the version declared by the repository dependencies.
- Do not casually upgrade Genesis while working on unrelated tasks, because internal APIs used here can change across releases.
- If a task depends on Genesis internals, verify behavior against the pinned version in `.venv` before changing compatibility code.
- Recent Genesis builds in this repo may use `quadrants` instead of `gstaichi` internally.
- When reproducing user issues, prefer `.venv/bin/python ...` so behavior matches the project environment.

## Monkey patches and compatibility shims

- This repository may need small compatibility shims for Genesis API differences between versions.
- Prefer localized monkey patches or wrappers inside `src/hsr_genesis/` instead of editing installed packages in `.venv`.
- Document the Genesis version or behavior being worked around near the patch site.
- Keep monkey patches minimal, targeted, and easy to remove once the upstream issue is fixed.
- When possible, guard compatibility code with feature detection or fallback imports instead of version-string checks.

## Taichi / gstaichi / quadrants

- Recent Genesis-related code may expose Taichi through `gstaichi` on older builds and `quadrants` on newer builds.
- Prefer compatibility imports such as `try: import gstaichi as ti` / `except Exception: import quadrants as ti` when working on Taichi-backed code in this repo.
- Keep Taichi kernels and Taichi helper functions small and focused; use `@ti.kernel` for bulk parallel work and `@ti.func` only as helpers called from kernels or other Taichi code.
- Avoid moving data back and forth between GPU and Python inside hot paths; batch work on-device whenever possible.
- Prefer Torch/Taichi tensors already on the active device and minimize repeated conversions like tensor-to-array, NumPy round-trips, or host-side loops.
- If data must leave the GPU, do it once at the boundary of the feature or debug path rather than repeatedly inside the main simulation/update loop.
