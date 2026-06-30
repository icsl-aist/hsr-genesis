from __future__ import annotations

import sys
from pathlib import Path

import pytest

root = Path(__file__).resolve().parents[1]
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

if "hsr_genesis" in sys.modules:
    del sys.modules["hsr_genesis"]

import genesis as gs  # noqa: E402
# Ensure the real genesis.sensors is registered in sys.modules before
# test_sensor_manager.py can register a stub via setdefault.  Genesis
# exposes sensors as an attribute (``gs.sensors``) but does not always
# register it as ``sys.modules["genesis.sensors"]``.
import genesis.options.sensors  # noqa: E402
sys.modules.setdefault("genesis.sensors", gs.sensors)

# Initialize Genesis once for the entire test session with a consistent
# backend. Individual test files also guard with
# `if not getattr(gs, "_initialized", False)`, so this central call wins
# and prevents backend conflicts (e.g. a CPU test initializing first, then
# a GPU test failing with a coupler error).
#
# Use GPU on local machines, CPU on CI (where GPU is unavailable).
if not getattr(gs, "_initialized", False):
    import os

    _on_ci = bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"))
    if _on_ci:
        gs.init(backend=gs.cpu, precision="32", logging_level="warning")
    else:
        try:
            gs.init(backend=gs.gpu, precision="32", logging_level="warning")
        except Exception:
            gs.init(backend=gs.cpu, precision="32", logging_level="warning")


def _ensure_genesis_initialized():
    """Re-initialize Genesis if a prior test module called gs.destroy()."""
    gs_mod = sys.modules.get("genesis")
    if gs_mod is not None and getattr(gs_mod, "_initialized", False):
        return

    import os

    _on_ci = bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"))
    if _on_ci:
        gs.init(backend=gs.cpu, precision="32", logging_level="warning")
    else:
        try:
            gs.init(backend=gs.gpu, precision="32", logging_level="warning")
        except Exception:
            gs.init(backend=gs.cpu, precision="32", logging_level="warning")


@pytest.fixture(scope="module", autouse=True)
def _reinit_genesis_if_destroyed():
    """Ensure Genesis is initialized before each test module."""
    _ensure_genesis_initialized()
    yield


def pytest_addoption(parser):
    """Add command-line options for pytest."""
    parser.addoption(
        "--visualize",
        action="store_true",
        default=False,
        help="Enable visualization for tests",
    )
