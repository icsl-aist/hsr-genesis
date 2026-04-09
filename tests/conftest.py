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


def pytest_addoption(parser):
    """Add command-line options for pytest."""
    parser.addoption(
        "--visualize",
        action="store_true",
        default=False,
        help="Enable visualization for tests",
    )
