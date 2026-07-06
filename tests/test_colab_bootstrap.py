"""Tests: Colab bootstrap — standalone file vs package-relative import."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETUP_PY = PROJECT_ROOT / "colab_setup.py"


def _clean_env_subprocess(code: str) -> subprocess.CompletedProcess:
    """Run *code* in a subprocess that excludes this project from ``sys.path``.

    This simulates a fresh Colab runtime where ``hsr_genesis`` is **not**
    installed, so any attempt to import from the package should fail.
    """
    env = {
        "PYTHONPATH": "",  # wipe any inherited paths
        "PATH": "/usr/bin:/bin",
    }
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )


# ---------------------------------------------------------------------------
# RED tests — the current broken pattern
# ---------------------------------------------------------------------------


def test_package_import_fails_on_clean_env():
    """``from hsr_genesis.colab_bootstrap import setup_colab`` **must** fail
    when the package is not installed (fresh Colab runtime)."""
    code = (
        "import sys;"
        "sys.path = [p for p in sys.path if 'hsr_genesis' not in p and 'src' not in p];"
        "from hsr_genesis.colab_bootstrap import setup_colab;"  # should raise
        "print('UNEXPECTED_SUCCESS')"
    )
    proc = _clean_env_subprocess(code)
    # The import MUST fail — if it succeeds, the test fails (no regression)
    assert proc.returncode != 0, (
        "Importing hsr_genesis.colab_bootstrap succeeded when it should have "
        "failed. This means the package happened to be on sys.path — the test "
        "environment may need adjustment."
    )
    assert "ModuleNotFoundError" in proc.stderr or "ImportError" in proc.stderr, (
        f"Expected ModuleNotFoundError/ImportError, got:\nstdout:{proc.stdout}\nstderr:{proc.stderr}"
    )


# ---------------------------------------------------------------------------
# RED tests — the new standalone file must work
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not SETUP_PY.exists(),
    reason="colab_setup.py not yet created — this test drives its creation",
)
class TestStandaloneBootstrap:
    """Tests for the top-level ``colab_setup.py`` that a notebook's first
    cell can fetch / import **without** the ``hsr_genesis`` package."""

    def test_standalone_file_is_importable(self):
        """``colab_setup.py`` can be imported as a top-level module (by path
        or exec) without ``hsr_genesis`` being on ``sys.path``."""
        code = (
            "import sys;"
            "sys.path = [p for p in sys.path if 'hsr_genesis' not in p and 'src' not in p];"
            f"import importlib.util;"
            f"spec = importlib.util.spec_from_file_location('colab_setup', r'{SETUP_PY}');"
            f"mod = importlib.util.module_from_spec(spec);"
            f"spec.loader.exec_module(mod);"
            f"print('has_setup_colab:', 'setup_colab' in dir(mod))"
        )
        proc = _clean_env_subprocess(code)
        assert proc.returncode == 0, f"Import failed:\nstdout:{proc.stdout}\nstderr:{proc.stderr}"
        assert "has_setup_colab: True" in proc.stdout, f"setup_colab not found in module:\n{proc.stdout}"

    def test_standalone_setup_colab_signature(self):
        """``setup_colab()`` in the standalone file has the expected signature
        (same default parameters as ``hsr_genesis.colab_bootstrap.setup_colab``)."""
        code = (
            "import sys;"
            "sys.path = [p for p in sys.path if 'hsr_genesis' not in p and 'src' not in p];"
            f"import importlib.util;"
            f"spec = importlib.util.spec_from_file_location('colab_setup', r'{SETUP_PY}');"
            f"mod = importlib.util.module_from_spec(spec);"
            f"spec.loader.exec_module(mod);"
            f"import inspect;"
            f"sig = inspect.signature(mod.setup_colab);"
            f"print('params:', list(sig.parameters.keys()));"
            f"print('defaults:', [p.default for p in sig.parameters.values()]);"
        )
        proc = _clean_env_subprocess(code)
        assert proc.returncode == 0, f"Signature check failed:\n{proc.stderr}"
        assert "params:" in proc.stdout
        assert "repo_url" in proc.stdout
        assert "repo_dir" in proc.stdout
        assert "genesis_version" in proc.stdout

    def test_exec_file_pattern(self):
        """Simulate the notebook first-cell pattern: exec the file contents,
        then call ``setup_colab()`` *without* having imported ``hsr_genesis``
        first. (This test does **not** actually run setup — it only checks
        that the module loads and the function is callable.)"""
        code = (
            "import sys;"
            "sys.path = [p for p in sys.path if 'hsr_genesis' not in p and 'src' not in p];"
            f"import importlib.util;"
            f"spec = importlib.util.spec_from_file_location('colab_setup', r'{SETUP_PY}');"
            f"mod = importlib.util.module_from_spec(spec);"
            f"spec.loader.exec_module(mod);"
            f"assert callable(mod.setup_colab), 'setup_colab not callable';"
            f"print('OK_callable')"
        )
        proc = _clean_env_subprocess(code)
        assert proc.returncode == 0, f"Exec pattern failed:\n{proc.stderr}"
        assert "OK_callable" in proc.stdout
