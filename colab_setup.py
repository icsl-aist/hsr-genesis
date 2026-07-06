"""Standalone Colab bootstrap (zero heavy deps, no ``hsr_genesis`` import).

On a fresh Colab runtime, ``genesis-world`` / ``torch`` / ``numpy`` are not
installed yet and neither is the ``hsr-genesis`` package itself.  This file
lives at the **repo root** (not under ``src/hsr_genesis/``) so it can be
fetched and executed by a notebook's **very first cell** — before the
package has been installed.

Usage in a notebook first-cell (fetch from raw GitHub)::

    import urllib.request
    url = "https://raw.githubusercontent.com/icsl-aist/hsr-genesis/main/colab_setup.py"
    exec(urllib.request.urlopen(url).read())
    setup_colab()

After that cell completes, ``import hsr_genesis`` and the rest will work.

If the repo is already cloned (e.g. local Jupyter), you can also import it
by path::

    import importlib.util
    spec = importlib.util.spec_from_file_location("colab_setup", "colab_setup.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.setup_colab()
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _log(msg: str, level: str = "INFO") -> None:
    """Print a tagged log line so users can trace setup progress."""
    print(f"[setup:{level}] {msg}")


# ---------------------------------------------------------------------------
# Colab setup (install, clone, EGL)
# ---------------------------------------------------------------------------


def setup_colab(
    *,
    repo_url: str = "https://github.com/icsl-aist/hsr-genesis.git",
    repo_dir: str = "/content/hsr-genesis",
    genesis_version: str = "0.4.6",
    force_reinstall: bool = False,
) -> None:
    """One-call Colab setup: install deps, clone repo, configure EGL.

    Idempotent — safe to re-run.  See ``src/hsr_genesis/colab_bootstrap.py``
    for the package-relative equivalent (identical logic).
    """
    import importlib

    # --- Step 1: pip install dependencies ---
    _log("Checking Python packages...")

    packages_to_check = {
        "genesis": "genesis-world",
        "mediapy": "mediapy",
    }
    need_install = force_reinstall
    for mod, pkg in packages_to_check.items():
        try:
            importlib.import_module(mod)
            _log(f"  {pkg} already installed", "OK")
        except ImportError:
            _log(f"  {pkg} not found, will install", "WARN")
            need_install = True

    try:
        import setuptools

        if int(setuptools.__version__.split(".")[0]) >= 82:
            _log("  setuptools >=82 detected, pinning <82 for torch compat", "WARN")
            need_install = True
        else:
            _log("  setuptools version OK", "OK")
    except Exception:
        pass

    if need_install:
        _log("Installing dependencies (this may take ~2 min)...")
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "setuptools<82",
            "jedi",
            f"genesis-world=={genesis_version}",
            "mediapy",
            "-q",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            _log(f"pip install failed (exit {result.returncode})", "ERROR")
            if result.stderr:
                print("  stderr:", result.stderr[-500:])
            _log("See the troubleshoot notebook for help.", "ERROR")
            raise RuntimeError("pip install failed")
        _log("Dependencies installed successfully.", "OK")
    else:
        _log("All dependencies already satisfied.", "OK")

    # --- Step 2: clone repo with submodules ---
    _log(f"Checking repo at {repo_dir}...")
    repo_path = Path(repo_dir)

    if repo_path.exists() and (repo_path / ".git").exists():
        _log("  Repo already cloned.", "OK")
        meshes_dir = repo_path / "data" / "urdf" / "hsrb_meshes"
        if not meshes_dir.exists() or not any(meshes_dir.iterdir() if meshes_dir.exists() else []):
            _log("  Submodules missing, fetching them...", "WARN")
            result = subprocess.run(
                ["git", "-C", str(repo_path), "submodule", "update", "--init", "--recursive", "--depth", "1"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                _log(f"  submodule update failed: {result.stderr[-300:]}", "ERROR")
                raise RuntimeError("git submodule update failed")
            _log("  Submodules fetched.", "OK")
        else:
            _log("  Submodules present.", "OK")
    else:
        _log(f"  Cloning {repo_url} (with submodules)...", "INFO")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "--recurse-submodules", "--shallow-submodules", repo_url, str(repo_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            _log(f"  git clone failed: {result.stderr[-300:]}", "ERROR")
            _log("  Check your internet connection or the repo URL.", "ERROR")
            raise RuntimeError("git clone failed")
        _log("  Repo cloned successfully.", "OK")

    # --- Step 3: install repo as editable package ---
    _log("Installing hsr-genesis (editable)...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(repo_path), "-q"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        _log(f"  pip install -e failed: {result.stderr[-300:]}", "WARN")
        _log("  Falling back to PYTHONPATH approach...", "WARN")
    else:
        _log("  hsr-genesis installed.", "OK")

    src_dir = str(repo_path / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
        _log(f"  Added {src_dir} to sys.path", "OK")

    # --- Step 4: verify import ---
    _log("Verifying hsr_genesis import...")
    try:
        import hsr_genesis  # noqa: F811

        _log(f"  hsr_genesis loaded from {hsr_genesis.__file__}", "OK")
    except ImportError as e:
        _log(f"  import hsr_genesis failed: {e}", "ERROR")
        _log("  Try: Runtime \u2192 Restart runtime, then re-run this cell.", "ERROR")
        raise

    urdf = repo_path / "data" / "urdf" / "hsrb4s.urdf"
    meshes = repo_path / "data" / "urdf" / "hsrb_meshes"
    if not urdf.exists():
        _log(f"  URDF not found at {urdf}", "ERROR")
        raise FileNotFoundError(f"URDF not found: {urdf}")
    _log(f"  URDF found: {urdf}", "OK")

    if meshes.exists() and any(meshes.iterdir()):
        _log(f"  Meshes found: {meshes}", "OK")
    else:
        _log(f"  Meshes directory empty or missing: {meshes}", "ERROR")
        _log("  Run: !git -C /content/hsr-genesis submodule update --init --recursive", "ERROR")
        raise FileNotFoundError(f"Meshes not found: {meshes}")

    # --- Step 5: EGL ICD config ---
    _log("Configuring NVIDIA EGL ICD...")
    icd_path = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    icd_content = (
        '{\n    "file_format_version" : "1.0.0",\n'
        '    "ICD" : {\n'
        '        "library_path" : "libEGL_nvidia.so.0"\n'
        '    }\n}\n'
    )
    try:
        os.makedirs(os.path.dirname(icd_path), exist_ok=True)
        with open(icd_path, "w") as f:
            f.write(icd_content)
        _log(f"  EGL ICD config written to {icd_path}", "OK")
    except PermissionError:
        _log(f"  Cannot write {icd_path} (permission denied).", "WARN")
        _log("  GPU rendering may fail. See troubleshoot notebook.", "WARN")

    # --- Step 6: check GPU ---
    _log("Checking GPU availability...")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip()
            _log(f"  GPU detected: {gpu_name}", "OK")
        else:
            _log("  nvidia-smi returned no GPU.", "WARN")
            _log("  Go to Runtime \u2192 Change runtime type \u2192 T4 GPU", "WARN")
    except Exception:
        _log("  nvidia-smi not found. GPU rendering may not work.", "WARN")

    _log("Setup complete! You can now call init_sim().", "DONE")
