"""ArtVIP dataset loader.

Loads articulated objects from the ArtVIP dataset
(``X-Humanoid/ArtVIP``, Apache-2.0, ICLR 2026).

ArtVIP provides 476 high-quality digital-twin articulated objects in USD
format, organized into 9 categories:

  - ``major_appliances``    (dishwasher, refrigerator, washing_machine, ...)
  - ``small_appliances``
  - ``large_furniture``
  - ``small_furniture``
  - ``household_items``
  - ``Ikea_furniture``
  - ``Medical_equipment``
  - ``industrial_machinery``
  - ``lab_items``

Each object directory contains:

  - ``model_<name>.usd``             -- main USD file (sublayers reference resource/)
  - ``resource/material.usd``        -- materials
  - ``resource/v_0/link_default_0.usd`` -- mesh geometry
  - ``resource/<name>_control.py``   -- Isaac Sim control script (joint metadata)

The USD files use relative sublayer paths, so the entire object directory must
be downloaded together (not just the main ``.usd``).

This module uses ``huggingface_hub`` for on-demand per-object downloads and
``pxr.Usd`` (via ``usd-core``) for joint introspection.  Genesis's built-in
``gs.morphs.USD`` handles the actual entity creation.

Usage
-----
    from hsr_genesis.artvip_loader import (
        list_artvip_categories,
        list_artvip_objects,
        download_artvip_object,
        load_artvip_object,
        parse_artvip_joint_info,
    )

    # List categories
    print(list_artvip_categories())

    # List objects in a category
    print(list_artvip_objects("major_appliances"))

    # Download + load as gs.morphs.USD
    morph = load_artvip_object(
        category="major_appliances",
        object_name="dishwasher_1",
        pos=[0.5, 0.0, 0.0],
    )

    # Or just download (returns the local path to model_*.usd)
    usd_path = download_artvip_object("major_appliances", "dishwasher_1")

    # Parse joint info from the USD
    joint_info = parse_artvip_joint_info(usd_path)
    for j in joint_info.joints:
        print(f"  {j.name}: {j.joint_type}, limits={j.limits}")
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

__all__ = [
    "ARTVIP_REPO_ID",
    "ARTVIP_CATEGORIES",
    "ArtVIPJointInfo",
    "ArtVIPJoint",
    "list_artvip_categories",
    "list_artvip_objects",
    "download_artvip_object",
    "load_artvip_object",
    "parse_artvip_joint_info",
    "parse_artvip_control_script",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: HuggingFace dataset repository ID.
ARTVIP_REPO_ID: str = "X-Humanoid/ArtVIP"

#: All articulated-object categories in ArtVIP.
ARTVIP_CATEGORIES: tuple[str, ...] = (
    "Ikea_furniture",
    "Medical_equipment",
    "household_items",
    "industrial_machinery",
    "lab_items",
    "large_furniture",
    "major_appliances",
    "small_appliances",
    "small_furniture",
)

#: Default cache directory for ArtVIP downloads.
_DEFAULT_CACHE_DIR: Path = Path(
    os.environ.get("ARTVIP_CACHE_DIR", str(Path.home() / ".cache" / "artvip"))
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArtVIPJoint:
    """Metadata for one joint in an ArtVIP USD articulation.

    Attributes
    ----------
    name : str
        Joint name from the USD prim (e.g. ``"RevoluteJoint_dishwasher_1_middle"``).
    joint_type : str
        ``"revolute"``, ``"prismatic"``, or ``"fixed"``.
    prim_path : str
        USD prim path of the joint.
    limits : tuple[float, float]
        ``(lower, upper)`` -- radians for revolute, meters for prismatic.
        ``(0.0, 0.0)`` for fixed joints.
    axis : np.ndarray
        Joint axis in the joint frame (shape ``(3,)``).
    body0 : str
        Prim path of the first body connected by the joint.
    body1 : str
        Prim path of the second body connected by the joint.
    """

    name: str
    joint_type: str
    prim_path: str
    limits: tuple[float, float]
    axis: np.ndarray
    body0: str
    body1: str


@dataclass(frozen=True)
class ArtVIPJointInfo:
    """Parsed joint information from an ArtVIP USD file.

    Attributes
    ----------
    usd_path : str
        Path to the USD file that was parsed.
    articulation_root : str
        Prim path of the articulation root (``ArticulationRootAPI``).
    joints : list[ArtVIPJoint]
        All joints in the articulation (including fixed).
    movable_joints : list[ArtVIPJoint]
        Subset of joints that are revolute or prismatic.
    up_axis : str
        Stage up axis (``"Z"`` or ``"Y"``).
    meters_per_unit : float
        Stage meters-per-unit scale.
    """

    usd_path: str
    articulation_root: str
    joints: list[ArtVIPJoint] = field(default_factory=list)
    movable_joints: list[ArtVIPJoint] = field(default_factory=list)
    up_axis: str = "Z"
    meters_per_unit: float = 1.0


# ---------------------------------------------------------------------------
# HuggingFace download
# ---------------------------------------------------------------------------

def list_artvip_categories() -> list[str]:
    """Return the list of ArtVIP articulated-object categories."""
    return list(ARTVIP_CATEGORIES)


def list_artvip_objects(
    category: str,
    *,
    object_type: Optional[str] = None,
    cache_dir: Optional[str | Path] = None,
) -> list[str]:
    """List object names within an ArtVIP category.

    ArtVIP uses a three-level hierarchy: ``category/object_type/object_instance``.
    For example, ``major_appliances/dishwasher/dishwasher_1``.

    If the category has already been downloaded to ``cache_dir``, the local
    directory is scanned.  Otherwise, the HuggingFace Hub API is queried.

    Parameters
    ----------
    category : str
        Category name (e.g. ``"major_appliances"``).
    object_type : str, optional
        Object type within the category (e.g. ``"dishwasher"``).  If None,
        all object types are scanned and the full ``type/instance`` paths
        are returned.
    cache_dir : str | Path, optional
        Local cache directory.  Defaults to ``~/.cache/artvip`` or the
        ``ARTVIP_CACHE_DIR`` environment variable.

    Returns
    -------
    list[str]
        Sorted list of object paths.  If ``object_type`` is given, returns
        just the instance names (e.g. ``["dishwasher_1", "dishwasher_2"]``).
        If ``object_type`` is None, returns ``type/instance`` paths
        (e.g. ``["dishwasher/dishwasher_1", "dishwasher/dishwasher_2"]``).
    """
    cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
    base = f"Articulated_objects/{category}"

    # Try local cache first.
    local = _find_snapshot_dir(cache_dir, base)
    if local is not None:
        if object_type is not None:
            type_dir = local / object_type
            if not type_dir.exists():
                return []
            return sorted(
                item.name for item in type_dir.iterdir()
                if item.is_dir() and not item.name.startswith(".")
            )
        else:
            objects = []
            for type_dir in local.iterdir():
                if not type_dir.is_dir() or type_dir.name.startswith("."):
                    continue
                for instance_dir in type_dir.iterdir():
                    if instance_dir.is_dir() and not instance_dir.name.startswith("."):
                        objects.append(f"{type_dir.name}/{instance_dir.name}")
            return sorted(objects)

    # Fall back to HF Hub API.
    from huggingface_hub import list_repo_tree

    if object_type is not None:
        api_base = f"{base}/{object_type}"
        entries = list_repo_tree(
            ARTVIP_REPO_ID,
            path_in_repo=api_base,
            repo_type="dataset",
            recursive=False,
        )
        objects = []
        for entry in entries:
            if entry.path.startswith(api_base + "/"):
                name = entry.path[len(api_base) + 1:]
                name = name.split("/")[0]
                if name and not name.startswith("."):
                    objects.append(name)
        return sorted(set(objects))
    else:
        # List all object types, then list instances in each.
        entries = list_repo_tree(
            ARTVIP_REPO_ID,
            path_in_repo=base,
            repo_type="dataset",
            recursive=False,
        )
        object_types = []
        for entry in entries:
            if entry.path.startswith(base + "/"):
                name = entry.path[len(base) + 1:]
                name = name.split("/")[0]
                if name and not name.startswith("."):
                    object_types.append(name)
        object_types = sorted(set(object_types))

        objects = []
        for otype in object_types:
            instances = list_artvip_objects(
                category, object_type=otype, cache_dir=cache_dir,
            )
            for inst in instances:
                objects.append(f"{otype}/{inst}")
        return sorted(objects)


def download_artvip_object(
    category: str,
    object_name: str,
    *,
    cache_dir: Optional[str | Path] = None,
    token: Optional[str] = None,
) -> Path:
    """Download a single ArtVIP object directory from HuggingFace.

    ArtVIP USD files use relative sublayer paths (e.g.
    ``./resource/material.usd``), so the entire object directory must be
    downloaded together for the USD to load correctly.

    Parameters
    ----------
    category : str
        Category name (e.g. ``"major_appliances"``).
    object_name : str
        Object path within the category.  This can be either:
        - ``"dishwasher_1"`` (instance name only, requires ``object_type``)
        - ``"dishwasher/dishwasher_1"`` (full ``type/instance`` path)
    cache_dir : str | Path, optional
        Local cache directory.  Defaults to ``~/.cache/artvip`` or the
        ``ARTVIP_CACHE_DIR`` environment variable.
    token : str, optional
        HuggingFace token for authenticated downloads (higher rate limits).

    Returns
    -------
    Path
        Path to the downloaded ``model_<name>.usd`` file.
    """
    from huggingface_hub import snapshot_download

    cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
    base = f"Articulated_objects/{category}/{object_name}"

    snapshot_path = snapshot_download(
        repo_id=ARTVIP_REPO_ID,
        allow_patterns=f"{base}/*",
        repo_type="dataset",
        cache_dir=str(cache_dir),
        token=token,
    )

    # Find the main model USD file.
    snapshot_dir = Path(snapshot_path)
    model_usd = _find_model_usd(snapshot_dir, base)
    if model_usd is None:
        raise FileNotFoundError(
            f"No model_*.usd found in downloaded ArtVIP object "
            f"{category}/{object_name} at {snapshot_dir}"
        )
    return model_usd


def _find_snapshot_dir(cache_dir: Path, base: str) -> Optional[Path]:
    """Find a previously-downloaded snapshot directory in the HF cache."""
    repo_cache = cache_dir / "datasets--X-Humanoid--ArtVIP"
    if not repo_cache.exists():
        return None
    snapshots = repo_cache / "snapshots"
    if not snapshots.exists():
        return None
    for snapshot in snapshots.iterdir():
        candidate = snapshot / base
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _find_model_usd(snapshot_dir: Path, base: str) -> Optional[Path]:
    """Find the main USD file in a downloaded ArtVIP object directory.

    ArtVIP objects use one of two naming conventions:
      - ``model_<name>.usd`` (most objects)
      - ``<Name>_<n>.usd`` (some small_furniture / household_items objects)

    This function prefers ``model_*.usd`` and falls back to any top-level
    ``.usd`` file that is not a sublayer reference (e.g. ``material.usd``).
    """
    target_dir = snapshot_dir / base
    if not target_dir.exists():
        return None

    usd_suffixes = (".usd", ".usda", ".usdc", ".usdz")

    # 1. Prefer model_*.usd at the top level.
    for p in sorted(target_dir.iterdir()):
        if p.is_file() and p.name.startswith("model_") and p.suffix in usd_suffixes:
            return p

    # 2. Fall back to any top-level .usd file, excluding known sublayer names.
    excluded = {"material.usd", "material.usda"}
    candidates = []
    for p in sorted(target_dir.iterdir()):
        if not p.is_file() or p.suffix not in usd_suffixes:
            continue
        if p.name.lower() in excluded:
            continue
        candidates.append(p)
    if candidates:
        return candidates[0]

    return None


# ---------------------------------------------------------------------------
# USD joint parsing
# ---------------------------------------------------------------------------

def parse_artvip_joint_info(usd_path: str | Path) -> ArtVIPJointInfo:
    """Parse joint information from an ArtVIP USD file.

    Uses ``pxr.Usd`` to introspect the USD stage and extract:
      - Articulation root prim path
      - All joints (revolute, prismatic, fixed) with limits and axes
      - Stage up-axis and meters-per-unit

    Parameters
    ----------
    usd_path : str | Path
        Path to the ArtVIP ``model_*.usd`` file.

    Returns
    -------
    ArtVIPJointInfo
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    usd_path = Path(usd_path)
    stage = Usd.Stage.Open(str(usd_path))

    up_axis = str(UsdGeom.GetStageUpAxis(stage))
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)

    # Find articulation root.
    articulation_root = ""
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_root = prim.GetPath().pathString
            break

    # Parse all joints.
    joints: list[ArtVIPJoint] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue

        joint_name = prim.GetName()
        prim_path = prim.GetPath().pathString

        # Determine joint type.
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joint_type = "revolute"
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            joint_type = "prismatic"
        elif prim.IsA(UsdPhysics.FixedJoint):
            joint_type = "fixed"
        else:
            # Generic joint -- skip unknown types.
            continue

        # Parse limits.
        limits = (0.0, 0.0)
        if joint_type == "revolute":
            rev = UsdPhysics.RevoluteJoint(prim)
            lower_attr = rev.GetLowerLimitAttr()
            upper_attr = rev.GetUpperLimitAttr()
            lower = lower_attr.Get() if lower_attr else 0.0
            upper = upper_attr.Get() if upper_attr else 0.0
            limits = (float(lower), float(upper))
        elif joint_type == "prismatic":
            pris = UsdPhysics.PrismaticJoint(prim)
            lower_attr = pris.GetLowerLimitAttr()
            upper_attr = pris.GetUpperLimitAttr()
            lower = lower_attr.Get() if lower_attr else 0.0
            upper = upper_attr.Get() if upper_attr else 0.0
            limits = (float(lower), float(upper))

        # Parse axis.
        # USD physics:axis is a token ("X", "Y", or "Z"), not a vector.
        axis = np.array([1.0, 0.0, 0.0])
        axis_attr = prim.GetAttribute("physics:axis")
        if axis_attr:
            val = axis_attr.Get()
            if val is not None:
                axis_token = str(val).upper()
                if axis_token == "X":
                    axis = np.array([1.0, 0.0, 0.0])
                elif axis_token == "Y":
                    axis = np.array([0.0, 1.0, 0.0])
                elif axis_token == "Z":
                    axis = np.array([0.0, 0.0, 1.0])

        # Parse connected bodies (stored as USD relationships, not attributes).
        body0 = ""
        body1 = ""
        body0_rel = prim.GetRelationship("physics:body0")
        body1_rel = prim.GetRelationship("physics:body1")
        if body0_rel:
            targets = body0_rel.GetTargets()
            if targets:
                body0 = str(targets[0])
        if body1_rel:
            targets = body1_rel.GetTargets()
            if targets:
                body1 = str(targets[0])

        joints.append(ArtVIPJoint(
            name=joint_name,
            joint_type=joint_type,
            prim_path=prim_path,
            limits=limits,
            axis=axis,
            body0=body0,
            body1=body1,
        ))

    movable = [j for j in joints if j.joint_type in ("revolute", "prismatic")]

    return ArtVIPJointInfo(
        usd_path=str(usd_path),
        articulation_root=articulation_root,
        joints=joints,
        movable_joints=movable,
        up_axis=up_axis,
        meters_per_unit=meters_per_unit,
    )


# ---------------------------------------------------------------------------
# Control script parsing (optional metadata source)
# ---------------------------------------------------------------------------

def parse_artvip_control_script(py_path: str | Path) -> dict[str, Any]:
    """Extract metadata from an ArtVIP Isaac Sim control script.

    ArtVIP objects ship with ``<name>_control.py`` scripts that contain
    constants like ``JOINT_NAMES``, ``JOINT_THRESHOLD``, and
    ``POSSIBLE_ASSET_ROOT_NAMES``.  These are useful for understanding the
    intended articulation behavior without loading the USD.

    This function uses regex parsing (not import) to avoid requiring Isaac
    Sim dependencies.

    Parameters
    ----------
    py_path : str | Path
        Path to the ``*_control.py`` file.

    Returns
    -------
    dict
        Keys: ``"joint_names"``, ``"joint_threshold"``, ``"asset_root_names"``.
        Missing values are omitted.
    """
    py_path = Path(py_path)
    text = py_path.read_text()

    result: dict[str, Any] = {}

    # JOINT_NAMES = ["joint1", "joint2", ...]
    m = re.search(r"JOINT_NAMES\s*=\s*\[([^\]]*)\]", text)
    if m:
        names = re.findall(r'["\']([^"\']+)["\']', m.group(1))
        result["joint_names"] = names

    # JOINT_THRESHOLD = 0.3
    m = re.search(r"JOINT_THRESHOLD\s*=\s*([0-9.]+)", text)
    if m:
        result["joint_threshold"] = float(m.group(1))

    # POSSIBLE_ASSET_ROOT_NAMES = ["name1", "name2"]
    m = re.search(r"POSSIBLE_ASSET_ROOT_NAMES\s*=\s*\[([^\]]*)\]", text)
    if m:
        names = re.findall(r'["\']([^"\']+)["\']', m.group(1))
        result["asset_root_names"] = names

    return result


# ---------------------------------------------------------------------------
# Genesis morph creation
# ---------------------------------------------------------------------------

def load_artvip_object(
    category: str,
    object_name: str,
    *,
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    euler: Optional[tuple[float, float, float]] = None,
    fixed: bool = True,
    cache_dir: Optional[str | Path] = None,
    token: Optional[str] = None,
    decimate: bool = True,
    convexify: bool = True,
    **usd_kwargs: Any,
):
    """Download (if needed) and load an ArtVIP object as ``gs.morphs.USD``.

    Parameters
    ----------
    category : str
        ArtVIP category (e.g. ``"major_appliances"``).
    object_name : str
        Object path within the category (e.g. ``"dishwasher/dishwasher_1"``).
    pos : tuple (3,)
        World-frame position.
    euler : tuple (3,), optional
        World-frame euler angles in degrees.
    fixed : bool
        Whether the base is fixed to the world.
    cache_dir : str | Path, optional
        Local cache directory for downloads.
    token : str, optional
        HuggingFace token for authenticated downloads.
    decimate : bool
        Whether to decimate meshes (default True, recommended for performance).
    convexify : bool
        Whether to convexify collision meshes (default True).
    **usd_kwargs
        Additional kwargs passed to ``gs.morphs.USD``.

    Returns
    -------
    gs.morphs.USD
        Morph ready for ``scene.add_entity(...)``.
    """
    import genesis as gs

    usd_path = download_artvip_object(
        category, object_name, cache_dir=cache_dir, token=token,
    )

    kwargs: dict[str, Any] = {
        "file": str(usd_path),
        "pos": tuple(pos),
        "fixed": fixed,
        "decimate": decimate,
        "convexify": convexify,
    }
    if euler is not None:
        kwargs["euler"] = tuple(euler)
    kwargs.update(usd_kwargs)
    return gs.morphs.USD(**kwargs)
