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
    "ArtVIPPart",
    "ArtVIPPartInfo",
    "list_artvip_categories",
    "list_artvip_objects",
    "download_artvip_object",
    "load_artvip_object",
    "parse_artvip_joint_info",
    "parse_artvip_part_info",
    "parse_artvip_control_script",
    "merge_fixed_meshes",
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


@dataclass(frozen=True)
class ArtVIPPart:
    """Semantic metadata for one part of an ArtVIP articulated object.

    ArtVIP USD files embed semantic labels via ``semantics:labels:class``
    attributes on link prims (e.g. ``"lid"``, ``"pedal"``, ``"handle"``,
    ``"door"``).  These labels identify functional parts that a robot would
    interact with, following the 35-category annotation scheme described in
    the ArtVIP paper (Table 5).

    Attributes
    ----------
    prim_path : str
        USD prim path of the labeled part (e.g. ``"/root/E_lid_1"``).
    name : str
        Prim name (e.g. ``"E_lid_1"``).
    label : str
        Semantic label from ``semantics:labels:class``
        (e.g. ``"lid"``, ``"pedal"``, ``"handle"``).
    is_link : bool
        True if this part is a top-level link (direct child of the
        articulation root).  Sub-parts (e.g. a handle nested inside a
        door link) have ``is_link=False``.
    parent_link : str
        Prim path of the top-level link that contains this part.
        For top-level links, this is the part's own path.
    n_meshes : int
        Number of mesh prims under this part.
    n_vertices : int
        Total vertex count across all meshes under this part.
    """

    prim_path: str
    name: str
    label: str
    is_link: bool
    parent_link: str
    n_meshes: int
    n_vertices: int


@dataclass(frozen=True)
class ArtVIPPartInfo:
    """Parsed part-level semantic metadata from an ArtVIP USD file.

    Attributes
    ----------
    usd_path : str
        Path to the USD file that was parsed.
    object_label : str
        Semantic label of the object itself (e.g. ``"trash_can"``,
        ``"dishwasher"``).
    parts : list[ArtVIPPart]
        All labeled parts in the object.
    labels : list[str]
        Sorted unique semantic labels across all parts.
    """

    usd_path: str
    object_label: str
    parts: list[ArtVIPPart] = field(default_factory=list)

    @property
    def labels(self) -> list[str]:
        return sorted({p.label for p in self.parts})

    def get_parts_by_label(self, label: str) -> list[ArtVIPPart]:
        """Return all parts with the given semantic label."""
        return [p for p in self.parts if p.label == label]

    def get_graspable_parts(self) -> list[ArtVIPPart]:
        """Return parts with labels commonly used for robotic grasping.

        These are parts like handles, lids, knobs, and buttons that a
        robot gripper would interact with, as opposed to passive parts
        like bodies or shelves.
        """
        graspable_labels = {
            "handle", "lid", "knob", "button", "door", "drawer",
            "pedal", "ball handle", "front cover",
        }
        return [p for p in self.parts if p.label in graspable_labels]


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
# Part-level semantic annotation parsing
# ---------------------------------------------------------------------------

def parse_artvip_part_info(usd_path: str | Path) -> ArtVIPPartInfo:
    """Parse part-level semantic annotations from an ArtVIP USD file.

    ArtVIP USD files embed semantic labels via ``semantics:labels:class``
    attributes on prims.  These labels follow a 35-category annotation
    scheme (Table 5 of the ArtVIP paper) that identifies functional parts
    such as ``"lid"``, ``"pedal"``, ``"handle"``, ``"door"``, ``"drawer"``,
    ``"shelf"``, ``"seat"``, ``"backrest"``, ``"wheel"``, etc.

    This function extracts all semantic labels, along with mesh counts and
    vertex counts per part, and identifies which parts are top-level
    articulation links vs. nested sub-parts.

    Parameters
    ----------
    usd_path : str | Path
        Path to the ArtVIP ``model_*.usd`` file.

    Returns
    -------
    ArtVIPPartInfo
        Parsed part metadata including the object-level label, all part
        labels, and convenience methods for filtering by label or
        identifying graspable parts.

    Example
    -------
        from hsr_genesis.artvip_loader import download_artvip_object, parse_artvip_part_info

        usd_path = download_artvip_object("household_items", "trash_can/stepping_dustbin_4")
        part_info = parse_artvip_part_info(usd_path)

        print(f"Object: {part_info.object_label}")
        for part in part_info.parts:
            print(f"  {part.name}: label={part.label}, meshes={part.n_meshes}, verts={part.n_vertices}")

        # Find graspable parts
        for part in part_info.get_graspable_parts():
            print(f"  Graspable: {part.name} ({part.label})")
    """
    from pxr import Usd, UsdGeom

    usd_path = Path(usd_path)
    stage = Usd.Stage.Open(str(usd_path))

    root = stage.GetDefaultPrim()
    if not root or not root.IsValid():
        root = stage.GetPseudoRoot()

    # Object-level label on the root prim.
    object_label = ""
    root_sem_attr = root.GetAttribute("semantics:labels:class")
    if root_sem_attr and root_sem_attr.Get():
        labels = list(root_sem_attr.Get())
        object_label = labels[0] if labels else ""

    # Identify top-level links (direct children of root that are Xforms).
    root_children_paths = {str(c.GetPath()) for c in root.GetChildren()}

    parts: list[ArtVIPPart] = []
    for prim in stage.Traverse():
        sem_attr = prim.GetAttribute("semantics:labels:class")
        if not sem_attr or not sem_attr.Get():
            continue

        labels_list = list(sem_attr.Get())
        if not labels_list:
            continue
        label = labels_list[0]

        prim_path = str(prim.GetPath())
        name = prim.GetName()

        # Skip the root prim (already captured as object_label).
        if prim_path == str(root.GetPath()):
            continue

        # Determine if this is a top-level link.
        is_link = prim_path in root_children_paths

        # Find the parent link (walk up to a root child).
        parent_link = prim_path
        parent = prim.GetParent()
        while parent and parent.IsValid():
            if str(parent.GetPath()) in root_children_paths:
                parent_link = str(parent.GetPath())
                break
            parent = parent.GetParent()

        # Count meshes and vertices under this part.
        n_meshes = 0
        n_vertices = 0
        for descendant in Usd.PrimRange(prim):
            if descendant.IsA(UsdGeom.Mesh) and descendant.IsActive():
                n_meshes += 1
                pts = UsdGeom.Mesh(descendant).GetPointsAttr().Get()
                if pts:
                    n_vertices += len(pts)

        parts.append(ArtVIPPart(
            prim_path=prim_path,
            name=name,
            label=label,
            is_link=is_link,
            parent_link=parent_link,
            n_meshes=n_meshes,
            n_vertices=n_vertices,
        ))

    return ArtVIPPartInfo(
        usd_path=str(usd_path),
        object_label=object_label,
        parts=parts,
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
# Mesh merging (unify meshes connected by fixed joints)
# ---------------------------------------------------------------------------

def merge_fixed_meshes(
    usd_path: str | Path,
    output_path: str | Path | None = None,
    *,
    decimate_face_num: int | None = None,
) -> Path:
    """Merge meshes connected by fixed joints into a single mesh per link.

    ArtVIP USD objects often have multiple mesh prims per link (e.g. a body
    link with separate meshes for the main body, handle, and group).  Genesis
    creates one collision geom per mesh prim, so 6 meshes → 6 geoms, increasing
    broad-phase collision pair count and slowing down physics.

    This function identifies rigid groups — sets of meshes that belong to the
    same link (connected by fixed joints or no joints at all, just nested
    Xforms) — and merges them into a single mesh per link.  Meshes on links
    connected by revolute or prismatic joints are kept separate.

    The merged USD preserves:
      - Joint prims (revolute, prismatic, fixed) and their properties
      - Link hierarchy and transforms
      - Material bindings (on the merged mesh)

    Parameters
    ----------
    usd_path : str | Path
        Path to the input ArtVIP ``model_*.usd`` file.
    output_path : str | Path, optional
        Path for the output USD.  If None, writes alongside the input with
        a ``_merged`` suffix (e.g. ``model_foo_merged.usd``).
    decimate_face_num : int, optional
        If given, decimate each merged mesh to this face count using
        ``trimesh``.  This is a simple uniform decimation — Genesis's own
        decimation (``decimate=True`` on ``gs.morphs.USD``) still applies
        on top.

    Returns
    -------
    Path
        Path to the merged USD file.

    Example
    -------
        from hsr_genesis.artvip_loader import download_artvip_object, merge_fixed_meshes

        usd_path = download_artvip_object("household_items", "trash_can/stepping_dustbin_4")
        merged_path = merge_fixed_meshes(usd_path)
        # merged_path has 3 meshes (one per link) instead of 6
    """
    from pxr import Usd, UsdGeom, UsdPhysics, Gf, Vt
    import shutil

    usd_path = Path(usd_path)
    if output_path is None:
        output_path = usd_path.with_name(
            usd_path.stem + "_merged" + usd_path.suffix
        )
    output_path = Path(output_path)

    # Copy the entire object directory to preserve sublayer references.
    # The merged USD replaces only the main model file; resource/ stays intact.
    if usd_path.parent != output_path.parent:
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        # Copy resource/ if it exists.
        resource = usd_path.parent / "resource"
        if resource.exists():
            dest_resource = output_dir / "resource"
            if not dest_resource.exists():
                shutil.copytree(resource, dest_resource)

    stage = Usd.Stage.Open(str(usd_path))

    # 1. Find all links (E_ prefixed Xforms that are direct children of root).
    root = stage.GetDefaultPrim()
    if not root or not root.IsValid():
        root = stage.GetPseudoRoot()

    links: dict[str, list] = {}  # link_path -> [(mesh_prim, world_to_link_matrix)]
    for link_prim in root.GetChildren():
        if not link_prim.IsA(UsdGeom.Xform):
            continue
        link_path = str(link_prim.GetPath())
        link_world = UsdGeom.Xformable(link_prim).ComputeLocalToWorldTransform(0.0)
        link_inv = link_world.GetInverse()

        # Recursively collect all meshes under this link (but not under
        # child links connected by movable joints).
        meshes = _collect_meshes_in_link(link_prim, link_inv)
        if meshes:
            links[link_path] = meshes

    # 2. For each link, merge all meshes into one.
    # Build a new stage that references the original but overrides mesh prims.
    # Strategy: edit the stage in-place (remove extra meshes, replace first
    # with merged geometry), then save to output_path.
    for link_path, mesh_list in links.items():
        if len(mesh_list) <= 1:
            continue

        # Merge vertex arrays, applying per-mesh transforms to bring vertices
        # into the link's local frame.
        all_points: list[Gf.Vec3f] = []
        all_face_counts: list[int] = []
        all_face_indices: list[int] = []
        vert_offset = 0

        for mesh_prim, to_link_matrix in mesh_list:
            mesh = UsdGeom.Mesh(mesh_prim)
            points = mesh.GetPointsAttr().Get()
            face_counts = mesh.GetFaceVertexCountsAttr().Get()
            face_indices = mesh.GetFaceVertexIndicesAttr().Get()

            if not points or not face_counts:
                continue

            # Transform points to link frame.
            for p in points:
                p_world = to_link_matrix.Transform(Gf.Vec3d(p))
                all_points.append(Gf.Vec3f(p_world))

            all_face_counts.extend(int(fc) for fc in face_counts)
            all_face_indices.extend(int(fi) + vert_offset for fi in face_indices)
            vert_offset += len(points)

        if not all_points:
            continue

        # Optional decimation.
        if decimate_face_num is not None and len(all_face_counts) > decimate_face_num:
            all_points, all_face_counts, all_face_indices = _decimate_mesh(
                all_points, all_face_counts, all_face_indices, decimate_face_num,
            )

        # Use the first mesh prim as the target; deactivate the rest.
        first_mesh_prim, _ = mesh_list[0]
        first_mesh = UsdGeom.Mesh(first_mesh_prim)

        # Set merged geometry on the first mesh.
        first_mesh.GetPointsAttr().Set(Vt.Vec3fArray(all_points))
        first_mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(all_face_counts))
        first_mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(all_face_indices))

        # Deactivate the extra mesh prims (they may be defined in sublayer
        # files, so RemovePrim doesn't work — use SetActive(False) instead).
        for mesh_prim, _ in mesh_list[1:]:
            mesh_prim.SetActive(False)

    # Export the modified stage to the output path.
    # We use Export (not Save) to avoid modifying the original file.
    stage.Export(str(output_path))

    return output_path


def _collect_meshes_in_link(
    link_prim,
    link_inv_matrix,
    *,
    _depth: int = 0,
) -> list:
    """Recursively collect all Mesh prims under a link, excluding child links.

    A "child link" is an Xform that has a PhysicsJoint (revolute, prismatic, or
    fixed connecting it to a different body).  We stop recursion at child links
    because those meshes belong to a different rigid body.

    Returns a list of (mesh_prim, matrix_to_link_frame) tuples.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    result = []
    for child in link_prim.GetChildren():
        # Skip joint prims.
        if child.IsA(UsdPhysics.Joint):
            continue

        # Check if this child is a separate link (has joints connecting it
        # to other bodies).
        has_movable_joint = False
        for grandchild in child.GetChildren():
            if grandchild.IsA(UsdPhysics.RevoluteJoint) or grandchild.IsA(UsdPhysics.PrismaticJoint):
                has_movable_joint = True
                break

        if has_movable_joint:
            continue  # This is a separate articulated link; skip its meshes.

        if child.IsA(UsdGeom.Mesh):
            # Compute this mesh's world transform, then transform to link frame.
            mesh_world = UsdGeom.Xformable(child).ComputeLocalToWorldTransform(0.0)
            to_link = link_inv_matrix * mesh_world
            result.append((child, to_link))
        elif child.IsA(UsdGeom.Xform):
            # Recurse into nested Xforms (e.g. E_Group_3, E_handle_4).
            result.extend(
                _collect_meshes_in_link(child, link_inv_matrix, _depth=_depth + 1)
            )

    return result


def _decimate_mesh(points, face_counts, face_indices, target_face_num):
    """Simple mesh decimation using trimesh.

    Converts the mesh to trimesh format, decimates, and returns the simplified
    geometry.  Falls back to the original if trimesh is unavailable.
    """
    try:
        import trimesh
    except ImportError:
        return points, face_counts, face_indices

    # Build trimesh from the mesh data.
    # First, triangulate non-triangle faces.
    verts = np.array([[p[0], p[1], p[2]] for p in points], dtype=np.float64)

    # Convert face_counts/face_indices to triangle faces.
    tri_faces = []
    idx = 0
    for fc in face_counts:
        face = face_indices[idx:idx + fc]
        idx += fc
        # Fan triangulation.
        for i in range(1, fc - 1):
            tri_faces.append([face[0], face[i], face[i + 1]])

    if not tri_faces:
        return points, face_counts, face_indices

    faces = np.array(tri_faces, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # Decimate.
    try:
        mesh = mesh.simplify_quadric_decimation(target_face_num)
    except Exception:
        try:
            mesh = mesh.simplify_faces_count(target_face_num)
        except Exception:
            return points, face_counts, face_indices

    # Convert back.
    from pxr import Gf
    new_points = [Gf.Vec3f(p[0], p[1], p[2]) for p in mesh.vertices]
    new_face_counts = [3] * len(mesh.faces)
    new_face_indices = [int(fi) for fi in mesh.faces.flatten()]

    return new_points, new_face_counts, new_face_indices


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
    merge_meshes: bool = False,
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
    merge_meshes : bool
        If True, merge meshes connected by fixed joints into a single mesh
        per link before loading.  This reduces the collision geom count
        (e.g. 6→3 for stepping_dustbin_4), which can improve physics
        throughput by reducing broad-phase collision pair count.
        Default False.
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

    if merge_meshes:
        usd_path = merge_fixed_meshes(usd_path)

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
