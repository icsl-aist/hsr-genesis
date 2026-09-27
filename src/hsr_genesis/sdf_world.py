"""Gazebo SDF world (incl. xacro) -> Genesis scene importer.

The ``tmc_wrs_gazebo`` submodule ships the WRS2020 arena as a *world* file
(``worlds/wrs2020.world.xacro``): an SDF document that ``<include>``s many
single-model SDFs at fixed poses.  Genesis has no SDF morph, so this module

1. expands the xacro world into plain SDF (``xacro`` is a Genesis dependency),
2. collects every ``<include>`` (instance name, model, pose, static flag),
3. spawns the referenced models with :mod:`hsr_genesis.sdf_parser`.

``model://`` URIs are resolved against the ``models/`` directory next to the
world (the standard Gazebo package layout) unless ``models_root`` is given.
Models whose geometry is an SDF ``<plane>`` (e.g. ``wrc_ground_plane``) become
``gs.morphs.Plane``, since Genesis represents ground as an infinite plane and
:func:`~hsr_genesis.sdf_parser.sdf_to_urdf` rejects ``<plane>`` geometry; the
plane model's ``<material>`` becomes the plane surface.
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from trimesh.transformations import euler_from_matrix

from hsr_genesis.sdf_parser import (
    _parse_pose,
    _resolve_sdf_file,
    load_sdf_model,
    sdf_materials,
)

__all__ = [
    "SDFWorldModel",
    "SDFWorld",
    "parse_sdf_world",
    "spawn_sdf_world",
]

# xacro args the upstream CMake build passes when generating this world.
# ``trofast_knob`` is referenced by ``$(arg trofast_knob)`` but not declared
# with ``<xacro:arg>``, so xacro raises "Undefined substitution argument"
# unless it is supplied explicitly.  ``false`` selects the ``trofast`` model,
# matching the upstream default / ``_fast`` world variants.
DEFAULT_XACRO_ARGS = {"trofast_knob": "false"}


@dataclass
class SDFWorldModel:
    """One ``<include>``d model instance inside an SDF world."""

    name: str             # instance name (<name>, defaults to the model name)
    model_name: str       # model directory name (from the ``model://`` URI)
    model_dir: Path       # absolute model directory
    pose: np.ndarray      # 4x4 world pose from <pose>
    static: bool          # <static> flag (false when absent)
    uri: str              # raw <uri> text
    is_plane: bool = False  # model geometry is an SDF <plane> only


@dataclass
class SDFWorld:
    """Parsed SDF world: model instances plus world-level physics settings."""

    name: str
    source: Path
    models: list[SDFWorldModel]
    gravity: Optional[np.ndarray] = None
    max_step_size: Optional[float] = None
    models_root: Optional[Path] = None


# ---------------------------------------------------------------------------
# World parsing
# ---------------------------------------------------------------------------

def _world_element(world_path: Path,
                   xacro_args: Optional[dict[str, str]]) -> ET.Element:
    """Return the ``<world>`` element of a ``.world`` / ``.world.xacro`` file."""
    if world_path.suffix == ".xacro":
        try:
            import xacro
        except ImportError as exc:  # pragma: no cover - xacro ships with Genesis
            raise RuntimeError(
                f"{world_path.name} is a xacro world; install the 'xacro' package"
                " (a Genesis dependency) or pre-process the file with"
                " 'rosrun xacro xacro' and pass the generated .world."
            ) from exc
        mappings = dict(DEFAULT_XACRO_ARGS)
        if xacro_args:
            mappings.update(xacro_args)
        doc = xacro.process_file(str(world_path), mappings=mappings)
        root = ET.fromstring(doc.toxml())
    else:
        root = ET.parse(world_path).getroot()
    world = root if root.tag == "world" else root.find("world")
    if world is None:
        raise ValueError(f"No <world> element found in {world_path}")
    return world


def _auto_models_root(world_path: Path) -> Optional[Path]:
    """Locate the ``models/`` directory of the Gazebo package owning the world."""
    world_dir = world_path.resolve().parent
    for candidate in (world_dir.parent / "models", world_dir / "models"):
        if candidate.is_dir():
            return candidate
    return None


def _resolve_model_dir(uri: str, world_dir: Path,
                       models_root: Optional[Path]) -> tuple[Path, str]:
    """Resolve an ``<include>`` URI to ``(model_dir, model_name)``."""
    if uri.startswith("model://"):
        rest = uri[len("model://"):]
        model_name = rest.split("/")[0]
        if models_root is None:
            raise FileNotFoundError(
                f"Cannot resolve {uri!r}: no 'models' directory found next to the"
                " world file; pass models_root=<path to Gazebo models dir>."
            )
        return models_root / model_name, model_name
    path = uri[len("file://"):] if uri.startswith("file://") else uri
    model_dir = Path(path)
    if not model_dir.is_absolute():
        model_dir = world_dir / model_dir
    model_dir = model_dir.resolve()
    return model_dir, model_dir.name


def _is_plane_model(model_dir: Path) -> bool:
    """Whether a model's geometry is exclusively SDF ``<plane>``."""
    root = ET.parse(_resolve_sdf_file(model_dir)).getroot()
    geometries = root.findall(".//geometry")
    planes = [g for g in geometries if g.find("plane") is not None]
    if planes and len(planes) != len(geometries):
        raise NotImplementedError(
            f"{model_dir} mixes <plane> geometry with other geometry; only"
            " plane-only models are supported (spawn gs.morphs.Plane and the"
            " remaining geometry separately)."
        )
    return bool(planes)


def parse_sdf_world(world_path: str | os.PathLike, *,
                    models_root: Optional[str | os.PathLike] = None,
                    xacro_args: Optional[dict[str, str]] = None) -> SDFWorld:
    """Parse a Gazebo SDF world (``.world`` or ``.world.xacro``).

    Parameters
    ----------
    world_path : str | PathLike
        Path to the world file.  ``.xacro`` files are expanded on the fly
        with the ``xacro`` package.
    models_root : str | PathLike, optional
        Directory containing the ``model://`` model directories.  Defaults to
        the ``models/`` directory next to the world file.
    xacro_args : dict[str, str], optional
        xacro ``name:=value`` mappings.  Merged over
        :data:`DEFAULT_XACRO_ARGS`, which supplies the ``trofast_knob`` arg
        this world references without declaring.

    Returns
    -------
    SDFWorld
        Model instances (with world poses and static flags), gravity, and the
        physics ``max_step_size`` if the world defines one.
    """
    world_path = Path(world_path)
    if not world_path.exists():
        raise FileNotFoundError(f"World file not found: {world_path}")
    world = _world_element(world_path, xacro_args)
    world_dir = world_path.resolve().parent
    if models_root is None:
        models_root = _auto_models_root(world_path)
    models_root_path = Path(models_root).resolve() if models_root is not None else None

    models: list[SDFWorldModel] = []
    for include in world.findall("include"):
        uri = (include.findtext("uri") or "").strip()
        if not uri:
            continue
        model_dir, model_name = _resolve_model_dir(uri, world_dir, models_root_path)
        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"Model {model_name!r} referenced by {world_path} not found at"
                f" {model_dir} (submodule not initialized?)"
            )
        name = (include.findtext("name") or model_name).strip()
        static_text = (include.findtext("static") or "").strip().lower()
        models.append(SDFWorldModel(
            name=name,
            model_name=model_name,
            model_dir=model_dir,
            pose=_parse_pose(include.find("pose")),
            static=static_text in ("1", "true"),
            uri=uri,
            is_plane=_is_plane_model(model_dir),
        ))

    gravity_text = (world.findtext("gravity") or "").strip()
    gravity = (
        np.array([float(v) for v in gravity_text.split()])
        if gravity_text else None
    )
    step_text = (world.findtext("physics/max_step_size") or "").strip()
    return SDFWorld(
        name=world.get("name", "default"),
        source=world_path,
        models=models,
        gravity=gravity,
        max_step_size=float(step_text) if step_text else None,
        models_root=models_root_path,
    )


# ---------------------------------------------------------------------------
# Spawning
# ---------------------------------------------------------------------------

def _pose_to_pos_euler(pose: np.ndarray) -> tuple[tuple[float, float, float],
                                                  tuple[float, float, float]]:
    """Convert a 4x4 SDF pose into Genesis ``pos`` and degree ``euler``."""
    rpy = euler_from_matrix(pose, axes="sxyz")
    return (
        tuple(float(v) for v in pose[:3, 3]),
        tuple(math.degrees(float(v)) for v in rpy),
    )


def _plane_surface(gs, model: SDFWorldModel,
                   models_root: Optional[Path]):
    """Genesis surface for a plane model, taken from its SDF ``<material>``.

    ``gs.morphs.Plane`` carries no surface of its own (Genesis hardcodes its
    own plane texture), so the arena floor's material script (wood texture +
    ambient color) has to be passed to ``scene.add_entity(surface=...)``.
    Returns ``None`` when the model declares no usable material.
    """
    material = next(
        (m for m in sdf_materials(model.model_dir, models_root=models_root)
         if m.texture is not None or m.color is not None),
        None,
    )
    if material is None:
        return None
    if material.texture is not None:
        # Genesis rejects a surface that sets both 'color' and
        # 'diffuse_texture'; the texture already carries the floor color.
        return gs.surfaces.Default(
            diffuse_texture=gs.textures.ImageTexture(image_path=material.texture),
        )
    if material.color is not None:
        return gs.surfaces.Default(color=material.color)
    return None


def spawn_sdf_world(scene, world_path: str | os.PathLike, *,
                    models_root: Optional[str | os.PathLike] = None,
                    xacro_args: Optional[dict[str, str]] = None,
                    fixed: Optional[bool] = None,
                    add_ground_plane: bool = True,
                    world: Optional[SDFWorld] = None) -> dict[str, Any]:
    """Add every model of an SDF world to a Genesis scene.

    Must be called *before* ``scene.build()``.  Each model is added at the pose
    written in the world, with ``fixed=True`` when the world marks it static
    (override with ``fixed``).  Plane-only models become ``gs.morphs.Plane``;
    skip them entirely with ``add_ground_plane=False``.  A plane model's SDF
    ``<material>`` (e.g. the WRS floor's wood texture) is applied as the
    entity surface.

    Parameters
    ----------
    scene : gs.Scene
        Scene to populate.
    world_path : str | PathLike
        World file (``.world`` or ``.world.xacro``).
    models_root, xacro_args
        Forwarded to :func:`parse_sdf_world`.
    fixed : bool, optional
        Force the fixed/static flag of every spawned entity.  ``None`` (the
        default) uses each model's ``<static>`` flag.
    add_ground_plane : bool
        Spawn plane-geometry models as ``gs.morphs.Plane`` (default) or skip.
    world : SDFWorld, optional
        Pre-parsed world, to avoid re-parsing for repeated spawns.

    Returns
    -------
    dict[str, Entity]
        Spawned entities keyed by instance name.
    """
    import genesis as gs

    if world is None:
        world = parse_sdf_world(
            world_path, models_root=models_root, xacro_args=xacro_args,
        )
    models_root = world.models_root if models_root is None else models_root

    entities: dict[str, Any] = {}
    for model in world.models:
        name = model.name
        if name in entities:
            raise ValueError(
                f"Duplicate model instance name {name!r} in {world.source}"
            )
        if model.is_plane and not add_ground_plane:
            continue
        pos, euler = _pose_to_pos_euler(model.pose)
        surface = None
        if model.is_plane:
            morph = gs.morphs.Plane(pos=pos, euler=euler)
            surface = _plane_surface(
                gs, model,
                Path(models_root) if models_root is not None else None,
            )
        else:
            morph = gs.morphs.URDF(
                file=load_sdf_model(model.model_dir, models_root=models_root),
                pos=pos,
                euler=euler,
                fixed=model.static if fixed is None else fixed,
            )
        entities[name] = scene.add_entity(morph, name=name, surface=surface)
    return entities
