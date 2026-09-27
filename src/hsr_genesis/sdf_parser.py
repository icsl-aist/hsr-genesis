"""Gazebo SDF (Simulation Description Format) -> ``urdfpy.URDF`` converter.

Genesis 0.4.6 has no native SDF morph (only URDF/MJCF/USD/Mesh/Drone).  This
module converts single-model Gazebo SDF files into in-memory
``genesis.ext.urdfpy.URDF`` objects that can be fed directly to
``gs.morphs.URDF(file=...)`` (``parse_urdf`` accepts a ``urdfpy.URDF``
instance as ``morph.file``).

Scope (matches the ``tmc_wrs_gazebo`` dataset):
  * Single-link models (all 93 models in the submodule are single-link).
  * Geometry: ``box``, ``cylinder``, ``sphere``, ``mesh``.
  * ``model://`` and ``file://`` URI resolution.
  * ``<static>`` -> fixed base.
  * ``model.config`` -> SDF file discovery.
  * ``<material>``: literal ``<color rgba="...">`` and ``<script>`` material
    references (Ogre ``.material`` scripts shipped with the model, or the
    built-in ``Gazebo/<Name>`` scripts) become URDF visual materials, so the
    arena furniture keeps the colors of the Gazebo world instead of rendering
    in Genesis' default white.
  * Collada (``.dae``) *collision* meshes are converted to a cached binary STL.
    Genesis hands URDF collision meshes to MuJoCo, which cannot decode Collada:
    any ``.dae`` collision mesh otherwise makes Genesis fall back to its legacy
    URDF parser (losing the model's textures and physics parameters).
  * Collada *visual* meshes whose textures live outside the mesh directory
    (trimesh refuses to resolve those, so the model would render untextured)
    are converted to a cached GLB that embeds them.  Converted files are cached
    under ``~/.cache/hsr_genesis`` (``HSR_GENESIS_CACHE_DIR`` overrides the
    location); the submodule's own files are never modified.

Out of scope (not present in this dataset; can be added later):
  * Multi-link articulations / SDF ``<joint>``.
  * ``<plane>`` geometry (use ``gs.morphs.Plane`` instead).

World-level SDF/xacro files (``<include>``ed models, world poses, physics) are
handled by :mod:`hsr_genesis.sdf_world`.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
from trimesh.transformations import euler_matrix

from genesis.ext import urdfpy as u

__all__ = ["SDFMaterial", "sdf_materials", "sdf_to_urdf", "load_sdf_model", "morph_from_sdf"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pose parsing
# ---------------------------------------------------------------------------

def _parse_pose(elem: Optional[ET.Element]) -> np.ndarray:
    """Parse an SDF ``<pose>x y z roll pitch yaw</pose>`` into a 4x4 transform.

    SDF uses xyz translation + roll-pitch-yaw (intrinsic XYZ) in radians.
    Returns a 4x4 homogeneous matrix (identity if ``elem`` is None).
    """
    if elem is None or elem.text is None or not elem.text.strip():
        return np.eye(4)
    vals = [float(v) for v in elem.text.split()]
    if len(vals) == 6:
        x, y, z, r, p, yw = vals
    elif len(vals) == 3:
        x, y, z = vals
        r = p = yw = 0.0
    else:
        raise ValueError(f"Unexpected pose with {len(vals)} values: {elem.text!r}")
    T = euler_matrix(r, p, yw, axes="sxyz")
    T[:3, 3] = [x, y, z]
    return T


# ---------------------------------------------------------------------------
# URI resolution
# ---------------------------------------------------------------------------

def _resolve_uri(uri: str, sdf_dir: str, models_root: Optional[str]) -> str:
    """Resolve an SDF mesh ``<uri>`` to an absolute filesystem path.

    Supported schemes:
      * ``model://<name>/<rest>`` -> ``<models_root>/<name>/<rest>``
      * ``file://<path>``         -> strip scheme (path may be absolute)
      * relative path             -> resolved against ``sdf_dir``
    """
    uri = uri.strip()
    if uri.startswith("model://"):
        rest = uri[len("model://"):]
        if models_root is None:
            # Fall back to the SDF file's parent ``models`` directory, which is
            # the common layout (e.g. tmc_wrs_gazebo_worlds/models/<name>/...).
            models_root = str(Path(sdf_dir).parent)
        name, _, sub = rest.partition("/")
        candidate = os.path.join(models_root, name, sub)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
        # Some SDFs reference the model by its own name from inside the model
        # dir (e.g. model://ycb_013_apple/meshes/... issued from within
        # ycb_013_apple/).  Try resolving against sdf_dir too.
        candidate2 = os.path.join(sdf_dir, sub)
        if os.path.exists(candidate2):
            return os.path.abspath(candidate2)
        return os.path.abspath(candidate)  # let the loader raise if missing
    if uri.startswith("file://"):
        path = uri[len("file://"):]
        if not os.path.isabs(path):
            path = os.path.join(sdf_dir, path)
        return os.path.abspath(path)
    # Plain relative / absolute path.
    if not os.path.isabs(uri):
        uri = os.path.join(sdf_dir, uri)
    return os.path.abspath(uri)


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------

#: Colors of the built-in Gazebo material scripts (``Gazebo/<Name>``) that the
#: Gazebo world files reference.  Values are the ``diffuse`` color (falling
#: back to ``ambient``) of gazebo-classic's
#: ``media/materials/scripts/gazebo.material`` (gazebo11 branch), which cannot
#: be read from disk here: the models reference it as
#: ``file://media/materials/scripts/gazebo.material``, a path that only exists
#: inside an installed Gazebo (``/usr/share/gazebo-11/media/...``).  Color-only
#: scripts are listed; texture-bearing ones (``Gazebo/Wood``, ...) are not, and
#: are reported as unknown.
_GAZEBO_SCRIPT_COLORS: dict[str, tuple[float, float, float, float]] = {
    "Gazebo/Black": (0.0, 0.0, 0.0, 1.0),
    "Gazebo/White": (1.0, 1.0, 1.0, 1.0),
    "Gazebo/FlatBlack": (0.1, 0.1, 0.1, 1.0),
    "Gazebo/Gray": (0.7, 0.7, 0.7, 1.0),
    "Gazebo/Grey": (0.7, 0.7, 0.7, 1.0),
    "Gazebo/DarkGray": (0.175, 0.175, 0.175, 1.0),
    "Gazebo/DarkGrey": (0.175, 0.175, 0.175, 1.0),
    "Gazebo/Red": (1.0, 0.0, 0.0, 1.0),
    "Gazebo/Green": (0.0, 1.0, 0.0, 1.0),
    "Gazebo/Blue": (0.0, 0.0, 1.0, 1.0),
    "Gazebo/Yellow": (1.0, 1.0, 0.0, 1.0),
    "Gazebo/Orange": (1.0, 0.5088, 0.0468, 1.0),
    "Gazebo/Purple": (1.0, 0.0, 1.0, 1.0),
    "Gazebo/Turquoise": (0.0, 1.0, 1.0, 1.0),
}

# Unknown script names are reported once, not once per visual.
_reported_unknown_materials: set[str] = set()


@dataclass(frozen=True)
class SDFMaterial:
    """A material resolved from an SDF ``<visual><material>`` element.

    Attributes
    ----------
    name : str | None
        Script material name (``Gazebo/Gray``) or the ``<material>`` element's
        ``name`` attribute, when either is present.
    color : tuple[float, float, float, float] | None
        RGBA in ``[0, 1]``, from ``<color rgba>`` or the referenced Ogre
        script's ``diffuse``/``ambient`` pass value.
    texture : str | None
        Absolute path to the diffuse texture referenced by an Ogre
        ``texture_unit``, if any.
    texture_scale : tuple[float, float] | None
        Ogre ``scale`` of the texture unit (texture repeats per meter).
    """

    name: Optional[str] = None
    color: Optional[tuple[float, float, float, float]] = None
    texture: Optional[str] = None
    texture_scale: Optional[tuple[float, float]] = None


_OGRE_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)


def _ogre_material_body(text: str, name: str) -> Optional[str]:
    """Return the ``{...}`` body of the Ogre script block ``material <name>``."""
    match = re.search(rf"\bmaterial\s+{re.escape(name)}\s*(?::[^{{]*)?\{{", text)
    if match is None:
        return None
    depth = 0
    for idx in range(match.end() - 1, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return text[match.end():idx]
    return None


def _ogre_color(body: str) -> Optional[tuple[float, float, float, float]]:
    """Return the RGBA of the first ``diffuse`` (else ``ambient``) pass value."""
    for keyword in ("diffuse", "ambient"):
        match = re.search(rf"\b{keyword}\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)(?:\s+(-?[\d.]+))?", body)
        if match is not None:
            r, g, b, a = (float(v) if v is not None else 1.0 for v in match.groups())
            return (r, g, b, a)
    return None


def _parse_material_script(path: str | os.PathLike,
                          name: str) -> Optional[SDFMaterial]:
    """Read one material out of an Ogre ``.material`` script file.

    Only the parts Gazebo worlds rely on are parsed: the ``diffuse``/``ambient``
    color of the first pass and the first ``texture_unit``'s texture (resolved
    relative to the script) plus its ``scale``.  Material inheritance
    (``material A : B``) is not followed.
    """
    text = _OGRE_COMMENT_RE.sub("", Path(path).read_text(errors="replace"))
    body = _ogre_material_body(text, name)
    if body is None:
        return None
    color = _ogre_color(body)
    texture = None
    texture_scale = None
    unit = re.search(r"\btexture_unit\s*\{(.*?)\}", body, re.DOTALL)
    if unit is not None:
        texture_match = re.search(r"\btexture\s+([^\s{}]+)", unit.group(1))
        if texture_match is not None:
            texture = str((Path(path).parent / texture_match.group(1)).resolve())
        scale_match = re.search(r"\bscale\s+(-?[\d.]+)\s+(-?[\d.]+)", unit.group(1))
        if scale_match is not None:
            texture_scale = (float(scale_match.group(1)), float(scale_match.group(2)))
    if color is None and texture is None:
        return None
    return SDFMaterial(name=name, color=color, texture=texture,
                       texture_scale=texture_scale)


def _parse_rgba(text: str) -> tuple[float, float, float, float]:
    """Parse an SDF ``rgba`` attribute ("r g b a") into a 4-tuple."""
    values = [float(v) for v in text.split()]
    if len(values) == 3:
        values.append(1.0)
    if len(values) != 4:
        raise ValueError(f"Expected 3 or 4 color components, got {text!r}")
    return (values[0], values[1], values[2], values[3])


def _parse_sdf_material(elem: Optional[ET.Element], sdf_dir: str,
                        models_root: Optional[str]) -> Optional[SDFMaterial]:
    """Convert an SDF ``<material>`` element into an :class:`SDFMaterial`."""
    if elem is None:
        return None
    color = None
    if (color_elem := elem.find("color")) is not None and color_elem.get("rgba"):
        color = _parse_rgba(color_elem.get("rgba"))

    script_name = None
    script_path = None
    script = elem.find("script")
    if script is not None:
        script_name = (script.findtext("name") or "").strip() or None
        uri = (script.findtext("uri") or "").strip()
        if uri:
            candidate = _resolve_uri(uri, sdf_dir, models_root)
            if os.path.isfile(candidate):
                script_path = candidate

    texture = None
    texture_scale = None
    if script_path is not None and script_name is not None:
        parsed = _parse_material_script(script_path, script_name)
        if parsed is not None:
            color = parsed.color if color is None else color
            texture = parsed.texture
            texture_scale = parsed.texture_scale

    if color is None and texture is None and script_name is not None:
        color = _GAZEBO_SCRIPT_COLORS.get(script_name)
        if color is None and script_name not in _reported_unknown_materials:
            _reported_unknown_materials.add(script_name)
            logger.warning(
                "Unknown SDF material script %r (no color applied); add it to "
                "_GAZEBO_SCRIPT_COLORS or ship the referenced .material file.",
                script_name,
            )

    if color is None and texture is None:
        return None
    return SDFMaterial(
        name=script_name or elem.get("name"),
        color=color,
        texture=texture,
        texture_scale=texture_scale,
    )


def sdf_materials(sdf_path: str | os.PathLike, *,
                  models_root: Optional[str | os.PathLike] = None) -> list[SDFMaterial]:
    """Return the visual materials declared by an SDF model, in document order.

    ``sdf_path`` may be a model directory (containing ``model.config``) or a
    ``.sdf`` file.  Visuals without a usable ``<material>`` are skipped, so the
    result may be shorter than the visual count.
    """
    sdf_path = Path(sdf_path)
    sdf_file = _resolve_sdf_file(sdf_path) if sdf_path.is_dir() else sdf_path
    sdf_dir = str(sdf_file.resolve().parent)
    root = ET.parse(sdf_file).getroot()
    model_elem = root if root.tag == "model" else root.find("model")
    if model_elem is None:
        raise ValueError(f"No <model> element found in {sdf_file}")
    resolved = (
        _parse_sdf_material(visual.find("material"), sdf_dir, os.fspath(models_root)
                            if models_root is not None else None)
        for visual in model_elem.iter("visual")
    )
    return [material for material in resolved if material is not None]


# ---------------------------------------------------------------------------
# Mesh format conversions
# ---------------------------------------------------------------------------

#: Mesh formats MuJoCo's asset decoder accepts (Collada is not one of them).
_MUJOCO_MESH_FORMATS = (".stl", ".obj", ".ply", ".msh")

#: Bump when the conversions below change to invalidate cached files.
_MESH_CACHE_VERSION = 1

#: Collada ``<init_from>`` elements (namespace-agnostic).
_COLLADA_INIT_FROM_RE = re.compile(r"^\{.*\}init_from$")


def _cache_dir(kind: str) -> Path:
    """Directory holding converted meshes of one kind (override with
    ``HSR_GENESIS_CACHE_DIR`` or ``XDG_CACHE_HOME``)."""
    override = os.environ.get("HSR_GENESIS_CACHE_DIR")
    if override:
        return Path(override) / f"{kind}_meshes"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "hsr_genesis" / f"{kind}_meshes"


def _cache_target(source: Path, kind: str, suffix: str) -> Path:
    """Cache path for a converted mesh, keyed on the source file's identity.

    Keying on path, size and mtime means the conversion happens once per source
    file; the source files themselves are never modified.
    """
    stat = source.stat()
    key = hashlib.sha1(
        f"{_MESH_CACHE_VERSION}:{source}:{stat.st_size}:{stat.st_mtime_ns}".encode()
    ).hexdigest()[:16]
    return _cache_dir(kind) / f"{source.stem}-{key}{suffix}"


def _export_cached(source: Path, target: Path, obj, file_type: str) -> str:
    """Export ``obj`` to ``target`` through a temporary file (atomic-ish)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp{target.suffix}")
    obj.export(str(tmp), file_type=file_type)
    os.replace(tmp, target)
    logger.debug("Converted mesh %s -> %s", source, target)
    return str(target)


def _convert_collision_mesh(path: str) -> str:
    """Return a cached binary STL of a collision mesh MuJoCo cannot decode.

    Only geometry is needed for collision, hence STL.
    """
    source = Path(path).resolve()
    target = _cache_target(source, "collision", ".stl")
    if target.exists():
        return str(target)
    mesh = trimesh.load(str(source), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):  # pragma: no cover - defensive
        mesh = mesh.dump(concatenate=True)
    return _export_cached(source, target, mesh, "stl")


def _collada_references_escaping_textures(path: str | os.PathLike) -> bool:
    """Whether a Collada mesh references a texture outside its own directory.

    trimesh (>= 5) resolves a Collada's ``<init_from>`` image paths with a
    ``FilePathResolver`` rooted at the *mesh's* directory and refuses paths
    that escape it (the WRS people model references
    ``../materials/textures/tshirt02_texture.png``), so the mesh silently loads
    without any texture.
    """
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:  # pragma: no cover - defensive
        return False
    mesh_dir = Path(path).resolve().parent
    for elem in root.iter():
        if not _COLLADA_INIT_FROM_RE.match(elem.tag) or not elem.text:
            continue
        reference = Path(elem.text.strip())
        target = reference if reference.is_absolute() else mesh_dir / reference
        if not target.resolve().is_relative_to(mesh_dir):
            return True
    return False


def _convert_visual_mesh(path: str) -> str:
    """Return a cached GLB of a Collada *visual* mesh with unloadable textures.

    The source is re-read with a ``FilePathResolver`` that allows the escaping
    texture paths and re-exported as GLB, which embeds the diffuse textures.
    Genesis loads GLB visuals through its glTF parser and keeps the textures;
    the mesh transform is unchanged.
    """
    source = Path(path).resolve()
    target = _cache_target(source, "visual", ".glb")
    if target.exists():
        return str(target)
    resolver = trimesh.resolvers.FilePathResolver(str(source.parent), allow_anywhere=True)
    scene = trimesh.load(str(source), force="scene", process=False, resolver=resolver)
    return _export_cached(source, target, scene, "glb")


def _collision_mesh_path(path: str) -> str:
    """Map an SDF collision mesh to a mesh MuJoCo can load."""
    if Path(path).suffix.lower() in _MUJOCO_MESH_FORMATS:
        return path
    try:
        return _convert_collision_mesh(path)
    except Exception as exc:
        logger.warning(
            "Could not convert collision mesh %s to a MuJoCo-loadable format "
            "(%s); Genesis may fall back to its legacy URDF parser.", path, exc,
        )
        return path


def _visual_mesh_path(path: str) -> str:
    """Map an SDF visual mesh to a mesh whose textures Genesis can load."""
    if Path(path).suffix.lower() != ".dae" or not _collada_references_escaping_textures(path):
        return path
    try:
        return _convert_visual_mesh(path)
    except Exception as exc:
        logger.warning("Could not convert visual mesh %s to GLB (%s).", path, exc)
        return path


# ---------------------------------------------------------------------------
# Geometry parsing
# ---------------------------------------------------------------------------

def _parse_geometry(geom_elem: ET.Element, sdf_dir: str,
                    models_root: Optional[str],
                    is_collision: bool = False) -> u.Geometry:
    """Convert an SDF ``<geometry>`` element into a ``urdfpy.Geometry``.

    ``is_collision`` marks collision geometry, whose meshes are handed to
    MuJoCo by Genesis and are therefore converted when MuJoCo cannot decode
    them (see :func:`_collision_mesh_path`).
    """
    children = list(geom_elem)
    if not children:
        raise ValueError("Empty <geometry> element")
    tag = children[0].tag
    if tag == "box":
        size_elem = children[0].find("size")
        size = [float(v) for v in (size_elem.text.split() if size_elem is not None else "1 1 1")]
        return u.Geometry(box=u.Box(size=size))
    if tag == "cylinder":
        radius = float(children[0].findtext("radius", "0.5"))
        length = float(children[0].findtext("length", "1.0"))
        return u.Geometry(cylinder=u.Cylinder(radius=radius, length=length))
    if tag == "sphere":
        radius = float(children[0].findtext("radius", "0.5"))
        return u.Geometry(sphere=u.Sphere(radius=radius))
    if tag == "mesh":
        uri_elem = children[0].find("uri")
        if uri_elem is None or not uri_elem.text:
            raise ValueError("Mesh geometry missing <uri>")
        path = _resolve_uri(uri_elem.text, sdf_dir, models_root)
        path = _collision_mesh_path(path) if is_collision else _visual_mesh_path(path)
        scale_elem = children[0].find("scale")
        scale = None
        if scale_elem is not None and scale_elem.text:
            parts = [float(v) for v in scale_elem.text.split()]
            if len(parts) == 1:
                scale = parts[0]
            elif len(parts) == 3:
                # urdfpy.Mesh.scale is a single float; take the mean for
                # anisotropic scales (rare in this dataset).
                scale = float(np.mean(parts))
        return u.Geometry(mesh=u.Mesh(filename=path, scale=scale))
    raise ValueError(f"Unsupported SDF geometry: <{tag}> (plane is not supported; use gs.morphs.Plane)")


# ---------------------------------------------------------------------------
# Inertial parsing
# ---------------------------------------------------------------------------

def _parse_inertial(elem: Optional[ET.Element]) -> Optional[u.Inertial]:
    if elem is None:
        return None
    mass = float(elem.findtext("mass", "1.0"))
    inertia_elem = elem.find("inertia")
    if inertia_elem is not None:
        ixx = float(inertia_elem.findtext("ixx", "0"))
        ixy = float(inertia_elem.findtext("ixy", "0"))
        ixz = float(inertia_elem.findtext("ixz", "0"))
        iyy = float(inertia_elem.findtext("iyy", "0"))
        iyz = float(inertia_elem.findtext("iyz", "0"))
        izz = float(inertia_elem.findtext("izz", "0"))
        inertia = np.array([[ixx, ixy, ixz],
                            [ixy, iyy, iyz],
                            [ixz, iyz, izz]], dtype=float)
    else:
        # Fallback: unit inertia so the link is dynamically valid.
        inertia = np.eye(3) * 1e-3
    origin = _parse_pose(elem.find("pose"))
    return u.Inertial(mass=mass, inertia=inertia, origin=origin)


# ---------------------------------------------------------------------------
# Link parsing
# ---------------------------------------------------------------------------

def _parse_link(elem: ET.Element, sdf_dir: str,
                models_root: Optional[str]) -> u.Link:
    name = elem.get("name", "link")
    inertial = _parse_inertial(elem.find("inertial"))
    visuals = []
    collisions = []
    for v in elem.findall("visual"):
        geom_elem = v.find("geometry")
        if geom_elem is None:
            continue
        visual_name = v.get("name") or name
        material = _parse_sdf_material(v.find("material"), sdf_dir, models_root)
        visuals.append(u.Visual(
            geometry=_parse_geometry(geom_elem, sdf_dir, models_root),
            name=v.get("name"),
            origin=_parse_pose(v.find("pose")),
            material=(
                u.Material(
                    name=material.name or f"{visual_name}_material",
                    color=np.asarray(material.color, dtype=np.float64),
                )
                if material is not None and material.color is not None
                else None
            ),
        ))
    for c in elem.findall("collision"):
        geom_elem = c.find("geometry")
        if geom_elem is None:
            continue
        collisions.append(u.Collision(
            name=c.get("name"),
            origin=_parse_pose(c.find("pose")),
            geometry=_parse_geometry(geom_elem, sdf_dir, models_root, is_collision=True),
        ))
    return u.Link(name=name, inertial=inertial, visuals=visuals, collisions=collisions)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sdf_to_urdf(sdf_path: str | os.PathLike,
                models_root: Optional[str | os.PathLike] = None) -> "u.URDF":
    """Convert a Gazebo SDF model file into a ``urdfpy.URDF`` object.

    Parameters
    ----------
    sdf_path : str | PathLike
        Path to the ``.sdf`` file.
    models_root : str | PathLike, optional
        Directory used to resolve ``model://<name>/...`` URIs.  If None,
        defaults to the parent of the SDF file's directory (i.e. the
        ``models/`` folder that contains sibling model directories).

    Returns
    -------
    urdfpy.URDF
        An in-memory URDF object suitable for ``gs.morphs.URDF(file=...)``.
        Mesh paths are absolutized so they load regardless of the current
        working directory.
    """
    sdf_path = os.fspath(sdf_path)
    sdf_dir = os.path.dirname(os.path.abspath(sdf_path))
    tree = ET.parse(sdf_path)
    root = tree.getroot()
    # SDF root is <sdf>; the <model> may be the root or a child.
    model_elem = root if root.tag == "model" else root.find("model")
    if model_elem is None:
        raise ValueError(f"No <model> element found in {sdf_path}")
    name = model_elem.get("name", "sdf_model")
    links = [_parse_link(le, sdf_dir, models_root) for le in model_elem.findall("link")]
    if not links:
        raise ValueError(f"SDF model {name!r} has no links")
    # Single-link models need no joints.  (Multi-link + joint support can be
    # added later; not required by the tmc_wrs_gazebo dataset.)
    return u.URDF(name=name, links=links, joints=[], materials=[])


def _resolve_sdf_file(model_dir: str | os.PathLike) -> Path:
    """Return the SDF file describing a Gazebo model directory.

    Resolves the ``<sdf>`` entry of ``model.config`` when present, otherwise
    falls back to the first ``*.sdf`` file in the directory.
    """
    model_dir = Path(model_dir)
    config = model_dir / "model.config"
    if config.exists():
        cfg = ET.parse(config).getroot()
        sdf_elem = cfg.find("sdf")
        if sdf_elem is not None and sdf_elem.text:
            sdf_file = model_dir / sdf_elem.text.strip()
            if sdf_file.exists():
                return sdf_file
    sdfs = sorted(model_dir.glob("*.sdf"))
    if not sdfs:
        raise FileNotFoundError(f"No SDF file found in {model_dir}")
    return sdfs[0]


def load_sdf_model(model_dir: str | os.PathLike,
                   models_root: Optional[str | os.PathLike] = None) -> "u.URDF":
    """Load a Gazebo model directory (with ``model.config``) as a URDF.

    Resolves the SDF file referenced by ``model.config`` and converts it.
    Falls back to any ``*.sdf`` file in the directory if no config exists.
    """
    return sdf_to_urdf(_resolve_sdf_file(model_dir), models_root=models_root)


def morph_from_sdf(sdf_path: str | os.PathLike,
                   models_root: Optional[str | os.PathLike] = None,
                   **urdf_kwargs):
    """Build a ``gs.morphs.URDF`` morph from an SDF file.

    Convenience wrapper around :func:`sdf_to_urdf` that returns a morph ready
    for ``scene.add_entity(...)``.  ``urdf_kwargs`` are forwarded to
    ``gs.morphs.URDF`` (e.g. ``pos``, ``euler``, ``fixed``, ``scale``).
    """
    import genesis as gs

    return gs.morphs.URDF(file=sdf_to_urdf(sdf_path, models_root=models_root), **urdf_kwargs)
