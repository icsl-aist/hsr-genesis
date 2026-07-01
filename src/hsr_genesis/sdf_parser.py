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

Out of scope (not present in this dataset; can be added later):
  * Multi-link articulations / SDF ``<joint>``.
  * ``<plane>`` geometry (use ``gs.morphs.Plane`` instead).
  * World-level xacro (lights, physics, multi-model poses).
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import numpy as np
from trimesh.transformations import euler_matrix

from genesis.ext import urdfpy as u

__all__ = ["sdf_to_urdf", "load_sdf_model", "morph_from_sdf"]


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
# Geometry parsing
# ---------------------------------------------------------------------------

def _parse_geometry(geom_elem: ET.Element, sdf_dir: str,
                    models_root: Optional[str]) -> u.Geometry:
    """Convert an SDF ``<geometry>`` element into a ``urdfpy.Geometry``."""
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
        visuals.append(u.Visual(
            geometry=_parse_geometry(geom_elem, sdf_dir, models_root),
            name=v.get("name"),
            origin=_parse_pose(v.find("pose")),
            material=None,
        ))
    for c in elem.findall("collision"):
        geom_elem = c.find("geometry")
        if geom_elem is None:
            continue
        collisions.append(u.Collision(
            name=c.get("name"),
            origin=_parse_pose(c.find("pose")),
            geometry=_parse_geometry(geom_elem, sdf_dir, models_root),
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


def load_sdf_model(model_dir: str | os.PathLike,
                   models_root: Optional[str | os.PathLike] = None) -> "u.URDF":
    """Load a Gazebo model directory (with ``model.config``) as a URDF.

    Resolves the SDF file referenced by ``model.config`` and converts it.
    Falls back to any ``*.sdf`` file in the directory if no config exists.
    """
    model_dir = Path(model_dir)
    config = model_dir / "model.config"
    sdf_file = None
    if config.exists():
        cfg = ET.parse(config).getroot()
        sdf_elem = cfg.find("sdf")
        if sdf_elem is not None and sdf_elem.text:
            sdf_file = model_dir / sdf_elem.text.strip()
    if sdf_file is None or not sdf_file.exists():
        sdfs = sorted(model_dir.glob("*.sdf"))
        if not sdfs:
            raise FileNotFoundError(f"No SDF file found in {model_dir}")
        sdf_file = sdfs[0]
    return sdf_to_urdf(sdf_file, models_root=models_root)


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
