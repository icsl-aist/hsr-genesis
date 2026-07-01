"""Tests for the SDF -> URDF converter and Genesis spawn path."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import genesis as gs

import hsr_genesis as hg
from hsr_genesis import sdf_parser

MODELS_DIR = (
    Path(__file__).resolve().parents[1]
    / "data" / "tmc_wrs_gazebo" / "tmc_wrs_gazebo_worlds" / "models"
)


def _require_submodule():
    if not MODELS_DIR.exists():
        pytest.skip("tmc_wrs_gazebo submodule not initialized")


# ---------------------------------------------------------------------------
# Conversion (no Genesis scene needed)
# ---------------------------------------------------------------------------

def test_convert_ycb_apple_mesh_model():
    _require_submodule()
    sdf = MODELS_DIR / "ycb_013_apple" / "model-1_4.sdf"
    robot = sdf_parser.sdf_to_urdf(sdf)
    assert robot.name == "ycb_013_apple"
    assert len(robot.links) == 1
    link = robot.links[0]
    assert link.name == "body"
    # Inertial parsed
    assert link.inertial is not None
    assert pytest.approx(link.inertial.mass, abs=1e-6) == 0.068
    # One visual + one collision, both meshes
    assert len(link.visuals) == 1
    assert len(link.collisions) == 1
    from genesis.ext.urdfpy import Mesh as UMesh

    assert isinstance(link.visuals[0].geometry.geometry, UMesh)
    assert isinstance(link.collisions[0].geometry.geometry, UMesh)
    # model:// URI resolved to an absolute existing path
    mesh_path = link.collisions[0].geometry.geometry.filename
    assert os.path.isabs(mesh_path)
    assert os.path.exists(mesh_path)
    assert mesh_path.endswith("nontextured.stl")


def test_convert_wrc_bin_green_box_primitives():
    _require_submodule()
    sdf = MODELS_DIR / "wrc_bin_green" / "model.sdf"
    robot = sdf_parser.sdf_to_urdf(sdf)
    assert robot.name == "wrc_bin_green"
    link = robot.links[0]
    # Multiple box collisions (bottom + 4 walls)
    assert len(link.collisions) >= 5
    from genesis.ext.urdfpy import Box as UBox

    for col in link.collisions:
        assert isinstance(col.geometry.geometry, UBox)


def test_load_sdf_model_via_config():
    _require_submodule()
    model_dir = MODELS_DIR / "ycb_013_apple"
    robot = sdf_parser.load_sdf_model(model_dir)
    assert robot.name == "ycb_013_apple"
    assert len(robot.links) == 1


def test_pose_parsing_identity_and_offset():
    import xml.etree.ElementTree as ET

    assert np.allclose(sdf_parser._parse_pose(None), np.eye(4))
    elem = ET.fromstring("<pose>0.1 0.2 0.3 0 0 1.5707963</pose>")
    T = sdf_parser._parse_pose(elem)
    assert np.allclose(T[:3, 3], [0.1, 0.2, 0.3])
    # 90 deg about Z
    assert np.allclose(T[:3, :3], np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]), atol=1e-6)


def test_uri_resolution_model_scheme():
    _require_submodule()
    models_root = str(MODELS_DIR)
    sdf_dir = str(MODELS_DIR / "ycb_013_apple")
    resolved = sdf_parser._resolve_uri(
        "model://ycb_013_apple/meshes/nontextured.stl", sdf_dir, models_root
    )
    assert os.path.exists(resolved)
    assert resolved.endswith("nontextured.stl")


# ---------------------------------------------------------------------------
# Genesis spawn (requires initialized Genesis)
# ---------------------------------------------------------------------------

@pytest.fixture
def scene():
    return gs.Scene()


def test_spawn_ycb_apple(scene):
    _require_submodule()
    sdf = MODELS_DIR / "ycb_013_apple" / "model-1_4.sdf"
    morph = sdf_parser.morph_from_sdf(sdf)
    entity = scene.add_entity(morph)
    assert entity is not None
    scene.build()
    # Single link, non-fixed base by default
    assert len(entity.links) == 1


def test_spawn_wrc_bin_green(scene):
    _require_submodule()
    sdf = MODELS_DIR / "wrc_bin_green" / "model.sdf"
    morph = sdf_parser.morph_from_sdf(sdf, fixed=True)
    entity = scene.add_entity(morph)
    assert entity is not None
    scene.build()


def test_morph_from_sdf_via_package_attr():
    """hg.morph_from_sdf is exported through the package __getattr__."""
    _require_submodule()
    sdf = MODELS_DIR / "ycb_013_apple" / "model-1_4.sdf"
    morph = hg.morph_from_sdf(sdf)
    assert morph is not None
