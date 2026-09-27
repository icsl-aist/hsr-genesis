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
# SDF materials
# ---------------------------------------------------------------------------

@pytest.fixture
def mesh_cache(tmp_path, monkeypatch):
    """Redirect converted-mesh caching into the test's tmp dir."""
    monkeypatch.setenv("HSR_GENESIS_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path / "cache"


def test_sdf_materials_resolve_gazebo_scripts():
    _require_submodule()
    trofast = sdf_parser.sdf_materials(MODELS_DIR / "trofast")
    assert [m.name for m in trofast] == ["Gazebo/Orange"]
    # gazebo-classic gazebo.material: Gazebo/Orange diffuse 1 0.5088 0.0468 1
    assert trofast[0].color == pytest.approx((1.0, 0.5088, 0.0468, 1.0))
    assert trofast[0].texture is None

    colors = {m.name: m.color for m in sdf_parser.sdf_materials(MODELS_DIR / "wrc_bin_green")}
    assert colors == {"Gazebo/Green": pytest.approx((0.0, 1.0, 0.0, 1.0))}


def test_sdf_materials_resolve_ogre_script_texture():
    _require_submodule()
    materials = sdf_parser.sdf_materials(MODELS_DIR / "wrc_ground_plane")
    assert len(materials) == 1
    material = materials[0]
    assert material.name == "WRS/WoodFloor"
    # wood.material: ambient 0.8 0.8 0.8 1 with a scaled texture_unit
    assert material.color == pytest.approx((0.8, 0.8, 0.8, 1.0))
    assert material.texture.endswith(".jpg")
    assert os.path.exists(material.texture)
    assert material.texture_scale == pytest.approx((0.2, 0.2))


def test_sdf_materials_resolve_literal_color(tmp_path):
    model_dir = tmp_path / "colored"
    model_dir.mkdir()
    (model_dir / "model.sdf").write_text(
        "<sdf version='1.6'><model name='colored'><link name='l'>"
        "<visual name='v'><geometry><box><size>1 1 1</size></box></geometry>"
        "<material name='red'><color rgba='0.25 0.5 0.75 1'/></material>"
        "</visual></link></model></sdf>"
    )
    materials = sdf_parser.sdf_materials(model_dir)
    assert materials == [
        sdf_parser.SDFMaterial(
            name="red", color=(0.25, 0.5, 0.75, 1.0), texture=None, texture_scale=None,
        )
    ]

    robot = sdf_parser.load_sdf_model(model_dir)
    visual = robot.links[0].visuals[0]
    assert visual.material is not None
    assert np.allclose(visual.material.color, [0.25, 0.5, 0.75, 1.0])


def test_parse_material_script_reads_color_texture_and_scale(tmp_path):
    script = tmp_path / "materials" / "test.material"
    script.parent.mkdir()
    script.write_text(
        "// leading comment\n"
        "material Foo/Bar // inline comment\n"
        "{\n"
        "  technique { pass {\n"
        "    ambient 0.1 0.2 0.3 1.0\n"
        "    diffuse 0.4 0.5 0.6\n"
        "    texture_unit { texture floor.jpg\n scale 0.25 0.5 }\n"
        "  } }\n"
        "}\n"
    )
    material = sdf_parser._parse_material_script(script, "Foo/Bar")
    # 'diffuse' wins over 'ambient'; a missing alpha defaults to 1.
    assert material.color == pytest.approx((0.4, 0.5, 0.6, 1.0))
    assert material.texture == str((script.parent / "floor.jpg").resolve())
    assert material.texture_scale == pytest.approx((0.25, 0.5))
    assert sdf_parser._parse_material_script(script, "Foo/Missing") is None


def test_unknown_material_script_is_reported_once(caplog):
    import xml.etree.ElementTree as ET

    sdf_parser._reported_unknown_materials.clear()
    elem = ET.fromstring(
        "<material><script><name>Gazebo/NoSuchScript</name></script></material>"
    )
    with caplog.at_level("WARNING", logger="hsr_genesis.sdf_parser"):
        assert sdf_parser._parse_sdf_material(elem, ".", None) is None
        assert sdf_parser._parse_sdf_material(elem, ".", None) is None
    assert sum("Unknown SDF material script" in r.message for r in caplog.records) == 1


# ---------------------------------------------------------------------------
# Collada meshes
# ---------------------------------------------------------------------------

def test_collada_collision_mesh_is_converted(mesh_cache):
    _require_submodule()
    from genesis.ext.urdfpy import Box as UBox

    robot = sdf_parser.load_sdf_model(MODELS_DIR / "person_standing")
    collisions = robot.links[0].collisions
    assert isinstance(collisions[0].geometry.geometry, UBox)
    collision_mesh = collisions[1].geometry.geometry.filename
    # standing.dae (Collada collision) -> cached STL, so MuJoCo can decode it.
    assert collision_mesh.endswith(".stl")
    assert os.path.exists(collision_mesh)
    assert str(mesh_cache) in collision_mesh
    # The Collada source itself is untouched.
    assert not (MODELS_DIR / "person_standing" / "meshes" / "standing.stl").exists()


def test_collada_visual_without_external_textures_is_kept(mesh_cache):
    _require_submodule()
    robot = sdf_parser.load_sdf_model(MODELS_DIR / "ycb_013_apple")
    visual_mesh = robot.links[0].visuals[0].geometry.geometry.filename
    # Its texture lives next to the mesh: no conversion needed.
    assert visual_mesh.endswith("textured.dae")


def test_collada_visual_with_external_textures_is_converted(mesh_cache):
    _require_submodule()
    assert sdf_parser._collada_references_escaping_textures(
        MODELS_DIR / "person_standing" / "meshes" / "standing.dae"
    )
    assert not sdf_parser._collada_references_escaping_textures(
        MODELS_DIR / "ycb_013_apple" / "meshes" / "textured.dae"
    )

    robot = sdf_parser.load_sdf_model(MODELS_DIR / "person_standing")
    visual_mesh = robot.links[0].visuals[0].geometry.geometry.filename
    assert visual_mesh.endswith(".glb")
    assert os.path.exists(visual_mesh)

    # The GLB keeps the model's textures and its orientation.
    import trimesh

    scene = trimesh.load(visual_mesh, force="scene", process=False)
    assert len(scene.geometry) == 7
    for geometry in scene.geometry.values():
        assert geometry.visual.material.baseColorTexture is not None


def test_escaped_texture_reference_detection(tmp_path):
    mesh_dir = tmp_path / "meshes"
    mesh_dir.mkdir()
    inside = mesh_dir / "inside.dae"
    inside.write_text(
        "<COLLADA xmlns='http://www.collada.org/2005/11/COLLADASchema'>"
        "<library_images><image><init_from>tex.png</init_from></image></library_images>"
        "</COLLADA>"
    )
    outside = mesh_dir / "outside.dae"
    outside.write_text(
        "<COLLADA xmlns='http://www.collada.org/2005/11/COLLADASchema'>"
        "<library_images><image><init_from>../textures/tex.png</init_from></image></library_images>"
        "</COLLADA>"
    )
    assert not sdf_parser._collada_references_escaping_textures(inside)
    assert sdf_parser._collada_references_escaping_textures(outside)


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
