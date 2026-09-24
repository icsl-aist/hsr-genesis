"""Tests for the SDF world (``.world`` / ``.world.xacro``) -> Genesis importer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import genesis as gs

import hsr_genesis as hg
from hsr_genesis import sdf_world

REPO_ROOT = Path(__file__).resolve().parents[1]
WORLDS_DIR = REPO_ROOT / "data" / "tmc_wrs_gazebo" / "tmc_wrs_gazebo_worlds"
MODELS_DIR = WORLDS_DIR / "models"
WORLD_XACRO = WORLDS_DIR / "worlds" / "wrs2020.world.xacro"


def _require_submodule():
    if not WORLD_XACRO.exists():
        pytest.skip("tmc_wrs_gazebo submodule not initialized")


def _write_world(tmp_path: Path, body: str) -> Path:
    world_dir = tmp_path / "worlds"
    world_dir.mkdir(parents=True, exist_ok=True)
    world = world_dir / "test.world"
    world.write_text(
        "<?xml version='1.0'?>\n"
        f"<sdf version='1.6'><world name='test'>{body}</world></sdf>\n"
    )
    return world


def _entity_pos(entity) -> np.ndarray:
    pos = entity.get_pos()
    if pos.ndim > 1:
        pos = pos[0]
    return np.asarray(pos, dtype=float)


# ---------------------------------------------------------------------------
# Parsing (no Genesis scene needed)
# ---------------------------------------------------------------------------

def test_parse_wrs2020_world():
    _require_submodule()
    world = sdf_world.parse_sdf_world(WORLD_XACRO)

    assert world.name == "default"
    assert world.source == WORLD_XACRO
    assert world.models_root == MODELS_DIR.resolve()
    assert np.allclose(world.gravity, [0.0, 0.0, -9.8])
    # ``fast_physics`` defaults to true in the xacro -> 0.003 s step size.
    assert world.max_step_size == pytest.approx(0.003)

    names = [m.name for m in world.models]
    assert len(names) == 18
    assert len(set(names)) == len(names)
    by_name = {m.name: m for m in world.models}

    # Ground plane: SDF <plane> geometry, static, at the origin.
    ground = by_name["wrc_ground_plane"]
    assert ground.is_plane and ground.static
    assert np.allclose(ground.pose[:3, 3], [0.0, 0.0, 0.0])
    assert ground.model_dir == MODELS_DIR.resolve() / "wrc_ground_plane"

    # Furniture: single-link SDF models, not planes, placed by the world pose.
    bookshelf = by_name["wrc_bookshelf"]
    assert not bookshelf.is_plane and bookshelf.static
    assert np.allclose(bookshelf.pose[:3, 3], [2.7, -1.0, 0.0], atol=1e-6)
    # yaw = -1.57 -> rotation by -90 deg about Z
    assert np.allclose(
        bookshelf.pose[:3, :3],
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        atol=1e-3,
    )

    # Two instances of the same model keep distinct names.
    assert by_name["wrc_long_table"].model_dir == by_name["wrc_long_table_0"].model_dir
    assert np.allclose(
        by_name["wrc_long_table_0"].pose[:3, 3], [-2.7, -0.3, 0.0], atol=1e-6,
    )

    # Included-without-<static> models stay dynamic.
    assert not by_name["person_standing"].static
    assert not by_name["trofast_1"].static

    # trofast_knob defaults to false -> the plain trofast model is used.
    assert by_name["trofast_1"].model_name == "trofast"
    assert by_name["trofast_1"].model_dir == MODELS_DIR.resolve() / "trofast"


def test_xacro_args_select_model_and_physics():
    _require_submodule()

    knob = sdf_world.parse_sdf_world(
        WORLD_XACRO, xacro_args={"trofast_knob": "true"},
    )
    assert {m.name: m.model_name for m in knob.models}["trofast_2"] == "trofast_knob"

    # ``fast_physics:=false`` (the upstream launch default) leaves the
    # <physics> block empty, so no step size is reported.
    plain = sdf_world.parse_sdf_world(
        WORLD_XACRO, xacro_args={"fast_physics": "false"},
    )
    assert plain.max_step_size is None
    assert np.allclose(plain.gravity, [0.0, 0.0, -9.8])


def test_parse_world_reads_pose_and_static(tmp_path):
    _require_submodule()
    world = _write_world(
        tmp_path,
        "<include><name>bin</name><uri>model://wrc_bin_green</uri>"
        "<pose>1 0.5 0 0 0 1.5707963</pose></include>",
    )
    parsed = sdf_world.parse_sdf_world(world, models_root=MODELS_DIR)
    assert [m.name for m in parsed.models] == ["bin"]
    assert np.allclose(parsed.models[0].pose[:3, 3], [1.0, 0.5, 0.0], atol=1e-6)
    assert not parsed.models[0].static


def test_parse_world_without_models_root_raises(tmp_path):
    world = _write_world(
        tmp_path,
        "<include><name>bin</name><uri>model://wrc_bin_green</uri></include>",
    )
    with pytest.raises(FileNotFoundError):
        sdf_world.parse_sdf_world(world)


def test_parse_world_missing_model_raises(tmp_path):
    world = _write_world(
        tmp_path,
        "<include><name>ghost</name><uri>model://does_not_exist</uri></include>",
    )
    with pytest.raises(FileNotFoundError):
        sdf_world.parse_sdf_world(world, models_root=MODELS_DIR)


def test_plane_detection_and_mixed_geometry_rejected(tmp_path):
    _require_submodule()
    assert sdf_world._is_plane_model(MODELS_DIR / "wrc_ground_plane")
    assert not sdf_world._is_plane_model(MODELS_DIR / "wrc_bin_green")

    model_dir = tmp_path / "mixed"
    model_dir.mkdir()
    (model_dir / "model.sdf").write_text(
        "<sdf version='1.6'><model name='mixed'><link name='l'>"
        "<visual name='v'><geometry><plane><normal>0 0 1</normal></plane>"
        "</geometry></visual>"
        "<collision name='c'><geometry><box><size>1 1 1</size></box>"
        "</geometry></collision>"
        "</link></model></sdf>"
    )
    with pytest.raises(NotImplementedError):
        sdf_world._is_plane_model(model_dir)


def test_package_exports():
    assert hg.parse_sdf_world is sdf_world.parse_sdf_world
    assert hg.spawn_sdf_world is sdf_world.spawn_sdf_world
    assert hg.SDFWorld is sdf_world.SDFWorld
    assert hg.SDFWorldModel is sdf_world.SDFWorldModel
    assert "parse_sdf_world" in dir(hg)


# ---------------------------------------------------------------------------
# Spawning (requires initialized Genesis)
# ---------------------------------------------------------------------------

@pytest.fixture
def scene():
    return gs.Scene()


def test_spawn_sdf_world_subset(scene, tmp_path):
    _require_submodule()
    world = _write_world(
        tmp_path,
        "<gravity>0 0 -9.8</gravity>"
        "<include><name>ground</name><static>1</static>"
        "<uri>model://wrc_ground_plane</uri></include>"
        "<include><name>bin</name><static>1</static>"
        "<uri>model://wrc_bin_green</uri>"
        "<pose>1 0.5 0 0 0 1.5707963</pose></include>"
        "<include><name>apple</name><uri>model://ycb_013_apple</uri>"
        "<pose>-0.5 0 0.5 0 0 0</pose></include>",
    )
    entities = sdf_world.spawn_sdf_world(scene, world, models_root=MODELS_DIR)
    assert sorted(entities) == ["apple", "bin", "ground"]
    scene.build()

    # The <plane> model became a plane at the origin; the static bin sits at
    # its world pose and the apple starts at its world pose.
    assert np.allclose(_entity_pos(entities["ground"]), [0.0, 0.0, 0.0], atol=1e-4)
    assert np.allclose(_entity_pos(entities["bin"]), [1.0, 0.5, 0.0], atol=1e-4)
    # Mesh models report their center-of-mass frame, which the SDF offsets from
    # the model origin via <inertial><pose> (the apple: ~3.6 cm up).
    assert np.allclose(
        _entity_pos(entities["apple"]), [-0.5, 0.0, 0.5], atol=0.05,
    )


    for _ in range(300):
        scene.step()

    # Static models are unaffected by gravity, the dynamic one settles on the
    # plane without falling through it.
    assert np.allclose(_entity_pos(entities["bin"]), [1.0, 0.5, 0.0], atol=1e-3)
    apple_z = _entity_pos(entities["apple"])[2]
    assert 0.0 < apple_z < 0.15


def test_spawn_sdf_world_skip_ground_plane(scene, tmp_path):
    _require_submodule()
    world = _write_world(
        tmp_path,
        "<include><name>ground</name><static>1</static>"
        "<uri>model://wrc_ground_plane</uri></include>"
        "<include><name>bin</name><static>1</static>"
        "<uri>model://wrc_bin_green</uri></include>",
    )
    entities = sdf_world.spawn_sdf_world(
        scene, world, models_root=MODELS_DIR, add_ground_plane=False,
    )
    assert sorted(entities) == ["bin"]


def test_spawn_sdf_world_fixed_override(scene, tmp_path):
    """``fixed=True`` freezes a model the world marks dynamic."""
    _require_submodule()
    world = _write_world(
        tmp_path,
        "<include><name>apple</name><uri>model://ycb_013_apple</uri>"
        "<pose>0 0 0.5 0 0 0</pose></include>",
    )
    entities = sdf_world.spawn_sdf_world(
        scene, world, models_root=MODELS_DIR, fixed=True,
    )
    scene.build()
    start = _entity_pos(entities["apple"])
    assert np.allclose(start, [0.0, 0.0, 0.5], atol=0.05)

    for _ in range(200):
        scene.step()
    assert np.allclose(_entity_pos(entities["apple"]), start, atol=1e-4)


def test_spawn_wrs2020_world(scene):
    _require_submodule()
    world = sdf_world.parse_sdf_world(WORLD_XACRO)
    entities = sdf_world.spawn_sdf_world(scene, WORLD_XACRO, world=world)
    assert len(entities) == len(world.models) == 18
    scene.build()

    assert np.allclose(_entity_pos(entities["wrc_ground_plane"]), 0.0, atol=1e-4)
    assert np.allclose(
        _entity_pos(entities["wrc_bookshelf"]), [2.7, -1.0, 0.0], atol=1e-3,
    )
