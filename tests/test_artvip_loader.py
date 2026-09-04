"""Tests for the ArtVIP dataset loader."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hsr_genesis.artvip_loader import (
    ARTVIP_CATEGORIES,
    ARTVIP_REPO_ID,
    ArtVIPJoint,
    ArtVIPJointInfo,
    ArtVIPPart,
    ArtVIPPartInfo,
    list_artvip_categories,
    merge_fixed_meshes,
    parse_artvip_control_script,
    parse_artvip_joint_info,
    parse_artvip_part_info,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

#: Path to the pre-downloaded dishwasher_1 USD in the HF cache.
_DISHWASHER_USD = (
    "/tmp/artvip_cache/datasets--X-Humanoid--ArtVIP/"
    "snapshots/d22f2095bc9db29fdd4b60f4a3c4f8177e1f5e9e/"
    "Articulated_objects/major_appliances/dishwasher/dishwasher_1/"
    "model_dishwasher_1.usd"
)


@pytest.fixture
def dishwasher_usd_path() -> str:
    """Path to the pre-downloaded dishwasher_1 USD, or skip if not available."""
    # Resolve symlinks (HF cache uses symlinks to blob storage).
    path = _DISHWASHER_USD
    if not os.path.exists(path):
        # Try resolving via readlink.
        if os.path.islink(path):
            real = os.readlink(path)
            if os.path.exists(real):
                return real
        pytest.skip(f"Dishwasher USD not found at {path}. Run download first.")
    return path


@pytest.fixture
def dishwasher_control_path() -> str:
    """Path to the dishwasher control script."""
    path = os.path.join(os.path.dirname(_DISHWASHER_USD), "resource", "dishwasher_control.py")
    if not os.path.exists(path):
        pytest.skip(f"Control script not found at {path}")
    return path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_artvip_repo_id():
    assert ARTVIP_REPO_ID == "X-Humanoid/ArtVIP"


def test_artvip_categories_count():
    assert len(ARTVIP_CATEGORIES) == 9


def test_artvip_categories_content():
    expected = {
        "Ikea_furniture", "Medical_equipment", "household_items",
        "industrial_machinery", "lab_items", "large_furniture",
        "major_appliances", "small_appliances", "small_furniture",
    }
    assert set(ARTVIP_CATEGORIES) == expected


def test_list_artvip_categories():
    cats = list_artvip_categories()
    assert len(cats) == 9
    assert "major_appliances" in cats


# ---------------------------------------------------------------------------
# ArtVIPJoint / ArtVIPJointInfo dataclasses
# ---------------------------------------------------------------------------

def test_artvip_joint_creation():
    j = ArtVIPJoint(
        name="test_joint",
        joint_type="revolute",
        prim_path="/root/body/joint",
        limits=(0.0, 90.0),
        axis=np.array([0.0, 0.0, 1.0]),
        body0="/root/body",
        body1="/root/door",
    )
    assert j.name == "test_joint"
    assert j.joint_type == "revolute"
    assert j.limits == (0.0, 90.0)
    assert np.allclose(j.axis, [0, 0, 1])


def test_artvip_joint_info_creation():
    j = ArtVIPJoint(
        name="j1", joint_type="prismatic", prim_path="/j1",
        limits=(-0.2, 0.0), axis=np.array([0, 1, 0]),
        body0="/root", body1="/shelf",
    )
    info = ArtVIPJointInfo(
        usd_path="/tmp/test.usd",
        articulation_root="/root",
        joints=[j],
        movable_joints=[j],
        up_axis="Z",
        meters_per_unit=1.0,
    )
    assert info.articulation_root == "/root"
    assert len(info.joints) == 1
    assert len(info.movable_joints) == 1
    assert info.up_axis == "Z"


# ---------------------------------------------------------------------------
# parse_artvip_joint_info (requires pxr + downloaded USD)
# ---------------------------------------------------------------------------

def test_parse_joint_info_dishwasher(dishwasher_usd_path):
    """Parse the real dishwasher_1 USD and verify joint structure."""
    info = parse_artvip_joint_info(dishwasher_usd_path)

    assert info.usd_path == dishwasher_usd_path
    assert info.up_axis == "Z"
    assert info.meters_per_unit == 1.0
    assert info.articulation_root == "/root/E_body_1"

    # Dishwasher_1 has 4 joints: 1 fixed, 1 revolute, 2 prismatic.
    assert len(info.joints) == 4
    assert len(info.movable_joints) == 3

    # Check joint types.
    types = {j.name: j.joint_type for j in info.joints}
    assert any(t == "fixed" for t in types.values())
    assert any(t == "revolute" for t in types.values())
    assert any(t == "prismatic" for t in types.values())


def test_parse_joint_info_revolute_limits(dishwasher_usd_path):
    """The dishwasher door revolute joint should have 0-90 degree limits."""
    info = parse_artvip_joint_info(dishwasher_usd_path)

    revolute = [j for j in info.movable_joints if j.joint_type == "revolute"]
    assert len(revolute) == 1
    j = revolute[0]
    assert "RevoluteJoint" in j.name
    assert j.limits[0] == 0.0
    assert j.limits[1] == 90.0  # degrees


def test_parse_joint_info_prismatic_limits(dishwasher_usd_path):
    """The dishwasher shelf prismatic joints should have -0.2 to 0 limits."""
    info = parse_artvip_joint_info(dishwasher_usd_path)

    prismatic = [j for j in info.movable_joints if j.joint_type == "prismatic"]
    assert len(prismatic) == 2
    for j in prismatic:
        assert "PrismaticJoint" in j.name
        assert j.limits[0] == pytest.approx(-0.2, abs=0.01)
        assert j.limits[1] == 0.0


def test_parse_joint_info_axis(dishwasher_usd_path):
    """Joints should have correct axis vectors."""
    info = parse_artvip_joint_info(dishwasher_usd_path)

    for j in info.movable_joints:
        # All axes should be unit vectors along X, Y, or Z.
        assert j.axis.shape == (3,)
        norm = np.linalg.norm(j.axis)
        assert norm == pytest.approx(1.0, abs=0.01)


def test_parse_joint_info_bodies(dishwasher_usd_path):
    """Movable joints should have body0 and body1 populated."""
    info = parse_artvip_joint_info(dishwasher_usd_path)

    for j in info.movable_joints:
        assert j.body0 != "", f"body0 empty for {j.name}"
        assert j.body1 != "", f"body1 empty for {j.name}"
        assert j.body0.startswith("/root/")
        assert j.body1.startswith("/root/")


# ---------------------------------------------------------------------------
# parse_artvip_control_script
# ---------------------------------------------------------------------------

def test_parse_control_script_dishwasher(dishwasher_control_path):
    """Parse the real dishwasher control script."""
    meta = parse_artvip_control_script(dishwasher_control_path)

    assert "joint_names" in meta
    assert "RevoluteJoint_dishwasher_1_middle" in meta["joint_names"]

    assert "joint_threshold" in meta
    assert meta["joint_threshold"] == 0.3

    assert "asset_root_names" in meta
    assert "E_body_1" in meta["asset_root_names"]


def test_parse_control_script_missing_file(tmp_path):
    """Should raise FileNotFoundError for non-existent file."""
    with pytest.raises(FileNotFoundError):
        parse_artvip_control_script(tmp_path / "nonexistent.py")


def test_parse_control_script_no_constants(tmp_path):
    """Should return empty dict for a script without ArtVIP constants."""
    p = tmp_path / "test_control.py"
    p.write_text("print('hello world')\n")
    meta = parse_artvip_control_script(p)
    assert meta == {}


def test_parse_control_script_all_constants(tmp_path):
    """Should parse all three constants."""
    p = tmp_path / "test_control.py"
    p.write_text(
        'JOINT_NAMES = ["joint_a", "joint_b"]\n'
        "JOINT_THRESHOLD = 0.5\n"
        'POSSIBLE_ASSET_ROOT_NAMES = ["root_link"]\n'
    )
    meta = parse_artvip_control_script(p)
    assert meta["joint_names"] == ["joint_a", "joint_b"]
    assert meta["joint_threshold"] == 0.5
    assert meta["asset_root_names"] == ["root_link"]


# ---------------------------------------------------------------------------
# parse_artvip_part_info
# ---------------------------------------------------------------------------

def test_parse_part_info_dishwasher(dishwasher_usd_path):
    """Parse part-level semantic annotations from the dishwasher USD."""
    info = parse_artvip_part_info(dishwasher_usd_path)

    assert info.object_label == "dishwasher"
    assert len(info.parts) >= 3

    # Should have door, handle, and rack labels.
    assert "door" in info.labels
    assert "handle" in info.labels
    assert "rack" in info.labels

    # The handle should be a nested sub-part (not a top-level link).
    handles = info.get_parts_by_label("handle")
    assert len(handles) == 1
    handle = handles[0]
    assert handle.is_link is False
    assert handle.parent_link != handle.prim_path

    # The door should be a top-level link.
    doors = info.get_parts_by_label("door")
    assert len(doors) == 1
    door = doors[0]
    assert door.is_link is True
    assert door.parent_link == door.prim_path

    # The handle's parent link should be the door.
    assert handle.parent_link == door.prim_path


def test_parse_part_info_mesh_counts(dishwasher_usd_path):
    """Each part should have non-zero mesh and vertex counts."""
    info = parse_artvip_part_info(dishwasher_usd_path)

    for part in info.parts:
        assert part.n_meshes >= 1, f"{part.name} has no meshes"
        assert part.n_vertices >= 1, f"{part.name} has no vertices"


def test_parse_part_info_labels_property(dishwasher_usd_path):
    """The labels property should return sorted unique labels."""
    info = parse_artvip_part_info(dishwasher_usd_path)

    labels = info.labels
    assert labels == sorted(labels)
    assert len(labels) == len(set(labels))


def test_parse_part_info_get_graspable_parts(dishwasher_usd_path):
    """get_graspable_parts should return handles, doors, etc. but not racks."""
    info = parse_artvip_part_info(dishwasher_usd_path)

    graspable = info.get_graspable_parts()
    graspable_labels = {p.label for p in graspable}

    # Door and handle are graspable; rack is not.
    assert "door" in graspable_labels
    assert "handle" in graspable_labels
    assert "rack" not in graspable_labels


def test_parse_part_info_get_parts_by_label(dishwasher_usd_path):
    """get_parts_by_label should filter correctly."""
    info = parse_artvip_part_info(dishwasher_usd_path)

    racks = info.get_parts_by_label("rack")
    assert len(racks) == 2
    for r in racks:
        assert r.label == "rack"

    # Non-existent label should return empty list.
    assert info.get_parts_by_label("nonexistent") == []


def test_artvip_part_dataclass():
    """ArtVIPPart should be a frozen dataclass with all fields."""
    p = ArtVIPPart(
        prim_path="/root/E_lid_1",
        name="E_lid_1",
        label="lid",
        is_link=True,
        parent_link="/root/E_lid_1",
        n_meshes=2,
        n_vertices=18364,
    )
    assert p.prim_path == "/root/E_lid_1"
    assert p.label == "lid"
    assert p.is_link is True
    assert p.n_meshes == 2
    assert p.n_vertices == 18364

    # Frozen — should not allow mutation.
    with pytest.raises(AttributeError):
        p.label = "door"


def test_artvip_part_info_dataclass():
    """ArtVIPPartInfo should support queries."""
    parts = [
        ArtVIPPart("/root/E_lid_1", "E_lid_1", "lid", True, "/root/E_lid_1", 1, 100),
        ArtVIPPart("/root/E_pedal_5", "E_pedal_5", "pedal", True, "/root/E_pedal_5", 1, 50),
    ]
    info = ArtVIPPartInfo(usd_path="test.usd", object_label="trash_can", parts=parts)

    assert info.object_label == "trash_can"
    assert info.labels == ["lid", "pedal"]
    assert len(info.get_parts_by_label("lid")) == 1
    assert len(info.get_graspable_parts()) == 2  # both lid and pedal are graspable


# ---------------------------------------------------------------------------
# download_artvip_object (mocked HF hub)
# ---------------------------------------------------------------------------

def test_download_artvip_object_mocked(tmp_path):
    """Test download with mocked huggingface_hub."""
    from hsr_genesis.artvip_loader import download_artvip_object

    # Create a fake snapshot directory structure.
    # ArtVIP structure: category/object_type/object_instance
    fake_snapshot = tmp_path / "snapshots" / "abc123"
    obj_dir = fake_snapshot / "Articulated_objects" / "major_appliances" / "dishwasher" / "dishwasher_1"
    obj_dir.mkdir(parents=True)
    (obj_dir / "model_dishwasher_1.usd").write_text("fake USD")

    with patch("huggingface_hub.snapshot_download") as mock_dl:
        mock_dl.return_value = str(fake_snapshot)
        result = download_artvip_object(
            "major_appliances", "dishwasher/dishwasher_1",
            cache_dir=tmp_path,
        )
        assert result == obj_dir / "model_dishwasher_1.usd"
        assert mock_dl.call_count == 1
        call_kwargs = mock_dl.call_args.kwargs
        assert call_kwargs["repo_id"] == ARTVIP_REPO_ID
        assert call_kwargs["repo_type"] == "dataset"
        assert "Articulated_objects/major_appliances/dishwasher/dishwasher_1/*" in call_kwargs["allow_patterns"]


def test_download_artvip_object_no_model_usd(tmp_path):
    """Should raise FileNotFoundError if no model_*.usd exists."""
    from hsr_genesis.artvip_loader import download_artvip_object

    fake_snapshot = tmp_path / "snapshots" / "abc123"
    obj_dir = fake_snapshot / "Articulated_objects" / "major_appliances" / "dishwasher" / "dishwasher_1"
    obj_dir.mkdir(parents=True)
    # No model_*.usd file.

    with patch("huggingface_hub.snapshot_download") as mock_dl:
        mock_dl.return_value = str(fake_snapshot)
        with pytest.raises(FileNotFoundError, match="No model_.*usd found"):
            download_artvip_object(
                "major_appliances", "dishwasher/dishwasher_1",
                cache_dir=tmp_path,
            )


# ---------------------------------------------------------------------------
# list_artvip_objects
# ---------------------------------------------------------------------------

def test_list_artvip_objects_local_cache(tmp_path):
    """list_artvip_objects should scan local cache if available."""
    from hsr_genesis.artvip_loader import list_artvip_objects

    # Create a fake cache with the 3-level structure: category/type/instance.
    base = tmp_path / "datasets--X-Humanoid--ArtVIP" / "snapshots" / "abc123" / "Articulated_objects" / "major_appliances"
    (base / "dishwasher" / "dishwasher_1").mkdir(parents=True)
    (base / "dishwasher" / "dishwasher_2").mkdir(parents=True)
    (base / "refrigerator" / "refrigerator_1").mkdir(parents=True)
    (base / ".thumbs").mkdir(parents=True)

    # Without object_type: returns type/instance paths.
    result = list_artvip_objects("major_appliances", cache_dir=tmp_path)
    assert "dishwasher/dishwasher_1" in result
    assert "dishwasher/dishwasher_2" in result
    assert "refrigerator/refrigerator_1" in result

    # With object_type: returns just instance names.
    result = list_artvip_objects("major_appliances", object_type="dishwasher", cache_dir=tmp_path)
    assert "dishwasher_1" in result
    assert "dishwasher_2" in result
    assert "refrigerator_1" not in result


# ---------------------------------------------------------------------------
# merge_fixed_meshes (requires pxr + downloaded USD)
# ---------------------------------------------------------------------------

#: Path to the pre-downloaded stepping_dustbin_4 USD in the HF cache.
_DUSTBIN_USD = (
    "/home/yosuke/.cache/artvip/datasets--X-Humanoid--ArtVIP/"
    "snapshots/d22f2095bc9db29fdd4b60f4a3c4f8177e1f5e9e/"
    "Articulated_objects/household_items/trash_can/stepping_dustbin_4/"
    "model_stepping_dustbin_4.usd"
)


@pytest.fixture
def dustbin_usd_path() -> str:
    """Path to the pre-downloaded stepping_dustbin_4 USD, or skip."""
    path = _DUSTBIN_USD
    if not os.path.exists(path):
        pytest.skip(f"Dustbin USD not found at {path}. Run download first.")
    return path


def _count_active_meshes(usd_path: str) -> tuple[int, int]:
    """Count active meshes and joints in a USD file."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    n_meshes = 0
    n_joints = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh) and prim.IsActive():
            n_meshes += 1
        if prim.IsA(UsdPhysics.Joint):
            n_joints += 1
    return n_meshes, n_joints


def test_merge_fixed_meshes_reduces_mesh_count(dustbin_usd_path, tmp_path):
    """Merging should reduce 6 meshes to 3 (one per link)."""
    output = tmp_path / "merged.usd"
    result = merge_fixed_meshes(dustbin_usd_path, output)

    assert result == output
    assert output.exists()

    n_before, n_joints_before = _count_active_meshes(dustbin_usd_path)
    n_after, n_joints_after = _count_active_meshes(str(output))

    # Original has 6 meshes across 3 links.
    assert n_before == 6
    # Merged should have 3 (one per link).
    assert n_after == 3
    # Joints should be preserved.
    assert n_joints_before == n_joints_after == 3


def test_merge_fixed_meshes_preserves_joints(dustbin_usd_path, tmp_path):
    """The merged USD should still have 2 revolute + 1 fixed joint."""
    from pxr import Usd, UsdPhysics

    output = tmp_path / "merged.usd"
    merge_fixed_meshes(dustbin_usd_path, output)

    stage = Usd.Stage.Open(str(output))
    joint_types = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            if prim.IsA(UsdPhysics.RevoluteJoint):
                joint_types.append("revolute")
            elif prim.IsA(UsdPhysics.FixedJoint):
                joint_types.append("fixed")
            elif prim.IsA(UsdPhysics.PrismaticJoint):
                joint_types.append("prismatic")

    assert sorted(joint_types) == ["fixed", "revolute", "revolute"]


def test_merge_fixed_meshes_default_output_path(dustbin_usd_path):
    """Without output_path, should write _merged.usd next to the input."""
    output = merge_fixed_meshes(dustbin_usd_path)
    try:
        assert output.exists()
        assert "_merged" in output.name
    finally:
        # Clean up.
        if output.exists():
            os.remove(output)


def test_merge_fixed_meshes_merged_vertices(dustbin_usd_path, tmp_path):
    """The first mesh in each link should contain the merged vertex count."""
    from pxr import Usd, UsdGeom

    output = tmp_path / "merged.usd"
    merge_fixed_meshes(dustbin_usd_path, output)

    stage = Usd.Stage.Open(str(output))
    root = stage.GetDefaultPrim() or stage.GetPseudoRoot()
    root_children = {str(c.GetPath()) for c in root.GetChildren()}

    link_vert_counts: dict[str, int] = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh) and prim.IsActive():
            # Walk up to the top-level link (direct child of root).
            parent = prim.GetParent()
            while parent and parent.IsValid():
                if str(parent.GetPath()) in root_children:
                    break
                parent = parent.GetParent()
            link_name = parent.GetName() if parent and parent.IsValid() else "unknown"
            pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            link_vert_counts[link_name] = len(pts) if pts else 0

    # Body link had 3 meshes (18030 + 2742 + 428 = 21200 verts).
    assert link_vert_counts.get("E_body_2", 0) == 21200
    # Lid link had 2 meshes (14132 + 4232 = 18364 verts).
    assert link_vert_counts.get("E_lid_1", 0) == 18364
    # Pedal link had 1 mesh (3190 verts, unchanged).
    assert link_vert_counts.get("E_pedal_5", 0) == 3190


def test_merge_fixed_meshes_does_not_modify_original(dustbin_usd_path, tmp_path):
    """The original USD file should not be modified."""
    import hashlib

    def file_hash(path):
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    original_hash = file_hash(dustbin_usd_path)
    output = tmp_path / "merged.usd"
    merge_fixed_meshes(dustbin_usd_path, output)
    assert file_hash(dustbin_usd_path) == original_hash
