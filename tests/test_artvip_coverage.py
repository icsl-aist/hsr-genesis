"""Test coverage for ArtVIP objects across all 9 categories.

This module verifies that the ArtVIP loader correctly handles the diverse
object structures found across the dataset:

  - Different USD naming conventions (``model_*.usd`` vs ``<Name>_<n>.usd``)
  - Different joint configurations (revolute-only, prismatic-only, mixed)
  - Different articulation complexities (1 to 21 movable joints)
  - Different category hierarchies (type-level USD vs type/instance USD)

The tests use a representative object from each of the 9 ArtVIP categories.
Objects are downloaded to the HuggingFace cache on first run and reused
from cache on subsequent runs.

Run:
    PYTHONPATH=src .venv/bin/python -m pytest tests/test_artvip_coverage.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from hsr_genesis.artvip_loader import (
    ARTVIP_CATEGORIES,
    ArtVIPJointInfo,
    download_artvip_object,
    parse_artvip_joint_info,
)


# ---------------------------------------------------------------------------
# Representative objects: one per ArtVIP category
# ---------------------------------------------------------------------------

#: One representative object from each of the 9 ArtVIP categories.
#: Tuple: (category, object_path, usd_filename, expected_min_movable_joints,
#:          expected_joint_types)
REPRESENTATIVE_OBJECTS = [
    (
        "Ikea_furniture",
        "EKET_Cabinet_with_door_brown_walnut_effect_35x35x35cm",
        "model_EKET_Cabinet_with_door_brown_walnut_effect_35x35x35cm.usd",
        1,
        {"revolute"},
    ),
    (
        "Medical_equipment",
        "AED",
        "model_AED_0.usd",
        3,  # At least 3 movable joints
        {"prismatic", "revolute"},
    ),
    (
        "household_items",
        "trash_can/stepping_dustbin_4",
        "model_stepping_dustbin_4.usd",
        1,
        {"revolute"},
    ),
    (
        "industrial_machinery",
        "ticket_gate/ticketgate001",
        "model_ticketgate001.usd",
        5,  # Has 10 movable joints, require at least 5
        {"revolute"},
    ),
    (
        "lab_items",
        "lab_cabinet/lab_cabinet001",
        "model_lab_cabinet001.usd",
        2,
        {"revolute"},
    ),
    (
        "large_furniture",
        "cupboard/cupboard_1",
        "model_cupboard_1.usd",
        10,  # Has 21 movable joints, require at least 10
        {"prismatic", "revolute"},
    ),
    (
        "major_appliances",
        "dishwasher/dishwasher_1",
        "model_dishwasher_1.usd",
        2,
        {"prismatic", "revolute"},
    ),
    (
        "small_appliances",
        "microwave/microwave_1",
        "model_microwave_1.usd",
        2,
        {"prismatic", "revolute"},
    ),
    (
        "small_furniture",
        "cabinet/D1_1",
        "Cabinet_3.usd",  # Note: non-model_ prefix
        1,
        {"prismatic"},
    ),
]


def _cache_dir() -> Path:
    """Get the ArtVIP cache directory, preferring /tmp/artvip_cache if it exists."""
    tmp_cache = Path("/tmp/artvip_cache")
    if tmp_cache.exists():
        return tmp_cache
    # Fall back to default.
    from hsr_genesis.artvip_loader import _DEFAULT_CACHE_DIR
    return _DEFAULT_CACHE_DIR


def _get_usd_path(category: str, object_path: str, usd_filename: str) -> str:
    """Get the USD path for a representative object, downloading if needed.

    Tries to find the file in the existing cache first.  If not found,
    downloads it from HuggingFace.
    """
    cache = _cache_dir()
    # Try to find in cache first (fast path, no network).
    from hsr_genesis.artvip_loader import _find_snapshot_dir
    base = f"Articulated_objects/{category}/{object_path}"
    obj_dir = _find_snapshot_dir(cache, base)
    if obj_dir is not None:
        usd_path = obj_dir / usd_filename
        if usd_path.exists():
            return str(usd_path)
        # Try resolving symlinks.
        if usd_path.is_symlink():
            real = Path(os.readlink(usd_path))
            if not real.is_absolute():
                real = usd_path.parent / real
            if real.exists():
                return str(real)

    # Download from HuggingFace.
    path = download_artvip_object(category, object_path, cache_dir=cache)
    return str(path)


@pytest.fixture(scope="module")
def artvip_cache_ready():
    """Ensure at least one object is available; skip entire module if not."""
    cache = _cache_dir()
    # Check if any representative object is available.
    for cat, obj, usd_name, _, _ in REPRESENTATIVE_OBJECTS:
        base = f"Articulated_objects/{cat}/{obj}"
        from hsr_genesis.artvip_loader import _find_snapshot_dir
        obj_dir = _find_snapshot_dir(cache, base)
        if obj_dir is not None and (obj_dir / usd_name).exists():
            return True
    # If nothing in cache, try downloading the first one.
    try:
        cat, obj, _, _, _ = REPRESENTATIVE_OBJECTS[0]
        download_artvip_object(cat, obj, cache_dir=cache)
        return True
    except Exception as e:
        pytest.skip(f"ArtVIP objects not available and download failed: {e}")


# ---------------------------------------------------------------------------
# Category coverage tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def downloaded_objects(artvip_cache_ready):
    """Download (or locate) all representative objects and return their paths."""
    results = []
    for cat, obj, usd_name, min_joints, expected_types in REPRESENTATIVE_OBJECTS:
        try:
            usd_path = _get_usd_path(cat, obj, usd_name)
            results.append((cat, obj, usd_name, min_joints, expected_types, usd_path))
        except Exception as e:
            results.append((cat, obj, usd_name, min_joints, expected_types, None))
    return results


@pytest.mark.parametrize(
    "category,object_path,usd_filename,min_movable,expected_types",
    REPRESENTATIVE_OBJECTS,
    ids=[f"{c}/{o}" for c, o, _, _, _ in REPRESENTATIVE_OBJECTS],
)
def test_all_categories_parse(
    artvip_cache_ready,
    category: str,
    object_path: str,
    usd_filename: str,
    min_movable: int,
    expected_types: set[str],
):
    """Every ArtVIP category should have at least one parseable object."""
    usd_path = _get_usd_path(category, object_path, usd_filename)
    info = parse_artvip_joint_info(usd_path)

    assert info.usd_path == usd_path
    assert info.up_axis in ("Y", "Z"), f"Unexpected up axis: {info.up_axis}"
    assert info.meters_per_unit > 0, "meters_per_unit should be positive"
    assert info.articulation_root, "Should have an articulation root"

    # Should have at least the minimum expected movable joints.
    actual_movable = len(info.movable_joints)
    assert actual_movable >= min_movable, (
        f"{category}/{object_path}: expected >={min_movable} movable joints, "
        f"got {actual_movable}"
    )

    # Should have the expected joint types.
    actual_types = {j.joint_type for j in info.movable_joints}
    assert expected_types.issubset(actual_types), (
        f"{category}/{object_path}: expected joint types {expected_types} "
        f"to be subset of {actual_types}"
    )


@pytest.mark.parametrize(
    "category,object_path,usd_filename,min_movable,expected_types",
    REPRESENTATIVE_OBJECTS,
    ids=[f"{c}/{o}" for c, o, _, _, _ in REPRESENTATIVE_OBJECTS],
)
def test_all_categories_joint_axes_valid(
    artvip_cache_ready,
    category: str,
    object_path: str,
    usd_filename: str,
    min_movable: int,
    expected_types: set[str],
):
    """All movable joints should have valid unit-length axes."""
    usd_path = _get_usd_path(category, object_path, usd_filename)
    info = parse_artvip_joint_info(usd_path)

    for j in info.movable_joints:
        assert j.axis.shape == (3,), (
            f"{j.name}: axis shape {j.axis.shape}, expected (3,)"
        )
        norm = float(np.linalg.norm(j.axis))
        assert norm == pytest.approx(1.0, abs=0.01), (
            f"{j.name}: axis {j.axis.tolist()} has norm {norm}, expected ~1.0"
        )


@pytest.mark.parametrize(
    "category,object_path,usd_filename,min_movable,expected_types",
    REPRESENTATIVE_OBJECTS,
    ids=[f"{c}/{o}" for c, o, _, _, _ in REPRESENTATIVE_OBJECTS],
)
def test_all_categories_joint_limits_valid(
    artvip_cache_ready,
    category: str,
    object_path: str,
    usd_filename: str,
    min_movable: int,
    expected_types: set[str],
):
    """All movable joints should have lower <= upper limits."""
    usd_path = _get_usd_path(category, object_path, usd_filename)
    info = parse_artvip_joint_info(usd_path)

    for j in info.movable_joints:
        lower, upper = j.limits
        assert lower <= upper, (
            f"{j.name}: lower limit {lower} > upper limit {upper}"
        )


@pytest.mark.parametrize(
    "category,object_path,usd_filename,min_movable,expected_types",
    REPRESENTATIVE_OBJECTS,
    ids=[f"{c}/{o}" for c, o, _, _, _ in REPRESENTATIVE_OBJECTS],
)
def test_all_categories_bodies_populated(
    artvip_cache_ready,
    category: str,
    object_path: str,
    usd_filename: str,
    min_movable: int,
    expected_types: set[str],
):
    """Movable joints should have body0 and body1 populated."""
    usd_path = _get_usd_path(category, object_path, usd_filename)
    info = parse_artvip_joint_info(usd_path)

    for j in info.movable_joints:
        # body0 may be empty for the root joint, but body1 should always be set.
        assert j.body1, (
            f"{j.name}: body1 is empty (body0={j.body0!r})"
        )


# ---------------------------------------------------------------------------
# Non-model_ USD naming convention
# ---------------------------------------------------------------------------

def test_non_model_prefix_usd_naming(artvip_cache_ready):
    """Objects with non-model_ USD naming should still be found."""
    # small_furniture/cabinet/D1_1 uses Cabinet_3.usd instead of model_*.usd
    usd_path = _get_usd_path("small_furniture", "cabinet/D1_1", "Cabinet_3.usd")
    assert "Cabinet_3.usd" in usd_path
    info = parse_artvip_joint_info(usd_path)
    assert len(info.movable_joints) >= 1


# ---------------------------------------------------------------------------
# Joint type distribution tests
# ---------------------------------------------------------------------------

def test_joint_type_distribution(artvip_cache_ready, downloaded_objects):
    """Across all categories, we should see both revolute and prismatic joints."""
    all_types = set()
    for cat, obj, usd_name, _, _, usd_path in downloaded_objects:
        if usd_path is None:
            continue
        info = parse_artvip_joint_info(usd_path)
        all_types.update(j.joint_type for j in info.movable_joints)

    assert "revolute" in all_types, "Should have revolute joints"
    assert "prismatic" in all_types, "Should have prismatic joints"


def test_joint_count_range(artvip_cache_ready, downloaded_objects):
    """Joint counts should vary across categories (not all the same)."""
    counts = []
    for cat, obj, usd_name, _, _, usd_path in downloaded_objects:
        if usd_path is None:
            continue
        info = parse_artvip_joint_info(usd_path)
        counts.append(len(info.movable_joints))

    assert len(counts) >= 8, "Should have data for at least 8 categories"
    # Should have a range of joint counts (not all identical).
    assert max(counts) > min(counts), (
        f"All objects have the same joint count ({counts}); expected variation"
    )
    # The most complex object should have at least 5 joints.
    assert max(counts) >= 5, f"Max joint count {max(counts)} is too low"


# ---------------------------------------------------------------------------
# Category completeness
# ---------------------------------------------------------------------------

def test_all_9_categories_represented(artvip_cache_ready, downloaded_objects):
    """All 9 ArtVIP categories should be represented in the test suite."""
    tested_categories = {cat for cat, _, _, _, _, usd_path in downloaded_objects if usd_path}
    all_categories = set(ARTVIP_CATEGORIES)

    missing = all_categories - tested_categories
    assert not missing, f"Missing categories: {missing}"


# ---------------------------------------------------------------------------
# Genesis loading tests (slow, marked separately)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize(
    "category,object_path,usd_filename,min_movable,expected_types",
    # Only test a subset for Genesis loading (the smaller objects).
    [
        ("household_items", "trash_can/stepping_dustbin_4", "model_stepping_dustbin_4.usd", 1, {"revolute"}),
        ("small_appliances", "microwave/microwave_1", "model_microwave_1.usd", 2, {"prismatic", "revolute"}),
        ("small_furniture", "cabinet/D1_1", "Cabinet_3.usd", 1, {"prismatic"}),
        ("Ikea_furniture", "EKET_Cabinet_with_door_brown_walnut_effect_35x35x35cm", "model_EKET_Cabinet_with_door_brown_walnut_effect_35x35x35cm.usd", 1, {"revolute"}),
    ],
    ids=["trash_can", "microwave", "cabinet_D1_1", "eket_cabinet"],
)
def test_genesis_loading_subset(
    artvip_cache_ready,
    category: str,
    object_path: str,
    usd_filename: str,
    min_movable: int,
    expected_types: set[str],
):
    """A subset of objects should load in Genesis and have the expected joints."""
    pytest.importorskip("genesis")

    import genesis as gs

    if not getattr(gs, "_initialized", False):
        try:
            gs.init(backend=gs.gpu, precision="32", logging_level="warning")
        except Exception:
            gs.init(backend=gs.cpu, precision="32", logging_level="warning")

    usd_path = _get_usd_path(category, object_path, usd_filename)
    info = parse_artvip_joint_info(usd_path)

    scene = gs.Scene(show_viewer=False, show_FPS=False)
    scene.add_entity(gs.morphs.Plane())
    entity = scene.add_entity(
        gs.morphs.USD(
            file=usd_path,
            pos=(0.0, 0.0, 0.0),
            fixed=True,
            decimate=True,
            convexify=True,
        ),
    )
    scene.build()

    # Genesis entity should have joints matching the USD parse.
    n_genesis_joints = len(entity.joints)
    n_parsed_movable = len(info.movable_joints)
    assert n_genesis_joints == n_parsed_movable, (
        f"Genesis loaded {n_genesis_joints} joints but parser found "
        f"{n_parsed_movable} movable joints"
    )

    # All joints should start at qpos=0.
    for joint in entity.joints:
        qpos = entity.get_dofs_position(joint.dofs_idx_local)
        for v in qpos.cpu().numpy():
            assert abs(v) < 1e-3, f"Joint {joint.name} starts at {v}, expected ~0"
