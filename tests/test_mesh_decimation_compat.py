"""Regression for face cleanup crossing Genesis's decimation target."""
import genesis as gs
import numpy as np
import trimesh

from hsr_genesis.genesis_patches import apply_mesh_decimation_patch


def test_mesh_cleanup_below_target_skips_simplification(monkeypatch):
    import fast_simplification

    apply_mesh_decimation_patch()
    original = gs.Mesh.decimate
    apply_mesh_decimation_patch()
    assert gs.Mesh.decimate is original
    box = trimesh.creation.box()
    # Before validation this has 120 faces; duplicate removal leaves only 12.
    dirty = trimesh.Trimesh(vertices=box.vertices, faces=np.tile(box.faces, (10, 1)), process=False)

    def unexpected_simplification(*args, **kwargs):
        raise AssertionError("Cleanup already brought the mesh below the target")

    monkeypatch.setattr(fast_simplification, "simplify", unexpected_simplification)
    mesh = gs.Mesh.from_trimesh(
        dirty, surface=gs.surfaces.Collision(), decimate=True,
        decimate_face_num=50, decimate_aggressiveness=7,
    )
    assert len(mesh.trimesh.faces) == 12
    assert not mesh.metadata.get("decimated", False)


def test_mesh_above_target_still_decimates():
    apply_mesh_decimation_patch()
    sphere = trimesh.creation.icosphere(subdivisions=2)
    assert len(sphere.faces) > 100
    mesh = gs.Mesh.from_trimesh(
        sphere, surface=gs.surfaces.Collision(), decimate=True,
        decimate_face_num=100, decimate_aggressiveness=7,
    )
    assert len(mesh.trimesh.faces) <= 100
    assert mesh.metadata["decimated"]
