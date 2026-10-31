"""Regression coverage for the Genesis 1.4.5 raycast compatibility layer."""
from types import SimpleNamespace

import genesis as gs
import pytest
import torch

from hsr_genesis.raycast_filter_patch import (
    apply_raycast_filter_patch, clear_raycast_ignore_list,
    set_raycast_ignore_list, update_raycast_ignore_list,
)



def test_explicit_solver_target_does_not_redirect_to_rigid_solver():
    from hsr_genesis.raycast_filter_patch import _resolve_solver, _get_state

    scene = gs.Scene(show_viewer=False)
    rigid = scene.sim.rigid_solver
    kinematic = scene.sim.kinematic_solver
    assert _resolve_solver(scene) is rigid
    assert _resolve_solver(scene.sim) is rigid
    assert _resolve_solver(rigid) is rigid
    assert _resolve_solver(kinematic) is kinematic
    set_raycast_ignore_list(kinematic, geom_indices=[7])
    try:
        assert _get_state(kinematic).ignore_list.geom_indices == {7}
        assert _get_state(rigid).ignore_list.geom_indices == set()
    finally:
        clear_raycast_ignore_list(kinematic)
        scene.destroy()


def test_attachment_masks_respect_solver_and_subtree():
    apply_raycast_filter_patch()
    from hsr_genesis.raycast_compat import _sensor_ignore_mask

    links = [SimpleNamespace(idx=i, root_idx=0 if i < 3 else 3, parent_idx=p) for i, p in enumerate([-1, 0, 1, -1])]
    entity = SimpleNamespace(idx=0)
    for link in links:
        link.name = f"link{link.idx}"
        link.entity = entity
    geoms = [SimpleNamespace(idx=i, link=link) for i, link in enumerate(links)]

    class Solver:
        pass
    solver = Solver()
    solver.links = links
    solver.geoms = solver.vgeoms = geoms
    other = Solver()
    other.links = links
    other.geoms = other.vgeoms = geoms
    metadata = SimpleNamespace(
        n_sensors=4,
        solver_groups=[SimpleNamespace(solver=solver, sensor_cols=torch.tensor([0, 1, 2]))],
        ignore_link_idx=torch.tensor([1, -1, -1, 0]),
        ignore_root_idx=torch.tensor([-1, 0, -1, -1]),
        ignore_subtree_root_link_idx=torch.tensor([-1, -1, 1, -1]),
    )
    expected = [[False, True, False, False], [True, True, True, False], [False, True, True, False], [False] * 4]
    for visual in (False, True):
        assert _sensor_ignore_mask(metadata, solver, visual).tolist() == expected
        assert not _sensor_ignore_mask(metadata, other, visual).any()
    set_raycast_ignore_list(solver, link_names=['link3'])
    assert _sensor_ignore_mask(metadata, solver, False)[:, 3].all()
    clear_raycast_ignore_list(solver)
    assert _sensor_ignore_mask(metadata, solver, False).tolist() == expected


@pytest.mark.parametrize('visual', [False, True])
def test_sensor_and_viewer_filters_reveal_next_hit(visual):
    apply_raycast_filter_patch()
    from genesis.vis.viewer_plugins.raycast import Raycaster

    scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(gravity=(0, 0, 0)))
    material = gs.materials.Rigid(use_visual_raycasting=visual)
    near = scene.add_entity(gs.morphs.Box(pos=(2, 0, 0), size=(1, 1, 1), fixed=True), material=material)
    far = scene.add_entity(gs.morphs.Box(pos=(4, 0, 0), size=(1, 1, 1)), material=material)
    pattern = gs.sensors.GridPattern(size=(0, 0), direction=(1, 0, 0))
    points = scene.add_sensor(gs.sensors.Raycaster(pattern=pattern))
    distances = scene.add_sensor(gs.sensors.Raycaster(pattern=pattern, return_points=False))
    attached = scene.add_sensor(gs.sensors.Raycaster(
        pattern=pattern, entity_idx=near.idx, pos_offset=(-2, 0, 0), ignore_self_link=True,
    ))
    same_root = scene.add_sensor(gs.sensors.Raycaster(
        pattern=pattern, entity_idx=near.idx, pos_offset=(-2, 0, 0), ignore_same_root=True,
    ))
    depth = scene.add_sensor(gs.sensors.DepthCamera(
        pattern=gs.sensors.DepthCameraPattern(res=(1, 1)), return_points=False,
    ))
    scene.build(n_envs=2, env_spacing=(10, 10))
    scene.step()
    viewer = Raycaster(scene, use_visual_geom=visual)

    def check(expected):
        assert points.read().distances.flatten().tolist() == pytest.approx([expected] * 2)
        assert depth.read_image().flatten().tolist() == pytest.approx([expected] * 2)
        assert distances.read().points is None
        assert distances.read().distances.flatten().tolist() == pytest.approx([expected] * 2)
        assert points.read().points[..., 0].flatten().tolist() == pytest.approx([expected] * 2)
        origin = scene.envs_offset[1]
        hit = viewer.cast(origin, (1, 0, 0), envs_idx=[1])
        assert hit.distance == pytest.approx(expected)
        assert hit.env_idx == 1
        assert hit.position[0] == pytest.approx(origin[0] + expected)

    try:
        check(1.5)
        assert attached.read().distances.flatten().tolist() == pytest.approx([3.5] * 2)
        assert same_root.read().distances.flatten().tolist() == pytest.approx([3.5] * 2)
        # Exercise replacement, addition, clearing and collision/visual mask caches.
        set_raycast_ignore_list(scene, geom_indices=[near.geoms[0].idx])
        scene.step()
        check(3.5)
        clear_raycast_ignore_list(scene)
        scene.step()
        check(1.5)
        set_raycast_ignore_list(scene, entity_indices=[near.idx])
        scene.step()
        check(3.5)
        clear_raycast_ignore_list(scene)
        update_raycast_ignore_list(scene, link_names=[near.links[0].name])
        scene.step()
        # Primitive boxes can share a link name, so every matching entity is ignored.
        assert points.read().distances.flatten().tolist() == pytest.approx([20.0] * 2)
        assert viewer.cast(scene.envs_offset[1], (1, 0, 0), envs_idx=[1]) is None
    finally:
        clear_raycast_ignore_list(scene)
        scene.destroy()
