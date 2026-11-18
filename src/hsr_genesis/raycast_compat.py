"""Genesis 1.4.5 raycast dispatch with geometry filtering.

The dispatch methods follow Genesis' Apache-2.0 sensor/viewer implementations
(see LICENSE.genesis.txt).
Masks are rejected inside BVH traversal, so an ignored nearest face never hides
an eligible face behind it. Sensor attachment filters only apply to their own
solver; dynamic filters apply to the solver selected by the ignore-list API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import genesis as gs
from genesis.utils.misc import qd_to_numpy, qd_to_torch
from genesis.utils.raycast import RayHit
from .raycast_ignore_kernel import kernel_cast_rays, kernel_cast_rays_visual
from .raycast_filter_patch import _build_ignore_geom_mask, _get_state

if TYPE_CHECKING:
    from genesis.engine.sensors.raycaster import RaycastContext, RaycasterSharedMetadata


def _dynamic_mask(solver, visual):
    state = _get_state(solver)
    # set/update/clear invalidate both collision and visual cached tensors.
    if state.dirty:
        state.mesh_masks = {}
        state.dirty = False
    if visual not in state.mesh_masks:
        state.mesh_masks[visual] = _build_ignore_geom_mask(solver, state.ignore_list, visual=visual)
    return state.mesh_masks[visual]


def _viewer_ignore_mask(solver, visual):
    mask = _dynamic_mask(solver, visual)
    if mask is None:
        return torch.zeros((1, len(solver.vgeoms if visual else solver.geoms)), device=gs.device, dtype=gs.tc_bool)
    return mask[None]


def _sensor_ignore_mask(metadata, solver, visual):
    key = (id(solver), visual)
    if not hasattr(metadata, "_hsr_geom_masks"):
        metadata._hsr_geom_masks = {}
    if key not in metadata._hsr_geom_masks:
        geoms = solver.vgeoms if visual else solver.geoms
        rows = [[False] * len(geoms) for _ in range(metadata.n_sensors)]
        # Attachment indices belong to each sensor's solver, not every cast target.
        sensor_cols = []
        for group in metadata.solver_groups:
            if group.solver is solver:
                sensor_cols.extend(group.sensor_cols.tolist())
        ignore_links = metadata.ignore_link_idx.tolist()
        ignore_roots = metadata.ignore_root_idx.tolist()
        ignore_subtrees = metadata.ignore_subtree_root_link_idx.tolist()
        for col in sensor_cols:
            for geom in geoms:
                link = geom.link
                ignore = link.idx == ignore_links[col] or (ignore_roots[col] >= 0 and link.root_idx == ignore_roots[col])
                if ignore_subtrees[col] >= 0:
                    cur = link
                    while cur is not None:
                        if cur.idx == ignore_subtrees[col]:
                            ignore = True
                            break
                        cur = solver.links[cur.parent_idx] if cur.parent_idx >= 0 else None
                rows[col][geom.idx] = ignore
        metadata._hsr_geom_masks[key] = torch.tensor(rows, device=gs.device, dtype=gs.tc_bool)
    static_mask = metadata._hsr_geom_masks[key]
    dynamic_mask = _dynamic_mask(solver, visual)
    return static_mask if dynamic_mask is None else static_mask | dynamic_mask[None]


def update_raw_data(
    cls, shared_context: RaycastContext, shared_metadata: RaycasterSharedMetadata, raw_data_T: torch.Tensor
):
    # The BVHs were already refreshed once this step by SensorManager (``RaycastContext.update``); read them here.
    bvh_contexts = shared_context.bvh_contexts

    # Allocate the link-pose scratch buffers on first cast (B and n_sensors are known here). Identity quat is baked
    # into the initial allocation so static sensors (entity_idx<0) leave their rows at identity, letting the cast
    # kernel apply pos_offset / euler_offset in world frame.
    if shared_metadata.links_pos is None:
        B = bvh_contexts[0].solver._B
        shared_metadata.links_pos = torch.zeros(
            B, shared_metadata.n_sensors, 3, device=gs.device, dtype=gs.tc_float
        )
        shared_metadata.links_quat = torch.zeros(
            B, shared_metadata.n_sensors, 4, device=gs.device, dtype=gs.tc_float
        )
        shared_metadata.links_quat[:, :, 0] = 1.0

    # Gather link poses per sensor. Sensors are pre-bucketed into shared_metadata.solver_groups at build time so
    # this loop issues one bulk get_links_pos / get_links_quat per solver with already-tensor-typed indices.
    links_pos = shared_metadata.links_pos
    links_quat = shared_metadata.links_quat
    for group in shared_metadata.solver_groups:
        pos = group.solver.get_links_pos(links_idx=group.links_idx)
        quat = group.solver.get_links_quat(links_idx=group.links_idx)
        if group.solver.n_envs == 0:
            pos = pos[None]
            quat = quat[None]
        links_pos[:, group.sensor_cols, :] = pos
        links_quat[:, group.sensor_cols, :] = quat

    # The two collision tree sets of a solver cast in one launch, a visual set in one of its own. The launches
    # chain into one output buffer: the first initializes every slot (is_merge=False), each subsequent one merges
    # in closer hits, and the final one (is_last) settles misses to no_hit_value - see write_ray_hit.
    launches = []
    for solver in dict.fromkeys(entry.solver for entry in bvh_contexts):
        collision_entries = [
            entry for entry in bvh_contexts if entry.solver is solver and entry.raycast_mask is None
        ]
        # The static and the movable set of a solver (see activate), cast in one launch (see kernel_cast_rays)
        assert len(collision_entries) <= 2
        if collision_entries:
            launches.append((solver, collision_entries))
        for entry in bvh_contexts:
            if entry.solver is solver and entry.raycast_mask is not None:
                launches.append((solver, [entry]))
    sensor_tables = (
        links_pos,
        links_quat,
        shared_metadata.ray_starts,
        shared_metadata.ray_dirs,
        shared_metadata.max_ranges,
        shared_metadata.no_hit_values,
        shared_metadata.return_world_frame,
        shared_metadata.sensor_cache_offsets,
        shared_metadata.sensor_point_offsets,
        shared_metadata.sensor_point_counts,
        shared_metadata.sensor_return_points,
        raw_data_T,
    )
    for i, (solver, entries) in enumerate(launches):
        entry_a, entry_b = entries[0], entries[-1]
        is_merge = i > 0
        is_last = i == len(launches) - 1
        # A tree serving several envs wants the env-major thread mapping (see kernel_cast_rays); a per-env tree
        # set in the launch, the movable one, decides for the ray-major mapping.
        is_env_major = all(entry.n_trees < solver._B for entry in entries)
        if entry_a.raycast_mask is None:
            kernel_cast_rays(
                shared_metadata.points_to_sensor_idx,
                entry_a.env_bvh_idx,
                entry_b.env_bvh_idx,
                *sensor_tables,
                solver.dyn_state,
                entry_a.bvh_state.tree,
                entry_b.bvh_state.tree,
                solver.dyn_info,
                eps=gs.EPS,
                ignore_geom_mask=_sensor_ignore_mask(shared_metadata, solver, entry_a.raycast_mask is not None),
                is_merge=is_merge,
                is_last=is_last,
                is_env_major=is_env_major,
                is_split=entry_b is not entry_a,
            )
        else:
            kernel_cast_rays_visual(
                shared_metadata.points_to_sensor_idx,
                entry_a.env_bvh_idx,
                *sensor_tables,
                solver.dyn_state,
                entry_a.bvh_state.tree,
                solver.dyn_info,
                eps=gs.EPS,
                ignore_geom_mask=_sensor_ignore_mask(shared_metadata, solver, entry_a.raycast_mask is not None),
                is_merge=is_merge,
                is_last=is_last,
                is_env_major=is_env_major,
            )


def cast_viewer(self, ray_origin, ray_direction, max_range: float = 1000.0, envs_idx=None) -> RayHit | None:
    """
    Cast a single ray against the BVH of each env in parallel and return the closest hit across envs and solvers.

    The ray and the hit are in the displayed world, where each env sits at its offset (see Scene.envs_offset):
    the physics of every env is in its own coordinates, so the ray is shifted into each env and the hit back out.
    `RayHit.env_idx` reports the env the hit comes from, None when the scene holds no parallel envs.

    Parameters
    ----------
    ray_origin : array-like, shape (3,)
        Ray origin in the displayed world.
    ray_direction : array-like, shape (3,)
        Normalized ray direction.
    max_range : float, optional
        Per-env BVH traversal max distance.
    envs_idx : optional
        Indices of envs to raycast. Defaults to all envs.
    """
    from .raycast_ignore_kernel import kernel_cast_ray

    ray_origin = torch.as_tensor(ray_origin, dtype=gs.tc_float, device=gs.device)
    ray_direction = torch.as_tensor(ray_direction, dtype=gs.tc_float, device=gs.device)
    rays_origin = ray_origin - self._envs_offset
    envs_idx = self.scene._sanitize_envs_idx(envs_idx)
    closest_hit: RayHit | None = None
    for target in self.targets:
        solver = target.solver
        is_visual = target.vfaces_mask is not None
        # Each pass caps its traversal at the best distance so far, so any hit it reports is closer by
        # construction and the closest hit across targets needs no further comparison.
        kernel_cast_ray(
            rays_origin,
            envs_idx,
            ray_direction,
            solver.dyn_state,
            target.bvh_state.tree,
            target.result,
            solver.dyn_info,
            max_range if closest_hit is None else closest_hit.distance,
            eps=gs.EPS,
            is_visual=is_visual,
            ignore_geom_mask=_viewer_ignore_mask(solver, is_visual),
        )

        # A no-hit env holds +inf, so argmin lands on the closest hitting env; geom_idx then rejects the
        # all-inf case, which would otherwise report env 0 as a hit.
        distances = qd_to_torch(target.result.distance, copy=None)
        winner = int(distances.argmin())
        geom_idx = int(qd_to_numpy(target.result.geom_idx, row_mask=winner, keepdim=False))
        if geom_idx < 0:
            continue

        distance = float(distances[winner])
        position = qd_to_numpy(target.result.hit_point, row_mask=winner, keepdim=False, transpose=True)
        position = position + self.scene.envs_offset[winner]
        # The normal escapes into the returned hit, which outlives the lock, while the next cast rewrites its buffer
        normal = qd_to_numpy(target.result.normal, row_mask=winner, keepdim=False, transpose=True, copy=True)
        geoms = solver.vgeoms if is_visual else solver.geoms
        geom = geoms[geom_idx] if 0 <= geom_idx < len(geoms) else None

        closest_hit = RayHit(distance, position, normal, geom, winner if self.scene.n_envs > 0 else None)
    return closest_hit
