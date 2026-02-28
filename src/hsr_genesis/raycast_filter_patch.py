"""Raycast ignore list monkey patches for Genesis.

Apply after ``gs.init()``. Supports ignoring geom indices, entity indices, or link names
for both sensor raycasters (Raycaster/DepthCamera) and viewer raycasts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable
import weakref


@dataclass
class RaycastIgnoreList:
    geom_indices: set[int] = field(default_factory=set)
    entity_indices: set[int] = field(default_factory=set)
    link_names: set[str] = field(default_factory=set)


@dataclass
class _SolverIgnoreState:
    solver_ref: weakref.ReferenceType
    ignore_list: RaycastIgnoreList = field(default_factory=RaycastIgnoreList)
    geom_mask: object | None = None
    dirty: bool = True


_SOLVER_IGNORE_STATE: dict[int, _SolverIgnoreState] = {}
_PATCH_APPLIED = False


def _resolve_solver(target):
    if target is None:
        return None
    if hasattr(target, "sim"):
        sim = getattr(target, "sim", None)
        return getattr(sim, "rigid_solver", None) if sim is not None else None
    if hasattr(target, "rigid_solver"):
        return getattr(target, "rigid_solver", None)
    return target


def _get_state(solver) -> _SolverIgnoreState | None:
    if solver is None:
        return None
    key = id(solver)
    state = _SOLVER_IGNORE_STATE.get(key)
    if state is None or state.solver_ref() is None:
        state = _SolverIgnoreState(solver_ref=weakref.ref(solver))
        _SOLVER_IGNORE_STATE[key] = state
    return state


def _normalize_ints(values: Iterable[int] | None) -> set[int]:
    if not values:
        return set()
    return {int(v) for v in values}


def _normalize_names(values: Iterable[str] | None) -> set[str]:
    if not values:
        return set()
    return {str(v) for v in values if str(v)}


def set_raycast_ignore_list(
    target,
    *,
    geom_indices: Iterable[int] | None = None,
    entity_indices: Iterable[int] | None = None,
    link_names: Iterable[str] | None = None,
) -> None:
    """Replace the ignore list for a scene or solver."""
    solver = _resolve_solver(target)
    state = _get_state(solver)
    if state is None:
        return
    state.ignore_list = RaycastIgnoreList(
        geom_indices=_normalize_ints(geom_indices),
        entity_indices=_normalize_ints(entity_indices),
        link_names=_normalize_names(link_names),
    )
    state.dirty = True


def update_raycast_ignore_list(
    target,
    *,
    geom_indices: Iterable[int] | None = None,
    entity_indices: Iterable[int] | None = None,
    link_names: Iterable[str] | None = None,
) -> None:
    """Add entries to the ignore list for a scene or solver."""
    solver = _resolve_solver(target)
    state = _get_state(solver)
    if state is None:
        return
    state.ignore_list.geom_indices.update(_normalize_ints(geom_indices))
    state.ignore_list.entity_indices.update(_normalize_ints(entity_indices))
    state.ignore_list.link_names.update(_normalize_names(link_names))
    state.dirty = True


def clear_raycast_ignore_list(target) -> None:
    """Clear the ignore list for a scene or solver."""
    solver = _resolve_solver(target)
    state = _get_state(solver)
    if state is None:
        return
    state.ignore_list = RaycastIgnoreList()
    state.dirty = True


def _build_ignore_geom_mask(solver, ignore_list: RaycastIgnoreList):
    import torch
    import genesis as gs

    geoms = list(getattr(solver, "geoms", []) or [])
    if not geoms:
        return None

    ignore_geom = set(ignore_list.geom_indices)

    if ignore_list.entity_indices or ignore_list.link_names:
        for geom in geoms:
            try:
                geom_idx = int(getattr(geom, "idx", -1))
            except Exception:
                continue
            link = getattr(geom, "link", None)
            link_name = getattr(link, "name", None)
            entity = getattr(link, "entity", None) if link is not None else None
            entity_idx = getattr(entity, "idx", None) if entity is not None else None
            if entity_idx is not None:
                try:
                    entity_idx = int(entity_idx)
                except Exception:
                    entity_idx = None

            if link_name in ignore_list.link_names:
                ignore_geom.add(geom_idx)
                continue
            if entity_idx is not None and entity_idx in ignore_list.entity_indices:
                ignore_geom.add(geom_idx)

    if not ignore_geom:
        return None

    mask = torch.zeros((len(geoms),), device=gs.device, dtype=gs.tc_bool)
    indices = [idx for idx in ignore_geom if 0 <= idx < len(geoms)]
    if indices:
        mask[torch.as_tensor(indices, device=gs.device, dtype=gs.tc_int)] = True
    return mask


def _get_ignore_geom_mask(solver):
    state = _get_state(solver)
    if state is None:
        return None
    if not state.dirty and state.geom_mask is not None:
        return state.geom_mask

    mask = _build_ignore_geom_mask(solver, state.ignore_list)
    state.geom_mask = mask
    if mask is None:
        if not (state.ignore_list.geom_indices or state.ignore_list.entity_indices or state.ignore_list.link_names):
            state.dirty = False
        else:
            state.dirty = True
        return None
    state.dirty = False
    return mask


def apply_raycast_filter_patch() -> None:
    """Patch Genesis raycasters to honor ignore lists."""
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    import genesis as gs
    import torch
    import genesis.engine.sensors.raycaster as raycaster_mod
    from genesis.utils import raycast_qd

    from .genesis_patches import apply_raycaster_ignore_patch
    from .raycast_ignore_kernel import (
        kernel_cast_ray_ignore_geom,
        kernel_cast_rays_ignore_geom,
    )

    apply_raycaster_ignore_patch()

    if getattr(raycaster_mod, "_hsr_ignore_filter_patch", False):
        _PATCH_APPLIED = True
        return

    original_update = raycaster_mod.RaycasterSensor._update_shared_ground_truth_cache

    @classmethod
    def _update_shared_ground_truth_cache(cls, shared_metadata, shared_ground_truth_cache):
        cls._update_bvh(shared_metadata)

        links_pos = shared_metadata.solver.get_links_pos(links_idx=shared_metadata.links_idx)
        links_quat = shared_metadata.solver.get_links_quat(links_idx=shared_metadata.links_idx)
        if shared_metadata.solver.n_envs == 0:
            links_pos = links_pos[None]
            links_quat = links_quat[None]

        if not hasattr(shared_metadata, "ignore_link_idx"):
            n_sensors = int(shared_metadata.sensor_point_counts.shape[0])
            shared_metadata.ignore_link_idx = torch.full((n_sensors,), -1, device=gs.device, dtype=gs.tc_int)
            shared_metadata.ignore_root_idx = torch.full((n_sensors,), -1, device=gs.device, dtype=gs.tc_int)
            shared_metadata.ignore_subtree_root_link_idx = torch.full(
                (n_sensors,), -1, device=gs.device, dtype=gs.tc_int
            )

        ignore_geom_mask = _get_ignore_geom_mask(shared_metadata.solver)
        has_sensor_ignore = (
            (
                hasattr(shared_metadata, "ignore_link_idx")
                and shared_metadata.ignore_link_idx is not None
                and (shared_metadata.ignore_link_idx >= 0).any()
            )
            or (
                hasattr(shared_metadata, "ignore_root_idx")
                and shared_metadata.ignore_root_idx is not None
                and (shared_metadata.ignore_root_idx >= 0).any()
            )
            or (
                hasattr(shared_metadata, "ignore_subtree_root_link_idx")
                and shared_metadata.ignore_subtree_root_link_idx is not None
                and (shared_metadata.ignore_subtree_root_link_idx >= 0).any()
            )
        )

        if ignore_geom_mask is None and not has_sensor_ignore:
            return original_update(shared_metadata, shared_ground_truth_cache)

        if ignore_geom_mask is None:
            ignore_geom_mask = torch.zeros(
                (len(shared_metadata.solver.geoms),),
                device=gs.device,
                dtype=gs.tc_bool,
            )

        output_hits = shared_ground_truth_cache.contiguous()
        kernel_cast_rays_ignore_geom(
            shared_metadata.solver.fixed_verts_state,
            shared_metadata.solver.free_verts_state,
            shared_metadata.solver.verts_info,
            shared_metadata.solver.faces_info,
            shared_metadata.solver.geoms_info,
            shared_metadata.solver.links_info,
            shared_metadata.bvh.nodes,
            shared_metadata.bvh.morton_codes,
            links_pos,
            links_quat,
            shared_metadata.ray_starts,
            shared_metadata.ray_dirs,
            shared_metadata.max_ranges,
            shared_metadata.no_hit_values,
            shared_metadata.return_world_frame,
            shared_metadata.points_to_sensor_idx,
            shared_metadata.sensor_cache_offsets,
            shared_metadata.sensor_point_offsets,
            shared_metadata.sensor_point_counts,
            shared_metadata.ignore_link_idx,
            shared_metadata.ignore_root_idx,
            shared_metadata.ignore_subtree_root_link_idx,
            ignore_geom_mask,
            output_hits,
            gs.EPS,
        )
        if not shared_ground_truth_cache.is_contiguous():
            shared_ground_truth_cache.copy_(output_hits)

    raycaster_mod.RaycasterSensor._update_shared_ground_truth_cache = _update_shared_ground_truth_cache

    original_cast = raycast_qd.Raycaster.cast

    def _cast(self, ray_origin, ray_direction, max_range: float = 1000.0, envs_idx=None):
        ignore_geom_mask = _get_ignore_geom_mask(self.solver)
        if ignore_geom_mask is None:
            ignore_geom_mask = torch.zeros(
                (len(self.solver.geoms),),
                device=gs.device,
                dtype=gs.tc_bool,
            )

        kernel_cast_ray_ignore_geom(
            self.solver.fixed_verts_state,
            self.solver.free_verts_state,
            self.solver.verts_info,
            self.solver.faces_info,
            self.bvh.nodes,
            self.bvh.morton_codes,
            gs.np.ascontiguousarray(ray_origin, dtype=gs.np_float),
            gs.np.ascontiguousarray(ray_direction, dtype=gs.np_float),
            float(max_range),
            envs_idx if envs_idx is not None else self.envs_idx,
            ignore_geom_mask,
            self.result,
            gs.EPS,
        )
        return self._raycast_from_result(self.result)

    raycast_qd.Raycaster.cast = _cast

    raycaster_mod._hsr_ignore_filter_patch = True
    _PATCH_APPLIED = True
