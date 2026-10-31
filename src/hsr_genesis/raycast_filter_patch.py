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
    mesh_masks: dict[bool, object] = field(default_factory=dict)
    dirty: bool = True


_SOLVER_IGNORE_STATE: dict[int, _SolverIgnoreState] = {}
_PATCH_APPLIED = False


def _resolve_solver(target):
    if target is None:
        return None
    # Genesis solvers expose .sim too; resolve an explicit solver before
    # interpreting .sim as a scene and selecting its default rigid solver.
    from genesis.engine.solvers.base_solver import Solver

    if isinstance(target, Solver):
        return target
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


def _build_ignore_geom_mask(solver, ignore_list: RaycastIgnoreList, *, visual=False):
    import torch
    import genesis as gs

    geoms = list(getattr(solver, "vgeoms" if visual else "geoms", []) or [])
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


def apply_raycast_filter_patch() -> None:
    """Install the Genesis 1.4.5 sensor and viewer filtering hooks."""
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return
    import genesis.engine.sensors.raycaster as raycaster_mod
    from genesis.vis.viewer_plugins.raycast import Raycaster
    from genesis.utils.misc import with_lock
    from .genesis_patches import apply_raycaster_ignore_patch
    from .raycast_compat import update_raw_data, cast_viewer

    apply_raycaster_ignore_patch()
    if not hasattr(raycaster_mod.RaycasterSensor, "_update_raw_data"):
        raise RuntimeError("Raycast filtering requires the Genesis 1.4.5 RaycastContext API")
    raycaster_mod.RaycasterSensor._update_raw_data = classmethod(update_raw_data)
    Raycaster.cast = with_lock(cast_viewer)
    raycaster_mod._hsr_ignore_filter_patch = True
    _PATCH_APPLIED = True
