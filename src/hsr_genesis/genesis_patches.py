"""Runtime patches for Genesis integration.

These are applied after ``gs.init()`` to avoid Genesis import-time assertions.
"""

from __future__ import annotations


def apply_entity_cls_override_patch() -> None:
    """Allow morphs to override the entity class via ``morph.entity_cls``."""
    import genesis as gs
    from genesis.engine.entities import DroneEntity, RigidEntity
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver

    if getattr(RigidSolver.add_entity, "_hsr_entity_cls_override", False):
        return

    def add_entity(self, idx, material, morph, surface, visualize_contact, name: str | None = None):
        # Handle heterogeneous morphs (list/tuple of morphs)
        morph_heterogeneous = []
        if isinstance(morph, (tuple, list)):
            morph, *morph_heterogeneous = morph
            self._enable_heterogeneous |= bool(morph_heterogeneous)

        entity_cls_override = getattr(morph, "entity_cls", None)
        if entity_cls_override is not None:
            EntityClass = entity_cls_override
        elif isinstance(morph, gs.morphs.Drone):
            EntityClass = DroneEntity
        else:
            EntityClass = RigidEntity

        morph._enable_mujoco_compatibility = self._enable_mujoco_compatibility

        entity = EntityClass(
            scene=self._scene,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            idx=idx,
            idx_in_solver=self.n_entities,
            link_start=self.n_links,
            joint_start=self.n_joints,
            q_start=self.n_qs,
            dof_start=self.n_dofs,
            geom_start=self.n_geoms,
            cell_start=self.n_cells,
            vert_start=self.n_verts,
            free_verts_state_start=self.n_free_verts,
            fixed_verts_state_start=self.n_fixed_verts,
            face_start=self.n_faces,
            edge_start=self.n_edges,
            vgeom_start=self.n_vgeoms,
            vvert_start=self.n_vverts,
            vface_start=self.n_vfaces,
            visualize_contact=visualize_contact,
            morph_heterogeneous=morph_heterogeneous,
            name=name,
        )
        assert isinstance(entity, RigidEntity)
        self._entities.append(entity)

        return entity

    add_entity._hsr_entity_cls_override = True
    RigidSolver.add_entity = add_entity


def apply_raycaster_ignore_patch() -> None:
    """Allow raycasters to ignore specific links/roots/subtrees during ray casting."""
    import torch
    import genesis as gs
    import genesis.engine.sensors.raycaster as raycaster_mod
    import genesis.options.sensors as sensors_pkg
    import genesis.options.sensors.options as sensor_options
    from genesis.engine.sensors.sensor_manager import SensorManager
    from genesis.utils.misc import concat_with_tensor

    if getattr(raycaster_mod, "_hsr_ignore_patch", False):
        return

    # Extend Raycaster options to accept ignore flags.
    OriginalOptions = raycaster_mod.RaycasterOptions
    OriginalDepthOptions = sensor_options.DepthCamera

    class HSRRaycasterOptions(OriginalOptions):
        ignore_self_link: bool = False
        ignore_same_root: bool = False
        ignore_parent_link: bool = False

    class HSRDepthCameraOptions(OriginalDepthOptions):
        ignore_self_link: bool = False
        ignore_same_root: bool = False
        ignore_parent_link: bool = False

    sensor_options.Raycaster = HSRRaycasterOptions
    sensors_pkg.Raycaster = HSRRaycasterOptions
    raycaster_mod.RaycasterOptions = HSRRaycasterOptions
    sensor_options.DepthCamera = HSRDepthCameraOptions
    sensors_pkg.DepthCamera = HSRDepthCameraOptions

    # Ensure sensor manager recognizes the new options class.
    if OriginalOptions in SensorManager.SENSOR_TYPES_MAP:
        SensorManager.SENSOR_TYPES_MAP[HSRRaycasterOptions] = SensorManager.SENSOR_TYPES_MAP[OriginalOptions]
    if OriginalDepthOptions in SensorManager.SENSOR_TYPES_MAP:
        SensorManager.SENSOR_TYPES_MAP[HSRDepthCameraOptions] = SensorManager.SENSOR_TYPES_MAP[OriginalDepthOptions]

    # Patch build to collect ignore settings.
    _orig_build = raycaster_mod.RaycasterSensor.build

    def _build(self):
        _orig_build(self)
        shared = self._shared_metadata

        if not hasattr(shared, "ignore_link_idx"):
            shared.ignore_link_idx = torch.empty((0,), device=gs.device, dtype=gs.tc_int)
            shared.ignore_root_idx = torch.empty((0,), device=gs.device, dtype=gs.tc_int)
            shared.ignore_subtree_root_link_idx = torch.empty((0,), device=gs.device, dtype=gs.tc_int)

        ignore_link = -1
        ignore_root = -1
        ignore_subtree_root = -1
        if self._link is not None:
            if getattr(self._options, "ignore_self_link", False):
                ignore_link = int(self._link.idx)
            if getattr(self._options, "ignore_same_root", False):
                ignore_root = int(self._link.root_idx)
            if getattr(self._options, "ignore_parent_link", False):
                cur = self._link
                for _ in range(64):
                    if any(joint.type is not gs.JOINT_TYPE.FIXED for joint in cur.joints):
                        ignore_subtree_root = int(cur.idx)
                        break
                    if cur.parent_idx == -1:
                        break
                    cur = self._manager._sim.rigid_solver.links[int(cur.parent_idx)]

        shared.ignore_link_idx = concat_with_tensor(shared.ignore_link_idx, ignore_link)
        shared.ignore_root_idx = concat_with_tensor(shared.ignore_root_idx, ignore_root)
        shared.ignore_subtree_root_link_idx = concat_with_tensor(
            shared.ignore_subtree_root_link_idx, ignore_subtree_root
        )

    raycaster_mod.RaycasterSensor.build = _build

    raycaster_mod._hsr_ignore_patch = True


def apply_runtime_patches() -> None:
    """Apply HSR runtime patches after ``gs.init()``."""
    import genesis as gs

    gs_version = getattr(gs, "__version__", None)
    if gs_version != "0.4.0":
        raise RuntimeError(
            f"genesis-world version mismatch: expected 0.4.0, found {gs_version}. Please install genesis-world==0.4.0."
        )

    apply_entity_cls_override_patch()
    apply_raycaster_ignore_patch()

    from .raycast_filter_patch import apply_raycast_filter_patch

    apply_raycast_filter_patch()
