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
