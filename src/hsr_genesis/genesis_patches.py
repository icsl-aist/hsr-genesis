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

    import inspect
    original_add_entity = RigidSolver.add_entity
    if "desc" in inspect.signature(original_add_entity).parameters:
        # Genesis 1.4 resolves an entity description before constructing entities.
        def add_entity_v14(self, idx, material, morph, surface, visualize_contact=False, name=None, desc=None):
            first = morph[0] if isinstance(morph, (list, tuple)) else morph
            entity_cls = getattr(first, "entity_cls", None)
            if entity_cls is None:
                return original_add_entity(self, idx, material, morph, surface, visualize_contact, name=name, desc=desc)
            if desc is not None:
                raise ValueError("HSR entity override requires a morph, not a pre-resolved description")
            morphs = tuple(morph) if isinstance(morph, (list, tuple)) else (morph,)
            desc = entity_cls._description_cls.resolve(
                morphs, material, surface, self._options, self._enable_mujoco_compatibility, visualize_contact, name
            )
            self._enable_heterogeneous |= bool(desc.variants)
            entity = entity_cls(self._scene, self, idx, desc, hsr_morph=first)
            self._entities.append(entity)
            if desc.attachment is not None:
                entity.attach(self._scene.get_entity(desc.attachment.entity_name), desc.attachment.link_name)
            return entity
        add_entity_v14._hsr_entity_cls_override = True
        RigidSolver.add_entity = add_entity_v14
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
                    cur = self._link.entity._solver.links[int(cur.parent_idx)]

        shared.ignore_link_idx = concat_with_tensor(shared.ignore_link_idx, ignore_link)
        shared.ignore_root_idx = concat_with_tensor(shared.ignore_root_idx, ignore_root)
        shared.ignore_subtree_root_link_idx = concat_with_tensor(
            shared.ignore_subtree_root_link_idx, ignore_subtree_root
        )

    raycaster_mod.RaycasterSensor.build = _build

    raycaster_mod._hsr_ignore_patch = True


def apply_rasterizer_camera_render_lock_patch() -> None:
    """Hold the interactive viewer's render_lock across RasterizerCameraSensor renders.

    Genesis 0.4.6 ``RasterizerCameraSensor._render_current_state`` temporarily
    subtracts ``envs_offset`` from every rigid node's pose (so each env renders
    at the local origin relative to the camera) and restores the poses after
    the offscreen render. On Linux the interactive viewer runs in a background
    thread by default (``run_in_thread=True``), and its ``on_draw`` can fire
    while the poses are still at the origin — outside the offscreen render's
    own ``render_lock`` window. This causes the viewer to intermittently draw
    all robots gathered at the origin, which is reported as flickering in
    https://github.com/icsl-aist/hsr-genesis/issues/10.

    This patch wraps ``_render_current_state`` so the viewer's ``render_lock``
    (an ``RLock`` — safe for the nested acquisition performed inside the
    offscreen render) is held across the entire save -> render -> restore
    sequence. When there is no interactive viewer (offscreen-only mode) the
    patch is a no-op.
    """
    import genesis as gs
    from genesis.engine.sensors.camera import RasterizerCameraSensor

    if getattr(RasterizerCameraSensor._render_current_state, "_hsr_render_lock_patch", False):
        return

    _orig_render_current_state = RasterizerCameraSensor._render_current_state

    def _render_current_state(self):
        renderer = self._shared_metadata.renderer
        viewer = getattr(renderer, "_viewer", None)
        # ``viewer.lock`` is a ViewerLock wrapping the pyrender viewer's
        # ``render_lock`` (an RLock). Only the interactive viewer needs
        # serialization; offscreen-only renderers have no viewer and no race.
        if viewer is None or getattr(viewer, "lock", None) is None:
            return _orig_render_current_state(self)

        with viewer.lock:
            return _orig_render_current_state(self)

    _render_current_state._hsr_render_lock_patch = True
    RasterizerCameraSensor._render_current_state = _render_current_state


def apply_mesh_decimation_patch() -> None:
    """Recheck Genesis 1.4.5's decimation target after mesh cleanup.

    Mesh.decimate checks its face count before process(validate=True), which
    removes duplicate/degenerate faces. Cleanup can bring the mesh below the
    target, causing fast_simplification to reject an attempted face increase.
    Pre-cleaning lets the original method's guard check the actual face count;
    its mutation decorator still handles cache invalidation and visual cleanup.
    """
    from functools import wraps

    from genesis.engine.mesh import Mesh

    original = Mesh.decimate
    if getattr(original, "_hsr_clean_before_decimation", False):
        return

    @wraps(original)
    def decimate(self, decimate_face_num, decimate_aggressiveness):
        if self._mesh.vertices.shape[0] > 3 and len(self._mesh.faces) > decimate_face_num:
            self._mesh.process(validate=True)
        return original(self, decimate_face_num, decimate_aggressiveness)

    decimate._hsr_clean_before_decimation = True
    Mesh.decimate = decimate


def apply_runtime_patches() -> None:
    """Apply HSR runtime patches after ``gs.init()``."""
    import genesis as gs

    gs_version = getattr(gs, "__version__", None)
    if gs_version != "1.4.5":
        raise RuntimeError(
            f"genesis-world version mismatch: expected 1.4.5, found {gs_version}. Please install genesis-world==1.4.5."
        )

    # Compatibility shim for Genesis 0.4.6 which doesn't have tc_int/tc_float
    if not hasattr(gs, "tc_int"):
        import torch
        gs.tc_int = torch.int32
    if not hasattr(gs, "tc_float"):
        import torch
        gs.tc_float = torch.float32

    apply_entity_cls_override_patch()
    apply_mesh_decimation_patch()
    apply_raycaster_ignore_patch()

    from .raycast_filter_patch import apply_raycast_filter_patch

    apply_raycast_filter_patch()

    # Import force_torque to register the custom sensor (requires gs.init())
    from . import force_torque as _force_torque  # noqa: F401

    apply_planar_rrt_patch()
    apply_rasterizer_camera_render_lock_patch()


def apply_planar_rrt_patch() -> None:
    """Support HSR planar roots in Genesis 1.4.5's scalar-coordinate RRT planners.

    FREE-joint qpos has seven coordinates but six DOFs. Build sampling bounds in
    qpos space, and sample/steer its orientation on the yaw-only unit circle before
    forward kinematics and collision checking.
    """
    import math

    import genesis as gs
    import quadrants as qd
    import torch
    from genesis.utils import array_class
    from genesis.utils.path_planning import PathPlanner, RRT, RRTConnect

    if getattr(PathPlanner, "_hsr_planar_rrt_patched", False):
        return

    original_init = PathPlanner.__init__
    original_limits = PathPlanner._get_dofs_limit
    original_sanitize = PathPlanner._sanitize_qposs
    original_check_collision = PathPlanner.check_collision
    original_rrt_step = RRT._kernel_rrt_step1
    original_connect_step = RRTConnect._kernel_rrt_connect_step1

    def project_qpos(planner, qpos, *, clamp_yaw=True):
        result = qpos.clone()
        start = planner._hsr_free_start
        quat = result[..., start + 3:start + 7]
        quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        w, x, y, z = quat.unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        if clamp_yaw:
            yaw = yaw.clamp(-planner._hsr_yaw_range, planner._hsr_yaw_range)
        result[..., start + 2] = 0
        result[..., start + 3] = torch.cos(yaw / 2)
        result[..., start + 4:start + 6] = 0
        result[..., start + 6] = torch.sin(yaw / 2)
        return result

    def patched_init(self, entity):
        if getattr(getattr(entity, "_morph", None), "hsr_base_mode", None) != "planar":
            return original_init(self, entity)
        free_joints = [joint for joint in entity.joints if joint.type == gs.JOINT_TYPE.FREE]
        if not free_joints:
            return original_init(self, entity)
        if len(free_joints) != 1 or any(j.type == gs.JOINT_TYPE.SPHERICAL for j in entity.joints):
            gs.raise_exception("Planar planning requires one FREE joint and no SPHERICAL joints.")
        self._entity = entity
        self._solver = entity._solver
        self.PENETRATION_EPS = 1e-5 if gs.qd_float == qd.f32 else 0.0
        self._hsr_free_start = min(free_joints[0].qs_idx_local)
        xy_range, yaw_range = getattr(entity, "_hsr_planar_sampling_limits", (2.0, 2 * math.pi))
        self._hsr_xy_range = xy_range
        # +/- pi covers every planar orientation once, without antipodal quaternions.
        self._hsr_yaw_range = min(yaw_range, math.pi)

    def patched_sanitize(self, qpos_goal, qpos_start, envs_idx):
        current, goal, start, envs_idx = original_sanitize(self, qpos_goal, qpos_start, envs_idx)
        if hasattr(self, "_hsr_free_start"):
            goal = project_qpos(self, goal, clamp_yaw=False)
            start = project_qpos(self, start, clamp_yaw=False)
        return current, goal, start, envs_idx

    def patched_limits(self, qpos_goal, envs_idx):
        if not hasattr(self, "_hsr_free_start"):
            return original_limits(self, qpos_goal, envs_idx)
        batch_dofs = self._solver._options.batch_dofs_info
        dof_lower, dof_upper = self._entity.get_dofs_limit(envs_idx=envs_idx if batch_dofs else None)
        shape = (len(envs_idx), self._entity.n_qs)
        lower = torch.zeros(shape, dtype=gs.tc_float, device=gs.device)
        upper = torch.zeros_like(lower)
        for joint in self._entity.joints:
            if joint.type == gs.JOINT_TYPE.FREE:
                continue
            qs = list(joint.qs_idx_local)
            dofs = list(joint.dofs_idx_local)
            if qs:
                lower[:, qs] = dof_lower[..., dofs]
                upper[:, qs] = dof_upper[..., dofs]
        s = self._hsr_free_start
        lower[:, s:s + 2] = -self._hsr_xy_range
        upper[:, s:s + 2] = self._hsr_xy_range
        lower[:, s + 3], upper[:, s + 3] = -gs.EPS, 1
        lower[:, s + 6], upper[:, s + 6] = -1, 1
        yaw = 2 * torch.atan2(qpos_goal[:, s + 6], qpos_goal[:, s + 3])
        invalid = ((qpos_goal < lower) | (qpos_goal > upper)).any(-1)
        invalid |= yaw.abs() > self._hsr_yaw_range + gs.EPS
        return lower.contiguous(), upper.contiguous(), invalid

    # One shared kernel handles both tree layouts. Explicit annotations are assigned
    # below because this module uses postponed annotations and qd resolves types now.
    def planar_step(
        planner, envs_idx, q_limit_lower, q_limit_upper, forward_pass, is_connect,
        free_start, yaw_range, dyn_state, dyn_info, rigid_info,
    ):
        for i_b_ in range(envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            if planner._rrt_is_active[i_b]:
                sample = qd.Vector([
                    q_limit_lower[i_b_, i_q] + qd.random(dtype=gs.qd_float)
                    * (q_limit_upper[i_b_, i_q] - q_limit_lower[i_b_, i_q])
                    for i_q in range(planner._entity.n_qs)
                ])
                yaw = (2 * qd.random(dtype=gs.qd_float) - 1) * yaw_range
                sample[free_start + 3] = qd.cos(yaw / 2)
                sample[free_start + 6] = qd.sin(yaw / 2)
                if qd.random() < planner._rrt_goal_bias:
                    for i_q in qd.static(range(planner._entity.n_qs)):
                        if forward_pass:
                            sample[i_q] = planner._rrt_goal_configuration[i_q, i_b]
                        else:
                            sample[i_q] = planner._rrt_start_configuration[i_q, i_b]
                nearest_idx = -1
                nearest_dist = gs.qd_float(1e30)
                for i_n in range(planner._rrt_tree_size[i_b]):
                    eligible = True
                    if qd.static(is_connect):
                        if forward_pass:
                            eligible = planner._rrt_node_info[i_n, i_b].parent_idx != -1
                        else:
                            eligible = planner._rrt_node_info[i_n, i_b].child_idx != -1
                    if eligible:
                        dist = (planner._rrt_node_info.configuration[i_n, i_b] - sample).norm_sqr()
                        if dist < nearest_dist:
                            nearest_dist = dist
                            nearest_idx = i_n
                nearest = planner._rrt_node_info.configuration[nearest_idx, i_b]
                result = qd.Vector.zero(gs.qd_float, planner._entity.n_qs)
                for i_q in qd.static(range(planner._entity.n_qs)):
                    direction = sample[i_q] - nearest[i_q]
                    direction = qd.max(-planner._rrt_max_step_size, qd.min(planner._rrt_max_step_size, direction))
                    result[i_q] = nearest[i_q] + direction
                # Steer yaw itself, instead of interpolating quaternion components.
                nearest_yaw = 2 * qd.atan2(nearest[free_start + 6], nearest[free_start + 3])
                sample_yaw = 2 * qd.atan2(sample[free_start + 6], sample[free_start + 3])
                delta_yaw = qd.max(-planner._rrt_max_step_size, qd.min(
                    planner._rrt_max_step_size, sample_yaw - nearest_yaw
                ))
                yaw = qd.max(-yaw_range, qd.min(yaw_range, nearest_yaw + delta_yaw))
                result[free_start + 2] = 0
                result[free_start + 3] = qd.cos(yaw / 2)
                result[free_start + 4] = 0
                result[free_start + 5] = 0
                result[free_start + 6] = qd.sin(yaw / 2)
                if planner._rrt_tree_size[i_b] < planner._rrt_max_nodes - 1:
                    i_new = planner._rrt_tree_size[i_b]
                    planner._rrt_node_info[i_new, i_b].configuration = result
                    if qd.static(is_connect):
                        if forward_pass:
                            planner._rrt_node_info[i_new, i_b].parent_idx = nearest_idx
                        else:
                            planner._rrt_node_info[i_new, i_b].child_idx = nearest_idx
                    else:
                        planner._rrt_node_info[i_new, i_b].parent_idx = nearest_idx
                    planner._rrt_tree_size[i_b] += 1
                    for i_q in qd.static(range(planner._entity.n_qs)):
                        rigid_info.qpos[i_q + planner._entity._q_start, i_b] = result[i_q]
                    i_base = dyn_info.entities.link_start[planner._entity._idx_in_solver]
                    I_base = [i_base, i_b] if qd.static(planner._solver.rigid_config.batch_links_info) else i_base
                    i_root = dyn_info.links.root_idx[I_base]
                    gs.engine.solvers.rigid.rigid_solver.func_forward_kinematics_root(
                        i_root, i_b, rigid_info.qpos, dyn_state, dyn_info, rigid_info,
                        planner._solver.rigid_config, is_backward=False,
                    )
                    for i_r in range(rigid_info.roots_link_idx.shape[0]):
                        gs.engine.solvers.rigid.rigid_solver.func_update_geoms_root(
                            rigid_info.roots_link_idx[i_r], i_b, dyn_state, dyn_info, rigid_info,
                            planner._solver.rigid_config, force_update_all_geoms=False, is_backward=False,
                        )

    planar_step.__annotations__ = {
        "planner": qd.template(), "envs_idx": qd.types.ndarray(),
        "q_limit_lower": qd.types.ndarray(), "q_limit_upper": qd.types.ndarray(),
        "forward_pass": qd.i32, "is_connect": qd.template(), "free_start": qd.template(),
        "yaw_range": gs.qd_float, "dyn_state": array_class.DynState,
        "dyn_info": array_class.DynInfo, "rigid_info": array_class.RigidInfo,
    }
    planar_step_kernel = qd.kernel(planar_step)

    def planar_finish(planner, envs_idx, obj_geom_start, obj_geom_end, ignore_geom_pairs,
                      collider_state, ignore_collision, is_plan_with_obj, is_connect, forward_pass):
        for i_b_ in range(envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            if planner._rrt_is_active[i_b]:
                collision = qd.cast(False, gs.qd_int)
                if not ignore_collision:
                    collision = planner._func_check_collision(
                        i_b, obj_geom_start, obj_geom_end, ignore_geom_pairs, collider_state, is_plan_with_obj
                    )
                i_node = planner._rrt_tree_size[i_b] - 1
                if collision:
                    planner._rrt_tree_size[i_b] -= 1
                    planner._rrt_node_info[i_node, i_b].configuration = 0.0
                    if qd.static(is_connect):
                        if forward_pass:
                            planner._rrt_node_info[i_node, i_b].parent_idx = -1
                        else:
                            planner._rrt_node_info[i_node, i_b].child_idx = -1
                    else:
                        planner._rrt_node_info[i_node, i_b].parent_idx = -1
                else:
                    if qd.static(is_connect):
                        for i_other in range(planner._rrt_tree_size[i_b]):
                            eligible = planner._rrt_node_info[i_other, i_b].child_idx != -1
                            if not forward_pass:
                                eligible = planner._rrt_node_info[i_other, i_b].parent_idx != -1
                            # At capacity step1 cannot append a node. Do not
                            # connect the preceding opposite-tree node to itself.
                            node_in_tree = planner._rrt_node_info[i_node, i_b].parent_idx != -1
                            if not forward_pass:
                                node_in_tree = planner._rrt_node_info[i_node, i_b].child_idx != -1
                            if eligible and node_in_tree and planner._rrt_is_active[i_b]:
                                connected = True
                                for i_q in qd.static(range(planner._entity.n_qs)):
                                    if qd.abs(planner._rrt_node_info[i_node, i_b].configuration[i_q]
                                              - planner._rrt_node_info[i_other, i_b].configuration[i_q]
                                              ) > planner._rrt_max_step_size:
                                        connected = False
                                if connected:
                                    planner._rrt_goal_reached_node_idx[i_b] = i_node
                                    if forward_pass:
                                        planner._rrt_node_info[i_node, i_b].child_idx = i_other
                                    else:
                                        planner._rrt_node_info[i_node, i_b].parent_idx = i_other
                                    planner._rrt_is_active[i_b] = False
                    else:
                        is_goal = True
                        for i_q in qd.static(range(planner._entity.n_qs)):
                            if qd.abs(planner._rrt_node_info[i_node, i_b].configuration[i_q]
                                      - planner._rrt_goal_configuration[i_q, i_b]) > planner._rrt_pos_tol:
                                is_goal = False
                        if is_goal:
                            planner._rrt_goal_reached_node_idx[i_b] = i_node
                            planner._rrt_is_active[i_b] = False

    planar_finish.__annotations__ = {
        "planner": qd.template(), "envs_idx": qd.types.ndarray(),
        "obj_geom_start": qd.i32, "obj_geom_end": qd.i32, "ignore_geom_pairs": qd.types.ndarray(),
        "collider_state": array_class.ColliderState, "ignore_collision": qd.i32, "is_plan_with_obj": qd.i32,
        "is_connect": qd.template(), "forward_pass": qd.i32,
    }
    planar_finish_kernel = qd.kernel(planar_finish)
    original_rrt_finish = RRT._kernel_rrt_step2
    original_connect_finish = RRTConnect._kernel_rrt_connect_step2

    def rrt_finish(self, *args):
        if not hasattr(self, "_hsr_free_start"):
            return original_rrt_finish(self, *args)
        # Read the just-written node: 1.4.5's RRT finish kernel captures qpos
        # without passing that qd.Tensor as a kernel argument.
        return planar_finish_kernel(self, *args, False, True)

    def connect_finish(self, envs_idx, obj_start, obj_end, pairs, collider_state, rigid_info,
                       forward_pass, ignore_collision, is_plan_with_obj):
        if not hasattr(self, "_hsr_free_start"):
            return original_connect_finish(self, envs_idx, obj_start, obj_end, pairs, collider_state,
                                           rigid_info, forward_pass, ignore_collision, is_plan_with_obj)
        return planar_finish_kernel(self, envs_idx, obj_start, obj_end, pairs, collider_state,
                                    ignore_collision, is_plan_with_obj, True, forward_pass)

    def rrt_step(self, envs_idx, lower, upper, dyn_state, dyn_info, rigid_info):
        if not hasattr(self, "_hsr_free_start"):
            return original_rrt_step(self, envs_idx, lower, upper, dyn_state, dyn_info, rigid_info)
        return planar_step_kernel(self, envs_idx, lower, upper, True, False,
                                  self._hsr_free_start, self._hsr_yaw_range, dyn_state, dyn_info, rigid_info)

    def connect_step(self, envs_idx, qpos, lower, upper, dyn_state, dyn_info, rigid_info, forward_pass):
        if not hasattr(self, "_hsr_free_start"):
            return original_connect_step(self, envs_idx, qpos, lower, upper, dyn_state, dyn_info, rigid_info, forward_pass)
        return planar_step_kernel(self, envs_idx, lower, upper, forward_pass, True,
                                  self._hsr_free_start, self._hsr_yaw_range, dyn_state, dyn_info, rigid_info)

    def check_collision(self, path, *args, **kwargs):
        if hasattr(self, "_hsr_free_start"):
            # Shortcuts and final waypoint resampling interpolate scalar channels.
            # Project BEFORE collision checks so checked and returned poses agree.
            path.copy_(project_qpos(self, path))
        return original_check_collision(self, path, *args, **kwargs)

    PathPlanner.__init__ = patched_init
    PathPlanner._sanitize_qposs = patched_sanitize
    PathPlanner._get_dofs_limit = patched_limits
    PathPlanner.check_collision = check_collision
    RRTConnect._kernel_rrt_connect_step2 = connect_finish
    RRT._kernel_rrt_step2 = rrt_finish
    RRT._kernel_rrt_step1 = rrt_step
    RRTConnect._kernel_rrt_connect_step1 = connect_step
    PathPlanner._hsr_planar_rrt_patched = True
