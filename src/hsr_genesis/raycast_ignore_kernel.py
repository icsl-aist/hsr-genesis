"""Filtered BVH traversal for the pinned Genesis 1.4.5 API.

Traversal/dispatch adapted from Genesis utils/raycast_qd.py (Apache-2.0; see LICENSE.genesis.txt).
Keep upstream's split/grouped trees, watertight intersections, visual geometry,
and packed sensor results; reject ignored leaves before selecting a nearest hit.
"""

import quadrants as qd
import genesis as gs
import genesis.utils.array_class as array_class
from genesis.utils.raycast_qd import (
    get_triangle_vertices, get_visual_triangle_vertices, ray_projection,
    ray_aabb_intersection, ray_triangle_intersection, triangle_face_normal,
    func_ray_of_thread, write_ray_hit,
)

@qd.func
def bvh_ray_cast(
    i_t: int,
    i_b: int,
    ray_start: qd.types.vector(3),
    ray_dir: qd.types.vector(3),
    max_range: float,
    dyn_state: array_class.DynState,
    bvh_tree_state: array_class.BVHTreeState,
    dyn_info: array_class.DynInfo,
    eps: float,
    ignore_geom_mask: qd.types.ndarray(ndim=2),
    ignore_sensor_idx: int,
):
    """
    Cast a ray through a BVH and find the closest intersection.

    ``i_t`` selects the tree and ``i_b`` the env whose verts back the leaf triangles: ``i_t == i_b`` for a per-env
    BVH, while a grouped static BVH shared by several envs routes ``i_t`` through ``env_bvh_idx`` (the routed envs
    have bit-identical verts, so each env reads its own).

    Returns
    -------
    hit_face : gs.qd_int
        index of the hit (global) triangle (-1 if no hit)
    hit_distance : gs.qd_float
        distance to hit point (unchanged max_range if no hit)
    hit_normal : qd.math.vec3
        normal vector at hit point (zero vector if no hit)
    """
    # The BVH's own leaf count. For a subset BVH this is the subset size, smaller than the solver's face count.
    n_triangles = bvh_tree_state.leaves_idx.shape[1]

    hit_face = -1
    closest_distance = gs.qd_float(max_range)
    hit_normal = qd.math.vec3(0.0, 0.0, 0.0)

    axes, shear, is_valid_dir = ray_projection(ray_dir, eps)

    # Stack for non-recursive BVH traversal
    # Stack depth, see func_bvh_query_aabb in genesis.engine.bvh
    STACK_SIZE = qd.static(64)
    node_stack = qd.Vector.zero(gs.qd_int, STACK_SIZE)
    node_stack[0] = 0  # Start at root node
    stack_idx = 1
    if not is_valid_dir:
        # A direction too short to define a ray leaves the stack empty, reporting no hit.
        stack_idx = 0

    while stack_idx > 0:
        stack_idx -= 1
        node_idx = node_stack[stack_idx]

        # Check if ray hits the node's bounding box
        aabb_t = ray_aabb_intersection(
            ray_start, ray_dir, bvh_tree_state.nodes_min[i_t, node_idx], bvh_tree_state.nodes_max[i_t, node_idx], eps
        )

        if aabb_t >= 0.0 and aabb_t < closest_distance:
            i_left = bvh_tree_state.nodes_left[i_t, node_idx]
            if i_left == -1:  # Leaf node
                # The leaf payload carries the global face index; see kernel_refresh_collision_bvh.
                i_f = bvh_tree_state.leaves_idx[i_t, node_idx - (n_triangles - 1)]
                if ignore_geom_mask[ignore_sensor_idx, dyn_info.faces.geom_idx[i_f]]:
                    continue

                # Get triangle vertices
                tri_vertices = get_triangle_vertices(i_f, i_b, dyn_state, dyn_info)
                v0, v1, v2 = tri_vertices[:, 0], tri_vertices[:, 1], tri_vertices[:, 2]

                # Perform ray-triangle intersection
                hit_distance = ray_triangle_intersection(axes, ray_start, shear, v0, v1, v2, eps)

                if hit_distance >= 0.0 and hit_distance < closest_distance:
                    closest_distance = hit_distance
                    hit_face = i_f
                    hit_normal = triangle_face_normal(v0, v1, v2)
            else:  # Internal node
                # Push children onto stack
                if stack_idx < STACK_SIZE - 2:
                    node_stack[stack_idx] = i_left
                    node_stack[stack_idx + 1] = bvh_tree_state.nodes_right[i_t, node_idx]
                    stack_idx += 2

    return hit_face, closest_distance, hit_normal


@qd.func
def bvh_ray_cast_visual(
    i_t,
    i_b,
    ray_start,
    ray_dir,
    max_range,
    dyn_state: array_class.DynState,
    bvh_tree_state: array_class.BVHTreeState,
    dyn_info: array_class.DynInfo,
    eps,
    ignore_geom_mask: qd.types.ndarray(ndim=2),
    ignore_sensor_idx: int,
):
    """Cast a single ray against the visual-mesh BVH; returns (hit_face, distance, normal).

    See bvh_ray_cast for the tree slot (i_t) / env (i_b) split.
    """
    n_triangles = bvh_tree_state.leaves_idx.shape[1]

    hit_face = -1
    closest_distance = gs.qd_float(max_range)
    hit_normal = qd.math.vec3(0.0, 0.0, 0.0)

    axes, shear, is_valid_dir = ray_projection(ray_dir, eps)

    # Stack depth, see func_bvh_query_aabb in genesis.engine.bvh
    STACK_SIZE = qd.static(64)
    node_stack = qd.Vector.zero(gs.qd_int, STACK_SIZE)
    node_stack[0] = 0
    stack_idx = 1
    if not is_valid_dir:
        stack_idx = 0

    while stack_idx > 0:
        stack_idx -= 1
        node_idx = node_stack[stack_idx]
        aabb_t = ray_aabb_intersection(
            ray_start, ray_dir, bvh_tree_state.nodes_min[i_t, node_idx], bvh_tree_state.nodes_max[i_t, node_idx], eps
        )

        if aabb_t >= 0.0 and aabb_t < closest_distance:
            i_left = bvh_tree_state.nodes_left[i_t, node_idx]
            if i_left == -1:
                i_f = bvh_tree_state.leaves_idx[i_t, node_idx - (n_triangles - 1)]
                if ignore_geom_mask[ignore_sensor_idx, dyn_info.vfaces.vgeom_idx[i_f]]:
                    continue

                tri_vertices = get_visual_triangle_vertices(i_f, i_b, dyn_state, dyn_info)
                v0, v1, v2 = tri_vertices[:, 0], tri_vertices[:, 1], tri_vertices[:, 2]

                hit_distance = ray_triangle_intersection(axes, ray_start, shear, v0, v1, v2, eps)

                if hit_distance >= 0.0 and hit_distance < closest_distance:
                    closest_distance = hit_distance
                    hit_face = i_f
                    hit_normal = triangle_face_normal(v0, v1, v2)
            else:
                if stack_idx < STACK_SIZE - 2:
                    node_stack[stack_idx] = i_left
                    node_stack[stack_idx + 1] = bvh_tree_state.nodes_right[i_t, node_idx]
                    stack_idx += 2

    return hit_face, closest_distance, hit_normal


@qd.kernel
def kernel_cast_ray(
    rays_start: qd.types.ndarray(ndim=2),  # (n_envs, 3)
    envs_idx: qd.types.ndarray(ndim=1),  # [n_envs]
    ray_direction: qd.types.ndarray(ndim=1),  # (3,)
    dyn_state: array_class.DynState,
    bvh_tree_state: array_class.BVHTreeState,
    result: array_class.RaycastResult,
    dyn_info: array_class.DynInfo,
    max_range: float,
    eps: float,
    is_visual: qd.template(),
    ignore_geom_mask: qd.types.ndarray(ndim=2),
):
    """
    Cast one ray of common direction against each env's BVH in parallel.

    Per-env: rays_start[i_b] is the start of the ray in the coordinates of env i_b, the ones its BVH and result[i_b] are
    in, and the closest hit on that env is written to result[i_b]; envs not in envs_idx are left as no-hit (geom_idx ==
    -1, distance == +inf). Aggregation across envs is intentionally out of scope, because cross-env reduction has no use
    beyond the viewer.

    `is_visual` selects the mesh the BVH covers: the visual mesh (vfaces, result.geom_idx holds the hit vgeom) or
    the collision mesh (faces, result.geom_idx holds the hit geom).
    """
    ray_direction_world = qd.math.vec3(ray_direction[0], ray_direction[1], ray_direction[2])

    for i_b in range(result.geom_idx.shape[0]):
        result.distance[i_b] = qd.math.inf
        result.geom_idx[i_b] = -1
        result.hit_point[i_b] = qd.math.vec3(0.0, 0.0, 0.0)
        result.normal[i_b] = qd.math.vec3(0.0, 0.0, 0.0)

    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        ray_start = qd.math.vec3(rays_start[i_b, 0], rays_start[i_b, 1], rays_start[i_b, 2])
        # Declared before the compile-time branch, whose body is a nested scope in quadrants.
        cur_hit_face = -1
        cur_distance = gs.qd_float(max_range)
        cur_hit_normal = qd.math.vec3(0.0, 0.0, 0.0)
        if qd.static(is_visual):
            cur_hit_face, cur_distance, cur_hit_normal = bvh_ray_cast_visual(
                i_b,
                i_b,
                ray_start,
                ray_direction_world,
                max_range,
                dyn_state,
                bvh_tree_state,
                dyn_info,
                eps,
                ignore_geom_mask=ignore_geom_mask,
                ignore_sensor_idx=0,
            )
        else:
            cur_hit_face, cur_distance, cur_hit_normal = bvh_ray_cast(
                i_b,
                i_b,
                ray_start,
                ray_direction_world,
                max_range,
                dyn_state,
                bvh_tree_state,
                dyn_info,
                eps,
                ignore_geom_mask=ignore_geom_mask,
                ignore_sensor_idx=0,
            )
        if cur_hit_face >= 0:
            result.distance[i_b] = cur_distance
            if qd.static(is_visual):
                result.geom_idx[i_b] = dyn_info.vfaces.vgeom_idx[cur_hit_face]
            else:
                result.geom_idx[i_b] = dyn_info.faces.geom_idx[cur_hit_face]
            result.normal[i_b] = cur_hit_normal
            result.hit_point[i_b] = ray_start + cur_distance * ray_direction_world


@qd.kernel
def kernel_cast_rays(
    points_to_sensor_idx: qd.types.ndarray(ndim=1),  # [n_points]
    env_bvh_idx_a: qd.types.ndarray(ndim=1),  # [n_env] - BVH tree slot each env casts against
    env_bvh_idx_b: qd.types.ndarray(ndim=1),
    links_pos: qd.types.ndarray(ndim=3),  # [n_env, n_sensors, 3]
    links_quat: qd.types.ndarray(ndim=3),  # [n_env, n_sensors, 4]
    ray_starts: qd.types.ndarray(ndim=2),  # [n_points, 3]
    ray_directions: qd.types.ndarray(ndim=2),  # [n_points, 3]
    max_ranges: qd.types.ndarray(ndim=1),  # [n_sensors]
    no_hit_values: qd.types.ndarray(ndim=1),  # [n_sensors]
    is_world_frame: qd.types.ndarray(ndim=1),  # [n_sensors]
    sensor_cache_offsets: qd.types.ndarray(ndim=1),  # [n_sensors] - cache start index for each sensor
    sensor_point_offsets: qd.types.ndarray(ndim=1),  # [n_sensors] - point start index for each sensor
    sensor_point_counts: qd.types.ndarray(ndim=1),  # [n_sensors] - number of points for each sensor
    sensor_return_points: qd.types.ndarray(ndim=1),  # [n_sensors] - True to store hit points, False for distances-only
    output_hits: qd.types.ndarray(ndim=2),  # [total_cache_size, n_env]
    dyn_state: array_class.DynState,
    bvh_tree_state_a: array_class.BVHTreeState,
    bvh_tree_state_b: array_class.BVHTreeState,
    dyn_info: array_class.DynInfo,
    eps: float,
    is_merge: qd.template(),
    is_last: qd.template(),
    is_env_major: qd.template(),
    is_split: qd.template(),
    ignore_geom_mask: qd.types.ndarray(ndim=2),
):
    """Cast every ray of every env against the collision BVH set of a solver, or against its two sets when its faces
    are split between static and movable links (see RaycastContext.activate), taking the closer hit.

    The second set casts with the distance of the first hit as its range, so it reports a hit only when closer.
    See write_ray_hit for `is_merge` / `is_last`, which chain the launches over several BVH sets into one output
    buffer. ``output_hits`` is a 2D array of shape (total_cache_size, n_env) where in the first dimension each
    sensor's data is stored as [sensor_points (n_points * 3), sensor_ranges (n_points)], the point block being present
    only for sensors with sensor_return_points set.

    env_bvh_idx routes each env to the BVH tree slot it casts against: identity for per-env trees, all-zero for one
    shared tree, group ids for a grouped static BVH (N distinct geometries, N <= n_env). is_env_major is a
    compile-time flag selecting the thread -> (ray, env) mapping matching that tree layout.
    """
    n_points = ray_starts.shape[0]
    n_envs = output_hits.shape[-1]
    for i_flat in range(n_points * n_envs):
        i_b, i_p, i_s, ray_start_world, ray_direction_world, ray_dir_local = func_ray_of_thread(
            i_flat,
            n_points,
            n_envs,
            points_to_sensor_idx,
            links_pos,
            links_quat,
            ray_starts,
            ray_directions,
            eps,
            is_env_major,
        )
        hit_face, hit_distance, _hit_normal = bvh_ray_cast(
            env_bvh_idx_a[i_b],
            i_b,
            ray_start_world,
            ray_direction_world,
            max_ranges[i_s],
            dyn_state,
            bvh_tree_state_a,
            dyn_info,
            eps,
            ignore_geom_mask=ignore_geom_mask,
            ignore_sensor_idx=i_s,
        )
        if qd.static(is_split):
            hit_range = max_ranges[i_s] if hit_face < 0 else hit_distance
            hit_face_b, hit_distance_b, _hit_normal_b = bvh_ray_cast(
                env_bvh_idx_b[i_b],
                i_b,
                ray_start_world,
                ray_direction_world,
                hit_range,
                dyn_state,
                bvh_tree_state_b,
                dyn_info,
                eps,
                ignore_geom_mask=ignore_geom_mask,
                ignore_sensor_idx=i_s,
            )
            if hit_face_b >= 0:
                hit_face = hit_face_b
                hit_distance = hit_distance_b
        write_ray_hit(
            i_b,
            i_p,
            i_s,
            hit_face,
            hit_distance,
            ray_start_world,
            ray_direction_world,
            ray_dir_local,
            is_world_frame,
            max_ranges,
            no_hit_values,
            sensor_cache_offsets,
            sensor_point_offsets,
            sensor_point_counts,
            sensor_return_points,
            output_hits,
            eps,
            is_merge,
            is_last,
        )


@qd.kernel
def kernel_cast_rays_visual(
    points_to_sensor_idx: qd.types.ndarray(ndim=1),
    env_bvh_idx: qd.types.ndarray(ndim=1),
    links_pos: qd.types.ndarray(ndim=3),
    links_quat: qd.types.ndarray(ndim=3),
    ray_starts: qd.types.ndarray(ndim=2),
    ray_directions: qd.types.ndarray(ndim=2),
    max_ranges: qd.types.ndarray(ndim=1),
    no_hit_values: qd.types.ndarray(ndim=1),
    is_world_frame: qd.types.ndarray(ndim=1),
    sensor_cache_offsets: qd.types.ndarray(ndim=1),
    sensor_point_offsets: qd.types.ndarray(ndim=1),
    sensor_point_counts: qd.types.ndarray(ndim=1),
    sensor_return_points: qd.types.ndarray(ndim=1),
    output_hits: qd.types.ndarray(ndim=2),
    dyn_state: array_class.DynState,
    bvh_tree_state: array_class.BVHTreeState,
    dyn_info: array_class.DynInfo,
    eps: float,
    is_merge: qd.template(),
    is_last: qd.template(),
    is_env_major: qd.template(),
    ignore_geom_mask: qd.types.ndarray(ndim=2),
):
    """Cast every ray of every env against one visual BVH set.

    See kernel_cast_rays for the arguments.
    """
    n_points = ray_starts.shape[0]
    n_envs = output_hits.shape[-1]
    for i_flat in range(n_points * n_envs):
        i_b, i_p, i_s, ray_start_world, ray_direction_world, ray_dir_local = func_ray_of_thread(
            i_flat,
            n_points,
            n_envs,
            points_to_sensor_idx,
            links_pos,
            links_quat,
            ray_starts,
            ray_directions,
            eps,
            is_env_major,
        )
        hit_face, hit_distance, _hit_normal = bvh_ray_cast_visual(
            env_bvh_idx[i_b],
            i_b,
            ray_start_world,
            ray_direction_world,
            max_ranges[i_s],
            dyn_state,
            bvh_tree_state,
            dyn_info,
            eps,
            ignore_geom_mask=ignore_geom_mask,
            ignore_sensor_idx=i_s,
        )
        write_ray_hit(
            i_b,
            i_p,
            i_s,
            hit_face,
            hit_distance,
            ray_start_world,
            ray_direction_world,
            ray_dir_local,
            is_world_frame,
            max_ranges,
            no_hit_values,
            sensor_cache_offsets,
            sensor_point_offsets,
            sensor_point_counts,
            sensor_return_points,
            output_hits,
            eps,
            is_merge,
            is_last,
        )
