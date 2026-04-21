"""Raycast kernel with link/root/subtree ignore support."""

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.engine.bvh import STACK_SIZE
from genesis.utils.raycast_qd import get_triangle_vertices, ray_aabb_intersection, ray_triangle_intersection
from genesis.utils.geom import qd_normalize, qd_transform_by_quat, qd_transform_by_trans_quat


@qd.kernel
def kernel_cast_rays_ignore(
    fixed_verts_state: array_class.VertsState,
    free_verts_state: array_class.VertsState,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    geoms_info: array_class.GeomsInfo,
    links_info: array_class.LinksInfo,
    bvh_nodes: qd.template(),
    bvh_morton_codes: qd.template(),
    links_pos: qd.types.ndarray(ndim=3),
    links_quat: qd.types.ndarray(ndim=3),
    ray_starts: qd.types.ndarray(ndim=2),
    ray_directions: qd.types.ndarray(ndim=2),
    max_ranges: qd.types.ndarray(ndim=1),
    no_hit_values: qd.types.ndarray(ndim=1),
    is_world_frame: qd.types.ndarray(ndim=1),
    points_to_sensor_idx: qd.types.ndarray(ndim=1),
    sensor_cache_offsets: qd.types.ndarray(ndim=1),
    sensor_point_offsets: qd.types.ndarray(ndim=1),
    sensor_point_counts: qd.types.ndarray(ndim=1),
    ignore_link_idx: qd.types.ndarray(ndim=1),
    ignore_root_idx: qd.types.ndarray(ndim=1),
    ignore_subtree_root_link_idx: qd.types.ndarray(ndim=1),
    output_hits: qd.types.ndarray(ndim=2),
    eps: gs.qd_float,
):
    n_points = ray_starts.shape[0]
    n_triangles = faces_info.verts_idx.shape[0]
    for i_p, i_b in qd.ndrange(n_points, output_hits.shape[-1]):
        i_s = points_to_sensor_idx[i_p]

        link_pos = qd.math.vec3(links_pos[i_b, i_s, 0], links_pos[i_b, i_s, 1], links_pos[i_b, i_s, 2])
        link_quat = qd.math.vec4(
            links_quat[i_b, i_s, 0], links_quat[i_b, i_s, 1], links_quat[i_b, i_s, 2], links_quat[i_b, i_s, 3]
        )

        ray_start_local = qd.math.vec3(ray_starts[i_p, 0], ray_starts[i_p, 1], ray_starts[i_p, 2])
        ray_start_world = qd_transform_by_trans_quat(ray_start_local, link_pos, link_quat)

        ray_dir_local = qd.math.vec3(ray_directions[i_p, 0], ray_directions[i_p, 1], ray_directions[i_p, 2])
        ray_direction_world = qd_normalize(qd_transform_by_quat(ray_dir_local, link_quat), gs.EPS)

        max_range = max_ranges[i_s]
        ignored_link = ignore_link_idx[i_s]
        ignored_root = ignore_root_idx[i_s]
        ignored_subtree_root = ignore_subtree_root_link_idx[i_s]

        hit_face = -1
        closest_distance = gs.qd_float(max_range)

        node_stack = qd.Vector.zero(gs.qd_int, qd.static(STACK_SIZE))
        node_stack[0] = 0
        stack_idx = 1

        while stack_idx > 0:
            stack_idx -= 1
            node_idx = node_stack[stack_idx]

            node = bvh_nodes[i_b, node_idx]
            aabb_t = ray_aabb_intersection(ray_start_world, ray_direction_world, node.bound.min, node.bound.max, eps)

            if aabb_t >= 0.0 and aabb_t < closest_distance:
                if node.left == -1:
                    sorted_leaf_idx = node_idx - (n_triangles - 1)
                    i_f = qd.cast(bvh_morton_codes[i_b, sorted_leaf_idx][1], gs.qd_int)

                    if ignored_link >= 0 or ignored_root >= 0 or ignored_subtree_root >= 0:
                        i_g = faces_info.geom_idx[i_f]
                        tri_link = geoms_info.link_idx[i_g]
                        if tri_link == ignored_link:
                            continue
                        if ignored_root >= 0:
                            tri_root = qd.cast(-1, gs.qd_int)
                            if qd.static(len(links_info.root_idx.shape) == 2):
                                tri_root = links_info.root_idx[tri_link, i_b]
                            else:
                                tri_root = links_info.root_idx[tri_link]
                            if tri_root == ignored_root:
                                continue

                        if ignored_subtree_root >= 0:
                            cur = tri_link
                            is_descendant = False
                            active = True
                            for _ in qd.static(range(64)):
                                if active:
                                    if cur == ignored_subtree_root:
                                        is_descendant = True
                                        active = False
                                    else:
                                        parent = qd.cast(-1, gs.qd_int)
                                        if qd.static(len(links_info.parent_idx.shape) == 2):
                                            parent = links_info.parent_idx[cur, i_b]
                                        else:
                                            parent = links_info.parent_idx[cur]
                                        if parent < 0:
                                            active = False
                                        else:
                                            cur = parent
                            if is_descendant:
                                continue

                    tri_vertices = get_triangle_vertices(
                        i_f, i_b, faces_info, verts_info, fixed_verts_state, free_verts_state
                    )
                    v0, v1, v2 = tri_vertices[:, 0], tri_vertices[:, 1], tri_vertices[:, 2]

                    hit_result = ray_triangle_intersection(ray_start_world, ray_direction_world, v0, v1, v2, eps)
                    if hit_result.w > 0.0 and hit_result.x < closest_distance and hit_result.x >= 0.0:
                        closest_distance = hit_result.x
                        hit_face = i_f
                else:
                    if stack_idx < qd.static(STACK_SIZE - 2):
                        node_stack[stack_idx] = node.left
                        node_stack[stack_idx + 1] = node.right
                        stack_idx += 2

        i_p_sensor = i_p - sensor_point_offsets[i_s]
        i_p_offset = sensor_cache_offsets[i_s]
        n_points_in_sensor = sensor_point_counts[i_s]

        i_p_dist = i_p_offset + n_points_in_sensor * 3 + i_p_sensor

        if hit_face >= 0:
            dist = closest_distance
            output_hits[i_p_dist, i_b] = dist

            if is_world_frame[i_s]:
                hit_point = ray_start_world + dist * ray_direction_world
                output_hits[i_p_offset + i_p_sensor * 3 + 0, i_b] = hit_point.x
                output_hits[i_p_offset + i_p_sensor * 3 + 1, i_b] = hit_point.y
                output_hits[i_p_offset + i_p_sensor * 3 + 2, i_b] = hit_point.z
            else:
                hit_point = dist * qd_normalize(
                    qd.math.vec3(ray_directions[i_p, 0], ray_directions[i_p, 1], ray_directions[i_p, 2]), gs.EPS
                )
                output_hits[i_p_offset + i_p_sensor * 3 + 0, i_b] = hit_point.x
                output_hits[i_p_offset + i_p_sensor * 3 + 1, i_b] = hit_point.y
                output_hits[i_p_offset + i_p_sensor * 3 + 2, i_b] = hit_point.z
        else:
            output_hits[i_p_offset + i_p_sensor * 3 + 0, i_b] = 0.0
            output_hits[i_p_offset + i_p_sensor * 3 + 1, i_b] = 0.0
            output_hits[i_p_offset + i_p_sensor * 3 + 2, i_b] = 0.0
            output_hits[i_p_dist, i_b] = no_hit_values[i_s]


@qd.func
def bvh_ray_cast_ignore_geom(
    ray_start: gs.qd_vec3,
    ray_dir: gs.qd_vec3,
    max_range: gs.qd_float,
    i_b: gs.qd_int,
    bvh_nodes: qd.template(),
    bvh_morton_codes: qd.template(),
    faces_info: array_class.FacesInfo,
    verts_info: array_class.VertsInfo,
    fixed_verts_state: array_class.VertsState,
    free_verts_state: array_class.VertsState,
    ignore_geom_mask: qd.types.ndarray(ndim=1),
    eps: gs.qd_float,
):
    n_triangles = faces_info.verts_idx.shape[0]

    hit_face = -1
    closest_distance = gs.qd_float(max_range)
    hit_normal = qd.math.vec3(0.0, 0.0, 0.0)

    node_stack = qd.Vector.zero(gs.qd_int, qd.static(STACK_SIZE))
    node_stack[0] = 0
    stack_idx = 1

    while stack_idx > 0:
        stack_idx -= 1
        node_idx = node_stack[stack_idx]

        node = bvh_nodes[i_b, node_idx]
        aabb_t = ray_aabb_intersection(ray_start, ray_dir, node.bound.min, node.bound.max, eps)

        if aabb_t >= 0.0 and aabb_t < closest_distance:
            if node.left == -1:
                sorted_leaf_idx = node_idx - (n_triangles - 1)
                i_f = qd.cast(bvh_morton_codes[i_b, sorted_leaf_idx][1], gs.qd_int)

                i_g = faces_info.geom_idx[i_f]
                if ignore_geom_mask[i_g]:
                    continue

                tri_vertices = get_triangle_vertices(
                    i_f, i_b, faces_info, verts_info, fixed_verts_state, free_verts_state
                )
                v0, v1, v2 = tri_vertices[:, 0], tri_vertices[:, 1], tri_vertices[:, 2]

                hit_result = ray_triangle_intersection(ray_start, ray_dir, v0, v1, v2, eps)

                if hit_result.w > 0.0 and hit_result.x < closest_distance and hit_result.x >= 0.0:
                    closest_distance = hit_result.x
                    hit_face = i_f
                    edge1 = v1 - v0
                    edge2 = v2 - v0
                    hit_normal = edge1.cross(edge2).normalized()
            else:
                if stack_idx < qd.static(STACK_SIZE - 2):
                    node_stack[stack_idx] = node.left
                    node_stack[stack_idx + 1] = node.right
                    stack_idx += 2

    return hit_face, closest_distance, hit_normal


@qd.kernel
def kernel_cast_ray_ignore_geom(
    fixed_verts_state: array_class.VertsState,
    free_verts_state: array_class.VertsState,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    bvh_nodes: qd.template(),
    bvh_morton_codes: qd.template(),
    ray_start: qd.types.ndarray(ndim=1),
    ray_direction: qd.types.ndarray(ndim=1),
    max_range: gs.qd_float,
    envs_idx: qd.types.ndarray(ndim=1),
    ignore_geom_mask: qd.types.ndarray(ndim=1),
    result: array_class.RaycastResult,
    eps: gs.qd_float,
):
    ray_start_world = qd.math.vec3(ray_start[0], ray_start[1], ray_start[2])
    ray_direction_world = qd.math.vec3(ray_direction[0], ray_direction[1], ray_direction[2])

    result.distance[None] = qd.math.nan
    result.geom_idx[None] = -1
    result.hit_point[None] = qd.math.vec3(0.0, 0.0, 0.0)
    result.normal[None] = qd.math.vec3(0.0, 0.0, 0.0)
    result.env_idx[None] = -1

    closest_distance = max_range
    hit_face = -1
    hit_env_idx = -1
    hit_normal = qd.math.vec3(0.0, 0.0, 0.0)

    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        cur_hit_face, cur_distance, cur_hit_normal = bvh_ray_cast_ignore_geom(
            ray_start=ray_start_world,
            ray_dir=ray_direction_world,
            max_range=closest_distance,
            i_b=i_b,
            bvh_nodes=bvh_nodes,
            bvh_morton_codes=bvh_morton_codes,
            faces_info=faces_info,
            verts_info=verts_info,
            fixed_verts_state=fixed_verts_state,
            free_verts_state=free_verts_state,
            ignore_geom_mask=ignore_geom_mask,
            eps=eps,
        )

        if cur_hit_face >= 0 and cur_distance < closest_distance:
            closest_distance = cur_distance
            hit_face = cur_hit_face
            hit_env_idx = i_b
            hit_normal = cur_hit_normal

    if hit_face >= 0:
        result.distance[None] = closest_distance
        i_g = faces_info.geom_idx[hit_face]
        result.geom_idx[None] = i_g
        hit_point = ray_start_world + closest_distance * ray_direction_world
        result.hit_point[None] = hit_point
        result.normal[None] = hit_normal
        result.env_idx[None] = hit_env_idx


@qd.kernel
def kernel_cast_rays_ignore_geom(
    fixed_verts_state: array_class.VertsState,
    free_verts_state: array_class.VertsState,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    geoms_info: array_class.GeomsInfo,
    links_info: array_class.LinksInfo,
    bvh_nodes: qd.template(),
    bvh_morton_codes: qd.template(),
    links_pos: qd.types.ndarray(ndim=3),
    links_quat: qd.types.ndarray(ndim=3),
    ray_starts: qd.types.ndarray(ndim=2),
    ray_directions: qd.types.ndarray(ndim=2),
    max_ranges: qd.types.ndarray(ndim=1),
    no_hit_values: qd.types.ndarray(ndim=1),
    is_world_frame: qd.types.ndarray(ndim=1),
    points_to_sensor_idx: qd.types.ndarray(ndim=1),
    sensor_cache_offsets: qd.types.ndarray(ndim=1),
    sensor_point_offsets: qd.types.ndarray(ndim=1),
    sensor_point_counts: qd.types.ndarray(ndim=1),
    ignore_link_idx: qd.types.ndarray(ndim=1),
    ignore_root_idx: qd.types.ndarray(ndim=1),
    ignore_subtree_root_link_idx: qd.types.ndarray(ndim=1),
    ignore_geom_mask: qd.types.ndarray(ndim=1),
    output_hits: qd.types.ndarray(ndim=2),
    eps: gs.qd_float,
):
    n_points = ray_starts.shape[0]
    n_triangles = faces_info.verts_idx.shape[0]
    for i_p, i_b in qd.ndrange(n_points, output_hits.shape[-1]):
        i_s = points_to_sensor_idx[i_p]

        link_pos = qd.math.vec3(links_pos[i_b, i_s, 0], links_pos[i_b, i_s, 1], links_pos[i_b, i_s, 2])
        link_quat = qd.math.vec4(
            links_quat[i_b, i_s, 0], links_quat[i_b, i_s, 1], links_quat[i_b, i_s, 2], links_quat[i_b, i_s, 3]
        )

        ray_start_local = qd.math.vec3(ray_starts[i_p, 0], ray_starts[i_p, 1], ray_starts[i_p, 2])
        ray_start_world = qd_transform_by_trans_quat(ray_start_local, link_pos, link_quat)

        ray_dir_local = qd.math.vec3(ray_directions[i_p, 0], ray_directions[i_p, 1], ray_directions[i_p, 2])
        ray_direction_world = qd_normalize(qd_transform_by_quat(ray_dir_local, link_quat), gs.EPS)

        max_range = max_ranges[i_s]
        ignored_link = ignore_link_idx[i_s]
        ignored_root = ignore_root_idx[i_s]
        ignored_subtree_root = ignore_subtree_root_link_idx[i_s]

        hit_face = -1
        closest_distance = gs.qd_float(max_range)

        node_stack = qd.Vector.zero(gs.qd_int, qd.static(STACK_SIZE))
        node_stack[0] = 0
        stack_idx = 1

        while stack_idx > 0:
            stack_idx -= 1
            node_idx = node_stack[stack_idx]

            node = bvh_nodes[i_b, node_idx]
            aabb_t = ray_aabb_intersection(ray_start_world, ray_direction_world, node.bound.min, node.bound.max, eps)

            if aabb_t >= 0.0 and aabb_t < closest_distance:
                if node.left == -1:
                    sorted_leaf_idx = node_idx - (n_triangles - 1)
                    i_f = qd.cast(bvh_morton_codes[i_b, sorted_leaf_idx][1], gs.qd_int)

                    i_g = faces_info.geom_idx[i_f]
                    if ignore_geom_mask[i_g]:
                        continue

                    if ignored_link >= 0 or ignored_root >= 0 or ignored_subtree_root >= 0:
                        tri_link = geoms_info.link_idx[i_g]
                        if tri_link == ignored_link:
                            continue
                        if ignored_root >= 0:
                            tri_root = qd.cast(-1, gs.qd_int)
                            if qd.static(len(links_info.root_idx.shape) == 2):
                                tri_root = links_info.root_idx[tri_link, i_b]
                            else:
                                tri_root = links_info.root_idx[tri_link]
                            if tri_root == ignored_root:
                                continue

                        if ignored_subtree_root >= 0:
                            cur = tri_link
                            is_descendant = False
                            active = True
                            for _ in qd.static(range(64)):
                                if active:
                                    if cur == ignored_subtree_root:
                                        is_descendant = True
                                        active = False
                                    else:
                                        parent = qd.cast(-1, gs.qd_int)
                                        if qd.static(len(links_info.parent_idx.shape) == 2):
                                            parent = links_info.parent_idx[cur, i_b]
                                        else:
                                            parent = links_info.parent_idx[cur]
                                        if parent < 0:
                                            active = False
                                        else:
                                            cur = parent
                            if is_descendant:
                                continue

                    tri_vertices = get_triangle_vertices(
                        i_f, i_b, faces_info, verts_info, fixed_verts_state, free_verts_state
                    )
                    v0, v1, v2 = tri_vertices[:, 0], tri_vertices[:, 1], tri_vertices[:, 2]

                    hit_result = ray_triangle_intersection(ray_start_world, ray_direction_world, v0, v1, v2, eps)
                    if hit_result.w > 0.0 and hit_result.x < closest_distance and hit_result.x >= 0.0:
                        closest_distance = hit_result.x
                        hit_face = i_f
                else:
                    if stack_idx < qd.static(STACK_SIZE - 2):
                        node_stack[stack_idx] = node.left
                        node_stack[stack_idx + 1] = node.right
                        stack_idx += 2

        i_p_sensor = i_p - sensor_point_offsets[i_s]
        i_p_offset = sensor_cache_offsets[i_s]
        n_points_in_sensor = sensor_point_counts[i_s]

        i_p_dist = i_p_offset + n_points_in_sensor * 3 + i_p_sensor

        if hit_face >= 0:
            dist = closest_distance
            output_hits[i_p_dist, i_b] = dist

            if is_world_frame[i_s]:
                hit_point = ray_start_world + dist * ray_direction_world
                output_hits[i_p_offset + i_p_sensor * 3 + 0, i_b] = hit_point.x
                output_hits[i_p_offset + i_p_sensor * 3 + 1, i_b] = hit_point.y
                output_hits[i_p_offset + i_p_sensor * 3 + 2, i_b] = hit_point.z
            else:
                hit_point = dist * qd_normalize(
                    qd.math.vec3(ray_directions[i_p, 0], ray_directions[i_p, 1], ray_directions[i_p, 2]), gs.EPS
                )
                output_hits[i_p_offset + i_p_sensor * 3 + 0, i_b] = hit_point.x
                output_hits[i_p_offset + i_p_sensor * 3 + 1, i_b] = hit_point.y
                output_hits[i_p_offset + i_p_sensor * 3 + 2, i_b] = hit_point.z
        else:
            output_hits[i_p_offset + i_p_sensor * 3 + 0, i_b] = 0.0
            output_hits[i_p_offset + i_p_sensor * 3 + 1, i_b] = 0.0
            output_hits[i_p_offset + i_p_sensor * 3 + 2, i_b] = 0.0
            output_hits[i_p_dist, i_b] = no_hit_values[i_s]
