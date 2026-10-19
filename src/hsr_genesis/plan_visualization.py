"""Visualization helpers for grasp planning.

Two things:

- ``capture_phase_frame`` / ``run_phases_with_captures`` — offscreen camera
  screenshots at pipeline phase boundaries (approach/descend/grasp/lift).
  Generalizes the loop that was duplicated across the RL example screenshot
  scripts.
- ``draw_sweep_debug`` — renders a :class:`SweepDiagnostics` from
  ``SweepGraspPlanner.last_sweep`` as debug objects in the scene: one arrow
  per swept approach-direction candidate (score-colored green→red, only
  reachable ones), and a debug frame at the winning grasp pose.

Debug drawing uses Genesis 0.4.6's ``scene.draw_debug_*`` API; positions are
expected in the *rendered env's* local frame — pass ``env_offset`` (e.g.
``scene.envs_offset[env_idx]``) to translate world → env-local.  When the
scene has no visualizer, ``draw_sweep_debug`` logs a warning and returns [].
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

import genesis as gs

logger = logging.getLogger(__name__)


def capture_phase_frame(camera, path: Path | str) -> Path:
    """Render the camera and save the RGB image as PNG."""
    rgb, _depth, _seg, _normal = camera.render()
    from PIL import Image
    path = Path(path)
    img = Image.fromarray(rgb)
    img.save(str(path))
    logger.info("Saved %s (%dx%d)", path, img.size[0], img.size[1])
    return path


def run_phases_with_captures(
    vec_env,
    camera,
    out_dir: Path | str,
    *,
    phase_boundaries: dict[str, int],
    zero_action: np.ndarray,
    n_envs: int = 1,
    cam_pos: np.ndarray | None = None,
    cam_lookat: np.ndarray | None = None,
    head_targets: tuple[torch.Tensor, torch.Tensor] | None = None,
    env=None,
    extra_steps: int = 10,
    per_phase_fn=None,
) -> dict[str, Path]:
    """Step the env with ``zero_action`` and capture a frame at each phase end.

    Args:
        vec_env: VecEnv wrapper (``step(action)`` advances the sim).
        camera: offscreen camera entity.
        out_dir: output directory (created if missing).
        phase_boundaries: {phase_name: step_index} — a frame is captured when
            ``step == boundary``.
        zero_action: (n_envs, action_dim) zero action applied each step.
        cam_pos / cam_lookat: world-frame camera pose set once before capture;
            None leaves the camera at its configured pose.
        head_targets: optional (pan, tilt) tensors re-asserted every step via
            ``env._pick_env.hsr.control_dofs_position`` (needs ``env``).
        env: the RL env (for head control); optional.
        extra_steps: run this many steps past the last boundary before
            stopping early.
        per_phase_fn: optional ``fn(step, phase_name)`` called after each
            capture (for per-phase logging by the caller).

    Returns:
        {phase_name: Path} for each captured frame.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    captured: dict[str, Path] = {}
    step = 0
    last_boundary = max(phase_boundaries.values())
    stop_at = last_boundary + extra_steps

    while step < stop_at and len(captured) < len(phase_boundaries):
        vec_env.step(zero_action)
        step += 1

        if head_targets is not None and env is not None:
            pan, tilt = head_targets
            pick_env = env._pick_env
            pick_env.hsr.control_dofs_position(
                pan, dofs_idx_local=[13], envs_idx=pick_env.envs_all,
            )
            pick_env.hsr.control_dofs_position(
                tilt, dofs_idx_local=[17], envs_idx=pick_env.envs_all,
            )

        for name, boundary in phase_boundaries.items():
            if step == boundary and name not in captured:
                if cam_pos is not None and cam_lookat is not None:
                    camera.set_pose(pos=cam_pos.tolist(), lookat=cam_lookat.tolist())
                path = capture_phase_frame(camera, out_dir / f"{name}.png")
                captured[name] = path
                if per_phase_fn is not None:
                    per_phase_fn(step, name)

    return captured


def draw_sweep_debug(
    scene,
    sweep,
    *,
    env_idx: int = 0,
    max_candidates: int = 64,
    arrow_len: float = 0.10,
):
    """Draw swept grasp candidates as debug objects in the scene.

    Args:
        scene: Genesis scene (must be built).
        sweep: ``SweepDiagnostics`` (``SweepGraspPlanner.last_sweep``).
        env_idx: which parallel env to draw (debug objects are drawn in that
            env's local frame, so world coords are shifted by
            ``scene.envs_offset[env_idx]``).
        max_candidates: cap on arrows drawn.
        arrow_len: arrow length in meters.

    Returns:
        List of created debug objects (``scene.clear_debug_objects()`` removes
        them).  Returns ``[]`` with a warning if there is no visualizer or the
        scene isn't built.
    """
    visualizer = getattr(scene, "_visualizer", None)
    if visualizer is None or getattr(visualizer, "context", None) is None:
        logger.warning("draw_sweep_debug: scene has no visualizer — skipping")
        return []
    if sweep is None:
        logger.warning("draw_sweep_debug: no sweep diagnostics recorded")
        return []

    # Env-local frame: subtract the env's world offset.
    offset = np.asarray(scene.envs_offset[env_idx])

    dirs = sweep.candidate_dirs[:, env_idx].detach().cpu().numpy()      # (C,3)
    poss = sweep.candidate_pos[:, env_idx].detach().cpu().numpy()       # (C,3)
    scores = sweep.candidate_scores[:, env_idx].detach().cpu().numpy()  # (C,)
    reach = getattr(sweep, "candidate_feasible", sweep.candidate_reachable)
    reach = reach[:, env_idx].detach().cpu().numpy().astype(bool)
    best = int(sweep.best_index[env_idx].item())

    # Normalize scores to [0, 1] over reachable candidates.
    finite = np.isfinite(scores) & reach
    if finite.any():
        lo, hi = scores[finite].min(), scores[finite].max()
    else:
        lo, hi = 0.0, 1.0
    rng = max(hi - lo, 1e-8)

    objs = []
    n = min(dirs.shape[0], max_candidates)
    for i in range(n):
        if not reach[i]:
            continue  # only draw reachable candidates
        t = float(np.clip((scores[i] - lo) / rng, 0.0, 1.0))
        color = (t, 1.0 - t, 0.0, 0.7)  # green (best score) → red (worst)
        origin = poss[i] - dirs[i] * arrow_len
        obj = scene.draw_debug_arrow(
            pos=(origin - offset).tolist(),
            vec=(dirs[i] * arrow_len).tolist(),
            radius=0.004,
            color=color,
        )
        if obj is not None:
            objs.append(obj)

    # Winning grasp: part-position sphere + hand-orientation frame.
    part = sweep.part_pos[env_idx].detach().cpu().numpy() - offset
    sphere = scene.draw_debug_sphere(part.tolist(), radius=0.015, color=(0.0, 0.4, 1.0, 0.9))
    if sphere is not None:
        objs.append(sphere)

    if 0 <= best < poss.shape[0]:
        best_pos = poss[best] - offset
        best_quat = sweep.candidate_quat[best, env_idx].detach().cpu().numpy()
        T = np.eye(4)
        T[:3, 3] = best_pos
        # WXYZ → rotation matrix.
        w, x, y, z = best_quat / np.linalg.norm(best_quat)
        T[:3, :3] = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])
        frame = scene.draw_debug_frame(T, axis_length=0.08, origin_size=0.01)
        if frame is not None:
            objs.append(frame)

    return objs


def draw_sweep_ghosts(
    scene,
    sweep,
    robot,
    *,
    env_idx: int = 0,
    max_ghosts: int | None = None,
    alpha: float = 0.25,
) -> list:
    """Draw IK-solved robot poses of sweep candidates as translucent
    "ghost" overlays.

    By default draws every solvable candidate (feasible and scored) —
    one merged mesh per candidate, colored by score rank: best green
    grading to worst red.  Set ``max_ghosts`` to cap the count (top
    ranked are kept).

    Args:
        scene: Genesis scene (must be built and have a visualizer).
        sweep: ``SweepDiagnostics`` with ``candidate_qpos`` populated
            (``SweepGraspPlanner.last_sweep``).
        robot: the HSR rigid entity the candidates were solved on.
        env_idx: which parallel env to draw.
        max_ghosts: cap on rendered candidates; None draws all solvable.
        alpha: mesh transparency (0=invisible, 1=opaque).

    Returns:
        List of created debug objects.
    """
    visualizer = getattr(scene, "_visualizer", None)
    if visualizer is None or getattr(visualizer, "context", None) is None:
        logger.warning("draw_sweep_ghosts: scene has no visualizer — skipping")
        return []
    if sweep is None or not hasattr(sweep, "candidate_qpos"):
        logger.warning("draw_sweep_ghosts: no sweep qpos diagnostics")
        return []

    import trimesh
    import genesis.utils.mesh as mu  # noqa: F401  (register debug material)

    offset = np.asarray(scene.envs_offset[env_idx])
    scores = sweep.candidate_scores[:, env_idx].detach().cpu().numpy()
    reach = getattr(sweep, "candidate_feasible", sweep.candidate_reachable)
    reach = reach[:, env_idx].detach().cpu().numpy().astype(bool)
    qposs = sweep.candidate_qpos[:, env_idx]  # (C, n_qs)

    # Solvable = feasible and scored finite; fall back to feasible alone
    # when every score is inf.
    solvable = reach & np.isfinite(scores)
    if not solvable.any():
        solvable = reach
    order = np.argsort(np.where(solvable, scores, np.inf), kind="stable")
    sel = [int(i) for i in order if solvable[i]]
    if max_ghosts is not None:
        sel = sel[:max_ghosts]
    if not sel:
        logger.warning("draw_sweep_ghosts: no solvable candidates")
        return []

    # Score-rank color gradient: best green -> worst red.
    n_sel = len(sel)
    t = np.linspace(0.0, 1.0, n_sel) if n_sel > 1 else np.zeros(1)
    rank_colors = [(float(0.2 + 0.8 * ti), float(1.0 - 0.8 * ti), 0.2, alpha)
                   for ti in t]

    objs = []
    envs = torch.tensor([env_idx], device=gs.device, dtype=torch.int64)
    for rank, ci in enumerate(sel):
        qpos = qposs[ci]
        if qpos.ndim == 1:
            qpos = qpos.unsqueeze(0)
        links_pos, links_quat = robot.forward_kinematics(qpos, envs_idx=envs)
        links_pos = links_pos[0].cpu().numpy()   # (n_links, 3)
        links_quat = links_quat[0].cpu().numpy() # (n_links, 4)
        color = rank_colors[min(rank, len(rank_colors) - 1)]

        meshes = []
        for li, link in enumerate(robot.links):
            for vg in link.vgeoms:
                T_link = np.eye(4)
                w, x, y, z = links_quat[li] / np.linalg.norm(links_quat[li])
                T_link[:3, :3] = np.array([
                    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                ])
                T_link[:3, 3] = links_pos[li]
                T_vg = np.eye(4)
                wv, xv, yv, zv = vg.init_quat
                T_vg[:3, :3] = np.array([
                    [1 - 2 * (yv * yv + zv * zv), 2 * (xv * yv - wv * zv), 2 * (xv * zv + wv * yv)],
                    [2 * (xv * yv + wv * zv), 1 - 2 * (xv * xv + zv * zv), 2 * (yv * zv - wv * xv)],
                    [2 * (xv * zv - wv * yv), 2 * (yv * zv + wv * xv), 1 - 2 * (xv * xv + yv * yv)],
                ])
                T_vg[:3, 3] = vg.init_pos
                T_world = T_link @ T_vg
                verts = np.asarray(vg.init_vverts)
                if verts.ndim != 2 or verts.shape[0] == 0:
                    continue
                v = trimesh.Trimesh(
                    vertices=verts, faces=np.asarray(vg.init_vfaces),
                    process=False,
                )
                v.visual = trimesh.visual.ColorVisuals(
                    mesh=v,
                    face_colors=np.tile(
                        (np.asarray(color[:4]) * 255).astype(np.uint8),
                        (len(v.faces), 1),
                    ),
                )
                v.apply_transform(T_world)
                meshes.append(v)
        if not meshes:
            continue
        merged = trimesh.util.concatenate(meshes)
        # draw in env-local frame via identity at origin minus offset.
        T_off = np.eye(4)
        T_off[:3, 3] = -offset
        merged_local = merged.copy()
        merged_local.apply_transform(T_off)
        obj = scene.draw_debug_mesh(merged_local, pos=np.zeros(3))
        if obj is not None:
            objs.append(obj)
    return objs
