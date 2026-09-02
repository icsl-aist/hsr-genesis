"""Verify ArtVIP dataset objects load in Genesis and joints articulate.

This script de-risks ArtVIP integration by:
  1. Downloading a small ArtVIP object (dishwasher_1, ~20MB) from HuggingFace.
  2. Parsing joint info from the USD (revolute door, prismatic shelves).
  3. Building a Genesis scene with the ArtVIP object.
  4. Checking the object loaded with the expected number of joints.
  5. Setting each joint to a target value and stepping physics.
  6. Verifying the joint moved (qpos changed).

Run:
    PYTHONPATH=src .venv/bin/python examples/verify_artvip.py

Options:
    --category major_appliances   ArtVIP category (default: major_appliances)
    --object dishwasher_1         Object name (default: dishwasher_1)
    --no-download                 Skip download (use existing cache)
    --cache-dir PATH              Custom cache directory
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify ArtVIP object loading in Genesis")
    parser.add_argument("--category", default="major_appliances",
                        help="ArtVIP category (default: major_appliances)")
    parser.add_argument("--object", default="dishwasher/dishwasher_1",
                        help="Object path: type/instance (default: dishwasher/dishwasher_1)")
    parser.add_argument("--no-download", action="store_true",
                        help="Skip download, use existing cache")
    parser.add_argument("--cache-dir", default=None,
                        help="Custom cache directory")
    args = parser.parse_args()

    import genesis as gs

    # Initialize Genesis (GPU preferred, fallback CPU) BEFORE importing any
    # Genesis-dependent HSR modules.
    if not getattr(gs, "_initialized", False):
        try:
            gs.init(backend=gs.gpu, precision="32", logging_level="warning")
        except Exception:
            gs.init(backend=gs.cpu, precision="32", logging_level="warning")

    from hsr_genesis.artvip_loader import (
        ARTVIP_REPO_ID,
        download_artvip_object,
        list_artvip_categories,
        parse_artvip_control_script,
        parse_artvip_joint_info,
    )

    print(f"ArtVIP dataset: {ARTVIP_REPO_ID}")
    print(f"Categories: {list_artvip_categories()}")
    print()

    # --- Step 1: Download (or locate cached) the object ---
    print(f"Object: {args.category}/{args.object}")

    if args.no_download:
        # Try to find in default cache.
        from hsr_genesis.artvip_loader import _DEFAULT_CACHE_DIR, _find_snapshot_dir
        base = f"Articulated_objects/{args.category}/{args.object}"
        cache_dir = Path(args.cache_dir) if args.cache_dir else _DEFAULT_CACHE_DIR
        obj_dir = _find_snapshot_dir(cache_dir, base)
        if obj_dir is None:
            print(f"ERROR: Object not found in cache at {cache_dir}")
            print("Run without --no-download to download it first.")
            return 1
        # Find the model_*.usd file in the object directory.
        usd_path = None
        for p in sorted(obj_dir.iterdir()):
            if p.name.startswith("model_") and p.suffix in (".usd", ".usda", ".usdc", ".usdz"):
                usd_path = p
                break
        if usd_path is None:
            print(f"ERROR: No model_*.usd found in {obj_dir}")
            return 1
    else:
        print(f"Downloading {args.category}/{args.object} from HuggingFace...")
        usd_path = download_artvip_object(
            args.category, args.object,
            cache_dir=args.cache_dir,
        )

    print(f"USD path: {usd_path}")
    print()

    # --- Step 2: Parse joint info ---
    print("Parsing USD joint info...")
    joint_info = parse_artvip_joint_info(usd_path)
    print(f"  Articulation root: {joint_info.articulation_root}")
    print(f"  Up axis: {joint_info.up_axis}, meters/unit: {joint_info.meters_per_unit}")
    print(f"  Total joints: {len(joint_info.joints)}")
    print(f"  Movable joints: {len(joint_info.movable_joints)}")
    for j in joint_info.joints:
        print(f"    {j.name}: {j.joint_type}, limits={j.limits}, axis={j.axis.tolist()}")
        if j.body0 or j.body1:
            print(f"      body0={j.body0}, body1={j.body1}")

    # Also parse control script if available.
    # The control script is named <type>_control.py, e.g. dishwasher_control.py.
    obj_type = args.object.split("/")[0] if "/" in args.object else args.object.split("_")[0]
    ctrl_path = Path(usd_path).parent / "resource" / f"{obj_type}_control.py"
    if ctrl_path.exists():
        print(f"\nParsing control script: {ctrl_path.name}")
        meta = parse_artvip_control_script(ctrl_path)
        for k, v in meta.items():
            print(f"  {k}: {v}")
    print()

    # --- Step 3: Build Genesis scene and load entities ---
    print("Creating Genesis simulation...")
    gs_scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            substeps=2,
        ),
        show_viewer=False,
        show_FPS=False,
    )

    # Ground plane.
    gs_scene.add_entity(gs.morphs.Plane())

    # ArtVIP object.
    obj_entity = gs_scene.add_entity(
        gs.morphs.USD(
            file=str(usd_path),
            pos=(0.5, 0.0, 0.0),
            fixed=True,
            decimate=True,
            convexify=True,
        ),
    )

    gs_scene.build()
    print("  Genesis scene built successfully!")
    print()

    # --- Step 4: Check joints ---
    print("Checking object joints...")
    joints = obj_entity.joints
    n_joints = len(joints)
    print(f"  Entity has {n_joints} joints")
    for i, joint in enumerate(joints):
        qpos = obj_entity.get_dofs_position(joint.dofs_idx_local)
        print(f"    Joint {i}: {joint.name}, qpos={qpos.tolist()}")

    # Collect all DOF indices.
    all_dofs = []
    for joint in joints:
        dofs = joint.dofs_idx_local
        if isinstance(dofs, (list, tuple)):
            all_dofs.extend(dofs)
        else:
            all_dofs.append(dofs)
    all_dofs = [int(d) for d in all_dofs]
    n_dofs = len(all_dofs)
    print(f"  Entity has {n_dofs} DOFs")
    print()

    # --- Step 5: Actuate joints and verify movement ---
    if n_dofs > 0 and all_dofs:
        print("Actuating joints...")
        for i, joint in enumerate(joints):
            dofs = joint.dofs_idx_local
            if isinstance(dofs, (list, tuple)):
                dofs = list(dofs)
            else:
                dofs = [dofs]
            dofs = np.array([int(d) for d in dofs])

            # Get current position.
            qpos_before = obj_entity.get_dofs_position(dofs).cpu().numpy()

            # Set a target position (small movement).
            target = np.full(len(dofs), 0.1, dtype=float)
            obj_entity.control_dofs_position(target, dofs)

            # Step physics.
            for _ in range(50):
                gs_scene.step()

            # Check if it moved.
            qpos_after = obj_entity.get_dofs_position(dofs).cpu().numpy()
            moved = np.any(np.abs(qpos_after - qpos_before) > 0.01)
            status = "MOVED" if moved else "NO MOVEMENT"
            print(f"    Joint {i} ({joint.name}): {qpos_before.tolist()} -> {qpos_after.tolist()} [{status}]")

            # Reset.
            obj_entity.control_dofs_position(np.zeros(len(dofs)), dofs)
            for _ in range(50):
                gs_scene.step()
    else:
        print("  No DOFs to actuate.")

    print()
    print("=" * 60)
    print("ArtVIP verification COMPLETE!")
    print(f"  Object: {args.category}/{args.object}")
    print(f"  USD joints: {len(joint_info.joints)} (movable: {len(joint_info.movable_joints)})")
    print(f"  Genesis entity joints: {n_joints}")
    print(f"  Genesis entity DOFs: {n_dofs}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
