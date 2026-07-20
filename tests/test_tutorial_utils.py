"""Tests for ``hsr_genesis.tutorial_utils`` — both pure functions and sim-integrated.

The file is split into two layers:

1. **Pure-function tests** (no Genesis scene required): test the math helpers,
   constants, and state-management logic that don't need a running simulator.
   These run on any platform (CPU-only CI included).

2. **Integration tests** (require a GPU-capable Taichi backend): call
   ``init_sim()``, exercise every public control function, and verify
   observable effects on the simulated HSR (base motion, arm motion, gripper,
   head, FK/IK consistency, frame capture, reset, etc.).

Run::

    PYTHONPATH=src .venv/bin/python -m pytest tests/test_tutorial_utils.py -v
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import genesis as gs

from hsr_genesis import tutorial_utils as tu


# ---------------------------------------------------------------------------
# GPU guard (mirrors test_hsr_ik_integration.py)
# ---------------------------------------------------------------------------


def _check_gpu() -> bool:
    """Return True if a GPU-capable backend is available.

    Uses ``torch.cuda.is_available()`` because Taichi's ``ti.cfg`` is ``None``
    before ``gs.init()`` is called, so checking ``ti.cfg.arch`` at module import
    time (when pytest evaluates ``skipif``) always fails.
    """
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


_gpu_required = pytest.mark.skipif(
    not _check_gpu(),
    reason="tutorial_utils integration tests require a GPU-capable Taichi backend",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def _clear_state():
    """Clear tutorial_utils singleton state before and after a test.

    Uses the shared ``_clear_sim_state()`` reset path (same as ``init_sim``
    and ``reset_sim``) so state-reset logic stays in one place.
    """
    tu._clear_sim_state()
    yield
    tu._clear_sim_state()


@pytest.fixture
def _sim():
    """Function-scoped initialized simulation.

    ``init_sim`` is idempotent (no-op if already initialized), so this is safe
    to call even if a prior test left state populated.  If a ``_clear_state``
    test ran in between, ``init_sim`` will rebuild the scene.
    """
    if not _check_gpu():
        pytest.skip("tutorial_utils integration tests require a GPU-capable Taichi backend")
    if not getattr(gs, "_initialized", False):
        gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    tu.init_sim(dt=0.02, cam_res=(160, 120))
    # init_sim defers scene.build() so spawn_* can add entities first.
    # For tests that need the built scene immediately, build now.
    tu._maybe_build()
    yield
    # Tear down: clear state so pure-function tests see a clean slate.
    tu._clear_sim_state()


@pytest.fixture
def _sim_unbuilt():
    """Like ``_sim`` but does NOT build the scene.

    Used by spawn tests that need to add entities before the first
    ``run()`` / ``step()`` call (Genesis disallows ``add_entity`` after build).
    """
    if not _check_gpu():
        pytest.skip("tutorial_utils integration tests require a GPU-capable Taichi backend")
    if not getattr(gs, "_initialized", False):
        gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    tu.init_sim(dt=0.02, cam_res=(160, 120))
    yield
    tu._clear_sim_state()


# ===========================================================================
# 1. Pure-function tests (no simulator)
# ===========================================================================


class TestEulerToQuaternion:
    """Tests for ``_euler_deg_to_quat_wxyz``."""

    def test_identity_rotation_is_unit_quaternion(self, _clear_state):
        q = tu._euler_deg_to_quat_wxyz(0, 0, 0)
        assert q.shape == (4,)
        assert q.dtype == np.float32
        # Identity quaternion: w=1, x=y=z=0
        np.testing.assert_allclose(q, [1, 0, 0, 0], atol=1e-6)

    def test_quaternion_is_unit_norm(self, _clear_state):
        """Result must always be a unit quaternion."""
        for roll, pitch, yaw in [(0, 0, 90), (30, 45, 60), (10, -20, 170), (180, 0, 0)]:
            q = tu._euler_deg_to_quat_wxyz(roll, pitch, yaw)
            norm = float(np.linalg.norm(q))
            assert abs(norm - 1.0) < 1e-5, f"norm={norm} for ({roll},{pitch},{yaw})"

    def test_180_yaw(self, _clear_state):
        """Yaw=180° → w≈0, z≈1 (rotation about Z)."""
        q = tu._euler_deg_to_quat_wxyz(0, 0, 180)
        # w = cos(90°) ≈ 0, z = sin(90°) = 1
        assert abs(q[0]) < 1e-5
        assert abs(q[3] - 1.0) < 1e-5

    def test_90_pitch(self, _clear_state):
        """Pitch=90° → w=cos(45°), y=sin(45°)."""
        q = tu._euler_deg_to_quat_wxyz(0, 90, 0)
        s = math.sin(math.radians(45))
        c = math.cos(math.radians(45))
        np.testing.assert_allclose(q, [c, 0, s, 0], atol=1e-6)


class TestQuatWxyzToYaw:
    """Tests for ``_quat_wxyz_to_yaw``."""

    def test_identity_quaternion_yields_zero_yaw(self, _clear_state):
        yaw = tu._quat_wxyz_to_yaw([1, 0, 0, 0])
        assert abs(yaw) < 1e-10

    def test_90_deg_yaw(self, _clear_state):
        """Quaternion for +90° yaw → yaw ≈ π/2."""
        s = math.sin(math.radians(45))
        c = math.cos(math.radians(45))
        yaw = tu._quat_wxyz_to_yaw([c, 0, 0, s])
        assert abs(yaw - math.pi / 2) < 1e-6

    def test_negative_90_deg_yaw(self, _clear_state):
        s = math.sin(math.radians(-45))
        c = math.cos(math.radians(-45))
        yaw = tu._quat_wxyz_to_yaw([c, 0, 0, s])
        assert abs(yaw + math.pi / 2) < 1e-6

    def test_accepts_torch_tensor(self, _clear_state):
        """Should accept a torch tensor as input."""
        q = torch.tensor([1.0, 0.0, 0.0, 0.0])
        yaw = tu._quat_wxyz_to_yaw(q)
        assert abs(yaw) < 1e-10

    def test_accepts_numpy_array(self, _clear_state):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        yaw = tu._quat_wxyz_to_yaw(q)
        assert abs(yaw) < 1e-10


class TestQuaternionFromEuler:
    """Tests for the public ``quaternion_from_euler`` (ROS xyzw convention)."""

    def test_identity(self, _clear_state):
        q = tu.quaternion_from_euler(0, 0, 0)
        # ROS convention: [x, y, z, w] → identity = [0, 0, 0, 1]
        np.testing.assert_allclose(q, [0, 0, 0, 1], atol=1e-6)

    def test_returns_xyzw_order(self, _clear_state):
        """Verify the output is in xyzw order, not wxyz."""
        q_wxyz = tu._euler_deg_to_quat_wxyz(0, 0, 90)
        q_xyzw = tu.quaternion_from_euler(0, 0, 90)
        np.testing.assert_allclose(q_xyzw, [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], atol=1e-6)

    def test_unit_norm(self, _clear_state):
        q = tu.quaternion_from_euler(30, 45, 60)
        assert abs(float(np.linalg.norm(q)) - 1.0) < 1e-5


class TestNamedPoses:
    """Tests for the ``ARM_NEUTRAL`` and ``ARM_INIT`` constants."""

    def test_arm_neutral_has_5_elements(self, _clear_state):
        assert len(tu.ARM_NEUTRAL) == 5

    def test_arm_init_has_5_elements(self, _clear_state):
        assert len(tu.ARM_INIT) == 5

    def test_arm_init_is_all_zeros(self, _clear_state):
        assert tu.ARM_INIT == [0.0, 0.0, 0.0, 0.0, 0.0]

    def test_arm_neutral_is_not_init(self, _clear_state):
        """Neutral should differ from init (otherwise it's a duplicate)."""
        assert tu.ARM_NEUTRAL != tu.ARM_INIT


class TestMoveBaseVelState:
    """Tests for ``move_base_vel`` / ``stop_base`` state management (no sim)."""

    def test_move_base_vel_sets_state_in_radians(self, _clear_state):
        tu.move_base_vel(0.1, 0.0, 90)
        assert tu._state.base_vel_cmd is not None
        vx, vy, vw_rad = tu._state.base_vel_cmd
        assert vx == pytest.approx(0.1)
        assert vy == pytest.approx(0.0)
        # 90 deg/s → π/2 rad/s
        assert vw_rad == pytest.approx(math.radians(90))

    def test_move_base_vel_converts_to_floats(self, _clear_state):
        tu.move_base_vel(1, 2, 45)
        vx, vy, vw = tu._state.base_vel_cmd
        assert isinstance(vx, float)
        assert isinstance(vy, float)
        assert isinstance(vw, float)

    def test_stop_base_zeros_velocity(self, _clear_state):
        tu.move_base_vel(0.5, 0.3, 30)
        tu.stop_base()
        assert tu._state.base_vel_cmd == (0.0, 0.0, 0.0)

    def test_move_base_vel_negative_rotation(self, _clear_state):
        tu.move_base_vel(0, 0, -45)
        _, _, vw = tu._state.base_vel_cmd
        assert vw == pytest.approx(math.radians(-45))


class TestClearFrames:
    """Tests for ``clear_frames`` (no sim needed)."""

    def test_clear_frames_empties_list(self, _clear_state):
        tu._state.frames = [1, 2, 3]
        tu.clear_frames()
        assert tu._state.frames == []


class TestMoveArmJointsValidation:
    """Tests for ``move_arm_joints`` input validation (no sim needed for the check)."""

    def test_too_few_angles_raises(self, _clear_state):
        with pytest.raises(ValueError, match="5 elements"):
            tu.move_arm_joints([0.1, 0.2])

    def test_too_many_angles_raises(self, _clear_state):
        with pytest.raises(ValueError, match="5 elements"):
            tu.move_arm_joints([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    def test_empty_list_raises(self, _clear_state):
        with pytest.raises(ValueError, match="5 elements"):
            tu.move_arm_joints([])


class TestArmTrajNames:
    """Tests for ``_arm_traj_names`` (pure function, no sim needed)."""

    def test_returns_list_of_strings(self, _clear_state):
        names = tu._arm_traj_names()
        assert isinstance(names, list)
        assert len(names) == 5
        for n in names:
            assert isinstance(n, str)

    def test_matches_joint_order(self, _clear_state):
        from hsr_genesis.analytic_ik import JOINT_ORDER

        assert tu._arm_traj_names() == list(JOINT_ORDER)

    def test_expected_joint_names(self, _clear_state):
        """The 5 arm joints should be in the canonical order."""
        expected = [
            "arm_lift_joint",
            "arm_flex_joint",
            "arm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
        ]
        assert tu._arm_traj_names() == expected


class TestLog:
    """Tests for ``_log`` (pure function, no sim needed)."""

    def test_prints_tagged_message(self, _clear_state, capsys):
        tu._log("hello world")
        captured = capsys.readouterr()
        assert "[setup:INFO]" in captured.out
        assert "hello world" in captured.out

    def test_custom_level(self, _clear_state, capsys):
        tu._log("something went wrong", level="ERROR")
        captured = capsys.readouterr()
        assert "[setup:ERROR]" in captured.out
        assert "something went wrong" in captured.out


class TestColabBootstrap:
    """Tests for ``colab_bootstrap`` standalone module.

    On a fresh Colab runtime, genesis / torch / numpy are not installed.
    The bootstrap module must be importable without them so that
    ``setup_colab()`` can install deps first.
    """

    def test_no_heavy_deps_imported_at_module_level(self):
        """``colab_bootstrap`` must NOT import genesis, torch, numpy, or
        ``hsr_genesis`` at *module level* — those aren't available on a fresh
        Colab runtime before ``setup_colab()`` runs.

        Imports inside function bodies (e.g. ``setup_colab()`` → ``import
        hsr_genesis`` after deps are installed) are fine — they execute only
        when the function is called, not at import time.

        This is a static AST check so it works regardless of test environment.
        """
        import ast
        import inspect
        from hsr_genesis import colab_bootstrap

        source = inspect.getsource(colab_bootstrap)
        tree = ast.parse(source)

        # Only check module-level body statements, not nested function/lambda bodies.
        imports: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split(".")[0])

        forbidden = {"genesis", "torch", "numpy", "hsr_genesis"}
        heavy = imports & forbidden
        assert not heavy, (
            f"colab_bootstrap imports forbidden deps at module level: {heavy}"
        )

    def test_setup_colab_importable_from_bootstrap(self):
        """``setup_colab`` must be importable from ``colab_bootstrap``."""
        from hsr_genesis.colab_bootstrap import setup_colab

        assert callable(setup_colab)

    def test_setup_colab_accepts_keyword_args(self):
        """``setup_colab`` must accept its documented keyword arguments."""
        from hsr_genesis.colab_bootstrap import setup_colab
        import inspect

        sig = inspect.signature(setup_colab)
        for param in ("repo_url", "repo_dir", "genesis_version", "force_reinstall"):
            assert param in sig.parameters, f"missing parameter: {param}"


class TestNotebookBootstrapPattern:
    """Verify every tutorial notebook's first import cell uses the standalone
    ``colab_bootstrap`` module instead of ``tutorial_utils``.

    Importing from ``tutorial_utils`` triggers ``import genesis`` at module
    level, which fails on a fresh Colab runtime.
    """

    EXEMPT: set[str] = {
        "7_troubleshoot_colab.ipynb",  # manual sys.path approach
        "IK_grasp_hsr_colab.ipynb",  # inline clone, no setup_colab
    }

    def _colab_notebooks(self) -> list[Path]:
        nb_dir = Path(__file__).resolve().parent.parent / "examples" / "tutorials"
        return sorted(nb_dir.glob("*_colab.ipynb"))

    def test_first_import_cell_uses_colab_bootstrap(self):
        """The first code cell that imports ``setup_colab`` must not import
        it from ``tutorial_utils`` (which triggers ``import genesis`` at
        module level and fails on a fresh Colab runtime).

        Accepted patterns:
        1. ``from hsr_genesis.colab_bootstrap import setup_colab``
        2. URL-fetch: ``exec(urllib.request.urlopen(...colab_setup.py...).read())``
           followed by ``setup_colab()``
        """
        import json

        for nb_path in self._colab_notebooks():
            if nb_path.name in self.EXEMPT:
                continue
            with open(nb_path) as f:
                nb = json.load(f)

            found_import_setup = False
            for cell in nb["cells"]:
                if cell["cell_type"] != "code":
                    continue
                src = "".join(cell["source"])
                if "import" not in src or "setup_colab" not in src:
                    continue
                found_import_setup = True

                # Must NOT import setup_colab from tutorial_utils.
                assert (
                    "colab_bootstrap" in src
                    or "colab_setup.py" in src
                ), (
                    f"{nb_path.name}: imports setup_colab from tutorial_utils, "
                    f"not colab_bootstrap or colab_setup.py.\n  Got: {src.strip()[:160]}"
                )
                break  # only check the first such cell

            if not found_import_setup:
                # Not all notebooks use setup_colab — skip silently
                pass

    def test_all_colab_notebooks_covered(self):
        """Every non-exempt notebook should contain a ``setup_colab`` call."""
        import json
        from pathlib import Path

        nb_dir = Path(__file__).resolve().parent.parent / "examples" / "tutorials"
        exempt_clone = {"IK_grasp_hsr_colab.ipynb", "7_troubleshoot_colab.ipynb"}
        for nb_path in sorted(nb_dir.glob("*_colab.ipynb")):
            if nb_path.name in exempt_clone:
                continue
            with open(nb_path) as f:
                nb = json.load(f)
            full_text = "".join(
                "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
            )
            assert "setup_colab" in full_text, (
                f"{nb_path.name} does not call setup_colab — "
                f"should it be added to EXEMPT?"
            )


class TestShowVideoNoFrames:
    """Tests for ``show_video`` / ``show_frame`` when no frames are captured."""

    def test_show_video_no_frames_prints_message(self, _clear_state, capsys):
        tu._state.frames = []
        tu.show_video()
        captured = capsys.readouterr()
        assert "No frames" in captured.out

    def test_show_frame_no_frames_prints_message(self, _clear_state, capsys):
        tu._state.frames = []
        tu.show_frame()
        captured = capsys.readouterr()
        assert "No frames" in captured.out


class TestFindUrdf:
    """Tests for ``_find_urdf``."""

    def test_finds_urdf_in_repo(self, _clear_state):
        """The URDF should be found relative to the repo root."""
        path = tu._find_urdf()
        assert path.exists()
        assert path.name == "hsrb4s.urdf"

    def test_returns_path_object(self, _clear_state):
        path = tu._find_urdf()
        assert isinstance(path, Path)


class TestNotebookSpawnOrder:
    """Regression: verify tutorial notebooks don't have ``run()`` followed by
    ``spawn_*`` in the same scene segment without an intervening ``init_sim()``.

    Each fresh-scene segment must start with ``init_sim()`` so that
    ``spawn_*`` operates on an unbuilt scene.
    """

    NB_DIR = Path(__file__).resolve().parent.parent / "examples" / "tutorials"

    def _check_notebook(self, nb_name: str) -> None:
        import json

        with open(self.NB_DIR / nb_name) as f:
            nb = json.load(f)
        sources = [
            "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
        ]

        last_init_idx = -1
        for i, src in enumerate(sources):
            if "init_sim(" in src:
                last_init_idx = i

            has_spawn = any(
                f"spawn_{s}(" in src for s in ("box", "sphere", "cylinder")
            )
            if not has_spawn:
                continue
            # Same cell has init_sim() — fine.
            if "init_sim(" in src:
                continue

            assert last_init_idx >= 0, (
                f"{nb_name} cell {i}: spawn_* before any init_sim() call. "
                f"Call init_sim() first to create a fresh unbuilt scene."
            )
            # Match ``run(...)`` as a statement-level call (not a substring
            # like ``duration_run(``) to avoid false positives.
            _run_call = re.compile(r"^\s*run\([^)]*\)\s*$", re.MULTILINE)
            has_run_after_init = any(
                _run_call.search(sources[j])
                for j in range(last_init_idx + 1, i)
            )
            assert not has_run_after_init, (
                f"{nb_name} cell {i}: spawn_* without init_sim() after "
                f"preceding run(). Add init_sim() between them to rebuild "
                f"the scene. Last init_sim at cell {last_init_idx}."
            )

    def test_1_basics_colab_spawn_order(self) -> None:
        """1_basics_colab.ipynb must have init_sim() before each spawn."""
        self._check_notebook("1_basics_colab.ipynb")

    def test_4_gripper_colab_spawn_order(self) -> None:
        """4_gripper_colab.ipynb must have init_sim() before each spawn."""
        self._check_notebook("4_gripper_colab.ipynb")

    def test_6_commands_colab_spawn_order(self) -> None:
        """6_commands_colab.ipynb must have init_sim() before each spawn."""
        self._check_notebook("6_commands_colab.ipynb")


class TestGettersBeforeInit:
    """Tests that getters return None before ``init_sim``."""

    def test_get_robot_returns_none_before_init(self, _clear_state):
        assert tu.get_robot() is None

    def test_get_scene_returns_none_before_init(self, _clear_state):
        assert tu.get_scene() is None

    def test_get_camera_returns_none_before_init(self, _clear_state):
        assert tu.get_camera() is None


# ===========================================================================
# 2. Integration tests (require GPU + simulator)
# ===========================================================================


@_gpu_required
class TestInitSim:
    """Tests for ``init_sim`` and ``reset_sim``."""

    def test_init_sim_populates_state(self, _sim):
        assert tu._state.scene is not None
        assert tu._state.hsr is not None
        assert tu._state.cam is not None
        assert tu._state.end_effector is not None
        assert tu._state.motor_idx is not None
        assert tu._state.head_idx is not None
        assert tu._state.gripper is not None
        assert len(tu._state.arm_dofs_idx) == 5

    def test_init_sim_rebuilds_when_called_twice(self, _sim):
        """Calling init_sim a second time should rebuild a fresh scene."""
        first_scene = tu._state.scene
        first_hsr = tu._state.hsr

        tu.init_sim(dt=0.02, cam_res=(160, 120))
        second_scene = tu._state.scene
        second_hsr = tu._state.hsr

        assert first_scene is not None
        assert second_scene is not None
        assert first_scene is not second_scene
        assert first_hsr is not second_hsr

    def test_init_sim_second_call_does_not_raise(self, _sim):
        """Calling init_sim a second time (without args) does not raise.

        Note: the second call does rebuild the scene (new objects), so the
        operation is *not* idempotent in the strict sense — this test only
        checks that no exception is thrown.
        """
        tu.init_sim()

    def test_get_robot_returns_entity(self, _sim):
        robot = tu.get_robot()
        assert robot is tu._state.hsr

    def test_get_scene_returns_scene(self, _sim):
        assert tu.get_scene() is tu._state.scene

    def test_get_camera_returns_camera(self, _sim):
        assert tu.get_camera() is tu._state.cam

    def test_end_effector_offset_is_set(self, _sim):
        """init_sim should set end_effector_offset on the HSR entity."""
        assert tu._state.hsr.end_effector_offset is not None
        assert tu._state.hsr.end_effector_offset[2] == pytest.approx(0.09)

    def test_reset_sim_rebuilds(self, _sim):
        """reset_sim should tear down and rebuild the scene."""
        old_scene = tu._state.scene
        tu.reset_sim(dt=0.02, cam_res=(160, 120))
        assert tu._state.scene is not None
        # A new scene object should be created (different identity).
        assert tu._state.scene is not old_scene


@_gpu_required
class TestRunAndStep:
    """Tests for ``run``, ``step``, and frame capture."""

    def test_run_advances_simulation(self, _sim):
        """run() should not raise and should produce the right number of steps."""
        tu.run(0.06, render=False)  # 3 steps at dt=0.02
        # No exception means success.

    def test_run_with_render_captures_frames(self, _sim):
        tu.clear_frames()
        tu.run(0.06, render=True)
        assert len(tu._state.frames) == 3

    def test_step_captures_frames(self, _sim):
        tu.clear_frames()
        tu.step(2, render=True)
        assert len(tu._state.frames) == 2

    def test_step_no_render_does_not_capture(self, _sim):
        tu.clear_frames()
        tu.step(5, render=False)
        assert len(tu._state.frames) == 0

    def test_clear_frames(self, _sim):
        tu.run(0.04, render=True)
        assert len(tu._state.frames) > 0
        tu.clear_frames()
        assert tu._state.frames == []


@_gpu_required
class TestBaseControl:
    """Tests for base velocity control and goal navigation."""

    def test_move_base_vel_then_run_moves_forward(self, _sim):
        """After commanding forward velocity and stepping, x should increase."""
        x0, _, _ = tu.get_base_pos()
        tu.move_base_vel(0.1, 0, 0)
        tu.run(0.5, render=False)  # 25 steps
        tu.stop_base()
        x1, _, _ = tu.get_base_pos()
        assert x1 > x0, f"base did not move forward: x0={x0}, x1={x1}"

    def test_get_base_pos_returns_three_floats(self, _sim):
        pos = tu.get_base_pos()
        assert len(pos) == 3
        for v in pos:
            assert isinstance(v, float)

    def test_stop_base_zeros_velocity(self, _sim):
        tu.move_base_vel(0.5, 0, 0)
        assert tu._state.base_vel_cmd is not None
        tu.stop_base()
        tu.run(0.04, render=False)
        # After stop, velocity command is (0, 0, 0).
        assert tu._state.base_vel_cmd == (0.0, 0.0, 0.0)

    def test_move_base_goal_cancels_velocity_command(self, _sim):
        """move_base_goal should clear any active velocity command."""
        tu.move_base_vel(0.2, 0, 0)
        assert tu._state.base_vel_cmd is not None
        dur = tu.move_base_goal(0.0, 0.0, 0.0, duration=1.0)
        assert dur == 1.0
        assert tu._state.base_vel_cmd is None

    def test_move_base_goal_returns_duration(self, _sim):
        dur = tu.move_base_goal(0.1, 0.0, 10.0, duration=2.5)
        assert dur == 2.5


@_gpu_required
class TestArmControl:
    """Tests for arm pose commands and joint control."""

    def test_move_arm_neutral_returns_duration(self, _sim):
        dur = tu.move_arm_neutral(duration=1.0)
        assert dur == 1.0

    def test_move_arm_init_returns_duration(self, _sim):
        dur = tu.move_arm_init(duration=1.5)
        assert dur == 1.5

    def test_move_arm_joints_returns_duration(self, _sim):
        dur = tu.move_arm_joints([0.0, -0.5, 0.0, -0.3, 0.0], duration=1.0)
        assert dur == 1.0

    def test_move_arm_joints_stepping_does_not_raise(self, _sim):
        tu.move_arm_joints([0.1, -0.3, 0.2, -0.2, 0.1], duration=1.0)
        tu.run(0.06, render=False)

    def test_move_arm_neutral_stepping_does_not_raise(self, _sim):
        tu.move_arm_neutral(duration=1.0)
        tu.run(0.06, render=False)


@_gpu_required
class TestWholeBodyIK:
    """Tests for ``move_wholebody_ik``."""

    def test_move_wholebody_ik_returns_duration(self, _sim):
        dur = tu.move_wholebody_ik(0.5, 0.0, 0.3, 0, 90, 0, duration=2.0)
        assert dur == 2.0

    def test_move_wholebody_ik_cancels_velocity(self, _sim):
        tu.move_base_vel(0.1, 0, 0)
        tu.move_wholebody_ik(0.5, 0.0, 0.3, 0, 90, 0)
        assert tu._state.base_vel_cmd is None

    def test_move_wholebody_ik_stepping_does_not_raise(self, _sim):
        tu.move_wholebody_ik(0.5, 0.0, 0.3, 0, 90, 0, duration=1.0)
        tu.run(0.06, render=False)

    def test_get_hand_pos_returns_three_floats(self, _sim):
        pos = tu.get_hand_pos()
        assert len(pos) == 3
        for v in pos:
            assert isinstance(v, float)


@_gpu_required
class TestForwardKinematics:
    """Tests for the ``forward_kinematics`` function."""

    def test_returns_dict_of_4x4_transforms(self, _sim):
        result = tu.forward_kinematics([0.0, 0.0, 0.0, 0.0, 0.0])
        assert isinstance(result, dict)
        assert len(result) > 0
        for name, T in result.items():
            assert T.shape == (4, 4)
            # Rotation part should be a valid rotation matrix (det ≈ 1).
            R = T[:3, :3]
            det = float(np.linalg.det(R))
            assert abs(det - 1.0) < 1e-3, f"det(R)={det} for {name}"

    def test_hand_palm_link_present(self, _sim):
        result = tu.forward_kinematics([0.0, 0.0, 0.0, 0.0, 0.0])
        assert "hand_palm_link" in result

    def test_base_footprint_at_origin_when_base_zero(self, _sim):
        result = tu.forward_kinematics([0.0, 0.0, 0.0, 0.0, 0.0], base_xyyaw=(0, 0, 0))
        T = result["base_footprint"]
        np.testing.assert_allclose(T[:3, 3], [0, 0, 0], atol=1e-3)

    def test_base_offset_propagates(self, _sim):
        """Shifting base_xyyaw should shift all link positions."""
        result0 = tu.forward_kinematics([0.0, -0.5, 0.0, -0.3, 0.0], base_xyyaw=(0, 0, 0))
        result1 = tu.forward_kinematics([0.0, -0.5, 0.0, -0.3, 0.0], base_xyyaw=(1.0, 0, 0))
        # All links should shift by ~1m in x.
        for name in result0:
            dx = result1[name][0, 3] - result0[name][0, 3]
            assert abs(dx - 1.0) < 1e-3, f"{name}: dx={dx}"

    def test_torso_lift_raises_hand(self, _sim):
        """Increasing torso_lift should raise the hand_palm_link z position."""
        low = tu.forward_kinematics([0.0, 0.0, 0.0, 0.0, 0.0], torso_lift=0.0)
        high = tu.forward_kinematics([0.0, 0.0, 0.0, 0.0, 0.0], torso_lift=0.1)
        dz = high["hand_palm_link"][2, 3] - low["hand_palm_link"][2, 3]
        assert dz > 0.01, f"torso lift did not raise hand: dz={dz}"


@_gpu_required
class TestGripperControl:
    """Tests for hand position control and force-controlled grasping."""

    def test_move_hand_open_does_not_raise(self, _sim):
        tu.move_hand(1.0)
        tu.run(0.04, render=False)

    def test_move_hand_close_does_not_raise(self, _sim):
        tu.move_hand(0.0)
        tu.run(0.04, render=False)

    def test_move_hand_deactivates_grasp(self, _sim):
        tu.grasp_object(3.0)
        assert tu._state.gripper_active is True
        tu.move_hand(1.0)
        assert tu._state.gripper_active is False

    def test_grasp_object_activates_grasp(self, _sim):
        tu.grasp_object(5.0)
        assert tu._state.gripper_active is True

    def test_grasp_object_with_run_does_not_raise(self, _sim):
        tu.grasp_object(3.0)
        tu.run(0.06, render=False)

    def test_release_object_deactivates_grasp(self, _sim):
        tu.grasp_object(3.0)
        assert tu._state.gripper_active is True
        tu.release_object()
        assert tu._state.gripper_active is False

    def test_release_object_runs_without_error(self, _sim):
        tu.release_object()
        tu.run(0.04, render=False)


@_gpu_required
class TestHeadControl:
    """Tests for ``move_head_tilt``."""

    def test_move_head_tilt_does_not_raise(self, _sim):
        tu.move_head_tilt(0.3)
        tu.run(0.04, render=False)

    def test_move_head_tilt_negative(self, _sim):
        tu.move_head_tilt(-0.5)
        tu.run(0.04, render=False)


class TestSpawnAfterBuild:
    """Regression: spawn helpers must raise clear error after scene is built."""

    @staticmethod
    def _assert_spawn_error(msg: str | None = None) -> None:
        """All three spawn functions should raise RuntimeError when built=True."""
        tu._state.built = True
        # Need a minimal scene mock so spawn_* can be reached (the built guard
        # runs before scene.add_entity, so a mock is sufficient).
        class _MockScene:
            def add_entity(self, *a, **kw):
                raise AssertionError("should not be called")

        tu._state.scene = _MockScene()
        for spawn_fn, kwargs in [
            (tu.spawn_box, {"pos": (0, 0, 0)}),
            (tu.spawn_sphere, {"pos": (0, 0, 0)}),
            (tu.spawn_cylinder, {"pos": (0, 0, 0)}),
        ]:
            with pytest.raises(RuntimeError) as excinfo:
                spawn_fn(**kwargs)
            if msg:
                assert msg in str(excinfo.value)
        tu._state.built = False
        tu._state.scene = None

    def test_all_spawn_raise_runtime_error_when_built_pure(self, _clear_state):
        """Pure test: set built=True, all spawn functions raise RuntimeError."""
        self._assert_spawn_error(msg="Cannot spawn")

    @_gpu_required
    def test_spawn_box_after_run_raises_runtime_error(self, _sim):
        """Integration test: build scene via run(), then spawn_box raises."""
        tu.run(0.02, render=False)  # builds scene
        with pytest.raises(RuntimeError, match="Cannot spawn"):
            tu.spawn_box((0.5, 0.0, 0.1))


class TestSpawnBeforeInit:
    """Regression: spawn helpers must raise clear error before init_sim()."""

    def test_all_spawn_raise_runtime_error_before_init(self, _clear_state):
        """Pure test: scene=None, all spawn functions raise RuntimeError."""
        assert tu._state.scene is None
        for spawn_fn, kwargs in [
            (tu.spawn_box, {"pos": (0, 0, 0)}),
            (tu.spawn_sphere, {"pos": (0, 0, 0)}),
            (tu.spawn_cylinder, {"pos": (0, 0, 0)}),
        ]:
            with pytest.raises(RuntimeError) as excinfo:
                spawn_fn(**kwargs)
            assert "init_sim" in str(excinfo.value)


@_gpu_required
class TestGetObjectPos:
    """Tests for ``get_object_pos`` using the HSR entity itself."""

    def test_get_object_pos_returns_three_floats(self, _sim):
        pos = tu.get_object_pos(tu._state.hsr)
        assert len(pos) == 3
        for v in pos:
            assert isinstance(v, float)


@_gpu_required
class TestSpawnBox:
    """Tests for ``spawn_box``.

    Uses ``_sim_unbuilt`` because Genesis disallows ``add_entity`` after
    ``scene.build()``.  The scene is built lazily on the first ``run()``.
    """

    def test_returns_entity(self, _sim_unbuilt):
        entity = tu.spawn_box((0.5, 0.0, 0.1))
        assert entity is not None

    def test_entity_at_requested_position(self, _sim_unbuilt):
        pos = (0.3, 0.2, 0.05)
        entity = tu.spawn_box(pos)
        tu.run(0.02, render=False)  # builds scene + one step to settle
        x, y, z = tu.get_object_pos(entity)
        assert x == pytest.approx(pos[0], abs=0.01)
        assert y == pytest.approx(pos[1], abs=0.01)

    def test_custom_size_does_not_raise(self, _sim_unbuilt):
        entity = tu.spawn_box((0.4, 0.0, 0.1), size=(0.1, 0.2, 0.3))
        assert entity is not None

    def test_custom_color_does_not_raise(self, _sim_unbuilt):
        entity = tu.spawn_box((0.4, 0.1, 0.1), color=(0.1, 0.2, 0.3, 1.0))
        assert entity is not None


@_gpu_required
class TestSpawnSphere:
    """Tests for ``spawn_sphere`` (uses ``_sim_unbuilt``, see TestSpawnBox)."""

    def test_returns_entity(self, _sim_unbuilt):
        entity = tu.spawn_sphere((0.5, 0.1, 0.1))
        assert entity is not None

    def test_entity_at_requested_position(self, _sim_unbuilt):
        pos = (0.3, -0.2, 0.05)
        entity = tu.spawn_sphere(pos)
        tu.run(0.02, render=False)
        x, y, z = tu.get_object_pos(entity)
        assert x == pytest.approx(pos[0], abs=0.01)
        assert y == pytest.approx(pos[1], abs=0.01)

    def test_custom_radius_does_not_raise(self, _sim_unbuilt):
        entity = tu.spawn_sphere((0.4, 0.0, 0.1), radius=0.08)
        assert entity is not None


@_gpu_required
class TestSpawnCylinder:
    """Tests for ``spawn_cylinder`` (uses ``_sim_unbuilt``, see TestSpawnBox)."""

    def test_returns_entity(self, _sim_unbuilt):
        entity = tu.spawn_cylinder((0.5, -0.1, 0.1))
        assert entity is not None

    def test_entity_at_requested_position(self, _sim_unbuilt):
        pos = (0.3, 0.3, 0.05)
        entity = tu.spawn_cylinder(pos)
        tu.run(0.02, render=False)
        x, y, z = tu.get_object_pos(entity)
        assert x == pytest.approx(pos[0], abs=0.01)
        assert y == pytest.approx(pos[1], abs=0.01)

    def test_custom_dimensions_do_not_raise(self, _sim_unbuilt):
        entity = tu.spawn_cylinder((0.4, -0.2, 0.1), radius=0.1, height=0.5)
        assert entity is not None


@_gpu_required
class TestSaveVideo:
    """Tests for ``save_video``."""

    def test_save_video_writes_file(self, _sim, tmp_path):
        tu.clear_frames()
        tu.run(0.06, render=True)  # 3 frames
        assert len(tu._state.frames) == 3
        out = tmp_path / "test_output.mp4"
        tu.save_video(str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_save_video_no_frames_prints_message(self, _sim, capsys, tmp_path):
        tu.clear_frames()
        out = tmp_path / "empty.mp4"
        tu.save_video(str(out))
        captured = capsys.readouterr()
        assert "No frames" in captured.out
        # File should not be created when there are no frames.
        assert not out.exists()
