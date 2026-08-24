"""Test that HSRPickEnv supports vis_options overrides and camera attachment."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "rl"))

import pytest
import genesis as gs


@pytest.fixture(scope="module")
def _genesis_initialized():
    if not getattr(gs, "_initialized", False):
        gs.init(backend=gs.gpu, logging_level="warning")
    yield


@pytest.mark.usefixtures("_genesis_initialized")
def test_env_with_camera_and_vis_overrides():
    """HSRPickEnv with camera_config creates a camera; vis_options_overrides applied."""
    from ycb_pick_ik_parallel import HSRPickEnv

    env = HSRPickEnv(
        n_envs=4,
        object_name="ycb_013_apple",
        show_viewer=False,
        seed=0,
        disable_visualizer=True,
        vis_options_overrides={
            "env_separate_rigid": False,
            "lights": [gs.options.vis.DirectionalLight(
                dir=(0.5, 0.5, -1), color=(1, 1, 1), intensity=3.0,
            )],
        },
        camera_config={
            "res": (640, 480),
            "pos": (2, -2, 1.5),
            "lookat": (0, 0, 0.5),
            "fov": 60,
            "far": 200.0,
        },
    )
    assert env.camera is not None, "camera should be created when camera_config is provided"
    assert env.camera.res == (640, 480)

    # Verify env_separate_rigid=False was applied
    assert env.scene.vis_options.env_separate_rigid is False

    # Clean up
    del env.scene
    import gc
    gc.collect()


@pytest.mark.usefixtures("_genesis_initialized")
def test_env_without_camera_has_none():
    """HSRPickEnv without camera_config has camera=None (backward compat)."""
    from ycb_pick_ik_parallel import HSRPickEnv

    env = HSRPickEnv(
        n_envs=4,
        object_name="ycb_013_apple",
        show_viewer=False,
        seed=0,
        disable_visualizer=True,
    )
    assert env.camera is None

    del env.scene
    import gc
    gc.collect()
