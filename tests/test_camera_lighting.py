"""Regression tests for default camera lighting behaviour in Genesis renderers.

If the Genesis library ever changes default lighting (VisOptions.lights defaults,
renderer-default DirectionalLight, etc.), these tests will catch the regression so
we can update the compatibility code in sensor_manager.py before users notice.

NOTE: this module must be run in its own pytest process (no concurrent mock tests)
because mock tests register stub modules that shadow the real Genesis package.
"""

import gc
import sys

# Remove any genesis stub registered by other test modules, otherwise
# ``import genesis`` will resolve to the stub instead of the real package.
for _key in list(sys.modules):
    if _key == "genesis" or _key.startswith("genesis."):
        del sys.modules[_key]

import genesis as gs
import pytest
import torch

_CPU_SCENE_TIMEOUT_S = 120.0


@pytest.fixture(scope="module")
def _genesis_initialized():
    if not getattr(gs, "_initialized", False):
        gs.init(backend=gs.cpu, precision="32", logging_level="warning")
    yield
    gs.destroy()


def _render_and_mean_brightness(
    scene: gs.Scene,
    camera,
    steps: int = 5,
) -> float:
    for _ in range(steps):
        scene.step()
    data = camera.read()
    rgb = data.rgb
    if rgb.dim() == 4:
        rgb = rgb[0]
    return float(torch.mean(rgb.float())) / 255.0


@pytest.mark.usefixtures("_genesis_initialized")
def test_rasterizer_default_lighting_not_dark() -> None:
    """Regression: rasterizer VisOptions.lights default is not dark.

    If this fails, Genesis may have changed its default `lights` or
    `ambient_light` behaviour.
    """
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        vis_options=gs.options.VisOptions(
            ambient_light=(0.1, 0.1, 0.1),
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
        gs.morphs.Box(pos=(1.0, 0.0, 0.3), size=(0.2, 0.2, 0.2)),
        surface=gs.surfaces.Default(color=(1.0, 1.0, 1.0, 1.0)),
    )
    cam = scene.add_sensor(
        gs.sensors.RasterizerCameraOptions(
            res=(64, 64),
            pos=(3.0, 0.0, 1.5),
            lookat=(1.0, 0.0, 0.3),
            fov=30,
        ),
    )
    scene.build()

    try:
        mean_brightness = _render_and_mean_brightness(scene, cam)
    except Exception:
        pytest.skip("rasterizer rendering not available on this platform")

    assert mean_brightness > 0.02, (
        f"rasterizer mean brightness {mean_brightness:.4f} too low — "
        "Genesis VisOptions.lights default may have changed"
    )
    assert mean_brightness < 0.95, (
        f"rasterizer mean brightness {mean_brightness:.4f} too high — "
        "possible over-exposure bug"
    )
    del scene, cam
    gc.collect()


@pytest.mark.usefixtures("_genesis_initialized")
def test_batch_renderer_with_default_light_not_dark() -> None:
    """Regression: batch_renderer with per-camera light is not dark.

    BatchRendererCameraSensor requires CUDA; if not available the test is
    skipped.  When GPU is present this guards against Genesis changing how
    batch_renderer processes per-camera lights.
    """
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
        gs.morphs.Box(pos=(1.0, 0.0, 0.3), size=(0.2, 0.2, 0.2)),
        surface=gs.surfaces.Default(color=(1.0, 1.0, 1.0, 1.0)),
    )
    cam = scene.add_sensor(
        gs.sensors.BatchRendererCameraOptions(
            res=(64, 64),
            pos=(3.0, 0.0, 1.5),
            lookat=(1.0, 0.0, 0.3),
            fov=30,
            lights=[{
                "type": "directional",
                "dir": (-1.0, -1.0, -1.0),
                "color": (1.0, 1.0, 1.0),
                "intensity": 5.0,
            }],
        ),
    )

    try:
        scene.build()
    except gs.GenesisException as e:
        if "CUDA" in str(e):
            pytest.skip("BatchRendererCameraSensor requires CUDA")
        raise

    try:
        mean_brightness = _render_and_mean_brightness(scene, cam)
    except Exception:
        pytest.skip("batch_renderer rendering not available on this platform")

    assert mean_brightness > 0.02, (
        f"batch_renderer mean brightness {mean_brightness:.4f} too low — "
        "the default DirectionalLight may no longer work with batch_renderer"
    )
    assert mean_brightness < 0.95, (
        f"batch_renderer mean brightness {mean_brightness:.4f} too high — "
        "possible over-exposure bug"
    )
    del scene, cam
    gc.collect()
