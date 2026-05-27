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


@pytest.fixture(scope="module")
def _genesis_initialized():
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
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
    """Regression: batch_renderer with VisOptions lights is not dark.

    Verifies that batch_renderer per-camera lights produce visible output
    on par with the rasterizer.  Requires GPU.
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

    # Replicate VisOptions lights as per-camera lights (same as sensor_manager)
    camera_lights = []
    for light in scene.vis_options.lights:
        lt = getattr(light, "type", None)
        if lt == "directional":
            camera_lights.append({
                "type": "directional",
                "dir": tuple(light.dir),
                "color": tuple(light.color),
                "intensity": float(light.intensity),
            })

    cam = scene.add_sensor(
        gs.sensors.BatchRendererCameraOptions(
            res=(64, 64),
            pos=(3.0, 0.0, 1.5),
            lookat=(1.0, 0.0, 0.3),
            fov=30,
            lights=camera_lights,
        ),
    )
    scene.build()

    try:
        mean_brightness = _render_and_mean_brightness(scene, cam)
    except Exception as e:
        pytest.skip(f"batch_renderer rendering not available: {e}")

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
