import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock


def _load_sensor_manager_module():
    module_name = "_test_sensor_manager_module"
    if module_name in sys.modules:
        return sys.modules[module_name]

    genesis_stub = types.ModuleType("genesis")
    genesis_stub.Scene = type("Scene", (), {})

    _sensors_stub = types.ModuleType("genesis.sensors")
    _sensors_stub.RasterizerCameraOptions = type(
        "RasterizerCameraOptions", (),
        {"__init__": lambda self, **kw: self.__dict__.update(kw)},
    )
    _sensors_stub.BatchRendererCameraOptions = type(
        "BatchRendererCameraOptions", (),
        {"__init__": lambda self, **kw: self.__dict__.update(kw)},
    )
    genesis_stub.sensors = _sensors_stub

    sys.modules.setdefault("genesis", genesis_stub)
    sys.modules.setdefault("genesis.sensors", _sensors_stub)

    hsr_pkg = types.ModuleType("hsr_genesis")
    hsr_pkg.__path__ = [
        str(Path(__file__).resolve().parents[1] / "src" / "hsr_genesis"),
    ]
    force_torque_stub = types.ModuleType("hsr_genesis.force_torque")

    sys.modules.setdefault("hsr_genesis", hsr_pkg)
    sys.modules.setdefault("hsr_genesis.force_torque", force_torque_stub)

    module_path = Path(__file__).resolve().parents[1] / "src" / "hsr_genesis" / "sensor_manager.py"
    spec = importlib.util.spec_from_file_location(
        module_name,
        module_path,
        submodule_search_locations=hsr_pkg.__path__,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "hsr_genesis"
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


sensor_manager = _load_sensor_manager_module()
_RateLimitedSensorProxy = sensor_manager._RateLimitedSensorProxy
parse_gazebo_sensors = sensor_manager.parse_gazebo_sensors


class _FakeScene:
    def __init__(self):
        self.t = 0.0


class _FakeCameraSensor:
    def __init__(self):
        self.read_calls = 0
        self.read_image_calls = 0

    def read(self):
        self.read_calls += 1
        return {"frame": self.read_calls}

    def read_image(self):
        self.read_image_calls += 1
        return [self.read_image_calls]


def test_parse_gazebo_sensors_reads_positive_update_rate(tmp_path: Path) -> None:
    urdf_path = tmp_path / "sensor.urdf"
    urdf_path.write_text(
        """
<robot name="test_robot">
  <gazebo reference="camera_link">
    <sensor name="front_camera" type="camera">
      <update_rate>15</update_rate>
      <camera>
        <horizontal_fov>1.0</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
        </image>
      </camera>
    </sensor>
  </gazebo>
</robot>
""".strip(),
        encoding="ascii",
    )

    specs = parse_gazebo_sensors(urdf_path)

    assert len(specs) == 1
    assert specs[0].name == "front_camera"
    assert specs[0].params["update_rate_hz"] == 15.0


def test_parse_gazebo_sensors_ignores_non_positive_update_rate(tmp_path: Path) -> None:
    urdf_path = tmp_path / "sensor.urdf"
    urdf_path.write_text(
        """
<robot name="test_robot">
  <gazebo reference="camera_link">
    <sensor name="front_camera" type="camera">
      <update_rate>0</update_rate>
      <camera />
    </sensor>
  </gazebo>
</robot>
""".strip(),
        encoding="ascii",
    )

    specs = parse_gazebo_sensors(urdf_path)

    assert len(specs) == 1
    assert "update_rate_hz" not in specs[0].params


def test_rate_limited_sensor_proxy_caches_until_next_period() -> None:
    scene = _FakeScene()
    sensor = _FakeCameraSensor()
    proxy = _RateLimitedSensorProxy(sensor, scene, update_rate_hz=5.0)

    first = proxy.read()
    second = proxy.read()

    assert first is second
    assert sensor.read_calls == 1

    scene.t = 0.19
    third = proxy.read()
    assert third is first
    assert sensor.read_calls == 1

    scene.t = 0.2
    fourth = proxy.read()
    assert fourth != first
    assert sensor.read_calls == 2


def test_rate_limited_sensor_proxy_tracks_read_image_separately() -> None:
    scene = _FakeScene()
    sensor = _FakeCameraSensor()
    proxy = _RateLimitedSensorProxy(sensor, scene, update_rate_hz=10.0)

    first = proxy.read_image()
    second = proxy.read_image()

    assert first is second
    assert sensor.read_image_calls == 1

    scene.t = 0.11
    third = proxy.read_image()

    assert third != first
    assert sensor.read_image_calls == 2


def test_rate_limited_sensor_proxy_invalidates_cache_when_time_rewinds() -> None:
    scene = _FakeScene()
    sensor = _FakeCameraSensor()
    proxy = _RateLimitedSensorProxy(sensor, scene, update_rate_hz=2.0)

    first = proxy.read()
    scene.t = 0.6
    second = proxy.read()
    assert second != first

    scene.t = 0.1
    third = proxy.read()

    assert third != second
    assert sensor.read_calls == 3


# ---------------------------------------------------------------------------
# batch_renderer default-light regression tests (mock-based)
# ---------------------------------------------------------------------------


class _TrackingScene:
    def __init__(self, lights=None):
        self.t = 0.0
        self.add_sensor_calls: list[Any] = []
        self.vis_options = _FakeVisOptions(lights)

    def add_sensor(self, options):
        self.add_sensor_calls.append(options)
        return MagicMock()


class _FakeVisOptions:
    def __init__(self, lights=None):
        self.lights = lights if lights is not None else [
            _StubDirectionalLight(dir=(-1.0, -1.0, -1.0), color=(1.0, 1.0, 1.0), intensity=5.0),
        ]


class _StubDirectionalLight:
    def __init__(self, *, dir, color, intensity):
        self.type = "directional"
        self.dir = dir
        self.color = color
        self.intensity = intensity


class _StubPointLight:
    def __init__(self, *, pos, color, intensity):
        self.type = "point"
        self.pos = pos
        self.color = color
        self.intensity = intensity


class _FakeLink:
    def __init__(self, idx_local=0):
        self.idx_local = idx_local


class _FakeEntity:
    def __init__(self, idx=0):
        self.idx = idx
        self._links: dict[str, _FakeLink] = {}

    def get_link(self, *, name):
        if name not in self._links:
            self._links[name] = _FakeLink()
        return self._links[name]


_MINIMAL_CAMERA_URDF = """\
<robot name="test_robot">
  <link name="camera_link"/>
  <gazebo reference="camera_link">
    <sensor name="front_camera" type="camera">
      <camera>
        <horizontal_fov>1.0</horizontal_fov>
        <image>
          <width>64</width>
          <height>48</height>
        </image>
      </camera>
    </sensor>
  </gazebo>
</robot>
"""


def test_batch_renderer_applies_vis_options_lights_to_camera(tmp_path):
    """Default VisOptions lights are replicated as per-camera lights."""
    URDFSensorManager = sensor_manager.URDFSensorManager

    scene = _TrackingScene()
    entity = _FakeEntity(idx=0)
    mgr = URDFSensorManager(scene=scene, entity=entity)

    urdf = tmp_path / "sensor.urdf"
    urdf.write_text(_MINIMAL_CAMERA_URDF, encoding="ascii")

    mgr.create_from_urdf(urdf, create_cameras=True, camera_backend="batch_renderer")

    assert len(scene.add_sensor_calls) >= 1
    options = scene.add_sensor_calls[0]
    assert len(options.lights) == 1
    light = options.lights[0]
    assert light["type"] == "directional"
    assert light["dir"] == (-1.0, -1.0, -1.0)
    assert light["color"] == (1.0, 1.0, 1.0)
    assert light["intensity"] == 5.0


def test_batch_renderer_honors_custom_vis_options_lights(tmp_path):
    """When user sets custom VisOptions lights, those are used for batch_renderer."""
    URDFSensorManager = sensor_manager.URDFSensorManager

    custom_lights = [
        _StubDirectionalLight(dir=(0.0, -1.0, 0.0), color=(0.8, 0.5, 0.3), intensity=3.0),
        _StubPointLight(pos=(2.0, 1.0, 3.0), color=(1.0, 0.0, 0.0), intensity=10.0),
    ]
    scene = _TrackingScene(lights=custom_lights)
    entity = _FakeEntity(idx=0)
    mgr = URDFSensorManager(scene=scene, entity=entity)

    urdf = tmp_path / "sensor.urdf"
    urdf.write_text(_MINIMAL_CAMERA_URDF, encoding="ascii")

    mgr.create_from_urdf(urdf, create_cameras=True, camera_backend="batch_renderer")

    assert len(scene.add_sensor_calls) >= 1
    options = scene.add_sensor_calls[0]
    assert len(options.lights) == 2

    dl = options.lights[0]
    assert dl["type"] == "directional"
    assert dl["dir"] == (0.0, -1.0, 0.0)
    assert dl["color"] == (0.8, 0.5, 0.3)
    assert dl["intensity"] == 3.0

    pl = options.lights[1]
    assert pl["type"] == "point"
    assert pl["pos"] == (2.0, 1.0, 3.0)
    assert pl["color"] == (1.0, 0.0, 0.0)
    assert pl["intensity"] == 10.0


def test_rasterizer_does_not_add_extra_per_camera_light(tmp_path):
    """Rasterizer gets lighting from VisOptions; no extra per-camera light."""
    URDFSensorManager = sensor_manager.URDFSensorManager

    scene = _TrackingScene()
    entity = _FakeEntity(idx=0)
    mgr = URDFSensorManager(scene=scene, entity=entity)

    urdf = tmp_path / "sensor.urdf"
    urdf.write_text(_MINIMAL_CAMERA_URDF, encoding="ascii")

    mgr.create_from_urdf(urdf, create_cameras=True, camera_backend="rasterizer")

    assert len(scene.add_sensor_calls) >= 1
    options = scene.add_sensor_calls[0]
    assert options.lights == [], (
        "rasterizer cameras must not receive extra per-camera lights"
    )


def test_batch_renderer_no_camera_when_cameras_disabled(tmp_path):
    """No cameras created when create_cameras=False (no lights added)."""
    URDFSensorManager = sensor_manager.URDFSensorManager

    scene = _TrackingScene()
    entity = _FakeEntity(idx=0)
    mgr = URDFSensorManager(scene=scene, entity=entity)

    urdf = tmp_path / "sensor.urdf"
    urdf.write_text(_MINIMAL_CAMERA_URDF, encoding="ascii")

    mgr.create_from_urdf(urdf, create_cameras=False, camera_backend="batch_renderer")

    assert len(scene.add_sensor_calls) == 0, (
        "no camera sensors should be created when create_cameras=False"
    )
