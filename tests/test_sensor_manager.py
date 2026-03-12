import importlib.util
import sys
import types
from pathlib import Path


def _load_sensor_manager_module():
    module_name = "_test_sensor_manager_module"
    if module_name in sys.modules:
        return sys.modules[module_name]

    genesis_stub = types.ModuleType("genesis")
    genesis_stub.Scene = type("Scene", (), {})

    hsr_pkg = types.ModuleType("hsr_genesis")
    hsr_pkg.__path__ = [
        str(Path(__file__).resolve().parents[1] / "src" / "hsr_genesis"),
    ]
    force_torque_stub = types.ModuleType("hsr_genesis.force_torque")

    sys.modules.setdefault("genesis", genesis_stub)
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
