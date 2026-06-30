"""URDF-driven sensor attachment helpers for HSR in Genesis.

License: Uses data conventions from hsrb_manipulation under BSD-compatible
terms. This package is released under the BSD 3-Clause License
(see `hsr_genesis/LICENSE.txt`).
"""

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection

import numpy as np

import genesis as gs


@dataclass(frozen=True)
class URDFSensorSpec:
    name: str
    type: str
    reference: str
    pose_xyz: tuple[float, float, float]
    pose_rpy: tuple[float, float, float]
    params: dict[str, Any]


class _RateLimitedSensorProxy:
    def __init__(self, sensor: Any, scene: gs.Scene, *, update_rate_hz: float):
        self._sensor = sensor
        self._scene = scene
        self._update_period = 1.0 / float(update_rate_hz)
        self._cache: dict[Any, tuple[float, Any]] = {}

    def _scene_time(self) -> float:
        return float(getattr(self._scene, "t", 0.0))

    def _make_cache_key(self, method_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        try:
            return (method_name, args, tuple(sorted(kwargs.items())))
        except TypeError:
            return None

    def _call_with_rate_limit(self, method_name: str, *args, **kwargs):
        method = getattr(self._sensor, method_name)
        cache_key = self._make_cache_key(method_name, args, kwargs)
        if cache_key is None:
            return method(*args, **kwargs)

        now = self._scene_time()
        cached = self._cache.get(cache_key)
        if cached is not None:
            last_time, last_value = cached
            if now < last_time:
                self._cache.pop(cache_key, None)
            elif now - last_time < self._update_period:
                return last_value

        value = method(*args, **kwargs)
        self._cache[cache_key] = (now, value)
        return value

    def read(self, *args, **kwargs):
        return self._call_with_rate_limit("read", *args, **kwargs)

    def read_image(self, *args, **kwargs):
        return self._call_with_rate_limit("read_image", *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sensor, name)


class _RasterizerCameraSensorProxy:
    def __init__(self, sensor: Any, scene: gs.Scene):
        self._sensor = sensor
        self._scene = scene
        self._frustum_disabled = False

    def _disable_viewer_camera_frustum_once(self) -> None:
        if self._frustum_disabled:
            return
        # Genesis 0.4 rasterizer cameras can fail on first RGB read when the
        # viewer frustum overlay is active, so disable that overlay lazily.
        viewer_context = getattr(getattr(self._scene, "visualizer", None), "context", None)
        off_camera_frustum = getattr(viewer_context, "off_camera_frustum", None)
        if callable(off_camera_frustum):
            off_camera_frustum()
        self._frustum_disabled = True

    def read(self, *args, **kwargs):
        self._disable_viewer_camera_frustum_once()
        return self._sensor.read(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sensor, name)


def _parse_float(text: str | None, default: float) -> float:
    if text is None:
        return float(default)
    return float(text.strip())


def _parse_int(text: str | None, default: int) -> int:
    if text is None:
        return int(default)
    return int(text.strip())


def _parse_positive_float(text: str | None) -> float | None:
    if text is None:
        return None
    value = float(text.strip())
    if value <= 0.0 or not math.isfinite(value):
        return None
    return value


def _pose_from_text(
    text: str | None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if not text:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    parts = [p for p in text.strip().split() if p]
    if len(parts) != 6:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    x, y, z, roll, pitch, yaw = (float(v) for v in parts)
    return (x, y, z), (roll, pitch, yaw)


def _rpy_rad_to_euler_deg(
    rpy: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(float(v) * 180.0 / math.pi for v in rpy)  # type: ignore[return-value]


def _rpy_xyz_to_t(
    xyz: tuple[float, float, float],
    rpy: tuple[float, float, float],
) -> np.ndarray:
    x, y, z = (float(v) for v in xyz)
    r, p, yw = (float(v) for v in rpy)

    cr = math.cos(r)
    sr = math.sin(r)
    cp = math.cos(p)
    sp = math.sin(p)
    cy = math.cos(yw)
    sy = math.sin(yw)

    rz = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    ry = np.array(
        [
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ],
        dtype=np.float32,
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ],
        dtype=np.float32,
    )

    rot = rz @ ry @ rx
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rot
    T[:3, 3] = np.array([x, y, z], dtype=np.float32)
    return T


_GAZEBO_TO_GENESIS_CAMERA_RPY = (math.pi / 2.0, 0.0, -math.pi / 2.0)
"""Rotation that maps Gazebo camera convention (+X forward) to Genesis (-Z forward; OpenGL)."""


def _rotation_to_rpy_rad(R: np.ndarray) -> tuple[float, float, float]:
    """Extract extrinsic ZYX RPY (roll, pitch, yaw) in radians from a 3x3 rotation matrix."""
    if abs(float(R[2, 0])) < 1.0 - 1e-6:
        pitch = math.asin(-float(R[2, 0]))
        roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
        yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    else:
        yaw = 0.0
        if R[2, 0] <= -1.0:
            pitch = math.pi / 2.0
            roll = math.atan2(float(-R[0, 1]), float(R[0, 2]))
        else:
            pitch = -math.pi / 2.0
            roll = math.atan2(float(R[0, 1]), float(-R[0, 2]))
    return (roll, pitch, yaw)


def _vertical_fov_from_horizontal(
    fov_h_rad: float,
    width: int,
    height: int,
) -> float:
    if width <= 0 or height <= 0:
        return float(fov_h_rad)
    return 2.0 * math.atan((float(height) / float(width)) * math.tan(0.5 * float(fov_h_rad)))


def _vis_lights_to_camera_dicts(lights: Any) -> list[dict[str, Any]]:
    """Convert VisOptions light objects to camera option dicts.

    Genesis VisOptions.lights are typed ``DirectionalLight`` / ``PointLight``
    objects; camera options expect ``{"type": ..., ...}`` dicts.  The rasterizer
    applies ``VisOptions.lights`` to its PBR shaders automatically, but
    ``batch_renderer`` cameras ignore scene-level lights, so we replicate them
    as per-camera ``lights=`` entries here.
    """
    result: list[dict[str, Any]] = []
    for light in lights:
        try:
            lt = getattr(light, "type", None)
        except Exception:
            continue
        if lt == "directional":
            result.append({
                "type": "directional",
                "dir": tuple(light.dir),
                "color": tuple(light.color),
                "intensity": float(light.intensity),
            })
        elif lt == "point":
            result.append({
                "type": "point",
                "pos": tuple(light.pos),
                "color": tuple(light.color),
                "intensity": float(light.intensity),
            })
    return result


class URDFSensorManager:
    def __init__(
        self,
        *,
        scene: gs.Scene,
        entity,
    ):
        self.scene = scene
        self.entity = entity
        self._sensors: dict[str, Any] = {}

    @property
    def sensors(self) -> dict[str, Any]:
        return dict(self._sensors)

    def get(self, name: str):
        return self._sensors.get(name)

    def _add(self, name: str, sensor: Any):
        if name in self._sensors:
            name = f"{name}_{len(self._sensors)}"
        self._sensors[name] = sensor

    def _wrap_sensor_with_update_rate(self, spec: URDFSensorSpec, sensor: Any) -> Any:
        update_rate_hz = spec.params.get("update_rate_hz")
        if update_rate_hz is None:
            return sensor
        return _RateLimitedSensorProxy(
            sensor,
            self.scene,
            update_rate_hz=float(update_rate_hz),
        )

    def create_from_urdf(
        self,
        urdf_path: str | Path,
        *,
        create_lidar: bool = True,
        create_cameras: bool = True,
        create_depth_cameras: bool = True,
        create_imu: bool = True,
        create_force_torque: bool = True,
        camera_backend: str = "rasterizer",
        depth_res_override: tuple[int, int] | None = None,
        draw_debug: bool = False,
        enabled_sensor_names: Collection[str] | None = None,
        disabled_sensor_names: Collection[str] | None = None,
    ) -> dict[str, Any]:
        specs = parse_gazebo_sensors(urdf_path)
        enabled_names = set(enabled_sensor_names or ())
        disabled_names = set(disabled_sensor_names or ())

        camera_res_override: tuple[int, int] | None = None
        if create_cameras and camera_backend == "batch_renderer":
            cam_resolutions: list[tuple[int, int]] = []
            for spec in specs:
                if spec.type not in ("camera", "depth") or not self._is_sensor_enabled(
                    spec.name,
                    enabled_names=enabled_names,
                    disabled_names=disabled_names,
                ):
                    continue
                width = int(spec.params.get("width", 320))
                height = int(spec.params.get("height", 240))
                cam_resolutions.append((width, height))
            if cam_resolutions:
                camera_res_override = max(
                    cam_resolutions,
                    key=lambda r: int(r[0]) * int(r[1]),
                )

        for spec in specs:
            if (
                spec.type == "ray"
                and create_lidar
                and self._is_sensor_enabled(
                    spec.name,
                    enabled_names=enabled_names,
                    disabled_names=disabled_names,
                )
            ):
                sensor = self._create_lidar(spec, draw_debug=draw_debug)
                if sensor is not None:
                    self._add(spec.name, self._wrap_sensor_with_update_rate(spec, sensor))
            elif spec.type == "camera":
                if create_cameras and self._is_sensor_enabled(
                    spec.name,
                    enabled_names=enabled_names,
                    disabled_names=disabled_names,
                ):
                    rgb = self._create_camera(
                        spec,
                        backend=camera_backend,
                        override_res=camera_res_override,
                    )
                    if rgb is not None:
                        self._add(spec.name, self._wrap_sensor_with_update_rate(spec, rgb))
            elif spec.type == "depth":
                if create_cameras and self._is_sensor_enabled(
                    spec.name,
                    enabled_names=enabled_names,
                    disabled_names=disabled_names,
                ):
                    rgb = self._create_camera(
                        spec,
                        backend=camera_backend,
                        override_res=camera_res_override,
                    )
                    if rgb is not None:
                        self._add(spec.name, self._wrap_sensor_with_update_rate(spec, rgb))
                depth_name = f"{spec.name}_depth"
                if create_depth_cameras and self._is_sensor_enabled(
                    depth_name,
                    enabled_names=enabled_names,
                    disabled_names=disabled_names,
                ):
                    depth = self._create_depth_camera(
                        spec,
                        draw_debug=draw_debug,
                        override_res=depth_res_override,
                    )
                    if depth is not None:
                        self._add(depth_name, self._wrap_sensor_with_update_rate(spec, depth))
            elif (
                spec.type == "force_torque"
                and create_force_torque
                and self._is_sensor_enabled(
                    spec.name,
                    enabled_names=enabled_names,
                    disabled_names=disabled_names,
                )
            ):
                ft = self._create_force_torque(spec)
                if ft is not None:
                    self._add(spec.name, self._wrap_sensor_with_update_rate(spec, ft))
            elif (
                spec.type == "imu"
                and create_imu
                and self._is_sensor_enabled(
                    spec.name,
                    enabled_names=enabled_names,
                    disabled_names=disabled_names,
                )
            ):
                imu = self._create_imu(spec)
                if imu is not None:
                    self._add(spec.name, self._wrap_sensor_with_update_rate(spec, imu))
        return dict(self._sensors)

    @staticmethod
    def _is_sensor_enabled(
        name: str,
        *,
        enabled_names: set[str],
        disabled_names: set[str],
    ) -> bool:
        if enabled_names and name not in enabled_names:
            return False
        if name in disabled_names:
            return False
        return True

    def _link_idx_local_from_reference(self, reference: str) -> int:
        link = self.entity.get_link(name=reference)
        return int(link.idx_local)

    def _create_lidar(
        self,
        spec: URDFSensorSpec,
        *,
        draw_debug: bool,
    ) -> Any | None:
        params = spec.params
        horizontal_samples = int(params.get("horizontal_samples", 128))
        min_angle_rad = float(params.get("horizontal_min_angle", -math.pi))
        max_angle_rad = float(params.get("horizontal_max_angle", math.pi))
        min_range = float(params.get("min_range", 0.0))
        max_range = float(params.get("max_range", 20.0))

        min_angle_deg = min_angle_rad * 180.0 / math.pi
        max_angle_deg = max_angle_rad * 180.0 / math.pi

        pattern = gs.sensors.raycaster.SphericalPattern(
            fov=((min_angle_deg, max_angle_deg), (0.0, 0.0)),
            n_points=(horizontal_samples, 1),
        )

        pos_offset = spec.pose_xyz
        euler_offset = _rpy_rad_to_euler_deg(spec.pose_rpy)

        link_idx_local = self._link_idx_local_from_reference(spec.reference)

        is_head_sensor = isinstance(spec.reference, str) and spec.reference.startswith("head_")
        ignore_parent_link = bool(is_head_sensor)

        return self.scene.add_sensor(
            gs.sensors.Raycaster(
                pattern=pattern,
                min_range=min_range,
                max_range=max_range,
                entity_idx=int(self.entity.idx),
                link_idx_local=link_idx_local,
                pos_offset=pos_offset,
                euler_offset=euler_offset,
                return_world_frame=True,
                ignore_self_link=True,
                ignore_same_root=not is_head_sensor,
                ignore_parent_link=ignore_parent_link,
                draw_debug=bool(draw_debug),
            )
        )

    def _create_force_torque(
        self,
        spec: URDFSensorSpec,
    ) -> Any | None:
        if not hasattr(gs.sensors, "ForceTorque"):
            return None
        pos_offset = spec.pose_xyz
        euler_offset = _rpy_rad_to_euler_deg(spec.pose_rpy)
        link_idx_local = self._link_idx_local_from_reference(spec.reference)

        # The ForceTorque sensor auto-discovers downstream links during
        # scene.build() by traversing the kinematic tree from the FT link.
        # No need to pass downstream_link_idxs_local explicitly.
        return self.scene.add_sensor(
            gs.sensors.ForceTorque(
                entity_idx=int(self.entity.idx),
                link_idx_local=link_idx_local,
                pos_offset=pos_offset,
                euler_offset=euler_offset,
            )
        )

    def _create_imu(
        self,
        spec: URDFSensorSpec,
    ) -> Any | None:
        pos_offset = spec.pose_xyz
        euler_offset = _rpy_rad_to_euler_deg(spec.pose_rpy)
        link_idx_local = self._link_idx_local_from_reference(spec.reference)
        return self.scene.add_sensor(
            gs.sensors.IMU(
                entity_idx=int(self.entity.idx),
                link_idx_local=link_idx_local,
                pos_offset=pos_offset,
                euler_offset=euler_offset,
            )
        )

    def _create_camera(
        self,
        spec: URDFSensorSpec,
        *,
        backend: str,
        override_res: tuple[int, int] | None = None,
    ) -> Any | None:
        backend = str(backend).lower().strip()
        if backend not in ("rasterizer", "batch_renderer"):
            raise ValueError(f"Unsupported camera_backend: {backend}")

        params = spec.params
        if override_res is not None:
            width, height = (int(override_res[0]), int(override_res[1]))
        else:
            width = int(params.get("width", 320))
            height = int(params.get("height", 240))
        fov_h_rad = float(params.get("horizontal_fov", math.radians(90.0)))
        fov_v_deg = _vertical_fov_from_horizontal(fov_h_rad, width, height) * 180.0 / math.pi

        offset_t = _rpy_xyz_to_t(spec.pose_xyz, spec.pose_rpy)
        correction = _rpy_xyz_to_t((0.0, 0.0, 0.0), _GAZEBO_TO_GENESIS_CAMERA_RPY)
        offset_t = offset_t @ correction

        if backend == "rasterizer":
            options_cls = gs.sensors.RasterizerCameraOptions
        else:
            options_cls = gs.sensors.BatchRendererCameraOptions

        link_idx_local = self._link_idx_local_from_reference(spec.reference)
        entity_idx = int(self.entity.idx)

        # batch_renderer ignores gs.options.VisOptions.lights defaults;
        # rasterizer PBR shaders receive them automatically, so replicate
        # scene-level lights as per-camera options to avoid dark images.
        camera_lights: list[dict[str, Any]] = []
        if backend == "batch_renderer":
            camera_lights = _vis_lights_to_camera_dicts(
                self.scene.vis_options.lights,
            )

        sensor_options = options_cls(
            res=(width, height),
            fov=float(fov_v_deg),
            entity_idx=entity_idx,
            link_idx_local=link_idx_local,
            offset_T=offset_t,
            lights=camera_lights,
        )
        sensor = self.scene.add_sensor(sensor_options)
        if backend == "rasterizer":
            return _RasterizerCameraSensorProxy(sensor, self.scene)
        return sensor

    def _create_depth_camera(
        self,
        spec: URDFSensorSpec,
        *,
        draw_debug: bool,
        override_res: tuple[int, int] | None = None,
    ) -> Any | None:
        params = spec.params
        if override_res is not None:
            width, height = (int(override_res[0]), int(override_res[1]))
        else:
            width = int(params.get("width", 320))
            height = int(params.get("height", 240))
        fov_h_rad = float(params.get("horizontal_fov", math.radians(90.0)))
        near = float(params.get("near", 0.05))
        far = float(params.get("far", 20.0))

        fov_horizontal_deg = float(fov_h_rad * 180.0 / math.pi)

        pattern = gs.sensors.raycaster.DepthCameraPattern(
            res=(width, height),
            fov_horizontal=fov_horizontal_deg,
        )

        pos_offset = spec.pose_xyz
        euler_offset = _rpy_rad_to_euler_deg(spec.pose_rpy)

        link_idx_local = self._link_idx_local_from_reference(spec.reference)

        return self.scene.add_sensor(
            gs.sensors.DepthCamera(
                pattern=pattern,
                min_range=float(near),
                max_range=float(far),
                entity_idx=int(self.entity.idx),
                link_idx_local=link_idx_local,
                pos_offset=pos_offset,
                euler_offset=euler_offset,
                ignore_self_link=True,
                ignore_parent_link=True,
                draw_debug=bool(draw_debug),
            )
        )


def parse_gazebo_sensors(urdf_path: str | Path) -> list[URDFSensorSpec]:
    urdf_path = Path(urdf_path)
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()

    joint_child_by_name: dict[str, str] = {}
    for joint in root.findall("joint"):
        joint_name = joint.attrib.get("name")
        if not joint_name:
            continue
        child = joint.find("child")
        if child is None:
            continue
        child_link = child.attrib.get("link")
        if not child_link:
            continue
        joint_child_by_name[str(joint_name)] = str(child_link)

    specs: list[URDFSensorSpec] = []

    for gazebo in root.findall("gazebo"):
        reference = gazebo.attrib.get("reference")

        for plugin in gazebo.findall("plugin"):
            filename = (plugin.attrib.get("filename") or "").strip()
            if filename != "libgazebo_ros_ft_sensor.so":
                continue
            joint_name = (plugin.findtext("joint_name") or "").strip()
            if not joint_name:
                continue
            child_link = joint_child_by_name.get(joint_name)
            if not child_link:
                continue

            specs.append(
                URDFSensorSpec(
                    name=f"ft_{joint_name}",
                    type="force_torque",
                    reference=child_link,
                    pose_xyz=(0.0, 0.0, 0.0),
                    pose_rpy=(0.0, 0.0, 0.0),
                    params={},
                )
            )

        if not reference:
            continue

        for sensor in gazebo.findall("sensor"):
            sensor_name = sensor.attrib.get("name", "")
            sensor_type = sensor.attrib.get("type", "")
            if not sensor_name or not sensor_type:
                continue

            pose_xyz, pose_rpy = _pose_from_text((sensor.findtext("pose") or "").strip())

            params: dict[str, Any] = {}
            update_rate_hz = _parse_positive_float(sensor.findtext("update_rate"))
            if update_rate_hz is not None:
                params["update_rate_hz"] = update_rate_hz

            if sensor_type == "ray":
                horiz = sensor.find("ray/scan/horizontal")
                if horiz is not None:
                    params["horizontal_samples"] = _parse_int(
                        horiz.findtext("samples"),
                        128,
                    )
                    params["horizontal_min_angle"] = _parse_float(
                        horiz.findtext("min_angle"),
                        -math.pi,
                    )
                    params["horizontal_max_angle"] = _parse_float(
                        horiz.findtext("max_angle"),
                        math.pi,
                    )

                range_node = sensor.find("ray/range")
                if range_node is not None:
                    params["min_range"] = _parse_float(
                        range_node.findtext("min"),
                        0.0,
                    )
                    params["max_range"] = _parse_float(
                        range_node.findtext("max"),
                        20.0,
                    )

            elif sensor_type in ("camera", "depth"):
                cam = sensor.find("camera")
                if cam is not None:
                    params["horizontal_fov"] = _parse_float(
                        cam.findtext("horizontal_fov"),
                        math.radians(90.0),
                    )

                    image = cam.find("image")
                    if image is not None:
                        params["width"] = _parse_int(image.findtext("width"), 320)
                        params["height"] = _parse_int(image.findtext("height"), 240)

                    clip = cam.find("clip")
                    if clip is not None:
                        params["near"] = _parse_float(clip.findtext("near"), 0.05)
                        params["far"] = _parse_float(clip.findtext("far"), 20.0)

            elif sensor_type == "force_torque":
                params = {}

            elif sensor_type == "imu":
                params = {}

            else:
                continue

            specs.append(
                URDFSensorSpec(
                    name=sensor_name,
                    type=sensor_type,
                    reference=reference,
                    pose_xyz=pose_xyz,
                    pose_rpy=pose_rpy,
                    params=params,
                )
            )

    return specs
