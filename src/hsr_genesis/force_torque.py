from dataclasses import dataclass
from typing import TYPE_CHECKING, Type

try:
    import gstaichi as ti
except Exception:
    import quadrants as ti
import numpy as np
import torch

import genesis as gs
from genesis.engine.sensors.base_sensor import (
    RigidSensorMetadataMixin,
    RigidSensorMixin,
    Sensor,
    SharedSensorMetadata,
)

try:
    from genesis.options.sensors import ForceTorque as ForceTorqueSensorOptions
except ImportError:
    ForceTorqueSensorOptions = None
from genesis.utils.geom import inv_transform_by_quat, transform_by_quat
from genesis.utils.misc import tensor_to_array

if TYPE_CHECKING:
    from genesis.ext.pyrender.mesh import Mesh
    from genesis.engine.sensors.sensor_manager import SensorManager
    from genesis.utils.ring_buffer import TensorRingBuffer


if ForceTorqueSensorOptions is not None:

    @dataclass
    class ForceTorqueSensorMetadata(RigidSensorMetadataMixin, SharedSensorMetadata):
        pass

    @ti.data_oriented
    class ForceTorqueSensor(RigidSensorMixin[ForceTorqueSensorMetadata], Sensor[ForceTorqueSensorMetadata]):
        def __init__(
            self,
            sensor_options: ForceTorqueSensorOptions,
            sensor_idx: int,
            data_cls: Type[tuple],
            sensor_manager: "SensorManager",
        ):
            super().__init__(sensor_options, sensor_idx, data_cls, sensor_manager)

            self._debug_force_object: "Mesh | None" = None
            self._debug_torque_object: "Mesh | None" = None

        def _get_return_format(self) -> tuple[int, ...]:
            return (6,)

        @classmethod
        def _get_cache_dtype(cls) -> torch.dtype:
            return gs.tc_float

        @classmethod
        def _update_shared_ground_truth_cache(
            cls, shared_metadata: ForceTorqueSensorMetadata, shared_ground_truth_cache: torch.Tensor
        ):
            assert shared_metadata.solver is not None

            contacts = shared_metadata.solver.collider.get_contacts(as_tensor=True, to_torch=True)
            force = contacts["force"]
            link_a = contacts["link_a"]
            link_b = contacts["link_b"]
            position = contacts["position"]

            if shared_metadata.solver.n_envs == 0:
                force, link_a, link_b, position = force[None], link_a[None], link_b[None], position[None]

            if link_a.shape[-1] == 0:
                shared_ground_truth_cache.zero_()
                return

            links_pos = shared_metadata.solver.get_links_pos()
            links_quat = shared_metadata.solver.get_links_quat()
            if shared_metadata.solver.n_envs == 0:
                links_pos = links_pos[None]
                links_quat = links_quat[None]

            sensor_links_idx = shared_metadata.links_idx
            sensors_pos = links_pos[:, sensor_links_idx]
            sensors_quat = links_quat[:, sensor_links_idx]

            mask_a = link_a[:, None] == sensor_links_idx[None, :, None]
            mask_b = link_b[:, None] == sensor_links_idx[None, :, None]
            sign = mask_b.to(dtype=gs.tc_float) - mask_a.to(dtype=gs.tc_float)

            force_signed = sign[..., None] * force[:, None]
            force_world = force_signed.sum(dim=2)

            r = position[:, None] - sensors_pos[:, :, None]
            torque_world = torch.cross(r, force_signed, dim=-1).sum(dim=2)

            force_local = inv_transform_by_quat(force_world, sensors_quat)
            torque_local = inv_transform_by_quat(torque_world, sensors_quat)

            out = shared_ground_truth_cache.reshape((max(shared_metadata.solver.n_envs, 1), -1, 6))
            out[..., :3] = force_local
            out[..., 3:] = torque_local

        @classmethod
        def _update_shared_cache(
            cls,
            shared_metadata: ForceTorqueSensorMetadata,
            shared_ground_truth_cache: torch.Tensor,
            shared_cache: torch.Tensor,
            buffered_data: "TensorRingBuffer",
        ):
            buffered_data.set(shared_ground_truth_cache)
            cls._apply_delay_to_shared_cache(shared_metadata, shared_cache, buffered_data)

        def _draw_debug(self, context, buffer_updates):
            env_idx = context.rendered_envs_idx[0] if self._manager._sim.n_envs > 0 else None
            if self._link is None:
                return

            if self._debug_force_object is not None:
                context.clear_debug_object(self._debug_force_object)
                self._debug_force_object = None
            if self._debug_torque_object is not None:
                context.clear_debug_object(self._debug_torque_object)
                self._debug_torque_object = None

            pos = self._link.get_pos(env_idx).reshape((3,))
            quat = self._link.get_quat(env_idx).reshape((4,))

            wrench = self.read(env_idx).reshape((6,))
            force_local = wrench[:3] * float(self._options.debug_force_scale)
            torque_local = wrench[3:] * float(self._options.debug_torque_scale)

            force_world = tensor_to_array(transform_by_quat(force_local, quat)).reshape((3,))
            torque_world = tensor_to_array(transform_by_quat(torque_local, quat)).reshape((3,))

            self._debug_force_object = context.draw_debug_arrow(
                pos=np.array(pos, dtype=float),
                vec=np.array(force_world, dtype=float),
                color=self._options.debug_force_color,
            )
            self._debug_torque_object = context.draw_debug_arrow(
                pos=np.array(pos, dtype=float),
                vec=np.array(torque_world, dtype=float),
                color=self._options.debug_torque_color,
            )

    # Register sensor manually for Genesis 0.4.6 (register_sensor decorator removed)
    from genesis.engine.sensors.sensor_manager import SensorManager
    SensorManager.SENSOR_TYPES_MAP[ForceTorqueSensorOptions] = (
        ForceTorqueSensorMetadata,
        tuple,
    )
