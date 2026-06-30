"""Force-Torque sensor for Genesis.

This module implements a 6-DOF wrench sensor that measures the net
force and torque at a fixed joint (e.g. a wrist FT sensor) using a
recursive Newton-Euler formulation over all downstream links.

The wrench includes contributions from gravity, inertial forces
(m*a, I*alpha, gyroscopic terms), and external contact forces on
the downstream chain.  The result is expressed in the sensor frame's
local coordinates.

Because upstream Genesis (as of 0.4.6) does not ship a
``ForceTorque`` sensor, this module defines both the options class
and the sensor class, then registers them with Genesis's
``SensorManager`` and exposes the options as ``gs.sensors.ForceTorque``.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from pydantic import Field

import genesis as gs
from genesis.engine.sensors.base_sensor import (
    RigidSensorMetadataMixin,
    RigidSensorMixin,
    Sensor,
    SharedSensorMetadata,
)
from genesis.options.sensors.options import RigidSensorOptionsMixin
from genesis.typing import PositiveFloat, UnitIntervalVec4Type
from genesis.utils.geom import quat_to_R, transform_by_quat
from genesis.utils.misc import qd_to_torch, tensor_to_array

if TYPE_CHECKING:
    from genesis.ext.pyrender.mesh import Mesh
    from genesis.engine.sensors.sensor_manager import SensorManager
    from genesis.utils.ring_buffer import TensorRingBuffer


# ---------------------------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------------------------


class ForceTorque(RigidSensorOptionsMixin["ForceTorqueSensor"]):
    """Options for a 6-DOF force-torque sensor.

    The sensor is attached to a RigidLink (the "FT sensor frame") and
    measures the wrench transmitted through that link to all
    *downstream* links (the hand/gripper chain).

    Parameters
    ----------
    downstream_link_idxs_local : list[int]
        Local indices of all links downstream of (and including) the FT
        sensor frame link.  The wrench is computed by summing
        gravity, inertial, and contact forces over these links.
    debug_force_scale : float, optional
        Scale factor for the debug force arrow. Defaults to 0.01.
    debug_force_color : array-like[float, float, float, float], optional
        RGBA color of the debug force arrow. Defaults to (1.0, 0.0, 0.0, 0.6).
    debug_torque_scale : float, optional
        Scale factor for the debug torque arrow. Defaults to 0.01.
    debug_torque_color : array-like[float, float, float, float], optional
        RGBA color of the debug torque arrow. Defaults to (0.0, 1.0, 0.0, 0.6).
    """

    downstream_link_idxs_local: list[int] = Field(default_factory=list)
    debug_force_scale: PositiveFloat = 0.01
    debug_force_color: UnitIntervalVec4Type = (1.0, 0.0, 0.0, 0.6)
    debug_torque_scale: PositiveFloat = 0.01
    debug_torque_color: UnitIntervalVec4Type = (0.0, 1.0, 0.0, 0.6)


ForceTorqueOptions = ForceTorque  # alias for internal use


# ---------------------------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------------------------


@dataclass
class ForceTorqueSensorMetadata(RigidSensorMetadataMixin, SharedSensorMetadata):
    """Shared metadata for all ForceTorque sensors of the same class."""

    # Per-sensor downstream link info (lists indexed by sensor idx).
    downstream_links_idx: list = field(default_factory=list)
    downstream_masses: list = field(default_factory=list)
    downstream_inertias: list = field(default_factory=list)


# ---------------------------------------------------------------------------------------------
# Sensor
# ---------------------------------------------------------------------------------------------


class ForceTorqueSensor(RigidSensorMixin[ForceTorqueSensorMetadata], Sensor[ForceTorque, ForceTorqueSensorMetadata]):
    """6-DOF force-torque sensor using recursive Newton-Euler.

    Returns ``[fx, fy, fz, tx, ty, tz]`` in the sensor frame's local
    coordinates.
    """

    def __init__(self, sensor_options: ForceTorque, sensor_idx: int, sensor_manager: "SensorManager"):
        super().__init__(sensor_options, sensor_idx, sensor_manager)
        self._debug_force_object: "Mesh | None" = None
        self._debug_torque_object: "Mesh | None" = None

    # --- build -------------------------------------------------------------

    def build(self):
        super().build()

        solver = self._shared_metadata.solver
        assert solver is not None

        entity_idx = self._options.entity_idx

        if entity_idx is None or entity_idx < 0 or self._link is None:
            # Static sensor — no downstream links.
            self._shared_metadata.downstream_links_idx.append(torch.empty(0, dtype=gs.tc_int, device=gs.device))
            self._shared_metadata.downstream_masses.append(torch.empty(0, dtype=gs.tc_float, device=gs.device))
            self._shared_metadata.downstream_inertias.append(
                torch.empty(0, 3, 3, dtype=gs.tc_float, device=gs.device)
            )
            return

        entity = solver.entities[entity_idx]
        link_start = int(entity.link_start)
        ft_link_idx_local = self._options.link_idx_local

        # Discover downstream links by traversing the kinematic tree.
        # If the user provided downstream_link_idxs_local, use that;
        # otherwise, auto-discover by finding all children of the FT link.
        downstream_local = list(self._options.downstream_link_idxs_local)
        if not downstream_local:
            downstream_local = self._discover_downstream_links(entity, ft_link_idx_local)

        if not downstream_local:
            self._shared_metadata.downstream_links_idx.append(torch.empty(0, dtype=gs.tc_int, device=gs.device))
            self._shared_metadata.downstream_masses.append(torch.empty(0, dtype=gs.tc_float, device=gs.device))
            self._shared_metadata.downstream_inertias.append(
                torch.empty(0, 3, 3, dtype=gs.tc_float, device=gs.device)
            )
            return

        # Convert local indices to global solver indices.
        downstream_global = torch.tensor(
            [li + link_start for li in downstream_local], dtype=gs.tc_int, device=gs.device
        )
        self._shared_metadata.downstream_links_idx.append(downstream_global)

        # Cache masses (constant after build).
        masses = solver.get_links_inertial_mass(links_idx=downstream_global)
        if masses.dim() == 0:
            masses = masses.reshape(1)
        self._shared_metadata.downstream_masses.append(masses.to(gs.tc_float))

        # Cache inertia tensors in link-local frame (will be rotated to world per-call).
        inertias = []
        for li in downstream_local:
            link = entity.links[li]
            I_local = np.array(link._inertial_i, dtype=np.float64)
            inertias.append(I_local)
        self._shared_metadata.downstream_inertias.append(
            torch.tensor(np.stack(inertias), device=gs.device, dtype=gs.tc_float)
        )

    @staticmethod
    def _discover_downstream_links(entity, root_idx_local: int) -> list[int]:
        """Traverse the kinematic tree to find all links downstream of root.

        Returns a list of local link indices, starting with root itself,
        followed by all children (depth-first).
        """
        downstream = []
        stack = [root_idx_local]
        while stack:
            li = stack.pop()
            downstream.append(li)
            for j, link in enumerate(entity.links):
                parent_local = int(link.parent_idx) - int(entity.link_start)
                if parent_local == li:
                    stack.append(j)
        return downstream

    # --- format / dtype ----------------------------------------------------

    def _get_return_format(self) -> tuple[int, ...]:
        return (6,)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    # --- ground truth ------------------------------------------------------

    @classmethod
    def _update_shared_ground_truth_cache(
        cls, shared_metadata: ForceTorqueSensorMetadata, shared_ground_truth_cache: torch.Tensor
    ):
        assert shared_metadata.solver is not None
        solver = shared_metadata.solver
        n_envs = max(solver.n_envs, 1)

        # Zero the cache first.
        shared_ground_truth_cache.zero_()

        # Gravity vector (world frame).
        gravity = solver.get_gravity()  # (3,) or (n_envs, 3)
        if gravity.dim() == 1:
            gravity = gravity.unsqueeze(0)  # (1, 3)
        gravity = gravity.to(gs.tc_float)

        # Read all link states once (world frame).
        # For n_envs == 0, these return (n_links, 3/4); we unsqueeze to (1, n_links, ...).
        all_links_pos_com = solver.get_links_pos(ref="link_com")  # (n_links, 3) or (n_envs, n_links, 3)
        all_links_quat = solver.get_links_quat()
        all_links_ang_vel = solver.get_links_ang()
        all_links_acc = solver.get_links_acc()
        all_links_acc_ang = solver.get_links_acc_ang()

        # Contact forces on all links.
        # solver.links_state.contact_force has shape (3, n_links) or (3, n_links, n_envs) in qd layout.
        # Use qd_to_torch for proper conversion.
        all_contact_forces = qd_to_torch(solver.links_state.contact_force, None, slice(None), transpose=True, copy=True)
        if solver.n_envs == 0:
            all_contact_forces = all_contact_forces[0]  # (n_links, 3)
        # else: (n_envs, n_links, 3)

        # Normalize to batched form (n_envs, n_links, ...).
        single_env = solver.n_envs == 0
        if single_env:
            all_links_pos_com = all_links_pos_com.unsqueeze(0)
            all_links_quat = all_links_quat.unsqueeze(0)
            all_links_ang_vel = all_links_ang_vel.unsqueeze(0)
            all_links_acc = all_links_acc.unsqueeze(0)
            all_links_acc_ang = all_links_acc_ang.unsqueeze(0)
            all_contact_forces = all_contact_forces.unsqueeze(0)

        # Reshape cache for per-sensor writes: (n_envs, n_sensors, 6) then permute to (n_sensors*6, n_envs).
        n_sensors = len(shared_metadata.downstream_links_idx)
        out = shared_ground_truth_cache.reshape(n_envs, n_sensors, 6)

        for s_idx in range(n_sensors):
            ds_global = shared_metadata.downstream_links_idx[s_idx]  # (n_ds,)
            masses = shared_metadata.downstream_masses[s_idx]  # (n_ds,)
            inertias = shared_metadata.downstream_inertias[s_idx]  # (n_ds, 3, 3)
            ft_global = shared_metadata.links_idx[s_idx]  # scalar tensor

            n_ds = ds_global.shape[0]
            if n_ds == 0:
                continue

            # Downstream link states (world frame).
            com_pos = all_links_pos_com[:, ds_global]  # (n_envs, n_ds, 3)
            quats = all_links_quat[:, ds_global]  # (n_envs, n_ds, 4)
            ang_vel = all_links_ang_vel[:, ds_global]  # (n_envs, n_ds, 3)
            lin_acc = all_links_acc[:, ds_global]  # (n_envs, n_ds, 3)
            ang_acc = all_links_acc_ang[:, ds_global]  # (n_envs, n_ds, 3)
            contact_forces = all_contact_forces[:, ds_global]  # (n_envs, n_ds, 3)

            # FT sensor frame pose.
            ft_pos = all_links_pos_com[:, [ft_global]]  # (n_envs, 1, 3)
            ft_quat = all_links_quat[:, [ft_global]]  # (n_envs, 1, 4)

            # World-frame inertia: I_world = R @ I_local @ R^T
            R = quat_to_R(quats)  # (n_envs, n_ds, 3, 3)
            I_local_b = inertias.unsqueeze(0).expand(n_envs, -1, -1, -1)
            I_world = R @ I_local_b @ R.transpose(-1, -2)  # (n_envs, n_ds, 3, 3)

            # Forces on each link (world frame):
            # F_joint = m * (a - g) - F_contact
            m_exp = masses.unsqueeze(0).unsqueeze(-1)  # (1, n_ds, 1)
            F_link = m_exp * (lin_acc - gravity.unsqueeze(1)) - contact_forces  # (n_envs, n_ds, 3)

            # Torques on each link (world frame, about FT sensor origin):
            # T = I*alpha + omega x (I*omega) + r x F
            I_alpha = (I_world @ ang_acc.unsqueeze(-1)).squeeze(-1)  # (n_envs, n_ds, 3)
            I_omega = (I_world @ ang_vel.unsqueeze(-1)).squeeze(-1)  # (n_envs, n_ds, 3)
            gyro = torch.cross(ang_vel, I_omega, dim=-1)  # (n_envs, n_ds, 3)
            r_rel = com_pos - ft_pos  # (n_envs, n_ds, 3)
            r_x_F = torch.cross(r_rel, F_link, dim=-1)  # (n_envs, n_ds, 3)
            T_link = I_alpha + gyro + r_x_F  # (n_envs, n_ds, 3)

            # Sum over downstream links.
            F_total = F_link.sum(dim=1)  # (n_envs, 3)
            T_total = T_link.sum(dim=1)  # (n_envs, 3)

            # Transform to FT sensor local frame.
            R_ft = quat_to_R(ft_quat).squeeze(1)  # (n_envs, 3, 3)
            F_local = (R_ft.transpose(-1, -2) @ F_total.unsqueeze(-1)).squeeze(-1)  # (n_envs, 3)
            T_local = (R_ft.transpose(-1, -2) @ T_total.unsqueeze(-1)).squeeze(-1)  # (n_envs, 3)

            out[:, s_idx, :3] = F_local
            out[:, s_idx, 3:] = T_local

    # --- measured cache (with delay) ---------------------------------------

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

    # --- debug visualization -----------------------------------------------

    def _draw_debug(self, context):
        """Draw force and torque arrows at the sensor location.

        Only draws for the first rendered environment.
        """
        env_idx = context.rendered_envs_idx[0] if self._manager._sim.n_envs > 0 else None
        if self._link is None:
            return

        # Clear previous debug objects.
        if self._debug_force_object is not None:
            context.clear_debug_object(self._debug_force_object)
            self._debug_force_object = None
        if self._debug_torque_object is not None:
            context.clear_debug_object(self._debug_torque_object)
            self._debug_torque_object = None

        pos = tensor_to_array(self._link.get_pos(env_idx)).reshape((3,))
        quat = self._link.get_quat(env_idx).reshape((4,))

        wrench = self.read(env_idx).reshape((6,))
        force_local = wrench[:3] * float(self._options.debug_force_scale)
        torque_local = wrench[3:] * float(self._options.debug_torque_scale)

        force_world = tensor_to_array(transform_by_quat(force_local, quat)).reshape((3,))
        torque_world = tensor_to_array(transform_by_quat(torque_local, quat)).reshape((3,))

        self._debug_force_object = context.draw_debug_arrow(
            pos=pos,
            vec=force_world,
            color=self._options.debug_force_color,
        )
        self._debug_torque_object = context.draw_debug_arrow(
            pos=pos,
            vec=torque_world,
            color=self._options.debug_torque_color,
        )


# ---------------------------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------------------------

# Auto-registration happens in Sensor.__init_subclass__ when the class is
# defined with Sensor[ForceTorque, ForceTorqueSensorMetadata].  We also
# expose the options class as gs.sensors.ForceTorque for convenience.
gs.sensors.ForceTorque = ForceTorque
