"""Force-Torque sensor for Genesis.

This module implements a 6-DOF wrench sensor that measures the net
force and torque at a fixed joint (e.g. a wrist FT sensor) using a
recursive Newton-Euler formulation over all downstream links.

The wrench includes contributions from gravity, inertial forces
(m*a, I*alpha, gyroscopic terms), and external contact forces on
the downstream chain.  The result is expressed in the sensor frame's
local coordinates.

Because upstream Genesis (as of 1.4.5) does not ship a
``ForceTorque`` sensor, this module defines both the options class
and the sensor class, then registers them with Genesis's
``SensorManager`` and exposes the options as ``gs.sensors.ForceTorque``.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from pydantic import Field

import genesis as gs
from genesis.engine.sensors.base_sensor import (
    RigidSensorMetadataMixin,
    RigidSensorMixin,
    SimpleSensor,
    SimpleSensorMetadata,
)
from genesis.options.sensors.options import RigidSensorOptionsMixin, SimpleSensorOptions
from genesis.typing import PositiveFloat, UnitIntervalVec4Type
from genesis.utils.geom import quat_to_R, transform_by_quat, transform_quat_by_quat
from genesis.utils.misc import concat_with_tensor, qd_to_torch, tensor_to_array

if TYPE_CHECKING:
    from genesis.ext.pyrender.mesh import Mesh


# ---------------------------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------------------------


class ForceTorque(RigidSensorOptionsMixin["ForceTorqueSensor"], SimpleSensorOptions["ForceTorqueSensor"]):
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
class ForceTorqueSensorMetadata(RigidSensorMetadataMixin, SimpleSensorMetadata):
    """Shared metadata for all ForceTorque sensors of the same class."""

    # Per-sensor downstream link info (lists indexed by sensor idx).
    downstream_links_idx: list = field(default_factory=list)
    downstream_masses: list = field(default_factory=list)
    downstream_inertias: list = field(default_factory=list)


# ---------------------------------------------------------------------------------------------
# Sensor
# ---------------------------------------------------------------------------------------------


class ForceTorqueSensor(
    RigidSensorMixin[ForceTorqueSensorMetadata],
    SimpleSensor[ForceTorque, None, ForceTorqueSensorMetadata, torch.Tensor],
):
    """6-DOF force-torque sensor using recursive Newton-Euler.

    Returns ``[fx, fy, fz, tx, ty, tz]`` in the sensor frame's local
    coordinates.
    """

    # --- build -------------------------------------------------------------

    def build(self):
        super().build()

        solver = self._shared_metadata.solver
        assert solver is not None

        entity_idx = self._options.entity_idx

        if entity_idx is None or entity_idx < 0 or self._link is None:
            # RigidSensorMixin only registers attached links; retain per-sensor alignment.
            self._shared_metadata.links_idx = concat_with_tensor(self._shared_metadata.links_idx, -1)
            # Static sensor — no downstream links.
            self._shared_metadata.downstream_links_idx.append(torch.empty(0, dtype=gs.tc_int, device=gs.device))
            self._shared_metadata.downstream_masses.append(torch.empty(0, dtype=gs.tc_float, device=gs.device))
            self._shared_metadata.downstream_inertias.append(
                torch.empty(0, 3, 3, dtype=gs.tc_float, device=gs.device)
            )
            return

        entity = self._manager._sim.entities[entity_idx]
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
        masses = solver.get_links_mass(links_idx=downstream_global)
        if masses.dim() == 0:
            masses = masses.reshape(1)
        self._shared_metadata.downstream_masses.append(masses.to(gs.tc_float))

        # Genesis 1.4.5 exposes inertias through the solver, in the inertial frame.
        self._shared_metadata.downstream_inertias.append(
            solver.get_links_inertia(links_idx=downstream_global).to(gs.tc_float)
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
    def _update_raw_data(
        cls, shared_context: None, shared_metadata: ForceTorqueSensorMetadata, raw_data_T: torch.Tensor
    ):
        assert shared_metadata.solver is not None
        solver = shared_metadata.solver
        n_envs = max(solver.n_envs, 1)

        # Zero the cache first.
        raw_data_T.zero_()
        if not any(links.numel() for links in shared_metadata.downstream_links_idx):
            return

        # Gravity vector (world frame).
        gravity = solver.get_gravity()  # (3,) or (n_envs, 3)
        if gravity.dim() == 1:
            gravity = gravity.unsqueeze(0)  # (1, 3)
        gravity = gravity.to(gs.tc_float)

        # Read all link states once (world frame).
        # For n_envs == 0, these return (n_links, 3/4); we unsqueeze to (1, n_links, ...).
        all_links_pos_com = solver.get_links_pos(ref=gs.link_ref_frame.link_COM)  # (n_links, 3) or (n_envs, n_links, 3)
        all_links_quat = solver.get_links_quat()
        all_links_pos = solver.get_links_pos()
        all_inertial_quat = qd_to_torch(solver.dyn_state.links.i_quat, transpose=True, copy=True)
        if solver.n_envs == 0:
            all_inertial_quat = all_inertial_quat[0]
        all_links_ang_vel = solver.get_links_ang()
        all_links_acc = solver.get_links_acc()
        all_links_acc_ang = solver.get_links_acc_ang()

        # Convert the device-side link contact field to batch-first Torch layout.
        all_contact_forces = qd_to_torch(solver.dyn_state.links.contact_force, None, slice(None), transpose=True, copy=True)
        if solver.n_envs == 0:
            all_contact_forces = all_contact_forces[0]  # (n_links, 3)
        # else: (n_envs, n_links, 3)

        # Normalize to batched form (n_envs, n_links, ...).
        single_env = solver.n_envs == 0
        if single_env:
            all_links_pos_com = all_links_pos_com.unsqueeze(0)
            all_links_pos = all_links_pos.unsqueeze(0)
            all_inertial_quat = all_inertial_quat.unsqueeze(0)
            all_links_quat = all_links_quat.unsqueeze(0)
            all_links_ang_vel = all_links_ang_vel.unsqueeze(0)
            all_links_acc = all_links_acc.unsqueeze(0)
            all_links_acc_ang = all_links_acc_ang.unsqueeze(0)
            all_contact_forces = all_contact_forces.unsqueeze(0)

        # get_links_acc measures the link origin; Newton-Euler uses acceleration at the COM.
        com_offset = all_links_pos_com - all_links_pos
        all_links_acc = all_links_acc + torch.cross(all_links_acc_ang, com_offset, dim=-1)
        all_links_acc = all_links_acc + torch.cross(
            all_links_ang_vel, torch.cross(all_links_ang_vel, com_offset, dim=-1), dim=-1
        )

        # Genesis 1.4.5 passes a transposed (sensor channels, environments) buffer.
        n_sensors = len(shared_metadata.downstream_links_idx)
        out = raw_data_T.T.view(n_envs, n_sensors, 6)

        for s_idx in range(n_sensors):
            ds_global = shared_metadata.downstream_links_idx[s_idx]  # (n_ds,)
            masses = shared_metadata.downstream_masses[s_idx]  # (n_ds,)
            inertias = shared_metadata.downstream_inertias[s_idx]  # (n_ds, 3, 3)

            n_ds = ds_global.shape[0]
            if n_ds == 0:
                continue

            ft_global = shared_metadata.links_idx[s_idx]

            # Downstream link states (world frame).
            com_pos = all_links_pos_com[:, ds_global]  # (n_envs, n_ds, 3)
            ang_vel = all_links_ang_vel[:, ds_global]  # (n_envs, n_ds, 3)
            lin_acc = all_links_acc[:, ds_global]  # (n_envs, n_ds, 3)
            ang_acc = all_links_acc_ang[:, ds_global]  # (n_envs, n_ds, 3)
            contact_forces = all_contact_forces[:, ds_global]  # (n_envs, n_ds, 3)

            # FT sensor frame pose.
            ft_link_quat = all_links_quat[:, ft_global]
            ft_pos = (all_links_pos[:, ft_global] + transform_by_quat(
                shared_metadata.offsets_pos[:, s_idx], ft_link_quat
            )).unsqueeze(1)
            ft_quat = transform_quat_by_quat(
                shared_metadata.offsets_quat[:, s_idx], ft_link_quat
            ).unsqueeze(1)

            # World-frame inertia: I_world = R @ I_local @ R^T
            R = quat_to_R(all_inertial_quat[:, ds_global])  # (n_envs, n_ds, 3, 3)
            I_local_b = inertias if inertias.ndim == 4 else inertias.unsqueeze(0)
            I_world = R @ I_local_b @ R.transpose(-1, -2)  # (n_envs, n_ds, 3, 3)

            # Forces on each link (world frame):
            # F_joint = m * (a - g) - F_contact
            m_exp = (masses if masses.ndim == 2 else masses.unsqueeze(0)).unsqueeze(-1)  # (1, n_ds, 1)
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

    # --- debug visualization -----------------------------------------------

    def _draw_debug(self, context, envs_idx: list[int]) -> list[tuple[None, "Mesh", None]]:
        """Make force and torque arrows in the first rendered environment."""
        if self._link is None or 0 not in envs_idx:
            return []
        env_idx = context.rendered_envs_idx[0] if self._manager._sim.n_envs > 0 else None
        quat = self._link.get_quat(env_idx, relative=False).reshape((4,))
        pos = self._link.get_pos(env_idx, relative=False).reshape((3,))
        pos = pos + transform_by_quat(self._shared_metadata.offsets_pos[0, self._idx], quat)
        quat = transform_quat_by_quat(self._shared_metadata.offsets_quat[0, self._idx], quat)
        wrench = self.read(env_idx).reshape((6,))
        force = tensor_to_array(transform_by_quat(wrench[:3] * self._options.debug_force_scale, quat))
        torque = tensor_to_array(transform_by_quat(wrench[3:] * self._options.debug_torque_scale, quat))
        arrows = (
            context.make_debug_arrow(pos=tensor_to_array(pos), vec=force, color=self._options.debug_force_color),
            context.make_debug_arrow(pos=tensor_to_array(pos), vec=torque, color=self._options.debug_torque_color),
        )
        return [(None, arrow, None) for arrow in arrows if arrow is not None]


# Register the options with Genesis through SimpleSensor.__init_subclass__.
gs.sensors.ForceTorque = ForceTorque
