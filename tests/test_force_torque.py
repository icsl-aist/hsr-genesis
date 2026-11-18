"""Wrench math, registration, and the native Genesis sensor lifecycle."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import genesis as gs
import pytest
import torch
from genesis.engine.sensors.base_sensor import SimpleSensor
from genesis.engine.sensors.sensor_manager import SensorManager


# Load directly so these focused checks do not depend on other HSR adapters.
spec = importlib.util.spec_from_file_location(
    "_test_force_torque_impl", Path(__file__).parents[1] / "src/hsr_genesis/force_torque.py"
)
ft = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ft
spec.loader.exec_module(ft)


def test_force_torque_registers_with_native_sensor_pipeline():
    assert issubclass(ft.ForceTorqueSensor, SimpleSensor)
    assert SensorManager.SENSOR_TYPES_MAP[ft.ForceTorque] is ft.ForceTorqueSensor
    assert ft.ForceTorqueSensor._return_data_cls is torch.Tensor
    assert ft.ForceTorque(entity_idx=None).entity_idx == -1


@pytest.mark.parametrize("n_envs", [0, 2])
def test_wrench_uses_new_solver_layout_and_transposed_cache(monkeypatch, n_envs):
    batch = max(n_envs, 1)
    tensor = lambda value: torch.tensor(value, dtype=gs.tc_float, device=gs.device)
    positions = tensor([[0, 0, 0], [1, 0, 0]]).expand(batch, -1, -1)
    quat = tensor([[1, 0, 0, 0], [1, 0, 0, 0]]).expand(batch, -1, -1)
    zeros = torch.zeros_like(positions)
    # One Newton of external upward contact on the downstream link.
    contacts = tensor([[0, 0, 0], [0, 0, 1]]).expand(batch, -1, -1)
    state = SimpleNamespace(contact_force=contacts, i_quat=quat)
    unbatch = lambda value: value[0] if n_envs == 0 else value
    solver = SimpleNamespace(
        n_envs=n_envs,
        dyn_state=SimpleNamespace(links=state),
        get_gravity=lambda: tensor([0, 0, -10]),
        get_links_pos=lambda **kwargs: unbatch(positions),
        get_links_quat=lambda: unbatch(quat),
        get_links_ang=lambda: unbatch(zeros),
        get_links_acc=lambda: unbatch(zeros),
        get_links_acc_ang=lambda: unbatch(zeros),
    )
    monkeypatch.setattr(ft, "qd_to_torch", lambda field, *args, **kwargs: field)
    metadata = ft.ForceTorqueSensorMetadata(
        solver=solver,
        links_idx=torch.tensor([0, -1], device=gs.device),
        downstream_links_idx=[torch.tensor([1], device=gs.device), torch.empty(0, dtype=torch.long, device=gs.device)],
        downstream_masses=[tensor([2]), tensor([])],
        downstream_inertias=[torch.eye(3, dtype=gs.tc_float, device=gs.device).unsqueeze(0), tensor([])],
        offsets_pos=torch.zeros(batch, 2, 3, dtype=gs.tc_float, device=gs.device),
        offsets_quat=tensor([1, 0, 0, 0]).expand(batch, 2, -1),
    )
    # Sensor zero reads 2*10 - 1 = 19 N upward and -19 Nm about Y.
    # Sensor one is static and returns zero. Multiple cache rows expose layout errors.
    raw_data_T = torch.full((12, batch), float("nan"), dtype=gs.tc_float, device=gs.device)
    ft.ForceTorqueSensor._update_raw_data(None, metadata, raw_data_T)
    expected = tensor([0, 0, 19, 0, -19, 0, 0, 0, 0, 0, 0, 0]).expand(batch, -1)
    torch.testing.assert_close(raw_data_T.T, expected)


@pytest.mark.parametrize("n_envs", [0, 2])
def test_wrench_transports_com_acceleration_and_rotates_sensor_frame(monkeypatch, n_envs):
    batch = max(n_envs, 1)
    tensor = lambda value: torch.tensor(value, dtype=gs.tc_float, device=gs.device)
    origin = tensor([[0, 0, 0], [1, 0, 0]]).expand(batch, -1, -1)
    com = tensor([[0, 0, 0], [2, 0, 0]]).expand(batch, -1, -1)
    identity = tensor([1, 0, 0, 0])
    quarter_turn = tensor([2**-0.5, 0, 0, 2**-0.5])
    quats = identity.expand(batch, 2, -1)
    inertial_quats = torch.stack((identity, quarter_turn)).expand(batch, -1, -1)
    zeros = torch.zeros_like(origin)
    unbatch = lambda value: value[0] if n_envs == 0 else value
    solver = SimpleNamespace(
        n_envs=n_envs,
        dyn_state=SimpleNamespace(links=SimpleNamespace(contact_force=zeros, i_quat=inertial_quats)),
        get_gravity=lambda: tensor([0, 0, 0]),
        get_links_pos=lambda **kwargs: unbatch(com if kwargs.get("ref") == gs.link_ref_frame.link_COM else origin),
        get_links_quat=lambda: unbatch(quats),
        get_links_ang=lambda: unbatch(tensor([[0, 0, 0], [0, 0, 2]]).expand(batch, -1, -1)),
        get_links_acc=lambda: unbatch(tensor([[0, 0, 0], [1, 2, 0]]).expand(batch, -1, -1)),
        get_links_acc_ang=lambda: unbatch(tensor([[0, 0, 0], [1, 0, 3]]).expand(batch, -1, -1)),
    )
    monkeypatch.setattr(ft, "qd_to_torch", lambda field, *args, **kwargs: field)
    masses = tensor([2]) if n_envs == 0 else tensor([[2], [3]])
    inertia = torch.diag(tensor([1, 2, 4])).unsqueeze(0)
    if n_envs:
        inertia = inertia.expand(batch, -1, -1, -1)
    metadata = ft.ForceTorqueSensorMetadata(
        solver=solver,
        links_idx=torch.tensor([0], device=gs.device),
        downstream_links_idx=[torch.tensor([1], device=gs.device)],
        downstream_masses=[masses], downstream_inertias=[inertia],
        offsets_pos=tensor([0.5, 0, 0]).expand(batch, 1, -1),
        offsets_quat=quarter_turn.expand(batch, 1, -1),
    )
    raw = torch.empty((6, batch), dtype=gs.tc_float, device=gs.device)
    ft.ForceTorqueSensor._update_raw_data(None, metadata, raw)
    # a_COM = (1,2,0) + alpha x (1,0,0) + w x (w x (1,0,0)) = (-3,5,0).
    # Rotated inertia yields I*alpha=(2,0,12); lever arm from sensor is (1.5,0,0).
    # Finally, the sensor's 90-degree rotation maps world (x,y,z) to (y,-x,z).
    expected = tensor([[10, 6, 0, 0, -2, 27]])
    if n_envs:
        expected = tensor([[10, 6, 0, 0, -2, 27], [15, 9, 0, 0, -2, 34.5]])
    torch.testing.assert_close(raw.T, expected, atol=1e-5, rtol=1e-5)


def test_force_torque_native_scene_lifecycle():
    scene = gs.Scene(show_viewer=False)
    box = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), fixed=True))
    # Add the static sensor first to exercise shared link-index alignment even
    # when the manager sorts the sensors by their attachment during build.
    static = scene.add_sensor(ft.ForceTorque())
    attached = scene.add_sensor(ft.ForceTorque(entity_idx=box.idx, link_idx_local=0))
    try:
        scene.build()
        scene.step()
        measured = attached.read()
        ground_truth = attached.read_ground_truth()
        assert measured.shape == ground_truth.shape == (6,)
        assert torch.isfinite(measured).all()
        torch.testing.assert_close(measured, ground_truth)
        torch.testing.assert_close(static.read(), torch.zeros_like(measured))
        torch.testing.assert_close(static.read_ground_truth(), torch.zeros_like(measured))
    finally:
        scene.destroy()
