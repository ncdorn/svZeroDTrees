import os
from types import SimpleNamespace

import numpy as np
import pytest

import svzerodtrees.simulation.simulation as simulation_module


class DummyInflow:
    def __init__(self, name="INFLOW", values=None, kind="periodic", steady_value=None):
        self.name = name
        self.kind = kind
        self.q = list(values) if values is not None else [4.0, 1.0, 3.0]
        self.rescale_calls = []
        self.added_flows = []
        self.steady_value = steady_value

    def rescale(self, **kwargs):
        self.rescale_calls.append(kwargs)

    def add_steady_flow(self, flow):
        self.added_flows.append(flow)
        self.q.append(flow)

    def to_dict(self):
        return {
            "bc_name": self.name,
            "bc_type": "FLOW",
            "bc_values": {"Q": self.q, "t": []},
        }


class DummyInflowFactory:
    def __init__(self):
        self.periodic_calls = []
        self.steady_calls = []
        self.default_q = [4.0, 1.0, 3.0]

    def periodic(self, path=None):
        inflow = DummyInflow(values=self.default_q, kind="periodic")
        inflow.path = path
        self.periodic_calls.append({"path": path, "inflow": inflow})
        return inflow

    def steady(self, value, name):
        inflow = DummyInflow(name=name, values=[value], kind="steady", steady_value=value)
        self.steady_calls.append({"value": value, "name": name, "inflow": inflow})
        return inflow


class DummySimulationDirectory:
    created = []

    def __init__(self, path, zerod_config=None, mesh_complete=None, **kwargs):
        self.path = path
        mesh_path = mesh_complete if mesh_complete is not None else os.path.join(path, "mesh.vtp")
        if isinstance(mesh_path, str):
            self.mesh_complete = SimpleNamespace(path=mesh_path, mesh_surfaces_dir=os.path.join(path, "surfaces"))
        else:
            self.mesh_complete = mesh_path
        self.generated = []
        self.runs = 0
        self.checked = False

    @classmethod
    def from_directory(cls, path, zerod_config=None, **kwargs):
        inst = cls(path=path, zerod_config=zerod_config, **kwargs)
        cls.created.append(inst)
        return inst

    def generate_steady_sim(self, flow_rate):
        self.generated.append(flow_rate)

    def run(self):
        self.runs += 1

    def check_simulation(self):
        self.checked = True

    def compute_pressure_drop(self):
        return 1.0, 2.0

    def flow_split(self, get_mean=True, verbose=False):
        return ({"LPA": {"q": 1.0}}, {"RPA": {"q": 2.0}})


@pytest.fixture
def simulation_env(monkeypatch, tmp_path):
    ct_state = {
        "mpa_p": [30.0, 15.0, 22.0],
        "rpa_split": 0.55,
        "wedge_p": 12.0,
        "q": 5.0,
        "rvot_flow": None,
        "ivc_flow": 1.2,
        "svc_flow": 0.8,
    }

    class DummyClinicalTargets:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        @classmethod
        def from_csv(cls, path):
            return cls(**ct_state)

    inflow_factory = DummyInflowFactory()

    DummySimulationDirectory.created = []

    monkeypatch.setattr(simulation_module, "ClinicalTargets", DummyClinicalTargets)
    monkeypatch.setattr(
        simulation_module,
        "Inflow",
        SimpleNamespace(periodic=inflow_factory.periodic, steady=inflow_factory.steady),
    )
    monkeypatch.setattr(simulation_module, "SimulationDirectory", DummySimulationDirectory)

    return SimpleNamespace(
        module=simulation_module,
        ct_state=ct_state,
        inflow_factory=inflow_factory,
        tmp_path=tmp_path,
    )


def test_simulation_rejects_unknown_bc_type(simulation_env):
    sim_mod = simulation_env.module
    with pytest.raises(ValueError):
        sim_mod.Simulation(path=simulation_env.tmp_path, bc_type="unknown")


def test_run_steady_sims_creates_three_cases(simulation_env):
    sim_mod = simulation_env.module
    simulation_env.ct_state["rvot_flow"] = None  # non-fontan

    sim = sim_mod.Simulation(path=simulation_env.tmp_path)
    start_idx = len(DummySimulationDirectory.created)

    sim.run_steady_sims()

    steady_dirs = DummySimulationDirectory.created[start_idx:]
    assert len(steady_dirs) == 3

    flows_recorded = sorted(d.generated[0] for d in steady_dirs)
    expected = sorted([max(sim.inflow.q), max(2.0, min(sim.inflow.q)), float(np.mean(sim.inflow.q))])
    assert flows_recorded == pytest.approx(expected, rel=1e-6, abs=1e-6)
    assert all(d.runs == 1 for d in steady_dirs)
    assert all(d.checked for d in steady_dirs)


def test_make_fontan_inflows_assigns_multiple_channels(simulation_env):
    sim_mod = simulation_env.module
    simulation_env.ct_state.update({"rvot_flow": 6.0, "ivc_flow": 1.5, "svc_flow": 0.7})

    sim = sim_mod.Simulation(path=simulation_env.tmp_path)
    sim.zerod_config = SimpleNamespace(inflows={})

    sim.make_fontan_inflows()

    inflows = sim.zerod_config.inflows
    assert set(inflows.keys()) == {"INFLOW", "INFLOW_SVC", "INFLOW_IVC"}

    mpa_inflow = inflows["INFLOW"]
    assert mpa_inflow.rescale_calls[-1] == {"cardiac_output": 6.0, "tsteps": 2000}

    svc_inflow = inflows["INFLOW_SVC"]
    ivc_inflow = inflows["INFLOW_IVC"]
    assert svc_inflow.steady_value == pytest.approx(0.7)
    assert ivc_inflow.steady_value == pytest.approx(1.5)
