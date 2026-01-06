import io
import builtins
import numpy as np
import pytest
from types import SimpleNamespace

import svzerodtrees.tune_bcs.rcr_tuner as rcr_module

from svzerodtrees.io.blocks import BoundaryCondition, SimParams, Vessel
from svzerodtrees.microvasculature import TreeParameters
from svzerodtrees.microvasculature.compliance import ConstantCompliance
from svzerodtrees.tune_bcs.clinical_targets import ClinicalTargets
import svzerodtrees.tune_bcs.pa_config as pa_config_module
from svzerodtrees.tune_bcs.pa_config import PAConfig
from svzerodtrees.io.config_handler import ConfigHandler
from svzerodtrees.tune_bcs.tune_space import (
    FreeParam,
    FixedParam,
    TiedParam,
    TuneSpace,
    positive,
)
from svzerodtrees.tune_bcs.impedance_tuner import ImpedanceTuner
from svzerodtrees.tune_bcs.rcr_tuner import RCRTuner
from svzerodtrees.microvasculature.structured_tree.asymmetry import alpha_beta_from_xi_eta
from tests.test_config_validation import _validate_config_connectivity


def _make_vessel(vessel_id, name, *, bc=None, length=5.0, resistance=100.0):
    config = {
        "vessel_id": vessel_id,
        "vessel_length": length,
        "vessel_name": name,
        "zero_d_element_type": "BloodVessel",
        "zero_d_element_values": {
            "R_poiseuille": resistance,
            "C": 1.0,
            "L": 1.0,
            "stenosis_coefficient": 0.0,
        },
    }
    if bc is not None:
        config["boundary_conditions"] = bc
    return Vessel.from_config(config)


def _build_pa_config(flow_profile):
    clinical_targets = ClinicalTargets(
        mpa_p=[30.0, 15.0, 22.0],
        rpa_split=0.55,
        wedge_p=12.0,
        q=5.0,
    )
    inflow_bc = BoundaryCondition.from_config(
        {
            "bc_name": "INFLOW",
            "bc_type": "FLOW",
            "bc_values": {
                "Q": flow_profile,
                "t": np.linspace(0.0, 1.0, num=len(flow_profile)).tolist(),
            },
        }
    )
    sim_params = SimParams(
        {
            "number_of_time_pts_per_cardiac_cycle": len(flow_profile),
            "number_of_cardiac_cycles": 1,
            "output_all_cycles": False,
        }
    )
    mpa = _make_vessel(0, "branch0_seg0")
    lpa_prox = _make_vessel(1, "branch1_seg0")
    rpa_prox = _make_vessel(3, "branch2_seg0")
    lpa_dist = _make_vessel(
        2, "branch1_seg1", bc={"outlet": "LPA_BC"}, resistance=150.0
    )
    rpa_dist = _make_vessel(
        4, "branch2_seg1", bc={"outlet": "RPA_BC"}, resistance=180.0
    )
    return PAConfig(
        simparams=sim_params,
        mpa=mpa,
        lpa_prox=lpa_prox,
        rpa_prox=rpa_prox,
        lpa_dist=lpa_dist,
        rpa_dist=rpa_dist,
        inflow=inflow_bc,
        wedge_p=clinical_targets.wedge_p * 1333.2,
        clinical_targets=clinical_targets,
        steady=True,
        compliance_model=None,
    )


class DummyImpedancePAConfig:
    def __init__(self, clinical_targets):
        self.clinical_targets = clinical_targets
        self.bcs = {
            "INFLOW": SimpleNamespace(Q=[5.0, 5.0], t=[0.0, 1.0])
        }
        self.created = None
        self.sim_calls = 0
        self.rpa_split = clinical_targets.rpa_split
        self.P_mpa = clinical_targets.mpa_p

    def create_impedance_trees(self, lpa_params, rpa_params, n_procs):
        self.created = (lpa_params, rpa_params, n_procs)
        self.bcs["LPA_BC"] = SimpleNamespace(Z=[1.0])
        self.bcs["RPA_BC"] = SimpleNamespace(Z=[1.0])

    def to_json(self, path):
        self.last_json = path

    def simulate(self):
        self.sim_calls += 1
        self.rpa_split = 0.60
        self.P_mpa = [32.0, 14.0, 25.0]

    def plot_mpa(self, path=None):
        self.last_plot = path


class DummyRCRBoundary:
    def __init__(self):
        self.R = 1000.0
        self.C = 1e-4


class DummyRCRPAConfig:
    last_instance = None

    def __init__(self, clinical_targets):
        self.clinical_targets = clinical_targets
        self.bcs = {"INFLOW": SimpleNamespace(Q=[10.0, 10.0])}
        self.sim_calls = 0
        self.rpa_split = clinical_targets.rpa_split
        self.P_mpa = clinical_targets.mpa_p
        DummyRCRPAConfig.last_instance = self

    @classmethod
    def from_pa_config(cls, config_handler, clinical_targets, compliance_model=None):
        return cls(clinical_targets)

    @classmethod
    def from_config_handler(cls, config_handler, clinical_targets, compliance_model=None):
        return cls(clinical_targets)

    def initialize_resistance_bcs(self):
        self.init_called = True
        self.bcs["LPA_BC"] = DummyRCRBoundary()
        self.bcs["RPA_BC"] = DummyRCRBoundary()

    def to_json(self, path):
        self.last_json = path

    def simulate(self):
        self.sim_calls += 1
        self.rpa_split = 0.60
        self.P_mpa = [31.0, 16.0, 23.0]

    def plot_mpa(self, path=None):
        self.last_plot = path


def test_pa_config_initializes_resistance_bcs_for_steady_inflow():
    pa_config = _build_pa_config([10.0, 10.0])
    pa_config.initialize_resistance_bcs()
    assert pa_config.bcs["LPA_BC"].type == "RESISTANCE"
    assert pa_config.bcs["RPA_BC"].type == "RESISTANCE"
    assert pa_config.bcs["LPA_BC"].values["Pd"] == pytest.approx(
        pa_config.clinical_targets.wedge_p * 1333.2
    )
    assert pa_config.bcs["RPA_BC"].values["R"] == pytest.approx(
        pa_config.bcs["LPA_BC"].values["R"]
    )


def test_pa_config_initializes_rcr_bcs_for_unsteady_inflow():
    pa_config = _build_pa_config([8.0, 10.0, 6.0])
    pa_config.initialize_resistance_bcs()
    assert pa_config.bcs["LPA_BC"].type == "RCR"
    assert pa_config.bcs["RPA_BC"].values["Rp"] == pytest.approx(100.0)
    assert pa_config.bcs["RPA_BC"].values["Rd"] == pytest.approx(900.0)


def test_pa_config_assemble_config_contains_all_blocks():
    pa_config = _build_pa_config([10.0, 10.0])
    pa_config.initialize_resistance_bcs()
    config = pa_config.config
    bc_names = {entry["bc_name"] for entry in config["boundary_conditions"]}
    assert bc_names == {"INFLOW", "LPA_BC", "RPA_BC"}
    assert len(config["vessels"]) == 5
    assert {junction["junction_name"] for junction in config["junctions"]} == {
        "J0",
        "J1",
        "J3",
    }


def test_tune_space_packs_and_resolves_parameters():
    tune_space = TuneSpace(
        free=[
            FreeParam(
                "comp.lpa.C",
                init=6.6e4,
                lb=1.0e4,
                ub=2.0e5,
                to_native=positive,
                from_native=np.log,
            ),
            FreeParam("lpa.alpha", init=0.9, lb=0.5, ub=0.99),
        ],
        fixed=[FixedParam("d_min", 0.01)],
        tied=[TiedParam("comp.rpa.C", other="comp.lpa.C", fn=lambda x: 2 * x)],
    )

    x0, bounds = tune_space.pack_init_and_bounds()
    assert pytest.approx(x0[0]) == np.log(6.6e4)
    assert bounds[0][0] == pytest.approx(np.log(1.0e4))
    params = tune_space.vector_to_param_dict(x0)
    assert params["comp.lpa.C"] == pytest.approx(6.6e4)
    assert params["lpa.alpha"] == pytest.approx(0.9)
    assert params["d_min"] == pytest.approx(0.01)
    assert params["comp.rpa.C"] == pytest.approx(2 * 6.6e4)


def test_pa_config_simulate_runs_with_impedance_trees(monkeypatch):
    pa_config = _build_pa_config([10.0, 10.0])

    class DummyStructuredTree:
        def __init__(self, name, time, simparams, compliance_model):
            self.name = name
            self.time = time
            self.simparams = simparams
            self.compliance_model = compliance_model

        def build(self, **kwargs):
            self._build_kwargs = kwargs

        def compute_olufsen_impedance(self, n_procs=1):
            self._n_procs = n_procs

        def create_impedance_bc(self, bc_name, outlet_id, pd, inductance=0.0):
            return BoundaryCondition.from_config(
                {
                    "bc_name": bc_name,
                    "bc_type": "IMPEDANCE",
                    "bc_values": {
                        "Z": [100.0],
                        "t": [0.0, 1.0],
                        "Pd": pd,
                        "L": inductance,
                    },
                }
            )

    def fake_simulate(config):
        import pandas as pd

        return pd.DataFrame(
            {
                "name": ["branch0_seg0", "branch3_seg0"],
                "flow_in": [0.0, 2.5],
                "flow_out": [5.0, 0.0],
                "pressure_in": [1333.2 * 20.0, 0.0],
            }
        )

    monkeypatch.setattr(pa_config_module, "StructuredTree", DummyStructuredTree)
    monkeypatch.setattr(pa_config_module.pysvzerod, "simulate", fake_simulate)

    lpa_params = TreeParameters(
        name="lpa",
        lrr=10.0,
        diameter=0.3,
        d_min=0.01,
        alpha=0.9,
        beta=0.6,
        compliance_model=ConstantCompliance(6.6e4),
        inductance=0.0,
    )
    rpa_params = TreeParameters(
        name="rpa",
        lrr=10.0,
        diameter=0.3,
        d_min=0.01,
        alpha=0.9,
        beta=0.6,
        compliance_model=ConstantCompliance(6.6e4),
        inductance=0.0,
    )

    pa_config.create_impedance_trees(lpa_params, rpa_params, n_procs=1)
    pa_config.simulate()

    assert pa_config.rpa_split == pytest.approx(0.5)
    assert pa_config.P_mpa[0] == pytest.approx(20.0)


def test_impedance_tuner_build_tree_params_uses_defaults():
    clinical_targets = ClinicalTargets(mpa_p=[30.0, 15.0, 22.0], rpa_split=0.6, wedge_p=12.0, q=5.0)
    tune_space = TuneSpace(free=[], fixed=[], tied=[])
    tuner = ImpedanceTuner(
        config_handler=SimpleNamespace(),
        mesh_surfaces_path="mesh",
        clinical_targets=clinical_targets,
        tune_space=tune_space,
        compliance_model="constant",
    )
    tuner._geom_defaults = {
        "lpa.default_diameter": 0.30,
        "rpa.default_diameter": 0.32,
    }
    params = {
        "comp.lpa.C": 6.6e4,
        "comp.rpa.C": 7.0e4,
        "lrr": 9.0,
        "d_min": 0.02,
        "lpa.alpha": 0.9,
        "lpa.beta": 0.6,
        "rpa.alpha": 0.85,
        "rpa.beta": 0.55,
    }
    lpa_params, rpa_params = tuner._build_tree_params(params)

    assert lpa_params.diameter == pytest.approx(0.30)
    assert rpa_params.diameter == pytest.approx(0.32)
    assert lpa_params.compliance_model.value == pytest.approx(6.6e4)
    assert rpa_params.compliance_model.value == pytest.approx(7.0e4)
    assert lpa_params.alpha == pytest.approx(0.9)
    assert rpa_params.beta == pytest.approx(0.55)


def test_impedance_tuning_with_inductance_builds_valid_threed_config(monkeypatch, tmp_path):
    pa_config = _build_pa_config([10.0, 10.0])

    class DummyStructuredTree:
        def __init__(self, name, time, simparams, compliance_model):
            self.name = name
            self.time = time
            self.simparams = simparams
            self.compliance_model = compliance_model

        def build(self, **kwargs):
            self._build_kwargs = kwargs

        def compute_olufsen_impedance(self, n_procs=1):
            self._n_procs = n_procs

        def create_impedance_bc(self, bc_name, outlet_id, pd, inductance=0.0):
            return BoundaryCondition.from_config(
                {
                    "bc_name": bc_name,
                    "bc_type": "IMPEDANCE",
                    "bc_values": {
                        "Z": [100.0],
                        "t": [0.0, 1.0],
                        "Pd": pd
                    },
                }
            )

    monkeypatch.setattr(pa_config_module, "StructuredTree", DummyStructuredTree)

    inductance_value = 0.05
    lpa_params = TreeParameters(
        name="lpa",
        lrr=10.0,
        diameter=0.3,
        d_min=0.01,
        alpha=0.9,
        beta=0.6,
        compliance_model=ConstantCompliance(6.6e4),
        inductance=inductance_value,
    )
    rpa_params = TreeParameters(
        name="rpa",
        lrr=10.0,
        diameter=0.3,
        d_min=0.01,
        alpha=0.9,
        beta=0.6,
        compliance_model=ConstantCompliance(6.6e4),
        inductance=inductance_value,
    )

    pa_config.create_impedance_trees(lpa_params, rpa_params, n_procs=1)
    config_handler = ConfigHandler(pa_config.config)

    class DummySurface:
        def __init__(self, filename):
            self.filename = filename

    class DummyMeshComplete:
        def __init__(self, filenames):
            self.mesh_surfaces = {
                f"surface_{idx}": DummySurface(name)
                for idx, name in enumerate(filenames)
            }

    mesh_complete = DummyMeshComplete(
        ["INFLOW.vtp", "LPA.vtp", "RPA.vtp"]
    )
    threed_coupler, _ = config_handler.generate_threed_coupler(
        simdir=str(tmp_path),
        inflow_from_0d=True,
        mesh_complete=mesh_complete,
        include_distal_vessel=True,
    )

    errors = _validate_config_connectivity(threed_coupler.config)
    assert not errors, "\n".join(errors)


def test_impedance_tuner_build_tree_params_uses_xi_eta_sym():
    clinical_targets = ClinicalTargets(mpa_p=[30.0, 15.0, 22.0], rpa_split=0.6, wedge_p=12.0, q=5.0)
    tune_space = TuneSpace(free=[], fixed=[], tied=[])
    tuner = ImpedanceTuner(
        config_handler=SimpleNamespace(),
        mesh_surfaces_path="mesh",
        clinical_targets=clinical_targets,
        tune_space=tune_space,
        compliance_model="constant",
    )
    tuner._geom_defaults = {
        "lpa.default_diameter": 0.28,
        "rpa.default_diameter": 0.31,
    }
    xi = 2.0
    eta = 0.7
    expected_alpha, expected_beta = alpha_beta_from_xi_eta(xi, eta)
    params = {
        "comp.lpa.C": 5.5e4,
        "comp.rpa.C": 5.5e4,
        "lrr": 8.0,
        "d_min": 0.02,
        "lpa.xi": xi,
        "lpa.eta_sym": eta,
        "rpa.xi": xi,
        "rpa.eta_sym": eta,
    }
    lpa_params, rpa_params = tuner._build_tree_params(params)

    assert lpa_params.alpha == pytest.approx(expected_alpha)
    assert lpa_params.beta == pytest.approx(expected_beta)
    assert rpa_params.alpha == pytest.approx(expected_alpha)
    assert rpa_params.beta == pytest.approx(expected_beta)
    assert lpa_params.xi == pytest.approx(xi)
    assert lpa_params.eta_sym == pytest.approx(eta)


def test_impedance_tuner_loss_fn_computes_weighted_loss():
    clinical_targets = ClinicalTargets(mpa_p=[30.0, 15.0, 22.0], rpa_split=0.55, wedge_p=12.0, q=5.0)
    tune_space = TuneSpace(
        free=[
            FreeParam("comp.lpa.C", init=6.6e4, lb=1.0e4, ub=2.0e5),
            FreeParam("comp.rpa.C", init=7.0e4, lb=1.0e4, ub=2.0e5),
            FreeParam("lpa.alpha", init=0.9, lb=0.5, ub=0.99),
            FreeParam("lpa.beta", init=0.6, lb=0.3, ub=0.9),
            FreeParam("rpa.alpha", init=0.88, lb=0.5, ub=0.99),
            FreeParam("rpa.beta", init=0.58, lb=0.3, ub=0.9),
            FreeParam("lrr", init=9.0, lb=4.0, ub=20.0),
        ],
        fixed=[FixedParam("d_min", 0.02)],
        tied=[],
    )
    tuner = ImpedanceTuner(
        config_handler=SimpleNamespace(),
        mesh_surfaces_path="mesh",
        clinical_targets=clinical_targets,
        tune_space=tune_space,
        compliance_model="constant",
    )
    tuner.n_procs = 2
    tuner._geom_defaults = {
        "lpa.default_diameter": 0.30,
        "rpa.default_diameter": 0.32,
        "n_outlets_scale": 1.0,
    }
    pa_config = DummyImpedancePAConfig(clinical_targets)
    x0, _ = tune_space.pack_init_and_bounds()

    loss = tuner.loss_fn(x0, pa_config)

    params = tune_space.vector_to_param_dict(x0)
    diff = np.abs(np.array(pa_config.P_mpa) - np.array(clinical_targets.mpa_p)) / clinical_targets.mpa_p
    pressure_components = diff ** 2
    weights = {"sys": 1.5, "dia": 1.0, "mean": 1.2}
    pressure_loss = (
        weights["sys"] * pressure_components[0] +
        weights["dia"] * pressure_components[1] +
        weights["mean"] * pressure_components[2]
    ) * 100.0
    flowsplit_loss = ((pa_config.rpa_split - clinical_targets.rpa_split) / clinical_targets.rpa_split) ** 2 * 100.0
    l2 = 1e-5 * (params["comp.lpa.C"] ** 2 + params["comp.rpa.C"] ** 2)
    expected = pressure_loss + flowsplit_loss + l2

    assert pa_config.created is not None
    assert pa_config.last_json == "pa_config_tuning_snapshot.json"
    assert loss == pytest.approx(expected)
    assert pa_config.created[2] == 2


def test_impedance_tuner_grid_search_init_selects_best_candidate(monkeypatch):
    clinical_targets = ClinicalTargets(mpa_p=[30.0, 15.0, 22.0], rpa_split=0.5, wedge_p=12.0, q=5.0)
    tune_space = TuneSpace(
        free=[
            FreeParam("comp.lpa.C", init=2.0, lb=0.5, ub=5.0),
            FreeParam("comp.rpa.C", init=2.0, lb=0.5, ub=5.0),
        ],
        fixed=[],
        tied=[],
    )
    tuner = ImpedanceTuner(
        config_handler=SimpleNamespace(),
        mesh_surfaces_path="mesh",
        clinical_targets=clinical_targets,
        tune_space=tune_space,
        compliance_model="constant",
    )
    tuner.grid_candidates_constant = (10.0, 5.0, 7.0)
    pa_config = DummyImpedancePAConfig(clinical_targets)

    def fake_loss(x, _):
        params = tune_space.vector_to_param_dict(x)
        return params["comp.lpa.C"]

    tuner.loss_fn = fake_loss
    x0, _ = tune_space.pack_init_and_bounds()
    best = tuner._grid_search_init(pa_config, x0)
    best_params = tune_space.vector_to_param_dict(best)

    assert best_params["comp.lpa.C"] == pytest.approx(5.0)
    assert best_params["comp.rpa.C"] == pytest.approx(5.0)


def test_rcr_tuner_capacitance_penalty_detects_imbalance():
    tuner = RCRTuner(
        config_handler=SimpleNamespace(vessel_map={}),
        mesh_surfaces_path="mesh",
        clinical_targets=ClinicalTargets(mpa_p=[30.0, 15.0, 22.0], rpa_split=0.5, wedge_p=12.0, q=5.0),
    )
    assert tuner.capacitance_penalty(1.0, 1.0) == 0.0
    assert tuner.capacitance_penalty(3.0, 1.0, ratio_max=2.0, penalty_scale=1.0) > 0.0
    assert tuner.capacitance_penalty(1.0, 3.0, ratio_max=2.0, penalty_scale=1.0) > 0.0


def test_rcr_tuner_tune_runs_with_stubbed_pa_config(monkeypatch):
    clinical_targets = ClinicalTargets(mpa_p=[30.0, 15.0, 22.0], rpa_split=0.55, wedge_p=12.0, q=5.0)
    config_handler = SimpleNamespace(vessel_map={i: i for i in range(5)})
    tuner = RCRTuner(
        config_handler=config_handler,
        mesh_surfaces_path="mesh",
        clinical_targets=clinical_targets,
    )

    monkeypatch.setattr(
        rcr_module,
        "vtp_info",
        lambda path, convert_to_cm, pulmonary: (
            {"r1": 1.0, "r2": 1.0},
            {"l1": 1.0, "l2": 1.0},
            None,
        ),
    )
    monkeypatch.setattr(rcr_module, "PAConfig", DummyRCRPAConfig)

    class _DummyFile(io.StringIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(builtins, "open", lambda *args, **kwargs: _DummyFile())

    captured = {}

    def fake_minimize(fun, x0, **kwargs):
        val = fun(np.array(x0))
        captured["loss"] = val
        return SimpleNamespace(x=np.array(x0, dtype=float), fun=val)

    monkeypatch.setattr(rcr_module, "minimize", fake_minimize)

    result = tuner.tune()

    instance = DummyRCRPAConfig.last_instance
    assert instance.init_called
    assert instance.sim_calls == 2
    assert instance.bcs["INFLOW"].Q == [5.0, 5.0]
    assert instance.last_plot is None

    diff = np.abs(np.array([31.0, 16.0, 23.0]) - np.array(clinical_targets.mpa_p)) / clinical_targets.mpa_p
    weights = np.array([1.5, 1.0, 1.2])
    pressure_loss = (np.dot(diff, weights)) ** 2 * 100.0
    flowsplit_loss = ((0.60 - clinical_targets.rpa_split) / clinical_targets.rpa_split) ** 2 * 100.0
    penalty = tuner.capacitance_penalty(1e-5, 1e-5)
    expected = pressure_loss + flowsplit_loss + penalty

    assert captured["loss"] == pytest.approx(expected)
    assert result.x.tolist() == [1000.0, 1e-05, 1000.0, 1e-05]
