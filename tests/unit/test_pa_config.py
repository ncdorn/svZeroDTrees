from __future__ import annotations

import numpy as np
import pytest

import svzerodtrees.tune_bcs.pa_config as pa_config_module
from svzerodtrees.io.blocks import BoundaryCondition, SimParams, Vessel
from svzerodtrees.microvasculature.compliance import ConstantCompliance
from svzerodtrees.microvasculature.treeparams import TreeParameters
from svzerodtrees.tune_bcs.clinical_targets import ClinicalTargets
from svzerodtrees.tune_bcs.pa_config import PAConfig


def _make_vessel(vessel_id, name, *, bc=None, resistance=100.0):
    config = {
        "vessel_id": vessel_id,
        "vessel_length": 5.0,
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


@pytest.fixture
def pa_config():
    targets = ClinicalTargets(
        mpa_p=[30.0, 15.0, 22.0],
        lpa_p=[25.0, 12.0, 18.0],
        rpa_p=[25.0, 12.0, 18.0],
        q=5.0,
        rpa_split=0.55,
        wedge_p=12.0,
    )
    inflow = BoundaryCondition.from_config(
        {
            "bc_name": "INFLOW",
            "bc_type": "FLOW",
            "bc_values": {"Q": [5.0, 5.0], "t": [0.0, 1.0]},
        }
    )
    return PAConfig(
        simparams=SimParams(
            {
                "number_of_time_pts_per_cardiac_cycle": 2,
                "number_of_cardiac_cycles": 1,
            }
        ),
        mpa=_make_vessel(99, "branch99_seg0"),
        lpa_prox=_make_vessel(98, "branch98_seg0"),
        rpa_prox=_make_vessel(97, "branch97_seg0"),
        lpa_dist=_make_vessel(96, "branch96_seg0", bc={"outlet": "LPA_BC"}),
        rpa_dist=_make_vessel(95, "branch95_seg0", bc={"outlet": "RPA_BC"}),
        inflow=inflow,
        wedge_p=targets.wedge_p,
        clinical_targets=targets,
        steady=True,
    )


def test_initialize_config_maps_assigns_ids_names_connectivity_and_junctions(pa_config):
    assert pa_config.mpa.id == 0
    assert pa_config.mpa.name == "branch0_seg0"
    assert pa_config.lpa_prox.id == 1
    assert pa_config.lpa_dist.id == 2
    assert pa_config.rpa_prox.id == 3
    assert pa_config.rpa_dist.id == 4

    assert pa_config.mpa.children == [pa_config.lpa_prox, pa_config.rpa_prox]
    assert pa_config.lpa_prox.children == [pa_config.lpa_dist]
    assert pa_config.rpa_prox.children == [pa_config.rpa_dist]
    assert set(pa_config.vessel_map) == {0, 1, 2, 3, 4}
    assert set(pa_config.junctions) == {"J0", "J1", "J3"}


def test_initialize_resistance_bcs_selects_resistance_for_steady_inflow(pa_config):
    pa_config.initialize_resistance_bcs()

    assert pa_config.bcs["LPA_BC"].type == "RESISTANCE"
    assert pa_config.bcs["RPA_BC"].type == "RESISTANCE"
    assert pa_config.bcs["LPA_BC"].values["R"] == pytest.approx(1000.0)
    assert pa_config.bcs["LPA_BC"].values["Pd"] == pytest.approx(12.0 * 1333.2)


def test_initialize_resistance_bcs_selects_rcr_for_unsteady_inflow(pa_config):
    pa_config.inflow.Q = [5.0, 6.0]
    pa_config.initialize_resistance_bcs()

    assert pa_config.bcs["LPA_BC"].type == "RCR"
    assert pa_config.bcs["RPA_BC"].type == "RCR"
    assert pa_config.bcs["LPA_BC"].values["Rp"] == pytest.approx(100.0)
    assert pa_config.bcs["LPA_BC"].values["Rd"] == pytest.approx(900.0)
    assert pa_config.bcs["LPA_BC"].values["C"] == pytest.approx(1e-4)


def test_assemble_config_emits_deterministic_blocks(pa_config):
    pa_config.initialize_resistance_bcs()
    pa_config.assemble_config()

    config = pa_config.config
    assert [vessel["vessel_id"] for vessel in config["vessels"]] == [0, 1, 3, 2, 4]
    assert [bc["bc_name"] for bc in config["boundary_conditions"]] == [
        "INFLOW",
        "RPA_BC",
        "LPA_BC",
    ]
    assert [junction["junction_name"] for junction in config["junctions"]] == [
        "J0",
        "J1",
        "J3",
    ]


def test_create_steady_trees_uses_structured_tree_resistance_bcs(monkeypatch, pa_config):
    created = []

    class FakeStructuredTree:
        def __init__(self, name, time, simparams=None, **_kwargs):
            self.name = name
            self.time = time
            self.simparams = simparams
            self.build_kwargs = None
            created.append(self)

        def build(self, **kwargs):
            self.build_kwargs = kwargs

        def create_resistance_bc(self, name, Pd=0.0):
            return BoundaryCondition.from_config(
                {
                    "bc_name": name,
                    "bc_type": "RESISTANCE",
                    "bc_values": {"R": 42.0 if name == "LPA_BC" else 84.0, "Pd": Pd},
                }
            )

    monkeypatch.setattr(pa_config_module, "StructuredTree", FakeStructuredTree)
    lpa_params = TreeParameters(
        name="lpa",
        lrr=2.0,
        diameter=0.5,
        d_min=0.1,
        alpha=0.9,
        beta=0.6,
        compliance_model=ConstantCompliance(1.0),
    )
    rpa_params = TreeParameters(
        name="rpa",
        lrr=3.0,
        diameter=0.6,
        d_min=0.2,
        alpha=0.8,
        beta=0.5,
        compliance_model=ConstantCompliance(1.0),
    )

    pa_config.create_steady_trees(lpa_params, rpa_params)

    assert [tree.name for tree in created] == ["lpa_tree", "rpa_tree"]
    assert created[0].build_kwargs == {
        "initial_d": 0.5,
        "d_min": 0.1,
        "lrr": 2.0,
        "alpha": 0.9,
        "beta": 0.6,
    }
    assert pa_config.bcs["LPA_BC"].R == pytest.approx(42.0)
    assert pa_config.bcs["RPA_BC"].R == pytest.approx(84.0)
