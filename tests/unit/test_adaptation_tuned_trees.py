"""Adaptation from the tuned full-PA per-cap structured-tree model."""

from __future__ import annotations

import numpy as np
import pytest

from svzerodtrees.adaptation.workflow import _apply_territory_homeostatic_update
from svzerodtrees.io.blocks.simulation_parameters import SimParams
from svzerodtrees.microvasculature.compliance.olufsen import OlufsenCompliance
from svzerodtrees.microvasculature.structured_tree.structuredtree import StructuredTree

TIME = [float(t) for t in np.linspace(0.0, 1.0, 16)]
SIMPARAMS = SimParams({})


def _small_tree(name="tree", diameter=0.05, max_nodes=2000):
    tree = StructuredTree(
        name=name,
        time=TIME,
        simparams=SIMPARAMS,
        compliance_model=OlufsenCompliance(2.6e5, -14.0, 1.0e5),
    )
    tree.build(
        initial_d=diameter,
        d_min=0.01,
        lrr=10.0,
        xi=2.7,
        eta_sym=0.8,
        max_nodes=max_nodes,
    )
    return tree


# ---- behavior locked before the per-cap change ---------------------------


def test_territory_homeostatic_update_scales_diameters_and_olufsen_stiffness():
    tree = _small_tree()
    d0 = np.asarray(tree.store.d, dtype=float).copy()

    metrics = _apply_territory_homeostatic_update(
        tree,
        preop_flow=10.0,
        postop_flow=15.0,
        preop_resistance=100.0,
        postop_resistance=80.0,
        iterations=2,
        wss_gain=1.0,
        ims_gain=1.0,
        compliance_gain=1.0,
    )

    flow_scale = 1.5 ** (1.0 / 3.0)
    ims_scale = 0.8 ** (1.0 / 4.0)
    total = (flow_scale * ims_scale) ** 2
    assert metrics["flow_ratio"] == pytest.approx(1.5)
    assert metrics["resistance_ratio"] == pytest.approx(0.8)
    assert metrics["total_scale"] == pytest.approx(total)
    np.testing.assert_allclose(tree.store.d, d0 * total)
    assert tree.compliance_model.k1 == pytest.approx(2.6e5 / total)
    assert tree.compliance_model.k2 == pytest.approx(-14.0)
    assert tree.compliance_model.k3 == pytest.approx(1.0e5 / total)
    assert tree.compliance_model.params["k3"] == pytest.approx(1.0e5 / total)


def test_unadapted_tree_metadata_round_trip_reproduces_tree():
    tree = _small_tree(max_nodes=500)
    rebuilt = StructuredTree.from_tree_metadata(tree.to_dict(), time=TIME, simparams=SIMPARAMS)

    assert rebuilt.max_nodes == 500
    np.testing.assert_array_equal(rebuilt.store.d, tree.store.d)


# ---- tuned per-cap model ---------------------------------------------------

import json
from pathlib import Path
from types import SimpleNamespace

import svzerodtrees.adaptation.workflow as workflow_module
from svzerodtrees.adaptation.tuned_trees import (
    MMHG_TO_DYN_PER_CM2,
    adaptation_clinical_targets,
    check_coupler_matches_tuned_model,
    load_tuned_tree_model,
    resolve_tuned_tree_budget,
    write_adapted_tuned_trees,
)
from svzerodtrees.adaptation.workflow import run_structured_tree_adaptation
from svzerodtrees.microvasculature.structured_tree.structuredtree import DEFAULT_MAX_NODES

# Caps sort as LPA_1, LPA_2, RPA_1, but the (centerline-ordered) seed names
# its BCs IMPEDANCE_0 -> RPA_1, IMPEDANCE_1 -> LPA_2, IMPEDANCE_2 -> LPA_1, so
# pairing by list position would put an LPA tree on the RPA cap.
CAPS = {
    "IMPEDANCE_0": ("/mesh/mesh-surfaces/RPA_1.vtp", "rpa", 0.06),
    "IMPEDANCE_1": ("/mesh/mesh-surfaces/LPA_2.vtp", "lpa", 0.04),
    "IMPEDANCE_2": ("/mesh/mesh-surfaces/LPA_1.vtp", "lpa", 0.05),
}
PD_MMHG = 9.988
TREE_NODES = 300


def _per_cap_trees():
    trees = []
    for bc_name, (cap, side, diameter) in sorted(CAPS.items(), key=lambda item: item[1][0]):
        tree = _small_tree(name=cap, diameter=diameter, max_nodes=TREE_NODES)
        tree.generation_mode = "per_outlet"
        tree.outlet_mapping = {
            "mode": "per_outlet",
            "side": side,
            "bc_names": [bc_name],
            "outlet_names": [cap],
        }
        trees.append(tree.to_dict())
    return trees


def _tuned_payload(trees, *, pd_mmhg=PD_MMHG, bc_order=("IMPEDANCE_0", "IMPEDANCE_1", "IMPEDANCE_2")):
    return {
        "boundary_conditions": [
            {"bc_name": "INFLOW", "bc_type": "FLOW", "bc_values": {"t": [0.0, 1.0], "Q": [1.0, 1.0]}}
        ]
        + [
            {
                "bc_name": bc,
                "bc_type": "IMPEDANCE",
                "bc_values": {"z": [1.0], "Pd": pd_mmhg * MMHG_TO_DYN_PER_CM2},
            }
            for bc in bc_order
        ],
        "trees": trees,
    }


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _mapping_payload(**overrides):
    pairs = []
    for bc_name, (cap, side, diameter) in CAPS.items():
        record = {
            "bc_name": bc_name,
            "cap_path": cap,
            "side": side,
            "raw_diameter": diameter,
            "scaled_diameter": diameter,
        }
        record.update(overrides.get(bc_name, {}))
        pairs.append(record)
    return {"version": 1, "strategy": "centerline", "pairs": pairs}


class FakeCoupler:
    def __init__(self, bc_names=tuple(CAPS), surfaces=None):
        if surfaces is None:
            surfaces = {bc: Path(CAPS[bc][0]).name for bc in bc_names}
        self.bcs = {"INFLOW": SimpleNamespace(name="INFLOW")}
        self.bcs.update({bc: SimpleNamespace(name=bc) for bc in bc_names})
        self.inflows = {"INFLOW": object()}
        self.coupling_blocks = {bc: SimpleNamespace(surface=surface) for bc, surface in surfaces.items()}
        self.tree_params = {"old": {"name": "old"}}
        self.path = "postop.json"

    def to_json(self, path):
        Path(path).write_text(
            json.dumps({name: getattr(bc, "values", {}) for name, bc in self.bcs.items()}),
            encoding="utf-8",
        )


def test_load_tuned_tree_model_pairs_bcs_from_metadata_not_list_order(tmp_path):
    trees = _per_cap_trees()  # serialized in cap order, unlike the BCs
    model = load_tuned_tree_model(_write_json(tmp_path / "tuned.json", _tuned_payload(trees)))

    assert model.per_outlet
    assert [tree.side for tree in model.trees] == ["lpa", "lpa", "rpa"]
    for bc_name, (cap, side, diameter) in CAPS.items():
        assert model.cap_for_bc(bc_name) == cap
        assert model.tree_for_bc(bc_name).side == side
        assert model.tree_for_bc(bc_name).metadata["initial_d"] == pytest.approx(diameter)
        assert model.tree_for_bc(bc_name).max_nodes == TREE_NODES
    assert model.outlet_pressure_mmhg == pytest.approx(PD_MMHG)


def test_load_tuned_tree_model_accepts_matching_outlet_cap_mapping(tmp_path):
    tuned = _write_json(tmp_path / "tuned.json", _tuned_payload(_per_cap_trees()))
    mapping = _write_json(tmp_path / "outlet_cap_mapping.json", _mapping_payload())

    model = load_tuned_tree_model(tuned, outlet_cap_mapping=mapping)

    assert model.outlet_cap_mapping == mapping


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"IMPEDANCE_0": {"cap_path": "/mesh/mesh-surfaces/LPA_1.vtp"}}, "mapping cap 'LPA_1.vtp'"),
        ({"IMPEDANCE_1": {"side": "rpa"}}, "mapping side 'rpa'"),
        ({"IMPEDANCE_2": {"scaled_diameter": 0.07}}, "mapping diameter"),
    ],
)
def test_outlet_cap_mapping_that_disagrees_with_tree_metadata_fails(tmp_path, overrides, message):
    tuned = _write_json(tmp_path / "tuned.json", _tuned_payload(_per_cap_trees()))

    with pytest.raises(ValueError, match=message):
        load_tuned_tree_model(tuned, outlet_cap_mapping=_mapping_payload(**overrides))


def test_tuned_config_without_tree_outlet_mapping_fails_instead_of_guessing(tmp_path):
    trees = _per_cap_trees()
    del trees[0]["outlet_mapping"]

    with pytest.raises(ValueError, match="never pairs caps and BCs by position"):
        load_tuned_tree_model(_write_json(tmp_path / "tuned.json", _tuned_payload(trees)))


def test_tuned_config_with_uncovered_impedance_bc_fails(tmp_path):
    payload = _tuned_payload(_per_cap_trees(), bc_order=("IMPEDANCE_0", "IMPEDANCE_1", "IMPEDANCE_2", "IMPEDANCE_3"))

    with pytest.raises(ValueError, match="IMPEDANCE_3"):
        load_tuned_tree_model(_write_json(tmp_path / "tuned.json", payload))


def test_tree_mapped_to_a_cap_of_the_other_side_fails(tmp_path):
    trees = _per_cap_trees()
    trees[0]["outlet_mapping"]["side"] = "rpa"  # LPA_1 cap

    with pytest.raises(ValueError, match="other side"):
        load_tuned_tree_model(_write_json(tmp_path / "tuned.json", _tuned_payload(trees)))


def test_tuned_impedance_bcs_with_different_pd_fail(tmp_path):
    payload = _tuned_payload(_per_cap_trees())
    payload["boundary_conditions"][1]["bc_values"]["Pd"] *= 1.01

    with pytest.raises(ValueError, match="disagree on Pd"):
        load_tuned_tree_model(_write_json(tmp_path / "tuned.json", payload))


def _clinical_targets_csv(tmp_path, *, wedge="7.0"):
    path = tmp_path / "clinical_targets.csv"
    path.write_text(
        "mpa_flow,mpa_pressure,rpa_split,wedge_pressure\n"
        f"50.0,40/3/16,0.6,{wedge}\n",
        encoding="utf-8",
    )
    return str(path)


def test_adaptation_targets_use_tuned_pd_instead_of_legacy_wedge_policy(tmp_path):
    tuned_pd = 7.0 + 0.332 * (16.0 - 7.0)  # precapillary_fraction
    targets, provenance = adaptation_clinical_targets(
        _clinical_targets_csv(tmp_path),
        tuned_outlet_pressure_dyn=tuned_pd * MMHG_TO_DYN_PER_CM2,
    )

    # The legacy default would be min(wedge, diastolic) = 3 mmHg.
    assert targets.wedge_p == pytest.approx(tuned_pd)
    assert provenance["source"] == "tuned_config_impedance_bcs"


def test_adaptation_targets_use_tuned_pd_when_no_wedge_is_measured(tmp_path):
    targets, _ = adaptation_clinical_targets(
        _clinical_targets_csv(tmp_path, wedge=""),
        tuned_outlet_pressure_dyn=1.0 * MMHG_TO_DYN_PER_CM2,  # diastolic_offset: 3 - 2
    )

    assert targets.wedge_p == pytest.approx(1.0)


def test_adaptation_targets_cross_check_the_tuning_policy(tmp_path):
    csv_path = _clinical_targets_csv(tmp_path)
    targets, provenance = adaptation_clinical_targets(
        csv_path,
        tuned_outlet_pressure_dyn=1.0 * MMHG_TO_DYN_PER_CM2,
        wedge_pressure_policy="diastolic_offset",
        diastolic_offset_mmhg=2.0,
    )
    assert targets.wedge_p == pytest.approx(1.0)
    assert provenance["policy_pd_mmhg"] == pytest.approx(1.0)

    with pytest.raises(ValueError, match="tuned with"):
        adaptation_clinical_targets(
            csv_path,
            tuned_outlet_pressure_dyn=1.0 * MMHG_TO_DYN_PER_CM2,
            wedge_pressure_policy="precapillary_fraction",
        )


def test_tuned_tree_budget_comes_from_tree_metadata(tmp_path):
    model = load_tuned_tree_model(_write_json(tmp_path / "tuned.json", _tuned_payload(_per_cap_trees())))
    tree = model.trees[0]

    assert resolve_tuned_tree_budget(tree, None) == (dict(tree.metadata), "tree_metadata")
    assert resolve_tuned_tree_budget(tree, TREE_NODES)[1] == "tree_metadata"
    with pytest.raises(ValueError, match="max_nodes=300"):
        resolve_tuned_tree_budget(tree, 100_000)

    legacy = type(tree)(tree.name, tree.side, tree.bc_names, tree.caps, {k: v for k, v in tree.metadata.items() if k != "max_nodes"})
    assert resolve_tuned_tree_budget(legacy, None)[0]["max_nodes"] == DEFAULT_MAX_NODES
    assert resolve_tuned_tree_budget(legacy, 5000)[0]["max_nodes"] == 5000


def test_adapted_diameter_scale_survives_metadata_rebuild():
    tree = _small_tree(max_nodes=500)
    tree.apply_diameter_scale(1.1)
    tree.apply_diameter_scale(1.2)
    metadata = tree.to_dict()

    assert metadata["adapted_diameter_scale"] == pytest.approx(1.32)
    rebuilt = StructuredTree.from_tree_metadata(metadata, time=TIME, simparams=SIMPARAMS)
    # store.d is float32: two scalings round differently from one combined one.
    np.testing.assert_allclose(rebuilt.store.d, tree.store.d, rtol=1e-6)
    z_adapted, _ = tree.compute_olufsen_impedance(tsteps=8)
    z_rebuilt, _ = rebuilt.compute_olufsen_impedance(tsteps=8)
    np.testing.assert_allclose(z_rebuilt, z_adapted, rtol=1e-5)

    tree.build(initial_d=0.05, d_min=0.01, lrr=10.0, xi=2.7, eta_sym=0.8, max_nodes=500)
    assert "adapted_diameter_scale" not in tree.to_dict()


def _side_scale_adapter(scales):
    def adapt(tuned_tree, tree):
        tree.apply_diameter_scale(scales[tuned_tree.side])
        return {"total_scale": scales[tuned_tree.side]}

    return adapt


def _expected_impedance(cap, diameter, scale):
    tree = _small_tree(name=cap, diameter=diameter, max_nodes=TREE_NODES)
    tree.apply_diameter_scale(scale)
    z, _ = tree.compute_olufsen_impedance(tsteps=8)
    return np.asarray(z)


def test_write_adapted_tuned_trees_puts_each_cap_tree_on_its_own_bc(tmp_path):
    model = load_tuned_tree_model(_write_json(tmp_path / "tuned.json", _tuned_payload(_per_cap_trees())))
    coupler = FakeCoupler()
    coupler.bcs.pop("INFLOW")
    scales = {"lpa": 0.9, "rpa": 1.2}

    metrics = write_adapted_tuned_trees(
        coupler, model, _side_scale_adapter(scales), time=TIME, kernel_steps=8
    )

    for bc_name, (cap, side, diameter) in CAPS.items():
        bc = coupler.bcs[bc_name]
        assert bc.values["Pd"] == pytest.approx(PD_MMHG * MMHG_TO_DYN_PER_CM2)
        np.testing.assert_allclose(bc.values["z"], _expected_impedance(cap, diameter, scales[side]))
        metadata = coupler.tree_params[cap]
        assert metadata["outlet_mapping"]["bc_names"] == [bc_name]
        assert metadata["initial_d"] == pytest.approx(diameter)
        assert metadata["max_nodes"] == TREE_NODES
        assert metadata["adapted_diameter_scale"] == pytest.approx(scales[side])
        assert metrics[cap]["max_nodes_source"] == "tree_metadata"
    assert "old" not in coupler.tree_params


def test_coupler_with_other_bcs_than_the_tuned_trees_fails(tmp_path):
    model = load_tuned_tree_model(_write_json(tmp_path / "tuned.json", _tuned_payload(_per_cap_trees())))
    coupler = FakeCoupler(bc_names=("IMPEDANCE_0", "IMPEDANCE_1", "IMPEDANCE_9"), surfaces={})

    with pytest.raises(ValueError, match=r"missing \['IMPEDANCE_2'\], unexpected \['IMPEDANCE_9'\]"):
        check_coupler_matches_tuned_model(coupler, model)


def test_coupler_that_couples_a_bc_to_another_cap_fails(tmp_path):
    model = load_tuned_tree_model(_write_json(tmp_path / "tuned.json", _tuned_payload(_per_cap_trees())))
    # Positional surface assignment: alphabetical caps against centerline BCs.
    coupler = FakeCoupler(
        surfaces={"IMPEDANCE_0": "LPA_1.vtp", "IMPEDANCE_1": "LPA_2.vtp", "IMPEDANCE_2": "RPA_1.vtp"}
    )

    with pytest.raises(ValueError, match="IMPEDANCE_0: coupled to 'LPA_1.vtp', tuned for 'RPA_1.vtp'"):
        check_coupler_matches_tuned_model(coupler, model)


class _SimDir:
    def __init__(self, path, lpa_flow, rpa_flow, lpa_res, rpa_res, coupler=None):
        self.path = path
        self._flows = (lpa_flow, rpa_flow)
        self._res = (lpa_res, rpa_res)
        self.svzerod_3Dcoupling = coupler
        self.zerod_config = None

    def flow_split(self, get_mean=True, verbose=False):
        return ({"lower": self._flows[0]}, {"lower": self._flows[1]})

    def _compute_pressure_drops(self, get_mean=True):
        return (0.0, 0.0, 0.0, *self._res)


def _run_m2(tmp_path, monkeypatch, **kwargs):
    adapted_dir = tmp_path / "adapted"
    adapted_dir.mkdir(exist_ok=True)
    simdirs = {
        "preop": _SimDir("preop", 10.0, 20.0, 100.0, 200.0),
        "postop": _SimDir("postop", 15.0, 15.0, 80.0, 250.0, coupler=FakeCoupler()),
        "adapted": SimpleNamespace(path=str(adapted_dir), svzerod_3Dcoupling=None),
    }
    monkeypatch.setattr(
        workflow_module.SimulationDirectory,
        "from_directory",
        lambda path, convert_to_cm=False: simdirs[Path(path).name],
    )
    monkeypatch.setattr(workflow_module, "_impedance_kernel_steps_from_config", lambda _cfg: 8)
    monkeypatch.setattr(workflow_module, "_resolve_inflow_time_array", lambda *_cfgs: TIME)
    tuned = _write_json(tmp_path / "tuned.json", _tuned_payload(_per_cap_trees()))
    kwargs.setdefault("parameter_set", {"iterations": 1, "wss_gain": 1.0, "ims_gain": 1.0})
    summary = run_structured_tree_adaptation(
        preop_dir=str(tmp_path / "preop"),
        postop_dir=str(tmp_path / "postop"),
        adapted_dir=str(adapted_dir),
        clinical_targets=_clinical_targets_csv(tmp_path),
        reduced_order_pa=str(tmp_path / "unused.json"),
        tree_params=str(tmp_path / "unused.csv"),
        output_root=str(tmp_path / "results"),
        tuned_config=tuned,
        outlet_cap_mapping=_write_json(tmp_path / "outlet_cap_mapping.json", _mapping_payload()),
        **kwargs,
    )
    return summary, simdirs["adapted"].svzerod_3Dcoupling


def test_m2_adapts_every_tuned_cap_tree_with_its_territory_scale(tmp_path, monkeypatch):
    summary, coupler = _run_m2(tmp_path, monkeypatch, model="M2")

    scales = {
        "lpa": (1.5 ** (1.0 / 3.0)) * (0.8 ** 0.25),
        "rpa": (0.75 ** (1.0 / 3.0)) * (1.25 ** 0.25),
    }
    assert summary["territory_deltas"]["lpa"]["total_scale"] == pytest.approx(scales["lpa"])
    assert summary["territory_deltas"]["lpa"]["n_trees"] == 2
    assert summary["territory_deltas"]["rpa"]["n_trees"] == 1
    assert "INFLOW" not in coupler.bcs
    for bc_name, (cap, side, diameter) in CAPS.items():
        bc = coupler.bcs[bc_name]
        assert bc.values["Pd"] == pytest.approx(PD_MMHG * MMHG_TO_DYN_PER_CM2)
        expected = _expected_impedance(cap, diameter, scales[side])
        # Compliance is scaled too (compliance_gain 1), so compare geometry
        # through the metadata and check the impedance moved accordingly.
        metadata = coupler.tree_params[cap]
        assert metadata["adapted_diameter_scale"] == pytest.approx(scales[side])
        assert metadata["compliance"]["params"]["k3"] == pytest.approx(1.0e5 / scales[side])
        assert np.asarray(bc.values["z"]).shape == expected.shape
        assert summary["tree_metrics"][cap]["max_nodes"] == TREE_NODES
    assert summary["outlet_pressure"]["pd_mmhg"] == pytest.approx(PD_MMHG)
    assert summary["tuned_model"]["per_outlet"] is True
    assert Path(summary["artifacts"]["adapted_coupler_json"]).exists()


def test_m2_rebuilt_adapted_trees_match_the_exported_impedance(tmp_path, monkeypatch):
    # Diameters and stiffness are both adapted (compliance_gain 1).
    _, coupler = _run_m2(tmp_path, monkeypatch, model="M2")

    for bc_name, (cap, _, _) in CAPS.items():
        rebuilt = StructuredTree.from_tree_metadata(coupler.tree_params[cap], time=TIME, simparams=SIMPARAMS)
        z, _ = rebuilt.compute_olufsen_impedance(tsteps=8)
        np.testing.assert_allclose(coupler.bcs[bc_name].values["z"], z)


def test_m2_rejects_a_node_budget_that_differs_from_the_tuned_trees(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="max_nodes=300"):
        _run_m2(tmp_path, monkeypatch, model="M2", parameter_set={"max_nodes": 100_000})


@pytest.mark.parametrize("model", ["M1", "M3"])
def test_reduced_pa_models_refuse_per_cap_tuned_trees(tmp_path, monkeypatch, model):
    with pytest.raises(ValueError, match="per-cap trees"):
        _run_m2(tmp_path, monkeypatch, model=model)


def test_coupler_without_a_coupling_block_for_an_outlet_bc_fails(tmp_path):
    model = load_tuned_tree_model(_write_json(tmp_path / "tuned.json", _tuned_payload(_per_cap_trees())))
    coupler = FakeCoupler()
    del coupler.coupling_blocks["IMPEDANCE_1"]

    with pytest.raises(ValueError, match="IMPEDANCE_1: no coupling block with a surface"):
        check_coupler_matches_tuned_model(coupler, model)


class CoupledFakeCoupler(FakeCoupler):
    """Coupler with the coupled timing used by kernel regeneration."""

    def __init__(self):
        super().__init__()
        self.simparams = SimpleNamespace(external_step_size=0.125)

    def _resolve_coupled_cardiac_period(self):
        return 1.0


def test_m2_exports_kernels_on_the_coupled_regeneration_grid(tmp_path, monkeypatch):
    coupled = CoupledFakeCoupler()
    monkeypatch.setattr(
        workflow_module, "_impedance_kernel_steps_from_config", lambda _cfg: pytest.fail("not used")
    )
    adapted_dir = tmp_path / "adapted"
    adapted_dir.mkdir()
    simdirs = {
        "preop": _SimDir("preop", 10.0, 20.0, 100.0, 200.0),
        "postop": _SimDir("postop", 15.0, 15.0, 80.0, 250.0, coupler=coupled),
        "adapted": SimpleNamespace(path=str(adapted_dir), svzerod_3Dcoupling=None),
    }
    monkeypatch.setattr(
        workflow_module.SimulationDirectory,
        "from_directory",
        lambda path, convert_to_cm=False: simdirs[Path(path).name],
    )
    run_structured_tree_adaptation(
        preop_dir=str(tmp_path / "preop"),
        postop_dir=str(tmp_path / "postop"),
        adapted_dir=str(adapted_dir),
        clinical_targets=_clinical_targets_csv(tmp_path),
        reduced_order_pa=str(tmp_path / "unused.json"),
        tree_params=str(tmp_path / "unused.csv"),
        output_root=str(tmp_path / "results"),
        tuned_config=_write_json(tmp_path / "tuned.json", _tuned_payload(_per_cap_trees())),
        model="M2",
    )

    coupler = simdirs["adapted"].svzerod_3Dcoupling
    grid = np.linspace(0.0, 1.0, 9).tolist()  # period 1.0, step 0.125 -> 8 kernel steps
    for bc_name, (cap, _, _) in CAPS.items():
        rebuilt = StructuredTree.from_tree_metadata(coupler.tree_params[cap], time=grid, simparams=SIMPARAMS)
        z, _ = rebuilt.compute_olufsen_impedance(tsteps=8)
        np.testing.assert_allclose(coupler.bcs[bc_name].values["z"], z)
