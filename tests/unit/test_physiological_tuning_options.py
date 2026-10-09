"""Physiological full-PA tuning controls (see docs/pulmonary_tuning_model.md)."""

import json
import math

import numpy as np
import pytest

from svzerodtrees.io.blocks.vessel import Vessel
from svzerodtrees.tune_bcs.clinical_targets import (
    ClinicalTargets,
    resolve_outlet_pressure,
    validate_outlet_pressure,
)
from svzerodtrees.tune_bcs.impedance_tuner import ImpedanceTuner, _clip_to_bounds_within_tolerance
from svzerodtrees.tune_bcs.objective import resolve_tuning_objective
from svzerodtrees.tune_bcs.pipeline_options import (
    SUPPORTED_IMPEDANCE_KEYS,
    resolve_polish,
    validate_tune_space_names,
)
from svzerodtrees.tune_bcs.tuning_diagnostics import (
    RIGID_COMPLIANCE_THRESHOLD,
    apply_proximal_compliance,
    matched_wall_elasticity_modulus,
    svpp_bracket,
)
from svzerodtrees.tuning.iteration import _resolve_impedance_config, evaluate_iteration_gate

TUNE_SPACE = {
    "free": [
        {"name": "lpa.xi", "init": 2.76, "lb": 2.33, "ub": 3.0},
        {"name": "comp.lpa.k3", "init": 1.0e5, "lb": 5.0e4, "ub": 4.0e5,
         "to_native": "positive", "from_native": "log"},
    ],
    "fixed": [{"name": "comp.lpa.k1", "value": 2.6e5}, {"name": "comp.lpa.k2", "value": -14.0}],
    "tied": [{"name": "comp.rpa.k3", "other": "comp.lpa.k3", "fn": "identity"}],
}


def _full_pa_config(**extra):
    return {
        "tuning_model": "full_pa",
        "outlet_mapping_mode": "cap_name",
        "objective_tree_policy": {"use_mean": True, "reference_diameter": "conductance_matched"},
        "tune_space": json.loads(json.dumps(TUNE_SPACE)),
        **extra,
    }


# -------------------------------------------------------------- outlet pressure

def test_precapillary_fraction_pressure():
    assert resolve_outlet_pressure([34.0, 3.0, 16.0], 7.0, "precapillary_fraction") == pytest.approx(
        7.0 + 0.332 * 9.0
    )


def test_diastolic_offset_needs_no_wedge():
    assert resolve_outlet_pressure([44.0, 14.0, 26.0], float("nan"), "diastolic_offset") == pytest.approx(12.0)
    assert resolve_outlet_pressure([44.0, 14.0, 26.0], None, "diastolic_offset",
                                   diastolic_offset_mmhg=3.0) == pytest.approx(11.0)


def test_legacy_policies_keep_nan_for_missing_wedge():
    # callers that only need the pressure targets must keep working
    assert math.isnan(resolve_outlet_pressure([34.0, 3.0, 16.0], float("nan"), "clamp_to_diastolic"))
    assert math.isnan(resolve_outlet_pressure([34.0, 3.0, 16.0], None, "measured"))


def test_precapillary_fraction_requires_wedge():
    with pytest.raises(ValueError, match="requires a measured"):
        resolve_outlet_pressure([34.0, 3.0, 16.0], float("nan"), "precapillary_fraction")


def test_outlet_pressure_at_or_above_mean_is_rejected():
    with pytest.raises(ValueError, match="mean MPA target"):
        resolve_outlet_pressure([20.0, 18.0, 15.0], 7.0, "diastolic_offset", diastolic_offset_mmhg=1.0)


def test_from_csv_without_wedge_column(tmp_path):
    csv = tmp_path / "targets.csv"
    csv.write_text("mpa_pressure,mpa_flow,rpa_split\n29/5/15,31.35,0.903\n")
    targets = ClinicalTargets.from_csv(str(csv), wedge_pressure_policy="diastolic_offset")
    assert targets.wedge_p == pytest.approx(3.0)
    assert math.isnan(targets.measured_wedge_p)
    validate_outlet_pressure(targets)


def test_validate_outlet_pressure_rejects_nan(tmp_path):
    csv = tmp_path / "targets.csv"
    csv.write_text("mpa_pressure,mpa_flow,rpa_split,wedge_pressure\n29/5/15,31.35,0.903,\n")
    targets = ClinicalTargets.from_csv(str(csv))
    with pytest.raises(ValueError, match="non-physical"):
        validate_outlet_pressure(targets)


# -------------------------------------------------------------- objective

class _Targets:
    mpa_p = [34.0, 3.0, 16.0]
    rpa_split = 0.87
    wedge_p = 9.99


def _tuner(objective=None, keep_dia=False):
    tuner = object.__new__(ImpedanceTuner)
    tuner.clinical_targets = _Targets()
    tuner.objective = resolve_tuning_objective(objective)
    tuner.keep_diastolic_target = keep_dia
    tuner._loss_weights = None
    return tuner


def test_likelihood_components_are_sigma_scaled():
    tuner = _tuner({"type": "likelihood"}, keep_dia=True)
    terms = tuner.objective_terms([36.0, 1.0, 16.0], 0.89, {})
    assert terms["components"]["sys"] == pytest.approx(1.0)
    assert terms["components"]["dia"] == pytest.approx(1.0)
    assert terms["components"]["mean"] == pytest.approx(0.0)
    assert terms["components"]["flow"] == pytest.approx(1.0)
    assert terms["unweighted_loss"] == pytest.approx(3.0)


def test_diastolic_term_dropped_below_outlet_unless_kept():
    assert _tuner()._pressure_weights()["dia"] == 0.0
    assert _tuner(keep_dia=True)._pressure_weights()["dia"] == 1.0


def test_likelihood_target_stop_uses_sigma():
    tuner = _tuner({"type": "likelihood", "target_sigma": 1.0}, keep_dia=True)
    inside = {"metrics": {"sys_pressure": 35.5, "dia_pressure": 4.5, "mean_pressure": 16.5, "rpa_split": 0.88}}
    outside = {"metrics": {"sys_pressure": 35.5, "dia_pressure": 5.5, "mean_pressure": 16.5, "rpa_split": 0.88}}
    assert tuner._objective_targets_met(inside, tolerance=0.025) is True
    assert tuner._objective_targets_met(outside, tolerance=0.025) is False


def test_objective_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown keys"):
        resolve_tuning_objective({"type": "likelihood", "sigma": 2.0})


# -------------------------------------------------------------- seeding

def test_seed_round_off_at_bound_is_clipped():
    overshoot = float(np.exp(np.log(4.0e5)))
    assert overshoot > 4.0e5
    assert _clip_to_bounds_within_tolerance(overshoot, 5.0e4, 4.0e5, param_name="k3", source="x") == 4.0e5
    with pytest.raises(ValueError, match="outside configured bounds"):
        _clip_to_bounds_within_tolerance(4.1e5, 5.0e4, 4.0e5, param_name="k3", source="x")


# -------------------------------------------------------------- proximal compliance

def _seed(c_values):
    return {
        "vessels": [
            {
                "vessel_name": f"v{i}",
                "vessel_length": 2.0,
                "geometric_params": {"inlet_area": 1.0, "outlet_area": 0.5},
                "zero_d_element_values": {"C": c, "R_poiseuille": 1.0, "L": 0.0, "stenosis_coefficient": 0.0},
            }
            for i, c in enumerate(c_values)
        ]
    }


def test_proximal_compliance_fills_rigid_vessels_idempotently():
    payload, summary = apply_proximal_compliance(_seed([0.0, 1e-10]), 5.0e4)
    expected = 3.0 * 0.75 * 2.0 / (2.0 * 5.0e4)
    assert [v["zero_d_element_values"]["C"] for v in payload["vessels"]] == pytest.approx([expected, expected])
    assert summary["n_applied"] == 2
    again, summary2 = apply_proximal_compliance(payload, 5.0e4)
    assert again == payload
    assert summary2["n_kept_existing"] == 2


def test_proximal_compliance_keeps_calibrated_compliance_without_geometry():
    seed = _seed([5e-6])
    del seed["vessels"][0]["geometric_params"]
    payload, summary = apply_proximal_compliance(seed, 5.0e4)
    assert payload["vessels"][0]["zero_d_element_values"]["C"] == 5e-6
    assert summary["n_kept_existing"] == 1 and summary["n_applied"] == 0
    assert 5e-6 > RIGID_COMPLIANCE_THRESHOLD


def test_proximal_compliance_rigid_vessel_needs_geometry():
    seed = _seed([0.0])
    del seed["vessels"][0]["geometric_params"]
    with pytest.raises(ValueError, match="geometric_params"):
        apply_proximal_compliance(seed, 5.0e4)


def test_proximal_compliance_takes_learned_connector_length_from_centerline():
    # Raw learnedZeroD seed: branch 1's R/L live on the junction; its
    # connectorEL vessel has zero length, and a split connector keeps 0.1 cm.
    seed = _seed([0.0, 1e-10, 0.0])
    names = ["branch0_seg0", "branch1_seg0_connectorEL", "branch1_seg0_connector0"]
    for index, (vessel, name, length) in enumerate(zip(seed["vessels"], names, [2.0, 0.0, 0.1])):
        vessel["vessel_id"] = index
        vessel["vessel_name"] = name
        vessel["vessel_length"] = length
    seed["vessels"][1]["zero_d_element_values"].update(R_poiseuille=0.0, L=0.0)
    seed["junctions"] = [
        {
            "junction_name": "J0",
            "junction_type": "BloodVesselJunction",
            "inlet_vessels": [0],
            "outlet_vessels": [1, 2],
            "junction_values": {"R_poiseuille": [7.0, 3.0], "L": [2.0, 1.0], "stenosis_coefficient": [0.0, 0.0]},
        }
    ]
    with pytest.raises(ValueError, match="outlet_mapping_centerline"):
        apply_proximal_compliance(seed, 5.0e4)
    payload, summary = apply_proximal_compliance(seed, 5.0e4, {0: 2.0, 1: 1.5})
    c = [v["zero_d_element_values"]["C"] for v in payload["vessels"]]
    assert c == pytest.approx([3.0 * 0.75 * length / (2.0 * 5.0e4) for length in (2.0, 1.4, 0.1)])
    assert summary["n_branch_length_from_centerline"] == 1
    assert summary["seed_volume_ml"] == pytest.approx(0.75 * 3.5)
    assert payload["vessels"][1]["vessel_length"] == 0.0
    # Branch 1's junction R/L now sit on its compliant connector vessel; the
    # split connector (non-zero R) keeps its junction values.
    assert summary["n_junction_values_moved_to_vessel"] == 1
    assert payload["vessels"][1]["zero_d_element_values"]["R_poiseuille"] == 7.0
    assert payload["vessels"][1]["zero_d_element_values"]["L"] == 2.0
    assert payload["junctions"][0]["junction_values"]["R_poiseuille"] == [0.0, 3.0]
    assert payload["junctions"][0]["junction_values"]["L"] == [0.0, 1.0]
    again, _ = apply_proximal_compliance(payload, 5.0e4, {0: 2.0, 1: 1.5})
    assert again == payload


def test_proximal_compliance_clamps_negative_learned_inductance():
    # learnedZeroD can fit a negative junction-outlet L; on a compliant vessel
    # it makes the solver's Newton iteration fail, so it is set to 0.
    seed = _seed([0.0, 1e-10, 5e-6])
    names = ["branch0_seg0", "branch1_seg0_connectorEL", "branch2_seg0"]
    for index, (vessel, name, length) in enumerate(zip(seed["vessels"], names, [2.0, 0.0, 1.0])):
        vessel["vessel_id"] = index
        vessel["vessel_name"] = name
        vessel["vessel_length"] = length
    seed["vessels"][1]["zero_d_element_values"].update(R_poiseuille=0.0, L=0.0)
    seed["vessels"][2]["zero_d_element_values"].update(L=-1.0)
    seed["junctions"] = [
        {
            "junction_name": "J0",
            "junction_type": "BloodVesselJunction",
            "inlet_vessels": [0],
            "outlet_vessels": [1, 2],
            "junction_values": {"R_poiseuille": [7.0, 0.0], "L": [-16.8, 0.0], "stenosis_coefficient": [0.0, 0.0]},
        }
    ]
    payload, summary = apply_proximal_compliance(seed, 5.0e4, {0: 2.0, 1: 1.5})
    values = payload["vessels"][1]["zero_d_element_values"]
    assert values["R_poiseuille"] == 7.0 and values["L"] == 0.0 and values["C"] > 0.0
    assert payload["junctions"][0]["junction_values"]["L"] == [0.0, 0.0]
    assert summary["n_negative_l_clamped"] == 1
    assert summary["negative_l_clamped"] == {"branch1_seg0_connectorEL": -16.8}
    # A vessel whose calibrated compliance is kept is left as it is.
    assert payload["vessels"][2]["zero_d_element_values"]["L"] == -1.0
    again, summary2 = apply_proximal_compliance(payload, 5.0e4, {0: 2.0, 1: 1.5})
    assert again == payload and summary2["n_negative_l_clamped"] == 0


def test_vessel_round_trip_preserves_geometric_params():
    config = _seed([0.0])["vessels"][0] | {"vessel_id": 0, "vessel_name": "branch0_seg0"}
    assert Vessel(config).to_dict()["geometric_params"] == {"inlet_area": 1.0, "outlet_area": 0.5}


# -------------------------------------------------------------- resolver contract

def test_resolver_accepts_new_controls():
    resolved = _resolve_impedance_config(_full_pa_config(
        wedge_pressure_policy="precapillary_fraction",
        keep_diastolic_target=True,
        objective={"type": "likelihood"},
        proximal_compliance={"wall_ehr": 5.0e4},
        tree_max_nodes=1_000_000,
        polish={"maxfev": 50, "initial_simplex_step": 0.05},
    ))
    assert resolved["objective"]["type"] == "likelihood"
    assert resolved["proximal_compliance"] == {"wall_ehr": 5.0e4}
    assert resolved["polish"] == {"maxfev": 50, "initial_simplex_step": 0.05}
    assert resolved["tree_max_nodes"] == 1_000_000
    assert resolved["keep_diastolic_target"] is True


def test_resolver_keeps_historical_shape_when_controls_are_omitted():
    resolved = _resolve_impedance_config(_full_pa_config())
    for key in ("objective", "proximal_compliance", "polish", "tree_max_nodes", "keep_diastolic_target",
                "precapillary_fraction", "diastolic_offset_mmhg"):
        assert key not in resolved


def test_resolver_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown keys"):
        _resolve_impedance_config(_full_pa_config(new_feature=True))
    assert "polish" in SUPPORTED_IMPEDANCE_KEYS


def test_resolver_rejects_unknown_tune_space_names():
    config = _full_pa_config()
    config["tune_space"]["fixed"].append({"name": "comp.lpa.k4", "value": 1.0})
    with pytest.raises(ValueError, match="does not use"):
        _resolve_impedance_config(config)
    validate_tune_space_names(["lpa.xi", "comp.rpa.k1", "lrr"])


def test_polish_requires_full_pa_and_objective_tree_policy():
    with pytest.raises(ValueError, match="objective_tree_policy"):
        resolve_polish({"maxfev": 10}, tuning_model="full_pa", objective_tree_policy=None)
    with pytest.raises(ValueError, match="full_pa"):
        resolve_polish({"maxfev": 10}, tuning_model="rri", objective_tree_policy={"use_mean": True})


def test_proximal_compliance_rejects_convert_to_cm():
    with pytest.raises(ValueError, match="cm-g-s"):
        _resolve_impedance_config(_full_pa_config(proximal_compliance={"wall_ehr": 5.0e4}, convert_to_cm=True))


# -------------------------------------------------------------- gate and diagnostics

def test_iteration_gate_sigma_mode():
    targets = {"mpa_p": [34.0, 3.0, 16.0], "rpa_split": 0.87}
    metrics = {"mpa_sys": 35.5, "mpa_dia": 4.5, "mpa_mean": 15.0, "rpa_split": 0.88}
    relative = evaluate_iteration_gate(metrics=metrics, clinical_targets=targets)
    sigma = evaluate_iteration_gate(
        metrics=metrics, clinical_targets=targets, sigma={"pressure_mmhg": 2.0, "split": 0.02}
    )
    assert relative["close_to_targets"] is False  # dia off by 50%
    assert sigma["close_to_targets"] is True and sigma["gate_mode"] == "sigma"


def test_svpp_bracket(tmp_path):
    inflow = tmp_path / "inflow.csv"
    t = np.linspace(0.0, 1.0, 201)
    q = 10.0 * np.sin(2 * np.pi * t) + 5.0
    inflow.write_text("t,q\n" + "\n".join(f"{a},{b}" for a, b in zip(t, q)))
    bracket = svpp_bracket(inflow, [30.0, 10.0, 18.0])
    assert bracket["lower_ml_per_mmhg"] < bracket["upper_ml_per_mmhg"]
    assert bracket["pulse_pressure_mmhg"] == pytest.approx(20.0)
    assert 0.0 < bracket["regurgitant_fraction"] < 1.0


# -------------------------------------------------------------- polish hand-off

class _FakeTuner:
    def __init__(self, fail=False):
        self.fail = fail
        self.objective_tree_policy = {"use_mean": True}
        self.grid_search_init = True
        self.stopping = None
        self.best_x = np.array([0.5])
        self.calls = []

    def tune(self, nm_iter=1, initial_params_csv=None, x0=None):
        self.calls.append({"nm_iter": nm_iter, "x0": x0, "policy": self.objective_tree_policy})
        if self.fail:
            raise RuntimeError("out of memory")
        _write_csv(self._out / "optimized_params.csv", 36.0)
        (self._out / "pa_config_tuning_snapshot.json").write_text("{\"polished\": true}")


def _write_csv(path, sys_p):
    path.write_text(
        "pa,flow_split,p_mpa,loss\n"
        f"lpa,0.13,[{sys_p} -4.0 15.0],10.0\n"
        f"rpa,0.87,[{sys_p} -4.0 15.0],10.0\n"
    )


@pytest.mark.parametrize("fail", [False, True])
def test_per_cap_polish_hand_off(tmp_path, fail):
    from svzerodtrees.tuning import iteration as it

    _write_csv(tmp_path / "optimized_params.csv", 40.0)
    (tmp_path / "pa_config_tuning_snapshot.json").write_text("{\"shared\": true}")
    log = tmp_path / "opt.log"
    tuner = _FakeTuner(fail=fail)
    tuner._out = tmp_path
    summary, _ = it._run_per_cap_polish(
        tuner=tuner,
        tuning={"polish": {"maxfev": 7, "initial_simplex_step": 0.05}, "stopping": {"maxfev": 200}},
        output_dir=tmp_path,
        opt_log=log,
    )
    assert (tmp_path / it.SHARED_OPTIMIZED_PARAMS_FILENAME).exists()
    assert summary["shared_fit"]["mpa_pressure_mmhg"][0] == pytest.approx(40.0)
    if fail:
        # the shared result is restored, never lost
        assert summary["used"] == "shared" and "out of memory" in summary["error"]
        assert "shared" in (tmp_path / "pa_config_tuning_snapshot.json").read_text()
        assert it._optimized_csv_fit(tmp_path / "optimized_params.csv")["mpa_pressure_mmhg"][0] == 40.0
    else:
        assert summary["used"] == "per_cap"
        assert summary["polished_fit"]["mpa_pressure_mmhg"][0] == pytest.approx(36.0)
    # one polish run on the final (per-cap) trees, started at the shared optimum
    # (the shallow copy shares the call log list with the original tuner)
    assert len(tuner.calls) == 1
    assert tuner.calls[0]["policy"] is None and tuner.calls[0]["nm_iter"] == 1
    assert np.allclose(tuner.calls[0]["x0"], [0.5])
    assert tuner.objective_tree_policy == {"use_mean": True}  # original tuner untouched


def test_tree_metadata_round_trips_max_nodes():
    from svzerodtrees.microvasculature.compliance.olufsen import OlufsenCompliance
    from svzerodtrees.microvasculature.structured_tree.structuredtree import StructuredTree
    from svzerodtrees.io.blocks.simulation_parameters import SimParams

    simparams = SimParams({"number_of_cardiac_cycles": 1, "number_of_time_pts_per_cardiac_cycle": 64})
    time = list(np.linspace(0.0, 1.0, 64))
    tree = StructuredTree(name="t", time=time, simparams=simparams,
                          compliance_model=OlufsenCompliance(2.6e5, -14.0, 1e5))
    tree.build(initial_d=0.3, d_min=0.01, lrr=10.0, xi=3.0, eta_sym=0.9, max_nodes=5000)
    metadata = tree.to_dict()
    assert metadata["max_nodes"] == 5000
    rebuilt = StructuredTree.from_tree_metadata(metadata, time=time, simparams=simparams)
    assert rebuilt.max_nodes == 5000
    assert rebuilt.store.d.size == tree.store.d.size


def test_inflow_consistency_flags_published_flow_mismatch(tmp_path):
    from svzerodtrees.io.inflow_handler import mean_flow_from_path
    from svzerodtrees.tune_bcs.tuning_diagnostics import _inflow_consistency

    inflow = tmp_path / "inflow.csv"
    t = np.linspace(0.0, 1.0, 101)
    q = 20.0 + 10.0 * np.sin(2.0 * np.pi * t)
    inflow.write_text("t,q\n" + "\n".join(f"{a},{b}" for a, b in zip(t, q)))
    reference = float(mean_flow_from_path(str(inflow)))
    same = _inflow_consistency({"inflow_mean_flow": reference * 1.002}, inflow)
    assert same["consistent"] is True
    off = _inflow_consistency({"inflow_mean_flow": reference * 1.5}, inflow)
    assert off["consistent"] is False
    assert off["rel_difference"] == pytest.approx(0.5)
    assert _inflow_consistency({"error": "did not converge"}, inflow) is None


def test_diagnostics_keep_report_when_published_fit_failed(tmp_path):
    from types import SimpleNamespace

    from svzerodtrees.tune_bcs.tuning_diagnostics import build_tuning_diagnostics

    targets = SimpleNamespace(mpa_p=[34.0, 3.0, 16.0], rpa_split=0.87, wedge_p=9.99)
    diagnostics = build_tuning_diagnostics(
        targets=targets,
        tuning={"wedge_pressure_policy": "precapillary_fraction"},
        tree_diagnostics=None,
        proximal_summary=None,
        inflow_path=None,
        published_fit={"error": "RuntimeError: boom"},
    )
    assert diagnostics["published_fit"] == {"error": "RuntimeError: boom"}
    assert diagnostics["outlet_pressure"]["pd_mmhg"] == pytest.approx(9.99)
    assert diagnostics["inflow_consistency"] is None


def test_proximal_summary_reports_compliance_matched_uniform_wall():
    seed = _seed([0.0, 0.0])
    seed["vessels"][1]["geometric_params"] = {"inlet_area": 4.0, "outlet_area": 4.0}
    _, summary = apply_proximal_compliance(seed, 5.0e4)
    areas, length = [0.75, 4.0], 2.0
    radii = [math.sqrt(a / math.pi) for a in areas]
    r_eff = sum(a * length * r for a, r in zip(areas, radii)) / sum(a * length for a in areas)
    assert summary["seed_volume_ml"] == pytest.approx(sum(areas) * length)
    assert summary["volume_weighted_radius_cm"] == pytest.approx(r_eff)
    assert summary["matched_uniform_wall_eh"] == pytest.approx(5.0e4 * r_eff)
    # A uniform wall with that E*h carries the same total compliance.
    uniform = sum(3.0 * a * length * r / (2.0 * summary["matched_uniform_wall_eh"]) for a, r in zip(areas, radii))
    assert uniform * 1333.2 == pytest.approx(summary["total_compliance_ml_per_mmhg"])


def test_matched_wall_elasticity_modulus_divides_matched_eh_by_thickness(tmp_path):
    seed = _seed([0.0, 0.0])
    _, summary = apply_proximal_compliance(seed, 5.0e4)
    path = tmp_path / "tuning_diagnostics.json"
    path.write_text(json.dumps({"proximal_compliance": summary}), encoding="utf-8")
    record = matched_wall_elasticity_modulus(path, 0.2)
    assert record["elasticity_modulus"] == pytest.approx(summary["matched_uniform_wall_eh"] / 0.2)
    assert record["volume_weighted_radius_cm"] == pytest.approx(summary["volume_weighted_radius_cm"])
    assert record["tuning_diagnostics"] == str(path)


def test_matched_wall_elasticity_modulus_requires_proximal_compliance(tmp_path):
    path = tmp_path / "tuning_diagnostics.json"
    path.write_text(json.dumps({"proximal_compliance": None}), encoding="utf-8")
    with pytest.raises(ValueError, match="matched_uniform_wall_eh"):
        matched_wall_elasticity_modulus(path, 0.2)
    with pytest.raises(ValueError, match="shell_thickness"):
        matched_wall_elasticity_modulus(path, 0.0)


# -------------------------------------------------------------- leaf resistance

def _tree(max_nodes=None, **kw):
    from svzerodtrees.microvasculature.compliance.olufsen import OlufsenCompliance
    from svzerodtrees.microvasculature.structured_tree.structuredtree import StructuredTree

    tree = StructuredTree(name="t", time=[0.0, 1.0], simparams=None,
                          compliance_model=OlufsenCompliance(2.6e5, -14.0, 1e5))
    tree.build(initial_d=kw.get("d", 0.2), d_min=0.01, lrr=10.0, xi=kw.get("xi", 2.8),
               eta_sym=kw.get("eta", 0.6), max_nodes=max_nodes)
    return tree


@pytest.mark.parametrize("terminal", [0.0, 1.0e6, 3.0e8])
def test_vectorized_dc_resistance_matches_recursive(terminal):
    tree = _tree()
    tree.terminal_resistance = terminal
    assert tree.dc_input_resistance() == pytest.approx(tree.equivalent_resistance(), rel=1e-12)


@pytest.mark.parametrize("fraction", [0.0, 0.332, 0.668, 0.9])
def test_leaf_resistance_carries_requested_share(fraction):
    tree = _tree(max_nodes=20_000)
    summary = tree.set_leaf_resistance_for_fraction(fraction)
    assert tree.terminal_resistance == summary["terminal_resistance"]
    assert summary["downstream_fraction"] == pytest.approx(fraction, abs=1e-9)
    arterial = tree.dc_input_resistance(0.0)
    assert tree.equivalent_resistance() == pytest.approx(arterial / (1.0 - fraction), rel=1e-9)
    assert (summary["terminal_resistance"] > 0.0) == (fraction > 0.0)


def test_leaf_resistance_survives_metadata_round_trip():
    from svzerodtrees.microvasculature.structured_tree.structuredtree import StructuredTree

    tree = _tree(max_nodes=5000)
    tree.set_leaf_resistance_for_fraction(0.668)
    rebuilt = StructuredTree.from_tree_metadata(tree.to_dict(), time=[0.0, 1.0], simparams=None)
    assert rebuilt.terminal_resistance == pytest.approx(tree.terminal_resistance)
    assert rebuilt.equivalent_resistance() == pytest.approx(tree.equivalent_resistance())


def test_resolver_accepts_leaf_resistance_with_measured_outlet():
    resolved = _resolve_impedance_config(_full_pa_config(
        wedge_pressure_policy="measured", leaf_resistance={"downstream_fraction": 0.668},
    ))
    assert resolved["leaf_resistance"] == {"downstream_fraction": 0.668}
    assert "leaf_resistance" not in _resolve_impedance_config(_full_pa_config())


@pytest.mark.parametrize(
    "change, match",
    [
        ({"wedge_pressure_policy": "precapillary_fraction"}, "requires wedge_pressure_policy 'measured'"),
        ({"leaf_resistance": {"downstream_fraction": 1.0}}, r"in \(0, 1\)"),
        ({"leaf_resistance": {"fraction": 0.5}}, "Unknown keys"),
    ],
)
def test_resolver_rejects_invalid_leaf_resistance(change, match):
    config = _full_pa_config(wedge_pressure_policy="measured", leaf_resistance={"downstream_fraction": 0.668})
    config.update(change)
    with pytest.raises(ValueError, match=match):
        _resolve_impedance_config(config)


def test_leaf_resistance_summary_reports_pvr_partition():
    from types import SimpleNamespace

    from svzerodtrees.tune_bcs.tuning_diagnostics import _leaf_resistance_summary

    targets = SimpleNamespace(wedge_p=7.0)
    trees = {
        "IMP_0": {"terminal_resistance": 3e8, "leaf_resistance": {"downstream_fraction": 0.8}},
        "IMP_1": {"terminal_resistance": 4e8, "leaf_resistance": {"downstream_fraction": 0.8}},
    }
    published = {
        "mpa_pressure_mmhg": [34.0, 3.0, 16.0],
        "outlet_means": {"IMP_0": {"pressure_mmhg": 13.0, "flow": 10.0},
                         "IMP_1": {"pressure_mmhg": 11.5, "flow": 30.0}},
    }
    summary = _leaf_resistance_summary({"leaf_resistance": {"downstream_fraction": 0.8}}, trees, published, targets)
    expected = (10.0 * 0.8 * 6.0 + 30.0 * 0.8 * 4.5) / (40.0 * 9.0)
    assert summary["model_capillary_venous_share_of_pvr"] == pytest.approx(expected)
    assert summary["model_arterial_share_of_pvr"] == pytest.approx(1.0 - expected)
    assert summary["terminal_resistance_min"] == 3e8 and summary["terminal_resistance_max"] == 4e8
    assert _leaf_resistance_summary({}, trees, published, targets) is None


@pytest.mark.parametrize("fraction", [0.0, 0.668])
def test_impedance_dc_includes_leaf_resistance_at_every_leaf(fraction):
    import contextlib
    import io

    from svzerodtrees.io.blocks.simulation_parameters import SimParams
    from svzerodtrees.microvasculature.compliance.olufsen import OlufsenCompliance
    from svzerodtrees.microvasculature.structured_tree.structuredtree import StructuredTree

    simparams = SimParams({"number_of_cardiac_cycles": 1, "number_of_time_pts_per_cardiac_cycle": 64})
    time = list(np.linspace(0.0, 0.8, 64))
    tree = StructuredTree(name="t", time=time, simparams=simparams,
                          compliance_model=OlufsenCompliance(2.6e5, -14.0, 1e5))
    # Asymmetric tree: most leaves end above the deepest generation.
    tree.build(initial_d=0.2, d_min=0.01, lrr=10.0, xi=2.8, eta_sym=0.5)
    gens = np.asarray(tree.store.gen)
    leaf = (np.asarray(tree.store.left) < 0) & (np.asarray(tree.store.right) < 0)
    assert np.count_nonzero(leaf & (gens < gens.max())) > 0
    tree.set_leaf_resistance_for_fraction(fraction)
    with contextlib.redirect_stdout(io.StringIO()):
        z_t, _ = tree.compute_olufsen_impedance(tsteps=63)
    assert float(np.sum(z_t)) == pytest.approx(tree.equivalent_resistance(), rel=1e-5)
