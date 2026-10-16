"""Full-PA evaluations run until the MPA pressure is periodic.

A cold-started (``steady_initial: False``) model whose compliance fills over
several cycles is still transient at the seed's 2 cycles.  The hermetic
solver below integrates such a model exactly (an RC Windkessel at the MPA,
the DC limit of an impedance outlet), so the periodic value is known.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from svzerodtrees.io import ConfigHandler
from svzerodtrees.tune_bcs.impedance_tuner import ImpedanceTuner
from svzerodtrees.tune_bcs.periodicity import (
    PeriodicEvaluation,
    cycle_metrics,
    resolve_periodic_evaluation,
)
from svzerodtrees.tune_bcs.tuning_diagnostics import _periodicity_summary
from svzerodtrees.tuning.iteration import (
    _resolve_impedance_config,
    summarize_pulmonary_zerod_config,
)

MMHG = 1333.2
PERIOD = 1.0
N_PTS = 51
R_TOTAL = 1000.0  # dyn s / cm^5
PD = 8.0 * MMHG
RPA_SHARE = 0.55


def _inflow(t):
    phase = np.mod(t, PERIOD)
    return np.where(phase < 0.35, 80.0 * np.sin(np.pi * phase / 0.35), 0.0)


def _tiny_pa_payload(cycles: int = 2, compliance: float = 1.0e-3) -> dict:
    t = np.linspace(0.0, PERIOD, N_PTS)

    def vessel(vessel_id, name, bcs=None):
        payload = {
            "vessel_id": vessel_id,
            "vessel_name": name,
            "vessel_length": 1.0,
            "zero_d_element_type": "BloodVessel",
            "zero_d_element_values": {"R_poiseuille": 1.0, "C": 0.0, "L": 0.0, "stenosis_coefficient": 0.0},
        }
        if bcs:
            payload["boundary_conditions"] = bcs
        return payload

    return {
        "simulation_parameters": {
            # learnedZeroD seeds: 2 cycles from a cold start, all cycles output.
            "number_of_cardiac_cycles": cycles,
            "output_all_cycles": True,
            "number_of_time_pts_per_cardiac_cycle": N_PTS,
            "steady_initial": False,
            "density": 1.06,
            "viscosity": 0.04,
            # Read only by the fake solver below.
            "test_compliance": compliance,
        },
        "boundary_conditions": [
            {"bc_name": "INFLOW", "bc_type": "FLOW", "bc_values": {"Q": _inflow(t).tolist(), "t": t.tolist()}},
            {"bc_name": "LPA_BC", "bc_type": "RESISTANCE", "bc_values": {"R": R_TOTAL / (1 - RPA_SHARE), "Pd": PD}},
            {"bc_name": "RPA_BC", "bc_type": "RESISTANCE", "bc_values": {"R": R_TOTAL / RPA_SHARE, "Pd": PD}},
        ],
        "vessels": [
            vessel(0, "branch0_seg0", {"inlet": "INFLOW"}),
            vessel(1, "branch1_seg0", {"outlet": "LPA_BC"}),
            vessel(2, "branch2_seg0", {"outlet": "RPA_BC"}),
        ],
        "junctions": [
            {"junction_name": "J0", "junction_type": "NORMAL_JUNCTION", "inlet_vessels": [0], "outlet_vessels": [1, 2]}
        ],
    }


def _rc_solver(payload: dict) -> pd.DataFrame:
    """Exact cold-start RC response: C dP/dt = Q(t) - (P - Pd)/R, P(0) = 0."""
    simparams = payload["simulation_parameters"]
    cycles = int(simparams["number_of_cardiac_cycles"])
    n_pts = int(simparams["number_of_time_pts_per_cardiac_cycle"])
    compliance = float(simparams.get("test_compliance", 1.0e-3))
    tau = R_TOTAL * compliance
    dt = PERIOD / (n_pts - 1)
    time = np.arange(cycles * (n_pts - 1) + 1) * dt
    flow = _inflow(time)
    pressure = np.zeros_like(time)
    decay = math.exp(-dt / tau)
    for i in range(1, time.size):
        q_mid = 0.5 * (flow[i - 1] + flow[i])
        p_inf = PD + R_TOTAL * q_mid
        pressure[i] = p_inf + (pressure[i - 1] - p_inf) * decay
    if not simparams.get("output_all_cycles", False):
        keep = time >= time.max() - PERIOD - 1e-12
        time, flow, pressure = time[keep], flow[keep], pressure[keep]
    distal = (pressure - PD) / R_TOTAL
    frames = []
    for name, q in (
        ("branch0_seg0", flow),
        ("branch1_seg0", (1 - RPA_SHARE) * distal),
        ("branch2_seg0", RPA_SHARE * distal),
    ):
        frames.append(
            pd.DataFrame(
                {"name": name, "time": time, "pressure_in": pressure, "pressure_out": pressure,
                 "flow_in": q, "flow_out": q}
            )
        )
    return pd.concat(frames, ignore_index=True)


def _periodic_reference(compliance: float = 1.0e-3) -> list[float]:
    payload = _tiny_pa_payload(cycles=80, compliance=compliance)
    payload["simulation_parameters"]["output_all_cycles"] = True
    result = _rc_solver(payload)
    mpa = result[result.name == "branch0_seg0"]
    last = cycle_metrics(mpa.time, mpa.pressure_in / MMHG, PERIOD)[-1]
    return [last["sys"], last["dia"], last["mean"]]


@pytest.fixture
def fake_solver(monkeypatch):
    calls: list[dict] = []

    def _simulate(payload):
        calls.append(copy.deepcopy(payload["simulation_parameters"]))
        return _rc_solver(payload)

    monkeypatch.setattr("svzerodtrees.tuning.iteration.simulate_pysvzerod", _simulate)
    monkeypatch.setattr("svzerodtrees.io.config_handler.simulate_pysvzerod", _simulate)
    return calls


def _mpa(summary):
    return [summary["mpa_sys"], summary["mpa_dia"], summary["mpa_mean"]]


def test_seed_cycle_count_is_transient_and_default_evaluation_is_periodic(fake_solver):
    payload = _tiny_pa_payload(cycles=2)
    original = copy.deepcopy(payload)
    reference = _periodic_reference()

    seed = summarize_pulmonary_zerod_config(payload, periodic_evaluation={"enabled": False})
    periodic = summarize_pulmonary_zerod_config(payload)

    # Cycle 2 of the cold start is still filling: diastole is far too low.
    assert reference[1] - seed["mpa_dia"] > 2.0
    assert seed["periodicity"]["cycles_used"] == 2
    assert seed["periodicity"]["enabled"] is False
    assert seed["periodicity"]["final_drift_mmhg"] > 1.0
    # The default evaluation reaches the periodic value.
    assert _mpa(periodic) == pytest.approx(reference, abs=0.1)
    info = periodic["periodicity"]
    assert info["converged"] is True
    assert info["final_drift_mmhg"] < 0.1
    assert info["cycles_used"] >= 6
    assert info["seed_number_of_cardiac_cycles"] == 2
    assert periodic["rpa_split"] == pytest.approx(RPA_SHARE, abs=1e-3)
    # The caller's config is never modified.
    assert payload == original
    assert fake_solver[0]["number_of_cardiac_cycles"] == 2
    assert fake_solver[1]["number_of_cardiac_cycles"] == 6
    assert fake_solver[1]["output_all_cycles"] is True


def test_slow_transient_extends_cycles_until_converged_or_capped(fake_solver):
    slow = _tiny_pa_payload(cycles=2, compliance=2.5e-3)
    reference = _periodic_reference(compliance=2.5e-3)

    periodic = summarize_pulmonary_zerod_config(slow)
    info = periodic["periodicity"]
    assert info["attempts"][:2] == [6, 12]
    assert info["converged"] is True
    # A slow decay leaves an offset larger than the drift; it is reported.
    offset = abs(periodic["mpa_mean"] - reference[2])
    assert offset < 0.3
    assert info["estimated_residual_mmhg"] == pytest.approx(offset, rel=0.25)

    capped = summarize_pulmonary_zerod_config(
        slow, periodic_evaluation={"min_cycles": 3, "max_cycles": 5}
    )
    assert capped["periodicity"]["attempts"] == [3, 5]
    assert capped["periodicity"]["cycles_used"] == 5
    assert capped["periodicity"]["converged"] is False
    assert capped["periodicity"]["final_drift_mmhg"] > 0.1


def test_full_pa_objective_overrides_cycles_only_for_the_solve(fake_solver, tmp_path: Path):
    handler = ConfigHandler(_tiny_pa_payload(cycles=2), is_pulmonary=True)
    tuner = object.__new__(ImpedanceTuner)
    tuner.periodic_evaluation = PeriodicEvaluation()
    reference = _periodic_reference()

    result, periodicity = tuner._simulate_full_pa_until_periodic(handler)
    metrics = tuner._compute_full_pa_metrics(handler, result)

    assert metrics["P_mpa"] == pytest.approx(reference, abs=0.1)
    assert periodicity["converged"] is True
    assert fake_solver[-1]["number_of_cardiac_cycles"] == periodicity["attempts"][-1]
    # The model (and so every snapshot written from it) keeps the seed's
    # simulation parameters.
    simparams = handler.simparams.to_dict()
    assert simparams["number_of_cardiac_cycles"] == 2
    assert simparams["output_all_cycles"] is True
    handler.to_json(str(tmp_path / "snapshot.json"))
    written = json.loads((tmp_path / "snapshot.json").read_text())["simulation_parameters"]
    assert written["number_of_cardiac_cycles"] == 2


def test_disabled_full_pa_objective_scores_the_seed_cycles(fake_solver):
    handler = ConfigHandler(_tiny_pa_payload(cycles=2), is_pulmonary=True)
    tuner = object.__new__(ImpedanceTuner)
    tuner.periodic_evaluation = resolve_periodic_evaluation({"enabled": False})

    result, periodicity = tuner._simulate_full_pa_until_periodic(handler)
    metrics = tuner._compute_full_pa_metrics(handler, result)

    assert _periodic_reference()[1] - metrics["P_mpa"][1] > 2.0
    assert periodicity["enabled"] is False
    assert fake_solver[-1]["number_of_cardiac_cycles"] == 2


def test_periodic_evaluation_config_contract():
    assert resolve_periodic_evaluation(None) == PeriodicEvaluation(6, 24, 0.1)
    assert resolve_periodic_evaluation({"enabled": False}) is None
    assert resolve_periodic_evaluation({"min_cycles": 8}) == PeriodicEvaluation(8, 24, 0.1)
    for bad in ({"min_cycles": 1}, {"min_cycles": 8, "max_cycles": 4},
                {"mean_pressure_tol_mmhg": 0.0}, {"cycles": 6}):
        with pytest.raises(ValueError):
            resolve_periodic_evaluation(bad)

    base = {
        "tuning_model": "full_pa",
        "tune_space": {
            "free": [{"name": "lpa.xi", "init": 2.3, "lb": 0.0, "ub": 6.0}],
            "fixed": [{"name": "lrr", "value": 10.0}],
            "tied": [],
        },
    }
    assert "periodic_evaluation" not in _resolve_impedance_config(base)
    resolved = _resolve_impedance_config({**base, "periodic_evaluation": {"enabled": False}})
    assert resolved["periodic_evaluation"] == {"enabled": False}
    resolved = _resolve_impedance_config({**base, "periodic_evaluation": {"max_cycles": 30}})
    assert resolved["periodic_evaluation"] == {
        "min_cycles": 6, "max_cycles": 30, "mean_pressure_tol_mmhg": 0.1
    }


def test_tuning_diagnostics_record_cycles_and_drift():
    optimizer = {"cycles_used": 6, "final_drift_mmhg": 0.02, "converged": True, "settings": {"x": 1}}
    published = {"periodicity": {"cycles_used": 12, "final_drift_mmhg": 0.05, "converged": True}}
    summary = _periodicity_summary({}, optimizer, published)
    assert summary["settings"] == {
        "enabled": True, "min_cycles": 6, "max_cycles": 24, "mean_pressure_tol_mmhg": 0.1
    }
    assert summary["optimizer"] == {"cycles_used": 6, "final_drift_mmhg": 0.02, "converged": True}
    assert summary["published"]["cycles_used"] == 12
    assert _periodicity_summary({"periodic_evaluation": {"enabled": False}}, None, None)["settings"] == {
        "enabled": False
    }


def test_full_pa_loss_scores_the_periodic_cycle(fake_solver, monkeypatch, tmp_path: Path):
    """loss_fn -> _evaluate_model scores periodic pressures; snapshot keeps 2 cycles."""
    from svzerodtrees.tune_bcs.clinical_targets import ClinicalTargets
    from svzerodtrees.tuning.iteration import _build_tune_space_from_config

    for name in ("construct_impedance_trees", "validate_boundary_condition_configs",
                 "validate_impedance_timing_config", "validate_flow_cardiac_output_config"):
        monkeypatch.setattr(f"svzerodtrees.tune_bcs.impedance_tuner.{name}", lambda *a, **k: None)
    monkeypatch.chdir(tmp_path)
    reference = _periodic_reference()
    handler = ConfigHandler(_tiny_pa_payload(cycles=2), is_pulmonary=True)
    tune_space = _build_tune_space_from_config(
        {"free": [{"name": "lpa.xi", "init": 2.3, "lb": 0.0, "ub": 6.0}],
         "fixed": [{"name": "lrr", "value": 10.0}, {"name": "lpa.eta_sym", "value": 0.6},
                   {"name": "rpa.xi", "value": 2.3}, {"name": "rpa.eta_sym", "value": 0.6},
                   {"name": "comp.lpa.k2", "value": -25.0},
                   {"name": "comp.rpa.k2", "value": -25.0}],
         "tied": []}
    )
    targets = ClinicalTargets(mpa_p=reference, rpa_split=RPA_SHARE, wedge_p=8.0)
    tuner = ImpedanceTuner(handler, str(tmp_path), targets, tune_space, compliance_model="olufsen",
                           tuning_model="full_pa", rescale_inflow=False)
    tuner._geom_defaults = {"lpa.default_diameter": 0.3, "rpa.default_diameter": 0.3}
    tuner._full_pa_base_config = handler
    tuner._augmented_mode = True
    x0, _ = tune_space.pack_init_and_bounds()

    tuner.loss_fn(x0, handler)

    metrics = tuner._last_loss_breakdown["metrics"]
    scored = [metrics["sys_pressure"], metrics["dia_pressure"], metrics["mean_pressure"]]
    assert scored == pytest.approx(reference, abs=0.1)
    assert tuner.last_periodicity["converged"] is True
    assert tuner.last_periodicity["cycles_used"] >= 6
    assert tuner._last_loss_breakdown["periodicity"] is tuner.last_periodicity
    snapshot = json.loads((tmp_path / "pa_config_tuning_snapshot.json").read_text())
    assert snapshot["simulation_parameters"]["number_of_cardiac_cycles"] == 2


def test_override_simparams_restores_absent_keys():
    from types import SimpleNamespace

    from svzerodtrees.tune_bcs.periodicity import override_simparams

    simparams = SimpleNamespace(number_of_cardiac_cycles=2)
    with override_simparams(simparams, 6):
        assert (simparams.number_of_cardiac_cycles, simparams.output_all_cycles) == (6, True)
    assert vars(simparams) == {"number_of_cardiac_cycles": 2}


def test_yaml_impedance_config_round_trips_periodic_evaluation():
    from svzerodtrees.config import _parse_impedance_config, impedance_config_to_mapping

    block = {
        "tuning_model": "full_pa",
        "tune_space": {"free": [{"name": "lpa.xi", "init": 2.3, "lb": 0.0, "ub": 6.0}],
                       "fixed": [], "tied": []},
    }
    assert "periodic_evaluation" not in impedance_config_to_mapping(_parse_impedance_config(block))
    parsed = _parse_impedance_config({**block, "periodic_evaluation": {"min_cycles": 8}})
    mapping = impedance_config_to_mapping(parsed)
    assert mapping["periodic_evaluation"] == {"min_cycles": 8, "max_cycles": 24, "mean_pressure_tol_mmhg": 0.1}
    assert _resolve_impedance_config(mapping)["periodic_evaluation"] == mapping["periodic_evaluation"]
