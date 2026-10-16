"""Real-solver regression: a small impedance model is transient at 2 cycles.

An MPA -> LPA/RPA model with structured-tree IMPEDANCE outlets and vessel
compliance, cold-started like a learnedZeroD seed.  At the seed's 2 cycles
its diastolic MPA pressure is ~8 mmHg below the periodic value; the periodic
evaluation used by full-PA tuning, the published fit, and
``summarize_pulmonary_zerod_config`` must report the periodic value.

Opt-in: ``SVZERODTREES_RUN_REAL_SOLVER=1`` and a pysvzerod build with IMPEDANCE
support (e.g. ``PYTHONPATH=src:../svZeroDSolver``).
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.external

N_PTS = 51
PERIOD = 1.0
MMHG = 1333.2


def _impedance_kernel(time, diameter):
    from svzerodtrees.io.blocks.simulation_parameters import SimParams
    from svzerodtrees.microvasculature import compliance as comp_mod
    from svzerodtrees.microvasculature.structured_tree.structuredtree import StructuredTree

    simparams = SimParams({"number_of_time_pts_per_cardiac_cycle": N_PTS, "density": 1.06, "viscosity": 0.04})
    tree = StructuredTree(
        name="T", time=list(time), simparams=simparams,
        compliance_model=comp_mod.OlufsenCompliance(k1=19992500.0, k2=-25.0, k3=0.0),
    )
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        tree.build(initial_d=diameter, d_min=0.05, lrr=10.0, xi=2.7, eta_sym=0.6)
        tree.compute_olufsen_impedance(tsteps=N_PTS - 1)
    return np.asarray(tree.Z_t, dtype=float).tolist()


def _impedance_model(cycles: int) -> dict:
    time = np.linspace(0.0, PERIOD, N_PTS)
    inflow = np.where(time < 0.35, 80.0 * np.sin(np.pi * time / 0.35), 0.0)
    compliance = 1.0e-3

    def vessel(vessel_id, name, resistance, c, bcs):
        return {
            "vessel_id": vessel_id, "vessel_name": name, "vessel_length": 1.0,
            "zero_d_element_type": "BloodVessel", "boundary_conditions": bcs,
            "zero_d_element_values": {"R_poiseuille": resistance, "C": c, "L": 0.0, "stenosis_coefficient": 0.0},
        }

    return {
        "simulation_parameters": {
            "number_of_cardiac_cycles": cycles, "number_of_time_pts_per_cardiac_cycle": N_PTS,
            "output_all_cycles": True, "steady_initial": False, "density": 1.06, "viscosity": 0.04,
        },
        "boundary_conditions": [
            {"bc_name": "INFLOW", "bc_type": "FLOW", "bc_values": {"Q": inflow.tolist(), "t": time.tolist()}},
            {"bc_name": "LPA_BC", "bc_type": "IMPEDANCE", "bc_values": {"z": _impedance_kernel(time, 0.5), "Pd": 8.0 * MMHG}},
            {"bc_name": "RPA_BC", "bc_type": "IMPEDANCE", "bc_values": {"z": _impedance_kernel(time, 0.4), "Pd": 8.0 * MMHG}},
        ],
        "vessels": [
            vessel(0, "branch0_seg0", 1.0, compliance, {"inlet": "INFLOW"}),
            vessel(1, "branch1_seg0", 2.0, compliance / 2, {"outlet": "LPA_BC"}),
            vessel(2, "branch2_seg0", 1.5, compliance / 2, {"outlet": "RPA_BC"}),
        ],
        "junctions": [
            {"junction_name": "J0", "junction_type": "NORMAL_JUNCTION", "inlet_vessels": [0], "outlet_vessels": [1, 2]}
        ],
    }


@pytest.fixture(scope="module")
def periodic_reference():
    if os.environ.get("SVZERODTREES_RUN_REAL_SOLVER") != "1":
        pytest.skip("opt-in actual-solver regression; set SVZERODTREES_RUN_REAL_SOLVER=1 to run")
    if importlib.util.find_spec("pysvzerod") is None:
        pytest.skip("optional pysvzerod is absent")
    from svzerodtrees._pysvzerod import simulate_pysvzerod
    from svzerodtrees.tune_bcs.periodicity import cycle_metrics

    try:
        result = simulate_pysvzerod(_impedance_model(cycles=40))
    except RuntimeError as exc:
        if "Invalid block type" in str(exc):
            pytest.skip(f"pysvzerod build lacks IMPEDANCE support: {exc}")
        raise
    mpa = result[result.name == "branch0_seg0"]
    last = cycle_metrics(mpa.time, np.asarray(mpa.pressure_in) / MMHG, PERIOD)[-1]
    return [last["sys"], last["dia"], last["mean"]]


def test_two_cycles_are_transient_and_summary_reports_periodic(periodic_reference):
    from svzerodtrees.tuning.iteration import summarize_pulmonary_zerod_config

    seed = summarize_pulmonary_zerod_config(_impedance_model(2), periodic_evaluation={"enabled": False})
    periodic = summarize_pulmonary_zerod_config(_impedance_model(2))

    assert periodic_reference[1] - seed["mpa_dia"] > 5.0
    assert periodic["periodicity"]["converged"] is True
    assert periodic["mpa_dia"] == pytest.approx(periodic_reference[1], abs=0.2)
    assert periodic["mpa_mean"] == pytest.approx(periodic_reference[2], abs=0.2)


def test_published_fit_and_objective_report_periodic(periodic_reference, tmp_path: Path):
    from svzerodtrees.io import ConfigHandler
    from svzerodtrees.tune_bcs.impedance_tuner import ImpedanceTuner
    from svzerodtrees.tune_bcs.periodicity import PeriodicEvaluation
    from svzerodtrees.tuning.iteration import _score_published_full_pa_model

    path = tmp_path / "svzerod_3d_coupling_tuned.json"
    path.write_text(json.dumps(_impedance_model(2)), encoding="utf-8")
    before = path.read_bytes()

    published = _score_published_full_pa_model(path)
    assert published["mpa_pressure_mmhg"] == pytest.approx(periodic_reference, abs=0.2)
    assert published["periodicity"]["cycles_used"] >= 6
    assert path.read_bytes() == before

    handler = ConfigHandler.from_json(str(path), is_pulmonary=True)
    tuner = object.__new__(ImpedanceTuner)
    tuner.periodic_evaluation = PeriodicEvaluation()
    result, periodicity = tuner._simulate_full_pa_until_periodic(handler)
    metrics = tuner._compute_full_pa_metrics(handler, result)
    assert metrics["P_mpa"] == pytest.approx(periodic_reference, abs=0.2)
    assert handler.simparams.number_of_cardiac_cycles == 2
