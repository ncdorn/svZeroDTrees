from types import SimpleNamespace

import numpy as np
import pytest

from svzerodtrees.tune_bcs.impedance_tuner import ImpedanceTuner
from svzerodtrees.tune_bcs.nm_stopping import (
    STOP_MAXFEV,
    STOP_STALLED,
    STOP_TARGET_MET,
    BoundsScaler,
    NelderMeadStopping,
    initial_simplex,
    resolve_nelder_mead_stopping,
    run_nelder_mead,
)
from svzerodtrees.tune_bcs.tune_space import FreeParam, TuneSpace
from svzerodtrees.tuning.iteration import _resolve_impedance_config


# ---- policy resolution ---- #

def test_resolve_stopping_absent_or_disabled_means_legacy():
    assert resolve_nelder_mead_stopping(None) is None
    assert resolve_nelder_mead_stopping({"enabled": False, "maxfev": 10}) is None


def test_resolve_stopping_fills_defaults_and_keeps_explicit_nulls():
    stopping = resolve_nelder_mead_stopping({"maxfev": 50, "target_tolerance": None})

    assert stopping.maxfev == 50
    assert stopping.target_tolerance is None
    assert stopping.stall_rel_improvement == NelderMeadStopping.stall_rel_improvement
    assert stopping.resolved_stall_window(6) == 30


@pytest.mark.parametrize(
    "payload, match",
    [
        ({"maxfev": 0}, "maxfev must be > 0"),
        ({"xatol": None}, "xatol cannot be null"),
        ({"target_tolerance": -0.1}, "target_tolerance must be > 0"),
        ({"stall_window": 0}, "stall_window must be > 0"),
        ({"initial_simplex_step": 0.9}, "initial_simplex_step must be <= 0.5"),
        ({"tolerance": 0.1}, "unknown keys"),
    ],
)
def test_resolve_stopping_rejects_invalid_values(payload, match):
    with pytest.raises(ValueError, match=match):
        resolve_nelder_mead_stopping(payload)


# ---- scaling and simplex ---- #

def test_initial_simplex_spans_range_fraction_and_steps_inward_at_bounds():
    # A zero init on a [0, 1e6] range must still get a meaningful step
    # (SciPy's default would step it by 2.5e-4).
    scaler = BoundsScaler([(0.0, 1e6), (-100.0, -1.0)])
    u0 = scaler.to_unit([0.0, -1.0])

    sim = scaler.from_unit(initial_simplex(u0, scaler, 0.1))

    assert sim[1] == pytest.approx([1e5, -1.0])
    assert sim[2] == pytest.approx([0.0, -10.9])


def test_bounds_scaler_leaves_infinite_axes_unscaled():
    scaler = BoundsScaler([(0.0, np.inf), (0.0, 2.0)])

    assert scaler.to_unit([3.0, 1.0]) == pytest.approx([3.0, 0.5])
    assert scaler.unit_bounds == [(0.0, np.inf), (0.0, 1.0)]


# ---- run_nelder_mead ---- #

def _quadratic(target):
    def _evaluate(x):
        loss = float(np.sum((np.asarray(x) - target) ** 2))
        return loss, {"metrics": {"x": np.asarray(x).copy()}, "unweighted_loss": loss}
    return _evaluate


def test_run_stops_as_soon_as_target_is_met():
    target = np.array([2.0, 3.0])
    stopping = NelderMeadStopping(target_tolerance=0.05, maxfev=500, stall_rel_improvement=None)

    def _met(breakdown, tol):
        return bool(np.all(np.abs(breakdown["metrics"]["x"] - target) <= tol * np.abs(target)))

    outcome = run_nelder_mead(_quadratic(target), [1.0, 1.0], [(0.0, 5.0), (0.0, 5.0)], stopping, _met)

    assert outcome.reason == STOP_TARGET_MET
    assert np.all(np.abs(outcome.x - target) <= 0.05 * target)
    assert outcome.n_evaluations < 100


def test_run_stops_when_best_loss_stalls():
    # A flat plateau: no evaluation improves on the first.
    stopping = NelderMeadStopping(
        target_tolerance=None, stall_window=10, stall_rel_improvement=0.01, maxfev=500,
        xatol=1e-12, fatol=1e-12,
    )

    def _flat(x):
        return 1.0, {"unweighted_loss": 1.0}

    outcome = run_nelder_mead(_flat, [1.0], [(0.0, 5.0)], stopping, lambda *_: False)

    assert outcome.reason == STOP_STALLED
    assert outcome.n_evaluations == 10 + 1 + 1


def test_run_respects_maxfev_and_ignores_failed_evaluations():
    stopping = NelderMeadStopping(
        target_tolerance=None, stall_rel_improvement=None, maxfev=15, xatol=1e-12, fatol=1e-12,
    )
    good = _quadratic(np.array([2.0, 3.0]))

    failed = []

    def _sometimes_fails(x):
        # The first simplex step along x[0] (4.5 -> 5.0) lands here.
        if x[0] > 4.8:
            failed.append(x)
            return 1e9, None
        return good(x)

    outcome = run_nelder_mead(_sometimes_fails, [4.5, 1.0], [(0.0, 5.0), (0.0, 5.0)], stopping, lambda *_: False)

    assert outcome.reason == STOP_MAXFEV
    assert failed
    assert outcome.x[0] <= 4.8
    assert outcome.loss < 1e9


# ---- ImpedanceTuner integration ---- #

TARGETS = SimpleNamespace(mpa_p=[34.0, 3.0, 16.0], rpa_split=0.87, wedge_p=7.0)


def _tuner(tmp_path, stopping, rpa_split=0.87):
    tuner = ImpedanceTuner(
        config_handler=SimpleNamespace(),
        mesh_surfaces_path=str(tmp_path / "mesh-surfaces"),
        clinical_targets=TARGETS,
        tune_space=TuneSpace(
            free=[
                FreeParam("lpa.xi", init=1.0, lb=0.0, ub=6.0),
                FreeParam("rpa.xi", init=1.0, lb=0.0, ub=6.0),
            ],
            fixed=[],
            tied=[],
        ),
        compliance_model="constant",
        grid_search_init=False,
        rescale_inflow=False,
        log_file=str(tmp_path / "tuning.log"),
        solver="Nelder-Mead",
        stopping=stopping,
    )
    tuner._prepare_geometry_defaults = lambda: None
    tuner._make_tuning_model = lambda: SimpleNamespace(
        bcs={"INFLOW": SimpleNamespace(Q=[6.0, 6.0])}, plot_mpa=lambda: None
    )
    tree = SimpleNamespace(to_csv_row=lambda **kwargs: {"pa": "x", "loss": kwargs["loss"]})
    calls = []

    def _evaluate_model(x, provided_model=None):
        calls.append(np.array(x))
        # sys/mean hit their targets at xi = (3, 2); diastolic is unreachable.
        metrics = {
            "P_mpa": [34.0 + 4.0 * (x[0] - 3.0), 12.0, 16.0 + 3.0 * (x[1] - 2.0)],
            "rpa_split": rpa_split,
        }
        params = {"lpa.xi": float(x[0]), "rpa.xi": float(x[1])}
        return None, metrics, tree, tree, params

    tuner._evaluate_model = _evaluate_model
    return tuner, calls


def test_target_check_ignores_diastolic_below_wedge(tmp_path):
    tuner, _ = _tuner(tmp_path, stopping=None)
    breakdown = {"metrics": {"sys_pressure": 34.5, "dia_pressure": 12.0, "mean_pressure": 16.2, "rpa_split": 0.88}}

    assert tuner._objective_targets_met(breakdown, 0.025)
    breakdown["metrics"]["mean_pressure"] = 17.0
    assert not tuner._objective_targets_met(breakdown, 0.025)


def test_tune_with_stopping_ends_after_target_met_without_restarts(tmp_path):
    tuner, calls = _tuner(tmp_path, stopping={"target_tolerance": 0.025, "maxfev": 200})

    result = tuner.tune(nm_iter=5)

    log = (tmp_path / "tuning.log").read_text(encoding="utf-8")
    assert "stop_reason=target_met" in log
    assert "Nelder-Mead run 2/5" not in log
    assert result.message == STOP_TARGET_MET
    sys_p = 34.0 + 4.0 * (result.x[0] - 3.0)
    mean_p = 16.0 + 3.0 * (result.x[1] - 2.0)
    assert abs(sys_p - 34.0) <= 0.025 * 34.0
    assert abs(mean_p - 16.0) <= 0.025 * 16.0
    assert len(calls) < 200


def test_tune_stops_restarting_when_a_run_barely_improves(tmp_path):
    tuner, _ = _tuner(
        tmp_path,
        stopping={
            "target_tolerance": None,
            "stall_rel_improvement": None,
            "maxfev": 60,
            "restart_min_rel_improvement": 0.05,
        },
        # A split the parameters cannot move puts a floor under the loss, so
        # a restart from the first optimum cannot improve it meaningfully.
        rpa_split=0.80,
    )

    tuner.tune(nm_iter=5)

    log = (tmp_path / "tuning.log").read_text(encoding="utf-8")
    assert "Nelder-Mead run 2/5" in log
    assert "Stopping restarts: run improved" in log
    assert "Nelder-Mead run 5/5" not in log


def test_tuner_rejects_stopping_for_other_solvers(tmp_path):
    with pytest.raises(ValueError, match="solver='Nelder-Mead'"):
        ImpedanceTuner(
            config_handler=SimpleNamespace(),
            mesh_surfaces_path=str(tmp_path),
            clinical_targets=TARGETS,
            tune_space=TuneSpace(free=[], fixed=[], tied=[]),
            compliance_model="constant",
            solver="Powell",
            stopping={},
        )


# ---- iteration config resolution ---- #

_TUNE_SPACE = {
    "free": [{"name": "lpa.xi", "init": 2.3, "lb": 0.0, "ub": 6.0}],
    "fixed": [],
    "tied": [],
}


def test_resolved_config_omits_stopping_when_absent():
    resolved = _resolve_impedance_config({"tune_space": _TUNE_SPACE})

    assert "stopping" not in resolved


def test_resolved_config_normalizes_stopping():
    resolved = _resolve_impedance_config({"tune_space": _TUNE_SPACE, "stopping": {"maxfev": 120}})

    assert resolved["stopping"]["maxfev"] == 120
    assert resolved["stopping"]["target_tolerance"] == NelderMeadStopping.target_tolerance


def test_resolved_config_rejects_stopping_for_other_solvers():
    with pytest.raises(ValueError, match="requires solver='Nelder-Mead'"):
        _resolve_impedance_config({"tune_space": _TUNE_SPACE, "solver": "Powell", "stopping": {}})
