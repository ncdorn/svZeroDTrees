"""Tied stiffness multiplier s (``stiffness_scale``) for full-PA impedance tuning.

proximal Eh/r = s * proximal_compliance.wall_ehr and tree k3 = s * tied_k3_ref;
the proximal C is re-applied on every objective evaluation.
"""

import copy
import json

import numpy as np
import pytest

from svzerodtrees.io.blocks.vessel import Vessel
from svzerodtrees.tune_bcs.impedance_tuner import ImpedanceTuner, _seed_x0_from_csv
from svzerodtrees.tune_bcs.pipeline_options import STIFFNESS_SCALE_PARAM, resolve_proximal_compliance
from svzerodtrees.tune_bcs.tied_stiffness import PROXIMAL_WALL_EHR_PARAM, TiedStiffness
from svzerodtrees.tune_bcs.tuning_diagnostics import apply_proximal_compliance
from svzerodtrees.tuning.iteration import _build_tune_space_from_config, _resolve_impedance_config

WALL_EHR_REF = 5.0e4
K3_REF = 1.0e5

TIED_TUNE_SPACE = {
    "free": [
        {"name": "lpa.xi", "init": 2.46, "lb": 2.1, "ub": 3.0},
        {"name": "rpa.xi", "init": 2.46, "lb": 2.1, "ub": 3.0},
        {"name": "lpa.eta_sym", "init": 0.64, "lb": 0.3, "ub": 1.0},
        {"name": "rpa.eta_sym", "init": 0.64, "lb": 0.3, "ub": 1.0},
        {"name": "stiffness_scale", "init": 7.0, "lb": 0.5, "ub": 20.0,
         "to_native": "positive", "from_native": "log"},
    ],
    "fixed": [
        {"name": "lrr", "value": 10.0},
        {"name": "d_min", "value": 0.01},
        {"name": "comp.lpa.k1", "value": 2.6e5},
        {"name": "comp.lpa.k2", "value": -14.0},
    ],
    "tied": [
        {"name": "comp.rpa.k1", "other": "comp.lpa.k1", "fn": "identity"},
        {"name": "comp.rpa.k2", "other": "comp.lpa.k2", "fn": "identity"},
    ],
}


def _config(tune_space=None, proximal=None, **extra):
    return {
        "tuning_model": "full_pa",
        "compliance_model": "olufsen",
        "outlet_mapping_mode": "cap_name",
        "proximal_compliance": (
            {"wall_ehr": WALL_EHR_REF, "tied_k3_ref": K3_REF} if proximal is None else proximal
        ),
        "tune_space": json.loads(json.dumps(TIED_TUNE_SPACE if tune_space is None else tune_space)),
        **extra,
    }


def _seed(c_values):
    return {
        "vessels": [
            {
                "vessel_id": i,
                "vessel_name": f"branch{i}_seg0",
                "vessel_length": 2.0 + i,
                "geometric_params": {"inlet_area": 1.0, "outlet_area": 0.5 + i},
                "zero_d_element_values": {"C": c, "R_poiseuille": 1.0, "L": 0.0, "stenosis_coefficient": 0.0},
            }
            for i, c in enumerate(c_values)
        ]
    }


def _thin_wall_c(vessel, wall_ehr):
    area = 0.5 * (vessel["geometric_params"]["inlet_area"] + vessel["geometric_params"]["outlet_area"])
    return 3.0 * area * vessel["vessel_length"] / (2.0 * wall_ehr)


# ------------------------------------------------------------- tied mapping

@pytest.mark.parametrize("s", [0.5, 1.0, 6.56, 20.0])
def test_tied_mapping_sets_wall_ehr_and_both_tree_k3(s):
    tied = TiedStiffness(WALL_EHR_REF, K3_REF, seed_payload=_seed([0.0]))
    params = tied.resolve_params({STIFFNESS_SCALE_PARAM: s, "comp.lpa.k1": 2.6e5, "comp.lpa.k2": -14.0})
    assert params[PROXIMAL_WALL_EHR_PARAM] == pytest.approx(s * 5.0e4)
    assert params["comp.lpa.k3"] == pytest.approx(s * 1.0e5)
    assert params["comp.rpa.k3"] == pytest.approx(s * 1.0e5)
    # k1 / k2 are not touched by the tie.
    assert params["comp.lpa.k1"] == 2.6e5 and params["comp.lpa.k2"] == -14.0


def test_tied_proximal_compliance_matches_thin_wall_at_scaled_ehr():
    seed = _seed([0.0, 0.0, 0.0])
    tied = TiedStiffness(WALL_EHR_REF, K3_REF, seed_payload=seed)
    for s in (0.5, 7.0):
        compliance = tied.proximal_compliance(s)
        expected = {v["vessel_name"]: _thin_wall_c(v, s * WALL_EHR_REF) for v in seed["vessels"]}
        assert compliance == pytest.approx(expected)
    # C scales exactly as 1 / s.
    assert tied.proximal_compliance(4.0)["branch1_seg0"] == pytest.approx(tied.proximal_compliance(1.0)["branch1_seg0"] / 4.0)


def test_tied_stiffness_overwrites_compliance_kept_from_earlier_iteration():
    # Calibrated seeds keep the previous iteration's C (R-only calibration);
    # with the tie, proximal C is a function of the current s instead.
    seed = _seed([5.0e-6, 1.0e-10])
    kept, _ = apply_proximal_compliance(seed, WALL_EHR_REF)
    assert kept["vessels"][0]["zero_d_element_values"]["C"] == 5.0e-6
    payload, summary = apply_proximal_compliance(seed, WALL_EHR_REF, overwrite=True)
    assert summary["n_kept_existing"] == 0 and summary["n_applied"] == 2
    assert summary["overwrite_existing"] is True
    assert payload["vessels"][0]["zero_d_element_values"]["C"] == pytest.approx(
        _thin_wall_c(seed["vessels"][0], WALL_EHR_REF)
    )


def test_tied_stiffness_rejects_nonpositive_scale():
    tied = TiedStiffness(WALL_EHR_REF, K3_REF, seed_payload=_seed([0.0]))
    with pytest.raises(ValueError, match="must be > 0"):
        tied.resolve_params({STIFFNESS_SCALE_PARAM: 0.0})
    with pytest.raises(ValueError, match="needs the 'stiffness_scale'"):
        tied.resolve_params({})


# ------------------------------------------------------------- bounds and contract

def test_resolver_keeps_stiffness_scale_log_bounds():
    resolved = _resolve_impedance_config(_config())
    assert resolved["proximal_compliance"] == {"wall_ehr": WALL_EHR_REF, "tied_k3_ref": K3_REF}
    space = _build_tune_space_from_config(resolved["tune_space"])
    x0, bounds = space.pack_init_and_bounds()
    names = [p.name for p in space.free]
    index = names.index(STIFFNESS_SCALE_PARAM)
    assert bounds[index] == pytest.approx((np.log(0.5), np.log(20.0)))
    assert x0[index] == pytest.approx(np.log(7.0))
    # Optimizer space is log(s): both bounds map back to the native range.
    param = space.free[index]
    assert param.to_native(bounds[index][0]) == pytest.approx(0.5)
    assert param.to_native(bounds[index][1]) == pytest.approx(20.0)
    for side in ("lpa", "rpa"):
        assert bounds[names.index(f"{side}.xi")] == pytest.approx((2.1, 3.0))
        assert bounds[names.index(f"{side}.eta_sym")] == pytest.approx((0.3, 1.0))


def test_tied_k3_ref_must_be_positive():
    with pytest.raises(ValueError, match="tied_k3_ref must be > 0"):
        resolve_proximal_compliance({"wall_ehr": 5.0e4, "tied_k3_ref": 0.0})


def test_stiffness_scale_requires_tied_k3_ref():
    with pytest.raises(ValueError, match="requires proximal_compliance.tied_k3_ref"):
        _resolve_impedance_config(_config(proximal={"wall_ehr": WALL_EHR_REF}))


def test_tied_k3_ref_requires_stiffness_scale():
    space = json.loads(json.dumps(TIED_TUNE_SPACE))
    space["free"] = [p for p in space["free"] if p["name"] != STIFFNESS_SCALE_PARAM]
    with pytest.raises(ValueError, match="requires a 'stiffness_scale'"):
        _resolve_impedance_config(_config(tune_space=space))


def test_stiffness_scale_rejects_explicit_k3():
    space = json.loads(json.dumps(TIED_TUNE_SPACE))
    space["fixed"].append({"name": "comp.lpa.k3", "value": 1.0e5})
    space["tied"].append({"name": "comp.rpa.k3", "other": "comp.lpa.k3", "fn": "identity"})
    with pytest.raises(ValueError, match="must not set"):
        _resolve_impedance_config(_config(tune_space=space))


def test_stiffness_scale_rejects_nonpositive_lower_bound_and_constant_compliance():
    space = json.loads(json.dumps(TIED_TUNE_SPACE))
    space["free"][-1].update({"lb": 0.0, "to_native": "identity", "from_native": "identity"})
    with pytest.raises(ValueError, match="must be > 0"):
        _resolve_impedance_config(_config(tune_space=space))
    with pytest.raises(ValueError, match="olufsen"):
        _resolve_impedance_config(_config(compliance_model="constant"))


def test_yaml_schema_accepts_the_same_contract():
    from svzerodtrees.config import _parse_impedance_config

    parsed = _parse_impedance_config(_yaml_impedance())
    assert parsed.proximal_compliance == {"wall_ehr": WALL_EHR_REF, "tied_k3_ref": K3_REF}
    with pytest.raises(ValueError, match="requires proximal_compliance.tied_k3_ref"):
        _parse_impedance_config(_yaml_impedance(proximal={"wall_ehr": WALL_EHR_REF}))


def _yaml_impedance(proximal=None):
    space = json.loads(json.dumps(TIED_TUNE_SPACE))
    for entry in space["free"]:
        entry.setdefault("to_native", "identity")
        entry.setdefault("from_native", "identity")
    data = _config(proximal=proximal)
    data["tune_space"] = space
    return data


# ------------------------------------------------------------- per-evaluation re-application

class _Simparams:
    number_of_cardiac_cycles = 6


class _Model:
    """Seed vessels only; the tuner's tree/solver steps are replaced below."""

    def __init__(self, seed):
        self.vessel_map = {v["vessel_id"]: Vessel(copy.deepcopy(v)) for v in seed["vessels"]}
        self.simparams = _Simparams()


def _tied_tuner(monkeypatch, seed, tmp_path):
    tuner = object.__new__(ImpedanceTuner)
    space = _build_tune_space_from_config(_resolve_impedance_config(_config())["tune_space"])
    reference, _ = apply_proximal_compliance(seed, WALL_EHR_REF, overwrite=True)
    tuner.__dict__.update(
        tune_space=space,
        tuning_model="full_pa",
        compliance_model="olufsen",
        tied_stiffness=TiedStiffness(WALL_EHR_REF, K3_REF, seed_payload=reference),
        _full_pa_base_config=_Model(reference),
        _geom_defaults={"lpa.default_diameter": 1.0, "rpa.default_diameter": 1.0},
        _augmented_mode=False,
        _loss_weights=None,
        _n_successful_evaluations=0,
        _last_evaluation_error=None,
        _opt_csv_path=str(tmp_path / "optimized_params.csv"),
        objective_tree_policy=None,
        use_mean=True,
        diameter_scale=0.0,
        diameter_std_cap=None,
        last_periodicity=None,
    )
    seen = []

    def _fake_construct(model, *_args, **_kwargs):
        lpa, rpa = _args[2], _args[3]
        seen.append({
            "C": {v.name: v.C for v in model.vessel_map.values()},
            "k3": (lpa.compliance_model.k3, rpa.compliance_model.k3),
        })

    monkeypatch.setattr("svzerodtrees.tune_bcs.impedance_tuner.construct_impedance_trees", _fake_construct)
    monkeypatch.setattr(ImpedanceTuner, "_write_and_validate_snapshot", lambda self, model: None)
    monkeypatch.setattr(ImpedanceTuner, "_impedance_bcs_are_finite", staticmethod(lambda model: True))
    monkeypatch.setattr(ImpedanceTuner, "_simulate_full_pa_until_periodic", lambda self, model: (None, None))
    monkeypatch.setattr(
        ImpedanceTuner,
        "_compute_full_pa_metrics",
        lambda self, model, result: {"P_mpa": [67.0, 9.0, 32.0], "rpa_split": 0.446},
    )
    monkeypatch.setattr(
        ImpedanceTuner,
        "objective_terms",
        lambda self, p, split, params: {"weighted_loss": 0.0, "unweighted_loss": 0.0, "components": {}},
    )
    for name in ("clinical_targets", "mesh_surfaces_path", "d_min", "convert_to_cm", "is_pulmonary",
                 "n_procs", "specify_diameter", "allow_ordered_outlet_mapping", "resolved_mapping",
                 "tree_max_nodes"):
        tuner.__dict__.setdefault(name, None)
    tuner.clinical_targets = type("T", (), {"wedge_p": 7.0})()
    return tuner, seen


def _x_at(tuner, s):
    x0, _ = tuner.tune_space.pack_init_and_bounds()
    index = [p.name for p in tuner.tune_space.free].index(STIFFNESS_SCALE_PARAM)
    x0[index] = np.log(s)
    return x0


def test_proximal_compliance_is_recomputed_on_every_evaluation(monkeypatch, tmp_path):
    seed = _seed([0.0, 5.0e-6, 0.0])  # one vessel carries C from an earlier iteration
    tuner, seen = _tied_tuner(monkeypatch, seed, tmp_path)
    base_before = {v.name: v.C for v in tuner._full_pa_base_config.vessel_map.values()}

    for s in (0.5, 6.56, 20.0, 6.56):
        tuner.loss_fn(_x_at(tuner, s), tuner._full_pa_base_config)

    assert len(seen) == 4
    for evaluation, s in zip(seen, (0.5, 6.56, 20.0, 6.56)):
        expected = {v["vessel_name"]: _thin_wall_c(v, s * WALL_EHR_REF) for v in seed["vessels"]}
        assert evaluation["C"] == pytest.approx(expected)
        assert evaluation["k3"] == pytest.approx((s * K3_REF, s * K3_REF))
    # Repeating an s gives the same C; the shared base model is never mutated.
    assert seen[1]["C"] == seen[3]["C"]
    assert {v.name: v.C for v in tuner._full_pa_base_config.vessel_map.values()} == base_before


def test_final_csv_records_stiffness_scale_and_seeds_next_iteration(monkeypatch, tmp_path):
    tuner, _ = _tied_tuner(monkeypatch, _seed([0.0]), tmp_path)
    tuner.loss_fn(_x_at(tuner, 6.56), tuner._full_pa_base_config, finalize=True)
    import pandas as pd

    frame = pd.read_csv(tmp_path / "optimized_params.csv")
    assert frame[STIFFNESS_SCALE_PARAM].tolist() == pytest.approx([6.56, 6.56])
    assert frame["proximal_wall_ehr"].tolist() == pytest.approx([6.56 * WALL_EHR_REF] * 2)
    assert frame["k3"].tolist() == pytest.approx([6.56 * K3_REF] * 2)

    x0, _ = tuner.tune_space.pack_init_and_bounds()
    seeded = _seed_x0_from_csv(str(tmp_path / "optimized_params.csv"), tuner.tune_space, x0, lambda _m: None)
    index = [p.name for p in tuner.tune_space.free].index(STIFFNESS_SCALE_PARAM)
    assert np.exp(seeded[index]) == pytest.approx(6.56)


def test_tuner_rejects_tied_stiffness_outside_full_pa_olufsen():
    tied = TiedStiffness(WALL_EHR_REF, K3_REF, seed_payload=_seed([0.0]))
    space = _build_tune_space_from_config(_resolve_impedance_config(_config())["tune_space"])
    with pytest.raises(ValueError, match="full_pa"):
        ImpedanceTuner(None, None, None, space, compliance_model="olufsen", tuning_model="rri",
                       tied_stiffness=tied)
    with pytest.raises(ValueError, match="olufsen"):
        ImpedanceTuner(None, None, None, space, compliance_model="constant", tuning_model="full_pa",
                       tied_stiffness=tied)
