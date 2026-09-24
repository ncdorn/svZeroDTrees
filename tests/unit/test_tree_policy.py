import pytest

from svzerodtrees.tune_bcs.tree_policy import resolve_objective_tree_policy

FINAL = {"use_mean": False, "diameter_scale": 1.0, "diameter_std_cap": 2.0}


def _resolve(raw, **overrides):
    options = {"tuning_model": "full_pa", **FINAL, **overrides}
    return resolve_objective_tree_policy(raw, **options)


def test_absent_policy_resolves_to_none():
    assert _resolve(None) is None


def test_omitted_fields_inherit_final_policy():
    assert _resolve({"use_mean": True, "reference_diameter": "conductance_matched"}) == {
        "use_mean": True,
        "diameter_scale": 1.0,
        "diameter_std_cap": 2.0,
        "reference_diameter": "conductance_matched",
    }


def test_explicit_null_std_cap_disables_cap():
    policy = _resolve({"use_mean": False, "diameter_scale": 0.5, "diameter_std_cap": None})

    assert policy["diameter_std_cap"] is None
    assert policy["diameter_scale"] == pytest.approx(0.5)
    assert policy["reference_diameter"] == "arithmetic_mean"


@pytest.mark.parametrize(
    ("raw", "overrides", "message"),
    [
        ({"use_mean": True}, {"tuning_model": "rri"}, "only for tuning_model='full_pa'"),
        ({"unknown": 1}, {}, "unknown keys"),
        ({"reference_diameter": "median"}, {}, "must be one of"),
        ({"diameter_scale": -1.0}, {}, "diameter_scale must be finite"),
        (
            {"use_mean": False, "reference_diameter": "conductance_matched"},
            {},
            "requires .*use_mean=true",
        ),
        (
            {"use_mean": True, "reference_diameter": "conductance_matched"},
            {"use_mean": True},
            "final policy to build per-outlet trees",
        ),
        (
            {"use_mean": True, "reference_diameter": "conductance_matched"},
            {"free_param_names": ["lrr", "lpa.diameter"]},
            "free \\['lpa.diameter'\\]",
        ),
    ],
)
def test_invalid_policy_is_rejected(raw, overrides, message):
    with pytest.raises(ValueError, match=message):
        _resolve(raw, **overrides)
