import pytest

from svzerodtrees.tune_bcs.clinical_targets import ClinicalTargets


@pytest.fixture
def targets_csv(tmp_path):
    path = tmp_path / "clinical_targets.csv"
    # Diastolic MPA target below the measured wedge, as with pulmonary regurgitation.
    path.write_text(
        "mpa_pressure,mpa_flow,rpa_split,wedge_pressure\n34/3/16,46.5,0.87,7.0\n",
        encoding="utf-8",
    )
    return path


def test_default_clamps_wedge_to_diastolic_target(targets_csv):
    targets = ClinicalTargets.from_csv(str(targets_csv))

    assert targets.wedge_p == 3.0
    assert targets.measured_wedge_p == 7.0
    assert targets.wedge_pressure_policy == "clamp_to_diastolic"


def test_measured_policy_keeps_measured_wedge(targets_csv):
    targets = ClinicalTargets.from_csv(str(targets_csv), wedge_pressure_policy="measured")

    assert targets.wedge_p == 7.0
    assert targets.measured_wedge_p == 7.0


def test_unknown_policy_is_rejected(targets_csv):
    with pytest.raises(ValueError, match="wedge_pressure_policy must be one of"):
        ClinicalTargets.from_csv(str(targets_csv), wedge_pressure_policy="mean")
