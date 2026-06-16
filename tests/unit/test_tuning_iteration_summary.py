from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import pandas as pd
import pytest

from svzerodtrees.tuning.iteration import summarize_pulmonary_zerod_config


def test_summarize_pulmonary_zerod_config_uses_last_cycle(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "simulation_parameters": {"cardiac_period": 1.0},
                "boundary_conditions": [],
            }
        ),
        encoding="utf-8",
    )

    class FakeHandler:
        def __init__(self, _payload, is_pulmonary: bool = False):
            assert is_pulmonary is True
            self.mpa = SimpleNamespace(name="branch0_seg0")
            self.lpa = SimpleNamespace(name="branch1_seg0")
            self.rpa = SimpleNamespace(name="branch2_seg0")

    result = pd.DataFrame(
        [
            {"name": "branch0_seg0", "time": 0.0, "pressure_in": 1333.2 * 10.0, "flow_in": 5.0},
            {"name": "branch0_seg0", "time": 1.0, "pressure_in": 1333.2 * 20.0, "flow_in": 10.0},
            {"name": "branch0_seg0", "time": 1.5, "pressure_in": 1333.2 * 30.0, "flow_in": 12.0},
            {"name": "branch0_seg0", "time": 2.0, "pressure_in": 1333.2 * 40.0, "flow_in": 14.0},
            {"name": "branch1_seg0", "time": 1.0, "pressure_in": 0.0, "flow_in": 4.0},
            {"name": "branch1_seg0", "time": 1.5, "pressure_in": 0.0, "flow_in": 5.0},
            {"name": "branch1_seg0", "time": 2.0, "pressure_in": 0.0, "flow_in": 6.0},
            {"name": "branch2_seg0", "time": 1.0, "pressure_in": 0.0, "flow_in": 6.0},
            {"name": "branch2_seg0", "time": 1.5, "pressure_in": 0.0, "flow_in": 7.0},
            {"name": "branch2_seg0", "time": 2.0, "pressure_in": 0.0, "flow_in": 8.0},
        ]
    )

    monkeypatch.setattr("svzerodtrees.tuning.iteration.ConfigHandler", FakeHandler)
    monkeypatch.setattr("svzerodtrees.tuning.iteration.simulate_pysvzerod", lambda _payload: result)

    summary = summarize_pulmonary_zerod_config(config_path)

    assert summary["window"] == "last_cycle"
    assert summary["mpa_sys"] == pytest.approx(40.0)
    assert summary["mpa_dia"] == pytest.approx(20.0)
    assert summary["mpa_mean"] == pytest.approx(30.0)
    assert summary["lpa_flow"] == pytest.approx(5.0)
    assert summary["rpa_flow"] == pytest.approx(7.0)
    assert summary["mpa_flow"] == pytest.approx(12.0)
    assert summary["rpa_split"] == pytest.approx(7.0 / 12.0)


def test_summarize_pulmonary_zerod_config_rejects_missing_branch_vessels(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"boundary_conditions": []}), encoding="utf-8")

    class FakeHandler:
        def __init__(self, _payload, is_pulmonary: bool = False):
            assert is_pulmonary is True
            self.mpa = SimpleNamespace(name="branch0_seg0")
            self.lpa = None
            self.rpa = SimpleNamespace(name="branch2_seg0")

    monkeypatch.setattr("svzerodtrees.tuning.iteration.ConfigHandler", FakeHandler)

    with pytest.raises(ValueError, match="could not identify MPA/LPA/RPA vessels"):
        summarize_pulmonary_zerod_config(config_path)
