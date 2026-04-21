from __future__ import annotations

from types import SimpleNamespace

import pytest

from svzerodtrees.api import (
    AdaptationWorkflow,
    ConstructTreesWorkflow,
    PipelineWorkflow,
    TuneBCsWorkflow,
    run_from_config_file,
)


def test_tune_bcs_workflow_requires_bcs_section():
    cfg = SimpleNamespace(
        paths=SimpleNamespace(
            zerod_config="model.json",
            clinical_targets="targets.csv",
            mesh_surfaces="mesh-surfaces",
        ),
        bcs=None,
        threed=None,
    )

    with pytest.raises(ValueError, match="bcs section is required"):
        TuneBCsWorkflow.from_config(cfg).run()


def test_construct_trees_workflow_requires_tree_section():
    cfg = SimpleNamespace(
        paths=SimpleNamespace(
            zerod_config="model.json",
            clinical_targets="targets.csv",
            mesh_surfaces="mesh-surfaces",
        ),
        bcs=SimpleNamespace(type="impedance"),
        trees=None,
        threed=None,
    )

    with pytest.raises(ValueError, match="trees section is required"):
        ConstructTreesWorkflow.from_config(cfg).run()


def test_adaptation_workflow_requires_simulation_directories():
    cfg = SimpleNamespace(
        paths=SimpleNamespace(
            preop_dir=None,
            postop_dir="postop",
            adapted_dir="adapted",
            clinical_targets="targets.csv",
            zerod_config="model.json",
        ),
        bcs=None,
        adaptation=None,
        threed=None,
    )

    with pytest.raises(ValueError, match="preop_dir"):
        AdaptationWorkflow.from_config(cfg).run()


def test_run_from_config_file_dispatches_pipeline(monkeypatch, tmp_path):
    calls = {}

    class DummyWorkflow:
        @classmethod
        def from_config(cls, cfg):
            calls["workflow"] = cfg.workflow
            return cls()

        def run(self):
            return {"status": "ok", "root": "case"}

    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        f"version: 1\nworkflow: pipeline\npaths:\n  root: {tmp_path}\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("svzerodtrees.api.WORKFLOW_MAP", {"pipeline": DummyWorkflow})

    assert run_from_config_file(str(cfg_path)) == {"status": "ok", "root": "case"}
    assert calls["workflow"] == "pipeline"


def test_pipeline_workflow_maps_defaults_when_sections_omitted(monkeypatch, tmp_path):
    calls = {}

    class DummySimulation:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def run_pipeline(self, **kwargs):
            calls["run"] = kwargs

    cfg = SimpleNamespace(
        paths=SimpleNamespace(
            root=str(tmp_path),
            clinical_targets=None,
            preop_dir=None,
            postop_dir=None,
            adapted_dir=None,
            zerod_config=None,
            inflow=None,
        ),
        bcs=None,
        adaptation=None,
        pipeline=None,
        threed=None,
    )

    monkeypatch.setattr("svzerodtrees.api.Simulation", DummySimulation)

    result = PipelineWorkflow.from_config(cfg).run()

    assert result == {"status": "ok", "root": str(tmp_path)}
    assert calls["init"]["preop_dir"] == "preop"
    assert calls["init"]["zerod_config"] == "zerod_config.json"
    assert calls["run"] == {
        "run_steady": True,
        "optimize_bcs": True,
        "run_threed": True,
        "adapt": True,
    }
