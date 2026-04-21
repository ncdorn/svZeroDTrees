import pytest

from svzerodtrees import cli


class RecordingWorkflow:
    seen = []

    def __init__(self, cfg):
        self.cfg = cfg

    @classmethod
    def from_config(cls, cfg):
        cls.seen.append(cfg)
        return cls(cfg)

    def run(self):
        return {"status": "ok"}


def test_cli_schema_renders_config_template(monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "argv", ["svzerodtrees", "schema"])

    assert cli.main() == 0

    rendered = capsys.readouterr().out
    assert "workflow: pipeline" in rendered
    assert "paths:" in rendered
    assert "bcs:" in rendered


def test_cli_dispatches_real_config_to_pipeline_workflow(monkeypatch, tmp_path):
    RecordingWorkflow.seen = []

    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        f"""
version: 1
workflow: pipeline
paths:
  root: {tmp_path}
pipeline:
  run_steady: false
  optimize_bcs: false
  run_threed: false
  adapt: false
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(cli, "WORKFLOW_MAP", {"pipeline": RecordingWorkflow})
    monkeypatch.setattr(cli.sys, "argv", ["svzerodtrees", "pipeline", str(cfg_path)])

    assert cli.main() == 0
    assert RecordingWorkflow.seen[0].workflow == "pipeline"
    assert RecordingWorkflow.seen[0].paths.root == str(tmp_path)


def test_cli_dispatches_real_config_to_construct_trees_workflow(monkeypatch, tmp_path):
    RecordingWorkflow.seen = []
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        f"""
version: 1
workflow: construct_trees
paths:
  root: {tmp_path}
  zerod_config: model.json
  clinical_targets: clinical_targets.csv
  mesh_surfaces: mesh-surfaces
bcs:
  type: rcr
  rcr_params: [1.0, 2.0, 3.0, 4.0]
trees:
  d_min: 0.01
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(cli, "WORKFLOW_MAP", {"construct_trees": RecordingWorkflow})
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["svzerodtrees", "construct-trees", str(cfg_path)],
    )

    assert cli.main() == 0
    assert RecordingWorkflow.seen[0].workflow == "construct_trees"


def test_cli_rejects_subcommand_workflow_mismatch(monkeypatch, tmp_path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        f"""
version: 1
workflow: pipeline
paths:
  root: {tmp_path}
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(cli.sys, "argv", ["svzerodtrees", "tune-bcs", str(cfg_path)])

    with pytest.raises(ValueError, match="does not match subcommand"):
        cli.main()


def test_run_from_config_helper_dispatches_correct_workflow(monkeypatch, tmp_path):
    class DummyWorkflow:
        seen = []

        @classmethod
        def from_config(cls, cfg):
            cls.seen.append(cfg)
            return cls()

        def run(self):
            return {"status": "ok"}

    monkeypatch.setattr(cli, "WORKFLOW_MAP", {"pipeline": DummyWorkflow})

    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        f"version: 1\nworkflow: pipeline\npaths:\n  root: {tmp_path}\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(cli.sys, "argv", ["svzerodtrees", "pipeline", str(cfg_path)])
    assert cli.main() == 0
    assert DummyWorkflow.seen[0].workflow == "pipeline"
