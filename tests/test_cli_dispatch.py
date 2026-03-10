import types
from svzerodtrees import cli


def test_cli_dispatches_correct_workflow(monkeypatch, tmp_path):
    class DummyWorkflow:
        def __init__(self):
            self.called = False

        @classmethod
        def from_config(cls, cfg):
            return cls()

        def run(self):
            self.called = True

    dummy_cfg = types.SimpleNamespace(workflow="pipeline")

    monkeypatch.setattr(cli, "load_config", lambda path: dummy_cfg)
    monkeypatch.setattr(cli, "WORKFLOW_MAP", {"pipeline": DummyWorkflow})

    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text("version: 1\nworkflow: pipeline\npaths: {root: .}\n")

    monkeypatch.setattr(cli.sys, "argv", ["svzerodtrees", "pipeline", str(cfg_path)])
    assert cli.main() == 0
