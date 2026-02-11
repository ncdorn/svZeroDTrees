from types import SimpleNamespace
from svzerodtrees.api import PipelineWorkflow


def test_pipeline_mapping(monkeypatch, tmp_path):
    called = {}

    class DummySimulation:
        def __init__(self, **kwargs):
            called["init"] = kwargs

        def run_pipeline(self, **kwargs):
            called["run"] = kwargs

    monkeypatch.setattr("svzerodtrees.api.Simulation", DummySimulation)

    cfg = SimpleNamespace(
        workflow="pipeline",
        paths=SimpleNamespace(
            root=str(tmp_path),
            zerod_config=str(tmp_path / "zerod_config.json"),
            clinical_targets=str(tmp_path / "clinical_targets.csv"),
            mesh_surfaces=str(tmp_path / "mesh-surfaces"),
            preop_dir=str(tmp_path / "preop"),
            postop_dir=str(tmp_path / "postop"),
            adapted_dir=str(tmp_path / "adapted"),
            inflow=None,
        ),
        bcs=SimpleNamespace(type="impedance", compliance_model="constant", tune_space=None),
        adaptation=SimpleNamespace(method="cwss", location="uniform", iterations=10),
        pipeline=SimpleNamespace(run_steady=True, optimize_bcs=False, run_threed=False, adapt=False),
        threed=SimpleNamespace(mesh_scale_factor=2.0, convert_to_cm=True),
    )

    PipelineWorkflow.from_config(cfg).run()

    assert called["init"]["bc_type"] == "impedance"
    assert called["init"]["compliance_model"] == "constant"
    assert called["init"]["mesh_scale_factor"] == 2.0
    assert called["init"]["convert_to_cm"] is True
    assert called["run"]["optimize_bcs"] is False
    assert called["run"]["run_threed"] is False
