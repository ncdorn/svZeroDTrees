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
        threed=SimpleNamespace(
            mesh_scale_factor=2.0,
            convert_to_cm=True,
            wall_model="deformable",
            elasticity_modulus=111.0,
            poisson_ratio=0.45,
            shell_thickness=0.25,
            prestress_file="auto",
            prestress_file_path=str(tmp_path / "prestress.vtu"),
            tissue_support=SimpleNamespace(
                enabled=True,
                type="uniform",
                stiffness=1000.0,
                damping=10000.0,
                apply_along_normal_direction=True,
                spatial_values_file_path=None,
            ),
        ),
    )

    PipelineWorkflow.from_config(cfg).run()

    assert called["init"]["bc_type"] == "impedance"
    assert called["init"]["compliance_model"] == "constant"
    assert called["init"]["mesh_scale_factor"] == 2.0
    assert called["init"]["convert_to_cm"] is True
    assert called["init"]["wall_model"] == "deformable"
    assert called["init"]["elasticity_modulus"] == 111.0
    assert called["init"]["poisson_ratio"] == 0.45
    assert called["init"]["shell_thickness"] == 0.25
    assert called["init"]["prestress_file"] == "auto"
    assert called["init"]["prestress_file_path"] == str(tmp_path / "prestress.vtu")
    assert called["init"]["tissue_support"]["stiffness"] == 1000.0
    assert called["init"]["tissue_support"]["damping"] == 10000.0
    assert called["init"]["adapted_dir"] is None
    assert called["run"]["optimize_bcs"] is False
    assert called["run"]["run_threed"] is False


def test_pipeline_mapping_adapt_enabled_uses_adapted_dir(monkeypatch, tmp_path):
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
            zerod_config=str(tmp_path / "simplified_nonlinear_zerod.json"),
            clinical_targets=str(tmp_path / "clinical_targets.csv"),
            mesh_surfaces=str(tmp_path / "mesh-surfaces"),
            preop_dir=str(tmp_path / "preop"),
            postop_dir=str(tmp_path / "postop"),
            adapted_dir=str(tmp_path / "adapted"),
            inflow=None,
        ),
        bcs=SimpleNamespace(type="impedance", compliance_model="olufsen", tune_space=None),
        adaptation=SimpleNamespace(method="cwss", location="uniform", iterations=10),
        pipeline=SimpleNamespace(run_steady=False, optimize_bcs=True, run_threed=False, adapt=True),
        threed=SimpleNamespace(
            mesh_scale_factor=1.0,
            convert_to_cm=False,
            wall_model="rigid",
            elasticity_modulus=5062674.563165,
            poisson_ratio=0.5,
            shell_thickness=0.12,
            prestress_file=None,
            prestress_file_path=None,
        ),
    )

    PipelineWorkflow.from_config(cfg).run()

    assert called["init"]["adapted_dir"] == "adapted"
    assert called["init"]["wall_model"] == "rigid"


def test_pipeline_mapping_without_threed_uses_simulation_defaults(monkeypatch, tmp_path):
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
        pipeline=SimpleNamespace(run_steady=True, optimize_bcs=True, run_threed=False, adapt=True),
        threed=None,
    )

    PipelineWorkflow.from_config(cfg).run()
    assert "wall_model" not in called["init"]


def test_pipeline_mapping_passes_threed_execution_config(monkeypatch, tmp_path):
    called = {}

    class DummySimulation:
        def __init__(self, **kwargs):
            called["init"] = kwargs

        def run_pipeline(self, **kwargs):
            called["run"] = kwargs

    monkeypatch.setattr("svzerodtrees.api.Simulation", DummySimulation)

    execution = SimpleNamespace(
        mode="local",
        executable="svmultiphysics",
        submit_command="sbatch",
        clean_command=None,
        slurm=SimpleNamespace(
            nodes=1,
            procs_per_node=2,
            memory=4,
            hours=1,
            partition="test",
            qos="debug",
        ),
    )
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
        bcs=None,
        adaptation=None,
        pipeline=SimpleNamespace(run_steady=False, optimize_bcs=False, run_threed=True, adapt=False),
        threed=SimpleNamespace(
            mesh_scale_factor=1.0,
            convert_to_cm=False,
            wall_model="rigid",
            elasticity_modulus=5062674.563165,
            poisson_ratio=0.5,
            shell_thickness=0.12,
            prestress_file=None,
            prestress_file_path=None,
            execution=execution,
        ),
    )

    PipelineWorkflow.from_config(cfg).run()

    assert called["init"]["execution_config"]["mode"] == "local"
    assert called["init"]["execution_config"]["executable"] == "svmultiphysics"
    assert called["init"]["execution_config"]["clean_command"] is None
    assert called["init"]["execution_config"]["slurm"].partition == "test"
