from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from svzerodtrees.config import ImpedanceConfig
from svzerodtrees.api import (
    AdaptationWorkflow,
    Calibrate0DFrom3DWorkflow,
    ConstructTreesWorkflow,
    PipelineWorkflow,
    PostprocessWorkflow,
    TuneBCsWorkflow,
    run_from_config_file,
)
from svzerodtrees.adaptation.workflow import _mean_resistances
from svzerodtrees.tune_bcs.tune_space import FreeParam, TuneSpace


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


def test_construct_trees_workflow_loads_rcr_params_from_csv(monkeypatch, tmp_path):
    optimized_csv = tmp_path / "optimized_rcr_params.csv"
    optimized_csv.write_text(
        "R_LPA,C_LPA,R_RPA,C_RPA\n1.0,2.0,3.0,4.0\n",
        encoding="utf-8",
    )

    calls = {}

    class DummyConfigHandler:
        def to_json(self, path):
            calls["output_path"] = path

    monkeypatch.setattr(
        "svzerodtrees.api.ConfigHandler",
        SimpleNamespace(from_json=lambda *_args, **_kwargs: DummyConfigHandler()),
    )
    monkeypatch.setattr(
        "svzerodtrees.api.ClinicalTargets",
        SimpleNamespace(from_csv=lambda *_args, **_kwargs: SimpleNamespace(wedge_p=12.0)),
    )

    def fake_assign_rcr_bcs(config_handler, mesh_surfaces_path, wedge_pressure, rcr_params, **kwargs):
        calls["mesh_surfaces_path"] = mesh_surfaces_path
        calls["wedge_pressure"] = wedge_pressure
        calls["rcr_params"] = rcr_params
        calls["kwargs"] = kwargs

    monkeypatch.setattr("svzerodtrees.api.assign_rcr_bcs", fake_assign_rcr_bcs)

    cfg = SimpleNamespace(
        paths=SimpleNamespace(
            root=str(tmp_path),
            zerod_config=str(tmp_path / "model.json"),
            clinical_targets=str(tmp_path / "targets.csv"),
            mesh_surfaces=str(tmp_path / "mesh-surfaces"),
            output_config=str(tmp_path / "svzerod_config_with_bcs.json"),
            optimized_params=None,
        ),
        bcs=SimpleNamespace(type="rcr", rcr_params=None, is_pulmonary=True),
        trees=SimpleNamespace(),
        threed=None,
    )

    result = ConstructTreesWorkflow.from_config(cfg).run()

    assert result == {
        "status": "ok",
        "output_config": str(tmp_path / "svzerod_config_with_bcs.json"),
    }
    assert calls["rcr_params"] == [1.0, 2.0, 3.0, 4.0]
    assert calls["kwargs"]["convert_to_cm"] is False
    assert calls["kwargs"]["is_pulmonary"] is True


def test_construct_trees_workflow_rejects_missing_rcr_params(monkeypatch, tmp_path):
    class DummyConfigHandler:
        def to_json(self, path):
            raise AssertionError("to_json should not be called when params are missing")

    monkeypatch.setattr(
        "svzerodtrees.api.ConfigHandler",
        SimpleNamespace(from_json=lambda *_args, **_kwargs: DummyConfigHandler()),
    )
    monkeypatch.setattr(
        "svzerodtrees.api.ClinicalTargets",
        SimpleNamespace(from_csv=lambda *_args, **_kwargs: SimpleNamespace(wedge_p=12.0)),
    )

    cfg = SimpleNamespace(
        paths=SimpleNamespace(
            root=str(tmp_path),
            zerod_config=str(tmp_path / "model.json"),
            clinical_targets=str(tmp_path / "targets.csv"),
            mesh_surfaces=str(tmp_path / "mesh-surfaces"),
            output_config=str(tmp_path / "svzerod_config_with_bcs.json"),
            optimized_params=None,
        ),
        bcs=SimpleNamespace(type="rcr", rcr_params=None, is_pulmonary=True),
        trees=SimpleNamespace(),
        threed=None,
    )

    with pytest.raises(ValueError, match="optimized_rcr_params.csv"):
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


def test_calibrate_0d_from_3d_workflow_requires_calibration_section():
    cfg = SimpleNamespace(
        paths=SimpleNamespace(
            zerod_config="model.json",
            output_config="calibrated.json",
        ),
        calibration=None,
    )

    with pytest.raises(ValueError, match="calibration section is required"):
        Calibrate0DFrom3DWorkflow.from_config(cfg).run()


def test_calibrate_0d_from_3d_workflow_returns_target_quality_identity(
    monkeypatch, tmp_path
):
    expected = {
        "status": "ok",
        "run_id": "run-001",
        "digests": {
            "normalized_input": "input",
            "observations": "observations",
            "solver_module": "solver",
            "output": "output",
        },
        "target_quality": {"status": "pass"},
    }
    calls = {}

    def fake_calibrate(**kwargs):
        calls.update(kwargs)
        return expected

    monkeypatch.setattr(
        "svzerodtrees.api.calibrate_0d_from_mapped_centerline", fake_calibrate
    )
    cfg = SimpleNamespace(
        paths=SimpleNamespace(
            zerod_config=str(tmp_path / "baseline.json"),
            output_config=str(tmp_path / "calibrated.json"),
        ),
        calibration=SimpleNamespace(),
    )

    result = Calibrate0DFrom3DWorkflow.from_config(cfg).run()

    assert result == expected
    assert calls == {
        "zerod_config_path": str(tmp_path / "baseline.json"),
        "output_config_path": str(tmp_path / "calibrated.json"),
        "calibration": cfg.calibration,
    }


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


def _full_pa_impedance_config():
    return ImpedanceConfig(
        tuning_model="full_pa",
        tune_space=TuneSpace(
            free=[FreeParam("lpa.alpha", init=0.9, lb=0.7, ub=0.99)],
            fixed=[],
            tied=[],
        ),
        outlet_mapping_mode="auto",
    )


def test_tune_bcs_full_pa_dispatches_canonical_iteration_service(monkeypatch, tmp_path):
    calls = {}
    expected = {
        "optimized_params_csv": str(tmp_path / "optimized_params.csv"),
        "pa_config_snapshot": str(tmp_path / "pa_config_tuning_snapshot.json"),
        "tuned_zerod_config": str(tmp_path / "svzerod_3d_coupling_tuned.json"),
        "outlet_cap_mapping": str(tmp_path / "outlet_cap_mapping.json"),
    }

    def fake_run(**kwargs):
        calls.update(kwargs)
        return expected

    monkeypatch.setattr("svzerodtrees.api.run_impedance_tuning_for_iteration", fake_run)
    cfg = SimpleNamespace(
        paths=SimpleNamespace(
            root=str(tmp_path),
            zerod_config=str(tmp_path / "full_pa_seed.json"),
            clinical_targets=str(tmp_path / "targets.csv"),
            mesh_surfaces=str(tmp_path / "mesh-surfaces"),
            inflow=str(tmp_path / "inflow.csv"),
        ),
        bcs=SimpleNamespace(type="impedance", impedance=_full_pa_impedance_config()),
        threed=None,
    )

    result = TuneBCsWorkflow.from_config(cfg).run()

    assert result == {"status": "ok", **expected}
    assert calls["iteration_dir"] == str(tmp_path)
    assert calls["seed_config"] == str(tmp_path / "full_pa_seed.json")
    assert calls["mesh_surfaces"] == str(tmp_path / "mesh-surfaces")
    assert calls["clinical_targets"] == str(tmp_path / "targets.csv")
    assert calls["inflow_path"] == str(tmp_path / "inflow.csv")
    assert calls["results_dir"] == str(tmp_path)
    assert calls["impedance_config"]["tuning_model"] == "full_pa"
    assert "tune_space" in calls["impedance_config"]


def test_tune_bcs_learned_full_pa_forwards_seed_and_reports_provenance(
    monkeypatch, tmp_path
):
    calls = {}
    learned_seed = tmp_path / "generated" / "learned_full_pa_seed.json"
    learned_metadata = tmp_path / "generated" / "learned_seed_metadata.json"

    monkeypatch.setattr(
        "svzerodtrees.api.generate_full_pa_learned_seed",
        lambda config: SimpleNamespace(
            seed_path=learned_seed, metadata_path=learned_metadata
        ),
    )

    def fake_run(**kwargs):
        calls.update(kwargs)
        return {"tuned_zerod_config": str(tmp_path / "tuned.json")}

    monkeypatch.setattr("svzerodtrees.api.run_impedance_tuning_for_iteration", fake_run)
    cfg = SimpleNamespace(
        paths=SimpleNamespace(
            root=str(tmp_path),
            zerod_config=None,
            clinical_targets=str(tmp_path / "targets.csv"),
            mesh_surfaces=str(tmp_path / "mesh-surfaces"),
            inflow=None,
        ),
        seed_generation=SimpleNamespace(),
        bcs=SimpleNamespace(type="impedance", impedance=_full_pa_impedance_config()),
        threed=None,
    )

    result = TuneBCsWorkflow.from_config(cfg).run()

    assert calls["seed_config"] == str(learned_seed)
    assert result["learned_seed"] == str(learned_seed)
    assert result["learned_seed_metadata"] == str(learned_metadata)


@pytest.mark.parametrize(
    ("missing_path", "expected_message"),
    [
        ("clinical_targets", "paths.clinical_targets is required"),
        ("mesh_surfaces", "paths.mesh_surfaces is required"),
    ],
)
def test_tune_bcs_validates_paths_before_learned_seed_generation(
    monkeypatch, tmp_path, missing_path, expected_message
):
    calls = []
    generated_dir = tmp_path / "generated"
    learned_seed = generated_dir / "learned_full_pa_seed.json"
    learned_metadata = generated_dir / "learned_seed_metadata.json"

    def fake_generate(config):
        calls.append(config)
        generated_dir.mkdir()
        learned_seed.write_text("{}", encoding="utf-8")
        learned_metadata.write_text("{}", encoding="utf-8")
        return SimpleNamespace(seed_path=learned_seed, metadata_path=learned_metadata)

    monkeypatch.setattr(
        "svzerodtrees.api.generate_full_pa_learned_seed", fake_generate
    )
    paths = SimpleNamespace(
        root=str(tmp_path),
        zerod_config=None,
        clinical_targets=str(tmp_path / "targets.csv"),
        mesh_surfaces=str(tmp_path / "mesh-surfaces"),
        inflow=None,
    )
    setattr(paths, missing_path, None)
    cfg = SimpleNamespace(
        paths=paths,
        seed_generation=SimpleNamespace(),
        bcs=SimpleNamespace(type="impedance", impedance=_full_pa_impedance_config()),
        threed=None,
    )

    with pytest.raises(ValueError, match=expected_message):
        TuneBCsWorkflow.from_config(cfg).run()

    assert calls == []
    assert not learned_seed.exists()
    assert not learned_metadata.exists()


def test_pipeline_passes_typed_full_pa_impedance_contract_to_simulation(
    monkeypatch, tmp_path
):
    calls = {}

    class DummySimulation:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def run_pipeline(self, **kwargs):
            calls["run"] = kwargs
            return {
                "tuned_zerod_config": str(tmp_path / "tuned.json"),
                "outlet_cap_mapping": str(tmp_path / "mapping.json"),
            }

    monkeypatch.setattr("svzerodtrees.api.Simulation", DummySimulation)
    cfg = SimpleNamespace(
        paths=SimpleNamespace(
            root=str(tmp_path),
            zerod_config=str(tmp_path / "full_pa_seed.json"),
            clinical_targets=str(tmp_path / "targets.csv"),
            mesh_surfaces=str(tmp_path / "mesh-surfaces"),
            preop_dir=str(tmp_path / "preop"),
            postop_dir=str(tmp_path / "postop"),
            adapted_dir=str(tmp_path / "adapted"),
            inflow=str(tmp_path / "inflow.csv"),
        ),
        bcs=SimpleNamespace(type="impedance", impedance=_full_pa_impedance_config()),
        adaptation=None,
        pipeline=SimpleNamespace(
            run_steady=False, optimize_bcs=True, run_threed=False, adapt=False
        ),
        threed=None,
    )

    result = PipelineWorkflow.from_config(cfg).run()

    assert result["status"] == "ok"
    assert result["tuned_zerod_config"].endswith("tuned.json")
    assert calls["init"]["zerod_config"] == str(tmp_path / "full_pa_seed.json")
    assert calls["init"]["impedance_config"]["tuning_model"] == "full_pa"
    assert calls["init"]["impedance_config"]["outlet_mapping_mode"] == "auto"
    assert calls["run"]["optimize_bcs"] is True


def test_pipeline_learned_full_pa_forwards_absolute_seed_and_reports_provenance(
    monkeypatch, tmp_path
):
    calls = {}
    learned_seed = tmp_path / "generated" / "learned_full_pa_seed.json"
    learned_metadata = tmp_path / "generated" / "learned_seed_metadata.json"

    monkeypatch.setattr(
        "svzerodtrees.api.generate_full_pa_learned_seed",
        lambda config: SimpleNamespace(
            seed_path=learned_seed, metadata_path=learned_metadata
        ),
    )

    class DummySimulation:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def run_pipeline(self, **kwargs):
            return {"learned_seed": "simulation-owned", "tuned": True}

    monkeypatch.setattr("svzerodtrees.api.Simulation", DummySimulation)
    cfg = SimpleNamespace(
        paths=SimpleNamespace(
            root=str(tmp_path),
            zerod_config=None,
            clinical_targets=None,
            preop_dir=None,
            postop_dir=None,
            adapted_dir=None,
            inflow=None,
        ),
        seed_generation=SimpleNamespace(),
        bcs=SimpleNamespace(type="impedance", impedance=_full_pa_impedance_config()),
        adaptation=None,
        pipeline=SimpleNamespace(
            run_steady=False, optimize_bcs=True, run_threed=False, adapt=False
        ),
        threed=None,
    )

    result = PipelineWorkflow.from_config(cfg).run()

    assert calls["init"]["zerod_config"] == str(learned_seed)
    assert result["learned_seed"] == "simulation-owned"
    assert result["learned_seed_metadata"] == str(learned_metadata)
    assert result["tuned"] is True


def test_learned_seed_generation_failure_prevents_downstream_dispatch(
    monkeypatch, tmp_path
):
    calls = {"simulation": 0}

    def fail_generation(config):
        raise RuntimeError("learned generation failed")

    monkeypatch.setattr("svzerodtrees.api.generate_full_pa_learned_seed", fail_generation)

    class DummySimulation:
        def __init__(self, **kwargs):
            calls["simulation"] += 1

    monkeypatch.setattr("svzerodtrees.api.Simulation", DummySimulation)
    cfg = SimpleNamespace(
        paths=SimpleNamespace(
            root=str(tmp_path),
            zerod_config=None,
            clinical_targets=None,
            preop_dir=None,
            postop_dir=None,
            adapted_dir=None,
            inflow=None,
        ),
        seed_generation=SimpleNamespace(),
        bcs=SimpleNamespace(type="impedance", impedance=_full_pa_impedance_config()),
        adaptation=None,
        pipeline=None,
        threed=None,
    )

    with pytest.raises(RuntimeError, match="learned generation failed"):
        PipelineWorkflow.from_config(cfg).run()
    assert calls["simulation"] == 0


def test_postprocess_workflow_dispatches_analysis(monkeypatch, tmp_path):
    calls = {}

    def fake_compute_pulmonary_resistance_map(**kwargs):
        calls["kwargs"] = kwargs
        return {"kind": "pulmonary_resistance_map", "summary_csv": "summary.csv"}

    monkeypatch.setattr(
        "svzerodtrees.api.compute_pulmonary_resistance_map",
        fake_compute_pulmonary_resistance_map,
    )

    cfg = SimpleNamespace(
        postprocess=SimpleNamespace(
            figures=[],
            analyses=[
                SimpleNamespace(
                    kind="pulmonary_resistance_map",
                    output=str(tmp_path / "results"),
                    options={
                        "svslicer_path": "/tmp/svslicer",
                        "centerline": "/tmp/centerlines.vtp",
                        "frames_csv": "/tmp/frames.csv",
                        "cycle_duration_s": 1.0,
                        "workers": "auto",
                    },
                )
            ],
        )
    )

    result = PostprocessWorkflow.from_config(cfg).run()

    assert calls["kwargs"]["output_dir"] == str(tmp_path / "results")
    assert calls["kwargs"]["workers"] == "auto"
    assert result["analysis_outputs"][0]["summary_csv"] == "summary.csv"


def test_postprocess_workflow_dispatches_pulmonary_threed_suite(monkeypatch, tmp_path):
    calls = {}

    def fake_run_suite(**kwargs):
        calls["kwargs"] = kwargs
        return {"kind": "pulmonary_threed_suite", "metadata_json": "suite.json"}

    monkeypatch.setattr(
        "svzerodtrees.api.run_pulmonary_threed_postprocess_suite",
        fake_run_suite,
    )

    cfg = SimpleNamespace(
        postprocess=SimpleNamespace(
            figures=[],
            analyses=[
                SimpleNamespace(
                    kind="pulmonary_threed_suite",
                    output=str(tmp_path / "postprocess"),
                    options={
                        "simulation_dir": "/tmp/preop",
                        "centerline": "/tmp/centerlines.vtp",
                        "svslicer_path": "/tmp/svslicer",
                        "clinical_targets": "/tmp/clinical_targets.csv",
                        "stage": "preop",
                        "inflow_csv": "/tmp/inflow.csv",
                        "resistance_map_workers": 2,
                    },
                )
            ],
        )
    )

    result = PostprocessWorkflow.from_config(cfg).run()

    assert calls["kwargs"]["output_dir"] == str(tmp_path / "postprocess")
    assert calls["kwargs"]["stage"] == "preop"
    assert calls["kwargs"]["resistance_map_workers"] == 2
    assert result["analysis_outputs"][0]["metadata_json"] == "suite.json"


def test_mean_resistances_falls_back_to_postprocessed_mpa_csv(tmp_path):
    csv_path = tmp_path / "results" / "postprocess" / "mpa_pressure_vs_time.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text("time_s,mpa_pressure_mmhg\n0.0,15.0\n1.0,15.0\n", encoding="utf-8")

    class DummyBlock:
        def __init__(self, surface):
            self.surface = surface

    class DummySvZeroDData:
        def get_result(self, block):
            pressure = np.array([13332.0, 13332.0]) if "lpa" in block.surface else np.array([10665.6, 10665.6])
            return np.array([0.0, 1.0]), np.array([2.0, 2.0]), pressure

    class DummySimDir:
        path = str(tmp_path / "preop")
        svzerod_data = DummySvZeroDData()
        svzerod_3Dcoupling = SimpleNamespace(
            coupling_blocks={
                "lpa": DummyBlock("lpa.vtp"),
                "rpa": DummyBlock("rpa.vtp"),
            }
        )

        def _compute_pressure_drops(self, get_mean=False):
            raise KeyError("branch0_seg0")

        def compute_pressure_drop(self, steady=True):
            raise KeyError("branch0_seg0")

        def flow_split(self, get_mean=True, verbose=False):
            return {"lpa": 2.0}, {"rpa": 4.0}

    lpa_resistance, rpa_resistance = _mean_resistances(DummySimDir())

    assert lpa_resistance == pytest.approx((15.0 * 1333.2 - 13332.0) / 2.0)
    assert rpa_resistance == pytest.approx((15.0 * 1333.2 - 10665.6) / 4.0)
