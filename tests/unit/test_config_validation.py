import os

import pytest

from svzerodtrees.config import load_config, render_schema


def test_load_valid_pipeline_config(tmp_path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        """
version: 1
workflow: pipeline
paths:
  root: .
  zerod_config: zerod_config.json
  clinical_targets: clinical_targets.csv
  mesh_surfaces: mesh-surfaces
  preop_dir: preop
  postop_dir: postop
  adapted_dir: adapted
bcs:
  type: impedance
  compliance_model: constant
  is_pulmonary: true
pipeline:
  run_steady: true
  optimize_bcs: false
  run_threed: false
  adapt: false
"""
    )
    cfg = load_config(str(cfg_path))
    assert cfg.workflow == "pipeline"
    assert cfg.paths.preop_dir is not None


def test_unknown_key_raises(tmp_path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        """
version: 1
workflow: pipeline
paths:
  root: .
  zerod_config: zerod_config.json
  clinical_targets: clinical_targets.csv
  mesh_surfaces: mesh-surfaces
  preop_dir: preop
  postop_dir: postop
  adapted_dir: adapted
unexpected_key: true
"""
    )
    with pytest.raises(ValueError):
        load_config(str(cfg_path))


def test_invalid_version_raises(tmp_path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text("version: 999\nworkflow: pipeline\npaths: {root: .}\n")

    with pytest.raises(ValueError, match="Unsupported config version"):
        load_config(str(cfg_path))


def test_invalid_workflow_raises(tmp_path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text("version: 1\nworkflow: unknown\npaths: {root: .}\n")

    with pytest.raises(ValueError, match="workflow must be one of"):
        load_config(str(cfg_path))


def test_load_valid_calibrate_0d_from_3d_config(tmp_path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        f"""
version: 1
workflow: calibrate_0d_from_3d
paths:
  root: {tmp_path}
  zerod_config: zerod.json
  output_config: calibrated.json
calibration:
  data_source:
    mode: mapped_centerline
    mapped_centerline_result: mapped.vtp
    centerline: centerline.vtp
  parameters:
    vessels:
      default: [R_poiseuille, C]
      overrides:
        branch0_seg0: [R_poiseuille]
    junctions:
      default: [R_poiseuille, L]
  solver:
    maximum_iterations: 12
""",
        encoding="utf-8",
    )

    cfg = load_config(str(cfg_path))

    assert cfg.workflow == "calibrate_0d_from_3d"
    assert cfg.calibration is not None
    assert cfg.calibration.data_source.mapped_centerline_result == str(tmp_path / "mapped.vtp")
    assert cfg.calibration.parameters.vessels.overrides["branch0_seg0"] == ["R_poiseuille"]
    assert cfg.calibration.solver.maximum_iterations == 12


def test_calibration_requires_mapped_centerline_source_fields(tmp_path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        """
version: 1
workflow: calibrate_0d_from_3d
paths:
  root: .
  zerod_config: zerod.json
  output_config: calibrated.json
calibration:
  data_source:
    mode: mapped_centerline
    centerline: centerline.vtp
  parameters:
    vessels: {}
    junctions: {}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mapped_centerline_result is required"):
        load_config(str(cfg_path))


def test_paths_resolve_relative_to_root(tmp_path):
    root = tmp_path / "case"
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        f"""
version: 1
workflow: pipeline
paths:
  root: {root}
  zerod_config: inputs/model.json
  clinical_targets: targets.csv
  mesh_surfaces: mesh-surfaces
"""
    )

    cfg = load_config(str(cfg_path))

    assert cfg.paths.root == str(root)
    assert cfg.paths.zerod_config == os.path.join(str(root), "inputs/model.json")
    assert cfg.paths.clinical_targets == os.path.join(str(root), "targets.csv")


def test_load_deformable_threed_config(tmp_path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        """
version: 1
workflow: pipeline
paths:
  root: .
  zerod_config: zerod_config.json
  clinical_targets: clinical_targets.csv
  mesh_surfaces: mesh-surfaces
  preop_dir: preop
  postop_dir: postop
  adapted_dir: adapted
threed:
  wall_model: deformable
  elasticity_modulus: 123.4
  poisson_ratio: 0.49
  shell_thickness: 0.22
  prestress_file: auto
  execution:
    mode: local
    executable: svmultiphysics
  tissue_support:
    enabled: true
    type: uniform
    stiffness: 1000.0
    damping: 10000.0
    apply_along_normal_direction: true
"""
    )
    cfg = load_config(str(cfg_path))
    assert cfg.threed is not None
    assert cfg.threed.wall_model == "deformable"
    assert cfg.threed.elasticity_modulus == pytest.approx(123.4)
    assert cfg.threed.poisson_ratio == pytest.approx(0.49)
    assert cfg.threed.shell_thickness == pytest.approx(0.22)
    assert cfg.threed.prestress_file == "auto"
    assert cfg.threed.tissue_support is not None
    assert cfg.threed.tissue_support.type == "uniform"
    assert cfg.threed.tissue_support.stiffness == pytest.approx(1000.0)
    assert cfg.threed.tissue_support.damping == pytest.approx(10000.0)
    assert cfg.threed.tissue_support.apply_along_normal_direction is True


def test_load_deformable_defaults_when_values_omitted(tmp_path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        """
version: 1
workflow: pipeline
paths:
  root: .
  zerod_config: zerod_config.json
  clinical_targets: clinical_targets.csv
  mesh_surfaces: mesh-surfaces
  preop_dir: preop
  postop_dir: postop
  adapted_dir: adapted
threed:
  wall_model: deformable
  execution:
    mode: local
    executable: svmultiphysics
"""
    )
    cfg = load_config(str(cfg_path))
    assert cfg.threed is not None
    assert cfg.threed.wall_model == "deformable"
    assert cfg.threed.elasticity_modulus == pytest.approx(5062674.563165)
    assert cfg.threed.poisson_ratio == pytest.approx(0.5)
    assert cfg.threed.shell_thickness == pytest.approx(0.12)


def test_threed_execution_local_config_parses(tmp_path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        """
version: 1
workflow: pipeline
paths:
  root: .
threed:
  execution:
    mode: local
    executable: svmultiphysics
    clean_command: null
"""
    )
    cfg = load_config(str(cfg_path))
    assert cfg.threed is not None
    assert cfg.threed.execution.mode == "local"
    assert cfg.threed.execution.executable == "svmultiphysics"
    assert cfg.threed.execution.clean_command is None
    assert cfg.threed.execution.slurm.nodes == 3


def test_threed_execution_slurm_config_parses(tmp_path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        """
version: 1
workflow: pipeline
paths:
  root: .
threed:
  execution:
    mode: slurm
    executable: /opt/sv/bin/svmultiphysics
    submit_command: sbatch
    slurm:
      nodes: 2
      procs_per_node: 8
      memory: 12
      hours: 4
      partition: test
      qos: debug
      mail_user: user@example.com
      mail_types: [fail, end]
"""
    )
    cfg = load_config(str(cfg_path))
    assert cfg.threed is not None
    assert cfg.threed.execution.mode == "slurm"
    assert cfg.threed.execution.executable == "/opt/sv/bin/svmultiphysics"
    assert cfg.threed.execution.slurm.nodes == 2
    assert cfg.threed.execution.slurm.procs_per_node == 8
    assert cfg.threed.execution.slurm.partition == "test"
    assert cfg.threed.execution.slurm.qos == "debug"
    assert cfg.threed.execution.slurm.mail_user == "user@example.com"
    assert cfg.threed.execution.slurm.mail_types == ["fail", "end"]


def test_invalid_threed_execution_mode_raises(tmp_path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        """
version: 1
workflow: pipeline
paths:
  root: .
threed:
  execution:
    mode: ssh
"""
    )
    with pytest.raises(ValueError, match="execution.mode"):
        load_config(str(cfg_path))


def test_missing_threed_execution_executable_raises(tmp_path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        """
version: 1
workflow: pipeline
paths:
  root: .
threed:
  wall_model: rigid
  execution:
    mode: local
"""
    )
    with pytest.raises(ValueError, match="threed.execution.executable"):
        load_config(str(cfg_path))


def test_missing_threed_execution_section_raises(tmp_path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        """
version: 1
workflow: pipeline
paths:
  root: .
threed:
  wall_model: rigid
"""
    )
    with pytest.raises(ValueError, match="threed.execution.executable"):
        load_config(str(cfg_path))


def test_unknown_threed_execution_key_raises(tmp_path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        """
version: 1
workflow: pipeline
paths:
  root: .
threed:
  execution:
    mode: local
    extra: true
"""
    )
    with pytest.raises(ValueError, match="threed.execution"):
        load_config(str(cfg_path))


def test_invalid_wall_model_raises(tmp_path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        """
version: 1
workflow: pipeline
paths:
  root: .
  zerod_config: zerod_config.json
  clinical_targets: clinical_targets.csv
  mesh_surfaces: mesh-surfaces
  preop_dir: preop
  postop_dir: postop
  adapted_dir: adapted
threed:
  wall_model: foobar
"""
    )
    with pytest.raises(ValueError, match="wall_model"):
        load_config(str(cfg_path))


def test_invalid_deformable_material_values_raise(tmp_path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        """
version: 1
workflow: pipeline
paths:
  root: .
  zerod_config: zerod_config.json
  clinical_targets: clinical_targets.csv
  mesh_surfaces: mesh-surfaces
  preop_dir: preop
  postop_dir: postop
  adapted_dir: adapted
threed:
  wall_model: deformable
  elasticity_modulus: -1.0
  poisson_ratio: 0.8
  shell_thickness: 0.0
"""
    )
    with pytest.raises(ValueError):
        load_config(str(cfg_path))


def test_tissue_support_requires_deformable_wall(tmp_path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        """
version: 1
workflow: pipeline
paths:
  root: .
threed:
  wall_model: rigid
  tissue_support:
    enabled: true
    type: uniform
    stiffness: 1000.0
    damping: 10000.0
"""
    )
    with pytest.raises(ValueError, match="tissue_support"):
        load_config(str(cfg_path))


def test_spatial_tissue_support_resolves_relative_path(tmp_path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        f"""
version: 1
workflow: pipeline
paths:
  root: {tmp_path}
threed:
  wall_model: deformable
  tissue_support:
    enabled: true
    type: spatial
    spatial_values_file_path: robin/values.vtp
  execution:
    mode: local
    executable: svmultiphysics
"""
    )
    cfg = load_config(str(cfg_path))
    assert cfg.threed is not None
    assert cfg.threed.tissue_support is not None
    assert cfg.threed.tissue_support.type == "spatial"
    assert cfg.threed.tissue_support.spatial_values_file_path == os.path.join(
        str(tmp_path), "robin/values.vtp"
    )


def test_invalid_mixed_tissue_support_raises(tmp_path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        """
version: 1
workflow: pipeline
paths:
  root: .
threed:
  wall_model: deformable
  tissue_support:
    enabled: true
    type: spatial
    stiffness: 1.0
    spatial_values_file_path: robin_values.vtp
"""
    )
    with pytest.raises(ValueError, match="forbids stiffness"):
        load_config(str(cfg_path))


def test_prestress_file_path_alias_resolves(tmp_path):
    cfg_path = tmp_path / "cfg.yml"
    prestress_path = tmp_path / "prestress" / "result_0003.vtu"
    cfg_path.write_text(
        f"""
version: 1
workflow: pipeline
paths:
  root: {tmp_path}
  zerod_config: zerod_config.json
  clinical_targets: clinical_targets.csv
  mesh_surfaces: mesh-surfaces
  preop_dir: preop
  postop_dir: postop
  adapted_dir: adapted
threed:
  wall_model: deformable
  prestress_file: {prestress_path}
  execution:
    mode: local
    executable: svmultiphysics
"""
    )
    cfg = load_config(str(cfg_path))
    assert cfg.threed is not None
    assert cfg.threed.prestress_file_path == str(prestress_path)


def test_tree_config_parses_constant_and_olufsen_compliance(tmp_path):
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
  type: impedance
trees:
  lpa:
    lrr: 10.0
    diameter: 0.3
    d_min: 0.01
    alpha: 0.9
    beta: 0.6
    inductance: 0.05
    compliance:
      model: constant
      params:
        value: 66000.0
  rpa:
    lrr: 9.0
    diameter: 0.32
    d_min: 0.01
    xi: 2.1
    eta_sym: 0.7
    compliance:
      model: olufsen
      params:
        k1: 1.0
        k2: -2.0
        k3: 3.0
"""
    )

    cfg = load_config(str(cfg_path))

    assert cfg.trees.lpa.compliance_model.value == pytest.approx(66000.0)
    assert cfg.trees.rpa.compliance_model.k1 == pytest.approx(1.0)
    assert cfg.trees.rpa.xi == pytest.approx(2.1)


def test_postprocess_config_resolves_inputs_and_outputs(tmp_path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        f"""
version: 1
workflow: postprocess
paths:
  root: {tmp_path}
postprocess:
  figures:
    - kind: generation_metrics
      input: tree.pkl
      output: figures/tree.png
      options:
        dpi: 150
"""
    )

    cfg = load_config(str(cfg_path))
    fig = cfg.postprocess.figures[0]

    assert fig.input == os.path.join(str(tmp_path), "tree.pkl")
    assert fig.output == os.path.join(str(tmp_path), "figures/tree.png")
    assert fig.options == {"dpi": 150}


def test_postprocess_analysis_config_resolves_paths_and_options(tmp_path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        f"""
version: 1
workflow: postprocess
paths:
  root: {tmp_path}
postprocess:
  analyses:
    - kind: pulmonary_resistance_map
      output: results/resistance_map
      options:
        svslicer_path: tools/svslicer
        centerline: centerlines.vtp
        frames_csv: frames.csv
        cycle_duration_s: 1.0
        workers: auto
        intermediate_dir: scratch/mapped
""",
        encoding="utf-8",
    )

    cfg = load_config(str(cfg_path))
    analysis = cfg.postprocess.analyses[0]

    assert analysis.output == os.path.join(str(tmp_path), "results/resistance_map")
    assert analysis.options["svslicer_path"] == os.path.join(str(tmp_path), "tools/svslicer")
    assert analysis.options["centerline"] == os.path.join(str(tmp_path), "centerlines.vtp")
    assert analysis.options["frames_csv"] == os.path.join(str(tmp_path), "frames.csv")
    assert analysis.options["intermediate_dir"] == os.path.join(str(tmp_path), "scratch/mapped")
    assert analysis.options["workers"] == "auto"


def test_postprocess_analysis_requires_core_options(tmp_path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        f"""
version: 1
workflow: postprocess
paths:
  root: {tmp_path}
postprocess:
  analyses:
    - kind: pulmonary_resistance_map
      output: results/resistance_map
      options:
        centerline: centerlines.vtp
        frames_csv: frames.csv
        cycle_duration_s: 1.0
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="options.svslicer_path"):
        load_config(str(cfg_path))


def test_postprocess_suite_analysis_config_resolves_paths(tmp_path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        f"""
version: 1
workflow: postprocess
paths:
  root: {tmp_path}
postprocess:
  analyses:
    - kind: pulmonary_threed_suite
      output: results/postprocess
      options:
        simulation_dir: preop
        centerline: centerlines.vtp
        svslicer_path: tools/svslicer
        clinical_targets: clinical_targets.csv
        stage: preop
        inflow_csv: inflow.csv
        resistance_map_workers: 2
""",
        encoding="utf-8",
    )

    cfg = load_config(str(cfg_path))
    analysis = cfg.postprocess.analyses[0]

    assert analysis.output == os.path.join(str(tmp_path), "results/postprocess")
    assert analysis.options["simulation_dir"] == os.path.join(str(tmp_path), "preop")
    assert analysis.options["centerline"] == os.path.join(str(tmp_path), "centerlines.vtp")
    assert analysis.options["svslicer_path"] == os.path.join(str(tmp_path), "tools/svslicer")
    assert analysis.options["clinical_targets"] == os.path.join(str(tmp_path), "clinical_targets.csv")
    assert analysis.options["inflow_csv"] == os.path.join(str(tmp_path), "inflow.csv")
    assert analysis.options["resistance_map_workers"] == 2


def test_postprocess_suite_requires_cycle_duration_or_inflow(tmp_path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        f"""
version: 1
workflow: postprocess
paths:
  root: {tmp_path}
postprocess:
  analyses:
    - kind: pulmonary_threed_suite
      output: results/postprocess
      options:
        simulation_dir: preop
        centerline: centerlines.vtp
        svslicer_path: tools/svslicer
        stage: preop
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cycle_duration_s or options.inflow_csv"):
        load_config(str(cfg_path))


def test_render_schema_includes_supported_workflows():
    schema = render_schema()
    assert "workflow: pipeline" in schema
    assert "construct_trees" in schema
    assert "postprocess:" in schema
