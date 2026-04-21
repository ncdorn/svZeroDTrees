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
"""
    )
    cfg = load_config(str(cfg_path))
    assert cfg.threed is not None
    assert cfg.threed.wall_model == "deformable"
    assert cfg.threed.elasticity_modulus == pytest.approx(123.4)
    assert cfg.threed.poisson_ratio == pytest.approx(0.49)
    assert cfg.threed.shell_thickness == pytest.approx(0.22)
    assert cfg.threed.prestress_file == "auto"


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
"""
    )
    cfg = load_config(str(cfg_path))
    assert cfg.threed is not None
    assert cfg.threed.wall_model == "deformable"
    assert cfg.threed.elasticity_modulus == pytest.approx(5062674.563165)
    assert cfg.threed.poisson_ratio == pytest.approx(0.5)
    assert cfg.threed.shell_thickness == pytest.approx(0.12)


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


def test_render_schema_includes_supported_workflows():
    schema = render_schema()
    assert "workflow: pipeline" in schema
    assert "construct_trees" in schema
    assert "postprocess:" in schema
