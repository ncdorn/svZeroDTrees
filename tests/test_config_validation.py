import pytest
from svzerodtrees.config import load_config


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
