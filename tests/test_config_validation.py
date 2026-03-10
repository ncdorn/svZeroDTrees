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
