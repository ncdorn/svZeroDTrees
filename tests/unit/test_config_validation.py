import os

import pytest

from svzerodtrees.config import (
    ImpedanceConfig,
    LearnedSeedGenerationConfig,
    impedance_config_to_mapping,
    load_config,
    render_schema,
)


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


def _full_pa_pipeline_yaml(tmp_path, impedance_fields=""):
    return f"""
version: 1
workflow: pipeline
paths:
  root: {tmp_path}
  zerod_config: seed.json
  clinical_targets: targets.csv
  mesh_surfaces: mesh-surfaces
  inflow: inflow.csv
bcs:
  type: impedance
  impedance:
    tuning_model: full_pa
    outlet_mapping_mode: auto
    tune_space:
      free:
        - name: lpa.alpha
          init: 0.9
          lb: 0.7
          ub: 0.99
      fixed: []
      tied: []
{impedance_fields}
pipeline:
  optimize_bcs: true
  run_threed: false
  adapt: false
"""


def _learned_seed_pipeline_yaml(
    tmp_path,
    *,
    missing_seed_field=None,
    seed_extra="",
    static_seed=False,
    bcs_type="impedance",
    is_pulmonary=True,
    tuning_model="full_pa",
    mapping_mode="serialized_cap_order",
    pipeline_extra="",
):
    seed_fields = [
        ("method", "learned_zerod"),
        ("anatomy", "pulmonary"),
        ("input_zerod_config", "inputs/source.json"),
        ("centerline", "inputs/centerline.vtp"),
        ("svzerodsolver", "tools/svzerodsolver"),
        ("output_dir", "generated/learned"),
    ]
    seed_fields = [
        (key, value) for key, value in seed_fields if key != missing_seed_field
    ]
    static_line = "  zerod_config: inputs/static.json\n" if static_seed else ""
    impedance_block = "" if bcs_type == "rcr" else f"""  impedance:
    tuning_model: {tuning_model}
    outlet_mapping_mode: {mapping_mode}
"""
    return f"""
version: 1
workflow: pipeline
paths:
  root: {tmp_path}
{static_line}  clinical_targets: inputs/targets.csv
  mesh_surfaces: inputs/mesh-surfaces
seed_generation:
""" + "".join(f"  {key}: {value}\n" for key, value in seed_fields) + f"""{seed_extra}
bcs:
  type: {bcs_type}
  is_pulmonary: {str(is_pulmonary).lower()}
{impedance_block}pipeline:
pipeline:
  optimize_bcs: true
{pipeline_extra}
"""


def test_learned_seed_generation_parses_typed_root_relative_paths(tmp_path):
    cfg_path = tmp_path / "learned.yml"
    cfg_path.write_text(
        _learned_seed_pipeline_yaml(tmp_path),
        encoding="utf-8",
    )

    cfg = load_config(str(cfg_path))

    assert isinstance(cfg.seed_generation, LearnedSeedGenerationConfig)
    assert cfg.paths.zerod_config is None
    assert cfg.seed_generation.method == "learned_zerod"
    assert cfg.seed_generation.anatomy == "pulmonary"
    assert cfg.seed_generation.input_zerod_config == str(tmp_path / "inputs/source.json")
    assert cfg.seed_generation.centerline == str(tmp_path / "inputs/centerline.vtp")
    assert cfg.seed_generation.svzerodsolver == str(tmp_path / "tools/svzerodsolver")
    assert cfg.seed_generation.output_dir == str(tmp_path / "generated/learned")
    assert cfg.seed_generation.learned_zerod_executable == "learned-zerod"
    assert cfg.seed_generation.output_filename == "learned_full_pa_seed.json"
    assert cfg.seed_generation.keep_tmp is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"seed_extra": "  unexpected: true\n"}, "Unknown keys in seed_generation"),
        ({"missing_seed_field": "centerline"}, "seed_generation.centerline is required"),
        ({"static_seed": True}, "mutually exclusive"),
        ({"tuning_model": "rri", "mapping_mode": ""}, "tuning_model='full_pa'"),
        ({"bcs_type": "rcr", "mapping_mode": ""}, "bcs.type='impedance'"),
        ({"is_pulmonary": False}, "is_pulmonary=true"),
        ({"mapping_mode": "auto"}, "serialized_cap_order.*explicit"),
    ],
)
def test_learned_seed_generation_rejects_invalid_contract(tmp_path, kwargs, message):
    cfg_path = tmp_path / "invalid-learned.yml"
    cfg_path.write_text(
        _learned_seed_pipeline_yaml(tmp_path, **kwargs),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_config(str(cfg_path))


def test_tuning_pipeline_requires_one_seed_source(tmp_path):
    cfg_path = tmp_path / "missing-seed.yml"
    cfg_path.write_text(
        """
version: 1
workflow: pipeline
paths:
  root: .
  clinical_targets: targets.csv
  mesh_surfaces: mesh-surfaces
bcs:
  type: impedance
  impedance:
    tuning_model: full_pa
    outlet_mapping_mode: serialized_cap_order
pipeline:
  optimize_bcs: true
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly one of paths.zerod_config or seed_generation"):
        load_config(str(cfg_path))


def test_full_pa_impedance_block_parses_and_round_trips_to_service_mapping(tmp_path):
    cfg_path = tmp_path / "full-pa.yml"
    cfg_path.write_text(
        _full_pa_pipeline_yaml(
            tmp_path,
            impedance_fields=(
                "    solver: Nelder-Mead\n"
                "    nm_iter: 7\n"
                "    n_procs: 3\n"
                "    diameter_scale: 0.5\n"
                "    outlet_mapping_mode: explicit\n"
                "    outlet_mapping:\n"
                "      lpa_cap.vtp: LPA_OUTLET\n"
                "      rpa_cap.vtp: RPA_OUTLET"
            ),
        ),
        encoding="utf-8",
    )

    cfg = load_config(str(cfg_path))

    assert cfg.bcs is not None
    assert isinstance(cfg.bcs.impedance, ImpedanceConfig)
    assert cfg.bcs.type == "impedance"
    assert cfg.bcs.impedance.tuning_model == "full_pa"
    assert cfg.bcs.impedance.nm_iter == 7
    assert cfg.bcs.impedance.use_mean is False
    assert cfg.bcs.impedance.diameter_scale == 0.5
    assert cfg.bcs.impedance.outlet_mapping == {
        "lpa_cap.vtp": "LPA_OUTLET",
        "rpa_cap.vtp": "RPA_OUTLET",
    }
    mapped = impedance_config_to_mapping(cfg.bcs.impedance)
    assert mapped["tuning_model"] == "full_pa"
    assert mapped["outlet_mapping_mode"] == "explicit"
    assert mapped["tune_space"]["free"][0]["name"] == "lpa.alpha"


def test_legacy_flat_impedance_fields_adapt_once_with_deprecation_warning(tmp_path):
    cfg_path = tmp_path / "legacy.yml"
    cfg_path.write_text(
        f"""
version: 1
workflow: tune_bcs
paths:
  root: {tmp_path}
  zerod_config: seed.json
  clinical_targets: targets.csv
  mesh_surfaces: mesh-surfaces
bcs:
  type: impedance
  compliance_model: constant
  allow_ordered_outlet_mapping: false
  tune_space:
    free:
      - name: lpa.alpha
        init: 0.9
        lb: 0.7
        ub: 0.99
    fixed: []
    tied: []
""",
        encoding="utf-8",
    )

    with pytest.warns(DeprecationWarning, match="flat bcs impedance fields"):
        cfg = load_config(str(cfg_path))

    assert cfg.bcs is not None
    assert cfg.bcs.impedance is not None
    assert cfg.bcs.impedance.compliance_model == "constant"
    assert cfg.bcs.impedance.tune_space is not None


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (
            "    outlet_mapping: {cap.vtp: OUTLET}\n",
            "requires outlet_mapping_mode='explicit'",
        ),
        ("    unexpected: true\n", "Unknown keys in bcs.impedance"),
        (
            "    outlet_mapping_mode: explicit\n",
            "requires outlet_mapping",
        ),
    ],
)
def test_full_pa_impedance_block_rejects_ambiguous_or_unknown_controls(
    tmp_path, extra, message
):
    cfg_path = tmp_path / "invalid.yml"
    cfg_path.write_text(
        _full_pa_pipeline_yaml(tmp_path, impedance_fields=extra),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_config(str(cfg_path))


def test_nested_impedance_rejects_legacy_flat_duplicate(tmp_path):
    cfg_path = tmp_path / "contradictory.yml"
    cfg_path.write_text(
        _full_pa_pipeline_yaml(
            tmp_path,
            impedance_fields="  compliance_model: constant\n",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot be combined with legacy flat"):
        load_config(str(cfg_path))


def test_existing_rri_yaml_remains_supported_without_full_pa_controls(tmp_path):
    cfg_path = tmp_path / "rri.yml"
    cfg_path.write_text(
        f"""
version: 1
workflow: tune_bcs
paths:
  root: {tmp_path}
  zerod_config: reduced.json
  clinical_targets: targets.csv
  mesh_surfaces: mesh-surfaces
bcs:
  type: impedance
  tune_space:
    free:
      - name: lpa.alpha
        init: 0.9
        lb: 0.7
        ub: 0.99
    fixed: []
    tied: []
""",
        encoding="utf-8",
    )

    with pytest.warns(DeprecationWarning):
        cfg = load_config(str(cfg_path))
    assert cfg.bcs is not None
    assert cfg.bcs.type == "impedance"
    assert cfg.bcs.impedance is not None
    assert cfg.bcs.impedance.tuning_model == "rri"
    assert cfg.bcs.impedance.outlet_mapping_mode is None


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
    flow_observation_type: flow
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
    assert cfg.calibration.data_source.flow_observation_type == "flow"
    assert cfg.calibration.data_source.area_array is None
    assert cfg.calibration.input_normalization.infinite_vessel_compliance == "error"
    assert cfg.calibration.observation_qc.vessel_flow_continuity_tolerance == 0.10
    assert cfg.calibration.observation_qc.minimum_usable_samples == 3
    assert cfg.calibration.parameters.vessels.overrides["branch0_seg0"] == ["R_poiseuille"]
    assert cfg.calibration.solver.maximum_iterations == 12
    assert cfg.calibration.solver.parameter_ratio_warning_threshold == 100.0
    assert cfg.calibration.solver.confirmation_absolute_tolerance == 1e-8
    assert cfg.calibration.solver.confirmation_relative_tolerance == 1e-6
    assert cfg.calibration.solver.pressure_bound_multiplier == 10.0
    assert cfg.calibration.solver.flow_bound_multiplier == 10.0
    assert cfg.calibration.solver.cycle_stability_tolerance == 1e-3


def _target_focused_calibration_yaml(
    tmp_path, *, targets: str | None = None, solver: str = ""
) -> str:
    target_section = "" if targets is None else f"\n  targets:\n{targets}"
    return f"""
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
    flow_observation_type: flow
  parameters:
    vessels: {{}}
    junctions: {{}}
  solver:
{solver or '    replay_minimum_cycles: 3'}
  observation_qc:
    enforcement: target_focused
{target_section}
"""


def test_legacy_calibration_defaults_to_strict_network(tmp_path):
    cfg_path = tmp_path / "legacy.yml"
    cfg_path.write_text(
        _target_focused_calibration_yaml(tmp_path, targets=None).replace(
            "  observation_qc:\n    enforcement: target_focused\n", "  observation_qc: {}\n"
        ),
        encoding="utf-8",
    )

    cfg = load_config(str(cfg_path))

    assert cfg.calibration is not None
    assert cfg.calibration.targets is None
    assert cfg.calibration.observation_qc.enforcement == "strict_network"
    assert cfg.calibration.solver.replay_minimum_cycles == 3
    assert cfg.calibration.solver.replay_maximum_cycles == 10
    assert cfg.calibration.solver.required_consecutive_stable_pairs == 1


def test_target_focused_calibration_parses_explicit_roles_and_replay_settings(tmp_path):
    cfg_path = tmp_path / "target-focused.yml"
    cfg_path.write_text(
        _target_focused_calibration_yaml(
            tmp_path,
            solver=(
                "    replay_minimum_cycles: 4\n"
                "    replay_maximum_cycles: 9\n"
                "    required_consecutive_stable_pairs: 2"
            ),
            targets=(
                "    mpa_pressure:\n"
                "      vessel: branch0_seg0\n"
                "      interface: external_upstream\n"
                "      weight: 1.0\n"
                "      normalized_rms_tolerance: 0.05\n"
                "    rpa_flow_split:\n"
                "      rpa_vessel: branch1_seg0\n"
                "      lpa_vessel: branch2_seg0\n"
                "      interface: external_downstream\n"
                "      weight: 2.0\n"
                "      absolute_tolerance: 0.02\n"
                "    require_improvement_over_baseline: false"
            ),
        ),
        encoding="utf-8",
    )

    cfg = load_config(str(cfg_path))

    assert cfg.calibration is not None
    assert cfg.calibration.observation_qc.enforcement == "target_focused"
    assert cfg.calibration.targets is not None
    assert cfg.calibration.targets.mpa_pressure.vessel == "branch0_seg0"
    assert cfg.calibration.targets.mpa_pressure.interface == "external_upstream"
    assert cfg.calibration.targets.rpa_flow_split.rpa_vessel == "branch1_seg0"
    assert cfg.calibration.targets.rpa_flow_split.lpa_vessel == "branch2_seg0"
    assert cfg.calibration.targets.rpa_flow_split.weight == 2.0
    assert cfg.calibration.targets.require_improvement_over_baseline is False
    assert cfg.calibration.solver.replay_minimum_cycles == 4
    assert cfg.calibration.solver.replay_maximum_cycles == 9
    assert cfg.calibration.solver.required_consecutive_stable_pairs == 2


def test_target_focused_requires_both_targets(tmp_path):
    cfg_path = tmp_path / "missing-target.yml"
    cfg_path.write_text(
        _target_focused_calibration_yaml(tmp_path, targets="    mpa_pressure: {}"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires both mpa_pressure and rpa_flow_split"):
        load_config(str(cfg_path))


@pytest.mark.parametrize(
    ("targets", "message"),
    [
        (
            "    mpa_pressure:\n"
            "      vessel: branch0_seg0\n"
            "      interface: external_upstream\n"
            "      weight: 1.0\n"
            "      normalized_rms_tolerance: 0.05\n"
            "      unexpected: true\n"
            "    rpa_flow_split:\n"
            "      rpa_vessel: branch1_seg0\n"
            "      lpa_vessel: branch2_seg0\n"
            "      interface: external_downstream",
            "Unknown keys in calibration.targets.mpa_pressure",
        ),
        (
            "    mpa_pressure:\n"
            "      vessel: branch1_seg0\n"
            "      interface: external_upstream\n"
            "    rpa_flow_split:\n"
            "      rpa_vessel: branch1_seg0\n"
            "      lpa_vessel: branch2_seg0\n"
            "      interface: external_downstream",
            "distinct MPA, LPA, and RPA",
        ),
        (
            "    mpa_pressure:\n"
            "      vessel: branch0_seg0\n"
            "      interface: invalid\n"
            "    rpa_flow_split:\n"
            "      rpa_vessel: branch1_seg0\n"
            "      lpa_vessel: branch2_seg0\n"
            "      interface: external_downstream",
            "interface must be one of",
        ),
    ],
)
def test_target_focused_rejects_ambiguous_target_configuration(tmp_path, targets, message):
    cfg_path = tmp_path / "invalid-target.yml"
    cfg_path.write_text(
        _target_focused_calibration_yaml(tmp_path, targets=targets),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_config(str(cfg_path))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("weight", 0.0),
        ("weight", "nan"),
        ("normalized_rms_tolerance", -0.01),
        ("absolute_tolerance", "inf"),
    ],
)
def test_target_focused_rejects_non_positive_or_non_finite_scores(tmp_path, field, value):
    mpa_weight = value if field in {"weight", "normalized_rms_tolerance"} else 1.0
    mpa_tolerance = value if field == "normalized_rms_tolerance" else 0.05
    rpa_weight = value if field == "weight" else 1.0
    rpa_tolerance = value if field == "absolute_tolerance" else 0.02
    cfg_path = tmp_path / "invalid-score.yml"
    cfg_path.write_text(
        _target_focused_calibration_yaml(
            tmp_path,
            targets=(
                "    mpa_pressure:\n"
                "      vessel: branch0_seg0\n"
                "      interface: external_upstream\n"
                f"      weight: {mpa_weight}\n"
                f"      normalized_rms_tolerance: {mpa_tolerance}\n"
                "    rpa_flow_split:\n"
                "      rpa_vessel: branch1_seg0\n"
                "      lpa_vessel: branch2_seg0\n"
                "      interface: external_downstream\n"
                f"      weight: {rpa_weight}\n"
                f"      absolute_tolerance: {rpa_tolerance}"
            ),
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be finite and positive"):
        load_config(str(cfg_path))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("replay_minimum_cycles", 2, "replay_minimum_cycles must be an integer at least 3"),
        ("replay_maximum_cycles", 2, "replay_maximum_cycles must be an integer at least 3"),
        (
            "replay_maximum_cycles",
            3,
            "replay_maximum_cycles must be at least replay_minimum_cycles",
        ),
        (
            "required_consecutive_stable_pairs",
            0,
            "required_consecutive_stable_pairs must be an integer at least 1",
        ),
        (
            "required_consecutive_stable_pairs",
            10,
            "required_consecutive_stable_pairs",
        ),
    ],
)
def test_calibration_rejects_inconsistent_replay_bounds(tmp_path, field, value, message):
    cfg_path = tmp_path / "invalid-replay.yml"
    cfg_path.write_text(
        _target_focused_calibration_yaml(
            tmp_path,
            solver=f"    replay_minimum_cycles: 4\n    {field}: {value}",
            targets=(
                "    mpa_pressure:\n"
                "      vessel: branch0_seg0\n"
                "      interface: external_upstream\n"
                "    rpa_flow_split:\n"
                "      rpa_vessel: branch1_seg0\n"
                "      lpa_vessel: branch2_seg0\n"
                "      interface: external_downstream"
            ),
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_config(str(cfg_path))


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
    flow_observation_type: flow
  parameters:
    vessels: {}
    junctions: {}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mapped_centerline_result is required"):
        load_config(str(cfg_path))


def test_calibration_rejects_unknown_flow_observation_type(tmp_path):
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
    flow_observation_type: flow
    flow_observation_type: bogus
  parameters:
    vessels: {{}}
    junctions: {{}}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="flow_observation_type must be one of flow\\|velocity"):
        load_config(str(cfg_path))


def test_calibration_requires_explicit_flow_observation_type(tmp_path):
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
    vessels: {{}}
    junctions: {{}}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="flow_observation_type is required"):
        load_config(str(cfg_path))


def test_calibration_accepts_opt_in_infinite_compliance_normalization(tmp_path):
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
  input_normalization:
    infinite_vessel_compliance: zero
  data_source:
    mode: mapped_centerline
    mapped_centerline_result: mapped.vtp
    centerline: centerline.vtp
    flow_observation_type: flow
  parameters:
    vessels: {{}}
    junctions: {{}}
""",
        encoding="utf-8",
    )

    cfg = load_config(str(cfg_path))

    assert cfg.calibration is not None
    assert cfg.calibration.input_normalization.infinite_vessel_compliance == "zero"


def test_calibration_rejects_unknown_infinite_compliance_normalization(tmp_path):
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
  input_normalization:
    infinite_vessel_compliance: nan
  data_source:
    mode: mapped_centerline
    mapped_centerline_result: mapped.vtp
    centerline: centerline.vtp
    flow_observation_type: flow
  parameters:
    vessels: {{}}
    junctions: {{}}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="infinite_vessel_compliance must be one of error\\|zero"):
        load_config(str(cfg_path))


def test_calibration_validates_observation_qc_thresholds(tmp_path):
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
    flow_observation_type: flow
  parameters:
    vessels: {{}}
    junctions: {{}}
  observation_qc:
    minimum_path_coverage: 1.1
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="minimum_path_coverage must be in \\(0, 1\\]"):
        load_config(str(cfg_path))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("confirmation_absolute_tolerance", -1.0),
        ("confirmation_relative_tolerance", "nan"),
    ],
)
def test_calibration_validates_confirmation_tolerances(tmp_path, field, value):
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
    flow_observation_type: flow
  parameters:
    vessels: {{}}
    junctions: {{}}
  solver:
    {field}: {value}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="confirmation tolerances must be finite and non-negative"):
        load_config(str(cfg_path))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("pressure_bound_multiplier", 0.0, "bound multipliers must be finite and positive"),
        ("flow_bound_multiplier", "inf", "bound multipliers must be finite and positive"),
        ("cycle_stability_tolerance", -1.0, "cycle stability tolerance must be finite and non-negative"),
    ],
)
def test_calibration_validates_replay_settings(tmp_path, field, value, message):
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
    flow_observation_type: flow
  parameters:
    vessels: {{}}
    junctions: {{}}
  solver:
    {field}: {value}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
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
    assert "enforcement: strict_network" in schema
    assert "normalized_rms_tolerance" in schema
    assert "required_consecutive_stable_pairs" in schema
