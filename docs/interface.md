# svzerodtrees YAML Interface (v1)

This document defines the YAML schema used by the CLI and Python API. The schema is strict: unknown keys raise errors.

**Workflows**
- `pipeline`: end-to-end run using `Simulation.run_pipeline`.
- `tune_bcs`: tune boundary conditions only.
- `construct_trees`: assign impedance or RCR BCs.
- `adapt`: run microvascular adaptation (impedance BCs).
- `adapt_benchmark`: run local reduced-PA adaptation benchmark studies from optimized preop/postop reduced RRI configs.
- `postprocess`: generate figures from saved tree pickles or compute standalone analysis artifacts.
- `calibrate_0d_from_3d`: run stage-1 0D calibration from a precomputed mapped centerline result.

**Workflow Requirements**

| Workflow | Required Sections | Optional Sections |
| --- | --- | --- |
| `pipeline` | `version`, `workflow`, `paths` | `bcs`, `adaptation`, `pipeline`, `threed` |
| `tune_bcs` | `version`, `workflow`, `paths`, `bcs` | `threed` |
| `construct_trees` | `version`, `workflow`, `paths`, `bcs`, `trees` | `threed` |
| `adapt` | `version`, `workflow`, `paths` | `bcs`, `adaptation`, `threed` |
| `adapt_benchmark` | `version`, `workflow`, `paths`, `adapt_benchmark` | none |
| `postprocess` | `version`, `workflow`, `paths`, `postprocess` | none |
| `calibrate_0d_from_3d` | `version`, `workflow`, `paths`, `calibration` | none |

**Path Resolution**
- `paths.root` is resolved to an absolute path.
- All other relative paths are resolved relative to `paths.root`.

**Top-Level Keys**
- `version`: must be `1`.
- `workflow`: one of `pipeline|tune_bcs|construct_trees|adapt|adapt_benchmark|postprocess|calibrate_0d_from_3d`.
- `paths`: required for all workflows.
- `bcs`, `trees`, `adaptation`, `adapt_benchmark`, `pipeline`, `threed`, `postprocess`, `calibration`: required only for certain workflows.

**Paths**
```yaml
paths:
  root: .
  zerod_config: zerod_config.json
  clinical_targets: clinical_targets.csv
  mesh_surfaces: mesh-complete/mesh-surfaces
  preop_dir: preop
  postop_dir: postop
  adapted_dir: adapted
  inflow: inflow.csv
  optimized_params: optimized_params.csv
  output_config: svzerod_config_with_bcs.json
```

For `calibrate_0d_from_3d`, `paths.zerod_config` and `paths.output_config` are required.

**Calibration**
```yaml
calibration:
  data_source:
    mode: mapped_centerline
    mapped_centerline_result: path/to/result_centerline.vtp
    metadata_json: path/to/result_centerline_metadata.json
    centerline: path/to/centerline.vtp
    pressure_array: pressure
    flow_array: flow
    flow_observation_type: flow  # required: flow | velocity
    area_array: null
    branch_id_array: BranchId
    path_array: Path
  parameters:
    vessels:
      default: [R_poiseuille, C, L]
      overrides:
        branch0_seg0: [R_poiseuille]
    junctions:
      default: [R_poiseuille, L]
      overrides:
        J0: [R_poiseuille]
  solver:
    initial_damping_factor: 1.0
    maximum_iterations: 100
    tolerance_gradient: 1e-6
    tolerance_increment: 1e-10
    parameter_ratio_warning_threshold: 100.0
    confirmation_absolute_tolerance: 1e-8
    confirmation_relative_tolerance: 1e-6
    pressure_bound_multiplier: 10.0
    flow_bound_multiplier: 10.0
    cycle_stability_tolerance: 1e-3
  input_normalization:
    infinite_vessel_compliance: error  # error | zero
  observation_qc:
    vessel_flow_continuity_tolerance: 0.10
    junction_mass_balance_tolerance: 0.10
    root_waveform_rms_tolerance: 0.10
    minimum_pressure_drop_fraction: 0.95
    minimum_path_coverage: 0.99
    minimum_usable_samples: 3
```

Stage-1 calibration constraints:

- `calibration.data_source.mode` must currently be `mapped_centerline`.
- `flow_observation_type` is required; an omitted value is rejected because mapped flow names are otherwise ambiguous.
- svSlicer stack output is configured as `flow_array: flow` and `flow_observation_type: flow`. Its `flow_0..N` arrays are integrated volumetric flow in `cm^3/s` and are never multiplied by cross-sectional area.
- Legacy svSlicer stacks named `velocity_0..N` are accepted with `flow_array: velocity` and `flow_observation_type: flow`; the values are still treated as already integrated flow and are not area-scaled.
- `flow_observation_type: velocity` is reserved for a true velocity field and explicitly converts it with `area_array`.
- Numbered series require `metadata_json`. The sidecar must identify the final-cycle artifact, match point/frame counts and frame indices, list the paired pressure/flow arrays, declare pressure units and volumetric-flow units, and provide ordered timestamps plus `cycle_duration_s`.
- For timeseries calibration, stage 1 derives `dy` from the recorded sidecar timestamps, including the periodic wrap interval. Nonuniform timing fails before calibration.
- Stage 1 supports multi-segment branches when each 0D vessel includes `vessel_length`, so internal segment interfaces can be placed along the branch `Path`.
- External boundary interfaces are qualified from the adjacent interior mapped cross-section: the upstream endpoint uses the second usable path sample and the downstream endpoint uses the penultimate usable path sample. Automatic qualification requires at least three unique usable branch paths; no endpoint extrapolation is performed.
- Interfaces attached to a 0D junction remain topology-derived path queries and are linearly interpolated at the requested `Path`, including interfaces between segments of one branch.
- The calibration result records `interface_sampling` for every assembled upstream/downstream interface, including `requested_path`, `selected_path`, `inset_distance`, `usable_sample_count`, pairing status, and quality status. A vessel can be explicitly excluded from calibration with an empty `calibration.parameters.vessels.overrides` list; such exclusions are recorded in `excluded_blocks`. Under-resolved endpoints otherwise fail before solver dispatch.
- Before solver dispatch, the workflow writes `calibration_observation_qc.json` beside the requested output config. The report contains deterministic metrics and checks for vessel continuity, junction mass balance, root inflow waveform agreement, pressure-drop direction, path coverage, and sampling resolution. Any failed check prevents the solver call and leaves the output solver JSON unwritten.
- The mapped centerline result and reference centerline must have matching point counts.
- Single-snapshot calibration still emits zero `dy` observations.
- The input 0D config must not contain non-finite numeric values such as `NaN` or `inf`.
  The opt-in `input_normalization.infinite_vessel_compliance: zero` policy converts
  only positive infinity at `vessels[*].zero_d_element_values.C` to `0.0`.
  Every other non-finite value fails with its JSON path, and the source file is
  never modified. The calibration result records the normalized paths and count.
- Non-finite values returned by the solver are treated as calibration failure and no output JSON is written.
- `resistance_map_mean.vtp` and `resistance_map_systolic.vtp` are summary artifacts and are rejected as timeseries sources. Use `centerline_timeseries_last_cycle.vtp` with its sidecar.
- After observation QC, the unchanged `pysvzerod.calibrate` API is invoked twice. The confirmation payload is a copy of the first payload with only vessel/junction parameter values replaced by the first result; observations, solver controls, and per-block `calibrate` selections remain unchanged.
- A selected scalar or list parameter is fixed-point confirmed when its confirmation delta satisfies `abs(delta) <= confirmation_absolute_tolerance + confirmation_relative_tolerance * max(abs(first), abs(confirmation))`. Non-finite results, missing selected parameters, changed inactive parameters, solver exceptions, and failed confirmation deltas prevent output publication.
- The calibrator is not required to return `calibration_diagnostics`. `parameter_ratio_warning_threshold` replaces the former hard maximum ratio, and negative selected parameters—including negative resistance—are accepted when the two-pass fixed-point check succeeds. Both passes, confirmation deltas, solver provenance, negative-parameter paths, and large-ratio paths are written to `calibration_confirmation.json`.
- After fixed-point confirmation, the workflow removes calibration-only fields and normalizes single-outlet `internal_junction` blocks to `NORMAL_JUNCTION`. Multi-outlet junction values are retained and are not calibrated unless explicitly selected.
- The normalized result is structurally checked and replayed through the unchanged `pysvzerod.simulate` API before publication. Replay uses a copy configured for at least two complete cycles with all time points emitted; the requested cycle settings in the published JSON are unchanged.
- Replay requires finite, positive `pressure_bound_multiplier` and `flow_bound_multiplier` settings and a finite, non-negative `cycle_stability_tolerance`. Every pressure and flow result must be finite, remain within its configured multiple of the corresponding observation scale, and have a final-cycle normalized RMS difference below the stability tolerance.
- The solver JSON is written with an atomic replacement only after replay passes. A negative calibrated resistance is therefore accepted when it is fixed-point confirmed and replay-stable. `calibration_replay.json` records boundedness and cycle-stability metrics, and `calibration_summary.json` combines provenance, normalization, QC, confirmation, warnings, and replay diagnostics.

**BCs**
```yaml
bcs:
  type: impedance  # impedance | rcr
  compliance_model: constant
  is_pulmonary: true
  tune_space:
    free:
      - name: lpa.alpha
        init: 0.9
        lb: 0.7
        ub: 0.99
        to_native: identity
        from_native: identity
    fixed:
      - name: d_min
        value: 0.01
    tied: []
  rcr_params: [R_LPA, C_LPA, R_RPA, C_RPA]
```

Supported transforms for `tune_space`:
- `to_native`: `identity|positive|unit_interval`
- `from_native`: `identity|log`

**Trees**
```yaml
trees:
  d_min: 0.01
  use_mean: true
  specify_diameter: true
  optimized_params_csv: optimized_params.csv
  lpa:
    lrr: 10.0
    diameter: 0.3
    d_min: 0.01
    alpha: 0.9
    beta: 0.6
    inductance: 0.0
    compliance:
      model: constant  # constant | olufsen
      params:
        value: 66000.0
  rpa:
    lrr: 10.0
    diameter: 0.3
    d_min: 0.01
    alpha: 0.9
    beta: 0.6
    inductance: 0.0
    compliance:
      model: constant
      params:
        value: 66000.0
```
If `optimized_params_csv` is present, `lpa` and `rpa` blocks are optional.

**Adaptation**
```yaml
adaptation:
  method: cwss
  location: uniform
  iterations: 10
```

**Reduced-PA Adaptation Benchmark**
```yaml
adapt_benchmark:
  study_id: tst-stan-1-reduced-pa
  output_dir: benchmark-results
  workers: 1
  models: [M1, M2, M3]
  tree_params_csv: path/to/optimized_params.csv
  clinical_targets_csv: path/to/clinical_targets.csv
  parameter_overrides:
    M1:
      wss_gain: 0.01
    M3:
      k_arr: [1.0, 1.0, 1.0, 1.0]
  scenarios:
    - name: baseline
      patient_id: tst-stan-1
      scenario_group: medium_dmin0p05
      perturbation_severity: medium
      preop_rri_config: path/to/preop_simplified_zerod_tuned_RRI.json
      postop_rri_config: path/to/postop_simplified_zerod_tuned_RRI.json
      parameter_overrides:
        M1:
          t_end: 3600.0
```

Scenario fields `patient_id`, `scenario_group`, and `perturbation_severity` are
optional metadata for robustness sweeps. They are copied into
`benchmark_summary.csv` and used for grouped overlay artifacts.

`workers` is optional and defaults to `1`. Values greater than `1` run scenario
and model jobs through a local `ProcessPoolExecutor`; output rows are sorted back
to YAML scenario/model order before writing aggregate artifacts.

Dynamic benchmark parameter overrides may also include stability-screen
thresholds and tree-size guards:

```yaml
max_nodes: 20000
collapse_split_floor: 0.01
collapse_split_ceiling: 0.99
radius_max_abs_relative_change_limit: 10.0
thickness_max_abs_relative_change_limit: 10.0
```

`max_nodes` caps structured-tree construction. Benchmark rows record
`lpa_tree_nodes`, `rpa_tree_nodes`, `lpa_tree_max_nodes_reached`, and
`rpa_tree_max_nodes_reached` so capped trees remain visible in run summaries.

Benchmark outputs include:

- `benchmark_summary.csv` and `benchmark_summary.json`
- `benchmark_convergence_table.csv`
- `benchmark_failure_table.csv`
- `benchmark_final_rpa_split.png`
- `benchmark_aggregate_final_rpa_split.png`
- `benchmark_rpa_split_overlay.png` and `benchmark_lpa_split_overlay.png`
- per-patient split overlays when `patient_id` is present

**Pipeline**
```yaml
pipeline:
  run_steady: true
  optimize_bcs: true
  run_threed: true
  adapt: true
```

**3D**
```yaml
threed:
  mesh_scale_factor: 1.0
  convert_to_cm: false
  wall_model: rigid  # rigid | deformable
  elasticity_modulus: 5062674.563165
  poisson_ratio: 0.5
  shell_thickness: 0.12
  prestress_file: auto  # auto | from_steady_mean | path/to/prestress_result.vtu
  prestress_file_path: path/to/prestress_result.vtu
  tissue_support:
    enabled: true
    type: uniform  # uniform | spatial
    stiffness: 1000.0
    damping: 10000.0
    apply_along_normal_direction: true
    spatial_values_file_path: null
  execution:
    mode: slurm  # slurm | local
    executable: /path/to/svmultiphysics  # required
    submit_command: sbatch
    clean_command: clean
    slurm:
      nodes: 3
      procs_per_node: 24
      memory: 16
      hours: 20
      partition: amarsden
      qos: normal
  solver_paths:
    svpre: svpre
    svsolver: svsolver
    svpost: postsolver
```

For deformable wall runs, setting `prestress_file: auto` creates and runs a `prestress/` simulation using mean wall traction from `steady/mean` results, then uses that resulting VTU as `Prestress_file_path`.

For deformable CMM wall runs, optional `tissue_support` writes svMultiPhysics
Robin support under the wall `Type=CMM` boundary condition. `uniform` mode writes
scalar `Stiffness` and `Damping`; `spatial` mode writes `Spatial_values_file_path`
for a VTP file containing `Stiffness` and `Damping` arrays. Rigid wall runs do
not support `tissue_support`.

`threed.execution.mode: local` runs the configured executable directly as
`svmultiphysics svFSIplus.xml` in each generated simulation directory.
`threed.execution.mode: slurm` writes `run_solver.sh` and submits it with
`submit_command`, defaulting to `sbatch`.
Optional `threed.execution.slurm.mail_user` and `mail_types` can be set when
Slurm email notifications are desired; otherwise the generated script omits
mail directives.

**Postprocess**
```yaml
postprocess:
  figures:
    - kind: generation_metrics
      input: path/to/tree.pkl
      output: figures/generation_metrics.png
      options:
        time_window: [0.0, 1.0]
        exclude_collapsed: true
  analyses:
    - kind: pulmonary_resistance_map
      output: results/resistance_map
      options:
        svslicer_path: ~/Documents/Stanford/PhD/Marsden_Lab/SimVascular/svSlicer/Release/svslicer
        centerline: centerlines.vtp
        frames_csv: frames.csv
        cycle_duration_s: 1.0
        keep_intermediate_centerlines: false
```

Supported figure `kind` values:
- `generation_metrics`
- `generation_waveforms`
- `visualize_hemodynamics`

Supported analysis `kind` values:
- `pulmonary_resistance_map`
- `pulmonary_threed_suite`

`pulmonary_resistance_map` options:
- `svslicer_path`: required path to the `svslicer` executable
- `centerline`: required `centerlines.vtp`
- `frames_csv`: required CSV with columns `path,time_s`; `timestep_id` is also
  accepted and is written automatically by the 3D postprocess suite for
  downstream frame selection
- `cycle_duration_s`: required last-cycle selection window. The resistance map
  now uses the exact temporal mean over the final full cardiac period defined
  as the half-open interval `[t_end - cycle_duration_s, t_end)` so the terminal
  phase endpoint is not double-counted.
- `max_frames`: optional explicit approximation cap on last-cycle frames passed
  through `svslicer`. By default this is unset, so all frames in the final full
  cycle window are included in the mean.
- `keep_intermediate_centerlines`: optional, default `false`
- `intermediate_dir`: optional directory for per-frame mapped VTPs
- `pressure_array`: optional, default `pressure`
- `flow_array`: optional, default `velocity`
- `branch_id_array`: optional, default `BranchId`
- `path_array`: optional, default `Path`

`pulmonary_threed_suite` options:
- `simulation_dir`: required 3D simulation directory containing `svFSIplus.xml`
  and `result_*.vtu`
- `centerline`: required `centerlines.vtp`
- `svslicer_path`: required path to the `svslicer` executable
- `stage`: required `preop|postop`
- `clinical_targets`: optional CSV path or in-memory mapping for target overlays
  and flow-split comparison. Mapping inputs may use `mpa_pressure` or legacy
  `mpa_p` plus `rpa_split`.
- `cycle_duration_s`: optional cardiac-cycle duration in seconds
- `inflow_csv`: optional inflow waveform used to infer cycle duration when
  `cycle_duration_s` is omitted

If `clinical_targets` is missing or malformed, the suite still generates the
pressure plot, flow-split plot, frame manifest, and resistance-map artifacts,
but omits target overlays/comparison values and records a warning in
`postprocess_suite_metadata.json`.

The suite writes both a mean resistance-map family and a systolic
resistance-map family. Systole is defined as the frame in the final full
cardiac cycle where the simulated MPA centerline pressure reaches its maximum;
ties are broken by earliest `timestep_id`. The systolic map reuses the mapped
centerline intermediates generated for the mean map instead of remapping the
selected frame. When callers provide camera vectors to
`run_pulmonary_threed_postprocess_suite`, the resistance-map PNG renders use
that view, resolve the framing against the latest simulation-surface bounds so
it matches the ParaView viz setup, and include a low-opacity anatomy surface
extracted from the latest state VTU. Additional suite outputs include:
- `resistance_map_mean.vtp` and `resistance_map_mean.png`
- `branch_resistance_summary.csv` and `ranked_stent_candidates.csv`
- `resistance_map_metadata.json`
- `resistance_map_systolic.vtp` and `resistance_map_systolic.png`
- `branch_resistance_summary_systolic.csv` and `ranked_stent_candidates_systolic.csv`
- `resistance_map_systolic_metadata.json`

Example:
```yaml
postprocess:
  analyses:
    - kind: pulmonary_threed_suite
      output: results/postprocess
      options:
        simulation_dir: preop
        centerline: centerlines.vtp
        svslicer_path: ~/bin/svslicer
        stage: preop
        inflow_csv: inflow.csv
        clinical_targets: clinical_targets.csv
```

**CLI Usage**
```bash
svzerodtrees pipeline config.yml
svzerodtrees tune-bcs config.yml
svzerodtrees construct-trees config.yml
svzerodtrees adapt config.yml
svzerodtrees postprocess config.yml
svzerodtrees schema
```
