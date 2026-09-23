# Full-PA tuning and calibration

This guide is the authoritative scientific-workflow contract for an explicitly
selected `full_pa` pulmonary 0D model. It covers the public YAML/CLI/Python
path, deterministic cap mapping, calibration-ready postprocess artifacts, and
3D-to-0D calibration. It does not define SSH, Slurm, staging, retries, or
manifest state. Those are orchestration concerns documented by
[`svzt-agent`](https://github.com/ncdorn/svzt-agent/blob/main/docs/EXECUTION.md).

## What the public path does

There is one domain service for full-PA tuning:
`run_impedance_tuning_for_iteration`. The public `tune_bcs` workflow and the
full-PA branch of `pipeline` adapt their validated YAML into this service;
they do not maintain a second reduced-RRI implementation. Programmatic
callers can use the same service directly:

```python
from svzerodtrees.tuning.iteration import run_impedance_tuning_for_iteration

result = run_impedance_tuning_for_iteration(
    iteration_dir="iteration-01",
    seed_config="input/full_pa_seed.json",
    mesh_surfaces="input/mesh-complete/mesh-surfaces",
    clinical_targets="input/clinical_targets.csv",
    inflow_path="input/inflow.csv",
    impedance_config={
        "tuning_model": "full_pa",
        "outlet_mapping_mode": "auto",
        # The centerline the seed was generated from; enables geometric mapping.
        "outlet_mapping_centerline": "input/centerlines.vtp",
        "tune_space": {
            "free": [
                {"name": "lpa.alpha", "init": 0.9, "lb": 0.7, "ub": 0.99},
                {"name": "rpa.alpha", "init": 0.9, "lb": 0.7, "ub": 0.99},
            ],
            "fixed": [{"name": "d_min", "value": 0.01}],
            "tied": [],
        },
    },
)
```

The returned result contains the canonical service artifacts:
`optimized_params_csv`, `stree_optimization_log`, `pa_config_snapshot`,
`tuned_zerod_config`, and, for `full_pa`, `outlet_cap_mapping`. The stable
filenames are `optimized_params.csv`,
`pa_config_tuning_snapshot.json`, `svzerod_3d_coupling_tuned.json`, and
`outlet_cap_mapping.json` in the result directory.

The equivalent CLI entrypoints are:

```bash
svzerodtrees tune-bcs path/to/full_pa.yml
svzerodtrees pipeline path/to/full_pa_pipeline.yml
svzerodtrees schema
```

`version: 1` is still the top-level YAML version. New full-PA and calibration
fields are additive within that version.

For a file-driven Python caller, `svzerodtrees.api.run_from_config_file` uses
the same loader and workflow dispatch as the CLI. The calibration API is
`svzerodtrees.calibration.calibrate_0d_from_mapped_centerline`; the
`postprocess_suite` adapter resolves its descriptor before entering that
existing calibrator. The supported postprocess producer and descriptor
validator are exported from `svzerodtrees.post_processing`.

## Full-PA seed and mapping contract

Select the model class explicitly with `bcs.impedance.tuning_model: full_pa`.
The seed must be the full pulmonary 0D JSON: it must contain vessel topology,
more than two non-inflow outlet boundary conditions, and one mapped pulmonary
cap for every outlet. A reduced two-outlet seed is rejected; it is never
silently upgraded or replaced with a reduced RRI model. The full topology,
vessel names, junctions, outlet attachments, and non-tuned fields are
preserved through tuning and snapshot validation.

The canonical nested configuration is:

```yaml
bcs:
  type: impedance
  is_pulmonary: true
  impedance:
    tuning_model: full_pa
    outlet_mapping_mode: auto
    # The centerline the seed was generated from (auto or centerline modes).
    # Defaults to seed_generation.centerline for a generated seed.
    outlet_mapping_centerline: input/centerlines.vtp
    # Include this only for outlet_mapping_mode: explicit.
    # outlet_mapping:
    #   lpa_cap_01: OUTLET_07
    #   rpa_cap_01: OUTLET_12
    use_mean: false
    diameter_scale: 1.0
    diameter_std_cap: null
    tune_space:
      free:
        - name: lpa.alpha
          init: 0.9
          lb: 0.7
          ub: 0.99
      fixed: []
      tied: []
```

`outlet_mapping_mode` accepts:

| Mode | Contract |
| --- | --- |
| `auto` | Try complete persisted tree metadata, then normalized cap-name matching, then `centerline` when `outlet_mapping_centerline` is set. It does not select serialized order implicitly. |
| `metadata` | Require a complete mapping retained in serialized tree metadata. |
| `cap_name` | Match normalized cap stems/names to normalized outlet BC names. |
| `centerline` | Pair each cap with an outlet BC through the centerline geometry (see below). Requires `outlet_mapping_centerline`. |
| `serialized_cap_order` | Pair canonical pulmonary cap order with the original serialized non-inflow BC order. This is an explicit choice. |
| `explicit` | Require a complete cap-to-BC mapping supplied in `outlet_mapping`. |

Canonical cap order is the existing pulmonary VTP filename order. Canonical BC
order is the original `boundary_conditions` JSON list after FLOW/inflow
entries are removed. Normalized names are used only for matching; original
paths and names remain in provenance. An explicit map may be a YAML mapping
from cap path/stem to BC name. It must be one-to-one, cover every cap and
every non-inflow outlet, and contain no unknown or duplicate entries.

### Geometric (`centerline`) mapping

A centerline-generated 0D model names its vessels `branch<N>_seg<K>`
(learnedZeroD appends suffixes such as `_connectorEL`), where `N` is the
centerline `BranchId`. The `centerline` strategy computes each cap's
area-weighted centroid, pairs it with the nearest centerline outlet endpoint,
and takes the outlet BC of the 0D vessel on that endpoint's branch. This is
the physically based mapping for centerline-generated seeds, whose BCs
(`RESISTANCE_<n>`) are numbered in branch order, unrelated to cap filename
order. Serialized order must not be used for these seeds.

The strategy fails with a `ValueError` naming every violated check, and never
falls back to another strategy, unless all of the following hold:

- The 0D branch set equals the centerline branch set.
- The seed is traceable to this centerline. When outlet vessels record
  `centerline_node_ids` (learnedZeroD seeds), each must end on its branch's
  terminal centerline node. Otherwise each 0D branch length must match the
  centerline branch length within 2%, after scaling the centerline by 0.1
  when `convert_to_cm` declares mm geometry.
- Each cap centroid lies within 1.0 cap radius of its matched endpoint, and
  the runner-up endpoint is at least 1.0 cap radius farther away. The cap
  radius is `sqrt(area / pi)` of the raw cap geometry.
- No two caps match the same endpoint.

`outlet_mapping_centerline` must be the centerline the seed was generated
from, in the same frame as the mesh caps. A centerline or mesh from another
model revision is rejected by these checks. Paths in YAML resolve against
`paths.root`. With `seed_generation`, the generated seed's own
`seed_generation.centerline` is used when `outlet_mapping_centerline` is
omitted and the mode is `auto` or `centerline`. The mapping provenance
records the centerline path and SHA-256, the check that tied the seed to the
centerline, the thresholds, and per-cap evidence (branch, vessel, centroid,
endpoint, offset, and margin in cap radii).

`ConfigHandler` does not retain `centerline_node_ids`, so the full-PA
service passes the serialized seed payload to the resolver. Direct callers
of `resolve_outlet_cap_mapping` or `validate_cap_to_bc_mapping` with a
learnedZeroD seed must pass `seed_payload`.

The cap determines the physical side: a cap identity must identify exactly one
of LPA or RPA. 0D graph labels are retained as an optional diagnostic and
never override the cap-derived side. One immutable `ResolvedOutletCapMapping`
is resolved before the tuner is created and reused for every optimizer
candidate, the snapshot, final tree construction, and publication.

For `full_pa`, omitted tree controls resolve to `use_mean: false` and
`diameter_scale: 1.0`. A per-cap construction diameter is computed from the
cap area as `2 * sqrt(area / pi)`, then optionally clipped by
`diameter_std_cap` and damped toward the LPA/RPA side mean by
`diameter_scale`. `use_mean: true` and `diameter_scale: 0.0` are explicit
compatibility controls for shared-by-side or fully mean-diameter studies.
`convert_to_cm` retains its existing geometry conversion meaning; solver
values remain on the established CGS contract.

### Objective tree policy

The controls above are the *final* policy used to publish
`tuned_zerod_config` (and the 3D BCs). By default every optimizer evaluation
also builds trees with that policy, so a per-outlet final policy rebuilds and
recomputes the impedance of every outlet tree at every Nelder-Mead step.
`objective_tree_policy` decouples the two:

```yaml
    use_mean: false                 # final: one tree per cap
    diameter_scale: 1.0
    objective_tree_policy:
      use_mean: true                # objective: one shared tree per side
      reference_diameter: conductance_matched
```

Parameters tuned with shared trees do not reproduce the tuned targets when
rebuilt per outlet at the arithmetic-mean diameter: tree conductance grows
super-linearly with root diameter, so `N * G(mean d) < sum_i G(d_i)` and the
objective model is too resistive (for caps spanning 0.08-0.35 cm the shared
trees carry about 67% of the per-outlet conductance). `conductance_matched`
builds the shared tree at `d_ref` solving `N * G(d_ref) = sum_i G(d_i)` per
side, where `d_i` are the per-outlet diameters defined by the policy's
`diameter_scale`/`diameter_std_cap` (inherited from the final policy). `G` is
the exact steady Poiseuille conductance of the structured tree
(`structured_tree_dc_resistance`, which reproduces
`StructuredTree.equivalent_resistance()` without building the tree). It
depends on `alpha`, `beta`, and `d_min` but not on `lrr` or viscosity, so
`d_ref` is recomputed cheaply for each candidate.

Limits of the correction:

- It matches only the DC (mean-flow) conductance of each side. Pulsatile
  impedance, compliance, and the flow distribution among outlets within a side
  still differ from the per-outlet trees.
- `G(d)` jumps when a generation crosses `d_min`; if a jump skips the target,
  the closest diameter is used and `relative_conductance_residual` reports the
  mismatch.
- It assumes untruncated trees and warns when a tree would exceed the
  `StructuredTree.build` `max_nodes` default.

`outlet_cap_mapping.json` records the final policy in `tree_options` and the
optimizer policy in `objective_tree_options` (`source` is `final_policy` or
`objective_tree_policy`). For `conductance_matched` it also records the
optimum `reference_diameters` per side. Because only DC conductance is matched,
it is still worth checking the final per-outlet `tuned_zerod_config` against
the clinical targets.

The published `outlet_cap_mapping.json` is version 1 and is intentionally
ordered. Each pair records the cap path/stem, cap-derived side, BC name and
indices, cap area, raw and scaled diameters, and any graph-side disagreement.
Its provenance records the mapping strategy, requested mode, geometry
conversion, tree options, and, for `centerline`, the geometric evidence. This artifact is the replay/audit record for
the mapping used by both optimization and final construction.

### Legacy input

`allow_ordered_outlet_mapping` is a deprecated input adapter. During the
compatibility window, configuration loading translates
`allow_ordered_outlet_mapping: true` to
`outlet_mapping_mode: serialized_cap_order` and emits a deprecation warning.
It may not be combined with a new mapping mode. The canonical validator and
iteration service use only `outlet_mapping_mode` and `outlet_mapping`.
Existing RRI tuning retains its reduced-seed expansion and historical
`use_mean: true`/`diameter_scale: 0.0` defaults; full-PA controls are not
inferred for RRI.

## Calibration-ready postprocess artifact

The pulmonary postprocess suite selects one last-cycle frame set. That same
set supplies resistance maps, timestamps, and the calibration centerline
series. Before intermediate mapped frames are removed,
`svZeroDTrees.post_processing.pulmonary_threed_suite` calls the supported
`publish_centerline_timeseries` producer.

The producer atomically publishes these stable files:

```text
postprocess_suite_metadata.json
centerline_timeseries_last_cycle.vtp
centerline_timeseries_last_cycle_metadata.json
```

The suite descriptor has an independently versioned `schema_version: "1.0"`
and an `artifacts.centerline_timeseries` record. The record contains:

- descriptor-relative `vtp`, `metadata`, and `reference_centerline` paths;
- SHA-256 digests for all three files (including the `digests` aliases);
- contiguous `frame_indices`, ordered `timestamps_s`, positive
  `cycle_duration_s`, and frame/point/cell counts;
- source pressure/flow array names; and
- the data contract, including `pressure` as `mmHg` and `flow` as integrated
  `volumetric_flow` in `cm^3/s`.

Callers producing a calibratable artifact must pass the exact tuned full-0D
JSON through `tuned_zerod_config_path`. The suite descriptor then carries one
top-level `lineage` object with the descriptor-relative
`tuned_zerod_config_path` and its SHA-256 digest:

```json
{
  "lineage": {
    "tuned_zerod_config_path": "../tuned/svzerod_3d_coupling_tuned.json",
    "tuned_zerod_config_sha256": "<sha256 of the exact input bytes>"
  }
}
```

The pulmonary suite writes `status: "completed"` only after the centerline,
mean resistance-map, and systolic resistance-map steps all succeed. A
descriptor left by a failed suite is diagnostic only and must not be used for
`postprocess_suite` calibration.

The VTP contains scalar point-data arrays `pressure_0`, `flow_0`, through the
last selected frame. Every frame has identical point coordinates and cell
connectivity to the reference centerline. The sidecar repeats the identity,
frame, geometry, source-array, and data-contract fields and records each
processed frame. Source flow is the svSlicer velocity-dot-normal integral;
it is already volumetric flow and must not be multiplied by area again.

Use `validate_centerline_timeseries_descriptor` as the descriptor boundary:

```python
from svzerodtrees.post_processing import validate_centerline_timeseries_descriptor

validated = validate_centerline_timeseries_descriptor(
    "postprocess/postprocess_suite_metadata.json"
)
print(validated["artifact"]["frame_indices"])
```

Validation resolves only the declared relative paths from the descriptor's
directory. It checks schema/kind, file existence, all declared digests,
sidecar agreement, frame order, timestamps, units, array counts and
finiteness, and VTP/reference geometry. Absolute paths, guessed filenames,
stale digests, `resistance_map_mean.vtp`, and non-monotone or incomplete
series are not accepted as substitutes.

A descriptor consumed by calibration must also declare the tuned full-0D
input lineage (an explicit path and/or digest under the descriptor's lineage,
provenance, source, or input metadata). Calibration compares that identity to
`paths.zerod_config`; a descriptor from another iteration fails before solver
dispatch. The descriptor is therefore portable across a staging boundary
without relying on its filename or the current working directory.

## Calibration data sources

The calibration workflow is available from YAML, the CLI, and the Python API:

```bash
svzerodtrees calibrate-0d-from-3d path/to/calibration.yml
```

Use `calibration.data_source.mode: postprocess_suite` for the canonical
descriptor handoff. It requires exactly one
`postprocess_metadata_json` value. The descriptor supplies the VTP, sidecar,
reference centerline, array names, units, timestamps, geometry, and digests;
all paths are resolved relative to the descriptor. Do not provide manual
mapped-centerline fields in the same request. Mixed fields fail fast instead
of choosing a precedence or falling back to a guessed path.

```yaml
version: 1
workflow: calibrate_0d_from_3d
paths:
  root: .
  zerod_config: results/svzerod_3d_coupling_tuned.json
  output_config: results/calibrated_full_pa_zerod.json

calibration:
  data_source:
    mode: postprocess_suite
    postprocess_metadata_json: postprocess/postprocess_suite_metadata.json
  parameters:
    vessels:
      default: [R_poiseuille]
      overrides: {}
    junctions:
      default: []
      overrides: {}
  observation_qc:
    enforcement: target_focused
  targets:
    mpa_pressure:
      vessel: branch0_seg0
      interface: external_upstream
      weight: 1.0
      normalized_rms_tolerance: 0.05
    rpa_flow_split:
      rpa_vessel: branch1_seg0
      lpa_vessel: branch2_seg0
      interface: external_downstream
      weight: 1.0
      absolute_tolerance: 0.02
    require_improvement_over_baseline: true
```

The established `mapped_centerline` mode remains supported. It requires
explicit `mapped_centerline_result` and `centerline` paths and an explicit
`flow_observation_type`. Numbered arrays require `metadata_json`. A true
velocity field uses `flow_observation_type: velocity` and an `area_array`;
svSlicer-integrated `flow_0..N` (and legacy `velocity_0..N` when explicitly
declared as `flow`) is already volumetric and is not area-scaled. Both modes
converge on the same observation assembly and calibration implementation.

`target_focused` is the production profile: configure distinct MPA, LPA, and
RPA vessel roles and explicit, topology-valid interfaces. Target-focused
calibration independently gates the MPA pressure-waveform normalized RMS
error and absolute RPA flow-split error. Their weighted,
tolerance-normalized composite is a post-calibration score, not a second
optimizer. Omitting `calibration.targets` retains `strict_network` behavior
during the version-1 compatibility window.

## Calibration quality gates and outputs

The unchanged standard `pysvzerod.calibrate(config)` API remains the only
branch-parameter optimizer. The workflow:

1. normalizes and validates the finite input model and descriptor observations;
2. writes `calibration_observation_qc.json` and blocks on fatal QC failures;
3. invokes calibration twice, replacing only selected parameter values in the
   second payload, and checks fixed-point deltas;
4. removes calibration-only fields and structurally validates the normalized
   solver model;
5. replays it through `pysvzerod.simulate` over a bounded settling horizon,
   checking finite, bounded pressure/flow and cycle stability;
6. evaluates configured MPA/RPA target gates and baseline-improvement policy;
   and
7. atomically publishes the solver JSON last.

The stable report set beside `paths.output_config` is:

```text
calibration_observation_qc.json
calibration_confirmation.json
calibration_replay.json
calibration_targets.json
calibration_summary.json
```

Reports and the returned result share a `run_id` and content digests for the
normalized input, observations, solver implementation, and published output.
Negative selected parameters and large parameter ratios are warnings when all
fixed-point, replay, target, and publication gates pass; they are not
standalone rejection rules. The calibrated solver JSON is not written when
input validation, descriptor lineage, observation QC, fixed-point
confirmation, replay, or target gates fail. Diagnostic reports remain
available to explain the failure.

The full-loop trapezoidal integration boundary is centralized in
`svzerodtrees.numerics.trapezoid`, selecting the available NumPy API without
changing the declared NumPy dependency range.

## Migration and rollback

The compatibility plan is additive for one coordinated release window:

- Keep top-level YAML `version: 1`.
- Prefer nested `bcs.impedance` over deprecated flat BCS impedance fields.
- Prefer `outlet_mapping_mode`/`outlet_mapping` over
  `allow_ordered_outlet_mapping`.
- Keep `mapped_centerline` available while adopting `postprocess_suite`.
- Make the full model and mapping policy explicit in every new full-PA case.
- Record and retain `outlet_cap_mapping.json` and suite descriptors with their
  input digests.

To reproduce a historical full-PA construction, set the explicit compatibility
controls (`use_mean: true` and/or `diameter_scale: 0.0`) and select the
historical mapping mode. A failed or intentionally disabled calibration does
not invalidate the tuned 0D or 3D artifacts; do not promote its output or
rewrite prior scientific artifacts. Seed selection, disabling automatic
calibration, and any legacy `legacy_rri_after_first` policy belong to
`svzt-agent`, whose execution and promotion contract is documented in its
[`EXECUTION.md`](https://github.com/ncdorn/svzt-agent/blob/main/docs/EXECUTION.md)
and operator guide.

When every calibration gate passes, the published calibrated model (for
example, `calibrated_full_pa_zerod.json`) is a lineage-valid candidate for the
next `full_pa` seed. Choosing that candidate and promoting it between
iterations remains an orchestration decision; the scientific workflow does not
silently replace the original seed.

## Ownership boundary

`svZeroDTrees` owns cap identity and tree construction, scientific 3D
postprocessing and VTP transformation, descriptor validation, observation QC,
calibration, replay, target evaluation, and calibrated-model publication.
`svzt-agent` may stage and record these artifacts and invoke the public
entrypoints, but it must not parse VTP, stack centerlines, infer mappings,
compute calibrated parameters, or duplicate scientific validation. Reviews of
mapping, units, numerical behavior, and artifact schemas start in this
repository; reviews of scheduling, staging, manifest state, resume, and seed
promotion start in `svzt-agent`.
