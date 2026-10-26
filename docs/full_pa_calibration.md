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
| `explicit` | Require a complete cap-to-BC mapping supplied in `outlet_mapping`. An `outlet_mapping_centerline` set alongside it only supplies proximal-compliance branch lengths; it never pairs caps (a later iteration reusing the previous iteration's validated mapping). |

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
- the data contract, including `pressure` units and `flow` as integrated
  `volumetric_flow` in `cm^3/s`. The pulmonary suite publishes svSlicer's
  `Pressure` field unconverted and declares it `dyn/cm^2` (the solver's CGS
  unit, which calibration passes to the solver as is); `mmHg` remains accepted.
  Target scoring converts each series by its declared unit. Descriptors from
  before 2026-10-07 declared `mmHg` for these dyn/cm^2 values; their target
  gates compare mismatched units and must be regenerated.

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
optimizer. `targets.gate_policy` (`absolute` by default) selects what must
pass: `absolute` requires both tolerances; `improvement_only` publishes when the
calibrated model's composite score is no worse than the uncalibrated model's
(both scored, replay passed) and reports the tolerance gates as advisory
(`component_gates_advisory` in `calibration_targets.json`). Omitting
`calibration.targets` retains `strict_network` behavior
during the version-1 compatibility window.

The 3D LPA and RPA target flows come from the target vessel's median
cross-section, not from its interface sample. Each target flow is the flow
trace of the centerline slice whose cycle-mean flow is the median over that
vessel's path range. The `interface` setting still selects the 0D replay column
and the solver observations. The reason: once the solution is periodic,
cycle-mean flow is the same at every slice of an unbranched vessel, but slices
next to a junction can cut through the parent lumen. On TST-STAN-1
(2026-10-20) the first 0.13 cm of the RPA and 0.33 cm of the LPA read the whole
MPA flow, so the junction-interface sample gave a 3D split of 0.50 against 0.68
from the outlets. The median is robust while fewer than half of a vessel's
slices are contaminated.

`observations_3d.{rpa,lpa}_flow` in `calibration_targets.json` records:
- `sampling`: the selected path, the candidate slice count, the slice-mean flow
  range, and the interface observation's mean (after the junction inset below)
  for comparison.
- `outlet_mass_balance`: the branch mean against the summed cycle-mean flows of
  the outlets the vessel drains. `consistent: false` beyond a 5% relative
  difference is advisory and flags a branch whose slices are still unreliable.

### Junction-interface flow observations

The same contamination biases the solver observations (`y`): a vessel's
R is fit from its pressure drop against its inlet flow, so a daughter whose
first slices read the parent flow gets an R that is too low. Every
junction-side (internal) interface therefore checks its flow sample:

- The reference is the vessel's lower-median slice cycle-mean flow over its
  path range.
- If the interface sample's cycle mean is within 5% of the reference, it is
  kept (`status: clean`).
- Otherwise the flow observation is the trace of the first slice within 5%,
  walking into the vessel from the junction (`status: inset`). The reference
  slice itself qualifies, so a slice is always found.
- The pressure observation stays at the interface. The 0D junction equates the
  pressures of the vessels it connects, and on TST-STAN-1 the pressure across
  the contaminated slices changes by about the vessel's own mean pressure drop.
  Any junction entry loss is therefore carried by the daughter's R.
- A vessel with fewer than three slices in its range cannot be checked and
  keeps the interface sample (`status: unverified_short_vessel`). These are
  mostly the 0.01 cm connectors that chain a multi-way junction into
  bifurcations. They sit inside the junction region, so they still read the
  parent flow.

`interface_sampling.<vessel>:<end>.flow_check` (also under
`observation_qc.selected_interfaces`) records the status, reference and
interface cycle-mean flows, the candidate slice count, and, for an inset, the
contaminated slice count, `flow_selected_path`, and `flow_inset_distance`.
`selected_path` stays the pressure path. `paired_observation` is `false` when
the flow comes from a different slice than the pressure.

On the 2026-10-16 runs (iteration 1, assembled 2026-10-20, no calibration):

| Run | Internal interfaces inset / clean / unverified | Median junction imbalance | RPA inlet flow (mL/s) | LPA inlet flow (mL/s) |
|---|---|---|---|---|
| TST-STAN-1 | 34 / 26 / 24 | 0.33 → 0.19 | 11.94 → 8.16 | 11.98 → 3.77 |
| TST-STAN-3 | 12 / 22 / 8 | 0.32 → 0.14 | 33.36 → 30.18 | 30.59 (clean) |
| TST-STAN-5 | 2 / 38 / 26 | 0.08 → 0.08 | clean | clean |

Junction imbalance is the relative RMS of the junction mass-balance QC metric.
The largest inset was 0.32 cm (TST-STAN-1 LPA). The implied steady R (mean
pressure drop over mean inlet flow) of the TST-STAN-1 RPA rises from 20.9 to
30.6, and that of the LPA from 61.9 to 196.5. The maximum junction imbalance
does not improve, and on TST-STAN-1 and TST-STAN-3 it grows. That residual
comes from the unverified connector chains: their outlets are now clean, but
the connector inlet still reads the parent flow. The split targets and the QC
status are unchanged.

### Vessels with a reversed 3D pressure drop

R-only calibration fits each vessel's pressure drop as R times its flow. If a
vessel's 3D cycle-mean pressure rises along its flow, the fit has to make R
negative. In that case the `pressure_drop_direction` QC reports the vessel
with `direction: wrong`. On short, low-flow vessels the slice-averaged 3D
pressure cannot resolve the viscous drop.

The evidence, from the 2026-10-16 runs:
- TST-STAN-1 iteration 2 had 22 of 57 vessels reversed, with a median drop of
  −0.02 mmHg and a worst case of −0.18 mmHg.
- All 21 of its negative calibrated resistances were among those 22, and the
  calibrated model failed its settled replay.
- Every calibration of those runs had 4 to 21 negative resistances.
- Total pressure (adding ½ρ(Q/A)²) does not explain them. It turns few
  positive (TST-STAN-5 10 → 2; TST-STAN-1 22 → 21). Most reversed vessels
  narrow downstream, and their reversal is far larger than the dynamic
  pressure.

`calibration.parameters.exclude_reversed_pressure_drop` therefore leaves
every non-target vessel with `direction: wrong` at its input parameters:
- Each such vessel's `calibrate` list is emptied, and the vessel is recorded
  under `exclusions` with the reason `reversed_3d_pressure_drop`.
- The MPA, LPA and RPA target vessels are always calibrated.
- The default (`null`) turns it on under `target_focused` enforcement and off
  otherwise. `true` or `false` overrides that.

On the same three calibrations it gave:

| Calibration | Excluded | Negative R | Replay | Composite |
|---|---|---|---|---|
| TST-STAN-1 iteration 2 | 22 | 21 → 0 | fail → pass | 15.3 → 14.3 |
| TST-STAN-9 iteration 1 | 3 | 5 → 2 | fail → pass | 25.6 → 2.4 |
| TST-STAN-2 iteration 1 | 9 | 10 → 1 | pass → pass | 21.2 → 2.3 (2.4 before) |

### Matching the 3D pulse pressure (proximal compliance multiplier)

R-only calibration cannot change the 0D pulse pressure much, because the
pulse is set by compliance. Fitting each vessel's C in the calibrator itself
is a weak alternative: each C is fit locally from the noisy difference between
inflow and outflow, never against the pulse pressure, and nothing keeps it
positive.

`calibration.parameters.proximal_compliance_scale` instead fits one multiplier
on every compliant vessel's C after the calibrator passes:

- **How it fits:** it bisects the multiplier in log space inside `bounds`
  (default `[0.2, 5.0]`). It stops when the settled replay's MPA pulse pressure
  is within `tolerance_mmhg` (default 0.05) of the 3D target observation's
  pulse pressure, or fails after `maximum_iterations` (default 40).
- **What it scales:** only vessels with C above the rigid threshold
  (`RIGID_COMPLIANCE_THRESHOLD`). A rigid model fails clearly, because nothing
  can be scaled.
- **What it leaves alone:** outlet boundary conditions (the structured-tree
  impedances) are never changed.
- **Units:** pressures are compared in CGS. The report also carries labeled
  mmHg values.
- **Gating:** the fit runs on the calibrated copy before the replay, so the
  replay, target gates and baseline-improvement policy judge the scaled model.
  If the 3D pulse is outside the 0D range over the bounds, or a replay fails,
  the fit writes its report and nothing is published.
- **Requirements:** it requires `calibration.targets` and is off by default.
  Use it where no later tuning will overwrite C, such as the 0D model that must
  match a final wall-corrected 3D run.

```yaml
calibration:
  parameters:
    vessels: {default: [R_poiseuille]}
    junctions: {default: []}
    proximal_compliance_scale:
      enabled: true
      bounds: [0.2, 5.0]
      tolerance_mmhg: 0.05
      maximum_iterations: 40
```

The result goes to `calibration_compliance_fit.json`, which records:
- the multiplier;
- the target, unscaled and fitted pulse pressures;
- the number of vessels scaled;
- the total C before and after (cm⁵/dyn and mL/mmHg);
- every evaluation.

It is also included as `compliance_fit` in `calibration_summary.json`.

## Calibration quality gates and outputs

The unchanged standard `pysvzerod.calibrate(config)` API remains the only
branch-parameter optimizer. The workflow:

1. normalizes and validates the finite input model and descriptor observations;
2. writes `calibration_observation_qc.json` and blocks on fatal QC failures;
3. invokes calibration twice, replacing only selected parameter values in the
   second payload, and checks fixed-point deltas;
4. removes calibration-only fields and structurally validates the normalized
   solver model, then, when `proximal_compliance_scale` is enabled, fits the
   compliance multiplier;
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
calibration_compliance_fit.json   # only when proximal_compliance_scale is enabled
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
