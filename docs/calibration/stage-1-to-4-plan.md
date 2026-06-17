# 0D Calibration From 3D Results Stage 1-4 Plan

Last updated: 2026-06-16

This file is an implementation plan for adding 0D calibration from 3D results
to `svZeroDTrees`, using the Levenberg-Marquardt calibrator available through
`pysvzerod.calibrate` in `svZeroDSolver` and aligning with the
`svSuperEstimator` workflow titled:

```text
Model calibration based on 3D result using Levenberg-Marquardt optimization
```

The goal is not to re-host `svSuperEstimator` wholesale inside
`svZeroDTrees`. The goal is to expose a clean, stable, file-based calibration
workflow that belongs in `svZeroDTrees` as a reusable domain interface and can
later be orchestrated by `svzt-agent`.

## Current State

`svZeroDTrees` already has:

- a strict YAML config parser and typed workflow surface
- raw 0D input path handling via `paths.zerod_config`
- output path handling via `paths.output_config`
- 3D postprocessing helpers that already accept:
  - `simulation_dir`
  - `centerline`
  - `svslicer_path`
  - `cycle_duration_s` or `inflow_csv`
- a local dependency on `pysvzerod` from the sibling `svZeroDSolver` repo

`svZeroDTrees` does not yet have:

- a first-class workflow for 0D calibration from 3D results
- config fields for calibration inputs and solver LM controls
- a stable contract for how mapped 3D centerline data becomes calibrator
  observations
- a documented path for `svzt-agent` to invoke this as part of the tuning
  pipeline

## Parameter Inventory

Parameters already represented in `svZeroDTrees` or directly reusable:

- `paths.zerod_config`
- `paths.output_config`
- `postprocess.analyses[].options.simulation_dir`
- `postprocess.analyses[].options.centerline`
- `postprocess.analyses[].options.svslicer_path`
- `postprocess.analyses[].options.cycle_duration_s`
- `postprocess.analyses[].options.inflow_csv`

Parameters that should be added for calibration:

- `calibration` section or equivalent workflow-specific section
- `mapped_centerline_result`
- `initial_damping_factor`
- `maximum_iterations`
- `tolerance_gradient`
- `tolerance_increment`
- parameter-selection policy for which 0D parameters are calibrated
- optional output artifact paths for generated mapped centerline intermediates
  and calibration summaries

Parameters from `svSuperEstimator` that should not be copied blindly:

- `project`
  - this is a SimVascular-project abstraction from `svSuperEstimator`, not a
    clean `svZeroDTrees` interface
- `set_capacitance_to_zero`
  - present in estimator examples, but not used by the least-squares task
- `calibrate_stenosis_coefficient`
  - the solver calibrator contract is centered on per-block `calibrate` lists,
    not a single global boolean
- `centerline_padding`
  - estimator least-squares code currently hardcodes padding behavior, so this
    should only be exposed in `svZeroDTrees` if we own and test its semantics

## Design Requirements

- Keep the public interface file-based and explicit.
- Reuse existing `svZeroDTrees` path resolution and validation patterns.
- Do not introduce `svzt-agent` orchestration logic into domain code.
- Prefer direct solver integration through `pysvzerod.calibrate` over taking a
  runtime dependency on `svSuperEstimator`.
- Preserve a staged contract:
  - Stage 1 proves the solver-facing calibration workflow.
  - Stage 2 automates the 3D preprocessing needed to feed Stage 1.
  - Stage 3 stabilizes parameter-selection and artifact/reporting behavior.
  - Stage 4 wires the finished workflow into the tuning pipeline and exposes a
    stable boundary for `svzt-agent`.

## Proposed Public Shape

Add a new workflow:

```text
calibrate_0d_from_3d
```

Suggested config outline:

```yaml
version: 1
workflow: calibrate_0d_from_3d

paths:
  root: .
  zerod_config: path/to/solver_0d.json
  output_config: path/to/calibrated_solver_0d.json

calibration:
  data_source:
    mode: mapped_centerline  # mapped_centerline | threed_simulation
    mapped_centerline_result: path/to/result_centerline.vtp
    centerline: path/to/centerlines.vtp
    simulation_dir: path/to/3d-simulation
    svslicer_path: path/to/svslicer
    cycle_duration_s: 1.0
    inflow_csv: path/to/inflow.csv
  parameters:
    vessels:
      default: [R_poiseuille, C, L]
    junctions:
      default: [R_poiseuille, C, L]
  solver:
    initial_damping_factor: 1.0
    maximum_iterations: 100
    tolerance_gradient: 1e-6
    tolerance_increment: 1e-10
  outputs:
    write_debug_plots: false
    mapped_centerline_output: path/to/mapped_centerline.vtp
    summary_csv: path/to/calibration_summary.csv
```

This shape is intentionally explicit. It avoids hiding the distinction between:

- a precomputed mapped centerline result
- a raw 3D simulation directory that still needs preprocessing

## Stage 1: Minimal Native Calibration Workflow

Status: planned

Goal: add the smallest stable `svZeroDTrees` workflow that can run the solver
calibrator from a precomputed mapped centerline result.

### Scope

- Add `calibrate_0d_from_3d` to the YAML workflow enum.
- Add a new typed config section for calibration.
- Require:
  - `paths.zerod_config`
  - `calibration.data_source.mode = mapped_centerline`
  - `calibration.data_source.mapped_centerline_result`
  - `calibration.data_source.centerline`
- Build solver observations:
  - `y`
  - `dy`
  - `calibration_parameters`
- Invoke `pysvzerod.calibrate`.
- Write calibrated solver JSON to `paths.output_config`.

### Stage 1 Deliverables

- config parser support
- workflow implementation in `src/svzerodtrees/api.py`
- a focused calibration module for observation assembly and solver dispatch
- unit tests for config validation
- at least one integration-style test for the workflow boundary with mocked
  mapping/calibration internals
- docs/interface updates for the new workflow

### Stage 1 Non-Goals

- no internal `svslicer` execution
- no raw 3D simulation directory handling
- no tuning-pipeline integration
- no `svzt-agent` integration
- no attempt to mirror every estimator report artifact

### Stage 1 Completion Requirements

- invalid configs fail early with actionable errors
- the workflow can consume an existing mapped centerline result and produce a
  calibrated 0D config deterministically
- the solver LM controls are configurable and documented
- the parameter-selection contract is explicit and test-covered

## Stage 2: Internal 3D-to-Centerline Preprocessing

Status: planned

Goal: let `svZeroDTrees` derive the mapped centerline calibration input from a
3D simulation directory instead of requiring callers to generate it first.

### Scope

- Extend `calibration.data_source.mode` with:

```text
threed_simulation
```

- Reuse existing 3D postprocess building blocks where practical:
  - frame selection from `cycle_duration_s` or `inflow_csv`
  - centerline mapping via `svslicer`
  - existing path-resolution rules
- Produce a stable mapped-centerline artifact that Stage 1 calibration code
  can consume unchanged.

### Stage 2 Deliverables

- preprocessing helpers that generate mapped centerline calibration inputs
- consistent artifact naming for generated mapped centerline files
- clear metadata describing:
  - source simulation directory
  - selected frames or averaging policy
  - source centerline
  - source pressure/flow field names when non-default
- integration tests around preprocessing selection and artifact contracts

### Stage 2 Non-Goals

- no tuning-pipeline wiring yet
- no `svzt-agent` orchestration yet
- no patient/workspace-specific path assumptions

### Stage 2 Completion Requirements

- callers can choose between supplying a precomputed mapped centerline result or
  a raw 3D simulation directory
- the same Stage 1 calibrator path is used once the mapped observation artifact
  exists
- generated artifacts are deterministic, documented, and machine-consumable

## Stage 3: Parameter Selection, Reporting, and Interface Hardening

Status: planned

Goal: stabilize the public calibration contract so it is safe to consume from
other workflows and from `svzt-agent`.

### Scope

- Finalize how users specify calibrated parameters.
- Prefer a contract that maps directly to solver block-level `calibrate` lists.
- Decide and document defaults for:
  - vessel parameters
  - junction parameters
  - whether stenosis terms are included
- Add structured outputs:
  - calibration summary JSON or CSV
  - effective config snapshot
  - generated mapped-centerline metadata when Stage 2 mode is used
- Add clear error behavior for:
  - missing branch mapping
  - malformed centerline arrays
  - inconsistent time resolution
  - unsupported parameter names

### Stage 3 Deliverables

- hardened parameter-selection schema
- output contract documentation
- focused tests for:
  - parameter-selection parsing
  - solver-input assembly
  - artifact presence and naming
- docs updates with copyable examples

### Stage 3 Completion Requirements

- there is one documented way to specify what gets calibrated
- output artifacts are stable enough for downstream automation
- the workflow no longer depends on estimator-only concepts like `project`
- the workflow contract is explicit enough for pipeline integration

## Stage 4: Tuning-Pipeline Integration and `svzt-agent` Boundary

Status: planned

Goal: make 0D calibration from 3D results part of the broader tuning pipeline
while keeping ownership boundaries clean between `svZeroDTrees` and
`svzt-agent`.

### Scope

- Add optional pipeline support so a run can include:
  - BC tuning
  - tree construction
  - 3D simulation
  - 3D-result-based 0D calibration
- Add a stable entrypoint that `svzt-agent` can call without filesystem poking
  or estimator-specific project conventions.
- Make calibration outputs available as structured pipeline artifacts for
  downstream steps.

### Recommended Pipeline Shape

Add a pipeline toggle such as:

```yaml
pipeline:
  run_steady: true
  optimize_bcs: true
  run_threed: true
  calibrate_0d_from_3d: true
  adapt: false
```

and a corresponding `calibration` section that is valid both for the standalone
workflow and for `pipeline`.

### `svZeroDTrees` Responsibilities in Stage 4

- validate calibration-related config
- run the domain workflow
- emit stable artifacts and machine-readable metadata
- document the supported interface

### `svzt-agent` Responsibilities in Stage 4

- decide when to schedule or skip calibration
- choose workspace-specific paths and runtime resources
- manage remote execution, staging, monitoring, and retries
- consume the structured outputs produced by `svZeroDTrees`

### Stage 4 Deliverables

- pipeline support for calibration
- structured outputs suitable for orchestration
- integration docs describing how `svzt-agent` should call the workflow
- focused tests that prove:
  - standalone calibration still works
  - pipeline-triggered calibration uses the same underlying contract
  - pipeline outputs expose the calibrated config path and summary metadata

### Stage 4 Completion Requirements

- a single `svZeroDTrees` config can express calibration as part of the tuning
  pipeline
- `svzt-agent` can invoke that interface using stable, documented fields
- no Slurm, SSH, rsync, or workspace policy logic is introduced into
  `svZeroDTrees`

## Recommended Implementation Order

1. Stage 1
2. Stage 2
3. Stage 3
4. Stage 4

Do not start with Stage 4. Pipeline and orchestration integration should sit on
top of a stable standalone calibration contract, not define it.

## Agenting Recommendation

Use staged delivery, not a single large implementation pass.

Recommended execution model:

- Stage 1: single agent
- Stage 2: single agent or tightly coordinated pair if the preprocessing reuse
  is deeper than expected
- Stage 3: single agent
- Stage 4: staged multi-repo effort across `svZeroDTrees` and `svzt-agent`

The main reason to avoid a one-shot implementation is that the clean public
contract is not finished yet. Stage 1 and Stage 2 should establish the domain
API first. Stage 4 should only begin once that API is explicit, documented, and
test-covered.
