# Adaptation Stage 2-5 Plan

Last updated: 2026-06-09

This file is an agent handoff plan for moving from the completed Stage 1 toy benchmark to the PA adaptation harness and then to full patient-specific `svZeroDTrees` and `svzt-agent` integration.

## Current State

Stage 1 is complete in the corrected toy topology:

- fixed upstream branch segment
- adaptive downstream BC segment
- fixed upstream perturbation held constant during integration
- downstream BC radii adapt for CWSS
- downstream BC radii and thickness adapt for CWSS-IMS

Stage 1 runner:

```text
examples/run_two_branch_terminal_resistance_cwss_comparison.py
```

Stage 1 outputs:

```text
examples/output/two-branch-terminal-resistance-cwss-comparison/
```

Important Stage 1 summary:

```text
examples/output/two-branch-terminal-resistance-cwss-comparison/comparison_summary.csv
```

Experiment notes:

```text
ADAPTATION_RUN_LOG.md
```

Use Stage 1 to preserve model intuition, not as a substitute for the PA harness.

Stage 2 is complete for the TST-STAN-1 reduced PA harness:

- The Stage 1 case set has been carried into an `M3`-only PA benchmark spec.
- The TST-STAN-1 preop source has been corrected to the Sherlock iter-04 tuned RRI model.
- Base adaptation gain magnitude has been raised to `1e-2` for the PA benchmark cases.
- Structured study outputs now include gain metadata, terminal/load metadata, convergence status, event time, final split, and radius/thickness relative-change summaries.
- Study-level plots now include RPA and LPA split overlays, final RPA split, and convergence/stability summaries.
- The final Stage 2 deep-tree run used `d_min = 0.01`.
- IMS thickness-gain multipliers `50x` and `100x` have been added as forward-looking cases for future models.

Stage 2 final TST-STAN-1 `d_min = 0.01` outputs:

```text
examples/output/stage2-tst-stan-1-pa-m3-cases-base1e-2-sherlock-preop-dmin0p01/
examples/output/stage2-tst-stan-1-pa-m3-cases-base1e-2-sherlock-preop-dmin0p01-thickness-addon/
```

## Benchmark Cases

Carry these cases forward unless a stage explicitly narrows scope:

```text
no_adaptation_fixed_bc
cwss_only_no_terminal_resistance
cwss_only_terminal_R_1e03
cwss_only_terminal_R_1e04
cwss_only_terminal_R_1e05
cwss_only_terminal_R_ref_adaptive
cwss_ims_equal_gains
cwss_ims_thickness_gains_3x
cwss_ims_thickness_gains_10x
cwss_ims_thickness_gains_30x
cwss_ims_thickness_gains_50x
cwss_ims_thickness_gains_100x
cwss_ims_radius_gains_0p1x_thickness_gains_1x
```

Naming requirements:

- Use `cwss_ims_thickness_gains_10x`, not `matlab_scaled`, in new specs and reports.
- Include `cwss_ims_thickness_gains_50x` and `cwss_ims_thickness_gains_100x` in future model comparisons unless a stage explicitly narrows scope.
- Keep `cwss_ims_equal_gains` for all four gains equal.
- Keep `no_adaptation_fixed_bc` as the hemodynamic control.
- Keep `cwss_only_terminal_R_ref_adaptive` as the mechanically anchored resistance control when an explicit patient-derived load is not yet available.

## Key Requirements Learned From Stage 1

- Stability depends on topology. The intended toy topology is fixed upstream perturbation plus adaptive downstream BC state.
- Stability depends on perturbation magnitude.
- Stability depends on downstream/terminal load.
- CWSS-only can be bounded for small upstream perturbations, but is not robust to large upstream perturbations without enough terminal resistance.
- CWSS-IMS with equal gains can be bounded for small perturbations, but is not robust to large upstream perturbations.
- CWSS-IMS with faster thickness gains improves large-perturbation robustness.
- Larger negative local eigenvalues are helpful but do not fully explain nonlinear finite-amplitude stability.
- Restoring downstream WSS or IMS does not necessarily restore 50/50 flow split when fixed upstream geometry remains asymmetric.

## Stage 2: PA Adaptation Harness

Status: complete for the TST-STAN-1 `M3` reduced PA benchmark.

Goal: reproduce the Stage 1 benchmark ideas in the reduced PA `adapt-benchmark` harness.

Primary repo:

```text
svZeroDTrees
```

Primary workflow:

```text
PYTHONPATH=src python -m svzerodtrees.cli adapt-benchmark <spec.yml>
```

Recommended initial target:

```text
examples/adapt_benchmark_tst_stan_1_local_m3.yml
```

or a new spec based on it:

```text
examples/adapt_benchmark_stage2_tst_stan_1.yml
```

### Stage 2 Implementation Tasks

1. Inspect current M3/CWSS-IMS model capabilities:
   - `src/svzerodtrees/adaptation/models/cwss.py`
   - `src/svzerodtrees/adaptation/models/cwss_ims.py`
   - `src/svzerodtrees/adaptation/benchmark.py`
   - `src/svzerodtrees/adaptation/workflow.py`

2. Identify how to represent the Stage 1 benchmark cases in the PA harness:
   - CWSS-only should map to radius/WSS-only adaptation.
   - CWSS terminal resistance cases should map to a clear downstream load parameter, not an undocumented hack.
   - CWSS-IMS equal gains should map to equal active gain channels.
   - `cwss_ims_thickness_gains_10x` should map to thickness gain channels 10x radius gain channels.

3. Add explicit benchmark scenario naming:
   - Use one scenario per benchmark condition if the current harness only supports one parameter set per scenario.
   - Alternatively extend the harness to support named case sets cleanly.
   - Do not overload existing scenario names in a way that makes output ambiguous.

4. Add or extend structured outputs:
   - final split
   - split history
   - termination reason
   - convergence flag
   - event time when present
   - final radius/adaptation state summary
   - final thickness summary for CWSS-IMS
   - terminal resistance/load metadata
   - gain metadata

5. Generate comparison plots:
   - all left/RPA split curves overlaid
   - all right/LPA split curves overlaid when available
   - final split bar or scatter by case
   - convergence/stability summary plot

### Stage 2 Completion Requirements

Stage 2 is complete. The requirements below have been met for the TST-STAN-1 reduced PA `M3` harness:

- A YAML spec exists for the Stage 2 PA benchmark.
- Running the spec locally produces deterministic output directories.
- `benchmark_summary.csv` includes the Stage 1 carry-forward cases and the documented `50x`/`100x` IMS thickness-gain add-on.
- Every row includes case name, gains, terminal/load setting, final split, termination reason, and convergence flag.
- Overlay plots exist for split histories.
- Results are summarized in `ADAPTATION_RUN_LOG.md`.
- Focused tests cover the new parser, config, and output-contract behavior added for the harness.
- No Sherlock, Slurm, SSH, or `svzt-agent` orchestration logic was added to `svZeroDTrees` for this stage.

Stage 2 final reference result:

- TST-STAN-1 `d_min = 0.01` tree size: 2,401 LPA vessels, 8,045 RPA vessels, 10,446 total tree vessels, and 20,892 `M3` state variables.
- Preop RPA split: `0.662969`.
- Fixed post-op RPA split: `0.510737`.
- Best adaptive `d_min = 0.01` case tested so far: `cwss_ims_thickness_gains_100x`, final RPA split `0.329348`.
- Interpretation: higher IMS thickness gains improve the final split relative to lower-gain IMS cases, but the `100x` case still undershoots the fixed post-op split and should be treated as a candidate for future-model testing, not a final accepted patient-specific model.

## Stage 3: PA Harness Robustness Sweep

Status: complete for the local TST-STAN-1/TST-STAN-9 patient-specific Stage 3 pilot.

Goal: determine which cases generalize across perturbation severities, patients, and load magnitudes.

Stage 3 should still run locally through `adapt-benchmark`.

Current Stage 3 pilot entrypoint:

```text
examples/adapt_benchmark_stage3_tst_stan_1_9_robustness.yml
```

Run from `svZeroDTrees/`:

```bash
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python -m svzerodtrees.cli adapt-benchmark examples/adapt_benchmark_stage3_tst_stan_1_9_robustness.yml
```

This pilot spec covers one selected patient-specific preop/postop simplified
nonlinear RRI pair per patient:

```text
TST-STAN-1 selected_iter04
TST-STAN-9 selected_iter03
```

The completed output directory is:

```text
examples/output/stage3-tst-stan-1-9-pa-m3-patient-specific-base1e-2-maxnodes20000/
```

Each patient-specific pair is tested across the carried-forward 13-case `M3`
set. The TST-STAN-9 tree construction is capped at `max_nodes = 20000`; cap hits
are recorded in the summary rather than treated as pre-adaptation errors.

Tree-size summary columns:

```text
lpa_tree_nodes
rpa_tree_nodes
lpa_tree_max_nodes_reached
rpa_tree_max_nodes_reached
```

The harness now supports optional scenario metadata fields:

```text
patient_id
scenario_group
perturbation_severity
```

The harness also supports `adapt_benchmark.workers` for parallel local runs and
guarded dynamic parameter overrides for Stage 3 screening:

```text
max_nodes
collapse_split_floor
collapse_split_ceiling
radius_max_abs_relative_change_limit
thickness_max_abs_relative_change_limit
```

Stage 3 outputs added to the benchmark runner:

```text
benchmark_convergence_table.csv
benchmark_failure_table.csv
benchmark_aggregate_final_rpa_split.png
benchmark_<patient_id>_rpa_split_overlay.png
benchmark_<patient_id>_lpa_split_overlay.png
```

`benchmark_summary.csv` now includes explicit screening fields:

```text
one_branch_collapse
radius_bounds_violation
thickness_bounds_violation
nonfinite_state_detected
nonphysical_terminal_load
stability_screen_failed
```

### Stage 3 Implementation Tasks

1. Expand the Stage 2 spec into a sweep:
   - at least TST-STAN-1
   - add other available patient configs once the Stage 2 contract is stable
   - include low, medium, and high adaptation difficulty or perturbation severity where the harness can express it

2. Keep benchmark cases consistent:
   - do not rename cases between patients
   - do not change gain values silently
   - include the `50x` and `100x` IMS thickness-gain cases from Stage 2 as standard forward tests
   - record any patient-specific terminal resistance estimates explicitly

3. Add postprocessing:
   - enriched CSV with one row per patient/scenario/case
   - all-case overlay per patient
   - aggregate final split plot
   - aggregate convergence table
   - failure table with solver status and termination reason

4. Add stability screening:
   - detect one-branch collapse
   - detect radius or thickness bounds violations
   - detect NaN/inf states
   - detect nonphysical terminal load settings

### Stage 3 Completion Requirements

Stage 3 is complete for the TST-STAN-1/TST-STAN-9 patient-specific pilot. The
completed run met the requirements below for the pilot scope:

- The sweep can be rerun from one YAML spec or one documented command sequence.
- At least one multi-case, multi-scenario summary CSV exists.
- Plots make it possible to compare the same benchmark case across scenarios.
- Collapse and nonphysical-state detection is explicit in outputs.
- `ADAPTATION_RUN_LOG.md` records which cases are robust, marginal, or rejected.
- The recommended shortlist for full patient-specific integration is stated clearly.

Stage 3 pilot reference result:

- Rows: 26 total, with 13 TST-STAN-1 rows and 13 TST-STAN-9 rows.
- Status: all 26 rows ran successfully.
- Stability-screen pass count: 14 total.
- TST-STAN-1 selected iter-04: only `no_adaptation_fixed_bc` passed the current
  screen. Higher IMS thickness-gain cases moved final RPA split toward the fixed
  post-op split, but failed the thickness-bound screen.
- TST-STAN-9 selected iter-03: all 13 rows passed the current screen, but both
  trees reached `max_nodes = 20000` for every case.
- No adaptive `M3` case currently passes the screen for both patients.

## Stage 4: Full Patient-Specific svZeroDTrees Integration

Goal: promote the successful benchmark cases into the full patient-specific adaptation workflow.

Primary repo:

```text
svZeroDTrees
```

Stage 4 is where model behavior becomes a reusable domain interface. Keep this separate from orchestration.

### Stage 4 Implementation Tasks

1. Define patient-specific inputs:
   - preop reduced config
   - postop reduced config
   - clinical targets
   - optimized tree parameters
   - terminal/load estimates
   - selected adaptation case name
   - selected gain vector

2. Add stable public configuration fields:
   - adaptation formulation
   - gain vector
   - terminal/load policy
   - radius bounds
   - thickness bounds
   - convergence settings
   - output directory

3. Make terminal resistance patient-aware:
   - prefer derived or measured load estimates over toy round numbers
   - still allow explicit override for controlled experiments
   - record the effective values in outputs

4. Harden model safety:
   - validate positive radii and thickness
   - validate nonnegative terminal resistance/load values
   - validate finite state vectors
   - fail with clear errors on invalid configuration

5. Standardize outputs:
   - adaptation summary JSON
   - metrics JSON
   - split history CSV
   - state history CSV
   - adaptation histories PNG
   - final state table
   - exact effective config snapshot

6. Update tests and docs:
   - focused unit tests for validation and output schema
   - integration test with a small patient config
   - docs/interface updates if config schema changes
   - README or example updates for the selected adaptation cases

### Stage 4 Completion Requirements

Stage 4 is complete only when:

- A patient-specific adaptation run can be launched through a documented `svZeroDTrees` API or CLI path.
- The selected benchmark cases from Stage 3 are supported without custom one-off scripts.
- Effective gains and terminal/load values are written to output.
- Invalid configs fail early with actionable errors.
- Outputs are deterministic enough for downstream automation.
- Tests cover at least one successful patient-specific run and key validation failures.
- Documentation describes the new config fields and output artifacts.

## Stage 5: svzt-agent Orchestration Integration

Goal: expose the stable `svZeroDTrees` patient-specific adaptation interface through `svzt-agent` planning, manifests, monitoring, and result collection.

Primary repo:

```text
svzt-agent
```

Do not implement orchestration inside `svZeroDTrees`.

### Stage 5 Implementation Tasks

1. Inspect `svzt-agent` adaptation workflow ownership:
   - `svzt-agent/src/svztagent/workflows/adapt.py`
   - `svzt-agent/src/svztagent/campaigns/adapt_benchmark.py`
   - `svzt-agent/src/svztagent/core/plan.py`
   - `svzt-agent/src/svztagent/core/manifest.py`

2. Add manifest fields for:
   - adaptation case name
   - formulation
   - gain vector
   - terminal/load policy
   - source preop/postop configs
   - selected patient
   - expected output artifacts

3. Add planning support:
   - render the exact `svZeroDTrees` command or API call
   - stage config snapshots
   - declare expected artifacts
   - preserve reproducibility metadata

4. Add monitoring support:
   - detect running/completed/failed states
   - surface termination reason
   - surface final split and convergence status
   - surface missing artifact errors cleanly

5. Add result collection:
   - copy structured summaries
   - copy split histories and plots
   - record output paths in manifest history
   - support local first, then Sherlock only through existing HPC abstractions

### Stage 5 Completion Requirements

Stage 5 is complete only when:

- `svzt-agent` can plan and execute a patient-specific adaptation case using the stable Stage 4 `svZeroDTrees` interface.
- The manifest captures all effective adaptation inputs.
- Status reporting includes convergence and final split.
- Fetch/postprocess commands collect the expected structured artifacts.
- Tests cover dry-run planning, manifest serialization, and at least one fake/local execution path.
- No `svZeroDTrees` domain code depends on `svzt-agent`.

## Reporting Standards

Every stage should report:

- exact spec or command used
- output directory
- model case names
- gain vectors
- terminal/load values
- starting/post-perturbation/final split
- convergence flag
- termination reason
- event time when present
- final radius state
- final thickness state for CWSS-IMS
- whether collapse or bounds violations occurred

## Agent Routing

Use these skills/workflows when available:

- `pa-toy-model` for toy experiments and Stage 1 comparisons.
- `adaptation-benchmark-testing` for Stage 2 and Stage 3 harness work.
- `pulmonary-0d-simulation` when checking patient-specific 0D config behavior.
- `pulmonary-bc-tuning-workflow-debugger` only when debugging BC tuning checkpoints.
- `sherlock-cluster-ops` only after local behavior is stable and the user explicitly needs Sherlock execution.

Repo ownership:

- `svZeroDTrees` owns model behavior, config schema, benchmark harness behavior, validation, and domain outputs.
- `svzt-agent` owns orchestration, manifests, planning, monitoring, local/remote execution, and artifact collection.
- `svz` owns workspace control-plane config, templates, and run artifacts.

## Immediate Next Step

Start Stage 3 by expanding the completed TST-STAN-1 Stage 2 benchmark into a robustness sweep. Keep the Stage 2 case names fixed, include the `50x` and `100x` IMS thickness-gain cases, and compare the same cases across additional patient configs, perturbation severities, and terminal/load settings where available.
