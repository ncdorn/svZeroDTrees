# svZeroDTrees
Structured tree boundary condition modeling for svZeroD cardiovascular simulations.

**Capabilities**
- Generate structured tree boundary conditions for svZeroD outlets.
- Tune boundary conditions (impedance or RCR) to clinical targets.
- Adapt microvasculature using CWSS or Pries/Secomb models.
- Optional 3D coupling pipeline via SimVascular tools.

**Requirements**
- Python >= 3.8.
- Validated on Sherlock with `python/3.12.1`.
- Runtime dependencies are installed via `python3 -m pip install -e .`.
- Solver-backed workflows additionally require `pysvzerod` from a sibling
  `svZeroDSolver` checkout.
- External tools for 3D coupling only: SimVascular `svpre`, `svsolver`, `svpost`
  in PATH, plus `svmultiphysics` for local 3D execution or `sbatch` for SLURM
  execution.
- Input data files:
- `zerod_config.json` (svZeroD config)
- `clinical_targets.csv`
- `mesh-surfaces` directory
- Optional `inflow.csv` for custom inflow

**Install**
```bash
git clone https://github.com/ncdorn/svZeroDTrees.git
git clone https://github.com/ncdorn/svZeroDSolver.git
python3 -m pip install -e svZeroDSolver
cd svZeroDTrees
python3 -m pip install -e .
```

If you only need non-solver code paths, `svZeroDTrees` can now be installed
before `pysvzerod`; solver-backed workflows raise an explicit runtime error
until the sibling solver checkout is installed.

For `uv` workflows in this workspace, the sibling `../svZeroDSolver` checkout is
still supported via the `solver` dependency group:
```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --group dev --group tests --group build --group solver
UV_CACHE_DIR=/tmp/uv-cache uv run python3 -c "import svzerodtrees, pysvzerod"
```

**Sherlock**
Building `svZeroDSolver` on Sherlock requires at least 16 GB of memory. Start
an `sdev` session or use another node with at least 16 GB before running the
helper script:
```bash
sdev -m 16GB
bash load_sherlock_modules.sh
```

Do not run the solver install from the default login node; the C++ build can be
killed for lack of memory there.

**Development**
Preferred package-validation workflows use Hatch-managed environments:
```bash
python3 -m pip install hatch
hatch run test:run
hatch run test:unit
hatch run test:integration
hatch run test:e2e
hatch run build:check
hatch run docs:serve
```

Direct `pytest` execution now assumes the package is already installed in the
active environment. The recommended path is `hatch run test:run`.

**Start Here**

New users usually want one of three things:

1. Build a tree in Python: `examples/tutorials/01_build_tree.py`
2. Run a zero-D simulation through a tree: `examples/tutorials/02_simulate_tree.py`
3. Apply a tree-derived load back to an `svZeroD` outlet: `examples/tutorials/03_apply_tree_bc.py`

The matching docs are:

- `docs/tutorial_build_tree.md`
- `docs/tutorial_simulate_tree.md`
- `docs/tutorial_apply_tree_bc.md`

If you want the broader YAML-driven workflow path instead, use:

```bash
svzerodtrees tune-bcs path/to/config.yml
svzerodtrees construct-trees path/to/config.yml
svzerodtrees calibrate-0d-from-3d path/to/config.yml
```

Generate a schema template:
```bash
svzerodtrees schema
```

**Workflows**
- `pipeline`: end-to-end run (0D setup, BC tuning, optional 3D, adaptation).
- `tune_bcs`: optimize impedance or RCR parameters only.
- `construct_trees`: assign impedance or RCR BCs to a svZeroD config.
- `adapt`: run microvascular adaptation using preop/postop results.
- `adapt-benchmark`: run local reduced-PA adaptation sweeps across `M1`, `M2`,
  and `M3` from optimized preop/postop reduced RRI configs and write
  study-level JSON/CSV/PNG summaries.
- `calibrate_0d_from_3d`: run fixed-point-confirmed Levenberg-Marquardt
  calibration from a precomputed mapped centerline result and write a
  calibrated 0D JSON only after stable replay. The
  mapped input may be a single scalar pressure/flow field pair or the ordered
  final-cycle timeseries emitted by svzt-agent. Numbered timeseries require a
  metadata sidecar; `flow_0..N` is integrated volumetric flow in `cm^3/s` and
  is consumed without area multiplication. A legacy `velocity_0..N` stack is
  accepted only when configured with `flow_observation_type: flow`, which also
  prevents area multiplication. A true velocity field must be explicitly
  declared with `flow_observation_type: velocity` and an area array. `dy` is
  derived from recorded sidecar timestamps, including periodic wraparound; if
  the solver returns
  non-finite calibrated parameters, or if the input 0D config contains
  unsupported non-finite numeric values, the workflow raises an error instead
  of writing an invalid JSON file. For rigid-vessel inputs that use positive
  infinity for compliance, opt in with
  `calibration.input_normalization.infinite_vessel_compliance: zero`; only
  `vessels[*].zero_d_element_values.C` is converted to `0.0`, and the result
  records the changed JSON paths. External interfaces are sampled from
  adjacent interior cross-sections and require at least three usable branch
  paths; internal junction interfaces remain topology-derived interpolations.
  An under-resolved vessel must be explicitly excluded with an empty vessel
  parameter override, and the result records selected interface paths and
  exclusions. After QC, the unchanged calibrator is invoked twice: the second
  invocation starts from the first calibrated block values and preserves the
  same observations, solver controls, and parameter selections. The workflow
  publishes only a fixed-point-confirmed result; confirmation tolerances are
  configured with `confirmation_absolute_tolerance` and
  `confirmation_relative_tolerance`. Negative selected parameters and large
  parameter ratios are recorded as warnings, not rejected solely for their
  sign or ratio. Before publication, calibration-only fields are removed,
  single-outlet `internal_junction` blocks are normalized to
  `NORMAL_JUNCTION`, and the result is replayed through the unchanged
  `pysvzerod.simulate` API. Replay uses a bounded settling horizon controlled
  by `replay_minimum_cycles` (default 3), `replay_maximum_cycles`, and
  `required_consecutive_stable_pairs`; it checks finite bounded pressure/flow
  values and final-cycle normalized RMS stability. The solver JSON is atomically published only
  after these checks pass, so a stable negative resistance is allowed while a
  divergent positive-resistance result is rejected.

- `postprocess`: generate figures from saved tree pickles or compute analysis artifacts such as svSlicer-based pulmonary resistance maps or the standardized pulmonary 3D postprocess suite.
  Pulmonary resistance-map configs may optionally set `workers: auto|<int>`, and
  pulmonary 3D suite configs may optionally set `resistance_map_workers`, to
  enable bounded frame-level parallelism during svSlicer centerline mapping.
  The 3D suite writes both a mean resistance map and a systolic resistance map,
  where systole is the maximum simulated MPA centerline pressure in the final
  full cardiac cycle, and the systolic map reuses the mapped centerline
  intermediates generated for the mean map instead of remapping the frame.

**Production calibration contract**

Version-1 calibration keeps a compatibility window for existing configs:
when `calibration.targets` is absent, observation QC defaults to
`strict_network` and the legacy behavior is retained. A production pulmonary
config should opt into `observation_qc.enforcement: target_focused` and define
both target blocks explicitly. The MPA pressure target and RPA flow-split
target name their vessel roles (`MPA`, `LPA`, and `RPA` must be distinct) and
their interfaces (`external_upstream`, `external_downstream`, `upstream`,
`downstream`, or an unambiguous `internal` endpoint). Anatomy is never
inferred from branch numbers or geometry.

Mapped observations must carry explicit pressure and volumetric-flow units,
ordered timestamps, and `cycle_duration_s` in the metadata sidecar. Integrated
flow is consumed as flow and is never multiplied by area. Target traces are
converted to common normalized units and a common periodic phase grid before
comparison; missing, ambiguous, non-finite, or inconsistent metadata is a
fatal input error.

Calibration success means that the complete observation contract passes its
configured QC policy, the unchanged `pysvzerod.calibrate(config)` callable
reaches a two-pass selected-parameter fixed point, and the normalized result
passes bounded settled replay through the unchanged
`pysvzerod.simulate(config)` callable. In `target_focused` mode,
data-contract and configured-target checks are fatal; whole-network
conservation, pressure-direction, and non-target diagnostics remain visible
as advisory metrics.

The MPA pressure waveform NRMSE and absolute RPA split error are independent
component gates. Their tolerance-normalized weighted composite is a
post-calibration score for reporting and gating only; it is never the
objective passed to `pysvzerod.calibrate`. When enabled, the baseline policy
requires the calibrated score not to regress from a stable baseline. A
negative calibrated resistance is a warning with its path and value recorded,
not an automatic failure when fixed-point, replay, and target gates pass.

The solver boundary requires only callable standard `calibrate` and `simulate`
APIs. Reports record the resolved module path, file metadata, optional version
or build identity, and a SHA-256 module digest so the run is reproducible.
Invalid input, QC, fixed-point, replay, or target gates raise an actionable
error and leave the calibrated solver JSON unpublished.

**Outputs**
Typical outputs are written under `paths.root` and include:
- `optimized_params.csv` or `optimized_rcr_params.csv` from tuning.
- `pa_config_tuning_snapshot.json` from pulmonary impedance tuning. This
  snapshot can be re-simulated later with the Python helper
  `svzerodtrees.tuning.summarize_pulmonary_zerod_config(...)` to recover
  pre-mapping MPA pressure and branch-flow metrics for iteration diagnostics.
- `svzerod_config_with_bcs.json` (or `paths.output_config`) from tree construction.
- calibrated 0D JSON at `paths.output_config` from `calibrate_0d_from_3d`.
- `calibration_observation_qc.json` beside the calibration output; failed
  conservation, inflow, pressure-direction, path-coverage, or sampling checks
  prevent solver dispatch and leave the output JSON unwritten.
- `calibration_confirmation.json` beside the calibration output; it records both
  black-box calibrator invocations, fixed-point deltas and tolerances, inactive
  parameter validation, solver provenance, and negative/large-ratio warnings.
- `calibration_replay.json` beside the calibration output; it records the
  bounded settling horizon, accepted cycle, observation-scale bounds,
  finite/bounded checks, and cycle-stability metrics.
- `calibration_targets.json` beside the calibration output; it records the
  explicit target roles and interfaces, normalized MPA-pressure/RPA-split
  component errors and gates, composite score, and baseline policy.
- `calibration_summary.json` beside the calibration output; it combines the
  input normalization, observation QC, output normalization, fixed-point
  confirmation, target evaluation, warnings, provenance, and replay
  diagnostics.
- `preop`, `postop`, `adapted` directories for pipeline/adaptation runs.
- Figures from postprocess workflow (PNG outputs you specify).
- Postprocess analysis artifacts such as `resistance_map_mean.vtp`,
  `resistance_map_systolic.vtp`, ranked CSV summaries, standardized
  `mpa_pressure_vs_time.csv`, flow-split comparison outputs, and metadata JSON
  files.

All calibration reports and the returned result carry the same `run_id` and
content-digest aliases (`normalized_input_digest`, `observation_digest`,
`solver_module_sha256`, and `output_config_digest`). The calibrated solver JSON
is written last with an atomic replacement, after every reportable check has
passed.

The copyable calibration example is
`examples/calibration/calibrate_svslicer_timeseries.yml`. Run it from the case
directory containing the referenced baseline, centerline, timeseries, and
metadata files:

```bash
svzerodtrees calibrate-0d-from-3d calibrate_svslicer_timeseries.yml
```

**Examples**
- Tutorial scripts: `examples/tutorials/01_build_tree.py`,
  `examples/tutorials/02_simulate_tree.py`,
  `examples/tutorials/03_apply_tree_bc.py`.
- YAML configs: `examples/pipeline_example.yml`, `examples/tune_bcs_example.yml`,
  `examples/construct_tree/construct_trees_example.yml`, `examples/adapt_example.yml`,
  `examples/adapt_benchmark_tst_stan_1.yml`, `examples/postprocess_example.yml`,
  `examples/postprocess_resistance_map_example.yml`,
  `examples/calibration/calibrate_svslicer_timeseries.yml`.
- Local BC tuning + preop 3D smoke case: `examples/bc-tuning/local_pipeline.yml`.
- Legacy construct-tree notes: `examples/construct_tree/README.md`.

**Reference**
- Schema reference: `docs/interface.md`.
- User guide: `docs/overview.md`.
