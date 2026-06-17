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
- Runtime dependencies are installed via `python -m pip install -e .`.
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
python -m pip install -e svZeroDSolver
cd svZeroDTrees
python -m pip install -e .
```

If you only need non-solver code paths, `svZeroDTrees` can now be installed
before `pysvzerod`; solver-backed workflows raise an explicit runtime error
until the sibling solver checkout is installed.

For `uv` workflows in this workspace, the sibling `../svZeroDSolver` checkout is
still supported via the `solver` dependency group:
```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --group dev --group tests --group build --group solver
UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import svzerodtrees, pysvzerod"
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
python -m pip install hatch
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
- `calibrate_0d_from_3d`: run stage-1 Levenberg-Marquardt calibration from a
  precomputed mapped centerline result and write a calibrated 0D JSON.
- `postprocess`: generate figures from saved tree pickles or compute analysis artifacts such as svSlicer-based pulmonary resistance maps or the standardized pulmonary 3D postprocess suite.
  Pulmonary resistance-map configs may optionally set `workers: auto|<int>`, and
  pulmonary 3D suite configs may optionally set `resistance_map_workers`, to
  enable bounded frame-level parallelism during svSlicer centerline mapping.
  The 3D suite writes both a mean resistance map and a systolic resistance map,
  where systole is the maximum simulated MPA centerline pressure in the final
  full cardiac cycle, and the systolic map reuses the mapped centerline
  intermediates generated for the mean map instead of remapping the frame.

**Outputs**
Typical outputs are written under `paths.root` and include:
- `optimized_params.csv` or `optimized_rcr_params.csv` from tuning.
- `pa_config_tuning_snapshot.json` from pulmonary impedance tuning. This
  snapshot can be re-simulated later with the Python helper
  `svzerodtrees.tuning.summarize_pulmonary_zerod_config(...)` to recover
  pre-mapping MPA pressure and branch-flow metrics for iteration diagnostics.
- `svzerod_config_with_bcs.json` (or `paths.output_config`) from tree construction.
- calibrated 0D JSON at `paths.output_config` from `calibrate_0d_from_3d`.
- `preop`, `postop`, `adapted` directories for pipeline/adaptation runs.
- Figures from postprocess workflow (PNG outputs you specify).
- Postprocess analysis artifacts such as `resistance_map_mean.vtp`,
  `resistance_map_systolic.vtp`, ranked CSV summaries, standardized
  `mpa_pressure_vs_time.csv`, flow-split comparison outputs, and metadata JSON
  files.

**Examples**
- Tutorial scripts: `examples/tutorials/01_build_tree.py`,
  `examples/tutorials/02_simulate_tree.py`,
  `examples/tutorials/03_apply_tree_bc.py`.
- YAML configs: `examples/pipeline_example.yml`, `examples/tune_bcs_example.yml`,
  `examples/construct_tree/construct_trees_example.yml`, `examples/adapt_example.yml`,
  `examples/adapt_benchmark_tst_stan_1.yml`, `examples/postprocess_example.yml`,
  `examples/postprocess_resistance_map_example.yml`.
- Local BC tuning + preop 3D smoke case: `examples/bc-tuning/local_pipeline.yml`.
- Legacy construct-tree notes: `examples/construct_tree/README.md`.

**Reference**
- Schema reference: `docs/interface.md`.
- User guide: `docs/overview.md`.
