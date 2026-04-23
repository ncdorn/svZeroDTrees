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
Use [load_sherlock_modules.sh](/Users/ndorn/Documents/Stanford/PhD/Marsden_Lab/SimVascular/svz/repos/svZeroDTrees/load_sherlock_modules.sh) to load the validated Python 3.12 module stack and install both sibling packages in the correct order.

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

**Quickstart**
CLI (use the example config):
```bash
svzerodtrees pipeline examples/pipeline_example.yml
```

Generate a schema template:
```bash
svzerodtrees schema
```

Python API:
```python
from svzerodtrees import load_config
from svzerodtrees.api import PipelineWorkflow

cfg = load_config("examples/pipeline_example.yml")
PipelineWorkflow.from_config(cfg).run()
```

**Workflows**
- `pipeline`: end-to-end run (0D setup, BC tuning, optional 3D, adaptation).
- `tune_bcs`: optimize impedance or RCR parameters only.
- `construct_trees`: assign impedance or RCR BCs to a svZeroD config.
- `adapt`: run microvascular adaptation using preop/postop results.
- `postprocess`: generate figures from saved tree pickles.

**Outputs**
Typical outputs are written under `paths.root` and include:
- `optimized_params.csv` or `optimized_rcr_params.csv` from tuning.
- `svzerod_config_with_bcs.json` (or `paths.output_config`) from tree construction.
- `preop`, `postop`, `adapted` directories for pipeline/adaptation runs.
- Figures from postprocess workflow (PNG outputs you specify).

**Examples**
- YAML configs: `examples/pipeline_example.yml`, `examples/tune_bcs_example.yml`,
  `examples/construct_tree/construct_trees_example.yml`, `examples/adapt_example.yml`,
  `examples/postprocess_example.yml`.
- Local BC tuning + preop 3D smoke case: `examples/bc-tuning/local_pipeline.yml`.
- Narrative example: `examples/construct_tree/README.md`.

**Reference**
- Schema reference: `docs/interface.md`.
- User guide: `docs/overview.md`.
