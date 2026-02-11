# svZeroDTrees
Structured tree boundary condition modeling for svZeroD cardiovascular simulations.

**Capabilities**
- Generate structured tree boundary conditions for svZeroD outlets.
- Tune boundary conditions (impedance or RCR) to clinical targets.
- Adapt microvasculature using CWSS or Pries/Secomb models.
- Optional 3D coupling pipeline via SimVascular tools.

**Requirements**
- Python >= 3.8.
- Python packages (installed via `pip install -e .`): `scipy`, `numpy`, `vtk`, `pandas`, `matplotlib`, `networkx`, `svsuperestimator`, `multiprocess`, `pysvzerod`, `pyyaml`.
- External tools for 3D coupling only: SimVascular `svpre`, `svsolver`, `svpost` in PATH.
- Input data files:
- `zerod_config.json` (svZeroD config)
- `clinical_targets.csv`
- `mesh-surfaces` directory
- Optional `inflow.csv` for custom inflow

**Install**
```bash
git clone https://github.com/ncdorn/svZeroDTrees.git
cd svZeroDTrees
pip install -e .
```

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
- Narrative example: `examples/construct_tree/README.md`.

**Reference**
- Schema reference: `docs/interface.md`.
- User guide: `docs/overview.md`.
