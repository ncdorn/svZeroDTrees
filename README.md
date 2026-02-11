# svZeroDTrees
This is a package to generate structured and morphometric trees to simulate microvascular adaptation in svZeroDPlus.

## Installing

1. Clone the repository with `git clone https://github.com/ncdorn/svZeroDTrees.git`
2. In desired python environment run

   `cd svZeroDTrees`
   
   `pip install -e .`

## Dependencies

clone and install:
* [svZeroDSolver](https://github.com/SimVascular/svZeroDSolver)
* [svSuperEstimator](https://github.com/StanfordCBCL/svSuperEstimator)

standard packages:
* scipy
* numpy
* vtk
* pandas
* matplotlib
* networkx


## Overview
Generate structured trees at the outlet of 0d models created in SimVascular based on outflow boundary conditions to simulate microvasculature. Adapt these trees to changes in flow at the outlet based on upstream changes, such as stenosis relief.
There are currently two microvascular adaptation schemes:
* constant wall shear stress assumption (Yang et al. 2016)
* Pries and Secomb model (Pries et al. 1998)

## todo
[] document all code

[] robustify documentation

[] add clinical targets to json file

## Modules
`simulation`: end-to-end pipeline orchestration (0D + 3D coupling)

`tune_bcs`: boundary condition tuning and assignment

`adaptation`: compute adaptation of trees

`post_processing`: post-processing utilities and plotting

## Classes
`ConfigHandler`: data handler for the 0D solver json config, with additional classes for each 0d element type

`ResultHandler`: data handler for the simulation result data

`StructuredTreeBC`: class for handling structured trees

`TreeVessel`: class for handling vessels of the structured tree
   

## building a simple tree
A simple tree with resistance 100.0 can be built with the following methods. steady through the tree is computed easily as well

```python
tree = StructuredTree(name='simple tree')
tree.optimize_tree_diameter(resistance=100.0)

# compute pressure and flow in the tree with inlet flow 10.0 cm3/s and distal pressure 100.0 dyn/cm2
tree_result = tree.simulate(Q_in = [10.0, 10.0], Pd=100.0)

```

## YAML Interface (v1)
svzerodtrees uses a YAML-first interface with explicit workflows. The CLI and Python API share the same schema.

### CLI
```\n+svzerodtrees pipeline config.yml\n+svzerodtrees tune-bcs config.yml\n+svzerodtrees construct-trees config.yml\n+svzerodtrees adapt config.yml\n+svzerodtrees postprocess config.yml\n+svzerodtrees schema\n+```\n+
### Python API
```python\n+from svzerodtrees import load_config\n+from svzerodtrees import PipelineWorkflow\n+\n+cfg = load_config(\"config.yml\")\n+PipelineWorkflow.from_config(cfg).run()\n+```\n+
### Quick YAML example (pipeline)
```yaml\n+version: 1\n+workflow: pipeline\n+paths:\n+  root: .\n+  zerod_config: zerod_config.json\n+  clinical_targets: clinical_targets.csv\n+  mesh_surfaces: mesh-complete/mesh-surfaces\n+  preop_dir: preop\n+  postop_dir: postop\n+  adapted_dir: adapted\n+  optimized_params: optimized_params.csv\n+bcs:\n+  type: impedance\n+  compliance_model: constant\n+  is_pulmonary: true\n+adaptation:\n+  method: cwss\n+  location: uniform\n+  iterations: 10\n+pipeline:\n+  run_steady: true\n+  optimize_bcs: true\n+  run_threed: true\n+  adapt: true\n+```\n+
See `docs/interface.md` for the full schema and additional examples.
