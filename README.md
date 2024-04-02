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

[] robustify tree and data visualization methods

## Overview of modules
   ### Modules
   `interface.py`: highest level commands for building trees and computing adaptation
   
   `preop.py`: build trees
   
   `operation.py`: perform repairs in zerod
   
   `adaptation.py`: compute adaptation of trees
   
   ### Classes
   `ConfigHandler`: data handler for the 0D solver json config, with additional classes for each 0d element type
   
   `ResultHandler`: data handler for the simulation result data
   
   `StructuredTreeBC`: class for handling structured trees
   
   `TreeVessel`: class for handling vessels of the structured tree
   

## json config
the json config file requires the following keys
* name: name of the experiment
* model: name of svZeroDSolver model config to build trees
* adapt: type of adaptation, either `cwss` (constant wall shear stress) or `ps` (pries and secomb)
* optimized: True if the boundary conditions have been optimized, False if the package should optimize boundary conditions
* is_full_pa_tree: True if the model is a pulmonary arterial tree. A different optimization algorithm must be used
* trees_exist: True if structured trees already exist for the pre-operative model
* mesh_surfaces_path: path to outlet mesh surfaces, required for PA tree optimization
* task: task to be run, either `repair`, `optimize_stent`, `threed_adaptation`

## Running a config file
The highest level command is `run_from_file(config.json)` examples of `config.json` are provided below:

### construct trees only
```json
{   "name": "foo_exp",
    "model": "foo",
    "adapt": "cwss",
    "optimized": false,
    "is_full_pa_tree": true,
    "trees_exist": false,
    "mesh_surfaces_path": "/foo/bar/Meshes/1.6M_elements/mesh-surfaces",
    "task": "construct_trees",
    "construct_trees": {
        "tree_type": "cwss"
        }
}
```

### 0d stent repair
```json
{   "name": "foo_exp",
    "model": "foo",
    "adapt": "cwss",
    "optimized": false,
    "is_full_pa_tree": true,
    "trees_exist": false,
    "mesh_surfaces_path": "/foo/bar/Meshes/1.6M_elements/mesh-surfaces",
    "task": "repair",
    "repair": {
        "type": "stent",
        "location": "proximal",
        "value": [0.5, 0.5]
        }
}
```

### stent optimization
```json
{   "name": "foo_stent_opt",
    "model": "foo",
    "adapt": "ps",
    "optimized": true,
    "is_full_pa_tree": true,
    "trees_exist": true,
    "mesh_surfaces_path": "/foo/bar/Meshes/1.6M_elements/mesh-surfaces",
    "repair": {
        "type": "optimize_stent",
        "location": "proximal",
        "value": [0.5, 0.5],
        "objective": "flow split"
        }
}
```

### 3D coupling
```json
{   "name": "foo_3d_coupling",
    "model": "foo",
    "adapt": "ps",
    "optimized": false,
    "is_full_pa_tree": true,
    "trees_exist": false,
    "mesh_surfaces_path": "/foo/bar/Meshes/1.6M_elements/mesh-surfaces",
    "task": "threed_adaptation",
    "threed_adaptation":
        {
        "preop_dir": "/foo/bar/threed_model/preop",
        "postop_dir": "/foo/bar/threed_model/postop",
        "adapted_dir": "/foo/bar/threed_model/adapted"
        }
}
```

