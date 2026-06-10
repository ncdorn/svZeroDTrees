# Apply a Tree as a Boundary Condition

There are two useful ways to think about "applying" a tree in this package:

1. derive a boundary condition from a tree object in Python
2. run the `construct_trees` workflow to assign tree-based BCs across a full
   `svZeroD` model

The first path is the easiest way to understand the interface. The second path
is the normal workflow for outlet-mapped impedance BC generation.

## Path 1: Direct Python Attachment

The matching script lives at `examples/tutorials/03_apply_tree_bc.py`.

It does three things:

1. loads a tiny one-outlet `svZeroD` config
2. builds a tree
3. replaces the outlet BC with a tree-derived resistance BC and writes a new
   config JSON

```python
from svzerodtrees import ConfigHandler, StructuredTree
from svzerodtrees.microvasculature.compliance.constant import ConstantCompliance

config = ConfigHandler.from_json("examples/construct_tree/simple_config.json")

tree = StructuredTree(
    name="demo_tree_bc",
    time=config.inflows["INFLOW"].t,
    simparams=config.simparams,
    compliance_model=ConstantCompliance(6.6e4),
)
tree.build(initial_d=0.30, d_min=0.01, lrr=10.0, alpha=0.9, beta=0.6)

config.bcs["BC"] = tree.create_resistance_bc("BC", Pd=0.0)
config.tree_params[tree.name] = tree.to_dict()
config.to_json("simple_config_with_tree_resistance.json")
```

This is the smallest complete example of taking a tree and turning it into an
outlet boundary condition in a solver-readable config.

## Path 2: Impedance BC Construction Across Model Outlets

For the full impedance workflow, use the CLI or workflow API:

```bash
svzerodtrees construct-trees path/to/config.yml
```

That path calls `construct_impedance_trees(...)`, which:

- inspects outlet surface areas from `mesh_surfaces`
- maps outlet caps to BC names deterministically
- builds either shared LPA/RPA trees or one tree per outlet
- computes impedance kernels
- writes updated BCs and serialized tree metadata into the output config

The YAML reference for this lives in [docs/interface.md](interface.md).

## Why The Repo Needs Both Paths

The direct Python path is ideal for learning and unit-level reasoning.

The YAML `construct_trees` path is ideal for real pulmonary model preparation
because it uses the outlet surface geometry and preserves the config-level
mapping metadata needed downstream.

## Current Example Status

The legacy `examples/construct_tree/` material shows the intended
`construct_trees` workflow shape, but it is not a fully self-contained
onboarding example because the required outlet-surface directory is not shipped
there as a minimal dataset.

For a new user, start with the Python example in `examples/tutorials/`, then
move to the YAML workflow once you have real `mesh_surfaces` data.
