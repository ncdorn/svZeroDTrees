# Legacy `construct_trees` Example

This directory shows the shape of the YAML-driven `construct_trees` workflow
around the tiny `simple_config.json` model.

The main files are:

1. `construct_trees_example.yml`: example workflow config
2. `construct_trees.py`: tiny runner script
3. `simple_config.json`: one-outlet `svZeroD` config used by the example

## Important Caveat

This is not the best onboarding example for a new user.

The full impedance-construction workflow depends on outlet surface data under
`mesh-surfaces/`, and that minimal dataset is not currently shipped here as a
complete self-contained tutorial case.

For first-time users, start with:

- `examples/tutorials/01_build_tree.py`
- `examples/tutorials/02_simulate_tree.py`
- `examples/tutorials/03_apply_tree_bc.py`

Then move to the YAML workflow once you have real `mesh-surfaces` data.
