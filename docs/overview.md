# svZeroDTrees Overview

`svZeroDTrees` builds structured microvascular trees and uses them to create or
adapt outlet boundary conditions for `svZeroD` models.

For a new user, the package is easiest to learn in three steps:

1. build a tree
2. simulate flow and pressure through that tree
3. apply the resulting tree-derived load back to an `svZeroD` outlet

Those steps are documented separately:

- [Build a tree](tutorial_build_tree.md)
- [Simulate a tree directly](tutorial_simulate_tree.md)
- [Apply a tree as a boundary condition](tutorial_apply_tree_bc.md)

## What The Package Covers

- direct `StructuredTree` construction and simulation
- impedance or RCR boundary-condition workflows
- optional BC tuning to clinical targets
- adaptation workflows built on top of structured trees
- optional 3D coupling and postprocessing utilities

## Main Inputs

- `zerod_config.json`: an `svZeroD` model configuration
- `clinical_targets.csv`: target hemodynamics for tuning or adaptation
- `mesh-surfaces`: outlet surfaces used by the full `construct_trees` workflow
- optional `inflow.csv`: custom inflow waveform

## Which Path To Start With

Start with the direct Python tutorials if your goal is to understand behavior.

Start with the YAML workflows if your goal is to prepare a real pulmonary
config for downstream simulation.

### Direct Python Path

- build only: `examples/tutorials/01_build_tree.py`
- build + simulate: `examples/tutorials/02_simulate_tree.py`
- attach a tree-derived BC to a tiny config: `examples/tutorials/03_apply_tree_bc.py`

### Workflow Path

- tune BC parameters: `svzerodtrees tune-bcs ...`
- assign tree BCs to a model: `svzerodtrees construct-trees ...`
- run the broader pipeline: `svzerodtrees pipeline ...`

## Important Caveat About Examples

Some of the older YAML examples are workflow-shaped examples rather than
minimal, self-contained onboarding examples. In particular, the impedance
construction flow depends on a real `mesh-surfaces` directory.

That is why the tutorial pages start with direct Python examples first.

## Reference Material

- strict YAML schema: [interface.md](interface.md)
- package API reference: `docs/api/` in the built Sphinx site
- performance notes: [performance_improvements.md](performance_improvements.md)
