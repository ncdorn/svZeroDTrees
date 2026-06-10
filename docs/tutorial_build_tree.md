# Build a Structured Tree

This tutorial covers the smallest useful task in `svZeroDTrees`: construct a
structured tree in Python, inspect its size, and compute its equivalent outlet
resistance.

## When To Use This Path

Use this path when you want to understand tree geometry and parameters before
you involve a full `svZeroD` model or any outlet-surface mapping.

This is the cleanest place to learn the core tree inputs:

- `initial_d`: root diameter
- `d_min`: collapse threshold for terminal branches
- `lrr`: length-to-radius ratio used to set vessel lengths
- `alpha`, `beta`: left/right daughter scaling
- `compliance_model`: wall model used later for impedance calculations

## Minimal Example

The matching script lives at `examples/tutorials/01_build_tree.py`.

```python
from svzerodtrees import StructuredTree
from svzerodtrees.io.blocks.simulation_parameters import SimParams
from svzerodtrees.microvasculature.compliance.constant import ConstantCompliance

tree = StructuredTree(
    name="demo_tree",
    time=[0.0, 0.5, 1.0],
    simparams=SimParams({}),
    compliance_model=ConstantCompliance(6.6e4),
)

tree.build(
    initial_d=0.30,
    d_min=0.01,
    lrr=10.0,
    alpha=0.9,
    beta=0.6,
)

print(tree.count_vessels())
print(tree.equivalent_resistance())
print(tree.to_dict())
```

## What `build()` Does

`StructuredTree.build()` stores the branch-scaling inputs and creates the
compact tree storage used by downstream impedance and simulation methods.

After `build()` you can immediately inspect:

- `tree.count_vessels()`
- `tree.segment_resistances()`
- `tree.equivalent_resistance()`
- `tree.to_dict()`

## Output You Should Expect

For a build-only workflow, the most useful outputs are not solver files. They
are the derived tree quantities:

- number of generated vessels
- equivalent resistance seen at the root
- serialized tree metadata from `tree.to_dict()`

The example script writes a short JSON summary under
`examples/tutorials/output/`.

## Notes

- This path does not require outlet surface files.
- This path does not require `pysvzerod`.
- If you want to run flow and pressure through the tree next, continue to
  [tutorial_simulate_tree.md](tutorial_simulate_tree.md).
