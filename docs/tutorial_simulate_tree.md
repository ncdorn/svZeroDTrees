# Simulate Flow Through a Structured Tree

This tutorial covers direct reduced-order simulation on a single
`StructuredTree` object.

## When To Use This Path

Use this when you already know the tree parameters and want to answer:

- what pressures and flows develop through the tree?
- what wall shear stress does the tree see?
- what is the homeostatic state for a chosen reference flow?

This is the most direct way to study tree behavior without attaching the tree
to a larger pulmonary model.

## Minimal Example

The matching script lives at `examples/tutorials/02_simulate_tree.py`.

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

result = tree.simulate(
    Q_in=[80.0, 80.0],
    Pd=12.0 * 1333.2,
)

print(result.head())
print(tree.results.pressure_in.shape)
```

## What `simulate()` Produces

`StructuredTree.simulate()` creates a small solver-ready config for the tree,
runs `pysvzerod`, and attaches the results back onto the tree object.

After simulation, the main data surface is `tree.results`, which contains
aligned arrays for:

- inlet and outlet flow
- inlet and outlet pressure
- vessel ids, names, generations, and diameters

That result object is what the postprocessing figure helpers expect.

## Practical Notes

- This path requires `pysvzerod`.
- Distal pressure `Pd` is expected in cgs pressure units used by the solver.
- If you want a steady load at the terminals instead of distal pressure, use
  `distal_bc_type="RESISTANCE"` and optionally pass `distal_resistance=...`.

## Homeostatic State

If you are preparing for adaptation work, the tree object can also cache a
reference homeostatic state:

```python
tree.compute_homeostatic_state(Q=80.0)
```

That populates mean wall shear stress and intramural stress profiles on the
tree for later adaptation logic.

## Next Step

If your goal is to place the tree back onto an `svZeroD` outlet boundary
condition, continue to [tutorial_apply_tree_bc.md](tutorial_apply_tree_bc.md).
