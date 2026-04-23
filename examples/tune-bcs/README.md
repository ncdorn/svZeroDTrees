# Local BC Tuning Example

This directory is a small local smoke case for the pipeline path that tunes
impedance boundary conditions and launches a preop `svmultiphysics` simulation
without SLURM.

Run from the repository root:

```bash
svzerodtrees pipeline examples/bc-tuning/local_pipeline.yml
```

The example writes outputs under `examples/bc-tuning/`, including
`optimized_params.csv`, `data/preop/svFSIplus.xml`,
`data/preop/svzerod_3Dcoupling.json`, and solver result files created by the
local `svmultiphysics` executable.

The mesh is intentionally tiny and is meant to verify local setup and dispatch,
not to provide a scientifically meaningful simulation.
