# svZeroDTrees Documentation

`svZeroDTrees` is a Python package for structured-tree boundary condition
modeling, tuning, adaptation, and related analysis for `svZeroD`
cardiovascular simulations.

This documentation is scoped to the package itself: the domain workflows,
interfaces, and package-level reference material that belong in this
repository.

:::{toctree}
:maxdepth: 2
:caption: Contents

overview
interface
performance_improvements
API Reference <api/index>
:::

## What This Covers

- The main package workflows, including `pipeline`, `tune_bcs`,
  `construct_trees`, `adapt`, and `postprocess`.
- The YAML configuration interface used by the CLI and Python API.
- Package internals exposed through generated API reference pages.
- Notes on implementation and performance work that affect this repository.

## What This Does Not Cover

- HPC orchestration, remote staging, Slurm submission, or cluster operations.
- Workspace-specific control-plane behavior owned by downstream tooling.

Those concerns should stay outside `svZeroDTrees` unless they correspond to a
clean, reusable package interface.

## Quick Links

- Repository overview: [overview.md](overview.md)
- YAML interface reference: [interface.md](interface.md)
- Package README, development guide, and contribution policy live in the
  repository root alongside this `docs/` directory.

## License

Copyright © 2026 Nick Dorn.

Released under the MIT License.
