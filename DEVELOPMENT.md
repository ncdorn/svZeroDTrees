# Development Guide

This document describes the local development workflows for `svZeroDTrees`.
It is intentionally focused on package development, validation, and repo
layout.

## Repository Layout

- `src/svzerodtrees`: package source code.
- `tests`: unit and package-level integration tests.
- `examples`: example configs and small workflow examples.
- `docs`: user and interface documentation.

## Python And Packaging

`svZeroDTrees` uses:

- `hatchling` as the build backend.
- `hatch` for managed build, test, and docs-preview environments.
- `uv` for fast local environment sync in this workspace.

The package requires Python `>=3.8`.
The validated Sherlock baseline is Python `3.12.1`.

## Local Setup

### Option 1: Hatch-managed validation

Use Hatch when you want isolated build or test runs that follow the configured
project scripts.

```bash
python -m pip install hatch
hatch run test:run
hatch run build:check
hatch run docs:serve
```

### Option 2: uv-managed local environment

Use `uv` when you want a working editable environment in this workspace. This
repo config resolves `pysvzerod` from the sibling `../svZeroDSolver` checkout
through the `solver` dependency group.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --group dev --group tests --group build --group solver
UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import svzerodtrees, pysvzerod"
```

### Option 3: pip-managed editable installs

Use plain `pip` when you want an explicit local environment without `uv`.
Install the sibling solver checkout first when you need solver-backed workflows.

```bash
python -m pip install -e ../svZeroDSolver
python -m pip install -e .
```

`svZeroDTrees` itself can be imported without `pysvzerod`, but any workflow
that actually calls the 0D solver will raise a clear runtime error until the
solver package is installed.

Direct `pytest` runs assume the package is already installed in the active
environment.

## Common Commands

Run the test suite:

```bash
hatch run test:run
```

Run individual test slices:

```bash
hatch run test:unit
hatch run test:integration
hatch run test:e2e
```

Run the test suite in your active environment:

```bash
pytest tests/unit tests/integration tests/e2e
pytest tests/unit tests/integration tests/e2e --cov=svzerodtrees --cov-report=term-missing --cov-report=xml
```

Run static checks:

```bash
ruff check .
black --check .
mypy src/svzerodtrees
```

Build and validate distributions:

```bash
hatch run build:check
```

Preview the docs directory locally:

```bash
hatch run docs:serve
```

## Sherlock Setup

Sherlock has a coherent Python 3.12 module stack for `svZeroDTrees`. The
checked-in helper script loads the validated modules and installs the sibling
solver first, but `svZeroDSolver` needs at least 16 GB of memory to build on
the cluster. Run it from an `sdev` session or another node with at least 16 GB,
not from the default login node:

```bash
sdev -m 16GB
bash load_sherlock_modules.sh
```

## CLI Smoke Checks

Useful quick checks while developing:

```bash
svzerodtrees schema
svzerodtrees pipeline examples/pipeline_example.yml
```

## Development Expectations

- Keep domain logic in `svZeroDTrees`; do not add HPC orchestration or
  cluster-specific control-plane behavior here.
- Prefer small, explicit interface changes over hidden side effects.
- Preserve deterministic outputs and clear error behavior when changing public
  workflows.
- Add or update the smallest test that proves a contract change or bug fix.

## Documentation Sync

When you change public behavior, update the relevant docs in:

- `README.md` for installation or entrypoint changes.
- `docs/interface.md` for interface or config changes.
- `docs/overview.md` and examples for workflow-facing behavior changes.
