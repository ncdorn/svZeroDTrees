# Contributing

Contributions are welcome. The most useful contributions are focused bug fixes,
test coverage improvements, interface clarifications, and documentation updates
that make `svZeroDTrees` a cleaner domain dependency.

## Before You Start

- Search existing issues before opening a new one.
- For bugs or feature requests, use the GitHub issue tracker:
  `https://github.com/ncdorn/svZeroDTrees/issues`
- If you plan to work on a non-trivial change, leave a short note on the issue
  so work does not overlap unnecessarily.

## Scope

Please keep changes aligned with the purpose of this repository:

- Good fits: domain logic fixes, public API improvements, config validation,
  deterministic outputs, and reusable package behavior.
- Poor fits: Slurm, SSH, rsync, remote staging, cluster path handling, or other
  orchestration-specific workflow logic.

## Local Setup

1. Fork `https://github.com/ncdorn/svZeroDTrees`.
2. Clone your fork.

```bash
git clone git@github.com:YOUR_GITHUB_USERNAME/svZeroDTrees.git
cd svZeroDTrees
```

3. Create a branch from `main`.

```bash
git checkout main
git checkout -b fix-short-description
```

4. Set up a development environment.

Hatch-managed workflow:

```bash
python -m pip install hatch
hatch run test:run
```

uv-managed local environment:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --group dev --group tests --group build
pytest
```

In this workspace, `uv` resolves `pysvzerod` from the sibling
`../svZeroDSolver` checkout.

## Quality Checks

Before opening a pull request, run the checks relevant to your change:

```bash
hatch run test:run
ruff check .
black --check .
mypy src/svzerodtrees
hatch run build:check
```

If your change affects docs or examples, update them in the same pull request.

## Pull Request Guidelines

- Keep pull requests focused.
- Add or update tests for behavior changes.
- Update docs when public APIs, config shape, outputs, or CLI behavior change.
- Use clear commit messages. Conventional commits are preferred, for example
  `fix: handle missing inflow file in pipeline config`.
- Explain any scientific, modeling, or reproducibility implications in the PR
  description when they are material to review.

## Reporting Bugs

When filing a bug, include:

- What you ran.
- The input/config involved, if shareable.
- The observed behavior or traceback.
- The behavior you expected.
- Version or commit information when available.

Small, reproducible examples make bugs much easier to diagnose.
