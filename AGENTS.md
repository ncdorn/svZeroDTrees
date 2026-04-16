# AGENTS.md

## How To Use This File
- Treat this file as a short rules-of-engagement brief for coding agents.
- Do not treat it as the full source of truth for package architecture, scientific methods, or user workflows.
- Keep it short. Put implementation detail in repo docs, code, and tests instead of expanding `AGENTS.md`.

## Repo Purpose
- `svZeroDTrees` is a domain/scientific workflow codebase for structured-tree boundary condition modeling, tuning, adaptation, and related analysis.
- In this workspace, it is primarily an upstream dependency and integration target for `svzt-agent`.
- It is not the control-plane layer for HPC execution, run planning, remote staging, monitoring, or workspace policy.
- If a problem is mainly about orchestration, operator workflow, or cluster behavior, start by assuming the fix belongs in `svzt-agent`, not here.

## When To Modify This Repo
- Modify this repo when `svzt-agent` needs a cleaner or more stable upstream boundary and that boundary properly belongs to `svZeroDTrees`.
- Acceptable changes include:
- integration fixes required for correct `svzt-agent` use of the package
- public API stabilization or explicit entrypoints needed for orchestration to call into the library cleanly
- deterministic output or artifact improvements that downstream tooling needs to consume reliably
- clearer config validation or error behavior that helps automation fail safely and predictably
- bug fixes in domain logic that block correct workflow execution
- Changes here should improve the package as a reusable domain dependency, not just patch a single workspace workflow.

## When Not To Modify This Repo
- Do not add HPC orchestration logic here.
- Do not add cluster-specific path logic here.
- Do not add Slurm, SSH, rsync, or remote-staging logic here.
- Do not move control-plane responsibilities out of `svzt-agent` and into this repo.
- Do not use this repo as a dumping ground for workflow-specific hacks, wrappers, or one-off workspace behavior.
- Do not solve operator UX problems here when the real issue is in `svzt-agent` CLI, manifests, planning, or monitoring.
- Do not hide environment-specific assumptions in domain code just because `svzt-agent` needs them today.

## Preferred Integration Posture
- Prefer adapting `svzt-agent` first.
- Change `svZeroDTrees` only when the upstream library boundary should become cleaner, more explicit, or more stable.
- Prefer explicit Python APIs, documented CLI entrypoints, and structured outputs over fragile filesystem poking or implicit side effects.
- Prefer small upstream changes that improve reuse over workspace-specific coupling.
- Preserve scientific clarity and reproducibility when exposing new integration surfaces.

## Output And Interface Expectations
- Favor deterministic behavior for the same inputs and configuration.
- Favor stable programmatic interfaces over undocumented internal coupling.
- Favor structured outputs and predictable artifact names or locations when downstream tooling depends on them.
- Favor clear, actionable errors over silent fallback behavior.
- Surface version or provenance information when it materially helps downstream automation reason about behavior.
- If public behavior changes, treat that as an interface change, not an incidental refactor.

## Safety And Scientific Hygiene
- Do not mix scientific or numerical logic with orchestration concerns.
- Preserve reproducibility. Avoid hidden side effects, ambient state, or behavior that depends on undeclared environment details.
- Keep domain behavior inspectable and understandable from inputs, config, and code.
- Avoid changes that silently alter downstream workflow expectations without documenting the contract change.
- If an integration need pressures this repo toward control-plane behavior, stop and move that logic back toward `svzt-agent`.

## Testing Expectations
- Preserve or improve tests when changing public behavior.
- Prefer focused unit tests and package-level integration tests.
- Avoid tests that require external cluster state, SSH access, Slurm, or workspace-specific infrastructure.
- Validate interface changes that `svzt-agent` depends on, including CLI dispatch, config validation, output contracts, and error behavior where relevant.
- When fixing an integration bug, add or update the smallest test that proves the contract.

## Documentation Sync
- This checklist is required when behavior changes.
- If you change public APIs or entrypoints, update the relevant `README.md` sections and any interface docs.
- If you change output artifact structure or naming, update the docs that describe outputs and examples that depend on them.
- If you change required inputs or config shape, update schema/interface documentation and examples.
- If you change error behavior that orchestration depends on, document the new expectation where operators or integrators will look for it.
- If the change affects how `svzt-agent` should integrate with this repo, note that in `svzt-agent` integration docs as needed instead of leaving the contract implicit.

## Authority Note
- `svzt-agent` owns orchestration policy, remote execution boundaries, and workspace workflow control.
- `svZeroDTrees` owns domain workflow behavior, scientific logic, and its own supported interfaces.
- This file is a boundary guide for coding agents, not a substitute for the repo's actual docs, public interfaces, or tests.
- When in doubt, prefer the cleaner ownership split over the quickest local patch.
