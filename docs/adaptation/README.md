# Adaptation Notes

Read this directory before changing adaptation models, benchmark specs, or patient-specific adaptation integration.

## Files

- `toy-stability-notes.md`: distilled learnings from the fixed-upstream/adaptive-downstream-BC toy investigation.
- `stage-2-to-5-plan.md`: staged implementation plan from toy benchmark to PA harness, full `svZeroDTrees` integration, and `svzt-agent` orchestration.
- `../../ADAPTATION_RUN_LOG.md`: chronological run log with raw experiment history and artifact paths.

## Current Direction

Use the corrected adaptation topology when reasoning from toy studies:

```text
fixed upstream branch segment -> adaptive downstream BC segment
```

The upstream perturbation is fixed during integration. Adaptation acts on downstream BC radii for CWSS and on downstream BC radii plus thickness for CWSS-IMS.

Do not generalize older no-load toy results where the perturbed branch radii were also the adaptive state. Those runs are useful historical counterexamples, but they do not represent the intended patient-specific adaptation topology.

## Stage 1 Artifacts

Runner:

```text
examples/run_two_branch_terminal_resistance_cwss_comparison.py
```

Output:

```text
examples/output/two-branch-terminal-resistance-cwss-comparison/
```

Summary:

```text
examples/output/two-branch-terminal-resistance-cwss-comparison/comparison_summary.csv
```

## Stage 3 Entry Point

Stage 3 has a completed local TST-STAN-1/TST-STAN-9 patient-specific pilot spec:

```text
examples/adapt_benchmark_stage3_tst_stan_1_9_robustness.yml
```

Run it from `svZeroDTrees/`:

```bash
MPLCONFIGDIR=/private/tmp PYTHONPATH=src python -m svzerodtrees.cli adapt-benchmark examples/adapt_benchmark_stage3_tst_stan_1_9_robustness.yml
```

Output:

```text
examples/output/stage3-tst-stan-1-9-pa-m3-patient-specific-base1e-2-maxnodes20000/
```

The Stage 3 summary CSV includes patient/scenario grouping fields, tree node counts, tree `max_nodes` cap-hit flags, and explicit stability-screen flags for branch collapse, radius/thickness relative-change bounds, non-finite metrics, and nonphysical terminal loads. The pilot uses only the selected patient-specific preop/postop simplified nonlinear RRI pair for each patient. It found no adaptive `M3` case that passes the current screen for both patients; TST-STAN-9 results are explicitly capped at 20,000 nodes per tree.
