# Toy Stability Notes

Last updated: 2026-06-05

These notes summarize the main lessons from the corrected two-branch toy adaptation investigation. For raw run history and exact artifact paths, see `../../ADAPTATION_RUN_LOG.md`.

## Correct Toy Topology

The intended Stage 1 toy topology is:

```text
fixed upstream branch segment -> adaptive downstream BC segment
```

The upstream branch radius perturbation is fixed during integration. The adaptive state is separate:

- CWSS adapts downstream BC radii.
- CWSS-IMS adapts downstream BC radii and downstream BC thickness.

This distinction matters. Earlier no-load toy experiments adapted the same branch radii that imposed the flow-split perturbation, which conflated fixed lesion geometry with adaptive downstream BC behavior.

## Benchmark Cases

Stage 1 currently includes:

```text
no_adaptation_fixed_bc
cwss_only_no_terminal_resistance
cwss_only_terminal_R_1e03
cwss_only_terminal_R_1e04
cwss_only_terminal_R_1e05
cwss_only_terminal_R_ref_adaptive
cwss_ims_equal_gains
cwss_ims_thickness_gains_3x
cwss_ims_thickness_gains_10x
cwss_ims_thickness_gains_30x
cwss_ims_radius_gains_0p1x_thickness_gains_1x
```

Use `cwss_ims_thickness_gains_10x` rather than `matlab_scaled` in new specs and reports.

## Perturbation Dependence

CWSS-only stability depends on upstream perturbation magnitude.

- Small perturbation, upstream `(0.105, 0.1)`: CWSS-only remains bounded and converges to a biased split.
- Large perturbation, upstream `(0.2, 0.1)`: CWSS-only collapses without enough terminal resistance.

CWSS-IMS with equal gains follows the same pattern:

- It can remain bounded for small upstream perturbations.
- It is not robust to large upstream perturbations.

Do not make blanket stability claims without specifying perturbation magnitude.

## Terminal Resistance

Terminal resistance stabilizes CWSS-only adaptation by reducing the hydraulic gain from downstream BC radius changes to flow split changes.

Observed qualitative behavior:

- Larger terminal resistance keeps final split closer to the no-adaptation split and often closer to 50/50.
- The resistance magnitude required for bounded behavior depends on upstream perturbation magnitude.
- Round-number terminal loads (`1e3`, `1e4`, `1e5`) are useful mechanism tests.
- Patient-specific work should include a derived or measured terminal/load estimate rather than relying only on round numbers.

## CWSS-IMS Gain Scaling

Equal CWSS-IMS gains create a radius/thickness degeneracy: if radius and thickness start proportionally, equal gain channels can move them together.

Thickness gain scaling breaks that degeneracy:

```text
radius gains    = 1x
thickness gains = 3x, 10x, or 30x
```

The `10x` and `30x` cases improve large-perturbation robustness relative to equal gains in the toy. They may still settle at biased flow splits because the fixed upstream geometry remains asymmetric.

The `radius_gains_0p1x_thickness_gains_1x` case helps separate relative thickness dominance from absolute gain size. It has the same 10x relative thickness/radius scaling as `thickness_gains_10x` but slower absolute dynamics.

## Eigenvalue Interpretation

In the corrected fixed-upstream/adaptive-BC topology, equal gains and thickness-scaled gains both have nonpositive local eigenvalues at the symmetric homeostatic point.

Thickness-scaled cases have more negative nonzero modes, so thickness responds faster. This helps explain improved robustness, but local eigenvalues alone do not determine finite-amplitude stability.

Finite-amplitude behavior depends on:

- topology
- perturbation magnitude
- downstream/terminal load
- gain ratios
- absolute gain magnitudes
- radius and thickness bounds

## Split Versus Signal Restoration

Flow split stabilization and homeostatic signal restoration are distinct.

Several cases approximately restore downstream WSS while converging to a biased flow split. That can be expected when the upstream geometry remains fixed and asymmetric.

For PA harness and patient-specific work, report both:

- final split
- WSS/IMS error or final adaptation signals

Do not treat one as a substitute for the other.

## Requirements For Future Stages

Future PA harness and patient-specific adaptation stages must report:

- topology or model formulation
- perturbation/source configuration
- terminal/load policy
- gain vector
- starting, post-perturbation, and final split
- final radius state
- final thickness state for CWSS-IMS
- convergence flag
- termination reason
- collapse or bounds violation status
