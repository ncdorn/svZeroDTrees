# Adaptation Run Log

Last updated: 2026-06-09

This is a running record of adaptation-model parameter sweeps and focused comparison runs in `svZeroDTrees`. The goal is to keep one place that answers:

- what was run
- which model form was tested
- which parameter sets behaved well enough to keep using
- where the saved artifacts live

## Current Decisions

- Keep the log-state ODE as the working model moving forward for toy adaptation studies.
- Treat `M3` harness "convergence" carefully. In the current implementation it is based on a moving window in `RPA` split, not proof that every adaptation state has reached a full equilibrium.
- For toy-model studies, use:
  - `starting_*` for the symmetric baseline before any imposed perturbation
  - `post_perturbation_*` for the state immediately after the imposed perturbation and before adaptation evolves

## What "Tested Well" Means Here

In this log, "tested well" means the run was numerically well behaved and useful for interpretation or comparison. It does not automatically mean the behavior is physiologically desirable.

## PA Harness Runs

### 2026-06-09: Stage 3 TST-STAN-1/TST-STAN-9 patient-specific pre/post-op pilot

- Spec: `examples/adapt_benchmark_stage3_tst_stan_1_9_robustness.yml`
- Output: `examples/output/stage3-tst-stan-1-9-pa-m3-patient-specific-base1e-2-maxnodes20000/`
- Models: `M3` only
- Command:
  ```bash
  MPLCONFIGDIR=/private/tmp PYTHONPATH=src python -m svzerodtrees.cli adapt-benchmark examples/adapt_benchmark_stage3_tst_stan_1_9_robustness.yml
  ```
- Execution notes:
  - Local process parallelism was enabled with `workers: 8`.
  - The run used the selected patient-specific preop/postop simplified nonlinear RRI pair for each patient.
  - TST-STAN-1 used selected preop iter-04 and postop from iter-04.
  - TST-STAN-9 used selected preop iter-03 and postop from iter-03.
  - Structured-tree construction was capped with `max_nodes: 20000`.
  - Rows were not rejected when a tree reached `max_nodes`; `benchmark_summary.csv` records `lpa_tree_max_nodes_reached` and `rpa_tree_max_nodes_reached`.
  - The run used early termination guards for RPA split collapse and radius/thickness relative-change bounds.
- Study-level artifacts:
  - `benchmark_summary.csv`
  - `benchmark_summary.json`
  - `benchmark_convergence_table.csv`
  - `benchmark_failure_table.csv`
  - `benchmark_aggregate_final_rpa_split.png`
  - `benchmark_tst-stan-1_rpa_split_overlay.png`
  - `benchmark_tst-stan-1_lpa_split_overlay.png`
  - `benchmark_tst-stan-9_rpa_split_overlay.png`
  - `benchmark_tst-stan-9_lpa_split_overlay.png`
- Aggregate results:
  - Total rows: 26.
  - Status counts: 26 `ok`.
  - Stability-screen counts: 14 passed, 12 failed.
  - Termination reasons:
    - `rpa_split_window_converged`: 8
    - `radius_bounds_violation`: 6
    - `thickness_bounds_violation`: 6
    - `t_end_reached`: 6
- TST-STAN-1 results:
  - Tree size: 2,401 LPA vessels and 8,045 RPA vessels; neither tree reached `max_nodes`.
  - Only `no_adaptation_fixed_bc` passed the screen, with final RPA split `0.510737`.
  - CWSS-only cases failed by radius bounds, with final RPA splits from `0.018342` to `0.022908`.
  - IMS thickness-gain cases moved the final RPA split toward the fixed post-op split as thickness gain increased, but all failed the thickness-bound screen.
  - The best TST-STAN-1 adaptive split was `cwss_ims_thickness_gains_100x`, final RPA split `0.496291`, rejected by thickness bounds.
- TST-STAN-9 results:
  - Tree size: 20,000 LPA vessels and 20,000 RPA vessels; both trees reached `max_nodes` for every case.
  - All 13 cases ran successfully and passed the current stability screen.
  - `no_adaptation_fixed_bc` final RPA split was `0.729130`.
  - CWSS-only and equal-gain IMS cases reached `t_end` with low final RPA splits around `0.060689` to `0.069584`.
  - IMS thickness-gain cases converged and increased final RPA split with thickness gain:
    - `3x`: `0.409185`
    - `10x`: `0.633666`
    - `30x`: `0.699146`
    - `50x`: `0.711412`
    - `100x`: `0.720378`
    - radius `0.1x` / thickness `1x`: `0.633720`
- Assessment:
  - No adaptive `M3` case passes the current screen for both patients.
  - The higher IMS thickness-gain cases remain the most promising direction by final split, especially `50x` and `100x`.
  - TST-STAN-1 is currently limited by thickness-bound rejection at the selected iter-04 tree size.
  - TST-STAN-9 results should be interpreted with the explicit caveat that both trees hit the 20,000-node construction cap in every case.

### 2026-06-05: Stage 2 TST-STAN-1 `d_min = 0.01` IMS thickness-gain add-on (`50x` and `100x`)

- Spec: `examples/adapt_benchmark_stage2_tst_stan_1_dmin0p01_thickness_addon.yml`
- Tree params: `../svz/runs/tst-stan-1-20260507/local_inputs/optimized_params_dmin_0p01.csv`
- Output: `examples/output/stage2-tst-stan-1-pa-m3-cases-base1e-2-sherlock-preop-dmin0p01-thickness-addon/`
- Models: `M3` only
- Tree sizes:
  - LPA: 2,401 vessels
  - RPA: 8,045 vessels
  - total tree vessels: 10,446
  - M3 state size: 20,892
- Study-level artifacts:
  - `benchmark_summary.csv`
  - `benchmark_summary.json`
  - `benchmark_rpa_split_overlay.png`
  - `benchmark_lpa_split_overlay.png`
  - `benchmark_final_rpa_split.png`
  - `benchmark_stability_convergence.png`
- Results:
  - Both add-on cases ran successfully.
  - Both add-on cases met the current `RPA` split moving-window convergence criterion by `1200 s`.
  - Reference splits:
    - preop `RPA`: `0.662969`
    - fixed post-op `RPA`: `0.510737`
  - Final `RPA` splits:
    - `cwss_ims_thickness_gains_50x`: `0.193480`
    - `cwss_ims_thickness_gains_100x`: `0.329348`
- Assessment:
  - Increasing the IMS thickness gains beyond `30x` continues to move the final split toward the fixed post-op split.
  - The `100x` case is the closest adaptive `d_min = 0.01` case tested so far, but it still undershoots the fixed post-op `RPA` split by about `0.181`.
  - The very large reported LPA thickness relative-change values indicate the thickness state is moving aggressively; interpret the split improvement together with state trajectories rather than as a fully settled physiological result.

### 2026-06-05: Stage 2 TST-STAN-1 PA benchmark with Sherlock preop source, base gains `1e-2`, and `d_min = 0.01`

- Spec: `examples/adapt_benchmark_stage2_tst_stan_1.yml`
- Tree params: `../svz/runs/tst-stan-1-20260507/local_inputs/optimized_params_dmin_0p01.csv`
- Output: `examples/output/stage2-tst-stan-1-pa-m3-cases-base1e-2-sherlock-preop-dmin0p01/`
- Models: `M3` only
- Tree sizes:
  - LPA: 2,401 vessels
  - RPA: 8,045 vessels
  - total tree vessels: 10,446
  - M3 state size: 20,892
- Study-level artifacts:
  - `benchmark_summary.csv`
  - `benchmark_summary.json`
  - `benchmark_rpa_split_overlay.png`
  - `benchmark_lpa_split_overlay.png`
  - `benchmark_final_rpa_split.png`
  - `benchmark_stability_convergence.png`
- Results:
  - All 11 cases ran successfully.
  - All 11 cases met the current `RPA` split moving-window convergence criterion by `1200 s`.
  - Final `RPA` splits:
    - `no_adaptation_fixed_bc`: `0.510737`
    - `cwss_only_no_terminal_resistance`: `0.00000123`
    - `cwss_only_terminal_R_1e03`: `0.00000129`
    - `cwss_only_terminal_R_1e04`: `0.00000120`
    - `cwss_only_terminal_R_1e05`: `0.00000286`
    - `cwss_only_terminal_R_ref_adaptive`: `0.00000120`
    - `cwss_ims_equal_gains`: `0.000000635`
    - `cwss_ims_thickness_gains_3x`: `0.0000869`
    - `cwss_ims_thickness_gains_10x`: `0.006967`
    - `cwss_ims_thickness_gains_30x`: `0.087154`
    - `cwss_ims_radius_gains_0p1x_thickness_gains_1x`: `0.007028`
- Assessment:
  - Reducing `d_min` from `0.05` to `0.01` increases the tree size from 208 to 10,446 vessels and changes the fixed post-op split from `0.295296` to `0.510737`.
  - The current convergence flag is especially misleading for this deep-tree run: the adaptive cases often converge numerically after collapsing toward near-zero `RPA` split.
  - Faster thickness gains still improve the final split relative to CWSS-only/equal-gain IMS, but even the `30x` case remains far below the fixed post-op split.

### 2026-06-05: Stage 2 TST-STAN-1 PA benchmark with Sherlock preop source, base gains `1e-2`, and `d_min = 0.05`

- Spec: `examples/adapt_benchmark_stage2_tst_stan_1.yml`
- Tree params: `../svz/runs/tst-stan-1-20260507/local_inputs/optimized_params_dmin_0p05.csv`
- Output: `examples/output/stage2-tst-stan-1-pa-m3-cases-base1e-2-sherlock-preop-dmin0p05/`
- Models: `M3` only
- Tree sizes:
  - LPA: 89 vessels
  - RPA: 119 vessels
  - total tree vessels: 208
  - M3 state size: 416
- Study-level artifacts:
  - `benchmark_summary.csv`
  - `benchmark_summary.json`
  - `benchmark_rpa_split_overlay.png`
  - `benchmark_lpa_split_overlay.png`
  - `benchmark_final_rpa_split.png`
  - `benchmark_stability_convergence.png`
- Results:
  - All 11 cases ran successfully.
  - 10 of 11 cases met the current `RPA` split moving-window convergence criterion by `1200 s`.
  - `cwss_only_terminal_R_1e05` reached `t_end` without meeting the current convergence criterion.
  - Final `RPA` splits:
    - `no_adaptation_fixed_bc`: `0.295296`
    - `cwss_only_no_terminal_resistance`: `0.000012`
    - `cwss_only_terminal_R_1e03`: `0.000013`
    - `cwss_only_terminal_R_1e04`: `0.000019`
    - `cwss_only_terminal_R_1e05`: `0.027460`
    - `cwss_only_terminal_R_ref_adaptive`: `0.000019`
    - `cwss_ims_equal_gains`: `0.000010`
    - `cwss_ims_thickness_gains_3x`: `0.109092`
    - `cwss_ims_thickness_gains_10x`: `0.221786`
    - `cwss_ims_thickness_gains_30x`: `0.267323`
    - `cwss_ims_radius_gains_0p1x_thickness_gains_1x`: `0.221834`
- Assessment:
  - Reducing `d_min` from `0.1` to `0.05` increases the tree size from 46 to 208 vessels and changes the fixed post-op split from `0.218470` to `0.295296`.
  - CWSS-only and equal-gain IMS still collapse toward the LPA with this corrected preop source.
  - Faster thickness gains remain the strongest stabilizing trend; `30x` ends closest to the fixed post-op split in this run.

### 2026-06-05: Stage 2 TST-STAN-1 PA benchmark with Sherlock preop source and base gains `1e-2`

- Spec: `examples/adapt_benchmark_stage2_tst_stan_1.yml`
- Output: `examples/output/stage2-tst-stan-1-pa-m3-cases-base1e-2-sherlock-preop/`
- Models: `M3` only
- Corrected preop source:
  - Sherlock: `/scratch/users/ndorn/svzt_runs/tst-stan-1-20260507/iterations/iter-04/preop/preop_simplified_zerod_tuned_RRI.json`
  - Local copy: `../svz/runs/tst-stan-1-20260507/iterations/iter-04/preop/preop_simplified_zerod_tuned_RRI.json`
- Study-level artifacts:
  - `benchmark_summary.csv`
  - `benchmark_summary.json`
  - `benchmark_rpa_split_overlay.png`
  - `benchmark_lpa_split_overlay.png`
  - `benchmark_final_rpa_split.png`
  - `benchmark_stability_convergence.png`
- Results:
  - All 11 cases ran successfully.
  - 9 of 11 cases met the current `RPA` split moving-window convergence criterion by `1200 s`.
  - The two non-converged cases were CWSS-only with no terminal resistance and CWSS-only with terminal resistance `1e3`; both reached `t_end`.
  - Final `RPA` splits:
    - `no_adaptation_fixed_bc`: `0.218470`
    - `cwss_only_no_terminal_resistance`: `0.000149`
    - `cwss_only_terminal_R_1e03`: `0.001525`
    - `cwss_only_terminal_R_1e04`: `0.106067`
    - `cwss_only_terminal_R_1e05`: `0.277781`
    - `cwss_only_terminal_R_ref_adaptive`: `0.106067`
    - `cwss_ims_equal_gains`: `0.000099`
    - `cwss_ims_thickness_gains_3x`: `0.146292`
    - `cwss_ims_thickness_gains_10x`: `0.192691`
    - `cwss_ims_thickness_gains_30x`: `0.209118`
    - `cwss_ims_radius_gains_0p1x_thickness_gains_1x`: `0.192737`
- Assessment:
  - This supersedes the earlier base `1e-2` Stage 2 run for TST-STAN-1 because the preop RRI model source is now corrected.
  - The corrected preop source reports a much higher preop `RPA` split (`0.567855`) than the post-op fixed value (`0.218470`), so no-load and low-load CWSS-only cases collapse toward the LPA over this horizon.
  - Faster thickness gains improve robustness: `10x`, `30x`, and radius `0.1x` / thickness `1x` stay much closer to the fixed post-op split than CWSS-only or equal-gain IMS.

### 2026-06-05: Stage 2 TST-STAN-1 PA benchmark with base gains increased to `1e-2`

- Spec: `examples/adapt_benchmark_stage2_tst_stan_1.yml`
- Output: `examples/output/stage2-tst-stan-1-pa-m3-cases-base1e-2/`
- Models: `M3` only
- Change from the first Stage 2 run:
  - active base gains increased from `1e-4` to `1e-2`
  - relative case definitions preserved
  - `cwss_ims_radius_gains_0p1x_thickness_gains_1x` uses radius gains `1e-3` and thickness gains `1e-2`
- Study-level artifacts:
  - `benchmark_summary.csv`
  - `benchmark_summary.json`
  - `benchmark_rpa_split_overlay.png`
  - `benchmark_lpa_split_overlay.png`
  - `benchmark_final_rpa_split.png`
  - `benchmark_stability_convergence.png`
- Results:
  - All 11 cases ran successfully.
  - All 11 cases met the current `RPA` split moving-window convergence criterion by `1200 s`.
  - Final `RPA` splits:
    - `no_adaptation_fixed_bc`: `0.218470`
    - `cwss_only_no_terminal_resistance`: `0.178195`
    - `cwss_only_terminal_R_1e03`: `0.182505`
    - `cwss_only_terminal_R_1e04`: `0.211306`
    - `cwss_only_terminal_R_1e05`: `0.325011`
    - `cwss_only_terminal_R_ref_adaptive`: `0.211306`
    - `cwss_ims_equal_gains`: `0.178207`
    - `cwss_ims_thickness_gains_3x`: `0.197669`
    - `cwss_ims_thickness_gains_10x`: `0.210492`
    - `cwss_ims_thickness_gains_30x`: `0.215561`
    - `cwss_ims_radius_gains_0p1x_thickness_gains_1x`: `0.210538`
- Assessment:
  - Raising the base magnitude to `1e-2` makes the Stage 2 cases converge within the current horizon.
  - CWSS-only without terminal load converges to a substantially lower `RPA` split than the no-adaptation baseline.
  - Increasing terminal resistance still shifts the loaded baseline and keeps final split higher.
  - Faster thickness gains again pull the final split closer to the no-adaptation value, with `30x` ending closest among the no-terminal-load adaptive cases.

### 2026-06-05: Stage 2 TST-STAN-1 PA benchmark carrying Stage 1 cases forward

- Spec: `examples/adapt_benchmark_stage2_tst_stan_1.yml`
- Output: `examples/output/stage2-tst-stan-1-pa-m3-cases/`
- Models: `M3` only
- Cases: all 11 Stage 1 benchmark case names carried forward as scenario names.
- Horizon and solver settings:
  - `t_end = 1200 s`
  - `max_step = 60 s`
  - `rtol = 1e-6`
  - `atol = 1e-7`
  - `solver_method = RK23`
- Study-level artifacts:
  - `benchmark_summary.csv`
  - `benchmark_summary.json`
  - `benchmark_rpa_split_overlay.png`
  - `benchmark_lpa_split_overlay.png`
  - `benchmark_final_rpa_split.png`
  - `benchmark_stability_convergence.png`
- Output contract update:
  - `benchmark_summary.csv` now includes case name, gain columns, terminal/load metadata, final LPA/RPA split, termination reason, convergence flag, event time, and radius/thickness relative-change summaries.
  - Per-case summaries preserve the full effective parameter set in `parameter_provenance`.
- Results:
  - All 11 cases ran successfully and wrote per-case summaries.
  - Current moving-window convergence criterion was met by 3 cases:
    - `no_adaptation_fixed_bc`: `final_rpa_split = 0.2184698441`, `event_time = 61.1111 s`
    - `cwss_only_terminal_R_1e05`: `final_rpa_split = 0.3270293630`, `event_time = 62.2881 s`
    - `cwss_ims_thickness_gains_30x`: `final_rpa_split = 0.2157599378`, `event_time = 850.7836 s`
  - The other 8 cases reached `t_end`.
  - Final `RPA` splits at `t_end` or convergence:
    - `cwss_only_no_terminal_resistance`: `0.209752`
    - `cwss_only_terminal_R_1e03`: `0.212469`
    - `cwss_only_terminal_R_1e04`: `0.232672`
    - `cwss_only_terminal_R_ref_adaptive`: `0.232672`
    - `cwss_ims_equal_gains`: `0.208931`
    - `cwss_ims_thickness_gains_3x`: `0.209946`
    - `cwss_ims_thickness_gains_10x`: `0.212499`
    - `cwss_ims_radius_gains_0p1x_thickness_gains_1x`: `0.217442`
- Assessment:
  - This is the first complete Stage 2 PA-harness baseline with Stage 1 case names.
  - `cwss_only_terminal_R_1e05` changes the loaded PA baseline materially (`postop_rpa_split = 0.327063`) and converges by the current split-window criterion, so it is useful as a high-load control rather than a direct no-load comparison.
  - Faster thickness gains move the final split closer to the no-adaptation value and reduce radius change magnitude over this horizon. The `30x` thickness-gain case is the only adaptive no-terminal-load case that formally converged by `1200 s`.
  - `cwss_only_terminal_R_ref_adaptive` currently uses the same explicit `1e4` terminal resistance as `cwss_only_terminal_R_1e04`; replace this with a patient-derived reference load in a later Stage 2/3 refinement.

### 2026-06-03: `M3`, only `k_tau_r` active, exact zeros for the other gains

- Spec: `examples/adapt_benchmark_tst_stan_1_m3_single_ktr_exact_zero.yml`
- Output: `examples/output/tst-stan-1-m3-single-ktr-exact-zero-20260603/`
- Effective gains: `k_arr = [1e-4, 0, 0, 0]`
- Result:
  - `postop_rpa_split = 0.2184698441`
  - `final_rpa_split = 0.1976579725`
  - `t95 = 1447.2371 s`
  - `n_rhs = 1808`
  - `termination_reason = t_end_reached`
- Assessment:
  - Useful baseline run.
  - Stable enough to integrate cleanly, but did not hit the harness convergence criterion by `3600 s`.
  - Good reference case for "shear-only, weak gain" behavior in the loaded PA system.

### 2026-06-03/04: longer `M3` single-`k_tau_r` run with the `tend7200` spec

- Spec: `examples/adapt_benchmark_tst_stan_1_m3_single_ktr_exact_zero_tend_7200.yml`
- Output: `examples/output/tst-stan-1-m3-single-ktr-exact-zero-20260603-tend7200/`
- Important reproducibility note:
  - The study and scenario names still say `single_ktr_1e-4_exact_zero_others`.
  - The current YAML on disk sets `k_arr: [1e-2, 0, 0, 0]`.
  - Interpret the saved `tend7200` artifact as a `k_tau_r = 1e-2` run unless the YAML is changed again.
- Result from the saved summary:
  - `postop_rpa_split = 0.2184698441`
  - `final_rpa_split = 0.1781946677`
  - `event_time = 371.0574 s`
  - `t95 = 71.0574 s`
  - `n_rhs = 197`
  - `termination_reason = rpa_split_window_converged`
- Assessment:
  - This tested well in the narrow harness sense: fast, stable, and formally converged by the current window criterion.
  - Because of the naming mismatch, do not cite this as a `1e-4` result.

### User-reported PA harness runs not yet fully logged to disk

- The user reported additional PA adaptation runs with:
  - `k_tau_r = 1e-3`
  - `k_tau_r = 1e-2`
- Reported behavior:
  - both converged
- Current status:
  - Keep these as qualitative notes until the exact output directories or summaries are pinned down in the repo.

## Toy Model Runs

All toy studies below use a two-branch, no-distal-load system with fixed total inflow unless noted otherwise.

### 2026-06-03: CWSS-only sweep, equal-and-opposite radius perturbation

- Runner: `examples/run_two_branch_no_load_cwss_toy_benchmark.py`
- Output: `examples/output/two-branch-no-load-cwss-toy/`
- Setup:
  - reference radii: `(0.1, 0.1)`
  - perturbed radii used in this older sweep: `(0.105, 0.095)`
  - model: CWSS only
  - ODE scale: log-state
- Cases and outcomes:
  - `k_tau_r = 1e-4`
    - `final_split_left = 0.6047003822`
    - `symmetry_break_amplified = true`
  - `k_tau_r = 1e-3`
    - `final_split_left = 0.6746858415`
    - `symmetry_break_amplified = true`
  - `k_tau_r = 1e-2`
    - `final_split_left = 1.0`
    - losing-branch radius collapsed to `1.27e-25`
    - `symmetry_break_amplified = true`
- Assessment:
  - Did not test well as a stabilizing model.
  - Useful only as a demonstration that the zero-load plant is unstable under shear-only adaptation.
  - Increasing `k_tau_r` accelerated runaway rather than damping it.

### 2026-06-03: CWSS-only vs full CWSS-IMS, one-sided `+5%` perturbation

- Runner: `examples/run_two_branch_no_load_cwss_ims_comparison_one_sided_5pct.py`
- Output: `examples/output/two-branch-no-load-cwss-ims-comparison-one-sided-5pct/`
- Setup:
  - `starting_radii = (0.1, 0.1)`
  - `post_perturbation_radii = (0.1, 0.105)`
  - thickness starts symmetric at `(0.01, 0.01)`
  - ODE scale: log-state
- CWSS only:
  - case: `cwss_only_k_tau_r_1e-2`
  - `starting_split_left = 0.5`
  - `post_perturbation_split_left = 0.4513641070`
  - `final_split_left = 3.4257427310e-58`
  - `n_rhs = 3644`
- CWSS-IMS:
  - case: `cwss_ims_all_gains_1e-2`
  - `starting_split_left = 0.5`
  - `post_perturbation_split_left = 0.4513641070`
  - `final_split_left = 3.9685026299e-197`
  - `final_thickness_ratio_left_to_right = 186.6202560846`
  - `n_rhs = 3740`
- Assessment:
  - Neither model stabilized the no-load toy.
  - Full CWSS-IMS was more aggressive than CWSS only.
  - Useful as a counterexample showing that the current sign structure does not rescue a hydraulically unstable zero-load parallel system.

### 2026-06-04: log-state vs non-log-state toy comparison

- Runner: `examples/run_two_branch_no_load_log_vs_nonlog_comparison.py`
- Output: `examples/output/two-branch-no-load-log-vs-nonlog/`
- Setup:
  - `starting_radii = (0.1, 0.1)`
  - `post_perturbation_radii = (0.1, 0.105)`
  - compared both CWSS-only and CWSS-IMS
- Results:
  - `cwss_only_log_k_tau_r_1e-2`
    - `final_split_left = 3.4257427310e-58`
    - `n_rhs = 3644`
  - `cwss_only_nonlog_k_tau_r_1e-2`
    - `final_split_left = 3.9685026299e-45`
    - `n_rhs = 4064`
    - losing-branch radius clipped at the toy floor `1e-12`
  - `cwss_ims_log_all_gains_1e-2`
    - `final_split_left = 3.9685026299e-197`
    - `n_rhs = 3740`
  - `cwss_ims_nonlog_all_gains_1e-2`
    - `final_split_left = 3.9685026299e-45`
    - `n_rhs = 4202`
    - losing-branch radius clipped at `1e-12`
- Assessment:
  - Log-state tested better than non-log-state.
  - Non-log variants collapsed faster and hit the radius floor, which made them less useful for interpretation.
  - This comparison is the basis for the current decision to keep the log-state model moving forward.

### 2026-06-05: fixed-upstream perturbation with adaptive downstream BCs

- Runner: `examples/run_two_branch_terminal_resistance_cwss_comparison.py`
- Output: `examples/output/two-branch-terminal-resistance-cwss-comparison/`
- Corrected topology:
  - fixed upstream branch segment, followed by adaptive downstream BC segment
  - adaptation acts only on downstream BC radii and, for CWSS-IMS, downstream BC thickness
  - upstream perturbation is held fixed during integration
  - this replaces the older toy interpretation where the same branch radii both imposed the split perturbation and adapted
- Baseline setup:
  - homeostatic upstream radii: `(0.1, 0.1)`
  - adaptive BC initial radii: `(0.1, 0.1)`
  - adaptive BC initial thickness: `(0.01, 0.01)` for CWSS-IMS
  - fixed total inflow: `1.0`
  - log-state ODEs
- Small upstream perturbation study:
  - perturbed upstream radii: `(0.105, 0.1)`
  - post-perturbation split before adaptation: `0.5231900764`
  - final splits:
    - CWSS only, no terminal load: `0.5769626736`
    - CWSS-IMS equal gains: `0.5577358525`
    - CWSS-IMS MATLAB-scaled gains: `0.5314001411`
    - CWSS only, terminal resistance `1e3`: `0.5657276870`
    - CWSS only, terminal resistance `1e4`: `0.5284267208`
    - CWSS only, terminal resistance `1e5`: `0.5042638118`
  - interpretation:
    - CWSS-only did not collapse for this small perturbation; it converged to a biased steady split.
    - CWSS-IMS with equal gains also remained bounded for this small perturbation.
    - MATLAB-scaled CWSS-IMS reduced the final split bias relative to equal gains by letting thickness respond faster than radius.
    - Terminal resistance reduced split sensitivity; higher terminal resistance kept the final split closer to 50/50.
- Large upstream perturbation study:
  - perturbed upstream radii: `(0.2, 0.1)`
  - post-perturbation split before adaptation: `0.6530612245`
  - final splits:
    - CWSS only, no terminal load: `1.0`
    - CWSS-IMS equal gains: `1.0`
    - CWSS-IMS MATLAB-scaled gains: `0.7242645857`
    - CWSS only, terminal resistance `1e3`: `1.0`
    - CWSS only, terminal resistance `1e4`: `0.7042775280`
    - CWSS only, terminal resistance `1e5`: `0.5234023622`
  - interpretation:
    - CWSS-only stability depends on perturbation magnitude. It is bounded for the small upstream perturbation above, but collapses for the large upstream perturbation without enough terminal load.
    - CWSS-IMS with all gains equal also depends on perturbation magnitude. It can remain bounded for a small perturbation but is unstable for the large upstream perturbation.
    - Terminal resistance stabilizes CWSS-only adaptation by reducing the hydraulic gain from adaptive BC radius to flow split. The resistance magnitude needed for stabilization depends on the upstream perturbation magnitude.
    - MATLAB-style CWSS-IMS gain scaling is required for bounded behavior under the large upstream perturbation in these tests. It does not restore 50/50 flow split, but it avoids the full one-branch collapse seen with equal gains over the same integration window.
- Local stability notes:
  - In the corrected fixed-upstream/adaptive-BC topology, equal gains and MATLAB-scaled gains both have nonpositive local eigenvalues at the symmetric homeostatic point.
  - The MATLAB-scaled case has more negative nonzero modes, so thickness responds faster and damps the radius/thickness coupling more strongly.
  - Negative eigenvalues alone do not fully explain the nonlinear large-perturbation behavior. Perturbation magnitude and downstream load strongly affect whether the finite-amplitude trajectory remains bounded.
- Requirements inferred from this toy experiment:
  - Stability claims must specify topology: fixed upstream perturbation plus adaptive downstream BCs is materially different from adapting the perturbed branch radii directly.
  - Stability claims must specify perturbation magnitude.
  - Stability claims must specify downstream/terminal resistance or other distal loading.
  - CWSS-only can be locally or small-perturbation stable in the corrected topology, but it is not robust to large upstream perturbations without adequate terminal resistance.
  - CWSS-IMS equal gains are not sufficient for robust large-perturbation stability.
  - CWSS-IMS gain scaling with faster thickness adaptation is an important requirement for large-perturbation robustness, though it may still converge to a biased split.
  - Flow split stabilization and homeostatic signal restoration are distinct. Several cases restore downstream WSS approximately while settling at a biased flow split caused by the fixed upstream geometry.

## Models / Variants That Tested Well

- `M3` PA harness with only `k_tau_r` active in the loaded PA system.
  - Reason: runs cleanly, produces interpretable `RPA`-split trajectories, and can meet the current harness convergence criterion at larger `k_tau_r`.
- Log-state toy ODEs.
  - Reason: still unstable in the no-load toy, but numerically smoother and more interpretable than non-log variants.
- Corrected fixed-upstream/adaptive-BC toy topology.
  - Reason: separates the imposed upstream lesion or dilation from the adaptive downstream BC state, matching the intended experiment better than adapting the perturbed branch radii directly.
- CWSS-only with sufficient terminal resistance in the corrected fixed-upstream/adaptive-BC toy.
  - Reason: terminal resistance reduces split sensitivity and can prevent collapse; required magnitude depends on perturbation size.
- CWSS-IMS with MATLAB-style gain scaling in the corrected fixed-upstream/adaptive-BC toy.
  - Reason: faster thickness adaptation improves robustness under large upstream perturbation relative to equal gains.

## Models / Variants That Did Not Test Well

- Older no-load toy where the perturbed branch radii are also the adaptive state.
  - Reason: conflates fixed upstream perturbation with adaptive downstream BC behavior and overstates instability for the intended experiment.
- CWSS-only without enough terminal resistance under large fixed upstream perturbations.
  - Reason: collapses to one-branch dominance for the `(0.2, 0.1)` upstream perturbation.
- CWSS-IMS with all gains equal under large fixed upstream perturbations.
  - Reason: also collapses to one-branch dominance for the `(0.2, 0.1)` upstream perturbation.
- Non-log toy ODEs.
  - Reason: faster collapse, more frequent clipping at the minimum-radius floor, and less useful dynamics for comparison.

## Follow-Up Notes

- If future PA harness runs are added here, record both:
  - the spec path used at run time
  - the exact `k_arr` on disk at the time of the run
- For future toy studies, prefer one-sided perturbations with:
  - `homeostatic_upstream_radii = (0.1, 0.1)`
  - `perturbed_upstream_radii = (...)`
  - `initial_adaptive_radii = (0.1, 0.1)`
  - adaptive BC state separate from the fixed upstream perturbation
- If a future study uses a finite distal load or backpressure, record it explicitly because that changes the stability interpretation materially.
