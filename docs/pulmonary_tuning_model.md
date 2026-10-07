# Physiological full-PA structured-tree tuning: model, justification, and contingencies

This document describes the full pulmonary-artery (PA) boundary-condition
model used for the PPAS tof-stent cohort, the justification and evidence for
each modeling choice and assumption, its known limitations, and what to change
if it does not fit a patient. Configuration syntax is in
[`interface.md`](interface.md); the operator view (svzt-agent workspace
settings, per-patient overrides, checks) is in
`svzt-agent/docs/TUNING_MODEL.md`. The evidence comes from a TST-STAN-5 study
in `examples/adaptation/tst5-physiological-tuning-study/` (`FINDINGS.md`,
`OPTION1.md`, `REPORT_*.md`); TST-STAN-5 has 54% pulmonary regurgitation (PR).

## 1. Model structure

```
inflow (MRI MPA flow, prescribed) -> proximal 0D model (3D-domain vessels)
                                   -> one structured tree per outlet cap -> constant Pd
```

- **Proximal model**: the learned / calibrated full-PA 0D seed (one 0D vessel
  per 3D-domain segment, outlet BCs paired to mesh caps geometrically).
- **Distal model**: an Olufsen structured tree per outlet cap, built at the
  cap's measured diameter, converted to a time-domain impedance kernel.
- **Targets**: catheter MPA systolic / diastolic / mean pressure and the RPA
  flow split. Four numbers, which bounds how many parameters can be identified.
- **Free parameters** (cohort default): xi and eta_sym per PA side, and the
  large-vessel stiffness k3 (5 parameters for 4 targets). Everything else is
  fixed from the literature or the patient's data.

## 2. Modeling choices

Each subsection: what, why, evidence, assumptions, and the alternative if it fails.

### 2.1 Objective: Gaussian likelihood (`objective.type: likelihood`)
- **What**: `chi2 = sum(((model - target) / sigma)^2)`, sigma = 2 mmHg for
  catheter pressures, 0.02 for the split.
- **Why**: the historical relative error divides each error by its target, so
  one mmHg of a 3 mmHg diastolic weighs ~100x a systolic mmHg. With proximal
  compliance enabled this made the optimizer accept 53-57/3/24-25 mmHg
  (perfect diastolic, systolic and mean 20+ mmHg off). The likelihood weights
  each target by its measurement error and is a statistically interpretable
  fit statistic (chi-square); restarts keep unit weights so it stays one.
- **Assumptions**: independent Gaussian measurement errors; sigma 2 mmHg is a
  typical fluid-filled catheter accuracy. Cath pressures and MRI flow are
  assumed to describe the same physiological state.
- **Stopping and gate**: the Nelder-Mead target stop and the iteration gate
  accept metrics within `target_sigma` (1) standard deviations.
- **If it fails**: report chi2 with the per-target errors; consider wider sigma
  for a target known to be unreliable rather than dropping it.

### 2.2 Diastolic target kept (`keep_diastolic_target: true`)
- **Why**: under PR the MPA pressure falls below the outlet pressure in
  diastole (backflow drains compliance), so a diastolic target below `Pd` is
  reachable and informative; the historical rule dropped it.
- **Checked**: the model's negative diastolic in TST-STAN-5 is not a catheter
  filtering artifact (it lasts ~190 ms during peak backflow and survives 6-20 Hz
  low-pass filtering within 0.2 mmHg).

### 2.3 Tree outlet pressure (`wedge_pressure_policy`)
The trees end at pre-capillary arterioles (r_min 50 um), so their constant
outlet pressure represents the pressure downstream of the arterial tree.
- **Regurgitant patients: `precapillary_fraction`**,
  `Pd = PCWP + 0.332 (mPAP - PCWP)`. TST-STAN-5: 7 + 0.332 x 9 = 9.99 mmHg. It
  raises mean and diastolic without adding resistance (hence without raising
  the pulse): chi2 33.5 -> 17.3 versus `Pd = PCWP`. The 0.332 is an empirically
  fitted outlet pressure, **not** Dong et al. 2021's partition (see below).
- **Non-regurgitant patients: `diastolic_offset`**, `Pd = PA diastolic - 2 mmHg`.
  Without backflow, a constant `Pd` must stay below the PA diastolic or the
  diastolic target is unreachable: the downstream fraction must be below
  DPG/TPG = (dia - LA) / (mean - LA), which 0.332 violates for TST-STAN-2
  (needs < 0.29) and TST-STAN-3 (< 0.12). With low end-diastolic flow the
  arterial tree carries little pressure drop, so PA diastolic approximates the
  downstream pressure (the reasoning behind the diastolic pulmonary gradient);
  `Pd = dia` exactly would make the target an asymptote and push compliance to
  its stiff bound, hence the 2 mmHg offset (a typical normal gradient). It
  needs no PCWP (three patients lack one).
- **Assumptions**: a constant (mean) outlet pressure, whereas real
  pre-capillary pressure pulsates; PCWP approximates LA. The implied
  downstream fraction for non-regurgitant patients varies (0.29 TST-STAN-2,
  0.12 TST-STAN-3).
- **Dong et al. 2021 / Raj & Chen 1986**: the pulmonary arteries (MPA to
  pre-capillary arterioles) carry 33.2 +/- 2% of PVR, the capillaries and veins
  66.8%. Dong et al. used this only for a steady, time-averaged analysis
  (sizing distal arterial trees, with a mean capillary pressure ~11.7 mmHg);
  their pulsatile 3D model used RCR outlets with distal pressure = LA. As a
  constant outlet pressure the partition gives PCWP + 0.668 (mPAP - PCWP)
  (TST-STAN-5 13.0, TST-STAN-1 18.0, TST-STAN-2 20.4, TST-STAN-3 23.4,
  TST-STAN-9 11.7 mmHg), 6-14 mmHg above PA diastolic for every patient, which
  the model can only reach with backflow. The partition describes mean
  pressures, not a constant boundary pressure.
- **Leaf resistance (evaluated 2026-10-06, not adopted)**: the pulsatile form
  of the partition is `Pd = PCWP` with the capillary + venous resistance at
  the tree leaves (`leaf_resistance: {downstream_fraction: f}`, requires
  `wedge_pressure_policy: measured`). TST-STAN-5, same budget (250 shared +
  40 per-cap evaluations), published model:

  | case | Pd | sys/dia/mean (34/3/16) | split | chi2 |
  |---|---|---|---|---|
  | `precapillary_fraction` 0.332 | 9.99 | 36.8/-4.7/14.6 | 0.875 | 17.2 |
  | leaf `f` = 0.668 | 7.0 | 46.0/-4.3/17.9 | 0.871 | 50.6 |
  | leaf `f` = 0.90 | 7.0 | 54.0/6.6/28.6 (shared stage) | 0.837 | ~145 |

  The leaf resistance sits behind the tree compliance and multiplies each
  tree's resistance by 1/(1 - f); the trees drain too slowly to absorb the
  regurgitant flow swing, so pulse pressure rises (41 -> 50 mmHg) while the
  54% backflow still drains diastolic. The fit is bound-limited in every case
  (xi = 3, k3 at 5e4), so softer or lower-resistance trees are not available
  within the cited ranges. With `f` = 0.668 the model's PVR splits 46%
  arterial / 54% capillary + venous (the proximal 3D-domain vessels are
  arterial too). For TST-STAN-5 the data favor a high outlet pressure with
  little resistance below the trees. The leaf resistance may behave
  differently without regurgitation (a longer RC decay raises diastolic); that
  is untested.
- **Alternatives**: the Gaar equation (capillary pressure 40% of the way from
  LA to mPAP; isolated dog lungs) places the pre-capillary pressure even higher
  and, as a constant outlet, is infeasible for more patients.

### 2.4 Proximal compliance (`proximal_compliance.wall_ehr: 5e4`)
- **What**: every rigid seed vessel gets `C = 3 A L / (2 Eh/r)`, the same
  linear thin-wall law the trees use, with Eh/r = 5e4 dyn/cm^2 (Krenz & Dawson,
  as cited by Qureshi et al. 2014). TST-STAN-5: 0.63 mL/mmHg over 15.6 mL of lumen.
- **Why**: the impedance in front of the compliance must stay below
  ~140 dyn s/cm^5 (pulse pressure / flow swing). The seed's ~122 dyn s/cm^5
  plus the characteristic impedance of control-stiffness tree roots
  (~100-185) exceeds it, so with a rigid seed the backflow cannot drain the
  tree compliance and diastolic goes strongly negative (chi2 267). Proximal
  compliance sits in front of the seed resistance: chi2 267 -> 61 (Eh/r 1e5)
  -> 35 (5e4). It is also physical: the 3D model the seed stands in for has
  deformable walls.
- **Assumptions**: linear compliance over the full pulse (at 31 mmHg pulse it
  already implies ~60-125% area change, so softer walls are not defensible
  without data); cm-g-s seed with `geometric_params`. Raw learnedZeroD seeds
  fold branch R and L into junctions and leave zero-length
  `branch{N}_..._connectorEL` vessels; these take the branch length from
  `outlet_mapping_centerline` (minus the branch's split connectors), which
  reproduces the calibrated seed's geometry (TST-STAN-5: 15.64 mL, 0.626 mL/mmHg).
  The branch's junction-outlet R and L move onto that vessel (the same model
  without C); a compliant R = L = 0 vessel in front of an IMPEDANCE outlet
  makes svZeroDSolver's Newton iteration fail at every evaluation. learnedZeroD
  can also fit negative junction-outlet L (a surrogate loss term, not an
  inertance); on a compliant vessel it is unstable, so vessels that receive
  proximal compliance have L < 0 set to 0 (`negative_l_clamped` in the
  summary; TST-STAN-1: two outlets, L = -16.8 and -10.6).
- **Consistency with the 3D wall**: the deformable 3D wall was softened
  (2026-10-06) from E 2.5e6 to 1.375e5 dyn/cm^2 at h 0.2 cm so that its total
  compliance matches the 0D proximal compliance. A uniform wall over the seed
  vessels has total compliance sum(3 A L r / (2 E h)), which equals
  sum(3 A L / (2 wall_ehr)) when E h = wall_ehr * r_eff, with
  r_eff = sum(A L r) / sum(A L) (TST-STAN-5: 0.55 cm, E h = 2.75e4 dyn/cm).
  The uniform wall places relatively more compliance in the largest vessels
  (Eh/r 3.5e4 in the MPA to 1.5e5 in the smallest branches). Each run reports
  r_eff and the matched E h in `tuning_diagnostics.json`
  (`proximal_compliance.matched_uniform_wall_eh`), and svzt-agent logs the
  ratio of the configured 3D wall to it. The large linear strain noted under
  Assumptions now applies to the 3D wall motion too, which exceeds the
  coupled-momentum method's small-strain assumption.
- **Later iterations**: compliance already present (calibrated against 3D) is
  kept; only rigid vessels are filled.

### 2.5 Tree wall stiffness (Olufsen form)
`Eh/r = k1 exp(k2 r) + k3`.
- **k1 = 2.6e5 dyn/cm^2, k2 = -14 cm^-1, fixed** (pulmonary small arteries,
  Colebank 2021 / Bartolo et al. 2022), replacing the systemic k1 = 2e7.
  k1 and k2 are not identifiable from pulmonary pressure data (Paun et al.
  2020); tuning k2 in [-50, -10] went to the bound for 0.3 mmHg.
- **k3: the single compliance knob, free in [5e4, 4e5] (log-scaled)**, the
  control pulmonary range (Krenz & Dawson to recent control large-artery values).
  It sets large-vessel stiffness, which holds most of the compliance, and pulse
  pressure varies 15-58 mmHg across the cohort.
- **Alternative: k3 fixed at 1e5** (Colebank/Bartolo). In TST-STAN-5 it cost
  +1.1 chi2 and put total compliance inside the SV/PP bracket (k3 free went to
  its 5e4 floor, compliance 1.8 vs <= 1.44 mL/mmHg). Prefer it when the free
  k3 sits at a bound or compliance leaves the bracket.
- **Rejected**: k3 = 0 with free k2 (historical). It fits (chi2 2-3) only with
  tree-root Eh/r 60-7,500 dyn/cm^2, k2 ~ -65 to -71 (literature -14 to -22.5),
  and 31-35 mL/mmHg of tree compliance (~25x the SV/PP estimate).

### 2.6 Morphometry: xi in [2.33, 3.0], eta_sym in [0.3, 1.0], per side
- **xi** (`r_p^xi = r_1^xi + r_2^xi`): 3 is Murray's law, 2.33-3 Uylings'
  theoretical range; measured pulmonary values are 2.76 (Huang 1996 casts,
  used by Olufsen 2012) and 2.1-2.6 (image-based, Miles et al. 2025). The
  bound also prevents the optimizer from growing trees past the node budget
  (unbounded xi -> 6 exploited truncation in an early study). xi > 3 improved
  TST-STAN-5 only modestly (chi2 18.3 -> 13.5 at 3.25) at the cost of
  compliance outside the SV/PP bracket.
- **eta_sym** (beta/alpha): Olufsen gamma 0.41 (eta 0.64); mouse micro-CT
  alpha 0.883 +/- 0.122, beta 0.666 +/- 0.141 (Chambers et al. 2020; mean
  eta ~0.75, spread to 1); 1.0 = symmetric. eta matters strongly below ~0.85
  (tree resistance -18%, compliance -31% from 0.6 to 0.84 at xi 3) and barely
  within 0.85-1.0, so it is constrained but not pinned there.
- **Per side**: xi and eta per PA set side resistance and the LPA/RPA split.

### 2.7 lrr = 10, d_min = 0.01 cm
- Olufsen et al. 2012 fit `l = 12.4 r^1.10` to Huang 1996 (lrr ~7-13 over these
  radii); mouse lrr ~13.4 (Chambers 2020). lrr 13 was worse at the
  pre-capillary outlet (extra resistance raises systolic); 15-20 helped only at
  `Pd = PCWP` and is above the morphometric range.
- r_min 50 um (d_min 0.01 cm) for pulmonary trees (Olufsen 2012).

### 2.8 Trees at measured cap diameters (`diameter_scale: 1.0`), 1M nodes
- No arbitrary shrinkage toward the mean; fit consistently better than 0.1
  (chi2 ~16-17 vs 49-56).
- Oversized caps (TST-STAN-5 `rpa_2`, d 0.854 cm) exceed 1M nodes and are
  truncated and flagged; 2M nodes changed diastolic by 0.25 mmHg.
- The impedance frequency chunk is sized automatically (~256 MB per array):
  identical results, 1.7x faster and far less memory for large trees.

### 2.9 Optimizer: shared trees, then a per-cap polish
- **Shared conductance-matched objective trees** (one per side) are fast but
  carry ~77% of the per-cap compliance at `diameter_scale` 1.0, so the
  optimizer misreports the published fit by ~2.6 mmHg.
- **Per-cap polish** (`polish.maxfev: 100`) re-tunes on the published trees
  from the shared optimum: same published fit (chi2 17.26 vs 17.22) when the
  optimum is bound-limited, at ~9x cost per evaluation (73 vs 19 min wall).
  It is essential if an optimum moves off the bounds.
- **Published scoring**: the exported model is always simulated once and
  reported (`published_fit`), so reported and exported fits agree.
- **Robustness**: 7/8 Sobol multi-starts reached the same optimum in
  TST-STAN-5 (chi2 17.2-17.6).

## 3. Validation and diagnostics (`tuning_diagnostics.json`)
- **SV/PP compliance bracket**: total compliance (tree static + proximal)
  should lie between regurgitant volume / PP (lower) and forward stroke volume
  / PP (upper). TST-STAN-5: 0.77-1.44 mL/mmHg; a Windkessel fit agrees
  (C >= ~1 needed; compliance is bounded below, not identified, by the targets).
- **Parameters at bounds**: flags free parameters within 1% of a bound; a
  bound-limited fit means the data want values beyond the cited range.
- **Truncation**: number and names of truncated trees.
- **Published vs optimizer fit** with per-target errors and chi2. A failure
  to score the published model is recorded as `published_fit.error`; the rest
  of the report is still written.
- **Inflow consistency**: mean flow of the exported model (which keeps the
  seed's inflow) vs the `inflow.csv` mean the optimizer used; more than 1%
  apart means the published and optimizer fits were computed at different
  flows (TST-STAN-5: 0.19%).

## 4. Assumptions (summary)
1. Prescribed MRI inflow (including backflow) describes the cath state.
2. Gaussian, independent measurement errors (2 mmHg, 0.02).
3. Linear thin-wall compliance (proximal and trees) over the full pulse.
4. Constant outlet pressure at the arteriolar level.
5. Literature morphometry and stiffness apply to these pediatric patients.
6. The calibrated seed's proximal resistance is correct.
7. The structured tree's frequency-domain impedance (linear, periodic) is
   adequate for the distal bed.

## 5. Known limitations (TST-STAN-5)
- Best defensible fit 36.8/-4.7/14.6 mmHg, split 0.875 (chi2 ~17): systolic,
  mean and split within measurement error, diastolic ~8 mmHg low under 54% PR.
- The fit is bound-limited (xi = 3, k3 at its floor): the data want larger or
  softer trees than the cited ranges allow; roughly two parameter combinations
  are identified (Jacobian singular values 32.7, 7.3, 0.17, 0.003).
- TST-STAN-1's model inflow is TST-STAN-5's waveform rescaled; inflow
  provenance must be confirmed per patient.
- Cath flow (2.79 L/min) differs from MRI net flow (1.55 L/min) in TST-STAN-5;
  tree resistance scales with the flow used.

## 6. If the model does not fit a patient: what to change

Check `tuning_diagnostics.json` first (published errors, bound flags,
compliance vs bracket, truncation), then:

| symptom | likely cause | change (most defensible first) |
|---|---|---|
| Diastolic far too low (regurgitant patient) | proximal impedance too high / proximal compliance too low | measure proximal stiffness from cine MRI area change (Eh/r ~ 1.5 PP / (dA/A)); recalibrate the seed's proximal resistance against 3D; model the inlet as RV + regurgitant valve instead of prescribed backflow; report as a limitation |
| Diastolic too high / pulse too small (no PR) | too much compliance or Pd too high | fix k3 at 1e5 or allow stiffer k3 (pulmonary-hypertension range up to ~8e6 for high-pressure patients, e.g. TST-STAN-3); lower `wall_ehr` stiffness only with data; check `diastolic_offset_mmhg` |
| Systolic too high | proximal / root impedance too high | as for low diastolic; check k3 at its upper bound |
| Mean off with xi or eta at a bound | total resistance out of range | check the flow source (MRI vs cath; consider rescaling the inflow to cath flow); lrr within 7-13; d_min; outlet policy |
| Split off | side resistance | free eta per side (already); check outlet mapping and split source |
| k3 at a bound or compliance outside bracket | compliance not identified | fix k3 at 1e5; add a compliance prior (future work) |
| Many truncated trees | very large caps / xi near 3 | raise `tree_max_nodes` (memory ~5 GB per 1M-node tree); `diameter_std_cap` for outlier caps |
| Published fit far from optimizer fit | surrogate bias | keep `polish`; or use per-cap objective trees throughout |
| Solver non-convergence | stiff proximal C / impedance coupling | published scoring already relaxes the tolerance; check negative calibrated R; reduce proximal compliance |
| Precapillary policy infeasible | Pd >= mean or missing PCWP | use `diastolic_offset`, or a measured PCWP |

### Future model changes (in priority order)
1. Patient-measured proximal stiffness (MRI pulsatility) instead of 5e4.
2. Time-resolved targets (cath pressure trace, LPA/RPA flow waveforms) to
   constrain where compliance sits.
3. RV + regurgitant valve inlet for PR patients.
4. Test `leaf_resistance` (Pd = PCWP + capillary/venous leaf resistance) on
   patients without regurgitation; if it fits there, it would replace the
   two outlet policies with one physiological rule (it worsened TST-STAN-5,
   §2.3).
5. Seed recalibration against 3D with the current BCs; 3D wall consistent with
   the 0D proximal compliance.
6. A compliance-matched shared surrogate (conductance and compliance) to keep
   shared-tree cost with per-cap fidelity.
7. Bayesian calibration (posterior over xi, eta, k3) instead of point estimates.
8. Nonlinear (strain-stiffening) wall law for large pulse pressures.
9. Pediatric / age-scaled morphometry; multi-patient validation.

## 7. Adaptation starts from the tuned model

Adaptation (`adaptation/workflow.py`; contract under **Adaptation** in
[`interface.md`](interface.md)) must start from the model that
produced the preop fit, otherwise the postop prediction mixes a different
distal bed into the comparison:
- **Trees**: the per-cap trees are rebuilt from the tuned config's `trees`
  metadata (xi, eta_sym, k1/k2/k3, lrr, d_min, measured cap diameter,
  `max_nodes` 1M), not from one LPA/RPA tree at the `optimized_params.csv`
  diameter (the earlier behavior, at a 100k default budget).
- **Pairing**: caps and BCs come from each tree's outlet mapping, checked
  against `outlet_cap_mapping.json` and the postop coupler's surfaces. The
  earlier list-position pairing could put an LPA impedance on an RPA cap
  with centerline-ordered BCs.
- **Pd**: the tuned IMPEDANCE `Pd` (from `wedge_pressure_policy`), not the
  legacy `min(wedge, diastolic)` (NaN without a measured wedge).

**M2 (default)** keeps its territory model: the LPA and RPA scale factors from
the 3D preop/postop flows and resistances are applied to every per-cap tree
on that side, so all caps of a side dilate (or constrict) by the same factor.
This is the direct generalization of the two-tree update; it assumes the
stimulus (flow and resistance change) is uniform within a territory.

**Open modeling decisions** (need the modeler's input):
1. *Territory vs per-cap stimulus for M2.* The 3D results give each cap's
   own preop and postop flow, so a per-cap update
   (`(Q_cap,post/Q_cap,pre)^(wss_gain/3)`) is possible; a stent changes the
   distribution within a side (e.g. lobar), which the territory update
   ignores. The IMS term needs a per-cap pressure drop (MPA to cap), which
   is not defined yet. `territory_scheme` is the intended switch.
2. *M1/M3 with per-cap trees.* Both integrate an ODE on one LPA and one RPA
   tree inside the reduced-order PA (RRI) model, and the RRI step assumes the
   5-vessel reduced PA, so neither can take the full-PA tuned config; they
   now refuse per-cap tuned models. Options: (a) a side surrogate tree
   (conductance-matched, as the tuning's shared stage) in the reduced PA,
   with the converged side result transferred to the per-cap trees as a
   uniform scale (CWSS equilibrium is a uniform scale (Q_post/Q_pre)^(1/3)
   when the flow fractions within a side do not change); (b) all per-cap trees in the ODE state
   (side BC = their parallel resistance), which needs the integrator to stop
   storing the full state history (23 x up to 1M vessels per step).
3. *Adapted geometry in 3D for M1/M3.* Their per-vessel radii cannot be
   stored in tree metadata, and the coupled-timing kernel regeneration
   rebuilds trees from metadata, so their adapted radii do not reach the 3D
   simulation. M2 stores its uniform factor (`adapted_diameter_scale`).

## References
- Olufsen et al. 2012, J Fluid Mech (pulmonary structured trees):
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3433075/
- Qureshi et al. 2014 (cites Krenz & Dawson 2003):
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4183203/
- Bartolo et al. 2022 (k1, k2, k3 from Colebank 2021):
  https://link.springer.com/article/10.1007/s10237-021-01538-1
- Paun et al. 2020 (stiffness identifiability):
  https://pmc.ncbi.nlm.nih.gov/articles/PMC7811590/
- Miles et al. 2025, Revisiting Murray's law in pulmonary arteries:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12834150/
- Chambers et al. 2020 (murine pulmonary morphometry):
  https://arxiv.org/html/2002.08416
- Multiscale pulmonary model 2026: https://arxiv.org/html/2607.19269
- Dong ML, Lan IS, Yang W, Rabinovitch M, Feinstein JA, Marsden AL (2021).
  Computational simulation-derived hemodynamic and biomechanical properties of
  the pulmonary arterial tree early in the course of ventricular septal
  defects. Biomech Model Mechanobiol 20:2471-2489.
  https://doi.org/10.1007/s10237-021-01519-4 (pulmonary arteries, MPA to
  pre-capillary arterioles, = 33.2 +/- 2% of PVR; used to size steady distal
  arterial trees, while the pulsatile 3D model used RCR outlets with distal
  pressure = LA).
- Raj JU, Chen P (1986). Micropuncture measurement of microvascular pressures
  in isolated lamb lungs during hypoxia. Circ Res 59(4):398-404.
  https://doi.org/10.1161/01.RES.59.4.398 (source of Dong et al.'s 33.2%).
- Cited via the above: Murray 1926; Uylings 1977; Huang et al. 1996;
  Krenz & Dawson 2003; Colebank et al. 2021; Gaar et al. 1967.
