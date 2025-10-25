from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..pa_config import PAConfig
from ..clinical_targets import ClinicalTargets
from ..utils import write_to_log
from ...io.config_handler import ConfigHandler
from ...io.blocks import BoundaryCondition
from ...microvasculature import TreeParameters
from ...microvasculature.compliance import (
    ComplianceModel,
    ConstantCompliance,
    OlufsenCompliance,
)

logger = logging.getLogger(__name__)


OLUFSEN_BASE = {"k1": 1.99925e7, "k2": -25.5267, "k3": 1.104531490909e6}
DEFAULT_CONSTANT_EH_OVER_R = 1.0e5
CONSTANT_EH_RANGE = (5.0e4, 1.5e5)
METRIC_ORDER: List[Tuple[str, str]] = [
    ("mpa_systolic", "MPA systolic (mmHg)"),
    ("mpa_diastolic", "MPA diastolic (mmHg)"),
    ("mpa_mean", "MPA mean (mmHg)"),
    ("lpa_split", "LPA flow split"),
    ("rpa_split", "RPA flow split"),
]


@dataclass
class SensitivityOptions:
    """Container for sensitivity sweep configuration."""

    alpha_values: np.ndarray
    beta_values: np.ndarray
    lrr_values: np.ndarray
    constant_eh_over_r: np.ndarray
    olufsen_k_ranges: Dict[str, np.ndarray]
    d_min: Optional[float] = None
    n_procs: int = 1


def generate_even_resistance_pa_config(
    config_path: Path | str,
    *,
    output_path: Optional[Path | str] = None,
    clinical_targets: Optional[ClinicalTargets] = None,
    is_pulmonary: bool = True,
) -> PAConfig:
    """
    Construct a reduced pulmonary artery configuration with balanced distal resistances.

    Parameters
    ----------
    config_path:
        Path to the baseline 0D configuration JSON.
    output_path:
        Optional path to persist the balanced PA config JSON.
    clinical_targets:
        Optional clinical target overrides. Defaults to nominal pulmonary values.
    is_pulmonary:
        Flag forwarded to ConfigHandler for correct anatomy handling.

    Returns
    -------
    PAConfig
        Prepared PA configuration with LPA/RPA outlets using identical resistance Bcs.
    """

    config_path = Path(config_path)
    config_handler = ConfigHandler.from_json(str(config_path), is_pulmonary=is_pulmonary)

    targets = clinical_targets or ClinicalTargets(
        wedge_p=5.0, mpa_p=[50.0, 10.0, 25.0], rpa_split=0.5
    )

    pa_config = _build_pa_config_with_even_resistance(config_handler, targets)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pa_config.to_json(str(output_path))

    return pa_config


def run_structured_tree_sensitivity(
    config_path: Path | str,
    *,
    output_dir: Path | str,
    clinical_targets: Optional[ClinicalTargets] = None,
    alpha_range: Optional[Iterable[float]] = None,
    beta_range: Optional[Iterable[float]] = None,
    lrr_range: Optional[Iterable[float]] = None,
    constant_eh_over_r: Optional[Iterable[float]] = None,
    olufsen_k_ranges: Optional[Dict[str, Iterable[float]]] = None,
    d_min: Optional[float] = None,
    n_procs: int = 1,
    log_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """
    Execute a one-at-a-time sensitivity sweep over structured tree parameters.

    The analysis perturbs alpha, beta, and l_rr for both the LPA and RPA trees,
    compares constant versus Olufsen compliance models, and varies the Olufsen
    compliance coefficients (k1, k2, k3). For each perturbation the routine
    records MPA systolic/diastolic/mean pressure along with the LPA/RPA flow split,
    and computes finite-difference sensitivity coefficients relative to the baseline.

    Parameters
    ----------
    config_path:
        Path to the baseline 0D JSON configuration.
    output_dir:
        Directory where CSV outputs and plots are written.
    clinical_targets:
        Optional clinical target overrides.
    alpha_range, beta_range:
        Sequences used for alpha/beta sweeps. Defaults to 0.1–0.8 with 0.02 spacing.
    lrr_range:
        Sequence for l_rr sweep. Defaults to 2–20 with step 2.
    constant_eh_over_r:
        Sequence of Eh/r values for constant compliance sensitivities. Defaults to
        [5e4, 7.5e4, 1e5, 1.25e5, 1.5e5].
    olufsen_k_ranges:
        Dictionary with keys ``k1``, ``k2``, ``k3`` providing sweep values. Defaults
        to ±50% around the canonical Olufsen coefficients.
    d_min:
        Optional minimum diameter used to build both structured trees. Defaults to
        10% of the distal vessel diameter extracted from the configuration.
    n_procs:
        Number of worker processes forwarded to the impedance computation.
    log_path:
        Optional log file path for progress updates.

    Returns
    -------
    pandas.DataFrame
        Tabulated sensitivity results for downstream analysis.
    """

    options = _assemble_options(
        alpha_range=alpha_range,
        beta_range=beta_range,
        lrr_range=lrr_range,
        constant_eh_over_r=constant_eh_over_r,
        olufsen_k_ranges=olufsen_k_ranges,
        d_min=d_min,
        n_procs=n_procs,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if log_path:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        write_to_log(str(log_path), "Starting structured tree sensitivity analysis\n")

    targets = clinical_targets or ClinicalTargets(
        wedge_p=5.0, mpa_p=[50.0, 10.0, 25.0], rpa_split=0.5
    )

    config_handler = ConfigHandler.from_json(str(config_path), is_pulmonary=True)

    mean_outlet_resistance = _mean_outlet_resistance(config_handler)

    def pa_config_factory() -> PAConfig:
        return _build_pa_config_with_even_resistance(
            config_handler, targets, mean_outlet_resistance
        )

    baseline_pa_config = pa_config_factory()
    even_config_path = output_dir / "pa_config_even_resistance.json"
    baseline_pa_config.to_json(str(even_config_path))

    baseline_tree_params = _baseline_tree_parameters(
        baseline_pa_config,
        options.d_min,
    )

    records: List[Dict[str, float]] = []

    for compliance_model_name, branch_params in baseline_tree_params.items():
        baseline_metrics = _simulate_metrics(
            pa_config_factory,
            branch_params["lpa"],
            branch_params["rpa"],
            options.n_procs,
        )

        if log_path:
            write_to_log(
                str(log_path),
                f"Baseline metrics for {compliance_model_name}: {baseline_metrics}\n",
            )

        # Geometric parameters (alpha, beta, l_rr)
        for param_name, sweep_values in (
            ("alpha", options.alpha_values),
            ("beta", options.beta_values),
            ("lrr", options.lrr_values),
        ):
            for branch in ("lpa", "rpa"):
                base_param_value = getattr(branch_params[branch], param_name)
                for value in sweep_values:
                    lpa_params = _clone_tree_params(branch_params["lpa"])
                    rpa_params = _clone_tree_params(branch_params["rpa"])

                    target_params = lpa_params if branch == "lpa" else rpa_params
                    setattr(target_params, param_name, float(value))

                    metrics = _simulate_metrics(
                        pa_config_factory, lpa_params, rpa_params, options.n_procs
                    )
                    _append_record(
                        records,
                        parameter=param_name,
                        branch=branch,
                        compliance_model=compliance_model_name,
                        value=float(value),
                        baseline_value=float(base_param_value),
                        metrics=metrics,
                        baseline_metrics=baseline_metrics,
                    )

        # Compliance perturbations
        if compliance_model_name == "constant":
            for branch in ("lpa", "rpa"):
                base_value = float(
                    branch_params[branch].compliance_model.value  # type: ignore[attr-defined]
                )
                for value in options.constant_eh_over_r:
                    lpa_params = _clone_tree_params(branch_params["lpa"])
                    rpa_params = _clone_tree_params(branch_params["rpa"])

                    target_params = lpa_params if branch == "lpa" else rpa_params
                    target_params.compliance_model = ConstantCompliance(value)

                    metrics = _simulate_metrics(
                        pa_config_factory, lpa_params, rpa_params, options.n_procs
                    )
                    _append_record(
                        records,
                        parameter="constant_Eh_over_r",
                        branch=branch,
                        compliance_model=compliance_model_name,
                        value=float(value),
                        baseline_value=base_value,
                        metrics=metrics,
                        baseline_metrics=baseline_metrics,
                    )
        elif compliance_model_name == "olufsen":
            for olufsen_param, sweep_values in options.olufsen_k_ranges.items():
                for branch in ("lpa", "rpa"):
                    base_model: OlufsenCompliance = branch_params[
                        branch
                    ].compliance_model  # type: ignore[assignment]
                    base_value = float(getattr(base_model, olufsen_param))
                    for value in sweep_values:
                        lpa_params = _clone_tree_params(branch_params["lpa"])
                        rpa_params = _clone_tree_params(branch_params["rpa"])

                        target_params = lpa_params if branch == "lpa" else rpa_params
                        new_model = OlufsenCompliance(
                            k1=float(value) if olufsen_param == "k1" else base_model.k1,
                            k2=float(value) if olufsen_param == "k2" else base_model.k2,
                            k3=float(value) if olufsen_param == "k3" else base_model.k3,
                        )
                        target_params.compliance_model = new_model

                        metrics = _simulate_metrics(
                            pa_config_factory, lpa_params, rpa_params, options.n_procs
                        )
                        _append_record(
                            records,
                            parameter=f"olufsen_{olufsen_param}",
                            branch=branch,
                            compliance_model=compliance_model_name,
                            value=float(value),
                            baseline_value=base_value,
                            metrics=metrics,
                            baseline_metrics=baseline_metrics,
                        )

    results_df = pd.DataFrame.from_records(records)

    if results_df.empty:
        logger.warning("Sensitivity analysis produced no results.")
        return results_df

    csv_path = output_dir / "structured_tree_sensitivity.csv"
    results_df.to_csv(csv_path, index=False)

    _plot_sensitivity(results_df, output_dir)

    if log_path:
        write_to_log(str(log_path), "Structured tree sensitivity analysis completed\n")

    return results_df


# --------------------------------------------------------------------------- #
# Latin Hypercube sensitivity analysis
# --------------------------------------------------------------------------- #


def run_structured_tree_lhs_sensitivity(
    config_path: Path | str,
    *,
    output_dir: Path | str,
    clinical_targets: Optional[ClinicalTargets] = None,
    num_samples: int = 60,
    random_seed: Optional[int] = None,
    d_min: Optional[float] = None,
    n_procs: int = 1,
) -> pd.DataFrame:
    """
    Perform a Latin Hypercube Sampling (LHS) sensitivity study across the structured
    tree geometry and compliance parameters for the LPA and RPA independently.
    The sampler perturbs ``alpha``, ``beta``, and ``l_rr`` separately for each branch,
    toggles branch-specific compliance models (constant vs Olufsen), and sweeps the
    associated compliance parameters (Eh/r for constant models, k1/k2/k3 for Olufsen).

    Outputs written to ``output_dir`` include:
        - ``structured_tree_lhs_samples.csv`` with the raw parameter/metric table.
        - ``structured_tree_lhs_sensitivity.csv`` containing absolute parameter/metric
          correlations (rows differentiate LPA/RPA parameters).
        - ``structured_tree_lhs_heatmap.png`` visualizing the combined sensitivities.

    Parameters
    ----------
    config_path:
        Path to the baseline 0D configuration JSON.
    output_dir:
        Destination directory for CSV files and plots.
    clinical_targets:
        Optional override for pulmonary clinical targets.
    num_samples:
        Number of Latin hypercube samples (default: 60).
    random_seed:
        Seed forwarded to the RNG for reproducibility.
    d_min:
        Optional override for minimum terminal diameter in the structured trees.
    n_procs:
        Worker count passed to impedance computations.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed=random_seed)

    targets = clinical_targets or ClinicalTargets(
        wedge_p=5.0, mpa_p=[50.0, 10.0, 25.0], rpa_split=0.5
    )

    config_handler = ConfigHandler.from_json(str(config_path), is_pulmonary=True)
    mean_outlet_resistance = _mean_outlet_resistance(config_handler)

    def pa_config_factory() -> PAConfig:
        return _build_pa_config_with_even_resistance(
            config_handler, targets, mean_outlet_resistance
        )

    baseline_pa_config = pa_config_factory()
    baseline_tree_params = _baseline_tree_parameters(
        baseline_pa_config,
        d_min,
    )

    lhs_samples = _latin_hypercube(num_samples, 16, rng)

    parameter_records: List[Dict[str, float]] = []

    for sample_idx, sample in enumerate(lhs_samples):
        alpha_lpa = _map_linear(sample[0], 0.1, 0.8)
        beta_lpa = _map_linear(sample[1], 0.1, 0.8)
        lrr_lpa = _map_linear(sample[2], 2.0, 20.0)

        alpha_rpa = _map_linear(sample[3], 0.1, 0.8)
        beta_rpa = _map_linear(sample[4], 0.1, 0.8)
        lrr_rpa = _map_linear(sample[5], 2.0, 20.0)

        comp_selector_lpa = sample[6]
        comp_selector_rpa = sample[7]

        eh_lpa_raw = _map_linear(sample[8], CONSTANT_EH_RANGE[0], CONSTANT_EH_RANGE[1])
        eh_rpa_raw = _map_linear(sample[9], CONSTANT_EH_RANGE[0], CONSTANT_EH_RANGE[1])

        k1_lpa_val = OLUFSEN_BASE["k1"] * _map_linear(sample[10], 0.5, 1.5)
        k2_lpa_val = OLUFSEN_BASE["k2"] * _map_linear(sample[11], 0.5, 1.5)
        k3_lpa_val = OLUFSEN_BASE["k3"] * _map_linear(sample[12], 0.5, 1.5)

        k1_rpa_val = OLUFSEN_BASE["k1"] * _map_linear(sample[13], 0.5, 1.5)
        k2_rpa_val = OLUFSEN_BASE["k2"] * _map_linear(sample[14], 0.5, 1.5)
        k3_rpa_val = OLUFSEN_BASE["k3"] * _map_linear(sample[15], 0.5, 1.5)

        lpa_params = _clone_tree_params(baseline_tree_params["constant"]["lpa"])
        rpa_params = _clone_tree_params(baseline_tree_params["constant"]["rpa"])

        lpa_params.alpha = alpha_lpa
        lpa_params.beta = beta_lpa
        lpa_params.lrr = lrr_lpa
        rpa_params.alpha = alpha_rpa
        rpa_params.beta = beta_rpa
        rpa_params.lrr = lrr_rpa

        if comp_selector_lpa < 0.5:
            constant_indicator_lpa = 1.0
            lpa_params.compliance_model = ConstantCompliance(eh_lpa_raw)
            eh_lpa = eh_lpa_raw
            k1_lpa = 0.0
            k2_lpa = 0.0
            k3_lpa = 0.0
            compliance_label_lpa = f"Constant ({eh_lpa:.2e})"
        else:
            constant_indicator_lpa = 0.0
            lpa_params.compliance_model = OlufsenCompliance(
                k1_lpa_val,
                k2_lpa_val,
                k3_lpa_val,
            )
            eh_lpa = 0.0
            k1_lpa = k1_lpa_val
            k2_lpa = k2_lpa_val
            k3_lpa = k3_lpa_val
            compliance_label_lpa = "Olufsen"

        if comp_selector_rpa < 0.5:
            constant_indicator_rpa = 1.0
            rpa_params.compliance_model = ConstantCompliance(eh_rpa_raw)
            eh_rpa = eh_rpa_raw
            k1_rpa = 0.0
            k2_rpa = 0.0
            k3_rpa = 0.0
            compliance_label_rpa = f"Constant ({eh_rpa:.2e})"
        else:
            constant_indicator_rpa = 0.0
            rpa_params.compliance_model = OlufsenCompliance(
                k1_rpa_val,
                k2_rpa_val,
                k3_rpa_val,
            )
            eh_rpa = 0.0
            k1_rpa = k1_rpa_val
            k2_rpa = k2_rpa_val
            k3_rpa = k3_rpa_val
            compliance_label_rpa = "Olufsen"

        metrics = _simulate_metrics(
            pa_config_factory,
            lpa_params,
            rpa_params,
            n_procs,
        )

        record = {
            "sample_index": sample_idx,
            "alpha_lpa": alpha_lpa,
            "beta_lpa": beta_lpa,
            "lrr_lpa": lrr_lpa,
            "alpha_rpa": alpha_rpa,
            "beta_rpa": beta_rpa,
            "lrr_rpa": lrr_rpa,
            "constant_indicator_lpa": constant_indicator_lpa,
            "constant_indicator_rpa": constant_indicator_rpa,
            "eh_lpa": eh_lpa,
            "eh_rpa": eh_rpa,
            "k1_lpa": k1_lpa,
            "k2_lpa": k2_lpa,
            "k3_lpa": k3_lpa,
            "k1_rpa": k1_rpa,
            "k2_rpa": k2_rpa,
            "k3_rpa": k3_rpa,
            "compliance_label_lpa": compliance_label_lpa,
            "compliance_label_rpa": compliance_label_rpa,
        }
        record.update(metrics)
        parameter_records.append(record)

    if not parameter_records:
        logger.warning("Latin hypercube sensitivity produced no samples.")
        return pd.DataFrame()

    samples_df = pd.DataFrame(parameter_records)
    samples_path = output_dir / "structured_tree_lhs_samples.csv"
    samples_df.to_csv(samples_path, index=False)

    parameter_columns = [
        "alpha_lpa",
        "beta_lpa",
        "lrr_lpa",
        "alpha_rpa",
        "beta_rpa",
        "lrr_rpa",
        "eh_lpa",
        "eh_rpa",
        "k1_lpa",
        "k2_lpa",
        "k3_lpa",
        "k1_rpa",
        "k2_rpa",
        "k3_rpa",
    ]

    parameter_matrix = samples_df[parameter_columns].copy()
    metric_matrix = samples_df[[name for name, _ in METRIC_ORDER]].copy()

    parameter_z = _zscore(parameter_matrix.to_numpy())
    metric_z = _zscore(metric_matrix.to_numpy())

    correlation = (parameter_z.T @ metric_z) / (len(samples_df) - 1)
    sensitivity_df = pd.DataFrame(
        np.abs(correlation),
        index=parameter_columns,
        columns=[name for name, _ in METRIC_ORDER],
    )
    sensitivity_df["combined"] = sensitivity_df.mean(axis=1)

    sensitivity_path = output_dir / "structured_tree_lhs_sensitivity.csv"
    sensitivity_df.to_csv(sensitivity_path)

    _plot_heatmap_sensitivity(
        sensitivity_df,
        output_dir / "structured_tree_lhs_heatmap.png",
    )

    return samples_df


# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #


def _assemble_options(
    *,
    alpha_range: Optional[Iterable[float]],
    beta_range: Optional[Iterable[float]],
    lrr_range: Optional[Iterable[float]],
    constant_eh_over_r: Optional[Iterable[float]],
    olufsen_k_ranges: Optional[Dict[str, Iterable[float]]],
    d_min: Optional[float],
    n_procs: int,
) -> SensitivityOptions:
    alpha_values = (
        np.array(list(alpha_range), dtype=float)
        if alpha_range is not None
        else np.round(np.arange(0.1, 0.8001, 0.02), 5)
    )
    beta_values = (
        np.array(list(beta_range), dtype=float)
        if beta_range is not None
        else np.round(np.arange(0.1, 0.8001, 0.02), 5)
    )
    lrr_values = (
        np.array(list(lrr_range), dtype=float)
        if lrr_range is not None
        else np.arange(2.0, 20.1, 2.0)
    )
    constant_values = (
        np.array(list(constant_eh_over_r), dtype=float)
        if constant_eh_over_r is not None
        else np.array([5.0e4, 7.5e4, 1.0e5, 1.25e5, 1.5e5], dtype=float)
    )

    if olufsen_k_ranges is not None:
        k_ranges = {key: np.array(list(vals), dtype=float) for key, vals in olufsen_k_ranges.items()}
    else:
        k_ranges = {
            "k1": np.linspace(0.5 * OLUFSEN_BASE["k1"], 1.5 * OLUFSEN_BASE["k1"], 13),
            "k2": np.linspace(1.5 * OLUFSEN_BASE["k2"], 0.5 * OLUFSEN_BASE["k2"], 13),
            "k3": np.linspace(0.5 * OLUFSEN_BASE["k3"], 1.5 * OLUFSEN_BASE["k3"], 13),
        }

    return SensitivityOptions(
        alpha_values=alpha_values,
        beta_values=beta_values,
        lrr_values=lrr_values,
        constant_eh_over_r=constant_values,
        olufsen_k_ranges=k_ranges,
        d_min=d_min,
        n_procs=n_procs,
    )


def _mean_outlet_resistance(config_handler: ConfigHandler) -> float:
    resistances = []
    for outlet_name in ("LPA_BC", "RPA_BC"):
        bc = config_handler.bcs.get(outlet_name)
        if bc is None:
            raise KeyError(f"Boundary condition '{outlet_name}' missing from configuration.")
        if bc.type in ("RESISTANCE", "RCR"):
            resistances.append(float(bc.R))
        elif bc.type == "IMPEDANCE":
            resistances.append(float(bc.Z[0]))
        else:
            raise ValueError(f"Unsupported boundary condition type '{bc.type}' for {outlet_name}.")
    return float(np.mean(resistances))


def _build_pa_config_with_even_resistance(
    config_handler: ConfigHandler,
    targets: ClinicalTargets,
    mean_resistance: Optional[float] = None,
) -> PAConfig:
    pa_config = PAConfig.from_pa_config(config_handler, targets)
    if mean_resistance is None:
        mean_resistance = _mean_outlet_resistance(config_handler)

    for name in ("LPA_BC", "RPA_BC"):
        pa_config.bcs[name] = BoundaryCondition.from_config(
            {
                "bc_name": name,
                "bc_type": "RESISTANCE",
                "bc_values": {"R": mean_resistance, "Pd": targets.wedge_p * 1333.2},
            }
        )

    return pa_config


def _baseline_tree_parameters(
    pa_config: PAConfig,
    d_min_override: Optional[float],
) -> Dict[str, Dict[str, TreeParameters]]:
    lpa_diameter = float(pa_config.lpa_dist.diameter)
    rpa_diameter = float(pa_config.rpa_dist.diameter)
    default_d_min = 0.1 * min(lpa_diameter, rpa_diameter)
    d_min = d_min_override or default_d_min

    baseline: Dict[str, Dict[str, TreeParameters]] = {}

    constant_model = ConstantCompliance(DEFAULT_CONSTANT_EH_OVER_R)
    baseline["constant"] = {
        "lpa": _make_tree_params(
            name="lpa",
            diameter=lpa_diameter,
            d_min=d_min,
            lrr=10.0,
            alpha=0.8,
            beta=0.6,
            compliance_model=constant_model,
        ),
        "rpa": _make_tree_params(
            name="rpa",
            diameter=rpa_diameter,
            d_min=d_min,
            lrr=10.0,
            alpha=0.8,
            beta=0.6,
            compliance_model=ConstantCompliance(DEFAULT_CONSTANT_EH_OVER_R),
        ),
    }

    olufsen_model_lpa = OlufsenCompliance(**OLUFSEN_BASE)
    olufsen_model_rpa = OlufsenCompliance(**OLUFSEN_BASE)
    baseline["olufsen"] = {
        "lpa": _make_tree_params(
            name="lpa",
            diameter=lpa_diameter,
            d_min=d_min,
            lrr=10.0,
            alpha=0.8,
            beta=0.6,
            compliance_model=olufsen_model_lpa,
        ),
        "rpa": _make_tree_params(
            name="rpa",
            diameter=rpa_diameter,
            d_min=d_min,
            lrr=10.0,
            alpha=0.8,
            beta=0.6,
            compliance_model=olufsen_model_rpa,
        ),
    }

    return baseline


def _make_tree_params(
    *,
    name: str,
    diameter: float,
    d_min: float,
    lrr: float,
    alpha: float,
    beta: float,
    compliance_model: ComplianceModel,
) -> TreeParameters:
    if isinstance(compliance_model, OlufsenCompliance):
        k_params = (
            compliance_model.k1,
            compliance_model.k2,
            compliance_model.k3,
        )
    else:
        k_params = (None, None, None)

    return TreeParameters(
        name=name,
        lrr=lrr,
        diameter=diameter,
        d_min=d_min,
        alpha=alpha,
        beta=beta,
        compliance_model=compliance_model,
        k1=k_params[0],
        k2=k_params[1],
        k3=k_params[2],
    )


def _clone_tree_params(params: TreeParameters) -> TreeParameters:
    compliance_copy = _copy_compliance_model(params.compliance_model)
    clone = TreeParameters(
        name=params.name,
        lrr=params.lrr,
        diameter=params.diameter,
        d_min=params.d_min,
        alpha=params.alpha,
        beta=params.beta,
        compliance_model=compliance_copy,
        k1=params.k1,
        k2=params.k2,
        k3=params.k3,
    )
    return clone


def _copy_compliance_model(model: ComplianceModel) -> ComplianceModel:
    if isinstance(model, OlufsenCompliance):
        return OlufsenCompliance(model.k1, model.k2, model.k3)
    if isinstance(model, ConstantCompliance):
        return ConstantCompliance(model.value)  # type: ignore[attr-defined]
    return model.__class__(**model.params)


def _simulate_metrics(
    pa_config_factory,
    lpa_params: TreeParameters,
    rpa_params: TreeParameters,
    n_procs: int,
) -> Dict[str, float]:
    pa_config = pa_config_factory()
    pa_config.create_impedance_trees(lpa_params, rpa_params, n_procs=n_procs)
    pa_config.simulate()

    metrics = {
        "mpa_systolic": float(pa_config.P_mpa[0]),
        "mpa_diastolic": float(pa_config.P_mpa[1]),
        "mpa_mean": float(pa_config.P_mpa[2]),
        "rpa_split": float(pa_config.rpa_split),
        "lpa_split": float(1.0 - pa_config.rpa_split),
    }
    return metrics


def _append_record(
    records: List[Dict[str, float]],
    *,
    parameter: str,
    branch: str,
    compliance_model: str,
    value: float,
    baseline_value: float,
    metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
) -> None:
    delta_param = value - baseline_value
    relative_param = np.nan
    if not np.isclose(baseline_value, 0.0):
        relative_param = delta_param / baseline_value

    for metric_name, metric_value in metrics.items():
        baseline_metric = baseline_metrics[metric_name]
        delta_metric = metric_value - baseline_metric
        if np.isclose(baseline_metric, 0.0):
            relative_metric = np.nan
        else:
            relative_metric = delta_metric / baseline_metric

        if np.isclose(delta_param, 0.0) or np.isnan(relative_param):
            sensitivity = 0.0
        elif np.isclose(relative_param, 0.0):
            sensitivity = np.nan
        else:
            sensitivity = relative_metric / relative_param

        records.append(
            {
                "parameter": parameter,
                "branch": branch,
                "compliance_model": compliance_model,
                "parameter_value": value,
                "baseline_parameter_value": baseline_value,
                "delta_parameter": delta_param,
                "relative_parameter_change": relative_param,
                "metric": metric_name,
                "metric_value": metric_value,
                "baseline_metric_value": baseline_metric,
                "delta_metric": delta_metric,
                "relative_metric_change": relative_metric,
                "sensitivity": sensitivity,
            }
        )


def _plot_sensitivity(results: pd.DataFrame, output_dir: Path) -> None:
    parameter_names = sorted(results["parameter"].unique())

    for parameter in parameter_names:
        param_df = results[results["parameter"] == parameter]
        if param_df.empty:
            continue

        fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
        axes = axes.flatten()

        legend_handles = {}

        for idx, (metric_name, metric_label) in enumerate(METRIC_ORDER):
            ax = axes[idx]
            metric_df = param_df[param_df["metric"] == metric_name]
            if metric_df.empty:
                ax.set_visible(False)
                continue

            for compliance in sorted(metric_df["compliance_model"].unique()):
                for branch in ("lpa", "rpa"):
                    subset = metric_df[
                        (metric_df["compliance_model"] == compliance)
                        & (metric_df["branch"] == branch)
                    ].sort_values("parameter_value")
                    if subset.empty:
                        continue
                    label = f"{compliance}-{branch}"
                    line, = ax.plot(
                        subset["parameter_value"],
                        subset["sensitivity"],
                        marker="o",
                        label=label,
                    )
                    legend_handles[label] = line

            ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.75)
            ax.set_ylabel("Sensitivity (Δmetric/Δparameter)")
            ax.set_title(metric_label)

        axes[-1].set_xlabel("Parameter value")
        axes[-2].set_xlabel("Parameter value")

        if legend_handles:
            fig.legend(
                legend_handles.values(),
                legend_handles.keys(),
                loc="upper center",
                ncol=4,
                fontsize="small",
            )

        fig.suptitle(f"Sensitivity for {parameter}")
        fig.tight_layout(rect=(0, 0, 1, 0.92))

        figure_path = output_dir / f"{parameter}_sensitivity.png"
        fig.savefig(figure_path, dpi=200)
        plt.close(fig)


def _plot_heatmap_sensitivity(sensitivity_df: pd.DataFrame, figure_path: Path) -> None:
    """Render heatmap for parameter sensitivities."""

    n_rows, n_cols = sensitivity_df.shape
    width = max(8.0, 1.2 * n_cols)
    height = max(4.5, 0.5 * n_rows + 1.5)

    fig, ax = plt.subplots(figsize=(width, height))
    data = sensitivity_df.to_numpy()
    im = ax.imshow(data, cmap="viridis", aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(sensitivity_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(sensitivity_df.index)

    threshold = np.nanmax(data) / 2.0 if np.isfinite(np.nanmax(data)) else 0.0
    for i in range(n_rows):
        for j in range(n_cols):
            value = data[i, j]
            if np.isnan(value):
                text = "nan"
                color = "black"
            else:
                text = f"{value:.2f}"
                color = "white" if value >= threshold else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize="small")

    fig.colorbar(im, ax=ax, label="|Correlation|")
    ax.set_title("Latin Hypercube Sensitivity Heatmap")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)


def _latin_hypercube(samples: int, dimensions: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a Latin hypercube sample matrix on [0, 1]."""
    cut_points = np.linspace(0.0, 1.0, samples + 1)
    points = np.empty((samples, dimensions))
    for dim in range(dimensions):
        low = cut_points[:-1]
        high = cut_points[1:]
        points[:, dim] = rng.uniform(low, high)
        rng.shuffle(points[:, dim])
    return points


def _map_linear(value: float, out_min: float, out_max: float, *, in_min: float = 0.0, in_max: float = 1.0) -> float:
    """Linearly map a scalar from one interval to another."""
    span_in = in_max - in_min
    if span_in == 0:
        return out_min
    ratio = (value - in_min) / span_in
    return out_min + ratio * (out_max - out_min)


def _zscore(array: np.ndarray) -> np.ndarray:
    """Compute column-wise z-scores, safeguarding zero variance."""
    mean = np.mean(array, axis=0)
    std = np.std(array, axis=0, ddof=0)
    std[std == 0.0] = 1.0
    return (array - mean) / std
