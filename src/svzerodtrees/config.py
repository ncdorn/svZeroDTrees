import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml
import numpy as np

from .tune_bcs.tune_space import FreeParam, FixedParam, TiedParam, TuneSpace, identity, positive, unit_interval
from .microvasculature.treeparams import TreeParameters
from .microvasculature.compliance.constant import ConstantCompliance
from .microvasculature.compliance.olufsen import OlufsenCompliance

CONFIG_VERSION = 1

_CALIBRATION_TARGET_INTERFACES = {
    "external_upstream",
    "external_downstream",
    "internal",
    # These short forms are convenient in YAML and map to the corresponding
    # interface direction during target resolution.
    "upstream",
    "downstream",
}


@dataclass
class PathsConfig:
    root: str = "."
    zerod_config: Optional[str] = None
    clinical_targets: Optional[str] = None
    mesh_surfaces: Optional[str] = None
    preop_dir: Optional[str] = None
    postop_dir: Optional[str] = None
    adapted_dir: Optional[str] = None
    inflow: Optional[str] = None
    optimized_params: Optional[str] = None
    output_config: Optional[str] = None


@dataclass
class BCSConfig:
    type: str
    compliance_model: str = "constant"
    tune_space: Optional[TuneSpace] = None
    is_pulmonary: bool = True
    rcr_params: Optional[List[float]] = None


@dataclass
class TreesConfig:
    d_min: float = 0.01
    use_mean: bool = True
    specify_diameter: bool = True
    optimized_params_csv: Optional[str] = None
    lpa: Optional[TreeParameters] = None
    rpa: Optional[TreeParameters] = None


@dataclass
class AdaptationConfig:
    model: str = "M2"
    method: str = "cwss"
    location: str = "uniform"
    iterations: int = 10
    territory_scheme: str = "lpa_rpa"
    mode: str = "predict"
    parameter_set: Optional[Dict[str, Any]] = None


@dataclass
class AdaptBenchmarkScenarioConfig:
    name: str
    preop_rri_config: str
    postop_rri_config: str
    patient_id: Optional[str] = None
    scenario_group: Optional[str] = None
    perturbation_severity: Optional[str] = None
    tree_params_csv: Optional[str] = None
    clinical_targets_csv: Optional[str] = None
    parameter_overrides: Optional[Dict[str, Dict[str, Any]]] = None


@dataclass
class AdaptBenchmarkConfig:
    study_id: str
    output_dir: str
    models: List[str] = field(default_factory=lambda: ["M1", "M2", "M3"])
    workers: int = 1
    tree_params_csv: Optional[str] = None
    clinical_targets_csv: Optional[str] = None
    parameter_overrides: Optional[Dict[str, Dict[str, Any]]] = None
    scenarios: List[AdaptBenchmarkScenarioConfig] = field(default_factory=list)


@dataclass
class PipelineConfig:
    run_steady: bool = True
    optimize_bcs: bool = True
    run_threed: bool = True
    adapt: bool = True


@dataclass
class SlurmExecutionConfig:
    nodes: int = 3
    procs_per_node: int = 24
    memory: int = 16
    hours: int = 20
    partition: str = "amarsden"
    qos: str = "normal"
    mail_user: Optional[str] = None
    mail_types: List[str] = field(default_factory=lambda: ["begin", "end"])


@dataclass
class ThreeDExecutionConfig:
    mode: str = "slurm"
    executable: Optional[str] = None
    submit_command: str = "sbatch"
    clean_command: Optional[str] = "clean"
    slurm: SlurmExecutionConfig = field(default_factory=SlurmExecutionConfig)


@dataclass
class TissueSupportConfig:
    enabled: bool = True
    type: str = "uniform"
    stiffness: Optional[float] = None
    damping: Optional[float] = None
    apply_along_normal_direction: bool = True
    spatial_values_file_path: Optional[str] = None


@dataclass
class ThreeDConfig:
    mesh_scale_factor: float = 1.0
    convert_to_cm: bool = False
    solver_paths: Optional[Dict[str, str]] = None
    wall_model: str = "rigid"
    elasticity_modulus: float = 5062674.563165
    poisson_ratio: float = 0.5
    shell_thickness: float = 0.12
    prestress_file: Optional[str] = None
    prestress_file_path: Optional[str] = None
    execution: ThreeDExecutionConfig = field(default_factory=ThreeDExecutionConfig)
    tissue_support: Optional[TissueSupportConfig] = None


@dataclass
class PostprocessFigure:
    kind: str
    input: str
    output: Optional[str] = None
    options: Optional[Dict[str, Any]] = None


@dataclass
class PostprocessAnalysis:
    kind: str
    output: str
    options: Optional[Dict[str, Any]] = None


@dataclass
class PostprocessConfig:
    figures: List[PostprocessFigure] = field(default_factory=list)
    analyses: List[PostprocessAnalysis] = field(default_factory=list)


@dataclass
class CalibrationDataSourceConfig:
    mode: str = "mapped_centerline"
    mapped_centerline_result: Optional[str] = None
    metadata_json: Optional[str] = None
    centerline: Optional[str] = None
    pressure_array: str = "pressure"
    flow_array: str = "flow"
    flow_observation_type: str = "flow"
    area_array: Optional[str] = None
    branch_id_array: str = "BranchId"
    path_array: str = "Path"


@dataclass
class CalibrationParameterSelectionConfig:
    default: List[str] = field(default_factory=list)
    overrides: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class CalibrationParametersConfig:
    vessels: CalibrationParameterSelectionConfig = field(default_factory=CalibrationParameterSelectionConfig)
    junctions: CalibrationParameterSelectionConfig = field(default_factory=CalibrationParameterSelectionConfig)


@dataclass
class CalibrationSolverConfig:
    initial_damping_factor: float = 1.0
    maximum_iterations: int = 100
    tolerance_gradient: float = 1e-6
    tolerance_increment: float = 1e-10
    parameter_ratio_warning_threshold: float = 100.0
    confirmation_absolute_tolerance: float = 1e-8
    confirmation_relative_tolerance: float = 1e-6
    pressure_bound_multiplier: float = 10.0
    flow_bound_multiplier: float = 10.0
    cycle_stability_tolerance: float = 1e-3
    replay_minimum_cycles: int = 3
    replay_maximum_cycles: int = 10
    required_consecutive_stable_pairs: int = 1


@dataclass
class CalibrationInputNormalizationConfig:
    infinite_vessel_compliance: str = "error"


@dataclass
class CalibrationObservationQCConfig:
    vessel_flow_continuity_tolerance: float = 0.10
    junction_mass_balance_tolerance: float = 0.10
    root_waveform_rms_tolerance: float = 0.10
    minimum_pressure_drop_fraction: float = 0.95
    minimum_path_coverage: float = 0.99
    minimum_usable_samples: int = 3
    enforcement: str = "strict_network"


@dataclass
class CalibrationMPAPressureTargetConfig:
    vessel: str
    interface: str
    weight: float = 1.0
    normalized_rms_tolerance: float = 0.05


@dataclass
class CalibrationRPAFlowSplitTargetConfig:
    rpa_vessel: str
    lpa_vessel: str
    interface: str
    weight: float = 1.0
    absolute_tolerance: float = 0.02


@dataclass
class CalibrationTargetsConfig:
    mpa_pressure: CalibrationMPAPressureTargetConfig
    rpa_flow_split: CalibrationRPAFlowSplitTargetConfig
    require_improvement_over_baseline: bool = True


@dataclass
class CalibrationConfig:
    data_source: CalibrationDataSourceConfig
    parameters: CalibrationParametersConfig
    solver: CalibrationSolverConfig = field(default_factory=CalibrationSolverConfig)
    input_normalization: CalibrationInputNormalizationConfig = field(
        default_factory=CalibrationInputNormalizationConfig
    )
    observation_qc: CalibrationObservationQCConfig = field(
        default_factory=CalibrationObservationQCConfig
    )
    targets: Optional[CalibrationTargetsConfig] = None


@dataclass
class BaseConfig:
    version: int
    workflow: str
    paths: PathsConfig
    bcs: Optional[BCSConfig] = None
    trees: Optional[TreesConfig] = None
    adaptation: Optional[AdaptationConfig] = None
    adapt_benchmark: Optional[AdaptBenchmarkConfig] = None
    pipeline: Optional[PipelineConfig] = None
    threed: Optional[ThreeDConfig] = None
    postprocess: Optional[PostprocessConfig] = None
    calibration: Optional[CalibrationConfig] = None


def _ensure_keys(data: Dict[str, Any], allowed: List[str], context: str) -> None:
    unknown = set(data.keys()) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown keys in {context}: {sorted(unknown)}")


def _resolve_path(root: str, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = os.path.expanduser(value)
    if os.path.isabs(value):
        return value
    return os.path.abspath(os.path.join(root, value))


def _resolve_postprocess_analysis_options(
    root: str,
    kind: str,
    options: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if options is None:
        return None
    if not isinstance(options, dict):
        raise ValueError(f"postprocess analysis '{kind}' options must be a mapping")

    resolved = dict(options)
    if kind == "pulmonary_resistance_map":
        for key in ("svslicer_path", "centerline", "frames_csv", "intermediate_dir"):
            if resolved.get(key):
                resolved[key] = _resolve_path(root, str(resolved[key]))
    elif kind == "pulmonary_threed_suite":
        for key in (
            "simulation_dir",
            "centerline",
            "svslicer_path",
            "clinical_targets",
            "inflow_csv",
        ):
            if resolved.get(key):
                resolved[key] = _resolve_path(root, str(resolved[key]))
    return resolved


_TRANSFORMS = {
    "identity": identity,
    "positive": positive,
    "unit_interval": unit_interval,
}

_FROM_NATIVE = {
    "identity": identity,
    "log": np.log,
}


def _parse_transform(name: Optional[str], mapping: Dict[str, Any], context: str):
    if name is None:
        return mapping["identity"]
    if name not in mapping:
        raise ValueError(f"Unknown transform '{name}' in {context}. Supported: {sorted(mapping.keys())}")
    return mapping[name]


def _parse_tune_space(data: Optional[Dict[str, Any]]) -> Optional[TuneSpace]:
    if data is None:
        return None
    _ensure_keys(data, ["free", "fixed", "tied"], "bcs.tune_space")
    free_params = []
    for entry in data.get("free", []) or []:
        _ensure_keys(entry, ["name", "init", "lb", "ub", "to_native", "from_native"], "bcs.tune_space.free")
        free_params.append(
            FreeParam(
                name=entry["name"],
                init=float(entry["init"]),
                lb=float(entry["lb"]),
                ub=float(entry["ub"]),
                to_native=_parse_transform(entry.get("to_native"), _TRANSFORMS, f"free param {entry['name']}.to_native"),
                from_native=_parse_transform(entry.get("from_native"), _FROM_NATIVE, f"free param {entry['name']}.from_native"),
            )
        )
    fixed_params = []
    for entry in data.get("fixed", []) or []:
        _ensure_keys(entry, ["name", "value"], "bcs.tune_space.fixed")
        fixed_params.append(FixedParam(name=entry["name"], value=float(entry["value"])))
    tied_params = []
    for entry in data.get("tied", []) or []:
        _ensure_keys(entry, ["name", "other", "fn"], "bcs.tune_space.tied")
        tied_params.append(
            TiedParam(
                name=entry["name"],
                other=entry["other"],
                fn=_parse_transform(entry.get("fn"), _TRANSFORMS, f"tied param {entry['name']}.fn"),
            )
        )
    return TuneSpace(free=free_params, fixed=fixed_params, tied=tied_params)


def _parse_compliance(model: str, params: Dict[str, Any]):
    model_l = model.lower()
    if model_l == "constant":
        if "value" not in params:
            raise ValueError("constant compliance requires 'value'")
        return ConstantCompliance(float(params["value"]))
    if model_l == "olufsen":
        for key in ("k1", "k2", "k3"):
            if key not in params:
                raise ValueError("olufsen compliance requires k1/k2/k3")
        return OlufsenCompliance(float(params["k1"]), float(params["k2"]), float(params["k3"]))
    raise ValueError(f"Unknown compliance model '{model}'. Use 'constant' or 'olufsen'.")


def _parse_tree_params(side: str, data: Dict[str, Any]) -> TreeParameters:
    _ensure_keys(
        data,
        [
            "lrr",
            "diameter",
            "d_min",
            "alpha",
            "beta",
            "xi",
            "eta_sym",
            "inductance",
            "compliance",
        ],
        f"trees.{side}",
    )
    if "compliance" not in data:
        raise ValueError(f"trees.{side}.compliance is required")
    comp = data["compliance"]
    _ensure_keys(comp, ["model", "params"], f"trees.{side}.compliance")
    compliance_model = _parse_compliance(comp["model"], comp.get("params") or {})
    return TreeParameters(
        name=side,
        lrr=float(data["lrr"]),
        diameter=float(data["diameter"]),
        d_min=float(data["d_min"]),
        alpha=float(data.get("alpha")) if data.get("alpha") is not None else None,
        beta=float(data.get("beta")) if data.get("beta") is not None else None,
        xi=float(data.get("xi")) if data.get("xi") is not None else None,
        eta_sym=float(data.get("eta_sym")) if data.get("eta_sym") is not None else None,
        inductance=float(data.get("inductance", 0.0)),
        compliance_model=compliance_model,
    )


def _parse_paths(data: Dict[str, Any]) -> PathsConfig:
    _ensure_keys(
        data,
        [
            "root",
            "zerod_config",
            "clinical_targets",
            "mesh_surfaces",
            "preop_dir",
            "postop_dir",
            "adapted_dir",
            "inflow",
            "optimized_params",
            "output_config",
        ],
        "paths",
    )
    root = data.get("root", ".")
    root_resolved = os.path.abspath(root)
    return PathsConfig(
        root=root_resolved,
        zerod_config=_resolve_path(root_resolved, data.get("zerod_config")),
        clinical_targets=_resolve_path(root_resolved, data.get("clinical_targets")),
        mesh_surfaces=_resolve_path(root_resolved, data.get("mesh_surfaces")),
        preop_dir=_resolve_path(root_resolved, data.get("preop_dir")),
        postop_dir=_resolve_path(root_resolved, data.get("postop_dir")),
        adapted_dir=_resolve_path(root_resolved, data.get("adapted_dir")),
        inflow=_resolve_path(root_resolved, data.get("inflow")),
        optimized_params=_resolve_path(root_resolved, data.get("optimized_params")),
        output_config=_resolve_path(root_resolved, data.get("output_config")),
    )


def _normalize_benchmark_models(models: Optional[List[Any]]) -> List[str]:
    selected = [str(model).upper() for model in (models or ["M1", "M2", "M3"])]
    invalid = [model for model in selected if model not in {"M1", "M2", "M3"}]
    if invalid:
        raise ValueError(
            "adapt_benchmark.models must contain only M1, M2, or M3"
        )
    if not selected:
        raise ValueError("adapt_benchmark.models must not be empty")
    return selected


def _parse_adapt_benchmark_parameter_overrides(
    data: Optional[Dict[str, Any]],
    *,
    context: str,
) -> Optional[Dict[str, Dict[str, Any]]]:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError(f"{context} must be a mapping of model name to parameter mapping")

    normalized: Dict[str, Dict[str, Any]] = {}
    for model_name, payload in data.items():
        resolved_model = str(model_name).upper()
        if resolved_model not in {"M1", "M2", "M3"}:
            raise ValueError(f"{context} contains unsupported model '{model_name}'")
        if payload is None:
            normalized[resolved_model] = {}
            continue
        if not isinstance(payload, dict):
            raise ValueError(f"{context}.{model_name} must be a mapping")
        normalized[resolved_model] = dict(payload)
    return normalized


def _parse_adapt_benchmark(
    root: str,
    data: Dict[str, Any],
) -> AdaptBenchmarkConfig:
    _ensure_keys(
        data,
        [
            "study_id",
            "output_dir",
            "models",
            "workers",
            "tree_params_csv",
            "clinical_targets_csv",
            "parameter_overrides",
            "scenarios",
        ],
        "adapt_benchmark",
    )
    study_id = str(data.get("study_id") or "").strip()
    if not study_id:
        raise ValueError("adapt_benchmark.study_id is required")

    output_dir_raw = data.get("output_dir")
    if not output_dir_raw:
        raise ValueError("adapt_benchmark.output_dir is required")

    scenarios_raw = data.get("scenarios") or []
    if not scenarios_raw:
        raise ValueError("adapt_benchmark.scenarios must contain at least one scenario")

    scenarios: List[AdaptBenchmarkScenarioConfig] = []
    for idx, entry in enumerate(scenarios_raw):
        if not isinstance(entry, dict):
            raise ValueError(f"adapt_benchmark.scenarios[{idx}] must be a mapping")
        _ensure_keys(
            entry,
            [
                "name",
                "patient_id",
                "scenario_group",
                "perturbation_severity",
                "preop_rri_config",
                "postop_rri_config",
                "tree_params_csv",
                "clinical_targets_csv",
                "parameter_overrides",
            ],
            f"adapt_benchmark.scenarios[{idx}]",
        )
        name = str(entry.get("name") or "").strip()
        if not name:
            raise ValueError(f"adapt_benchmark.scenarios[{idx}].name is required")
        preop_rri_config = entry.get("preop_rri_config")
        postop_rri_config = entry.get("postop_rri_config")
        if not preop_rri_config:
            raise ValueError(
                f"adapt_benchmark.scenarios[{idx}].preop_rri_config is required"
            )
        if not postop_rri_config:
            raise ValueError(
                f"adapt_benchmark.scenarios[{idx}].postop_rri_config is required"
            )
        scenarios.append(
            AdaptBenchmarkScenarioConfig(
                name=name,
                preop_rri_config=_resolve_path(root, str(preop_rri_config)),
                postop_rri_config=_resolve_path(root, str(postop_rri_config)),
                patient_id=(
                    str(entry["patient_id"]).strip()
                    if entry.get("patient_id") is not None
                    else None
                ),
                scenario_group=(
                    str(entry["scenario_group"]).strip()
                    if entry.get("scenario_group") is not None
                    else None
                ),
                perturbation_severity=(
                    str(entry["perturbation_severity"]).strip()
                    if entry.get("perturbation_severity") is not None
                    else None
                ),
                tree_params_csv=(
                    _resolve_path(root, str(entry["tree_params_csv"]))
                    if entry.get("tree_params_csv")
                    else None
                ),
                clinical_targets_csv=(
                    _resolve_path(root, str(entry["clinical_targets_csv"]))
                    if entry.get("clinical_targets_csv")
                    else None
                ),
                parameter_overrides=_parse_adapt_benchmark_parameter_overrides(
                    entry.get("parameter_overrides"),
                    context=f"adapt_benchmark.scenarios[{idx}].parameter_overrides",
                ),
            )
        )

    return AdaptBenchmarkConfig(
        study_id=study_id,
        output_dir=_resolve_path(root, str(output_dir_raw)),
        models=_normalize_benchmark_models(data.get("models")),
        workers=max(1, int(data.get("workers") or 1)),
        tree_params_csv=(
            _resolve_path(root, str(data["tree_params_csv"]))
            if data.get("tree_params_csv")
            else None
        ),
        clinical_targets_csv=(
            _resolve_path(root, str(data["clinical_targets_csv"]))
            if data.get("clinical_targets_csv")
            else None
        ),
        parameter_overrides=_parse_adapt_benchmark_parameter_overrides(
            data.get("parameter_overrides"),
            context="adapt_benchmark.parameter_overrides",
        ),
        scenarios=scenarios,
    )


def _parse_slurm_execution(data: Optional[Dict[str, Any]]) -> SlurmExecutionConfig:
    if data is None:
        return SlurmExecutionConfig()
    _ensure_keys(
        data,
        ["nodes", "procs_per_node", "memory", "hours", "partition", "qos", "mail_user", "mail_types"],
        "threed.execution.slurm",
    )
    mail_user = data.get("mail_user")
    if mail_user is not None:
        mail_user = str(mail_user)
    mail_types = data.get("mail_types", ["begin", "end"])
    if mail_types is None:
        mail_types = []
    if not isinstance(mail_types, list):
        raise ValueError("threed.execution.slurm.mail_types must be a list when provided")
    return SlurmExecutionConfig(
        nodes=int(data.get("nodes", 3)),
        procs_per_node=int(data.get("procs_per_node", 24)),
        memory=int(data.get("memory", 16)),
        hours=int(data.get("hours", 20)),
        partition=str(data.get("partition", "amarsden")),
        qos=str(data.get("qos", "normal")),
        mail_user=mail_user,
        mail_types=[str(mail_type) for mail_type in mail_types],
    )


def _parse_threed_execution(data: Optional[Dict[str, Any]]) -> ThreeDExecutionConfig:
    if data is None:
        return ThreeDExecutionConfig()
    _ensure_keys(
        data,
        ["mode", "executable", "submit_command", "clean_command", "slurm"],
        "threed.execution",
    )
    mode = str(data.get("mode", "slurm")).lower()
    if mode not in {"local", "slurm"}:
        raise ValueError("threed.execution.mode must be one of local|slurm")
    executable = data.get("executable")
    if executable is None or not str(executable).strip():
        raise ValueError("threed.execution.executable is required")
    clean_command = data.get("clean_command", "clean")
    if clean_command is not None:
        clean_command = str(clean_command)
    return ThreeDExecutionConfig(
        mode=mode,
        executable=str(executable),
        submit_command=str(data.get("submit_command", "sbatch")),
        clean_command=clean_command,
        slurm=_parse_slurm_execution(data.get("slurm")),
    )


def _parse_tissue_support(
    root: str,
    data: Optional[Dict[str, Any]],
    *,
    wall_model: str,
) -> Optional[TissueSupportConfig]:
    if data is None:
        return None
    _ensure_keys(
        data,
        [
            "enabled",
            "type",
            "stiffness",
            "damping",
            "apply_along_normal_direction",
            "spatial_values_file_path",
        ],
        "threed.tissue_support",
    )
    if wall_model != "deformable":
        raise ValueError("threed.tissue_support is only valid with threed.wall_model=deformable")

    enabled = bool(data.get("enabled", True))
    support_type = str(data.get("type", "uniform")).lower()
    if support_type not in {"uniform", "spatial"}:
        raise ValueError("threed.tissue_support.type must be one of uniform|spatial")

    stiffness = data.get("stiffness")
    damping = data.get("damping")
    spatial_values_file_path = data.get("spatial_values_file_path")

    if enabled and support_type == "uniform":
        if stiffness is None or damping is None:
            raise ValueError("uniform threed.tissue_support requires stiffness and damping")
        stiffness = float(stiffness)
        damping = float(damping)
        if stiffness < 0.0 or damping < 0.0:
            raise ValueError("threed.tissue_support stiffness and damping must be non-negative")
        if spatial_values_file_path is not None:
            raise ValueError("uniform threed.tissue_support forbids spatial_values_file_path")
    elif enabled and support_type == "spatial":
        if not spatial_values_file_path:
            raise ValueError("spatial threed.tissue_support requires spatial_values_file_path")
        if stiffness is not None or damping is not None:
            raise ValueError("spatial threed.tissue_support forbids stiffness and damping")
        spatial_values_file_path = _resolve_path(root, str(spatial_values_file_path))
    else:
        stiffness = float(stiffness) if stiffness is not None else None
        damping = float(damping) if damping is not None else None
        spatial_values_file_path = (
            _resolve_path(root, str(spatial_values_file_path))
            if spatial_values_file_path
            else None
        )

    return TissueSupportConfig(
        enabled=enabled,
        type=support_type,
        stiffness=stiffness,
        damping=damping,
        apply_along_normal_direction=bool(data.get("apply_along_normal_direction", True)),
        spatial_values_file_path=spatial_values_file_path,
    )


def _parse_calibration_parameter_selection(
    data: Optional[Dict[str, Any]],
    *,
    context: str,
) -> CalibrationParameterSelectionConfig:
    if data is None:
        return CalibrationParameterSelectionConfig()
    _ensure_keys(data, ["default", "overrides"], context)

    default_raw = data.get("default") or []
    if not isinstance(default_raw, list):
        raise ValueError(f"{context}.default must be a list of parameter names")
    default = [str(name) for name in default_raw]

    overrides_raw = data.get("overrides") or {}
    if not isinstance(overrides_raw, dict):
        raise ValueError(f"{context}.overrides must be a mapping of block name to parameter list")
    overrides: Dict[str, List[str]] = {}
    for block_name, names in overrides_raw.items():
        if not isinstance(names, list):
            raise ValueError(f"{context}.overrides.{block_name} must be a list of parameter names")
        overrides[str(block_name)] = [str(name) for name in names]

    return CalibrationParameterSelectionConfig(
        default=default,
        overrides=overrides,
    )


def _finite_positive_float(value: Any, *, context: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be finite and positive") from exc
    if not np.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{context} must be finite and positive")
    return converted


def _positive_integer(value: Any, *, context: str, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be an integer at least {minimum}")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be an integer at least {minimum}") from exc
    if (
        not np.isfinite(converted)
        or converted != int(converted)
        or int(converted) < minimum
    ):
        raise ValueError(f"{context} must be an integer at least {minimum}")
    return int(converted)


def _target_vessel_name(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty vessel name")
    return value.strip()


def _target_interface(value: Any, *, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"{context} must be one of {sorted(_CALIBRATION_TARGET_INTERFACES)}"
        )
    interface = value.strip().lower()
    if interface not in _CALIBRATION_TARGET_INTERFACES:
        raise ValueError(
            f"{context} must be one of {sorted(_CALIBRATION_TARGET_INTERFACES)}"
        )
    return interface


def _parse_calibration_targets(
    data: Optional[Dict[str, Any]],
    *,
    context: str = "calibration.targets",
) -> Optional[CalibrationTargetsConfig]:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError(f"{context} must be a mapping")
    _ensure_keys(
        data,
        ["mpa_pressure", "rpa_flow_split", "require_improvement_over_baseline"],
        context,
    )

    mpa_raw = data.get("mpa_pressure")
    rpa_raw = data.get("rpa_flow_split")
    if not isinstance(mpa_raw, dict) or not isinstance(rpa_raw, dict):
        raise ValueError(
            f"{context} requires both mpa_pressure and rpa_flow_split mappings"
        )
    _ensure_keys(
        mpa_raw,
        ["vessel", "interface", "weight", "normalized_rms_tolerance"],
        f"{context}.mpa_pressure",
    )
    _ensure_keys(
        rpa_raw,
        ["rpa_vessel", "lpa_vessel", "interface", "weight", "absolute_tolerance"],
        f"{context}.rpa_flow_split",
    )

    mpa = CalibrationMPAPressureTargetConfig(
        vessel=_target_vessel_name(mpa_raw.get("vessel"), context=f"{context}.mpa_pressure.vessel"),
        interface=_target_interface(
            mpa_raw.get("interface"), context=f"{context}.mpa_pressure.interface"
        ),
        weight=_finite_positive_float(
            mpa_raw.get("weight", 1.0), context=f"{context}.mpa_pressure.weight"
        ),
        normalized_rms_tolerance=_finite_positive_float(
            mpa_raw.get("normalized_rms_tolerance", 0.05),
            context=f"{context}.mpa_pressure.normalized_rms_tolerance",
        ),
    )
    rpa = CalibrationRPAFlowSplitTargetConfig(
        rpa_vessel=_target_vessel_name(
            rpa_raw.get("rpa_vessel"), context=f"{context}.rpa_flow_split.rpa_vessel"
        ),
        lpa_vessel=_target_vessel_name(
            rpa_raw.get("lpa_vessel"), context=f"{context}.rpa_flow_split.lpa_vessel"
        ),
        interface=_target_interface(
            rpa_raw.get("interface"), context=f"{context}.rpa_flow_split.interface"
        ),
        weight=_finite_positive_float(
            rpa_raw.get("weight", 1.0), context=f"{context}.rpa_flow_split.weight"
        ),
        absolute_tolerance=_finite_positive_float(
            rpa_raw.get("absolute_tolerance", 0.02),
            context=f"{context}.rpa_flow_split.absolute_tolerance",
        ),
    )
    roles = {mpa.vessel, rpa.rpa_vessel, rpa.lpa_vessel}
    if len(roles) != 3:
        raise ValueError(
            f"{context} requires distinct MPA, LPA, and RPA vessel roles"
        )

    require_improvement = data.get("require_improvement_over_baseline", True)
    if not isinstance(require_improvement, bool):
        raise ValueError(
            f"{context}.require_improvement_over_baseline must be a boolean"
        )
    return CalibrationTargetsConfig(
        mpa_pressure=mpa,
        rpa_flow_split=rpa,
        require_improvement_over_baseline=require_improvement,
    )


def _parse_calibration(root: str, data: Dict[str, Any]) -> CalibrationConfig:
    _ensure_keys(
        data,
        [
            "data_source",
            "parameters",
            "solver",
            "input_normalization",
            "observation_qc",
            "targets",
        ],
        "calibration",
    )

    data_source_raw = data.get("data_source")
    if not isinstance(data_source_raw, dict):
        raise ValueError("calibration.data_source is required")
    _ensure_keys(
        data_source_raw,
        [
            "mode",
            "mapped_centerline_result",
            "metadata_json",
            "centerline",
            "pressure_array",
            "flow_array",
            "flow_observation_type",
            "area_array",
            "branch_id_array",
            "path_array",
        ],
        "calibration.data_source",
    )
    mode = str(data_source_raw.get("mode", "mapped_centerline"))
    if mode != "mapped_centerline":
        raise ValueError(
            "calibration.data_source.mode must be 'mapped_centerline' for calibrate_0d_from_3d stage 1"
        )
    if data_source_raw.get("mapped_centerline_result") in (None, ""):
        raise ValueError("calibration.data_source.mapped_centerline_result is required")
    if data_source_raw.get("centerline") in (None, ""):
        raise ValueError("calibration.data_source.centerline is required")
    if "flow_observation_type" not in data_source_raw:
        raise ValueError(
            "calibration.data_source.flow_observation_type is required; "
            "declare flow for svSlicer integrated flow or velocity for a true velocity field"
        )
    flow_observation_type = str(data_source_raw["flow_observation_type"]).lower()
    if flow_observation_type not in {"flow", "velocity"}:
        raise ValueError(
            "calibration.data_source.flow_observation_type must be one of flow|velocity"
        )
    area_array = data_source_raw.get(
        "area_array",
        "CenterlineSectionArea" if flow_observation_type == "velocity" else None,
    )
    if area_array in ("", None):
        area_array = None
    if flow_observation_type == "velocity" and area_array is None:
        raise ValueError(
            "calibration.data_source.area_array is required when "
            "calibration.data_source.flow_observation_type=velocity"
        )
    data_source = CalibrationDataSourceConfig(
        mode=mode,
        mapped_centerline_result=_resolve_path(root, str(data_source_raw["mapped_centerline_result"])),
        metadata_json=_resolve_path(
            root,
            str(data_source_raw["metadata_json"])
            if data_source_raw.get("metadata_json") not in (None, "")
            else None,
        ),
        centerline=_resolve_path(root, str(data_source_raw["centerline"])),
        pressure_array=str(data_source_raw.get("pressure_array", "pressure")),
        flow_array=str(data_source_raw.get("flow_array", "flow")),
        flow_observation_type=flow_observation_type,
        area_array=str(area_array) if area_array is not None else None,
        branch_id_array=str(data_source_raw.get("branch_id_array", "BranchId")),
        path_array=str(data_source_raw.get("path_array", "Path")),
    )

    parameters_raw = data.get("parameters")
    if not isinstance(parameters_raw, dict):
        raise ValueError("calibration.parameters is required")
    _ensure_keys(parameters_raw, ["vessels", "junctions"], "calibration.parameters")
    parameters = CalibrationParametersConfig(
        vessels=_parse_calibration_parameter_selection(
            parameters_raw.get("vessels"),
            context="calibration.parameters.vessels",
        ),
        junctions=_parse_calibration_parameter_selection(
            parameters_raw.get("junctions"),
            context="calibration.parameters.junctions",
        ),
    )

    solver_raw = data.get("solver") or {}
    if not isinstance(solver_raw, dict):
        raise ValueError("calibration.solver must be a mapping")
    _ensure_keys(
        solver_raw,
        [
            "initial_damping_factor",
            "maximum_iterations",
            "tolerance_gradient",
            "tolerance_increment",
            "parameter_ratio_warning_threshold",
            "confirmation_absolute_tolerance",
            "confirmation_relative_tolerance",
            "pressure_bound_multiplier",
            "flow_bound_multiplier",
            "cycle_stability_tolerance",
            "replay_minimum_cycles",
            "replay_maximum_cycles",
            "required_consecutive_stable_pairs",
        ],
        "calibration.solver",
    )
    solver = CalibrationSolverConfig(
        initial_damping_factor=float(solver_raw.get("initial_damping_factor", 1.0)),
        maximum_iterations=int(solver_raw.get("maximum_iterations", 100)),
        tolerance_gradient=float(solver_raw.get("tolerance_gradient", 1e-6)),
        tolerance_increment=float(solver_raw.get("tolerance_increment", 1e-10)),
        parameter_ratio_warning_threshold=float(
            solver_raw.get("parameter_ratio_warning_threshold", 100.0)
        ),
        confirmation_absolute_tolerance=float(
            solver_raw.get("confirmation_absolute_tolerance", 1e-8)
        ),
        confirmation_relative_tolerance=float(
            solver_raw.get("confirmation_relative_tolerance", 1e-6)
        ),
        pressure_bound_multiplier=float(solver_raw.get("pressure_bound_multiplier", 10.0)),
        flow_bound_multiplier=float(solver_raw.get("flow_bound_multiplier", 10.0)),
        cycle_stability_tolerance=float(solver_raw.get("cycle_stability_tolerance", 1e-3)),
        replay_minimum_cycles=_positive_integer(
            solver_raw.get("replay_minimum_cycles", 3),
            context="calibration.solver.replay_minimum_cycles",
            minimum=3,
        ),
        replay_maximum_cycles=_positive_integer(
            solver_raw.get("replay_maximum_cycles", 10),
            context="calibration.solver.replay_maximum_cycles",
            minimum=3,
        ),
        required_consecutive_stable_pairs=_positive_integer(
            solver_raw.get("required_consecutive_stable_pairs", 1),
            context="calibration.solver.required_consecutive_stable_pairs",
        ),
    )
    if (
        not np.isfinite(solver.initial_damping_factor)
        or solver.initial_damping_factor <= 0.0
        or solver.maximum_iterations < 1
        or not np.isfinite(solver.tolerance_gradient)
        or solver.tolerance_gradient < 0.0
        or not np.isfinite(solver.tolerance_increment)
        or solver.tolerance_increment < 0.0
        or not np.isfinite(solver.parameter_ratio_warning_threshold)
        or solver.parameter_ratio_warning_threshold < 1.0
        or not np.isfinite(solver.confirmation_absolute_tolerance)
        or solver.confirmation_absolute_tolerance < 0.0
        or not np.isfinite(solver.confirmation_relative_tolerance)
        or solver.confirmation_relative_tolerance < 0.0
        or not np.isfinite(solver.pressure_bound_multiplier)
        or solver.pressure_bound_multiplier <= 0.0
        or not np.isfinite(solver.flow_bound_multiplier)
        or solver.flow_bound_multiplier <= 0.0
        or not np.isfinite(solver.cycle_stability_tolerance)
        or solver.cycle_stability_tolerance < 0.0
        or solver.replay_maximum_cycles < solver.replay_minimum_cycles
        or solver.required_consecutive_stable_pairs > solver.replay_maximum_cycles - 1
    ):
        raise ValueError(
            "calibration.solver requires a positive damping factor and at least "
            "one iteration; tolerances must be finite and non-negative; "
            "parameter_ratio_warning_threshold must be finite and at least 1; "
            "confirmation tolerances must be finite and non-negative; pressure and "
            "flow bound multipliers must be finite and positive; cycle stability "
            "tolerance must be finite and non-negative; replay_maximum_cycles must "
            "be at least replay_minimum_cycles and leave enough cycle pairs for "
            "required_consecutive_stable_pairs"
        )

    normalization_raw = data.get("input_normalization") or {}
    if not isinstance(normalization_raw, dict):
        raise ValueError("calibration.input_normalization must be a mapping")
    _ensure_keys(
        normalization_raw,
        ["infinite_vessel_compliance"],
        "calibration.input_normalization",
    )
    infinite_vessel_compliance = str(
        normalization_raw.get("infinite_vessel_compliance", "error")
    ).lower()
    if infinite_vessel_compliance not in {"error", "zero"}:
        raise ValueError(
            "calibration.input_normalization.infinite_vessel_compliance "
            "must be one of error|zero"
        )

    observation_qc_raw = data.get("observation_qc") or {}
    if not isinstance(observation_qc_raw, dict):
        raise ValueError("calibration.observation_qc must be a mapping")
    _ensure_keys(
        observation_qc_raw,
        [
            "vessel_flow_continuity_tolerance",
            "junction_mass_balance_tolerance",
            "root_waveform_rms_tolerance",
            "minimum_pressure_drop_fraction",
            "minimum_path_coverage",
            "minimum_usable_samples",
            "enforcement",
        ],
        "calibration.observation_qc",
    )
    observation_qc = CalibrationObservationQCConfig(
        vessel_flow_continuity_tolerance=float(
            observation_qc_raw.get("vessel_flow_continuity_tolerance", 0.10)
        ),
        junction_mass_balance_tolerance=float(
            observation_qc_raw.get("junction_mass_balance_tolerance", 0.10)
        ),
        root_waveform_rms_tolerance=float(
            observation_qc_raw.get("root_waveform_rms_tolerance", 0.10)
        ),
        minimum_pressure_drop_fraction=float(
            observation_qc_raw.get("minimum_pressure_drop_fraction", 0.95)
        ),
        minimum_path_coverage=float(
            observation_qc_raw.get("minimum_path_coverage", 0.99)
        ),
        minimum_usable_samples=int(
            observation_qc_raw.get("minimum_usable_samples", 3)
        ),
        enforcement=str(observation_qc_raw.get("enforcement", "strict_network")).lower(),
    )
    if (
        not np.isfinite(observation_qc.vessel_flow_continuity_tolerance)
        or observation_qc.vessel_flow_continuity_tolerance < 0.0
        or not np.isfinite(observation_qc.junction_mass_balance_tolerance)
        or observation_qc.junction_mass_balance_tolerance < 0.0
        or not np.isfinite(observation_qc.root_waveform_rms_tolerance)
        or observation_qc.root_waveform_rms_tolerance < 0.0
    ):
        raise ValueError(
            "calibration.observation_qc error tolerances must be finite and non-negative"
        )
    if not 0.0 <= observation_qc.minimum_pressure_drop_fraction <= 1.0:
        raise ValueError(
            "calibration.observation_qc.minimum_pressure_drop_fraction must be between 0 and 1"
        )
    if not 0.0 < observation_qc.minimum_path_coverage <= 1.0:
        raise ValueError(
            "calibration.observation_qc.minimum_path_coverage must be in (0, 1]"
        )
    if observation_qc.minimum_usable_samples < 3:
        raise ValueError(
            "calibration.observation_qc.minimum_usable_samples must be at least 3"
        )

    if observation_qc.enforcement not in {"strict_network", "target_focused"}:
        raise ValueError(
            "calibration.observation_qc.enforcement must be one of "
            "strict_network|target_focused"
        )
    targets = _parse_calibration_targets(data.get("targets"))
    if observation_qc.enforcement == "target_focused" and targets is None:
        raise ValueError(
            "calibration.observation_qc.enforcement=target_focused requires "
            "calibration.targets"
        )

    return CalibrationConfig(
        data_source=data_source,
        parameters=parameters,
        solver=solver,
        input_normalization=CalibrationInputNormalizationConfig(
            infinite_vessel_compliance=infinite_vessel_compliance
        ),
        observation_qc=observation_qc,
        targets=targets,
    )


def load_config(path: str) -> BaseConfig:
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}

    _ensure_keys(
        raw,
        [
            "version",
            "workflow",
            "paths",
            "bcs",
            "trees",
            "adaptation",
            "adapt_benchmark",
            "pipeline",
            "threed",
            "postprocess",
            "calibration",
        ],
        "config",
    )

    version = int(raw.get("version", 0))
    if version != CONFIG_VERSION:
        raise ValueError(f"Unsupported config version {version}. Expected {CONFIG_VERSION}.")

    workflow = raw.get("workflow")
    if workflow not in {
        "pipeline",
        "tune_bcs",
        "construct_trees",
        "adapt",
        "adapt_benchmark",
        "postprocess",
        "calibrate_0d_from_3d",
    }:
        raise ValueError(
            "workflow must be one of "
            "pipeline|tune_bcs|construct_trees|adapt|adapt_benchmark|postprocess|calibrate_0d_from_3d"
        )

    if "paths" not in raw or raw["paths"] is None:
        raise ValueError("paths section is required")
    paths = _parse_paths(raw["paths"])

    bcs = None
    if raw.get("bcs") is not None:
        data = raw["bcs"]
        _ensure_keys(data, ["type", "compliance_model", "tune_space", "is_pulmonary", "rcr_params"], "bcs")
        bcs = BCSConfig(
            type=data["type"],
            compliance_model=data.get("compliance_model", "constant"),
            tune_space=_parse_tune_space(data.get("tune_space")),
            is_pulmonary=bool(data.get("is_pulmonary", True)),
            rcr_params=data.get("rcr_params"),
        )

    trees = None
    if raw.get("trees") is not None:
        data = raw["trees"]
        _ensure_keys(data, ["d_min", "use_mean", "specify_diameter", "optimized_params_csv", "lpa", "rpa"], "trees")
        trees = TreesConfig(
            d_min=float(data.get("d_min", 0.01)),
            use_mean=bool(data.get("use_mean", True)),
            specify_diameter=bool(data.get("specify_diameter", True)),
            optimized_params_csv=_resolve_path(paths.root, data.get("optimized_params_csv")) if data.get("optimized_params_csv") else None,
            lpa=_parse_tree_params("lpa", data["lpa"]) if data.get("lpa") is not None else None,
            rpa=_parse_tree_params("rpa", data["rpa"]) if data.get("rpa") is not None else None,
        )

    adaptation = None
    if raw.get("adaptation") is not None:
        data = raw["adaptation"]
        _ensure_keys(
            data,
            ["model", "method", "location", "iterations", "territory_scheme", "mode", "parameter_set"],
            "adaptation",
        )
        adaptation = AdaptationConfig(
            model=str(data.get("model", "M2")),
            method=data.get("method", "cwss"),
            location=data.get("location", "uniform"),
            iterations=int(data.get("iterations", 10)),
            territory_scheme=str(data.get("territory_scheme", "lpa_rpa")),
            mode=str(data.get("mode", "predict")),
            parameter_set=data.get("parameter_set"),
        )

    adapt_benchmark = None
    if raw.get("adapt_benchmark") is not None:
        adapt_benchmark = _parse_adapt_benchmark(paths.root, raw["adapt_benchmark"])

    pipeline = None
    if raw.get("pipeline") is not None:
        data = raw["pipeline"]
        _ensure_keys(data, ["run_steady", "optimize_bcs", "run_threed", "adapt"], "pipeline")
        pipeline = PipelineConfig(
            run_steady=bool(data.get("run_steady", True)),
            optimize_bcs=bool(data.get("optimize_bcs", True)),
            run_threed=bool(data.get("run_threed", True)),
            adapt=bool(data.get("adapt", True)),
        )

    threed = None
    if raw.get("threed") is not None:
        data = raw["threed"]
        _ensure_keys(
            data,
            [
                "mesh_scale_factor",
                "convert_to_cm",
                "solver_paths",
                "wall_model",
                "elasticity_modulus",
                "poisson_ratio",
                "shell_thickness",
                "prestress_file",
                "prestress_file_path",
                "execution",
                "tissue_support",
            ],
            "threed",
        )
        wall_model = str(data.get("wall_model", "rigid")).lower()
        if wall_model not in {"rigid", "deformable"}:
            raise ValueError("threed.wall_model must be one of rigid|deformable")

        elasticity_modulus = float(data.get("elasticity_modulus", 5062674.563165))
        poisson_ratio = float(data.get("poisson_ratio", 0.5))
        shell_thickness = float(data.get("shell_thickness", 0.12))
        prestress_file = data.get("prestress_file")
        if isinstance(prestress_file, bool):
            prestress_file = "auto" if prestress_file else None
        elif prestress_file is not None:
            prestress_file = str(prestress_file)

        prestress_file_path = (
            _resolve_path(paths.root, data.get("prestress_file_path"))
            if data.get("prestress_file_path")
            else None
        )
        if prestress_file and prestress_file.lower() not in {"auto", "from_steady_mean"}:
            # allow prestress_file to also directly carry a path
            prestress_file_path = _resolve_path(paths.root, prestress_file)
        if wall_model == "deformable":
            if elasticity_modulus <= 0.0:
                raise ValueError("threed.elasticity_modulus must be > 0 for deformable wall model")
            if shell_thickness <= 0.0:
                raise ValueError("threed.shell_thickness must be > 0 for deformable wall model")
            if not (-1.0 < poisson_ratio <= 0.5):
                raise ValueError("threed.poisson_ratio must satisfy -1.0 < v <= 0.5 for deformable wall model")
        tissue_support = _parse_tissue_support(
            paths.root,
            data.get("tissue_support"),
            wall_model=wall_model,
        )
        if data.get("execution") is None:
            raise ValueError("threed.execution.executable is required when threed is provided")
        threed = ThreeDConfig(
            mesh_scale_factor=float(data.get("mesh_scale_factor", 1.0)),
            convert_to_cm=bool(data.get("convert_to_cm", False)),
            solver_paths=data.get("solver_paths"),
            wall_model=wall_model,
            elasticity_modulus=elasticity_modulus,
            poisson_ratio=poisson_ratio,
            shell_thickness=shell_thickness,
            prestress_file=prestress_file,
            prestress_file_path=prestress_file_path,
            execution=_parse_threed_execution(data.get("execution")),
            tissue_support=tissue_support,
        )

    postprocess = None
    if raw.get("postprocess") is not None:
        data = raw["postprocess"]
        _ensure_keys(data, ["figures", "analyses"], "postprocess")
        figures = []
        for entry in data.get("figures", []) or []:
            _ensure_keys(entry, ["kind", "input", "output", "options"], "postprocess.figures")
            figures.append(
                PostprocessFigure(
                    kind=entry["kind"],
                    input=_resolve_path(paths.root, entry["input"]),
                    output=_resolve_path(paths.root, entry.get("output")) if entry.get("output") else None,
                    options=entry.get("options"),
                )
            )
        analyses = []
        for entry in data.get("analyses", []) or []:
            _ensure_keys(entry, ["kind", "output", "options"], "postprocess.analyses")
            kind = str(entry["kind"])
            options = entry.get("options")
            if kind == "pulmonary_resistance_map":
                if options is None:
                    raise ValueError(
                        "postprocess analysis 'pulmonary_resistance_map' requires options"
                    )
                _ensure_keys(
                    options,
                    [
                        "svslicer_path",
                        "centerline",
                        "frames_csv",
                        "cycle_duration_s",
                        "workers",
                        "keep_intermediate_centerlines",
                        "intermediate_dir",
                        "pressure_array",
                        "flow_array",
                        "branch_id_array",
                        "path_array",
                    ],
                    "postprocess.analyses.options",
                )
                for required_key in ("svslicer_path", "centerline", "frames_csv", "cycle_duration_s"):
                    if options.get(required_key) in (None, ""):
                        raise ValueError(
                            "postprocess analysis 'pulmonary_resistance_map' requires "
                            f"options.{required_key}"
                        )
            elif kind == "pulmonary_threed_suite":
                if options is None:
                    raise ValueError(
                        "postprocess analysis 'pulmonary_threed_suite' requires options"
                    )
                _ensure_keys(
                    options,
                    [
                        "simulation_dir",
                        "centerline",
                        "svslicer_path",
                        "clinical_targets",
                        "stage",
                        "cycle_duration_s",
                        "inflow_csv",
                        "pressure_field",
                        "already_mmhg",
                        "resistance_map_workers",
                    ],
                    "postprocess.analyses.options",
                )
                for required_key in ("simulation_dir", "centerline", "svslicer_path", "stage"):
                    if options.get(required_key) in (None, ""):
                        raise ValueError(
                            "postprocess analysis 'pulmonary_threed_suite' requires "
                            f"options.{required_key}"
                        )
                if options.get("cycle_duration_s") in (None, "") and options.get("inflow_csv") in (None, ""):
                    raise ValueError(
                        "postprocess analysis 'pulmonary_threed_suite' requires one of "
                        "options.cycle_duration_s or options.inflow_csv"
                    )
            analyses.append(
                PostprocessAnalysis(
                    kind=kind,
                    output=_resolve_path(paths.root, entry["output"]),
                    options=_resolve_postprocess_analysis_options(paths.root, kind, options),
                )
            )
        if not figures and not analyses:
            raise ValueError("postprocess requires at least one figure or analysis")
        postprocess = PostprocessConfig(figures=figures, analyses=analyses)

    calibration = None
    if raw.get("calibration") is not None:
        calibration = _parse_calibration(paths.root, raw["calibration"])

    return BaseConfig(
        version=version,
        workflow=workflow,
        paths=paths,
        bcs=bcs,
        trees=trees,
        adaptation=adaptation,
        adapt_benchmark=adapt_benchmark,
        pipeline=pipeline,
        threed=threed,
        postprocess=postprocess,
        calibration=calibration,
    )


def render_schema() -> str:
    return """
# svzerodtrees config (v1)
version: 1
workflow: pipeline  # pipeline | tune_bcs | construct_trees | adapt | adapt_benchmark | postprocess | calibrate_0d_from_3d

paths:
  root: .
  zerod_config: path/to/zerod_config.json
  clinical_targets: path/to/clinical_targets.csv
  mesh_surfaces: path/to/mesh-surfaces
  preop_dir: path/to/preop
  postop_dir: path/to/postop
  adapted_dir: path/to/adapted
  inflow: path/to/inflow.csv
  optimized_params: path/to/optimized_params.csv
  output_config: path/to/output_config.json

calibration:
  data_source:
    mode: mapped_centerline  # mapped_centerline
    mapped_centerline_result: path/to/result_centerline.vtp
    metadata_json: path/to/result_centerline_metadata.json
    centerline: path/to/centerline.vtp
    pressure_array: pressure
    flow_array: flow
    flow_observation_type: flow  # flow | velocity; required
    area_array: null
    branch_id_array: BranchId
    path_array: Path
  parameters:
    vessels:
      default: [R_poiseuille, C, L]
      overrides: {}
    junctions:
      default: [R_poiseuille, L]
      overrides: {}
  solver:
    initial_damping_factor: 1.0
    maximum_iterations: 100
    tolerance_gradient: 1e-6
    tolerance_increment: 1e-10
    parameter_ratio_warning_threshold: 100.0
    confirmation_absolute_tolerance: 1e-8
    confirmation_relative_tolerance: 1e-6
    pressure_bound_multiplier: 10.0
    flow_bound_multiplier: 10.0
    cycle_stability_tolerance: 1e-3
    replay_minimum_cycles: 3
    replay_maximum_cycles: 10
    required_consecutive_stable_pairs: 1
  input_normalization:
    infinite_vessel_compliance: error  # error | zero
  observation_qc:
    vessel_flow_continuity_tolerance: 0.10
    junction_mass_balance_tolerance: 0.10
    root_waveform_rms_tolerance: 0.10
    minimum_pressure_drop_fraction: 0.95
    minimum_path_coverage: 0.99
    minimum_usable_samples: 3
    enforcement: strict_network  # strict_network | target_focused
  # Optional during the version-1 compatibility window. Required when
  # observation_qc.enforcement is target_focused.
  targets:
    mpa_pressure:
      vessel: branch0_seg0
      interface: external_upstream
      weight: 1.0
      normalized_rms_tolerance: 0.05
    rpa_flow_split:
      rpa_vessel: branch1_seg0
      lpa_vessel: branch2_seg0
      interface: external_downstream
      weight: 1.0
      absolute_tolerance: 0.02
    require_improvement_over_baseline: true

bcs:
  type: impedance  # impedance | rcr
  compliance_model: constant
  is_pulmonary: true
  tune_space:
    free:
      - name: lpa.alpha
        init: 0.9
        lb: 0.7
        ub: 0.99
        to_native: identity
        from_native: identity
    fixed:
      - name: d_min
        value: 0.01
    tied: []
  rcr_params: [R_LPA, C_LPA, R_RPA, C_RPA]

trees:
  d_min: 0.01
  use_mean: true
  specify_diameter: true
  optimized_params_csv: optimized_params.csv
  lpa:
    lrr: 10.0
    diameter: 0.3
    d_min: 0.01
    alpha: 0.9
    beta: 0.6
    inductance: 0.0
    compliance:
      model: constant
      params:
        value: 66000.0
  rpa:
    lrr: 10.0
    diameter: 0.3
    d_min: 0.01
    alpha: 0.9
    beta: 0.6
    inductance: 0.0
    compliance:
      model: constant
      params:
        value: 66000.0

adaptation:
  model: M2
  method: cwss
  location: uniform
  iterations: 10
  territory_scheme: lpa_rpa
  mode: predict
  parameter_set: {}  # e.g. {max_nodes: 200000, wss_gain: 0.01}

adapt_benchmark:
  study_id: tst-stan-1-reduced-pa
  output_dir: benchmark-results
  models: [M1, M2, M3]
  tree_params_csv: path/to/optimized_params.csv
  clinical_targets_csv: path/to/clinical_targets.csv
  parameter_overrides:
    M1:
      wss_gain: 0.01
    M3:
      k_arr: [1.0, 1.0, 1.0, 1.0]
  scenarios:
    - name: baseline
      patient_id: tst-stan-1
      scenario_group: medium_dmin0p05
      perturbation_severity: medium
      preop_rri_config: path/to/preop_simplified_zerod_tuned_RRI.json
      postop_rri_config: path/to/postop_simplified_zerod_tuned_RRI.json
      parameter_overrides:
        M1:
          t_end: 3600.0

pipeline:
  run_steady: true
  optimize_bcs: true
  run_threed: true
  adapt: true

threed:
  mesh_scale_factor: 1.0
  convert_to_cm: false
  wall_model: rigid  # rigid | deformable
  elasticity_modulus: 5062674.563165
  poisson_ratio: 0.5
  shell_thickness: 0.12
  prestress_file: auto  # auto | from_steady_mean | path/to/prestress_result.vtu
  prestress_file_path: path/to/prestress_result.vtu
  tissue_support:
    enabled: true
    type: uniform  # uniform | spatial
    stiffness: 1000.0
    damping: 10000.0
    apply_along_normal_direction: true
    spatial_values_file_path: null
  execution:
    mode: slurm  # slurm | local
    executable: /path/to/svmultiphysics  # required
    submit_command: sbatch
    clean_command: clean
    slurm:
      nodes: 3
      procs_per_node: 24
      memory: 16
      hours: 20
      partition: amarsden
      qos: normal
  solver_paths:
    svpre: svpre
    svsolver: svsolver
    svpost: postsolver
    svzerodsolver_build_dir: /path/to/svZeroDSolver-build
    svzerod_interface_library: /path/to/svZeroDSolver-build/src/interface/libsvzero_interface.so

postprocess:
  figures:
    - kind: generation_metrics
      input: path/to/tree.pkl
      output: figures/generation_metrics.png
      options:
        time_window: [0.0, 1.0]
        exclude_collapsed: true
  analyses:
    - kind: pulmonary_resistance_map
      output: results/resistance_map
      options:
        svslicer_path: ~/Documents/Stanford/PhD/Marsden_Lab/SimVascular/svSlicer/Release/svslicer
        centerline: path/to/centerlines.vtp
        frames_csv: path/to/frames.csv
        cycle_duration_s: 1.0
        workers: auto
        keep_intermediate_centerlines: false
    - kind: pulmonary_threed_suite
      output: results/postprocess
      options:
        simulation_dir: path/to/preop
        centerline: path/to/centerlines.vtp
        svslicer_path: ~/Documents/Stanford/PhD/Marsden_Lab/SimVascular/svSlicer/Release/svslicer
        clinical_targets: path/to/clinical_targets.csv
        stage: preop
        inflow_csv: path/to/inflow.csv
        resistance_map_workers: auto
""".strip() + "\n"
