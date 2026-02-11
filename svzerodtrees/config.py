import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml
import numpy as np

from .tune_bcs.tune_space import FreeParam, FixedParam, TiedParam, TuneSpace, identity, positive, unit_interval
from .microvasculature.treeparams import TreeParameters
from .microvasculature.compliance.constant import ConstantCompliance
from .microvasculature.compliance.olufsen import OlufsenCompliance

CONFIG_VERSION = 1


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
    method: str = "cwss"
    location: str = "uniform"
    iterations: int = 10


@dataclass
class PipelineConfig:
    run_steady: bool = True
    optimize_bcs: bool = True
    run_threed: bool = True
    adapt: bool = True


@dataclass
class ThreeDConfig:
    mesh_scale_factor: float = 1.0
    convert_to_cm: bool = False
    solver_paths: Optional[Dict[str, str]] = None


@dataclass
class PostprocessFigure:
    kind: str
    input: str
    output: Optional[str] = None
    options: Optional[Dict[str, Any]] = None


@dataclass
class PostprocessConfig:
    figures: List[PostprocessFigure]


@dataclass
class BaseConfig:
    version: int
    workflow: str
    paths: PathsConfig
    bcs: Optional[BCSConfig] = None
    trees: Optional[TreesConfig] = None
    adaptation: Optional[AdaptationConfig] = None
    pipeline: Optional[PipelineConfig] = None
    threed: Optional[ThreeDConfig] = None
    postprocess: Optional[PostprocessConfig] = None


def _ensure_keys(data: Dict[str, Any], allowed: List[str], context: str) -> None:
    unknown = set(data.keys()) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown keys in {context}: {sorted(unknown)}")


def _resolve_path(root: str, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if os.path.isabs(value):
        return value
    return os.path.abspath(os.path.join(root, value))


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
            "pipeline",
            "threed",
            "postprocess",
        ],
        "config",
    )

    version = int(raw.get("version", 0))
    if version != CONFIG_VERSION:
        raise ValueError(f"Unsupported config version {version}. Expected {CONFIG_VERSION}.")

    workflow = raw.get("workflow")
    if workflow not in {"pipeline", "tune_bcs", "construct_trees", "adapt", "postprocess"}:
        raise ValueError("workflow must be one of pipeline|tune_bcs|construct_trees|adapt|postprocess")

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
        _ensure_keys(data, ["method", "location", "iterations"], "adaptation")
        adaptation = AdaptationConfig(
            method=data.get("method", "cwss"),
            location=data.get("location", "uniform"),
            iterations=int(data.get("iterations", 10)),
        )

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
        _ensure_keys(data, ["mesh_scale_factor", "convert_to_cm", "solver_paths"], "threed")
        threed = ThreeDConfig(
            mesh_scale_factor=float(data.get("mesh_scale_factor", 1.0)),
            convert_to_cm=bool(data.get("convert_to_cm", False)),
            solver_paths=data.get("solver_paths"),
        )

    postprocess = None
    if raw.get("postprocess") is not None:
        data = raw["postprocess"]
        _ensure_keys(data, ["figures"], "postprocess")
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
        postprocess = PostprocessConfig(figures=figures)

    return BaseConfig(
        version=version,
        workflow=workflow,
        paths=paths,
        bcs=bcs,
        trees=trees,
        adaptation=adaptation,
        pipeline=pipeline,
        threed=threed,
        postprocess=postprocess,
    )


def render_schema() -> str:
    return """
# svzerodtrees config (v1)
version: 1
workflow: pipeline  # pipeline | tune_bcs | construct_trees | adapt | postprocess

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
  method: cwss
  location: uniform
  iterations: 10

pipeline:
  run_steady: true
  optimize_bcs: true
  run_threed: true
  adapt: true

threed:
  mesh_scale_factor: 1.0
  convert_to_cm: false
  solver_paths:
    svpre: svpre
    svsolver: svsolver
    svpost: postsolver

postprocess:
  figures:
    - kind: generation_metrics
      input: path/to/tree.pkl
      output: figures/generation_metrics.png
      options:
        time_window: [0.0, 1.0]
        exclude_collapsed: true
""".strip() + "\n"
