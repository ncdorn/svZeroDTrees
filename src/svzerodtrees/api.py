from dataclasses import asdict, is_dataclass
from typing import Any, Dict
import os
import pandas as pd

from .config import (
    AdaptationConfig,
    BCSConfig,
    BaseConfig,
    ImpedanceConfig,
    PathsConfig,
    PipelineConfig,
    ThreeDConfig,
    TreesConfig,
    impedance_config_to_mapping,
    load_config,
)
from .simulation.simulation import Simulation
from .io import ConfigHandler
from .tune_bcs.assign_bcs import assign_rcr_bcs, construct_impedance_trees
from .tune_bcs.clinical_targets import ClinicalTargets
from .tune_bcs.impedance_tuner import ImpedanceTuner
from .tune_bcs.rcr_tuner import RCRTuner, write_rcr_params_csv
from .microvasculature.treeparams import TreeParameters
from .adaptation.microvascular_adaptor import MicrovascularAdaptor
from .adaptation.benchmark import run_adaptation_benchmark_study
from .adaptation.workflow import run_structured_tree_adaptation
from .calibration import calibrate_0d_from_mapped_centerline
from .simulation.simulation_directory import SimulationDirectory
from .tuning.iteration import run_impedance_tuning_for_iteration
from .tuning import generate_full_pa_learned_seed
import pickle


def compute_pulmonary_resistance_map(**kwargs):
    from .post_processing.resistance_map import compute_pulmonary_resistance_map as _impl

    return _impl(**kwargs)


def run_pulmonary_threed_postprocess_suite(**kwargs):
    from .post_processing.pulmonary_threed_suite import (
        run_pulmonary_threed_postprocess_suite as _impl,
    )

    return _impl(**kwargs)


_RCR_PARAM_COLUMNS = ("R_LPA", "C_LPA", "R_RPA", "C_RPA")


def _load_rcr_params_csv(csv_path: str) -> list[float]:
    df = pd.read_csv(csv_path)
    missing = [column for column in _RCR_PARAM_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"RCR params CSV {csv_path} is missing required columns: {', '.join(missing)}"
        )
    if df.empty:
        raise ValueError(f"RCR params CSV {csv_path} does not contain any rows")
    row = df.iloc[0]
    return [float(row[column]) for column in _RCR_PARAM_COLUMNS]


def _resolve_rcr_params(paths: PathsConfig, bcs: BCSConfig) -> list[float]:
    if bcs.rcr_params is not None:
        values = [float(value) for value in bcs.rcr_params]
        if len(values) != 4:
            raise ValueError("bcs.rcr_params must contain exactly 4 values")
        return values

    if paths.optimized_params is not None:
        return _load_rcr_params_csv(paths.optimized_params)

    default_csv = os.path.join(paths.root, "optimized_rcr_params.csv")
    if os.path.exists(default_csv):
        return _load_rcr_params_csv(default_csv)

    raise ValueError(
        "bcs.rcr_params is required for rcr tree construction unless "
        "paths.optimized_params or paths.root/optimized_rcr_params.csv is available"
    )


def _impedance_config_mapping(bcs: BCSConfig) -> Dict[str, Any]:
    """Return one serializable impedance contract for workflow dispatch."""

    impedance = getattr(bcs, "impedance", None)
    if isinstance(impedance, ImpedanceConfig) or impedance is not None:
        return impedance_config_to_mapping(impedance)

    # SimpleNamespace-based callers and older programmatic clients may still
    # construct BCSConfig-shaped objects without the new nested block.  Keep
    # that boundary compatible while making the service input explicit.
    tune_space = getattr(bcs, "tune_space", None)
    payload: Dict[str, Any] = {
        "tuning_model": "rri",
        "compliance_model": getattr(bcs, "compliance_model", "constant"),
        "tune_space": None,
    }
    if tune_space is not None:
        payload["tune_space"] = {
            "free": [
                {
                    "name": item.name,
                    "init": float(item.init),
                    "lb": float(item.lb),
                    "ub": float(item.ub),
                    "to_native": getattr(item.to_native, "__name__", "identity"),
                    "from_native": getattr(item.from_native, "__name__", "identity"),
                }
                for item in tune_space.free
            ],
            "fixed": [
                {"name": item.name, "value": float(item.value)}
                for item in tune_space.fixed
            ],
            "tied": [
                {
                    "name": item.name,
                    "other": item.other,
                    "fn": getattr(item.fn, "__name__", "identity"),
                }
                for item in tune_space.tied
            ],
        }
    return payload


def _resolve_seed_source(config: BaseConfig) -> tuple[Any, Dict[str, Any]]:
    """Resolve the static or generated seed used by a workflow.

    Static seed paths retain their existing value.  Learned generation is
    deliberately resolved once at the API boundary so a failed generation
    cannot reach Simulation or the full-PA tuning service.
    """

    paths = config.paths
    seed_generation = getattr(config, "seed_generation", None)
    if seed_generation is None:
        return paths.zerod_config, {}

    generated = generate_full_pa_learned_seed(seed_generation)
    seed_path = getattr(generated, "seed_path", None)
    if seed_path is None:
        seed_path = getattr(generated, "generated_seed_path", None)
    metadata_path = getattr(generated, "metadata_path", None)
    if isinstance(generated, dict):
        seed_path = generated.get("seed_path", generated.get("generated_seed_path"))
        metadata_path = generated.get("metadata_path")
    if seed_path is None or metadata_path is None:
        raise TypeError(
            "generate_full_pa_learned_seed must return seed_path and metadata_path"
        )

    # Simulation and the tuning service have historically received strings.
    # Do not resolve, basename, or otherwise rewrite the generated path.
    return os.fspath(seed_path), {
        "learned_seed": os.fspath(seed_path),
        "learned_seed_metadata": os.fspath(metadata_path),
    }


class PipelineWorkflow:
    def __init__(self, config: BaseConfig):
        self.config = config

    @classmethod
    def from_config(cls, config: BaseConfig) -> "PipelineWorkflow":
        return cls(config)

    def run(self) -> Dict[str, Any]:
        cfg = self.config
        paths = cfg.paths
        seed_config, seed_provenance = _resolve_seed_source(cfg)
        bcs = cfg.bcs
        adaptation = cfg.adaptation
        pipeline = cfg.pipeline
        threed = cfg.threed

        sim_kwargs: Dict[str, Any] = {
            "path": paths.root,
            "clinical_targets": paths.clinical_targets,
            "preop_dir": os.path.basename(paths.preop_dir) if paths.preop_dir else "preop",
            "postop_dir": os.path.basename(paths.postop_dir) if paths.postop_dir else "postop",
            "zerod_config": os.path.basename(seed_config) if seed_config else "zerod_config.json",
        }
        if pipeline is not None and not pipeline.adapt:
            # Avoid initializing adapted simulation directories when adaptation is disabled.
            sim_kwargs["adapted_dir"] = None
        else:
            sim_kwargs["adapted_dir"] = (
                os.path.basename(paths.adapted_dir) if paths.adapted_dir else "adapted"
            )

        if bcs is not None:
            bcs_type = getattr(bcs, "type", None)
            if bcs_type is None and getattr(bcs, "impedance", None) is not None:
                bcs_type = "impedance"
            sim_kwargs["bc_type"] = bcs_type
            impedance = getattr(bcs, "impedance", None)
            if impedance is not None:
                # Nested impedance is canonical.  Pass its legacy RRI
                # controls through to Simulation as objects where that
                # implementation still expects them; full_pa consumes the
                # serialized contract below.
                sim_kwargs["compliance_model"] = (
                    impedance.get("compliance_model", "constant")
                    if isinstance(impedance, dict)
                    else getattr(impedance, "compliance_model", "constant")
                )
                nested_tune_space = (
                    impedance.get("tune_space")
                    if isinstance(impedance, dict)
                    else getattr(impedance, "tune_space", None)
                )
                if nested_tune_space is not None:
                    sim_kwargs["tune_space"] = nested_tune_space
            else:
                sim_kwargs["compliance_model"] = getattr(
                    bcs, "compliance_model", "constant"
                )
                if getattr(bcs, "tune_space", None) is not None:
                    sim_kwargs["tune_space"] = bcs.tune_space
            if str(bcs_type).lower() == "impedance":
                sim_kwargs["impedance_config"] = _impedance_config_mapping(bcs)
                if (
                    sim_kwargs["impedance_config"].get("tuning_model") == "full_pa"
                    and seed_config is not None
                ):
                    # Preserve a seed nested below paths.root.  The legacy
                    # Simulation path mapping used only basenames, which
                    # would silently point full_pa at the wrong seed.
                    sim_kwargs["zerod_config"] = seed_config

        if adaptation is not None:
            if is_dataclass(adaptation):
                sim_kwargs["adaptation_config"] = asdict(adaptation)
            elif isinstance(adaptation, dict):
                sim_kwargs["adaptation_config"] = adaptation
            else:
                sim_kwargs["adaptation_config"] = vars(adaptation)

        if threed is not None:
            tissue_support = getattr(threed, "tissue_support", None)
            if is_dataclass(tissue_support):
                tissue_support = asdict(tissue_support)
            elif tissue_support is not None and not isinstance(tissue_support, dict):
                tissue_support = vars(tissue_support)
            sim_kwargs["mesh_scale_factor"] = threed.mesh_scale_factor
            sim_kwargs["convert_to_cm"] = threed.convert_to_cm
            sim_kwargs["wall_model"] = threed.wall_model
            sim_kwargs["elasticity_modulus"] = threed.elasticity_modulus
            sim_kwargs["poisson_ratio"] = threed.poisson_ratio
            sim_kwargs["shell_thickness"] = threed.shell_thickness
            sim_kwargs["prestress_file"] = threed.prestress_file
            sim_kwargs["prestress_file_path"] = threed.prestress_file_path
            sim_kwargs["tissue_support"] = tissue_support
            if getattr(threed, "execution", None) is not None:
                sim_kwargs["execution_config"] = (
                    asdict(threed.execution)
                    if is_dataclass(threed.execution)
                    else vars(threed.execution)
                )

        if paths.inflow is not None:
            sim_kwargs["inflow_path"] = paths.inflow

        sim = Simulation(**sim_kwargs)
        pipeline_result = sim.run_pipeline(
            run_steady=pipeline.run_steady if pipeline else True,
            optimize_bcs=pipeline.optimize_bcs if pipeline else True,
            run_threed=pipeline.run_threed if pipeline else True,
            adapt=pipeline.adapt if pipeline else True,
        )

        result = {
            "status": "ok",
            "root": paths.root,
        }
        if isinstance(pipeline_result, dict):
            result.update(pipeline_result)
        for key, value in seed_provenance.items():
            result.setdefault(key, value)
        return result


class TuneBCsWorkflow:
    def __init__(self, config: BaseConfig):
        self.config = config

    @classmethod
    def from_config(cls, config: BaseConfig) -> "TuneBCsWorkflow":
        return cls(config)

    def run(self) -> Dict[str, Any]:
        cfg = self.config
        paths = cfg.paths
        bcs = cfg.bcs
        threed = cfg.threed

        if bcs is None:
            raise ValueError("bcs section is required for tune_bcs workflow")
        if paths.zerod_config is None and getattr(cfg, "seed_generation", None) is None:
            raise ValueError("paths.zerod_config is required for tune_bcs workflow")
        if paths.clinical_targets is None:
            raise ValueError("paths.clinical_targets is required for tune_bcs workflow")
        if paths.mesh_surfaces is None:
            raise ValueError("paths.mesh_surfaces is required for tune_bcs workflow")
        seed_config, seed_provenance = _resolve_seed_source(cfg)
        if seed_config is None:
            raise ValueError(
                "paths.zerod_config or seed_generation is required for tune_bcs workflow"
            )

        convert_to_cm = threed.convert_to_cm if threed else False

        bcs_type = getattr(bcs, "type", None)
        if bcs_type is None and getattr(bcs, "impedance", None) is not None:
            bcs_type = "impedance"

        if bcs_type == "impedance":
            impedance_config = _impedance_config_mapping(bcs)
            if impedance_config.get("tuning_model") == "full_pa":
                result = run_impedance_tuning_for_iteration(
                    iteration_dir=paths.root,
                    seed_config=seed_config,
                    mesh_surfaces=paths.mesh_surfaces,
                    clinical_targets=paths.clinical_targets,
                    inflow_path=paths.inflow,
                    impedance_config=impedance_config,
                    results_dir=paths.root,
                )
                workflow_result = {"status": "ok", **result}
                for key, value in seed_provenance.items():
                    workflow_result.setdefault(key, value)
                return workflow_result
            reduced_config = ConfigHandler.from_json(
                seed_config, is_pulmonary=getattr(bcs, "is_pulmonary", True)
            )
            targets = ClinicalTargets.from_csv(paths.clinical_targets)
            impedance = getattr(bcs, "impedance", None)
            tune_space = getattr(bcs, "tune_space", None)
            compliance_model = getattr(bcs, "compliance_model", "constant")
            if impedance is not None:
                if isinstance(impedance, dict):
                    tune_space = impedance.get("tune_space", tune_space)
                    compliance_model = impedance.get(
                        "compliance_model", compliance_model
                    )
                else:
                    tune_space = getattr(impedance, "tune_space", tune_space)
                    compliance_model = getattr(
                        impedance, "compliance_model", compliance_model
                    )
            if tune_space is None:
                raise ValueError("bcs.tune_space is required for impedance tuning")
            tuner = ImpedanceTuner(
                reduced_config,
                paths.mesh_surfaces,
                targets,
                tune_space,
                rescale_inflow=True,
                convert_to_cm=convert_to_cm,
                compliance_model=compliance_model,
                solver="Nelder-Mead",
                grid_search_init=True,
                log_file=os.path.join(paths.root, "stree_impedance_optimization.log"),
                n_procs=24,
                inflow_path=paths.inflow,
            )
            tuner.tune(nm_iter=5)
            output_csv = os.path.join(paths.root, "optimized_params.csv")
        elif bcs_type == "rcr":
            reduced_config = ConfigHandler.from_json(
                seed_config, is_pulmonary=getattr(bcs, "is_pulmonary", True)
            )
            targets = ClinicalTargets.from_csv(paths.clinical_targets)
            tuner = RCRTuner(
                reduced_config,
                paths.mesh_surfaces,
                targets,
                rescale_inflow=True,
                convert_to_cm=convert_to_cm,
                n_procs=24,
            )
            result = tuner.tune()
            output_csv = os.path.join(paths.root, "optimized_rcr_params.csv")
            write_rcr_params_csv(result.x, output_csv)
        else:
            raise ValueError("bcs.type must be 'impedance' or 'rcr'")

        return {
            "status": "ok",
            "optimized_params": output_csv,
        }


class ConstructTreesWorkflow:
    def __init__(self, config: BaseConfig):
        self.config = config

    @classmethod
    def from_config(cls, config: BaseConfig) -> "ConstructTreesWorkflow":
        return cls(config)

    def _load_tree_params(self, trees: TreesConfig, paths: PathsConfig) -> Dict[str, TreeParameters]:
        if trees.optimized_params_csv:
            df = pd.read_csv(trees.optimized_params_csv)
            lpa = TreeParameters.from_row(df[df["pa"].str.lower() == "lpa"])
            rpa = TreeParameters.from_row(df[df["pa"].str.lower() == "rpa"])
            return {"lpa": lpa, "rpa": rpa}
        if trees.lpa is None or trees.rpa is None:
            raise ValueError("trees.lpa and trees.rpa are required when optimized_params_csv is not provided")
        return {"lpa": trees.lpa, "rpa": trees.rpa}

    def run(self) -> Dict[str, Any]:
        cfg = self.config
        paths = cfg.paths
        bcs = cfg.bcs
        trees = cfg.trees
        threed = cfg.threed

        if bcs is None:
            raise ValueError("bcs section is required for construct_trees workflow")
        if trees is None:
            raise ValueError("trees section is required for construct_trees workflow")
        if paths.zerod_config is None:
            raise ValueError("paths.zerod_config is required for construct_trees workflow")
        if paths.clinical_targets is None:
            raise ValueError("paths.clinical_targets is required for construct_trees workflow")
        if paths.mesh_surfaces is None:
            raise ValueError("paths.mesh_surfaces is required for construct_trees workflow")

        convert_to_cm = threed.convert_to_cm if threed else False

        config_handler = ConfigHandler.from_json(paths.zerod_config, is_pulmonary=bcs.is_pulmonary)
        targets = ClinicalTargets.from_csv(paths.clinical_targets)

        if bcs.type == "impedance":
            params = self._load_tree_params(trees, paths)
            construct_impedance_trees(
                config_handler,
                paths.mesh_surfaces,
                targets.wedge_p,
                params["lpa"],
                params["rpa"],
                d_min=trees.d_min,
                convert_to_cm=convert_to_cm,
                is_pulmonary=bcs.is_pulmonary,
                use_mean=trees.use_mean,
                specify_diameter=trees.specify_diameter,
            )
        elif bcs.type == "rcr":
            assign_rcr_bcs(
                config_handler,
                paths.mesh_surfaces,
                targets.wedge_p,
                _resolve_rcr_params(paths, bcs),
                convert_to_cm=convert_to_cm,
                is_pulmonary=bcs.is_pulmonary,
            )
        else:
            raise ValueError("bcs.type must be 'impedance' or 'rcr'")

        output_path = paths.output_config or os.path.join(paths.root, "svzerod_config_with_bcs.json")
        config_handler.to_json(output_path)

        return {
            "status": "ok",
            "output_config": output_path,
        }


class AdaptationWorkflow:
    def __init__(self, config: BaseConfig):
        self.config = config

    @classmethod
    def from_config(cls, config: BaseConfig) -> "AdaptationWorkflow":
        return cls(config)

    def run(self) -> Dict[str, Any]:
        cfg = self.config
        paths = cfg.paths
        bcs = cfg.bcs
        adaptation = cfg.adaptation
        threed = cfg.threed

        if paths.preop_dir is None or paths.postop_dir is None or paths.adapted_dir is None:
            raise ValueError("paths.preop_dir, paths.postop_dir, paths.adapted_dir are required for adapt workflow")
        if paths.clinical_targets is None:
            raise ValueError("paths.clinical_targets is required for adapt workflow")
        if paths.zerod_config is None:
            raise ValueError("paths.zerod_config is required for adapt workflow")

        convert_to_cm = threed.convert_to_cm if threed else False

        preop = SimulationDirectory.from_directory(paths.preop_dir, convert_to_cm=convert_to_cm)
        postop = SimulationDirectory.from_directory(paths.postop_dir, convert_to_cm=convert_to_cm)
        adapted = SimulationDirectory.from_directory(paths.adapted_dir, convert_to_cm=convert_to_cm)

        targets = ClinicalTargets.from_csv(paths.clinical_targets)

        tree_params = paths.optimized_params
        if tree_params is None:
            tree_params = os.path.join(paths.root, "optimized_params.csv")

        if bcs and bcs.type != "impedance":
            raise ValueError("adapt workflow currently supports impedance BCs only")
        return run_structured_tree_adaptation(
            preop_dir=paths.preop_dir,
            postop_dir=paths.postop_dir,
            adapted_dir=paths.adapted_dir,
            clinical_targets=paths.clinical_targets,
            reduced_order_pa=paths.zerod_config,
            tree_params=tree_params,
            model=adaptation.model if adaptation else "M2",
            territory_scheme=adaptation.territory_scheme if adaptation else "lpa_rpa",
            parameter_set=adaptation.parameter_set if adaptation else None,
            mode=adaptation.mode if adaptation else "predict",
            convert_to_cm=convert_to_cm,
            output_root=paths.adapted_dir,
        )


class AdaptBenchmarkWorkflow:
    def __init__(self, config: BaseConfig):
        self.config = config

    @classmethod
    def from_config(cls, config: BaseConfig) -> "AdaptBenchmarkWorkflow":
        return cls(config)

    def run(self) -> Dict[str, Any]:
        benchmark = self.config.adapt_benchmark
        if benchmark is None:
            raise ValueError("adapt_benchmark section is required for adapt_benchmark workflow")
        return run_adaptation_benchmark_study(benchmark)


class PostprocessWorkflow:
    def __init__(self, config: BaseConfig):
        self.config = config

    @classmethod
    def from_config(cls, config: BaseConfig) -> "PostprocessWorkflow":
        return cls(config)

    def _load_tree(self, path: str):
        with open(path, "rb") as fh:
            return pickle.load(fh)

    def run(self) -> Dict[str, Any]:
        cfg = self.config
        postprocess = cfg.postprocess
        if postprocess is None:
            raise ValueError("postprocess section is required for postprocess workflow")

        outputs = []
        for fig in postprocess.figures:
            options = fig.options or {}
            if fig.kind == "generation_metrics":
                from .post_processing.tree_figures import generation_metrics

                tree = self._load_tree(fig.input)
                fig_obj, _, _, _ = generation_metrics.plot_generation_metrics_for_tree(tree, **options)
            elif fig.kind == "generation_waveforms":
                from .post_processing.tree_figures import generation_waveforms

                tree = self._load_tree(fig.input)
                fig_obj, _, _ = generation_waveforms.plot_generation_waveforms_for_tree(tree, **options)
            elif fig.kind == "visualize_hemodynamics":
                from .post_processing.tree_figures import visualize_hemodynamics

                tree = self._load_tree(fig.input)
                fig_obj, _ = visualize_hemodynamics.plot_tree_metrics_by_generation(
                    results=tree.results,
                    **options,
                )
            else:
                raise ValueError(f"Unknown postprocess kind '{fig.kind}'")

            if fig.output:
                fig_obj.savefig(fig.output, dpi=options.get("dpi", 300), bbox_inches="tight")
                outputs.append(fig.output)
            else:
                outputs.append(None)

        analysis_outputs = []
        for analysis in postprocess.analyses:
            options = analysis.options or {}
            if analysis.kind == "pulmonary_resistance_map":
                analysis_outputs.append(
                    compute_pulmonary_resistance_map(
                        output_dir=analysis.output,
                        **options,
                    )
                )
            elif analysis.kind == "pulmonary_threed_suite":
                analysis_outputs.append(
                    run_pulmonary_threed_postprocess_suite(
                        output_dir=analysis.output,
                        **options,
                    )
                )
            else:
                raise ValueError(f"Unknown postprocess analysis kind '{analysis.kind}'")

        return {
            "status": "ok",
            "outputs": outputs,
            "analysis_outputs": analysis_outputs,
        }


class Calibrate0DFrom3DWorkflow:
    def __init__(self, config: BaseConfig):
        self.config = config

    @classmethod
    def from_config(cls, config: BaseConfig) -> "Calibrate0DFrom3DWorkflow":
        return cls(config)

    def run(self) -> Dict[str, Any]:
        paths = self.config.paths
        calibration = self.config.calibration

        if paths.zerod_config is None:
            raise ValueError("paths.zerod_config is required for calibrate_0d_from_3d workflow")
        if paths.output_config is None:
            raise ValueError("paths.output_config is required for calibrate_0d_from_3d workflow")
        if calibration is None:
            raise ValueError("calibration section is required for calibrate_0d_from_3d workflow")

        return calibrate_0d_from_mapped_centerline(
            zerod_config_path=paths.zerod_config,
            output_config_path=paths.output_config,
            calibration=calibration,
        )


WORKFLOW_MAP = {
    "pipeline": PipelineWorkflow,
    "tune_bcs": TuneBCsWorkflow,
    "construct_trees": ConstructTreesWorkflow,
    "adapt": AdaptationWorkflow,
    "adapt_benchmark": AdaptBenchmarkWorkflow,
    "postprocess": PostprocessWorkflow,
    "calibrate_0d_from_3d": Calibrate0DFrom3DWorkflow,
}


def run_from_config_file(path: str) -> Dict[str, Any]:
    cfg = load_config(path)
    workflow_cls = WORKFLOW_MAP[cfg.workflow]
    return workflow_cls.from_config(cfg).run()
