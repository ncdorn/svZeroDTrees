from dataclasses import asdict, is_dataclass
from typing import Any, Dict
import os
import pandas as pd

from .config import BaseConfig, PathsConfig, BCSConfig, TreesConfig, AdaptationConfig, PipelineConfig, ThreeDConfig
from .config import load_config
from .simulation import Simulation
from .io import ConfigHandler
from .tune_bcs import ImpedanceTuner, RCRTuner, ClinicalTargets, construct_impedance_trees, assign_rcr_bcs
from .microvasculature.treeparams import TreeParameters
from .adaptation import MicrovascularAdaptor
from .simulation.simulation_directory import SimulationDirectory
from .post_processing.tree_figures import generation_metrics, generation_waveforms
from .post_processing.tree_figures import visualize_hemodynamics
import pickle


class PipelineWorkflow:
    def __init__(self, config: BaseConfig):
        self.config = config

    @classmethod
    def from_config(cls, config: BaseConfig) -> "PipelineWorkflow":
        return cls(config)

    def run(self) -> Dict[str, Any]:
        cfg = self.config
        paths = cfg.paths
        bcs = cfg.bcs
        adaptation = cfg.adaptation
        pipeline = cfg.pipeline
        threed = cfg.threed

        sim_kwargs: Dict[str, Any] = {
            "path": paths.root,
            "clinical_targets": paths.clinical_targets,
            "preop_dir": os.path.basename(paths.preop_dir) if paths.preop_dir else "preop",
            "postop_dir": os.path.basename(paths.postop_dir) if paths.postop_dir else "postop",
            "zerod_config": os.path.basename(paths.zerod_config) if paths.zerod_config else "zerod_config.json",
        }
        if pipeline is not None and not pipeline.adapt:
            # Avoid initializing adapted simulation directories when adaptation is disabled.
            sim_kwargs["adapted_dir"] = None
        else:
            sim_kwargs["adapted_dir"] = (
                os.path.basename(paths.adapted_dir) if paths.adapted_dir else "adapted"
            )

        if bcs is not None:
            sim_kwargs["bc_type"] = bcs.type
            sim_kwargs["compliance_model"] = bcs.compliance_model
            if bcs.tune_space is not None:
                sim_kwargs["tune_space"] = bcs.tune_space

        if adaptation is not None:
            if is_dataclass(adaptation):
                sim_kwargs["adaptation_config"] = asdict(adaptation)
            elif isinstance(adaptation, dict):
                sim_kwargs["adaptation_config"] = adaptation
            else:
                sim_kwargs["adaptation_config"] = vars(adaptation)

        if threed is not None:
            sim_kwargs["mesh_scale_factor"] = threed.mesh_scale_factor
            sim_kwargs["convert_to_cm"] = threed.convert_to_cm
            sim_kwargs["wall_model"] = threed.wall_model
            sim_kwargs["elasticity_modulus"] = threed.elasticity_modulus
            sim_kwargs["poisson_ratio"] = threed.poisson_ratio
            sim_kwargs["shell_thickness"] = threed.shell_thickness
            sim_kwargs["prestress_file"] = threed.prestress_file
            sim_kwargs["prestress_file_path"] = threed.prestress_file_path

        if paths.inflow is not None:
            sim_kwargs["inflow_path"] = paths.inflow

        sim = Simulation(**sim_kwargs)
        sim.run_pipeline(
            run_steady=pipeline.run_steady if pipeline else True,
            optimize_bcs=pipeline.optimize_bcs if pipeline else True,
            run_threed=pipeline.run_threed if pipeline else True,
            adapt=pipeline.adapt if pipeline else True,
        )

        return {
            "status": "ok",
            "root": paths.root,
        }


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
        if paths.zerod_config is None:
            raise ValueError("paths.zerod_config is required for tune_bcs workflow")
        if paths.clinical_targets is None:
            raise ValueError("paths.clinical_targets is required for tune_bcs workflow")
        if paths.mesh_surfaces is None:
            raise ValueError("paths.mesh_surfaces is required for tune_bcs workflow")

        convert_to_cm = threed.convert_to_cm if threed else False

        reduced_config = ConfigHandler.from_json(paths.zerod_config, is_pulmonary=bcs.is_pulmonary)
        targets = ClinicalTargets.from_csv(paths.clinical_targets)

        if bcs.type == "impedance":
            if bcs.tune_space is None:
                raise ValueError("bcs.tune_space is required for impedance tuning")
            tuner = ImpedanceTuner(
                reduced_config,
                paths.mesh_surfaces,
                targets,
                bcs.tune_space,
                rescale_inflow=True,
                convert_to_cm=convert_to_cm,
                compliance_model=bcs.compliance_model,
                solver="Nelder-Mead",
                grid_search_init=True,
                log_file=os.path.join(paths.root, "stree_impedance_optimization.log"),
                n_procs=24,
                inflow_path=paths.inflow,
            )
            tuner.tune(nm_iter=5)
            output_csv = os.path.join(paths.root, "optimized_params.csv")
        elif bcs.type == "rcr":
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
            pd.DataFrame([result.x], columns=["R_LPA", "C_LPA", "R_RPA", "C_RPA"]).to_csv(output_csv, index=False)
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
            if bcs.rcr_params is None:
                raise ValueError("bcs.rcr_params is required for rcr tree construction")
            assign_rcr_bcs(
                config_handler,
                paths.mesh_surfaces,
                targets.wedge_p,
                bcs.rcr_params,
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

        adaptor = MicrovascularAdaptor(
            preop,
            postop,
            adapted,
            targets,
            reduced_order_pa=paths.zerod_config,
            tree_params=tree_params,
            method=adaptation.method if adaptation else "cwss",
            location=adaptation.location if adaptation else "uniform",
            n_iter=adaptation.iterations if adaptation else 10,
            bc_type="impedance",
            convert_to_cm=convert_to_cm,
        )
        adaptor.adapt()

        return {
            "status": "ok",
            "adapted_dir": paths.adapted_dir,
        }


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
                tree = self._load_tree(fig.input)
                fig_obj, _, _, _ = generation_metrics.plot_generation_metrics_for_tree(tree, **options)
            elif fig.kind == "generation_waveforms":
                tree = self._load_tree(fig.input)
                fig_obj, _, _ = generation_waveforms.plot_generation_waveforms_for_tree(tree, **options)
            elif fig.kind == "visualize_hemodynamics":
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

        return {
            "status": "ok",
            "outputs": outputs,
        }


WORKFLOW_MAP = {
    "pipeline": PipelineWorkflow,
    "tune_bcs": TuneBCsWorkflow,
    "construct_trees": ConstructTreesWorkflow,
    "adapt": AdaptationWorkflow,
    "postprocess": PostprocessWorkflow,
}


def run_from_config_file(path: str) -> Dict[str, Any]:
    cfg = load_config(path)
    workflow_cls = WORKFLOW_MAP[cfg.workflow]
    return workflow_cls.from_config(cfg).run()
