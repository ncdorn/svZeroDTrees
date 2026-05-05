from .base import BoundaryConditionTuner
import copy
import numpy as np
from scipy.optimize import minimize
from ..simulation.threedutils import pa_outlet_scale_from_branch_counts, vtp_info
from ..tune_bcs.pa_config import PAConfig
from ..tune_bcs.assign_bcs import construct_impedance_trees
from ..microvasculature import TreeParameters, compliance as comp_mod
from ..microvasculature.structured_tree.asymmetry import resolve_branch_scaling
from ..tune_bcs.tune_space import TuneSpace
from ..io.inflow_handler import mean_flow_from_path
from ..io.blocks.boundary_condition import (
    validate_boundary_condition_configs,
    validate_flow_cardiac_output_config,
    validate_impedance_timing_config,
)
import csv, json, math, os

class ImpedanceTuner(BoundaryConditionTuner):
    def __init__(self,
                 config_handler,
                 mesh_surfaces_path,
                 clinical_targets,
                 tune_space: TuneSpace,             # <— NEW
                 compliance_model: str = "constant",
                 grid_search_init: bool = True,
                 grid_candidates_constant = (3.3e4, 6.6e4, 1.0e5, 1.3e5),
                 grid_candidates_olufsen = (-10.0, -25.0, -50.0, -75.0),
                 rescale_inflow=True,
                 n_procs=24,
                 tol=0.01,
                 is_pulmonary=True,
                 convert_to_cm=True,
                 log_file=None,
                 maxiter=200,
                 solver="Powell",
                 inflow_path=None,
                 tuning_model="rri",
                 d_min=0.01,
                 use_mean=True,
                 specify_diameter=True,
                 diameter_scale=0.0,
                 diameter_std_cap=None,
                 allow_ordered_outlet_mapping=False):
        super().__init__(config_handler, mesh_surfaces_path, clinical_targets)
        self.tune_space = tune_space
        self.compliance_model = (compliance_model or "").lower()
        if self.compliance_model not in ("olufsen", "constant"):
            raise ValueError(f"Unknown compliance model: {self.compliance_model}")
        self.tuning_model = str(tuning_model or "rri").strip().lower()
        if self.tuning_model not in {"rri", "full_pa"}:
            raise ValueError("tuning_model must be one of rri|full_pa")

        self.rescale_inflow = rescale_inflow
        self.n_procs = n_procs
        self.tol = tol
        self.is_pulmonary = is_pulmonary
        self.convert_to_cm = convert_to_cm
        self.log_file = log_file
        self.maxiter = maxiter
        self.solver = solver
        self.inflow_path = os.path.abspath(inflow_path) if inflow_path is not None else None
        self.d_min = float(d_min)
        self.use_mean = bool(use_mean)
        self.specify_diameter = bool(specify_diameter)
        self.diameter_scale = float(diameter_scale)
        self.diameter_std_cap = None if diameter_std_cap is None else float(diameter_std_cap)
        self.allow_ordered_outlet_mapping = bool(allow_ordered_outlet_mapping)

        # grid search params
        self.grid_search_init = grid_search_init
        self.grid_candidates_constant = tuple(grid_candidates_constant)
        self.grid_candidates_olufsen = tuple(grid_candidates_olufsen)

        # will be filled in tune()
        self._geom_defaults = {}
        self._augmented_mode = False
        self._loss_weights = None
        self._last_loss_breakdown = {}
        self._opt_csv_path = None
        self._expected_snapshot_cardiac_output = None
        self._full_pa_base_config = None

    def _tree_assignment_options_for_objective(self):
        if self.tuning_model != "full_pa":
            return {
                "use_mean": self.use_mean,
                "diameter_scale": self.diameter_scale,
                "diameter_std_cap": self.diameter_std_cap,
            }
        return {
            "use_mean": True,
            "diameter_scale": 0.0,
            "diameter_std_cap": None,
        }


    # ---- Internal helpers ---- #

    def _find_free(self, name: str):
        for i, p in enumerate(self.tune_space.free):
            if p.name == name:
                return i, p
        return None, None
    
    def _set_free_native(self, x, name: str, native_value: float):
        idx, p = self._find_free(name)
        if p is None:
            return x  # silently ignore if that knob isn't free
        x = np.array(x, dtype=float, copy=True)
        x[idx] = p.from_native(native_value)
        return x

    def _make_pa_config(self):
        if len(self.config_handler.vessel_map.values()) == 5:
            pa_config = PAConfig.from_pa_config(self.config_handler, self.clinical_targets, self.compliance_model)
        else:
            pa_config = PAConfig.from_config_handler(self.config_handler, self.clinical_targets, self.compliance_model)
            if self.convert_to_cm:
                pa_config.convert_to_cm()
        return pa_config

    def _make_tuning_model(self):
        if self.tuning_model == "full_pa":
            return copy.deepcopy(self.config_handler)
        return self._make_pa_config()

    def _prepare_geometry_defaults(self):
        rpa_info, lpa_info, _ = vtp_info(self.mesh_surfaces_path, convert_to_cm=self.convert_to_cm, pulmonary=True)
        rpa_mean_d = np.mean([(area / np.pi) ** 0.5 * 2 for area in rpa_info.values()])
        lpa_mean_d = np.mean([(area / np.pi) ** 0.5 * 2 for area in lpa_info.values()])
        # clamp for safety
        rpa_mean_d = rpa_mean_d
        lpa_mean_d = lpa_mean_d
        self._geom_defaults = {
            "rpa.default_diameter": rpa_mean_d,
            "lpa.default_diameter": lpa_mean_d,
            "n_outlets_scale": pa_outlet_scale_from_branch_counts(
                len(lpa_info),
                len(rpa_info),
            ),
        }

    def _compute_inflow_cardiac_output(self, inflow_bc) -> float:
        if self.inflow_path is not None:
            return mean_flow_from_path(self.inflow_path)
        flow = np.asarray(inflow_bc.Q, dtype=float)
        return float(np.mean(flow))

    @staticmethod
    def _compute_boundary_condition_mean_flow(inflow_bc) -> float:
        flow = np.asarray(inflow_bc.Q, dtype=float)
        return float(np.mean(flow))

    def _resolve_expected_snapshot_cardiac_output(self, inflow_bc) -> float:
        if self.inflow_path is not None:
            cardiac_output = mean_flow_from_path(self.inflow_path)
        elif not self.rescale_inflow:
            cardiac_output = self._compute_boundary_condition_mean_flow(inflow_bc)
        else:
            raise ValueError(
                "rescale_inflow=True requires inflow_path so snapshot scaling uses "
                "the patient inflow.csv mean flow as the source of truth"
            )
        if not self.rescale_inflow:
            return cardiac_output
        if self.tuning_model == "full_pa":
            return cardiac_output

        scale = float(self._geom_defaults["n_outlets_scale"])
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"invalid pulmonary outlet scale for inflow rescaling: {scale}"
            )
        return cardiac_output / scale

    def _build_compliance(self, side: str, params: dict[str, float]):
        if self.compliance_model == "olufsen":
            k1 = 19992500.0
            k2 = params[f"comp.{side}.k2"]
            k3 = 0.0
            return comp_mod.OlufsenCompliance(k1=k1, k2=k2, k3=k3)
        else:
            cval = params[f"comp.{side}.C"]
            return comp_mod.ConstantCompliance(cval)

    def _build_tree_params(self, params: dict[str, float]) -> tuple[TreeParameters, TreeParameters]:
        # fall back to geometry defaults if diameter not in params
        lpa_d = params.get("lpa.diameter", self._geom_defaults["lpa.default_diameter"])
        rpa_d = params.get("rpa.diameter", self._geom_defaults["rpa.default_diameter"])
        lrr   = params.get("lrr", 10.0)
        alpha_l, beta_l = resolve_branch_scaling(
            alpha=params.get("lpa.alpha"),
            beta=params.get("lpa.beta"),
            xi=params.get("lpa.xi"),
            eta_sym=params.get("lpa.eta_sym"),
        )
        alpha_r, beta_r = resolve_branch_scaling(
            alpha=params.get("rpa.alpha"),
            beta=params.get("rpa.beta"),
            xi=params.get("rpa.xi"),
            eta_sym=params.get("rpa.eta_sym"),
        )
        d_min   = params.get("d_min", 0.01)
        lpa_inductance = params.get("lpa.inductance", 0.0)
        rpa_inductance = params.get("rpa.inductance", 0.0)

        lpa_comp = self._build_compliance("lpa", params)
        rpa_comp = self._build_compliance("rpa", params)

        lpa_params = TreeParameters(name="lpa", lrr=lrr, diameter=lpa_d, d_min=d_min,
                                    alpha=alpha_l, beta=beta_l, compliance_model=lpa_comp,
                                    xi=params.get("lpa.xi"), eta_sym=params.get("lpa.eta_sym"),
                                    inductance=lpa_inductance)
        rpa_params = TreeParameters(name="rpa", lrr=lrr, diameter=rpa_d, d_min=d_min,
                                    alpha=alpha_r, beta=beta_r, compliance_model=rpa_comp,
                                    xi=params.get("rpa.xi"), eta_sym=params.get("rpa.eta_sym"),
                                    inductance=rpa_inductance)
        return lpa_params, rpa_params
    
    def _grid_search_init(self, pa_config, x0: np.ndarray) -> np.ndarray:
        # detect which compliance knobs exist
        if self.compliance_model == "constant":
            lname, rname = "comp.lpa.C", "comp.rpa.C"
            candidates = self.grid_candidates_constant
        else:
            lname, rname = "comp.lpa.k2", "comp.rpa.k2"
            candidates = self.grid_candidates_olufsen

        l_idx, l_p = self._find_free(lname)
        r_idx, r_p = self._find_free(rname)
        if l_p is None and r_p is None:
            # no compliance knobs are free -> nothing to grid-search
            return x0

        best_x = x0
        best_f = np.inf

        # probe candidates
        print(f"[grid] probing {len(candidates)} candidates for {lname} and {rname}")
        for c in candidates:
            trial = x0
            
            # set both (if present) to same candidate
            trial = self._set_free_native(trial, lname, c)
            trial = self._set_free_native(trial, rname, c)
            f = self.loss_fn(trial, pa_config)
            if f < best_f:
                best_f, best_x = f, trial

        print(f"[grid] best init loss={best_f:.3f} for candidate={best_x[-1]}")
        return best_x

    @staticmethod
    def _vessel_result(result, vessel):
        by_name = result[result.name == vessel.name]
        if not by_name.empty:
            return by_name
        branch_name = f"branch{vessel.branch}_seg0"
        by_branch = result[result.name == branch_name]
        if not by_branch.empty:
            return by_branch
        raise ValueError(f"simulation result missing vessel '{vessel.name}'")

    def _compute_full_pa_metrics(self, config_handler, result):
        if not hasattr(config_handler, "mpa") or not hasattr(config_handler, "rpa"):
            raise ValueError("full_pa tuning requires pulmonary mpa and rpa vessels")

        mpa_result = self._vessel_result(result, config_handler.mpa)
        rpa_result = self._vessel_result(result, config_handler.rpa)

        cycle_duration = getattr(config_handler.simparams, "cardiac_period", None)
        if cycle_duration is None:
            inflow_times = getattr(config_handler.bcs["INFLOW"], "t", None)
            if inflow_times:
                inflow_arr = np.asarray(inflow_times, dtype=float)
                if inflow_arr.size >= 2:
                    cycle_duration = float(inflow_arr.max() - inflow_arr.min())

        if cycle_duration is not None and np.isfinite(cycle_duration) and cycle_duration > 0.0:
            mpa_time = np.asarray(mpa_result.time, dtype=float)
            rpa_time = np.asarray(rpa_result.time, dtype=float)
            mpa_result = mpa_result[mpa_time > mpa_time.max() - float(cycle_duration)]
            rpa_result = rpa_result[rpa_time > rpa_time.max() - float(cycle_duration)]

        if mpa_result.empty or rpa_result.empty:
            raise ValueError("full_pa simulation result has no usable last-cycle data")

        pressure = np.asarray(mpa_result.pressure_in, dtype=float) / 1333.2
        mpa_flow = np.asarray(mpa_result.flow_in, dtype=float)
        rpa_flow = np.asarray(rpa_result.flow_in, dtype=float)
        mpa_time = np.asarray(mpa_result.time, dtype=float)
        rpa_time = np.asarray(rpa_result.time, dtype=float)

        if mpa_time.size >= 2 and rpa_time.size >= 2:
            total_flow = float(np.trapz(mpa_flow, mpa_time))
            rpa_total = float(np.trapz(rpa_flow, rpa_time))
        else:
            total_flow = float(np.mean(mpa_flow))
            rpa_total = float(np.mean(rpa_flow))
        if total_flow == 0.0:
            raise ValueError("full_pa MPA flow is zero; cannot compute RPA split")

        return {
            "P_mpa": [float(np.max(pressure)), float(np.min(pressure)), float(np.mean(pressure))],
            "rpa_split": rpa_total / total_flow,
        }

    @staticmethod
    def _impedance_bcs_are_finite(model):
        for bc in model.bcs.values():
            if getattr(bc, "type", None) != "IMPEDANCE":
                continue
            z_values = getattr(bc, "Z", None)
            if z_values is not None and np.asarray(z_values, dtype=float).size:
                if not np.all(np.isfinite(np.asarray(z_values, dtype=float))):
                    return False
        return True

    def _write_and_validate_snapshot(self, model):
        model.to_json('pa_config_tuning_snapshot.json')
        with open('pa_config_tuning_snapshot.json', encoding='utf-8') as ff:
            snapshot_payload = json.load(ff)
        validate_boundary_condition_configs(snapshot_payload.get("boundary_conditions", []))
        validate_impedance_timing_config(snapshot_payload)
        expected_snapshot_cardiac_output = self._expected_snapshot_cardiac_output
        if expected_snapshot_cardiac_output is None:
            expected_snapshot_cardiac_output = self._compute_inflow_cardiac_output(
                model.bcs["INFLOW"]
            )
        validate_flow_cardiac_output_config(
            snapshot_payload,
            expected_cardiac_output=expected_snapshot_cardiac_output,
        )

    def _evaluate_model(self, x: np.ndarray, provided_model=None):
        params = self.tune_space.vector_to_param_dict(x)
        lpa_params, rpa_params = self._build_tree_params(params)

        if self.tuning_model == "full_pa":
            base_model = self._full_pa_base_config if self._full_pa_base_config is not None else provided_model
            if base_model is None:
                raise ValueError("full_pa tuning model has not been initialized")
            model = copy.deepcopy(base_model)
            objective_tree_options = self._tree_assignment_options_for_objective()
            construct_impedance_trees(
                model,
                self.mesh_surfaces_path,
                self.clinical_targets.wedge_p,
                lpa_params,
                rpa_params,
                d_min=self.d_min,
                convert_to_cm=self.convert_to_cm,
                is_pulmonary=self.is_pulmonary,
                n_procs=self.n_procs,
                use_mean=objective_tree_options["use_mean"],
                specify_diameter=self.specify_diameter,
                diameter_scale=objective_tree_options["diameter_scale"],
                diameter_std_cap=objective_tree_options["diameter_std_cap"],
                allow_ordered_outlet_mapping=self.allow_ordered_outlet_mapping,
                verbose=False,
            )
            self._write_and_validate_snapshot(model)
            if not self._impedance_bcs_are_finite(model):
                raise ValueError("full_pa impedance BC contains non-finite values")
            result = model.simulate()
            metrics = self._compute_full_pa_metrics(model, result)
            print(
                f'full PA config SIMULATED, rpa split: {metrics["rpa_split"]}, '
                f'p_mpa = {metrics["P_mpa"]}\n params: {params}'
            )
            return model, metrics, lpa_params, rpa_params, params

        model = self._full_pa_base_config if self._full_pa_base_config is not None else provided_model
        if model is None:
            raise ValueError("RRI tuning model has not been initialized")
        lpa_params, rpa_params = self._build_tree_params(params)
        model.create_impedance_trees(lpa_params, rpa_params, self.n_procs)
        self._write_and_validate_snapshot(model)
        if (np.isnan(model.bcs['LPA_BC'].Z[0]) or
            np.isnan(model.bcs['RPA_BC'].Z[0])):
            raise ValueError("RRI impedance BC contains non-finite values")
        model.simulate()
        metrics = {
            "P_mpa": model.P_mpa,
            "rpa_split": model.rpa_split,
        }
        print(
            f'pa config SIMULATED, rpa split: {model.rpa_split}, '
            f'p_mpa = {model.P_mpa}\n params: {params}'
        )
        return model, metrics, lpa_params, rpa_params, params
    

    # ---- Main tuning routine ---- #

    def tune(self, nm_iter: int = 1):
        # set log path if not provided
        if self.log_file is None:
            base_dir = getattr(self.config_handler, "path", None)
            if base_dir is not None:
                base_dir = os.path.dirname(os.path.abspath(base_dir))
            else:
                base_dir = os.getcwd()
            self.log_file = os.path.join(base_dir, "impedance_tuning.log")
        else:
            base_dir = os.path.dirname(os.path.abspath(self.log_file))
        self._opt_csv_path = os.path.join(base_dir, "optimized_params.csv")

        def _append_log(msg: str):
            with open(self.log_file, "a") as lf:
                lf.write(msg + "\n")

        self._prepare_geometry_defaults()
        pa_config = self._make_tuning_model()
        self._expected_snapshot_cardiac_output = self._resolve_expected_snapshot_cardiac_output(
            pa_config.bcs["INFLOW"]
        )

        if self.rescale_inflow:
            current_mean_flow = self._compute_boundary_condition_mean_flow(pa_config.bcs["INFLOW"])
            if not np.isfinite(current_mean_flow) or current_mean_flow == 0.0:
                raise ValueError(
                    f"invalid inflow mean flow for rescaling: {current_mean_flow}"
                )
            scale_factor = self._expected_snapshot_cardiac_output / current_mean_flow
            pa_config.bcs['INFLOW'].Q = [q * scale_factor for q in pa_config.bcs['INFLOW'].Q]
        self._full_pa_base_config = pa_config

        x0, bounds = self.tune_space.pack_init_and_bounds()

        # ——— Grid search init on compliance ———
        if self.grid_search_init:
            x0 = self._grid_search_init(pa_config, x0)

        # ——— Optionally run Nelder-Mead multiple times ———
        repeats = nm_iter if self.solver == "Nelder-Mead" else 1
        max_runs = max(1, repeats)
        x_init = x0
        result = None
        best_x = None
        best_unweighted_loss = np.inf
        # augmented Lagrangian-style weight updating
        self._augmented_mode = True
        self._loss_weights = {
            "sys": 1.0,
            "dia": 1.0,
            "mean": 1.0,
            "flow": 1.0,
            "reg": 1.0,
        }
        self._last_loss_breakdown = {}
        penalty_growth = 5.0
        max_penalty = 1e6

        for run_idx in range(max_runs):
            result = minimize(
                fun=lambda x: self.loss_fn(x, pa_config),
                x0=x_init,
                method=self.solver,
                bounds=bounds if self.solver in ("Nelder-Mead", "L-BFGS-B", "Powell", "TNC", "SLSQP", "trust-constr") else None,
                options={"maxiter": self.maxiter}
            )
            unweighted_loss = self._last_loss_breakdown.get("unweighted_loss", np.inf)
            metrics = self._last_loss_breakdown.get("metrics", {})
            accepted = True
            if best_x is None:
                best_x = result.x
                best_unweighted_loss = unweighted_loss
            elif unweighted_loss > best_unweighted_loss:
                accepted = False
            else:
                best_x = result.x
                best_unweighted_loss = unweighted_loss
            print(
                f"Nelder-Mead run {run_idx + 1}/{max_runs} complete: "
                f"weighted loss={result.fun}, unweighted loss={unweighted_loss}, weights={self._loss_weights}"
            )
            _append_log(
                f"Nelder-Mead run {run_idx + 1}/{max_runs} complete: "
                f"weighted loss={result.fun:.6e}, unweighted loss={unweighted_loss:.6e}, "
                f"weights={self._loss_weights}, "
                f"pressures={metrics.get('sys_pressure', np.nan):.6f}/"
                f"{metrics.get('dia_pressure', np.nan):.6f}/"
                f"{metrics.get('mean_pressure', np.nan):.6f} mmHg, "
                f"pressure_targets={self.clinical_targets.mpa_p[0]:.6f}/"
                f"{self.clinical_targets.mpa_p[1]:.6f}/"
                f"{self.clinical_targets.mpa_p[2]:.6f} mmHg, "
                f"rpa_split={metrics.get('rpa_split', np.nan):.6f}, "
                f"rpa_split_target={self.clinical_targets.rpa_split:.6f}"
            )
            if accepted:
                x_init = result.x
                self.loss_fn(result.x, pa_config, finalize=True)
            else:
                x_init = best_x
                _append_log(
                    "Nelder-Mead run rejected: "
                    f"unweighted loss {unweighted_loss:.6e} exceeded previous {best_unweighted_loss:.6e}"
                )
            if unweighted_loss < 1e-5:
                break
            if run_idx == max_runs - 1:
                break

            components = self._last_loss_breakdown.get("components", {})
            finite_components = {k: v for k, v in components.items() if np.isfinite(v)}
            total_residual = sum(finite_components.values()) if finite_components else 0.0
            for key in ["sys", "dia", "mean", "flow"]:
                residual = components.get(key, np.inf)
                if not np.isfinite(residual) or total_residual <= 0.0:
                    continue
                share = residual / total_residual
                updated_weight = self._loss_weights[key] * (1.0 + penalty_growth * share)
                self._loss_weights[key] = min(updated_weight, max_penalty)

            print(f"Updated loss weights for next run: {self._loss_weights}")
            _append_log(f"Updated loss weights for next run: {self._loss_weights}")
            if not np.isfinite(unweighted_loss):
                break

        print(f"[ImpedanceTuner] Optimized: {best_x}  f={best_unweighted_loss:.3f}")
        # final simulate & plot
        _ = self.loss_fn(best_x, pa_config, finalize=True)
        if self.tuning_model == "rri":
            pa_config.plot_mpa()
        return result
    

    # ---- Loss function ---- #

    def loss_fn(self, x: np.ndarray, pa_config, finalize: bool=False) -> float:
        try:
            model, metrics, lpa_params, rpa_params, params = self._evaluate_model(
                x,
                provided_model=pa_config,
            )
        except Exception as e:
            params = self.tune_space.vector_to_param_dict(x)
            print(f"[loss_fn] simulation error: {e} params={params}")
            return 1e9

        # ---- Loss: weighted MPA pressure (sys/dia/mean separately) + flow split + mild L2 on compliance ----
        pressure_weights = (
            {"sys": 1.5, "dia": 1.0, "mean": 1.2}
            if (self.clinical_targets.mpa_p[1] >= self.clinical_targets.wedge_p)
            else {"sys": 1.0, "dia": 0.0, "mean": 1.0}
        )
        p_mpa = metrics["P_mpa"]
        rpa_split = metrics["rpa_split"]
        pressure_diff = np.abs(np.array(p_mpa) - np.array(self.clinical_targets.mpa_p)) / self.clinical_targets.mpa_p
        pressure_components = {
            "sys": pressure_diff[0] ** 2,
            "dia": pressure_diff[1] ** 2,
            "mean": pressure_diff[2] ** 2,
        }
        pressure_contrib = {
            "sys": pressure_weights["sys"] * pressure_components["sys"] * 100.0,
            "dia": pressure_weights["dia"] * pressure_components["dia"] * 100.0,
            "mean": pressure_weights["mean"] * pressure_components["mean"] * 100.0,
        }
        pressure_loss = (
            pressure_contrib["sys"] +
            pressure_contrib["dia"] +
            pressure_contrib["mean"]
        )
        flowsplit_loss = ((rpa_split - self.clinical_targets.rpa_split) / self.clinical_targets.rpa_split)**2 * 100.0

        if self.compliance_model == "olufsen":
            l2 = 1e-3 * (params["comp.lpa.k2"]**2 + params["comp.rpa.k2"]**2)
        else:
            l2 = 1e-5 * (params["comp.lpa.C"]**2 + params["comp.rpa.C"]**2)

        base_total = float(pressure_loss + flowsplit_loss + l2)
        loss_weights = self._loss_weights or {"sys": 1.0, "dia": 1.0, "mean": 1.0, "flow": 1.0, "reg": 1.0}
        weighted_total = (
            loss_weights.get("sys", 1.0) * pressure_contrib["sys"] +
            loss_weights.get("dia", 1.0) * pressure_contrib["dia"] +
            loss_weights.get("mean", 1.0) * pressure_contrib["mean"] +
            loss_weights.get("flow", 1.0) * flowsplit_loss +
            loss_weights.get("reg", 1.0) * l2
        )
        unweighted_loss = pressure_loss + flowsplit_loss + l2

        if self._augmented_mode:
            self._last_loss_breakdown = {
                "weighted_loss": weighted_total,
                "unweighted_loss": unweighted_loss,
                "components": {
                    "sys": pressure_contrib["sys"],
                    "dia": pressure_contrib["dia"],
                    "mean": pressure_contrib["mean"],
                    "flow": flowsplit_loss,
                    "reg": l2,
                },
                "metrics": {
                    "sys_pressure": p_mpa[0],
                    "dia_pressure": p_mpa[1],
                    "mean_pressure": p_mpa[2],
                    "rpa_split": rpa_split,
                },
                "params": params,
            }

        if finalize:
            final_loss = weighted_total if self._augmented_mode else base_total
            # Write once on final call
            rows = [
                lpa_params.to_csv_row(loss=final_loss, flow_split=1 - rpa_split, p_mpa=p_mpa),
                rpa_params.to_csv_row(loss=final_loss, flow_split=rpa_split, p_mpa=p_mpa)
            ]
            keys = sorted({k for r in rows for k in r})
            csv_path = self._opt_csv_path or "optimized_params.csv"
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

        return weighted_total if self._augmented_mode else base_total
