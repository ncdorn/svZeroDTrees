from .base import BoundaryConditionTuner
import numpy as np
from scipy.optimize import minimize
from ..simulation.threedutils import vtp_info
from ..tune_bcs.pa_config import PAConfig
from ..microvasculature import TreeParameters, compliance as comp_mod
from ..microvasculature.structured_tree.asymmetry import resolve_branch_scaling
from ..tune_bcs.tune_space import TuneSpace
import csv, math, os

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
                 solver="Powell"):
        super().__init__(config_handler, mesh_surfaces_path, clinical_targets)
        self.tune_space = tune_space
        self.compliance_model = (compliance_model or "").lower()
        if self.compliance_model not in ("olufsen", "constant"):
            raise ValueError(f"Unknown compliance model: {self.compliance_model}")

        self.rescale_inflow = rescale_inflow
        self.n_procs = n_procs
        self.tol = tol
        self.is_pulmonary = is_pulmonary
        self.convert_to_cm = convert_to_cm
        self.log_file = log_file
        self.maxiter = maxiter
        self.solver = solver

        # grid search params
        self.grid_search_init = grid_search_init
        self.grid_candidates_constant = tuple(grid_candidates_constant)
        self.grid_candidates_olufsen = tuple(grid_candidates_olufsen)

        # will be filled in tune()
        self._geom_defaults = {}
        self._augmented_mode = False
        self._loss_weights = None
        self._last_loss_breakdown = {}


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
            "n_outlets_scale": float((len(lpa_info) + len(rpa_info)))/2.0
        }

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

        lpa_comp = self._build_compliance("lpa", params)
        rpa_comp = self._build_compliance("rpa", params)

        lpa_params = TreeParameters(name="lpa", lrr=lrr, diameter=lpa_d, d_min=d_min,
                                    alpha=alpha_l, beta=beta_l, compliance_model=lpa_comp,
                                    xi=params.get("lpa.xi"), eta_sym=params.get("lpa.eta_sym"))
        rpa_params = TreeParameters(name="rpa", lrr=lrr, diameter=rpa_d, d_min=d_min,
                                    alpha=alpha_r, beta=beta_r, compliance_model=rpa_comp,
                                    xi=params.get("rpa.xi"), eta_sym=params.get("rpa.eta_sym"))
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

        def _append_log(msg: str):
            with open(self.log_file, "a") as lf:
                lf.write(msg + "\n")

        self._prepare_geometry_defaults()
        pa_config = self._make_pa_config()

        if self.rescale_inflow:
            scale = self._geom_defaults["n_outlets_scale"]
            if scale and scale > 0:
                pa_config.bcs['INFLOW'].Q = [q / scale for q in pa_config.bcs['INFLOW'].Q]

        x0, bounds = self.tune_space.pack_init_and_bounds()

        # ——— Grid search init on compliance ———
        if self.grid_search_init:
            x0 = self._grid_search_init(pa_config, x0)

        # ——— Optionally run Nelder-Mead multiple times ———
        repeats = nm_iter if self.solver == "Nelder-Mead" else 1
        max_runs = max(1, repeats)
        x_init = x0
        result = None
        # augmented Lagrangian-style weight updating
        self._augmented_mode = True
        self._loss_weights = {
            "pressure": 1.0,
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
            x_init = result.x
            unweighted_loss = self._last_loss_breakdown.get("unweighted_loss", np.inf)
            metrics = self._last_loss_breakdown.get("metrics", {})
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
            if unweighted_loss < 1e-5:
                break
            if run_idx == max_runs - 1:
                break

            components = self._last_loss_breakdown.get("components", {})
            for key in ["pressure", "flow"]:
                residual = components.get(key, np.inf)
                if not np.isfinite(residual):
                    continue
                updated_weight = self._loss_weights[key] * (1.0 + penalty_growth * residual)
                self._loss_weights[key] = min(updated_weight, max_penalty)

            print(f"Updated loss weights for next run: {self._loss_weights}")
            _append_log(f"Updated loss weights for next run: {self._loss_weights}")
            if not np.isfinite(unweighted_loss):
                break

        print(f"[ImpedanceTuner] Optimized: {result.x}  f={result.fun:.3f}")
        # final simulate & plot
        _ = self.loss_fn(result.x, pa_config, finalize=True)
        pa_config.plot_mpa()
        return result
    

    # ---- Loss function ---- #

    def loss_fn(self, x: np.ndarray, pa_config, finalize: bool=False) -> float:
        params = self.tune_space.vector_to_param_dict(x)

        try:
            lpa_params, rpa_params = self._build_tree_params(params)
            pa_config.create_impedance_trees(lpa_params, rpa_params, self.n_procs)
            pa_config.to_json('pa_config_tuning_snapshot.json')
            # sanity check
            if (np.isnan(pa_config.bcs['LPA_BC'].Z[0]) or
                np.isnan(pa_config.bcs['RPA_BC'].Z[0])):
                return 1e9
            pa_config.simulate()
            print(f'pa config SIMULATED, rpa split: {pa_config.rpa_split}, p_mpa = {pa_config.P_mpa}\n params: {params}')
        except Exception as e:
            print(f"[loss_fn] simulation error: {e} params={params}")
            return 1e9

        # ---- Loss: weighted MPA pressure + flow split + mild L2 on compliance ----
        weights = np.array([1.5, 1.0, 1.2]) if (self.clinical_targets.mpa_p[1] >= self.clinical_targets.wedge_p) else np.array([1.0, 0.0, 1.0])
        pressure_loss = np.sum(np.dot(np.abs(np.array(pa_config.P_mpa) - np.array(self.clinical_targets.mpa_p)) / self.clinical_targets.mpa_p, weights))**2 * 100.0
        flowsplit_loss = ((pa_config.rpa_split - self.clinical_targets.rpa_split) / self.clinical_targets.rpa_split)**2 * 100.0

        if self.compliance_model == "olufsen":
            l2 = 1e-3 * (params["comp.lpa.k2"]**2 + params["comp.rpa.k2"]**2)
        else:
            l2 = 1e-5 * (params["comp.lpa.C"]**2 + params["comp.rpa.C"]**2)

        base_total = float(pressure_loss + flowsplit_loss + l2)
        loss_weights = self._loss_weights or {"pressure": 1.0, "flow": 1.0, "reg": 1.0}
        weighted_total = (
            loss_weights.get("pressure", 1.0) * pressure_loss +
            loss_weights.get("flow", 1.0) * flowsplit_loss +
            loss_weights.get("reg", 1.0) * l2
        )
        unweighted_loss = pressure_loss + flowsplit_loss + l2

        if self._augmented_mode:
            self._last_loss_breakdown = {
                "weighted_loss": weighted_total,
                "unweighted_loss": unweighted_loss,
                "components": {
                    "pressure": pressure_loss,
                    "flow": flowsplit_loss,
                    "reg": l2,
                },
                "metrics": {
                    "sys_pressure": pa_config.P_mpa[0],
                    "dia_pressure": pa_config.P_mpa[1],
                    "mean_pressure": pa_config.P_mpa[2],
                    "rpa_split": pa_config.rpa_split,
                },
                "params": params,
            }

        if finalize:
            final_loss = weighted_total if self._augmented_mode else base_total
            # Write once on final call
            rows = [
                lpa_params.to_csv_row(loss=final_loss, flow_split=1 - pa_config.rpa_split, p_mpa=pa_config.P_mpa),
                rpa_params.to_csv_row(loss=final_loss, flow_split=pa_config.rpa_split, p_mpa=pa_config.P_mpa)
            ]
            keys = sorted({k for r in rows for k in r})
            with open("optimized_params.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

        return weighted_total if self._augmented_mode else base_total
