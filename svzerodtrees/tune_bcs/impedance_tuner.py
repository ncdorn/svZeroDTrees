from .base import BoundaryConditionTuner
import numpy as np
from scipy.optimize import minimize, Bounds
from ..simulation.threedutils import vtp_info
from ..tune_bcs.pa_config import PAConfig  # assuming your project structure
import os

class ImpedanceTuner(BoundaryConditionTuner):
    def __init__(self, config_handler, mesh_surfaces_path, clinical_targets,
                 rescale_inflow=True, n_procs=24, d_min=0.01, tol=0.01,
                 compliance="olufsen", is_pulmonary=True, convert_to_cm=True,
                 log_file=None):
        super().__init__(config_handler, mesh_surfaces_path, clinical_targets)
        self.rescale_inflow = rescale_inflow
        self.n_procs = n_procs
        self.d_min = d_min
        self.tol = tol
        self.compliance = compliance
        self.is_pulmonary = is_pulmonary
        self.convert_to_cm = convert_to_cm
        self.log_file = log_file


    def tune(self):
        # --- Geometry and Model Checks ---
        if self.is_pulmonary:
            rpa_info, lpa_info, inflow_info = vtp_info(self.mesh_surfaces_path, convert_to_cm=self.convert_to_cm, pulmonary=True)

        if self.compliance.lower() not in ['olufsen', 'constant']:
            raise ValueError(f'Unknown compliance model: {self.compliance}')

        rpa_mean_dia = np.mean([(area / np.pi)**0.5 * 2 for area in rpa_info.values()])
        lpa_mean_dia = np.mean([(area / np.pi)**0.5 * 2 for area in lpa_info.values()])
        print(f'RPA mean diameter: {rpa_mean_dia:.3f}, LPA mean diameter: {lpa_mean_dia:.3f}')

        # --- Configuration Setup ---
        if len(self.config_handler.vessel_map.values()) == 5:
            pa_config = PAConfig.from_pa_config(self.config_handler, self.clinical_targets)
        else:
            pa_config = PAConfig.from_config_handler(self.config_handler, self.clinical_targets)
            if self.convert_to_cm:
                pa_config.convert_to_cm()

        if self.rescale_inflow:
            scale = ((len(lpa_info.values()) + len(rpa_info.values())) // 2)
            pa_config.bcs['INFLOW'].Q = [q / scale for q in pa_config.bcs['INFLOW'].Q]

        # --- Objective Function ---
        def loss_fn(params, grid_search=False):
            k2_l, k2_r, lpa_d, rpa_d, lrr = params
            k1_l = k1_r = 19992500.0
            k3_l = k3_r = 0.0
            alpha = 0.9
            beta = 0.6

            tree_params = {
                'lpa': [k1_l, k2_l, k3_l, lrr, alpha, beta],
                'rpa': [k1_r, k2_r, k3_r, lrr, alpha, beta]
            }

            try:
                pa_config.create_impedance_trees(lpa_d, rpa_d, [self.d_min, self.d_min], tree_params, self.n_procs)
                pa_config.to_json(f'pa_config_test_tuning.json')
                pa_config.simulate()
                print(f'pa config SIMULATED, rpa split: {pa_config.rpa_split}, p_mpa = {pa_config.P_mpa}\n params: {params}')
                pa_config.plot_mpa(path='figures/pa_config_plot.png')
            except:
                return (5e5, 5e5, 1e6) if grid_search else 1e6

            # check for NaNs
            if np.isnan(pa_config.bcs['LPA_BC'].Z[0]) or np.isnan(pa_config.bcs['RPA_BC'].Z[0]):
                return (5e5, 5e5, 1e6) if grid_search else 1e6

            weights = np.array([1.5, 1, 1.2]) if self.clinical_targets.mpa_p[1] >= self.clinical_targets.wedge_p else np.array([1, 0, 1])
            pressure_loss = np.sum(np.dot(np.abs(np.array(pa_config.P_mpa) - np.array(self.clinical_targets.mpa_p)) / self.clinical_targets.mpa_p, weights))**2 * 100
            flowsplit_loss = ((pa_config.rpa_split - self.clinical_targets.rpa_split) / self.clinical_targets.rpa_split)**2 * 100
            total_loss = pressure_loss + flowsplit_loss

            with open('optimized_params.csv', 'w') as f:
                f.write("pa,k1,k2,k3,lrr,diameter,loss,flow_split,p_mpa\n")
                # do unique for lpa, rpa
                f.write(f'lpa,{k1_l},{k2_l},{k3_l},{lrr},{lpa_mean_dia},{loss},{1 - pa_config.rpa_split},[{pa_config.P_mpa[0]} {pa_config.P_mpa[1]} {pa_config.P_mpa[2]}]\n')
                f.write(f'rpa,{k1_r},{k2_r},{k3_r},{lrr},{rpa_mean_dia},{loss},{pa_config.rpa_split},[{pa_config.P_mpa[0]} {pa_config.P_mpa[1]} {pa_config.P_mpa[2]}]\n')
            
            print(f'\n***PRESSURE LOSS: {pressure_loss}, FS LOSS: {flowsplit_loss}, TOTAL LOSS: {loss} ***\n')

            if grid_search:
                return pressure_loss, flowsplit_loss, total_loss
            return total_loss

        # --- Grid Search for Initial K2 ---
        min_loss = 1e5
        k2_guess = 0
        for k2 in [-10, -25, -50, -75]:
            p_loss, _, loss = loss_fn([k2, k2, lpa_mean_dia, rpa_mean_dia, 10.0], grid_search=True)
            if p_loss < min_loss:
                min_loss = p_loss
                k2_guess = k2

        # --- Optimization ---
        initial_guess = [k2_guess, k2_guess, lpa_mean_dia, rpa_mean_dia, 10.0]
        bounds = Bounds([-np.inf, -np.inf, 0.01, 0.01, 1.0], [np.inf]*5)
        result = minimize(loss_fn, initial_guess, method='Nelder-Mead', bounds=bounds, options={'maxiter': 100})

        print(f"Optimized impedance parameters: {result.x}")
        pa_config.simulate()
        pa_config.plot_mpa()

        return result