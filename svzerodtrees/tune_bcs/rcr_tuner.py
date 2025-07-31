from .base import BoundaryConditionTuner
import numpy as np
from scipy.optimize import minimize, Bounds
from ..simulation.threedutils import vtp_info
from ..tune_bcs.pa_config import PAConfig  # assuming your project structure
import os

class RCRTuner(BoundaryConditionTuner):
    def __init__(self, config_handler, mesh_surfaces_path, clinical_targets,
                 rescale_inflow=True, n_procs=24, tol=0.01,
                 is_pulmonary=True, convert_to_cm=True,
                 log_file=None):
        super().__init__(config_handler, mesh_surfaces_path, clinical_targets)
        self.rescale_inflow = rescale_inflow
        self.n_procs = n_procs
        self.tol = tol
        self.is_pulmonary = is_pulmonary
        self.convert_to_cm = convert_to_cm
        self.log_file = log_file

    def tune(self):
        # --- Geometry and Model Checks ---
        if self.is_pulmonary:
            rpa_info, lpa_info, inflow_info = vtp_info(self.mesh_surfaces_path, convert_to_cm=self.convert_to_cm, pulmonary=True)

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

        pa_config.initialize_resistance_bcs() # will recognize inflow and create RCR BCs

        # --- Objective Function ---
        def loss_fn(params):
            # R_LPA, C_LPA, R_RPA, C_RPA
            pa_config.bcs['LPA_BC'].R, pa_config.bcs['LPA_BC'].C, pa_config.bcs['RPA_BC'].R, pa_config.bcs['RPA_BC'].C = params

            try:
                pa_config.to_json(f'pa_config_test_tuning.json')
                pa_config.simulate()
                print(f'pa config SIMULATED, rpa split: {pa_config.rpa_split}, p_mpa = {pa_config.P_mpa}\n params: {params}')
                pa_config.plot_mpa(path='figures/pa_config_plot.png')
            except:
                return 1e6
            
            # Loss Function
            weights = np.array([1.5, 1, 1.2]) if self.clinical_targets.mpa_p[1] >= self.clinical_targets.wedge_p else np.array([1, 0, 1])
            pressure_loss = np.sum(np.dot(np.abs(np.array(pa_config.P_mpa) - np.array(self.clinical_targets.mpa_p)) / self.clinical_targets.mpa_p, weights))**2 * 100
            flowsplit_loss = ((pa_config.rpa_split - self.clinical_targets.rpa_split) / self.clinical_targets.rpa_split)**2 * 100
            total_loss = pressure_loss + flowsplit_loss

            with open('optimized_params.csv', 'w') as f:
                f.write(f"R_LPA,C_LPA,R_RPA,C_RPA,Pressure_Loss,Flow_Split_Loss,Total_Loss\n")
                f.write(f"{pa_config.bcs['LPA_BC'].R},{pa_config.bcs['LPA_BC'].C},{pa_config.bcs['RPA_BC'].R},{pa_config.bcs['RPA_BC'].C},{pressure_loss},{flowsplit_loss},{total_loss}\n")

            print(f'\n***PRESSURE LOSS: {pressure_loss}, FS LOSS: {flowsplit_loss}, TOTAL LOSS: {total_loss} ***\n')

            return total_loss

        # --- Optimization Constraints ---
        constraints = [ # constrain capacitances to not be more than twice each other.
            {"type": "ineq", "fun": lambda x: 2 * x[1] - x[3]},  # x[0] ≤ 2 * x[1]
            {"type": "ineq", "fun": lambda x: 2 * x[3] - x[1]}   # x[1] ≤ 2 * x[0]
        ]
        # --- Optimization ---
        initial_guess = [1000.0, 1e-5, 1000.0, 1e-5]  # Initial guess for R and C values
        bounds = Bounds([0.0, 1e-10, 0.0, 1e-10], [np.inf, 1.0, np.inf, 1.0])  # Bounds for R and C values
        result = minimize(loss_fn, initial_guess, method='SLSQP', bounds=bounds, options={'maxiter': 100}, constraints=constraints)

        print(f"Optimized parameters: {result.x}")
        pa_config.simulate()
        pa_config.plot_mpa()

        return result