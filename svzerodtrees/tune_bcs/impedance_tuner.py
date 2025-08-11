from .base import BoundaryConditionTuner
import numpy as np
from scipy.optimize import minimize, Bounds
from ..simulation.threedutils import vtp_info
from ..tune_bcs.pa_config import PAConfig  # assuming your project structure
from ..microvasculature import TreeParameters, compliance
import csv

class ImpedanceTuner(BoundaryConditionTuner):
    def __init__(self, 
                 config_handler, 
                 mesh_surfaces_path, 
                 clinical_targets,
                 initial_guess=None,
                 rescale_inflow=True, 
                 n_procs=24, 
                 d_min=0.01, 
                 alpha=0.9,
                 beta=0.6,
                 tol=0.01,
                 compliance_model: str = None, 
                 is_pulmonary=True, 
                 convert_to_cm=True,
                 log_file=None):
        super().__init__(config_handler, mesh_surfaces_path, clinical_targets)
        self.rescale_inflow = rescale_inflow
        self.n_procs = n_procs
        self.d_min = d_min
        self.alpha = alpha
        self.beta = beta
        self.tol = tol
        self.compliance_model = compliance_model
        self.is_pulmonary = is_pulmonary
        self.convert_to_cm = convert_to_cm
        self.log_file = log_file


    def tune(self):
        # --- Geometry and Model Checks ---
        if self.is_pulmonary:
            rpa_info, lpa_info, inflow_info = vtp_info(self.mesh_surfaces_path, convert_to_cm=self.convert_to_cm, pulmonary=True)

        if self.compliance_model.lower() not in ['olufsen', 'constant']:
            raise ValueError(f'Unknown compliance model: {self.compliance}')

        rpa_mean_dia = np.mean([(area / np.pi)**0.5 * 2 for area in rpa_info.values()])
        lpa_mean_dia = np.mean([(area / np.pi)**0.5 * 2 for area in lpa_info.values()])
        print(f'RPA mean diameter: {rpa_mean_dia:.3f}, LPA mean diameter: {lpa_mean_dia:.3f}')


        if lpa_mean_dia > 0.4:
            print(f'LPA mean diameter: {lpa_mean_dia:.3f} too large, rescaling to 0.4 cm')
            lpa_mean_dia = 0.4

        if rpa_mean_dia > 0.4:
            print(f'RPA mean diameter: {rpa_mean_dia:.3f} too large, rescaling to 0.4 cm')
            rpa_mean_dia = 0.4
        


        # --- Configuration Setup ---
        if len(self.config_handler.vessel_map.values()) == 5:
            pa_config = PAConfig.from_pa_config(self.config_handler, self.clinical_targets, self.compliance_model)
        else:
            pa_config = PAConfig.from_config_handler(self.config_handler, self.clinical_targets, self.compliance_model)
            if self.convert_to_cm:
                pa_config.convert_to_cm()

        if self.rescale_inflow:
            scale = ((len(lpa_info.values()) + len(rpa_info.values())) // 2)
            pa_config.bcs['INFLOW'].Q = [q / scale for q in pa_config.bcs['INFLOW'].Q]

        
        if initial_guess:
            print(f'Using initial guess: {initial_guess}')
        else:
            # --- Grid Search for Initial K2 for olufsen compliance ---
            if self.compliance_model.lower() == 'olufsen':
                min_loss = 1e5
                k2_guess = 0
                for k2 in [-10, -25, -50, -75]:
                    p_loss, _, loss = self.loss_fn([k2, k2, lpa_mean_dia, rpa_mean_dia, 10.0], pa_config, grid_search=True)
                    if p_loss < min_loss:
                        min_loss = p_loss
                        k2_guess = k2
                
                initial_guess = [k2_guess, k2_guess, lpa_mean_dia, rpa_mean_dia, 10.0]
                bounds = Bounds([-np.inf, -np.inf, 0.01, 0.01, 1.0], [np.inf]*5)
            
            elif self.compliance_model.lower() == 'constant':
                min_loss = 1e5
                compliance = 0
                for compliance in [3.3e4, 6.6e4, 1e5, 1.3e5]: # 25, 50, 75, 100 mmHg
                    p_loss, _, loss = self.loss_fn([compliance, compliance, lpa_mean_dia, rpa_mean_dia, 10.0], pa_config, grid_search=True)
                    if p_loss < min_loss:
                        min_loss = p_loss
                        compliance_guess = compliance

                initial_guess = [compliance_guess, compliance_guess, lpa_mean_dia, rpa_mean_dia, 10.0]
                bounds = Bounds([0.0, 0.0, 0.01, 0.01, 1.0], [np.inf]*5)

        result = minimize(self.loss_fn, initial_guess, args=(pa_config, False), method='Nelder-Mead', bounds=bounds, options={'maxiter': 100})

        print(f"Optimized impedance parameters: {result.x}")
        pa_config.simulate()
        pa_config.plot_mpa()

        return result

    def loss_fn(self, params, pa_config, grid_search=False):
        # params should take the form [compliance_lpa, compliance_rpa, lpa_mean_dia, rpa_mean_dia, lrr]
        lpa_mean_dia = params[-3]
        rpa_mean_dia = params[-2]
        lrr = params[-1]

        if self.compliance_model.lower() == 'olufsen':
            k1_l = k1_r = 19992500.0
            k3_l = k3_r = 0.0
            lpa_compliance = compliance.OlufsenCompliance(k1=k1_l, k2=params[0], k3=k3_l)
            rpa_compliance = compliance.OlufsenCompliance(k1=k1_r, k2=params[1], k3=k3_r)
        
        elif self.compliance_model.lower() == 'constant': # NEED TO MOVE THIS TO CONSTANT COMPLIANCE LOSS
            lpa_compliance = compliance.ConstantCompliance(params[0])
            rpa_compliance = compliance.ConstantCompliance(params[1])

        else:
            raise ValueError(f'Unknown compliance model: {self.compliance_model}')

        lpa_params = TreeParameters(name='lpa',
                                    lrr=lrr,
                                    diameter=lpa_mean_dia,
                                    d_min=self.d_min,
                                    alpha=self.alpha,
                                    beta=self.beta,
                                    compliance_model=lpa_compliance)

        rpa_params = TreeParameters(name='rpa',
                                    lrr=lrr,
                                    diameter=rpa_mean_dia,
                                    d_min=self.d_min,
                                    alpha=self.alpha,
                                    beta=self.beta,
                                    compliance_model=rpa_compliance)

        try:
            pa_config.create_impedance_trees(lpa_params, rpa_params, self.n_procs)
            pa_config.to_json(f'pa_config_test_tuning.json')
            # check for NaNs
            if np.isnan(pa_config.bcs['LPA_BC'].Z[0]) or np.isnan(pa_config.bcs['RPA_BC'].Z[0]):
                print("NaN detected in boundary conditions, returning high loss")
                return (5e5, 5e5, 1e6) if grid_search else 1e6
            else:
                pa_config.simulate()
            print(f'pa config SIMULATED, rpa split: {pa_config.rpa_split}, p_mpa = {pa_config.P_mpa}\n params: {params}')
            pa_config.plot_mpa(path='figures/pa_config_plot.png')
        except:
            return (5e5, 5e5, 1e6) if grid_search else 1e6

        

        weights = np.array([1.5, 1, 1.2]) if self.clinical_targets.mpa_p[1] >= self.clinical_targets.wedge_p else np.array([1, 0, 1])
        pressure_loss = np.sum(np.dot(np.abs(np.array(pa_config.P_mpa) - np.array(self.clinical_targets.mpa_p)) / self.clinical_targets.mpa_p, weights))**2 * 100
        flowsplit_loss = ((pa_config.rpa_split - self.clinical_targets.rpa_split) / self.clinical_targets.rpa_split)**2 * 100
        total_loss = pressure_loss + flowsplit_loss

        output_params_rows = [
            lpa_params.to_csv_row(loss=total_loss, flow_split=1 - pa_config.rpa_split, p_mpa=pa_config.P_mpa),
            rpa_params.to_csv_row(loss=total_loss, flow_split=pa_config.rpa_split, p_mpa=pa_config.P_mpa)
        ]

        # Determine all keys across all rows
        all_keys = sorted({key for row in output_params_rows for key in row})

        with open("optimized_params.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            for row in output_params_rows:
                writer.writerow(row)

        print(f'\n***PRESSURE LOSS: {pressure_loss}, FS LOSS: {flowsplit_loss}, TOTAL LOSS: {total_loss} ***\n')

        if grid_search:
            return pressure_loss, flowsplit_loss, total_loss
        return total_loss

