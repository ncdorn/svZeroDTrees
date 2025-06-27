
from .setup import *
from .integrator import run_adaptation
from .models import CWSSIMSAdaptation
from ..tune_bcs import ClinicalTargets
import pandas as pd

'''
this file is for setting up adaptation experiments
'''

def run_single_gain_cwss_ims_adaptation(
        preop_config_path: str,
        postop_config_path: str,
        optimized_tree_params_csv: str,
        clinical_targets_csv: str,
        gain: float,
) -> pd.DataFrame:
    
    print(f"Running single gain adaptation with gain {gain}.")

    preop_pa, postop_pa = initialize_from_paths(
        preop_config_path,
        postop_config_path,
        optimized_tree_params_csv,
        clinical_targets_csv
    )

    K_arrs = [
        [0.0, gain, 0.0, 0.0],
        [gain, 0.0, 0.0, 0.0],
        [0.0, 0.0, gain, 0.0],
        [0.0, 0.0, 0.0, gain]
    ]

    # initialize result list
    rows = []

    for K_arr in K_arrs:
        print(f"Running adaptation with K_arr: {K_arr}")
        result, flow_log, sol, postop_pa = run_adaptation(preop_pa, postop_pa, CWSSIMSAdaptation, K_arr)

        print(f"Adaptation result: {result}")
        
        # Save results
        rows.append(result)

    print("Adaptation complete")


    # create dataframe from results
    results_df = pd.DataFrame(rows)

    return results_df

