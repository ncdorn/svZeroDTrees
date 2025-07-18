
from .setup import *
from .integrator import run_adaptation
from .models import CWSSIMSAdaptation
from ..tune_bcs import ClinicalTargets
import pandas as pd
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

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


def run_single_Karr_worker(args) -> Tuple[List[float], pd.DataFrame]:
    K_arr, preop_config_path, postop_config_path, optimized_tree_params_csv, clinical_targets_csv = args
    print(f"Running parallel adaptation with K_arr {K_arr}")

    # Generate fresh preop/postop objects for this run
    preop_pa, postop_pa = initialize_from_paths(
        preop_config_path,
        postop_config_path,
        optimized_tree_params_csv,
        clinical_targets_csv
    )

    result, flow_log, sol, postop_pa = run_adaptation(
        preop_pa,
        postop_pa,
        CWSSIMSAdaptation,
        K_arr
    )

    # Package result with the tested K_arr
    df = pd.DataFrame([result])
    for i in range(4):
        df[f'K_{i}'] = K_arr[i]
    return (K_arr, df)


def run_parallel_gain_combinations(
    relatives: List[float],
    base_gain: float,
    preop_config_path: str,
    postop_config_path: str,
    optimized_tree_params_csv: str,
    clinical_targets_csv: str,
    max_workers: int = None,
    combinations_csv_path: str = 'K_arr_combinations.csv'
) -> pd.DataFrame:
    """
    Sweep over all combinations of relative gains scaled by base_gain
    for the 4 parameters in K_arr, but keep only combinations
    with at least 2 active (non-zero) gains. Run in parallel.
    """
    # Create all possible combinations
    all_combinations = list(itertools.product(relatives, repeat=4))
    
    # Filter to only those with at least 2 non-zero entries
    valid_combinations = [combo for combo in all_combinations if sum(1 for x in combo if x != 0) >= 2]

    print(f"Generated {len(valid_combinations)} valid combinations (at least 2 active gains) out of {len(all_combinations)} total.")

    # Scale by base_gain
    scaled_K_arrs = [[base_gain * x for x in combo] for combo in valid_combinations]

    combined_df = run_parallel_gains(
        scaled_K_arrs,
        preop_config_path,
        postop_config_path,
        optimized_tree_params_csv,
        clinical_targets_csv,
        max_workers=max_workers,
        combinations_csv_path=combinations_csv_path
    )

    return combined_df


def run_parallel_gains(
    scaled_K_arrs: List[List[float]],
    preop_config_path: str,
    postop_config_path: str,
    optimized_tree_params_csv: str,
    clinical_targets_csv: str,
    max_workers: int = None,
    combinations_csv_path: str = 'K_arr_combinations.csv'
) -> pd.DataFrame:
    """
    Run adaptations for all scaled K_arrs in parallel.
    """
    # Write CSV manifest
    df_manifest = pd.DataFrame(scaled_K_arrs, columns=['K_0', 'K_1', 'K_2', 'K_3'])
    df_manifest.to_csv(combinations_csv_path, index=False)
    print(f"Saved tested K_arr vectors to {combinations_csv_path}")

    # Prepare args for parallel processing
    args_list = [
        (K_arr, preop_config_path, postop_config_path, optimized_tree_params_csv, clinical_targets_csv)
        for K_arr in scaled_K_arrs
    ]


    all_results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_Karr_worker, args): args[0] for args in args_list}

        for future in as_completed(futures):
            K_arr = futures[future]
            try:
                K_values, df = future.result()
                all_results.append(df)
            except Exception as e:
                print(f"K_arr {K_arr} failed with error: {e}")

    combined_df = pd.concat(all_results, ignore_index=True)

    return combined_df

import itertools
import pandas as pd
from typing import List, Tuple

def run_gains(
    scaled_K_arrs: List[List[float]],
    preop_config_path: str,
    postop_config_path: str,
    optimized_tree_params_csv: str,
    clinical_targets_csv: str,
) -> pd.DataFrame:
    """
    Run gain sweeps for the CWSS-IMS adaptation model, testing combinations of gains
    without parallelization. 

    Parameters:
    - preop_config_path: path to preop JSON config
    - postop_config_path: path to postop JSON config
    - optimized_tree_params_csv: CSV of optimized geometry parameters
    - clinical_targets_csv: CSV of clinical flow splits and pressures
    - relative_values: list of scaling values to multiply by base_gain (1e-7)
    - min_active_gains: minimum number of gains that must be non-zero

    Returns:
    - DataFrame logging results for each gain combination
    """

    # Logging list of dicts
    results = []

    for K_arr in scaled_K_arrs:
        
        # Initialize patient-specific PA objects
        preop_pa, postop_pa = initialize_from_paths(
            preop_config_path,
            postop_config_path,
            optimized_tree_params_csv,
            clinical_targets_csv
        )
        
        # Run adaptation
        result, flow_log, sol, postop_pa = run_adaptation(preop_pa, postop_pa, CWSSIMSAdaptation, K_arr)

        results.append(result)

    # Convert to DataFrame for easy analysis / CSV
    results_df = pd.DataFrame(results)

    return results_df