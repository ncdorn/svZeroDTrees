
from .setup import *
from .integrator import run_adaptation, run_adaptation_outsidesim
from .models import CWSSAdaptation, CWSSIMSAdaptation
from ..microvasculature.structured_tree.structuredtree import StructuredTree
from ..tune_bcs.clinical_targets import ClinicalTargets
import pandas as pd
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple
from .utils import (
    append_result_to_csv,
    estimate_steady_tree_hemodynamics,
    pack_state,
    rel_change,
    time_to_95,
    unpack_state,
)
import os
import copy

import numpy as np
from scipy.integrate import solve_ivp
from types import SimpleNamespace

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
        result, flow_log, sol, postop_pa, hists = run_adaptation(
            preop_pa,
            postop_pa,
            CWSSIMSAdaptation,
            K_arr,
        )

        print(f"Adaptation result: {result}")
        
        # Save results
        rows.append(result)

    print("Adaptation complete")


    # create dataframe from results
    results_df = pd.DataFrame(rows)

    return results_df


def run_single_Karr_worker(args) -> Tuple[List[float], pd.DataFrame]:
    K_arr, preop_config_path, postop_config_path, optimized_tree_params_csv, clinical_targets_csv, output_csv_path = args
    print(f"Running adaptation with K_arr {K_arr}")

    try:
        preop_pa, postop_pa = initialize_from_paths(
            preop_config_path, postop_config_path, optimized_tree_params_csv, clinical_targets_csv
        )

        result, flow_log, sol, postop_pa, hists = run_adaptation(
            preop_pa, postop_pa, CWSSIMSAdaptation, K_arr
        )

        df = pd.DataFrame([result])
        # for i in range(4):
        #     df[f'K_{i}'] = K_arr[i]

        # print(f"saving result with adapted split {postop_pa.rpa_split}")

        # append_result_to_csv(df, output_csv_path)
        return (K_arr, df)

    except Exception as e:
        print(f"Worker failed for K_arr {K_arr}: {e}")
        raise


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
    combinations_csv_path: str = 'K_arr_combinations.csv',
    output_csv_path: str = 'K_arr_results.csv'
) -> pd.DataFrame:

    df_manifest = pd.DataFrame(scaled_K_arrs, columns=['K_0', 'K_1', 'K_2', 'K_3'])
    df_manifest.to_csv(combinations_csv_path, index=False)
    print(f"Saved tested K_arr vectors to {combinations_csv_path}")

    args_list = [
        (K_arr, preop_config_path, postop_config_path, optimized_tree_params_csv, clinical_targets_csv, output_csv_path)
        for K_arr in scaled_K_arrs
    ]

    all_results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_Karr_worker, args): args[0] for args in args_list}

        for future in as_completed(futures):
            K_arr = futures[future]
            try:
                K_values, df = future.result()
                all_results.append(df)  # Optional: still return combined DataFrame
            except Exception as e:
                print(f"K_arr {K_arr} failed with error: {e}")

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()


def run_single_tree_cwss_debug(
    tree,
    *,
    q_homeostatic: float,
    q_target: float,
    wss_gain: float = 0.01,
    t_end: float = 3600.0,
    rtol: float = 1e-6,
    atol: float = 1e-7,
    max_step: float = 60.0,
    method: str = "RK23",
) -> dict:
    """
    Debug a single structured tree with the stabilized WSS-only ODE.

    This helper is intentionally standalone so gain tuning and convergence
    inspection can happen without the full bilateral PA workflow.
    """
    debug_tree = copy.deepcopy(tree)
    debug_tree.compute_homeostatic_state(q_homeostatic)

    y0 = pack_state(debug_tree)
    y_initial = y0.copy()
    last_update_y = y0.copy()
    event_reference_y = y0.copy()
    last_t_holder = [-float("inf")]
    residual_log: list[tuple[float, float]] = []

    def _rhs(t, y):
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 1:
            raise ValueError(f"State vector must be 1-D, received shape {y.shape}.")
        if not np.all(y > 0.0):
            raise ValueError("Single-tree CWSS debug requires a strictly positive state.")

        unpack_state(y, debug_tree)
        pd_val = float(debug_tree.Pd) if getattr(debug_tree, "Pd", None) is not None else 0.0
        hemo = estimate_steady_tree_hemodynamics(
            debug_tree,
            root_flow=float(q_target),
            distal_pressure=pd_val,
        )
        tau = hemo.wall_shear_stress
        tau_h = np.asarray(debug_tree.homeostatic_wss, dtype=np.float64)
        if t > last_t_holder[0] + 1e-12:
            last_update_y[:] = y
            last_t_holder[0] = t
            residual_log.append((float(t), float(np.mean(np.abs(tau - tau_h)))))

        dydt = np.zeros_like(y)
        dydt[0::2] = float(wss_gain) * (tau - tau_h) * y[0::2]
        return dydt

    def _event(t, y, triggered=[False], was_positive=[False]):
        rel_geom_r = np.mean(np.abs((y[0::2] - event_reference_y[0::2]) / event_reference_y[0::2]))
        converged = rel_geom_r < 1e-6

        if converged:
            triggered[0] = True

        if not triggered[0]:
            was_positive[0] = True
            return 1.0
        if was_positive[0]:
            was_positive[0] = False
            return -1.0
        was_positive[0] = True
        return 1.0

    _event.terminal = True
    _event.direction = -1

    sol = solve_ivp(
        _rhs,
        (0.0, float(t_end)),
        y0,
        events=_event,
        method=str(method),
        rtol=float(rtol),
        atol=float(atol),
        max_step=float(max_step),
    )

    y_final = sol.y[:, -1]
    unpack_state(y_final, debug_tree)
    geom_err = rel_change(y_final, y_initial)

    return {
        "tree": debug_tree,
        "solution": sol,
        "residual_log": residual_log,
        "metrics": {
            "geom_err": geom_err,
            "t95": time_to_95(sol),
            "stable": int(sol.status == 1),
            "n_rhs": sol.nfev,
            "wss_gain": float(wss_gain),
            "solver_t_end": float(t_end),
            "integration_method": str(method),
        },
    }


class _MiniPADebug:
    def __init__(self, lpa_tree, rpa_tree, *, q_total: float, lpa_upstream_R: float, rpa_upstream_R: float, wedge_p: float):
        self.lpa_tree = lpa_tree
        self.rpa_tree = rpa_tree
        self.q_total = float(q_total)
        self.lpa_upstream_R = float(lpa_upstream_R)
        self.rpa_upstream_R = float(rpa_upstream_R)
        self.clinical_targets = SimpleNamespace(wedge_p=float(wedge_p))
        self.result = None
        self.rpa_split = None

    def update_bcs(self):
        return None

    def simulate(self):
        lpa_R = self.lpa_upstream_R + float(self.lpa_tree.equivalent_resistance())
        rpa_R = self.rpa_upstream_R + float(self.rpa_tree.equivalent_resistance())
        lpa_G = 1.0 / max(lpa_R, 1e-12)
        rpa_G = 1.0 / max(rpa_R, 1e-12)
        total_G = lpa_G + rpa_G
        lpa_flow = self.q_total * lpa_G / total_G
        rpa_flow = self.q_total * rpa_G / total_G
        self.rpa_split = float(rpa_flow / max(lpa_flow + rpa_flow, 1e-12))
        self.result = pd.DataFrame(
            {
                "name": ["branch2_seg0", "branch4_seg0"],
                "flow_out": [lpa_flow, rpa_flow],
            }
        )
        return self.result


def run_minipa_cwss_debug(
    *,
    initial_d: float = 0.3,
    d_min: float = 0.1,
    alpha: float = 0.9,
    beta: float = 0.6,
    lrr: float = 10.0,
    Pd: float = 8.0 * 1333.2,
    q_total_preop: float = 2.0,
    q_total_postop: float = 2.0,
    preop_upstream_R: tuple[float, float] = (500.0, 500.0),
    postop_upstream_R: tuple[float, float] = (250.0, 750.0),
    wss_gain: float = 0.01,
    t_end: float = 200.0,
    rtol: float = 1e-6,
    atol: float = 1e-7,
    max_step: float = 1.0,
    method: str = "RK23",
) -> dict:
    """
    Run the full CWSS adaptation path on a tiny local bilateral PA surrogate
    built from real structured trees with a large terminal diameter.
    """

    def _build_tree(name: str):
        tree = StructuredTree(name=name, time=[0.0, 1.0], simparams=None, Pd=Pd)
        tree.build(initial_d=initial_d, d_min=d_min, alpha=alpha, beta=beta, lrr=lrr)
        return tree

    preop_lpa = _build_tree("debug_lpa")
    preop_rpa = _build_tree("debug_rpa")
    preop_pa = _MiniPADebug(
        preop_lpa,
        preop_rpa,
        q_total=q_total_preop,
        lpa_upstream_R=preop_upstream_R[0],
        rpa_upstream_R=preop_upstream_R[1],
        wedge_p=Pd / 1333.2,
    )
    preop_pa.simulate()
    lpa_q_homeostatic = float(preop_pa.result[preop_pa.result.name == "branch2_seg0"]["flow_out"].mean())
    rpa_q_homeostatic = float(preop_pa.result[preop_pa.result.name == "branch4_seg0"]["flow_out"].mean())
    preop_pa.lpa_tree.compute_homeostatic_state(lpa_q_homeostatic)
    preop_pa.rpa_tree.compute_homeostatic_state(rpa_q_homeostatic)

    postop_pa = _MiniPADebug(
        copy.deepcopy(preop_pa.lpa_tree),
        copy.deepcopy(preop_pa.rpa_tree),
        q_total=q_total_postop,
        lpa_upstream_R=postop_upstream_R[0],
        rpa_upstream_R=postop_upstream_R[1],
        wedge_p=Pd / 1333.2,
    )

    result, flow_log, sol, adapted_pa, hists = run_adaptation(
        preop_pa,
        postop_pa,
        CWSSAdaptation,
        [float(wss_gain), 0.0, 0.0, 0.0],
        t_end=float(t_end),
        rtol=float(rtol),
        atol=float(atol),
        max_step=float(max_step),
        method=str(method),
    )

    return {
        "result": result,
        "solution": sol,
        "flow_log": flow_log,
        "adapted_pa": adapted_pa,
        "figures": hists,
        "metrics": {
            **result,
            "solver_diagnostics": result["solver_diagnostics"],
        },
    }

import itertools
import pandas as pd
from typing import List, Tuple

def run_gains(
    scaled_K_arrs: List[List[float]],
    preop_config_path: str,
    postop_config_path: str,
    optimized_tree_params_csv: str,
    clinical_targets_csv: str,
    fig_dir: str = 'figures',
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
        result, flow_log, sol, postop_pa, hists = run_adaptation(preop_pa, postop_pa, CWSSIMSAdaptation, K_arr)

        if fig_dir is not None:
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)

            for i, hist in enumerate(hists):
                hist_path = os.path.join(fig_dir, f'histogram_{i}.png')
                hist.savefig(hist_path)
                print(f"Saved histogram to {hist_path}")
            
            # save sol.y to CSV
            sol_y_path = os.path.join(fig_dir, 'sol_y.csv')
            pd.DataFrame(sol.y).to_csv(sol_y_path, index=False)
            print(f"Saved sol.y to {sol_y_path}")

        results.append(result)

    # Convert to DataFrame for easy analysis / CSV
    results_df = pd.DataFrame(results)

    return results_df
