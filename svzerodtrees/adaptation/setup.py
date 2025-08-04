import csv
import pandas as pd
import numpy as np
from ..microvasculature import TreeParameters
from ..io import ConfigHandler
from ..tune_bcs import PAConfig
from ..tune_bcs.clinical_targets import ClinicalTargets

'''
this file is used to set up objects for adaptation computation
'''

def load_optimized_params(path: csv) -> TreeParameters:
    """
    Load optimized parameters from a file.

    :param path: Path to the csv containing optimized parameters.
    :return: Dictionary of optimized parameters.
    """

    opt_params = pd.read_csv(path)

    lpa_params = TreeParameters.from_row("lpa", opt_params[opt_params["pa"] == "lpa"])
    rpa_params = TreeParameters.from_row("rpa", opt_params[opt_params["pa"] == "rpa"])

    return lpa_params, rpa_params


def create_preop_model(
        config_path: str,
        clinical_targets: ClinicalTargets,
        lpa_params: TreeParameters,
        rpa_params: TreeParameters,
) -> PAConfig:
    '''
    create the preop reduced pa config 
    '''

    preop_pa_config_handler = ConfigHandler.from_json(config_path, is_pulmonary=True)
    preop_pa = PAConfig.from_pa_config(preop_pa_config_handler, clinical_targets)
    preop_pa.create_steady_trees(lpa_params, rpa_params)
    print(f"number of vessels in lpa tree: {preop_pa.lpa_tree.count_vessels()}")
    print(f"number of vessels in rpa tree: {preop_pa.rpa_tree.count_vessels()}")

    return preop_pa


def create_postop_model(
        config_path: str,
        clinical_targets: ClinicalTargets
) -> PAConfig:

    postop_pa_config_handler = ConfigHandler.from_json(config_path, is_pulmonary=True)
    postop_pa = PAConfig.from_pa_config(postop_pa_config_handler, clinical_targets)

    return postop_pa


def simulate_homeostatic_state(preop_pa: PAConfig) -> None:
    """
    Simulate the homeostatic state of the preop pa model

    :param preop_pa: The preoperative PAConfig object.
    """

    preop_pa.simulate()
    print(f"preop rpa split: {preop_pa.rpa_split}")
    # get flow and simulate homeostatic state for trees
    preop_lpa_flow = np.mean(preop_pa.result[preop_pa.result.name=='branch2_seg0']['flow_out'])
    preop_rpa_flow = np.mean(preop_pa.result[preop_pa.result.name=='branch4_seg0']['flow_out'])

    preop_pa.lpa_tree.compute_homeostatic_state(preop_lpa_flow)
    preop_pa.rpa_tree.compute_homeostatic_state(preop_rpa_flow)
    print("homeostatic state computed for trees.")


def initialize_from_paths(
        preop_config_path: str,
        postop_config_path: str,
        optimized_tree_params_csv: str,
        clinical_targets_csv: str
) -> tuple[PAConfig, PAConfig]:
    """
    Initialize the preoperative and postoperative PAConfig objects from their respective paths.

    :param preop_config_path: Path to the preoperative configuration file.
    :param postop_config_path: Path to the postoperative configuration file.
    :param optimized_tree_params_csv: Path to the optimized tree parameters CSV file.
    :param clinical_targets_csv: Path to the clinical targets CSV file.
    :return: A tuple containing the preoperative and postoperative PAConfig objects.
    """

    clinical_targets = ClinicalTargets.from_csv(clinical_targets_csv)
    lpa_params, rpa_params = load_optimized_params(optimized_tree_params_csv)

    # RESET D_MIN FOR TESTING
    lpa_params.d_min = 0.05
    rpa_params.d_min = 0.05

    preop_pa = create_preop_model(preop_config_path, clinical_targets, lpa_params, rpa_params)
    simulate_homeostatic_state(preop_pa)

    postop_pa = create_postop_model(postop_config_path, clinical_targets)

    return preop_pa, postop_pa