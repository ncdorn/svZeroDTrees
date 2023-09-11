import csv
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import pandas as pd
import copy
import svzerodplus

# utilities for working with structured trees


def get_pressure(result_array, branch):
    '''
    get the time series, systolic, diastolic and mean pressure for a given branch

    :param result_array: result array from svzerodplus
    :param branch: branch number

    :return pressures: time series of pressure
    :return systolic_p: systolic pressure
    :return diastolic_p: diastolic pressure value
    :return mean_p: time average pressure
    '''
    pressures = get_branch_result(result_array, 'pressure_in', branch, steady=False)
    systolic_p = np.min(pressures)
    diastolic_p = np.max(pressures)
    mean_p = np.mean(pressures)

    return pressures, systolic_p, diastolic_p, mean_p


def get_outlet_data(config: dict, result_array, data_name: str, steady=False):
    '''
    get a result at the outlets of a model

    :param config: svzerodplus config file
    :param result_array: svzerodplus result array
    :param data_name: data type to get out of the result array (q, p, wss)
    :param steady: True if the model has steady inflow

    :return data_out: list of lists of outlet data
    '''
    outlet_vessels, outlet_d = find_outlets(config)

    if 'wss' in data_name:
        data_out = []
        for i, branch in enumerate(outlet_vessels):
            q_out = get_branch_result(result_array, 'flow_out', branch, steady)
            if steady:
                data_out.append(q_out * 4 * config["simulation_parameters"]["viscosity"] / (np.pi * outlet_d[i]))
            else:
                data_out.append([q * 4 * config["simulation_parameters"]["viscosity"] / (np.pi * outlet_d[i]) for q in q_out])
                
    else:
        data_out = [get_branch_result(result_array, data_name, branch, steady) for branch in outlet_vessels]

    return data_out


def get_wss(config: dict, result_array, branch, steady=False):
    '''
    get the wss of a branch

    :param config: svzerodplus config dict
    :param result_array: svzerodplus result array
    :param branch: branch id
    :param steady: True if the model has steady inflow

    :return wss: wss array for the branch
    '''
    
    d = get_branch_d(config, branch)

    q_out = get_branch_result(result_array, 'flow_out', branch, steady)
    if steady:
        wss = q_out * 4 * config["simulation_parameters"]["viscosity"] / (np.pi * d)
    else:
        wss = [q * 4 * config["simulation_parameters"]["viscosity"] / (np.pi * d) for q in q_out]

    return wss


def get_branch_d(config, branch):
    '''
    get the diameter of a branch

    :param config: svzerodplus config dict
    :param branch: branch id

    :return d: branch diameter
    '''
    R = 0
    l = 0
    for vessel_config in config["vessels"]:
        if get_branch_id(vessel_config) == branch:
            # get total resistance of branch
            R += vessel_config["zero_d_element_values"].get("R_poiseuille")
            l += vessel_config["vessel_length"]
            break

    d = ((128 * config["simulation_parameters"]["viscosity"] * l) / (np.pi * R)) ** (1 / 4)

    return d


def find_outlets(config):
    '''
    find the outlet vessels in a model, return the vessel id and diameter

    :param config: svzerodplus config dict

    :return outlet_vessels: list of outlet vessel branch ids
    :return outlet_d: list of diameters corresponding to outlet_vessels
    '''
    outlet_vessels = []
    outlet_d = []
    for vessel_config in config["vessels"]:
        if "boundary_conditions" in vessel_config:
            if "outlet" in vessel_config["boundary_conditions"]:
                branch_id = get_branch_id(vessel_config)
                outlet_vessels.append(branch_id)
                # calculate the diameter
                d = ((128 * config["simulation_parameters"]["viscosity"] * vessel_config["vessel_length"]) /
                     (np.pi * vessel_config["zero_d_element_values"].get("R_poiseuille"))) ** (1 / 4)
                outlet_d.append(d)

    return outlet_vessels, outlet_d


def get_branch_result(result_array, data_name: str, branch: int, steady: bool=False):
    '''
    get the flow, pressure or wss result for a model branch

    :param result_array: svzerodplus result array
    :param data_name: q, p or wss
    :param branch: branch id to get result for
    :param steady: True if the model inflow is steady

    :return: result array for branch and QoI
    '''

    if steady:
        return result_array[data_name][branch][-1]
    else:
        return result_array[data_name][branch]


def get_resistances(config):
    '''
    get the outlet bc resistances from a svzerodplus config

    :param config: svzerodplus config dict

    :return resistance: list of outflow bc resistances
    '''
    resistance = []
    for bc_config in config["boundary_conditions"]:
        if bc_config["bc_type"] == 'RESISTANCE':
            resistance.append(bc_config['bc_values'].get('R'))
        if bc_config["bc_type"] == 'RCR':
            resistance.append(bc_config['bc_values'].get('Rp') + bc_config['bc_values'].get('Rd'))
    np.array(resistance)
    return resistance


def write_resistances(config, resistances):
    '''
    write a list of resistances to the outlet bcs of a config dict

    :param config: svzerodplus config dict
    :param resistances: list of resistances, ordered by outlet in the config
    '''
    idx = 0
    for bc_config in config["boundary_conditions"]:
        if bc_config["bc_type"] == 'RESISTANCE':
            bc_config['bc_values']['R'] = resistances[idx]
            idx += 1


def get_value_from_csv(csv_file, name):
    '''
    get a value from a csv file with a name in the same row

    :param csv_file: path to csv file
    :param name: name of the value in the same row as the int or float value

    ;return: value from csv

    '''
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if name.lower() in row[0].lower():
                return row[1]  # Return the value in the same row

    return None  # Return None if the name is not found


def get_resistance_idx(vessel_config):
    '''
    get the index of a resistance boundary condition

    :param vessel_config: config dict of the vessel (taken from the master config in a for loop)

    :return: integer index of the resistance boundary condition
    '''
    name = vessel_config["boundary_conditions"]["outlet"]
    str_idx = 10
    idx = name[str_idx:]
    while not idx.isdigit():
        str_idx += 1
        idx = name[str_idx:]

    return int(idx)


def make_inflow_steady(config, Q=97.3):
    '''
    convert unsteady inflow to steady

    :param config: input config dict
    :param Q: mean inflow value, default is 97.3
    '''
    for bc_config in config["boundary_conditions"]:
        if bc_config["bc_name"] == "INFLOW":
            bc_config["bc_values"]["Q"] = [Q, Q]
            bc_config["bc_values"]["t"] = [0.0, 1.0]


def convert_RCR_to_R(config, Pd=10 * 1333.22):
    '''
    Convert RCR boundary conditions to Resistance.

    :param config: input config dict
    :param Pd: distal pressure value for resistance bc. default value is 10 mmHg (converted to barye)

    :return: Pd and updated config
    '''
    for bc_config in config["boundary_conditions"]:
        if "RCR" in bc_config["bc_type"]:
            R = bc_config["bc_values"].get("Rp") + bc_config["bc_values"].get("Rd")
            bc_config["bc_type"] = "RESISTANCE"
            bc_config["bc_values"] = {"R": R, "Pd": Pd}

            return Pd

def add_Pd(config, Pd = 10 * 1333.22):
    '''
    add the distal pressure to the boundary conditions of a config file

    :param config: svzerodplus config dict
    :param Pd: distal pressure value [=] barye
    '''
    for bc_config in config["boundary_conditions"]:
        if "RESISTANCE" in bc_config["bc_type"]:
            bc_config["bc_values"]["Pd"] = Pd


def log_optimization_results(log_file, result, name: str=None):
    '''
    print optimization result to a log file

    :param log_file: path to log file
    :param result: optimizer result
    :param name: optimization name
    '''

    write_to_log(log_file, name + " optimization completed! \n")
    write_to_log(log_file, "     Optimization solution: " + str(result.x) + "\n")
    write_to_log(log_file, "     Objective function value: " + str(result.fun) + "\n")
    write_to_log(log_file, "     Number of iterations: " + str(result.nit) + "\n")


def plot_optimization_progress(fun, save=False, path=None):
    '''
    plot optimization progress by objective function value

    :param fun: list of objective function values to plot
    :param save: save the plot after optimization is complete
    :param path: path to figures directory to save the optimization plot
    '''
    plt.clf()
    plt.plot(range(len(fun)), fun, marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Objective Function Value')
    plt.title('Optimization Progress')
    plt.yscale('log')
    plt.pause(0.001)
    if save:
        plt.savefig(str(path) + '/optimization_result.png')


def assign_flow_to_root(result_array, root, steady=False):
    '''
    assign flow values to each TreeVessel instance in a StructuredTreOutlet tree

    :param result_array: svzerodplus result array of the structured tree
    :param root: root TreeVessel instance
    :param steady: True if the model has steady inflow
    '''
    def assign_flow(vessel):
        if vessel:
            # assign flow values to the vessel
            vessel.Q = get_branch_result(result_array, 'flow_in', vessel.id, steady=steady)
            vessel.P_in = get_branch_result(result_array, 'pressure_in', vessel.id, steady=steady)
            vessel.t_w = vessel.Q * 4 * vessel.eta / (np.pi * vessel.d)
            # recursive step
            assign_flow(vessel.left)
            assign_flow(vessel.right)
    
    assign_flow(root)


def run_svzerodplus(config):
    """Run the svzerodplus solver and return a dict of results.

    :param config: svzerodplus config dict

    :return output: the result of the simulation as a dict of dicts with each array denoted by its branch id
    """

    output = svzerodplus.simulate(config)
    result = pd.read_csv(StringIO(output))

    output = {
        "pressure_in": {},
        "pressure_out": {},
        "flow_in": {},
        "flow_out": {},
    }

    last_seg_id = 0

    for vessel in config["vessels"]:
        name = vessel["vessel_name"]
        branch_id, seg_id = name.split("_")
        branch_id, seg_id = int(branch_id[6:]), int(seg_id[3:])

        if seg_id == 0:
            output["pressure_in"][branch_id] = np.array(
                result[result.name == name]["pressure_in"]
            )
            output["flow_in"][branch_id] = np.array(
                result[result.name == name]["flow_in"]
            )
            output["pressure_out"][branch_id] = np.array(
                result[result.name == name]["pressure_out"]
            )
            output["flow_out"][branch_id] = np.array(
                result[result.name == name]["flow_out"]
            )
        elif seg_id > last_seg_id:
            output["pressure_out"][branch_id] = np.array(
                result[result.name == name]["pressure_out"]
            )
            output["flow_out"][branch_id] = np.array(
                result[result.name == name]["flow_out"]
            )

        last_seg_id = seg_id

    return output


def write_to_log(log_file, message: str, write=False):
    '''
    write a message to a log file

    :param log_file: path to log file
    :param message: message to print to the log file
    :param write: True if you would like to write to the log file (erasing previous log file data)
    '''
    if log_file is not None:
        if write:
            with open(log_file, "w") as log:
                log.write(message +  "\n")
        else:
            with open(log_file, "a") as log:
                log.write(message +  "\n")


def get_branch_id(vessel_config):
    '''
    get the integer id of a branch for a given vessel

    :param vessel_config: config dict of a vessel

    :return: integer branch id
    '''
    branch_id, seg_id = vessel_config["vessel_name"].split("_")

    return int(branch_id[6:])

def get_clinical_targets(clinical_targets: csv, log_file: str):
    '''
    get the clinical target values from a csv file

    :param clinical_targets: path to csv file with clinical targets
    :param log_file: path to log file
    '''
    write_to_log(log_file, "Getting clinical target values...")

    bsa = float(get_value_from_csv(clinical_targets, 'bsa'))
    cardiac_index = float(get_value_from_csv(clinical_targets, 'cardiac index'))
    q = bsa * cardiac_index * 16.667 # cardiac output in L/min. convert to cm3/s
    mpa_pressures = get_value_from_csv(clinical_targets, 'mpa pressures') # mmHg
    mpa_sys_p_target = int(mpa_pressures[0:2])
    mpa_dia_p_target = int(mpa_pressures[3:5])
    mpa_mean_p_target = int(get_value_from_csv(clinical_targets, 'mpa mean pressure'))
    target_ps = np.array([
        mpa_sys_p_target,
        mpa_dia_p_target,
        mpa_mean_p_target
    ])
    target_ps = target_ps * 1333.22 # convert to barye

    return q, cardiac_index, mpa_pressures, target_ps


def config_flow(preop_config, q):
    for bc_config in preop_config["boundary_conditions"]:
        if bc_config["bc_name"] == "INFLOW":
            bc_config["bc_values"]["Q"] = [q, q]
            bc_config["bc_values"]["t"] = [0.0, 1.0]


def create_pa_optimizer_config(preop_config, log_file, clinical_targets):
    '''
    create a config dict for the pa optimizer
    
    :param preop_config: preoperative config dict
    :param log_file: path to log file
    :param clinical_targets: path to csv file with clinical targets
    
    :return pa_config: config dict for the pa optimizer
    '''
    pa_config = copy.deepcopy(preop_config)

    

    pass