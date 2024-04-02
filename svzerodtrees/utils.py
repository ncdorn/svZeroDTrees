import csv
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import pandas as pd
import copy
import pysvzerod

# utilities for working with zero D trees

def get_pressure(result_array, branch, convert_to_mmHg=False):
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

    if convert_to_mmHg:
        pressures = np.array(pressures) / 1333.22

    systolic_p = np.min(pressures)
    diastolic_p = np.max(pressures)
    mean_p = np.mean(pressures)

    return pressures, systolic_p, diastolic_p, mean_p

def plot_result(result_df, quantity, filepath):
    '''
    plot the result from a result dataframe

    :param result_df: result dataframe
    :param quantity: quantity to plot
    :param filepath: path to save the plot
    '''

    plt.clf()
    plt.plot(result_df['time'], result_df[quantity])
    plt.xlabel('time')
    plt.ylabel(quantity)
    plt.title(quantity)
    plt.pause(0.001)
    plt.savefig(filepath)


def plot_pressure(result_array, branch, save=False, fig_dir=None):
    '''
    plot the pressure time series for a given branch

    :param result_array: result array from svzerodplus
    :param branch: branch number
    :param save: save the plot after optimization is complete
    :param fig_dir: path to figures directory to save the optimization plot
    '''
    pressures, systolic_p, diastolic_p, mean_p = get_pressure(result_array, branch)

    plt.clf()
    plt.plot(range(len(pressures)), pressures, marker='o')
    plt.xlabel('Time')
    plt.ylabel('Pressure')
    plt.title('Pressure Time Series')
    plt.pause(0.001)
    if save:
        plt.savefig(str(fig_dir) + '/pressure_branch_' + str(branch) + '.png')
    else:
        plt.show()


def get_outlet_data(config: dict, result_array, data_name: str, steady=True):
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


def get_wss(vessels, viscosity, result_array, branch, steady=False):
    '''
    get the wss of a branch

    :param vessel: vessel config dict
    :param result_array: svzerodplus result array from result handler
    :param branch: branch id
    :param steady: True if the model has steady inflow

    :return wss: wss array for the branch
    '''
    
    d = get_branch_d(vessels, viscosity, branch)

    r = d / 2

    q_out = get_branch_result(result_array, 'flow_out', branch, steady)
    if steady:
        wss = q_out * 4 * viscosity / (np.pi * r ** 3)
    else:
        wss = [q * 4 * viscosity / (np.pi * r ** 3) for q in q_out]

    return wss

def get_branch_d(config, viscosity, branch):
    '''
    this is the worst method ever made, I'm sorry to anyone that is reading this. Will update soon.
    get the diameter of a branch in units of cm

    :param config: svzerodplus config dict
    :param branch: branch id

    :return d: branch diameter
    '''
    R = 0
    l = 0
    for vessel_config in config["vessels"]:
        if get_branch_id(vessel_config)[0] == branch:
            # get total resistance of branch if it is split into multiple segments
            R += vessel_config["zero_d_element_values"].get("R_poiseuille")
            l += vessel_config["vessel_length"]
            break

    d = ((128 * viscosity * l) / (np.pi * R)) ** (1 / 4)

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
                branch_id = get_branch_id(vessel_config)[0]
                outlet_vessels.append(branch_id)
                # calculate the diameter
                d = ((128 * config["simulation_parameters"]["viscosity"] * vessel_config["vessel_length"]) /
                     (np.pi * vessel_config["zero_d_element_values"].get("R_poiseuille"))) ** (1 / 4)
                outlet_d.append(d)

    return outlet_vessels, outlet_d


def get_branch_result(result_array, data_name: str, branch: int, steady: bool=False):
    '''
    get the flow, pressure or wss result for a model branch form an unformatted result

    :param result_array: svzerodplus result array
    :param data_name: q, p or wss
    :param branch: branch id to get result for
    :param steady: True if the model inflow is steady or youw want to get the average value

    :return: result array for branch and QoI
    '''

    if steady:
        return np.mean(result_array[data_name][branch])
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


def get_rcrs(config, one_to_nine=False):
    '''
    get the outlet rcr bc values from a svzerodplus config

    :param config: svzerodplus config dict

    :return rcrs: list of outflow bc rcr values as a flattened array [Rp, C, Rd]
    '''
    rcrs = []
    for bc_config in config["boundary_conditions"]:
        if bc_config["bc_type"] == 'RCR':
            rcrs.append([bc_config['bc_values'].get('Rp'), bc_config['bc_values'].get('C'), bc_config['bc_values'].get('Rd')])
    if one_to_nine:
        for rcr in rcrs:
            rcr[2] = rcr[0] * 9
    rcr = np.array(rcrs).flatten()

    return rcr


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


def write_rcrs(config, rcrs):
    '''
    write a list of rcrs to the outlet bcs of a config dict
    
    :param config: svzerodplus config dict
    :param rcrs: list of rcrs, ordered by outlet in the config
    '''
    idx = 0
    for bc_config in config["boundary_conditions"]:
        if bc_config["bc_type"] == 'RCR':
            # proximal resistance
            bc_config['bc_values']['Rp'] = rcrs[3 * idx]
            # capacitance
            bc_config['bc_values']['C'] = rcrs[3 * idx + 1]
            # distal resistance
            bc_config['bc_values']['Rd'] = rcrs[3 * idx + 2]
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
            if name.lower() in row[0].lower() and name.lower()[0] == row[0].lower()[0]:
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


def run_svzerodplus(config: dict, dtype='ndarray'):
    """Run the svzerodplus solver and return a dict of results.

    :param config: svzerodplus config dict
    :param dtype: data type of the result arrays, either dict or ndarray. default is ndarray.

    :return output: the result of the simulation as a dict of dicts with each array denoted by its branch id
    """

    result = pysvzerod.simulate(config)

    output = {
        "pressure_in": {},
        "pressure_out": {},
        "flow_in": {},
        "flow_out": {},
    }

    last_seg_id = 0

    for vessel in config["vessels"]:
        name = vessel["vessel_name"]
        branch_id, seg_id = get_branch_id(vessel)

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

    if dtype == 'dict':
        for field in output.keys():
            for branch in output[field].keys():
                output[field][branch] = output[field][branch].tolist()

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

    br, seg = vessel_config["vessel_name"].split("_")
    br = int(br[6:])
    seg = int(seg[3:])

    return br, seg

def get_clinical_targets(clinical_targets: csv, log_file: str):
    '''
    get the clinical target values from a csv file

    :param clinical_targets: path to csv file with clinical targets
    :param log_file: path to log file

    :return q: cardiac output [cm3/s]
    :return mpa_ps: mpa systolic, diastolic, mean pressures [mmHg]
    :return rpa_ps: rpa systolic, diastolic, mean pressures [mmHg]
    :return lpa_ps: lpa systolic, diastolic, mean pressures [mmHg]
    :return wedge_p: wedge pressure [mmHg]
    '''
    write_to_log(log_file, "Getting clinical target values...")

    bsa = float(get_value_from_csv(clinical_targets, 'bsa'))
    cardiac_index = float(get_value_from_csv(clinical_targets, 'cardiac index'))
    q = bsa * cardiac_index * 16.667 # cardiac output in L/min. convert to cm3/s
    # get important mpa pressures
    mpa_pressures = get_value_from_csv(clinical_targets, 'mpa pressures') # mmHg
    mpa_sys_p, mpa_dia_p = mpa_pressures.split("/")
    mpa_sys_p = int(mpa_sys_p)
    mpa_dia_p = int(mpa_dia_p)
    mpa_mean_p = int(get_value_from_csv(clinical_targets, 'mpa mean pressure'))
    mpa_ps = np.array([
        mpa_sys_p,
        mpa_dia_p,
        mpa_mean_p
    ])

    # get important rpa pressures
    rpa_pressures = get_value_from_csv(clinical_targets, 'rpa pressures') # mmHg
    rpa_sys_p, rpa_dia_p = rpa_pressures.split("/")
    rpa_sys_p = int(rpa_sys_p)
    rpa_dia_p = int(rpa_dia_p)
    rpa_mean_p = int(get_value_from_csv(clinical_targets, 'rpa mean pressure'))
    rpa_ps = np.array([
        rpa_sys_p,
        rpa_dia_p,
        rpa_mean_p
    ])

    # get important lpa pressures
    lpa_pressures = get_value_from_csv(clinical_targets, 'lpa pressures') # mmHg
    lpa_sys_p, lpa_dia_p = lpa_pressures.split("/")
    lpa_sys_p = int(lpa_sys_p)
    lpa_dia_p = int(lpa_dia_p)
    lpa_mean_p = int(get_value_from_csv(clinical_targets, 'lpa mean pressure'))
    lpa_ps = np.array([
        lpa_sys_p,
        lpa_dia_p,
        lpa_mean_p
    ])

    # get wedge pressure
    wedge_p = int(get_value_from_csv(clinical_targets, 'wedge pressure'))

    # get RPA flow split
    rpa_split = float(get_value_from_csv(clinical_targets, 'pa flow split')[0:2]) / 100

    return q, mpa_ps, rpa_ps, lpa_ps, wedge_p, rpa_split


def config_flow(preop_config, q):
    for bc_config in preop_config["boundary_conditions"]:
        if bc_config["bc_name"] == "INFLOW":
            bc_config["bc_values"]["Q"] = [q, q]
            bc_config["bc_values"]["t"] = [0.0, 1.0]


def create_pa_optimizer_config(config_handler, q, wedge_p, log_file=None):
    '''
    create a config dict for the pa optimizer
    
    :param config_handler: config_handler
    :param q: cardiac output
    :param wedge_p: wedge pressure for the distal pressure bc
    :param log_file: path to log file
    
    :return pa_config: config dict for the pa optimizer
    '''

    write_to_log(log_file, "Creating PA optimizer config...")

    pa_config = {'boundary_conditions': [],
                 'simulation_parameters': [], 
                 'vessels': [],
                 'junctions': []}

    # copy the inflow boundary condition
    pa_config['boundary_conditions'].append(
        {
            "bc_name": "INFLOW",
            "bc_type": "FLOW",
            "bc_values": {
                "Q": [q, q],
                "t": [0.0, 1.0]
            }
        }
    )

    # set the outflow boundary conditions
    pa_config['boundary_conditions'].append(
        {
            "bc_name": "RPA_BC",
            "bc_type": "RESISTANCE",
            "bc_values": {
                "R": 300.0,
                "Pd": wedge_p * 1333.22
            }
        }
    )

    pa_config['boundary_conditions'].append(
        {
            "bc_name": "LPA_BC",
            "bc_type": "RESISTANCE",
            "bc_values": {
                "R": 300.0,
                "Pd": wedge_p * 1333.22
            }
        }
    )

    # copy the simulation parameters
    pa_config['simulation_parameters'] = config_handler.config['simulation_parameters']

    # create the MPA, RPA and LPA vessels
    pa_config['vessels'] = [
        # MPA
        config_handler.config['vessels'][0],
        
        { # RPA proximal
            "vessel_id": 1,
            "vessel_length": 10.0,
            "vessel_name": "branch1_seg0",
            "zero_d_element_type": "BloodVessel",
            "zero_d_element_values": {
                "C": 0.0,
                "L": 0.0,
                "R_poiseuille": config_handler.rpa.zero_d_element_values.get("R_poiseuille"), # R_RPA_proximal
                "stenosis_coefficient": 0.0
            }
        },
        {   # LPA proximal
            "vessel_id": 2,
            "vessel_length": 10.0,
            "vessel_name": "branch2_seg0",
            "zero_d_element_type": "BloodVessel",
            "zero_d_element_values": {
                "C": 0.0,
                "L": 0.0,
                "R_poiseuille": config_handler.lpa.zero_d_element_values.get("R_poiseuille"), # R_LPA_proximal
                "stenosis_coefficient": 0.0
            }
        },
        {      # RPA distal
            "boundary_conditions":{
                "outlet": "RPA_BC"
            },
            "vessel_id": 3,
            "vessel_length": 10.0,
            "vessel_name": "branch3_seg0",
            "zero_d_element_type": "BloodVessel",
            "zero_d_element_values": {
                "C": 0.0,
                "L": 0.0,
                "R_poiseuille": config_handler.rpa.R_eq - config_handler.rpa.zero_d_element_values.get("R_poiseuille"), # R_RPA_distal
                "stenosis_coefficient": 0.0
            }
        },
        {   # LPA distal
            "boundary_conditions":{
                "outlet": "LPA_BC"
            },
            "vessel_id": 4,
            "vessel_length": 10.0,
            "vessel_name": "branch4_seg0",
            "zero_d_element_type": "BloodVessel",
            "zero_d_element_values": {
                "C": 0.0,
                "L": 0.0,
                "R_poiseuille": config_handler.lpa.R_eq - config_handler.lpa.zero_d_element_values.get("R_poiseuille"), # R_LPA_distal
                "stenosis_coefficient": 0.0
            }
        }
    ]
    
    
    # create the junctions
    pa_config['junctions'] = [
        {
            "inlet_vessels": [
                0
            ],
            "junction_name": "J0",
            "junction_type": "internal_junction",
            "outlet_vessels": [
                1,
                2
            ]
        },
        {
            "inlet_vessels": [
                1
            ],
            "junction_name": "J1",
            "junction_type": "internal_junction",
            "outlet_vessels": [
                3
            ]
        },
        {
            "inlet_vessels": [
                2
            ],
            "junction_name": "J2",
            "junction_type": "internal_junction",
            "outlet_vessels": [
                4
            ]
        }]
    
    return pa_config


def create_pa_optimizer_config_NEW(config_handler, q, wedge_p, log_file=None):
    '''
    create the reduced pa config for the bc optimizer\
    '''

    write_to_log(log_file, "Creating PA optimizer config...")

    # initialize the config dict
    pa_config = {'boundary_conditions': [],
                 'simulation_parameters': [], 
                 'vessels': [],
                 'junctions': []}

    # copy the inflow boundary condition
    pa_config['boundary_conditions'].append(
        {
            "bc_name": "INFLOW",
            "bc_type": "FLOW",
            "bc_values": {
                "Q": [q, q],
                "t": [0.0, 1.0]
            }
        }
    )

    # set the outflow boundary conditions
    pa_config['boundary_conditions'].append(
        {
            "bc_name": "RPA_BC",
            "bc_type": "RESISTANCE",
            "bc_values": {
                "R": 300.0,
                "Pd": wedge_p * 1333.22
            }
        }
    )

    pa_config['boundary_conditions'].append(
        {
            "bc_name": "LPA_BC",
            "bc_type": "RESISTANCE",
            "bc_values": {
                "R": 300.0,
                "Pd": wedge_p * 1333.22
            }
        }
    )

    # copy the simulation parameters
    pa_config['simulation_parameters'] = config_handler.simparams.to_dict()

    # add in the MPA
    pa_config['vessels'].extend(config_handler.get_vessels('mpa', dtype='dict'))

    # add in the RPA
    pa_config['vessels'].extend(config_handler.get_vessels('rpa', dtype='dict'))


def loss_function_bound_penalty(value, target, lb=None, ub=None):
    '''
    loss function penalty for optimization with bounds
    
    :param value: observed value
    :param target: target value
    :param lb: optional lower bound
    :param ub: optional upper bound
    
    :return penalty: penalty value
    '''

    if lb is None:
        lb = target - 10 # default lower bound is 10 less than target
    if ub is None:
        ub = target + 10 # default upper bound is 10 more than target

    # Normalized by the tuning bounds
    g1 = (lb - value) / (ub - lb)
    g2 = (value - ub) / (ub - lb)

    if np.any(g1 >= 0) or np.any(g2 >= 0):
        return -np.inf
    else:
        return 0.1 * np.sum(np.log(-g1) + np.log(-g2))


def get_pa_config_resistances(pa_config):
    '''
    get the important resistance values from a reduced pa config dict

    :param pa_config: reduced pa config dict

    :return: list of resistances [R_RPA_proximal, R_LPA_proximal, R_RPA_distal, R_LPA_distal, R_RPA_BC, R_LPA_BC]
    '''

    # R_RPA_proximal = pa_config['vessels'][1]['zero_d_element_values']['R_poiseuille']
    # R_LPA_proximal = pa_config['vessels'][2]['zero_d_element_values']['R_poiseuille']
    # R_RPA_distal = pa_config['vessels'][3]['zero_d_element_values']['R_poiseuille']
    # R_LPA_distal = pa_config['vessels'][4]['zero_d_element_values']['R_poiseuille']
    R_RPA_BC = pa_config['boundary_conditions'][1]['bc_values']['R']
    R_LPA_BC = pa_config['boundary_conditions'][2]['bc_values']['R']

    return [R_RPA_BC, R_LPA_BC] # R_RPA_distal, R_LPA_distal,

def write_pa_config_resistances(pa_config, resistances):
    '''
    write the important resistance values to a reduced pa config dict

    :param pa_config: reduced pa config dict
    :param resistances: list of resistances [R_RPA_proximal, R_LPA_proximal, R_RPA_distal, R_LPA_distal, R_RPA_BC, R_LPA_BC]
    '''

    # pa_config['vessels'][1]['zero_d_element_values']['R_poiseuille'] = resistances[0]
    # pa_config['vessels'][2]['zero_d_element_values']['R_poiseuille'] = resistances[1]
    # pa_config['vessels'][3]['zero_d_element_values']['R_poiseuille'] = resistances[2]
    # pa_config['vessels'][4]['zero_d_element_values']['R_poiseuille'] = resistances[3]
    pa_config['boundary_conditions'][1]['bc_values']['R'] = resistances[0]
    pa_config['boundary_conditions'][2]['bc_values']['R'] = resistances[1]

def get_pa_optimization_values(result):
    '''
    get the fucntion values for the pa optimization from a result array

    :param result: result array from svzerodplus

    :return: list of targets [Q_rpa, P_mpa, P_rpa, P_lpa], pressures in mmHg
    '''

    # rpa flow, for flow split optimization
    Q_rpa = get_branch_result(result, 'flow_in', 1, steady=True)

    # mpa pressure
    P_mpa = get_branch_result(result, 'pressure_in', 0, steady=True) /  1333.2 

    # rpa pressure
    P_rpa = get_branch_result(result, 'pressure_out', 1, steady=True) / 1333.2

    # lpa pressure
    P_lpa = get_branch_result(result, 'pressure_out', 2, steady=True) / 1333.2

    return np.array([Q_rpa, P_mpa, P_rpa, P_lpa])

def calc_WU_m2(vessel, viscosity):
    '''
    calculate woods units by m2 for a given vessel config
    '''

    return np.sqrt(8 * viscosity * vessel["vessel_length"] * np.pi * vessel["zero_d_element_values"].get("R_poiseuille"))


def find_lpa_rpa_branches(config):
    '''
    find the LPA and RPA branches in a config dict. 
    We assume that this is the first junction in the config with 2 distinct outlet vessels.
    
    :param config: svzerodplus config dict
    
    :return rpa_lpa_branch: list of ints of RPA, LPA branch id
    '''
    
    junction_id = 0
    junction_found = False
    # search for the first junction with 2 outlet vessels (LPA and RPA)
    while not junction_found:
        if len(config["junctions"][junction_id]["outlet_vessels"]) == 2:
            lpa_rpa_id = config["junctions"][junction_id]["outlet_vessels"]
            junction_found = True
        
        elif len(config["junctions"][junction_id]["outlet_vessels"]) != 2:
            junction_id += 1

    # initialize the list of branch ids
    branches = []

    for i, id in enumerate(lpa_rpa_id):
        for vessel_config in config["vessels"]:
            if vessel_config["vessel_id"] == id:
                branches.append(get_branch_id(vessel_config)[0])
    
    lpa_branch = branches[0]
    rpa_branch = branches[1]

    return lpa_branch, rpa_branch

def rebuild_trees(config: dict):
        '''
        build a list of StructuredTreeOutlet instances from a config_w_trees

        :param config_w_trees: config dict with trees
        
        :return trees: list of StructuredTreeOutlet instances
        '''

        trees = []
        for vessel_config in config['vessels']:
            if 'tree' in vessel_config:
                
                pass


### UNIT CONVERSIONS ###
            
def m2d(mmHg):
    '''
    convert mmHg to dynes/cm2
    '''

    return mmHg * 1333.22

def d2m(dynes):
    '''
    convert dynes/cm2 to mmHg
    '''

    return dynes / 1333.22

def nlmin2cm3s(nlmin):
    '''
    convert nl/min to cm3/s
    '''

    return nlmin * 1.667e-8