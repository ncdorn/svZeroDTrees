import csv
import numpy as np
import matplotlib.pyplot as plt
from .io.utils import *
import pandas as pd
import copy
import pysvzerod

'''
global utils
'''

def run_svzerodplus(config: dict, dtype='ndarray'):
    """Run the svzerodplus solver and return a dict of results.

    :param config: svzerodplus config dict
    :param dtype: data type of the result arrays, either dict or ndarray. default is ndarray.

    :return output: the result of the simulation as a dict of dicts with each array denoted by its branch id
    """

    result = pysvzerod.simulate(config)

    output = {
        "time": result["time"],
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
