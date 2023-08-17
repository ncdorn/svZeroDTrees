import csv
import numpy as np
import matplotlib.pyplot as plt
import math
from svzerodsolver import runner

# utilities for working with structured trees


def get_mpa_pressure(result_df, branch_name):
    # get the mpa pressure array for a given branch name from a zerod solver output file
    pressures = get_df_data(result_df, 'pressure_in', branch_name)
    systolic_p = np.min(pressures)
    diastolic_p = np.max(pressures)
    mean_p = np.mean(pressures)
    return pressures, systolic_p, diastolic_p, mean_p


def get_outlet_data(config, result_df, data_name, steady=False):
    # get data at the outlets of a model
    outlet_vessels, outlet_d = find_outlets(config)

    if 'wss' in data_name:
        data_out = []
        for i, branch_name in enumerate(outlet_vessels):
            q_out = get_df_data(result_df, 'flow_out', branch_name, steady=steady)
            if steady:
                data_out.append(q_out * 4 * config["simulation_parameters"]["viscosity"] / (np.pi * outlet_d[i]))
            else:
                data_out.append([q * 4 * config["simulation_parameters"]["viscosity"] / (np.pi * outlet_d[i]) for q in q_out])
                
    else:
        data_out = [get_df_data(result_df, data_name, branch_name, steady=steady) for branch_name in outlet_vessels]

    return data_out

def find_outlets(config):
    # find the outlet vessels in a model, return the vessel id and diameter
    outlet_vessels = []
    outlet_d = []
    for vessel_config in config["vessels"]:
        if "boundary_conditions" in vessel_config:
            if "outlet" in vessel_config["boundary_conditions"]:
                outlet_vessels.append('V' + str(vessel_config["vessel_id"]))
                d = ((128 * config["simulation_parameters"]["viscosity"] * vessel_config["vessel_length"]) /
                     (np.pi * vessel_config["zero_d_element_values"].get("R_poiseuille"))) ** (1 / 4)
                outlet_d.append(d)
    return outlet_vessels, outlet_d


def get_df_data(result_df, data_name, branch_name, steady=False):
    # get data from a dataframe based on a data name and branch name
    data = []
    for idx, name in enumerate(result_df['name']):
        if str(branch_name) == name:
            data.append(result_df.loc[idx, data_name])
    if steady:
        return data[-1]
    else:
        return data



def get_resistances(config):
    # get the resistances from an input config file
    # return a list of resistances
    resistance = []
    for bc_config in config["boundary_conditions"]:
        if bc_config["bc_type"] == 'RESISTANCE':
            resistance.append(bc_config['bc_values'].get('R'))
    np.array(resistance)
    return resistance


def write_resistances(config, resistances):
    # write resistances to the config file
    idx = 0
    for bc_config in config["boundary_conditions"]:
        if bc_config["bc_type"] == 'RESISTANCE':
            bc_config['bc_values']['R'] = resistances[idx]
            idx += 1


def get_value_from_csv(csv_file, name):
    '''
    get a value from a csv file with a name in the same row
    Args:
        csv_file: path to csv file
        name: name of the value in the same row as the int or float value

    Returns:

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
    Args:
        vessel_config: config dict of the vessel (taken from the master config in a for loop)

    Returns:
        integer index of the resistance boundary condition
    '''
    name = vessel_config["boundary_conditions"]["outlet"]
    str_idx = 10
    idx = name[str_idx:]
    while not idx.isdigit():
        str_idx += 1
        idx = name[str_idx:]

    return int(idx)


def repair_stenosis(config, vessels: str or list = None, degree: float = 1.0, log_file=None):
    '''
    Repair the stenoses in a pulmonary arterial tree
    Args:
        config: dict. solver input dictionary
        vessels: str or list of ints. stenosis repair strategy. 'proximal' repairs just the LPA and RPA. 'extensive'
        repairs all stenosis coefficients. an array of integers will repair those vessels
        degree: float. the degree to which the stenosis is repaired
        log_file: path to log file for output observation and debugging

    Returns:
        edited config file and log file
    '''
    if vessels is None:
        with open(log_file, "a") as log:
            log.write("** repairing all stenoses ** \n")
        for vessel_config in config["vessels"]:
            if vessel_config["zero_d_element_values"]["stenosis_coefficient"] > 0:
                vessel_config["zero_d_element_values"]["stenosis_coefficient"] = vessel_config["zero_d_element_values"].get("stenosis_coefficient") * (1-degree)
                with open(log_file, "a") as log:
                    log.write("     vessel " + str(vessel_config["vessel_id"]) + " has been repaired \n")
    else:
        with open(log_file, "a") as log:
            log.write("** repairing stenoses in vessels " + str(vessels) + " ** \n")
        for vessel_config in config["vessels"]:
            if vessel_config["vessel_id"] in vessels:
                vessel_config["zero_d_element_values"]["stenosis_coefficient"] = vessel_config["zero_d_element_values"].get("stenosis_coefficient") * (1-degree)
                with open(log_file, "a") as log:
                    log.write("     vessel " + str(vessel_config["vessel_id"]) + " has been repaired \n")


def make_inflow_steady(config, Q=97.3):
    '''
    convert unsteady inflow to steady
    Args:
        config: input config dict
        Q: mean inflow value, default is 97.3
    Returns:
        updated bc_config
    '''
    for bc_config in config["boundary_conditions"]:
        if bc_config["bc_name"] == "INFLOW":
            bc_config["bc_values"]["Q"] = [Q, Q]
            bc_config["bc_values"]["t"] = [0.0, 1.0]


def convert_RCR_to_R(config, Pd=10 * 1333.22):
    '''
    Convert RCR boundary conditions to Resistance.
    Args:
        config: input config dict
        Pd: distal pressure value for resistance bc. default value is 10 mmHg (converted to barye)
    Returns:
        Pd and updated config
    '''
    for bc_config in config["boundary_conditions"]:
        if "RCR" in bc_config["bc_type"]:
            R = bc_config["bc_values"].get("Rp")
            bc_config["bc_type"] = "RESISTANCE"
            bc_config["bc_values"] = {"R": R, "Pd": Pd}

            return Pd

def add_Pd(config, Pd = 10 * 1333.22):
    # add the distal pressure to the boundary conditions of a config file
    for bc_config in config["boundary_conditions"]:
        if "RESISTANCE" in bc_config["bc_type"]:
            bc_config["bc_values"]["Pd"] = Pd

# this is some random code that may need to be used in the non-steady state case
def log_optimization_results(log_file, result, name: str=None):

    with open(log_file, "a") as log:
        log.write(name + "optimization completed! \n")
        log.write("     Optimization solution: " + str(result.x) + "\n")
        log.write("     Objective function value: " + str(result.fun) + "\n")
        log.write("     Number of iterations: " + str(result.nit) + "\n")


def plot_optimization_progress(fun, save=False, path=None):
    plt.clf()
    plt.plot(range(len(fun)), fun, marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Objective Function Value')
    plt.title('Optimization Progress')
    plt.yscale('log')
    plt.pause(0.001)
    if save:
        plt.savefig(str(path) + '/optimization_result.png')

def calculate_tree_flow(simparams: dict, tree_config: dict, flow_out: float, P_d: float):
    # create a config file for the tree and calculate flow through it
    if len(flow_out) == 1:
        flow_out = [flow_out[0],] * 2
    tree_calc_config = {
        "boundary_conditions": [
            {
                "bc_name": "INFLOW",
                "bc_type": "FLOW",
                "bc_values": {
                    "Q": flow_out,
                    "t": np.linspace(0.0, 1.0, num=len(flow_out)).tolist()
                }
            },
            {
                "bc_name": "P_d",
                "bc_type": "PRESSURE",
                "bc_values": {
                    "P": [P_d,] * 2,
                    "t": [
                        0.0,
                        1.0
                    ]
                }
            },
        ],
        "junctions": tree_config["junctions"],
        "simulation_parameters": simparams,
        "vessels": tree_config["vessels"]
    }

    tree_result = runner.run_from_config(tree_calc_config)

    return tree_result

def assign_flow_to_root(result, root, steady=False):
    # assign flow values to the TreeVessel recursion object
    def assign_flow(vessel):
        if vessel:
            # assign flow values to the vessel
            vessel.Q = get_df_data(result, 'flow_out', 'V' + str(vessel.id), steady=steady)
            vessel.P_in = get_df_data(result, 'pressure_in', 'V' + str(vessel.id), steady=steady)
            vessel.t_w = vessel.Q * 4 * vessel.eta / (np.pi * vessel.d)
            # recursive step
            assign_flow(vessel.left)
            assign_flow(vessel.right)
    
    assign_flow(root)


def optimize_pries_secomb(ps_params, trees, q_outs, dt=0.01, P_d=1333.2, steady=True):
    
    # initialize and calculate the Pries and Secomb parameters in the TreeVessel objects via a postorder traversal
    dD_list = [] # initialize the list of dDs for the outlet calculation
    for i, tree in enumerate(trees):
        SS_dD = 0.0 # sum of squared dDs initial guess
        converged = False
        threshold = 10 ** -5
        while not converged:
            tree_solver_config = tree.tree_solver_input([q_outs[i]], P_d)
            tree_result = runner.run_from_config(tree_solver_config)
            assign_flow_to_root(tree_result, tree.root, steady=steady)

            next_SS_dD = 0.0 # initializing sum of squared dDs, value to minimize
            def stimulate(vessel):
                if vessel:
                    stimulate(vessel.left)
                    stimulate(vessel.right)
                    vessel_dD = vessel.adapt_pries_secomb(ps_params, dt)
                    nonlocal next_SS_dD
                    next_SS_dD += vessel_dD ** 2
            stimulate(tree.root)

            dD_diff = abs(next_SS_dD ** 2 - SS_dD ** 2)
            if dD_diff < threshold:
                converged = True
            
            SS_dD = next_SS_dD
            print(dD_diff)
        print('Pries and Secomb integration completed!')
        dD_list.append(next_SS_dD)

    SSE = sum(dD ** 2 for dD in dD_list)

    return SSE




'''
def scale_inflow(config, Q_target):
for bc_config in config['boundary_conditions']:
        if bc_config["bc_name"] == 'inflow':

    Q_avg = np.mean(inflow_wave)
    scale_ratio = Q_target / Q_avg
    scaled_inflow = scale_ratio * inflow_wave
    return scaled_inflow
'''