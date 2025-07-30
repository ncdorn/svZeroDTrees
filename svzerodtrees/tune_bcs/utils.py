import matplotlib.pyplot as plt
import numpy as np
from ..utils import write_to_log
from ..io.utils import get_branch_result
from ..io.blocks import BoundaryCondition


'''
utils for tuning bcs
'''

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


def generate_outlet_rcr(resistance, capacitance, wedge_p):

    return BoundaryCondition.from_config({
                    "bc_name": "RPA_BC",
                    "bc_type": "RCR",
                    "bc_values": {
                        "Rp": resistance* 0.1,
                        "C": capacitance,
                        "Rd": resistance * 0.9,
                        "Pd": wedge_p
                    }
                })