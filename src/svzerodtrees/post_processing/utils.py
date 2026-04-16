import numpy as np
from ..io.utils import get_branch_id

'''
postprocessing utils
'''

def calc_WU_m2(vessel, viscosity):
    '''
    calculate woods units by m2 for a given vessel config
    '''

    return np.sqrt(8 * viscosity * vessel["vessel_length"] * np.pi * vessel["zero_d_element_values"].get("R_poiseuille"))





def get_branch_d(vessels, branch, viscosity=0.04):
    '''
    this is the worst method ever made, I'm sorry to anyone that is reading this. Will update soon.
    get the diameter of a branch in units of cm

    :param config: svzerodplus config dict
    :param branch: branch id

    :return d: branch diameter
    '''
    R = 0
    l = 0
    for vessel_config in vessels:
        if get_branch_id(vessel_config)[0] == branch:
            # get total resistance of branch if it is split into multiple segments
            R += vessel_config["zero_d_element_values"].get("R_poiseuille")
            l += vessel_config["vessel_length"]

    d = ((128 * viscosity * l) / (np.pi * R)) ** (1 / 4)

    return d
