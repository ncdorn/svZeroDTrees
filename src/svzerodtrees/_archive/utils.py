from ..utils import get_branch_id
'''
archived utils functions
'''

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


