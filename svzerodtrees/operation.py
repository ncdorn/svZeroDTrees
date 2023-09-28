from svzerodtrees.utils import *
from svzerodtrees.results_handler import ResultHandler

import copy

def repair_stenosis_coefficient(preop_config: dict, result_handler: ResultHandler, repair_config=None, log_file=None):
    '''
    perform a virtual stenosis repair in 0d by adjusting the stenosis coefficient according to the repair config

    :param preop_config: preop config dict
    :param repair_config: repair config dict, with keys 'location' and 'repair degrees'. If location is a list, adjust those vessels

    :return postop_config: config dict with post-operative changes
    :return postop_result: postop flow result
    '''

    # just point to the preop config
    postop_config = preop_config

    write_to_log(log_file, 'making repair according to repair config: ' + str(repair_config))

    # initialize list of vessels to repair
    repair_vessels = []

    # proximal stenosis repair case
    if repair_config['location'] == 'proximal': 
        # repair only the LPA and RPA (should be the outlets of the first junction in the config file)
        repair_config['vessels'] = find_rpa_lpa_branches(preop_config)

        # if an improper number of repair degrees are specified
        if len(repair_config['degree']) != 2: 
            raise Exception("repair config must specify 2 degrees for LPA and RPA")
        
        write_to_log(log_file, "** repairing stenoses in vessels " + str(repair_vessels) + " **")

    # extensive repair case (all vessels)
    elif repair_config['location'] == 'extensive': 

        # get list of vessels with no duplicates
        repair_config['vessels'] = list(set([get_branch_id(vessel) for vessel in postop_config["vessels"]])) 

        # match the length of the repair degrees to the number of vessels
        repair_config['degree'] *= len(repair_config['vessels']) 

        write_to_log(log_file, "** repairing all stenoses **")

    # custom repair case
    elif type(repair_config['location']) is list: 

        # repair vessels specified in the repair config
        repair_config['vessels'] = repair_config['location']
        
        # repair degree specified in the repair config
        repair_config['degree'] = repair_config['degree']

        write_to_log(log_file, "** repairing stenoses in vessels " + str(repair_vessels) + " **")

    # perform the virtual stenosis repair
    for vessel_config in postop_config["vessels"]:

        if get_branch_id(vessel_config) in repair_config['vessels']:

            # get the repair index to match with the appropriate repair degree
            repair_idx = repair_config['vessels'].index(get_branch_id(vessel_config))

            # adjust the stenosis coefficient
            vessel_config["zero_d_element_values"]["stenosis_coefficient"] = vessel_config["zero_d_element_values"].get("stenosis_coefficient") * (1-repair_config['degree'][repair_idx])
            
            # if we write to the log file for every vessel in the extensive case we could clog the log file
            if repair_config['location'] != 'extensive': 
                write_to_log(log_file, "     vessel " + str(vessel_config["vessel_id"]) + " has been repaired")
                
    write_to_log(log_file, 'all stenosis repairs completed')

    postop_result = run_svzerodplus(postop_config)

    result_handler.add_unformatted_result(postop_result, 'postop')

    return postop_config, result_handler


def repair_stenosis_resistance(preop_config: dict, repair_config=None, log_file=None):
    # adjust the resistance of the stenosed vessel, as opposed to the stenosis coefficient
    # to be implemented
    pass