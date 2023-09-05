from svzerodtrees.utils import *
import copy

def repair_stenosis_coefficient(preop_config: dict, repair_config=None, log_file=None):
    '''
    :param preop_config: preop config dict
    :repair_config: repair config dict, with keys 'location' and 'repair degrees'. If location is a list, adjust those vessels
    '''

    postop_config = copy.deepcopy(preop_config)  # deepcopy in order to not mess with the preop config

    repair_vessels = []
    repair_degree = []
    if repair_config['location'] == 'proximal': # proximal stenosis repair case
        # repair only the LPA and RPA (outlets of the first junction in the config)
        repair_config['vessels'] = [1, 2]
        if len(repair_config['degree']) != 2: # if an improper numeber of repair degrees are specified
            raise "repair config must specify 2 degrees for LPA and RPA"
        write_to_log(log_file, "** repairing stenoses in vessels " + str(repair_vessels) + " **")

    elif repair_config['location'] == 'extensive': # extensive repair case (all vessels)
        repair_config['vessels'] = [get_branch_id(vessel) for vessel in postop_config["vessels"]] # NEED TO REMOVE DUPLICATES
        repair_config['degree'] *= len(repair_config['vessels']) # match the length of the repair degrees to the number of vessels
        write_to_log(log_file, "** repairing all stenoses **")

    elif type(repair_config['location']) is list: # custom repair case
        repair_config['vessels'] = repair_config['location']
        repair_config['degree'] = repair_config['degree']
        write_to_log(log_file, "** repairing stenoses in vessels " + str(repair_vessels) + " **")

    # perform the virtual stenosis repair
    for vessel_config in postop_config["vessels"]:
        if get_branch_id(vessel_config) in repair_config['vessels']:
            repair_idx = repair_config['vessels'].index(get_branch_id(vessel_config)) # this will count up too high, need to adjust this
            vessel_config["zero_d_element_values"]["stenosis_coefficient"] = vessel_config["zero_d_element_values"].get("stenosis_coefficient") * (1-repair_config['degree'][repair_idx])
            if repair_config['location'] != 'extensive': # if we write to the log file for every vessel in the extensive case we could clog the log file
                write_to_log(log_file, "     vessel " + str(vessel_config["vessel_id"]) + " has been repaired")
    write_to_log(log_file, 'all stenosis repairs completed')

    postop_result = run_svzerodplus(postop_config)

    return postop_result, postop_config


def repair_stenosis_resistance(preop_config: dict, repair_config=None, log_file=None):
    # adjust the resistance of the stenosed vessel, as opposed to the stenosis coefficient
    # to be implemented
    pass