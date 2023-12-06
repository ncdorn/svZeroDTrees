from svzerodtrees.utils import *
from svzerodtrees._result_handler import ResultHandler
from svzerodtrees._config_handler import ConfigHandler

import copy

def repair_stenosis_coefficient(config_handler: ConfigHandler, result_handler: ResultHandler, repair_config=None, log_file=None):
    '''
    perform a virtual stenosis repair in 0d by adjusting the stenosis coefficient according to the repair config

    :param preop_config: preop config dict
    :param repair_config: repair config dict, with keys 'location' and 'repair degrees'. If location is a list, adjust those vessels

    :return postop_config: config dict with post-operative changes
    :return postop_result: postop flow result
    '''

    write_to_log(log_file, 'making repair according to repair config: ' + str(repair_config))

    # initialize list of vessels to repair
    repair_vessels = []

    # proximal stenosis repair case
    if repair_config['location'] == 'proximal': 
        # repair only the LPA and RPA (should be the outlets of the first junction in the config file)
        repair_config['vessels'] = [result_handler.lpa_branch, result_handler.rpa_branch]

        # if an improper number of repair degrees are specified
        if len(repair_config['degree']) != 2: 
            raise Exception("repair config must specify 2 degrees for LPA and RPA")
        
        write_to_log(log_file, "** repairing stenoses in vessels " + str(repair_config['vessels']) + " **")

    # extensive repair case (all vessels)
    elif repair_config['location'] == 'extensive': 

        # get list of vessels with no duplicates
        repair_config['vessels'] = list(set([get_branch_id(vessel)[0] for vessel in config_handler.config["vessels"]])) 

        # match the length of the repair degrees to the number of vessels
        repair_config['degree'] *= len(repair_config['vessels']) 

        write_to_log(log_file, "** repairing all stenoses **")

    # custom repair case
    elif type(repair_config['location']) is list: 

        repair_config['vessels'] = []
        # add lpa and rpa branch if rpa and lpa are specified
        for location in repair_config['location']:
            if location == 'rpa':
                repair_config['vessels'].append(result_handler.rpa_branch)

            elif location == 'lpa':
                repair_config['vessels'].append(result_handler.lpa_branch)

            else:
                repair_config['vessels'].append(location)
        
        # repair degree specified in the repair config
        # repair_config['degree'] = repair_config['degree']

        write_to_log(log_file, "** repairing stenoses in vessels " + str(repair_config['vessels']) + " **")

    # perform the virtual stenosis repair
    for vessel_config in config_handler.config["vessels"]:


        if get_branch_id(vessel_config)[0] in repair_config['vessels']:

            # get the repair index to match with the appropriate repair degree
            repair_idx = repair_config['vessels'].index(get_branch_id(vessel_config)[0])

            # adjust the stenosis coefficient
            if repair_config['type'] == 'stenosis_coefficient':
                vessel_config["zero_d_element_values"]["stenosis_coefficient"] = vessel_config["zero_d_element_values"].get("stenosis_coefficient") * (1-repair_config['degree'][repair_idx])
                # if we write to the log file for every vessel in the extensive case we could clog the log file
                if repair_config['location'] != 'extensive': 
                    write_to_log(log_file, "     vessel " + str(vessel_config["vessel_id"]) + " has been repaired by degree " + str(repair_config['degree'][repair_idx]))
                    write_to_log(log_file, "     the new stenosis coefficient is " + str(vessel_config["zero_d_element_values"]["stenosis_coefficient"]))
            
            # adjust the vessel resistance by some scaLing factor
            if repair_config['type'] == 'resistance':
                vessel_config["zero_d_element_values"]["R_poiseuille"] = vessel_config["zero_d_element_values"].get("R_poiseuille") * (repair_config['degree'][repair_idx])
                if repair_config['location'] != 'extensive': 
                    write_to_log(log_file, "     vessel " + str(vessel_config["vessel_id"]) + " resistance has been scaled by " + str(repair_config['degree'][repair_idx]))
                    write_to_log(log_file, "     the new resistance is " + str(vessel_config["zero_d_element_values"]["R_poiseuille"]))
            
            # adjust the vessel resistance by stent diameter (in cm)
            if repair_config['type'] == 'stent':
                R_old = vessel_config["zero_d_element_values"].get("R_poiseuille") + vessel_config["zero_d_element_values"].get("stenosis_coefficient")
                vessel_config["zero_d_element_values"]["R_poiseuille"] = (8 * config_handler.config["simulation_parameters"]["viscosity"] * vessel_config["vessel_length"]) / (np.pi * (repair_config["diameter"][repair_idx] / 2) ** 4)
                vessel_config["zero_d_element_values"]["stenosis_coefficient"] = 0.0

                write_to_log(log_file, "     vessel " + str(vessel_config["vessel_id"]) + " has been repaired by " + str(repair_config["diameter"][repair_idx] * 10) + " mm stent")
                write_to_log(log_file, "     the change in resistance is " + str(R_old - vessel_config["zero_d_element_values"]["R_poiseuille"]))
        
    write_to_log(log_file, 'all stenosis repairs completed')

    postop_result = run_svzerodplus(config_handler.config)

    result_handler.add_unformatted_result(postop_result, 'postop')


def repair_stenosis_resistance(preop_config: dict, repair_config=None, log_file=None):
    '''
    repair the stenosis by manually reducing the resistance by some value

    :param preop_config: preop config dict
    :param repair_config: repair config dict, with keys 'location' and 'resistance'. If location is a list, adjust those vessels
    '''
    pass
    

    
    