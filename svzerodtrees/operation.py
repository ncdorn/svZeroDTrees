from svzerodtrees.utils import *
from svzerodtrees._result_handler import ResultHandler
from svzerodtrees._config_handler import ConfigHandler

import copy

def repair_stenosis(config_handler: ConfigHandler, result_handler: ResultHandler, repair_config=None, log_file=None):
    '''
    repair the stenosis
    '''

    # proximal stenosis repair case
    if repair_config['location'] == 'proximal': 
        # repair only the LPA and RPA (should be the outlets of the first junction in the config file)
        repair_branches = ['lpa', 'rpa']

        # if an improper number of repair degrees are specified
        if len(repair_config['value']) != 2: 
            raise Exception("repair config must specify 2 degrees for LPA and RPA")
        
        write_to_log(log_file, "** repairing stenoses in vessels " + str(repair_branches) + " **")

    # extensive repair case (all vessels)
    elif repair_config['location'] == 'extensive': 

        # get list of vessels with no duplicates
        repair_branches = list(config_handler.branch_map.keys())

        # match the length of the repair degrees to the number of vessels
        repair_config['value'] *= len(repair_branches)

        write_to_log(log_file, "** repairing all stenoses **")

    # custom repair case
    elif type(repair_config['location']) is list: 
        repair_branches = repair_config['location']

    for branch, value in zip(repair_branches, repair_config['value']):
        branch_stenosis = Stenosis.create(config_handler, branch, repair_config['type'], value)
        branch_stenosis.repair()
    
    config_handler.simulate(result_handler, 'postop')


class Stenosis:
    '''
    a class to handle stenoses in 0D
    '''

    def __init__(self, vessels: list, branch: int, repair_type: str, repair_value: float, viscosity: float, log_file=None):
        '''
        :param vessel_config: the vessel config dict or a list if multiple segments
        :param repair_config: the repair config dict
        :param log_file: the log file to write to'''
        self.branch = branch
        self.ids = [vessel.id for vessel in vessels]
        self.repair_type = repair_type
        self.repair_value = repair_value
        self.log_file = log_file
        self.vessels = vessels
        self.viscosity = viscosity

    @classmethod
    def create(cls, config_handler: ConfigHandler, branch: int or str, repair_type, repair_value):
        '''
        create a stenosis from a config handler

        :param config_handler: the config handler
        :param branch: the branch id
        :param repair: the repair dict with type and value
        '''

        if branch == 'lpa':
            branch = config_handler.lpa.branch
        elif branch == 'rpa':
            branch = config_handler.rpa.branch

        # get the vessels in the branch
        vessels = config_handler.get_segments(branch)

        return cls(vessels, branch, repair_type, repair_value, config_handler.simparams.viscosity)

    
    def repair(self):
        '''repair the stenosis according to the specs'''
        
        if self.repair_type == 'stenosis_coefficient':
            self.sc_repair()
        elif self.repair_type == 'stent':
            self.stent_repair()
        elif self.repair_type == 'resistance':
            self.resistance_repair()
            
    
    def sc_repair(self):
        '''
        repair the stenosis by adjusting the stenosis coefficient
        '''

        # change the stenosis coefficient
        for vessel in self.vessels:
            vessel.zero_d_element_values['stenosis_coefficient'] = vessel.zero_d_element_values['stenosis_coefficient'] * (1 - self.repair_value)
    

    def stent_repair(self):
        '''
        repair the stenosis by changing the diameter according to stent diameter'''

        for vessel in self.vessels:
            # set stenosis coefficient to zero
            R_old = vessel.zero_d_element_values['R_poiseuille']
            vessel.zero_d_element_values['stenosis_coefficient'] = 0.0
            vessel.zero_d_element_values["R_poiseuille"] = (8 * self.viscosity * vessel.length) / (np.pi * (self.repair_value / 2) ** 4)
            R_change = R_old - vessel.zero_d_element_values['R_poiseuille']
    

    def resistance_repair(self):
        '''
        repair the stenosis by adjusting the resistance
        '''

        for vessel in self.vessels:
            vessel.zero_d_element_values['R_poiseuille'] = vessel.zero_d_element_values['R_poiseuille'] * self.repair_value