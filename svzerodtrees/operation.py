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
        repair_branches = [config_handler.lpa.branch, config_handler.rpa.branch]

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

    stenoses = [] # list of stenosis objects

    for branch, value in zip(repair_branches, repair_config['value']):

        # create the stenosis object and repair it
        branch_stenosis = Stenosis.create(config_handler, branch, repair_config['type'], value, log_file=log_file)
        branch_stenosis.repair()
        # create the list of stenosis objects for analysis later
        stenoses.append(branch_stenosis)
    
    # compute the postop simulation result
    config_handler.simulate(result_handler, 'postop')

    # return the list of stenoses for analysis (e.g. optimization)
    return stenoses




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
        self.diameters = [vessel.diameter for vessel in vessels] # original diameters
        self.repair_value = repair_value
        self.log_file = log_file
        self.vessels = vessels
        self.viscosity = viscosity

    @classmethod
    def create(cls, config_handler: ConfigHandler, branch, repair_type, repair_value, log_file=None):
        '''
        create a stenosis from a config handler

        :param config_handler: the config handler
        :param branch: the branch id (int or str)
        :param repair: the repair dict with type and value
        '''

        if branch == 'lpa':
            branch = config_handler.lpa.branch
        elif branch == 'rpa':
            branch = config_handler.rpa.branch

        # get the vessels in the branch
        vessels = config_handler.get_segments(branch)

        return cls(vessels, branch, repair_type, repair_value, config_handler.simparams.viscosity, log_file)

    
    def repair(self):
        '''repair the stenosis according to the specs'''
        
        if self.repair_type == 'stenosis_coefficient':
            write_to_log(self.log_file, "adjusting stenosis coefficient in branch " + str(self.branch) + " with stenosis coefficient " + str(self.repair_value))
            self.sc_repair()
        elif self.repair_type == 'stent':
            write_to_log(self.log_file, "repairing stenosis in branch " + str(self.branch) + " with stent diameter " + str(self.repair_value))
            self.stent_repair()
        elif self.repair_type == 'resistance':
            write_to_log(self.log_file, "repairing stenosis in branch " + str(self.branch) + " with resistance " + str(self.repair_value))
            self.resistance_repair()
            
    
    def sc_repair(self):
        '''
        repair the stenosis by adjusting the stenosis coefficient
        '''

        # change the stenosis coefficient
        for vessel in self.vessels:
            vessel.stenosis_coefficient *= (1 - self.repair_value)

    def stent_repair(self):
        '''
        repair the stenosis by changing the diameter according to stent diameter'''

        for vessel in self.vessels:
            # set stenosis coefficient to zero
            # R_old = vessel.R
            # vessel.stenosis_coefficient = 0.0
            # vessel.R = (8 * self.viscosity * vessel.length) / (np.pi * (self.repair_value / 2) ** 4)
            # R_change = R_old - vessel.R
            vessel.stenosis_coefficient = 0.0
            vessel.diameter = self.repair_value


    

    def resistance_repair(self):
        '''
        repair the stenosis by adjusting the resistance
        '''

        for vessel in self.vessels:
            vessel.R *= self.repair_value