from svzerodtrees.utils import *
import json
import pickle


class ConfigHandler():
    '''
    class to handle configs with and without trees
    '''

    def __init__(self, config: dict):
        self.config = config
        self.trees = []


    def to_json(self, file_name: str):
        '''
        write the desired config to file

        :param file_name: name of the file to write to
        '''
        with open(file_name, 'w') as ff:
            json.dump(self.config, ff)

    def to_json_w_trees(self, file_name: str):
        '''
        write the desired config to file

        :param file_name: name of the file to write to
        '''

        outlet_idx = 0
        for vessel_config in self.config["vessels"]:
            if "boundary_conditions" in vessel_config:
                if "outlet" in vessel_config["boundary_conditions"]:
                    for bc_config in self.config["boundary_conditions"]:
                        if vessel_config["boundary_conditions"]["outlet"] in bc_config["bc_name"]:
                            vessel_config["tree"] = self.trees[outlet_idx].block_dict

                    outlet_idx += 1


        with open(file_name, 'w') as ff:
            json.dump(self.config, ff)
        
        self.clear_config_trees()


    def to_file(self, file_name: str):
        '''
        write the desired config to a binary file via pickle

        :param file_name: name of the file to write to
        '''
        with open(file_name, 'wb') as ff:
            pickle.dump(self.config, ff)

    def to_file_w_trees(self, file_name: str):
        '''
        write the desired config with trees to a binary file via pickle

        :param file_name: name of the file to write to
        '''

        outlet_idx = 0
        for vessel_config in self.config["vessels"]:
            if "boundary_conditions" in vessel_config:
                if "outlet" in vessel_config["boundary_conditions"]:
                    for bc_config in self.config["boundary_conditions"]:
                        if vessel_config["boundary_conditions"]["outlet"] in bc_config["bc_name"]:
                            vessel_config["tree"] = self.trees[outlet_idx]

                    outlet_idx += 1

        with open(file_name, 'wb') as ff:
            pickle.dump(self.config, ff)

        self.clear_config_trees()
    

    def from_file_w_trees(self, file_name: str):
        '''
        load in a config dict with trees from a binary file via pickle
        '''

        with open(file_name, 'rb') as ff:
            self.config = pickle.load(ff)
        
        self.trees = []
        for vessel_config in self.config["vessels"]:
            if "boundary_conditions" in vessel_config:
                if "outlet" in vessel_config["boundary_conditions"]:
                    for bc_config in self.config["boundary_conditions"]:
                        if vessel_config["boundary_conditions"]["outlet"] in bc_config["bc_name"]:
                            self.trees.append(vessel_config["tree"])

        self.clear_config_trees()


    def convert_struct_trees_to_dict(self):
        '''
        convert the StructuredTreeOutlet instances into dict instances
        '''

        pass

    def clear_config_trees(self):
        '''
        clear the trees from the config
        '''

        for vessel_config in self.config["vessels"]:
            if "boundary_conditions" in vessel_config:
                if "outlet" in vessel_config["boundary_conditions"]:
                    for bc_config in self.config["boundary_conditions"]:
                        if vessel_config["boundary_conditions"]["outlet"] in bc_config["bc_name"]:
                            vessel_config["tree"] = {}

    
