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


    @classmethod
    def from_json(cls, file_name: str):
        '''
        load in a config dict from json file

        :param file_name: name of the file to load from
        '''

        with open(file_name) as ff:
            config = json.load(ff)

        return ConfigHandler(config)
    
    @classmethod
    def from_file(cls, file_name: str):
        '''
        load in a config dict from binary file via pickle
        '''

        with open(file_name, 'rb') as ff:
            config = pickle.load(ff)

        return ConfigHandler(config)


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
                        if vessel_config["boundary_conditions"]["outlet"] == bc_config["bc_name"]:
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
                        if vessel_config["boundary_conditions"]["outlet"] == bc_config["bc_name"]:
                            vessel_config["tree"] = {}
    

    def update_stree_hemodynamics(self, current_result):
        '''
        update the hemodynamics of the StructuredTreeOutlet instances
        '''

        # get the outlet flowrate
        q_outs = get_outlet_data(self.config, current_result, "flow_out", steady=True)
        p_outs = get_outlet_data(self.config, current_result, "pressure_out", steady=True)
        outlet_idx = 0 # need this when iterating through outlets 
        # get the outlet vessel
        for vessel_config in self.config["vessels"]:
            if "boundary_conditions" in vessel_config:
                if "outlet" in vessel_config["boundary_conditions"]:
                    for bc_config in self.config["boundary_conditions"]:
                        if vessel_config["boundary_conditions"]["outlet"] == bc_config["bc_name"]:
                            # update the outlet flowrate and pressure for the tree at that outlet
                            print(outlet_idx, len(self.trees))
                            self.trees[outlet_idx].add_hemodynamics_from_outlet([q_outs[outlet_idx]], [p_outs[outlet_idx]])

                            # re-integrate pries and secomb --- this is not necesary at the moment because I think it will send us into an
                            # endless loop of re-integration

                            # self.trees[outlet_idx].integrate_pries_secomb()

                            # count up
                            outlet_idx += 1


    def get_time_series(self):
        '''
        get the time series from the config
        '''

        t_min = min(self.config["boundary_conditions"][0]["bc_values"]["t"])
        t_max = max(self.config["boundary_conditions"][0]["bc_values"]["t"])

        return np.linspace(min(self.config["boundary_conditions"][0]["bc_values"]["t"]), 
                           max(self.config["boundary_conditions"][0]["bc_values"]["t"]), 
                           self.config["simulation_parameters"]["number_of_time_pts_per_cardiac_cycle"])
    

    def get_branch_resistances(self):
        '''
        get the branch resistances from the config for centerline projection
        '''
        
        for vessel in self.config["vessels"]:
            br, seg = vessel["vessel_name"].split("_")
            R = vessel["zero_d_element_values"]["R_poiseuille"]
            

### below is a set of routines for loading in a pulmonary artery model and creating a tree structure from it

    def load_pa_model(self):
        pass


    def map_vessels_to_branches(self):
        '''
        map each vessel id to a branch id
        '''

        for vessel in self.vessels:
            self.vessel_branch_map[vessel['vessel_id']] = get_branch_id(vessel)


    def combine_vessels(self, vessel1, vessel2):
        '''
        combine two vessels of the same branch into one vessel
        :param vessel1: first vessel
        :param vessel2: second vessel
        '''
        vessel_config = dict(
            label = vessel1.label,
            vessel_id = vessel1.branch,
            vessel_length = vessel1.length + vessel2.length,
            vessel_name = 'branch' + str(vessel1.branch) + '_seg0',
            zero_d_element_values = {
                'R_poiseuille': vessel1.zero_d_element_values['R_poiseuille'] + vessel2.zero_d_element_values['R_poiseuille'],
                # need to update this to be actual math
                'C': vessel1.zero_d_element_values['C'] + vessel2.zero_d_element_values['C'],
                # need to update this to be actual math
                'L': vessel1.zero_d_element_values['L'] + vessel2.zero_d_element_values['L'],
                # take the max, there is definitely some math to be done here
                'stenosis_coefficient': max(vessel1.zero_d_element_values['stenosis_coefficient'], vessel2.zero_d_element_values['stenosis_coefficient'])
            }
        )

        return self.Vessel(vessel_config)


    def build_tree_map(self):
        '''
        Build a recursive map of the tree structure using self.Vessel objects.
        Useful for dfs and vessel distance analysis

        :return self.vessel_map: a dict where the keys are branch ids and the values are Vessel objects
        '''

        # initialize the vessel map (dict of branches)
        for vessel_config in self.config['vessels']:
            branch = get_branch_id(vessel_config)
            if branch not in self.vessel_map.keys():
                self.vessel_map[branch] = self.Vessel(vessel_config)
            else: 
                self.vessel_map[branch] = self.combine_vessels(self.vessel_map[branch], self.Vessel(vessel_config))
            
            self.vessel_map[branch].d = get_branch_d(self.config, self.config["simulation_parameters"]["viscosity"], branch)
            # map vessel id to branch
        
        # loop through junctions and add children to parent vessels
        for junction in self.config['junctions']:
            # map inlet vessel id to branches
            inlets = [self.vessel_branch_map[inlet] for inlet in junction['inlet_vessels']]
            
            if len(inlets) > 1:
                raise Exception("there is more than one inlet to this junction")
            
            # map outlet vessel id to branches
            outlets = [self.vessel_branch_map[outlet] for outlet in junction['outlet_vessels']]

            # if the inlet and outlet branches are the same, skip
            if inlets == outlets:
                continue

            parent_vessel = self.vessel_map[inlets[0]]

            # connect all the vessel instances
            for outlet in outlets:
                child_vessel = self.vessel_map[outlet]
                child_vessel.parent = parent_vessel
                parent_vessel.children.append(child_vessel)

        # find the root vessel
        self.root = None
        for vessel in self.vessel_map.values():
            if not any(vessel in child_vessel.children for child_vessel in self.vessel_map.values()):
                self.root = vessel
        
        # organize the children in numerical order
        # we assume that the branches are sorted alphabetically, and therefore the lpa comes first.
        self.root.children.sort(key=lambda x: x.branch)


        # label the mpa, rpa and lpa
        self.root.label = 'mpa'
        self.root.children[0].label = 'lpa'
        self.root.children[1].label = 'rpa'
        
        # add these in incase we index using the mpa, lpa, rpa strings
        self.mpa = self.root
        self.lpa = self.root.children[0]
        self.rpa = self.root.children[1]

        # modify the self.result array to include the mpa, rpa, lpa with numerical ids
        self.result[str(self.mpa.branch)] = self.result.pop('mpa')
        self.result[str(self.rpa.branch)] = self.result.pop('rpa')
        self.result[str(self.lpa.branch)] = self.result.pop('lpa')

        keys = list(self.result.keys())
        keys.sort(key=int)

        self.result = {key: self.result[key] for key in keys}

        self.find_vessel_paths()

    def get_R_eq(self):
        '''
        calculate the equivalent resistance for a vessel

        :param vessel: vessel to calculate resistance for
        '''

        # get the resistance of the children
        def calc_R_eq(vessel):
            if len(vessel.children) != 0:
                calc_R_eq(vessel.children[0])
                calc_R_eq(vessel.children[1])
                vessel.R_eq = vessel.zero_d_element_values['R_poiseuille'] + (1 / sum([1 / child.R_eq for child in vessel.children]))
            else:
                vessel.R_eq = vessel.zero_d_element_values['R_poiseuille']
        
        calc_R_eq(self.root)

    class Vessel:
        '''
        class to handle tree structure creation and dfs on the tree

        '''
        def __init__(self, config: dict):
            # optional name, such as mpa, lpa, rpa to classify the vessel
            self.label= None
            # list of child vessels
            self.children = []
            self.parent = None
            # path to root
            self.path = []
            # generation from root
            self.gen = None
            # get info from vessel config
            self.length = config['vessel_length']
            self.id = config['vessel_id']
            self.branch = get_branch_id(config)
            self.zero_d_element_values = config['zero_d_element_values']
            self.R_eq = 0.0
            # get the branch diameter
            self.d = 0.0