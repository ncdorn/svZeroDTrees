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

        self.map_vessels_to_branches()
        self.build_tree_map()
        self.compute_R_eq()


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
    

    def set_inflow(self, Q_in):
        '''
        set the inflow for the config
        '''

        for bc in self.config["boundary_conditions"]:
            if bc["bc_name"] == "INFLOW":
                bc["bc_values"]["Q"] = [Q_in] * len(bc["bc_values"]["t"])


    def map_vessels_to_branches(self):
        '''
        map each vessel id to a branch id to deal with the case of multiple vessel ids in the same branch
        '''

        self.vessel_branch_map = {}
        for vessel in self.config["vessels"]:
            self.vessel_branch_map[vessel['vessel_id']] = get_branch_id(vessel)[0]


    def find_vessel_paths(self):
            '''
            find the path from the root to each vessel
            '''

            # helper function for depth-first search
            def dfs(vessel, path):

                if vessel is None:
                    return
                
                # add current vessel to the path
                path.append(vessel.branch)

                vessel.path = path.copy()
                
                vessel.gen = len(path) - 1

                for child in vessel.children:
                    dfs(child, path.copy())

            dfs(self.root, [])


    def build_vessel(self, branch):
        '''build a vessel object given a branch id'''

        for vessel_config in self.config['vessels']:
            br, seg = get_branch_id(vessel_config)
            if br == branch and seg == 0:
                self.vessel_map[br] = self.Vessel(vessel_config)
            elif br == branch and seg != 0:
                self.vessel_map[br].add_segment(vessel_config)


    def build_tree_map(self):
        '''
        Build a recursive map of the tree structure using self.Vessel objects.
        Useful for dfs and vessel distance analysis

        :return self.vessel_map: a dict where the keys are branch ids and the values are Vessel objects
        '''
        self.vessel_map = {}
        # initialize the vessel map (dict of branches)
        for vessel_config in self.config['vessels']:
            br, seg = get_branch_id(vessel_config)
            if seg == 0:
                self.vessel_map[br] = self.Vessel(vessel_config)
            else: 
                self.vessel_map[br].add_segment(vessel_config)
        
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


        self.find_vessel_paths()


    def compute_R_eq(self):
        '''
        calculate the equivalent resistance for a vessel

        :param vessel: vessel to calculate resistance for
        '''

        # get the resistance of the children
        def calc_R_eq(vessel):
            if len(vessel.children) != 0:
                for child in vessel.children:
                    calc_R_eq(child)
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
            # segments in the branch
            segs = [0]
            # get info from vessel config
            self.length = config['vessel_length']
            self.ids = [config['vessel_id']]
            self.branch = get_branch_id(config)[0]
            self.zero_d_element_values = config['zero_d_element_values']
            self.R_eq = 0.0
        

        def add_segment(self, config: dict):
            '''
            add a segment to the vessel
            '''
            # add the length
            self.length += config['vessel_length']
            # add the vessel id of the segment
            self.ids.append(config['vessel_id'])
            # add zero d element values
            self.zero_d_element_values['R_poiseuille'] += config['zero_d_element_values']['R_poiseuille']
            self.zero_d_element_values['C'] = 1 / ((1 / self.zero_d_element_values['C']) + (1 / config['zero_d_element_values']['C']))
            self.zero_d_element_values['L'] += config['zero_d_element_values']['L']
            self.zero_d_element_values['stenosis_coefficient'] += config['zero_d_element_values']['stenosis_coefficient']
            # add the segment number
            self.segs.append(get_branch_id(config)[1])
