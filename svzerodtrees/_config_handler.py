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

        # initialize config maps
        self.branch_map = {}
        self.junctions = {}
        self.bcs = {}
        self.simparams = None

        # build the config maps
        self.map_vessels_to_branches()
        self.build_config_map()

        # compute equivalent resistance
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


    def assemble_config(self):
        '''
        assemble the config dict from the config maps
        '''

        # this is a separate config for debugging purposes
        self.assembled_config = {}
        # add the boundary conditions
        self.assembled_config['boundary_conditions'] = [bc.to_dict() for bc in self.bcs.values()]

        # add the junctions
        self.assembled_config['junctions'] = [junction.to_dict() for junction in self.junctions.values()]

        # add the simulation parameters
        self.assembled_config['simulation_parameters'] = self.simparams.to_dict()

        # add the vessels
        self.assembled_config['vessels'] = [vessel.to_dict() for vessel in self.branch_map.values()]


    def convert_struct_trees_to_dict(self):
        '''
        convert the StructuredTreeOutlet instances into dict instances
        '''

        pass


    def clear_config_trees(self):
        '''
        clear the trees from the config dict
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
    

    def set_inflow(self, Q_in):
        '''
        set the inflow for the config
        '''


        self.bcs["INFLOW"].bc_values["Q"] = [Q_in] * len(self.bcs["INFLOW"].bc_values["t"])


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
                self.branch_map[br] = self.Vessel(vessel_config)
            elif br == branch and seg != 0:
                self.branch_map[br].add_segment(vessel_config)


    def build_config_map(self):
        '''
        Build a recursive map of the tree structure using self.Vessel objects.
        Useful for dfs and vessel distance analysis

        :return self.branch_map: a dict where the keys are branch ids and the values are Vessel objects
        return self.junction_mapL a dict where the keys are junction ids and the values are Junction objects
        '''
        self.branch_map = {}
        # initialize the vessel map (dict of branches)
        for vessel_config in self.config['vessels']:
            br, seg = get_branch_id(vessel_config)
            if seg == 0:
                self.branch_map[br] = Vessel.from_config(vessel_config)
            else: 
                self.branch_map[br].add_segment(vessel_config)
        
        # initialize the junction map (dict of junctions)
        self.junctions = {}
        for junction_config in self.config['junctions']:
            if junction_config['junction_type'] == 'NORMAL_JUNCTION':
                self.junctions[junction_config['junction_name']] = Junction.from_config(junction_config, self.vessel_branch_map)
        
        # initialize the boundary condition map (dict of boundary conditions)
        self.bcs = {}
        for bc_config in self.config['boundary_conditions']:
            self.bcs[bc_config['bc_name']] = BoundaryCondition.from_config(bc_config)

        # initialize the simulation parameters
        self.simparams = SimParams(self.config['simulation_parameters'])

        # loop through junctions and add children to parent vessels
        for junction in self.junctions.values():

            if len(junction.inlet_branches) > 1:
                raise Exception("there is more than one inlet to this junction")

            parent_vessel = self.branch_map[junction.inlet_branches[0]]

            # connect all the vessel instances
            for outlet in junction.outlet_branches:
                child_vessel = self.branch_map[outlet]
                child_vessel.parent = parent_vessel
                parent_vessel.children.append(child_vessel)

        # find the root vessel
        self.root = None
        for vessel in self.branch_map.values():
            if not any(vessel in child_vessel.children for child_vessel in self.branch_map.values()):
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


    def change_zerod_element_value(self, branch: id, param: dict):
        '''
        change the value of a zero d element in a branch

        :param branch: id of the branch to change
        :param param: dict of the parameter to change and the new value
        '''

        vessel = self.branch_map[branch]

        seg_ids = vessel.ids



        for name, value in param.items():
            vessel.zero_d_element_values[name] = value
        





class Vessel:
    '''
    class to handle BloodVessel LPN tree structure creation and dfs on the tree
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
        self.segs = [0]
        # boundary condition
        if 'boundary_conditions' in config:
            self.bc = config['boundary_conditions']
        else:
            self.bc = None
        # get info from vessel config
        self.length = config['vessel_length']
        self.ids = [config['vessel_id']]
        self.branch = get_branch_id(config)[0]
        self.zero_d_element_values = config['zero_d_element_values']
        self.R_eq = 0.0
    
    @classmethod
    def from_config(cls, config):
        '''
        create a vessel from a config dict

        :param config: config dict
        '''

        return cls(config)
    
    def to_dict(self):
        '''
        convert the vessel to a dict for zerod solver use
        '''

        if self.bc is None:
            return {
                'vessel_id': self.branch,
                'vessel_length': self.length,
                'vessel_name': 'branch' + str(self.branch) + '_seg0',
                'zero_d_element_type': "BloodVessel",
                'zero_d_element_values': self.zero_d_element_values,
            }
        
        else:
            return {
                'boundary_conditions': self.bc,
                'vessel_id': self.branch,
                'vessel_length': self.length,
                'vessel_name': 'branch' + str(self.branch) + '_seg0',
                'zero_d_element_type': "BloodVessel",
                'zero_d_element_values': self.zero_d_element_values,
            }

        return 

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


class Junction:
    '''
    class to handle junction LPN blocks
    '''

    def __init__(self, config, in_branches, out_branches):
        self.inlet_branches = in_branches
        self.outlet_branches = out_branches
        self.name = config['junction_name']
        self.areas = config['areas']
        self.type = config['junction_type']
    
    @classmethod
    def from_config(cls, config, vessel_branch_map):
        '''
        create a junction from a config dict

        :param config: config dict
        :param vessel_branch_map: dict mapping vessel ids to branch ids
        '''

        in_branches = [vessel_branch_map[inlet] for inlet in config['inlet_vessels']]
        out_branches = [vessel_branch_map[outlet] for outlet in config['outlet_vessels']]

        return cls(config, in_branches, out_branches)
    
    def to_dict(self):
        '''
        convert the junction to a dict for zerod solver use
        '''

        return {
            'junction_name': self.name,
            'junction_type': self.type,
            'inlet_vessels': self.inlet_branches,
            'outlet_vessels': self.outlet_branches,
            'areas': self.areas
        }
    

class BoundaryCondition:
    '''
    class to handle boundary conditions
    '''

    def __init__(self, config: dict):
        self.name = config['bc_name']
        self.type = config['bc_type']
        self.values = config['bc_values']
    
    @classmethod
    def from_config(cls, config):
        '''
        create a boundary condition from a config dict

        :param config: config dict
        '''

        return cls(config)
    
    def to_dict(self):
        '''
        convert the boundary condition to a dict for zerod solver use
        '''

        return {
            'bc_name': self.name,
            'bc_type': self.type,
            'bc_values': self.values
        }
    

class SimParams:
    '''class to handle simulation parameters'''

    def __init__(self, config: dict):
        self.density = config['density']
        self.model_name = config['model_name']
        self.number_of_cardiac_cycles = config['number_of_cardiac_cycles']
        self.number_of_time_pts_per_cardiac_cycle = config['number_of_time_pts_per_cardiac_cycle']
        self.viscosity = config['viscosity']

    @classmethod
    def from_config(cls, config):
        '''
        create a simulation parameters object from a config dict

        :param config: config dict
        '''

        return cls(config)
    
    def to_dict(self):
        '''
        convert the simulation parameters to a dict for zerod solver use
        '''

        return {
            'density': self.density,
            'model_name': self.model_name,
            'number_of_cardiac_cycles': self.number_of_cardiac_cycles,
            'number_of_time_pts_per_cardiac_cycle': self.number_of_time_pts_per_cardiac_cycle,
            'viscosity': self.viscosity
        }
