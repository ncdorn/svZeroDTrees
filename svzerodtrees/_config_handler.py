from svzerodtrees.utils import *
from svzerodtrees._result_handler import ResultHandler
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
        self.vessel_map = {}
        self.junctions = {}
        self.bcs = {}
        self.simparams = None

        # build the config maps
        self.map_vessels_to_branches()
        self.build_config_map()

        # compute equivalent resistance
        # self.compute_R_eq()


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


    def simulate(self, result_handler: ResultHandler, label: str):
        '''
        run the simulation

        :param result_handler: result handler instance to add the result to
        :param label: label for the result e.g. preop, postop, final
        '''

        # assemble the config
        self.assemble_config()

        # run the simulation
        result = run_svzerodplus(self.config)

        # add the result to the result handler
        result_handler.add_unformatted_result(result, label)


    def assemble_config(self):
        '''
        assemble the config dict from the config maps
        '''

        # this is a separate config for debugging purposes
        self.config = {}
        # add the boundary conditions
        self.config['boundary_conditions'] = [bc.to_dict() for bc in self.bcs.values()]

        # add the junctions
        self.config['junctions'] = [junction.to_dict() for junction in self.junctions.values()]

        # add the simulation parameters
        self.config['simulation_parameters'] = self.simparams.to_dict()

        # add the vessels
        self.config['vessels'] = [vessel.to_dict() for vessel in self.vessel_map.values()]


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


        self.bcs["INFLOW"].values["Q"] = [Q_in] * len(self.bcs["INFLOW"].values["t"])


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


    def build_config_map(self):
        '''
        Build a recursive map of the tree structure using self.Vessel objects.
        Useful for dfs and vessel distance analysis

        :return self.branch_map: a dict where the keys are branch ids and the values are Vessel objects
        return self.junction_mapL a dict where the keys are junction ids and the values are Junction objects
        '''

        # initialize the vessel map (dict of branches)
        for vessel_config in self.config['vessels']:
            self.vessel_map[vessel_config['vessel_id']] = Vessel.from_config(vessel_config)
            br, seg = get_branch_id(vessel_config)
            if seg == 0:
                self.branch_map[br] = Vessel.from_config(vessel_config)
            else: 
                self.branch_map[br].add_segment(vessel_config)
        
        # initialize the junction map (dict of junctions)
        for junction_config in self.config['junctions']:
            self.junctions[junction_config['junction_name']] = Junction.from_config(junction_config)
        
        # initialize the boundary condition map (dict of boundary conditions)
        for bc_config in self.config['boundary_conditions']:
            self.bcs[bc_config['bc_name']] = BoundaryCondition.from_config(bc_config)

        # initialize the simulation parameters
        self.simparams = SimParams(self.config['simulation_parameters'])

        # loop through junctions and add children to parent BRANCHES
        for junction in self.junctions.values():
            # make sure we ignore internal junctions since we are just dealing with branches
            if junction.type == 'NORMAL_JUNCTION':

                if len(junction.inlet_branches) > 1:
                    raise Exception("there is more than one inlet to this junction")

                parent_vessel = self.branch_map[self.vessel_branch_map[junction.inlet_branches[0]]]

                # connect all the vessel instances
                for outlet in [self.vessel_branch_map[outlet] for outlet in junction.outlet_branches]:
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
                    # we assume here that the stenosis coefficient is linear, which is not true but a reasonable approximation
                vessel.R_eq = vessel.R + vessel.zero_d_element_values['stenosis_coefficient'] + (1 / sum([1 / child.R_eq for child in vessel.children]))
            else:
                vessel.R_eq = vessel.R + vessel.zero_d_element_values['stenosis_coefficient']
        
        calc_R_eq(self.root)


    def change_zerod_element_value(self, branch_id: int, param: dict):
        '''
        change the value of a zero d element in a branch

        :param branch: id of the branch to change
        :param param: dict of the parameter to change and the new value
        '''
        
        # get the branch object and the segments in that branch
        branch = self.branch_map[branch_id]
        vessels = self.get_segments(branch_id)


    def get_segments(self, branch: int or str, dtype: str = 'vessel', junctions=False):
        '''
        get the vessels in a branch

        :param branch: id of the branch to get the vessels of
        :param dtype: type of data to return, either 'Vessel' class or 'dict'
        '''
        if branch == 'mpa':
            branch = self.mpa.branch
        elif branch == 'lpa':
            branch = self.lpa.branch
        elif branch == 'rpa':
            branch = self.rpa.branch

        if dtype == 'vessel':
            return [self.vessel_map[id] for id in self.branch_map[branch].ids]
        if dtype == 'dict':
            return [self.vessel_map[id].to_dict() for id in self.branch_map[branch].ids]

    

class Vessel:
    '''
    class to handle BloodVessel LPN tree structure creation and dfs on the tree
    used for both vessels (vessel map) and branches (branch map)
    '''

    def __init__(self, config: dict):
        # optional name, such as mpa, lpa, rpa to classify the vessel
        self.label= None
        self.name = config['vessel_name']
        # list of child vessels
        self._children = []
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
        self.id = config['vessel_id']
        # list of ids if multiple segments
        self.ids = [config['vessel_id']]
        self.branch = get_branch_id(config)[0]
        self.zero_d_element_values = config['zero_d_element_values']
        self._R = config['zero_d_element_values']['R_poiseuille']
        self._C = config['zero_d_element_values']['C']
        self._L = config['zero_d_element_values']['L']
        # get equivalent values, initilized with the values of the vessel
        self._R_eq = self._R
        self._C_eq = self._C
        self._L_eq = self._L
        # get diameter with viscosity 0.04
        self.diameter = ((128 * 0.04 * self.length) / (np.pi * self.zero_d_element_values['R_poiseuille'])) ** (1 / 4)
    
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
                'vessel_id': self.id,
                'vessel_length': self.length,
                'vessel_name': self.name,
                'zero_d_element_type': "BloodVessel",
                'zero_d_element_values': {
                    'R_poiseuille': self.R,
                    'C': self.C,
                    'L': self.L,
                },
            }
        
        else:
            return {
                'boundary_conditions': self.bc,
                'vessel_id': self.id,
                'vessel_length': self.length,
                'vessel_name': self.name,
                'zero_d_element_type': "BloodVessel",
                'zero_d_element_values': {
                    'R_poiseuille': self.R,
                    'C': self.C,
                    'L': self.L,
                },
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

        # add bc
        if 'boundary_conditions' in config:
            self.bc = config['boundary_conditions']

    #### property setters for dynamically updating the equivalent values ####
    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, new_R):
        self._R = new_R

    @property
    def R_eq(self):
        if len(self.children) != 0:
            self._update_R_eq()
        return self._R_eq

    def _update_R_eq(self):
        self._R_eq = self._R + (1 / sum([1 / child.R_eq for child in self.children]))
    
    @property
    def C(self):
        return self._C
    
    @C.setter
    def C(self, new_C):
        self._C = new_C

    @property
    def C_eq(self):
        if len(self.children) != 0:
            self._update_C_eq()
        return self._C_eq
    
    def _update_C_eq(self):
        self._C_eq = 1 / ((1 / self._C) + (1 / sum(child.R_eq for child in self.children)))

    @property
    def L(self):
        return self._L

    @L.setter
    def L(self, new_L):
        self._L = new_L

    @property
    def L_eq(self):
        if len(self.children) != 0:
            self._update_L_eq()
        return self._L_eq

    def _update_L_eq(self):
        self._L_eq = self._L + (1 / sum([1 / child.R_eq for child in self.children]))
        
    @property
    def children(self):
        return self._children
    
    @children.setter
    def children(self, new_children):
        for child in new_children:
            child.parent = self
        self._children = new_children


class Junction:
    '''
    class to handle junction LPN blocks
    '''

    def __init__(self, config):
        self.inlet_branches = config['inlet_vessels']
        self.outlet_branches = config['outlet_vessels']
        self.name = config['junction_name']
        if 'areas' in config:
            self.areas = config['areas']
        else:
            self.areas = None
        self.type = config['junction_type']

    @classmethod
    def from_config(cls, config):
        '''
        create a junction from a config dict

        :param config: config dict
        :param vessel_branch_map: dict mapping vessel ids to branch ids
        '''

        return cls(config)
    
    @classmethod
    def from_vessel(cls, inlet_vessel: Vessel):
        '''
        generate a junction from inlet vessel and a list of outlet vessels'''
        # determine if tehere are outlet vessels
        if len(inlet_vessel.children) == 0:
            return None
        # determine if it is a normal junction or internal junction
        if len(inlet_vessel.children) == 1:
            junction_type = 'internal_junction'
        else:
            junction_type = 'NORMAL_JUNCTION'
        config = {
            'junction_name': 'J' + str(inlet_vessel.id),
            'junction_type': junction_type,
            'inlet_vessels': [inlet_vessel.id],
            'outlet_vessels': [outlet_vessel.id for outlet_vessel in inlet_vessel.children],
            'areas': [outlet_vessel.diameter ** 2 * np.pi / 4 for outlet_vessel in inlet_vessel.children]
        }

        return cls(config)


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
        if self.type == 'RESISTANCE':
            self._R = self.values['R']
    
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
    
    # a setter so we can change the resistances in the BC easier
    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, new_R):
        self._R = new_R
        self.values['R'] = new_R
    

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