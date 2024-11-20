import json
import pickle
import os
from svzerodtrees.utils import *
from svzerodtrees._result_handler import ResultHandler
import pysvzerod



class ConfigHandler():
    '''
    class to handle configs with and without trees
    '''

    def __init__(self, config: dict, is_pulmonary=True, is_threed_interface=False, closed_loop=False):
        self._config = config

        self.trees = {} # list of StructuredTree instances

        # initialize config maps
        self.branch_map = {} # {branch id: Vessel instance}
        self.vessel_map = {} # {vessel id: Vessel instance}
        self.junctions = {} # {junction name: Junction instance}
        self.bcs = {} # {bc name: BoundaryCondition instance}

        self.simparams = None

        self.is_pulmonary = is_pulmonary
        self.threed_interface = is_threed_interface

        # build the config maps
        self.map_vessels_to_branches()
        self.build_config_map()

        if closed_loop:
            print('this is a closed loop simulation!')

    #### I/O METHODS ####
    @classmethod
    def from_json(cls, file_name: str, is_pulmonary=True, is_threed_interface=False):
        '''
        load in a config dict from json file

        :param file_name: name of the file to load from
        '''

        with open(file_name) as ff:
            config = json.load(ff)
        if "external_solver_coupling_blocks" in config:
            is_threed_interface = True

        return ConfigHandler(config, is_pulmonary, is_threed_interface)
    
    @classmethod
    def from_file(cls, file_name: str, is_pulmonary=True):
        '''
        load in a config dict from binary file via pickle
        '''

        with open(file_name, 'rb') as ff:
            config = pickle.load(ff)
        
        if "external_solver_coupling_blocks" in config:
            is_threed_interface = True

        return ConfigHandler(config, is_pulmonary, is_threed_interface)


    def to_json(self, file_name: str):
        '''
        write the desired config to file

        :param file_name: name of the file to write to
        '''
        with open(file_name, 'w') as ff:
            json.dump(self.config, ff, indent=4)


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
                            vessel_config["tree"] = list(self.trees.values())[outlet_idx].block_dict

                    outlet_idx += 1


        with open(file_name, 'w') as ff:
            json.dump(self.config, ff, indent=4)
        
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
        write the desired config handler with trees to a binary file via pickle

        :param file_name: name of the file to write to
        '''

        with open(file_name, 'wb') as ff:
            pickle.dump(self, ff)
    

    def from_file_w_trees(self, file_name: str):
        '''
        Load a config dictionary with trees from a binary file via pickle.

        Parameters:
            file_name (str): The name of the binary file to load.

        Returns:
            None
        '''
        with open(file_name, 'rb') as ff:
            self = pickle.load(ff)
    
    #### END OF I/O METHODS ####
            
    def simulate(self, result_handler: ResultHandler = None, label: str=None):
        '''
        run the simulation

        :param result_handler: result handler instance to add the result to
        :param label: label for the result e.g. preop, postop, adapted
        '''

        # assemble the config
        self.assemble_config()

        # run the simulation
        result = run_svzerodplus(self.config)

        result["time"] = self.get_time_series()

        if result_handler is not None:
            # add the vessels to the result handler
            result_handler.vessels[label] = self.config['vessels']
            # add the result to the result handler
            result_handler.add_unformatted_result(result, label)

        else:
            result_df = pysvzerod.simulate(self.config)
            
            return result_df
        


    def assemble_config(self):
        '''
        assemble the config dict from the config maps
        '''

        # this is a separate config for debugging purposes
        self._config = {}
        # add the boundary conditions
        self._config['boundary_conditions'] = [bc.to_dict() for bc in self.bcs.values()]

        # add the junctions
        self._config['junctions'] = [junction.to_dict() for junction in self.junctions.values()]

        # add the simulation parameters
        self._config['simulation_parameters'] = self.simparams.to_dict()

        # add the valves
        if hasattr(self, 'valves'):
            self._config['valves'] = [valve.to_dict() for valve in self.valves.values()]

        # add the chambers
        if hasattr(self, 'chambers'):
            self._config['chambers'] = [chamber.to_dict() for chamber in self.chambers.values()]

        # add the vessels
        self._config['vessels'] = [vessel.to_dict() for vessel in self.vessel_map.values()]

        if self.threed_interface:
            self._config['external_solver_coupling_blocks'] = [coupling_block.to_dict() for coupling_block in self.coupling_blocks.values()]


    def plot_inflow(self):
        '''
        plot the inflow
        '''
        plt.figure(2)
        plt.plot(self.config['boundary_conditions'][0]['bc_values']['t'], self.config['boundary_conditions'][0]['bc_values']['Q'])
        plt.show()


    def update_bcs(self, vals: list, rcr: bool):
        '''
        adjust the boundary conditions for the tree by changing the R or RCR values

        :param vals: list of values to change the boundary conditions to, with either R or Rp, C, Rd values
        :param rcr: bool to indicate if changing RCR or R BCs
        '''

        if rcr:
            for idx, bc in enumerate(list(self.bcs.values())[1:]):
                bc.Rp = vals[idx * 3]
                bc.C = vals[idx * 3 + 1]
                bc.Rd = vals[idx * 3 + 2]
        else:
            for idx, bc in enumerate(list(self.bcs.values())[1:]):
                bc.R = vals[idx]


    def convert_struct_trees_to_dict(self):
        '''
        convert the StructuredTree instances into dict instances
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
        update the hemodynamics of the StructuredTree instances
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
                            list(self.trees.values())[outlet_idx].add_hemodynamics_from_outlet([q_outs[outlet_idx]], [p_outs[outlet_idx]])

                            # re-integrate pries and secomb --- this is not necesary at the moment because I think it will send us into an
                            # endless loop of re-integration

                            # self.trees[outlet_idx].integrate_pries_secomb()

                            # count up
                            outlet_idx += 1


    def get_time_series(self):
        '''
        get the time series from the config
        '''

        return np.linspace(min(self.config["boundary_conditions"][0]["bc_values"]["t"]), 
                           max(self.config["boundary_conditions"][0]["bc_values"]["t"]), 
                           self.config["simulation_parameters"]["number_of_time_pts_per_cardiac_cycle"])
    

    def set_inflow(self, inflow):
        '''
        set the inflow for the config

        inflow: Inflow instance
        '''


        self.bcs["INFLOW"] = inflow.to_bc()


    def map_vessels_to_branches(self):
        '''
        map each vessel id to a branch id to deal with the case of multiple vessel ids in the same branch
        '''

        self.vessel_branch_map = {}
        for vessel in self._config["vessels"]:
            self.vessel_branch_map[vessel['vessel_id']] = get_branch_id(vessel)[0]

        self.branch_vessel_map = {}
        for k, v in self.vessel_branch_map.items():
            self.branch_vessel_map[v] = self.branch_vessel_map.get(v, []) + [k]


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
        for vessel_config in self._config['vessels']:
            self.vessel_map[vessel_config['vessel_id']] = Vessel.from_config(vessel_config)
            br, seg = get_branch_id(vessel_config)
            if seg == 0:
                self.branch_map[br] = Vessel.from_config(vessel_config)
            else: 
                self.branch_map[br].add_segment(vessel_config)
        
        # initialize the junction map (dict of junctions)
        for junction_config in self._config['junctions']:
            self.junctions[junction_config['junction_name']] = Junction.from_config(junction_config)
        
        # initialize the boundary condition map (dict of boundary conditions)
        for bc_config in self._config['boundary_conditions']:
            self.bcs[bc_config['bc_name']] = BoundaryCondition.from_config(bc_config)

        # initialize the simulation parameters
        self.simparams = SimParams(self._config['simulation_parameters'])

        # loop through junctions and add children to parent BRANCHES
        for junction in self.junctions.values():
            # make sure all junctions are of type NORMAL_JUNCTION
            if junction.type != 'NORMAL_JUNCTION':
                junction.type = 'NORMAL_JUNCTION'
            # make sure we ignore internal junctions since we are just dealing with branches
            if len(junction.inlet_branches) > 1 or len(junction.outlet_branches) > 1:

                # if len(junction.inlet_branches) > 1:
                #     raise Exception("there is more than one inlet to this junction")

                parent_branch = self.branch_map[self.vessel_branch_map[junction.inlet_branches[0]]]
                # from the vessel map
                parent_vessel = self.vessel_map[junction.inlet_branches[0]]

                # connect all the vessel instances
                for outlet in junction.outlet_branches:
                    parent_vessel.children.append(self.vessel_map[outlet])
                    self.vessel_map[outlet].parent = parent_vessel
                
                # connect branches
                for outlet in [self.vessel_branch_map[outlet] for outlet in junction.outlet_branches]:
                    child_branch = self.branch_map[outlet]
                    child_branch.parent = parent_branch
                    parent_branch.children.append(child_branch)
            
            else:
                # internal junctions are just a single vessel
                parent_vessel = self.vessel_map[junction.inlet_branches[0]]
                child_vessel = self.vessel_map[junction.outlet_branches[0]]
                child_vessel.parent = parent_vessel
                parent_vessel.children.append(child_vessel)

        if 'chambers' in self._config.keys():
            self.chambers = {}
            for chamber in self._config['chambers']:  
                self.chambers[chamber['name']] = Chamber.from_config(chamber)

        if 'valves' in self._config.keys():
            self.valves = {}
            for valve in self._config['valves']:
                self.valves[valve['name']] = Valve.from_config(valve)

        # find the root vessel
        if self._config['vessels'] != []:
            self.root = None
            for vessel in self.vessel_map.values():
                # this takes from the vessel map as opposed to the branch map. since the vessels are nonlinear we may have to use the vessel
                # map in the future
                if not any(vessel in child_vessel.children for child_vessel in self.vessel_map.values()):
                    root_br = self.vessel_branch_map[vessel.id]
                    # account for multiple vessels in root branch
                    self.root = self.vessel_map[self.branch_vessel_map[self.vessel_branch_map[vessel.id]][-1]]
            # organize the children in numerical order
            # we assume that the branches are sorted alphabetically, and therefore the lpa comes first.
            self.root.children.sort(key=lambda x: x.branch)
            # find the vessel paths
            self.find_vessel_paths()

        # label the mpa, rpa and lpa
        if self.is_pulmonary:
            self.root.label = 'mpa'
            self.root.children[0].label = 'lpa'
            self.root.children[1].label = 'rpa'
            
        # add these in incase we index using the mpa, lpa, rpa strings
        if self.is_pulmonary:
            self.mpa = self.root
            self.lpa = self.root.children[0]
            self.rpa = self.root.children[1]

        if self.threed_interface:
            self.coupling_blocks = {}
            for coupling_block in self._config["external_solver_coupling_blocks"]:
                # create a mapping from connected block name to coupling block
                self.coupling_blocks[coupling_block['connected_block']] = CouplingBlock.from_config(coupling_block)

        self.assemble_config()


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
                vessel.R_eq = vessel.R + vessel.stenosis_coefficient + (1 / sum([1 / child.R_eq for child in vessel.children]))
            else:
                vessel.R_eq = vessel.R + vessel.stenosis_coefficient
        
        calc_R_eq(self.root)


    def change_branch_resistance(self, branch_id: int, value, remove_stenosis_coefficient=True):
        '''
        change the value of a zero d element in a branch

        :param branch: id of the branch to change
        :param value: a list of values to change the resistance for the zero d elements
        :param remove_stenosis_coefficient: bool to keep or remove the stenosis coefficient
        '''
        

        if type(value) == list:
            # we are given a list here, so we will distribute each R to each vessel in the branch
            for idx, vessel in enumerate(self.get_segments(branch_id)):
                if remove_stenosis_coefficient:
                    vessel.stenosis_coefficient = 0.0
                vessel.R = value[idx]
        else:
            # we have a single value so we will distribute it amongst the branchs
            for idx, vessel in enumerate(self.get_segments(branch_id)):
                if remove_stenosis_coefficient:
                    vessel.stenosis_coefficient = 0.0
                vessel.R = value * (vessel.R / self.branch_map[branch_id].R)
            
            # set the branch resistance to the value
            self.branch_map[branch_id].R = value


    def get_branch_resistance(self, branch_id: int):
        '''
        get the resistance of a branch

        :param branch: id of the branch to get the resistance of
        '''

        return sum(vessel.R for vessel in self.get_segments(branch_id))


    def get_segments(self, branch, dtype: str = 'vessel', junctions=False):
        '''
        get the vessels in a branch

        :param branch: id of the branch to get the vessels of
        :param dtype: type of data to return, either 'Vessel' class or 'dict'

        :returns: list of names of coupling blocks
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
        
    
    def generate_threed_coupler(self, simdir, inflow_from_0d=False):
        '''
        create a 3D-0D coupling blocks config from the boundary conditions and save it to a json

        :param simdir: directory to save the json to

        :return coupling_block_list: list of coupling block names
        '''
        threed_coupler = ConfigHandler(
            {
                "simulation_parameters": {
                    "density": 1.06,
                    "viscosity": 0.04,
                    "coupled_simulation": True,
                    "number_of_time_pts": 2,
                    "output_all_cycles": True,
                    "steady_initial": False
                },
                "external_solver_coupling_blocks": [],
                "boundary_conditions": [],
                "vessels": [],
                "junctions": []
            },
            is_pulmonary=False,
            is_threed_interface=True
        )

        # copy over the bcs
        threed_coupler.bcs = self.bcs
        if inflow_from_0d:
            # need to add a vessel between the inflwo bc and coupling block to allow effective coupling
            threed_coupler.vessel_map[0] = Vessel.from_config(
                {
                   "boundary_conditions": {
                        "inlet": "INFLOW"
                    },
                    "vessel_id": 0,
                    "vessel_length": 10.0,
                    "vessel_name": "branch0_seg0",
                    "zero_d_element_type": "BloodVessel",
                    "zero_d_element_values": {
                        "C": 0.0000001,
                        "L": 0.0,
                        "R_poiseuille": 0.0000001,
                        "stenosis_coefficient": 0.0
                    }
                }
            )

            threed_coupler.coupling_blocks["inflow"] = CouplingBlock(
                {
                    "name": "inflow",
                    "type": "FLOW",
                    "location": "outlet",
                    "connected_block": "branch0_seg0",
                    "periodic": False,
                    "values": {
                            "t": [
                                0,
                                max(self.bcs["INFLOW"].values['t'])
                            ],
                            "Q": [
                                1.0,
                                1.0
                            ]
                    }
                }
            )


        else:
            del threed_coupler.bcs["INFLOW"] 
        

        # create the coupling blocks
        for i, bc in enumerate(threed_coupler.bcs.values()):
            if 'inflow' not in bc.name.lower():
                block_name = f'coupled_bc{i}'
                threed_coupler.coupling_blocks[block_name] = CouplingBlock.from_bc(bc)

        print('writing svzerod_3Dcoupling.json...')
        threed_coupler.to_json(os.path.join(simdir, 'svzerod_3Dcoupling.json'))

        coupling_block_list = [coupling_block.name for coupling_block in threed_coupler.coupling_blocks.values()]

        return coupling_block_list


    def generate_inflow_file(self, simdir):
        '''
        generate and inflow.flow file from the inflow bc of the zerod model'''

        print('writing inflow.flow...')

        with open(os.path.join(simdir, 'inflow.flow'), 'w') as ff:
            for t, q in zip(self.bcs["INFLOW"].values['t'], [q * -1 for q in self.bcs["INFLOW"].values['Q']]):
                ff.write(f'{t} {q}\n')

        return max(self.bcs["INFLOW"].values['t'])

    @property
    def config(self):
        self.assemble_config()
        return self._config


class Vessel():
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
        self._stenosis_coefficient = config['zero_d_element_values']['stenosis_coefficient']
        self._R = config['zero_d_element_values']['R_poiseuille']
        self._C = config['zero_d_element_values']['C']
        self._L = config['zero_d_element_values']['L']
        # get equivalent values, initilized with the values of the vessel
        self._R_eq = self._R
        self._C_eq = self._C
        self._L_eq = self._L
        # get diameter with viscosity 0.04
        self._diameter = ((128 * 0.04 * self.length) / (np.pi * self._R)) ** (1 / 4)
    
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
                    'stenosis_coefficient': self.stenosis_coefficient
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
                    'stenosis_coefficient': self.stenosis_coefficient
                },
            }


    def add_segment(self, config: dict):
        '''
        add a segment to the vessel
        '''
        # add the length
        self.length += config['vessel_length']
        # add the vessel id of the segment
        self.ids.append(config['vessel_id'])
        # add zero d element values
        self.R += config['zero_d_element_values']['R_poiseuille']
        self.C = 1 / ((1 / self._C) + (1 / config['zero_d_element_values']['C']))
        self.L += config['zero_d_element_values']['L']
        self._stenosis_coefficient += config['zero_d_element_values']['stenosis_coefficient']
        # add the segment number
        self.segs.append(get_branch_id(config)[1])

        self._update_diameter()

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
        self._update_diameter()

    @property
    def R_eq(self):
        if len(self.children) != 0:
            self._update_R_eq()
        else:
            self._R_eq = self._R
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
        self._L_eq = self._L + (1 / sum([1 / child.L_eq for child in self.children]))

    @property
    def stenosis_coefficient(self):
        return self._stenosis_coefficient
    
    @stenosis_coefficient.setter
    def stenosis_coefficient(self, new_stenosis_coefficient):
        self._stenosis_coefficient = new_stenosis_coefficient
        
    @property
    def children(self):
        return self._children
    
    @children.setter
    def children(self, new_children):
        for child in new_children:
            child.parent = self
        self._children = new_children

    @property
    def diameter(self):
        return self._diameter
    
    @diameter.setter
    def diameter(self, new_diameter):
        self._diameter = new_diameter
        self.R = 8 * 0.04 * self.length / (np.pi * (self._diameter / 2) ** 4)

    def _update_diameter(self):
        self._diameter = ((128 * 0.04 * self.length) / (np.pi * self.R)) ** (1 / 4)


class Junction():
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
        # always a normal junction

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
    

class BoundaryCondition():
    '''
    class to handle boundary conditions
    '''

    def __init__(self, config: dict):
        self.name = config['bc_name']
        self.type = config['bc_type']
        self.values = config['bc_values']
        if self.type == 'RESISTANCE':
            self._R = self.values['R']
        
        if self.type == 'RCR':
            self._Rp = self.values['Rp']
            self._Rd = self.values['Rd']
            self._C = self.values['C']
        
        if self.type == 'FLOW':
            self._Q = self.values['Q']
            self._t = self.values['t']
        
        if self.type == 'PRESSURE':
            self._P = self.values['P']
            self._t = self.values['t']
        
        if self.type == 'IMPEDANCE':
            self._Z = self.values['Z']
            self._t = self.values['t']
    
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
    
    def RCR_to_R(self):
        '''
        change the boundary condition to a resistance
        '''
        self.values = {'R': self.values['Rd'] + self.values['Rp'],
                       'Pd': self.values['Pd']}

        self.type = 'RESISTANCE'

        self._R = self.values['R']
    
    def Z_to_R(self):
        '''
        change from impedance boundary condition to resistance'''

        self.values = {'R': self.Z[0],
                       'Pd': self.values['Pd']}
        
        self.type = 'RESISTANCE'
        self._R = self.values['R']


    def R_to_RCR(self):
        '''
        change the boundary condition to RCR
        '''
        self.values = {'Rp': 0.1 * self.values['R'],
                       'Rd': 0.9 * self.values['R'],
                       'C': 1e-5,
                       'Pd': self.values['Pd']}
        
        self.type = 'RCR'

        self._Rp = self.values['Rp']
        self._Rd = self.values['Rd']
        self._C = self.values['C']
    
    # a setter so we can change the resistances in the BC easier
    @property
    def R(self):
        if self.type == 'RESISTANCE':
            return self._R
        elif self.type == 'RCR':
            return self._Rp + self._Rd

    @R.setter
    def R(self, new_R):
        if self.type == 'RESISTANCE':
            self._R = new_R
            self.values['R'] = new_R
        if self.type == 'RCR':
            self._Rp = 0.1 * new_R
            self._Rd = 0.9 * new_R
            self.values['Rp'] = self._Rp
            self.values['Rd'] = self._Rd
    
    @property
    def Rp(self):
        return self._Rp
    
    @Rp.setter
    def Rp(self, new_Rp):
        self._Rp = new_Rp
        self.values['Rp'] = new_Rp

    @property
    def Rd(self):
        return self._Rd
    
    @Rd.setter
    def Rd(self, new_Rd):
        self._Rd = new_Rd
        self.values['Rd'] = new_Rd

    @property
    def C(self):
        return self._C
    
    @C.setter
    def C(self, new_C):
        self._C = new_C
        self.values['C'] = new_C

    @property
    def Q(self):
        return self._Q
    
    @Q.setter
    def Q(self, new_Q):
        self._Q = new_Q
        self.values['Q'] = new_Q

    @property
    def P(self):
        return self._P
    
    @P.setter
    def P(self, new_P):
        self._P = new_P
        self.values['P'] = new_P

    @property
    def t(self):
        return self._t
    
    @t.setter
    def t(self, new_t):
        self._t = new_t
        self.values['t'] = new_t
    
    @property
    def Z(self):
        return self._Z
    
    @Z.setter
    def Z(self, new_Z):
        self._Z = new_Z
        self.values['Z'] = new_Z
   

class SimParams():
    '''class to handle simulation parameters'''

    def __init__(self, config: dict, threed_coupled=False):
        
        # this is probably an inefficient method but can be reimplemented later
        if 'coupled_simulation' in config.keys():
            self.coupled_simulation = config['coupled_simulation']
        if 'number_of_time_pts' in config.keys():
            self.number_of_time_pts = config["number_of_time_pts"]
        if 'output_all_cycles' in config.keys():
            self.output_all_cycles = config["output_all_cycles"]
        if 'steady_initial' in config.keys():
            self.steady_initial = config["steady_initial"]
        if 'density' in config.keys():
            self.density = config['density']
        if 'model_name' in config.keys():
            self.model_name = config['model_name']
        if 'number_of_cardiac_cycles' in config.keys():
            self.number_of_cardiac_cycles = config['number_of_cardiac_cycles']
        if 'number_of_time_pts_per_cardiac_cycle' in config.keys():
            self.number_of_time_pts_per_cardiac_cycle = config['number_of_time_pts_per_cardiac_cycle']
        if 'viscosity' in config.keys():
            self.viscosity = config['viscosity']

    @classmethod
    def from_config(cls, config):
        '''
        create a simulation parameters object from a config dict

        :param config: config dict
        '''
        if "coupled_simulation" in config.keys():
            # this is probably a 3d coupled simulation and the simulation parameters will be different
            threed_coupled = config["coupled_simulation"]
        else:
            threed_coupled = False

        return cls(config, threed_coupled=threed_coupled)
    
    def to_dict(self):
        '''
        convert the simulation parameters to a dict for zerod solver use
        '''

        return self.__dict__
    

class CouplingBlock():
    '''class to handle coupling blocks for 3d-0d coupling'''

    def __init__(self, config: dict):
        self.name = config['name']
        self.type = config['type']
        self.location = config['location']
        self.connected_block = config['connected_block']
        self.periodic = config['periodic']
        self.values = config['values']
        # to be added later
        self.surface = None
    
    @classmethod
    def from_config(cls, config):
        '''
        create a coupling block from a config dict

        :param config: config dict
        '''

        return cls(config)
    
    @classmethod
    def from_bc(cls, bc: BoundaryCondition, coupling_type='FLOW', location='inlet', periodic=False):
        '''
        create a coupling block from a boundary condition

        :param bc: boundary condition to create the coupling block from
        '''
        config = {
            'name': bc.name.replace('_', ''),
            'type': coupling_type,
            'location': location,
            'connected_block': bc.name,
            'periodic': periodic,
            'values': {
                        "t": [0.0, 1.0],
                        "Q": [1.0, 1.0]
                    }
        }

        return cls(config)
        
    
    def to_dict(self):
        '''
        convert the coupling block to a dict for zerod solver use
        '''

        return self.__dict__
    

class Chamber():
    '''class to handle chamber blocks'''

    def __init__(self, config: dict):
        self.name = config['name']
        self.type = config['type']
        self.values = config['values']
    
    @classmethod
    def from_config(cls, config):
        '''
        create a chamber from a config dict

        :param config: config dict
        '''

        return cls(config)
    
    def to_dict(self):
        '''
        convert the chamber to a dict for zerod solver use
        '''

        return self.__dict__
    

class Valve:
    '''class to handle valve blocks'''

    def __init__(self, config: dict):
        self.name = config['name']
        self.type = config['type']
        self.params = config['params']
    
    @classmethod
    def from_config(cls, config):
        '''
        create a valve from a config dict

        :param config: config dict
        '''

        return cls(config)
    
    def to_dict(self):
        '''
        convert the valve to a dict for zerod solver use
        '''

        return self.__dict__