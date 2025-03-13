from svzerodtrees.utils import *
import numpy as np



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

    def convert_to_cm(self):
        '''
        convert the vessel parameters to cm
        multiply R by 1000
        divide C by 10000
        multiply L by 10
        '''

        self.R *= 1000

        self.C /= 10000

        self.L *= 10



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
            if 'tree' in self.values.keys():
                self.tree = self.values['tree']
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
        self.surface = config['surface']

        # simulation result
        if 'result' in config.keys():
            self.result = config['result']
        else:
            self.result = {}
    
    @classmethod
    def from_config(cls, config):
        '''
        create a coupling block from a config dict

        :param config: config dict
        '''

        return cls(config)
    
    @classmethod
    def from_bc(cls, bc: BoundaryCondition, coupling_type='FLOW', location='inlet', periodic=False, surface=None):
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
                    },
            'surface': surface
        }

        return cls(config)
        
    def add_result(self, svzerod_data):
        '''
        :param result: svZeroDdata class instance
        '''

        # get the result array which corresponds to the coupling block name
        time, flow, pressure = svzerod_data.get_result(self)

        self.result = {}
        self.result['time'] = time
        self.result['flow'] = flow
        self.result['pressure'] = pressure




    def to_dict(self, with_result=False):
        '''
        convert the coupling block to a dict for zerod solver use
        '''

        self_dict = self.__dict__

        # don't want to include the result arrays in the output dict
        if 'result' in self_dict.keys() and not with_result:
            del self_dict['result']

        return self_dict
    

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

