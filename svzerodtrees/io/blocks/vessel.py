import numpy as np
from ..utils import get_branch_id


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
        self._diameter = self._calculate_diameter(self._R)
    
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

    # @property
    # def L_eq(self):
    #     if len(self.children) != 0:
    #         self._update_L_eq()
    #     return self._L_eq

    # def _update_L_eq(self):
    #     self._L_eq = self._L + (1 / sum([1 / child.L_eq for child in self.children]))

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
        self._diameter = self._calculate_diameter(self.R)

    def _calculate_diameter(self, resistance):
        # avoid division by zero when the resistance is zero
        if resistance == 0:
            return np.inf
        return ((128 * 0.04 * self.length) / (np.pi * resistance)) ** (1 / 4)
