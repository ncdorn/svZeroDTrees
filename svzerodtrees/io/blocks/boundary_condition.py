
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
   
