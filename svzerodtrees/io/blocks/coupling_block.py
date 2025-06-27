from .boundary_condition import BoundaryCondition
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
    