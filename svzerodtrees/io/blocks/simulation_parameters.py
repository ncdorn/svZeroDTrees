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
    
