
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

