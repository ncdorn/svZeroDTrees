
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
    
