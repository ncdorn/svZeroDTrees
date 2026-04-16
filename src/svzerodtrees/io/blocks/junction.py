import numpy as np
from .vessel import Vessel

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
   