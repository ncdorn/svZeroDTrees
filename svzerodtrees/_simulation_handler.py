from svzerodtrees.utils import *
from svzerodtrees.threedutils import *
import json
import pickle

class SimulationHandler:
    def __init__(self):
        '''
        initialize the simulation handler which handles threed simulation data'''

        self.n_timesteps = 0
        self.timestep = 0
        