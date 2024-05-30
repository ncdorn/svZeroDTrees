from svzerodtrees.utils import *
from svzerodtrees.threedutils import *
import json
import pickle


class ClosedLoopSimulation():
    def __init__(self, cl_config, pa_config, dim=0):
        '''
        initialize the closed loop handler object

        :param closed_loop_config: the closed loop config dict
        :param pa_config: the pulmonary artery config dict or the 3d coupled BCs
        :param dim: the dimension of the simulation (0, 3)
        '''

        self.cl_config = cl_config
        self.pa_config = pa_config
        self._config = None # TODO: implement combination of config handlers

        pass
    

    


