import os
from abc import ABC, abstractmethod

class SimulationFile(ABC):
    '''
    abstract super class for simulation files'''

    def __init__(self, path):
        '''
        initialize the simulation file'''

        self.path = os.path.abspath(path)

        self.directory = os.path.dirname(path)

        self.filename = os.path.basename(path)

        if os.path.exists(path):
            self.initialize()
            self.is_written = True
        else:
            self.is_written = False

    @abstractmethod
    def initialize(self):
        '''
        initialize the file from some pre-existing file'''
        raise NotImplementedError

    @abstractmethod
    def write(self):
        '''
        write the file'''
        raise NotImplementedError
