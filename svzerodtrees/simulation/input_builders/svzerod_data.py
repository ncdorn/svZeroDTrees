from .simulation_file import SimulationFile
import numpy as np
import pandas as pd

class SvZeroDdata(SimulationFile):
    '''
    class to handle the svZeroD_data file in the simulation directory'''
    def __init__(self, path):
        '''
        initialize the svZeroD_data object'''
        super().__init__(path)

    def initialize(self):
        '''
        initialize the svZeroD_data object'''
        self.df = pd.read_csv(self.path, sep='\s+')

        self.df.rename({self.df.columns[0]: 'time'}, axis=1, inplace=True)

    def write(self):
        '''
        write the svZeroD_data file'''

        pass

    def get_result(self, block):
        '''
        get the pressure and flow from the svZeroD_data DataFrame for a given CouplingBlock
        
        :returns: time, flow, pressure'''

        if block.location == 'inlet':
            return self.df['time'], self.df[f'flow:{block.name}:{block.connected_block}'], self.df[f'pressure:{block.name}:{block.connected_block}']
        
        elif block.location == 'outlet':
            return self.df['time'], self.df[f'flow:{block.connected_block}:{block.name}'], self.df[f'pressure:{block.connected_block}:{block.name}']

    def get_flow(self, block):
        '''
        integrate the flow at the outlet over the last period
        
        :coupling_block: name of the coupling block
        :block_name: name of the block to integrate the flow over'''

        time, flow, pressure = self.get_result(block)

        # only get times and flows over the last cardiac period 1.0s
        if time.max() > 1.0:
            # unsteady simulation, get last period of the pandas dataframd
            time = time[time > time.max() - 1.0]
            # use the indices of the time to get the flow
            flow = flow[time.index]
            return np.trapz(flow, time)
        else:
            # steady simulation, only get last flow value in the pandas dataframe
            flow = flow.iloc[-1]
            return flow