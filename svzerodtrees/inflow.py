import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz
import scipy
import sys
from ._config_handler import BoundaryCondition

class Inflow():
    '''
    a class to handle inflow for pulmonary trees
    '''
    def __init__(self, q, t, t_per, n_periods):
        '''
        initialize the inflow object

        :param q: the flow rate
        :param t: the time points
        '''

        self.q = q
        self.t = t
        self.dt = t[1] - t[0]
        self.t_per = t_per
        self.n_periods = n_periods
        self.n_tsteps = len(t)
    

    @classmethod
    def periodic(cls, path, t_per=1.0, n_periods=1, flip_sign=False):
        '''
        create a periodic inflow
        '''
    
       #  with open(path) as ff:

        inflow = pd.read_table(path, sep='\s+', header=None)
        if len(inflow.columns) == 2:
            # we have both flowrate and time in this file
            inflow.rename(columns={0: 't', 1: 'q'}, inplace=True)

            t_per = inflow.t.max()
        else:
            # we only have the flowrate and need to make the time array
            inflow.rename(columns={0: 'q'}, inplace=True)
            inflow['t'] = np.linspace(0, t_per, len(inflow.q))
        
        if flip_sign:
            inflow['q'] = inflow['q'] * -1
        
        q = inflow.q.to_list() * n_periods
        t = []
        for n in range(n_periods):
            t += [time + t_per * n for time in inflow.t.to_list()]
        
        
        return cls(q, t, t_per, n_periods)
    
    @classmethod
    def steady(cls, q, t_per=1.0, n_periods=1, n_tsteps=2):
        '''
        create a steady inflow
        '''

        q = [q] * n_tsteps * n_periods
        t = np.linspace(0, t_per * n_periods, n_tsteps * n_periods)

        return cls(q, t.tolist(), t_per, n_periods)
    

    def rescale(self, cardiac_output=None, t_per=None, tsteps=None):
        '''
        rescale the inflow to a given cardiac output and period

        :param cardiac_output: the desired cardiac output in ml/s
        '''
        if cardiac_output is not None:
            curr_cardiac_output = trapz(self.period()[0], self.period()[1])
            scale_factor = cardiac_output / curr_cardiac_output
            self.q = [q * scale_factor for q in self.q]

        if tsteps is not None:
            self.q = scipy.signal.resample(self.q, tsteps)
            self.n_tsteps = tsteps
            if t_per is not None:
                self.t = np.linspace(0, t_per, tsteps)
            else:
                self.t = np.linspace(0, max(self.t), tsteps)

        if t_per is not None:
            self.t_per = t_per
            self.t = np.linspace(0, t_per * self.n_periods, self.n_tsteps * self.n_periods)

        

    def plot(self, ax):
        '''
        plot the inflow
        '''
        ax.plot(self.t, self.q)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('flow rate (ml/s)')
        ax.set_title('Inflow')
        ax.grid()


    def period(self):
        '''
        return one period of the inflow
        '''
        
        nt_period = self.n_tsteps // self.n_periods

        return self.q[:nt_period], self.t[:nt_period]

    def to_dict(self):
        '''
        return the inflow as a key-value pair
        '''
        if type(self.q) is np.ndarray:
            self.q = self.q.tolist()
        inflow_dict = {
            "bc_name": "INFLOW",
            "bc_type": "FLOW",
            "bc_values": {
                "Q": self.q,
                "t": self.t.tolist()
            }
        }

        return inflow_dict
    
    def to_bc(self):
        '''
        return the inflow as a BoundaryCondition object'''

        inflow_bc = BoundaryCondition(self.to_dict())

        return inflow_bc
    
    def to_df(self):
        '''
        return the inflow as a dataframe
        '''

        inflow_df = pd.DataFrame({'q': self.q, 't': self.t})

        return inflow_df
    
    def save(self, path):
        '''
        save the inflow to a file
        '''

        inflow_df = self.to_df()

        inflow_df.to_csv(path, index=False, sep='\t')
        
        print(f'Inflow saved to {path}')
        


if __name__=='__main__':

    cardiac_output = float(sys.argv[1])
    t_per = float(sys.argv[2])
    patient = sys.argv[3]

    inflow = Inflow.periodic(path='../tests/cases/olufsen_impedance/inflow.flow')

    inflow.rescale(cardiac_output=cardiac_output, t_per=t_per)

    path = f'../../threed_models/flow-files/inflow_{patient}.csv'

    inflow.save(path)

    fig, ax = plt.subplots()
    inflow.plot(ax)
    plt.show()



    