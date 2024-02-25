import matplotlib.pyplot as plt
from svzerodtrees.utils import *
import pickle
import json


class PA_Plotter:

    def __init__(self, config, result, fig_dir: str):
        self.config = config
        self.result = result
        self.fig_dir = fig_dir

    @classmethod
    def from_file(cls, config_file: str, result_file: str, fig_dir: str):
        '''
        class method to generate the results handler with vessel and config information

        :param config_file: 0d config file
        :param result_file: 0d result file
        :param fig_dir: path to directory to save figures

        :return: ResultHandler instance
        '''
        with open(config_file) as ff:
            config = pickle.load(ff)
        
        print(config)

        with open(result_file) as ff:
            result = json.load(ff)

        return cls(config, result, fig_dir)

    def plot_distal_wss(self):
        '''
        plot the wss at the outlets of the model

        :param config: config dict
        '''


        plt.hist(get_outlet_data(self.config, self.result, "wss"), bins=20)

        plt.title('Histogram of PA distal vessel WSS')

        plt.tight_layout()
        plt.savefig(str(self.fig_dir + '/') + 'distal_wss' + '.png')


    def plot_micro_wss(trees, max_d=.015):
        '''
        plot the wss in the distal vessels of the tree, below a certain diameter threshold

        :param trees: list of StructuredTreeOutlet instances
        :param max_d: maximum diameter threshold for plotting distal wss
        '''

        wss_list = []

        for tree in trees:
            
            def get_distal_wss(vessel):
                if vessel:
                    if vessel.d < max_d:
                        wss_list.append(vessel.wss)
                    get_distal_wss(vessel.left)
                    get_distal_wss(vessel.right)

            get_distal_wss(tree.root)

        plt.hist(wss_list, bins=100)
        plt.title('Histogram of wss for vessels < ' + str(max_d * 1000) + ' um')


    def plot_outlet_flow_histogram():

        pass

    def plot_flow_adaptation():

        pass
    

