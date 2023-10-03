import matplotlib.pyplot as plt
from svzerodtrees.utils import *
import pickle
import json


class PAPlotter:
    '''
    plotter class for pulmonary tree analysis
    '''

    def __init__(self, config, result, fig_dir: str):
        self.config = config
        self.result = result
        self.fig_dir = fig_dir
        self.vessels = config['vessels']

        # root of the vessel map for tree structure
        self.root = None

    @classmethod
    def from_file(cls, config_file: str, result_file: str, fig_dir: str):
        '''
        class method to generate the results handler with vessel and config information

        :param config_file: path to 0d config file
        :param result_file: path to 0d result file
        :param fig_dir: path to directory to save figures

        :return: ResultHandler instance
        '''
        print('loading in config file')
        if config_file.endswith('.json'):
            with open(config_file) as ff:
                config = json.load(ff)
        else:
            with open(config_file, 'rb') as ff:
                config = pickle.load(ff)
        
        print('config file loaded!')   

        with open(result_file) as ff:
            result = json.load(ff)

        return cls(config, result, fig_dir)
    
    def remove_trees(self):
        '''
        remove the structured tree objects from the config and dump to a json file
        '''

        config = self.config

        for vessel in config['vessels']:
            if 'tree' in vessel:
                del vessel['tree']

        with open('config_no_trees.json', 'w') as ff:
            json.dump(config, ff)

    def make_figure(self, result: dict, fig_title: str, filename: str, sharex=False, sharey=False):
        '''
        make a figure with the given list of subfigures

        :param result: dict of lists of result arrays, with one for each subplot
        :param fig_title: title of figure
        :param filename: filename to save figure
        :param sharex: bool to share xlabel
        :param sharey: bool to share ylabel
        '''

        # initialize the figure
        # fig = plt.figure()

        fig, axs = plt.subplots(1, len(result), sharex=sharex, sharey=sharey)

        # make an array so indexing still works
        if len(result) == 1:
            axs = [axs]

        # intialize the subplot axes depending on the length of result dict
        # axs = fig.subplots(1, len(result), sharex=sharex, sharey=sharey)

        # loop through reult dict and make subplots
        for i, (title, info) in enumerate(result.items()):
            
            # if only one result, no subplot title
            if len(result) == 1:
                title=None
            
            # type of plot (scatter, histogram, bar)
            plot_type = info['type']

            # data to plot
            data = info['data']

            # if the result dict specifies labels, get the labels
            if 'labels' in info:
                label = True
                labels = info['labels']
            else:
                label = False
            
            # make a scatterplot
            if plot_type == 'scatter':
                self.make_scatterplot(axs[i], data[0], data[1], title, label, labels['x'], labels['y'])
            
            # make a histogram
            elif plot_type == 'hist':
                self.make_histogram(axs[i], data, label, title, labels['x'], labels['y'])
            
            # make a barplot
            elif plot_type == 'bar':
                self.make_barplot(axs[i], data[0], data[1], label, title, labels['y'], labels['x'])
            
            # if the plot type is not recognized, raise an error
            else:
                raise ValueError('Plot type not recognized')

        # set the figure title
        fig.suptitle(fig_title)

        plt.tight_layout()
        plt.savefig(self.fig_dir + '/' + filename)
    
    def make_subfigure(self, result, title, ylabel, xlabel):
        
        pass

    def make_scatterplot(self, ax, result_x, result_y, title=None, labels=False, ylabel=None, xlabel=None):
        '''
        make a pyplot scatterplot given a list

        :param ax: pyplot axis
        :param result_x: list of x values
        :param result_y: list of y values
        :param title: title of plot
        :param ylabel: y axis label
        :param xlabel: x axis label
        '''

        # initialize the scatterplot
        plot = ax.scatter(result_x, result_y)
        
        # set the title
        ax.set_title(title)

        # if labels, get x and y labels
        if labels:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        # return the plot object
        return plot
    
    def make_histogram(self, ax, result, labels=False, title=None, ylabel=None, xlabel=None):
        '''
        make a histogram given a list, with auto binning

        :param ax: pyplot axis
        :param result: list of values
        :param title: title of plot
        :param ylabel: y axis label
        :param xlabel: x axis label
        '''

        plot = ax.hist(result, bins=20)

        if labels:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
        
        return plot
    
    def make_barplot(self, ax, x, result, labels=False, title=None, ylabel=None, xlabel=None):
        '''
        make a barplot given a list

        :param ax: pyplot axis
        :param x: list of x values (bar labels)
        :param result: list of values
        :param title: title of plot
        :param ylabel: y axis label
        :param xlabel: x axis label
        '''

        ax.bar(x, result)

        if labels:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
        
        # return plot

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


    def plot_outlet_flow_histogram(self):
        outlet_vessels, outlet_d = find_outlets(self.config)

        outlet_flows = []
        for vessel in outlet_vessels:
            outlet_flows.append(self.result[str(vessel)]["q_out"]["final"])

        result_dict = {
            'flow_adaptation':
            {
                'type': 'hist',
                'data': outlet_flows,
                'labels': {
                    'x': 'flowrate',
                    'y': 'number of outlet vessels'
                }
            }
        }

        self.make_figure(result_dict, 'Histogram of outlet flowrate', 'outlet_flowrate.png')

    def plot_flow_adaptation(self, vessels, filename='flow_adaptation.png'):
        '''
        plot a bar chart of the flow adaptation in the large vessels

        :param vessels: list of vessels


        '''
        if vessels == 'all':
            vessels = self.result.keys()
        if vessels == 'outlets':
            outlet_vessels, outlet_d = find_outlets(self.config)
            vessels = [str(vessel) for vessel in outlet_vessels]

        percents_adapt = []
        for vessel in vessels:
            postop_q = self.result[vessel]["q_out"]["postop"]
            adapted_q = self.result[vessel]["q_out"]["final"]

            percent_adapt = (adapted_q - postop_q) / postop_q * 100
            if vessels == 'all':
                # set some threshold to reduce the numebr of bars
                if abs(percent_adapt) > 100:
                    percents_adapt.append(percent_adapt)
            else:
                percents_adapt.append(percent_adapt)
        
        result_dict = {
            'flow_adaptation':
            {
                'type': 'bar',
                'data': [vessels, percents_adapt],
                'labels': {
                    'x': 'vessel',
                    'y': '% flow adaptation'
                }
            }
        }
        
        self.make_figure(result_dict, '% flow adaptation in vessels', filename, sharex=False, sharey=False)


    def label_lpa_or_rpa(self):
        '''
        label each vessel as either lpa or rpa
        
        '''

        pass

    def build_tree_map(self):
        '''
        build a map of the tree structure

        '''

        class Vessel:
            def __init__(self, id: int):
                self.id = id
                # optional name, such as mpa, lpa, rpa
                self.label= None
                # list of child vessels
                self.children = []
                # path to root
                self.path = []
        
        vessel_map = {}

        # initialize the vessel map (dict of vessels)
        for vessel in self.vessels:
            vessel_map[vessel['vessel_id']] = Vessel(vessel['vessel_id'])
        
        # loop through junctions and add children to parent vessels
        for junction in self.config['junctions']:
            inlets = junction['inlet_vessels']
            
            if len(inlets) > 1:
                raise Exception("there is more than one inlet to this junction")
            
            outlets = junction['outlet_vessels']

            parent_vessel = vessel_map[inlets[0]]

            for outlet in outlets:
                child_vessel = vessel_map[outlet]
                parent_vessel.children.append(child_vessel)
            
        # find the root vessel
        self.root = None
        for vessel in vessel_map.values():
            if not any(vessel in child_vessel.children for child_vessel in vessel_map.values()):
                self.root = vessel
        
        # label the mpa, rpa and lpa
        self.root.label = 'mpa'
        self.root.children[0].label = 'rpa'
        self.root.children[1].label = 'lpa'

        self.find_vessel_paths()


    
    def find_vessel_paths(self):
        '''
        find the path from the root to each vessel
        '''

        # helper function for depth-first search
        def dfs(vessel, path):
            if vessel is None:
                return
            
            # add current vessel to the path
            path.append(vessel.id)

            vessel.path = path.copy()

            for child in vessel.children:
                dfs(child, path.copy())

        dfs(self.root, [])




    
    

