import matplotlib.pyplot as plt
import numpy as np
from svzerodtrees.utils import *
import pandas as pd
import pickle
import json
import os


class PAanalyzer:
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

        # maps of vessels and branches
        self.vessel_map = {}
        self.branch_map = {}
        self.vessel_branch_map = {}

        # initialize indices of mpa, rpa, lpa
        self.mpa = None
        self.rpa = None
        self.lpa = None

        # build tree map, organize by branches, and map vessels to branches
        self.map_vessels_to_branches()
        self.build_tree_map()
        self.map_branches_to_vessels()
        

    @classmethod
    def from_files(cls, config_file: str, result_file: str, fig_dir: str):
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


    def make_figure(self, plot_config: dict, fig_title: str, filename: str, sharex=False, sharey=False):
        '''
        make a figure with the given list of subfigures

        :param plot_config: dict of lists of result arrays, with one for each subplot
        :param fig_title: title of figure
        :param filename: filename to save figure
        :param sharex: bool to share xlabel
        :param sharey: bool to share ylabel
        '''

        # initialize the figure
        # fig = plt.figure()

        fig, axs = plt.subplots(1, len(plot_config), sharex=sharex, sharey=sharey)

        # make an array so indexing still works
        if len(plot_config) == 1:
            axs = [axs]

        # intialize the subplot axes depending on the length of result dict
        # axs = fig.subplots(1, len(result), sharex=sharex, sharey=sharey)

        # loop through reult dict and make subplots
        for i, (title, info) in enumerate(plot_config.items()):
            
            # if only one result, no subplot title
            if len(plot_config) == 1:
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

        :param trees: list of StructuredTree instances
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
            outlet_flows.append(self.result[str(vessel)]["q_out"]['adapted'])

        plot_config = {
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

        self.make_figure(plot_config, 'Histogram of outlet flowrate', 'outlet_flowrate.png')


    def plot_lpa_rpa_diff(self):
        '''
        plot the difference in flowrate, pressure and wss between the LPA and RPA
        '''

        fig = plt.figure(figsize=(10, 4))
        ax = fig.subplots(1, 4)

        # plot the changes in q, p, wss in subfigures
        # self.plot_changes_subfig([self.lpa.branch, self.rpa.branch],
        #                          'q_out',
        #                          title='outlet flowrate',
        #                          ylabel='q (cm3/s)',
        #                          ax=ax[0])
        
        self.plot_flow_split(ax[0])
        
        print([self.lpa.branch, self.rpa.branch])

        self.plot_changes_subfig([self.lpa.branch, self.rpa.branch],
                                 'p_in',
                                 title='pressure in',
                                 ylabel='p (mmHg)',
                                 ax=ax[1])

        self.plot_changes_subfig([self.lpa.branch, self.rpa.branch],
                                 'p_out',
                                 title='pressure out',
                                 ylabel='p (mmHg)',
                                 ax=ax[2])
        
        # set pressure ylims equal to each other
        ax[2].set_ylim(ax[1].get_ylim())
        
        self.plot_changes_subfig([self.lpa.branch, self.rpa.branch],
                                 'wss',
                                 title='wall shear stress',
                                 ylabel='wss (dynes/cm2)',
                                 ax=ax[3])

        plt.suptitle('Hemodynamic changes in LPA and RPA')
        plt.tight_layout()
        plt.savefig(self.fig_dir + '/lpa_rpa_diff_w_bar.png')


    def plot_flow_split(self, ax):
        '''
        plot the flow split between the LPA and RPA as a stacked bar graph

        :param ax: pyplot axis
        '''
        
        timesteps = ['preop', 'postop', 'adapted']

        # get flow splits
        preop_q = self.get_result([self.lpa.branch, self.rpa.branch], 'q_out', 'preop', type='np')
        postop_q = self.get_result([self.lpa.branch, self.rpa.branch], 'q_out', 'postop', type='np')
        final_q = self.get_result([self.lpa.branch, self.rpa.branch], 'q_out', 'adapted', type='np')

        preop_split = preop_q / sum(preop_q)
        postop_split = postop_q / sum(postop_q)
        final_split = final_q / sum(final_q)

        percent = {'lpa': [preop_split[0] * 100, postop_split[0] * 100, final_split[0] * 100],
                     'rpa': [preop_split[1] * 100, postop_split[1] * 100, final_split[1] * 100]}
        
        q = {'lpa': [preop_q[0], postop_q[0], final_q[0]],
             'rpa': [preop_q[1], postop_q[1], final_q[1]]}

        # plot the stacked bar graph
        bottom = np.zeros(len(timesteps))
        colors = {'lpa': 'tomato', 'rpa': 'cornflowerblue'}
        for vessel, values in q.items():
            ax.bar(timesteps, values, label=vessel, bottom=bottom, color=colors[vessel])
            for value in values:
                ax.text(timesteps[values.index(value)], value / 2 + bottom[values.index(value)], str(int(percent[vessel][values.index(value)])) + '%', ha='center', va='center')
            bottom += values

        ax.set_ylim([0, 1.2 * (final_q[0] + final_q[1])])
        ax.set_title('flow split')
        ax.set_ylabel('mean flow (cm3/s)')
        ax.legend(ncols=2)


    def plot_changes_subfig(self, branches, qoi, title, ylabel, xlabel=None, ax=None):
        '''
        plot the changes in the LPA and RPA flow, pressure and wss as a grouped bar graph

        :param summary_values: summarized results dict for a given QOI, from postop.summarize_results
        :param branches: list of str containing the branches to plot
        :param qoi: str containing the data name to plot
        :param title: figure title
        :param ylabel: figure ylabel
        :param xlabel: figure xlabel
        :param ax: figure ax object
        :param condition: experimental condition name

        '''

        timesteps = ['preop', 'postop', 'adapted']

        bar_width = 1 / (len(branches) + 1)

        x = np.arange(len(timesteps))

        bar_width = 0.25
        shift = 0

        # Plotting the grouped bar chart
        colors = ['tomato', 'cornflowerblue']
        color = 0
        for branch, qois in self.result.items():
            if int(branch) in branches:
                values = [qois[qoi][timestep] for timestep in timesteps]
                offset = bar_width * shift
                ax.bar(x + offset, values, bar_width, label=branch, color=colors[color])
                shift += 1
                color += 1
        
        # set x and y axis ranges
        max_y_val = max([self.result[str(branch)][qoi][timestep] for branch in branches for timestep in timesteps])
        ax.set_ylim((0, 1.2 * max_y_val))

        # Set labels, title, and legend
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks([0, 1, 2], timesteps)
        ax.legend(['lpa', 'rpa'], ncols=2)


    def plot_mpa_pressure(self):
        '''
        plot the inlet and outlet pressure in the mpa for the preop, postop and final cases
        '''

        # initialize timesteps
        timesteps = ['preop', 'postop', 'adapted']

        # get the inlet pressure
        p_in = [self.result[str(self.mpa.branch)]['p_in'][timestep] for timestep in timesteps]

        # get the outlet pressure
        p_out = [self.result[str(self.mpa.branch)]['p_out'][timestep] for timestep in timesteps]

        # make a barplot of the inlet and outlet pressure
        fig, ax = plt.subplots(1, 2, sharey=True)

        p_i = ax[0].bar(timesteps, p_in)
        ax[0].bar_label(p_i, label_type='center')
        ax[0].set_title('MPA inlet pressure')

        p_o = ax[1].bar(timesteps, p_out)
        ax[1].bar_label(p_o, label_type='center')
        ax[1].set_title('MPA outlet pressure')

        ax[0].set_ylabel('pressure (mmHg)')

        plt.tight_layout()

        plt.savefig(self.fig_dir + '/mpa_pressure.png')


    def plot_lpa_rpa_adaptation(self):
        '''
        make a barplot of the flow adaptation in the lpa and rpa
        '''

        percent_adapt = self.get_qoi_adaptation(['lpa', 'rpa'], 'q_out')

        fig, ax = plt.subplots()

        ax.bar(['lpa', 'rpa'], percent_adapt)

        ax.set_title('flow adaptation in LPA and RPA')
        ax.set_ylabel('percent flow adaptation (%)')

        plt.tight_layout()

        plt.savefig(self.fig_dir + '/lpa_rpa_flow_adaptation.png')


    def get_qoi_adaptation(self, vessels, qoi: str,  threshold=0):
        '''
        get a list of the percent flow adapatation for a given list of vessels

        :param vessels: list of branches
        :param qoi: quantity of interest, either 'p_in', 'p_out', 'q_in', 'q_out', or 'wss'
        :param threshold: threshold for flow adaptation
        '''

        if 'lpa' in vessels:
            vessels[vessels.index('lpa')] = self.lpa.branch
        if 'rpa' in vessels:
            vessels[vessels.index('rpa')] = self.rpa.branch

        postop = self.get_result(vessels, qoi, 'postop', type='np')
        adapted = self.get_result(vessels, qoi, 'adapted', type='np')

        percent_adapt = np.subtract(adapted, postop) / postop * 100

        percent_adapt = percent_adapt[abs(percent_adapt) >= threshold]

        print(percent_adapt)
        
        return percent_adapt


    def scatter_qoi_vs_distance(self, branches, qoi, filename=None):
        '''
        scatter plot of vessel resistance vs distance from MPA

        :param branches: list of branches to plot
        :param filename: filename to save figure
        '''

        # get branches as list of str and vessels as list
        branches, vessels = self.get_vessels(branches)
        
        # handle qoi names such as "q_out postop"
        if ' ' in qoi:
            qoi, timestep = qoi.split(' ')
        else:
            timestep = ''

        print(branches, vessels)
        # assess qoi and make list
        qois = self.get_qoi(qoi, vessels, branches, timestep=timestep)

        distances = self.get_distance_from_mpa(vessels)

        # color the vessels red or blue depending on if they are LPA or RPA
        rpa_distances, lpa_distances = self.sort_into_rpa_lpa(vessels, distances)
        rpa_qois, lpa_qois = self.sort_into_rpa_lpa(vessels, qois)
        
        fig, ax = plt.subplots()
        ax.scatter(rpa_distances, rpa_qois, s=100, c='red')
        ax.scatter(lpa_distances, lpa_qois, s=100, c='blue')

        # add horizontal line at zero
        # ax.axhline(0, color='red', linestyle='--', label='0% adaptation')

        # add grid
        ax.grid(True)

        ax.set_xlabel('distance from MPA (cm)')
        ax.set_ylabel(qoi)

        # symmetric log y axis
        ax.set_yscale('log')

        ax.legend(['RPA', 'LPA'])
        ax.set_title(qoi + ' ' + timestep + ' vs distance from MPA')

        if filename is None:
            filename = qoi + '_' + timestep + '_vs_distance.png'
        
        plt.savefig(str(self.fig_dir + '/') + filename)


    def plot_3d_scatter(self, branches, qoi1, qoi2, qoi3, filename=None):
        '''
        plot a 3d scatter plot of tree qois for a given list of branches

        :param branches: list of branches to plot
        :param qoi1: quantity of interest 1 to be plotted on the x axis
        :param qoi2: quantity of interest 2 to be plotted on the y axis
        :param qoi3: quantity of interest 3 to be plotted on the z axis
        :param filename: filename to save figure
        '''

        # get branches as list of str and vessels as list
        branches, vessels = self.get_vessels(branches)

        # get qois
        qois = []
        for qoi in [qoi1, qoi2, qoi3]:
            qois.append(self.get_qoi(qoi, vessels, branches))
        

    def scatter_qoi_vs_qoi(self, branches, qoi1, qoi2, yscale = None, filename=None):
        '''
        plot a scatter plot of qoi1 vs qoi2 for a given list of branches

        :param branches: list of branches to plot
        :param qoi1: quantity of interest 1 to be plotted on the x axis
        :param qoi2: quantity of interest 2 to be plotted on the y axis
        :param filename: filename to save figure
        '''

        # get branches as list of str and vessels as list
        branches, vessels = self.get_vessels(branches)

        # get qois
        rpa_qois = []
        lpa_qois = []
        for qoi in [qoi1, qoi2]:
            all_qois = self.get_qoi(qoi, vessels, branches)
            rpa_qoi, lpa_qoi = self.sort_into_rpa_lpa(vessels, all_qois)
            rpa_qois.append(rpa_qoi)
            lpa_qois.append(lpa_qoi)
        
        fig, ax = plt.subplots()
        ax.scatter(rpa_qois[0], rpa_qois[1], s=100, c='red')
        ax.scatter(lpa_qois[0], lpa_qois[1], s=100, c='blue')

        # add grid
        ax.grid(True)

        ax.set_xlabel(qoi1)
        ax.set_ylabel(qoi2)

        if yscale is not None:
            ax.set_yscale(yscale)

        ax.legend(['RPA', 'LPA'])

        ax.set_title(qoi1 + ' vs ' + qoi2)

        if filename is None:
            filename = qoi1 + '_vs_' + qoi2 + '.png'
        
        plt.savefig(str(self.fig_dir + '/') + filename)


    def plot_flow_adaptation(self, vessel_ids, filename='flow_adaptation.png', threshold=0.0):
        '''
        plot a bar chart of the flow adaptation in the large vessels

        :param vessels: list of vessel ids
        :param filename: filename to save figure
        :param threshold: threshold for flow adaptation
        '''
        if vessel_ids == 'all':
            vessel_ids = list(self.result.keys())
        if vessel_ids == 'outlets':
            outlet_vessels, outlet_d = find_outlets(self.config)
            vessel_ids = [str(vessel) for vessel in outlet_vessels]

        if vessel_ids == [self.lpa.branch, self.rpa.branch]:
            vessel_ids = ['lpa', 'rpa']
        
        vessels = [self.vessel_map[int(vessel)] for vessel in vessel_ids]

        percents_adapt = self.get_qoi_adaptation(vessel_ids, 'q_out', threshold=threshold)
        
        rpa_qoi, lpa_qoi = self.sort_into_rpa_lpa(vessels, percents_adapt)

        print(f'lpa: {self.lpa.branch} / rpa: {self.rpa.branch}')

        lpa_qoi = pd.Series(lpa_qoi)
        rpa_qoi = pd.Series(rpa_qoi)

        plt.figure()
        plt.bar(lpa_qoi.index, lpa_qoi, label='LPA', color='tomato')
        plt.bar(rpa_qoi.index, rpa_qoi, label='RPA', color='cornflowerblue')
        plt.ylabel(f'% change in flow')
        plt.xticks([])
        plt.legend()
        plt.title('Outlet adaptation')
        plt.savefig(os.path.join(self.fig_dir, filename))



    def scatter_qoi_adaptation_distance(self, branches, qoi: str, filename= 'adaptation_scatter.png', threshold=0.0):
        '''
        create a scatterplot of flow adaptation vs distance from MPA where the points are colored red and blue for
        LPA vs RPA

        '''

        branches, vessels = self.get_vessels(branches)
        percents_adapt = self.get_qoi_adaptation(branches, qoi, threshold=threshold)
        distances = self.get_distance_from_mpa(vessels)

        # color the vessels red or blue depending on if they are LPA or RPA
        rpa_distances, lpa_distances = self.sort_into_rpa_lpa(vessels, distances)
        rpa_percents_adapt, lpa_percents_adapt = self.sort_into_rpa_lpa(vessels, percents_adapt)
        
        fig, ax = plt.subplots()
        ax.scatter(rpa_distances, rpa_percents_adapt, s=100, c='red')
        ax.scatter(lpa_distances, lpa_percents_adapt, s=100, c='blue')

        # add horizontal line at zero
        # ax.axhline(0, color='red', linestyle='--', label='0% adaptation')

        # add grid
        ax.grid(True)

        ax.set_xlabel('distance from MPA (cm)')
        ax.set_ylabel('% ' + qoi + ' adaptation')

        # symmetric log y axis
        # ax.set_yscale('symlog')

        ax.legend(['RPA', 'LPA'])
        ax.set_title(qoi + ' adaptation vs distance from MPA')

        plt.savefig(str(self.fig_dir + '/') + qoi + '_' + filename)


    def label_lpa_or_rpa(self):
        '''
        label each vessel as either lpa or rpa
        
        '''

        pass
    

    def map_vessels_to_branches(self):
        '''
        map each vessel id to a branch id
        '''

        for vessel in self.vessels:
            self.vessel_branch_map[vessel['vessel_id']] = get_branch_id(vessel)[0]


    def build_tree_map(self):
        '''
        Build a recursive map of the tree structure using self.Vessel objects.
        Useful for dfs and vessel distance analysis

        :return self.vessel_map: a dict where the keys are branch ids and the values are Vessel objects
        '''

        # initialize the vessel map (dict of branches)
        for vessel_config in self.vessels:
            branch = get_branch_id(vessel_config)[0]
            if branch not in self.vessel_map.keys():
                self.vessel_map[branch] = self.Vessel(vessel_config)
            else: 
                self.vessel_map[branch] = self.combine_vessels(self.vessel_map[branch], self.Vessel(vessel_config))
            
            self.vessel_map[branch].d = get_branch_d(self.config['vessels'], branch)
            # map vessel id to branch
        
        # loop through junctions and add children to parent vessels
        for junction in self.config['junctions']:
            # map inlet vessel id to branches
            inlets = [self.vessel_branch_map[inlet] for inlet in junction['inlet_vessels']]
            
            if len(inlets) > 1:
                raise Exception("there is more than one inlet to this junction")
            
            # map outlet vessel id to branches
            outlets = [self.vessel_branch_map[outlet] for outlet in junction['outlet_vessels']]

            # if the inlet and outlet branches are the same, skip
            if inlets == outlets:
                continue

            parent_vessel = self.vessel_map[inlets[0]]

            # connect all the vessel instances
            for outlet in outlets:
                child_vessel = self.vessel_map[outlet]
                child_vessel.parent = parent_vessel
                parent_vessel.children.append(child_vessel)

        # find the root vessel
        self.root = None
        for vessel in self.vessel_map.values():
            if not any(vessel in child_vessel.children for child_vessel in self.vessel_map.values()):
                self.root = vessel
        
        # organize the children in numerical order
        # we assume that the branches are sorted alphabetically, and therefore the lpa comes first.
        self.root.children.sort(key=lambda x: x.branch)


        # label the mpa, rpa and lpa
        self.root.label = 'mpa'
        self.root.children[0].label = 'lpa'
        self.root.children[1].label = 'rpa'
        
        # add these in incase we index using the mpa, lpa, rpa strings
        self.mpa = self.root
        self.lpa = self.root.children[0]
        self.rpa = self.root.children[1]

        # modify the self.result array to include the mpa, rpa, lpa with numerical ids
        self.result[str(self.mpa.branch)] = self.result.pop('mpa')
        self.result[str(self.rpa.branch)] = self.result.pop('rpa')
        self.result[str(self.lpa.branch)] = self.result.pop('lpa')

        keys = list(self.result.keys())
        keys.sort(key=int)

        self.result = {key: self.result[key] for key in keys}

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
            path.append(vessel.branch)

            vessel.path = path.copy()
            
            vessel.gen = len(path) - 1

            for child in vessel.children:
                dfs(child, path.copy())

        dfs(self.root, [])


    def get_distance_from_mpa(self, vessels):
        '''
        get the distance from the mpa for a given list of vessel ids

        :param vessels: vessels to get distance for
        '''
        distances = []
        for vessel in vessels:
            if vessel.label == 'mpa':
                distances.append(0)
            else:
                distances.append(vessel.length + self.get_distance_from_mpa([vessel.parent])[0])
            
        return distances
    

    def sort_into_rpa_lpa(self, vessels, data):
        rpa_data = []
        lpa_data = []
        for vessel, datum in zip(vessels, data):
            if self.rpa.branch in vessel.path:
                rpa_data.append(datum)
            elif self.lpa.branch in vessel.path:
                lpa_data.append(datum)
        
        return rpa_data, lpa_data


    def map_branches_to_vessels(self):
        '''
        organize the vessel map by branch
        '''

        branches = list(self.result.keys())

        for branch_id in branches:
            # create a branch map where each branch id keeps a list of vessels in that branch if multiple segments have been made
            self.branch_map[branch_id] = [vessel for vessel in self.vessel_map.values() if vessel.branch == branch_id]
            
            # sort the list from least to most distal vessels
            self.branch_map[branch_id].sort(key=lambda x: x.gen)


    def get_result(self, branches, qoi: str, time: str, type='np'):
        '''
        get a result for a given qoi for a list of branches
        
        :param branches: list of branch ids
        :param qoi: quantity of interest
        :param type: datatype of result array (np or list)
        '''

        if qoi not in ['p_in', 'p_out', 'q_in', 'q_out', 'wss']:
            raise ValueError('qoi not recognized')

        if type not in ['np', 'list']:
            raise ValueError('type not recognized')
        
        if time not in ['preop', 'postop', 'adapted']:
            raise ValueError('time not recognized')
        
        values = []
        for branch in branches:
            values.append(self.result[str(branch)][qoi][time])
        
        if type == 'np':
            return np.array(values)
        else:
            return values


    def get_qoi(self, qoi: str, vessels, branches=None, timestep='adapted'):
        ''' get a list of qois depending on a string input name
        
        :param qoi: string describing the quantity of interest
        :param vessels: list of vessels to get qoi for
        :param branches: optional param describing list of branches to get qoi for
        :param timestep: optional param describing timestep to get qoi for
        '''

        # split the input qoi if we are looking for adaptation
        if ' ' in qoi:
            qoi, adapt = qoi.split(' ')
            adapt = True
        else:
            adapt = False

        # current supported qois: 
        # [resistance, diameter, aspect (d / l), generation, 
        #  tree_resistance, [p_in, p_out, q_in, q_out, wss] with adaptation, 
        #  length, distance, Re, dist_index]

        if qoi == 'resistance':
            qois = [vessel.zero_d_element_values['R_poiseuille'] for vessel in vessels]
        elif qoi == 'diameter':
            qois = [vessel.d for vessel in vessels]
        elif qoi == 'aspect':
            qois = [vessel.d / vessel.length for vessel in vessels]
        elif qoi == 'gen':
            qois = [vessel.gen for vessel in vessels]
        elif qoi == 'tree_resistance':
            qois = []
            for vessel in self.config['vessels']:
                if str(get_branch_id(vessel)[0]) in branches:
                    if 'tree' in vessel.keys():
                        qois.append(vessel["tree"].root.R_eq)
        elif qoi in ['p_in', 'p_out', 'q_in', 'q_out', 'wss']:
            if adapt:
                qois = self.get_qoi_adaptation(branches, qoi, threshold=0)
            else:
                qois = self.get_result(branches, qoi, timestep, type='np')
        elif qoi == 'length':
            qois = [vessel.length for vessel in vessels]
        elif qoi== 'distance':
            qois = self.get_distance_from_mpa(vessels)
        elif qoi == 'Re':
            qois = []
            for vessel in vessels:
                q = self.get_result([vessel.branch], 'q_in', timestep, type='np')
                r = vessel.d / 2
                print(q, r)
                rho = self.config['simulation_parameters']['density']
                mu = self.config['simulation_parameters']['viscosity']
                qois.append(abs(2 * rho * q / (np.pi * r * mu)))
        elif qoi == 'dist_index':
            qois = []
            for vessel in vessels:
                dist = self.get_distance_from_mpa([vessel])
                if vessel.gen == 0:
                    qois.append(0)
                else:
                    qois.append(dist[0] / vessel.gen)
        else:
            raise Exception('qoi not recognized')
        
        return qois
    

    def get_vessels(self, branches):
        '''
        get a list of vessels from a list of branches

        :param branches: list of branches, or str
        '''

        if branches == 'all':
            branches = list(self.result.keys())
            vessels = list(self.vessel_map.values())

        elif branches == 'outlets':
            outlet_branches, outlet_d = find_outlets(self.config)
            branches = [str(branch) for branch in outlet_branches]
            vessels = [self.vessel_map[branch] for branch in outlet_branches]
        else:
            if 'rpa' in branches:
                branches[branches.index('rpa')] = self.rpa.branch
            if 'lpa' in branches:
                branches[branches.index('lpa')] = self.lpa.branch

            vessels = [self.vessel_map[branch] for branch in branches]
            branches = [str(branch) for branch in branches]
        
        return branches, vessels
        

    def combine_vessels(self, vessel1, vessel2):
        '''
        combine two vessels of the same branch into one vessel
        :param vessel1: first vessel
        :param vessel2: second vessel
        '''
        vessel_config = dict(
            label = vessel1.label,
            vessel_id = vessel1.branch,
            vessel_length = vessel1.length + vessel2.length,
            vessel_name = 'branch' + str(vessel1.branch) + '_seg0',
            zero_d_element_values = {
                'R_poiseuille': vessel1.zero_d_element_values['R_poiseuille'] + vessel2.zero_d_element_values['R_poiseuille'],
                # need to update this to be actual math
                'C': vessel1.zero_d_element_values['C'] + vessel2.zero_d_element_values['C'],
                # need to update this to be actual math
                'L': vessel1.zero_d_element_values['L'] + vessel2.zero_d_element_values['L'],
                # take the max, there is definitely some math to be done here
                'stenosis_coefficient': max(vessel1.zero_d_element_values['stenosis_coefficient'], vessel2.zero_d_element_values['stenosis_coefficient'])
            }
        )

        return self.Vessel(vessel_config)


    def get_R_eq(self):
        '''
        calculate the equivalent resistance for a vessel

        :param vessel: vessel to calculate resistance for
        '''

        # get the resistance of the children
        def calc_R_eq(vessel):
            if len(vessel.children) != 0:
                calc_R_eq(vessel.children[0])
                calc_R_eq(vessel.children[1])
                vessel.R_eq = vessel.zero_d_element_values['R_poiseuille'] + (1 / sum([1 / child.R_eq for child in vessel.children]))
            else:
                vessel.R_eq = vessel.zero_d_element_values['R_poiseuille']
        
        calc_R_eq(self.root)
    
    class Vessel:
        '''
        class to handle tree structure creation and dfs on the tree

        '''
        def __init__(self, config: dict):
            # optional name, such as mpa, lpa, rpa to classify the vessel
            self.label= None
            # list of child vessels
            self.children = []
            self.parent = None
            # path to root
            self.path = []
            # generation from root
            self.gen = None
            # get info from vessel config
            self.length = config['vessel_length']
            self.id = config['vessel_id']
            self.branch = get_branch_id(config)[0]
            self.zero_d_element_values = config['zero_d_element_values']
            self.R_eq = 0.0
            # get the branch diameter
            self.d = 0.0





    
    

