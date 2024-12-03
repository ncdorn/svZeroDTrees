from svzerodtrees.utils import *
import pickle
import json
from scipy.integrate import trapz

class ResultHandler:
    '''
    class to handle preop, postop and post adaptation results from the structured tree simulation
    '''

    def __init__(self, vessels, lpa_branch=None, rpa_branch=None, viscosity=None):

        self.lpa_branch = lpa_branch
        self.rpa_branch = rpa_branch
        # dict to store preop, postop, adapted vessels
        self.vessels = {'preop': vessels}
        self.viscosity = viscosity # for wss calculations
        self.results = {} # of the form [timestep][field][branch_id]
        self.clean_results = {} # of the form [branch_id][field][timestep]
    
    @classmethod
    def from_config(cls, config):
        '''
        class method to generate the results handler with vessel and config information

        :param config: 0d config dict

        :return: ResultHandler instance
        '''

        # get rpa_lpa_branch ids
        lpa_branch, rpa_branch = find_lpa_rpa_branches(config)

        # get vessel info and vessel ids (vessel info necessary for wss calculations)
        vessels = []
        for vessel_config in config['vessels']:
            # add to the vessel list
            vessels.append(vessel_config)

        # get the viscosity
        viscosity = config['simulation_parameters']['viscosity']

        return ResultHandler(vessels, lpa_branch, rpa_branch, viscosity)

    @classmethod
    def from_config_handler(cls, config_handler):
        '''
        class method to generate the results handler from a config handler
        
        :param config_handler: ConfigHandler instance
        
        :return: ResultHandler instance
        '''
        # get rpa_lpa_branch ids
        if config_handler.is_pulmonary:
            lpa_branch, rpa_branch = find_lpa_rpa_branches(config_handler.config)
        else:
            lpa_branch, rpa_branch = None, None

        # get vessel info and vessel ids (vessel info necessary for wss calculations)
        # may need to delete this
        vessels = []
        for vessel_config in config_handler.config['vessels']:
            # add to the vessel list
            vessels.append(vessel_config)

        

        # get the viscosity
        viscosity = config_handler.config['simulation_parameters']['viscosity']

        return ResultHandler(config_handler.config['vessels'], lpa_branch, rpa_branch, viscosity)


    def format_results(self, is_pulmonary=True):
        '''
        format the results into preop, postop and adapted for each branch, for use in visualization
        '''

        # get summary results for the MPA
        if is_pulmonary:
            self.clean_results['mpa'] = self.format_branch_result(0)

            # get summary results for the RPA
            self.clean_results['rpa'] = self.format_branch_result(self.rpa_branch)

            # get summary results for the LPA
            self.clean_results['lpa'] = self.format_branch_result(self.lpa_branch)

            # create a flow split attribute
            self.flow_split = {
                'rpa': self.clean_results['rpa']['q_out']['adapted'] / (self.clean_results['rpa']['q_out']['adapted'] + self.clean_results['lpa']['q_out']['adapted']),
                'lpa': self.clean_results['lpa']['q_out']['adapted'] / (self.clean_results['rpa']['q_out']['adapted'] + self.clean_results['lpa']['q_out']['adapted'])
            }

        # get summary results for all other vessels
        for vessel_config in self.vessels['preop']:
            id = get_branch_id(vessel_config)[0]
            # if id not in [0, self.lpa_branch, self.rpa_branch]:
            self.clean_results[id] = self.format_branch_result(id)

    
    def format_branch_result(self, branch: int):
        '''
        get a dict containing the preop, postop and adapted q, p, wss for a specified branch

        :param branch: branch id

        :return branch_summary: dict with preop, postop and adapted outlet q, p, wss
        '''
        
        # initialize branch summary dict
        branch_result = {}

        # get the inlet flowrates preop, postop, and post adaptation
        preop_q = get_branch_result(self.results['preop'], 'flow_in', branch, steady=True)
        postop_q = get_branch_result(self.results['postop'], 'flow_in', branch, steady=True)
        final_q = get_branch_result(self.results['adapted'], 'flow_in', branch, steady=True)
        branch_result['q_in'] = {'preop': preop_q, 'postop': postop_q, 'adapted': final_q}

        # get the outlet flowrates preop, postop, and post adaptation
        preop_q = get_branch_result(self.results['preop'], 'flow_out', branch, steady=True)
        postop_q = get_branch_result(self.results['postop'], 'flow_out', branch, steady=True)
        final_q = get_branch_result(self.results['adapted'], 'flow_out', branch, steady=True)
        branch_result['q_out'] = {'preop': preop_q, 'postop': postop_q, 'adapted': final_q}

        # get the inlet pressures preop, postop and post adaptation, in mmHg
        preop_p = get_branch_result(self.results['preop'], 'pressure_in', branch, steady=True) / 1333.22
        postop_p = get_branch_result(self.results['postop'], 'pressure_in', branch, steady=True) / 1333.22
        final_p = get_branch_result(self.results['adapted'], 'pressure_in', branch, steady=True) / 1333.22
        branch_result['p_in'] = {'preop': preop_p, 'postop': postop_p, 'adapted': final_p}

        # get the outlet pressures preop, postop and post adaptation, in mm Hg
        preop_p = get_branch_result(self.results['preop'], 'pressure_out', branch, steady=True) / 1333.22
        postop_p = get_branch_result(self.results['postop'], 'pressure_out', branch, steady=True) / 1333.22
        final_p = get_branch_result(self.results['adapted'], 'pressure_out', branch, steady=True) / 1333.22
        branch_result['p_out'] = {'preop': preop_p, 'postop': postop_p, 'adapted': final_p}

        # get the wall shear stress at the outlet
        preop_wss = get_wss(self.vessels['preop'], self.viscosity, self.results['preop'], branch, steady=True)
        postop_wss = get_wss(self.vessels['postop'], self.viscosity, self.results['postop'], branch, steady=True)
        final_wss = get_wss(self.vessels['adapted'], self.viscosity, self.results['adapted'], branch, steady=True)
        branch_result['wss'] = {'preop': preop_wss, 'postop': postop_wss, 'adapted': final_wss}

        

        return branch_result

    def add_unformatted_result(self, result, name):
        '''
        add an unformatted svzerodplus result to the result handler

        :param result: the result to add
        :param name: the name of the result (preop, postop, final)
        '''

        self.results[name] = result

    def to_file(self, file_name: str):
        '''
        write the result handler to a pickle file

        :param file_name: name of the file to write to
        '''

        with open(file_name, 'wb') as ff:
            pickle.dump(self, ff)
    
    def to_json(self, file_name: str):
        '''
        write the result handler to a json file

        :param file_name: name of the file to write to
        '''

        self.format_results()

        with open(file_name, 'w') as ff:
            json.dump(self.clean_results, ff)


    def format_result_for_cl_projection(self, timestep):
        '''format a pressure or flow result for the centerline projection

        Args:
            timestep (str): timestep to format, ['preop', 'postop', 'adapted']
        
        
        '''

        cl_mappable_result = {"flow": {}, "pressure": {}, "wss": {}, "distance": {},"time": {}, "resistance": {}, "WU m2": {}, "repair": {}, "diameter": {}}

        branches = list(self.clean_results.keys())
        for branch in branches:
            if branch == 'mpa':
                branches[branches.index(branch)] = 0
            elif branch == 'lpa':
                branches[branches.index(branch)] = self.lpa_branch
            elif branch == 'rpa':
                branches[branches.index(branch)] = self.rpa_branch

        fields = list(self.results['preop'].keys())

        fields.sort() # should be ['flow_in', 'flow_out', 'pressure_in', 'pressure_out']

        if timestep == 'adaptation':
            for field in ['flow', 'pressure', 'wss']:
                if field == 'wss':
                    cl_mappable_result[field] = {
                        branch: [[(self.clean_results[branch]['wss']['postop'] - 
                                    self.clean_results[branch]['wss']['adapted']) / 
                                    self.clean_results[branch]['wss']['postop']] * 10,
                                [(self.clean_results[branch]['wss']['postop'] - # need 2 entries for the oulet and inlet
                                    self.clean_results[branch]['wss']['adapted']) / 
                                    self.clean_results[branch]['wss']['postop']] * 10]
                                for branch in branches}
                else:
                    cl_mappable_result[field] = {
                        branch: [(self.results['postop'][field + "_in"][branch] - 
                                    self.results['adapted'][field + "_in"][branch]) / 
                                    self.results['postop'][field + "_in"][branch], 
                                (self.results['postop'][field + "_out"][branch] - 
                                    self.results['adapted'][field + "_out"][branch]) / 
                                    self.results['postop'][field + "_out"][branch], ]
                                for branch in branches}
        else:
            for field in ['flow', 'pressure', 'wss']:
                if field == 'wss':
                    # language is different in the clean result vs the general result, need to fix this

                    # the clean_result currently computes one value for wss, when there could be a time-varying quantity
                    # we will leave it as constant for now
                    cl_mappable_result[field] = {
                        branch: [[self.clean_results[branch]['wss'][timestep]] * 10] * 2 for branch in branches}
                else:
                    cl_mappable_result[field] = {
                        branch: [self.results[timestep][field + "_in"][branch], 
                                self.results[timestep][field + "_out"][branch]]
                        for branch in branches}
                # now, change the branch id for mpa, lpa, rpa
                for branch in cl_mappable_result[field].keys():
                    if branch == 'mpa':
                        cl_mappable_result[field][0] = cl_mappable_result[field].pop(branch)
                    elif branch == 'lpa':
                        cl_mappable_result[field][self.lpa_branch] = cl_mappable_result[field].pop(branch)
                    elif branch == 'rpa':
                        cl_mappable_result[field][self.rpa_branch] = cl_mappable_result[field].pop(branch)
        
        # construct the distance dict

        return cl_mappable_result


    def results_to_dict(self):
        '''
        convert the results to dict which are json serializeable
        '''

        for timestep in self.results.keys():
            for field in self.results[timestep].keys():
                self.results[timestep][field] = {key: value.tolist() for key, value in self.results[timestep][field].items()}


    def get_cardiac_output(self, branch: int, timestep: str='preop'):
        '''
        get the cardiac output for the preop, postop and adapted simulations

        :param branch: the branch id of the mpa
        :param timestep: preop, postop, adapted
        '''

        # get the flow in the mpa
        q_in = self.results[timestep]['flow_in'][branch]
        
        t = self.results[timestep]['time']

        cardiac_output = trapz(q_in, t)

        return cardiac_output
    

    def plot(self, timestep, field, branches, filepath=None, show_mean=False):
        '''
        plot a field for a specified branch and timestep

        :param timestep: the timestep to plot
        :param field: the field to plot
        :param branches: list of branch ids to plot
        :param save_path: the path to save the plot
        :param show_mean: whether to show the mean value on the plot
        '''
        data = []
        for branch in branches:
            data.append(self.results[timestep][field][branch])

        plt.clf()
        for datum in data:
            if np.log10(np.array(datum).mean()) >= 3:
                datum = np.array(datum) / 1333.22
            plt.plot(self.results[timestep]['time'], datum)
            if show_mean:
                plt.axhline(y=np.mean(datum), linestyle='--', label='mean')

        plt.xlabel('time')
        plt.ylabel(field)
        plt.title(f"branch {branches} {field} {timestep}")
        plt.legend(branches)
        plt.pause(0.001)
        if filepath == None:
            plt.show()
        else:
            plt.savefig(filepath)