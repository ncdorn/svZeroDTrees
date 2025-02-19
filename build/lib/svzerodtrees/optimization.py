from svzerodtrees.utils import *
from svzerodtrees.result_handler import ResultHandler
from svzerodtrees.config_handler import ConfigHandler
from svzerodtrees import preop, operation, adaptation
from scipy.optimize import minimize, Bounds
import pickle


class StentOptimization:
    '''
    a class to handle optimization of stent diameters
    '''

    def __init__(self, config_handler, result_handler, repair_config, adapt='ps', log_file=None, n_procs=12, trees_exist=True):
        '''
        :param config_handler: ConfigHandler instance
        :param result_handler: ResultHandler instance
        :param repair_config: repair config dict containing 
                            "type"="optimize stent", 
                            "location"="proximal" or some list of locations, 
                            "objective"="flow split" or "mpa pressure" or "all"
        :param adaptation: adaptation scheme to use, either 'ps' or 'cwss'
        :param n_procs: number of processors to use for tree construction
        '''

        if trees_exist:
            # compute preop result
            preop_result = run_svzerodplus(config_handler.config)
            # add the preop result to the result handler
            result_handler.add_unformatted_result(preop_result, 'preop')

        else:
            # construct trees
            if adapt == 'ps':
                # pries and secomb adaptationq
                preop.construct_pries_trees(config_handler,
                                            result_handler,
                                            n_procs=n_procs,
                                            log_file=log_file,
                                            d_min=.01)
                
            elif adapt == 'cwss':
                # constant
                preop.construct_cwss_trees(config_handler,
                                                result_handler,
                                                n_procs=n_procs,
                                                log_file=log_file,
                                                d_min=.0049) # THIS NEEDS TO BE .0049 FOR REAL SIMULATIONS

            # save preop config to as pickle, with StructuredTree objects
            write_to_log(log_file, 'saving preop config with' + adapt + 'trees...')

            # pickle the config handler with the trees
            config_handler.to_file_w_trees('config_w_' + adapt + '_trees.in')

        # define the class attributes
        self.preop_config_handler = config_handler # config handler
        self.result_handler = result_handler # result handler
        self.repair_config = repair_config # repair config dict, ideally won't need this
        self.adapt = adapt # adaptation scheme
        self.log_file = log_file # log file
        self.n_procs = n_procs # n processors to use
        if repair_config['location'] == 'proximal':
            self.branches = [config_handler.lpa.branch, config_handler.rpa.branch]
        else:
            self.branches = repair_config['location']
        self.value = repair_config['value'] # initial values for stent optimization
        self.objective_type = repair_config['objective'] # objective function type to use

        # attributes we need for choosing branches
        self.current_stents = None
        self.all_wss = {}
        self.high_wss = {}

    
    def minimize_nm(self):
        '''
        run the optimization
        '''

        # create a lower bound for the optimization that is the original vessel diameter
        
        if len(self.branches) == len(self.repair_config['value']):
            stent = Bounds(lb=[self.preop_config_handler.branch_map[branch].diameter for branch in self.branches],
                            ub=[2.2 for branch in self.branches])
        else:
            stent = Bounds(lb=[0.2] * len(self.repair_config['value']),
                            ub=[2.2] * len(self.repair_config['value']))
        
        result = minimize(self.simple_objective, self.repair_config["value"], args=(self.branches),bounds=stent, method='Nelder-Mead')

        print('optimized stent diameters: ' + str(result.x))
        # write this to the log file as well
        write_to_log(self.log_file, 'stent locations: ' + str(self.repair_config['location']))
        write_to_log(self.log_file, 'optimized stent diameters: ' + str(result.x))

    def simple_objective(self, diameters, branches):
        '''
        compute the objective function based on the input stent diameters and branches
        '''
        # copy the preop config handler so we do not change it on accident
        # self.config_handler = copy.deepcopy(self.preop_config_handler)
        # loding the config handler from pickle is much faster
        with open('config_w_' + self.adapt + '_trees.in', 'rb') as ff:
            self.config_handler = pickle.load(ff)

        # generate repair config
        self.repair_config['type'] = 'stent'
        self.repair_config['location'] = branches
        self.repair_config['value'] = diameters

        # perform repair
        self.current_stents = operation.repair_stenosis(self.config_handler, 
                                                        self.result_handler,
                                                        self.repair_config, 
                                                        self.log_file)

        # compute adaptation
        if self.adapt == 'cwss':
            adaptation.adapt_constant_wss(self.config_handler,
                                          self.result_handler,
                                          self.log_file)
        elif self.adapt == 'ps':
            adaptation.adapt_pries_secomb(self.config_handler,
                                          self.result_handler,
                                          self.log_file)

        # reformat the results so we can use it for the loss function
        self.result_handler.format_results()

        # get quantities of interest
        rpa_split = self.result_handler.flow_split['rpa']
        mpa_pressure = self.result_handler.clean_results['mpa']['p_in']['adapted']

        target_wss = 10.0 # dyn/cm2, 100 dyn/cm2 is the value from Shinohara et al. (2024) but we reduce this to see how it improves the optimization

        self.all_wss = {branch: self.result_handler.clean_results[branch]['wss']['adapted'] for branch in self.config_handler.branch_map.keys()}

        self.high_wss = {branch: wss_val for branch, wss_val in self.all_wss.items() if wss_val > 100.0}


        # compute loss
        for vessel, wss in self.high_wss.items():
            flow = get_branch_result(self.result_handler.results['adapted'], 'flow_in', vessel, steady=True)
            print(f'{vessel} has high wss: {wss} dyn/cm2, flow: {flow}, radius: {self.config_handler.vessel_map[self.config_handler.branch_map[vessel].ids[0]].diameter / 2} \n')


        flow_split_objective = self.sse(rpa_split, 0.5)

        mpa_pressure_objective = self.sse(mpa_pressure, 10.0)

        # wss_objective = self.exp_loss(list(self.all_wss.values()), [target_wss])

        wss_objective = self.mse(list(self.all_wss.values()), [target_wss])


        objective = 0

        if self.repair_config['objective'] == 'flow split':
            print(f'stent diameters: {diameters}, rpa split: {rpa_split}, objective: {flow_split_objective} \n')
            return flow_split_objective
        elif self.repair_config['objective'] == 'mpa pressure':
            return mpa_pressure_objective
        elif self.repair_config['objective'] == 'wss':
            print('stent diameters: ' + str(diameters) + '\n')
            print(f'rpa flow split: {rpa_split}')
            print('wss objective: ' + str(wss_objective))
            return wss_objective
        elif self.repair_config['objective'] == 'all':
            print(f"stent diameters: {diameters}")
            print(f'rpa flow split: {rpa_split}')
            
            objective = 10 * flow_split_objective + wss_objective

            print(f"objective: {objective}")

            return objective
        else:
            raise Exception('invalid objective function specified')


    def choose_branches(self, n_vessels, method='Req'):
        '''
        choose the branches to optimize based on some factor

        :param method: method to use to choose branches, either 'Req' or 'WSS'
        '''


    def compute_Req_contribution(self, n_vessels):
        '''
        compute the contribution of change in resistance for each vessel to Req'''

        data_dict = {branch: {'lpa change': 0.0, 
                          'rpa change': 0.0, 
                          'diameter change': 0.0} for branch in self.config_handler.vessel_branch_map.values()}

        for name, vessel in self.config_handler.vessel_map.items():
            intial_lpa_Req = self.config_handler.lpa.R_eq
            intial_rpa_Req = self.config_handler.rpa.R_eq

            initial_d = vessel.diameter

            # adjust vessel diameter
            vessel.diameter = 0.5

            stented_lpa_Req = self.config_handler.lpa.R_eq
            stented_rpa_Req = self.config_handler.rpa.R_eq

            lpa_Req_diff = intial_lpa_Req - stented_lpa_Req
            rpa_Req_diff = intial_rpa_Req - stented_rpa_Req

            diameter_diff = vessel.diameter - initial_d

            data_dict[self.config_handler.vessel_branch_map[name]]['lpa change'] += lpa_Req_diff
            data_dict[self.config_handler.vessel_branch_map[name]]['rpa change'] += rpa_Req_diff

            # reset vessel diameter
            vessel.diameter = initial_d


        lpa_vessels = sorted(data_dict, key=lambda x: data_dict[x]['lpa change'], reverse=True)
        rpa_vessels = sorted(data_dict, key=lambda x: data_dict[x]['rpa change'], reverse=True)

        return lpa_vessels[:n_vessels], rpa_vessels[:n_vessels]


    def mse(self, values, target):
        '''
        mean squared error between values and target
        '''

        if values is list:
            values = np.array(values)
            if len(target) != len(values):
                target = np.repeat(target, len(values))

        return np.mean(np.subtract(values, target) ** 2)
    
    def sse(self, values, target):
        '''
        sum of squared errors between values and target
        '''

        if values is list:
            values = np.array(values)
            if len(target) != len(values):
                target = np.repeat(target, len(values))

        return np.sum(np.subtract(values, target) ** 2)

    def exp_loss(self, values, target):
        '''
        objective function to minimize based on input stent diameters
        '''

        if values is list or np.ndarray:
            values = np.array(values)
            if len(target) != len(values):
                target = np.repeat(target, len(values))

        return np.sum([np.exp(value - target) for value, target in zip(values, target)])
        



        





# this to be deprecated and removed once the class is made
def optimize_stent_diameter(config_handler, result_handler, repair_config: dict, adapt='ps' or 'cwss', log_file=None, n_procs=12, trees_exist=True):
    '''
    optimize stent diameter based on some objective function containing flow splits or pressures
    
    :param config_handler: ConfigHandler instance
    :param result_handler: ResultHandler instance
    :param repair_config: repair config dict containing 
                        "type"="optimize stent", 
                        "location"="proximal" or some list of locations, 
                        "objective"="flow split" or "mpa pressure" or "all"
    :param adaptation: adaptation scheme to use, either 'ps' or 'cwss'
    :param n_procs: number of processors to use for tree construction
    '''

    if trees_exist:
        # compute preop result
        preop_result = run_svzerodplus(config_handler.config)
        # add the preop result to the result handler
        result_handler.add_unformatted_result(preop_result, 'preop')

    else:
        # construct trees
        if adapt == 'ps':
            # pries and secomb adaptationq
            preop.construct_pries_trees(config_handler,
                                        result_handler,
                                        n_procs=n_procs,
                                        log_file=log_file,
                                        d_min=.01)
            
        elif adapt == 'cwss':
            # constant
            preop.construct_cwss_trees(config_handler,
                                            result_handler,
                                            n_procs=n_procs,
                                            log_file=log_file,
                                            d_min=.0049) # THIS NEEDS TO BE .0049 FOR REAL SIMULATIONS

        # save preop config to as pickle, with StructuredTree objects
        write_to_log(log_file, 'saving preop config with' + adapt + 'trees...')

        # pickle the config handler with the trees
        config_handler.to_file_w_trees('config_w_' + adapt + '_trees.in')

    

    def objective_function(diameters, result_handler, repair_config, adapt):
        '''
        objective function to minimize based on input stent diameters
        '''

        repair_config['type'] = 'stent'
        repair_config['value'] = diameters

        with open('config_w_' + adapt + '_trees.in', 'rb') as ff:
            config_handler = pickle.load(ff)

        # perform repair. this needs to be updated to accomodate a list of repairs > length 1
        # print(f"preop diameter: {config_handler.branch_map[1].diameter}")
        operation.repair_stenosis(config_handler, 
                                  result_handler,
                                  repair_config, 
                                  log_file)
        # print(f"postop diameter: {config_handler.branch_map[1].diameter}")

        # adapt trees
        if adapt == 'cwss':
            adaptation.adapt_constant_wss(config_handler,
                                          result_handler,
                                          log_file)
        elif adapt == 'ps':
            adaptation.adapt_pries_secomb(config_handler,
                                          result_handler,
                                          log_file)

        rpa_split = get_branch_result(result_handler.results['adapted'], 'flow_in', config_handler.rpa.branch, steady=True) / get_branch_result(result_handler.results['adapted'], 'flow_in', config_handler.mpa.branch, steady=True)

        mpa_pressure = get_branch_result(result_handler.results['adapted'], 'pressure_in', config_handler.mpa.branch, steady=True)

        # reformat the results so we can use it for the loss function
        result_handler.format_results()

        target_wss = 50.0 # dyn/cm2, 100 dyn/cm2 is the value from Shinohara et al. (2024) but we reduce this to see how it improves the optimization

        all_wss = {branch: result_handler.clean_results[branch]['wss']['adapted'] for branch in config_handler.branch_map.keys()}

        high_wss = {branch: wss_val for branch, wss_val in all_wss.items() if wss_val > target_wss}

        for vessel, wss in high_wss.items():
            flow = get_branch_result(result_handler.results['adapted'], 'flow_in', vessel, steady=True)
            print(f'{vessel} has wss: {wss} dyn/cm2, flow: {flow}, radius: {config_handler.vessel_map[config_handler.branch_map[vessel].ids[0]].diameter / 2} \n')


        flow_split_objective = (rpa_split - 0.5) ** 2

        mpa_pressure_objective = (mpa_pressure - 20) ** 2

        wss_objective = sum([10**(wss - target_wss) for wss in all_wss.values()])

        diameter_check = np.sum([np.exp(-1 * diameter) for diameter in diameters])



        if repair_config['objective'] == 'flow split':
            return flow_split_objective
        elif repair_config['objective'] == 'mpa pressure':
            return mpa_pressure_objective
        elif repair_config['objective'] == 'wss':
            print('stent diameters: ' + str(diameters) + '\n')
            print(f'rpa flow split: {rpa_split}')
            print('wss objective: ' + str(wss_objective))
            return wss_objective
        elif repair_config['objective'] == 'all':
            print('stent diameters: ' + str(diameters) + '\n')
            print(f'rpa flow split: {rpa_split}')
            print('flow split objective: ' + str(100 * flow_split_objective) + '\n')
            print('wss objective: ' + str(wss_objective))
            return 10 * flow_split_objective + wss_objective + diameter_check
        else:
            raise Exception('invalid objective function specified')
    
    result = minimize(objective_function, repair_config["value"], args=(result_handler, repair_config, adapt), method='Nelder-Mead')

    print('optimized stent diameters: ' + str(result.x))
    # write this to the log file as well
    write_to_log(log_file, 'stent locations: ' + str(repair_config['location']))
    write_to_log(log_file, 'optimized stent diameters: ' + str(result.x))

    rpa_split = get_branch_result(result_handler.results['adapted'], 'flow_in', config_handler.rpa.branch, steady=True) / get_branch_result(result_handler.results['adapted'], 'flow_in', config_handler.mpa.branch, steady=True)

    write_to_log(log_file, 'RPA flow split: ' + str(rpa_split))