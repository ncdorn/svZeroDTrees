'''
we may not ever use this. I kind of want to delete this.
'''

from ..microvasculature import StructuredTree

class PriesnSecomb():
    '''
    class to perform Pries and Secomb integration on a structured tree
    '''
    def __init__(self, tree: StructuredTree,
                 k_p = 0.68,
                 k_m = .70,
                 k_c = 2.45,
                 k_s = 1.72,
                 L = 1.73,
                 J0 = 27.9,
                 tau_ref = .103,
                 Q_ref = 3.3 * 10 ** -8,
                 dt=0.01, 
                 tol = .01, 
                 time_avg_q=True):
        '''
        :param tree: StructuredTree instance
        :param ps_params: pries and secomb empirical parameters. in the form [k_p, k_m, k_c, k_s, L (cm), S_0, tau_ref, Q_ref]
            units:
                k_p, k_m, k_c, k_s [=] dimensionless
                L [=] cm
                J0 [=] dimensionless
                tau_ref [=] dyn/cm2
                Q_ref [=] cm3/s
        :param dt: time step for explicit euler integration
        :param tol: tolerance (relative difference in function value) for euler integration convergence
        :param time_avg_q: True if the flow in the vessels is assumed to be steady
        '''
        self.tree = tree
        self.k_p = k_p
        self.k_m = k_m
        self.k_c = k_c
        self.k_s = k_s
        self.L = L
        self.J0 = J0
        self.tau_ref = tau_ref
        self.Q_ref = Q_ref
        self._ps_params = [self.k_p, self.k_m, self.k_c, self.k_s, self.L, self.J0, self.tau_ref, self.Q_ref]
        self.dt = dt
        self.tol = tol
        self.time_avg_q = time_avg_q
        self.H_d = 0.45 # hematocrit

    def integrate(self):
        '''
        integrate pries and secomb diff eq by Euler integration for the tree until dD reaches some tolerance (default 10^-5)
        '''
        # initialize sum of squared dD
        SS_dD = 0.0

        # initialize converged stop condition
        converged = False

        # intialize iteration count
        iter = 0

        og_d = self.tree.root.d
        # begin euler integration
        while not converged:
            
            # create solver config from StructuredTree and get tree flow result
            self.tree.create_bcs()
            tree_result = run_svzerodplus(self.tree.block_dict)

            # assign flow result to TreeVessel instances to allow for pries and secomb downstream calculation
            assign_flow_to_root(tree_result, self.tree.root, steady=self.time_avg_q)

            # initialize sum of squared dDs, to check dD against tolerance
            self.sumsq_dD = 0.0 
            def stimulate(vessel):
                '''
                postorder traversal to adapt each vessel according to Pries and Secomb equations

                :param vessel: TreeVessel instance
                '''
                if vessel:

                    # postorder traversal step
                    stimulate(vessel.left)
                    stimulate(vessel.right)

                    # adapt vessel diameter
                    vessel_dD = vessel.adapt_pries_secomb(self.k_p, 
                                                          self.k_m, 
                                                          self.k_c, 
                                                          self.k_s, 
                                                          self.L,
                                                          self.J0,
                                                          self.tau_ref,
                                                          self.Q_ref,
                                                          self.dt,
                                                          self.H_d)

                    # update sum of squared dD
                    self.sumsq_dD += vessel_dD ** 2
            
            # begin traversal
            stimulate(self.tree.root)

            # check if dD is below the tolerance. if so, end integration
            dD_diff = abs(self.sumsq_dD ** 2 - SS_dD ** 2)
            if iter == 0:
                first_dD = dD_diff

            if dD_diff / first_dD < self.tol:
                converged = True
            
            # if not converged, continue integration
            SS_dD = self.sumsq_dD

            # increase iteration count
            iter += 1

        print('Pries and Secomb integration completed in ' + str(iter) + ' iterations! R = ' + str(self.tree.root.R_eq) + ', dD = ' + str(og_d - self.tree.root.d))


    def optimize_params(self):
        '''
        optimize the pries and secomb parameters for stable adaptation with pre-inerventional hemodynamics
        '''

        # print the initial parameters
        print('default parameters: ' + str(self.ps_params))

        param_bounds = Bounds(lb=[0, 0, 0, 0, 0, 0, 0, 0], keep_feasible=True)
        minimize(self.stimulate_vessels, self.ps_params, args=(True), method='Nelder-Mead', bounds=param_bounds)

        # print the optimized parameters
        print('optimized parameters: ' + str(self.ps_params))


    def stimulate_vessels(self, ps_params, optimizing_params):
        '''
        stimulate the vessels and compute adaptation
        '''
        self.sumsq_dD = 0.0 

        # convert the ps_params list into individual parameters
        self.k_p = ps_params[0]
        self.k_m = ps_params[1]
        self.k_c = ps_params[2]
        self.k_s = ps_params[3]
        self.L = ps_params[4]
        self.J0 = ps_params[5]
        self.tau_ref = ps_params[6]
        self.Q_ref = ps_params[7]

        # create solver config from StructuredTree and get tree flow result
        self.tree.create_bcs()
        tree_result = run_svzerodplus(self.tree.block_dict)

        # assign flow result to TreeVessel instances to allow for pries and secomb downstream calculation
        assign_flow_to_root(tree_result, self.tree.root, steady=True)



        def stimulate(vessel):
            '''
            postorder traversal to adapt each vessel according to Pries and Secomb equations

            :param vessel: TreeVessel instance
            '''
            if vessel:

                # postorder traversal step
                stimulate(vessel.left)
                stimulate(vessel.right)

                # compute the change in vessel diameter due to adaptation
                vessel_dD = vessel.adapt_pries_secomb(self.k_p, 
                                                        self.k_m, 
                                                        self.k_c, 
                                                        self.k_s, 
                                                        self.L,
                                                        self.J0,
                                                        self.tau_ref,
                                                        self.Q_ref,
                                                        self.dt,
                                                        self.H_d,
                                                        optimizing_params=optimizing_params)

                # update sum of squared dD
                self.sumsq_dD += vessel_dD ** 2
            
        # begin traversal
        stimulate(self.tree.root)

        return self.sumsq_dD


    # property decorators
    @property
    def ps_params(self):
        self._ps_params = [self.k_p, self.k_m, self.k_c, self.k_s, self.L, self.J0, self.tau_ref, self.Q_ref]
        return self._ps_params