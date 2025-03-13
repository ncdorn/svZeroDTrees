import numpy as np
import random
from scipy.optimize import minimize, Bounds, LinearConstraint
from svzerodtrees.treevessel import TreeVessel
from svzerodtrees.utils import *
from svzerodtrees.config_handler import ConfigHandler
from svzerodtrees.blocks import *
from multiprocessing import Pool
import math
import scipy
import json
import pickle
from functools import partial
import time
from numba import jit

class StructuredTree():
    """
    Structured tree which represents microvascular adaptation at the outlets of a 0D Windkessel model.
    utilizes the TreeVessel class which is recursive by nature to handle recursive tasks
    """
    def __init__(self, params: dict = None, 
                 diameter: float = 0.5, # default value that should be optimized later
                 R: float = None,
                 C: float = None,
                 P_in: list = None,
                 Q_in: list = None,
                 time: list = None,
                 Pd: float = 0.0,
                 name: str = None, tree_config: dict = None, simparams: SimParams = None, root: TreeVessel = None):
        """
        Create a new StructuredTree instance
        
        :param params: dict of 0D Windkessel parameters for the StructuredTree class. 
            contains length, R, C, L, stenosis coeff, viscosity, inlet pressure and flow, bc values
        :param name: name of the StructuredTree instance, e.g. OutletTree3
        :param tree_config: optional tree config dict, used to create a StructuredTree instance from a pre-existing tree which has
            been saved in the model 0D config dict
        :param simparams: simulation parameters from the 0D model config file
        :param root: TreeVessel instance, required if the StructuredTree instance is built from a pre-existing tree

        :return: None
        """
        # parameter attributes
        self.params = params # TODO: remove this attribute such that all parameters are indiividual class attributes since these are fixed
        self.diameter = diameter

        # windkessel parameters 
        self._R = R
        self.C = C

        # simulation parameters
        self.viscosity = 0.049
        self.density = 1.055
        self.simparams = simparams

        # flow and pressure attributes
        self.P_in = P_in
        self.Q_in = Q_in
        self.time = time
        self.Pd = Pd

        ### non-diensional parameters ###
        self.q = 10.0 # characteristic flow rate, cm3/s
        self.Lr = 1.0 # characteristic length scale, cm
        self.g = 981.0 # acceleration due to gravity

        #number of generations
        self.generations = 0

        # set up empty block dict if not generated from pre-existing tree
        if tree_config is None:
            self.name = name
            self.block_dict = {'name': name, 
                               'initial_d': diameter, 
                               'P_in': P_in,
                               'Q_in': Q_in,
                               'boundary_conditions': [],
                               'simulation_parameters': simparams.to_dict() if simparams is not None else None,
                               'vessels': [], 
                               'junctions': [], 
                               'adaptations': 0}
            # initialize the root of the structured tree
            self.root = None
        else:
            # set up parameters from pre-existing tree config
            self.name = tree_config["name"]
            self.block_dict = tree_config
            # initialize the root of the structured tree
            if root is None:
                # if no TreeVessel instance is provided to create the new tree
                raise Exception('No root TreeVessel instance provided!')
            # add the root instance to the StructuredTree
            self.root = root


    @classmethod
    def from_outlet_vessel(cls, 
                           vessel: Vessel, 
                           simparams: SimParams,
                           bc: BoundaryCondition,
                           tree_exists=False, 
                           root: TreeVessel = None, 
                           P_outlet: list=[0.0], 
                           Q_outlet: list=[0.0],
                           time: list=[0.0]) -> "StructuredTree":
        """
        Class method to creat an instance from the config dictionary of an outlet vessel

        :param config: config file of outlet vessel
        :param simparams: config file of simulation parameters to get viscosity
        :param bc_config: config file of the outlet boundary condition
        :param tree_exists: True if the StructuredTree is being created from a pre-existing tree (applicable in the adaptation 
            and postop steps of the simulation pipeline)
        :param root: TreeVessel instance, required if tree_exists = True
        :param P_outlet: pressure at the outlet of the 0D model, which is the inlet of this StructuredTree instance
        :param Q_outlet: flow at the outlet of the 0D model, which is the inlet of this StructuredTree instance

        :return: StructuredTree instance
        """
        # if steady state, make the Q_outlet and P_outlet into a list of length two for svzerodplus config BC compatibility
        if type(Q_outlet) is not list:
            # steady flow
            Q_outlet = [Q_outlet,] * 2
            time = [0.0, 1.0]
        
        if type(P_outlet) is not list:
            # steady pressure
            P_outlet = [P_outlet,] * 2
            time = [0.0, 1.0]
        

        params = dict(
            # need vessel length to determine vessel diameter
            diameter = vessel.diameter,
            eta=simparams.viscosity,
            P_in = P_outlet,
            Q_in = Q_outlet,
            bc_values = bc.values
        )

        # convert bc_values to class parameters
        if "Rp" in bc.values.keys():
            R_bc = bc.values["Rp"] + bc.values["Rd"]
            C_bc = bc.values["C"]
        else:
            R_bc = bc.values["R"]
            C_bc = None
        

        if tree_exists:
            print("tree exists")
            # not sure if this works, probably need to change to a tree config parameter
            return cls(params=params, 
                       diameter = vessel.diameter,
                       R = R_bc,
                       C = C_bc,
                       P_in = P_outlet,
                       Q_in = Q_outlet, 
                       Pd = bc.values["Pd"],
                       viscosity = simparams.viscosity, 
                       config = vessel, simparams=simparams, root=root)
        
        else:
            # this works
            return cls(params=params, 
                       diameter = vessel.diameter,
                       R = R_bc,
                       C = C_bc,
                       P_in = P_outlet,
                       Q_in = Q_outlet, 
                       time=time,
                       Pd = bc.values["Pd"],
                       name="OutletTree" + str(vessel.branch), simparams=simparams)


    @classmethod
    def from_bc_config(cls,
                       bc: BoundaryCondition,
                       simparams: SimParams,
                       diameter: float,
                       P_outlet: list=[0.0], 
                       Q_outlet: list=[0.0],
                       time: list=[0.0]) -> "StructuredTree":
        '''
        Class method to create an instance from the config dictionary of an outlet boundary condition
        '''

        # if steady state, make the Q_outlet and P_outlet into a list of length two for svzerodplus config BC compatibility
        if type(Q_outlet) is not list:
            # steady flow
            Q_outlet = [Q_outlet,] * 2
            time = [0.0, 1.0]
        
        if type(P_outlet) is not list:
            # steady pressure
            P_outlet = [P_outlet,] * 2
            time = [0.0, 1.0]

        params = dict(
            # need vessel length to determine vessel diameter
            diameter = diameter,
            eta=simparams.viscosity,
            P_in = P_outlet,
            Q_in = Q_outlet,
            bc_values = bc.values
        )
        
        return cls(params=params, 
                   diameter = diameter,
                   viscosity=simparams.viscosity,
                   P_in = P_outlet,
                   Q_in = Q_outlet, 
                   time=time,
                   name="OutletTree" + str(bc.name), simparams=simparams)

# **** I/O METHODS ****

    def to_dict(self):
        '''
        convert the StructuredTree instance parameters to a dictionary
        '''

        params = {
            "name": self.name,
            "initial_d": self.initial_d,
            "d_min": self.d_min,
            "k1": self.k1,
            "k2": self.k2,
            "k3": self.k3,
            "n_procs": self.n_procs,

        }
        return params

    def to_json(self, filename):
        '''
        write the structured tree to a json file

        :param filename: name of the json file
        '''
        with open(filename, 'w') as f:
            json.dump(self.block_dict, f, indent=4)


    def to_pickle(self, filename):
        '''
        write the structured tree to a pickle file

        :param filename: name of the pickle file
        '''
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


# **** END OF I/O METHODS ****


    def reset_tree(self, keep_root=False):
        """
        reset the block dict if you are generating many iterations of the structured tree to optimize the diameter

        :param keep_root: bool to decide whether to keep the root TreeVessel instance
        """
        if keep_root:
            pass
        else:
            self.root = None
        self.block_dict["vessels"] = []
        self.block_dict["junctions"] = []
        self.vesselDlist = []


    def create_block_dict(self):
        '''
        create the block dict from a pre-existing root, 
        for example in the case of adapting the diameter of the vessels
        '''
        self.reset_tree(keep_root=True)
        self.block_dict["vessels"].append(self.root.params)
        queue = [self.root]

        while len(queue) > 0:
            q_id = 0
            current_vessel = queue.pop(q_id)
            # create the block dict for the left vessel
            if not current_vessel.collapsed:
                queue.append(current_vessel.left)
                self.block_dict["vessels"].append(current_vessel.left.params)
                # create the block dict for the right vessel
                queue.append(current_vessel.right)
                self.block_dict["vessels"].append(current_vessel.right.params)


    def build_tree(self, initial_d=None, d_min=0.0049, optimizing=False, asym=0.4048, xi=2.7, alpha=None, beta=None, lrr=50.0):
        '''
        recursively build the structured tree

        :param initial_d: root vessel diameter
        :param d_min: diameter at which the vessel is considered "collapsed" and the tree terminates [cm]. default is 100 um
        :param optimizing: True if the tree is being built as part of an optimization scheme, so the block_dict will be
            reset for each optimization iteration
        :param asym: asymmetry ratio, used in the place of alpha and beta
        :param xi: junction scaling coefficient
        :param alpha: left vessel scaling factor (see Olufsen et al. 2012)
        :param beta: right vessel scaling factor
        '''

        if optimizing:
            self.reset_tree() # reset block dict if making the tree many times

        if initial_d is None: # set default value of initial_r
            initial_d = self.initialD

        if d_min <= 0:
            raise ValueError("The min diameter must be greater than 0.")

        if initial_d <= 0:
            raise Exception("initial_d is invalid, " + str(initial_d))
        
        if alpha is None and beta is None:
            alpha = (1 + asym**(xi/2))**(-1/xi)
            beta = asym**(1/2) * alpha
            print(f"alpha: {alpha}, beta: {beta}")

        # make self params
        self.initial_d = initial_d
        self.alpha = alpha
        self.beta = beta
        self.lrr = lrr
        # add r_min into the block dict
        self.d_min = d_min
        # initialize counting values
        vessel_id = 0
        junc_id = 0
        # initialize the root vessel of the tree
        self.root = TreeVessel.create_vessel(0, 0, initial_d, density=1.055, lrr=lrr)
        self.root.name = self.name
        # add inflow boundary condition
        self.root.params["boundary_conditions"] = {"inlet": "INFLOW"}
        self.block_dict["vessels"].append(self.root.params)
        queue = [self.root]

        while len(queue) > 0:
            q_id = 0
            current_vessel = queue.pop(q_id)
            creating_vessels = True
            while current_vessel.collapsed:
                # remove the collapsed vessels without creating new ones
                if len(queue) == 0:
                    creating_vessels = False
                    break
                current_vessel = queue.pop(q_id)
            if not creating_vessels:
                # end the loop
                break

            if not current_vessel.collapsed:
                # create new left and right vessels
                next_gen = current_vessel.gen + 1
                # create left vessel
                vessel_id += 1
                left_dia = alpha * current_vessel.d
                # assume pressure is conserved at the junction. 
                # Could later replace this with a function to account for pressure loss
                current_vessel.left = TreeVessel.create_vessel(vessel_id, next_gen, left_dia, density=1.055, lrr=lrr)
                if left_dia < d_min:
                    current_vessel.left.collapsed = True
                    if current_vessel.left.gen > self.generations:
                        self.generations = current_vessel.left.gen
                else:
                    queue.append(current_vessel.left)
                self.block_dict["vessels"].append(current_vessel.left.params)
                

                # create right vessel
                vessel_id += 1
                right_dia = beta * current_vessel.d
                current_vessel.right = TreeVessel.create_vessel(vessel_id, next_gen, right_dia, density=1.055, lrr=lrr)
                if right_dia < d_min:
                    current_vessel.right.collapsed = True
                    if current_vessel.right.gen > self.generations:
                        self.generations = current_vessel.right.gen
                else:
                    queue.append(current_vessel.right)
                self.block_dict["vessels"].append(current_vessel.right.params)
                

                # add a junction
                junction_config = {"junction_name": "J" + str(junc_id),
                                 "junction_type": "NORMAL_JUNCTION",
                                 "inlet_vessels": [current_vessel.id],
                                 "outlet_vessels": [current_vessel.left.id, current_vessel.right.id]
                                #  "junction_values": {"C": [0, 0, 0],  
                                #                      "L": [0, 0, 0],      MAYBE THIS WILL BE FOR ANOTHER TIME
                                #                      "R_poiseuille": [0, 0, 0], 
                                #                      "stenosis_coefficient": [0, 0, 0]},
                                 }
                self.block_dict["junctions"].append(junction_config)
                junc_id += 1

    # @jit
    def compute_olufsen_impedance(self,
                                  k1 = 19992500, # g/cm/s^2
                                  k2 = -25.5267, # 1/cm 
                                  k3 = 1104531.4909089999, # g/cm/s^2
                                  n_procs=None
                                  ):
        '''
        compute the impedance of the structured tree accordin to Olufsen et al. (2000)
        '''

        # initialize class params
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.n_procs = n_procs

        # inflow to tree must be periodic!!
        # if sum(np.gradient(self.Q_in)) == 0.0:
        #     raise ValueError("Tree inflow must be periodic to compute the impedance")
        
        # tsteps = self.simparams.number_of_time_pts_per_cardiac_cycle

        # needs to be a multiple of 2 (we use 0.6 because it is 1.2/2)
        # tsteps = int(math.ceil(self.simparams.number_of_time_pts_per_cardiac_cycle * 0.6)) * 2

        tsteps = len(self.time)

        # print(f'timesteps for impedance: {tsteps}')

        # print(f'number of time points per cardiac cycle: {self.simparams.number_of_time_pts_per_cardiac_cycle}, len(self.time): {len(self.time)}')

        # print(f'k1: {k1}, k2: {k2}, k3: {k3}')
        # tsteps = int(512 * 0.6) * 2

        # tsteps = len(self.time)


        period = max(self.time) * self.q / self.Lr**3

        df = 1 / period

        omega = [i * df * 2 * np.pi for i in range(-tsteps//2, tsteps//2)] # angular frequency vector


        # we need to remove the zero frequency, because we will insert that later at the first index


        # initialize the impedance
        Z_om = np.zeros(len(omega), dtype=complex)
        # loop through POSITIVE frequencies and calculate impedance
        
        ## PARALLELIZED ##
        # with Pool(24) as p:       
        #     Z_om = p.map(self.root.z0, omega)

        
        
        if n_procs is not None:
            start = time.time()
            # make the partial function
            z0_w_stiffness = partial(self.root.z0_olufsen, k1=k1, k2=k2, k3=k3)
            # parallelize the computation of the impedance
            with Pool(n_procs) as p:
                Z_om[:tsteps//2+1] = np.conjugate(p.map(z0_w_stiffness, [abs(w) for w in omega[:tsteps//2+1]]))
            
            end = time.time()
            print(f'this parallelized process took {end - start} seconds for d_root = {self.initial_d} and d_min = {self.d_min}')
        else:
            ## UNPARALLELIZED ##
            start = time.time()
            for k in range(0, tsteps//2+1):
                # if (k) % 100 == 0:
                # print(f'computing root impedance for timestep {k} of {tsteps//2}')
                # compute impedance at the root vessel
                # we cannot have a negative number here so we take positive frequency and then conjugate
                Z_om[k] = np.conjugate(self.root.z0_olufsen(abs(omega[k]),
                                                            k1 = k1,
                                                            k2 = k2,
                                                            k3 = k3
                                                            ))
            
            end = time.time()
            # print(f'this UNparallelized process took {end - start} seconds')

            
        # apply self-adjoint property of the impedance
        Z_om_half = Z_om[:tsteps//2]
        # add negative frequencies
        Z_om[tsteps//2+1:] = np.conjugate(np.flipud(Z_om_half[:-1]))


        # dimensionalize omega
        omega = [w * self.q / self.Lr**3 for w in omega]

        Z_om = np.fft.ifftshift(Z_om)

        print(f'Z(w=0) = {Z_om[0]}')

        Z_t = np.fft.ifft(Z_om)

        self.Z_t = np.real(Z_t)

        return self.Z_t, self.time


    def create_impedance_bc(self, name, Pd: float = 0.0):
        '''
        create an impedance BC object
        
        :param name: name of the boundary condition
        :param Pd: distal pressure in dyn/cm2'''

        impedance_bc = BoundaryCondition({
            "bc_name": f"{name}",
            "bc_type": "IMPEDANCE",
            "bc_values": {
                "tree": self.name,
                "Z": self.Z_t.tolist(),
                "t": self.time,
                "Pd": Pd,
            }
        })

        return impedance_bc


    def match_RCR_to_impedance(self):
        '''
        find the RCR parameters to match the impedance from an impedance tree.'''

        # get the impedance of the structured tree if self.Z_t is None
        if self.Z_t is None:
            self.compute_olufsen_impedance()

        
        # loss function for optimizing RCR parameters
        def loss_function(params):
            '''
            loss function for optimizing RCR parameters

            :param params: RCR parameters [Rp, C, Rd]
            '''
            # compute impedance from the RCR parameters
            Rp, C, Rd = params
            # calculate the impedance from the RCR parameters
            tsteps = len(self.time)
            period = max(self.time) * self.q / self.Lr**3
            df = 1 / period
            omega = [i * df * 2 * np.pi for i in range(-tsteps//2, tsteps//2)] # angular frequency vector

            Z_om = np.zeros(len(omega), dtype=complex)

            # Z_om[:tsteps//2+1] = np.conjugate([((1j * w * Rp * Rd * C) + (Rp + Rd)) / ((1j * w * Rd * C) + 1) for w in omega[:tsteps//2+1]])
            # def Z(w, Rp, C, Rd):
            #     return ((1j * w * Rp * Rd * C) + (Rp + Rd)) / ((1j * w * Rd * C) + 1)
            Z_om[:tsteps//2+1] = np.array([np.sqrt(((Rd + Rp) ** 2 + (w * Rp * Rd * C) ** 2) / (1 + (w * Rd * C) ** 2)) for w in omega[:tsteps//2+1]])

            # apply self-adjoint property of the impedance
            Z_om_half = Z_om[:tsteps//2]
            # add negative frequencies
            Z_om[tsteps//2+1:] = np.conjugate(np.flipud(Z_om_half[:-1]))


            # dimensionalize omega
            omega = [w * self.q / self.Lr**3 for w in omega]

            Z_om = np.fft.ifftshift(Z_om)

            print(f'Z(w=0) = {Z_om[0]}')

            Z_rcr = np.fft.ifft(Z_om)

            self.Z_rcr = np.real(Z_rcr)

            # calculate the squared difference between the impedance from the RCR parameters and the impedance from the structured tree
            loss = np.sum((self.Z_t - Z_rcr)**2)

            print(f'loss: {loss}')

            return loss
        
        # initial guess for RCR parameters
        initial_guess = [100.0, 0.0001, 900.0]

        # optimize the RCR parameters
        bounds = Bounds(lb=[0.0, 0.0, 0.0], ub=[np.inf, np.inf, np.inf])
        # bounds = Bounds(lb=[-np.inf, -np.inf, -np.inf], ub=[np.inf, np.inf, np.inf])
        result = minimize(loss_function, initial_guess, method='Nelder-Mead', bounds=bounds)

        print(f'optimized RCR parameters: {result.x}')

        return result.x


    def adapt_constant_wss(self, Q, Q_new):
        R_old = self.root.R_eq  # calculate pre-adaptation resistance

        def constant_wss(d, Q=Q, Q_new=Q_new):
            '''
            function for recursive algorithm to update the vessel diameter based on constant wall shear stress assumption

            :param d: diameter of the vessel
            :param Q: original flowrate through the vessel
            :param Q_new: post-operative flowrate through the model
            
            :return: length of the updated diameter
            '''
            # adapt the diameter of the vessel based on the constant shear stress assumption

            return (Q_new / Q) ** (1 / 3) * d

        def update_diameter(vessel, update_func):
            '''
            preorder traversal to update the diameters of all the vessels in the tree  
            
            :param vessel: TreeVessel instance
            :param update_func: function to update vessel diameter based on constant wall shear stress asssumption
            '''

            if vessel:
                # recursive step
                update_diameter(vessel.left, update_func)
                update_diameter(vessel.right, update_func)

                vessel.d = update_func(vessel.d)

        
        # recursive step
        update_diameter(self.root, constant_wss)

        self.create_block_dict()

        R_new = self.root.R_eq  # calculate post-adaptation resistance

        return R_old, R_new


    def optimize_tree_diameter(self, resistance=None, log_file=None, d_min=0.0049, pries_secomb=False):
        """ 
        Use Nelder-Mead to optimize the diameter and number of vessels with respect to the desired resistance
        
        :param resistance: resistance value to optimize against
        :param log_file: optional path to log file
        :param d_min: minimum diameter of the vessels
        :param pries_secomb: True if the pries and secomb model is used to adapt the vessels, so pries and secomb integration
            is performed at every optimization iteration
        """
        
        # write_to_log(log_file, "Optimizing tree diameter for resistance " + str(self.params["bc_values"]["R"]) + " with d_min = " + str(d_min) + "...")
        # initial guess is oulet r
        d_guess = self.diameter / 2

        # get the resistance if it is RCR or Resistance BC
        if resistance is None:
            if "Rp" in self.params["bc_values"].keys():
                R0 = self.params["bc_values"]["Rp"] + self.params["bc_values"]["Rd"]
            else:
                R0 = self.params["bc_values"]["R"]
        else:
            R0 = resistance

        # define the objective function to be minimized
        def r_min_objective(diameter, d_min, R0):
            '''
            objective function for optimization

            :param diameter: inlet diameter of the structured tree

            :return: squared difference between target resistance and built tree resistance
            '''
            # build tree
            self.build_tree(diameter[0], d_min=d_min, optimizing=True)


            # get equivalent resistance
            R = self.root.R_eq

            # calculate squared relative difference
            loss = ((R0 - R) / R0) ** 2
            return loss

        # define optimization bound (lower bound = r_min, which is the termination diameter)
        bounds = Bounds(lb=0.005)

        # perform Nelder-Mead optimization
        d_final = minimize(r_min_objective,
                           d_guess,
                           args=(d_min, R0),
                           options={"disp": True},
                           method='Nelder-Mead',
                           bounds=bounds)
        
        R_final = self.root.R_eq

        # write_to_log(log_file, "     Resistance after optimization is " + str(R_final))
        # write_to_log(log_file, "     the optimized diameter is " + str(d_final.x[0]))
        write_to_log(log_file, "     the number of vessels is " + str(len(self.block_dict["vessels"])) + "\n")

        if pries_secomb:
            self.pries_n_secomb = PriesnSecomb(self)


        return d_final.x, R_final
    

    def add_hemodynamics_from_outlet(self, Q_outlet, P_outlet):
        '''
        add hemodynamics from the outlet of the 0D model to the structured tree
        
        :param Q_outlet: flow at the outlet of the 0D model
        :param P_outlet: pressure at the outlet of the 0D model
        '''

        # make the array length 2 for steady state bc
        if len(Q_outlet) == 1:
            Q_outlet = [Q_outlet[0],] * 2
        
        if len(P_outlet) == 1:
            P_outlet = [P_outlet[0],] * 2

        # add the flow and pressure values to the structured tree
        self.params["Q_in"] = Q_outlet
        self.params["P_in"] = P_outlet

        # this is redundant but whatever
        self.block_dict["Q_in"] = Q_outlet
        self.block_dict["P_in"] = P_outlet
    

    def optimize_alpha_beta(self, Resistance=5.0, log_file=None):
        """ 
        use constrained optimization to optimize the diameter, alpha and beta values of the tree
        
        :param Resistance: resistance value to optimize against
        :param log_file: optional path to log file
        """

        def r_min_objective(params):
            '''
            objective function for optimization

            :param radius: inlet radius of the structured tree

            :return: squared difference between target resistance and built tree resistance
            '''

            # build structured tree
            self.build_tree(params[0], optimizing=True, alpha=params[1], beta=params[2])
            
            # get the equivalent resistance
            R = self.root.R_eq
            
            # calculate squared difference to minimize
            R_diff = (Resistance - R) ** 2

            return R_diff

        # initial guess is outlet r and alpha, beta values from literature
        r_guess = self.initialD / 2
        params_guess = np.array([r_guess, 0.9, 0.6]) # r, alpha, beta initial guess

        # define optimization constraints
        param_constraints = LinearConstraint([[0, 0, 0], [0, 1, 1], [0, 1, -1.5]], [0.0, 1, 0], [np.inf, np.inf, 0])
        param_bounds = Bounds(lb=[0.049, 0, 0], ub=[np.inf, 1, 1], keep_feasible=True)

        # optimization step: use trust-constr since the optimization is constrained
        r_final = minimize(r_min_objective,
                           params_guess,
                           options={"disp": True},
                           method='trust-constr',
                           constraints=param_constraints,
                           bounds=param_bounds)
        
        R_final = self.root.R_eq

        # write the optimization results to log file
        write_to_log(log_file, "     Resistance after optimization is " + str(R_final) + "\n")
        write_to_log(log_file, "     the optimized radius is " + str(r_final.x[0]) + "\n")
        write_to_log(log_file, "     the optimized alpha value is " + str(r_final.x[1])  + "\n")
        write_to_log(log_file, "     the optimized alpha value is " + str(r_final.x[2])  + "\n")

        return r_final.x[0], R_final


    def create_bcs(self):
        ''''
        create the inflow and distal pressure BCs. This function will prepare a block_dict to be run by svzerodplus
        '''
        self.block_dict["boundary_conditions"] = [] # erase the previous boundary conditions
        timesteps = len(self.Q_in) # identify the number of timesteps in the flow boundary condition

        self.block_dict["boundary_conditions"].append(
            {
                    "bc_name": "INFLOW",
                    "bc_type": "FLOW",
                    "bc_values": {
                        "Q": self.Q_in,
                        "t": np.linspace(0.0, 1.0, num=timesteps).tolist()
                    }
                },
        )

        for vessel_config in self.block_dict["vessels"]:
            if "boundary_conditions" in vessel_config:
                if "outlet" in vessel_config["boundary_conditions"]:
                    self.block_dict["boundary_conditions"].append(
                        {
                        "bc_name": "P_d" + str(vessel_config["vessel_id"]),
                        "bc_type": "PRESSURE",
                        "bc_values": {
                            "P": [self.Pd,] * 2,
                            "t": np.linspace(0.0, 1.0, num=timesteps).tolist()
                            }
                        }
                    )


    def count_vessels(self):
        '''
            count the number vessels in the tree
        '''
        return len(self.block_dict["vessels"])


    def plot_stiffness(self, path='stiffness_plot.png'):
        '''
        plot the value of Eh/r from d_root to d_min for the tree stiffness value
        '''

        d = np.linspace(self.initial_d, self.d_min, 100)
        Eh_r = np.zeros(len(d))

        # compute Eh/r for each d
        for i in range(len(d)):
            Eh_r[i] = self.k1 * np.exp(self.k2 * d[i] / 2) + self.k3

        plt.plot(d, Eh_r)
        plt.yscale('log')
        plt.xlabel('diameter (cm)')
        plt.ylabel('Eh/r')
        plt.title('Eh/r vs. diameter')
        plt.savefig(path)



        
    @property
    def R(self):
        '''
        :return: the equivalent resistance of the tree

        tree.root.R_eq may work better in most cases since that is a value rather than a method
        '''
        if self.root is not None:
            self._R = self.root.R_eq
        return self._R
    

    def adapt_pries_secomb(self):
        '''
        integrate pries and secomb diff eq by Euler integration for the tree until dD reaches some tolerance (default 10^-5)

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

        :return: equivalent resistance of the tree
        '''




        return self.root.R_eq


    def simulate(self, 
                 Q_in: list = [1.0, 1.0],
                 Pd: float = 1.0,
                 density = 1.06,
                 number_of_cardiac_cycles=1,
                 number_of_time_pts_per_cardiac_cycle = 10,
                 viscosity = 0.04):
        '''
        simulate the structured tree

        :param Q_in: flow at the inlet of the tree
        :param P_d: pressure at the distal outlets of the tree
        ** simulation parameters: **
        :param density: density of the blood [g/cm3]
        :param number_of_cardiac_cycles: number of cardiac cycles to simulate
        :param number_of_time_pts_per_cardiac_cycle: number of time points per cardiac cycle
        :param viscosity: viscosity of the blood [g/cm/s]
        '''

        if self.simparams is None:
            self.simparams = SimParams({
                "density": density,
                "model_name": self.name,
                "number_of_cardiac_cycles": number_of_cardiac_cycles,
                "number_of_time_pts_per_cardiac_cycle": number_of_time_pts_per_cardiac_cycle,
                "viscosity": viscosity
            })
            self.block_dict["simulation_parameters"] = self.simparams.to_dict()

        if self.Q_in is None:
            self.Q_in = Q_in

        if self.Pd is None:
            self.Pd = Pd
        
        # create solver config from StructuredTree and get tree flow result
        self.create_bcs()

        result = run_svzerodplus(self.block_dict)

        # assign flow result to TreeVessel instances to allow for visualization, adaptation, etc.

        assign_flow_to_root(result, self.root)

        return result
    

    def visualize(self):
        '''
        TO BE IMPLEMENTED
        '''
        pass


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