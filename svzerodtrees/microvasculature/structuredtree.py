import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
from .treevessel import TreeVessel
from ..utils import *
from .utils import *
from ..io.blocks import *
from .compliance import *
from multiprocessing import Pool
import json
import pickle
from functools import partial
import time

class StructuredTree:
    """
    Structured tree representing microvascular adaptation at the outlet of a 0D model.
    Can be initialized from primitive inputs or from a pre-defined config tree.

    Required inputs:
    - name: a unique identifier for the structured tree
    - time: time vector (list or np.ndarray) used for pressure and flow inputs
    - simparams: simulation parameter object containing physical constants
    """

    def __init__(self,
                 name: str,
                 time: list[float],
                 simparams: SimParams,
                 diameter: float = 0.5,
                 compliance_model: ComplianceModel = None,
                 R: float = None,
                 C: float = None,
                 Pd: float = 0.0,
                 P_in: list[float] = None,
                 Q_in: list[float] = None,
                 tree_config: dict = None,
                 root: TreeVessel = None):
        # --- Required core parameters ---
        self.name = name
        self.time = time
        self.simparams = simparams

        # --- Physical constants ---
        self.viscosity = simparams.viscosity
        self.density = 1.055  # fixed unless overridden in vessel-level params

        # --- Geometry and hemodynamics ---
        self.diameter = diameter
        self._R = R
        self.C = C
        self.Pd = Pd
        self.P_in = P_in
        self.Q_in = Q_in

        # --- Dimensionless reference values ---
        self.q = 10.0    # reference flow [cm³/s]
        self.Lr = 1.0    # reference length [cm]
        self.g = 981.0   # gravity [cm/s²]

        # --- Tree structure ---
        self.generations = 0
        self.root = None

        # --- Compliance model ---
        self.compliance_model = compliance_model if compliance_model else ConstantCompliance(1e5)

        if tree_config:
            if root is None:
                raise ValueError("tree_config provided but no root TreeVessel instance passed.")
            self.root = root
            self.block_dict = tree_config
        else:
            self.block_dict = {
                "name": name,
                "initial_d": diameter,
                "P_in": P_in,
                "Q_in": Q_in,
                "boundary_conditions": [],
                "simulation_parameters": simparams.to_dict(),
                "vessels": [],
                "junctions": [],
                "adaptations": 0
            }


    @classmethod
    def from_outlet_vessel(cls,
                        vessel: Vessel,
                        simparams: SimParams,
                        bc: BoundaryCondition,
                        tree_exists: bool = False,
                        root: TreeVessel = None,
                        P_outlet=0.0,
                        Q_outlet=0.0,
                        time: list = None) -> "StructuredTree":
        """
        Create StructuredTree from a 0D outlet vessel and boundary condition.
        """
        P_outlet, _ = _ensure_list_signal(P_outlet)
        Q_outlet, time = _ensure_list_signal(Q_outlet, time or [0.0, 1.0])

        if "Rp" in bc.values:
            R = bc.values["Rp"] + bc.values["Rd"]
            C = bc.values.get("C", 0.0)
        else:
            R = bc.values["R"]
            C = None

        name = f"OutletTree{vessel.branch}"
        Pd = bc.values.get("Pd", 0.0)

        return cls(
            name=name,
            diameter=vessel.diameter,
            R=R,
            C=C,
            Pd=Pd,
            P_in=P_outlet,
            Q_in=Q_outlet,
            time=time,
            simparams=simparams,
            tree_config=vessel if tree_exists else None,
            root=root if tree_exists else None
        )

    @classmethod
    def from_bc_config(cls,
                    bc: BoundaryCondition,
                    simparams: SimParams,
                    diameter: float,
                    P_outlet=0.0,
                    Q_outlet=0.0,
                    time: list = None) -> "StructuredTree":
        """
        Create StructuredTree from a boundary condition only (no vessel metadata).
        """
        P_outlet, _ = _ensure_list_signal(P_outlet)
        Q_outlet, time = _ensure_list_signal(Q_outlet, time or [0.0, 1.0])

        if "Rp" in bc.values:
            R = bc.values["Rp"] + bc.values["Rd"]
            C = bc.values.get("C", 0.0)
        else:
            R = bc.values["R"]
            C = None

        name = f"OutletTree_{bc.name}"
        Pd = bc.values.get("Pd", 0.0)

        return cls(
            name=name,
            diameter=diameter,
            R=R,
            C=C,
            Pd=Pd,
            P_in=P_outlet,
            Q_in=Q_outlet,
            time=time,
            simparams=simparams
        )

# **** I/O METHODS ****

    def to_dict(self):
        '''
        convert the StructuredTree instance parameters to a dictionary
        '''

        params = {
            "name": self.name,
            "initial_d": self.initial_d,
            "d_min": self.d_min,
            "lrr": self.lrr,
            "n_procs": self.n_procs,
            "compliance": {
                "model": self.compliance_model.description(),
                "params": self.compliance_model.params,
            }
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
        self.root = TreeVessel.create_vessel(0, 0, initial_d, density=1.055, lrr=lrr, compliance_model=self.compliance_model)
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
                current_vessel.left = TreeVessel.create_vessel(vessel_id, next_gen, left_dia, density=1.055, lrr=lrr, compliance_model=self.compliance_model)
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
                current_vessel.right = TreeVessel.create_vessel(vessel_id, next_gen, right_dia, density=1.055, lrr=lrr, compliance_model=self.compliance_model)
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
                                  n_procs=None,
                                  tsteps=None
                                  ):
        '''
        compute the impedance of the structured tree accordin to Olufsen et al. (2000)
        '''

        # initialize class params
        # self.k1 = k1
        # self.k2 = k2
        # self.k3 = k3
        self.n_procs = n_procs

        if tsteps is None:
            tsteps = len(self.time)
        else:
            tsteps = int(tsteps)

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
            z0_w_stiffness = partial(self.root.z0_olufsen)
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
                Z_om[k] = np.conjugate(self.root.z0_olufsen(abs(omega[k])))

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


    def create_impedance_bc(self, name, tree_id, Pd: float = 0.0):
        '''
        create an impedance BC object
        
        :param name: name of the boundary condition
        :param tree_id: id of the tree in the list of trees
        :param Pd: distal pressure in dyn/cm2'''

        print(f'creating impedance bc for tree {self.name}')

        impedance_bc = BoundaryCondition({
            "bc_name": f"{name}",
            "bc_type": "IMPEDANCE",
            "bc_values": {
                "tree": tree_id,
                "Z": self.Z_t.tolist(),
                "t": self.time,
                "Pd": Pd,
            }
        })

        return impedance_bc
    
    def create_resistance_bc(self, name, Pd: float = 0.0):
        '''
        create a resistance bc from the tree using the trees root equivalent resistance'''

        self.Pd = Pd

        resistance_bc = BoundaryCondition({
            "bc_name": f"{name}",
            "bc_type": "RESISTANCE",
            "bc_values": {
                "R": self.root.R_eq,
                "Pd": Pd,
            }
        })

        return resistance_bc
    
    def compute_homeostatic_state(self, Q):
        '''
        compute the homeostatic state of the structured tree by adapting the diameter of the vessels to a given flow rate Q
        '''
    
        print("computing homeostatic state for structured tree...")
        # simulation with initial flow
        preop_result = self.simulate(Q_in=[Q, Q], Pd=self.Pd)

        # assign homeostatic wss and ims
        def assign_homeostatic_wss_ims(vessel):
            '''
            recursive step to assign homeostatic wss and ims to the vessel
            '''

            if vessel:
                # get flow from postop result
                vessel_name = f"branch{vessel.id}_seg0"
                Q = get_branch_result(preop_result, 'flow_in', vessel_name)
                P = get_branch_result(preop_result, 'pressure_in', vessel_name)
                # adapt the diameter based on the flow
                vessel.wss_h = vessel.wall_shear_stress(Q)
                vessel.ims_h = vessel.intramural_stress(P)

                assign_homeostatic_wss_ims(vessel.left)
                assign_homeostatic_wss_ims(vessel.right)

        assign_homeostatic_wss_ims(self.root)

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


    def adapt_constant_wss(self, Q, Q_new, method='cwss', n_iter=1):
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
        
        def constant_wss_ims(d, Q=Q, Q_new=Q_new):
            '''
            update the diameter based on constant wall shear stress and intramural stress'''

            # dDdt = K_tau_d * ()

            pass

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
        for i in range(n_iter):
            print(f'performing cwss adaptation iteration {i}')
            self.initial_d = constant_wss(self.initial_d, Q=Q, Q_new=Q_new)
            update_diameter(self.root, constant_wss)

        self.create_block_dict()

        R_new = self.root.R_eq  # calculate post-adaptation resistance

        return R_old, R_new


    def adapt_wss_ims(self, Q, Q_new, n_iter=100):
        '''
        adapt the diameter of the structured tree based on the flowrate through the model

        :param Q: original flowrate through the vessel
        :param Q_new: post-operative flowrate through the model
        :param method: adaptation method to use
        :param n_iter: number of iterations to perform

        :return: pre-adaptation and post-adaptation resistance
        '''

        preop_result = self.simulate(Q_in=[Q, Q], Pd=self.Pd)

        assign_flow_to_root(preop_result, self.root)

        postop_result = self.simulate(Q_in=[Q_new, Q_new], Pd=self.Pd)

        def adapt_vessel(vessel):
            '''
            recursive step to adapt the vessel diameter'''

            if vessel:
                # get flow from postop result
                vessel_name = f"branch{vessel.id}_seg0"
                Q_new = get_branch_result(postop_result, 'flow_in', vessel_name)
                P_new = get_branch_result(postop_result, 'pressure_in', vessel_name)
                # adapt the diameter based on the flow
                vessel.adapt_cwss_ims(Q_new, P_new, n_iter=n_iter, verbose=False)
                # recursive step
                adapt_vessel(vessel.left)
                adapt_vessel(vessel.right)
        
        print(f"adapting tree diameter with Q = {Q} Q_new = {Q_new}")

        adapt_vessel(self.root)

        adapted_result = self.simulate(Q_in=[Q_new, Q_new], Pd=self.Pd)
        assign_flow_to_root(adapted_result, self.root)


    def adapt_wss_ims_method2(self, Q, Q_new, n_iter=100):
        '''
        adapt the diameter of the structured tree based on the flowrate through the model using a different method

        :param Q: original flowrate through the vessel
        :param Q_new: post-operative flowrate through the model
        :param n_iter: number of iterations to perform

        :return: pre-adaptation and post-adaptation resistance
        '''
        print(f"running preop tree simulation with Q = {Q} and Pd = {self.Pd}")
        preop_result = self.simulate(Q_in=[Q, Q], Pd=self.Pd)
        assign_flow_to_root(preop_result, self.root)

        print(f"running postop tree simulation with Q = {Q_new} and Pd = {self.Pd}")
        iteration_result = self.simulate(Q_in=[Q_new, Q_new], Pd=self.Pd)

        def adapt_vessel(vessel):
            '''
            recursive step to adapt the vessel diameter'''

            if vessel:
                # get flow from postop result
                vessel_name = f"branch{vessel.id}_seg0"
                Q_new = get_branch_result(iteration_result, 'flow_in', vessel_name)
                P_new = get_branch_result(iteration_result, 'pressure_in', vessel_name)
                # adapt the diameter based on the flow
                vessel.adapt_cwss_ims(Q_new, P_new, n_iter=1)
                # recursive step
                adapt_vessel(vessel.left)
                adapt_vessel(vessel.right)
        
        print(f"adapting tree diameter with Q = {Q} Q_new = {Q_new}")

        for i in range(n_iter):
            print(f"adapting {self.count_vessels()} vessel diameters for tree {self.name}...")
            adapt_vessel(self.root)
            # after adapting the vessel diameters, we need to simulate the tree again to get the new flow and pressure values
            iteration_result = self.simulate(Q_in=[Q_new, Q_new], Pd=self.Pd)
            # assign_flow_to_root(iteration_result, self.root)

            print(f"adaptation iteration {i+1} of {n_iter}, root diameter = {self.root.d}, root resistance = {self.root.R_eq}")
        


        adapted_result = self.simulate(Q_in=[Q_new, Q_new], Pd=self.Pd)
        assign_flow_to_root(adapted_result, self.root)


    def optimize_tree_diameter(self, resistance=None, log_file=None, d_min=0.01, pries_secomb=False):
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
            # self.pries_n_secomb = PriesnSecomb(self)
            pass


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
            Eh_r[i] = self.compliance_model.evaluate(d[i] / 2)

        if path is None:
            return d, Eh_r
        else:
            plt.figure()
            plt.plot(d, Eh_r)
            plt.yscale('log')
            plt.xlabel('diameter (cm)')
            plt.ylabel('Eh/r (mmHg)')
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

        self.Q_in = Q_in

        self.Pd = Pd
        
        # create solver config from StructuredTree and get tree flow result
        self.create_bcs()

        # result = run_svzerodplus(self.block_dict)

        result = pysvzerod.simulate(self.block_dict)

        # assign flow result to TreeVessel instances to allow for visualization, adaptation, etc.
        # currently this conflicts with adaptation computation, where we do not want to assign flow to root every time we simulate
        # assign_flow_to_root(result, self.root)

        return result

    def enumerate_vessels(self, start_idx=0):
        """Return a deterministic DFS ordering and stamp each vessel with .idx."""

        vessel_order = []
        def _dfs(v):
            if v is None:
                return
            v.idx = len(vessel_order) + start_idx       # store once, forever
            vessel_order.append(v)
            _dfs(v.left)
            _dfs(v.right)
        
        _dfs(self.root)
        return vessel_order


def _ensure_list_signal(signal, fallback_time=[0.0, 1.0]):
    """Ensure signal is a list; return default time vector if needed."""
    if not isinstance(signal, list):
        return [signal] * 2, fallback_time
    return signal, fallback_time if len(signal) == 1 else None