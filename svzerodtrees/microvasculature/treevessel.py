import numpy as np
import math
from scipy.special import jv
from numba import jit

# class for recursive generation of a structured tree

class TreeVessel:
    # blood vessel class for binary tree recusion operations
    def __init__(self, params: dict, name: str = None, lrr=50.0):
        '''
        :param params: dictionary of TreeVessel class parameters
        :param name: name of the vessel, which follows the svzerodplus naming convention
        '''
        self.name = name # name of the, tree only has a str in the root node
        self.params = params # vessel params dict
        self._d = self.params["vessel_D"] # diameter in cm
        self._r = self._d / 2 # radius in cm
        self.l = self.params["vessel_length"] # length in cm
        self.id = self.params["vessel_id"]
        self.gen = self.params["generation"]
        self.eta = self.params["viscosity"]
        self.density = self.params["density"]
        self._R = self.params["zero_d_element_values"].get("R_poiseuille")
        self._R_eq = self._R # this is the initial value, not dependent on left and right
        self.C = self.params["zero_d_element_values"].get("C")
        self.L = self.params["zero_d_element_values"].get("L")
        self.a = np.pi * self._d ** 2 / 4 # cross sectional area

        self.h = self._r / 10 # currently assume thin wall, could be higher with remodelling!

        ### *** PARAMETERS FOR ADAPTATION *** ###
        # diameter gain parameters for remodeling
        self.K_tau_r = 1e-6
        self.K_sig_r = 1e-6
        # thickness gain parameters
        self.K_tau_h = 1e-5
        self.K_sig_h = 1e-5

        # homeostatic values
        self.wss_h = None # homeostatic wall shear stress
        self.ims_h = None # homeostatic intramural stress

        # global array idk
        self.idx = None # index in the flattened array of vessels, to be set later

        # adaptation parameters
        self.r_adapt = self._r
        self.h_adapt = self.h
        self.dt = 1e-5 # timestep for integration


        self.lrr = lrr

        # flow values, based on Poiseulle assumption
        self.P_in = 0.0
        self.Q = 0.0
        self.P_out = 0.0

        # daughter segments for recursion
        self._left = None
        self._right = None

        # if vessel is collapsed, no daughters
        self._collapsed = False


    @classmethod
    def create_vessel(cls, id, gen, diameter, density, lrr=None):
        '''
        class method to create a TreeVessel instance

        :param id: int describing the vessel id
        :param gen: int describing the generation of the tree in which the TreeVessel exists
        :param diameter: diameter of the vessel
        :param density: density of the blood within the vessel

        :return: TreeVessel instance
        '''

        # Viscosity is always governed by fahraeus lindqvist effect
        # viscosity = cls.fl_visc(cls, diameter, H_d=H_d)

        viscosity = 0.049 # poise, cm2/s
        
        # initialize the 0D parameters of the tree
        r = diameter / 2
        if lrr is None:
            l = 12.4 * r ** 1.1  # from ingrid's paper - does this remain constant throughout adaptation?
        else:
            l = lrr * r
        R = 8 * viscosity * l / (np.pi * r ** 4)
        C = 0.0  # to implement later
        L = 0.0  # to implement later

        # create name to match svzerodplus naming convention
        name = "branch" + str(id) + "_seg0" 

        # generate a config file to run svzerodplus on the TreeVessel instances
        vessel_params = {"vessel_id": id,  # mimic input json file
                       "vessel_length": l,
                       "vessel_name": name,
                       "zero_d_element_type": "BloodVessel",
                       "zero_d_element_values": {
                           "R_poiseuille": R,
                           "C": C,
                           "L": L,
                           "stenosis_coefficient": 0.0
                       },
                       "vessel_D": diameter,
                       "generation": gen,
                       "viscosity": viscosity,
                       "density": density,
                       }

        return cls(params=vessel_params, lrr=lrr)


    ####### beginning of property setters to dynamically update various class properties #######

    @property
    def left(self):
        return self._left

    # update R_eq based on the added left and right vessels
    @left.setter 
    def left(self, new_left):
        self._left = new_left
        if self.right is not None:
            self._update_R_eq()

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, new_right):
        self._right = new_right
        self._update_R_eq()

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, new_R):
        self._R = new_R
        if self.left is not None and self.right is not None:
            self._update_R_eq() # update R_eq if R is updated

    @property
    def R_eq(self):
        if self.left is not None and self.right is not None:
            self._update_R_eq()
        else:
            self._R_eq = self._R
        return self._R_eq

    def _update_R_eq(self):
        self._R_eq = self._R + (1 / (self._left.R_eq ** -1 + self._right.R_eq ** -1))

    @property
    def collapsed(self):
        return self._collapsed
    # add the distal pressure reference bc if the vessel is determined to be collapsed
    @collapsed.setter
    def collapsed(self, new_collapsed):
        self._collapsed = new_collapsed
        self.add_collapsed_bc()

    # method to change d and zero d parameters which are dependent on d
    @property
    def d(self):
        return self._d

    @d.setter
    def d(self, new_d):
        self._d = new_d
        self._r = new_d / 2
        self.update_vessel_params()

    @property 
    def r(self):
        return self._r
    
    @r.setter
    def r(self, new_r):
        self._r = new_r
        self._d = new_r * 2
        self.update_vessel_params()

    ####### end of property setters used to dynamically update various class attributes #######

    def wall_shear_stress(self, Q=None, r=None, mean=True):
        '''
        calculate the wall shear stress in the vessel
        :param mean: boolean to determine if the mean wall shear stress should be calculated
        :return: wall shear stress
        '''

        if Q is None:
            Q = self.Q

        if r is None:
            r = self.r

        if mean:
            t_w = np.mean(Q * 4 * self.eta / (np.pi * r ** 3))
        else:
            t_w = Q * 4 * self.eta / (np.pi * r ** 3)

        return t_w
    
    def intramural_stress(self, P=None, h=None, mean=True):
        '''
        calculate the intramural stress in the vessel
        :return: intramural stress
        '''

        if P is None:
            P = self.P_in

        if h is None:
            h = self.h

        # intramural stress
        if mean:
            sigma_theta = np.mean(P * self.r / self.h)
        else:
            sigma_theta = P * self.r / self.h

        return sigma_theta
    
    def adapt_cwss_ims(self, Q_new, P_new, n_iter=100, verbose=False):

        '''
        adapt the diameter of the vessel based on the wall shear stress and intramural stress
        :param Q_new: new flow rate
        :param P_new: new pressure
        :return: change in diameter
        '''

        r_initial = self.r
        h_initial = self.h
        # calculate homeostatic values
        wss_h = self.wall_shear_stress(Q=self.Q)
        ims_h = self.intramural_stress(P=self.P_in)

        def integrate_adaptation(Q_new, P_new, iter):
            # calculate new wall shear stress and intramural stress
            wss_new = self.wall_shear_stress(Q=Q_new)
            ims_new = self.intramural_stress(P=P_new)

            # calculate change in diameter and thickness
            dr = (k_tau_r * (wss_new - wss_h) + k_sig_r * (ims_new - ims_h)) * self.dt # * self.r 
            dh = (-k_tau_h * (wss_new - wss_h) + k_sig_h * (ims_new - ims_h)) * self.dt # * self.h

            # update radius and thickness
            self.r += dr
            self.h += dh

            if self.r < 0.0:
                raise ValueError(f"Adaptation resulted in negative radius: {self.r}. Check adaptation parameters or initial conditions.")
            if self.h < 0.0:
                raise ValueError(f"Adaptation resulted in negative thickness: {self.h}. Check adaptation parameters or initial conditions.")

            if verbose:
                if (iter + 1) % 100 == 0:
                    print(f"\nAdaptation step {iter + 1}: wss_h={wss_h}, wss_new={wss_new}, ims_h={ims_h}, ims_new={ims_new}")
                    print(f"dr={dr}, dh={dh}, new radius={self.r}, new thickness={self.h}")
                    print(f"one iteration of cwss adaptation would give r = {np.mean((Q_new / self.Q) ** (1 / 3) * r_initial)}\n")
            
            return dr, dh
        
        if verbose:
            print(f"adapting with Q_old={np.mean(self.Q)}, P_old={np.mean(self.P_in)}, Q_new={np.mean(Q_new)}, P_new={np.mean(P_new)}\n")
        for i in range(n_iter):  # adapt for 100 timesteps
            dr, dh = integrate_adaptation(Q_new, P_new, i)

        if verbose:
            print(f" ***ADAPTATION RESULTS FOR VESSEL {self.name} after {n_iter} iterations *** ")
            print(f"adaptation parameters: dt: {self.dt}, k_wss_r={k_wss_r}, k_ims_r={k_ims_r}, k_wss_h={k_wss_h}, k_ims_h={k_ims_h}")
            print(f"Q_old={np.mean(self.Q)}, Q_new={np.mean(Q_new)}, P_old={np.mean(self.P_in)}, P_new={np.mean(P_new)}")
            print(f"initial radius: {r_initial}, adapted radius: {self.r}. initial thickness: {h_initial}, adapted thickness: {self.h}")
            print(f"final dr: {dr}, final dh: {dh}\n")

        # Approach 1: treat each individual vessel in isolation and adapt each vessel until it converges
        # update wss and ims with each new radius
        

        # Approach 2: update the flow in the network with every timestep. treat the flow from the outlet of hte model but you redistribute based on how vessels are adapting

        # Approach 3: rerun the 3D model every time

        

        # return dr, dh
    




    def calc_zero_d_values(self, diameter, mu):
        '''
        calculate 0D Windkessel parameters based on a vessel diameter

        :param vesselD: vessel diameter
        :param eta: vessel viscosity

        :return: resistance, capacitance, inductiance and vessel length
        '''

        r = diameter / 2
        if self.lrr is None:
            l = 12.4 * r ** 1.1  # from ingrid's paper - does this remain constant throughout adaptation?
        else:
            l = self.lrr * r
        R = 8 * mu * l / (np.pi * r ** 4)
        C = 0.0  # to implement later
        L = 0.0  # to implement later

        return R, C, L, l


    def update_vessel_params(self):
        '''
        update vessel params dict based on changes to vessel diameter
        '''

        # update viscosity
        # self.eta = self.fl_visc(self.d)
        # self.params["viscosity"] = self.eta

        self.eta = 0.049
        
        R, C, L, l = self.calc_zero_d_values(self.d, self.eta)
        self.params["vessel_length"] = l
        self.params["vessel_D"] = self.d
        self.params["zero_d_element_values"]["R_poiseuille"] = R
        self.params["zero_d_element_values"]["C"] = C
        self.params["zero_d_element_values"]["L"] = L
        self.R = R
        if not self.collapsed:
            self._update_R_eq()


    def add_collapsed_bc(self):
        '''
        if the vessel is collapsed, add a distal pressure outlet boundary condition to the vessel config
        '''
        
        self.params["boundary_conditions"] = {
        
            "outlet": "P_d" + str(self.id)
        }


    def adapt_pries_secomb(self, 
                            k_p = 0.68,
                            k_m = .70,
                            k_c = 2.45,
                            k_s = 1.72,
                            L = 1.73,
                            J0 = 27.9,
                            tau_ref = .103,
                            Q_ref = 3.3 * 10 ** -8,
                            dt = 0.01, 
                            H_d=0.45,
                            optimizing_params=False):
        '''
        calculate the diameter change in the vessel based on pries and secomb parameters
        :param ps_params: pries and secomb parameters in the following form [k_p, k_m, k_c, k_s, L (cm), S_0, tau_ref, Q_ref]
            units:
                k_p, k_m, k_c, k_s [=] dimensionless
                L [=] cm
                J0 [=] dimensionless
                tau_ref [=] dyn/cm2
                Q_ref [=] cm3/s
        :param dt: timestep size for euler integration
        :param H_d: hematocrit
        :return: change in vessel diameter
        '''

        ## pries and secomb equations ##

        self.k_p = k_p
        self.k_m = k_m
        self.k_c = k_c
        self.k_s = k_s
        self.L = L
        self.J0 = J0
        self.tau_ref = tau_ref
        self.Q_ref = Q_ref

        self. H_d = H_d # hematocrit

        if self.Q < 0.0:
            raise ValueError("Q must be positive")

        self.S_m = self.k_m * math.log(self.Q_ref / (self.Q * self.H_d) + 1)

        self.tau_e = 50 / 86 * (100 - 86 * math.exp(-5000 * math.log(math.log(4.5 * self.P_in + 10)) ** 5.4) - 14) + 1

        self.S_tau = math.log(self.t_w + self.tau_ref)

        self.S_p = -self.k_p * math.log(self.tau_e)
        
        self.Sbar_c = 0.0
        
        if not self.collapsed:
            if self.left.Sbar_c > 0:
                self.Sbar_c = self.left.S_m + self.right.S_m + self.left.Sbar_c * math.exp(-self.left.l / self.L) + self.right.Sbar_c * math.exp(-self.right.l / self.L)
            else:
                self.Sbar_c = self.left.S_m + self.right.S_m

        self.S_c = self.k_c * (self.Sbar_c / (self.Sbar_c + self.J0))

        self.S_s = -self.k_s

        self.S_tot = self.S_tau + self.S_p + self.S_m + self.S_c + self.S_s

        self.dD = self.d * self.S_tot * dt

        # make sure that the diameter change is positive
        if not optimizing_params:
            if self.d + self.dD > 0.0:
                self.d += self.dD
            else:
                pass

        return self.dD


    def fl_visc(self, diameter, H_d=0.45):
        '''
        calculate the viscosity within a vessel of diameter < 300 um based on empirical relationship describing 
        fahraeus-lindqvist effect

        :param diameter: vessel diameter in cm
        :param H_d: hematocrit
        '''

        diameter = diameter * 100 # convert to um

        u_45 = 6 * math.exp(-0.085 * diameter) + 3.2 - 2.44 * math.exp(-0.06 * diameter ** 0.645)
        C = (0.8 + math.exp(-0.075 * diameter)) * (-1 + (1 + 10 ** -11 * diameter ** 12) ** -1) + (1 + 10 ** -11 * diameter ** 12) ** -1
        rel_viscosity = .012 * (1 + (u_45 - 1) * (((1 - H_d) ** C - 1) / ((1 - 0.45) ** C - 1)) * (diameter / (diameter - 1.1)) ** 2) * (diameter / (diameter - 1.1)) ** 2

        plasma_viscosity = .012 # poise

        viscosity = plasma_viscosity * rel_viscosity

        return viscosity


    def z0(self, omega):
        '''
        calculate the characteristic impedance of the vessel
        '''

        ## computing z_L (impedance at the end of the vessel)
        if self.collapsed:
            # Z_L is terminal resistance
            z_L = 0.0

        else:
            # z_0_left = self.left.z0
            # z_0_right = self.right.z0
            z_L = 1 / (self.left.z0(omega) ** -1 + self.right.z0(omega) ** -1)
        

        if omega == 0.0:
            # if frequency=0, just equivalent resistance with no contribution from compliance
            z_0 = self.R_eq 
        else:
            # compute impedance
        
            ## computing z_0 (impedance at the beginning of the vessel)
            # first, compute Eh/r using empirical relationship from Olufsen et al. (1999)
            Eh_r = self.k1 * np.exp(self.k2 * self.d / 2) + self.k3 # update with new h and r
            # compute compliance using Eh/r relationship
            self.C = 3 * self.a / 2 / Eh_r

            # OTHER PARAMS FROM OLUFSEN MATLAB CODE
            E = 3800
            h = 0.12

            # womersley number
            wom2 = omega * self.density * self.d ** 2 / 4 / self.eta
            wom = self.d / 2 * np.sqrt(omega * self.density / self.eta)
            w_0 = np.sqrt(1j ** 3 * wom2)

            # f_j, bessel function factor
            f_j = 2 * jv(1, w_0) / w_0 / jv(0, w_0)

            # wave propagation velocity
            c = np.sqrt(self.a * (1 - f_j) / (self.density * self.C))

            # scalar value for equation simplification
            g = c * self.C

            z_0 = (1j * g ** -1 * np.sin(omega * self.l / c) + z_L * np.cos(omega * self.l / c)) / \
            (np.cos(omega * self.l / c) + 1j * g * z_L * np.sin(omega * self.l / c))

        # if math.isnan(z_0):
        #     raise Exception('z0 is nan')

        return z_0

    # @jit(forceobj=True)
    def z0_olufsen(self, omega,
                   k1 = 19992500, # g/cm/s^2
                   k2 = -25.5267, # 1/cm 
                   k3 = 1104531.4909089999 # g/cm/s^2
                   ):
        '''
        calculate the characteristic impedance of the vessel
        this function is based on Olufsen's Fortran code.

        :param omega: frequency
        :param trmrst: terminal resistance
        '''

        g = 981.0 # m/s^2
        # rho = 1.055 # kg/m^3
        Lr = 1.0 # characteristic length
        q = 10.0 # characteristic flowrate


        ## computing z_L (impedance at the end of the vessel)
        if self.collapsed:
            # Z_L is terminal resistance
            z_L = 0.0
        else:
            # z_0_left = self.left.z0
            # z_0_right = self.right.z0
            z_L = 1 / (1/self.left.z0_olufsen(omega) + 1/self.right.z0_olufsen(omega))
        

        if omega == 0.0:
            # if frequency=0, just equivalent resistance with no contribution from compliance
            z_0 = self.R_eq 
        else:
            # compute impedance
        
            ## computing z_0 (impedance at the beginning of the vessel)
            # first, compute Eh/r using empirical relationship from Olufsen et al. (1999)
            Eh_r = k1 * np.exp(k2 * self.d / 2) + k3
            # compute compliance using Eh/r relationship
            self.C = 3 * self.a / 2 / Eh_r # compliance / distensibility
            wom = self.d / 2 * np.sqrt(omega * self.density / self.eta) # womersley parameter

            if wom > 3.0:
                g_omega = np.sqrt(self.C*self.a/self.density)* np.sqrt(1.0-2.0/1j**(0.5)/wom*(1.0+1.0/2.0/wom)) 
                c_omega = np.sqrt(self.a/self.C/self.density) * np.sqrt(1.0-2.0/1j**(0.5)/wom*(1.0+1.0/2.0/wom))
            elif wom > 2.0:
                g_omega = np.sqrt(self.C*self.a/self.density)*((3.0-wom)* \
                    np.sqrt(1j*wom**2.0/8.0+wom**4.0/48.0) + \
                    (wom-2.0)* np.sqrt(1.0-2.0/1j**(0.5)/wom*(1.0+1.0/2.0/wom)))
                c_omega = np.sqrt(self.a/self.C/self.density)*((3.0-wom)* \
                    np.sqrt(1j*wom**2.0/8.0+wom**4.0/48.0) + \
                        (wom-2.0)* \
                        np.sqrt(1.0-2.0/1j**(0.5)/wom*(1.0+1.0/2.0/wom)))
            elif wom == 0.0:
                g_omega = 0.0
                c_omega = 0.0
            else:
                g_omega = np.sqrt(self.C*self.a/self.density)*np.sqrt(1j*wom**2/8+wom**4/48)
                c_omega = np.sqrt(self.a/self.C/self.density)*np.sqrt(1j*wom**2/8+wom**4/48)

            kappa = omega*self.l/c_omega

            t1 = 1j*np.sin(kappa)/g_omega + np.cos(kappa)*z_L
            t2 = np.cos(kappa) + 1j*g_omega*z_L*np.sin(kappa)
            z_0 = (t1/t2)

        return z_0
    
