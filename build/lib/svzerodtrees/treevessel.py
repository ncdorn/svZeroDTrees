import numpy as np
import math

# class for recursive generation of a structured tree

class TreeVessel:
    # blood vessel class for binary tree recusion operations
    def __init__(self, info: dict, name: str = None):
        '''
        :param info: dictionary of TreeVessel class parameters
        :param name: name of the vessel, which follows the svzerodplus naming convention
        '''
        self.name = name # name of the, tree only has a str in the root node
        self.info = info # vessel info dict
        self._d = self.info["vessel_D"] # diameter in CM
        self.l = self.info["vessel_length"]
        self.id = self.info["vessel_id"]
        self.gen = self.info["generation"]
        self.eta = self.info["viscosity"]
        self._R = self.info["zero_d_element_values"].get("R_poiseuille")
        self._R_eq = self._R # this is the initial value, not dependent on left and right
        self.C = self.info["zero_d_element_values"].get("C")
        self.L = self.info["zero_d_element_values"].get("L")

        # flow values, based on Poiseulle assumption
        self.P_in = 0.0
        self.Q = 0.0
        self.t_w = 0.0

        # daughter segments for recursion
        self._left = None
        self._right = None

        # if vessel is collapsed, no daughters
        self._collapsed = False


    @classmethod
    def create_vessel(cls, id, gen, diameter, eta):
        '''
        class method to create a TreeVessel instance

        :param id: int describing the vessel id
        :param gen: int describing the generation of the tree in which the TreeVessel exists
        :param diameter: diameter of the vessel
        :param eta: viscosity of the blood within the vessel

        :return: TreeVessel instance
        '''
        if diameter > .03 or id == 0:
            # if we allow the first vessel to experience FL effects, the optimization does not work as the resistance is
            # decreased by an order of magnitude. Therefore the first vessel automatically does not experience FL.
            viscosity = eta
        else: 
            # Implemented Fahraeus-Lindqvist effect according to empirical relationship in Lan et al. and Pries and Secomb
            viscosity = cls.fl_visc(cls, diameter)
        
        # initialize the 0D parameters of the treee
        R, C, L, l = cls.calc_zero_d_values(cls, diameter, viscosity)

        # create name to match svzerodplus naming convention
        name = "branch" + str(id) + "_seg0" 

        # generate a config file to run svzerodplus on the TreeVessel instances
        vessel_info = {"vessel_id": id,  # mimic input json file
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
                       }

        return cls(info=vessel_info)


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
        return self._R + (1 / (self._left._R_eq ** -1 + self._right._R_eq ** -1))

    def _update_R_eq(self):
        self._R_eq = self._R + (1 / (self._left._R_eq ** -1 + self._right._R_eq ** -1))

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
        self.update_vessel_info()

    ####### end of property setters used to dynamically update various class properties #######


    def calc_zero_d_values(self, vesselD, eta):
        '''
        calculate 0D Windkessel parameters based on a vessel diameter

        :param vesselD: vessel diameter
        :param eta: vessel viscosity

        :return: resistance, capacitance, inductiance and vessel length
        '''

        r = vesselD / 2
        l = 12.4 * r ** 1.1  # from ingrid's paper - does this remain constant throughout adaptation?
        R = 8 * eta * l / (np.pi * r ** 4)
        C = 0.0  # to implement later
        L = 0.0  # to implement later

        return R, C, L, l

    def update_vessel_info(self):
        '''
        update vessel info dict based on changes to vessel diameter
        '''

        if self._d < .03 and self.id > 0: 
        # if we allow the first vessel to experience FL effects, the optimization does not work as the resistance is decreased
        # by an order of magnitude
            self.eta = self.fl_visc(self.d)
            self.info["viscosity"] = self.eta
        R, C, L, l = self.calc_zero_d_values(self._d, self.eta)
        self.info["vessel_length"] = l
        self.info["vessel_D"] = self.d
        self.info["zero_d_element_values"]["R_poiseulle"] = R
        self.info["zero_d_element_values"]["C"] = C
        self.info["zero_d_element_values"]["L"] = L
        self.R = R
        if not self.collapsed:
            self._update_R_eq()

    def add_collapsed_bc(self):
        '''
        if the vessel is collapsed, add a distal pressure outlet boundary condition to the vessel config
        '''
        self.info["boundary_conditions"] = {
        
            "outlet": "P_d" + str(self.id)
        }

    def adapt_pries_secomb(self, ps_params, dt, H_d=0.45):
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

        self.k_p, self.k_m, self.k_c, self.k_s, self.L, self.S_0, self.tau_ref, self.Q_ref = tuple(ps_params)
        self. H_d = H_d # hematocrit

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

        self.S_c = self.k_c * (self.Sbar_c / (self.Sbar_c + self.S_0))

        self.S_s = -self.k_s

        self.S_tot = self.S_tau + self.S_p + self.S_m + self.S_c + self.S_s

        self.dD = self.d * self.S_tot * dt

        # make sure that the diameter change is positive
        if self.d + self.dD > 0.0:
            self.d += self.dD
        else:
            pass
            # potentially add collapsed 

        return self.dD

    def fl_visc(self, diameter):
        '''
        calculate the viscosity within a vessel of diameter < 300 um based on empirical relationship describing 
        fahraeus-lindqvist effect

        :param diameter: vessel diameter
        '''
        H_d = 0.45 # hematocrit
        u_45 = 6 * math.exp(-0.085 * diameter) + 3.2 - 2.44 * math.exp(-0.06 * diameter ** 0.645)
        C = (0.8 + math.exp(-0.075 * diameter)) * (-1 + (1 + 10 ** -11 * diameter ** 12) ** -1) + (1 + 10 ** -11 * diameter ** 12) ** -1
        viscosity = .012 * (1 + (u_45 - 1) * (((1 - H_d) ** C - 1) / ((1 - 0.45) ** C - 1)) * (diameter / (diameter - 1.1)) ** 2) * (diameter / (diameter - 1.1)) ** 2

        return viscosity