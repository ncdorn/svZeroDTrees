import numpy as np
import math

# class for recursive generation of a structured tree

class TreeVessel:
    # class to construct binary tree of blood vessels using recursion
    def __init__(self, info: dict, name: str = None):
        self.name = name # name of the, tree only has a str in the root node
        self.info = info # vessel info dict
        self._d = self.info["vessel_D"] # diameter
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
        # pries and secomb parameters
        self.pries_secomb = True


    @classmethod
    def create_vessel(cls, id, gen, diameter, eta):
        if diameter > .3:
            viscosity = eta
        else: # Implemented Fahraeus-Lindqvist effect according to empirical relationship in Lan et al. and Pries and Secomb
            H_d = 0.45 # hematocrit
            u_45 = 6 * math.exp(-0.085 * diameter) + 3.2 - 2.44 * math.exp(-0.06 * diameter ** 0.645)
            C = (0.8 + math.exp(-0.075 * diameter)) * (-1 + (1 + 10 ** -11 * diameter ** 12) ** -1) + (1 + 10 ** -11 * diameter ** 12) ** -1
            viscosity = .012 * (1 + (u_45 - 1) * (((1 - H_d) ** C - 1) / ((1 - 0.45) ** C - 1)) * (diameter / (diameter - 1.1)) ** 2) * (diameter / (diameter - 1.1)) ** 2
        R, C, L, l = cls.calc_zero_d_values(cls, diameter, viscosity)
        # print(R, C, L, l)
        name = " "  # to implement later

        # generate essentially a config file for the BloodVessel instances
        vessel_info = {"vessel_id": id,  # mimic input json file
                       "vessel_length": l,
                       "vessel_D": diameter,
                       "vessel_name": name,
                       "generation": gen,
                       "viscosity": viscosity,
                       "zero_d_element_type": "BloodVessel",
                       "zero_d_element_values": {
                           "R_poiseuille": R,
                           "C": C,
                           "L": L,
                           "stenosis_coefficient": 0.0
                       }}

        return cls(info=vessel_info)

    # property setters to dynamically update equivalent resistance
    @property
    def left(self):
        return self._left

    @left.setter # this method updates R_eq based on the left and right vessels
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

    def calc_zero_d_values(self, vesselD, eta):
        # calculate zero_d values based on an arbitrary vessel diameter
        r = vesselD / 2
        l = 12.4 * r ** 1.1  # from ingrid's paper, does this remain constant throughout adaptation?
        R = 8 * eta * l / (np.pi * r ** 4)
        C = 0.0  # to implement later
        L = 0.0  # to implement later

        return R, C, L, l

    def update_vessel_info(self):
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
        self.info["boundary_conditions"] = {
            "outlet": "P_d"
        }

    # def initialize_pries_secomb(self, ps_params, H_d=0.45):
    #     # intialize the pries and secomb parameters which are required for upstream adaptation calculations
    #     # namely, S_m = f(k_m, Q_ref, H_D)
    #     # this will have to be done in a postorder traversal
    #     # ps_params in the following form [k_p, k_m, k_c, k_s, S_0, tau_ref, Q_ref, L]
    #     self.k_p, self.k_m, self.k_c, self.k_s, self.S_0, self.tau_ref, self.Q_ref, self.L = tuple(ps_params)
    #     self. H_d = H_d # hematocrit
    #
    #
    #     self.S_m = self.k_m * math.log(self.Q_ref / (self.Q * self.H_d) + 1)
    #     self.Sbar_c = 0.0 # initialize sbar_c
    #
    #     if not self.collapsed:
    #         if self.left.Sbar_c > 0:
    #             self.Sbar_c = self.left.S_m + self.right.S_m + self.left.Sbar_c * math.exp(-self.left.l / self.L) + self.right.Sbar_c * math.exp(-self.right.l / self.L)
    #         else:
    #             self.Sbar_c = self.left.S_m + self.right.S_m



    def adapt_pries_secomb(self, ps_params, dt, H_d=0.45):
        # calculate the pries and secomb parameters for microvascular adaptation
        # this will have to be done in a postorder traversal
        # ps_params in the following form [k_p, k_m, k_c, k_s, S_0, tau_ref, Q_ref, L]
        # adapt the tree based the pries and secomb model for diameter change

        self.k_p, self.k_m, self.k_c, self.k_s, self.S_0, self.tau_ref, self.Q_ref, self.L = tuple(ps_params)
        self. H_d = H_d # hematocrit

        self.S_m = self.k_m * math.log(self.Q_ref / (self.Q * self.H_d) + 1)

        self.tau_e = 50 / 86 * (100 - 86 * math.exp(-5000 * math.log(math.log(4.5 * self.P_in + 10)) ** 5.4) - 14) + 1

        self.S_tau = math.log(self.t_w + self.tau_ref)

        self.S_p = -self.k_p * math.log(self.tau_e)
        
        self.Sbar_c = 0.0 # initialize sbar_c
        if not self.collapsed:
            if self.left.Sbar_c > 0:
                self.Sbar_c = self.left.S_m + self.right.S_m + self.left.Sbar_c * math.exp(-self.left.l / self.L) + self.right.Sbar_c * math.exp(-self.right.l / self.L)
            else:
                self.Sbar_c = self.left.S_m + self.right.S_m

        self.S_c = self.k_c * (self.Sbar_c / (self.Sbar_c + self.S_0))

        self.S_s = -self.k_s

        self.S_tot = self.S_tau + self.S_p + self.S_m + self.S_c + self.S_s

        self.dD = self.d * self.S_tot * dt

        if self.d + self.dD > 0.0:
            self.d += self.dD
        else:
            pass

        # if self.d < 0.005:
        #     self.collapsed = True
        #     self.d = 0.0
        # else:
        #     self.collapsed = False


        return self.dD
