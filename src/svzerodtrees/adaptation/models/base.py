from abc import ABC, abstractmethod

import numpy as np

class AdaptationModel(ABC):
    def __init__(self, K_arr):
        self.K_arr = K_arr

    def encode_state(self, y):
        return np.asarray(y, dtype=np.float64)

    def decode_state(self, y):
        return np.asarray(y, dtype=np.float64)

    @abstractmethod
    def compute_rhs(self, t, y, simple_pa, vessels, last_update_y, last_t_holder, flow_log, solver_trace):
        pass

    @abstractmethod
    def event(self, t, y, simple_pa, *args):
        pass
