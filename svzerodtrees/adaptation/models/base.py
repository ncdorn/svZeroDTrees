from abc import ABC, abstractmethod

class AdaptationModel(ABC):
    def __init__(self, K_arr):
        self.K_arr = K_arr

    @abstractmethod
    def compute_rhs(self, t, y, simple_pa, vessels, last_update_y, last_t_holder, flow_log):
        pass

    @abstractmethod
    def event(self, t, y, simple_pa, *args):
        pass