from .base import AdaptationModel
from ..utils import unpack_state, simulate_outlet_trees
import numpy as np
from typing import Sequence

class CWSSIMSAdaptation(AdaptationModel):
    
    def __init__(self, K_arr: Sequence[float]):
        if not isinstance(K_arr, (list, tuple, np.ndarray)):
            raise TypeError(f"K_arr must be a list, tuple, or ndarray, but got {type(K_arr)}.")
        if len(K_arr) != 4:
            raise ValueError(
                f"K_arr must contain exactly 4 elements: "
                f"[K_tau_r, K_sig_r, K_tau_h, K_sig_h], but got {len(K_arr)}."
            )
        super().__init__(K_arr)
    
    def compute_rhs(self, t, y, simple_pa, vessels, last_update_y, last_t_holder, flow_log):
        unpack_state(y, vessels)
        simple_pa.update_bcs()
        simple_pa.simulate()

        geom_change = np.max(np.abs((y - last_update_y) / last_update_y))
        if geom_change > 1e-3 and t > last_t_holder[0] + 1e-12:
            print(f"Geometry change at t={t:.2f} s: {geom_change:.3e} and rpa split: {simple_pa.rpa_split:.3f}")
            simulate_outlet_trees(simple_pa)
            last_update_y[:] = y
            last_t_holder[0] = t
            flow_log.append((t, simple_pa.rpa_split))

        mu = 0.04
        dydt = np.empty_like(y)
        for v in vessels:
            base = 2*v.idx
            tau = np.mean(4*mu*v.Q/(np.pi*v.r**3))
            sig = np.mean(v.P_in*v.r/v.h)
            dydt[base]   = self.K_arr[0]*(tau - v.wss_h) + self.K_arr[1]*(sig - v.ims_h)
            dydt[base+1] = -self.K_arr[2]*(tau - v.wss_h) + self.K_arr[3]*(sig - v.ims_h)
        return dydt

    def event(self, t, y, *args):
        gchange = np.max(np.abs((y - args[1]) / args[1]))  # args[1] = last_update_y
        return gchange - 5e-7
    event.terminal = True
    event.direction = -1