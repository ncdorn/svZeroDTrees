from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

# Simple transforms to keep parameters in valid domains
def identity(x): return x
def positive(x): return np.exp(x)        # optimize log(value)
def unit_interval(x): return 1/(1+np.exp(-x))  # sigmoid

@dataclass(frozen=True)
class FreeParam:
    name: str                  # e.g. "lpa.diameter", "rpa.alpha", "comp.lpa.k2", "lrr"
    init: float
    lb: float
    ub: float
    to_native: Callable[[float], float] = identity     # x (free space) -> native space
    from_native: Callable[[float], float] = identity   # native -> free space (for init)
    
@dataclass(frozen=True)
class FixedParam:
    name: str
    value: float

@dataclass(frozen=True)
class TiedParam:
    name: str
    other: str
    fn: Callable[[float], float] = identity   # native value of other -> native value of this

@dataclass
class TuneSpace:
    free: List[FreeParam]
    fixed: List[FixedParam]
    tied: List[TiedParam]

    def pack_init_and_bounds(self) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        x0, bnds = [], []
        for p in self.free:
            x0.append(p.from_native(p.init))
            bnds.append((p.from_native(p.lb), p.from_native(p.ub)))
        return np.array(x0, dtype=float), bnds

    def vector_to_param_dict(self, x: np.ndarray) -> Dict[str, float]:
        # start with free
        state: Dict[str, float] = {}
        for i, p in enumerate(self.free):
            state[p.name] = p.to_native(float(x[i]))
        # add fixed
        for p in self.fixed:
            state[p.name] = p.value
        # add tied
        for t in self.tied:
            if t.other not in state:
                raise ValueError(f"TiedParam depends on '{t.other}' which is not resolved yet.")
            state[t.name] = t.fn(state[t.other])
        return state
    


#### PRESETS FOR EXAMPLE PURPOSES ####



def preset_fix_root_vary_alpha_beta_and_k2(geom_defaults) -> TuneSpace:
    # share k2 across LPA/RPA, fix diameters to geometry defaults, vary alpha/beta and lrr
    return TuneSpace(
        free=[
            FreeParam("lpa.alpha", init=0.9, lb=0.7, ub=0.99),
            FreeParam("lpa.beta",  init=0.6, lb=0.3, ub=0.9),
            FreeParam("rpa.alpha", init=0.9, lb=0.7, ub=0.99),
            FreeParam("rpa.beta",  init=0.6, lb=0.3, ub=0.9),
            FreeParam("lrr",       init=10.0, lb=4.0,  ub=25.0),
            FreeParam("comp.lpa.k2", init=-25.0, lb=-100.0, ub=-1.0),
        ],
        fixed=[
            FixedParam("lpa.diameter", geom_defaults["lpa.default_diameter"]),
            FixedParam("rpa.diameter", geom_defaults["rpa.default_diameter"]),
            FixedParam("d_min", 0.01),
        ],
        tied=[
            TiedParam("comp.rpa.k2", other="comp.lpa.k2")
        ]
    )

def preset_vary_root_diameters_and_C() -> TuneSpace:
    # constant compliance with separate C per side, vary diameters too
    return TuneSpace(
        free=[
            FreeParam("lpa.diameter", init=0.30, lb=0.12, ub=0.40, to_native=positive, from_native=np.log),
            FreeParam("rpa.diameter", init=0.30, lb=0.12, ub=0.40, to_native=positive, from_native=np.log),
            FreeParam("comp.lpa.C",   init=6.6e4, lb=1.0e4, ub=2.0e5, to_native=positive, from_native=np.log),
            FreeParam("comp.rpa.C",   init=6.6e4, lb=1.0e4, ub=2.0e5, to_native=positive, from_native=np.log),
            FreeParam("lrr",          init=10.0,  lb=4.0,   ub=25.0),
            FreeParam("lpa.alpha",    init=0.9,   lb=0.7,   ub=0.99),
            FreeParam("lpa.beta",     init=0.6,   lb=0.3,   ub=0.9),
            FreeParam("rpa.alpha",    init=0.9,   lb=0.7,   ub=0.99),
            FreeParam("rpa.beta",     init=0.6,   lb=0.3,   ub=0.9),
        ],
        fixed=[FixedParam("d_min", 0.01)],
        tied=[]
    )