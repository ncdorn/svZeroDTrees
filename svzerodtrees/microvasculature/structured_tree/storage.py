from dataclasses import dataclass
import numpy as np

@dataclass
class StructuredTreeStorage:
    # node fields (SoA)
    ids: np.ndarray         # int32
    gen: np.ndarray         # int16
    d: np.ndarray           # float32
    parent: np.ndarray      # int32, -1 for root
    left: np.ndarray        # int32, -1 if none/collapsed
    right: np.ndarray       # int32, -1 if none/collapsed
    collapsed: np.ndarray   # bool
    # tree-level scalars/defaults
    lrr: float
    density: float
    compliance_model: str | int | object  # whatever you use
    name: str
    eta: float

    def n_nodes(self) -> int:
        return self.ids.size