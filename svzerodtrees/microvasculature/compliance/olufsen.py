import numpy as np
from .base import ComplianceModel

class OlufsenCompliance(ComplianceModel):
    def __init__(self, k1: float, k2: float, k3: float):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

        self.params['k1'] = k1
        self.params['k2'] = k2
        self.params['k3'] = k3

    def evaluate(self, r: float) -> float:
        return self.k1 * np.exp(self.k2 * r) + self.k3