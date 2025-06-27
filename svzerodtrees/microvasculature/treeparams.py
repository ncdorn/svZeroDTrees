import pandas as pd
from pathlib import Path

class TreeParameters:
    """
    Structured parameters for one vascular tree (LPA or RPA).
    """
    def __init__(self, pa: str, row: pd.Series):
        self.side = pa
        self.k1 = row["k1"].values[0]
        self.k2 = row["k2"].values[0]
        self.k3 = row["k3"].values[0]
        self.lrr = row["lrr"].values[0]
        self.diameter = row["diameter"].values[0]
        self.alpha = 0.9
        self.beta = 0.6
        self.d_min = row["d_min"].values[0]

    def as_list(self) -> list:
        return [self.k1, self.k2, self.k3, self.lrr, self.alpha, self.beta]

    def __repr__(self):
        return (
            f"TreeParameters(pa={self.side}, "
            f"k1={self.k1:.3g}, k2={self.k2:.3g}, k3={self.k3:.3g}, "
            f"lrr={self.lrr:.3g}, alpha={self.alpha}, beta={self.beta}, "
            f"diameter={self.diameter:.3f})"
        )

