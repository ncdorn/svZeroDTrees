import pandas as pd
from pathlib import Path
from .compliance import *

class TreeParameters:
    """
    Structured parameters for one vascular tree (LPA or RPA).
    """
    def __init__(self, 
                 name: str,
                 lrr: float,
                 diameter: float,
                 d_min: float,
                 alpha: float,
                 beta: float,
                 compliance_model: ComplianceModel,
                 k1: float = None,
                 k2: float = None,
                 k3: float = None, # want to eventually deprecate k1, k2, k3.
                 ):
        
        self.name = name
        self.lrr = lrr
        self.diameter = diameter
        self.d_min = d_min
        self.alpha = alpha
        self.beta = beta
        self.compliance_model = compliance_model 


        # want to eventually deprecate!! and only rely on compliance model class
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
    
    @classmethod
    def from_row(cls, pa: str, row: pd.Series):
        """
        Create a TreeParameters instance from a DataFrame row. basically a backwards compatible function.

        :param pa: 'lpa' or 'rpa'
        :param row: DataFrame row containing tree parameters
        """

        k1 = row["k1"].values[0]
        k2 = row["k2"].values[0]
        k3 = row["k3"].values[0]
        lrr = row["lrr"].values[0]
        diameter = row["diameter"].values[0]
        alpha = 0.9
        beta = 0.6
        d_min = row["d_min"].values[0]
        compliance_model = ComplianceModel(k1=k1, k2=k2, k3=k3) if k1 is not None else None

        return cls(pa, lrr, diameter, d_min, alpha, beta, compliance_model)

    def as_list(self) -> list:
        return [self.k1, self.k2, self.k3, self.lrr, self.alpha, self.beta]

    def __repr__(self):
        return (
            f"TreeParameters(pa={self.side}, "
            f"k1={self.k1:.3g}, k2={self.k2:.3g}, k3={self.k3:.3g}, "
            f"lrr={self.lrr:.3g}, alpha={self.alpha}, beta={self.beta}, "
            f"diameter={self.diameter:.3f})"
        )
    
    def to_csv_row(self, loss, flow_split, p_mpa):
        '''
        Get a CSV row representation of the tree parameters.
        :param loss: Loss value for the optimization.
        :param flow_split: Flow split value from the optimization.
        :param p_mpa: Pressure values from the optimization.
        '''
        row = {
            "pa": self.name,
            "compliance model": self.compliance_model.description(),
            "lrr": self.lrr,
            "diameter": self.diameter,
            "loss": loss,
            "flow_split": flow_split,
            "p_mpa": f"[{p_mpa[0]} {p_mpa[1]} {p_mpa[2]}]",
        }

        # Add compliance-specific parameters
        if isinstance(self.compliance_model, OlufsenCompliance):
            row["k1"] = self.compliance_model.k1
            row["k2"] = self.compliance_model.k2
            row["k3"] = self.compliance_model.k3
        elif isinstance(self.compliance_model, ConstantCompliance):
            row["Eh/r"] = self.compliance_model.value
        else:
            raise ValueError(f"Unsupported compliance model: {self.compliance_model.description()}")

        return row

