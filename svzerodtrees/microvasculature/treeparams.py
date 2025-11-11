import pandas as pd
from pathlib import Path
from .compliance import *
from .structured_tree.asymmetry import resolve_branch_scaling

class TreeParameters:
    """
    Structured parameters for one vascular tree (LPA or RPA).
    """
    def __init__(self, 
                 name: str,
                 lrr: float,
                 diameter: float,
                 d_min: float,
                 alpha: float = None,
                 beta: float = None,
                 compliance_model: ComplianceModel = None,
                 k1: float = None,
                 k2: float = None,
                 k3: float = None, # want to eventually deprecate k1, k2, k3.
                 xi: float = None,
                 eta_sym: float = None,
                 ):
        
        if compliance_model is None:
            raise ValueError("TreeParameters requires a compliance_model.")

        self.name = name
        self.lrr = lrr
        self.diameter = diameter
        self.d_min = d_min
        self.xi = xi
        self.eta_sym = eta_sym
        self.alpha, self.beta = resolve_branch_scaling(
            alpha, beta, xi, eta_sym, default_alpha=None, default_beta=None)
        if self.eta_sym is None and self.alpha:
            self.eta_sym = self.beta / self.alpha
        self.compliance_model = compliance_model 


        # want to eventually deprecate!! and only rely on compliance model class
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

    @classmethod
    def from_row(cls, row: pd.Series):
        """
        Create a TreeParameters instance from a DataFrame row with new compliance model.

        :param row: DataFrame row containing tree parameters
        """
        name = row["pa"].values[0]
        lrr = row["lrr"].values[0]
        diameter = row["diameter"].values[0]
        d_min = row["d_min"].values[0]
        alpha = row["alpha"].values[0] if "alpha" in row else None
        beta = row["beta"].values[0] if "beta" in row else None
        xi = row["xi"].values[0] if "xi" in row else None
        eta_sym = row["eta_sym"].values[0] if "eta_sym" in row else None

        if row["compliance model"].values[0] == "ConstantCompliance":
            compliance_model = ConstantCompliance(row["Eh/r"].values[0])
        elif row["compliance model"].values[0] == "OlufsenCompliance":
            compliance_model = OlufsenCompliance(k1=row["k1"].values[0],
                                                 k2=row["k2"].values[0],
                                                 k3=row["k3"].values[0])
        else:
            raise ValueError(f"Unknown compliance model: {row['compliance model'].values[0]}")

        return cls(name, lrr, diameter, d_min, alpha, beta, compliance_model, xi=xi, eta_sym=eta_sym)

    def as_list(self) -> list:
        return [self.k1, self.k2, self.k3, self.lrr, self.alpha, self.beta]

    def __repr__(self):
        return (
            f"TreeParameters(pa={self.side}, "
            f"k1={self.k1:.3g}, k2={self.k2:.3g}, k3={self.k3:.3g}, "
            f"lrr={self.lrr:.3g}, alpha={self.alpha}, beta={self.beta}, "
            f"diameter={self.diameter:.3f})"
        )
    
    def summary(self):
        """
        return a string of important paramters for the tree
        """

        xi_str = f"{self.xi:.3f}" if self.xi is not None else "n/a"
        eta_str = f"{self.eta_sym:.3f}" if self.eta_sym is not None else "n/a"
        return (
            f"{self.compliance_model.description()} with params {self.compliance_model.params}, "
            f"diameter: {self.diameter:.3f}, d_min: {self.d_min:.3f}, l_rr: {self.lrr:.3f}, "
            f"alpha: {self.alpha:.3f}, beta: {self.beta:.3f}, xi: {xi_str}, eta_sym: {eta_str}"
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
            "d_min": self.d_min,
            "alpha": self.alpha,
            "beta": self.beta,
            "xi": self.xi,
            "eta_sym": self.eta_sym,
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
