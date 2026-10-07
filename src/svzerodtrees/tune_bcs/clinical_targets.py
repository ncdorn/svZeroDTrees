import math

import pandas as pd
import csv
from ..utils import write_to_log

# How the outlet (distal) pressure Pd of the structured-tree BCs is derived.
# The trees end at pre-capillary arterioles, so Pd stands for the pressure
# downstream of the modelled arterial tree.
#   clamp_to_diastolic: min(wedge, diastolic MPA target) (historical default)
#   measured: the measured wedge pressure, even when it exceeds the diastolic
#             target (e.g. pulmonary regurgitation)
#   precapillary_fraction: wedge + f * (mean MPA - wedge), the pre-capillary
#             pressure when a fraction f of the MPA-to-LA drop lies downstream
#             of the trees (capillaries and veins).  Intended for patients with
#             pulmonary regurgitation, whose backflow lets MPA pressure fall
#             below Pd in diastole.
#   diastolic_offset: diastolic MPA target - offset.  Intended for patients
#             without regurgitation: with low end-diastolic flow the arterial
#             tree carries little pressure drop, so PA diastolic approximates
#             the downstream pressure; the offset keeps the diastolic target
#             reachable (with a constant Pd and no backflow, MPA pressure only
#             decays toward Pd).  Does not need a measured wedge pressure.
WEDGE_PRESSURE_POLICIES = (
    "clamp_to_diastolic",
    "measured",
    "precapillary_fraction",
    "diastolic_offset",
)
DEFAULT_PRECAPILLARY_FRACTION = 0.332
DEFAULT_DIASTOLIC_OFFSET_MMHG = 2.0


def resolve_outlet_pressure(
    mpa_p,
    measured_wedge_p,
    policy,
    *,
    precapillary_fraction=DEFAULT_PRECAPILLARY_FRACTION,
    diastolic_offset_mmhg=DEFAULT_DIASTOLIC_OFFSET_MMHG,
):
    """Return the outlet pressure Pd [mmHg] for ``policy``.

    ``mpa_p`` is [systolic, diastolic, mean]; ``measured_wedge_p`` may be NaN
    or None only for ``diastolic_offset``.
    """
    if policy not in WEDGE_PRESSURE_POLICIES:
        raise ValueError(
            "wedge_pressure_policy must be one of " + "|".join(WEDGE_PRESSURE_POLICIES)
        )
    dia, mean = float(mpa_p[1]), float(mpa_p[2])
    wedge = float("nan") if measured_wedge_p is None else float(measured_wedge_p)
    if policy in ("clamp_to_diastolic", "measured"):
        # Legacy policies keep their historical behavior: a missing wedge gives
        # NaN here (callers that only need targets keep working) and tuning
        # rejects a non-finite outlet pressure (validate_outlet_pressure).
        if not math.isfinite(wedge):
            return float("nan")
        return float(min(wedge, dia) if policy == "clamp_to_diastolic" else wedge)
    if policy == "diastolic_offset":
        offset = float(diastolic_offset_mmhg)
        if not math.isfinite(offset) or offset < 0.0:
            raise ValueError("diastolic_offset_mmhg must be finite and >= 0")
        pd_value = dia - offset
    else:
        if not math.isfinite(wedge):
            raise ValueError(
                "wedge_pressure_policy 'precapillary_fraction' requires a measured "
                "wedge_pressure in clinical_targets.csv; use 'diastolic_offset' when it "
                "is unavailable"
            )
        fraction = float(precapillary_fraction)
        if not math.isfinite(fraction) or not 0.0 <= fraction < 1.0:
            raise ValueError("precapillary_fraction must be in [0, 1)")
        pd_value = wedge + fraction * (mean - wedge)
    _check_outlet_pressure(pd_value, mean, policy)
    return float(pd_value)


def _check_outlet_pressure(pd_value, mean, policy):
    if not math.isfinite(pd_value) or pd_value < 0.0:
        raise ValueError(
            f"wedge_pressure_policy '{policy}' gives a non-physical outlet pressure "
            f"{pd_value} mmHg"
        )
    if mean is not None and pd_value >= float(mean):
        raise ValueError(
            f"wedge_pressure_policy '{policy}' gives an outlet pressure {pd_value:.2f} mmHg "
            f">= the mean MPA target {float(mean):.2f} mmHg; no pressure drop would drive "
            "the mean flow"
        )


def validate_outlet_pressure(targets):
    """Raise unless ``targets.wedge_p`` is a usable tuning outlet pressure."""
    mpa_p = getattr(targets, "mpa_p", None)
    _check_outlet_pressure(
        float(targets.wedge_p),
        None if mpa_p is None else float(mpa_p[2]),
        getattr(targets, "wedge_pressure_policy", "clamp_to_diastolic"),
    )
class ClinicalTargets():
    '''
    class to handle clinical target values
    '''

    def __init__(self, mpa_p=None, lpa_p=None, rpa_p=None, q=None, rpa_split=None, wedge_p=None, t=None, steady=False,
                 rvot_flow=None, ivc_flow=None, svc_flow=None):
        '''
        initialize the clinical targets object
        '''
        
        self.t = t
        self.mpa_p = mpa_p
        self.lpa_p = lpa_p
        self.rpa_p = rpa_p
        self.q = q

        # fontan flows
        self.rvot_flow = rvot_flow
        self.ivc_flow = ivc_flow
        self.svc_flow = svc_flow

        self.rpa_split = rpa_split
        if q is not None and rpa_split is not None:
            self.q_rpa = q * rpa_split
        self.wedge_p = wedge_p
        self.steady = steady


    @classmethod
    def from_csv(
        cls,
        clinical_targets: csv,
        steady=True,
        wedge_pressure_policy="clamp_to_diastolic",
        precapillary_fraction=DEFAULT_PRECAPILLARY_FRACTION,
        diastolic_offset_mmhg=DEFAULT_DIASTOLIC_OFFSET_MMHG,
    ):
        '''
        initialize from a csv file

        :param wedge_pressure_policy: one of WEDGE_PRESSURE_POLICIES; sets how
            the outlet pressure ``wedge_p`` [mmHg] is derived
        :param precapillary_fraction: downstream fraction for
            ``precapillary_fraction``
        :param diastolic_offset_mmhg: offset for ``diastolic_offset``
        '''
        if wedge_pressure_policy not in WEDGE_PRESSURE_POLICIES:
            raise ValueError(
                "wedge_pressure_policy must be one of " + "|".join(WEDGE_PRESSURE_POLICIES)
            )
        # get the flowrate
        df = pd.read_csv(clinical_targets)
        df.columns = map(str.lower, df.columns)

        # get the mpa flowrate
        q = float(df.loc[0,'mpa_flow'])

        if "rvot_flow" in df.columns and "ivc_flow" in df.columns and "svc_flow" in df.columns:
            print("RVOT, IVC, SVC BCs detected")
            rvot_flow = float(df.loc[0,"rvot_flow"])
            ivc_flow = float(df.loc[0,"ivc_flow"])
            svc_flow = float(df.loc[0,"svc_flow"])
        else:
            rvot_flow = None
            ivc_flow = None
            svc_flow = None

        # get the mpa pressures
        mpa_p = [float(p) for p in df.loc[0,"mpa_pressure"].split("/")] # sys, dia, mean

        # get wedge pressure (may be absent for diastolic_offset)
        measured_wedge_p = float("nan")
        if "wedge_pressure" in df.columns:
            raw_wedge = df.loc[0, "wedge_pressure"]
            if not (isinstance(raw_wedge, str) and not raw_wedge.strip()) and not pd.isna(raw_wedge):
                measured_wedge_p = float(raw_wedge)
        wedge_p = resolve_outlet_pressure(
            mpa_p,
            measured_wedge_p,
            wedge_pressure_policy,
            precapillary_fraction=precapillary_fraction,
            diastolic_offset_mmhg=diastolic_offset_mmhg,
        )

        # get RPA flow split
        rpa_split = float(df.loc[0,"rpa_split"])

        instance = cls(
            mpa_p,
            q=q,
            rpa_split=rpa_split,
            wedge_p=wedge_p,
            steady=steady,
            rvot_flow=rvot_flow,
            ivc_flow=ivc_flow,
            svc_flow=svc_flow,
        )
        instance.path = str(clinical_targets)
        instance.measured_wedge_p = measured_wedge_p
        instance.wedge_pressure_policy = wedge_pressure_policy
        instance.precapillary_fraction = float(precapillary_fraction)
        instance.diastolic_offset_mmhg = float(diastolic_offset_mmhg)
        return instance

        
    def log_clinical_targets(self, log_file):

        write_to_log(log_file, "*** clinical targets ****")
        write_to_log(log_file, "Q: " + str(self.q))
        write_to_log(log_file, "MPA pressures: " + str(self.mpa_p))
        write_to_log(log_file, "RPA pressures: " + str(self.rpa_p))
        write_to_log(log_file, "LPA pressures: " + str(self.lpa_p))
        write_to_log(log_file, "wedge pressure: " + str(self.wedge_p))
        write_to_log(log_file, "RPA flow split: " + str(self.rpa_split))
