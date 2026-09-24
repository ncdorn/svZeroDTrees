import pandas as pd
import csv
from ..utils import write_to_log

# How the measured wedge pressure becomes the distal pressure (Pd) of outlet BCs.
#   clamp_to_diastolic: min(wedge, diastolic MPA target) (historical default)
#   measured: the measured wedge pressure, even when it exceeds the diastolic
#             target (e.g. pulmonary regurgitation)
WEDGE_PRESSURE_POLICIES = ("clamp_to_diastolic", "measured")
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
    def from_csv(cls, clinical_targets: csv, steady=True, wedge_pressure_policy="clamp_to_diastolic"):
        '''
        initialize from a csv file

        :param wedge_pressure_policy: one of WEDGE_PRESSURE_POLICIES; sets how
            the measured wedge pressure [mmHg] becomes ``wedge_p``
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

        # get wedge pressure
        measured_wedge_p = float(df.loc[0,"wedge_pressure"])
        if wedge_pressure_policy == "clamp_to_diastolic":
            # ensure wedge pressure is not greater than diastolic MPA pressure
            wedge_p = min(measured_wedge_p, mpa_p[1])
        else:
            wedge_p = measured_wedge_p

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
        return instance

        
    def log_clinical_targets(self, log_file):

        write_to_log(log_file, "*** clinical targets ****")
        write_to_log(log_file, "Q: " + str(self.q))
        write_to_log(log_file, "MPA pressures: " + str(self.mpa_p))
        write_to_log(log_file, "RPA pressures: " + str(self.rpa_p))
        write_to_log(log_file, "LPA pressures: " + str(self.lpa_p))
        write_to_log(log_file, "wedge pressure: " + str(self.wedge_p))
        write_to_log(log_file, "RPA flow split: " + str(self.rpa_split))
