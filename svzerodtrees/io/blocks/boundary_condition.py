
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


_ALLOWED_IMPEDANCE_KEYS = {"z", "Pd", "convolution_mode", "num_kernel_terms"}
_FORBIDDEN_IMPEDANCE_KEYS = {"Z", "tree", "t"}


def _ensure_finite_numeric(value, *, bc_name: str, field_name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"IMPEDANCE boundary condition '{bc_name}' field '{field_name}' must be numeric"
        ) from exc
    if not math.isfinite(numeric):
        raise ValueError(
            f"IMPEDANCE boundary condition '{bc_name}' field '{field_name}' must be finite"
        )
    return numeric


def _ensure_positive_int(value, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if numeric <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return numeric


def validate_impedance_bc_values(values: Mapping, *, bc_name: str) -> dict:
    if not isinstance(values, Mapping):
        raise ValueError(
            f"IMPEDANCE boundary condition '{bc_name}' bc_values must be a mapping"
        )

    keys = set(values.keys())
    invalid_keys = sorted((keys - _ALLOWED_IMPEDANCE_KEYS) | (keys & _FORBIDDEN_IMPEDANCE_KEYS))
    if invalid_keys:
        raise ValueError(
            f"IMPEDANCE boundary condition '{bc_name}' contains unsupported keys: "
            + ", ".join(invalid_keys)
        )

    missing = [key for key in ("z", "Pd") if key not in values]
    if missing:
        raise ValueError(
            f"IMPEDANCE boundary condition '{bc_name}' is missing required keys: "
            + ", ".join(missing)
        )

    kernel = values["z"]
    if isinstance(kernel, (str, bytes)):
        raise ValueError(
            f"IMPEDANCE boundary condition '{bc_name}' field 'z' must be a non-empty numeric sequence"
        )
    try:
        normalized_kernel = [
            _ensure_finite_numeric(entry, bc_name=bc_name, field_name="z")
            for entry in kernel
        ]
    except TypeError as exc:
        raise ValueError(
            f"IMPEDANCE boundary condition '{bc_name}' field 'z' must be a non-empty numeric sequence"
        ) from exc
    if not normalized_kernel:
        raise ValueError(
            f"IMPEDANCE boundary condition '{bc_name}' field 'z' must be a non-empty numeric sequence"
        )

    normalized = {
        "z": normalized_kernel,
        "Pd": _ensure_finite_numeric(values["Pd"], bc_name=bc_name, field_name="Pd"),
    }

    mode = values.get("convolution_mode")
    if mode is not None:
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in {"exact", "truncated"}:
            raise ValueError(
                f"IMPEDANCE boundary condition '{bc_name}' field 'convolution_mode' "
                "must be 'exact' or 'truncated'"
            )
        normalized["convolution_mode"] = normalized_mode

    num_terms = values.get("num_kernel_terms")
    if num_terms is not None:
        if normalized.get("convolution_mode") != "truncated":
            raise ValueError(
                f"IMPEDANCE boundary condition '{bc_name}' field 'num_kernel_terms' "
                "requires convolution_mode='truncated'"
            )
        try:
            normalized_terms = int(num_terms)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"IMPEDANCE boundary condition '{bc_name}' field 'num_kernel_terms' must be a positive integer"
            ) from exc
        if normalized_terms <= 0:
            raise ValueError(
                f"IMPEDANCE boundary condition '{bc_name}' field 'num_kernel_terms' must be a positive integer"
            )
        normalized["num_kernel_terms"] = normalized_terms

    return normalized


def validate_boundary_condition_config(config: Mapping) -> dict:
    if not isinstance(config, Mapping):
        raise ValueError("boundary condition config must be a mapping")
    bc_name = str(config.get("bc_name", "")).strip() or "<unknown>"
    bc_type = config.get("bc_type")
    values = config.get("bc_values")
    if bc_type == "IMPEDANCE":
        values = validate_impedance_bc_values(values, bc_name=bc_name)
    return {
        "bc_name": bc_name,
        "bc_type": bc_type,
        "bc_values": dict(values) if isinstance(values, Mapping) else values,
    }


def validate_boundary_condition_configs(configs: Sequence[Mapping]) -> list[dict]:
    return [validate_boundary_condition_config(config) for config in configs]


def resolve_impedance_timepoint_contract(simparams: Mapping) -> tuple[str, int, int]:
    if not isinstance(simparams, Mapping):
        raise ValueError("simulation_parameters must be a mapping")

    points_per_cycle = simparams.get("number_of_time_pts_per_cardiac_cycle")
    if points_per_cycle is None:
        raise ValueError(
            "configs with IMPEDANCE boundary conditions require "
            "simulation_parameters.number_of_time_pts_per_cardiac_cycle"
        )

    resolved_points_per_cycle = _ensure_positive_int(
        points_per_cycle,
        field_name="simulation_parameters.number_of_time_pts_per_cardiac_cycle",
    )
    if resolved_points_per_cycle < 2:
        raise ValueError(
            "simulation_parameters.number_of_time_pts_per_cardiac_cycle must be >= 2"
        )
    return (
        "number_of_time_pts_per_cardiac_cycle",
        resolved_points_per_cycle,
        resolved_points_per_cycle - 1,
    )


def validate_impedance_timing_config(config: Mapping) -> None:
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")

    validated_bcs = validate_boundary_condition_configs(
        config.get("boundary_conditions", [])
    )
    impedance_bcs = [
        bc_config for bc_config in validated_bcs if bc_config.get("bc_type") == "IMPEDANCE"
    ]
    if not impedance_bcs:
        return

    simparams = config.get("simulation_parameters")
    if not isinstance(simparams, Mapping):
        raise ValueError(
            "configs with IMPEDANCE boundary conditions require simulation_parameters"
        )

    coupled = bool(simparams.get("coupled_simulation", False))
    mode_label = "coupled" if coupled else "non-coupled"

    number_of_time_pts = simparams.get("number_of_time_pts")
    if coupled:
        if number_of_time_pts is None:
            raise ValueError(
                "coupled config with IMPEDANCE boundary conditions requires "
                "simulation_parameters.number_of_time_pts = 2"
            )
        resolved_number_of_time_pts = _ensure_positive_int(
            number_of_time_pts,
            field_name="simulation_parameters.number_of_time_pts",
        )
        if resolved_number_of_time_pts != 2:
            raise ValueError(
                "coupled config with IMPEDANCE boundary conditions requires "
                "simulation_parameters.number_of_time_pts = 2; got "
                f"{resolved_number_of_time_pts}"
            )
    else:
        resolved_number_of_time_pts = (
            _ensure_positive_int(
                number_of_time_pts,
                field_name="simulation_parameters.number_of_time_pts",
            )
            if number_of_time_pts is not None
            else None
        )

    if "number_of_time_pts_per_cardiac_cycle" not in simparams:
        raise ValueError(
            f"{mode_label} config with IMPEDANCE boundary conditions requires "
            "simulation_parameters.number_of_time_pts_per_cardiac_cycle"
        )
    _, points_per_cycle, _ = resolve_impedance_timepoint_contract(simparams)

    for bc_config in impedance_bcs:
        bc_name = bc_config["bc_name"]
        kernel_size = len(bc_config["bc_values"]["z"])
        if kernel_size + 1 != points_per_cycle:
            raise ValueError(
                f"{mode_label} IMPEDANCE boundary condition '{bc_name}' requires "
                "simulation_parameters.number_of_time_pts_per_cardiac_cycle = len(z) + 1; "
                f"got number_of_time_pts={resolved_number_of_time_pts}, "
                f"number_of_time_pts_per_cardiac_cycle={points_per_cycle}, "
                f"len(z)={kernel_size}"
            )


class BoundaryCondition():
    '''
    class to handle boundary conditions
    '''

    def __init__(self, config: dict):
        validated = validate_boundary_condition_config(config)
        self.name = validated['bc_name']
        self.type = validated['bc_type']
        self.values = validated['bc_values']
        if self.type == 'RESISTANCE':
            self._R = self.values['R']
        
        if self.type == 'RCR':
            self._Rp = self.values['Rp']
            self._Rd = self.values['Rd']
            self._C = self.values['C']
        
        if self.type == 'FLOW':
            self._Q = self.values['Q']
            self._t = self.values['t']
        
        if self.type == 'PRESSURE':
            self._P = self.values['P']
            self._t = self.values['t']
        
        if self.type == 'IMPEDANCE':
            self._Z = self.values['z']
    
    @classmethod
    def from_config(cls, config):
        '''
        create a boundary condition from a config dict

        :param config: config dict
        '''

        return cls(config)
    
    def to_dict(self):
        '''
        convert the boundary condition to a dict for zerod solver use
        '''

        return {
            'bc_name': self.name,
            'bc_type': self.type,
            'bc_values': self.values
        }
    
    def RCR_to_R(self):
        '''
        change the boundary condition to a resistance
        '''
        self.values = {'R': self.values['Rd'] + self.values['Rp'],
                       'Pd': self.values['Pd']}

        self.type = 'RESISTANCE'

        self._R = self.values['R']
    
    def Z_to_R(self):
        '''
        change from impedance boundary condition to resistance'''

        self.values = {'R': self.Z[0],
                       'Pd': self.values['Pd']}
        
        self.type = 'RESISTANCE'
        self._R = self.values['R']


    def R_to_RCR(self):
        '''
        change the boundary condition to RCR
        '''
        self.values = {'Rp': 0.1 * self.values['R'],
                       'Rd': 0.9 * self.values['R'],
                       'C': 1e-5,
                       'Pd': self.values['Pd']}
        
        self.type = 'RCR'

        self._Rp = self.values['Rp']
        self._Rd = self.values['Rd']
        self._C = self.values['C']
    
    # a setter so we can change the resistances in the BC easier
    @property
    def R(self):
        if self.type == 'RESISTANCE':
            return self._R
        elif self.type == 'RCR':
            return self._Rp + self._Rd

    @R.setter
    def R(self, new_R):
        if self.type == 'RESISTANCE':
            self._R = new_R
            self.values['R'] = new_R
        if self.type == 'RCR':
            self._Rp = 0.1 * new_R
            self._Rd = 0.9 * new_R
            self.values['Rp'] = self._Rp
            self.values['Rd'] = self._Rd
    
    @property
    def Rp(self):
        return self._Rp
    
    @Rp.setter
    def Rp(self, new_Rp):
        self._Rp = new_Rp
        self.values['Rp'] = new_Rp

    @property
    def Rd(self):
        return self._Rd
    
    @Rd.setter
    def Rd(self, new_Rd):
        self._Rd = new_Rd
        self.values['Rd'] = new_Rd

    @property
    def C(self):
        return self._C
    
    @C.setter
    def C(self, new_C):
        self._C = new_C
        self.values['C'] = new_C

    @property
    def Q(self):
        return self._Q
    
    @Q.setter
    def Q(self, new_Q):
        self._Q = new_Q
        self.values['Q'] = new_Q

    @property
    def P(self):
        return self._P
    
    @P.setter
    def P(self, new_P):
        self._P = new_P
        self.values['P'] = new_P

    @property
    def t(self):
        return self._t
    
    @t.setter
    def t(self, new_t):
        self._t = new_t
        self.values['t'] = new_t
    
    @property
    def Z(self):
        return self._Z
    
    @Z.setter
    def Z(self, new_Z):
        normalized = validate_impedance_bc_values(
            {**self.values, 'z': new_Z},
            bc_name=self.name,
        )
        self._Z = normalized['z']
        self.values = normalized
   
