
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np


_ALLOWED_IMPEDANCE_KEYS = {"z", "Pd", "convolution_mode", "num_kernel_terms"}
_FORBIDDEN_IMPEDANCE_KEYS = {"Z", "tree", "t"}
_COUPLED_IMPEDANCE_SIMPARAM_KEYS = {
    "coupled_simulation",
    "number_of_time_pts",
    "output_all_cycles",
    "steady_initial",
    "density",
    "viscosity",
    "external_step_size",
    "cardiac_period",
}


def _ensure_finite_numeric(value, *, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    return numeric


def _ensure_numeric_sequence(values, *, bc_name: str, field_name: str, bc_type: str) -> list[float]:
    if isinstance(values, (str, bytes)):
        raise ValueError(
            f"{bc_type} boundary condition '{bc_name}' field '{field_name}' must be a non-empty numeric sequence"
        )
    try:
        normalized = [
            _ensure_finite_numeric(
                entry,
                label=f"{bc_type} boundary condition '{bc_name}' field '{field_name}'",
            )
            for entry in values
        ]
    except TypeError as exc:
        raise ValueError(
            f"{bc_type} boundary condition '{bc_name}' field '{field_name}' must be a non-empty numeric sequence"
        ) from exc
    if not normalized:
        raise ValueError(
            f"{bc_type} boundary condition '{bc_name}' field '{field_name}' must be a non-empty numeric sequence"
        )
    return normalized


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
            _ensure_finite_numeric(
                entry,
                label=f"IMPEDANCE boundary condition '{bc_name}' field 'z'",
            )
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
        "Pd": _ensure_finite_numeric(
            values["Pd"],
            label=f"IMPEDANCE boundary condition '{bc_name}' field 'Pd'",
        ),
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


def _find_boundary_condition_config(config: Mapping, *, bc_name: str) -> Mapping:
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")

    boundary_conditions = config.get("boundary_conditions", [])
    if isinstance(boundary_conditions, (str, bytes)) or not isinstance(boundary_conditions, Sequence):
        raise ValueError("config boundary_conditions must be a sequence")

    resolved_name = str(bc_name).strip() or "<unknown>"
    for entry in boundary_conditions:
        if not isinstance(entry, Mapping):
            raise ValueError("boundary condition config must be a mapping")
        if str(entry.get("bc_name", "")).strip() == resolved_name:
            return entry

    raise ValueError(f"config must include boundary condition '{resolved_name}'")


def _resolve_flow_cardiac_output_config(config: Mapping, *, bc_name: str = "INFLOW") -> float:
    flow_config = _find_boundary_condition_config(config, bc_name=bc_name)
    resolved_name = str(bc_name).strip() or "<unknown>"
    bc_type = str(flow_config.get("bc_type", "")).strip()
    if bc_type != "FLOW":
        raise ValueError(
            f"boundary condition '{resolved_name}' must have bc_type 'FLOW'; got '{bc_type or '<missing>'}'"
        )

    values = flow_config.get("bc_values")
    if not isinstance(values, Mapping):
        raise ValueError(
            f"FLOW boundary condition '{resolved_name}' bc_values must be a mapping"
        )

    missing = [key for key in ("Q", "t") if key not in values]
    if missing:
        raise ValueError(
            f"FLOW boundary condition '{resolved_name}' is missing required keys: "
            + ", ".join(missing)
        )

    flow_values = _ensure_numeric_sequence(
        values.get("Q"),
        bc_name=resolved_name,
        field_name="Q",
        bc_type="FLOW",
    )
    time_values = _ensure_numeric_sequence(
        values.get("t"),
        bc_name=resolved_name,
        field_name="t",
        bc_type="FLOW",
    )
    if len(flow_values) != len(time_values):
        raise ValueError(
            f"FLOW boundary condition '{resolved_name}' fields 'Q' and 't' must have the same length"
        )
    if len(flow_values) < 2:
        raise ValueError(
            f"FLOW boundary condition '{resolved_name}' requires at least 2 samples to compute cardiac output"
        )

    return float(np.trapz(np.asarray(flow_values, dtype=float), np.asarray(time_values, dtype=float)))


def _resolve_flow_mean_config(config: Mapping, *, bc_name: str = "INFLOW") -> float:
    flow_config = _find_boundary_condition_config(config, bc_name=bc_name)
    resolved_name = str(bc_name).strip() or "<unknown>"
    bc_type = str(flow_config.get("bc_type", "")).strip()
    if bc_type != "FLOW":
        raise ValueError(
            f"boundary condition '{resolved_name}' must have bc_type 'FLOW'; got '{bc_type or '<missing>'}'"
        )

    values = flow_config.get("bc_values")
    if not isinstance(values, Mapping):
        raise ValueError(
            f"FLOW boundary condition '{resolved_name}' bc_values must be a mapping"
        )

    missing = [key for key in ("Q", "t") if key not in values]
    if missing:
        raise ValueError(
            f"FLOW boundary condition '{resolved_name}' is missing required keys: "
            + ", ".join(missing)
        )

    flow_values = _ensure_numeric_sequence(
        values.get("Q"),
        bc_name=resolved_name,
        field_name="Q",
        bc_type="FLOW",
    )
    time_values = _ensure_numeric_sequence(
        values.get("t"),
        bc_name=resolved_name,
        field_name="t",
        bc_type="FLOW",
    )
    if len(flow_values) != len(time_values):
        raise ValueError(
            f"FLOW boundary condition '{resolved_name}' fields 'Q' and 't' must have the same length"
        )
    if len(flow_values) < 2:
        raise ValueError(
            f"FLOW boundary condition '{resolved_name}' requires at least 2 samples to compute cardiac output"
        )
    return float(np.mean(np.asarray(flow_values, dtype=float)))


def validate_flow_cardiac_output_config(
    config: Mapping,
    *,
    expected_cardiac_output,
    bc_name: str = "INFLOW",
    rel_tol: float = 1e-5,
    abs_tol: float = 1e-8,
) -> float:
    measured_cardiac_output = _resolve_flow_mean_config(config, bc_name=bc_name)
    try:
        expected = float(expected_cardiac_output)
    except (TypeError, ValueError) as exc:
        raise ValueError("expected_cardiac_output must be numeric") from exc
    if not math.isfinite(expected):
        raise ValueError("expected_cardiac_output must be finite")

    if not math.isclose(measured_cardiac_output, expected, rel_tol=rel_tol, abs_tol=abs_tol):
        raise ValueError(
            f"FLOW boundary condition '{bc_name}' cardiac output mismatch: "
            f"expected {expected:.12g}, got {measured_cardiac_output:.12g}"
        )
    return measured_cardiac_output


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


def _resolve_flow_sample_count_config(config: Mapping, *, bc_name: str = "INFLOW") -> int:
    flow_config = _find_boundary_condition_config(config, bc_name=bc_name)
    resolved_name = str(bc_name).strip() or "<unknown>"
    bc_type = str(flow_config.get("bc_type", "")).strip()
    if bc_type != "FLOW":
        raise ValueError(
            f"boundary condition '{resolved_name}' must have bc_type 'FLOW'; got '{bc_type or '<missing>'}'"
        )

    values = flow_config.get("bc_values")
    if not isinstance(values, Mapping):
        raise ValueError(
            f"FLOW boundary condition '{resolved_name}' bc_values must be a mapping"
        )

    missing = [key for key in ("Q", "t") if key not in values]
    if missing:
        raise ValueError(
            f"FLOW boundary condition '{resolved_name}' is missing required keys: "
            + ", ".join(missing)
        )

    flow_values = _ensure_numeric_sequence(
        values.get("Q"),
        bc_name=resolved_name,
        field_name="Q",
        bc_type="FLOW",
    )
    time_values = _ensure_numeric_sequence(
        values.get("t"),
        bc_name=resolved_name,
        field_name="t",
        bc_type="FLOW",
    )
    if len(flow_values) != len(time_values):
        raise ValueError(
            f"FLOW boundary condition '{resolved_name}' fields 'Q' and 't' must have the same length"
        )
    if len(time_values) < 2:
        raise ValueError(
            f"FLOW boundary condition '{resolved_name}' requires at least 2 samples"
        )
    return len(time_values)


def resolve_coupled_impedance_timepoint_contract(config: Mapping) -> tuple[int, int]:
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")

    simparams = config.get("simulation_parameters")
    if not isinstance(simparams, Mapping):
        raise ValueError(
            "configs with IMPEDANCE boundary conditions require simulation_parameters"
        )

    simparam_keys = set(simparams.keys())
    missing_keys = sorted(_COUPLED_IMPEDANCE_SIMPARAM_KEYS - simparam_keys)
    if missing_keys:
        raise ValueError(
            "coupled config with IMPEDANCE boundary conditions requires simulation_parameters keys: "
            + ", ".join(missing_keys)
        )
    extra_keys = sorted(simparam_keys - _COUPLED_IMPEDANCE_SIMPARAM_KEYS)
    if extra_keys:
        raise ValueError(
            "coupled config with IMPEDANCE boundary conditions does not allow extra "
            "simulation_parameters keys: "
            + ", ".join(extra_keys)
        )

    if not bool(simparams.get("coupled_simulation", False)):
        raise ValueError(
            "resolve_coupled_impedance_timepoint_contract requires coupled_simulation = true"
        )

    resolved_number_of_time_pts = _ensure_positive_int(
        simparams.get("number_of_time_pts"),
        field_name="simulation_parameters.number_of_time_pts",
    )
    if resolved_number_of_time_pts != 2:
        raise ValueError(
            "coupled config with IMPEDANCE boundary conditions requires "
            "simulation_parameters.number_of_time_pts = 2; got "
            f"{resolved_number_of_time_pts}"
        )

    if simparams.get("external_step_size") is None:
        raise ValueError(
            "coupled config with IMPEDANCE boundary conditions requires "
            "simulation_parameters.external_step_size"
        )
    external_step_size = _ensure_finite_numeric(
        simparams.get("external_step_size"),
        label="simulation_parameters.external_step_size",
    )
    if external_step_size <= 0.0:
        raise ValueError("simulation_parameters.external_step_size must be > 0")

    if simparams.get("cardiac_period") is None:
        raise ValueError(
            "coupled config with IMPEDANCE boundary conditions requires "
            "simulation_parameters.cardiac_period"
        )
    cardiac_period = _ensure_finite_numeric(
        simparams.get("cardiac_period"),
        label="simulation_parameters.cardiac_period",
    )
    if cardiac_period <= 0.0:
        raise ValueError("simulation_parameters.cardiac_period must be > 0")

    validated_bcs = validate_boundary_condition_configs(
        config.get("boundary_conditions", [])
    )
    impedance_bcs = [
        bc_config for bc_config in validated_bcs if bc_config.get("bc_type") == "IMPEDANCE"
    ]
    if not impedance_bcs:
        raise ValueError(
            "coupled config with IMPEDANCE boundary conditions requires at least one IMPEDANCE boundary condition"
        )

    sample_count = _resolve_flow_sample_count_config(config)
    kernel_sizes = {len(bc_config["bc_values"]["z"]) for bc_config in impedance_bcs}
    if len(kernel_sizes) != 1:
        raise ValueError(
            "coupled IMPEDANCE boundary conditions must all use the same len(z); got "
            + ", ".join(str(size) for size in sorted(kernel_sizes))
        )

    return sample_count, next(iter(kernel_sizes))


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
        resolve_coupled_impedance_timepoint_contract(config)
        return
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
   
