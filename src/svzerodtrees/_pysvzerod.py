from __future__ import annotations

import importlib
import inspect
from functools import lru_cache
import json
import os
from pathlib import Path
import time

REQUIRED_CALIBRATION_CAPABILITIES = frozenset(
    {
        "per_block_parameter_selection",
    }
)

_INSTALL_HINT = (
    "pysvzerod is required for solver-backed svZeroDTrees workflows. "
    "Install the sibling svZeroDSolver checkout first with "
    "`python3 -m pip install -e ../svZeroDSolver` "
    "(or `python3 -m pip install -e /home/users/ndorn/svZeroDSolver` on Sherlock)."
)

_last_calibration_provenance: dict[str, object] | None = None


@lru_cache(maxsize=1)
def require_pysvzerod():
    try:
        return importlib.import_module("pysvzerod")
    except ModuleNotFoundError as exc:
        if exc.name != "pysvzerod":
            raise
        raise ModuleNotFoundError(_INSTALL_HINT) from exc


def _build_identity(module) -> object:
    identity = getattr(module, "build_identity", None)
    if callable(identity):
        return identity()
    return getattr(module, "__build_identity__", None)


def pysvzerod_provenance(module=None) -> dict[str, object]:
    """Return the loaded solver's import location and build provenance."""
    if module is None:
        module = require_pysvzerod()
    module_path = getattr(module, "__file__", None)
    return {
        "module_path": str(Path(module_path).resolve()) if module_path else "<unknown>",
        "version": getattr(module, "__version__", None),
        "build_identity": _build_identity(module),
    }


def require_calibration_capabilities() -> dict[str, object]:
    """Load pysvzerod and verify the calibration API contract."""
    module = require_pysvzerod()
    provenance = pysvzerod_provenance(module)
    capabilities_fn = getattr(module, "capabilities", None)
    try:
        capabilities = capabilities_fn() if callable(capabilities_fn) else None
    except Exception as exc:
        capabilities = f"<capabilities() failed: {type(exc).__name__}: {exc}>"

    if not isinstance(capabilities, dict):
        capabilities_for_error = capabilities
        missing = sorted(REQUIRED_CALIBRATION_CAPABILITIES)
    else:
        missing = sorted(
            capability
            for capability in REQUIRED_CALIBRATION_CAPABILITIES
            if capabilities.get(capability) is not True
        )
        capabilities_for_error = capabilities

    if missing:
        raise RuntimeError(
            "Incompatible pysvzerod calibrator: missing required capability "
            f"{', '.join(missing)}. Imported module path: "
            f"{provenance['module_path']}; version: {provenance['version']!r}; "
            f"build identity: {provenance['build_identity']!r}; capabilities: "
            f"{capabilities_for_error!r}. Reinstall the pinned sibling solver "
            "with `python3 -m pip install -e ../svZeroDSolver` "
            "(or `python3 -m pip install -e /home/users/ndorn/svZeroDSolver` "
            "on Sherlock)."
        )

    return {
        **provenance,
        "capabilities": capabilities,
    }


def last_calibration_provenance() -> dict[str, object] | None:
    """Return provenance for the most recent successful calibration dispatch."""
    return _last_calibration_provenance


def clear_calibration_provenance() -> None:
    """Clear the provenance cache before beginning a calibration dispatch."""
    global _last_calibration_provenance
    _last_calibration_provenance = None


def _trace_destination() -> Path | None:
    raw = os.environ.get("SVZEROD_TRACE_FILE", "").strip()
    return Path(raw).expanduser() if raw else None


def _caller_payload() -> dict[str, object]:
    for frame_info in inspect.stack()[2:]:
        module = inspect.getmodule(frame_info.frame)
        module_name = module.__name__ if module is not None else ""
        if module_name != __name__:
            return {
                "module": module_name or None,
                "function": frame_info.function,
                "file": frame_info.filename,
                "line": frame_info.lineno,
            }
    return {}


def _config_payload(config) -> dict[str, object]:
    payload: dict[str, object] = {
        "config_type": type(config).__name__,
    }
    if not isinstance(config, dict):
        return payload

    boundary_conditions = config.get("boundary_conditions") or []
    bc_types: dict[str, int] = {}
    bc_names: list[str] = []
    inflow_points = None
    for bc in boundary_conditions:
        if not isinstance(bc, dict):
            continue
        bc_name = bc.get("bc_name")
        if isinstance(bc_name, str):
            bc_names.append(bc_name)
        bc_type = str(bc.get("bc_type", "UNKNOWN"))
        bc_types[bc_type] = bc_types.get(bc_type, 0) + 1
        if bc_name == "INFLOW":
            values = bc.get("bc_values") or {}
            t_values = values.get("t")
            if isinstance(t_values, list):
                inflow_points = len(t_values)

    vessels = config.get("vessels") or []
    vessel_names = [
        vessel.get("vessel_name")
        for vessel in vessels
        if isinstance(vessel, dict) and isinstance(vessel.get("vessel_name"), str)
    ]
    simparams = config.get("simulation_parameters") or {}

    payload.update(
        {
            "boundary_condition_count": len(boundary_conditions),
            "boundary_condition_types": bc_types,
            "boundary_condition_names": bc_names,
            "vessel_count": len(vessels),
            "vessel_names": vessel_names[:8],
            "junction_count": len(config.get("junctions") or []),
            "inflow_points": inflow_points,
            "time_points_per_cycle": simparams.get("number_of_time_pts_per_cardiac_cycle"),
            "cardiac_cycles": simparams.get("number_of_cardiac_cycles"),
            "coupled_simulation": simparams.get("coupled_simulation"),
            "steady_initial": simparams.get("steady_initial"),
        }
    )
    return payload


def _write_trace(event: dict[str, object]) -> None:
    destination = _trace_destination()
    if destination is None:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def simulate_pysvzerod(config):
    call_id = f"{os.getpid()}-{time.time_ns()}"
    base_event = {
        "call_id": call_id,
        "pid": os.getpid(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "caller": _caller_payload(),
        "config": _config_payload(config),
    }
    _write_trace(
        {
            **base_event,
            "phase": "enter",
        }
    )
    try:
        result = require_pysvzerod().simulate(config)
    except Exception as exc:
        _write_trace(
            {
                **base_event,
                "phase": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        raise
    row_count = len(result) if hasattr(result, "__len__") else None
    _write_trace(
        {
            **base_event,
            "phase": "return",
            "result_type": type(result).__name__,
            "result_len": row_count,
        }
    )
    return result


def calibrate_pysvzerod(config):
    global _last_calibration_provenance
    module = require_pysvzerod()
    provenance = require_calibration_capabilities()
    calibrate = getattr(module, "calibrate", None)
    if not callable(calibrate):
        raise RuntimeError(
            "Incompatible pysvzerod calibrator: imported module does not expose "
            "calibrate(). Imported module path: "
            f"{pysvzerod_provenance(module)['module_path']}. "
            f"{_INSTALL_HINT}"
        )
    result = calibrate(config)
    _last_calibration_provenance = provenance
    return result
