from __future__ import annotations

import hashlib
import importlib
import inspect
from functools import lru_cache
import json
import os
from pathlib import Path
import time

REQUIRED_SOLVER_CALLABLES = ("calibrate", "simulate")

_INSTALL_HINT = (
    "pysvzerod is required for solver-backed svZeroDTrees workflows. "
    "Install the sibling svZeroDSolver checkout first with `python3 -m pip "
    "install -e "
    "../svZeroDSolver` (or `python3 -m pip install -e "
    "/home/users/ndorn/svZeroDSolver` on Sherlock), or install the pinned "
    "solver with `uv sync --group solver`."
)

_last_calibration_provenance: dict[str, object] | None = None


@lru_cache(maxsize=1)
def require_pysvzerod():
    try:
        return importlib.import_module("pysvzerod")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import pysvzerod. Resolved module state: "
            f"<unresolved; missing {exc.name or 'import dependency'}>; "
            "required callables: calibrate(config), "
            f"simulate(config). {_INSTALL_HINT} Original error: {exc}"
        ) from exc
    except ImportError as exc:
        raise ImportError(
            "Could not import pysvzerod. Resolved module state: "
            "<import failed>; required callables: calibrate(config), "
            f"simulate(config). {_INSTALL_HINT} Original error: {exc}"
        ) from exc


def _build_identity(module) -> object:
    identity = getattr(module, "build_identity", None)
    if callable(identity):
        try:
            return identity()
        except Exception as exc:
            return f"<build_identity() failed: {type(exc).__name__}: {exc}>"
    return getattr(module, "__build_identity__", None)


def _module_file_provenance(module) -> tuple[str, dict[str, object], str | None]:
    module_path = getattr(module, "__file__", None)
    if not module_path:
        return "<unknown>", {
            "size": None,
            "mtime_ns": None,
            "mode": None,
            "readable": False,
        }, None

    resolved_path = Path(module_path).expanduser().resolve()
    metadata: dict[str, object] = {
        "size": None,
        "mtime_ns": None,
        "mode": None,
        "readable": False,
    }
    digest = None
    try:
        stat_result = resolved_path.stat()
        metadata.update(
            {
                "size": stat_result.st_size,
                "mtime_ns": stat_result.st_mtime_ns,
                "mode": stat_result.st_mode,
                "readable": os.access(resolved_path, os.R_OK),
            }
        )
        if metadata["readable"] and resolved_path.is_file():
            file_hash = hashlib.sha256()
            with resolved_path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    file_hash.update(chunk)
            digest = file_hash.hexdigest()
    except OSError:
        # Importable extension modules can be represented by a path that is
        # no longer readable after import. Keep the identity inspectable while
        # making the missing digest explicit.
        pass
    return str(resolved_path), metadata, digest


def pysvzerod_provenance(module=None) -> dict[str, object]:
    """Return the loaded solver's import location and build provenance.

    The module digest is best-effort because some import loaders expose a
    path for an extension that cannot be read after it has been loaded. The
    path and stat metadata remain useful diagnostics in that case.
    """
    if module is None:
        module = require_pysvzerod()
    module_path, file_metadata, module_sha256 = _module_file_provenance(module)
    return {
        "module_path": module_path,
        "module_sha256": module_sha256,
        "file_metadata": file_metadata,
        "version": getattr(module, "__version__", None),
        "build_identity": _build_identity(module),
    }


def _solver_identity_for_error(provenance: dict[str, object]) -> str:
    return (
        f"module path: {provenance['module_path']}; "
        f"module SHA-256: {provenance['module_sha256']!r}; "
        f"file metadata: {provenance['file_metadata']!r}; "
        f"version: {provenance['version']!r}; "
        f"build identity: {provenance['build_identity']!r}"
    )


def require_pysvzerod_api() -> dict[str, object]:
    """Load pysvzerod and verify its standard calibration API contract."""
    module = require_pysvzerod()
    provenance = pysvzerod_provenance(module)

    missing = [
        name
        for name in REQUIRED_SOLVER_CALLABLES
        if not callable(getattr(module, name, None))
    ]
    if missing:
        missing_list = ", ".join(f"{name}(config)" for name in missing)
        raise RuntimeError(
            "Incompatible pysvzerod module: required callable API missing: "
            f"{missing_list}. Resolved module state: "
            f"{_solver_identity_for_error(provenance)}. {_INSTALL_HINT}"
        )

    return provenance


def require_calibration_capabilities() -> dict[str, object]:
    """Backward-compatible name for standard solver API validation.

    Calibration no longer depends on a custom ``capabilities()`` method. The
    name remains available for callers that imported this helper before the
    standard API contract was adopted.
    """
    return require_pysvzerod_api()


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
        module = require_pysvzerod()
        provenance = pysvzerod_provenance(module)
        simulate = getattr(module, "simulate", None)
        if not callable(simulate):
            raise RuntimeError(
                "Incompatible pysvzerod module: required callable API missing: "
                f"simulate(config). Resolved module state: "
                f"{_solver_identity_for_error(provenance)}. {_INSTALL_HINT}"
            )
        result = simulate(config)
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
    provenance = require_pysvzerod_api()
    module = require_pysvzerod()
    calibrate = module.calibrate
    result = calibrate(config)
    _last_calibration_provenance = provenance
    return result
