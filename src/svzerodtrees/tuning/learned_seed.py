"""Helpers for preparing reduced seeds and generating learned full-PA seeds."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping
import json
import math
from numbers import Real
import os
import shutil
import subprocess
import tempfile
import time

import numpy as np
from scipy.optimize import Bounds, minimize

from ..io.blocks import BoundaryCondition
from ..io.config_handler import ConfigHandler
from ..config import LearnedSeedGenerationConfig
from ..numerics import trapezoid
from ..tune_bcs.clinical_targets import ClinicalTargets
from ..tune_bcs.pa_config import PAConfig

MMHG_TO_BARYE = 1333.2
DEFAULT_SIDE_BC_RESISTANCE = 1000.0
DEFAULT_SIDE_BC_PD = 0.0


@dataclass(frozen=True)
class LearnedSeedResult:
    """Paths and provenance for a successfully generated full-PA seed."""

    seed_path: Path
    metadata_path: Path
    metadata: dict[str, Any]

    @property
    def generated_seed_path(self) -> Path:
        """Alias used by callers that prefer an explicit generated-path name."""

        return self.seed_path

    def __getitem__(self, key: str) -> Any:
        """Provide a small mapping-compatible surface for workflow adapters."""

        if key in {"seed_path", "generated_seed_path", "seed"}:
            return self.seed_path
        if key in {"metadata_path", "learned_seed_metadata"}:
            return self.metadata_path
        if key == "metadata":
            return self.metadata
        raise KeyError(key)


def _absolute_file(path: str | os.PathLike[str], *, label: str, executable: bool = False) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found or is not a file: {resolved}")
    if not os.access(resolved, os.R_OK):
        raise PermissionError(f"{label} is not readable: {resolved}")
    if executable and not os.access(resolved, os.X_OK):
        raise PermissionError(f"{label} is not executable: {resolved}")
    return resolved


def _resolve_executable(value: str | os.PathLike[str], *, label: str) -> Path:
    """Resolve a path or PATH command and require an executable file."""

    text = os.fspath(value)
    path_like = os.path.isabs(text) or os.path.dirname(text)
    if path_like:
        return _absolute_file(text, label=label, executable=True)

    resolved = shutil.which(text)
    if resolved is None:
        raise FileNotFoundError(
            f"{label} '{text}' is not available as an executable on PATH"
        )
    return _absolute_file(resolved, label=label, executable=True)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _learned_zerod_version() -> str | None:
    """Return package metadata when installed, without importing learnedZeroD."""

    for distribution in ("learnedzerod", "learned-zerod", "learnedZeroD"):
        try:
            return importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            continue
    return None


def _full_pa_outlet_names(payload: Mapping[str, Any]) -> tuple[list[str], list[Mapping[str, Any]]]:
    boundary_conditions = payload.get("boundary_conditions")
    if not isinstance(boundary_conditions, list):
        raise ValueError(
            "learned full-PA seed must contain a boundary_conditions list"
        )

    outlet_bcs: list[Mapping[str, Any]] = []
    names: list[str] = []
    for bc in boundary_conditions:
        if not isinstance(bc, Mapping):
            raise ValueError("learned full-PA boundary conditions must be mappings")
        name = bc.get("bc_name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "learned full-PA boundary conditions require non-empty bc_name values"
            )
        normalized_name = name.strip()
        bc_type = str(bc.get("bc_type", "")).strip().upper()
        if normalized_name.upper() == "INFLOW" or bc_type == "FLOW":
            continue
        names.append(normalized_name)
        outlet_bcs.append(bc)

    if len(names) <= 2:
        raise ValueError(
            "learned full-PA seed requires more than two non-inflow outlet "
            f"boundary conditions; found {len(names)}"
        )
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(
            "learned full-PA seed has duplicate outlet boundary-condition names: "
            + ", ".join(duplicates)
        )
    return names, outlet_bcs


def _validate_full_pa_seed_payload(payload: Any) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("learned full-PA seed JSON must contain an object at the top level")

    vessels = payload.get("vessels")
    if not isinstance(vessels, list) or not vessels:
        raise ValueError("learned full-PA seed JSON must contain a non-empty vessels list")
    if any(not isinstance(vessel, Mapping) for vessel in vessels):
        raise ValueError("learned full-PA seed vessels must be mappings")

    outlet_names, _ = _full_pa_outlet_names(payload)
    outlet_set = set(outlet_names)
    attached: list[str] = []
    for vessel in vessels:
        vessel_bcs = vessel.get("boundary_conditions")
        if not isinstance(vessel_bcs, Mapping):
            continue
        outlet = vessel_bcs.get("outlet")
        if outlet is None:
            continue
        if not isinstance(outlet, str) or not outlet.strip():
            raise ValueError(
                "learned full-PA seed vessel outlet attachments must be non-empty strings"
            )
        attached.append(outlet.strip())

    unknown = sorted(set(attached) - outlet_set)
    if unknown:
        raise ValueError(
            "learned full-PA seed has vessel outlet attachments without matching "
            "boundary conditions: " + ", ".join(unknown)
        )
    attached_set = set(attached)
    missing = sorted(outlet_set - attached_set)
    duplicates = sorted({name for name in attached if attached.count(name) > 1})
    if missing or duplicates or len(attached) != len(outlet_names):
        details = []
        if missing:
            details.append("missing=" + ", ".join(missing))
        if duplicates:
            details.append("duplicate=" + ", ".join(duplicates))
        raise ValueError(
            "learned full-PA seed must provide exactly one vessel outlet attachment "
            + ("(" + "; ".join(details) + ")" if details else "")
        )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON to a sibling temporary file and atomically publish it."""

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _subprocess_failure(
    *,
    command: list[str],
    error: BaseException,
    stdout: str = "",
    stderr: str = "",
) -> RuntimeError:
    command_text = " ".join(command)
    details = [f"learned-zerod failed for command: {command_text}"]
    if isinstance(error, subprocess.CalledProcessError):
        details.append(f"exit status: {error.returncode}")
        stdout = str(error.stdout or stdout or "")
        stderr = str(error.stderr or stderr or "")
    else:
        details.append(str(error))
    if stdout.strip():
        details.append("stdout: " + stdout.strip())
    if stderr.strip():
        details.append("stderr: " + stderr.strip())
    return RuntimeError("; ".join(details))


def generate_full_pa_learned_seed(
    config: LearnedSeedGenerationConfig,
) -> LearnedSeedResult:
    """Generate and publish a validated learned full pulmonary 0D seed.

    learnedZeroD is intentionally treated as an external executable.  The
    generated JSON is validated before it is moved into the configured output
    directory, and the source artifacts are only read for validation/digests.
    """

    if not isinstance(config, LearnedSeedGenerationConfig):
        raise TypeError(
            "generate_full_pa_learned_seed requires a LearnedSeedGenerationConfig"
        )
    if config.method != "learned_zerod":
        raise ValueError("seed_generation.method must be 'learned_zerod'")
    if config.anatomy != "pulmonary":
        raise ValueError("seed_generation.anatomy must be 'pulmonary'")

    input_path = _absolute_file(config.input_zerod_config, label="input 0D JSON")
    centerline_path = _absolute_file(config.centerline, label="centerline VTP")
    solver_path = _resolve_executable(config.svzerodsolver, label="svZeroDSolver")
    learned_executable = _resolve_executable(
        config.learned_zerod_executable, label="learned-zerod"
    )

    input_bytes = input_path.read_bytes()
    try:
        json.loads(input_bytes)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"input 0D JSON is not valid JSON: {input_path}") from exc
    # Reading the centerline here makes unreadable/special input fail before
    # the external process is invoked, while leaving it otherwise opaque.
    centerline_path.read_bytes()
    input_digest = _sha256_file(input_path)
    centerline_digest = _sha256_file(centerline_path)
    solver_digest = _sha256_file(solver_path)

    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = str(config.output_filename).strip()
    if (
        not output_filename
        or Path(output_filename).name != output_filename
        or output_filename in {".", ".."}
    ):
        raise ValueError(
            "seed_generation.output_filename must be a simple filename, not a path"
        )
    final_seed_path = output_dir / output_filename
    metadata_path = output_dir / "learned_seed_metadata.json"

    staging_path = Path(tempfile.mkdtemp(prefix=".learned_seed-", dir=str(output_dir)))
    command: list[str] = []
    started_at = _timestamp()
    started_clock = time.perf_counter()
    published_seed = False
    published_metadata = False
    try:
        command = [
            str(learned_executable),
            "--anatomy",
            "pulmonary",
            "--zerod-json",
            str(input_path),
            "--centerline-vtp",
            str(centerline_path),
            "--svzerod",
            str(solver_path),
            "--output-dir",
            str(staging_path),
            "--output-filename",
            output_filename,
        ]

        # A stable manifest must never describe an earlier successful run if
        # this attempted run fails after process invocation begins.
        metadata_path.unlink(missing_ok=True)
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                shell=False,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise _subprocess_failure(command=command, error=exc) from exc

        return_code = getattr(completed, "returncode", 0)
        if return_code not in (None, 0):
            raise _subprocess_failure(
                command=command,
                error=subprocess.CalledProcessError(
                    return_code,
                    command,
                    output=getattr(completed, "stdout", ""),
                    stderr=getattr(completed, "stderr", ""),
                ),
            )

        staged_seed_path = staging_path / output_filename
        if not staged_seed_path.is_file():
            raise RuntimeError(
                "learned-zerod completed successfully but did not produce the "
                f"requested output: {staged_seed_path}"
            )
        try:
            generated_payload = json.loads(staged_seed_path.read_bytes())
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError(
                f"learned-zerod output is not valid JSON: {staged_seed_path}"
            ) from exc
        _validate_full_pa_seed_payload(generated_payload)

        generated_digest = _sha256_file(staged_seed_path)
        # os.replace is atomic when source and destination share output_dir's
        # filesystem; the staging directory is deliberately created there.
        os.replace(staged_seed_path, final_seed_path)
        published_seed = True
        finished_at = _timestamp()
        duration_seconds = time.perf_counter() - started_clock
        paths = {
            "input_zerod_config": str(input_path),
            "centerline": str(centerline_path),
            "svzerodsolver": str(solver_path),
            "seed": str(final_seed_path),
            "metadata": str(metadata_path),
        }
        digests = {
            "input_zerod_config": input_digest,
            "input_json": input_digest,
            "centerline": centerline_digest,
            "svzerodsolver": solver_digest,
            "solver": solver_digest,
            "generated_seed": generated_digest,
            "generated_json": generated_digest,
        }
        metadata: dict[str, Any] = {
            "schema": "svzerodtrees.learned_seed_metadata",
            "schema_version": 1,
            "version": 1,
            "method": "learned_zerod",
            "status": "success",
            "success": True,
            "anatomy": "pulmonary",
            "paths": paths,
            "source_paths": {
                "input_zerod_config": str(input_path),
                "centerline": str(centerline_path),
                "svzerodsolver": str(solver_path),
            },
            "output_paths": {
                "seed": str(final_seed_path),
                "metadata": str(metadata_path),
            },
            "digests": digests,
            "sha256": digests,
            "input_json_sha256": input_digest,
            "centerline_sha256": centerline_digest,
            "generated_json_sha256": generated_digest,
            "solver_sha256": solver_digest,
            "command": {
                "argv": command,
                "executable": str(learned_executable),
                "identity": str(learned_executable),
            },
            "learned_zerod_version": _learned_zerod_version(),
            "command_argv": command,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": duration_seconds,
        }
        _atomic_write_json(metadata_path, metadata)
        published_metadata = True
        return LearnedSeedResult(
            seed_path=final_seed_path,
            metadata_path=metadata_path,
            metadata=metadata,
        )
    except Exception:
        # The seed is published only immediately before its metadata.  If
        # metadata publication fails, remove that just-published seed so
        # callers cannot mistake an un-described artifact for a success.
        if published_seed and not published_metadata:
            final_seed_path.unlink(missing_ok=True)
        raise
    finally:
        if not bool(config.keep_tmp):
            shutil.rmtree(staging_path, ignore_errors=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(entry) for entry in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(entry) for entry in value]
    return value


def _normalize_resistance_bc(
    bc: float | dict[str, Any] | BoundaryCondition | None,
    *,
    name: str,
) -> BoundaryCondition:
    if bc is None:
        config = {
            "bc_name": name,
            "bc_type": "RESISTANCE",
            "bc_values": {
                "R": DEFAULT_SIDE_BC_RESISTANCE,
                "Pd": DEFAULT_SIDE_BC_PD,
            },
        }
    elif isinstance(bc, BoundaryCondition):
        config = {
            "bc_name": name,
            "bc_type": bc.type,
            "bc_values": deepcopy(bc.values),
        }
    elif isinstance(bc, Real):
        config = {
            "bc_name": name,
            "bc_type": "RESISTANCE",
            "bc_values": {
                "R": float(bc),
                "Pd": DEFAULT_SIDE_BC_PD,
            },
        }
    elif isinstance(bc, dict):
        bc_copy = deepcopy(bc)
        if "bc_values" in bc_copy:
            config = {
                "bc_name": name,
                "bc_type": bc_copy.get("bc_type", "RESISTANCE"),
                "bc_values": bc_copy["bc_values"],
            }
        else:
            values = dict(bc_copy)
            values.setdefault("Pd", DEFAULT_SIDE_BC_PD)
            config = {
                "bc_name": name,
                "bc_type": "RESISTANCE",
                "bc_values": values,
            }
    else:
        raise TypeError(f"{name} must be a resistance value, dict, BoundaryCondition, or None")

    normalized = BoundaryCondition.from_config(config)
    if normalized.type != "RESISTANCE":
        raise ValueError(f"{name} must be a RESISTANCE boundary condition")
    return normalized


def _cycle_duration_from_handler(config_handler: ConfigHandler) -> float | None:
    period = getattr(config_handler.simparams, "cardiac_period", None)
    if period is not None:
        try:
            period_float = float(period)
        except (TypeError, ValueError):
            period_float = None
        if period_float is not None and math.isfinite(period_float) and period_float > 0.0:
            return period_float

    inflow = config_handler.bcs.get("INFLOW")
    times = getattr(inflow, "t", None)
    if times is None:
        return None
    times_array = np.asarray(times, dtype=float)
    if times_array.size < 2:
        return None
    duration = float(times_array.max() - times_array.min())
    if not math.isfinite(duration) or duration <= 0.0:
        return None
    return duration


def _slice_last_cycle(result, cycle_duration: float | None):
    if cycle_duration is None or "time" not in result:
        return result
    time = np.asarray(result["time"], dtype=float)
    if time.size < 2:
        return result
    cutoff = float(time.max()) - float(cycle_duration)
    sliced = result[time >= cutoff]
    return sliced if not sliced.empty else result


def _vessel_result(result, vessel):
    by_name = result[result.name == vessel.name]
    if not by_name.empty:
        return by_name
    branch_name = f"branch{vessel.branch}_seg0"
    by_branch = result[result.name == branch_name]
    if not by_branch.empty:
        return by_branch
    raise ValueError(f"simulation result missing vessel '{vessel.name}'")


def _series(frame, preferred: str, fallback: str) -> np.ndarray:
    column = preferred if preferred in frame else fallback
    if column not in frame:
        raise ValueError(f"simulation result missing '{preferred}'/'{fallback}' columns")
    return np.asarray(frame[column], dtype=float)


def _integral_or_mean(values: np.ndarray, time: np.ndarray | None) -> float:
    if time is not None and time.size == values.size and values.size >= 2:
        return float(trapezoid(values, time))
    return float(np.mean(values))


def _extract_pa_metrics(
    result,
    *,
    mpa_vessel,
    rpa_vessel,
    cycle_duration: float | None,
    mpa_flow_column: str,
    rpa_flow_column: str,
) -> dict[str, Any]:
    mpa_result = _slice_last_cycle(_vessel_result(result, mpa_vessel), cycle_duration)
    rpa_result = _slice_last_cycle(_vessel_result(result, rpa_vessel), cycle_duration)
    if mpa_result.empty or rpa_result.empty:
        raise ValueError("simulation result has no usable PA data")

    pressure = np.asarray(mpa_result["pressure_in"], dtype=float) / MMHG_TO_BARYE
    mpa_flow = _series(mpa_result, mpa_flow_column, "flow_out")
    rpa_flow = _series(rpa_result, rpa_flow_column, "flow_out")

    mpa_time = np.asarray(mpa_result["time"], dtype=float) if "time" in mpa_result else None
    rpa_time = np.asarray(rpa_result["time"], dtype=float) if "time" in rpa_result else None
    total_flow = _integral_or_mean(mpa_flow, mpa_time)
    rpa_total = _integral_or_mean(rpa_flow, rpa_time)
    if not math.isfinite(total_flow) or total_flow == 0.0:
        raise ValueError("MPA flow is zero or non-finite; cannot compute RPA split")

    return {
        "P_mpa": [
            float(np.max(pressure)),
            float(np.min(pressure)),
            float(np.mean(pressure)),
        ],
        "rpa_split": float(rpa_total / total_flow),
        "mean_mpa_flow": float(np.mean(mpa_flow)),
        "mean_rpa_flow": float(np.mean(rpa_flow)),
    }


def _reduced_metrics(pa_config: PAConfig) -> dict[str, Any]:
    return _extract_pa_metrics(
        pa_config.result,
        mpa_vessel=pa_config.mpa,
        rpa_vessel=pa_config.rpa_prox,
        cycle_duration=_cycle_duration_from_inflow(pa_config.inflow),
        mpa_flow_column="flow_out",
        rpa_flow_column="flow_in",
    )


def _cycle_duration_from_inflow(inflow: BoundaryCondition) -> float | None:
    times = getattr(inflow, "t", None)
    if times is None:
        return None
    times_array = np.asarray(times, dtype=float)
    if times_array.size < 2:
        return None
    duration = float(times_array.max() - times_array.min())
    if not math.isfinite(duration) or duration <= 0.0:
        return None
    return duration


def _is_steady_inflow(inflow: BoundaryCondition) -> bool:
    flow = np.asarray(getattr(inflow, "Q", []), dtype=float)
    return bool(flow.size > 0 and np.allclose(flow, flow[0]))


def _loss(
    pa_config: PAConfig,
    reference_metrics: dict[str, Any],
    resistances: np.ndarray,
) -> float:
    if not np.all(np.isfinite(resistances)) or np.any(resistances < 0.0):
        return 1.0e12

    pa_config.lpa_prox.R = float(resistances[0])
    pa_config.rpa_prox.R = float(resistances[1])

    try:
        pa_config.simulate()
        metrics = _reduced_metrics(pa_config)
    except Exception:
        return 1.0e12

    p_ref = np.asarray(reference_metrics["P_mpa"], dtype=float)
    p_fit = np.asarray(metrics["P_mpa"], dtype=float)
    p_scale = np.maximum(np.abs(p_ref), 1.0)
    pressure_components = ((p_fit - p_ref) / p_scale) ** 2
    pressure_weights = np.asarray([1.5, 1.0, 1.2], dtype=float)
    pressure_loss = float(np.dot(pressure_weights, pressure_components) * 100.0)

    split_ref = float(reference_metrics["rpa_split"])
    split_scale = max(abs(split_ref), 1.0e-6)
    split_loss = float(((float(metrics["rpa_split"]) - split_ref) / split_scale) ** 2 * 100.0)
    return pressure_loss + split_loss


def _initial_resistances(pa_config: PAConfig) -> np.ndarray:
    initial = np.asarray([pa_config.lpa_prox.R, pa_config.rpa_prox.R], dtype=float)
    if not np.all(np.isfinite(initial)) or np.any(initial <= 0.0):
        initial = np.asarray([1.0, 1.0], dtype=float)
    return initial


def prepare_reduced_rri_seed_from_learned(
    *,
    learned_config: str | Path,
    output_config: str | Path | None = None,
    reduced_template: str | Path | None = None,
    metrics_path: str | Path | None = None,
    lpa_bc: float | dict[str, Any] | BoundaryCondition | None = None,
    rpa_bc: float | dict[str, Any] | BoundaryCondition | None = None,
    maxiter: int = 200,
    solver: str = "Nelder-Mead",
) -> dict[str, Any]:
    """Fit a reduced RRI PAConfig seed to a learned full 0D reference.

    The learned config is simulated first and used as the target source for MPA
    pressure and RPA flow split. The reduced RRI model uses fixed LPA/RPA
    resistance boundary conditions: default ``R=1000.0, Pd=0.0`` per side, or
    caller-provided resistance BCs.
    """

    learned_path = Path(learned_config).expanduser()
    if not learned_path.exists():
        raise FileNotFoundError(f"learned 0D config not found: {learned_path}")

    learned_handler = ConfigHandler.from_json(str(learned_path), is_pulmonary=True)
    learned_result = learned_handler.simulate()
    reference_metrics = _extract_pa_metrics(
        learned_result,
        mpa_vessel=learned_handler.mpa,
        rpa_vessel=learned_handler.rpa,
        cycle_duration=_cycle_duration_from_handler(learned_handler),
        mpa_flow_column="flow_in",
        rpa_flow_column="flow_in",
    )

    clinical_targets = ClinicalTargets(
        mpa_p=reference_metrics["P_mpa"],
        q=reference_metrics["mean_mpa_flow"],
        rpa_split=reference_metrics["rpa_split"],
        wedge_p=0.0,
        steady=_is_steady_inflow(learned_handler.bcs["INFLOW"]),
    )
    pa_config = PAConfig.from_config_handler(
        learned_handler,
        clinical_targets,
        steady=clinical_targets.steady,
    )
    pa_config.bcs = {
        "INFLOW": pa_config.inflow,
        "RPA_BC": _normalize_resistance_bc(rpa_bc, name="RPA_BC"),
        "LPA_BC": _normalize_resistance_bc(lpa_bc, name="LPA_BC"),
    }

    x0 = _initial_resistances(pa_config)
    result = minimize(
        fun=lambda x: _loss(pa_config, reference_metrics, np.asarray(x, dtype=float)),
        x0=x0,
        method=solver,
        bounds=Bounds([0.0, 0.0], [np.inf, np.inf]),
        options={"maxiter": int(maxiter)},
    )

    final_x = np.asarray(result.x if result.x is not None else x0, dtype=float)
    pa_config.lpa_prox.R = float(final_x[0])
    pa_config.rpa_prox.R = float(final_x[1])
    pa_config.simulate()
    optimized_metrics = _reduced_metrics(pa_config)

    optimized_config = pa_config.config
    output_path = Path(output_config).expanduser() if output_config is not None else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pa_config.to_json(output_path)

    fit_residuals = {
        "P_mpa": (
            np.asarray(optimized_metrics["P_mpa"], dtype=float)
            - np.asarray(reference_metrics["P_mpa"], dtype=float)
        ).tolist(),
        "rpa_split": float(optimized_metrics["rpa_split"] - reference_metrics["rpa_split"]),
    }
    metrics: dict[str, Any] = {
        "method": "rri_from_learned_reference",
        "learned_config": str(learned_path),
        "source_template": str(Path(reduced_template).expanduser()) if reduced_template else None,
        "output_config": str(output_path) if output_path is not None else None,
        "reference_metrics": reference_metrics,
        "optimized_metrics": optimized_metrics,
        "fit_residuals": fit_residuals,
        "optimizer": {
            "success": bool(result.success),
            "message": str(result.message),
            "fun": float(result.fun) if result.fun is not None else None,
            "nit": int(result.nit) if hasattr(result, "nit") else None,
            "nfev": int(result.nfev) if hasattr(result, "nfev") else None,
            "x": final_x.tolist(),
            "solver": solver,
            "maxiter": int(maxiter),
        },
        "boundary_conditions": {
            "LPA_BC": pa_config.bcs["LPA_BC"].to_dict(),
            "RPA_BC": pa_config.bcs["RPA_BC"].to_dict(),
        },
        "optimized_config": optimized_config,
    }
    metrics = _json_safe(metrics)

    if metrics_path is not None:
        metrics_out = Path(metrics_path).expanduser()
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        metrics_out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    return metrics
