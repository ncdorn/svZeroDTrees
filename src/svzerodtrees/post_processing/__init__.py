"""Post-processing interfaces."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "compute_pulmonary_resistance_map",
    "render_resistance_map_png",
    "run_pulmonary_threed_postprocess_suite",
    "write_flow_split_comparison_artifacts",
    "write_frames_csv_for_simulation",
    "write_mpa_pressure_timeseries_csv",
]

_LAZY_EXPORTS = {
    "compute_pulmonary_resistance_map": (
        "svzerodtrees.post_processing.resistance_map",
        "compute_pulmonary_resistance_map",
    ),
    "render_resistance_map_png": (
        "svzerodtrees.post_processing.pulmonary_threed_suite",
        "render_resistance_map_png",
    ),
    "run_pulmonary_threed_postprocess_suite": (
        "svzerodtrees.post_processing.pulmonary_threed_suite",
        "run_pulmonary_threed_postprocess_suite",
    ),
    "write_flow_split_comparison_artifacts": (
        "svzerodtrees.post_processing.pulmonary_threed_suite",
        "write_flow_split_comparison_artifacts",
    ),
    "write_frames_csv_for_simulation": (
        "svzerodtrees.post_processing.pulmonary_threed_suite",
        "write_frames_csv_for_simulation",
    ),
    "write_mpa_pressure_timeseries_csv": (
        "svzerodtrees.post_processing.pulmonary_threed_suite",
        "write_mpa_pressure_timeseries_csv",
    ),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
