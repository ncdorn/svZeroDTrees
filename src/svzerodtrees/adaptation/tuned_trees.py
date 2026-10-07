"""The tuned structured-tree model that adaptation starts from.

Preop impedance tuning exports, in the tuned config's top-level ``trees``
section, one structured tree per outlet BC (one per cap for the
physiological full-PA model; one per side for shared-tree models).  Each
entry carries its rebuild parameters, its node budget (``max_nodes``) and its
outlet mapping (``side``, ``bc_names``, ``outlet_names``).  The IMPEDANCE BCs
carry the outlet pressure ``Pd`` the trees were tuned with, and full-PA
tuning also writes the cap-to-BC pairing to ``outlet_cap_mapping.json``.

This module loads and checks that model and writes adapted trees back by BC
name.  Caps and BCs are never paired by list position: learned seeds name
their BCs in centerline order, which need not match the cap file order.
"""

from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from ..microvasculature.structured_tree.structuredtree import DEFAULT_MAX_NODES, StructuredTree
from ..tune_bcs.clinical_targets import ClinicalTargets
from ..tune_bcs.outlet_mapping import cap_side, mapping_key

# Tuning converts Pd [mmHg] to the IMPEDANCE BC value with this factor.
MMHG_TO_DYN_PER_CM2 = 1333.2
_PD_RTOL = 1e-6
_DIAMETER_RTOL = 1e-6
_REBUILD_KEYS = ("name", "initial_d", "d_min", "lrr", "compliance")


@dataclass(frozen=True)
class TunedTree:
    """One tuned structured tree and the outlet BCs (and caps) it serves."""

    name: str
    side: str
    bc_names: tuple[str, ...]
    caps: tuple[str, ...]
    metadata: Mapping[str, Any]

    @property
    def max_nodes(self) -> int | None:
        value = self.metadata.get("max_nodes")
        return None if value is None else int(value)


@dataclass(frozen=True)
class TunedTreeModel:
    trees: tuple[TunedTree, ...]
    outlet_pressure_dyn: float
    source: str
    outlet_cap_mapping: str | None = None

    @property
    def outlet_pressure_mmhg(self) -> float:
        return self.outlet_pressure_dyn / MMHG_TO_DYN_PER_CM2

    @property
    def bc_names(self) -> tuple[str, ...]:
        return tuple(bc for tree in self.trees for bc in tree.bc_names)

    @property
    def per_outlet(self) -> bool:
        """True when some side has more than one tree (per-cap trees)."""
        return any(len(self.trees_for_side(side)) > 1 for side in ("lpa", "rpa"))

    def trees_for_side(self, side: str) -> tuple[TunedTree, ...]:
        return tuple(tree for tree in self.trees if tree.side == side)

    def tree_for_bc(self, bc_name: str) -> TunedTree:
        for tree in self.trees:
            if bc_name in tree.bc_names:
                return tree
        raise KeyError(bc_name)

    def cap_for_bc(self, bc_name: str) -> str:
        tree = self.tree_for_bc(bc_name)
        return tree.caps[tree.bc_names.index(bc_name)]

    def provenance(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "outlet_cap_mapping": self.outlet_cap_mapping,
            "outlet_pressure_mmhg": self.outlet_pressure_mmhg,
            "n_trees": len(self.trees),
            "per_outlet": self.per_outlet,
            "trees": [
                {
                    "name": tree.name,
                    "side": tree.side,
                    "bc_names": list(tree.bc_names),
                    "caps": [Path(cap).name for cap in tree.caps],
                    "initial_d": float(tree.metadata["initial_d"]),
                    "max_nodes": tree.max_nodes,
                }
                for tree in self.trees
            ],
        }


def _config_parts(source) -> tuple[list, dict[str, tuple[str, Mapping[str, Any]]], str]:
    """Return (tree entries, {bc_name: (type, values)}, label) from a config."""
    if isinstance(source, (str, os.PathLike)):
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
        trees = list(payload.get("trees") or [])
        bcs = {
            str(bc.get("bc_name")): (
                str(bc.get("bc_type", "")).upper(),
                dict(bc.get("bc_values") or {}),
            )
            for bc in payload.get("boundary_conditions") or []
        }
        return trees, bcs, str(source)
    trees = list((getattr(source, "tree_params", None) or {}).values())
    bcs = {
        str(name): (str(getattr(bc, "type", "")).upper(), dict(getattr(bc, "values", None) or {}))
        for name, bc in (getattr(source, "bcs", None) or {}).items()
    }
    return trees, bcs, str(getattr(source, "path", None) or "<config>")


def _parse_trees(entries, impedance_bcs: set[str], label: str) -> tuple[TunedTree, ...]:
    problems: list[str] = []
    trees: list[TunedTree] = []
    seen: dict[str, str] = {}
    for entry in entries:
        name = str(entry.get("name")) if isinstance(entry, Mapping) else "<invalid>"
        if not isinstance(entry, Mapping):
            problems.append("a tree entry is not a mapping")
            continue
        missing = [key for key in _REBUILD_KEYS if key not in entry]
        if missing:
            problems.append(f"tree {name!r} lacks rebuild fields {missing}")
        mapping = entry.get("outlet_mapping")
        if not isinstance(mapping, Mapping):
            problems.append(f"tree {name!r} has no outlet_mapping")
            continue
        bc_names = [str(bc) for bc in mapping.get("bc_names") or []]
        caps = [str(cap) for cap in mapping.get("outlet_names") or []]
        side = str(mapping.get("side") or "").strip().lower()
        if not bc_names or len(caps) != len(bc_names):
            problems.append(
                f"tree {name!r} outlet_mapping needs one outlet_names entry per bc_names entry"
            )
            continue
        if side not in {"lpa", "rpa"}:
            problems.append(f"tree {name!r} outlet_mapping side must be lpa or rpa, got {side!r}")
            continue
        for bc, cap in zip(bc_names, caps):
            if bc in seen:
                problems.append(f"BC {bc!r} is mapped by trees {seen[bc]!r} and {name!r}")
            seen[bc] = name
            if bc not in impedance_bcs:
                problems.append(f"tree {name!r} maps {bc!r}, which is not an IMPEDANCE BC")
            try:
                if cap_side(cap) != side:
                    problems.append(f"tree {name!r} ({side}) maps cap {Path(cap).name!r} of the other side")
            except ValueError as exc:
                problems.append(str(exc))
        trees.append(
            TunedTree(
                name=name,
                side=side,
                bc_names=tuple(bc_names),
                caps=tuple(caps),
                metadata=copy.deepcopy(dict(entry)),
            )
        )
    uncovered = sorted(impedance_bcs - set(seen))
    if uncovered:
        problems.append(f"IMPEDANCE BCs without tree metadata: {uncovered}")
    if problems:
        raise ValueError(
            f"tuned config {label} cannot be paired to outlet caps without guessing "
            "(adaptation never pairs caps and BCs by position); regenerate it with the "
            "current svZeroDTrees: " + "; ".join(problems)
        )
    return tuple(trees)


def _common_outlet_pressure(bcs, bc_names, label: str) -> float:
    values = {}
    for bc in bc_names:
        try:
            values[bc] = float(bcs[bc][1]["Pd"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"tuned config {label}: IMPEDANCE BC {bc!r} has no numeric Pd") from None
    pd_values = list(values.values())
    if not all(math.isfinite(value) for value in pd_values):
        raise ValueError(f"tuned config {label}: IMPEDANCE BC Pd values must be finite")
    if max(pd_values) - min(pd_values) > _PD_RTOL * max(1.0, abs(pd_values[0])):
        raise ValueError(
            f"tuned config {label}: IMPEDANCE BCs disagree on Pd "
            f"({min(pd_values):.6g} to {max(pd_values):.6g} dyn/cm^2); "
            "adaptation needs the single tuned outlet pressure"
        )
    return pd_values[0]


def _check_outlet_cap_mapping(trees: tuple[TunedTree, ...], mapping_payload, label: str) -> None:
    pairs = mapping_payload.get("pairs") if isinstance(mapping_payload, Mapping) else None
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"outlet cap mapping {label} has no pairs list")
    records = {}
    for record in pairs:
        if not isinstance(record, Mapping) or "bc_name" not in record or "cap_path" not in record:
            raise ValueError(f"outlet cap mapping {label} has a pair without bc_name/cap_path")
        records[str(record["bc_name"])] = record
    tree_bcs = {bc for tree in trees for bc in tree.bc_names}
    if set(records) != tree_bcs:
        raise ValueError(
            f"outlet cap mapping {label} covers BCs {sorted(records)} but the tuned trees "
            f"cover {sorted(tree_bcs)}"
        )
    problems = []
    for tree in trees:
        for bc, cap in zip(tree.bc_names, tree.caps):
            record = records[bc]
            if mapping_key(record["cap_path"]) != mapping_key(cap):
                problems.append(
                    f"{bc}: mapping cap {Path(str(record['cap_path'])).name!r}, "
                    f"tree cap {Path(cap).name!r}"
                )
            if record.get("side") is not None and str(record["side"]).lower() != tree.side:
                problems.append(f"{bc}: mapping side {record['side']!r}, tree side {tree.side!r}")
            scaled = record.get("scaled_diameter")
            if scaled is not None and not math.isclose(
                float(scaled), float(tree.metadata["initial_d"]), rel_tol=_DIAMETER_RTOL
            ):
                problems.append(
                    f"{bc}: mapping diameter {float(scaled):.6g}, tree initial_d "
                    f"{float(tree.metadata['initial_d']):.6g}"
                )
    if problems:
        raise ValueError(
            f"outlet cap mapping {label} disagrees with the tuned tree metadata: "
            + "; ".join(problems)
        )


def load_tuned_tree_model(source, *, outlet_cap_mapping=None) -> TunedTreeModel:
    """Load the tuned trees, their outlet BCs/caps, and Pd from a tuned config.

    ``source`` is a config path (e.g. ``svzerod_3d_coupling_tuned.json``) or a
    loaded ``ConfigHandler`` (e.g. the preop 3D coupler).  When
    ``outlet_cap_mapping`` (path or payload of ``outlet_cap_mapping.json``) is
    given it must cover the same BCs with the same caps, sides and diameters.
    Raises ``ValueError`` when any IMPEDANCE BC cannot be tied to a tree and a
    cap from saved metadata.
    """
    entries, bcs, label = _config_parts(source)
    impedance_bcs = {name for name, (bc_type, _) in bcs.items() if bc_type == "IMPEDANCE"}
    if not impedance_bcs:
        raise ValueError(
            f"tuned config {label} has no IMPEDANCE boundary conditions; adaptation "
            "starts from the impedance-tuned structured trees"
        )
    if not entries:
        raise ValueError(
            f"tuned config {label} has no 'trees' metadata; regenerate it with the "
            "current svZeroDTrees (adaptation rebuilds the tuned trees from it)"
        )
    trees = _parse_trees(entries, impedance_bcs, label)
    outlet_pressure = _common_outlet_pressure(bcs, [bc for tree in trees for bc in tree.bc_names], label)
    mapping_label = None
    if outlet_cap_mapping is not None:
        if isinstance(outlet_cap_mapping, (str, os.PathLike)):
            mapping_label = str(outlet_cap_mapping)
            payload = json.loads(Path(outlet_cap_mapping).read_text(encoding="utf-8"))
        else:
            mapping_label = "<payload>"
            payload = outlet_cap_mapping
        _check_outlet_cap_mapping(trees, payload, mapping_label)
    return TunedTreeModel(
        trees=trees,
        outlet_pressure_dyn=outlet_pressure,
        source=label,
        outlet_cap_mapping=mapping_label,
    )


def adaptation_clinical_targets(
    clinical_targets_csv: str,
    *,
    tuned_outlet_pressure_dyn: float,
    wedge_pressure_policy: str | None = None,
    precapillary_fraction: float | None = None,
    diastolic_offset_mmhg: float | None = None,
) -> tuple[ClinicalTargets, dict[str, Any]]:
    """Clinical targets whose ``wedge_p`` is the tuned outlet pressure [mmHg].

    The tuned trees were fitted with the IMPEDANCE BC ``Pd``; adaptation uses
    that value everywhere (reduced-PA BCs, steady tree hemodynamics, adapted
    IMPEDANCE BCs).  An optional ``wedge_pressure_policy`` is resolved like
    tuning does and must reproduce it.
    """
    kwargs: dict[str, Any] = {}
    if wedge_pressure_policy is not None:
        kwargs["wedge_pressure_policy"] = str(wedge_pressure_policy)
        if precapillary_fraction is not None:
            kwargs["precapillary_fraction"] = float(precapillary_fraction)
        if diastolic_offset_mmhg is not None:
            kwargs["diastolic_offset_mmhg"] = float(diastolic_offset_mmhg)
    targets = ClinicalTargets.from_csv(clinical_targets_csv, **kwargs)
    tuned_pd_mmhg = float(tuned_outlet_pressure_dyn) / MMHG_TO_DYN_PER_CM2
    provenance: dict[str, Any] = {
        "pd_mmhg": tuned_pd_mmhg,
        "source": "tuned_config_impedance_bcs",
        "wedge_pressure_policy": wedge_pressure_policy,
    }
    if wedge_pressure_policy is not None:
        policy_pd = float(targets.wedge_p)
        provenance["policy_pd_mmhg"] = policy_pd
        if not math.isclose(policy_pd, tuned_pd_mmhg, rel_tol=_PD_RTOL, abs_tol=1e-9):
            raise ValueError(
                f"wedge_pressure_policy {wedge_pressure_policy!r} gives Pd {policy_pd:.4f} mmHg "
                f"but the tuned trees use Pd {tuned_pd_mmhg:.4f} mmHg; adaptation must use the "
                "outlet pressure the trees were tuned with"
            )
    targets.wedge_p = tuned_pd_mmhg
    return targets, provenance


def resolve_tuned_tree_budget(tree: TunedTree, override: int | None) -> tuple[dict, str]:
    """Return tree metadata with the node budget to rebuild with, and its source.

    The tuned budget in the metadata wins; an ``override`` (adaptation
    ``parameter_set.max_nodes``) only fills it in for metadata written before
    ``max_nodes`` was recorded, and must agree with it otherwise.
    """
    metadata = copy.deepcopy(dict(tree.metadata))
    if tree.max_nodes is not None:
        if override is not None and int(override) != tree.max_nodes:
            raise ValueError(
                f"tuned tree {tree.name!r} was built with max_nodes={tree.max_nodes}; "
                f"adaptation parameter_set max_nodes={int(override)} would rebuild a "
                "different tree (remove it or set it to the tuned budget)"
            )
        return metadata, "tree_metadata"
    metadata["max_nodes"] = int(override) if override is not None else DEFAULT_MAX_NODES
    return metadata, "parameter_set" if override is not None else "default"


def check_coupler_matches_tuned_model(coupler, model: TunedTreeModel) -> None:
    """Fail unless ``coupler`` has exactly the tuned outlet BCs on the mapped caps."""
    outlet_bcs = {
        str(name)
        for name, bc in (getattr(coupler, "bcs", None) or {}).items()
        if "inflow" not in str(getattr(bc, "name", name)).lower()
    }
    tuned_bcs = set(model.bc_names)
    if outlet_bcs != tuned_bcs:
        raise ValueError(
            "coupler outlet BCs do not match the tuned trees: missing "
            f"{sorted(tuned_bcs - outlet_bcs)}, unexpected {sorted(outlet_bcs - tuned_bcs)}"
        )
    blocks = getattr(coupler, "coupling_blocks", None) or {}
    mismatches = []
    for bc in model.bc_names:
        surface = getattr(blocks.get(bc), "surface", None)
        if not surface:
            mismatches.append(f"{bc}: no coupling block with a surface")
        elif mapping_key(surface) != mapping_key(model.cap_for_bc(bc)):
            mismatches.append(
                f"{bc}: coupled to {Path(str(surface)).name!r}, tuned for "
                f"{Path(model.cap_for_bc(bc)).name!r}"
            )
    if mismatches:
        raise ValueError(
            "the 3D coupler couples outlet BCs to different caps than the tuned outlet "
            "mapping, so adapted trees would land on the wrong caps: " + "; ".join(mismatches)
        )


def write_adapted_tuned_trees(
    coupler,
    model: TunedTreeModel,
    adapt_tree: Callable[[TunedTree, StructuredTree], Mapping[str, Any]],
    *,
    time,
    kernel_steps: int,
    max_nodes_override: int | None = None,
    n_procs: int = 24,
) -> dict[str, dict[str, Any]]:
    """Rebuild each tuned tree, adapt it, and write its BCs into ``coupler``.

    Trees are processed one at a time (per-cap trees can each hold ~1M
    vessels).  ``adapt_tree(tuned_tree, tree)`` mutates the rebuilt tree and
    returns its adaptation metrics.  Every BC the tree serves gets the adapted
    impedance at the tuned Pd; ``coupler.tree_params`` is replaced by the
    adapted metadata, which keeps the tuned outlet mapping.  Returns
    per-tree metrics keyed by tree name.
    """
    check_coupler_matches_tuned_model(coupler, model)
    tree_params = {}
    metrics: dict[str, dict[str, Any]] = {}
    for tuned in model.trees:
        metadata, budget_source = resolve_tuned_tree_budget(tuned, max_nodes_override)
        tree = StructuredTree.from_tree_metadata(metadata, time=time, simparams=None)
        tree_metrics = dict(adapt_tree(tuned, tree) or {})
        tree.compute_olufsen_impedance(n_procs=n_procs, tsteps=kernel_steps)
        for index, bc_name in enumerate(tuned.bc_names):
            coupler.bcs[bc_name] = tree.create_impedance_bc(
                bc_name, index, model.outlet_pressure_dyn, verbose=False
            )
        adapted_metadata = tree.to_dict()
        adapted_metadata["adaptation"] = copy.deepcopy(tree_metrics)
        tree_params[tuned.name] = adapted_metadata
        metrics[tuned.name] = {
            "side": tuned.side,
            "bc_names": list(tuned.bc_names),
            "initial_d": float(metadata["initial_d"]),
            "max_nodes": int(metadata["max_nodes"]),
            "max_nodes_source": budget_source,
            "n_nodes": int(tree.store.n_nodes()),
            "truncated": bool(tree.store.n_nodes() >= int(metadata["max_nodes"])),
            **tree_metrics,
        }
        del tree
    coupler.tree_params = tree_params
    return metrics


__all__ = [
    "MMHG_TO_DYN_PER_CM2",
    "TunedTree",
    "TunedTreeModel",
    "adaptation_clinical_targets",
    "check_coupler_matches_tuned_model",
    "load_tuned_tree_model",
    "resolve_tuned_tree_budget",
    "write_adapted_tuned_trees",
]
