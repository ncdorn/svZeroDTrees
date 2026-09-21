"""Deterministic mapping between pulmonary mesh caps and outlet BCs.

The mapping used to be reconstructed from ``ConfigHandler.bcs`` each time a
tree was built.  That is convenient for interactive callers, but dictionary
iteration is not a sufficient provenance contract for a full 0D model.  This
module keeps the resolution rules in one place and exposes an immutable,
serializable result for callers that need to freeze the pairing.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any


MAPPING_VERSION = 1
MAPPING_MODES = (
    "auto",
    "metadata",
    "cap_name",
    "serialized_cap_order",
    "explicit",
)


def mapping_key(value: Any) -> str:
    """Return the comparison key used for cap and BC names.

    Paths and VTP suffixes are deliberately ignored.  The key is only used
    for matching; the original path/name is always retained in provenance.
    """

    stem = Path(str(value)).stem
    return re.sub(r"[^a-z0-9]+", "", stem.lower())


def cap_side(value: Any) -> str:
    """Classify a cap from its physical cap identity.

    A cap must identify exactly one pulmonary side.  In particular, this does
    not inspect 0D vessel graph labels or branch membership.
    """

    value_lower = Path(str(value)).stem.lower()
    has_lpa = "lpa" in value_lower
    has_rpa = "rpa" in value_lower
    if has_lpa == has_rpa:
        raise ValueError(
            f"cap '{value}' must identify exactly one pulmonary side using LPA or RPA"
        )
    return "lpa" if has_lpa else "rpa"


def _finite_float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _optional_float(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, label=label)


@dataclass(frozen=True)
class ResolvedCapOutlet:
    """One ordered, immutable cap-to-BC pairing.

    ``cap_index`` and ``bc_index`` are indices in the filtered canonical cap
    and outlet orders.  The serialized indices retain their source-list
    positions when available, which makes an order-based mapping auditable.
    """

    cap_path: str
    cap_stem: str
    side: str
    bc_name: str
    cap_index: int
    bc_index: int
    area: float
    raw_diameter: float
    scaled_diameter: float | None = None
    serialized_cap_index: int | None = None
    serialized_bc_index: int | None = None
    graph_side: str | None = None
    graph_side_disagreement: bool = False

    def __post_init__(self) -> None:
        if self.cap_index < 0 or self.bc_index < 0:
            raise ValueError("cap_index and bc_index must be non-negative")
        if self.serialized_cap_index is not None and self.serialized_cap_index < 0:
            raise ValueError("serialized_cap_index must be non-negative")
        if self.serialized_bc_index is not None and self.serialized_bc_index < 0:
            raise ValueError("serialized_bc_index must be non-negative")
        normalized_side = str(self.side).strip().lower()
        if normalized_side not in {"lpa", "rpa"}:
            raise ValueError(f"side must be 'lpa' or 'rpa', got {self.side!r}")
        object.__setattr__(self, "side", normalized_side)
        object.__setattr__(self, "cap_path", str(self.cap_path))
        object.__setattr__(self, "cap_stem", str(self.cap_stem))
        object.__setattr__(self, "bc_name", str(self.bc_name))
        object.__setattr__(self, "area", _finite_float(self.area, label="cap area"))
        if self.area <= 0.0:
            raise ValueError("cap area must be > 0")
        object.__setattr__(
            self,
            "raw_diameter",
            _finite_float(self.raw_diameter, label="raw cap diameter"),
        )
        if self.raw_diameter <= 0.0:
            raise ValueError("raw cap diameter must be > 0")
        object.__setattr__(
            self,
            "scaled_diameter",
            _optional_float(self.scaled_diameter, label="scaled cap diameter"),
        )
        if self.graph_side is not None:
            graph = str(self.graph_side).lower()
            if graph not in {"lpa", "rpa"}:
                raise ValueError("graph_side must be 'lpa', 'rpa', or None")
            object.__setattr__(self, "graph_side", graph)

    @property
    def cap_name(self) -> str:
        """Backward-compatible spelling for the cap path."""

        return self.cap_path

    @property
    def cap_side(self) -> str:
        return self.side

    @property
    def outlet_name(self) -> str:
        return self.bc_name

    @property
    def diameter(self) -> float:
        """The construction diameter, falling back to the raw geometry value."""

        return self.raw_diameter if self.scaled_diameter is None else self.scaled_diameter

    def with_scaled_diameter(self, scaled_diameter: float) -> "ResolvedCapOutlet":
        return replace(
            self,
            scaled_diameter=_finite_float(
                scaled_diameter, label="scaled cap diameter"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cap_path": self.cap_path,
            "cap_stem": self.cap_stem,
            "side": self.side,
            "bc_name": self.bc_name,
            "cap_index": self.cap_index,
            "bc_index": self.bc_index,
            "serialized_cap_index": self.serialized_cap_index,
            "serialized_bc_index": self.serialized_bc_index,
            "area": self.area,
            "raw_diameter": self.raw_diameter,
            "scaled_diameter": self.scaled_diameter,
            "graph_side": self.graph_side,
            "graph_side_disagreement": self.graph_side_disagreement,
        }


# The longer name is useful to callers that want to make the record type
# explicit, while the shorter name is easier to use in implementation code.
ResolvedOutletCapRecord = ResolvedCapOutlet


@dataclass(frozen=True)
class ResolvedOutletCapMapping(Mapping[str, str]):
    """Frozen, ordered, bijective cap-to-outlet mapping.

    The object implements the read-only Mapping interface so existing direct
    callers that index the legacy mapping continue to work.  ``records`` and
    ``to_dict`` expose the richer contract needed for replay and provenance.
    """

    strategy: str
    records: tuple[ResolvedCapOutlet, ...]
    cap_order: tuple[str, ...] = field(init=False)
    bc_order: tuple[str, ...] = field(init=False)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    version: int = MAPPING_VERSION

    def __post_init__(self) -> None:
        strategy = str(self.strategy).strip().lower()
        if strategy not in MAPPING_MODES[1:]:
            raise ValueError(f"invalid resolved mapping strategy: {self.strategy!r}")
        records = tuple(self.records)
        if not records:
            raise ValueError("resolved outlet-cap mapping cannot be empty")
        cap_order = tuple(record.cap_path for record in records)
        bc_order = tuple(record.bc_name for record in records)
        if len(set(mapping_key(cap) for cap in cap_order)) != len(records):
            raise ValueError("resolved mapping contains duplicate cap entries")
        if len(set(bc_order)) != len(records):
            raise ValueError("resolved mapping contains duplicate outlet BC entries")
        if tuple(record.cap_index for record in records) != tuple(range(len(records))):
            raise ValueError("resolved mapping cap indices must be complete and ordered")
        if sorted(record.bc_index for record in records) != list(range(len(records))):
            raise ValueError("resolved mapping BC indices must form a bijection")
        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "cap_order", cap_order)
        object.__setattr__(self, "bc_order", bc_order)
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def __getitem__(self, cap: str) -> str:
        key = mapping_key(cap)
        matches = [record.bc_name for record in self.records if mapping_key(record.cap_path) == key]
        if len(matches) != 1:
            raise KeyError(cap)
        return matches[0]

    def __iter__(self):
        return iter(self.cap_order)

    def __len__(self) -> int:
        return len(self.records)

    def items(self):
        return tuple((record.cap_path, record.bc_name) for record in self.records)

    def keys(self):
        return self.cap_order

    def values(self):
        return self.bc_order

    @property
    def pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple((record.cap_path, record.bc_name) for record in self.records)

    @property
    def ordered_pairs(self) -> tuple[tuple[str, str], ...]:
        return self.pairs

    @property
    def cap_records(self) -> tuple[ResolvedCapOutlet, ...]:
        return self.records

    @property
    def cap_paths(self) -> tuple[str, ...]:
        return self.cap_order

    @property
    def outlet_names(self) -> tuple[str, ...]:
        return self.bc_order

    @property
    def mapping(self) -> Mapping[str, str]:
        return MappingProxyType(OrderedDict(self.items()))

    def record_for_cap(self, cap: str) -> ResolvedCapOutlet:
        key = mapping_key(cap)
        matches = [record for record in self.records if mapping_key(record.cap_path) == key]
        if len(matches) != 1:
            raise KeyError(cap)
        return matches[0]

    def record_for_bc(self, bc_name: str) -> ResolvedCapOutlet:
        matches = [record for record in self.records if record.bc_name == str(bc_name)]
        if len(matches) != 1:
            raise KeyError(bc_name)
        return matches[0]

    def with_scaled_diameters(
        self, scaled_diameters: Mapping[str, float]
    ) -> "ResolvedOutletCapMapping":
        records = tuple(
            record.with_scaled_diameter(scaled_diameters[record.cap_path])
            if record.cap_path in scaled_diameters
            else record
            for record in self.records
        )
        return replace(self, records=records)

    def to_dict(self) -> dict[str, Any]:
        # Lists, rather than dictionaries, are intentional: order is part of
        # the mapping contract and must survive JSON round trips.
        return {
            "version": self.version,
            "strategy": self.strategy,
            "cap_order": list(self.cap_order),
            "bc_order": list(self.bc_order),
            "pairs": [record.to_dict() for record in self.records],
            "provenance": _json_safe(self.provenance),
        }

    as_dict = to_dict
    to_payload = to_dict

    def serialize(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def to_json(self, path: str | Path | None = None, *, indent: int | None = 2) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, indent=indent)
        if path is not None:
            Path(path).write_text(payload + "\n", encoding="utf-8")
        return payload


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(entry) for entry in value]
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("mapping provenance cannot contain non-finite values")
        return value
    return str(value)


def _serialized_boundary_conditions(config_handler: Any) -> list[tuple[int, str, str | None]]:
    """Return ``(serialized_index, name, type)`` in original JSON order."""

    raw_config = getattr(config_handler, "_config", None)
    if isinstance(raw_config, Mapping):
        entries = raw_config.get("boundary_conditions")
        if isinstance(entries, Sequence) and not isinstance(entries, (str, bytes)):
            result = []
            for index, entry in enumerate(entries):
                if isinstance(entry, Mapping) and entry.get("bc_name") is not None:
                    result.append(
                        (index, str(entry["bc_name"]), entry.get("bc_type"))
                    )
            if result:
                return result

    bcs = getattr(config_handler, "bcs", {})
    if isinstance(bcs, Mapping):
        return [
            (index, str(name), getattr(bc, "type", None))
            for index, (name, bc) in enumerate(bcs.items())
        ]
    raise ValueError("config handler does not expose serialized boundary_conditions")


def serialized_outlet_names(config_handler: Any) -> tuple[tuple[str, int], ...]:
    """Return non-inflow BC names and their original serialized indices."""

    result = []
    for serialized_index, name, bc_type in _serialized_boundary_conditions(config_handler):
        if str(bc_type or "").upper() == "FLOW" or "inflow" in name.lower():
            continue
        result.append((name, serialized_index))
    return tuple(result)


def _bc_objects(config_handler: Any) -> Mapping[str, Any]:
    bcs = getattr(config_handler, "bcs", None)
    if not isinstance(bcs, Mapping):
        return {}
    return bcs


def _graph_side(config_handler: Any, bc_name: str) -> str | None:
    """Best-effort diagnostic graph side; never used for physical mapping."""

    for vessel in getattr(config_handler, "vessel_map", {}).values():
        bc = getattr(vessel, "bc", None)
        if isinstance(bc, Mapping) and bc.get("outlet") == bc_name:
            label = str(getattr(vessel, "label", "") or "").lower()
            if "lpa" in label:
                return "lpa"
            if "rpa" in label:
                return "rpa"
    return None


def _normalise_cap_info(cap_info: Mapping[str, Any]) -> tuple[tuple[str, float, str], ...]:
    if not isinstance(cap_info, Mapping) or not cap_info:
        raise ValueError("cap_info must be a non-empty mapping of cap path to area")
    result = []
    seen = set()
    for index, (cap_path, area) in enumerate(cap_info.items()):
        path = str(cap_path)
        key = mapping_key(path)
        if not key:
            raise ValueError(f"cap '{path}' has an empty normalized name")
        if key in seen:
            raise ValueError(f"duplicate cap names are ambiguous: '{path}'")
        seen.add(key)
        side = cap_side(path)
        area_value = _finite_float(area, label=f"area for cap '{path}'")
        if area_value <= 0.0:
            raise ValueError(f"area for cap '{path}' must be > 0")
        result.append((path, area_value, side))
    return tuple(result)


def _extract_metadata_pairs(config_handler: Any) -> list[tuple[Any, Any]]:
    payloads = []
    tree_params = getattr(config_handler, "tree_params", {})
    if isinstance(tree_params, Mapping) and tree_params:
        # ConfigHandler mirrors the original ``trees`` list into tree_params;
        # reading both would duplicate every metadata pair and make an
        # otherwise valid mapping appear ambiguous.
        payloads.extend(tree_params.values())
    else:
        raw_config = getattr(config_handler, "_config", None)
        if isinstance(raw_config, Mapping) and isinstance(raw_config.get("trees"), Sequence):
            payloads.extend(raw_config["trees"])

    pairs: list[tuple[Any, Any]] = []
    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        mapping_payload = payload.get("outlet_mapping")
        if not isinstance(mapping_payload, Mapping):
            continue
        direct = mapping_payload.get("cap_to_bc") or mapping_payload.get("mapping")
        if isinstance(direct, Mapping):
            pairs.extend(direct.items())
            continue
        raw_pairs = mapping_payload.get("pairs")
        if isinstance(raw_pairs, Sequence) and not isinstance(raw_pairs, (str, bytes)):
            for pair in raw_pairs:
                if isinstance(pair, Mapping):
                    cap = pair.get("cap_path", pair.get("cap_name", pair.get("outlet_name")))
                    bc = pair.get("bc_name")
                    if cap is not None and bc is not None:
                        pairs.append((cap, bc))
                elif isinstance(pair, Sequence) and len(pair) == 2:
                    pairs.append((pair[0], pair[1]))
            continue
        outlet_names = mapping_payload.get("outlet_names") or []
        bc_names = mapping_payload.get("bc_names") or []
        if isinstance(outlet_names, Sequence) and isinstance(bc_names, Sequence):
            if len(outlet_names) == len(bc_names):
                pairs.extend(zip(outlet_names, bc_names))
    return pairs


def _candidate_from_pairs(
    pairs: Sequence[tuple[Any, Any]],
    caps: tuple[tuple[str, float, str], ...],
    outlets: tuple[tuple[str, int], ...],
    *,
    strategy: str,
    config_handler: Any,
    provenance: Mapping[str, Any] | None = None,
) -> ResolvedOutletCapMapping:
    cap_by_key = {mapping_key(path): (path, area, side) for path, area, side in caps}
    outlet_by_name = {name: (index, serialized_index) for index, (name, serialized_index) in enumerate(outlets)}
    if len(pairs) != len(caps):
        raise ValueError(
            f"{strategy} mapping must contain exactly one entry per cap; "
            f"got {len(pairs)} for {len(caps)} caps"
        )

    resolved: list[tuple[str, str, float, str, int, int]] = []
    used_caps = set()
    used_bcs = set()
    for raw_cap, raw_bc in pairs:
        cap_key = mapping_key(raw_cap)
        if cap_key not in cap_by_key:
            raise ValueError(f"{strategy} mapping references unknown cap '{raw_cap}'")
        cap_path, area, side = cap_by_key[cap_key]
        if cap_key in used_caps:
            raise ValueError(f"{strategy} mapping contains duplicate cap '{raw_cap}'")
        bc_name = str(raw_bc)
        if bc_name not in outlet_by_name:
            raise ValueError(
                f"{strategy} mapping references unknown non-inflow outlet BC '{bc_name}'"
            )
        if bc_name in used_bcs:
            raise ValueError(f"{strategy} mapping contains duplicate outlet BC '{bc_name}'")
        used_caps.add(cap_key)
        used_bcs.add(bc_name)
        bc_index, serialized_index = outlet_by_name[bc_name]
        resolved.append((cap_path, bc_name, area, side, bc_index, serialized_index))

    if used_caps != set(cap_by_key):
        missing = sorted(set(cap_by_key) - used_caps)
        raise ValueError(f"{strategy} mapping is incomplete; missing caps: {', '.join(missing)}")
    if used_bcs != set(outlet_by_name):
        missing = sorted(set(outlet_by_name) - used_bcs)
        raise ValueError(
            f"{strategy} mapping is incomplete; missing non-inflow outlet BCs: {', '.join(missing)}"
        )

    resolved_by_cap = {mapping_key(item[0]): item for item in resolved}
    records = []
    # Always emit canonical cap order, even when metadata or an explicit map
    # was supplied in a different order.  This is what makes serialization
    # replayable and independent of dictionary insertion order.
    for cap_index, (canonical_cap_path, area, side) in enumerate(caps):
        cap_path, bc_name, _, _, bc_index, serialized_index = resolved_by_cap[
            mapping_key(canonical_cap_path)
        ]
        graph_side = _graph_side(config_handler, bc_name)
        records.append(
            ResolvedCapOutlet(
                cap_path=cap_path,
                cap_stem=Path(cap_path).stem,
                side=side,
                bc_name=bc_name,
                cap_index=cap_index,
                bc_index=bc_index,
                serialized_cap_index=cap_index,
                serialized_bc_index=serialized_index,
                area=area,
                raw_diameter=2.0 * math.sqrt(area / math.pi),
                graph_side=graph_side,
                graph_side_disagreement=(graph_side is not None and graph_side != side),
            )
        )
    return ResolvedOutletCapMapping(
        strategy=strategy,
        records=tuple(records),
        provenance={"strategy_attempt": strategy, **dict(provenance or {})},
    )


def _name_pairs(caps: tuple[tuple[str, float, str], ...], outlets, config_handler):
    bc_by_key: dict[str, list[str]] = {}
    bcs = _bc_objects(config_handler)
    for name, _ in outlets:
        keys = {mapping_key(name), mapping_key(getattr(bcs.get(name), "name", name))}
        for key in keys:
            if key:
                bc_by_key.setdefault(key, []).append(name)
    pairs = []
    for path, _, _ in caps:
        matches = bc_by_key.get(mapping_key(path), [])
        if len(matches) != 1:
            if not matches:
                raise ValueError(f"cap_name mapping has no BC match for cap '{path}'")
            raise ValueError(
                f"cap_name mapping is ambiguous for cap '{path}': {', '.join(matches)}"
            )
        pairs.append((path, matches[0]))
    return pairs


def _explicit_pairs(explicit_mapping: Any) -> list[tuple[Any, Any]]:
    if isinstance(explicit_mapping, Mapping):
        for key in ("cap_to_bc", "mapping", "pairs"):
            if key in explicit_mapping and key != "pairs":
                return _explicit_pairs(explicit_mapping[key])
        if "pairs" in explicit_mapping:
            return _explicit_pairs(explicit_mapping["pairs"])
        return list(explicit_mapping.items())
    if isinstance(explicit_mapping, Sequence) and not isinstance(explicit_mapping, (str, bytes)):
        pairs = []
        for item in explicit_mapping:
            if isinstance(item, Mapping):
                cap = item.get("cap_path", item.get("cap_name", item.get("cap_stem")))
                bc = item.get("bc_name")
                if cap is None or bc is None:
                    raise ValueError("explicit mapping pair requires cap and bc_name")
                pairs.append((cap, bc))
            elif isinstance(item, Sequence) and len(item) == 2:
                pairs.append((item[0], item[1]))
            else:
                raise ValueError("explicit mapping pairs must be [cap, bc_name]")
        return pairs
    raise ValueError("explicit outlet_mapping must be a mapping or sequence of pairs")


def resolve_outlet_cap_mapping(
    config_handler: Any,
    cap_info: Mapping[str, Any],
    *,
    bc_prefix: str = "IMPEDANCE",
    mode: str = "auto",
    outlet_mapping_mode: str | None = None,
    mapping_mode: str | None = None,
    strategy: str | None = None,
    explicit_mapping: Any = None,
    explicit_map: Any = None,
    outlet_mapping: Any = None,
    allow_ordered_outlet_mapping: bool = False,
    allow_serialized_fallback: bool = False,
) -> ResolvedOutletCapMapping:
    """Resolve and validate one complete cap-to-outlet mapping.

    ``bc_prefix`` remains accepted for compatibility with the old resolver;
    non-inflow BCs are intentionally ordered from the serialized seed rather
    than from ``ConfigHandler.bcs``.
    """

    selected = outlet_mapping_mode
    if selected is None:
        selected = mapping_mode
    if selected is None:
        selected = strategy
    if selected is None:
        selected = mode
    selected = str(selected or "auto").strip().lower()
    if selected not in MAPPING_MODES:
        raise ValueError(
            "outlet mapping mode must be one of " + "|".join(MAPPING_MODES)
        )
    if allow_ordered_outlet_mapping:
        if selected not in {"auto", "serialized_cap_order"}:
            raise ValueError("allow_ordered_outlet_mapping conflicts with mode")
        if selected == "auto":
            selected = "serialized_cap_order"
    supplied_mapping_inputs = [
        name
        for name, value in (
            ("explicit_mapping", explicit_mapping),
            ("explicit_map", explicit_map),
            ("outlet_mapping", outlet_mapping),
        )
        if value is not None
    ]
    if len(supplied_mapping_inputs) > 1:
        raise ValueError(
            "conflicting outlet mapping inputs: "
            + ", ".join(supplied_mapping_inputs)
            + "; provide only one"
        )
    if explicit_mapping is None:
        explicit_mapping = outlet_mapping
    if explicit_mapping is None:
        explicit_mapping = explicit_map

    if explicit_mapping is not None and selected != "explicit":
        raise ValueError(
            "outlet_mapping requires outlet_mapping_mode='explicit'; "
            "select explicit mode before supplying a mapping"
        )

    caps = _normalise_cap_info(cap_info)
    outlets = serialized_outlet_names(config_handler)
    if len(outlets) != len(caps):
        raise ValueError(
            "number of outlet boundary conditions does not match number of cap surfaces: "
            f"bcs={len(outlets)}, caps={len(caps)}"
        )
    if selected == "explicit" and explicit_mapping is None:
        raise ValueError("outlet_mapping_mode='explicit' requires outlet_mapping")

    if selected != "auto":
        strategies = [selected]
    elif allow_serialized_fallback:
        strategies = ["metadata", "cap_name", "serialized_cap_order"]
    else:
        # Direct tree-construction callers historically required an explicit
        # ordered-mapping opt-in.  Keep that behavior while full_pa callers
        # can use the resolver's documented auto strategy.
        strategies = ["metadata", "cap_name"]
    errors = []
    for strategy in strategies:
        try:
            if strategy == "metadata":
                pairs = _extract_metadata_pairs(config_handler)
                if not pairs:
                    raise ValueError("no complete outlet mapping metadata was found")
            elif strategy == "cap_name":
                pairs = _name_pairs(caps, outlets, config_handler)
            elif strategy == "serialized_cap_order":
                pairs = [(cap_path, outlets[index][0]) for index, (cap_path, _, _) in enumerate(caps)]
            elif strategy == "explicit":
                pairs = _explicit_pairs(explicit_mapping)
            else:  # pragma: no cover - guarded by mode validation
                raise ValueError(f"unsupported mapping strategy: {strategy}")
            return _candidate_from_pairs(
                pairs,
                caps,
                outlets,
                strategy=strategy,
                config_handler=config_handler,
                provenance={"bc_prefix": bc_prefix, "requested_mode": selected},
            )
        except ValueError as exc:
            errors.append(f"{strategy}: {exc}")

    detail = "; ".join(errors)
    if selected == "auto" and not allow_serialized_fallback:
        detail += (
            "; auto mapping does not use serialized_cap_order; select "
            "outlet_mapping_mode='serialized_cap_order' explicitly when order "
            "is the intended contract"
        )
    raise ValueError("could not deterministically resolve outlet-cap mapping: " + detail)


def coerce_resolved_mapping(
    mapping: ResolvedOutletCapMapping | Mapping[str, str],
    config_handler: Any,
    cap_info: Mapping[str, Any],
    *,
    bc_prefix: str = "IMPEDANCE",
) -> ResolvedOutletCapMapping:
    """Validate a supplied mapping or turn a legacy mapping dict into one."""

    if isinstance(mapping, ResolvedOutletCapMapping):
        expected_caps = tuple(str(key) for key in cap_info)
        if {mapping_key(key) for key in expected_caps} != {
            mapping_key(key) for key in mapping.cap_order
        }:
            raise ValueError("supplied resolved mapping does not cover the current cap set")
        expected_outlets = {name for name, _ in serialized_outlet_names(config_handler)}
        if set(mapping.bc_order) != expected_outlets:
            raise ValueError("supplied resolved mapping does not cover current outlet BCs")
        return mapping
    if not isinstance(mapping, Mapping):
        raise TypeError("resolved mapping must implement Mapping")
    return resolve_outlet_cap_mapping(
        config_handler,
        cap_info,
        bc_prefix=bc_prefix,
        mode="explicit",
        explicit_mapping=mapping,
    )


__all__ = [
    "MAPPING_MODES",
    "MAPPING_VERSION",
    "ResolvedCapOutlet",
    "ResolvedOutletCapRecord",
    "ResolvedOutletCapMapping",
    "cap_side",
    "coerce_resolved_mapping",
    "mapping_key",
    "resolve_outlet_cap_mapping",
    "serialized_outlet_names",
]
