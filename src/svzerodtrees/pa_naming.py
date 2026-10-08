"""Pulmonary side (LPA/RPA) of a cap, surface, or BC name.

Names usually contain ``lpa``/``rpa`` (``lpa_2.vtp``, ``RPA_1``).  Some meshes
separate the letters (``l_pa_1_x.vtp``, ``r-pa_3``); those are matched only as
a whole ``l_pa`` / ``r_pa`` token so unrelated words are never misread.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

_SEPARATED = {
    "lpa": re.compile(r"(?:^|[^a-z])l[_-]pa(?:[^a-z]|$)"),
    "rpa": re.compile(r"(?:^|[^a-z])r[_-]pa(?:[^a-z]|$)"),
}


def pa_side(name: Any) -> str | None:
    """Return ``"lpa"``, ``"rpa"``, or ``None`` if the name names neither or both."""
    stem = Path(str(name)).stem.lower()
    has_lpa, has_rpa = "lpa" in stem, "rpa" in stem
    if not (has_lpa or has_rpa):
        has_lpa = bool(_SEPARATED["lpa"].search(stem))
        has_rpa = bool(_SEPARATED["rpa"].search(stem))
    if has_lpa == has_rpa:
        return None
    return "lpa" if has_lpa else "rpa"
