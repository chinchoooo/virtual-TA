"""Academic terms. IITM runs three terms a year, identified here by their start month:

    YYYY-01  Jan term  (Jan-Apr)   a.k.a. "T1", "term1-YYYY", "[TDS Jan YYYY]"
    YYYY-05  May term  (May-Aug)   a.k.a. "T2"
    YYYY-09  Sep term  (Sep-Dec)   a.k.a. "T3"

This matches the folder names used by the course repository (``2025-01/`` etc.).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime

TERM_MONTHS = (1, 5, 9)
_MONTH_BY_NAME = {"jan": 1, "may": 5, "sep": 9}
_TERM_ID = re.compile(r"^(20\d\d)-(01|05|09)$")
# Discourse tags such as "term1-2025" / "t3-2024"
_TAG = re.compile(r"^(?:term|t)([123])[-_]?(20\d\d)$", re.IGNORECASE)
# Title markers such as "[TDS Jan 2025]", "TDS May'25", "(Sep 2024 term)"
_TITLE = re.compile(
    r"(?:\btds\W{0,3}|\[[^\]]{0,40}?)\b(jan|may|sep)[a-z]*\W{0,3}(20\d\d|\d\d)\b",
    re.IGNORECASE,
)


def is_term(value: str) -> bool:
    return bool(_TERM_ID.match(value))


def make_term(year: int, month: int) -> str:
    return f"{year}-{month:02d}"


def term_for_date(dt: datetime | str) -> str:
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
    return make_term(dt.year, max(m for m in TERM_MONTHS if m <= dt.month))


def term_from_tags(tags: Iterable[str]) -> str | None:
    for tag in tags:
        if m := _TAG.match(tag):
            return make_term(int(m.group(2)), TERM_MONTHS[int(m.group(1)) - 1])
    return None


def term_from_title(title: str) -> str | None:
    if m := _TITLE.search(title):
        year = int(m.group(2))
        return make_term(year + 2000 if year < 100 else year, _MONTH_BY_NAME[m.group(1).lower()])
    return None


def term_label(term: str) -> str:
    """'2025-01' -> 'Jan 2025'."""
    year, month = term.split("-")
    names = {v: k.title() for k, v in _MONTH_BY_NAME.items()}
    return f"{names[int(month)]} {year}"
