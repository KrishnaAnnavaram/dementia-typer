"""One source of truth for the diagnosis groups.

`CLASSES` fixes the class order. The model stores the class names, and predictions return names,
never bare integers. `map_diagnosis` changes an OASIS-3 `dx1` text into a group with ordered,
word-boundary rules. A text that no rule matches gives None and is reported, not guessed.
"""
from __future__ import annotations

import re

import pandas as pd

CLASSES: tuple[str, ...] = ("Normal", "MCI", "Alzheimer's", "Vascular", "Other dementia")
CLASS_INDEX = {name: i for i, name in enumerate(CLASSES)}

# (group, pattern). The first match wins. The order matters and the tests cover it.
RULES: tuple[tuple[str, str], ...] = (
    # Impairment without a confirmed dementia: uncertain, incipient, questionable, very mild.
    ("MCI", r"\buncertain|\bincipient|0\.5 in memory|\bques\w*\.? ?impair|\bquestionable|impair\w* reversible"
            r"|\bmci\b|mild cognitive|w/o dement"),
    ("Normal", r"cognitively normal|\bno dementia\b|^normal$"),
    ("Other dementia", r"\bnon[- ]?ad\b"),
    ("Vascular", r"\bvascular\b|\bvad\b"),
    ("Alzheimer's", r"\bad\b|\bdat\b|alzheimer"),
    ("Other dementia", r"dlbd|lewy|frontotemporal|\bftd\b|\bpdd\b|parkinson|dementia/pd|huntington|prion"
                       r"|alcohol|other primary|\bdem\w*\b"),
)
_COMPILED = tuple((group, re.compile(pattern)) for group, pattern in RULES)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def map_diagnosis(text) -> str | None:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    norm = normalize(text)
    if not norm:
        return None
    for group, pattern in _COMPILED:
        if pattern.search(norm):
            return group
    return None


def map_series(texts: pd.Series) -> pd.Series:
    return texts.map(map_diagnosis)


def unmapped(texts: pd.Series) -> list[str]:
    """Return the distinct texts that no rule maps, so a person can extend the rules."""
    values = texts.dropna().astype(str)
    return sorted({v for v in values.unique() if map_diagnosis(v) is None})


def encode(groups: pd.Series):
    return groups.map(CLASS_INDEX).astype(int).to_numpy()


def decode(indices) -> list[str]:
    return [CLASSES[int(i)] for i in indices]
