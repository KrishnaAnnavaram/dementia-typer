"""Feature sets for the leakage ablation.

- `A`: demographics and MRI volumes (normalized by intracranial volume). No cognitive scores.
  This is the main model for an objective, early signal.
- `B`: A plus MMSE, a short cognitive test.
- `C`: B plus the CDR boxes, CDR-SB and global CDR. Clinicians use CDR to assign the diagnosis,
  so C is a leakage ceiling for reference only. It is never the main result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import MRI_VOLUMES

DEMOGRAPHIC = ["age", "sex_male", "education_years"]
MRI = ["icv_l"] + [f"{v}_icv" for v in MRI_VOLUMES]
COGNITIVE = ["MMSE"]
CDR = ["memory", "orient", "judgment", "commun", "homehobb", "perscare", "CDRSUM", "CDRTOT"]
FEATURE_SETS = {"A": DEMOGRAPHIC + MRI, "B": DEMOGRAPHIC + MRI + COGNITIVE, "C": DEMOGRAPHIC + MRI + COGNITIVE + CDR}
LEAKAGE_FEATURES = frozenset(CDR)
SET_DESCRIPTIONS = {
    "A": "demographics + MRI volumes",
    "B": "A + MMSE",
    "C": "B + CDR (leakage ceiling, reference only)",
}


def make_features(table: pd.DataFrame) -> pd.DataFrame:
    """Return all candidate feature columns. Missing source columns give NaN columns."""
    out = pd.DataFrame(index=table.index)
    out["age"] = pd.to_numeric(table.get("age at visit"), errors="coerce")
    sex = table.get("sex", pd.Series(np.nan, index=table.index)).astype(str).str.upper().str[0]
    out["sex_male"] = np.where(sex == "M", 1.0, np.where(sex == "F", 0.0, np.nan))
    out["education_years"] = pd.to_numeric(table.get("education_years", np.nan), errors="coerce")
    icv = pd.to_numeric(table.get("IntraCranialVol", np.nan), errors="coerce")
    out["icv_l"] = icv / 1e6
    for vol in MRI_VOLUMES:
        values = pd.to_numeric(table.get(vol, np.nan), errors="coerce")
        out[f"{vol}_icv"] = values / icv * 1000.0
    for col in COGNITIVE + CDR:
        out[col] = pd.to_numeric(table.get(col, np.nan), errors="coerce")
    return out


def columns_for(feature_set: str) -> list[str]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature set must be one of {', '.join(FEATURE_SETS)}, got {feature_set!r}")
    return list(FEATURE_SETS[feature_set])
