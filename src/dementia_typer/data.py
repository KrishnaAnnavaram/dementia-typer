"""Load the OASIS-3 tables, check their schema and join each clinical visit to an MR session.

Join rule: for each clinical visit, take the FreeSurfer session of the same participant with the
smallest distance in days, if the distance is not more than the window (default 365 days). A visit
with no session in the window keeps empty MRI values. Only sessions with a passed QC status are used.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .labels import map_series

CLINICAL_REQUIRED = ["OASISID", "days_to_visit", "age at visit", "dx1"]
CLINICAL_OPTIONAL = ["MMSE", "memory", "orient", "judgment", "commun", "homehobb", "perscare", "CDRSUM", "CDRTOT"]
MRI_REQUIRED = ["OASISID", "days_to_scan", "IntraCranialVol"]
MRI_VOLUMES = [
    "TotalGrayVol", "CortexVol", "CorticalWhiteMatterVol", "SubCortGrayVol", "Left-Hippocampus",
    "Right-Hippocampus", "Left-Lateral-Ventricle", "Right-Lateral-Ventricle", "CSF",
]
DEMOGRAPHICS_REQUIRED = ["OASISID", "sex", "education_years"]
QC_PASSED = {"passed", "pass", "ok"}


class SchemaError(ValueError):
    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


def _require(frame: pd.DataFrame, columns: list[str], name: str) -> list[str]:
    return [f"{name}: missing column {c!r}" for c in columns if c not in frame.columns]


def day_from_label(label: str) -> float:
    """Read the day number from an OASIS label such as `OAS30001_MR_d0129`."""
    match = re.search(r"_d(\d+)$", str(label))
    return float(match.group(1)) if match else np.nan


def validate_clinical(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "days_to_visit" not in out.columns and "OASIS_session_label" in out.columns:
        out["days_to_visit"] = out["OASIS_session_label"].map(day_from_label)
    problems = _require(out, CLINICAL_REQUIRED, "clinical")
    if problems:
        raise SchemaError(problems)
    if out[["OASISID", "days_to_visit"]].duplicated().any():
        problems.append("clinical: (OASISID, days_to_visit) must be unique")
    if "MMSE" in out.columns:
        mmse = pd.to_numeric(out["MMSE"], errors="coerce")
        if ((mmse < 0) | (mmse > 30)).any():
            problems.append("clinical: MMSE must be between 0 and 30")
    if problems:
        raise SchemaError(problems)
    return out


def validate_mri(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "days_to_scan" not in out.columns and "MR_session" in out.columns:
        out["days_to_scan"] = out["MR_session"].map(day_from_label)
    problems = _require(out, MRI_REQUIRED, "mri")
    if not problems:
        icv = pd.to_numeric(out["IntraCranialVol"], errors="coerce")
        if (icv <= 0).any():
            problems.append("mri: IntraCranialVol must be positive")
    if problems:
        raise SchemaError(problems)
    if "FS QC Status" in out.columns:
        out = out[out["FS QC Status"].astype(str).str.strip().str.lower().isin(QC_PASSED)]
    return out


def validate_demographics(frame: pd.DataFrame) -> pd.DataFrame:
    problems = _require(frame, DEMOGRAPHICS_REQUIRED, "demographics")
    if not problems and frame["OASISID"].duplicated().any():
        problems.append("demographics: OASISID must be unique")
    if problems:
        raise SchemaError(problems)
    return frame


def join_visits(clinical: pd.DataFrame, mri: pd.DataFrame | None, demographics: pd.DataFrame | None,
                window_days: int = 365) -> pd.DataFrame:
    out = clinical.sort_values(["days_to_visit"]).copy()
    out["days_to_visit"] = out["days_to_visit"].astype(float)
    if mri is not None and len(mri):
        cols = ["OASISID", "days_to_scan"] + [c for c in ["IntraCranialVol", *MRI_VOLUMES] if c in mri.columns]
        scans = mri[cols].copy()
        scans["days_to_scan"] = scans["days_to_scan"].astype(float)
        scans = scans.sort_values("days_to_scan")
        out = pd.merge_asof(out, scans, left_on="days_to_visit", right_on="days_to_scan", by="OASISID",
                            direction="nearest", tolerance=float(window_days))
        out["scan_gap_days"] = (out["days_to_scan"] - out["days_to_visit"]).abs()
    if demographics is not None:
        out = out.merge(demographics, on="OASISID", how="left", validate="many_to_one")
    out["group"] = map_series(out["dx1"])
    return out.sort_values(["OASISID", "days_to_visit"]).reset_index(drop=True)


def load_tables(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    clinical_path = data_dir / "clinical.csv"
    if not clinical_path.exists():
        raise FileNotFoundError(f"No clinical.csv in {data_dir}. Read data/README.md, or run `dementia-typer synth`.")
    tables = {"clinical": validate_clinical(pd.read_csv(clinical_path))}
    if (data_dir / "freesurfer.csv").exists():
        tables["mri"] = validate_mri(pd.read_csv(data_dir / "freesurfer.csv"))
    if (data_dir / "demographics.csv").exists():
        tables["demographics"] = validate_demographics(pd.read_csv(data_dir / "demographics.csv"))
    return tables


def build_table(data_dir: str | Path, window_days: int = 365) -> pd.DataFrame:
    tables = load_tables(data_dir)
    return join_visits(tables["clinical"], tables.get("mri"), tables.get("demographics"), window_days)
