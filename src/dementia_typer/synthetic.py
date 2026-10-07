"""Synthetic OASIS-3-like tables for the offline demo and the tests.

The generator writes `clinical.csv`, `freesurfer.csv` and `demographics.csv` with the column names
that dementia-typer reads. Participants have several visits, some move from Normal to MCI to
Alzheimer's, MRI volumes shrink with age and disease, and each participant has a fixed brain size.
CDR scores follow the group almost exactly, which shows the CDR leakage. The values are random.
They are not OASIS data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DX_TEXT = {
    "Normal": ["Cognitively normal", "No dementia"],
    "MCI": ["0.5 in memory only", "uncertain dementia", "Incipient demt PTP", "Unc: ques. Impairment"],
    "Alzheimer's": ["AD Dementia", "AD dem w/depresss- contribut", "DAT"],
    "Vascular": ["Vascular Demt- primary", "Vascular Demt- secondary"],
    "Other dementia": ["Non AD dem- Other primary", "DLBD- primary", "Frontotemporal demt. prim"],
}
START = {"Normal": 0.62, "MCI": 0.14, "Alzheimer's": 0.15, "Vascular": 0.04, "Other dementia": 0.05}
SEVERITY = {"Normal": 0.0, "MCI": 0.5, "Alzheimer's": 1.0, "Vascular": 0.8, "Other dementia": 0.9}


def _cdr(sev: float, rng) -> dict:
    boxes = {}
    for name in ("memory", "orient", "judgment", "commun", "homehobb", "perscare"):
        raw = sev * (1.4 if name == "memory" else 1.0) + rng.normal(0, 0.15)
        boxes[name] = float(min(3.0, max(0.0, np.round(raw * 2) / 2)))
    boxes["CDRSUM"] = float(sum(boxes.values()))
    boxes["CDRTOT"] = float(0.0 if sev == 0 else 0.5 if sev <= 0.5 else 1.0 if sev < 1 else 2.0)
    return boxes


def generate(n_subjects: int = 300, seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    clinical, mri, demo = [], [], []
    groups = list(START)
    for s in range(n_subjects):
        sid = f"OAS3{s:04d}"
        sex = "M" if rng.random() < 0.43 else "F"
        demo.append({"OASISID": sid, "sex": sex, "education_years": int(rng.integers(8, 21))})
        icv = rng.normal(1.55e6 if sex == "M" else 1.38e6, 0.09e6)
        brain = rng.normal(0, 1)  # fixed participant effect on all volumes
        group = str(rng.choice(groups, p=[START[g] for g in groups]))
        age0 = float(rng.uniform(55, 85))
        day = 0
        for visit in range(int(rng.integers(1, 6))):
            if visit:
                day += int(rng.integers(300, 900))
                if group == "Normal" and rng.random() < 0.12:
                    group = "MCI"
                elif group == "MCI" and rng.random() < 0.3:
                    group = "Alzheimer's"
            age = age0 + day / 365.25
            sev = SEVERITY[group]
            atrophy = 0.004 * (age - 70) + 0.06 * sev + 0.015 * brain * -1
            hippo_drop = 0.002 * (age - 70) + (0.12 if group == "Alzheimer's" else 0.05 if group == "MCI" else 0.02 * sev)
            mmse = float(np.clip(np.round(29.3 - 9 * sev - 0.05 * (age - 70) + rng.normal(0, 1.4)), 0, 30))
            clinical.append({
                "OASISID": sid, "OASIS_session_label": f"{sid}_ClinicalData_d{day:04d}", "days_to_visit": day,
                "age at visit": round(age, 1), "MMSE": mmse if rng.random() > 0.05 else np.nan,
                "dx1": str(rng.choice(DX_TEXT[group])), **_cdr(sev, rng),
            })
            if rng.random() < 0.8:
                scan_day = max(0, day + int(rng.integers(-200, 200)))
                wm_extra = 0.08 if group == "Vascular" else 0.0
                vol = {
                    "TotalGrayVol": icv * (0.46 - atrophy * 0.5) * rng.normal(1, 0.01),
                    "CortexVol": icv * (0.37 - atrophy * 0.45) * rng.normal(1, 0.01),
                    "CorticalWhiteMatterVol": icv * (0.33 - atrophy * 0.2 - wm_extra * 0.3) * rng.normal(1, 0.015),
                    "SubCortGrayVol": icv * (0.038 - atrophy * 0.01) * rng.normal(1, 0.02),
                    "Left-Hippocampus": icv * 0.0027 * (1 - hippo_drop) * rng.normal(1, 0.03),
                    "Right-Hippocampus": icv * 0.0028 * (1 - hippo_drop) * rng.normal(1, 0.03),
                    "Left-Lateral-Ventricle": icv * (0.009 + atrophy * 0.04 + wm_extra * 0.02) * rng.normal(1, 0.08),
                    "Right-Lateral-Ventricle": icv * (0.0085 + atrophy * 0.04 + wm_extra * 0.02) * rng.normal(1, 0.08),
                    "CSF": icv * (0.0009 + atrophy * 0.002) * rng.normal(1, 0.1),
                }
                mri.append({"OASISID": sid, "MR_session": f"{sid}_MR_d{scan_day:04d}", "days_to_scan": scan_day,
                            "IntraCranialVol": icv, **{k: round(v, 1) for k, v in vol.items()},
                            "FS QC Status": "Passed" if rng.random() > 0.03 else "Failed"})
    clin = pd.DataFrame(clinical)
    mri_frame = pd.DataFrame(mri).drop_duplicates(["OASISID", "days_to_scan"])
    return {"clinical": clin, "freesurfer": mri_frame, "demographics": pd.DataFrame(demo)}


def write(out_dir: str | Path, n_subjects: int = 300, seed: int = 0) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, frame in generate(n_subjects, seed).items():
        paths[name] = out / f"{name}.csv"
        frame.to_csv(paths[name], index=False)
    return paths
