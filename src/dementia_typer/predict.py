"""Schema-validated prediction for one visit.

The input is a JSON object with named fields. The pydantic model rejects unknown fields and values
out of range, and the bundle gives the feature order. A hand-written feature vector is never used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .evaluate import full_proba
from .features import make_features
from .labels import CLASSES


class Visit(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    age: float = Field(ge=18, le=110, alias="age at visit")
    sex: str | None = Field(default=None, pattern="^[MF]$")
    education_years: float | None = Field(default=None, ge=0, le=30)
    IntraCranialVol: float | None = Field(default=None, gt=0)
    TotalGrayVol: float | None = Field(default=None, gt=0)
    CortexVol: float | None = Field(default=None, gt=0)
    CorticalWhiteMatterVol: float | None = Field(default=None, gt=0)
    SubCortGrayVol: float | None = Field(default=None, gt=0)
    left_hippocampus: float | None = Field(default=None, gt=0, alias="Left-Hippocampus")
    right_hippocampus: float | None = Field(default=None, gt=0, alias="Right-Hippocampus")
    left_lateral_ventricle: float | None = Field(default=None, gt=0, alias="Left-Lateral-Ventricle")
    right_lateral_ventricle: float | None = Field(default=None, gt=0, alias="Right-Lateral-Ventricle")
    CSF: float | None = Field(default=None, gt=0)
    MMSE: float | None = Field(default=None, ge=0, le=30)
    memory: float | None = Field(default=None, ge=0, le=3)
    orient: float | None = Field(default=None, ge=0, le=3)
    judgment: float | None = Field(default=None, ge=0, le=3)
    commun: float | None = Field(default=None, ge=0, le=3)
    homehobb: float | None = Field(default=None, ge=0, le=3)
    perscare: float | None = Field(default=None, ge=0, le=3)
    CDRSUM: float | None = Field(default=None, ge=0, le=18)
    CDRTOT: float | None = Field(default=None, ge=0, le=3)


def predict_visit(bundle: dict, payload: dict) -> dict:
    visit = Visit.model_validate(payload)
    row = pd.DataFrame([visit.model_dump(by_alias=True)])
    X = make_features(row)[bundle["features"]]
    missing = [c for c in bundle["features"] if X[c].isna().all()]
    pipe = bundle["pipeline"]
    proba = full_proba(pipe.predict_proba(X), pipe.classes_)[0]
    return {
        "prediction": CLASSES[int(np.argmax(proba))],
        "probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, proba)},
        "feature_set": bundle["feature_set"],
        "imputed_features": missing,
        "note": "Research output only. A clinician must make the diagnosis.",
    }
