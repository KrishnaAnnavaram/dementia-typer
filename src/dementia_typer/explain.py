"""SHAP values on held-out visits (optional extra: `pip install -e ".[explain]"`).

The core package uses permutation importance on the test participants (see `train.run_experiment`).
"""
from __future__ import annotations

import pandas as pd


def shap_values(pipeline, X: pd.DataFrame, background: pd.DataFrame | None = None):
    try:
        import shap  # noqa: PLC0415 - optional dependency
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError('SHAP needs the optional extra: pip install -e ".[explain]"') from exc
    background = background if background is not None else X
    explainer = shap.Explainer(pipeline.predict_proba, background)
    return explainer(X)
