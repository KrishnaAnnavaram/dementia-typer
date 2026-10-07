"""Subject-level splits, pipelines and nested cross-validation.

- Every split groups the visits by participant (`OASISID`).
- The imputer, scaler and feature selector are steps of one pipeline. They are fit inside each fold.
- The inner grouped CV selects the model and its parameters by macro-F1. The held-out test
  participants are used once, after the selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODELS = ("majority", "logreg", "forest", "boosting")
SELECTORS = ("none", "mi", "rfe")


class LeakageError(AssertionError):
    """A participant has visits on both sides of a split."""


def subject_labels(groups: pd.Series, y: np.ndarray) -> pd.Series:
    """The label of the last visit of each participant, used to stratify the participant split."""
    frame = pd.DataFrame({"g": np.asarray(groups), "y": y})
    return frame.groupby("g", sort=False)["y"].last()


def holdout_split(y: np.ndarray, groups: pd.Series, test_fraction: float = 0.2, seed: int = 42):
    """Hold out whole participants, stratified by the label of their last visit."""
    n_splits = max(2, int(round(1 / test_fraction)))
    per_subject = subject_labels(groups, y)
    counts = per_subject.value_counts()
    strata = per_subject.where(per_subject.map(counts) >= n_splits, -1)  # rare labels share one stratum
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, test_subjects = next(splitter.split(np.zeros(len(strata)), strata.to_numpy()))
    test_ids = set(per_subject.index[test_subjects])
    is_test = np.asarray(groups.isin(test_ids))
    train_idx, test_idx = np.flatnonzero(~is_test), np.flatnonzero(is_test)
    assert_disjoint(groups, train_idx, test_idx)
    return train_idx, test_idx


def assert_disjoint(groups, train_idx, test_idx) -> None:
    g = np.asarray(groups)
    shared = set(g[train_idx]) & set(g[test_idx])
    if shared:
        raise LeakageError(f"{len(shared)} participants are in train and test")


def build_pipeline(model: str, selector: str = "none", k: int = 8, seed: int = 42) -> Pipeline:
    steps = [("impute", SimpleImputer(strategy="median", add_indicator=False)), ("scale", StandardScaler())]
    if selector == "mi":
        steps.append(("select", SelectKBest(partial(mutual_info_classif, random_state=seed), k=k)))
    elif selector == "rfe":
        steps.append(("select", RFE(LogisticRegression(max_iter=2000), n_features_to_select=k)))
    elif selector != "none":
        raise ValueError(f"selector must be one of {', '.join(SELECTORS)}")
    if model == "majority":
        est = DummyClassifier(strategy="most_frequent")
    elif model == "logreg":
        est = LogisticRegression(max_iter=3000, class_weight="balanced")
    elif model == "forest":
        est = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", min_samples_leaf=3,
                                     random_state=seed, n_jobs=1)
    elif model == "boosting":
        est = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, class_weight="balanced",
                                             random_state=seed)
    else:
        raise ValueError(f"model must be one of {', '.join(MODELS)}")
    steps.append(("model", est))
    return Pipeline(steps)


GRIDS = {
    "majority": {},
    "logreg": {"model__C": [0.1, 1.0, 10.0]},
    "forest": {"model__max_depth": [None, 8], "model__min_samples_leaf": [1, 5]},
    "boosting": {"model__learning_rate": [0.05, 0.1], "model__max_leaf_nodes": [15, 31]},
}


@dataclass
class Candidate:
    model: str
    selector: str
    k: int


def candidates(models=("logreg", "forest"), selectors=("none", "mi"), k: int = 8) -> list[Candidate]:
    return [Candidate(m, s, k) for m in models for s in selectors]


def inner_search(X: pd.DataFrame, y: np.ndarray, groups: pd.Series, cands: list[Candidate], n_splits: int,
                 seed: int) -> tuple[Pipeline, dict, list[dict]]:
    """Grid search each candidate with grouped inner CV. Return the refit best pipeline and a log."""
    log = []
    best_score, best_est, best_info = -np.inf, None, {}
    for cand in cands:
        k = min(cand.k, X.shape[1])
        grid = GridSearchCV(build_pipeline(cand.model, cand.selector, k, seed), GRIDS[cand.model] or {},
                            scoring="f1_macro", cv=GroupKFold(n_splits=n_splits), refit=True, n_jobs=1)
        grid.fit(X, y, groups=np.asarray(groups))
        info = {"model": cand.model, "selector": cand.selector, "k": k, "inner_macro_f1": float(grid.best_score_),
                "params": {key: (v if not isinstance(v, np.generic) else v.item()) for key, v in grid.best_params_.items()}}
        log.append(info)
        if grid.best_score_ > best_score:
            best_score, best_est, best_info = grid.best_score_, grid.best_estimator_, info
    return best_est, best_info, log


def nested_cv(X: pd.DataFrame, y: np.ndarray, groups: pd.Series, cands: list[Candidate], outer_splits: int = 5,
              inner_splits: int = 3, seed: int = 42, scorer=None) -> list[dict]:
    """Outer grouped CV around `inner_search`. It estimates the error of the full selection procedure."""
    outer = StratifiedGroupKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    results = []
    for fold, (tr, te) in enumerate(outer.split(X, y, np.asarray(groups))):
        assert_disjoint(groups, tr, te)
        est, info, _ = inner_search(X.iloc[tr], y[tr], groups.iloc[tr], cands, inner_splits, seed)
        proba = est.predict_proba(X.iloc[te])
        results.append({"fold": fold, "chosen": info, "score": scorer(y[te], proba, est) if scorer else None})
    return results
