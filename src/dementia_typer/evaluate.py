"""Metrics for an imbalanced 5-class problem, with participant-level bootstrap intervals."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .labels import CLASSES


def full_proba(proba: np.ndarray, model_classes) -> np.ndarray:
    """Put the probability columns in `CLASSES` order. A class that the model never saw gets 0."""
    out = np.zeros((proba.shape[0], len(CLASSES)))
    for j, c in enumerate(model_classes):
        out[:, int(c)] = proba[:, j]
    return out


def multiclass_brier(y: np.ndarray, proba: np.ndarray) -> float:
    onehot = np.eye(len(CLASSES))[y]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def top_label_ece(y: np.ndarray, proba: np.ndarray, bins: int = 10) -> float:
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())
    return float(ece)


def metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    labels = list(range(len(CLASSES)))
    present = sorted(int(c) for c in np.unique(y))
    pred = proba.argmax(axis=1)
    recall = recall_score(y, pred, labels=labels, average=None, zero_division=0)
    out = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        # Macro means use the classes that occur in y. An absent class has no defined recall.
        "macro_f1": float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
        "absent_classes": [CLASSES[c] for c in labels if c not in present],
        "recall": {CLASSES[c]: (float(recall[c]) if c in present else float("nan")) for c in labels},
        "precision": dict(zip(CLASSES, map(float, precision_score(y, pred, labels=labels, average=None,
                                                                   zero_division=0)))),
        "support": dict(zip(CLASSES, map(int, np.bincount(y, minlength=len(CLASSES))))),
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "brier": multiclass_brier(y, proba),
        "ece": top_label_ece(y, proba),
    }
    both = [c for c in labels if 0 < (y == c).sum() < len(y)]
    aucs = {CLASSES[c]: float(roc_auc_score((y == c).astype(int), proba[:, c])) for c in both}
    out["auroc_ovr"] = aucs
    out["macro_auroc_ovr"] = float(np.mean(list(aucs.values()))) if aucs else float("nan")
    return out


def subject_bootstrap(y: np.ndarray, proba: np.ndarray, groups, key: str = "macro_f1", n_boot: int = 200,
                      seed: int = 42) -> tuple[float, float]:
    g = np.asarray(groups)
    uniq, inverse = np.unique(g, return_inverse=True)
    members = [np.flatnonzero(inverse == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))])
        pred = proba[idx].argmax(axis=1)
        if key == "macro_f1":
            values.append(f1_score(y[idx], pred, labels=np.unique(y[idx]), average="macro", zero_division=0))
        elif key == "balanced_accuracy":
            values.append(balanced_accuracy_score(y[idx], pred))
        else:
            raise ValueError(f"unknown metric {key!r}")
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)
