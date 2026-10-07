"""Experiments: participant hold-out, nested selection, one test evaluation, model bundle and card."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from . import __version__
from .evaluate import full_proba, metrics, subject_bootstrap
from .features import LEAKAGE_FEATURES, SET_DESCRIPTIONS, columns_for, make_features
from .labels import CLASSES, encode
from .modeling import build_pipeline, candidates, holdout_split, inner_search, nested_cv

SUMMARY = ("accuracy", "balanced_accuracy", "macro_f1", "macro_auroc_ovr", "brier", "ece")


@dataclass
class Experiment:
    feature_set: str
    features: list[str]
    pipeline: object
    chosen: dict
    search_log: list[dict]
    test: dict
    baseline_test: dict
    ci: dict
    nested: list[dict] = field(default_factory=list)
    importance: pd.DataFrame | None = None
    n_train_subjects: int = 0
    n_test_subjects: int = 0
    n_train_visits: int = 0
    n_test_visits: int = 0


def labelled(table: pd.DataFrame) -> pd.DataFrame:
    return table[table["group"].notna()].reset_index(drop=True)


def _macro_f1_scorer(y, proba, est):
    return float(f1_score(y, full_proba(proba, est.classes_).argmax(axis=1), labels=np.unique(y),
                          average="macro", zero_division=0))


def run_experiment(table: pd.DataFrame, feature_set: str = "A", models=("logreg", "forest"),
                   selectors=("none", "mi"), k: int = 8, seed: int = 42, n_splits: int = 5, nested: bool = False,
                   n_boot: int = 200) -> Experiment:
    data = labelled(table)
    cols = columns_for(feature_set)
    X = make_features(data)[cols]
    y = encode(data["group"])
    groups = data["OASISID"]
    tr, te = holdout_split(y, groups, seed=seed)
    cands = candidates(models, selectors, k)
    inner = min(n_splits, groups.iloc[tr].nunique())
    nested_scores = nested_cv(X.iloc[tr], y[tr], groups.iloc[tr], cands, outer_splits=n_splits,
                              inner_splits=3, seed=seed, scorer=_macro_f1_scorer) if nested else []
    best, chosen, log = inner_search(X.iloc[tr], y[tr], groups.iloc[tr], cands, inner, seed)
    proba = full_proba(best.predict_proba(X.iloc[te]), best.classes_)
    test = metrics(y[te], proba)
    ci = {key: subject_bootstrap(y[te], proba, groups.iloc[te], key, n_boot, seed)
          for key in ("macro_f1", "balanced_accuracy")} if n_boot else {}
    base = build_pipeline("majority").fit(X.iloc[tr], y[tr])
    baseline = metrics(y[te], full_proba(base.predict_proba(X.iloc[te]), base.classes_))
    imp = permutation_importance(best, X.iloc[te], y[te], scoring="f1_macro", n_repeats=3, random_state=seed)
    importance = pd.DataFrame({"feature": cols, "macro_f1_drop": imp.importances_mean, "std": imp.importances_std})
    return Experiment(
        feature_set=feature_set, features=cols, pipeline=best, chosen=chosen, search_log=log, test=test,
        baseline_test=baseline, ci=ci, nested=nested_scores,
        importance=importance.sort_values("macro_f1_drop", ascending=False).reset_index(drop=True),
        n_train_subjects=int(groups.iloc[tr].nunique()), n_test_subjects=int(groups.iloc[te].nunique()),
        n_train_visits=len(tr), n_test_visits=len(te),
    )


def leakage_check(table: pd.DataFrame, feature_set: str = "A", seed: int = 42) -> dict:
    """Macro-F1 of one fixed forest with a visit-level split vs. a participant-level split."""
    data = labelled(table)
    X = make_features(data)[columns_for(feature_set)]
    y = encode(data["group"])
    out = {}
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
    pipe = build_pipeline("forest", seed=seed).fit(X.iloc[tr], y[tr])
    out["visit_split"] = _macro_f1_scorer(y[te], pipe.predict_proba(X.iloc[te]), pipe)
    tr, te = holdout_split(y, data["OASISID"], seed=seed)
    pipe = build_pipeline("forest", seed=seed).fit(X.iloc[tr], y[tr])
    out["participant_split"] = _macro_f1_scorer(y[te], pipe.predict_proba(X.iloc[te]), pipe)
    out["gap"] = out["visit_split"] - out["participant_split"]
    return out


def model_card(exp: Experiment) -> str:
    t, b = exp.test, exp.baseline_test
    lines = [
        f"# Model card: dementia-typer, feature set {exp.feature_set}",
        "",
        f"- Package version {__version__}, date {date.today().isoformat()}.",
        f"- Feature set {exp.feature_set}: {SET_DESCRIPTIONS[exp.feature_set]}.",
        f"- Features ({len(exp.features)}): {', '.join(exp.features)}.",
        f"- Chosen by inner grouped CV (macro-F1): {exp.chosen['model']} with selector {exp.chosen['selector']}, "
        f"parameters {exp.chosen['params']}.",
        f"- Participants: {exp.n_train_subjects} train, {exp.n_test_subjects} test. "
        f"Visits: {exp.n_train_visits} train, {exp.n_test_visits} test. No participant is in both.",
        "",
        "## Intended use",
        "",
        "Research on objective markers of dementia groups. It is not a diagnostic tool. A clinician must make",
        "every diagnosis. The training data come from one research cohort and do not represent all populations.",
        "",
    ]
    if LEAKAGE_FEATURES & set(exp.features):
        lines += ["**Warning:** this feature set contains CDR scores, which clinicians use to assign the diagnosis.",
                  "Its results are a leakage ceiling, not a valid performance estimate.", ""]
    lines += ["## Test metrics", "", "| Metric | Model | Majority baseline |", "|---|---|---|"]
    for key in SUMMARY:
        lines.append(f"| {key} | {t[key]:.3f} | {b[key]:.3f} |")
    lines += ["", "Recall for each class: " + ", ".join(f"{k} {v:.3f}" for k, v in t["recall"].items()) + "."]
    if t["absent_classes"]:
        lines.append("Classes with no test visit (recall not defined): " + ", ".join(t["absent_classes"]) + ".")
    if exp.ci:
        lo, hi = exp.ci["macro_f1"]
        lines.append(f"Macro-F1 95% participant bootstrap CI: {lo:.3f} to {hi:.3f}.")
    return "\n".join(lines) + "\n"


def save(exp: Experiment, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"model": out / "model.joblib", "metrics": out / "metrics.json", "card": out / "model_card.md"}
    joblib.dump({"pipeline": exp.pipeline, "features": exp.features, "classes": list(CLASSES),
                 "feature_set": exp.feature_set, "chosen": exp.chosen, "version": __version__}, paths["model"])
    payload = {"feature_set": exp.feature_set, "chosen": exp.chosen, "search_log": exp.search_log, "test": exp.test,
               "baseline_test": exp.baseline_test, "ci": exp.ci, "nested": exp.nested,
               "importance": exp.importance.to_dict(orient="records") if exp.importance is not None else []}
    paths["metrics"].write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    paths["card"].write_text(model_card(exp), encoding="utf-8")
    return paths


def load(path: str | Path) -> dict:
    bundle = joblib.load(path)
    if tuple(bundle["classes"]) != CLASSES:
        raise ValueError("the model was saved with a different class list")
    return bundle
