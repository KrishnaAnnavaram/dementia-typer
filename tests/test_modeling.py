"""Problems 2 (participant leakage), 4 (selection on test), 5 (preprocessing before split),
7 (wrong model for a selector) and 8 (accuracy only)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from dementia_typer.evaluate import full_proba, metrics, multiclass_brier, subject_bootstrap, top_label_ece
from dementia_typer.features import columns_for, make_features
from dementia_typer.labels import CLASSES, encode
from dementia_typer.modeling import (
    LeakageError,
    assert_disjoint,
    build_pipeline,
    candidates,
    holdout_split,
    inner_search,
    nested_cv,
)
from dementia_typer.train import labelled, leakage_check


def xy(table, fs="A"):
    data = labelled(table)
    return make_features(data)[columns_for(fs)], encode(data["group"]), data["OASISID"]


def test_holdout_split_keeps_participants_apart(table):
    X, y, g = xy(table)
    tr, te = holdout_split(y, g, seed=1)
    assert not set(g.iloc[tr]) & set(g.iloc[te])
    with pytest.raises(LeakageError):
        assert_disjoint(pd.Series(["a", "a"]), np.array([0]), np.array([1]))


def test_preprocessing_lives_inside_the_pipeline():
    pipe = build_pipeline("logreg", "mi", k=4)
    assert isinstance(pipe, Pipeline)
    assert [name for name, _ in pipe.steps] == ["impute", "scale", "select", "model"]
    with pytest.raises(ValueError):
        build_pipeline("svm")
    with pytest.raises(ValueError):
        build_pipeline("logreg", "pca")


def test_each_selector_keeps_its_own_fitted_pipeline(table):
    X, y, g = xy(table, "B")
    mi = build_pipeline("logreg", "mi", k=5).fit(X, y)
    rfe = build_pipeline("logreg", "rfe", k=5).fit(X, y)
    assert mi.named_steps["select"].get_support().sum() == 5
    assert rfe.named_steps["select"].get_support().sum() == 5
    assert mi.predict_proba(X).shape == rfe.predict_proba(X).shape == (len(X), len(np.unique(y)))


def test_inner_search_uses_grouped_cv_and_logs_each_candidate(table):
    X, y, g = xy(table)
    best, info, log = inner_search(X, y, g, candidates(("logreg",), ("none", "mi"), 6), n_splits=3, seed=0)
    assert len(log) == 2 and info in log
    assert info["inner_macro_f1"] == max(r["inner_macro_f1"] for r in log)
    assert hasattr(best, "predict_proba")


def test_nested_cv_reports_each_outer_fold(table):
    X, y, g = xy(table)
    folds = nested_cv(X, y, g, candidates(("logreg",), ("none",)), outer_splits=3, inner_splits=2, seed=0,
                      scorer=lambda yt, p, est: float((full_proba(p, est.classes_).argmax(1) == yt).mean()))
    assert len(folds) == 3 and all(0 <= f["score"] <= 1 for f in folds)


def test_experiment_metrics_beat_baseline(experiment):
    t, b = experiment.test, experiment.baseline_test
    assert set(t["recall"]) == set(CLASSES)
    assert t["macro_f1"] > b["macro_f1"] + 0.2
    assert t["balanced_accuracy"] > b["balanced_accuracy"]
    lo, hi = experiment.ci["macro_f1"]
    assert lo <= t["macro_f1"] <= hi
    assert experiment.n_train_subjects + experiment.n_test_subjects == 160


def test_importance_is_computed_on_test_rows(experiment):
    imp = experiment.importance
    assert list(imp.columns) == ["feature", "macro_f1_drop", "std"]
    assert set(imp["feature"]) == set(experiment.features)


def test_cdr_set_is_a_ceiling(table):
    from dementia_typer.train import run_experiment

    a = run_experiment(table, "A", models=("logreg",), selectors=("none",), seed=0, n_splits=3, n_boot=0)
    c = run_experiment(table, "C", models=("logreg",), selectors=("none",), seed=0, n_splits=3, n_boot=0)
    assert c.test["balanced_accuracy"] > a.test["balanced_accuracy"]


def test_visit_split_is_optimistic(table):
    r = leakage_check(table, "A", seed=0)
    assert set(r) == {"visit_split", "participant_split", "gap"}
    assert r["gap"] == pytest.approx(r["visit_split"] - r["participant_split"])


def test_metric_helpers():
    y = np.array([0, 1, 2, 2])
    proba = np.eye(5)[[0, 1, 2, 3]] * 0.9 + 0.02
    m = metrics(y, proba)
    assert m["support"]["Normal"] == 1 and m["support"]["Vascular"] == 0
    assert np.asarray(m["confusion_matrix"]).sum() == 4
    assert multiclass_brier(y, np.eye(5)[y]) == 0.0
    assert top_label_ece(y, np.eye(5)[y]) == 0.0
    aligned = full_proba(np.array([[0.3, 0.7]]), np.array([1, 4]))
    assert aligned.tolist() == [[0.0, 0.3, 0.0, 0.0, 0.7]]
    lo, hi = subject_bootstrap(y, proba, ["a", "a", "b", "c"], n_boot=20)
    assert 0 <= lo <= hi <= 1
    with pytest.raises(ValueError):
        subject_bootstrap(y, proba, ["a", "b", "c", "d"], key="auc", n_boot=2)


def test_holdout_contains_every_group_with_enough_participants(table):
    X, y, g = xy(table)
    tr, te = holdout_split(y, g, seed=3)
    per_subject = pd.Series(y).groupby(g.to_numpy()).last()
    for cls, count in per_subject.value_counts().items():
        if count >= 5:
            assert (y[te] == cls).any(), CLASSES[cls]


def test_absent_class_has_no_recall_and_no_macro_weight():
    y = np.array([0, 0, 1, 1])
    proba = np.eye(5)[[0, 0, 1, 1]]
    m = metrics(y, proba)
    assert m["macro_f1"] == 1.0
    assert np.isnan(m["recall"]["Vascular"]) and "Vascular" in m["absent_classes"]
