"""Optional SHAP explanation. It skips when `shap` is not installed (for example in CI)."""
import pytest

from dementia_typer.explain import shap_values
from dementia_typer.features import make_features
from dementia_typer.train import labelled


def test_shap_on_held_out_rows(experiment, table):
    pytest.importorskip("shap")
    X = make_features(labelled(table))[experiment.features].head(30)
    values = shap_values(experiment.pipeline, X, background=X.head(10))
    assert values.values.shape[0] == 30
