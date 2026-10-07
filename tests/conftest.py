import pytest

from dementia_typer.data import join_visits, validate_clinical, validate_demographics, validate_mri
from dementia_typer.synthetic import generate


@pytest.fixture(scope="session")
def tables():
    return generate(n_subjects=160, seed=5)


@pytest.fixture(scope="session")
def table(tables):
    return join_visits(validate_clinical(tables["clinical"]), validate_mri(tables["freesurfer"]),
                       validate_demographics(tables["demographics"]), window_days=365)


@pytest.fixture(scope="session")
def experiment(table):
    from dementia_typer.train import run_experiment

    return run_experiment(table, "A", models=("logreg",), selectors=("none",), seed=0, n_splits=3, n_boot=30)
