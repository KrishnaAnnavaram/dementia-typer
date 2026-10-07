"""Schema checks, the visit-to-MR join and problem 1 (CDR leakage) through the feature sets."""
import numpy as np
import pandas as pd
import pytest

from dementia_typer.config import Settings
from dementia_typer.data import (
    SchemaError,
    day_from_label,
    join_visits,
    load_tables,
    validate_clinical,
    validate_mri,
)
from dementia_typer.features import CDR, FEATURE_SETS, LEAKAGE_FEATURES, columns_for, make_features
from dementia_typer.synthetic import write


def test_day_from_label():
    assert day_from_label("OAS30001_MR_d0129") == 129
    assert np.isnan(day_from_label("bad"))


def test_join_takes_nearest_scan_in_window():
    clinical = pd.DataFrame({"OASISID": ["s1", "s1", "s2"], "days_to_visit": [0, 1000, 0],
                             "age at visit": [70, 73, 80], "dx1": ["Cognitively normal"] * 3})
    mri = pd.DataFrame({"OASISID": ["s1", "s1", "s2"], "days_to_scan": [30, 900, 600],
                        "IntraCranialVol": [1.5e6, 1.4e6, 1.3e6]})
    out = join_visits(clinical, mri, None, window_days=365).set_index(["OASISID", "days_to_visit"])
    assert out.loc[("s1", 0.0), "IntraCranialVol"] == 1.5e6
    assert out.loc[("s1", 1000.0), "IntraCranialVol"] == 1.4e6
    assert np.isnan(out.loc[("s2", 0.0), "IntraCranialVol"])  # 600 days is outside the window
    assert (out["group"] == "Normal").all()


def test_join_never_crosses_participants():
    clinical = pd.DataFrame({"OASISID": ["a"], "days_to_visit": [0], "age at visit": [70], "dx1": ["DAT"]})
    mri = pd.DataFrame({"OASISID": ["b"], "days_to_scan": [0], "IntraCranialVol": [1.5e6]})
    assert np.isnan(join_visits(clinical, mri, None)["IntraCranialVol"].iloc[0])


def test_failed_qc_sessions_are_dropped():
    mri = pd.DataFrame({"OASISID": ["a", "a"], "days_to_scan": [0, 10], "IntraCranialVol": [1e6, 1e6],
                        "FS QC Status": ["Passed", "Failed"]})
    assert len(validate_mri(mri)) == 1


def test_schema_errors():
    with pytest.raises(SchemaError, match="dx1"):
        validate_clinical(pd.DataFrame({"OASISID": ["a"], "days_to_visit": [0], "age at visit": [70]}))
    with pytest.raises(SchemaError, match="MMSE"):
        validate_clinical(pd.DataFrame({"OASISID": ["a"], "days_to_visit": [0], "age at visit": [70],
                                        "dx1": ["DAT"], "MMSE": [45]}))
    with pytest.raises(SchemaError, match="unique"):
        validate_clinical(pd.DataFrame({"OASISID": ["a", "a"], "days_to_visit": [0, 0], "age at visit": [70, 70],
                                        "dx1": ["DAT", "DAT"]}))
    with pytest.raises(SchemaError, match="positive"):
        validate_mri(pd.DataFrame({"OASISID": ["a"], "days_to_scan": [0], "IntraCranialVol": [0]}))


def test_days_from_session_label_when_column_missing():
    clinical = pd.DataFrame({"OASISID": ["a"], "OASIS_session_label": ["a_ClinicalData_d0042"],
                             "age at visit": [70], "dx1": ["DAT"]})
    assert validate_clinical(clinical)["days_to_visit"].iloc[0] == 42


def test_load_tables(tmp_path):
    write(tmp_path, n_subjects=20, seed=1)
    assert set(load_tables(tmp_path)) == {"clinical", "mri", "demographics"}
    with pytest.raises(FileNotFoundError):
        load_tables(tmp_path / "none")


def test_feature_sets_and_leakage_flags(table):
    assert not set(FEATURE_SETS["A"]) & LEAKAGE_FEATURES
    assert "MMSE" not in FEATURE_SETS["A"] and "MMSE" in FEATURE_SETS["B"]
    assert set(CDR) <= set(FEATURE_SETS["C"])
    feats = make_features(table)
    assert set(columns_for("C")) <= set(feats.columns)
    ratio = feats["Left-Hippocampus_icv"].dropna()
    assert 1.0 < ratio.median() < 4.0  # per mille of the intracranial volume
    with pytest.raises(ValueError):
        columns_for("D")


def test_settings(monkeypatch):
    monkeypatch.setenv("DEMENTIA_JOIN_WINDOW_DAYS", "180")
    assert Settings.from_env().join_window_days == 180
    monkeypatch.setenv("DEMENTIA_N_SPLITS", "1")
    with pytest.raises(ValueError):
        Settings.from_env()
