"""Problem 3 (wrong class names in the demo) and problem 9 (hard-coded feature order), plus the CLI."""
import json

import pytest
from pydantic import ValidationError

from dementia_typer.cli import main
from dementia_typer.predict import Visit, predict_visit
from dementia_typer.train import load, save

HEALTHY = {"age at visit": 68, "sex": "F", "education_years": 16, "IntraCranialVol": 1.40e6,
           "TotalGrayVol": 0.645e6, "CortexVol": 0.52e6, "CorticalWhiteMatterVol": 0.465e6,
           "SubCortGrayVol": 0.053e6, "Left-Hippocampus": 3800, "Right-Hippocampus": 3900,
           "Left-Lateral-Ventricle": 12000, "Right-Lateral-Ventricle": 11500, "CSF": 1250, "MMSE": 30}


def test_saved_bundle_predicts_names(experiment, tmp_path):
    paths = save(experiment, tmp_path)
    bundle = load(paths["model"])
    result = predict_visit(bundle, HEALTHY)
    assert result["prediction"] in result["probabilities"]
    assert set(result["probabilities"]) == {"Normal", "MCI", "Alzheimer's", "Vascular", "Other dementia"}
    assert abs(sum(result["probabilities"].values()) - 1) < 1e-3
    assert "clinician" in result["note"]
    card = paths["card"].read_text(encoding="utf-8")
    assert "not a diagnostic tool" in card


def test_healthy_profile_is_not_called_severe_dementia(experiment, tmp_path):
    bundle = load(save(experiment, tmp_path)["model"])
    probs = predict_visit(bundle, HEALTHY)["probabilities"]
    assert probs["Normal"] > probs["Alzheimer's"]


def test_schema_rejects_bad_input():
    with pytest.raises(ValidationError):
        Visit.model_validate({**HEALTHY, "MMSE": 45})
    with pytest.raises(ValidationError):
        Visit.model_validate({**HEALTHY, "shoe_size": 9})
    with pytest.raises(ValidationError):
        Visit.model_validate({"sex": "F"})


def test_feature_order_comes_from_the_bundle(experiment, tmp_path):
    bundle = load(save(experiment, tmp_path)["model"])
    shuffled = dict(reversed(list(HEALTHY.items())))
    assert predict_visit(bundle, shuffled)["probabilities"] == predict_visit(bundle, HEALTHY)["probabilities"]


def test_cli_end_to_end(tmp_path, capsys):
    data = tmp_path / "d"
    assert main(["synth", "--out", str(data), "--subjects", "80", "--seed", "3"]) == 0
    assert main(["validate", "--data", str(data)]) == 0
    assert "unmapped 0" in capsys.readouterr().out
    out = tmp_path / "run"
    assert main(["train", "--data", str(data), "--models", "logreg", "--selectors", "none", "--bootstrap", "10",
                 "--out", str(out)]) == 0
    payload = tmp_path / "visit.json"
    payload.write_text(json.dumps(HEALTHY), encoding="utf-8")
    capsys.readouterr()
    assert main(["predict", "--model", str(out / "model.joblib"), "--input", str(payload)]) == 0
    assert '"prediction"' in capsys.readouterr().out
    payload.write_text(json.dumps({**HEALTHY, "MMSE": 99}), encoding="utf-8")
    assert main(["predict", "--model", str(out / "model.joblib"), "--input", str(payload)]) == 1
    assert main(["leakage", "--data", str(data)]) == 0


def test_validate_flags_unmapped_text(tmp_path, capsys):
    data = tmp_path / "d"
    main(["synth", "--out", str(data), "--subjects", "10", "--seed", "1"])
    path = data / "clinical.csv"
    text = path.read_text(encoding="utf-8").replace("Cognitively normal", "Stroke", 1)
    if "Stroke" not in text:
        text = path.read_text(encoding="utf-8").replace("No dementia", "Stroke", 1)
    path.write_text(text, encoding="utf-8")
    assert main(["validate", "--data", str(data)]) == 2
    assert "Stroke" in capsys.readouterr().out


def test_missing_data_is_a_clean_error(tmp_path, capsys):
    assert main(["validate", "--data", str(tmp_path / "x")]) == 1
    assert "dementia-typer synth" in capsys.readouterr().err
