"""Problem 3 (labels decoded wrongly) and problem 6 (fragile substring mapping)."""
import pandas as pd
import pytest

from dementia_typer.labels import CLASSES, decode, encode, map_diagnosis, unmapped


@pytest.mark.parametrize("text, group", [
    ("Cognitively normal", "Normal"),
    ("No dementia", "Normal"),
    ("0.5 in memory only", "MCI"),
    ("uncertain dementia", "MCI"),
    ("Incipient demt PTP", "MCI"),
    ("Incipient Non-AD dem", "MCI"),
    ("uncertain- possible NON AD dem", "MCI"),
    ("Unc: ques. Impairment", "MCI"),
    ("Unc: impair reversible", "MCI"),
    ("ProAph w/o dement", "MCI"),
    ("AD Dementia", "Alzheimer's"),
    ("AD dem w/CVD contribut", "Alzheimer's"),
    ("DAT", "Alzheimer's"),
    ("Vascular Demt- primary", "Vascular"),
    ("Non AD dem- Other primary", "Other dementia"),
    ("Non AD dementia", "Other dementia"),
    ("DLBD- primary", "Other dementia"),
    ("Frontotemporal demt. prim", "Other dementia"),
    ("Dementia/PD- primary", "Other dementia"),
])
def test_known_dx_texts(text, group):
    assert map_diagnosis(text) == group


@pytest.mark.parametrize("text", ["Depression", "", None, float("nan"), "headache"])
def test_unknown_texts_are_not_guessed(text):
    assert map_diagnosis(text) is None


def test_substrings_inside_words_do_not_match():
    # "ad" inside "headache" or "dat" inside "update" must not give Alzheimer's.
    assert map_diagnosis("update headache") is None


def test_unmapped_report():
    assert unmapped(pd.Series(["AD Dementia", "Stroke", "Stroke", None])) == ["Stroke"]


def test_encode_decode_round_trip_uses_one_class_list():
    assert CLASSES == ("Normal", "MCI", "Alzheimer's", "Vascular", "Other dementia")
    codes = encode(pd.Series(["Vascular", "Normal"]))
    assert list(codes) == [3, 0]
    assert decode(codes) == ["Vascular", "Normal"]
