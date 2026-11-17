import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import data  # noqa: E402

HAS_DATA = (data.RAW_DIR / "c40502_efficacy.csv").exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="raw data not present in data/raw")


def test_no_forbidden_features():
    assert set(data.FEATURES).isdisjoint(data.FORBIDDEN)
    assert data.FEATURES == [
        "agecat", "sex_id", "racecat", "strat1_taxane", "anyvisceral",
        "dficat", "clinstage4", "hormone", "her2",
    ]


@needs_data
def test_dataset_shape_and_labels():
    X, y, info = data.build_dataset()
    assert list(X.columns) == data.FEATURES
    assert set(y.unique()) == {0, 1}
    assert X.index.is_unique and (X.index == y.index).all()
    assert info["dropped_grade_minus1"] == 3
    assert 0.44 < y.mean() < 0.50


@needs_data
def test_dose_audit_flags_known_quirks():
    audit = data.audit_dose_file()
    assert len(audit["patients_not_in_baseline"]) == 5
    assert 2928 in audit["rows_dropped_implausible_cycleno"]
