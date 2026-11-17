"""Data preparation for the CIPN proof-of-pipeline project (CALGB 40502, paclitaxel arm).

Everything that touches the raw CSVs lives here so the leakage rules are
enforced in exactly one place.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

SENSORY_EVENT = "Neuropathy: sensory"
POSITIVE_GRADE = 2  # target: max sensory-neuropathy grade >= 2
UNKNOWN = "unknown"

# Baseline features, and nothing else. Each entry:
#   column in the efficacy CSV (lower-cased) -> (label, {code: option label})
# Codes follow c40502_datadictionary.pdf. Missing values become an explicit
# "unknown" category so the model and the web form share the same inputs.
FEATURE_SPECS: dict[str, dict] = {
    "agecat": {
        "label": "Age group",
        "options": {"1": "20–49", "2": "50–69", "3": "70+"},
    },
    "sex_id": {
        "label": "Sex",
        "options": {"1": "Male", "2": "Female"},
    },
    "racecat": {
        "label": "Race",
        "options": {"1": "White", "2": "Black", "3": "Other"},
        "unknown_codes": [9],  # dictionary code 9 = Unknown
    },
    "strat1_taxane": {
        "label": "Prior adjuvant taxane",
        "options": {"1": "Yes", "2": "No"},
    },
    "anyvisceral": {
        "label": "Visceral disease",
        "options": {"1": "Yes", "0": "No"},
    },
    "dficat": {
        "label": "Disease-free interval",
        "options": {"0": "De novo", "1": "Up to 1 yr", "2": "1–2 yrs", "3": "2+ yrs"},
    },
    "clinstage4": {
        "label": "Clinical stage",
        "options": {"1": "Stage IV", "0": "Not stage IV"},
    },
    "hormone": {
        "label": "Hormone receptor",
        "options": {"1": "Positive", "0": "Negative"},
    },
    "her2": {
        "label": "HER2",
        "options": {"1": "Positive", "0": "Negative"},
    },
}
FEATURES = list(FEATURE_SPECS)

# Outcome-related columns that must never be used as inputs.
FORBIDDEN = [
    "survmos", "survstat", "fumos", "pfsmos", "pfsstat", "ttfmos", "ttfstat",
    "bestresp", "txendreas", "sstat", "pac_dc", "nab_dc", "ixa_dc",
]
assert not set(FEATURES) & set(FORBIDDEN)


def load_baseline(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "c40502_efficacy.csv")
    df.columns = [c.lower() for c in df.columns]
    return df


def sensory_max_grade(raw_dir: Path = RAW_DIR) -> pd.Series:
    """Max sensory-neuropathy grade per patient (after dropping duplicate AE rows)."""
    ae = pd.read_csv(raw_dir / "c40502_ae.csv").drop_duplicates()
    ev = ae[ae["eventname"].fillna("").str.strip() == SENSORY_EVENT]
    return ev.groupby("MASK_ID")["grade_ID"].max()


def build_dataset(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Return (X, y, info). X is all-categorical strings; y is 0/1."""
    base = load_baseline(raw_dir)
    max_grade = sensory_max_grade(raw_dir)

    info = {"patients_in_baseline_file": int(len(base))}

    # Patients who never received protocol therapy have no AE records, so they
    # cannot be labelled; keep only treated patients.
    base = base[base["treated"] == 1].copy()
    info["after_dropping_untreated"] = int(len(base))

    # No sensory-neuropathy event on record -> grade 0.
    base["max_grade"] = base["mask_id"].map(max_grade).fillna(0)

    # grade -1 means "unknown event": no usable label.
    info["dropped_grade_minus1"] = int((base["max_grade"] == -1).sum())
    base = base[base["max_grade"] != -1].copy()

    y = (base["max_grade"] >= POSITIVE_GRADE).astype(int)
    y.index = base["mask_id"].to_numpy()

    X = base[FEATURES].copy()
    for col, spec in FEATURE_SPECS.items():
        for code in spec.get("unknown_codes", []):
            X.loc[X[col] == code, col] = np.nan
        X[col] = X[col].astype("Int64").astype("string").fillna(UNKNOWN)
    X.index = base["mask_id"].to_numpy()

    info["n_patients"] = int(len(y))
    info["n_positive"] = int(y.sum())
    info["positive_rate"] = float(y.mean())
    return X, y, info


def audit_dose_file(raw_dir: Path = RAW_DIR, max_cycle: int = 100) -> dict:
    """Clean the per-cycle dose file and report its quirks.

    The dose file is NOT a source of model features (its discontinuation flags
    leak the outcome); this audit only documents and handles the known quirks:
    an implausible cycleno outlier (2928) and patients absent from the baseline file.
    """
    cyc = pd.read_csv(raw_dir / "c40502_cycles.csv")
    base_ids = set(load_baseline(raw_dir)["mask_id"])

    orphans = sorted(set(cyc["MASK_ID"]) - base_ids)
    cyc = cyc[cyc["MASK_ID"].isin(base_ids)]
    bad = cyc[(cyc["cycleno"] < 1) | (cyc["cycleno"] > max_cycle)]
    cyc = cyc.drop(bad.index)  # dropped (not capped): a cap would invent a cycle number
    return {
        "rows_after_cleaning": int(len(cyc)),
        "patients_not_in_baseline": orphans,
        "rows_dropped_implausible_cycleno": bad["cycleno"].tolist(),
    }
