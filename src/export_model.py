"""Fit the baseline logistic regression on all patients and export it for the web page.

Score = sigmoid(intercept + sum of the selected option's coefficient per feature).
For the plain-language breakdown, each coefficient is re-expressed relative to the
training-sample average of its feature:

    contribution = coef[selected] - sum_c freq[c] * coef[c]
    baseline_logit = intercept + sum_features sum_c freq[c] * coef[c]

so score logit = baseline_logit + sum of contributions, and exp(contribution) is
the odds multiplier of that input versus the average training patient.
"""
import json
from pathlib import Path

from data import FEATURE_SPECS, FEATURES, UNKNOWN, build_dataset
from models import BASELINE, logistic_regression

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "model.json"

# How each feature is entered on the page. Numeric entries are binned into the
# dataset's categories (upper bound inclusive, checked in order).
INPUTS = {
    "agecat": {
        "type": "number", "label": "Age", "unit": "years", "min": 18, "max": 100,
        "bins": [{"max": 49.999, "value": "1"}, {"max": 69.999, "value": "2"}, {"max": None, "value": "3"}],
        "help": "Grouped as 20–49, 50–69, 70+",
    },
    "dficat": {
        "type": "number", "label": "Disease-free interval", "unit": "years", "min": 0, "max": 40, "step": 0.1,
        "bins": [{"max": 0, "value": "0"}, {"max": 1, "value": "1"}, {"max": 2, "value": "2"}, {"max": None, "value": "3"}],
        "help": "Years from first diagnosis to first recurrence; 0 = de novo",
    },
}


def main() -> None:
    X, y, info = build_dataset()
    pipe = logistic_regression().fit(X, y)
    enc = pipe.named_steps["prep"].named_transformers_["onehot"]
    clf = pipe.named_steps["clf"]

    coef_by_feature, offset = {f: {} for f in FEATURES}, 0
    for feat, cats in zip(FEATURES, enc.categories_):
        for cat in cats:
            coef_by_feature[feat][str(cat)] = float(clf.coef_[0][offset])
            offset += 1

    cv = json.loads((ROOT / "results" / "cv_results.json").read_text(encoding="utf-8"))

    baseline_logit = float(clf.intercept_[0])
    fields = []
    for feat in FEATURES:
        spec, coefs = FEATURE_SPECS[feat], coef_by_feature[feat]
        freq = X[feat].value_counts(normalize=True)
        mean = sum(float(freq.get(c, 0.0)) * v for c, v in coefs.items())
        baseline_logit += mean
        labels = {**spec["options"], UNKNOWN: "Unknown"}
        options = [
            {"value": c, "label": l, "contribution": coefs[c] - mean}
            for c, l in labels.items() if c in coefs  # only categories seen in training
        ]
        inp = INPUTS.get(feat, {"type": "select", "label": spec["label"]})
        fields.append({"key": feat, "name": spec["label"], **inp, "options": options})

    model = {
        "baseline_logit": baseline_logit,
        "fields": fields,
        "base_rate": info["positive_rate"],
        "n_patients": info["n_patients"],
        "n_positive": info["n_positive"],
        "n_enrolled": info["patients_in_baseline_file"],
        "served_model": BASELINE,
        "majority_accuracy": cv["majority_class_accuracy"],
        "cv": cv["cv"],
        "cv_models": {
            n: {m: r[m] for m in ("AUC", "Accuracy", "Brier")} for n, r in cv["models"].items()
        },
        "auc_diff": cv["auc_difference_vs_baseline"],
    }
    OUT.write_text(json.dumps(model, indent=1, ensure_ascii=False), encoding="utf-8")

    # Reference predictions for checking the browser implementation (git-ignored).
    ref = X.head(5)
    (ROOT / "results" / "reference_predictions.json").write_text(json.dumps([
        {"inputs": row.to_dict(), "probability": float(p)}
        for (_, row), p in zip(ref.iterrows(), pipe.predict_proba(ref)[:, 1])
    ], indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
