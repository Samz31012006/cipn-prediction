"""Fit the logistic regression on all patients and export it for the web page.

The exported JSON holds the form definition plus one coefficient per option, so
the browser computes score = sigmoid(intercept + sum of selected coefficients).
"""
import json
from pathlib import Path

from data import FEATURE_SPECS, FEATURES, UNKNOWN, build_dataset
from models import logistic_regression

OUT = Path(__file__).resolve().parent.parent / "web" / "model.json"


def main() -> None:
    X, y, info = build_dataset()
    pipe = logistic_regression().fit(X, y)
    enc = pipe.named_steps["prep"].named_transformers_["onehot"]
    clf = pipe.named_steps["clf"]

    coef_by_feature = {f: {} for f in FEATURES}
    offset = 0
    for feat, cats in zip(FEATURES, enc.categories_):
        for cat in cats:
            coef_by_feature[feat][str(cat)] = float(clf.coef_[0][offset])
            offset += 1

    # Cross-validated AUC (from evaluate.py) is shown on the page so the score is never presented without it.
    cv_file = OUT.parent.parent / "results" / "cv_results.json"
    cv_auc = None
    if cv_file.exists():
        a = json.loads(cv_file.read_text(encoding="utf-8"))["models"]["Logistic regression"]["AUC"]
        cv_auc = {"estimate": a["estimate"], "ci95": a["ci95"]}

    fields = []
    for feat in FEATURES:
        spec = FEATURE_SPECS[feat]
        labels = {**spec["options"], UNKNOWN: "Unknown"}
        options = [
            {"value": code, "label": label, "coef": coef_by_feature[feat][code]}
            for code, label in labels.items()
            if code in coef_by_feature[feat]  # only categories seen in training
        ]
        fields.append({"key": feat, "label": spec["label"], "options": options})

    model = {
        "intercept": float(clf.intercept_[0]),
        "fields": fields,
        "base_rate": info["positive_rate"],
        "n_patients": info["n_patients"],
        "cv_auc": cv_auc,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(model, indent=2), encoding="utf-8")

    # Reference predictions so the browser implementation can be checked.
    ref = X.head(5)
    (OUT.parent.parent / "results" / "reference_predictions.json").write_text(json.dumps([
        {"inputs": row.to_dict(), "probability": float(p)}
        for (_, row), p in zip(ref.iterrows(), pipe.predict_proba(ref)[:, 1])
    ], indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
