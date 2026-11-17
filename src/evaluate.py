"""Repeated stratified k-fold CV with bootstrap confidence intervals.

Procedure
---------
1. For each of R repeats, run stratified 5-fold CV and keep the out-of-fold
   predicted probability for every patient -> preds[r, i].
2. Point estimate of a metric = mean over repeats of the metric on all patients.
3. 95% CI = percentile bootstrap over *patients* (the dominant source of
   uncertainty with n~270). Each bootstrap draw resamples patients once and
   applies the same resample to all R repeats, then averages over repeats.
   Naively taking percentiles across CV folds/repeats would give far-too-narrow
   intervals because the folds share training data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from data import audit_dose_file, build_dataset
from models import MODELS

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
N_SPLITS, N_REPEATS, N_BOOT, SEED = 5, 10, 2000, 42

METRICS = {
    "AUC": roc_auc_score,
    "Accuracy": lambda y, p: accuracy_score(y, p >= 0.5),
    "Brier": brier_score_loss,
}


def oof_predictions(make_model, X, y) -> np.ndarray:
    preds = np.empty((N_REPEATS, len(y)))
    for r in range(N_REPEATS):
        cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED + r)
        preds[r] = cross_val_predict(make_model(), X, y, cv=cv, method="predict_proba")[:, 1]
    return preds


def metric_over_repeats(fn, y, preds, idx=None) -> float:
    if idx is not None:
        y, preds = y[idx], preds[:, idx]
    return float(np.mean([fn(y, p) for p in preds]))


def bootstrap_indices(n: int, y: np.ndarray, rng) -> list[np.ndarray]:
    """Stratified bootstrap so every resample keeps both classes."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    return [
        np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for _ in range(N_BOOT)
    ]


def ci(samples) -> list[float]:
    return [float(v) for v in np.percentile(samples, [2.5, 97.5])]


def main() -> None:
    X, y_s, info = build_dataset()
    y = y_s.to_numpy()
    rng = np.random.default_rng(SEED)
    boots = bootstrap_indices(len(y), y, rng)

    all_preds, results = {}, {}
    for name, make in MODELS.items():
        print(f"CV: {name}")
        preds = oof_predictions(make, X, y)
        all_preds[name] = preds
        results[name] = {}
        for m, fn in METRICS.items():
            est = metric_over_repeats(fn, y, preds)
            samples = [metric_over_repeats(fn, y, preds, idx) for idx in boots]
            results[name][m] = {"estimate": est, "ci95": ci(samples)}

    # Paired difference in AUC (forest - logistic) on the same bootstrap draws.
    a, b = all_preds["Random forest"], all_preds["Logistic regression"]
    diff = [
        metric_over_repeats(roc_auc_score, y, a, idx) - metric_over_repeats(roc_auc_score, y, b, idx)
        for idx in boots
    ]
    diff_est = metric_over_repeats(roc_auc_score, y, a) - metric_over_repeats(roc_auc_score, y, b)

    majority = max(y.mean(), 1 - y.mean())
    out = {
        "data": info,
        "dose_file_audit": audit_dose_file(),
        "cv": {"splits": N_SPLITS, "repeats": N_REPEATS, "bootstrap": N_BOOT, "seed": SEED},
        "majority_class_accuracy": float(majority),
        "models": results,
        "auc_difference_forest_minus_logistic": {"estimate": diff_est, "ci95": ci(diff)},
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "cv_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (RESULTS_DIR / "cv_results.md").write_text(render_markdown(out), encoding="utf-8")
    print(render_markdown(out))


def render_markdown(out: dict) -> str:
    d, cv = out["data"], out["cv"]
    lines = [
        "| Model | AUC (95% CI) | Accuracy (95% CI) | Brier (95% CI) |",
        "|---|---|---|---|",
    ]
    for name, res in out["models"].items():
        cells = [
            f"{res[m]['estimate']:.3f} ({res[m]['ci95'][0]:.3f}–{res[m]['ci95'][1]:.3f})"
            for m in ("AUC", "Accuracy", "Brier")
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    dd = out["auc_difference_forest_minus_logistic"]
    lines += [
        "",
        f"n = {d['n_patients']} patients, {d['n_positive']} positive ({d['positive_rate']:.1%}). "
        f"Majority-class accuracy = {out['majority_class_accuracy']:.3f}; chance AUC = 0.5.",
        f"{cv['repeats']}×{cv['splits']}-fold stratified CV; {cv['bootstrap']}-draw stratified patient bootstrap CIs.",
        f"AUC difference (forest − logistic): {dd['estimate']:+.3f} "
        f"({dd['ci95'][0]:+.3f} to {dd['ci95'][1]:+.3f}).",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
