# CIPN Prediction: a proof of pipeline

A small academic machine-learning project. It builds an end-to-end pipeline on real
public trial data: cleaning, label construction, leakage-safe features, model
comparison with confidence intervals, and a one-page web form that serves the model.

**This is a proof of pipeline, not a medical tool.** It makes no clinical claims and
must not be used for any decision about patient care.

## Data

[CALGB 40502](https://clinicaltrials.gov/study/NCT00785291) paclitaxel (control) arm,
from [Project Data Sphere](https://www.projectdatasphere.org/). **The raw data is not
included in this repository**; it is subject to Project Data Sphere's terms, so get your own
copy there and place these files in `data/raw/` (git-ignored):

- `c40502_efficacy.csv`: baseline file, 283 patients, one row each
- `c40502_ae.csv`: adverse events, one row per event
- `c40502_cycles.csv`: per-cycle dose records (audited only, see below)

## Target

Whether a patient ever has a grade ≥ 2 sensory neuropathy event ("Neuropathy: sensory" in
the AE file), using each patient's maximum grade.

1. Drop duplicate AE rows (4,141 of 8,436).
2. Max grade per patient; a treated patient with no sensory event on record is grade 0.
3. Drop the 3 patients whose max grade is -1 (unknown event).
4. Drop the 8 patients who never started treatment (`treated = 0`). They have no AE
   records at all, so labelling them "no neuropathy" would be an artifact, not a
   finding. With this exclusion the positive rate is 128 / 272 = 47.1%.

## Features (baseline only)

`agecat`, `sex_id`, `racecat`, `strat1_taxane` (prior taxane), `anyvisceral`, `dficat`
(disease-free interval, 4 levels), `clinstage4`, `hormone`, `her2`. All are treated as
categorical; missing values become an explicit "unknown" level.

Never used, because they leak the outcome or postdate baseline: `survmos`, `survstat`,
`pfsmos`, `pfsstat`, `ttfmos`, `ttfstat`, `bestresp`, `txendreas`, `sstat`, and the
`*_dc` discontinuation flags in the dose file (`fumos` too). A test enforces this.

Dose file quirks: `src/data.py::audit_dose_file` drops the implausible `cycleno`
values (2928, and a stray -2; dropped rather than capped, since a cap would invent a cycle
number) and the 5 patients absent from the baseline file. The audit is reported in
`results/cv_results.json`; the dose file feeds no model features.

## Models and evaluation

Five candidates, all on the same nine baseline features. The list and every hyperparameter
were fixed before the second pass was run, and nothing was tuned on the CV folds:

1. Logistic regression, L2, C = 1 (first-pass baseline)
2. Logistic regression, L2, C = 0.1 (stronger regularization)
3. Logistic regression, C = 0.1, plus four a-priori interaction terms
   (age × prior taxane, race × prior taxane, disease-free interval × prior taxane,
   visceral × clinical stage)
4. Random forest (300 trees, depth 4, min leaf 5)
5. Gradient boosting (100 trees, depth 2, learning rate 0.05, subsample 0.8)

Evaluation: 10 × repeated stratified 5-fold CV. 95% confidence intervals are percentile
intervals from a 2,000-draw stratified bootstrap over patients, applied to the out-of-fold
predictions (see `src/evaluate.py` for why per-fold percentiles would be too narrow).
The rule for which model is served was also fixed in advance: a candidate replaces the baseline
only if the paired bootstrap interval of its AUC difference excludes zero.

| Model | AUC (95% CI) | Accuracy (95% CI) | Brier (95% CI) |
|---|---|---|---|
| Logistic regression (C=1) | 0.511 (0.451–0.568) | 0.524 (0.483–0.564) | 0.264 (0.252–0.277) |
| Logistic regression (C=0.1) | 0.485 (0.429–0.541) | 0.527 (0.491–0.563) | 0.255 (0.250–0.261) |
| Logistic + interactions (C=0.1) | 0.505 (0.450–0.565) | 0.527 (0.489–0.568) | 0.257 (0.249–0.264) |
| Random forest | 0.500 (0.441–0.563) | 0.544 (0.504–0.586) | 0.255 (0.248–0.262) |
| Gradient boosting | 0.510 (0.450–0.570) | 0.526 (0.482–0.570) | 0.267 (0.253–0.282) |

Majority-class accuracy is 0.529; chance AUC is 0.5. Paired AUC differences against the
baseline: C=0.1 −0.026 (−0.041 to −0.009); interactions −0.005 (−0.037 to +0.025); forest
−0.011 (−0.036 to +0.014); boosting −0.001 (−0.022 to +0.020).

**Result: nothing beats the baseline, and nothing beats chance.** Every AUC interval includes
0.5, and every accuracy interval includes the majority-class rate. Strong regularization is
nominally worse than the baseline (its paired difference excludes zero, on the downside). The simplest model
(baseline logistic regression) is therefore kept. The pipeline works; these nine baseline
variables carry no usable signal in this dataset. The web page shows its score next to this
caveat.

## Web page

`web/index.html` is a single static page: a form (numeric inputs for age and disease-free
interval, dropdowns for the rest), the score, a per-input breakdown, and an About section.
It loads `web/model.json` (baseline logistic regression coefficients and the CV table) and
computes everything in the browser; there is no backend. Each input's contribution is its
coefficient relative to the training-sample average for that feature, so the breakdown
reads as "odds ×" versus the average training patient. Live at
https://cipn-prediction.vercel.app (Vercel; `vercel.json` serves `web/`).

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest tests          # data tests skip if data/raw is empty
python src/evaluate.py          # CV + CIs  -> results/cv_results.{json,md}
python src/export_model.py      # fit on all patients -> web/model.json
```
