"""Candidate models.

The candidate list and every hyperparameter below were fixed BEFORE the second
modeling pass was run, and nothing is tuned on the cross-validation folds. The
rule for which model is served on the web page was also fixed in advance:
a candidate replaces the baseline logistic regression only if the paired
bootstrap interval for its AUC difference against the baseline excludes zero.
Otherwise the simplest model (the baseline) is kept.
"""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from data import FEATURES

# A few interaction terms chosen a priori (not searched for); they only combine
# the same nine baseline features.
INTERACTIONS = [
    ("agecat", "strat1_taxane"),
    ("racecat", "strat1_taxane"),
    ("dficat", "strat1_taxane"),
    ("anyvisceral", "clinstage4"),
]


def _add_interactions(X):
    X = X.copy()
    for a, b in INTERACTIONS:
        X[f"{a}*{b}"] = X[a].astype(str) + "|" + X[b].astype(str)
    return X


def _encoder(columns) -> ColumnTransformer:
    return ColumnTransformer(
        [("onehot", OneHotEncoder(handle_unknown="ignore"), list(columns))]
    )


def _plain(clf) -> Pipeline:
    return Pipeline([("prep", _encoder(FEATURES)), ("clf", clf)])


def logistic_regression() -> Pipeline:
    """Baseline from the first pass."""
    return _plain(LogisticRegression(C=1.0, max_iter=1000))


def logistic_strong_l2() -> Pipeline:
    return _plain(LogisticRegression(C=0.1, max_iter=1000))


def logistic_interactions() -> Pipeline:
    cols = FEATURES + [f"{a}*{b}" for a, b in INTERACTIONS]
    return Pipeline([
        ("interact", FunctionTransformer(_add_interactions)),
        ("prep", _encoder(cols)),
        ("clf", LogisticRegression(C=0.1, max_iter=1000)),
    ])


def random_forest() -> Pipeline:
    return _plain(RandomForestClassifier(
        n_estimators=300, max_depth=4, min_samples_leaf=5, random_state=0, n_jobs=-1,
    ))


def gradient_boosting() -> Pipeline:
    return _plain(GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=2, subsample=0.8,
        min_samples_leaf=5, random_state=0,
    ))


BASELINE = "Logistic regression (C=1)"
MODELS = {
    BASELINE: logistic_regression,
    "Logistic regression (C=0.1)": logistic_strong_l2,
    "Logistic + interactions (C=0.1)": logistic_interactions,
    "Random forest": random_forest,
    "Gradient boosting": gradient_boosting,
}
