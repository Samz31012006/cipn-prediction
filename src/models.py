"""Model definitions: one logistic regression, one random forest.

Hyperparameters are fixed up front (no tuning) so the cross-validation
estimates are not optimistically biased by selection on the same folds.
"""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data import FEATURES


def _encoder() -> ColumnTransformer:
    return ColumnTransformer(
        [("onehot", OneHotEncoder(handle_unknown="ignore"), FEATURES)]
    )


def logistic_regression() -> Pipeline:
    return Pipeline([
        ("prep", _encoder()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000)),
    ])


def random_forest() -> Pipeline:
    return Pipeline([
        ("prep", _encoder()),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=4, min_samples_leaf=5,
            random_state=0, n_jobs=-1,
        )),
    ])


MODELS = {
    "Logistic regression": logistic_regression,
    "Random forest": random_forest,
}
