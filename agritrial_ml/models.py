"""Model registry; every model is a scikit-learn Pipeline so preprocessing is fitted inside folds (FR-5)."""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

_ESTIMATORS = {
    "baseline": lambda p: DummyRegressor(strategy="mean"),
    "linear": lambda p: LinearRegression(**p),
    "ridge": lambda p: Ridge(**p),
    "svr": lambda p: SVR(**p),
    "random_forest": lambda p: RandomForestRegressor(**p),
}


class TopKSpearman(BaseEstimator, TransformerMixin):
    """Keeps the k columns with the largest |Spearman rho| with y, computed on the data passed to fit only."""

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X, float)
        rho = np.array([abs(spearmanr(X[:, j], y)[0]) if np.std(X[:, j]) > 0 else 0.0 for j in range(X.shape[1])])
        rho = np.nan_to_num(rho)
        self.selected_ = np.argsort(-rho, kind="stable")[: self.k]
        return self

    def transform(self, X):
        return np.asarray(X, float)[:, self.selected_]


def make_model(spec: dict, top_k: int | None = None) -> Pipeline:
    kind = spec["type"]
    if kind not in _ESTIMATORS:
        raise ValueError(f"unknown model type {kind!r}; known: {sorted(_ESTIMATORS)}")
    steps = []
    if top_k is not None:
        steps.append(("select", TopKSpearman(k=top_k)))
    if spec.get("scale", kind in ("linear", "ridge", "svr")):
        steps.append(("scale", StandardScaler()))
    steps.append(("model", _ESTIMATORS[kind](dict(spec.get("params", {})))))
    return Pipeline(steps)
