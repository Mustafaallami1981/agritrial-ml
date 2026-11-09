"""Grouped cross-validation, permutation tests and validation-design comparison (FR-6)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, LeaveOneGroupOut


class LeakageError(RuntimeError):
    pass


def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = y - p
    return {"R2": float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()),
            "RMSE": float(np.sqrt((e ** 2).mean())), "MAE": float(np.abs(e).mean())}


def grouped_predictions(model, X, y, groups, splitter=None):
    """Out-of-fold predictions; every fold is checked so that no group is in both train and test."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    splitter = splitter or LeaveOneGroupOut()
    pred = np.full(len(y), np.nan)
    for tr, te in splitter.split(X, y, groups):
        if set(groups[tr]) & set(groups[te]):
            raise LeakageError("a treatment appears in both training and test rows")
        m = clone(model).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    if np.isnan(pred).any():
        raise RuntimeError("some rows were never predicted")
    return pred


def random_predictions(model, X, y, n_splits=5, seed=0):
    X, y = np.asarray(X, float), np.asarray(y, float)
    pred = np.empty(len(y))
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        pred[te] = clone(model).fit(X[tr], y[tr]).predict(X[te])
    return pred


def y_permutation(model, X, y, groups, n_perm, seed):
    rng = np.random.default_rng(seed)
    obs = metrics(y, grouped_predictions(model, X, y, groups))["R2"]
    hits = 0
    for _ in range(n_perm):
        yp = y[rng.permutation(len(y))]
        hits += metrics(yp, grouped_predictions(model, X, yp, groups))["R2"] >= obs
    return obs, (hits + 1) / (n_perm + 1)


def block_permutation(model, X: pd.DataFrame, y, groups, fixed_cols, permuted_cols, n_perm, seed):
    """Permute the rows of `permuted_cols` across units, keeping `fixed_cols` and y in place."""
    rng = np.random.default_rng(seed)
    obs = metrics(y, grouped_predictions(model, X.to_numpy(float), y, groups))["R2"]
    F, P = X[fixed_cols].to_numpy(float), X[permuted_cols].to_numpy(float)
    hits = 0
    for _ in range(n_perm):
        Xp = np.hstack([F, P[rng.permutation(len(y))]])
        hits += metrics(y, grouped_predictions(model, Xp, y, groups))["R2"] >= obs
    return obs, (hits + 1) / (n_perm + 1)
