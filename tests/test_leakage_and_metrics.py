import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from agritrial_ml.evaluation import LeakageError, grouped_predictions, metrics
from agritrial_ml.models import TopKSpearman, make_model


def test_metrics_match_sklearn():
    rng = np.random.default_rng(0); y = rng.normal(size=30); p = y + rng.normal(0, .5, 30)
    m = metrics(y, p)
    assert m["R2"] == pytest.approx(r2_score(y, p)) and m["MAE"] == pytest.approx(mean_absolute_error(y, p))


def test_loo_mean_baseline_closed_form():
    # leave-one-out mean predictor: residual = (y_i - ybar) * n/(n-1)  =>  R2 = 1 - (n/(n-1))^2
    y = np.arange(10, dtype=float); g = np.arange(10).astype(str)
    p = grouped_predictions(make_model({"type": "baseline"}), np.zeros((10, 1)), y, g)
    assert metrics(y, p)["R2"] == pytest.approx(1 - (10 / 9) ** 2)


class LeakySplitter:
    def split(self, X, y, groups):
        idx = np.arange(len(y))
        yield idx, idx[:2]


def test_group_overlap_is_refused():
    with pytest.raises(LeakageError):
        grouped_predictions(make_model({"type": "ridge"}), np.ones((6, 1)), np.arange(6.), np.array(list("aabbcc")), LeakySplitter())


class Spy(BaseEstimator, TransformerMixin):
    seen = []
    def fit(self, X, y=None):
        Spy.seen.append(len(X)); return self
    def transform(self, X):
        return X


def test_preprocessing_fitted_inside_folds():
    Spy.seen = []
    X = np.random.default_rng(1).normal(size=(12, 2)); y = X[:, 0]; g = np.repeat(list("abcd"), 3)
    grouped_predictions(Pipeline([("spy", Spy()), ("m", LinearRegression())]), X, y, g)
    assert Spy.seen == [9, 9, 9, 9]          # never the 12 rows of the full data


def test_topk_selection_uses_training_rows_only():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(20, 4)); y = X[:, 2] + rng.normal(0, .1, 20)
    sel = TopKSpearman(k=1).fit(X[:10], y[:10]).selected_
    assert list(sel) == [2]
    y2 = y.copy(); y2[10:] = X[10:, 0] * 10      # change only the held-out rows
    assert list(TopKSpearman(k=1).fit(X[:10], y2[:10]).selected_) == [2]
