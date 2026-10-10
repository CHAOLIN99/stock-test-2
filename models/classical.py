"""Multivariate classical ML with walk-forward TimeSeriesSplit."""

from __future__ import annotations

import sys
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor

from config import TIER3_NAMES

warnings.filterwarnings("ignore")

# XGBoost wheels can segfault on Python 3.14+ with some real matrices (C extension issue).
# Use sklearn's histogram booster as a drop-in fallback with similar inductive bias.
USE_HGB_AS_XGB = sys.version_info >= (3, 14)

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None  # type: ignore
    USE_HGB_AS_XGB = True

try:
    from pyearth import Earth

    HAS_MARS = True
except ImportError:
    HAS_MARS = False


def _ts_cv_predict(
    X: np.ndarray,
    y: np.ndarray,
    model_factory,
    n_splits: int = 5,
    max_train_rows: Optional[int] = None,
) -> np.ndarray:
    """Out-of-fold predictions on train for stacking."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof = np.zeros(len(y))
    for train_idx, val_idx in tscv.split(X):
        tr_x, tr_y = X[train_idx], y[train_idx]
        if max_train_rows:
            tr_x, tr_y = tr_x[-max_train_rows:], tr_y[-max_train_rows:]
        m = model_factory()
        m.fit(tr_x, tr_y)
        oof[val_idx] = m.predict(X[val_idx])
    return oof


def train_sklearn_with_oof(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_splits: int,
    svr_max_rows: int,
    rf_estimators: int,
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """Returns fitted model, test predictions, oof train predictions."""

    def factory():
        if name == "Ridge":
            return Ridge(alpha=1.0, random_state=42)
        if name == "MARS":
            if not HAS_MARS:
                return Ridge(alpha=1.0, random_state=42)
            return Earth(max_degree=1, penalty=3.0)
        if name == "RandomForest":
            return RandomForestRegressor(
                n_estimators=rf_estimators,
                max_depth=8,
                min_samples_leaf=50,
                n_jobs=1,
                random_state=42,
            )
        if name == "SVR":
            return SVR(kernel="rbf", C=100.0, epsilon=0.01, gamma="scale")
        if name == "MLP":
            return MLPRegressor(
                hidden_layer_sizes=(64, 32),
                early_stopping=True,
                random_state=42,
                max_iter=500,
            )
        if name == "XGBoost":
            if USE_HGB_AS_XGB:
                return HistGradientBoostingRegressor(
                    max_iter=500,
                    max_depth=4,
                    learning_rate=0.05,
                    random_state=42,
                )
            return XGBRegressor(
                n_estimators=500,
                max_depth=4,
                tree_method="hist",
                device="cpu",
                random_state=42,
                n_jobs=1,
            )
        if name == "LightGBM":
            return LGBMRegressor(
                n_estimators=500,
                max_depth=4,
                num_leaves=31,
                random_state=42,
                n_jobs=1,
                verbose=-1,
            )
        raise ValueError(name)

    # SVR is O(n^2): fit only on the most recent rows
    cap = svr_max_rows if name == "SVR" else None
    oof = _ts_cv_predict(X_train, y_train, factory, n_splits=n_splits, max_train_rows=cap)

    model = factory()
    model.fit(X_train[-cap:] if cap else X_train, y_train[-cap:] if cap else y_train)
    return model, model.predict(X_test), oof


def fit_all_tier3(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_splits: int,
    svr_max_rows: int,
    rf_estimators: int,
) -> Dict[str, Tuple[Any, np.ndarray, np.ndarray]]:
    return {
        name: train_sklearn_with_oof(name, X_train, y_train, X_test, n_splits, svr_max_rows, rf_estimators)
        for name in TIER3_NAMES
    }
