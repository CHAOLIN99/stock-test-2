"""Voting and stacking ensembles over Tier-3 models."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


def validation_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def pick_top3_weights(
    oof_preds: Dict[str, np.ndarray],
    y_train: np.ndarray,
) -> Tuple[List[str], np.ndarray]:
    """Select top-3 Tier3 models by OOF RMSE; weights inverse RMSE (normalized)."""
    scores = []
    for name, pred in oof_preds.items():
        m = validation_rmse(y_train, pred)
        scores.append((name, m))
    scores.sort(key=lambda x: x[1])
    top = scores[:3]
    inv = np.array([1.0 / (s[1] + 1e-12) for s in top], dtype=float)
    w = inv / inv.sum()
    return [s[0] for s in top], w


def voting_predict(
    test_preds: Dict[str, np.ndarray],
    top_names: List[str],
    weights: np.ndarray,
) -> np.ndarray:
    stack = np.zeros_like(next(iter(test_preds.values())))
    for n, w in zip(top_names, weights):
        stack = stack + w * test_preds[n]
    return stack


def stacking_train_predict(
    X_oof: np.ndarray,
    y_train: np.ndarray,
    X_test_stack: np.ndarray,
) -> Tuple[Ridge, np.ndarray]:
    meta = Ridge(alpha=1.0, random_state=42)
    meta.fit(X_oof, y_train)
    pred = meta.predict(X_test_stack)
    return meta, pred


def build_oof_matrix(
    oof_dict: Dict[str, np.ndarray],
    names: List[str],
) -> np.ndarray:
    return np.column_stack([oof_dict[n] for n in names])


def ts_cv_rmse_for_weights(
    y: np.ndarray,
    preds: Dict[str, np.ndarray],
    n_splits: int,
) -> Dict[str, float]:
    """Mean RMSE per model across CV folds (optional diagnostic)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    out: Dict[str, float] = {}
    for name, p in preds.items():
        errs = []
        for tr, va in tscv.split(y):
            errs.append(validation_rmse(y[va], p[va]))
        out[name] = float(np.mean(errs))
    return out
