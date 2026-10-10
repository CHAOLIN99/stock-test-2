"""Voting and stacking ensembles over Tier-3 models."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


def pick_top3_weights(oof_preds: Dict[str, np.ndarray], y_train: np.ndarray) -> Tuple[List[str], np.ndarray]:
    """Select the top-3 models by out-of-fold RMSE; weights are normalized inverse RMSE."""
    rmse = {name: np.sqrt(mean_squared_error(y_train, pred)) for name, pred in oof_preds.items()}
    top = sorted(rmse, key=rmse.get)[:3]
    inv = np.array([1.0 / (rmse[n] + 1e-12) for n in top])
    return top, inv / inv.sum()


def voting_predict(preds: Dict[str, np.ndarray], top_names: List[str], weights: np.ndarray) -> np.ndarray:
    return sum(w * preds[n] for n, w in zip(top_names, weights))


def stacking_train_predict(X_oof: np.ndarray, y_train: np.ndarray, X_test_stack: np.ndarray) -> Tuple[Ridge, np.ndarray]:
    meta = Ridge(alpha=1.0, random_state=42)
    meta.fit(X_oof, y_train)
    return meta, meta.predict(X_test_stack)


def build_oof_matrix(oof_dict: Dict[str, np.ndarray], names: List[str]) -> np.ndarray:
    return np.column_stack([oof_dict[n] for n in names])
