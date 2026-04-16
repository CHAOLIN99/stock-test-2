"""Metrics, bootstrap CIs, financial stats vs benchmark."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from config import BOOTSTRAP_N, TRANSACTION_COST


def rmse_over_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.mean(np.abs(y_true))
    if denom <= 0:
        return float("nan")
    return float(np.sqrt(mean_squared_error(y_true, y_pred)) / denom)


def directional_accuracy(y_true_ret: np.ndarray, y_pred_ret: np.ndarray) -> float:
    """Sign agreement on returns."""
    t = np.sign(y_true_ret)
    p = np.sign(y_pred_ret)
    mask = (t != 0) & (p != 0)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(t[mask] == p[mask]))


def information_coefficient_series(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 5:
        return float("nan")
    r, _ = spearmanr(y_true, y_pred, nan_policy="omit")
    return float(r) if np.isfinite(r) else float("nan")


def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    stat_fn,
    n_boot: int = BOOTSTRAP_N,
    seed: int = 42,
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats.append(stat_fn(y_true[idx], y_pred[idx]))
    stats = np.array(stats, dtype=float)
    return float(np.nanmean(stats)), float(np.nanpercentile(stats, 2.5)), float(np.nanpercentile(stats, 97.5))


def strategy_returns_from_prices(
    pred_prices: np.ndarray,
    actual_prices: np.ndarray,
) -> np.ndarray:
    """Long day i when pred[i] > actual[i-1] (predicted up); pay round-trip cost on position changes."""
    pred_prices = np.asarray(pred_prices, dtype=float).ravel()
    actual_prices = np.asarray(actual_prices, dtype=float).ravel()
    n = len(pred_prices)
    rets = np.zeros(n)
    prev_pos = 0.0
    for i in range(1, n):
        sig = 1.0 if pred_prices[i] > actual_prices[i - 1] else 0.0
        day_r = (actual_prices[i] - actual_prices[i - 1]) / actual_prices[i - 1]
        cost = TRANSACTION_COST * abs(sig - prev_pos)
        rets[i] = sig * day_r - cost
        prev_pos = sig
    return rets


def annualized_sharpe(daily_rets: np.ndarray, periods: int = 252) -> float:
    mu = np.nanmean(daily_rets)
    sd = np.nanstd(daily_rets, ddof=1)
    if sd <= 0:
        return float("nan")
    return float(np.sqrt(periods) * mu / sd)


def max_drawdown(equity_curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    return float(np.min(dd))


def calmar_ratio(daily_rets: np.ndarray, equity: np.ndarray) -> float:
    ann_ret = float(np.mean(daily_rets) * 252)
    mdd = max_drawdown(equity)
    if mdd >= -1e-9:
        return float("nan")
    return ann_ret / abs(mdd)


def classification_5d(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = (y_true[mask] > 0).astype(int)
    yp = (y_pred[mask] > 0).astype(int)
    if len(yt) < 5:
        return {
            "acc": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "auc": float("nan"),
        }
    try:
        auc = roc_auc_score(yt, y_pred[mask]) if len(np.unique(yt)) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    return {
        "acc": float(accuracy_score(yt, yp)),
        "precision": float(precision_score(yt, yp, zero_division=0)),
        "recall": float(recall_score(yt, yp, zero_division=0)),
        "f1": float(f1_score(yt, yp, zero_division=0)),
        "auc": float(auc) if np.isfinite(auc) else float("nan"),
    }


def compute_icir(ic_series: np.ndarray) -> float:
    ic_series = np.asarray(ic_series, dtype=float)
    ic_series = ic_series[np.isfinite(ic_series)]
    if len(ic_series) < 2:
        return float("nan")
    m = np.mean(ic_series)
    s = np.std(ic_series, ddof=1)
    if s <= 0:
        return float("nan")
    return float(m / s)


def rolling_ic(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    out = []
    for i in range(window, len(y_true)):
        r, _ = spearmanr(y_true[i - window : i], y_pred[i - window : i])
        out.append(r)
    return np.array(out, dtype=float)


def full_metrics_block(
    y_price_true: np.ndarray,
    y_price_pred: np.ndarray,
    y_ret_true: Optional[np.ndarray] = None,
    y_ret_pred: Optional[np.ndarray] = None,
    y_class_true: Optional[np.ndarray] = None,
    y_class_pred: Optional[np.ndarray] = None,
    bench_rets: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    y_price_true = np.asarray(y_price_true, dtype=float).ravel()
    y_price_pred = np.asarray(y_price_pred, dtype=float).ravel()
    m: Dict[str, float] = {}
    m["rmse_mean"] = rmse_over_mean(y_price_true, y_price_pred)
    m["mae"] = float(mean_absolute_error(y_price_true, y_price_pred))
    m["rmse_dollar"] = float(np.sqrt(mean_squared_error(y_price_true, y_price_pred)))
    m["r2"] = float(r2_score(y_price_true, y_price_pred))

    if y_ret_true is not None and y_ret_pred is not None:
        yt = np.asarray(y_ret_true, dtype=float).ravel()
        yp = np.asarray(y_ret_pred, dtype=float).ravel()
        m["dir_acc"] = directional_accuracy(yt, yp)
        m["ic"] = information_coefficient_series(yt, yp)
        ric = rolling_ic(yt, yp, window=min(20, max(5, len(yt) // 5)))
        m["icir"] = compute_icir(ric)
    else:
        m["dir_acc"] = float("nan")
        m["ic"] = float("nan")
        m["icir"] = float("nan")

    if y_class_true is not None and y_class_pred is not None:
        cls = classification_5d(y_class_true, y_class_pred)
        m.update({f"cls_{k}": v for k, v in cls.items()})
    else:
        for k in ("acc", "precision", "recall", "f1", "auc"):
            m[f"cls_{k}"] = float("nan")

    rets = strategy_returns_from_prices(y_price_pred, y_price_true)
    eq = np.cumprod(1.0 + rets)
    m["ann_return"] = float(np.mean(rets) * 252)
    m["sharpe"] = annualized_sharpe(rets)
    m["max_dd"] = max_drawdown(eq)
    m["calmar"] = calmar_ratio(rets, eq)

    if bench_rets is not None:
        br = np.asarray(bench_rets, dtype=float)
        m["sharpe_bench"] = annualized_sharpe(br)
        xs = rets[: len(br)] - br[: len(rets)]
        m["sharpe_vs_bench"] = annualized_sharpe(xs)
    else:
        m["sharpe_bench"] = float("nan")
        m["sharpe_vs_bench"] = float("nan")

    return m


def _to_py(x: Any) -> Any:
    """Cast numpy/pandas scalars to JSON-safe Python types."""
    if x is None:
        return None
    if isinstance(x, (np.floating,)):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, float):
        return x if np.isfinite(x) else None
    if isinstance(x, (str, int, bool)):
        return x
    if isinstance(x, (pd.Timestamp,)):
        return str(x.date())
    return x


def _atomic_write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    os.replace(tmp, path)


def strategy_log_returns_from_prices(pred_prices: np.ndarray, actual_next: np.ndarray) -> np.ndarray:
    """
    Daily strategy *log* returns derived from the existing price-based signal:
    long when pred[t] > actual[t-1]. Uses the same transaction cost model as evaluate.py,
    then converts net simple returns to log returns via log1p.
    """
    pred_prices = np.asarray(pred_prices, dtype=float).ravel()
    actual_next = np.asarray(actual_next, dtype=float).ravel()
    simple = strategy_returns_from_prices(pred_prices, actual_next)
    out = np.full_like(simple, np.nan, dtype=float)
    for i, r in enumerate(simple):
        if not np.isfinite(r) or r <= -0.999999:
            out[i] = np.nan
        else:
            out[i] = float(np.log1p(r))
    return out


def feature_importance_from_model(model: Any, feature_cols: list[str], top_k: int = 20) -> list[dict[str, Any]]:
    """
    Best-effort feature importance:
    - sklearn trees: feature_importances_
    - linear models: coef_
    Returned as [{"feature": str, "importance": float}, ...]
    """
    if model is None or not feature_cols:
        return []
    imp = None
    if hasattr(model, "feature_importances_"):
        try:
            imp = np.asarray(getattr(model, "feature_importances_"), dtype=float).ravel()
        except Exception:
            imp = None
    elif hasattr(model, "coef_"):
        try:
            coef = np.asarray(getattr(model, "coef_"), dtype=float)
            imp = np.abs(coef).ravel()
        except Exception:
            imp = None
    if imp is None or len(imp) != len(feature_cols):
        return []
    imp = np.where(np.isfinite(imp), imp, 0.0)
    order = np.argsort(-imp)[:top_k]
    out = []
    for i in order:
        v = float(imp[int(i)])
        if v <= 0:
            continue
        out.append({"feature": str(feature_cols[int(i)]), "importance": float(v)})
    return out


def write_dashboard_data(
    *,
    tickers: list[str],
    per_ticker: dict[str, dict[str, Any]],
    out_dir: str | Path = "results",
    generated_at: str,
) -> None:
    """
    Write frontend dashboard payloads. This is intentionally file-system oriented
    and uses atomic replaces to avoid partial writes on crashes.

    Expected per_ticker[ticker] shape matches charts_data.json schema.
    """
    out_dir = Path(out_dir)
    charts_path = out_dir / "charts_data.json"
    payload = {
        "tickers": tickers,
        "generated_at": generated_at,
        "data": {},
    }
    for t in tickers:
        td = per_ticker.get(t) or {}
        payload["data"][t] = td

    # also provide split outputs for debugging/optional use
    fi_payload = {t: (per_ticker.get(t) or {}).get("feature_importance", {}) for t in tickers}
    reg_payload = {
        t: {
            "dates": (per_ticker.get(t) or {}).get("dates", []),
            "regime": (per_ticker.get(t) or {}).get("regime", []),
            "regime_names": (per_ticker.get(t) or {}).get("regime_names", []),
        }
        for t in tickers
    }

    # Ensure JSON-safe types
    def normalize(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): normalize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [normalize(v) for v in obj]
        return _to_py(obj)

    _atomic_write_json(charts_path, normalize(payload))
    _atomic_write_json(out_dir / "feature_importance.json", normalize(fi_payload))
    _atomic_write_json(out_dir / "regime_labels.json", normalize(reg_payload))
