"""
Live next-day forecasts from the saved Tier-3 models + stacking ensemble.

Rebuilds the exact training feature frame (data_pipeline.build_supervised_frame)
on freshly downloaded prices, then applies the saved scaler and models to the
latest row. Backtest stats come from results/charts_data.json.
"""

from __future__ import annotations

import json
import threading
import time
import warnings
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from config import RESULTS_DIR, SAVED_MODELS_DIR, TICKERS, TIER3_NAMES
from data_pipeline import add_technical_features, load_fred_features, merge_macro

PRICE_TTL_S = 15 * 60
MACRO_TTL_S = 6 * 60 * 60
SIGNAL_THRESHOLD = 0.0025
DISCLAIMER = "Educational model output, not financial advice."

_lock = threading.Lock()
_cache: Dict[str, tuple[float, Any]] = {}


def _cached(key: str, ttl: float, fn):
    now = time.time()
    with _lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
    val = fn()
    with _lock:
        _cache[key] = (now, val)
    return val


def _download_prices(ticker: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=TICKERS[ticker],
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df.empty:
        raise RuntimeError(f"No price data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df = df.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]]
    df.index.name = "Date"
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.apply(pd.to_numeric, errors="coerce").ffill(limit=2)
    return df.dropna(subset=["Close"])


def _macro() -> pd.DataFrame:
    end = (pd.Timestamp.today() + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    return _cached("macro", MACRO_TTL_S, lambda: load_fred_features(min(TICKERS.values()), end))


def _models(ticker: str) -> Dict[str, Any]:
    def load():
        out = {"scaler": joblib.load(SAVED_MODELS_DIR / f"scaler_{ticker}.pkl")}
        out["stacking"] = joblib.load(SAVED_MODELS_DIR / f"stacking_{ticker}.pkl")
        out["tier3"] = {n: joblib.load(SAVED_MODELS_DIR / f"{ticker}_{n}.pkl") for n in TIER3_NAMES}
        return out

    return _cached(f"models:{ticker}", float("inf"), load)


def _backtest(ticker: str) -> Dict[str, Any]:
    def load():
        path = RESULTS_DIR / "charts_data.json"
        if not path.exists():
            return {}
        with open(path) as f:
            return json.load(f).get("data", {})

    data = _cached("charts_data", 60 * 60, load).get(ticker)
    if not data:
        return {"metrics": {}}
    metrics = data.get("metrics", {})
    best = min(metrics, key=lambda n: metrics[n].get("rmse_mean_test", float("inf")), default=None)
    regime = None
    labels, names = data.get("regime") or [], data.get("regime_names") or []
    if labels and names and labels[-1] is not None:
        regime = {"label": names[int(labels[-1])], "as_of": data["dates"][-1]}
    return {
        "metrics": metrics,
        "stacking": metrics.get("StackingEnsemble"),
        "best_model": best,
        "best_metrics": metrics.get(best) if best else None,
        "regime": regime,
        "top_features": [f["feature"] for f in (data.get("feature_importance", {}).get("LightGBM") or [])[:5]],
        "test_start": data["dates"][0] if data.get("dates") else None,
    }


def _signal(pred_ret: float, agreement: float) -> str:
    if pred_ret > SIGNAL_THRESHOLD and agreement >= 0.625:
        return "BUY"
    if pred_ret < -SIGNAL_THRESHOLD and agreement <= 0.375:
        return "SELL"
    return "HOLD"


def predict_next(ticker: str) -> Dict[str, Any]:
    ticker = ticker.upper()
    if ticker not in TICKERS:
        raise KeyError(ticker)

    ohlcv = _cached(f"px:{ticker}", PRICE_TTL_S, lambda: _download_prices(ticker))
    models = _models(ticker)
    scaler = models["scaler"]

    frame = merge_macro(add_technical_features(ohlcv), _macro())
    feature_cols = list(frame.columns)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)
    if frame.empty:
        raise RuntimeError(f"Not enough data to build features for {ticker}")
    if len(feature_cols) != scaler.n_features_in_:
        raise RuntimeError(f"Feature mismatch: built {len(feature_cols)}, scaler expects {scaler.n_features_in_}")

    row = frame.iloc[[-1]]
    x = scaler.transform(row[feature_cols].values.astype(np.float64))
    last_close = float(row["Close"].iloc[0])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        per_model = {n: float(m.predict(x)[0]) for n, m in models["tier3"].items()}
    stack_ret = float(models["stacking"].predict(np.array([[per_model[n] for n in TIER3_NAMES]]))[0])
    all_rets = list(per_model.values()) + [stack_ret]
    agreement = float(np.mean([r > 0 for r in all_rets]))

    backtest = _backtest(ticker)
    metrics = backtest.pop("metrics")
    return {
        "ticker": ticker,
        "as_of": row.index[0].strftime("%Y-%m-%d"),
        "last_close": last_close,
        "predicted_close": last_close * float(np.exp(stack_ret)),
        "predicted_return": float(np.expm1(stack_ret)),
        "signal": _signal(stack_ret, agreement),
        "models_bullish": agreement,
        "models": {
            n: {
                "predicted_return": float(np.expm1(r)),
                "predicted_close": last_close * float(np.exp(r)),
                "backtest_dir_acc": (metrics.get(n) or {}).get("dir_acc"),
                "backtest_rmse": (metrics.get(n) or {}).get("rmse_mean_test"),
            }
            for n, r in {**per_model, "StackingEnsemble": stack_ret}.items()
        },
        "backtest": backtest,
        "generated_at": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "disclaimer": DISCLAIMER,
    }
