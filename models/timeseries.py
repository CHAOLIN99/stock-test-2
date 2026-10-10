"""Univariate price forecasts (Buy & Hold, SMA cross, Holt-Winters, ARIMA) and stationarity tests."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")


def adf_kpss_summary(series: pd.Series) -> dict:
    s = series.dropna()
    out = {"adf_p": np.nan, "kpss_p": np.nan}
    if len(s) < 30:
        return out
    try:
        out["adf_p"] = float(adfuller(s, autolag="AIC")[1])
    except Exception:
        pass
    try:
        out["kpss_p"] = float(kpss(s, regression="c", nlags="auto")[1])
    except Exception:
        pass
    return out


def fit_predict_holt_winters(train_close: pd.Series, test_index: pd.DatetimeIndex) -> pd.Series:
    """Additive-trend Holt-Winters with 5-day seasonality; falls back to no seasonality, then last price."""
    y = train_close.astype(float).values
    if len(y) < 50:
        return pd.Series(index=test_index, dtype=float)
    for seasonal, periods in (("add", 5), (None, None)):
        try:
            hw = ExponentialSmoothing(
                y,
                trend="add",
                seasonal=seasonal,
                seasonal_periods=periods,
                initialization_method="estimated",
            ).fit(optimized=True)
            return pd.Series(hw.forecast(steps=len(test_index)), index=test_index)
        except Exception:
            continue
    return pd.Series(train_close.iloc[-1], index=test_index)


def fit_predict_arima(train_close: pd.Series, test_index: pd.DatetimeIndex) -> pd.Series:
    """auto_arima on daily log returns, compounded forward from the last training price."""
    c = train_close.astype(float)
    log_ret = np.log(c / c.shift(1)).dropna()
    if len(log_ret) < 100:
        return pd.Series(index=test_index, dtype=float)
    try:
        model = auto_arima(
            log_ret,
            seasonal=False,
            information_criterion="aic",
            suppress_warnings=True,
            error_action="ignore",
            max_p=5,
            max_q=5,
            max_order=10,
        )
        fc_ret = np.asarray(model.predict(n_periods=len(test_index)), dtype=float)
    except Exception:
        return pd.Series(index=test_index, dtype=float)
    return pd.Series(float(c.iloc[-1]) * np.exp(np.cumsum(fc_ret)), index=test_index)


def buy_hold_forecast(train_close: pd.Series, test_index: pd.DatetimeIndex) -> pd.Series:
    """Compound the mean training log return forward from the last training price."""
    mu = float(np.log(train_close / train_close.shift(1)).dropna().mean())
    steps = np.arange(1, len(test_index) + 1)
    return pd.Series(float(train_close.iloc[-1]) * np.exp(mu * steps), index=test_index)


def sma_cross_forecast(train_close: pd.Series, test_index: pd.DatetimeIndex) -> pd.Series:
    """
    Recursive forecast: each step drifts at the training mean return while SMA50 > SMA200,
    otherwise at -0.5x that magnitude. SMAs are recomputed on the forecast path.
    """
    mu_long = float(np.log(train_close / train_close.shift(1)).dropna().mean())
    mu_short = -abs(mu_long) * 0.5
    prices = list(train_close.astype(float).values)
    preds = []
    for _ in test_index:
        recent = np.array(prices[-200:])
        mu = mu_long if len(recent) < 200 or recent[-50:].mean() > recent.mean() else mu_short
        prices.append(prices[-1] * float(np.exp(mu)))
        preds.append(prices[-1])
    return pd.Series(preds, index=test_index)
