"""Holt-Winters, ARIMA, stationarity tests."""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")


def adf_kpss_summary(series: pd.Series) -> dict:
    s = series.dropna()
    out: dict = {}
    if len(s) < 30:
        return {"adf_p": np.nan, "kpss_p": np.nan}
    try:
        out["adf_p"] = float(adfuller(s, autolag="AIC")[1])
    except Exception:
        out["adf_p"] = np.nan
    try:
        out["kpss_p"] = float(kpss(s, regression="c", nlags="auto")[1])
    except Exception:
        out["kpss_p"] = np.nan
    return out


def fit_predict_holt_winters(
    train_close: pd.Series,
    test_index: pd.DatetimeIndex,
) -> pd.Series:
    y = train_close.astype(float).values
    if len(y) < 50:
        return pd.Series(index=test_index, dtype=float)
    for seasonal in ("add", None):
        try:
            if seasonal:
                hw = ExponentialSmoothing(
                    y,
                    seasonal_periods=5,
                    trend="add",
                    seasonal="add",
                    initialization_method="estimated",
                ).fit(optimized=True)
            else:
                hw = ExponentialSmoothing(
                    y,
                    trend="add",
                    seasonal=None,
                    initialization_method="estimated",
                ).fit(optimized=True)
            fc = hw.forecast(steps=len(test_index))
            return pd.Series(fc, index=test_index)
        except Exception:
            continue
    return pd.Series(train_close.iloc[-1], index=test_index)


def fit_predict_arima(
    train_close: pd.Series,
    test_index: pd.DatetimeIndex,
) -> pd.Series:
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
    p0 = float(c.iloc[-1])
    prices = p0 * np.exp(np.cumsum(fc_ret))
    return pd.Series(prices, index=test_index)


def naive_forecast(train_close: pd.Series, test_close: pd.Series) -> pd.Series:
    full = pd.concat([train_close, test_close])
    return full.shift(1).loc[test_close.index]


def buy_hold_forecast(train_close: pd.Series, test_index: pd.DatetimeIndex) -> pd.Series:
    mu = float(np.log(train_close / train_close.shift(1)).dropna().mean())
    last = float(train_close.iloc[-1])
    out = []
    p = last
    for _ in range(len(test_index)):
        p = p * np.exp(mu)
        out.append(p)
    return pd.Series(out, index=test_index)


def sma_cross_forecast(
    train_close: pd.Series,
    test_index: pd.DatetimeIndex,
) -> pd.Series:
    c = pd.concat([train_close, pd.Series(index=test_index, dtype=float)])
    c = c.sort_index()
    mu_long = float(np.log(train_close / train_close.shift(1)).dropna().mean())
    mu_short = -abs(mu_long) * 0.5
    prices = list(train_close.astype(float).values)
    preds = []
    for dt in test_index:
        ser = pd.Series(prices[-250:], dtype=float)
        if len(ser) >= 200:
            s50 = ser.rolling(50).mean().iloc[-1]
            s200 = ser.rolling(200).mean().iloc[-1]
            mu = mu_long if s50 > s200 else mu_short
        else:
            mu = mu_long
        nxt = prices[-1] * float(np.exp(mu))
        prices.append(nxt)
        preds.append(nxt)
    return pd.Series(preds, index=test_index)
