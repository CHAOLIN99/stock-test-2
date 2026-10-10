"""Gaussian HMM on [log return, volume change]: regime-conditional price forecast and regime labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from config import TRAIN_END

REGIME_NAMES = ["bull", "bear", "volatile", "sideways"]


def _hmm_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    c, v = ohlcv["Close"].astype(float), ohlcv["Volume"].astype(float)
    X = pd.concat([np.log(c / c.shift(1)), v.pct_change()], axis=1, keys=["lr", "vc"])
    return X.replace([np.inf, -np.inf], np.nan).dropna()


def _fit_hmm(X: np.ndarray):
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    hmm = GaussianHMM(n_components=4, covariance_type="diag", n_iter=200, random_state=42)
    hmm.fit(Xs)
    return sc, hmm, hmm.predict(Xs)


def fit_predict_hmm_price(train_ohlcv: pd.DataFrame, test_ohlcv: pd.DataFrame) -> pd.Series:
    """
    Recursive forecast: each test day, classify the regime from the previous forecast return and the
    observed volume change, then step the price forward by that regime's mean training log return.
    """
    X = _hmm_features(train_ohlcv)
    if len(X) < 100:
        return pd.Series(index=test_ohlcv.index, dtype=float)
    try:
        sc, hmm, states = _fit_hmm(X.values)
    except Exception:
        return pd.Series(index=test_ohlcv.index, dtype=float)

    lr = X["lr"].values
    regime_mu = {s: float(lr[states == s].mean()) if np.any(states == s) else float(lr.mean()) for s in range(4)}

    volume = pd.concat([train_ohlcv, test_ohlcv]).sort_index()["Volume"].astype(float)
    vol_change = volume.pct_change().fillna(0.0).replace([np.inf, -np.inf], 0.0)
    close = train_ohlcv["Close"].astype(float)
    price, prev_price = float(close.iloc[-1]), float(close.iloc[-2])

    preds = []
    for dt in test_ohlcv.index:
        log_ret = np.log(price / prev_price) if prev_price > 0 else 0.0
        state = int(hmm.predict(sc.transform([[log_ret, vol_change.loc[dt]]]))[0])
        prev_price, price = price, price * float(np.exp(regime_mu[state]))
        preds.append(price)
    return pd.Series(preds, index=test_ohlcv.index)


def label_regimes(ohlcv: pd.DataFrame, dates: pd.DatetimeIndex) -> tuple[list[str], list[int | None]]:
    """
    Fit the HMM on the training period and label `dates`. States are named by their training
    log-return stats: highest mean = bull, lowest = bear, higher-variance remaining = volatile,
    last = sideways.
    """
    X = _hmm_features(ohlcv)
    train_X, test_X = X[X.index <= TRAIN_END], X.reindex(dates).dropna()
    if len(train_X) <= 120 or len(test_X) <= 20:
        return REGIME_NAMES, [None] * len(dates)
    try:
        sc, hmm, states = _fit_hmm(train_X.values)
    except Exception:
        return REGIME_NAMES, [None] * len(dates)

    lr = train_X["lr"].values
    mu = np.array([lr[states == s].mean() if np.any(states == s) else 0.0 for s in range(4)])
    var = np.array([lr[states == s].var() if np.any(states == s) else 0.0 for s in range(4)])
    bull, bear = int(np.argmax(mu)), int(np.argmin(mu))
    rest = [s for s in range(4) if s not in (bull, bear)]
    volatile = max(rest, key=lambda s: var[s])
    sideways = next(s for s in rest if s != volatile)
    mapping = {bull: 0, bear: 1, volatile: 2, sideways: 3}

    test_states = hmm.predict(sc.transform(test_X.values))
    by_date = {d: mapping[int(s)] for d, s in zip(test_X.index, test_states)}
    return REGIME_NAMES, [by_date.get(d) for d in dates]
