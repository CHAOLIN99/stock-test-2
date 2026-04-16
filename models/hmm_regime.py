"""Gaussian HMM on returns + volume change — regime-conditional mean return forecast."""

from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


def fit_predict_hmm_price(
    train_ohlcv: pd.DataFrame,
    test_ohlcv: pd.DataFrame,
) -> pd.Series:
    """
    Train HMM on [log_ret, vol_change]. Next-day price via regime-conditional
    mean log return estimated on training states.
    """
    c = train_ohlcv["Close"].astype(float)
    v = train_ohlcv["Volume"].astype(float)
    log_ret = np.log(c / c.shift(1))
    vol_ch = v.pct_change()
    Xdf = pd.concat([log_ret, vol_ch], axis=1, keys=["lr", "vc"]).dropna()
    Xdf = Xdf.replace([np.inf, -np.inf], np.nan).dropna()
    if len(Xdf) < 100:
        return pd.Series(index=test_ohlcv.index, dtype=float)

    sc = StandardScaler()
    Xs = sc.fit_transform(Xdf.values)
    try:
        hmm = GaussianHMM(
            n_components=4,
            covariance_type="diag",
            n_iter=200,
            random_state=42,
        )
        hmm.fit(Xs)
    except Exception:
        return pd.Series(index=test_ohlcv.index, dtype=float)

    states = hmm.predict(Xs)
    lr_vals = Xdf["lr"].values
    regime_mu: dict[int, float] = {}
    for s in range(4):
        mask = states == s
        regime_mu[s] = float(np.mean(lr_vals[mask])) if mask.sum() else float(np.mean(lr_vals))

    full = pd.concat([train_ohlcv, test_ohlcv]).sort_index()
    c_all = full["Close"].astype(float)
    v_all = full["Volume"].astype(float)

    price = float(c_all.iloc[len(train_ohlcv) - 1])
    prev_price = float(c_all.iloc[len(train_ohlcv) - 2]) if len(train_ohlcv) > 1 else price

    preds = []
    for dt in test_ohlcv.index:
        log_ret_t = np.log(price / prev_price) if prev_price > 0 else 0.0
        v_prev = float(v_all.shift(1).loc[dt]) if pd.notna(v_all.shift(1).loc[dt]) else float(v_all.loc[dt])
        v_cur = float(v_all.loc[dt])
        vol_ch_t = (v_cur - v_prev) / v_prev if v_prev > 0 else 0.0
        try:
            x_t = sc.transform(np.array([[log_ret_t, vol_ch_t]]))
            st = int(hmm.predict(x_t)[0])
        except Exception:
            st = 0
        mu = regime_mu.get(st, float(np.mean(lr_vals)))
        prev_price = price
        price = price * float(np.exp(mu))
        preds.append(price)
    return pd.Series(preds, index=test_ohlcv.index)
