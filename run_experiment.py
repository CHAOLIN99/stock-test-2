#!/usr/bin/env python3
"""
End-to-end training and evaluation for five tickers.
Run: python run_experiment.py
"""

from __future__ import annotations

import gc
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

from config import (
    BOOTSTRAP_N,
    CHARTS_DIR,
    N_SPLITS_CV,
    RESULTS_DIR,
    RF_ESTIMATORS,
    SAVED_MODELS_DIR,
    SVR_MAX_ROWS,
    TEST_START,
    TICKERS,
    TIER3_NAMES,
    TRAIN_END,
)
from sklearn.preprocessing import MinMaxScaler
from data_pipeline import (
    build_supervised_frame,
    download_yahoo,
    fit_save_scaler,
    load_fred_features,
    prepare_univariate_series,
    set_seeds,
    train_test_split_df,
    validate_price_df,
)
from evaluate import bootstrap_metric, full_metrics_block, rmse_over_mean
from models import classical as classical_mod
from models import deep_learning as dl_mod
from models import ensemble as ens_mod
from models import hmm_regime as hmm_mod
from models import timeseries as ts_mod

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_PLOT = True
except Exception:
    HAS_PLOT = False

try:
    import shap

    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.ffill(limit=2)
    return df


def log_returns_from_prices(close: pd.Series) -> pd.Series:
    return np.log(close.astype(float) / close.astype(float).shift(1))


def prices_from_log_pred(
    close_at_t: np.ndarray,
    pred_log_ret: np.ndarray,
) -> np.ndarray:
    return close_at_t * np.exp(pred_log_ret)


def train_dl_price_preds(
    model_name: str,
    train_close: pd.Series,
    full_close: pd.Series,
    train_len: int,
    lookback: int,
    price_scaler,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    train_scaled = price_scaler.transform(train_close.values.reshape(-1, 1)).ravel()
    full_scaled = price_scaler.transform(full_close.values.reshape(-1, 1)).ravel()
    pred_test_s, model = dl_mod.train_dl_model(
        model_name,
        train_scaled,
        full_scaled,
        train_len,
        lookback,
        device,
    )
    if model is None or len(pred_test_s) == 0:
        dl_mod.cleanup_dl(None, device)
        return np.array([]), np.array([])
    train_preds = []
    with torch.no_grad():
        model.eval()
        for i in range(lookback, train_len):
            seq = train_scaled[i - lookback : i].reshape(1, lookback, 1)
            xt = torch.from_numpy(seq.astype(np.float32)).to(device)
            ps = float(model(xt).cpu().numpy().ravel()[0])
            train_preds.append(ps)
    train_prices = price_scaler.inverse_transform(np.array(train_preds).reshape(-1, 1)).ravel()
    test_prices = price_scaler.inverse_transform(pred_test_s.reshape(-1, 1)).ravel()
    dl_mod.cleanup_dl(model, device)
    return train_prices, test_prices


def ml_chain_5d_class(
    model: Any,
    X_test: np.ndarray,
    scaler,
    test_index: pd.DatetimeIndex,
    close: pd.Series,
) -> np.ndarray:
    """5-step chain of 1d log-return predictions for direction vs close[t+5]-close[t]."""
    Xs = scaler.transform(X_test)
    preds = []
    for i in range(len(test_index) - 5):
        p = float(close.loc[test_index[i]])
        for k in range(5):
            r = float(model.predict(Xs[i + k : i + k + 1])[0])
            p = p * np.exp(r)
        preds.append(np.sign(p - float(close.loc[test_index[i]])))
    return np.array(preds)


def compute_metrics_row(
    ticker: str,
    model: str,
    source: str,
    actual_next: np.ndarray,
    pred_next: np.ndarray,
    close_at_t: np.ndarray,
    y_class: np.ndarray,
    cls_pred: Optional[np.ndarray],
    bench_rets: np.ndarray,
    train_rmse_mean: float,
) -> Dict[str, Any]:
    m = len(actual_next)
    actual_next = actual_next[:m]
    pred_next = pred_next[:m]
    close_at_t = close_at_t[:m]
    y_class = y_class[:m]

    ret_true = np.log(actual_next / close_at_t)
    ret_pred = np.log(pred_next / close_at_t)

    if cls_pred is None:
        cls_pred = np.sign(ret_pred)
    else:
        cls_pred = cls_pred[:m]
        if len(cls_pred) < m:
            cls_pred = np.pad(cls_pred, (0, m - len(cls_pred)), constant_values=np.nan)
        cls_pred = np.where(np.isfinite(cls_pred), cls_pred, np.sign(ret_pred))

    mets = full_metrics_block(
        actual_next,
        pred_next,
        y_ret_true=ret_true,
        y_ret_pred=ret_pred,
        y_class_true=y_class,
        y_class_pred=cls_pred,
        bench_rets=bench_rets[:m],
    )

    b_mean, b_low, b_high = bootstrap_metric(
        actual_next,
        pred_next,
        lambda a, b: rmse_over_mean(a, b),
        n_boot=BOOTSTRAP_N,
    )

    mets_out = {k: v for k, v in mets.items() if k != "rmse_mean"}
    return {
        "ticker": ticker,
        "model": model,
        "source": source,
        "rmse_mean_train": train_rmse_mean,
        "rmse_mean_test": mets["rmse_mean"],
        "bootstrap_rmse_mean": b_mean,
        "bootstrap_rmse_low": b_low,
        "bootstrap_rmse_high": b_high,
        **mets_out,
    }


def run_ticker(
    ticker: str,
    start_date: str,
    macro: pd.DataFrame,
    bench_close: pd.Series,
    device: torch.device,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    set_seeds()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    raw, source = download_yahoo(ticker, start_date=start_date)
    raw = _clean_ohlcv(raw)
    validate_price_df(raw)
    ohlcv = raw.copy()

    sup, feature_cols = build_supervised_frame(ticker, ohlcv, macro)
    train_df, test_df = train_test_split_df(sup, ticker)
    train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)
    test_df = test_df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)
    assert len(train_df) > 500 and len(test_df) > 20

    close = ohlcv["Close"].astype(float)

    X_train = train_df[feature_cols].values.astype(np.float64)
    X_test = test_df[feature_cols].values.astype(np.float64)
    y_train = train_df["y_reg"].values.astype(np.float64)
    y_test = test_df["y_reg"].values.astype(np.float64)
    y_class_train = train_df["y_class"].values
    y_class_test = test_df["y_class"].values

    scaler = fit_save_scaler(X_train, ticker)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    train_close = prepare_univariate_series(ohlcv[ohlcv.index <= TRAIN_END])
    test_close = prepare_univariate_series(ohlcv[ohlcv.index >= TEST_START])
    full_close = prepare_univariate_series(ohlcv)

    idx_test = test_df.index
    close_at_t = close.reindex(idx_test).values.astype(float)
    actual_next = close.shift(-1).reindex(idx_test).values.astype(float)
    bench_rets = log_returns_from_prices(bench_close.reindex(idx_test)).values

    rows: List[Dict[str, Any]] = []

    # --- Baselines ---
    def _align(s: pd.Series) -> np.ndarray:
        return s.reindex(idx_test).ffill().bfill().values.astype(float)

    pred_naive = _align(ts_mod.naive_forecast(train_close, test_close))
    pred_bh = _align(ts_mod.buy_hold_forecast(train_close, test_close.index))
    pred_sma = _align(ts_mod.sma_cross_forecast(train_close, test_close.index))

    # --- Holt-Winters, ARIMA ---
    pred_hw = _align(ts_mod.fit_predict_holt_winters(train_close, test_close.index))
    pred_arima = _align(ts_mod.fit_predict_arima(train_close, test_close.index))

    # --- Tier 3 ---
    tier3 = classical_mod.fit_all_tier3(
        X_train_s,
        y_train,
        X_test_s,
        N_SPLITS_CV,
        SVR_MAX_ROWS,
        RF_ESTIMATORS,
    )
    pred_ml: Dict[str, np.ndarray] = {}
    train_pred_ml: Dict[str, np.ndarray] = {}
    oof_ml: Dict[str, np.ndarray] = {}
    models_save: Dict[str, Any] = {}
    for name in TIER3_NAMES:
        m, pt, oof = tier3[name]
        models_save[name] = m
        pred_ml[name] = pt.astype(np.float64)
        oof_ml[name] = oof
        train_pred_ml[name] = m.predict(X_train_s).astype(np.float64)

    # Ensembles
    top3, w3 = ens_mod.pick_top3_weights(oof_ml, y_train)
    pred_ml["VotingEnsemble"] = ens_mod.voting_predict(pred_ml, top3, w3).astype(np.float64)
    X_oof = ens_mod.build_oof_matrix(oof_ml, TIER3_NAMES)
    X_test_stack = np.column_stack([pred_ml[n] for n in TIER3_NAMES])
    meta_model, pred_stack = ens_mod.stacking_train_predict(X_oof, y_train, X_test_stack)
    pred_ml["StackingEnsemble"] = pred_stack.astype(np.float64)
    joblib.dump(meta_model, SAVED_MODELS_DIR / f"stacking_{ticker}.pkl")

    v_vote = np.zeros_like(train_pred_ml[top3[0]], dtype=np.float64)
    for i, n in enumerate(top3):
        v_vote = v_vote + train_pred_ml[n] * float(w3[i])
    train_pred_ml["VotingEnsemble"] = v_vote
    train_pred_ml["StackingEnsemble"] = meta_model.predict(
        np.column_stack([train_pred_ml[n] for n in TIER3_NAMES])
    ).astype(np.float64)

    # --- HMM ---
    pred_hmm = _align(
        hmm_mod.fit_predict_hmm_price(
            ohlcv[ohlcv.index <= TRAIN_END],
            ohlcv[ohlcv.index >= TEST_START],
        )
    )

    # --- DL ---
    price_scaler = MinMaxScaler((0, 1))
    price_scaler.fit(train_close.values.reshape(-1, 1))
    train_len = len(train_close)
    lookback = 7
    dl_names = ["RNN", "LSTM_A", "LSTM_B", "BiLSTM"]
    pred_dl_test: Dict[str, np.ndarray] = {}
    pred_dl_train: Dict[str, np.ndarray] = {}
    for dn in dl_names:
        tr_p, te_p = train_dl_price_preds(
            dn,
            train_close,
            full_close,
            train_len,
            lookback,
            price_scaler,
            device,
        )
        if te_p.size == 0:
            continue
        pred_dl_train[dn] = tr_p
        pred_dl_test[dn] = te_p

    # Train close for next-day alignment (ML): predict price at t+1 from row t
    train_close_at_t = close.reindex(train_df.index).values.astype(float)
    train_actual_next = close.shift(-1).reindex(train_df.index).values.astype(float)

    def train_rmse_ml(pred_log: np.ndarray) -> float:
        p = prices_from_log_pred(train_close_at_t, pred_log)
        mask = np.isfinite(train_actual_next) & np.isfinite(p)
        if mask.sum() < 10:
            return float("nan")
        return rmse_over_mean(train_actual_next[mask], p[mask])

    def train_rmse_dl(train_prices: np.ndarray) -> float:
        actual = train_close.values[lookback:train_len]
        m = min(len(actual), len(train_prices))
        mask = np.isfinite(actual[:m]) & np.isfinite(train_prices[:m])
        return rmse_over_mean(actual[:m][mask[:m]], train_prices[:m][mask[:m]])

    # Collect all test predictions (price next)
    ts_preds = {
        "Naive": pred_naive,
        "BuyHold": pred_bh,
        "SMA_Cross": pred_sma,
        "HoltWinters": pred_hw,
        "ARIMA": pred_arima,
        "HMM_Regime": pred_hmm,
    }
    for name, pv in ts_preds.items():
        rows.append(
            compute_metrics_row(
                ticker,
                name,
                source,
                actual_next,
                pv,
                close_at_t,
                y_class_test,
                np.sign(np.log(pv / close_at_t)),
                bench_rets,
                float("nan"),
            )
        )

    for name in TIER3_NAMES:
        pt = prices_from_log_pred(close_at_t, pred_ml[name])
        tr = train_rmse_ml(train_pred_ml[name])
        cls_p = None
        if name in models_save and len(test_df) > 6:
            ch = ml_chain_5d_class(models_save[name], X_test, scaler, idx_test, close)
            cls_p = np.pad(ch, (0, len(idx_test) - len(ch)), constant_values=np.nan)
        rows.append(
            compute_metrics_row(
                ticker,
                name,
                source,
                actual_next,
                pt,
                close_at_t,
                y_class_test,
                cls_p,
                bench_rets,
                tr,
            )
        )

    for name in ("VotingEnsemble", "StackingEnsemble"):
        pt = prices_from_log_pred(close_at_t, pred_ml[name])
        tr = train_rmse_ml(train_pred_ml[name])
        rows.append(
            compute_metrics_row(
                ticker,
                name,
                source,
                actual_next,
                pt,
                close_at_t,
                y_class_test,
                None,
                bench_rets,
                tr,
            )
        )

    for dn in pred_dl_test:
        pt = pred_dl_test[dn]
        tr = train_rmse_dl(pred_dl_train[dn])
        # align DL test length to test_df
        m = min(len(actual_next), len(pt))
        rows.append(
            compute_metrics_row(
                ticker,
                dn,
                source,
                actual_next[:m],
                pt[:m],
                close_at_t[:m],
                y_class_test[:m],
                np.sign(np.log(pt[:m] / close_at_t[:m])),
                bench_rets[:m],
                tr,
            )
        )

    df_res = pd.DataFrame(rows)
    meta = {
        "ticker": ticker,
        "data_source": source,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "voting_top3": top3,
        "adf_kpss": ts_mod.adf_kpss_summary(log_returns_from_prices(train_close).dropna()),
    }

    if HAS_SHAP and "RandomForest" in models_save:
        try:
            import matplotlib.pyplot as plt

            x_s = X_test_s[: min(200, len(X_test_s))]
            explainer = shap.TreeExplainer(models_save["RandomForest"])
            sv = explainer.shap_values(x_s)
            shap.summary_plot(sv, x_s, feature_names=feature_cols, show=False, max_display=15)
            plt.tight_layout()
            plt.savefig(CHARTS_DIR / f"shap_{ticker}.png", dpi=120)
            plt.close()
        except Exception:
            pass

    if HAS_PLOT:
        best = df_res.loc[df_res["rmse_mean_test"].idxmin()]
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(x=test_close.index, y=test_close.values, ax=ax, label="Actual")
        ax.set_title(f"{ticker} test Adj Close (best test RMSE/mean: {best['model']})")
        ax.legend()
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / f"price_{ticker}.png", dpi=120)
        plt.close()

    for n, m in models_save.items():
        joblib.dump(m, SAVED_MODELS_DIR / f"{ticker}_{n}.pkl")

    with open(RESULTS_DIR / f"meta_{ticker}.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return df_res, meta


def main() -> None:
    set_seeds()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    start_all = min(TICKERS.values())
    macro = load_fred_features(start_all, "2027-01-01")
    bench_raw, _ = download_yahoo("^GSPC", start_date=start_all)
    bench_close = _clean_ohlcv(bench_raw)["Close"]

    device = dl_mod.get_device()
    all_dfs: List[pd.DataFrame] = []
    single = os.environ.get("SINGLE_TICKER")
    tick_iter = (
        {single: TICKERS[single]}.items()
        if single and single in TICKERS
        else TICKERS.items()
    )
    for ticker, sd in tqdm(tick_iter, desc="Tickers"):
        df_t, _ = run_ticker(ticker, sd, macro, bench_close, device)
        all_dfs.append(df_t)
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    out = pd.concat(all_dfs, ignore_index=True)
    out.to_csv(RESULTS_DIR / "results_all.csv", index=False)
    print(out.groupby("ticker")["rmse_mean_test"].min())
    try:
        from report import generate_report

        generate_report()
    except Exception as e:
        print("report skipped:", e)


if __name__ == "__main__":
    main()
