#!/usr/bin/env python3
"""
End-to-end training and evaluation for five tickers.
Run: python run_experiment.py
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

from config import (
    BOOTSTRAP_N,
    CHARTS_DIR,
    LOOKBACK,
    N_SPLITS_CV,
    RESULTS_DIR,
    RF_ESTIMATORS,
    SAVED_MODELS_DIR,
    SVR_MAX_ROWS,
    TEST_START,
    TICKERS,
    TIER3_NAMES,
    TRAIN_END,
    YF_END,
)
from data_pipeline import (
    build_supervised_frame,
    clean_ohlcv,
    download_yahoo,
    fit_save_scaler,
    load_fred_features,
    prepare_univariate_series,
    set_seeds,
    train_test_split_df,
    validate_price_df,
)
from evaluate import (
    bootstrap_metric,
    feature_importance_from_model,
    full_metrics_block,
    rmse_over_mean,
    strategy_log_returns_from_prices,
    write_dashboard_data,
)
from models import classical as classical_mod
from models import deep_learning as dl_mod
from models import ensemble as ens_mod
from models import hmm_regime as hmm_mod
from models import timeseries as ts_mod
from report import generate_report


def log_returns_from_prices(close: pd.Series) -> pd.Series:
    return np.log(close.astype(float) / close.astype(float).shift(1))


def prices_from_log_pred(close_at_t: np.ndarray, pred_log_ret: np.ndarray) -> np.ndarray:
    return close_at_t * np.exp(pred_log_ret)


def compute_metrics_row(
    ticker: str,
    model: str,
    source: str,
    actual_next: np.ndarray,
    pred_next: np.ndarray,
    close_at_t: np.ndarray,
    y_class: np.ndarray,
    bench_rets: np.ndarray,
    train_rmse_mean: float,
) -> Dict[str, Any]:
    ret_true = np.log(actual_next / close_at_t)
    ret_pred = np.log(pred_next / close_at_t)

    mets = full_metrics_block(
        actual_next,
        pred_next,
        y_ret_true=ret_true,
        y_ret_pred=ret_pred,
        y_class_true=y_class,
        y_class_pred=np.sign(ret_pred),
        bench_rets=bench_rets,
    )
    b_mean, b_low, b_high = bootstrap_metric(actual_next, pred_next, rmse_over_mean, n_boot=BOOTSTRAP_N)

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
    ohlcv = clean_ohlcv(raw)
    validate_price_df(ohlcv)

    sup, feature_cols = build_supervised_frame(ticker, ohlcv, macro)
    train_df, test_df = train_test_split_df(sup, ticker)
    train_df = train_df.dropna(subset=feature_cols)
    test_df = test_df.dropna(subset=feature_cols)
    assert len(train_df) > 500 and len(test_df) > 20

    close = ohlcv["Close"].astype(float)

    X_train = train_df[feature_cols].values.astype(np.float64)
    X_test = test_df[feature_cols].values.astype(np.float64)
    y_train = train_df["y_reg"].values.astype(np.float64)
    y_test = test_df["y_reg"].values.astype(np.float64)
    y_class_test = test_df["y_class"].values

    scaler = fit_save_scaler(X_train, ticker)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    train_close = prepare_univariate_series(ohlcv[ohlcv.index <= TRAIN_END])
    test_close = prepare_univariate_series(ohlcv[ohlcv.index >= TEST_START])

    idx_test = test_df.index
    close_at_t = close.reindex(idx_test).values.astype(float)
    actual_next = close.shift(-1).reindex(idx_test).values.astype(float)
    bench_rets = log_returns_from_prices(bench_close).shift(-1).reindex(idx_test).values

    # --- Baselines ---
    def _align(s: pd.Series) -> np.ndarray:
        return s.reindex(idx_test).ffill().bfill().values.astype(float)

    pred_naive = close_at_t.copy()
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

    train_pred_ml["VotingEnsemble"] = ens_mod.voting_predict(train_pred_ml, top3, w3)
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

    # --- DL: one-step-ahead log returns from windows of the scaled feature matrix ---
    X_all_s = np.vstack([X_train_s, X_test_s])
    y_all = np.concatenate([y_train, y_test])
    n_train = len(X_train_s)
    pred_dl_test: Dict[str, np.ndarray] = {}
    pred_dl_train: Dict[str, np.ndarray] = {}
    dl_meta: Dict[str, Any] = {}
    for dn, spec in dl_mod.PRESETS.items():
        res = dl_mod.train_seq_model(spec, X_all_s, y_all, n_train, LOOKBACK, device)
        if res.model is None:
            continue
        pred_dl_train[dn] = res.pred[:n_train]
        pred_dl_test[dn] = res.pred[n_train:]
        dl_meta[dn] = {
            "n_params": res.n_params,
            "best_epoch": res.best_epoch,
            "epochs_run": res.epochs_run,
            "lookback": LOOKBACK,
            "history": res.history,
        }
        dl_mod.cleanup_dl(res.model, device)

    # Train-period price error for log-return models (predict close[t+1] from row t)
    train_close_at_t = close.reindex(train_df.index).values.astype(float)
    train_actual_next = close.shift(-1).reindex(train_df.index).values.astype(float)

    def train_rmse(pred_log: np.ndarray) -> float:
        p = prices_from_log_pred(train_close_at_t, pred_log)
        mask = np.isfinite(train_actual_next) & np.isfinite(p)
        return rmse_over_mean(train_actual_next[mask], p[mask]) if mask.sum() >= 10 else float("nan")

    rows: List[Dict[str, Any]] = []

    def score(name: str, pred_price: np.ndarray, train_rmse_mean: float = float("nan")) -> None:
        rows.append(
            compute_metrics_row(
                ticker, name, source, actual_next, pred_price, close_at_t,
                y_class_test, bench_rets, train_rmse_mean,
            )
        )

    for name, pv in [
        ("Naive", pred_naive),
        ("BuyHold", pred_bh),
        ("SMA_Cross", pred_sma),
        ("HoltWinters", pred_hw),
        ("ARIMA", pred_arima),
        ("HMM_Regime", pred_hmm),
    ]:
        score(name, pv)
    for name in (*TIER3_NAMES, "VotingEnsemble", "StackingEnsemble"):
        score(name, prices_from_log_pred(close_at_t, pred_ml[name]), train_rmse(train_pred_ml[name]))
    for name in pred_dl_test:
        score(name, prices_from_log_pred(close_at_t, pred_dl_test[name]), train_rmse(pred_dl_train[name]))

    df_res = pd.DataFrame(rows)
    meta = {
        "ticker": ticker,
        "data_source": source,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "voting_top3": top3,
        "deep_learning": dl_meta,
        "adf_kpss": ts_mod.adf_kpss_summary(log_returns_from_prices(train_close).dropna()),
    }

    try:
        x_s = X_test_s[:200]
        sv = shap.TreeExplainer(models_save["RandomForest"]).shap_values(x_s)
        shap.summary_plot(sv, x_s, feature_names=feature_cols, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / f"shap_{ticker}.png", dpi=120)
        plt.close()
    except Exception as e:
        print(f"SHAP plot skipped for {ticker}: {e}")

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

    # --- Dashboard payload: per-date series, NaN -> None for JSON ---
    def _as_list(a: np.ndarray) -> List[Optional[float]]:
        return [float(v) if np.isfinite(v) else None for v in np.asarray(a, dtype=float)]

    price_preds = {
        "Naive": pred_naive,
        "BuyHold": pred_bh,
        "SMA_Cross": pred_sma,
        "HoltWinters": pred_hw,
        "ARIMA": pred_arima,
        "HMM_Regime": pred_hmm,
        **{n: prices_from_log_pred(close_at_t, r) for n, r in {**pred_ml, **pred_dl_test}.items()},
    }
    metrics = {
        r["model"]: {k: (float(r[k]) if pd.notna(r[k]) else None) for k in ("rmse_mean_test", "dir_acc", "sharpe", "r2", "max_dd")}
        for _, r in df_res.iterrows()
    }
    fi = {}
    for name, model in models_save.items():
        if name not in ("SVR", "MLP"):
            fi_list = feature_importance_from_model(model, feature_cols, top_k=20)
            if fi_list:
                fi[name] = fi_list

    regime_names, regime = hmm_mod.label_regimes(ohlcv, idx_test)
    dash = {
        "dates": [str(d.date()) for d in idx_test],
        "actual": _as_list(actual_next),
        "predictions": {n: _as_list(p) for n, p in price_preds.items()},
        "metrics": metrics,
        "regime": regime,
        "regime_names": regime_names,
        "feature_importance": fi,
        "strategy_log_returns": {
            n: _as_list(strategy_log_returns_from_prices(p, actual_next)) for n, p in price_preds.items()
        },
    }

    return df_res, dash


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_experiment.py",
        description="Run the end-to-end stock prediction pipeline and export results + dashboard payload.",
    )
    p.add_argument(
        "--ticker",
        action="append",
        type=str.upper,
        choices=list(TICKERS),
        help="Run only this ticker (repeatable). Example: --ticker NVDA",
    )
    p.add_argument(
        "--list-tickers",
        action="store_true",
        help="Print supported tickers and exit.",
    )
    p.add_argument(
        "--skip-dashboard",
        action="store_true",
        help="Skip writing results/charts_data.json (dashboard payload).",
    )
    p.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip generating report.md at the end.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.list_tickers:
        print("Supported tickers:")
        for t in sorted(TICKERS.keys()):
            print(" -", t)
        return 0

    set_seeds()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    start_all = min(TICKERS.values())
    macro = load_fred_features(start_all, YF_END)
    bench_raw, _ = download_yahoo("^GSPC", start_date=start_all)
    bench_close = clean_ohlcv(bench_raw)["Close"]

    device = dl_mod.get_device()
    tickers = args.ticker or list(TICKERS)
    all_dfs: List[pd.DataFrame] = []
    dash_all: Dict[str, Dict[str, Any]] = {}
    for ticker in tqdm(tickers, desc="Tickers"):
        df_t, dash_all[ticker] = run_ticker(ticker, TICKERS[ticker], macro, bench_close, device)
        all_dfs.append(df_t)
        gc.collect()

    out = pd.concat(all_dfs, ignore_index=True)
    out.to_csv(RESULTS_DIR / "results_all.csv", index=False)
    if not args.skip_dashboard:
        write_dashboard_data(
            tickers=tickers,
            per_ticker=dash_all,
            out_dir=RESULTS_DIR,
            generated_at=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        print("wrote:", RESULTS_DIR / "charts_data.json")
    print(out.groupby("ticker")["rmse_mean_test"].min())
    if not args.skip_report:
        generate_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
