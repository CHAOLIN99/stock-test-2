#!/usr/bin/env python3
"""
Deep-learning experiment runner: sweeps architecture / lookback / feature-set /
regularization variants and logs train, validation, and test error per run.

Run:  python dl_experiments.py --grid baseline --ticker NVDA
Out:  results/dl_experiments.csv (appended), results/dl_curves/<run_id>.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import asdict, replace
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import LOOKBACK, RESULTS_DIR, TICKERS, YF_END
from data_pipeline import (
    build_supervised_frame,
    clean_ohlcv,
    download_yahoo,
    load_fred_features,
    train_test_split_df,
)
from evaluate import rmse_over_mean
from models import deep_learning as dl

warnings.filterwarnings("ignore")

MACRO_COLS = {"FEDFUNDS", "T10Y2Y", "VIXCLS", "UNRATE", "DGS10", "CPIAUCSL_yoy"}

FEATURE_SETS = {
    "returns": lambda cols: [c for c in cols if c.startswith("ret_")],
    "technical": lambda cols: [c for c in cols if c not in MACRO_COLS],
    "all": lambda cols: list(cols),
}

BASE = dict(spec=dl.PRESETS["LSTM_A"], lookback=LOOKBACK, features="all", lr=1e-3, weight_decay=0.0)


def _v(name: str, **kw) -> Dict[str, Any]:
    out = dict(BASE, name=name)
    out.update(kw)
    return out


GRIDS: Dict[str, List[Dict[str, Any]]] = {
    "baseline": [_v(n, spec=s) for n, s in dl.PRESETS.items()],
    "cells": [
        _v("rnn_128_64", spec=dl.SeqSpec(cell="rnn", hidden=(128, 64))),
        _v("gru_128_64", spec=dl.SeqSpec(cell="gru", hidden=(128, 64))),
        _v("lstm_128_64", spec=dl.SeqSpec(cell="lstm", hidden=(128, 64))),
    ],
    "size": [
        _v("lstm_32", spec=dl.SeqSpec(hidden=(32,))),
        _v("lstm_64", spec=dl.SeqSpec(hidden=(64,))),
        _v("lstm_64_32", spec=dl.SeqSpec(hidden=(64, 32))),
        _v("lstm_256_128", spec=dl.SeqSpec(hidden=(256, 128))),
    ],
    "lookback": [_v(f"lookback_{k}", lookback=k) for k in (5, 7, 21, 63)],
    "features": [_v(f"features_{f}", features=f) for f in FEATURE_SETS],
    "regularization": [
        _v("dropout_0", spec=replace(dl.PRESETS["LSTM_A"], dropout=0.0)),
        _v("dropout_0.4", spec=replace(dl.PRESETS["LSTM_A"], dropout=0.4)),
        _v("wd_1e-4", weight_decay=1e-4),
        _v("wd_1e-3", weight_decay=1e-3),
        _v("lr_3e-4", lr=3e-4),
    ],
}


def load_frame(ticker: str, macro: pd.DataFrame):
    raw, _ = download_yahoo(ticker, start_date=TICKERS[ticker])
    ohlcv = clean_ohlcv(raw)
    sup, cols = build_supervised_frame(ticker, ohlcv, macro)
    train, test = train_test_split_df(sup, ticker)
    train, test = train.dropna(subset=cols), test.dropna(subset=cols)
    close = ohlcv["Close"].astype(float)
    return train, test, cols, close


def run_one(cfg: Dict[str, Any], ticker: str, train, test, cols, close, device) -> Dict[str, Any]:
    feats = FEATURE_SETS[cfg["features"]](cols)
    scaler = MinMaxScaler().fit(train[feats].values)
    X = np.vstack([scaler.transform(train[feats].values), scaler.transform(test[feats].values)])
    y = np.concatenate([train["y_reg"].values, test["y_reg"].values])
    n = len(train)

    t0 = time.time()
    res = dl.train_seq_model(
        cfg["spec"], X, y, n, cfg["lookback"], device, lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    secs = time.time() - t0

    def split_stats(pred, rows, prefix):
        mask = np.isfinite(pred)
        ret_true, ret_pred = rows["y_reg"].values[mask], pred[mask]
        c_t = close.reindex(rows.index).values[mask]
        return {
            f"{prefix}_ret_rmse": float(np.sqrt(np.mean((ret_pred - ret_true) ** 2))),
            f"{prefix}_zero_ret_rmse": float(np.sqrt(np.mean(ret_true**2))),
            f"{prefix}_rmse_mean": rmse_over_mean(c_t * np.exp(ret_true), c_t * np.exp(ret_pred)),
            f"{prefix}_dir_acc": float(np.mean(np.sign(ret_pred) == np.sign(ret_true))),
            f"{prefix}_frac_pred_up": float(np.mean(ret_pred > 0)),
            f"{prefix}_pred_std": float(np.std(ret_pred)),
            f"{prefix}_corr": float(np.corrcoef(ret_pred, ret_true)[0, 1]) if np.std(ret_pred) > 0 else float("nan"),
        }

    run_id = f"{ticker}_{cfg['name']}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
    row = {
        "run_id": run_id,
        "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
        "ticker": ticker,
        "name": cfg["name"],
        "cell": cfg["spec"].cell,
        "hidden": "-".join(map(str, cfg["spec"].hidden)),
        "bidirectional": cfg["spec"].bidirectional,
        "dropout": cfg["spec"].dropout,
        "lookback": cfg["lookback"],
        "features": cfg["features"],
        "n_features": len(feats),
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "n_params": res.n_params,
        "best_epoch": res.best_epoch,
        "epochs_run": res.epochs_run,
        "best_val_loss": float(min(res.history["val_loss"])) if res.history else float("nan"),
        "train_seconds": round(secs, 1),
        **split_stats(res.pred[:n], train, "train"),
        **split_stats(res.pred[n:], test, "test"),
    }
    curves_dir = RESULTS_DIR / "dl_curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    with open(curves_dir / f"{run_id}.json", "w") as f:
        json.dump({"config": {**cfg, "spec": asdict(cfg["spec"])}, "history": res.history, "metrics": row}, f, indent=1)
    dl.cleanup_dl(res.model, device)
    return row


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", choices=list(GRIDS) + ["all"], default="baseline")
    p.add_argument("--ticker", action="append", choices=sorted(TICKERS), help="Repeatable; default: all tickers")
    args = p.parse_args(argv)

    macro = load_fred_features(min(TICKERS.values()), YF_END)
    configs = [c for g in GRIDS.values() for c in g] if args.grid == "all" else GRIDS[args.grid]
    device = dl.get_device()
    out_path = RESULTS_DIR / "dl_experiments.csv"

    for ticker in args.ticker or sorted(TICKERS):
        train, test, cols, close = load_frame(ticker, macro)
        for cfg in configs:
            row = run_one(cfg, ticker, train, test, cols, close, device)
            pd.DataFrame([row]).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
            print(
                f"{ticker} {cfg['name']:<16} params={row['n_params']:>7} "
                f"epochs={row['best_epoch']}/{row['epochs_run']} "
                f"train={row['train_ret_rmse']:.5f} test={row['test_ret_rmse']:.5f} "
                f"(zero {row['test_zero_ret_rmse']:.5f}) dir={row['test_dir_acc']:.3f} "
                f"up={row['test_frac_pred_up']:.2f} corr={row['test_corr']:.3f}",
                flush=True,
            )
    print("wrote:", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
