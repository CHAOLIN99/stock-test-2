"""
Data download, validation, feature engineering, and train/test splits.
Prices: Yahoo Finance (cached parquet, Stooq fallback). Macro: FRED CSV endpoints.
"""

from __future__ import annotations

import io
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

from config import (
    DATA_DIR,
    FRED_SERIES,
    RANDOM_SEED,
    SAVED_MODELS_DIR,
    TEST_START,
    TRAIN_END,
    YF_END,
)

warnings.filterwarnings("ignore", category=FutureWarning)


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def download_yahoo(
    ticker: str,
    start_date: str,
    end_date: str = YF_END,
    cache_path: Optional[Path] = None,
    refresh: bool = False,
) -> Tuple[pd.DataFrame, str]:
    """
    Download OHLCV from Yahoo Finance with auto_adjust=True (Adj Close in 'Close').
    Returns (df, source_label).
    """
    _ensure_dirs()
    if cache_path is None:
        cache_path = DATA_DIR / f"raw_{ticker.replace('^', '_')}.parquet"

    if cache_path.exists() and not refresh:
        df = pd.read_parquet(cache_path)
        return df, "parquet_cache"

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    if df.empty:
        # Stooq fallback (daily CSV; symbol format e.g. aapl.us)
        try:
            sym = ticker.replace("^", "").lower() + ".us"
            url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
            with urllib.request.urlopen(url, timeout=60) as resp:
                df = pd.read_csv(io.StringIO(resp.read().decode("utf-8")))
            if df.empty or "Date" not in df.columns:
                raise RuntimeError("Empty Stooq data")
            df = df.set_index(pd.to_datetime(df.pop("Date"))).sort_index()
            source = "stooq_fallback"
        except Exception as e:
            raise RuntimeError(f"yfinance and Stooq failed for {ticker}: {e}") from e
    else:
        source = "yfinance"
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)

    # With auto_adjust=True, Close is the adjusted close
    df = df.rename(columns=str.title)
    df = df[[c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]]
    df.index.name = "Date"
    df = df.sort_index()
    df.to_parquet(cache_path)
    return df, source


def validate_price_df(df: pd.DataFrame, train_end: str = TRAIN_END) -> None:
    """Run assertions after download."""
    assert df.index.is_monotonic_increasing
    assert not df.index.duplicated().any()
    train = df[df.index <= train_end]
    vol = train["Volume"].astype(float)
    pct_pos = (vol > 0).mean()
    assert pct_pos > 0.98, f"Volume > 0 on {pct_pos:.2%} of days (need >98%)"


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.ffill(limit=2)


FRED_CACHE = DATA_DIR / "fred_raw.parquet"
# Monthly series are dated at the start of the reference month but published 4-6 weeks later.
FRED_MONTHLY = ["FEDFUNDS", "CPIAUCSL", "UNRATE", "CPIAUCSL_yoy"]
FRED_MONTHLY_RELEASE_LAG = pd.Timedelta(days=45)


def download_fred() -> pd.DataFrame:
    """Fetch raw FRED series via the public graph CSV endpoints (no API key) and cache them."""
    _ensure_dirs()
    parts = []
    for sid in FRED_SERIES:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
        with urllib.request.urlopen(url, timeout=60) as resp:
            s = pd.read_csv(io.StringIO(resp.read().decode("utf-8")), index_col=0, parse_dates=True)
        parts.append(pd.to_numeric(s.iloc[:, 0], errors="coerce").rename(sid))
    fred = pd.concat(parts, axis=1).sort_index()
    fred.to_parquet(FRED_CACHE)
    return fred


def load_fred_features(start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    """Daily macro features (monthly series forward-filled). Uses data/fred_raw.parquet when present."""
    fred = pd.read_parquet(FRED_CACHE) if use_cache and FRED_CACHE.exists() else download_fred()
    idx = pd.date_range(start=start, end=end, freq="D")
    # CPI is monthly: 12 observations = one year (must be computed before daily reindexing)
    fred["CPIAUCSL_yoy"] = fred["CPIAUCSL"].dropna().pct_change(periods=12) * 100.0
    cols = list(fred.columns)
    released = fred[FRED_MONTHLY].dropna(how="all").shift(freq=FRED_MONTHLY_RELEASE_LAG)
    fred = fred.drop(columns=FRED_MONTHLY).join(released, how="outer")[cols]
    return fred.reindex(idx).ffill()


# Kept in the frame for targets/prices but excluded from model inputs: price and CPI levels trend
# upward, so test-period values fall outside the training range.
NON_FEATURE_COLS = {"Open", "High", "Low", "Close", "Volume", "CPIAUCSL", "y_reg", "y_class"}


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if col not in NON_FEATURE_COLS]


def add_technical_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Multivariate features, all scale-free (returns, ratios, or % of price).
    Day-t bar features (ret_0d, oc_ret, hl_range, vol_chg) are known at the close of t,
    when the t+1 forecast is made. Rolling statistics are shifted by 1 so they use data through t-1.
    """
    df = ohlcv.copy()
    c = df["Close"].astype(float)
    o, h, low, v = (df[k].astype(float) for k in ("Open", "High", "Low", "Volume"))

    log_c = np.log(c.replace(0, np.nan))
    df["ret_0d"] = log_c.diff()
    df["oc_ret"] = np.log(c / o.replace(0, np.nan))
    df["hl_range"] = np.log(h / low.replace(0, np.nan))
    df["vol_chg"] = np.log(v.replace(0, np.nan)).diff()

    df["ret_1d"] = log_c.diff().shift(1)
    df["ret_5d"] = log_c.diff(5).shift(1)
    df["ret_21d"] = log_c.diff(21).shift(1)

    for w in (10, 20, 50, 200):
        sma = c.rolling(w).mean().shift(1)
        df[f"close_over_SMA_{w}"] = c.shift(1) / sma.replace(0, np.nan)

    for span in (12, 26):
        ema = c.ewm(span=span, adjust=False).mean().shift(1)
        df[f"close_over_EMA_{span}"] = c.shift(1) / ema.replace(0, np.nan)

    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta.clip(upper=0.0))
    rs = gain.rolling(14).mean() / loss.rolling(14).mean().replace(0, np.nan)
    df["RSI_14"] = (100 - (100 / (1 + rs))).shift(1)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    df["MACD_line_pct"] = (macd_line / c).shift(1)
    df["MACD_signal_pct"] = (signal / c).shift(1)
    df["MACD_hist_pct"] = ((macd_line - signal) / c).shift(1)

    ma20 = c.rolling(20).mean().shift(1)
    sd20 = c.rolling(20).std().shift(1)
    upper = ma20 + 2 * sd20
    lower = ma20 - 2 * sd20
    df["BB_width"] = ((upper - lower) / ma20.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    tr = pd.concat(
        [
            (h - low),
            (h - c.shift(1)).abs(),
            (low - c.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["ATR_14_pct"] = (tr.rolling(14).mean() / c).shift(1)

    lr = log_c.diff()
    df["realized_vol_21"] = (lr.rolling(21).std() * np.sqrt(252)).shift(1)

    sign_ret = np.sign(c.diff().fillna(0))
    obv = (sign_ret * v).cumsum()
    df["OBV_20d"] = (obv.diff(20) / v.rolling(20).sum().replace(0, np.nan)).shift(1)
    vol_ma20 = v.rolling(20).mean().shift(1)
    df["vol_ratio"] = (v.shift(1) / vol_ma20.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    vm = v.shift(1)
    df["vol_z"] = (vm - vm.rolling(20).mean()) / vm.rolling(20).std().replace(0, np.nan)

    roll_max = c.rolling(252, min_periods=50).max().shift(1)
    roll_min = c.rolling(252, min_periods=50).min().shift(1)
    df["pct_52w_high"] = ((c.shift(1) - roll_max) / roll_max.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    df["pct_52w_low"] = ((c.shift(1) - roll_min) / roll_min.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    df["dow"] = df.index.dayofweek.astype(float)
    df["month"] = df.index.month.astype(float)
    df["is_month_end"] = df.index.is_month_end.astype(int)
    df["is_quarter_end"] = df.index.is_quarter_end.astype(int)

    return df


def merge_macro(df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    out = df.join(macro, how="left")
    return out.ffill(limit=5)


def build_supervised_frame(
    ticker: str,
    ohlcv: pd.DataFrame,
    macro: pd.DataFrame,
) -> Tuple[pd.DataFrame, list[str]]:
    """Returns frame with targets and feature list for ML."""
    base = add_technical_features(ohlcv)
    full = merge_macro(base, macro)
    c = ohlcv["Close"].astype(float)
    full["y_reg"] = np.log(c.shift(-1) / c)
    full["y_class"] = np.sign(c.shift(-5) - c)
    full = full.replace([np.inf, -np.inf], np.nan).dropna(subset=["y_reg", "y_class"])
    return full, feature_columns(full)


def train_test_split_df(
    df: pd.DataFrame,
    ticker: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df.index <= TRAIN_END].copy()
    test = df[df.index >= TEST_START].copy()
    assert train.index.max() < test.index.min(), "Temporal leakage — stop"
    assert len(train) > 500, f"Train too short for {ticker}"
    assert len(test) > 20, f"Test too short for {ticker}"
    return train, test


def fit_save_scaler(
    X_train: np.ndarray,
    ticker: str,
) -> MinMaxScaler:
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    scaler = MinMaxScaler((0, 1))
    scaler.fit(X_train)
    joblib.dump(scaler, SAVED_MODELS_DIR / f"scaler_{ticker}.pkl")
    return scaler


def prepare_univariate_series(
    df: pd.DataFrame,
) -> pd.Series:
    """Adj Close series (column 'Close' when auto_adjust=True)."""
    return df["Close"].astype(float).copy()


def set_seeds() -> None:
    import torch

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
