"""
Data download, validation, feature engineering, and train/test splits.
Primary: Yahoo Finance (cached parquet). Secondary: FRED via pandas_datareader.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Tuple

import io
import urllib.error
import urllib.request

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
    TICKERS,
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
) -> Tuple[pd.DataFrame, str]:
    """
    Download OHLCV from Yahoo Finance with auto_adjust=True (Adj Close in 'Close').
    Returns (df, source_label).
    """
    _ensure_dirs()
    if cache_path is None:
        cache_path = DATA_DIR / f"raw_{ticker.replace('^', '_')}.parquet"

    if cache_path.exists():
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
                raw = resp.read().decode("utf-8")
            df = pd.read_csv(io.StringIO(raw))
            if not df.empty and "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date").sort_index()
                df.index = df.index.tz_localize(None)
                df = df.rename(
                    columns={
                        "Open": "Open",
                        "High": "High",
                        "Low": "Low",
                        "Close": "Close",
                        "Volume": "Volume",
                    }
                )
                source = "stooq_fallback"
            else:
                raise RuntimeError("Empty Stooq data")
        except Exception as e:
            raise RuntimeError(f"yfinance and Stooq failed for {ticker}: {e}") from e
    else:
        source = "yfinance"
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)

    df = df.rename(columns=str.title)
    # yfinance with auto_adjust: Close is adjusted
    col_map = {"Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"}
    for k in list(df.columns):
        if k not in col_map:
            df = df.drop(columns=[k], errors="ignore")
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
    assert train.index.max() <= pd.Timestamp(train_end)
    assert (train.index <= pd.Timestamp(train_end)).all()
    vol = train["Volume"].astype(float)
    pct_pos = (vol > 0).mean()
    assert pct_pos > 0.98, f"Volume > 0 on {pct_pos:.2%} of days (need >98%)"


def forward_fill_limited(s: pd.Series, max_gap: int = 2) -> pd.Series:
    """Forward-fill only up to `max_gap` consecutive NaNs."""
    s = s.copy()
    is_na = s.isna()
    if not is_na.any():
        return s
    groups = (~is_na).cumsum()
    runlength = s.groupby(groups).transform("count")
    # mask runs of NA longer than max_gap: leave as NA
    # simpler: ffill with limit
    return s.ffill(limit=max_gap)


def load_fred_features(start: str, end: str) -> pd.DataFrame:
    """
    Fetch FRED macro series via public graph CSV endpoints (no API key).
    Forward-fill monthly series to daily.
    """
    idx = pd.date_range(start=start, end=end, freq="D")
    parts: list[pd.DataFrame] = []
    for sid in FRED_SERIES:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            s = pd.read_csv(io.StringIO(raw), index_col=0, parse_dates=True)
            s = s.rename(columns={s.columns[0]: sid})
            parts.append(s)
        except (urllib.error.URLError, ValueError, pd.errors.ParserError):
            parts.append(pd.DataFrame(index=idx, columns=[sid]))

    fred = pd.concat(parts, axis=1)
    fred = fred.sort_index()
    fred = fred.reindex(idx).ffill()
    if "CPIAUCSL" in fred.columns:
        cpi = fred["CPIAUCSL"]
        fred["CPIAUCSL_yoy"] = cpi.pct_change(periods=12) * 100.0
    return fred.ffill()


def add_technical_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Multivariate features. Rolling statistics are shifted by 1 so row t uses
    only information available through t-1 (no same-bar leakage).
    """
    df = ohlcv.copy()
    c = df["Close"].astype(float)
    h, low, v = df["High"].astype(float), df["Low"].astype(float), df["Volume"].astype(float)

    log_c = np.log(c.replace(0, np.nan))
    df["ret_1d"] = log_c.diff().shift(1)
    df["ret_5d"] = log_c.diff(5).shift(1)
    df["ret_21d"] = log_c.diff(21).shift(1)

    for w in (10, 20, 50, 200):
        df[f"SMA_{w}"] = c.rolling(w).mean().shift(1)

    df["EMA_12"] = c.ewm(span=12, adjust=False).mean().shift(1)
    df["EMA_26"] = c.ewm(span=26, adjust=False).mean().shift(1)

    sma50 = c.rolling(50).mean().shift(1)
    sma200 = c.rolling(200).mean().shift(1)
    df["price_over_SMA50"] = (c.shift(1) / sma50.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    df["price_over_SMA200"] = (c.shift(1) / sma200.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta.clip(upper=0.0))
    rs = gain.rolling(14).mean() / loss.rolling(14).mean().replace(0, np.nan)
    df["RSI_14"] = (100 - (100 / (1 + rs))).shift(1)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    df["MACD_line"] = macd_line.shift(1)
    df["MACD_signal"] = signal.shift(1)
    df["MACD_hist"] = (macd_line - signal).shift(1)

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
    df["ATR_14"] = tr.rolling(14).mean().shift(1)

    lr = log_c.diff()
    df["realized_vol_21"] = (lr.rolling(21).std() * np.sqrt(252)).shift(1)

    sign_ret = np.sign(c.diff().fillna(0))
    obv = (sign_ret * v).cumsum()
    df["OBV"] = obv.shift(1)
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
    feature_cols = [
        col
        for col in full.columns
        if col not in ("y_reg", "y_class")
    ]
    return full, feature_cols


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
    np.random.seed(RANDOM_SEED)
    try:
        import torch

        torch.manual_seed(RANDOM_SEED)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(RANDOM_SEED)
    except ImportError:
        pass
