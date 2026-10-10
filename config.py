"""Global configuration for stock prediction experiments."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SAVED_MODELS_DIR = ROOT / "saved_models"
RESULTS_DIR = ROOT / "results"
CHARTS_DIR = RESULTS_DIR / "charts"

# Ticker -> first trading date to download
TICKERS = {
    "META": "2012-05-18",
    "NVDA": "2000-01-02",
    "AMD": "2000-01-02",
    "MSFT": "2000-01-02",
    "QCOM": "2000-01-02",
}

TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"
# yfinance end is exclusive; extend past "present" so 2025–2026 out-of-sample is included
YF_END = "2027-01-01"

LOOKBACK = 7
EPOCHS = 100
BATCH_SIZE = 64
PATIENCE = 10
RANDOM_SEED = 42

TRANSACTION_COST = 0.001
N_SPLITS_CV = 5
BOOTSTRAP_N = 200

RF_ESTIMATORS = 200
SVR_MAX_ROWS = 2000

FRED_SERIES = ["FEDFUNDS", "T10Y2Y", "VIXCLS", "CPIAUCSL", "UNRATE", "DGS10"]

TIER3_NAMES = ["Ridge", "MARS", "RandomForest", "SVR", "MLP", "XGBoost", "LightGBM"]
