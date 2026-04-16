"""Global configuration for stock prediction experiments."""

from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SAVED_MODELS_DIR = ROOT / "saved_models"
RESULTS_DIR = ROOT / "results"
CHARTS_DIR = RESULTS_DIR / "charts"

# Universe (5 stocks + benchmarks)
TICKERS = {
    "META": "2012-05-18",
    "NVDA": "2000-01-02",
    "AMD": "2000-01-02",
    "MSFT": "2000-01-02",
    "QCOM": "2000-01-02",
}
BENCHMARKS = ["^GSPC", "^VIX"]

TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"
# yfinance end is exclusive; extend past "present" so 2025–2026 out-of-sample is included
YF_END = "2027-01-01"

LOOKBACK = 7
LOOKBACK_ALT = 21

EPOCHS = 100
BATCH_SIZE = 64
PATIENCE = 10
RANDOM_SEED = 42

TRANSACTION_COST = 0.001
N_SPLITS_CV = 5
BOOTSTRAP_N = 200

RF_ESTIMATORS = 200
SVR_MAX_ROWS = 2000

# FRED series (pandas_datareader)
FRED_SERIES = ["FEDFUNDS", "T10Y2Y", "VIXCLS", "CPIAUCSL", "UNRATE", "DGS10"]

# Model names for reporting (order matters)
BASELINE_NAMES = ["Naive", "BuyHold", "SMA_Cross"]
TIER1_NAMES = ["HoltWinters"]
TIER2_NAMES = ["ARIMA"]
TIER3_NAMES = [
    "Ridge",
    "MARS",
    "RandomForest",
    "SVR",
    "MLP",
    "XGBoost",
    "LightGBM",
]
TIER4_NAMES = ["RNN", "LSTM_A", "LSTM_B", "BiLSTM", "HMM_Regime"]
TIER5_NAMES = ["VotingEnsemble", "StackingEnsemble"]

ALL_MODEL_NAMES = (
    BASELINE_NAMES
    + TIER1_NAMES
    + TIER2_NAMES
    + TIER3_NAMES
    + TIER4_NAMES
    + TIER5_NAMES
)
