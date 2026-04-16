# Stock price prediction (5 US tech stocks) — reproducible Mac/MPS pipeline

End-to-end, reproducible stock prediction system for **META, NVDA, AMD, MSFT, QCOM** with strict train/test splits and out-of-sample evaluation on **2025+** data. Designed to run locally on a MacBook Air (Apple Silicon) using **PyTorch + MPS** (no paid services).

## Scope

- **Universe (fixed)**: `META`, `NVDA`, `AMD`, `MSFT`, `QCOM`
- **Benchmarks downloaded**: `^GSPC` (S&P 500). (FRED macro series are also fetched.)
- **Frequency**: daily OHLCV
- **Targets**:
  - **Primary (regression)**: next-day prediction evaluated on **Adj Close level** (via next-day log return → price)
  - **Secondary (classification)**: **5-day return direction**
- **Train/Test**:
  - `NVDA/AMD/MSFT/QCOM`: train `2000-01-02` → `2024-12-31`
  - `META`: train `2012-05-18` → `2024-12-31` (**no pre-IPO padding**)
  - All: test `2025-01-01` → present

## Models included (reported for every ticker)

- **Tier 0 baselines**: Naive (t-1), Buy-and-hold, SMA 50/200 crossover
- **Tier 1**: Holt–Winters
- **Tier 2**: ARIMA (auto on log returns; 1-day horizon)
- **Tier 3 (multivariate ML)**: Ridge, MARS (if available), RandomForest, SVR (capped), MLP, XGBoost-like, LightGBM
- **Tier 4 (DL; univariate Adj Close)**: RNN, LSTM-A, LSTM-B, BiLSTM (PyTorch + MPS/CPU)
- **HMM**: GaussianHMM for regime-conditional next-day forecast (regime labeling + simple forecasting)
- **Tier 5 ensembles**: weighted voting (top-3 by OOF RMSE), stacking (Ridge meta-learner)

## Setup (recommended: virtualenv)

```bash
cd "/Users/weichao/Desktop/stock test 2"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Run all 5 tickers:

```bash
python run_experiment.py
python report.py
```

Run a single ticker (faster iteration):

```bash
SINGLE_TICKER=META python run_experiment.py
```

## Outputs

- **Cached raw data**: `data/raw_<TICKER>.parquet` (download once; subsequent runs load parquet)
- **Saved scalers/models**: `saved_models/`
- **Metrics**: `results/results_all.csv`
- **Per-ticker metadata**: `results/meta_<TICKER>.json`
- **Charts**: `results/charts/` (price plot; SHAP summary for RandomForest when available)
- **Dashboard data**: `results/charts_data.json` (offline frontend payload; written atomically)
- **Report**: `report.md`

## Dashboard (offline frontend)

The dashboard is a self-contained HTML/CSS/JS app (no Node, no build step). It reads `results/charts_data.json`.

1) Generate data (also writes `results/charts_data.json`):

```bash
python run_experiment.py
```

2) Start the local server:

```bash
python server.py
```

3) Open:

- `http://localhost:8080/dashboard.html`

## Notes (Mac + Python 3.14)

- **Device selection** is automatic in `models/deep_learning.py`:
  - `mps` if available, otherwise `cpu`
- **Memory/runtime constraints** are respected (single-ticker DL training; batch size 64; SVR row cap).
- **XGBoost on Python 3.14**: some XGBoost wheels can **segfault** on real feature matrices. On Python **3.14+**, `models/classical.py` uses `sklearn.ensemble.HistGradientBoostingRegressor` as a histogram-boosting stand-in and keeps the output label as `XGBoost` (documented in `report.md`).

## Repo structure

```
config.py
data_pipeline.py
evaluate.py
report.py
run_experiment.py
models/
  classical.py
  deep_learning.py
  ensemble.py
  hmm_regime.py
  timeseries.py
data/
saved_models/
results/
```

