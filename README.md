# Stock Prediction Dashboard

End-to-end, reproducible next-day price prediction for **META, NVDA, AMD, MSFT, QCOM** (and an `^GSPC` benchmark), with an offline browser dashboard. Training runs on the BYU supercomputer (Slurm, see [section 2b](#2b-training-on-the-supercomputer)); the dashboard and prediction server run locally. No paid services, no build step.

---

## 1. Quick start (local: view results, no training)

```bash
git pull                                    # fetch results/ + saved_models/ produced on the supercomputer
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python server.py --open
```

This serves the trained results at <http://127.0.0.1:8080/dashboard.html> and a live prediction API ([section 3b](#3b-prediction-api)). Training itself runs on the cluster ([section 2b](#2b-training-on-the-supercomputer)).

---

## 2. Everyday commands

| Goal | Command |
| --- | --- |
| Run all 5 tickers + write report | `python run_experiment.py` |
| Run a single ticker (faster) | `python run_experiment.py --ticker NVDA` |
| List supported tickers | `python run_experiment.py --list-tickers` |
| Skip dashboard payload | `python run_experiment.py --skip-dashboard` |
| Skip report regeneration | `python run_experiment.py --skip-report` |
| Regenerate the report only | `python report.py` |
| Start the dashboard server | `python server.py` |
| Start & open the browser | `python server.py --open` |
| Use a different port | `python server.py --port 9000` |

### Deep-learning experiments

`dl_experiments.py` sweeps recurrent-model variants and appends one row per run to `results/dl_experiments.csv`, with per-epoch train/validation loss curves in `results/dl_curves/`.

| Command | Varies |
| --- | --- |
| `python dl_experiments.py --grid baseline --ticker NVDA` | The four `run_experiment.py` architectures |
| `python dl_experiments.py --grid cells` | RNN vs. GRU vs. LSTM |
| `python dl_experiments.py --grid size` | Layer count and width |
| `python dl_experiments.py --grid lookback` | Window length (5, 7, 21, 63 days) |
| `python dl_experiments.py --grid features` | Returns only vs. technical vs. all 35 features |
| `python dl_experiments.py --grid regularization` | Dropout, weight decay, learning rate |
| `python dl_experiments.py --grid all` | Everything above |

Omit `--ticker` to run all five tickers. Add new variants by editing `GRIDS` in the script. Compare `test_ret_rmse` against `test_zero_ret_rmse` (always predicting a 0% move), and check `test_frac_pred_up` and `test_corr`: a run that predicts "up" on ~100% of days with near-zero correlation has collapsed to the average return.

---

## 2b. Training on the supercomputer

All training runs on the cluster, not locally. Compute nodes may have no internet, so every input (Yahoo prices and FRED macro data) is cached in `data/` and committed to the repo; `fetch_data.py` refreshes it.

```bash
# on the login node
git clone https://github.com/CHAOLIN99/stock-test-2.git && cd stock-test-2
bash scripts/setup.sh                       # uv venv (Python 3.12) + requirements + data caches

sbatch scripts/sbatch_train.sh              # full pipeline, all tickers, one A100 job
sbatch scripts/sbatch_dl_experiments.sh     # DL sweep; pass a grid name, e.g. ... sbatch_dl_experiments.sh lookback

squeue -u $USER                             # job status; logs in slurm_logs/
```

When the job finishes, commit and push `results/`, `saved_models/`, and `report.md` from the cluster, then `git pull` locally and run `python server.py --open` to view the dashboard.

Deep-learning models run on CUDA when available (then MPS, then CPU). On Python 3.12 the real XGBoost is used; the `HistGradientBoostingRegressor` stand-in only applies on Python 3.14+.

---

## 3. Using the dashboard

Open `http://127.0.0.1:8080/dashboard.html`. The URL's hash decides the page, so everything is bookmarkable:

| Route | What it shows |
| --- | --- |
| `#/overview` | Cross-ticker best-model summary, RMSE bar chart, directional-accuracy heatmap |
| `#/ticker/NVDA` | Per-ticker price vs. predictions, equity curves, feature importance, regime timeline |
| `#/compare` | Small-multiples price charts (one per ticker) |
| `#/methodology` | System architecture + guarantees |

### Keyboard shortcuts

| Keys | Action |
| --- | --- |
| `g` then `o` / `t` / `c` / `m` | Jump to Overview / Ticker / Compare / Methodology |
| `←` / `→` / `Home` / `End` on ticker pills | Move between tickers |
| `Shift` + mouse wheel on a chart | Zoom the x-axis |
| Pinch / drag on trackpad | Pan/zoom the x-axis |
| `Enter` in a date input | Apply the date range |

### Controls in the header

- **Dark** toggle — remembered in `localStorage`; falls back to OS preference.
- **Regimes** toggle — overlays HMM regime bands on the price chart.
- **Date range** — filters all per-ticker charts. Non-trading dates snap to the nearest trading day automatically.
- **Export PNG** — downloads the current price or RMSE chart as PNG.
- **Reset zoom** — only visible after you've zoomed/panned.

### Loading a different dataset

Append a `data` query param: `dashboard.html?data=/path/to/other_charts_data.json`. The file must match the schema described in [`results/charts_data.json`](results/charts_data.json).

---

## 3b. Prediction API

`python server.py` also exposes a JSON API for live next-day forecasts.

**What the forecast means:** the server downloads fresh daily prices and FRED macro data, rebuilds the 35 training features, and runs the saved Tier-3 models plus the stacking ensemble on the latest trading day. The headline number is the stacking ensemble. **BUY/SELL** appears only when it expects a move over 0.25% *and* at least 5 of 8 models agree (at most 3 of 8 for SELL); otherwise **HOLD**. Backtest hit-rate and regime come from the last `run_experiment.py` run. Only the five trained tickers are covered; others return "not covered". This is educational output, not financial advice.

**API:**

| Route | Returns |
| --- | --- |
| `GET /api/health` | `{"ok": true}` |
| `GET /api/tickers` | Covered tickers |
| `GET /api/predict/<TICKER>` | Live forecast, per-model forecasts, backtest stats |

Prices are cached for 15 minutes and macro data for 6 hours, so the first request for a ticker takes a few seconds and later ones are instant.

---

## 4. What gets produced

| Path | Purpose |
| --- | --- |
| `data/raw_<TICKER>.parquet` | Cached Yahoo Finance OHLCV |
| `saved_models/` | Fitted scalers, Tier-3 models, and stacking meta-models (joblib) |
| `results/results_all.csv` | Long-form metrics per (ticker × model × split) |
| `results/meta_<TICKER>.json` | Per-ticker pipeline metadata (hyperparams, runtime) |
| `results/charts/` | Static matplotlib charts (price, SHAP when available) |
| `results/charts_data.json` | **Dashboard payload** — the only file the frontend reads |
| `report.md` | Markdown report summarizing metrics and choices |

---

## 5. Models

- **Baselines (Tier 0):** Naive (tomorrow = today's close), Buy & Hold, SMA 50/200 cross
- **Classical (Tier 1–2):** Holt–Winters, ARIMA on log returns
- **Multivariate ML (Tier 3):** Ridge, MARS (if `pyearth` available), RandomForest, SVR (row-capped), MLP, XGBoost-like, LightGBM
- **Deep learning (Tier 4):** RNN, LSTM-A, LSTM-B, BiLSTM (PyTorch, MPS if available). Each predicts the next-day log return from a 7-day window of the same 35 scaled features as Tier 3, one step ahead from observed data. Early stopping uses the last 15% of the training period as a chronological validation set and restores the best weights.
- **HMM:** `GaussianHMM` regime labels + regime-conditional next-day forecast
- **Ensembles (Tier 5):** weighted voting (top-3 by OOF RMSE), stacking (Ridge meta-learner)

### Train / test splits

- `NVDA / AMD / MSFT / QCOM`: train `2000-01-02` → `2024-12-31`
- `META`: train `2012-05-18` → `2024-12-31` (no pre-IPO padding)
- All: test `2025-01-01` → present

---

## 6. Troubleshooting

| Symptom | Fix |
| --- | --- |
| Dashboard shows `Dashboard data not found … HTTP 404` | Run `python run_experiment.py` to generate `results/charts_data.json`, then reload. |
| Browser caches an old version | The server sets `Cache-Control: no-store`, but a hard reload (`⌘⇧R`) forces it. |
| Port 8080 already in use | `python server.py --port 9000` (or set `PORT=9000`). |
| `XGBoost segfaulted` on Python 3.14 | Expected — `models/classical.py` transparently substitutes `HistGradientBoostingRegressor` and keeps the label `XGBoost`. |
| Job fails with a network error on a compute node | Run `python fetch_data.py` on the login node so `data/` holds every price and FRED cache. |
| Regime overlay shows nothing | The ticker's `charts_data.json` must include `regime` + `regime_names`; an empty HMM run leaves them `null`. |

---

## 7. Project layout

```
.
├── config.py              # tickers, date ranges, hyperparam defaults
├── data_pipeline.py       # Yahoo download, FRED macro, feature frames, train/test split
├── fetch_data.py          # pre-download all inputs into data/ (for offline compute nodes)
├── run_experiment.py      # orchestrates training + evaluation + dashboard export
├── dl_experiments.py      # deep-learning experiment sweeps
├── scripts/               # setup.sh + Slurm job scripts (sbatch_*.sh)
├── evaluate.py            # metric computation (RMSE/mean, dir acc, Sharpe, drawdown, R²)
├── report.py              # renders report.md from results/results_all.csv
├── server.py              # local server: dashboard files + /api/* prediction routes
├── predict_service.py     # live next-day forecasts from saved models
├── models/
│   ├── classical.py       # Ridge, MARS, RF, SVR, MLP, XGBoost, LightGBM
│   ├── deep_learning.py   # RNN / LSTM variants (PyTorch)
│   ├── ensemble.py        # voting + stacking
│   ├── hmm_regime.py      # Gaussian HMM regime model
│   └── timeseries.py      # Holt-Winters, ARIMA
├── dashboard.html         # dashboard shell (no build step)
├── css/                   # @layer base/components/utilities
├── js/
│   ├── utils.js           # pure helpers (format, clamp, debounce, raf-throttle)
│   ├── data-loader.js     # fetch + validate + one-time preprocess
│   ├── components.js      # DOM components (diff-updating)
│   ├── charts.js          # Chart.js wrappers (upsert-in-place)
│   └── app.js             # state, routing, render dispatch
├── data/                  # cached prices + FRED parquet
├── saved_models/          # scalers, Tier-3 and stacking models
└── results/               # metrics + dashboard payload + static PNG charts
```

---

## 8. Design notes (frontend)

The dashboard is intentionally a no-build vanilla JS app. Smoothness comes from:

- **Scoped render dispatch** — `DIRTY.{DATA,ROUTE,TICKER,MODELS,DATE,REGIME,THEME}` flags are OR-ed into a single RAF-batched paint. Toggling a model never re-renders the whole page.
- **In-place chart updates** — `upsertChart` swaps `chart.data` / `chart.options` then calls `chart.update("none")` instead of `destroy()` + `new Chart()`.
- **Memoized slicing** — per-ticker date-range slices are cached in an LRU of 32; non-trading dates snap to the nearest trading day via a binary search on the dates array.
- **Lazy ticker sections** — only the initial ticker's DOM is built up front; others are cloned from a `<template>` on first visit.
- **Event-driven zoom reset** — no polling; the Chart.js zoom plugin's `onZoomComplete` / `onPanComplete` drive the Reset button's visibility.
- **CSS var cache** — invalidated on theme flip; chart colors are then re-derived without recreating charts.

No PostgreSQL, no Node, no React: results are static JSON, there's nothing to persist, and the component tree is small enough that a framework would add cost without paying it back.
