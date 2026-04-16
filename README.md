# Stock Prediction Dashboard

End-to-end, reproducible next-day price prediction for **META, NVDA, AMD, MSFT, QCOM** (and an `^GSPC` benchmark), with an offline browser dashboard. Runs locally on a MacBook Air (Apple Silicon, MPS) — no paid services, no build step.

---

## 1. Quick start (4 commands)

```bash
cd "/Users/weichao/Desktop/stock test 2"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_experiment.py && python server.py --open
```

The fourth command trains every model, writes `results/charts_data.json`, starts the local server, and opens the dashboard at <http://127.0.0.1:8080/dashboard.html>.

> First run downloads daily OHLCV from Yahoo Finance (cached to `data/raw_<TICKER>.parquet`). Subsequent runs load from parquet and are much faster.

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

You can also set `SINGLE_TICKER=NVDA python run_experiment.py` if you prefer env vars (overridden by `--ticker`).

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

## 4. What gets produced

| Path | Purpose |
| --- | --- |
| `data/raw_<TICKER>.parquet` | Cached Yahoo Finance OHLCV |
| `saved_models/` | Fitted scalers + torch state dicts |
| `results/results_all.csv` | Long-form metrics per (ticker × model × split) |
| `results/meta_<TICKER>.json` | Per-ticker pipeline metadata (hyperparams, runtime) |
| `results/charts/` | Static matplotlib charts (price, SHAP when available) |
| `results/charts_data.json` | **Dashboard payload** — the only file the frontend reads |
| `report.md` | Markdown report summarizing metrics and choices |

---

## 5. Models

- **Baselines (Tier 0):** Naive (t-1), Buy & Hold, SMA 50/200 cross
- **Classical (Tier 1–2):** Holt–Winters, ARIMA on log returns
- **Multivariate ML (Tier 3):** Ridge, MARS (if `pyearth` available), RandomForest, SVR (row-capped), MLP, XGBoost-like, LightGBM
- **Deep learning (Tier 4):** RNN, LSTM-A, LSTM-B, BiLSTM (PyTorch, MPS if available)
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
| MPS errors on deep-learning models | Device selection in `models/deep_learning.py` falls back to CPU automatically if MPS is unavailable. |
| Regime overlay shows nothing | The ticker's `charts_data.json` must include `regime` + `regime_names`; an empty HMM run leaves them `null`. |

---

## 7. Project layout

```
.
├── config.py              # tickers, date ranges, hyperparam defaults
├── data_pipeline.py       # Yahoo download, FRED macro, feature frames, train/test split
├── run_experiment.py      # orchestrates training + evaluation + dashboard export
├── evaluate.py            # metric computation (RMSE/mean, dir acc, Sharpe, drawdown, R²)
├── report.py              # renders report.md from results/results_all.csv
├── server.py              # local static server (no-store headers, CORS opt-in)
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
├── data/                  # cached parquet
├── saved_models/          # scalers + torch weights
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
