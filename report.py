"""
Generate report.md from results/results_all.csv and per-ticker metadata.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from config import RESULTS_DIR, ROOT

REPORT_COLUMNS = [
    "model", "rmse_mean_train", "rmse_mean_test", "bootstrap_rmse_low", "bootstrap_rmse_high",
    "mae", "rmse_dollar", "r2", "dir_acc", "ic", "icir", "cls_acc", "cls_f1", "cls_auc",
    "ann_return", "sharpe", "max_dd", "calmar", "sharpe_vs_bench",
]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="report.py", description="Generate report.md from results/results_all.csv.")
    p.add_argument(
        "--results-dir",
        default=str(RESULTS_DIR),
        help="Directory containing results_all.csv (default: results/).",
    )
    p.add_argument(
        "--out",
        default=str(ROOT / "report.md"),
        help="Output markdown path (default: ./report.md).",
    )
    return p.parse_args(argv)


def generate_report(*, results_dir: str | Path = RESULTS_DIR, out_path: str | Path = ROOT / "report.md") -> Path:
    results_dir = Path(results_dir)
    out_path = Path(out_path)
    csv_path = results_dir / "results_all.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Run run_experiment.py first — missing {csv_path}")

    df = pd.read_csv(csv_path)
    lines = [
        "# Stock price prediction — out-of-sample 2025+ report\n",
        "## Methodology\n",
        "- **Data**: Yahoo Finance (Adj Close via `auto_adjust=True`), FRED macro features for multivariate ML.",
        "- **Split**: train through 2024-12-31 (META from IPO window); test from 2025-01-01 onward.",
        "- **Primary metric**: RMSE / mean(|Adj Close|) on the test window; bootstrap 95% CIs (n=200).",
        "- **Naive baseline**: tomorrow's close = today's close (persistence).",
        "- **Deep learning**: RNN / LSTM / BiLSTM predict the next-day log return from a 7-day window of the "
        "same 39 scaled features as Tier 3, one step ahead from observed data; early stopping on a chronological "
        "validation set (last 15% of training) with best-weight restore.",
        "- **Secondary**: 5-day direction is the sign of the sum of five consecutive one-day log-return "
        "forecasts from the Tier-3 models.",
        "- **Implementation note**: on Python 3.14+ the XGBoost wheel can segfault, so `XGBoost` is backed by "
        "`sklearn.ensemble.HistGradientBoostingRegressor` with comparable depth/iterations.",
        "\n## Limitations\n",
        "- **ARIMA / HW**: Gaussian innovations assumption; volatility clustering in tech stocks violates constant variance.",
        "- **ANN / LSTM**: Black-box risk per Polamuri et al.; mitigated partially via SHAP on tree models.",
        "- **RNN**: Vanishing gradients limit effective memory to ~10 steps despite longer lookback windows.",
        "- **All models**: No news, earnings surprises, or macro shock dummies — structural breaks remain unexplained.",
    ]

    lines.append("\n## Best model per ticker (lowest test RMSE/mean)\n")
    best = df.loc[df.groupby("ticker")["rmse_mean_test"].idxmin()]
    lines.append(best[["ticker", "model", "rmse_mean_test", "dir_acc", "cls_acc"]].to_markdown(index=False))
    lines.append("\n")

    lines.append("## Full results (all models)\n")
    for t in sorted(df["ticker"].unique()):
        sub = df[df["ticker"] == t].sort_values("rmse_mean_test")
        lines.append(f"### {t}\n")
        cols = [c for c in REPORT_COLUMNS if c in sub.columns]
        lines.append(sub[cols].to_markdown(index=False))
        lines.append("\n")

    lines.append("## Honest comparison\n")
    lines.append(
        "If no model beats **Naive** on RMSE/mean for a ticker, that is reported in the table — "
        "markets are hard and short out-of-sample windows are noisy.\n"
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    p = generate_report(results_dir=args.results_dir, out_path=args.out)
    print(f"Wrote {p}")
