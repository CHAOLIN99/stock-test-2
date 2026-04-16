"""
Generate report.md from results/results_all.csv and per-ticker metadata.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import RESULTS_DIR, ROOT


def generate_report() -> Path:
    csv_path = RESULTS_DIR / "results_all.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Run run_experiment.py first — missing {csv_path}")

    df = pd.read_csv(csv_path)
    lines: list[str] = []
    lines.append("# Stock price prediction — out-of-sample 2025+ report\n")
    lines.append("## Methodology\n")
    lines.append(
        "- **Data**: Yahoo Finance (Adj Close via `auto_adjust=True`), FRED macro features for multivariate ML.\n"
    )
    lines.append(
        "- **Split**: train through 2024-12-31 (META from IPO window); test from 2025-01-01 onward.\n"
    )
    lines.append(
        "- **Primary metric**: RMSE / mean(|Adj Close|) on the test window; bootstrap 95% CIs (n=200).\n"
    )
    lines.append(
        "- **Secondary**: 5-day direction uses a 5-step chain of one-day log-return forecasts for Tier-3 models.\n"
    )
    lines.append(
        "- **Implementation note**: On Python 3.14+, the XGBoost wheel can segfault on some real feature matrices; "
        "the code uses `sklearn.ensemble.HistGradientBoostingRegressor` with comparable depth/iterations as a "
        "histogram-boosting stand-in (same `XGBoost` label in outputs).\n"
    )
    lines.append("\n## Limitations (required)\n")
    lines.append(
        "- **ARIMA / HW**: Gaussian innovations assumption; volatility clustering in tech stocks violates constant variance.\n"
    )
    lines.append(
        "- **ANN / LSTM**: Black-box risk per Polamuri et al.; mitigated partially via SHAP on tree models.\n"
    )
    lines.append(
        "- **RNN**: Vanishing gradients limit effective memory to ~10 steps despite longer lookback windows.\n"
    )
    lines.append(
        "- **All models**: No news, earnings surprises, or macro shock dummies — structural breaks remain unexplained.\n"
    )

    lines.append("\n## Best model per ticker (lowest test RMSE/mean)\n")
    best = df.loc[df.groupby("ticker")["rmse_mean_test"].idxmin()]
    lines.append(best[["ticker", "model", "rmse_mean_test", "dir_acc", "cls_acc"]].to_markdown(index=False))
    lines.append("\n")

    lines.append("## Full results (all models)\n")
    for t in sorted(df["ticker"].unique()):
        sub = df[df["ticker"] == t].sort_values("rmse_mean_test")
        lines.append(f"### {t}\n")
        cols = [
            "model",
            "rmse_mean_train",
            "rmse_mean_test",
            "bootstrap_rmse_low",
            "bootstrap_rmse_high",
            "mae",
            "rmse_dollar",
            "r2",
            "dir_acc",
            "ic",
            "icir",
            "cls_acc",
            "cls_f1",
            "cls_auc",
            "ann_return",
            "sharpe",
            "max_dd",
            "calmar",
            "sharpe_vs_bench",
        ]
        cols = [c for c in cols if c in sub.columns]
        lines.append(sub[cols].to_markdown(index=False))
        lines.append("\n")

    lines.append("## Honest comparison\n")
    lines.append(
        "If no model beats **Naive** on RMSE/mean for a ticker, that is reported in the table — "
        "markets are hard and short out-of-sample windows are noisy.\n"
    )

    out = ROOT / "report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


if __name__ == "__main__":
    p = generate_report()
    print(f"Wrote {p}")
