#!/usr/bin/env python3
"""
Download every input the pipeline needs into data/ so training can run without internet
(e.g. on supercomputer compute nodes). Run on a machine/login node with internet access.

Run:  python fetch_data.py            # fill missing caches
      python fetch_data.py --refresh  # re-download everything
"""

from __future__ import annotations

import argparse

from config import TICKERS
from data_pipeline import FRED_CACHE, download_fred, download_yahoo


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--refresh", action="store_true", help="Re-download even if cached files exist")
    args = p.parse_args()

    start_all = min(TICKERS.values())
    for ticker, start in [*TICKERS.items(), ("^GSPC", start_all)]:
        df, source = download_yahoo(ticker, start_date=start, refresh=args.refresh)
        print(f"{ticker:<6} {len(df):>5} rows  {df.index[0].date()} -> {df.index[-1].date()}  ({source})")

    if args.refresh or not FRED_CACHE.exists():
        download_fred()
    print(f"FRED   cached at {FRED_CACHE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
