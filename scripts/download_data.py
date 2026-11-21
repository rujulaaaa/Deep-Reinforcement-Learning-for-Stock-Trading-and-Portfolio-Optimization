"""Download NIFTY50 OHLCV data (cached in data/raw/) and print the split summary.

    python scripts/download_data.py [--force]
"""

from common import base_parser, get_config

from finsearch.data import FEATURES, build_dataset, describe

if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--force", action="store_true", help="re-download even if cached")
    args = p.parse_args()
    cfg = get_config(args)
    splits = build_dataset(cfg, force_download=args.force)
    print(describe(splits).to_string())
    total = sum(len(d.dates) - d.trade_start for d in splits.values())
    n = splits["train"].n_assets
    print(f"\n{total} trading days x {n} stocks = {total * n:,} stock-days "
          f"({total * n * 5:,} OHLCV values) -> {len(FEATURES)} features per stock-day")
