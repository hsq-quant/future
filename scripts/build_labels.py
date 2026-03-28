from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.market_labels import build_weekly_labels, stitch_continuous_near_month
from src.utils.io import read_dataframe, read_yaml, write_dataframe


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build weekly labels from normalized INE market data.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs/market.yaml"),
        help="Market configuration YAML path.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    root = ROOT
    config = read_yaml(args.config)
    market_daily = read_dataframe(root / "data/intermediate/market_daily.parquet")
    stitched = stitch_continuous_near_month(
        market_daily,
        roll_days_before_last_trade=config["market"]["roll_days_before_last_trade"],
    )
    weekly_labels = build_weekly_labels(stitched)
    write_dataframe(stitched, root / "data/intermediate/continuous_daily.parquet")
    write_dataframe(weekly_labels, root / config["market"]["processed_output"])
    print(root / config["market"]["processed_output"])


if __name__ == "__main__":
    main()
