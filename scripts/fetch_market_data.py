from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.market_source import MarketFetchConfig, load_market_data
from src.utils.io import read_yaml, write_dataframe


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and normalize INE market data.")
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
    market_cfg = config["market"]
    fetch_cfg = MarketFetchConfig(
        source=market_cfg["source"],
        start_date=market_cfg["start_date"],
        end_date=market_cfg["end_date"],
        exchange=market_cfg["exchange"],
        local_input_glob=str(root / market_cfg["local_input_glob"]),
        contract_metadata_file=str(root / market_cfg["contract_metadata_file"]),
    )
    market_df = load_market_data(fetch_cfg)
    write_dataframe(market_df, root / "data/intermediate/market_daily.parquet")
    print(root / "data/intermediate/market_daily.parquet")


if __name__ == "__main__":
    main()
