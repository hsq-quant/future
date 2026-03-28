from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.rss_ingest import (
    allowed_language_prefixes_from_feeds,
    clean_articles,
    ingest_rss_feeds,
    load_feed_config,
)
from src.utils.io import read_dataframe, read_yaml, write_dataframe


def _resolve_date_window(root: Path, args: argparse.Namespace) -> tuple[str | None, str | None]:
    if args.start_date or args.end_date:
        return args.start_date, args.end_date

    market_config = read_yaml(root / "configs/market.yaml")
    market = market_config["market"]
    start_date = (
        pd.Timestamp(market["start_date"]) - pd.Timedelta(days=int(market.get("news_lookback_days", 7)))
    ).strftime("%Y-%m-%d")
    weekly_labels_path = root / market["processed_output"]
    if weekly_labels_path.exists():
        weekly_labels = read_dataframe(weekly_labels_path)
        end_date = pd.Timestamp(weekly_labels["week_end_date"].max()).strftime("%Y-%m-%d")
    else:
        end_date = market["end_date"]
    return start_date, end_date


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Chinese RSS feeds for the INE SC project.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs/feeds.yaml"),
        help="Feed configuration YAML path.",
    )
    parser.add_argument("--max-entries-per-feed", type=int, default=None, help="Small-batch test cap per feed.")
    parser.add_argument("--start-date", type=str, default=None, help="Inclusive start date in YYYY-MM-DD.")
    parser.add_argument("--end-date", type=str, default=None, help="Inclusive end date in YYYY-MM-DD.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = ROOT
    start_date, end_date = _resolve_date_window(root, args)
    raw_articles = ingest_rss_feeds(
        args.config,
        max_entries_per_feed=args.max_entries_per_feed,
    )
    write_dataframe(raw_articles, root / "data/intermediate/articles_raw.parquet")
    feeds, keywords = load_feed_config(args.config)
    allowed_language_prefixes = allowed_language_prefixes_from_feeds(feeds)
    trading_calendar = read_dataframe(root / "data/intermediate/continuous_daily.parquet")[["trade_date"]].drop_duplicates()
    cleaned = clean_articles(
        raw_articles,
        trading_calendar=trading_calendar,
        energy_keywords=keywords,
        start_date=start_date,
        end_date=end_date,
        allowed_language_prefixes=allowed_language_prefixes,
    )
    write_dataframe(cleaned, root / "data/intermediate/articles_clean.parquet")
    print(root / "data/intermediate/articles_clean.parquet")
    print(f"date_window={start_date}..{end_date}")
    print(f"raw_articles={len(raw_articles)}")
    print(f"clean_articles={len(cleaned)}")


if __name__ == "__main__":
    main()
