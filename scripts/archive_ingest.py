from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.archive_ingest import run_archive_ingest
from src.data.rss_ingest import clean_articles, load_feed_config
from src.utils.io import read_dataframe, read_yaml, write_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run archive-first article ingestion with fallback providers.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs/archive.yaml"))
    parser.add_argument("--max-articles", type=int, default=None)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = ROOT
    archive_cfg = read_yaml(args.config).get("archive", {})

    raw_articles, manifest = run_archive_ingest(
        config_path=args.config,
        max_articles=args.max_articles,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    write_dataframe(raw_articles, root / "data/intermediate/articles_archive_raw.parquet")
    write_dataframe(manifest, root / "data/intermediate/archive_manifest.csv")

    _, fallback_keywords = load_feed_config(str(root / "configs/feeds.yaml"))
    keywords = archive_cfg.get("keyword_filters", fallback_keywords)
    allowed_language_prefixes = tuple(archive_cfg.get("allowed_language_prefixes", ["zh"]))
    trading_calendar = read_dataframe(root / "data/intermediate/continuous_daily.parquet")[["trade_date"]].drop_duplicates()
    cleaned = clean_articles(
        raw_articles,
        trading_calendar=trading_calendar,
        energy_keywords=keywords,
        start_date=args.start_date,
        end_date=args.end_date,
        allowed_language_prefixes=allowed_language_prefixes,
    )
    write_dataframe(cleaned, root / "data/intermediate/articles_archive_clean.parquet")

    print(root / "data/intermediate/articles_archive_clean.parquet")
    print(f"raw_articles={len(raw_articles)}")
    print(f"clean_articles={len(cleaned)}")
    if not manifest.empty:
        print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
