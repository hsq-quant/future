from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from newsplease.crawler.commoncrawl_crawler import crawl_from_commoncrawl

from src.data.newsplease_archive import (
    append_jsonl_records,
    article_matches_filters,
    articles_to_jsonl_records,
    ensure_nltk_resources,
)
from src.data.rss_ingest import load_feed_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch historical news from Common Crawl via news-please.")
    parser.add_argument("--start-date", required=True, help="Article publish start bound, YYYY-MM-DD or ISO datetime.")
    parser.add_argument("--end-date", required=True, help="Article publish end bound, YYYY-MM-DD or ISO datetime.")
    parser.add_argument(
        "--warc-start-date",
        default=None,
        help="WARC listing start bound, YYYY-MM-DD or ISO datetime. Defaults to start-date.",
    )
    parser.add_argument(
        "--warc-end-date",
        default=None,
        help="WARC listing end bound, YYYY-MM-DD or ISO datetime. Defaults to end-date + 1 day.",
    )
    parser.add_argument("--output", default=str(ROOT / "data/raw/archive/newsplease/commoncrawl_sample.jsonl"))
    parser.add_argument("--download-dir", default=str(ROOT / "data/tmp/cc_warc"))
    parser.add_argument("--max-articles", type=int, default=20)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument(
        "--valid-host",
        action="append",
        default=[],
        help="Optional host filter, can be passed multiple times, e.g. --valid-host caixin.com",
    )
    parser.add_argument("--show-download-progress", action="store_true")
    parser.add_argument("--delete-warc-after-extraction", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _as_datetime(value: str) -> dt.datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.to_pydatetime()


def main() -> None:
    args = parse_args()
    _, keywords = load_feed_config(str(ROOT / "configs/feeds.yaml"))
    ensure_nltk_resources(ROOT / "data/tmp/nltk")

    start_date = _as_datetime(args.start_date)
    end_date = _as_datetime(args.end_date)
    warc_start_date = _as_datetime(args.warc_start_date) if args.warc_start_date else start_date
    warc_end_date = _as_datetime(args.warc_end_date) if args.warc_end_date else end_date + dt.timedelta(days=1)

    saved_records: list[dict[str, object]] = []
    max_articles = max(args.max_articles, 1)
    log_level = logging.INFO if args.verbose else logging.WARNING

    def on_valid_article_extracted(article: object) -> None:
        article_dict = getattr(article, "__dict__", {})
        if not article_matches_filters(article_dict, keywords=keywords, language_prefix="zh"):
            return
        saved_records.extend(articles_to_jsonl_records([article_dict]))
        if len(saved_records) >= max_articles:
            raise KeyboardInterrupt("Reached max_articles limit for sample run.")

    def on_warc_completed(*_: object) -> None:
        return None

    try:
        crawl_from_commoncrawl(
            on_valid_article_extracted,
            callback_on_warc_completed=on_warc_completed,
            valid_hosts=args.valid_host,
            start_date=start_date,
            end_date=end_date + dt.timedelta(days=1),
            warc_files_start_date=warc_start_date,
            warc_files_end_date=warc_end_date,
            strict_date=True,
            reuse_previously_downloaded_files=True,
            local_download_dir_warc=args.download_dir,
            continue_after_error=True,
            show_download_progress=args.show_download_progress,
            number_of_extraction_processes=args.processes,
            log_level=log_level,
            delete_warc_after_extraction=args.delete_warc_after_extraction,
            continue_process=False,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        pass

    if saved_records:
        append_jsonl_records(saved_records[:max_articles], args.output)

    print(args.output)
    print(f"saved_records={len(saved_records[:max_articles])}")
    print(f"article_window={args.start_date}..{args.end_date}")
    print(f"warc_window={warc_start_date.date()}..{warc_end_date.date()}")
    print(f"valid_hosts={args.valid_host}")


if __name__ == "__main__":
    main()
