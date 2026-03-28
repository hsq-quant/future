from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.article_sampling import sample_articles_by_month
from src.utils.io import read_dataframe, write_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample cleaned articles by calendar month.")
    parser.add_argument("--input", type=str, required=True, help="Full clean-article parquet/csv path.")
    parser.add_argument("--output", type=str, required=True, help="Sampled clean-article output path.")
    parser.add_argument("--sample-frac", type=float, default=0.5, help="Within-month sample fraction.")
    parser.add_argument(
        "--max-articles",
        type=int,
        default=0,
        help="Optional global cap applied after monthly stratified sampling. Use 0 for no cap.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for deterministic sampling.")
    parser.add_argument(
        "--timestamp-column",
        type=str,
        default="published_at",
        help="Timestamp column used for monthly stratification.",
    )
    parser.add_argument(
        "--ensure-weekly-coverage",
        action="store_true",
        help="Backfill one article for any week present in the full clean set but absent after sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    articles = read_dataframe(args.input)
    sampled = sample_articles_by_month(
        articles,
        sample_frac=args.sample_frac,
        max_articles=args.max_articles if args.max_articles and args.max_articles > 0 else None,
        random_state=args.random_state,
        timestamp_column=args.timestamp_column,
        ensure_weekly_coverage=args.ensure_weekly_coverage,
    )
    write_dataframe(sampled, args.output)
    month_counts = sampled[args.timestamp_column].pipe(lambda s: s.dt.to_period("M").astype(str).value_counts().sort_index())
    print(args.output)
    print(f"input_rows={len(articles)}")
    print(f"sampled_rows={len(sampled)}")
    print(month_counts.to_string())


if __name__ == "__main__":
    main()
