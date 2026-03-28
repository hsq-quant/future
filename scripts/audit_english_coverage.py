from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.english_coverage_audit import summarize_english_coverage
from src.utils.io import read_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit English weekly coverage against supervised weeks.")
    parser.add_argument(
        "--clean-input",
        type=str,
        default=str(ROOT / "data/intermediate/articles_archive_clean_english.parquet"),
    )
    parser.add_argument(
        "--weekly-labels",
        type=str,
        default=str(ROOT / "data/processed/weekly_labels.parquet"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "reports/english_coverage_audit.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean = read_dataframe(args.clean_input)
    weekly_labels = read_dataframe(args.weekly_labels)[["week_end_date"]]
    summary = summarize_english_coverage(clean, weekly_labels)

    by_year = clean.copy()
    by_year["published_at"] = pd.to_datetime(by_year["published_at"], errors="coerce", utc=True)
    yearly = by_year.dropna(subset=["published_at"]).assign(year=lambda df: df["published_at"].dt.year).groupby("year").size()

    lines = [
        "# English Coverage Audit",
        "",
        f"- Total Articles: {summary['total_articles']}",
        f"- Supervised Weeks: {summary['supervised_weeks']}",
        f"- Weeks With >=1 Article: {summary['covered_weeks_ge_1']}",
        f"- Weeks With >=3 Articles: {summary['covered_weeks_ge_3']}",
        f"- Longest Zero-Article Gap (weeks): {summary['longest_zero_article_gap_weeks']}",
        f"- Coverage Window: {summary['min_week']} ~ {summary['max_week']}",
        "",
        "## Articles By Year",
    ]
    for year, count in yearly.items():
        lines.append(f"- {int(year)}: {int(count)}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
