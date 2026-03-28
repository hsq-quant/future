from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.hf_financial_multisource import (
    normalize_hf_financial_multisource,
    normalize_hf_financial_news_2024,
)
from src.utils.io import read_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize a local HF financial-news export into archive format.")
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument(
        "--dataset-kind",
        choices=["multisource", "financial-news-2024"],
        default="multisource",
    )
    parser.add_argument("--start-date", default="2019-12-30")
    parser.add_argument("--end-date", default="2026-03-13")
    parser.add_argument("--output", default="data/raw/archive/english_global/financial_news_multisource.parquet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames: list[pd.DataFrame] = []
    normalize = (
        normalize_hf_financial_multisource
        if args.dataset_kind == "multisource"
        else normalize_hf_financial_news_2024
    )
    for input_path in args.input:
        path = Path(input_path)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            continue
        frames.append(
            normalize(
                read_dataframe(path),
                start_date=args.start_date,
                end_date=args.end_date,
            )
        )

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty:
        combined = combined.drop_duplicates(subset=["url", "title", "published_at", "source"]).sort_values(
            ["published_at", "title"]
        )

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output, index=False)
    print(f"rows={len(combined)}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
