from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.external_news_providers import fetch_akshare_news
from src.utils.io import read_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch historical Chinese finance news via AKShare.")
    parser.add_argument("--config", default="configs/external_news.yaml")
    parser.add_argument("--keyword", action="append", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = read_yaml(config_path).get("external_news", {})
    ak_cfg = config.get("akshare", {})

    start_date = args.start_date or config.get("start_date", "2019-12-30")
    end_date = args.end_date or config.get("end_date", "2026-03-13")
    keywords = args.keyword or ak_cfg.get("keywords", ["原油"])
    output_dir = ROOT / ak_cfg.get("output_dir", "data/raw/archive/akshare")
    output = Path(args.output) if args.output else output_dir / "akshare_news.parquet"
    if not output.is_absolute():
        output = ROOT / output

    frames: list[pd.DataFrame] = []
    for keyword in keywords:
        frame = fetch_akshare_news(keyword, start_date=start_date, end_date=end_date)
        if not frame.empty:
            frame = frame.copy()
            frame["retrieval_keyword"] = keyword
            frames.append(frame)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty:
        combined = combined.drop_duplicates(subset=["url", "title", "published_at"]).sort_values(
            ["published_at", "title"]
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output, index=False)
    print(f"rows={len(combined)}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
