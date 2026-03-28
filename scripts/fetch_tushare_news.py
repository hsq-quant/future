from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.external_news_providers import (
    fetch_tushare_major_news_batched,
    fetch_tushare_news_batched,
)
from src.utils.io import read_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch historical Chinese finance news via Tushare Pro.")
    parser.add_argument("--config", default="configs/external_news.yaml")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--mode", choices=["news", "major_news", "both"], default="both")
    parser.add_argument("--source", action="append", default=None)
    parser.add_argument("--keyword", action="append", default=None)
    parser.add_argument("--batch-days", type=int, default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _get_token() -> str:
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise SystemExit("Missing TUSHARE_TOKEN")
    return token


def main() -> None:
    args = parse_args()
    token = _get_token()
    config_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = read_yaml(config_path).get("external_news", {})
    ts_cfg = config.get("tushare", {})

    start_date = args.start_date or config.get("start_date", "2019-12-30")
    end_date = args.end_date or config.get("end_date", "2026-03-13")
    output_dir = ROOT / ts_cfg.get("output_dir", "data/raw/archive/tushare")
    batch_days = int(args.batch_days or ts_cfg.get("batch_days", 120))
    keywords = args.keyword or ts_cfg.get("keyword_filters", [])
    output = Path(args.output) if args.output else output_dir / "tushare_news.parquet"
    if not output.is_absolute():
        output = ROOT / output

    frames: list[pd.DataFrame] = []
    if args.mode in {"news", "both"}:
        sources = args.source or ts_cfg.get("news", {}).get("sources", [])
        limit = int(ts_cfg.get("news", {}).get("limit_per_call", 1500))
        for source in sources:
            frame = fetch_tushare_news_batched(
                token,
                start_date=start_date,
                end_date=end_date,
                src=source,
                limit_per_call=limit,
                batch_days=batch_days,
                keywords=keywords,
            )
            if not frame.empty:
                frame = frame.copy()
                frame["retrieval_source"] = source
                frame["retrieval_dataset"] = "news"
                frames.append(frame)

    if args.mode in {"major_news", "both"}:
        sources = args.source or ts_cfg.get("major_news", {}).get("sources", [])
        limit = int(ts_cfg.get("major_news", {}).get("limit_per_call", 400))
        for source in sources:
            frame = fetch_tushare_major_news_batched(
                token,
                start_date=start_date,
                end_date=end_date,
                src=source,
                limit_per_call=limit,
                batch_days=batch_days,
                keywords=keywords,
            )
            if not frame.empty:
                frame = frame.copy()
                frame["retrieval_source"] = source
                frame["retrieval_dataset"] = "major_news"
                frames.append(frame)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty:
        combined = combined.drop_duplicates(subset=["url", "title", "published_at", "source"]).sort_values(
            ["published_at", "title"]
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output, index=False)
    print(f"rows={len(combined)}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
