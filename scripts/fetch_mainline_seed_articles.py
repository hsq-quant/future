from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
from pathlib import Path

import pandas as pd
import requests
import trafilatura

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.mainline_seed_fetch import decode_response_text, normalize_extracted_article, read_seed_urls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch real historical articles from curated mainline Chinese-source URLs.")
    parser.add_argument(
        "--seed-file",
        default="data/seeds/mainline_historical_urls.csv",
        help="CSV file containing source_name,url rows.",
    )
    parser.add_argument(
        "--output",
        default="data/raw/finnewshunter_exports/mainline_seed_export.parquet",
        help="Output parquet for fetched articles.",
    )
    parser.add_argument("--max-articles", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def fetch_one(url: str, timeout: int) -> dict[str, str] | None:
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    html = decode_response_text(response)
    extracted = trafilatura.extract(
        html,
        url=url,
        output_format="json",
        with_metadata=True,
        favor_precision=True,
    )
    if not extracted:
        return None
    return json.loads(extracted)


def main() -> None:
    args = parse_args()
    seed_path = Path(args.seed_file)
    if not seed_path.is_absolute():
        seed_path = ROOT / args.seed_file
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / args.output

    seeds = read_seed_urls(seed_path)
    if args.max_articles is not None:
        seeds = seeds.head(args.max_articles).copy()

    rows: list[dict[str, str]] = []

    def _fetch_row(row: pd.Series) -> dict[str, str] | None:
        source_name = row["source_name"]
        url = row["url"]
        extracted = fetch_one(url, timeout=args.timeout)
        if not extracted:
            raise ValueError("no_extract")
        return normalize_extracted_article(extracted, source_name=source_name, fallback_url=url)

    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as executor:
        future_map = {
            executor.submit(_fetch_row, row): (row["source_name"], row["url"])
            for _, row in seeds.iterrows()
        }
        for future in as_completed(future_map):
            source_name, url = future_map[future]
            try:
                row = future.result()
                rows.append(row)
                print(f"ok source={source_name} url={url}")
            except Exception as exc:  # pragma: no cover - network dependent
                print(f"error source={source_name} url={url} reason={exc}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    print(f"rows={len(rows)}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
