from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.sina_search_api import SinaSearchConfig, discover_sina_search_urls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover Sina Finance historical URLs from the site's JSON search API.")
    parser.add_argument("--keywords", nargs="+", default=["原油", "油价", "上海原油", "原油期货", "OPEC", "库存"])
    parser.add_argument("--start-date", default="2019-12-30")
    parser.add_argument("--end-date", default="2026-03-13")
    parser.add_argument("--page-size", type=int, default=20)
    parser.add_argument("--max-results-per-keyword", type=int, default=200)
    parser.add_argument("--sort", type=int, default=1)
    parser.add_argument("--output", default="data/seeds/sina_search_discovered.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output

    discovered = discover_sina_search_urls(
        SinaSearchConfig(
            keywords=args.keywords,
            start_date=args.start_date,
            end_date=args.end_date,
            page_size=args.page_size,
            max_results_per_keyword=args.max_results_per_keyword,
            sort=args.sort,
        )
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    discovered.to_csv(output, index=False)
    print(f"rows={len(discovered)}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
