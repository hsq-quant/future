from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.sina_roll import SinaRollConfig, discover_sina_roll_urls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover Sina Finance historical URLs from roll pages.")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--max-pages", type=int, default=500)
    parser.add_argument("--keywords", nargs="*", default=["原油", "油价", "上海原油", "期货", "OPEC", "库存", "能源"])
    parser.add_argument("--output", default="data/raw/finnewshunter_exports/sina_roll_discovered.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output

    discovered = discover_sina_roll_urls(
        SinaRollConfig(
            start_date=args.start_date,
            end_date=args.end_date,
            max_pages=args.max_pages,
            keywords=tuple(args.keywords),
        )
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    discovered.to_csv(output, index=False)
    print(f"rows={len(discovered)}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
