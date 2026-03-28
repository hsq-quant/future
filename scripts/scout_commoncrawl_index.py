from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.china_sources import load_source_catalog, prioritized_domains
from src.data.commoncrawl_index import query_index, save_index_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scout Common Crawl index for prioritized Chinese finance domains.")
    parser.add_argument("--source-config", default="configs/china_sources.yaml")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--limit-per-domain", type=int, default=200)
    parser.add_argument("--max-domains", type=int, default=None)
    parser.add_argument(
        "--output",
        default="data/intermediate/commoncrawl_index_scout.parquet",
        help="Scout output table path.",
    )
    parser.add_argument(
        "--index-api",
        default="https://index.commoncrawl.org/CC-MAIN-2025-13-index",
        help="Common Crawl index endpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.source_config)
    if not config_path.is_absolute():
        config_path = ROOT / args.source_config

    catalog = load_source_catalog(config_path)
    domains = prioritized_domains(catalog)
    if args.max_domains is not None:
        domains = domains[: args.max_domains]

    frames: list[pd.DataFrame] = []
    for domain in domains:
        try:
            frame = query_index(
                domain=domain,
                start_date=args.start_date,
                end_date=args.end_date,
                limit=args.limit_per_domain,
                index_api=args.index_api,
            )
            if frame.empty:
                continue
            frame["domain"] = domain
            frames.append(frame)
            print(f"domain={domain} rows={len(frame)}")
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"domain={domain} error={exc}")

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / args.output
    save_index_results(result, output_path)

    print(f"domains_scanned={len(domains)}")
    print(f"rows={len(result)}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
