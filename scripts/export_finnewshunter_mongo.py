from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.finnewshunter_mongo import (
    build_collection_specs,
    normalize_mongo_documents,
    read_mongo_export_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export or normalize FinnewsHunter Mongo-style documents.")
    parser.add_argument("--source-name", required=True, help="Source name, e.g. CNStock/JRJ/NBD.")
    parser.add_argument(
        "--input-json",
        required=True,
        help="Path to a Mongo export file in JSON or JSONL format.",
    )
    parser.add_argument(
        "--output",
        default="data/raw/finnewshunter_exports/exported.parquet",
        help="Output parquet path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_json)
    if not input_path.is_absolute():
        input_path = ROOT / args.input_json

    docs = read_mongo_export_file(input_path)
    normalized = normalize_mongo_documents(docs, source_name=args.source_name)

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_parquet(output, index=False)

    print(f"source={args.source_name}")
    print(f"documents={len(docs)}")
    print(f"rows={len(normalized)}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
