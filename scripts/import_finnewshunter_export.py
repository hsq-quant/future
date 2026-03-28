from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.archive_ingest import _load_table
from src.data.finnewshunter_import import normalize_finnewshunter_export


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize FinnewsHunter-style local exports into archive files.")
    parser.add_argument(
        "--input-glob",
        default="data/raw/finnewshunter_exports/*",
        help="Glob for local CSV/Parquet/JSONL exports.",
    )
    parser.add_argument(
        "--output",
        default="data/raw/archive/finnewshunter/normalized.parquet",
        help="Normalized archive file path.",
    )
    parser.add_argument("--max-articles", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pattern = Path(args.input_glob)
    if not pattern.is_absolute():
        pattern = ROOT / args.input_glob

    frames: list[pd.DataFrame] = []
    for file_path in sorted(pattern.parent.glob(pattern.name)):
        frames.append(_load_table(file_path))

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    normalized = normalize_finnewshunter_export(combined)
    if args.max_articles is not None:
        normalized = normalized.head(args.max_articles).copy()

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_parquet(output, index=False)

    print(f"input_rows={len(combined)}")
    print(f"normalized_rows={len(normalized)}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
