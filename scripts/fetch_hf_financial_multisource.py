from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.hf_dataset_fetch import (
    choose_parquet_urls,
    download_file,
    fetch_hf_parquet_urls,
    fetch_hf_rows,
    fetch_hf_splits,
    missing_download_targets,
    resolve_hf_token,
    select_parquet_urls_by_ids,
)
from src.utils.io import write_dataframe


DATASET = "Brianferrell787/financial-news-multisource"
DEFAULT_CONFIG = "data"
DEFAULT_SPLIT = "train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch rows or parquet shards from Brianferrell787/financial-news-multisource.")
    parser.add_argument("--mode", choices=["splits", "rows", "parquet"], default="rows")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--max-files", type=int, default=2)
    parser.add_argument("--file-ids", type=int, nargs="*", default=None)
    parser.add_argument("--tail", action="store_true")
    parser.add_argument("--all-files", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--output", default="data/tmp/hf_financial_multisource_rows.parquet")
    parser.add_argument("--download-dir", default="data/raw/hf_financial_multisource")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = resolve_hf_token()

    if args.mode == "splits":
        splits = fetch_hf_splits(dataset=DATASET, token=token)
        print("\n".join(splits))
        return

    if args.mode == "rows":
        rows = fetch_hf_rows(
            dataset=DATASET,
            config=args.config,
            split=args.split,
            offset=args.offset,
            length=args.length,
            token=token,
        )
        output = Path(args.output)
        if not output.is_absolute():
            output = ROOT / output
        write_dataframe(rows, output)
        print(f"rows={len(rows)}")
        print(f"output={output}")
        return

    parquet_urls = fetch_hf_parquet_urls(
        dataset=DATASET,
        config=args.config,
        split=args.split,
        token=token,
    )
    download_dir = Path(args.download_dir)
    if not download_dir.is_absolute():
        download_dir = ROOT / download_dir
    download_dir.mkdir(parents=True, exist_ok=True)

    selected_urls = choose_parquet_urls(
        parquet_urls,
        max_files=args.max_files,
        tail=args.tail,
        all_files=args.all_files,
    )
    if args.file_ids:
        selected_urls = select_parquet_urls_by_ids(parquet_urls, args.file_ids)
    if args.skip_existing:
        targets = missing_download_targets(selected_urls, download_dir)
    else:
        targets = []
        for index, url in enumerate(selected_urls, start=1):
            suffix = Path(url).name or f"part-{index}.parquet"
            targets.append((url, download_dir / suffix))

    for url, output in targets:
        download_file(url, output, token=token)
        print(output)


if __name__ == "__main__":
    main()
