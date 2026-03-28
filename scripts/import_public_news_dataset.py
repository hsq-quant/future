from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.finnewshunter_import import normalize_finnewshunter_export
from src.utils.io import read_dataframe, read_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize downloaded public finance-news datasets into archive format.")
    parser.add_argument("--config", default="configs/external_news.yaml")
    parser.add_argument("--input", action="append", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = read_yaml(config_path).get("external_news", {})
    dataset_cfg = config.get("public_datasets", {})

    input_paths = args.input or sorted(str(path) for path in (ROOT / dataset_cfg.get("input_glob", "data/raw/public_datasets/*")).parent.glob((ROOT / dataset_cfg.get("input_glob", "data/raw/public_datasets/*")).name))
    output_dir = ROOT / dataset_cfg.get("output_dir", "data/raw/archive/public_datasets")
    output = Path(args.output) if args.output else output_dir / "public_datasets.parquet"
    if not output.is_absolute():
        output = ROOT / output

    frames: list[pd.DataFrame] = []
    for input_path in input_paths:
        path = Path(input_path)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            continue
        frames.append(normalize_finnewshunter_export(read_dataframe(path)))

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty and "published_at" in combined.columns:
        combined["published_at"] = pd.to_datetime(combined["published_at"], errors="coerce")
        combined = combined.dropna(subset=["published_at"])
        combined = combined.drop_duplicates(subset=["url", "title", "published_at"]).sort_values(
            ["published_at", "title"]
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output, index=False)
    print(f"rows={len(combined)}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
