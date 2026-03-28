from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.scoring_shards import merge_scored_shard_frames, select_articles_for_shard, shard_output_paths
from src.utils.io import read_dataframe, read_yaml, write_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize parallel scoring shards from existing scored caches.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed-path", action="append", default=[], help="Existing scored parquet to seed from.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_yaml(args.config)
    clean_input = ROOT / config["artifacts"]["clean_articles_input"]
    articles = read_dataframe(clean_input)

    seed_frames = [read_dataframe(path) for path in args.seed_path if Path(path).exists()]
    merged_seed = merge_scored_shard_frames(seed_frames)
    if not merged_seed.empty and "article_id" in merged_seed.columns:
        valid_ids = set(articles["article_id"].astype(str))
        merged_seed = merged_seed[merged_seed["article_id"].astype(str).isin(valid_ids)].copy()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for shard_index in range(args.num_shards):
        shard_articles = select_articles_for_shard(articles, num_shards=args.num_shards, shard_index=shard_index)
        shard_ids = set(shard_articles["article_id"].astype(str))
        shard_seed = (
            merged_seed[merged_seed["article_id"].astype(str).isin(shard_ids)].copy()
            if not merged_seed.empty and "article_id" in merged_seed.columns
            else pd.DataFrame()
        )
        scored_path, failure_path = shard_output_paths(output_dir, shard_index)
        write_dataframe(shard_seed, scored_path)
        write_dataframe(pd.DataFrame(), failure_path)
        print(f"shard={shard_index} target_rows={len(shard_articles)} seed_rows={len(shard_seed)}")


if __name__ == "__main__":
    main()
