from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.scoring_shards import merge_scored_shard_frames, shard_output_paths
from src.utils.io import read_dataframe, write_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge shard-parallel scoring outputs.")
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--failure-output", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    scored_frames = []
    failure_frames = []
    for shard_index in range(args.num_shards):
        scored_path, failure_path = shard_output_paths(input_dir, shard_index)
        if scored_path.exists():
            scored_frames.append(read_dataframe(scored_path))
        if failure_path.exists():
            failure_frames.append(read_dataframe(failure_path))

    merged_scored = merge_scored_shard_frames(scored_frames)
    merged_failures = merge_scored_shard_frames(failure_frames)
    write_dataframe(merged_scored, args.output)
    write_dataframe(merged_failures, args.failure_output)
    print(args.output)
    print(f"scored_rows={len(merged_scored)}")
    print(args.failure_output)
    print(f"failed_rows={len(merged_failures)}")


if __name__ == "__main__":
    main()
