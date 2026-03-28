from __future__ import annotations

import argparse
import sys

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.scoring_shards import summarize_parallel_progress


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor shard-parallel scoring progress.")
    parser.add_argument("--target-input", type=str, required=True)
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_parallel_progress(args.target_input, args.input_dir, num_shards=args.num_shards)
    print(f"target_rows={summary['target_rows']}")
    print(f"scored_rows={summary['scored_rows']}")
    print(f"failed_rows={summary['failed_rows']}")
    print(f"completed_rows={summary['completed_rows']}")
    print(f"remaining_rows={summary['remaining_rows']}")
    print(f"completion_ratio={summary['completion_ratio']:.4f}")
    for shard in summary["per_shard"]:
        print(
            f"shard={shard['shard_index']} scored={shard['scored_rows']} failed={shard['failed_rows']} "
            f"completed={shard['completed_rows']} remaining={shard['remaining_rows']} stalled={shard['is_stalled']}"
        )


if __name__ == "__main__":
    main()
