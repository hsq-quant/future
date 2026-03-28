from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.scoring_monitor import summarize_scoring_progress
from src.utils.io import read_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor long-running article scoring progress.")
    parser.add_argument("--config", type=str, required=True, help="Model config used by score_articles.py")
    parser.add_argument("--target-input", type=str, default=None, help="Override target clean-article input path.")
    parser.add_argument("--stall-seconds", type=int, default=1800, help="Mark stalled if no file update within this many seconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_yaml(args.config)
    target_input = Path(args.target_input) if args.target_input else ROOT / config["artifacts"]["clean_articles_input"]
    scored_path = ROOT / config["artifacts"]["scored_articles"]
    failures_path = scored_path.with_name("articles_scoring_failures.parquet")

    summary = summarize_scoring_progress(
        target_input,
        scored_path,
        failures_path,
        stall_seconds=args.stall_seconds,
    )
    print(f"target_rows={summary.target_rows}")
    print(f"scored_rows={summary.scored_rows}")
    print(f"failed_rows={summary.failed_rows}")
    print(f"completed_rows={summary.completed_rows}")
    print(f"remaining_rows={summary.remaining_rows}")
    print(f"completion_ratio={summary.completion_ratio:.4f}")
    print(f"last_update_iso={summary.last_update_iso}")
    print(f"seconds_since_update={summary.seconds_since_update}")
    print(f"is_stalled={summary.is_stalled}")


if __name__ == "__main__":
    main()
