from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_parent, read_dataframe, read_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Snapshot current multilingual scoring inputs into the isolated V2 paths.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs/model_v2.yaml"))
    parser.add_argument(
        "--source-scored",
        type=str,
        default=str(ROOT / "data/intermediate/articles_scored.parquet"),
    )
    parser.add_argument(
        "--source-failures",
        type=str,
        default=str(ROOT / "data/intermediate/articles_scoring_failures.parquet"),
    )
    parser.add_argument(
        "--source-clean",
        type=str,
        default=str(ROOT / "data/intermediate/articles_archive_clean_multilingual.parquet"),
    )
    parser.add_argument(
        "--language-prefixes",
        nargs="*",
        default=None,
        help="Optional language prefixes to keep, such as en or zh.",
    )
    return parser.parse_args()


def _filter_languages(df, prefixes: list[str] | None):
    if prefixes is None or len(prefixes) == 0 or df.empty or "language" not in df.columns:
        return df
    prefix_tuple = tuple(str(prefix) for prefix in prefixes)
    return df[df["language"].fillna("").astype(str).str.startswith(prefix_tuple)].copy()


def _copy_if_exists(source: Path, target: Path) -> None:
    if not source.exists():
        return
    ensure_parent(target)
    shutil.copy2(source, target)


def main() -> None:
    args = parse_args()
    config = read_yaml(args.config)
    source_scored = Path(args.source_scored)
    source_failures = Path(args.source_failures)
    source_clean = Path(args.source_clean)

    target_scored = ROOT / config["artifacts"]["scored_articles"]
    target_clean = target_scored.with_name("articles_archive_clean_multilingual.parquet")
    target_failures = target_scored.with_name("articles_scoring_failures.parquet")
    prefixes = args.language_prefixes

    if prefixes:
        scored_df = _filter_languages(read_dataframe(source_scored), prefixes)
        clean_df = _filter_languages(read_dataframe(source_clean), prefixes)
        ensure_parent(target_scored)
        scored_df.to_parquet(target_scored, index=False)
        ensure_parent(target_clean)
        clean_df.to_parquet(target_clean, index=False)
        if source_failures.exists():
            failures_df = _filter_languages(read_dataframe(source_failures), prefixes)
            ensure_parent(target_failures)
            failures_df.to_parquet(target_failures, index=False)
    else:
        _copy_if_exists(source_scored, target_scored)
        _copy_if_exists(source_clean, target_clean)
        _copy_if_exists(source_failures, target_failures)

    print(target_scored)
    print(target_clean)
    if target_failures.exists():
        print(target_failures)


if __name__ == "__main__":
    main()
