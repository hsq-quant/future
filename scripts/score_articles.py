from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.article_inputs import resolve_clean_articles_path
from src.features.scoring import (
    filter_unscored_articles,
    iter_article_score_attempts,
    mock_score_article,
    restrict_scores_to_articles,
    score_article_with_qwen,
)
from src.pipeline.scoring_shards import select_articles_for_shard
from src.utils.io import read_dataframe, read_yaml, write_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score cleaned articles with Qwen sentiment dimensions.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs/model.yaml"))
    parser.add_argument("--input", type=str, default=None, help="Optional cleaned-article input path.")
    parser.add_argument("--max-articles", type=int, default=None, help="Optional cap for smoke testing.")
    parser.add_argument("--mock", action="store_true", help="Use deterministic local mock scores instead of calling Qwen.")
    parser.add_argument("--save-every", type=int, default=10, help="Persist intermediate scores every N articles.")
    parser.add_argument("--max-workers", type=int, default=1, help="Parallel workers for article scoring.")
    parser.add_argument("--output", type=str, default=None, help="Optional scored-article output override.")
    parser.add_argument("--failure-output", type=str, default=None, help="Optional failed-article output override.")
    parser.add_argument("--num-shards", type=int, default=1, help="Deterministic article shard count.")
    parser.add_argument("--shard-index", type=int, default=0, help="Shard index in [0, num_shards).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = ROOT
    config = read_yaml(args.config)
    input_path = resolve_clean_articles_path(
        root,
        preferred_input=args.input,
        default_input=config["artifacts"].get("clean_articles_input"),
    )
    output_path = Path(args.output) if args.output else root / config["artifacts"]["scored_articles"]
    failure_path = Path(args.failure_output) if args.failure_output else output_path.with_name("articles_scoring_failures.parquet")
    articles = read_dataframe(input_path)
    articles = select_articles_for_shard(
        articles,
        num_shards=max(args.num_shards, 1),
        shard_index=max(args.shard_index, 0),
    )
    if args.max_articles is not None:
        articles = articles.head(args.max_articles).copy()
    existing = read_dataframe(output_path) if output_path.exists() else pd.DataFrame()
    failures = read_dataframe(failure_path) if failure_path.exists() else pd.DataFrame()
    if not existing.empty:
        existing = restrict_scores_to_articles(existing, articles)
    articles = filter_unscored_articles(articles, existing if not args.mock else pd.DataFrame())
    if not failures.empty and "article_id" in articles.columns and "article_id" in failures.columns:
        failed_ids = set(failures["article_id"].dropna().astype(str))
        articles = articles[~articles["article_id"].astype(str).isin(failed_ids)].copy()

    def _score_record(article: dict[str, object]) -> dict[str, object]:
        if args.mock:
            return mock_score_article(article)
        return score_article_with_qwen(
            article,
            model=config["qwen"]["model"],
            base_url=config["qwen"]["base_url"],
            api_key_env=config["qwen"]["api_key_env"],
            timeout_seconds=config["qwen"].get("timeout_seconds", 60),
        )

    rows: list[dict[str, object]] = []
    failed_rows: list[dict[str, object]] = []
    for record, score, error in iter_article_score_attempts(
        articles,
        scorer=_score_record,
        max_workers=max(args.max_workers, 1),
    ):
        if error is None and score is not None:
            rows.append({**record, **score})
        else:
            failed_rows.append({**record, "error_message": error or "Unknown scoring error"})

        processed = len(rows) + len(failed_rows)
        if processed and processed % max(args.save_every, 1) == 0:
            partial = pd.DataFrame(rows)
            combined = pd.concat([existing, partial], ignore_index=True) if not existing.empty else partial
            if "article_id" in combined.columns:
                combined = combined.drop_duplicates(subset=["article_id"], keep="last").reset_index(drop=True)
            write_dataframe(combined, output_path)

            if failed_rows:
                partial_failures = pd.DataFrame(failed_rows)
                combined_failures = (
                    pd.concat([failures, partial_failures], ignore_index=True) if not failures.empty else partial_failures
                )
                if "article_id" in combined_failures.columns:
                    combined_failures = combined_failures.drop_duplicates(subset=["article_id"], keep="last").reset_index(drop=True)
                write_dataframe(combined_failures, failure_path)
    scored = pd.DataFrame(rows)
    combined = pd.concat([existing, scored], ignore_index=True) if not existing.empty else scored
    if not combined.empty and "article_id" in combined.columns:
        combined = combined.drop_duplicates(subset=["article_id"], keep="last").reset_index(drop=True)
    write_dataframe(combined, output_path)
    if failed_rows:
        partial_failures = pd.DataFrame(failed_rows)
        combined_failures = pd.concat([failures, partial_failures], ignore_index=True) if not failures.empty else partial_failures
        if "article_id" in combined_failures.columns:
            combined_failures = combined_failures.drop_duplicates(subset=["article_id"], keep="last").reset_index(drop=True)
        write_dataframe(combined_failures, failure_path)
    print(f"input={input_path}")
    print(f"articles={len(articles)}")
    print(output_path)
    if failed_rows:
        print(failure_path)
        print(f"failed_articles={len(failed_rows)}")


if __name__ == "__main__":
    main()
