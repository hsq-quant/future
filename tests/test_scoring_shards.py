from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.pipeline.scoring_shards import (
    merge_scored_shard_frames,
    select_articles_for_shard,
    summarize_parallel_progress,
)


def test_select_articles_for_shard_is_deterministic_and_disjoint() -> None:
    articles = pd.DataFrame(
        {
            "article_id": [f"a{i}" for i in range(12)],
            "title": [f"t{i}" for i in range(12)],
        }
    )

    shards = [
        select_articles_for_shard(articles, num_shards=3, shard_index=idx)["article_id"].tolist()
        for idx in range(3)
    ]

    assert all(shards)
    assert sum(len(ids) for ids in shards) == 12
    assert len(set().union(*[set(ids) for ids in shards])) == 12

    shards_again = [
        select_articles_for_shard(articles, num_shards=3, shard_index=idx)["article_id"].tolist()
        for idx in range(3)
    ]
    assert shards == shards_again


def test_merge_scored_shard_frames_deduplicates_article_id() -> None:
    shard_a = pd.DataFrame({"article_id": ["a1", "a2"], "relevance": [0.1, 0.2]})
    shard_b = pd.DataFrame({"article_id": ["a2", "a3"], "relevance": [0.25, 0.3]})

    merged = merge_scored_shard_frames([shard_a, shard_b])

    assert merged["article_id"].tolist() == ["a1", "a2", "a3"]
    assert merged.loc[merged["article_id"] == "a2", "relevance"].iloc[0] == 0.25


def test_summarize_parallel_progress_aggregates_shard_outputs(tmp_path: Path) -> None:
    target_path = tmp_path / "target.parquet"
    pd.DataFrame({"article_id": [f"a{i}" for i in range(6)]}).to_parquet(target_path, index=False)

    shard_dir = tmp_path / "scoring"
    shard_dir.mkdir()
    pd.DataFrame({"article_id": ["a1", "a2"]}).to_parquet(shard_dir / "articles_scored_shard_0.parquet", index=False)
    pd.DataFrame({"article_id": ["a3"]}).to_parquet(shard_dir / "articles_scored_shard_1.parquet", index=False)
    pd.DataFrame({"article_id": ["a4"]}).to_parquet(shard_dir / "articles_scoring_failures_shard_1.parquet", index=False)

    summary = summarize_parallel_progress(target_path, shard_dir, num_shards=2)

    assert summary["target_rows"] == 6
    assert summary["scored_rows"] == 3
    assert summary["failed_rows"] == 1
    assert summary["remaining_rows"] == 2
    assert summary["per_shard"][0]["scored_rows"] == 2
    assert summary["per_shard"][1]["failed_rows"] == 1
