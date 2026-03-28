from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.pipeline.scoring_monitor import summarize_scoring_progress


def _stable_shard(article_id: str, num_shards: int) -> int:
    digest = hashlib.md5(article_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % int(num_shards)


def select_articles_for_shard(
    articles: pd.DataFrame,
    *,
    num_shards: int,
    shard_index: int,
    id_column: str = "article_id",
) -> pd.DataFrame:
    frame = articles.copy()
    if num_shards <= 1:
        return frame.reset_index(drop=True)
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards).")
    if id_column not in frame.columns:
        raise ValueError(f"Missing shard key column: {id_column}")
    mask = frame[id_column].astype(str).map(lambda value: _stable_shard(value, num_shards) == shard_index)
    return frame[mask].reset_index(drop=True)


def shard_output_paths(output_dir: str | Path, shard_index: int) -> tuple[Path, Path]:
    directory = Path(output_dir)
    return (
        directory / f"articles_scored_shard_{shard_index}.parquet",
        directory / f"articles_scoring_failures_shard_{shard_index}.parquet",
    )


def merge_scored_shard_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    materialized = [frame for frame in frames if frame is not None and not frame.empty]
    if not materialized:
        return pd.DataFrame()
    merged = pd.concat(materialized, ignore_index=True)
    if "article_id" in merged.columns:
        merged = merged.drop_duplicates(subset=["article_id"], keep="last").reset_index(drop=True)
    return merged


def summarize_parallel_progress(target_path: str | Path, shard_dir: str | Path, *, num_shards: int) -> dict[str, object]:
    target = Path(target_path)
    base = Path(shard_dir)
    target_frame = pd.read_parquet(target) if target.exists() else pd.DataFrame()
    per_shard: list[dict[str, object]] = []
    scored_total = 0
    failed_total = 0
    for shard_index in range(num_shards):
        scored_path, failure_path = shard_output_paths(base, shard_index)
        summary = summarize_scoring_progress(target, scored_path, failure_path)
        shard_target_rows = (
            int(len(select_articles_for_shard(target_frame, num_shards=num_shards, shard_index=shard_index)))
            if not target_frame.empty
            else 0
        )
        shard_completed_rows = min(summary.completed_rows, shard_target_rows)
        per_shard.append(
            {
                "shard_index": shard_index,
                "target_rows": shard_target_rows,
                "scored_rows": summary.scored_rows,
                "failed_rows": summary.failed_rows,
                "completed_rows": shard_completed_rows,
                "remaining_rows": max(shard_target_rows - shard_completed_rows, 0),
                "completion_ratio": (shard_completed_rows / shard_target_rows) if shard_target_rows else 0.0,
                "last_update_iso": summary.last_update_iso,
                "seconds_since_update": summary.seconds_since_update,
                "is_stalled": summary.is_stalled,
            }
        )
        scored_total += summary.scored_rows
        failed_total += summary.failed_rows

    target_rows = int(len(target_frame))
    completed_rows = scored_total + failed_total
    remaining_rows = max(target_rows - completed_rows, 0)
    return {
        "target_rows": target_rows,
        "scored_rows": scored_total,
        "failed_rows": failed_total,
        "completed_rows": completed_rows,
        "remaining_rows": remaining_rows,
        "completion_ratio": (completed_rows / target_rows) if target_rows else 0.0,
        "per_shard": per_shard,
    }
