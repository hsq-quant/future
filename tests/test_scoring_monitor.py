from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.pipeline.scoring_monitor import summarize_scoring_progress


def test_summarize_scoring_progress_computes_progress_and_remaining(tmp_path: Path) -> None:
    target_path = tmp_path / "target.parquet"
    scored_path = tmp_path / "scored.parquet"
    failures_path = tmp_path / "failures.parquet"

    pd.DataFrame({"article_id": ["a1", "a2", "a3", "a4"]}).to_parquet(target_path, index=False)
    pd.DataFrame({"article_id": ["a1", "a2"]}).to_parquet(scored_path, index=False)
    pd.DataFrame({"article_id": ["a3"]}).to_parquet(failures_path, index=False)

    summary = summarize_scoring_progress(target_path, scored_path, failures_path)

    assert summary.target_rows == 4
    assert summary.scored_rows == 2
    assert summary.failed_rows == 1
    assert summary.remaining_rows == 1
    assert summary.completed_rows == 3
    assert summary.completion_ratio == 0.75
    assert summary.is_stalled is False


def test_summarize_scoring_progress_marks_missing_outputs_as_zero(tmp_path: Path) -> None:
    target_path = tmp_path / "target.parquet"
    pd.DataFrame({"article_id": ["a1", "a2"]}).to_parquet(target_path, index=False)

    summary = summarize_scoring_progress(
        target_path,
        tmp_path / "missing_scored.parquet",
        tmp_path / "missing_failures.parquet",
    )

    assert summary.scored_rows == 0
    assert summary.failed_rows == 0
    assert summary.remaining_rows == 2


def test_summarize_scoring_progress_flags_stalled_when_last_update_is_old(tmp_path: Path) -> None:
    target_path = tmp_path / "target.parquet"
    scored_path = tmp_path / "scored.parquet"
    pd.DataFrame({"article_id": ["a1", "a2"]}).to_parquet(target_path, index=False)
    pd.DataFrame({"article_id": ["a1"]}).to_parquet(scored_path, index=False)

    summary = summarize_scoring_progress(
        target_path,
        scored_path,
        tmp_path / "missing_failures.parquet",
        stall_seconds=0,
    )

    assert summary.is_stalled is True
