from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ScoringProgress:
    target_rows: int
    scored_rows: int
    failed_rows: int
    completed_rows: int
    remaining_rows: int
    completion_ratio: float
    last_update_iso: str | None
    seconds_since_update: float | None
    is_stalled: bool


def _row_count(path: Path) -> int:
    if not path.exists():
        return 0
    return int(len(pd.read_parquet(path)))


def summarize_scoring_progress(
    target_path: str | Path,
    scored_path: str | Path,
    failures_path: str | Path,
    *,
    stall_seconds: int = 1800,
) -> ScoringProgress:
    target = Path(target_path)
    scored = Path(scored_path)
    failures = Path(failures_path)

    target_rows = _row_count(target)
    scored_rows = _row_count(scored)
    failed_rows = _row_count(failures)
    completed_rows = scored_rows + failed_rows
    remaining_rows = max(target_rows - completed_rows, 0)
    completion_ratio = (completed_rows / target_rows) if target_rows else 0.0

    candidate_paths = [path for path in (scored, failures) if path.exists()]
    last_update_iso: str | None = None
    seconds_since_update: float | None = None
    is_stalled = False
    if candidate_paths:
        last_mtime = max(path.stat().st_mtime for path in candidate_paths)
        last_dt = datetime.fromtimestamp(last_mtime, tz=timezone.utc)
        last_update_iso = last_dt.isoformat()
        seconds_since_update = max((datetime.now(timezone.utc) - last_dt).total_seconds(), 0.0)
        is_stalled = remaining_rows > 0 and seconds_since_update > float(stall_seconds)

    return ScoringProgress(
        target_rows=target_rows,
        scored_rows=scored_rows,
        failed_rows=failed_rows,
        completed_rows=completed_rows,
        remaining_rows=remaining_rows,
        completion_ratio=float(completion_ratio),
        last_update_iso=last_update_iso,
        seconds_since_update=seconds_since_update,
        is_stalled=is_stalled,
    )
