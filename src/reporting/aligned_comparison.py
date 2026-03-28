from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.strategy import summarize_strategy_vs_benchmark


def summarize_aligned_strategy_window(
    weekly_path: Path,
    *,
    start_date: str,
    end_date: str,
) -> dict[str, float]:
    weekly = pd.read_csv(weekly_path)
    weekly["week_end_date"] = pd.to_datetime(weekly["week_end_date"]).dt.normalize()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    window = weekly[(weekly["week_end_date"] >= start) & (weekly["week_end_date"] <= end)].copy()
    if window.empty:
        raise ValueError(f"No rows found in aligned window {start_date} ~ {end_date} for {weekly_path}")
    return summarize_strategy_vs_benchmark(window)
