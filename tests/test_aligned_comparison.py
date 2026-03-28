from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.reporting.aligned_comparison import summarize_aligned_strategy_window


def test_summarize_aligned_strategy_window_filters_to_requested_dates(tmp_path: Path) -> None:
    weekly_path = tmp_path / "weekly.csv"
    pd.DataFrame(
        [
            {
                "week_end_date": "2021-01-15",
                "strategy_return": 0.10,
                "position": 1.0,
                "benchmark_return": 0.08,
                "benchmark_position": 1.0,
            },
            {
                "week_end_date": "2021-01-22",
                "strategy_return": -0.02,
                "position": -1.0,
                "benchmark_return": 0.01,
                "benchmark_position": 1.0,
            },
            {
                "week_end_date": "2021-01-29",
                "strategy_return": 0.03,
                "position": 1.0,
                "benchmark_return": 0.02,
                "benchmark_position": 1.0,
            },
            {
                "week_end_date": "2021-02-05",
                "strategy_return": 0.05,
                "position": 1.0,
                "benchmark_return": 0.04,
                "benchmark_position": 1.0,
            },
        ]
    ).to_csv(weekly_path, index=False)

    summary = summarize_aligned_strategy_window(
        weekly_path,
        start_date="2021-01-22",
        end_date="2021-01-29",
    )

    assert summary["strategy_num_weeks"] == 2
    assert round(summary["strategy_cumulative_return"], 6) == round((1 - 0.02) * (1 + 0.03) - 1, 6)
    assert round(summary["benchmark_cumulative_return"], 6) == round((1 + 0.01) * (1 + 0.02) - 1, 6)
