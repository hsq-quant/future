from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.models.strategy import evaluate_regime_comparison


START_DATE = "2021-01-22"
END_DATE = "2025-09-05"
REGIMES = [
    {"regime": "Recovery Bull", "start": "2021-01-22", "end": "2022-02-18"},
    {"regime": "War Spike", "start": "2022-02-25", "end": "2022-06-24"},
    {"regime": "Post-Spike Bear", "start": "2022-07-01", "end": "2023-06-30"},
    {"regime": "OPEC-Supported Range", "start": "2023-07-07", "end": "2024-12-27"},
    {"regime": "Oversupply Bear", "start": "2025-01-03", "end": "2026-03-13"},
]


def _load_and_clip(path: str, *, variant: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if variant is not None:
        frame = frame.loc[frame["variant"] == variant].reset_index(drop=True)
    frame["week_end_date"] = pd.to_datetime(frame["week_end_date"]).dt.normalize()
    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)
    return frame[(frame["week_end_date"] >= start) & (frame["week_end_date"] <= end)].copy().reset_index(drop=True)


def _build_rows() -> list[dict[str, object]]:
    specs = [
        ("V1 Chinese classification", "classification", "always_in", "/Users/hsq/Desktop/codex/future/reports/iterations/2026-03-28-v1-classification-mapping-comparison/variant_weekly_positions.csv"),
        ("V1 Chinese classification", "classification", "long_only", "/Users/hsq/Desktop/codex/future/reports/iterations/2026-03-28-v1-classification-mapping-comparison/variant_weekly_positions.csv"),
        ("V3 English classification", "classification", "always_in", "/Users/hsq/Desktop/codex/future/reports/iterations/2026-03-28-v3-classification-mapping-comparison/variant_weekly_positions.csv"),
        ("V3 English classification", "classification", "long_only", "/Users/hsq/Desktop/codex/future/reports/iterations/2026-03-28-v3-classification-mapping-comparison/variant_weekly_positions.csv"),
        ("V1 Chinese regression", "regression", "baseline_tanh", "/Users/hsq/Desktop/codex/future/reports/iterations/2026-03-28-v1-regression-mapping-comparison/variant_weekly_positions.csv"),
        ("V1 Chinese regression", "regression", "long_only_factor", "/Users/hsq/Desktop/codex/future/reports/iterations/2026-03-28-v1-regression-mapping-comparison/variant_weekly_positions.csv"),
        ("V3 English regression", "regression", "baseline_tanh", "/Users/hsq/Desktop/codex/future/reports/iterations/2026-03-28-v3-english-mapping-comparison/variant_weekly_positions.csv"),
        ("V3 English regression", "regression", "long_only_factor", "/Users/hsq/Desktop/codex/future/reports/iterations/2026-03-28-v3-english-mapping-comparison/variant_weekly_positions.csv"),
    ]
    rows: list[dict[str, object]] = []
    for label, task, execution, path in specs:
        window = _load_and_clip(path, variant=execution)
        regime_df = evaluate_regime_comparison(window, REGIMES)
        for _, row in regime_df.iterrows():
            rows.append(
                {
                    "label": label,
                    "task": task,
                    "execution": execution,
                    "window_start": START_DATE,
                    "window_end": END_DATE,
                    "regime": row["regime"],
                    "strategy_num_weeks": row["strategy_num_weeks"],
                    "strategy_cumulative_return": row["strategy_cumulative_return"],
                    "strategy_annualized_return": row["strategy_annualized_return"],
                    "strategy_sharpe_ratio": row["strategy_sharpe_ratio"],
                    "strategy_max_drawdown": row["strategy_max_drawdown"],
                    "benchmark_cumulative_return": row["benchmark_cumulative_return"],
                    "cumulative_return_diff": row["cumulative_return_diff"],
                    "information_ratio": row["information_ratio"],
                }
            )
    return rows


def main() -> None:
    output_dir = ROOT / "reports/final"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(_build_rows())
    csv_path = output_dir / "regime_comparison_aligned.csv"
    md_path = output_dir / "regime_comparison_aligned.md"
    comparison.to_csv(csv_path, index=False)

    lines = [
        "# Aligned Regime Comparison",
        "",
        f"Common window: `{START_DATE}` to `{END_DATE}`.",
        "",
    ]
    for regime, regime_df in comparison.groupby("regime", sort=False):
        lines.append(f"## {regime}")
        lines.append("")
        lines.append("| Label | Task | Execution | Weeks | Strategy Cum | Sharpe | Benchmark Cum | Cum Diff | IR |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
        for _, row in regime_df.iterrows():
            lines.append(
                "| {label} | {task} | {execution} | {weeks:.0f} | {scum:.2%} | {sharpe:.4f} | {bcum:.2%} | {diff:.2%} | {ir:.4f} |".format(
                    label=row["label"],
                    task=row["task"],
                    execution=row["execution"],
                    weeks=row["strategy_num_weeks"],
                    scum=row["strategy_cumulative_return"],
                    sharpe=row["strategy_sharpe_ratio"],
                    bcum=row["benchmark_cumulative_return"],
                    diff=row["cumulative_return_diff"],
                    ir=row["information_ratio"],
                )
            )
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
