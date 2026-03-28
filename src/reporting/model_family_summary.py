from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ExperimentSpec:
    label: str
    family: str
    language_scope: str
    task: str
    execution: str
    cv_metrics_path: Path
    strategy_metrics_path: Path
    strategy_variant: str | None = None


def _load_cv_summary(path: Path, *, task: str) -> dict[str, float]:
    frame = pd.read_csv(path)
    if task == "classification":
        return {
            "mean_auc": float(frame["auc"].mean()),
            "mean_accuracy": float(frame["accuracy"].mean()),
            "mean_ic": float(frame["ic"].mean()),
            "mean_directional_accuracy": float(frame["accuracy"].mean()),
            "mean_mse": 0.0,
            "mean_mae": 0.0,
        }
    return {
        "mean_auc": 0.0,
        "mean_accuracy": 0.0,
        "mean_ic": float(frame["ic"].mean()),
        "mean_directional_accuracy": float(frame["directional_accuracy"].mean()),
        "mean_mse": float(frame["mse"].mean()),
        "mean_mae": float(frame["mae"].mean()),
    }


def _load_strategy_summary(path: Path, *, variant: str | None) -> dict[str, float]:
    frame = pd.read_csv(path)
    if variant is not None and "variant" in frame.columns:
        frame = frame.loc[frame["variant"] == variant].reset_index(drop=True)
    row = frame.iloc[0]
    num_weeks = float(row["strategy_num_weeks"]) if "strategy_num_weeks" in row else float(row["num_weeks"])
    return {
        "strategy_num_weeks": num_weeks,
        "strategy_cumulative_return": float(row["strategy_cumulative_return"]),
        "strategy_annualized_return": float(row["strategy_annualized_return"]),
        "strategy_annualized_volatility": float(row["strategy_annualized_volatility"]),
        "strategy_sharpe_ratio": float(row["strategy_sharpe_ratio"]),
        "strategy_max_drawdown": float(row["strategy_max_drawdown"]),
        "benchmark_cumulative_return": float(row["benchmark_cumulative_return"]),
        "cumulative_return_diff": float(row["cumulative_return_diff"]),
        "information_ratio": float(row["information_ratio"]),
    }


def build_model_family_summary(specs: list[ExperimentSpec]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in specs:
        cv_summary = _load_cv_summary(spec.cv_metrics_path, task=spec.task)
        strategy_summary = _load_strategy_summary(spec.strategy_metrics_path, variant=spec.strategy_variant)
        rows.append(
            {
                "label": spec.label,
                "family": spec.family,
                "language_scope": spec.language_scope,
                "task": spec.task,
                "execution": spec.execution,
                **cv_summary,
                **strategy_summary,
            }
        )
    return pd.DataFrame(rows)


def render_model_family_markdown(summary: pd.DataFrame) -> str:
    lines = [
        "# Model Family Comparison",
        "",
        "| Label | Task | Execution | Weeks | AUC | Accuracy | IC | Directional Acc | Cum Return | Sharpe | Benchmark Cum | Cum Diff |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| {label} | {task} | {execution} | {weeks:.0f} | {auc:.4f} | {acc:.4f} | {ic:.4f} | {dacc:.4f} | {cum:.2%} | {sharpe:.4f} | {bench:.2%} | {diff:.2%} |".format(
                label=row["label"],
                task=row["task"],
                execution=row["execution"],
                weeks=row["strategy_num_weeks"],
                auc=row["mean_auc"],
                acc=row["mean_accuracy"],
                ic=row["mean_ic"],
                dacc=row["mean_directional_accuracy"],
                cum=row["strategy_cumulative_return"],
                sharpe=row["strategy_sharpe_ratio"],
                bench=row["benchmark_cumulative_return"],
                diff=row["cumulative_return_diff"],
            )
        )
    return "\n".join(lines) + "\n"
