from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.models.strategy import (
    apply_weekly_trading_costs,
    build_long_benchmark,
    build_weekly_factor_strategy,
    build_weekly_strategy,
    evaluate_regime_comparison,
    render_strategy_report,
    summarize_strategy_vs_benchmark,
    summarize_weekly_strategy,
)
from src.utils.io import read_dataframe, read_yaml, write_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run weekly strategy evaluation from model predictions.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs/model.yaml"))
    parser.add_argument("--strategy-config", type=str, default=str(ROOT / "configs/strategy.yaml"))
    return parser.parse_args()


def main() -> None:
    root = ROOT
    args = parse_args()
    config = read_yaml(args.config)
    strategy_config = read_yaml(args.strategy_config).get("strategy", {})
    predictions = read_dataframe(root / config["artifacts"]["predictions"])
    weekly_labels = read_dataframe(root / "data/processed/weekly_labels.parquet")[
        ["week_end_date", "weekly_close"]
    ].copy()
    strategy_mode = str(strategy_config.get("mode", "classification"))
    if strategy_mode == "factor":
        factor_cfg = strategy_config.get("factor_mapping", {})
        strategy = build_weekly_factor_strategy(
            predictions,
            prediction_column=str(strategy_config.get("prediction_column", "pred_value")),
            mapping_method=str(factor_cfg.get("method", "tanh")),
            scale=float(factor_cfg.get("scale", 0.5)),
            k=float(factor_cfg.get("k", 1.0)),
        )
    else:
        strategy = build_weekly_strategy(predictions)
    strategy = strategy.merge(weekly_labels, on="week_end_date", how="left")
    strategy = strategy.merge(
        build_long_benchmark(strategy)[["week_end_date", "benchmark_position", "benchmark_return", "benchmark_cum_return"]],
        on="week_end_date",
        how="left",
    )
    strategy_path = root / config["artifacts"]["strategy"]
    metrics_path = root / config["artifacts"].get("strategy_metrics", "reports/strategy_metrics.parquet")
    report_path = root / config["artifacts"].get("strategy_report", "reports/strategy_report.md")
    scenarios_path = root / config["artifacts"].get("strategy_scenarios", "reports/strategy_scenarios.parquet")
    regimes_path = root / config["artifacts"].get("strategy_regimes", "reports/strategy_regimes.parquet")

    summary = summarize_weekly_strategy(strategy)
    benchmark_summary = summarize_weekly_strategy(
        strategy,
        return_column="benchmark_return",
        position_column="benchmark_position",
    )
    comparison_summary = summarize_strategy_vs_benchmark(strategy)
    metrics = pd.DataFrame([{**summary, **{f"benchmark_{k}": v for k, v in benchmark_summary.items()}, **comparison_summary}])

    contract = strategy_config.get("contract", {})
    scenario_rows: list[dict[str, object]] = []
    benchmark_regime_frames: list[pd.DataFrame] = []
    for scenario in strategy_config.get("scenarios", []):
        scenario_name = scenario["name"]
        if scenario.get("apply_costs", False):
            scenario_strategy = apply_weekly_trading_costs(
                strategy,
                commission_per_lot=float(contract.get("commission_per_lot", 40.0)),
                slippage_ticks=int(scenario.get("slippage_ticks", 0)),
                tick_size=float(contract.get("tick_size", 0.1)),
                contract_size=float(contract.get("contract_size", 1000.0)),
            )
            scenario_benchmark = apply_weekly_trading_costs(
                scenario_strategy,
                commission_per_lot=float(contract.get("commission_per_lot", 40.0)),
                slippage_ticks=int(scenario.get("slippage_ticks", 0)),
                tick_size=float(contract.get("tick_size", 0.1)),
                contract_size=float(contract.get("contract_size", 1000.0)),
                gross_return_column="benchmark_return",
                net_return_column="net_benchmark_return",
                position_column="benchmark_position",
                cum_return_column="net_benchmark_cum_return",
            )
            strategy_return_column = "net_strategy_return"
            benchmark_return_column = "net_benchmark_return"
        else:
            scenario_strategy = strategy.copy()
            scenario_benchmark = scenario_strategy.copy()
            strategy_return_column = "strategy_return"
            benchmark_return_column = "benchmark_return"
        scenario_summary = summarize_weekly_strategy(scenario_strategy, return_column=strategy_return_column)
        scenario_benchmark_summary = summarize_weekly_strategy(
            scenario_benchmark,
            return_column=benchmark_return_column,
            position_column="benchmark_position",
        )
        scenario_comparison = summarize_strategy_vs_benchmark(
            scenario_benchmark,
            strategy_return_column=strategy_return_column,
            benchmark_return_column=benchmark_return_column,
        )
        scenario_rows.append(
            {
                "scenario": scenario_name,
                "slippage_ticks": int(scenario.get("slippage_ticks", 0)),
                "commission_per_lot": float(contract.get("commission_per_lot", 40.0)),
                **{f"strategy_{key}": value for key, value in scenario_summary.items()},
                **{f"benchmark_{key}": value for key, value in scenario_benchmark_summary.items()},
                **scenario_comparison,
            }
        )
        regime_df = evaluate_regime_comparison(
            scenario_benchmark,
            list(strategy_config.get("regimes", [])),
            strategy_return_column=strategy_return_column,
            benchmark_return_column=benchmark_return_column,
        )
        if not regime_df.empty:
            regime_df.insert(0, "scenario", scenario_name)
            regime_df.insert(1, "slippage_ticks", int(scenario.get("slippage_ticks", 0)))
            benchmark_regime_frames.append(regime_df)

    scenarios = pd.DataFrame(scenario_rows)
    regimes = pd.concat(benchmark_regime_frames, ignore_index=True) if benchmark_regime_frames else pd.DataFrame()

    start_week = str(pd.to_datetime(strategy["week_end_date"]).min().date())
    end_week = str(pd.to_datetime(strategy["week_end_date"]).max().date())
    configured_regimes = list(strategy_config.get("regimes", []))
    if configured_regimes:
        latest_regime_end = max(pd.Timestamp(item["end"]).normalize() for item in configured_regimes)
        strategy_end = pd.to_datetime(strategy["week_end_date"]).max().normalize()
        if strategy_end < latest_regime_end:
            regime_coverage_note = (
                f"Current sample ends at {strategy_end.date()}, earlier than the configured regime end "
                f"{latest_regime_end.date()}; later regimes are only partially covered or absent."
            )
        else:
            regime_coverage_note = "Current sample covers the full configured regime window."
    else:
        regime_coverage_note = None

    write_dataframe(strategy, strategy_path)
    write_dataframe(metrics, metrics_path)
    write_dataframe(scenarios, scenarios_path)
    write_dataframe(regimes, regimes_path)
    strategy.to_csv(strategy_path.with_suffix(".csv"), index=False)
    metrics.to_csv(metrics_path.with_suffix(".csv"), index=False)
    scenarios.to_csv(scenarios_path.with_suffix(".csv"), index=False)
    regimes.to_csv(regimes_path.with_suffix(".csv"), index=False)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_lines = ["", "## Scenario Comparison"]
    for row in scenario_rows:
        scenario_lines.append(
            f"- {row['scenario']}: strategy={row['strategy_cumulative_return']:.2%}, "
            f"benchmark={row['benchmark_cumulative_return']:.2%}, "
            f"diff={row['cumulative_return_diff']:.2%}, "
            f"IR={row['information_ratio']:.4f}"
        )
    regime_lines = ["", "## Regime Comparison"]
    for _, row in regimes.iterrows():
        regime_lines.append(
            f"- {row['scenario']} / {row['regime']}: strategy={row['strategy_cumulative_return']:.2%}, "
            f"benchmark={row['benchmark_cumulative_return']:.2%}, "
            f"diff={row['cumulative_return_diff']:.2%}, "
            f"IR={row['information_ratio']:.4f}"
        )
    report_path.write_text(
        render_strategy_report(
            summary,
            benchmark_summary=benchmark_summary,
            comparison_summary=comparison_summary,
            date_range=(start_week, end_week),
            regime_coverage_note=regime_coverage_note,
        )
        + "\n"
        + "\n".join(scenario_lines + regime_lines),
        encoding="utf-8",
    )

    print(strategy_path)
    print(metrics_path)
    print(scenarios_path)
    print(regimes_path)
    print(report_path)


if __name__ == "__main__":
    main()
