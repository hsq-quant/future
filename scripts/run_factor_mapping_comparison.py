from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.strategy import (
    apply_weekly_trading_costs,
    build_long_benchmark,
    build_weekly_factor_strategy,
    evaluate_regime_comparison,
    summarize_strategy_vs_benchmark,
)
from src.utils.io import read_dataframe, read_yaml, write_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiple factor-mapping variants for one prediction set.")
    parser.add_argument("--config", type=str, required=True, help="Model config with prediction artifact paths.")
    parser.add_argument("--strategy-config", type=str, required=True, help="Strategy config with factor_variants.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for comparison outputs.")
    return parser.parse_args()


def _load_strategy_base(predictions: pd.DataFrame, weekly_labels: pd.DataFrame, variant: dict[str, object], prediction_column: str) -> pd.DataFrame:
    factor_cfg = dict(variant.get("factor_mapping", {}))
    strategy = build_weekly_factor_strategy(
        predictions,
        prediction_column=prediction_column,
        mapping_method=str(factor_cfg.get("method", "tanh")),
        scale=float(factor_cfg.get("scale", 0.5)),
        k=float(factor_cfg.get("k", 1.0)),
        long_scale=float(factor_cfg.get("long_scale", 1.0)),
        short_scale=float(factor_cfg.get("short_scale", 1.0)),
        short_threshold=float(factor_cfg.get("short_threshold", 0.0)),
    )
    strategy = strategy.merge(weekly_labels, on="week_end_date", how="left")
    strategy = strategy.merge(
        build_long_benchmark(strategy)[["week_end_date", "benchmark_position", "benchmark_return", "benchmark_cum_return"]],
        on="week_end_date",
        how="left",
    )
    strategy.insert(0, "variant", str(variant["name"]))
    strategy.insert(1, "variant_label", str(variant.get("label", variant["name"])))
    return strategy


def main() -> None:
    args = parse_args()
    model_cfg = read_yaml(args.config)
    strategy_cfg = read_yaml(args.strategy_config).get("strategy", {})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = read_dataframe(ROOT / model_cfg["artifacts"]["predictions"])
    weekly_labels = read_dataframe(ROOT / "data/processed/weekly_labels.parquet")[["week_end_date", "weekly_close"]].copy()
    prediction_column = str(strategy_cfg.get("prediction_column", "pred_value"))
    variants = list(strategy_cfg.get("factor_variants", []))
    if not variants:
        raise ValueError("strategy.factor_variants is required for mapping comparison.")

    contract = strategy_cfg.get("contract", {})
    scenarios = list(strategy_cfg.get("scenarios", []))
    regimes = list(strategy_cfg.get("regimes", []))

    variant_frames: list[pd.DataFrame] = []
    variant_rows: list[dict[str, object]] = []
    scenario_rows: list[dict[str, object]] = []
    regime_frames: list[pd.DataFrame] = []

    for variant in variants:
        variant_name = str(variant["name"])
        base_strategy = _load_strategy_base(predictions, weekly_labels, variant, prediction_column)
        variant_frames.append(base_strategy)
        comparison_summary = summarize_strategy_vs_benchmark(base_strategy)
        variant_rows.append({"variant": variant_name, "variant_label": str(variant.get("label", variant_name)), **comparison_summary})

        for scenario in scenarios:
            if scenario.get("apply_costs", False):
                scenario_strategy = apply_weekly_trading_costs(
                    base_strategy,
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
                scenario_strategy = base_strategy.copy()
                scenario_benchmark = base_strategy.copy()
                strategy_return_column = "strategy_return"
                benchmark_return_column = "benchmark_return"

            scenario_summary = summarize_strategy_vs_benchmark(
                scenario_benchmark,
                strategy_return_column=strategy_return_column,
                benchmark_return_column=benchmark_return_column,
            )
            scenario_rows.append(
                {
                    "variant": variant_name,
                    "variant_label": str(variant.get("label", variant_name)),
                    "scenario": scenario["name"],
                    "slippage_ticks": int(scenario.get("slippage_ticks", 0)),
                    **scenario_summary,
                }
            )

            regime_df = evaluate_regime_comparison(
                scenario_benchmark,
                regimes,
                strategy_return_column=strategy_return_column,
                benchmark_return_column=benchmark_return_column,
            )
            if not regime_df.empty:
                regime_df.insert(0, "variant", variant_name)
                regime_df.insert(1, "variant_label", str(variant.get("label", variant_name)))
                regime_df.insert(2, "scenario", str(scenario["name"]))
                regime_df.insert(3, "slippage_ticks", int(scenario.get("slippage_ticks", 0)))
                regime_frames.append(regime_df)

    variant_metrics = pd.DataFrame(variant_rows)
    scenario_metrics = pd.DataFrame(scenario_rows)
    regime_metrics = pd.concat(regime_frames, ignore_index=True) if regime_frames else pd.DataFrame()
    weekly_positions = pd.concat(variant_frames, ignore_index=True) if variant_frames else pd.DataFrame()

    write_dataframe(variant_metrics, output_dir / "variant_metrics.parquet")
    write_dataframe(scenario_metrics, output_dir / "variant_scenarios.parquet")
    write_dataframe(regime_metrics, output_dir / "variant_regimes.parquet")
    write_dataframe(weekly_positions, output_dir / "variant_weekly_positions.parquet")
    variant_metrics.to_csv(output_dir / "variant_metrics.csv", index=False)
    scenario_metrics.to_csv(output_dir / "variant_scenarios.csv", index=False)
    regime_metrics.to_csv(output_dir / "variant_regimes.csv", index=False)
    weekly_positions.to_csv(output_dir / "variant_weekly_positions.csv", index=False)

    lines = [
        "# Factor Mapping Comparison",
        "",
        "## Gross Summary",
    ]
    gross = scenario_metrics[scenario_metrics["scenario"] == "gross"].copy()
    if not gross.empty:
        gross = gross.sort_values("information_ratio", ascending=False).reset_index(drop=True)
        for _, row in gross.iterrows():
            lines.append(
                f"- {row['variant']}: strategy={row['strategy_cumulative_return']:.2%}, "
                f"benchmark={row['benchmark_cumulative_return']:.2%}, "
                f"diff={row['cumulative_return_diff']:.2%}, "
                f"IR={row['information_ratio']:.4f}, "
                f"Sharpe={row['strategy_sharpe_ratio']:.4f}"
            )
    (output_dir / "variant_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(output_dir / "variant_metrics.parquet")
    print(output_dir / "variant_scenarios.parquet")
    print(output_dir / "variant_regimes.parquet")
    print(output_dir / "variant_weekly_positions.parquet")
    print(output_dir / "variant_comparison.md")


if __name__ == "__main__":
    main()
