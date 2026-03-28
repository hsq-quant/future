from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.strategy import (
    build_long_benchmark,
    build_weekly_classification_strategy,
    evaluate_regime_comparison,
    summarize_strategy_vs_benchmark,
)
from src.utils.io import read_dataframe, read_yaml, write_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiple classification-mapping variants for one prediction set.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--strategy-config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def _read_prefer_csv(path: Path) -> pd.DataFrame:
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return read_dataframe(path)


def main() -> None:
    args = parse_args()
    model_cfg = read_yaml(args.config)
    strategy_cfg = read_yaml(args.strategy_config).get("strategy", {})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = _read_prefer_csv(ROOT / model_cfg["artifacts"]["predictions"])
    weekly_labels = read_dataframe(ROOT / "data/processed/weekly_labels.parquet")[["week_end_date", "weekly_close"]].copy()
    variants = list(strategy_cfg.get("classification_variants", []))
    regimes = list(strategy_cfg.get("regimes", []))

    variant_rows: list[dict[str, object]] = []
    scenario_rows: list[dict[str, object]] = []
    regime_frames: list[pd.DataFrame] = []
    weekly_frames: list[pd.DataFrame] = []

    for variant in variants:
        base = build_weekly_classification_strategy(
            predictions,
            mapping_method=str(variant.get("mapping_method", "always_in")),
            short_probability_threshold=float(variant.get("short_probability_threshold", 0.25)),
        )
        base = base.merge(weekly_labels, on="week_end_date", how="left")
        base = base.merge(
            build_long_benchmark(base)[["week_end_date", "benchmark_position", "benchmark_return", "benchmark_cum_return"]],
            on="week_end_date",
            how="left",
        )
        base.insert(0, "variant", str(variant["name"]))
        base.insert(1, "variant_label", str(variant.get("label", variant["name"])))
        weekly_frames.append(base)
        summary = summarize_strategy_vs_benchmark(base)
        variant_rows.append({"variant": variant["name"], "variant_label": variant.get("label", variant["name"]), **summary})
        scenario_rows.append({"variant": variant["name"], "variant_label": variant.get("label", variant["name"]), "scenario": "gross", **summary})
        regime_df = evaluate_regime_comparison(base, regimes)
        if not regime_df.empty:
            regime_df.insert(0, "variant", str(variant["name"]))
            regime_df.insert(1, "variant_label", str(variant.get("label", variant["name"])))
            regime_df.insert(2, "scenario", "gross")
            regime_frames.append(regime_df)

    variant_metrics = pd.DataFrame(variant_rows)
    variant_scenarios = pd.DataFrame(scenario_rows)
    variant_regimes = pd.concat(regime_frames, ignore_index=True) if regime_frames else pd.DataFrame()
    weekly_positions = pd.concat(weekly_frames, ignore_index=True) if weekly_frames else pd.DataFrame()

    write_dataframe(variant_metrics, output_dir / "variant_metrics.parquet")
    write_dataframe(variant_scenarios, output_dir / "variant_scenarios.parquet")
    write_dataframe(variant_regimes, output_dir / "variant_regimes.parquet")
    write_dataframe(weekly_positions, output_dir / "variant_weekly_positions.parquet")
    variant_metrics.to_csv(output_dir / "variant_metrics.csv", index=False)
    variant_scenarios.to_csv(output_dir / "variant_scenarios.csv", index=False)
    variant_regimes.to_csv(output_dir / "variant_regimes.csv", index=False)
    weekly_positions.to_csv(output_dir / "variant_weekly_positions.csv", index=False)

    lines = ["# Classification Mapping Comparison", "", "## Gross Summary"]
    for _, row in variant_metrics.sort_values("strategy_sharpe_ratio", ascending=False).iterrows():
        lines.append(
            f"- {row['variant']}: strategy={row['strategy_cumulative_return']:.2%}, "
            f"benchmark={row['benchmark_cumulative_return']:.2%}, "
            f"diff={row['cumulative_return_diff']:.2%}, "
            f"IR={row['information_ratio']:.4f}, "
            f"Sharpe={row['strategy_sharpe_ratio']:.4f}"
        )
    (output_dir / "variant_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(output_dir / "variant_metrics.parquet")


if __name__ == "__main__":
    main()
