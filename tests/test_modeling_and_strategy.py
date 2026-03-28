from __future__ import annotations

import math

import pandas as pd

from src.models.strategy import (
    apply_weekly_trading_costs,
    build_long_benchmark,
    build_weekly_classification_strategy,
    build_weekly_factor_strategy,
    build_weekly_strategy,
    compute_causal_zscore,
    evaluate_regime_comparison,
    evaluate_strategy_regimes,
    map_factor_signal_to_position,
    render_strategy_report,
    summarize_strategy_vs_benchmark,
    summarize_weekly_strategy,
)
from src.models.thresholds import apply_class_share_threshold


def test_apply_class_share_threshold_matches_training_positive_share() -> None:
    validation_probs = pd.Series([0.9, 0.7, 0.4, 0.3, 0.2], dtype=float)

    threshold, labels = apply_class_share_threshold(
        validation_probs=validation_probs,
        training_positive_share=0.4,
    )

    assert math.isclose(threshold, 0.7)
    assert labels.tolist() == [1, 1, 0, 0, 0]


def test_build_weekly_strategy_uses_oos_labels_to_create_positions_and_returns() -> None:
    predictions = pd.DataFrame(
        {
            "week_end_date": pd.to_datetime(["2026-01-09", "2026-01-16", "2026-01-23"]),
            "pred_label": [1, 0, 1],
            "actual_return": [0.10, -0.05, 0.02],
        }
    )

    strategy = build_weekly_strategy(predictions)

    assert strategy["position"].tolist() == [1, -1, 1]
    assert strategy["strategy_return"].round(6).tolist() == [0.10, 0.05, 0.02]
    assert strategy["cum_return"].round(4).tolist() == [0.10, 0.155, 0.1781]


def test_build_weekly_classification_strategy_supports_long_only_and_threshold_short_only() -> None:
    predictions = pd.DataFrame(
        {
            "week_end_date": pd.to_datetime(["2026-01-09", "2026-01-16", "2026-01-23", "2026-01-30"]),
            "pred_prob": [0.72, 0.41, 0.18, 0.49],
            "pred_label": [1, 0, 0, 1],
            "threshold_used": [0.5, 0.5, 0.5, 0.5],
            "actual_return": [0.10, -0.05, 0.08, 0.02],
        }
    )

    long_only = build_weekly_classification_strategy(predictions, mapping_method="long_only")
    thresholded = build_weekly_classification_strategy(
        predictions,
        mapping_method="threshold_short_only",
        short_probability_threshold=0.25,
    )

    assert long_only["position"].tolist() == [1.0, 0.0, 0.0, 1.0]
    assert thresholded["position"].tolist() == [1.0, 0.0, -1.0, 1.0]
    assert thresholded["strategy_return"].round(6).tolist() == [0.10, -0.0, -0.08, 0.02]


def test_build_long_benchmark_uses_realized_returns_as_always_long_baseline() -> None:
    predictions = pd.DataFrame(
        {
            "week_end_date": pd.to_datetime(["2026-01-09", "2026-01-16", "2026-01-23"]),
            "actual_return": [0.10, -0.05, 0.02],
        }
    )

    benchmark = build_long_benchmark(predictions)

    assert benchmark["benchmark_position"].tolist() == [1, 1, 1]
    assert benchmark["benchmark_return"].round(6).tolist() == [0.10, -0.05, 0.02]
    assert benchmark["benchmark_cum_return"].round(4).tolist() == [0.10, 0.045, 0.0659]


def test_summarize_weekly_strategy_returns_core_metrics() -> None:
    predictions = pd.DataFrame(
        {
            "week_end_date": pd.to_datetime(["2026-01-09", "2026-01-16", "2026-01-23", "2026-01-30"]),
            "pred_label": [1, 0, 1, 0],
            "actual_return": [0.10, -0.05, -0.02, 0.03],
        }
    )
    strategy = build_weekly_strategy(predictions)

    summary = summarize_weekly_strategy(strategy)

    assert summary["num_weeks"] == 4
    assert math.isclose(summary["cumulative_return"], float(strategy["cum_return"].iloc[-1]))
    assert math.isclose(summary["win_rate"], 0.5)
    assert math.isclose(summary["long_share"], 0.5)
    assert math.isclose(summary["short_share"], 0.5)
    assert math.isclose(summary["position_change_rate"], 0.75)
    assert summary["max_drawdown"] <= 0.0
    assert "annualized_return" in summary
    assert "annualized_volatility" in summary
    assert "sharpe_ratio" in summary


def test_summarize_strategy_vs_benchmark_returns_active_metrics() -> None:
    predictions = pd.DataFrame(
        {
            "week_end_date": pd.to_datetime(["2026-01-09", "2026-01-16", "2026-01-23", "2026-01-30"]),
            "pred_label": [1, 0, 1, 0],
            "actual_return": [0.10, -0.05, -0.02, 0.03],
        }
    )
    strategy = build_weekly_strategy(predictions)
    strategy = strategy.merge(
        build_long_benchmark(strategy)[["week_end_date", "benchmark_position", "benchmark_return"]],
        on="week_end_date",
        how="left",
    )

    comparison = summarize_strategy_vs_benchmark(strategy)

    assert "strategy_cumulative_return" in comparison
    assert "benchmark_cumulative_return" in comparison
    assert "cumulative_return_diff" in comparison
    assert "information_ratio" in comparison
    assert math.isclose(
        comparison["cumulative_return_diff"],
        comparison["strategy_cumulative_return"] - comparison["benchmark_cumulative_return"],
    )


def test_render_strategy_report_contains_key_sections() -> None:
    summary = {
        "num_weeks": 10,
        "cumulative_return": 0.12,
        "annualized_return": 0.15,
        "annualized_volatility": 0.2,
        "sharpe_ratio": 0.75,
        "max_drawdown": -0.1,
        "win_rate": 0.6,
        "long_share": 0.55,
        "short_share": 0.45,
        "position_change_rate": 0.3,
        "long_cumulative_return": 0.08,
        "short_cumulative_return": 0.04,
        "long_mean_return": 0.01,
        "short_mean_return": 0.005,
    }

    report = render_strategy_report(summary)

    assert "Strategy Summary" in report
    assert "Annualized Return" in report
    assert "Long/Short Split" in report
    assert "12.00%" in report

    benchmark_summary = {
        "cumulative_return": 0.10,
        "annualized_return": 0.12,
        "annualized_volatility": 0.18,
        "sharpe_ratio": 0.66,
        "max_drawdown": -0.08,
        "win_rate": 0.55,
    }
    comparison_summary = {
        "cumulative_return_diff": 0.02,
        "annualized_return_diff": 0.03,
        "mean_active_return": 0.001,
        "annualized_active_return": 0.052,
        "tracking_error": 0.11,
        "information_ratio": 0.47,
        "active_win_rate": 0.57,
        "sharpe_ratio_diff": 0.09,
    }
    detailed_report = render_strategy_report(
        summary,
        benchmark_summary=benchmark_summary,
        comparison_summary=comparison_summary,
    )
    assert "Always-Long Benchmark" in detailed_report
    assert "Active Return vs Benchmark" in detailed_report


def test_apply_weekly_trading_costs_adds_cost_columns_and_net_returns() -> None:
    strategy = pd.DataFrame(
        {
            "week_end_date": pd.to_datetime(["2026-01-09", "2026-01-16", "2026-01-23"]),
            "position": [1, 1, -1],
            "strategy_return": [0.10, 0.05, 0.02],
            "weekly_close": [100.0, 100.0, 100.0],
        }
    )

    net = apply_weekly_trading_costs(
        strategy,
        commission_per_lot=1.0,
        slippage_ticks=0,
        tick_size=1.0,
        contract_size=1.0,
        include_terminal_exit=True,
    )

    assert net["side_trades"].tolist() == [1, 0, 3]
    assert net["cost_return"].round(4).tolist() == [0.01, 0.0, 0.03]
    assert net["net_strategy_return"].round(4).tolist() == [0.09, 0.05, -0.01]
    assert "net_cum_return" in net.columns

    benchmark = strategy.copy()
    benchmark["benchmark_position"] = [1, 1, 1]
    benchmark["benchmark_return"] = [0.10, 0.05, 0.02]
    benchmark_net = apply_weekly_trading_costs(
        benchmark,
        commission_per_lot=1.0,
        slippage_ticks=0,
        tick_size=1.0,
        contract_size=1.0,
        include_terminal_exit=True,
        gross_return_column="benchmark_return",
        net_return_column="net_benchmark_return",
        position_column="benchmark_position",
        cum_return_column="net_benchmark_cum_return",
    )
    assert benchmark_net["side_trades"].tolist() == [1, 0, 1]
    assert benchmark_net["net_benchmark_return"].round(4).tolist() == [0.09, 0.05, 0.01]


def test_evaluate_strategy_regimes_splits_metrics_by_period() -> None:
    strategy = pd.DataFrame(
        {
            "week_end_date": pd.to_datetime(["2026-01-09", "2026-01-16", "2026-01-23", "2026-01-30"]),
            "net_strategy_return": [0.10, -0.05, 0.02, 0.01],
            "position": [1, -1, 1, -1],
        }
    )
    regimes = [
        {"regime": "bull", "start": "2026-01-09", "end": "2026-01-16"},
        {"regime": "range", "start": "2026-01-23", "end": "2026-01-30"},
    ]

    result = evaluate_strategy_regimes(strategy, regimes, return_column="net_strategy_return")

    assert result["regime"].tolist() == ["bull", "range"]
    assert result["num_weeks"].tolist() == [2, 2]
    assert "sharpe_ratio" in result.columns
    assert "max_drawdown" in result.columns


def test_evaluate_regime_comparison_reports_strategy_and_benchmark_metrics() -> None:
    strategy = pd.DataFrame(
        {
            "week_end_date": pd.to_datetime(["2026-01-09", "2026-01-16", "2026-01-23", "2026-01-30"]),
            "strategy_return": [0.10, -0.05, 0.02, 0.01],
            "benchmark_return": [0.08, -0.01, 0.01, 0.00],
            "position": [1, -1, 1, -1],
            "benchmark_position": [1, 1, 1, 1],
        }
    )
    regimes = [
        {"regime": "bull", "start": "2026-01-09", "end": "2026-01-16"},
        {"regime": "range", "start": "2026-01-23", "end": "2026-01-30"},
    ]

    result = evaluate_regime_comparison(strategy, regimes)

    assert result["regime"].tolist() == ["bull", "range"]
    assert "strategy_cumulative_return" in result.columns
    assert "benchmark_cumulative_return" in result.columns
    assert "cumulative_return_diff" in result.columns


def test_compute_causal_zscore_uses_only_past_predictions() -> None:
    predictions = pd.Series([0.10, 0.20, 0.30, 0.40], dtype=float)

    z = compute_causal_zscore(predictions)

    assert z.iloc[0] == 0.0
    assert z.iloc[1] == 0.0
    assert round(z.iloc[2], 6) == 3.0
    assert round(z.iloc[3], 6) == 2.44949


def test_map_factor_signal_to_position_supports_linear_clip_and_tanh() -> None:
    z = pd.Series([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=float)

    linear = map_factor_signal_to_position(z, method="linear_clip", scale=0.5)
    tanh = map_factor_signal_to_position(z, method="tanh", k=1.0)

    assert linear.round(3).tolist() == [-1.0, -0.5, 0.0, 0.5, 1.0]
    assert tanh.iloc[0] < -0.99
    assert tanh.iloc[-1] > 0.99


def test_map_factor_signal_to_position_supports_long_only_variant() -> None:
    z = pd.Series([-2.0, -0.5, 0.0, 1.0, 3.0], dtype=float)

    long_only = map_factor_signal_to_position(z, method="long_only_clip", scale=0.5)

    assert long_only.round(3).tolist() == [0.0, 0.0, 0.0, 0.5, 1.0]


def test_map_factor_signal_to_position_supports_asymmetric_long_short_variant() -> None:
    z = pd.Series([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=float)

    asymmetric = map_factor_signal_to_position(
        z,
        method="asymmetric_tanh",
        k=1.0,
        long_scale=1.0,
        short_scale=0.25,
    )

    assert asymmetric.iloc[0] < 0.0
    assert asymmetric.iloc[0] > -0.3
    assert asymmetric.iloc[-1] > 0.99
    assert asymmetric.iloc[0] != -asymmetric.iloc[-1]


def test_map_factor_signal_to_position_supports_threshold_short_only_variant() -> None:
    z = pd.Series([-2.0, -0.5, 0.0, 1.0, 2.0], dtype=float)

    thresholded = map_factor_signal_to_position(
        z,
        method="threshold_short_only",
        k=1.0,
        long_scale=1.0,
        short_scale=0.5,
        short_threshold=1.0,
    )

    assert thresholded.iloc[0] < 0.0
    assert thresholded.iloc[1] == 0.0
    assert thresholded.iloc[2] == 0.0
    assert thresholded.iloc[3] > 0.0
    assert thresholded.iloc[4] > thresholded.iloc[3]


def test_build_weekly_factor_strategy_maps_prediction_strength_to_continuous_position() -> None:
    predictions = pd.DataFrame(
        {
            "week_end_date": pd.to_datetime(["2026-01-09", "2026-01-16", "2026-01-23", "2026-01-30"]),
            "pred_value": [0.10, 0.20, 0.30, 0.40],
            "actual_return": [0.10, -0.05, 0.02, 0.01],
        }
    )

    strategy = build_weekly_factor_strategy(predictions, mapping_method="linear_clip", scale=0.5)

    assert "zscore_signal" in strategy.columns
    assert "position" in strategy.columns
    assert strategy["position"].iloc[0] == 0.0
    assert strategy["position"].iloc[2] > 0.0
    assert strategy["strategy_return"].iloc[2] == strategy["position"].iloc[2] * strategy["actual_return"].iloc[2]
