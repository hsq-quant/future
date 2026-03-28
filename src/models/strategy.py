from __future__ import annotations

import math

import pandas as pd


def compute_causal_zscore(predictions: pd.Series) -> pd.Series:
    pred = predictions.astype(float).reset_index(drop=True)
    expanding_mean = pred.expanding().mean().shift(1)
    expanding_std = pred.expanding().std(ddof=0).shift(1)
    safe_std = expanding_std.mask(expanding_std == 0.0)
    z = ((pred - expanding_mean) / safe_std).astype(float).fillna(0.0)
    return z


def map_factor_signal_to_position(
    zscore_signal: pd.Series,
    *,
    method: str = "linear_clip",
    scale: float = 0.5,
    k: float = 1.0,
    long_scale: float = 1.0,
    short_scale: float = 1.0,
    short_threshold: float = 0.0,
) -> pd.Series:
    signal = zscore_signal.astype(float)
    if method == "linear_clip":
        return (signal * float(scale)).clip(-1.0, 1.0)
    if method == "tanh":
        return signal.map(lambda value: math.tanh(float(k) * float(value)))
    if method == "long_only_clip":
        return (signal.clip(lower=0.0) * float(scale)).clip(0.0, 1.0)
    if method == "asymmetric_tanh":
        def _map_asymmetric(value: float) -> float:
            numeric = float(value)
            if numeric >= 0.0:
                return min(1.0, math.tanh(float(k) * numeric) * float(long_scale))
            return max(-1.0, -math.tanh(float(k) * abs(numeric)) * float(short_scale))

        return signal.map(_map_asymmetric)
    if method == "threshold_short_only":
        threshold = float(short_threshold)

        def _map_thresholded(value: float) -> float:
            numeric = float(value)
            if numeric >= 0.0:
                return min(1.0, math.tanh(float(k) * numeric) * float(long_scale))
            if abs(numeric) <= threshold:
                return 0.0
            return max(-1.0, -math.tanh(float(k) * (abs(numeric) - threshold)) * float(short_scale))

        return signal.map(_map_thresholded)
    raise ValueError(f"Unsupported factor mapping method: {method}")


def build_weekly_factor_strategy(
    predictions: pd.DataFrame,
    *,
    prediction_column: str = "pred_value",
    mapping_method: str = "linear_clip",
    scale: float = 0.5,
    k: float = 1.0,
    long_scale: float = 1.0,
    short_scale: float = 1.0,
    short_threshold: float = 0.0,
) -> pd.DataFrame:
    weekly = predictions.copy()
    weekly["week_end_date"] = pd.to_datetime(weekly["week_end_date"]).dt.normalize()
    weekly = weekly.sort_values("week_end_date").reset_index(drop=True)
    weekly["zscore_signal"] = compute_causal_zscore(weekly[prediction_column])
    weekly["position"] = map_factor_signal_to_position(
        weekly["zscore_signal"],
        method=mapping_method,
        scale=scale,
        k=k,
        long_scale=long_scale,
        short_scale=short_scale,
        short_threshold=short_threshold,
    )
    weekly["strategy_return"] = weekly["position"] * weekly["actual_return"].astype(float)
    weekly["cum_return"] = (1.0 + weekly["strategy_return"]).cumprod() - 1.0
    return weekly


def build_weekly_strategy(predictions: pd.DataFrame) -> pd.DataFrame:
    """Create the weekly always-in paper strategy from out-of-sample labels."""
    weekly = predictions.copy()
    weekly["week_end_date"] = pd.to_datetime(weekly["week_end_date"]).dt.normalize()
    weekly = weekly.sort_values("week_end_date").reset_index(drop=True)
    weekly["position"] = weekly["pred_label"].map({1: 1.0, 0: -1.0}).astype(float)
    weekly["strategy_return"] = weekly["position"] * weekly["actual_return"]
    weekly["cum_return"] = (1.0 + weekly["strategy_return"]).cumprod() - 1.0
    return weekly


def build_weekly_classification_strategy(
    predictions: pd.DataFrame,
    *,
    mapping_method: str = "always_in",
    short_probability_threshold: float = 0.25,
) -> pd.DataFrame:
    weekly = predictions.copy()
    weekly["week_end_date"] = pd.to_datetime(weekly["week_end_date"]).dt.normalize()
    weekly = weekly.sort_values("week_end_date").reset_index(drop=True)

    if mapping_method == "always_in":
        positions = weekly["pred_label"].map({1: 1.0, 0: -1.0}).astype(float)
    elif mapping_method == "long_only":
        positions = weekly["pred_label"].astype(float)
    elif mapping_method == "threshold_short_only":
        if "pred_prob" not in weekly.columns:
            raise ValueError("pred_prob is required for threshold_short_only mapping.")
        positions = pd.Series(0.0, index=weekly.index, dtype=float)
        positions.loc[weekly["pred_label"] == 1] = 1.0
        positions.loc[weekly["pred_prob"].astype(float) <= float(short_probability_threshold)] = -1.0
    else:
        raise ValueError(f"Unsupported classification mapping method: {mapping_method}")

    weekly["position"] = positions.astype(float)
    weekly["strategy_return"] = weekly["position"] * weekly["actual_return"].astype(float)
    weekly["cum_return"] = (1.0 + weekly["strategy_return"]).cumprod() - 1.0
    return weekly


def build_long_benchmark(predictions_or_strategy: pd.DataFrame) -> pd.DataFrame:
    """Create a same-horizon always-long benchmark from realized weekly returns."""
    weekly = predictions_or_strategy.copy()
    weekly["week_end_date"] = pd.to_datetime(weekly["week_end_date"]).dt.normalize()
    weekly = weekly.sort_values("week_end_date").reset_index(drop=True)
    weekly["benchmark_position"] = 1
    weekly["benchmark_return"] = weekly["actual_return"].astype(float)
    weekly["benchmark_cum_return"] = (1.0 + weekly["benchmark_return"]).cumprod() - 1.0
    return weekly


def summarize_weekly_strategy(
    strategy: pd.DataFrame,
    periods_per_year: int = 52,
    return_column: str = "strategy_return",
    position_column: str = "position",
) -> dict[str, float | int]:
    weekly = strategy.sort_values("week_end_date").reset_index(drop=True).copy()
    returns = weekly[return_column].astype(float)
    n_periods = len(weekly)

    wealth = (1.0 + returns).cumprod()
    cumulative_return = float(wealth.iloc[-1] - 1.0) if n_periods else 0.0
    annualized_return = (float(wealth.iloc[-1]) ** (periods_per_year / n_periods) - 1.0) if n_periods else 0.0
    annualized_volatility = float(returns.std(ddof=0) * math.sqrt(periods_per_year)) if n_periods else 0.0
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else float("nan")
    drawdown = wealth / wealth.cummax() - 1.0 if n_periods else pd.Series(dtype=float)
    positions = weekly[position_column].astype(float)
    position_changes = positions.diff().fillna(0.0).abs() > 1e-12

    long_mask = positions > 0
    short_mask = positions < 0
    long_returns = weekly.loc[long_mask, return_column].astype(float)
    short_returns = weekly.loc[short_mask, return_column].astype(float)

    def _compound(series: pd.Series) -> float:
        if series.empty:
            return 0.0
        return float((1.0 + series).prod() - 1.0)

    def _mean(series: pd.Series) -> float:
        return float(series.mean()) if not series.empty else 0.0

    return {
        "num_weeks": int(n_periods),
        "cumulative_return": cumulative_return,
        "annualized_return": float(annualized_return),
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
        "win_rate": float((returns > 0).mean()) if n_periods else 0.0,
        "long_share": float(long_mask.mean()) if n_periods else 0.0,
        "short_share": float(short_mask.mean()) if n_periods else 0.0,
        "position_change_rate": float(position_changes.mean()) if n_periods else 0.0,
        "long_cumulative_return": _compound(long_returns),
        "short_cumulative_return": _compound(short_returns),
        "long_mean_return": _mean(long_returns),
        "short_mean_return": _mean(short_returns),
    }


def apply_weekly_trading_costs(
    strategy: pd.DataFrame,
    *,
    commission_per_lot: float,
    slippage_ticks: int,
    tick_size: float,
    contract_size: float,
    include_terminal_exit: bool = True,
    gross_return_column: str = "strategy_return",
    net_return_column: str = "net_strategy_return",
    position_column: str = "position",
    cum_return_column: str = "net_cum_return",
) -> pd.DataFrame:
    weekly = strategy.sort_values("week_end_date").reset_index(drop=True).copy()
    if "weekly_close" not in weekly.columns:
        raise ValueError("weekly_close is required to convert fixed trading costs into return space.")

    if gross_return_column not in weekly.columns:
        raise ValueError(f"{gross_return_column} is required to subtract trading costs.")
    if position_column not in weekly.columns:
        raise ValueError(f"{position_column} is required to estimate turnover costs.")

    one_way_cost_rmb = float(commission_per_lot) + float(slippage_ticks) * float(tick_size) * float(contract_size)
    side_trades = []
    positions = weekly[position_column].astype(float)
    for idx in range(len(weekly)):
        if idx == 0:
            trades = abs(float(positions.iloc[idx]))
        else:
            trades = abs(float(positions.iloc[idx]) - float(positions.iloc[idx - 1]))
        if include_terminal_exit and idx == len(weekly) - 1:
            trades += abs(float(positions.iloc[idx]))
        side_trades.append(trades)

    weekly["side_trades"] = side_trades
    weekly["one_way_cost_rmb"] = one_way_cost_rmb
    weekly["cost_rmb"] = weekly["side_trades"] * weekly["one_way_cost_rmb"]
    weekly["entry_notional_rmb"] = weekly["weekly_close"].astype(float) * float(contract_size)
    weekly["cost_return"] = weekly["cost_rmb"] / weekly["entry_notional_rmb"]
    weekly[net_return_column] = weekly[gross_return_column].astype(float) - weekly["cost_return"]
    weekly[cum_return_column] = (1.0 + weekly[net_return_column]).cumprod() - 1.0
    return weekly


def evaluate_strategy_regimes(
    strategy: pd.DataFrame,
    regimes: list[dict[str, str]],
    *,
    return_column: str = "strategy_return",
    periods_per_year: int = 52,
) -> pd.DataFrame:
    weekly = strategy.copy()
    weekly["week_end_date"] = pd.to_datetime(weekly["week_end_date"]).dt.normalize()
    rows: list[dict[str, float | int | str]] = []
    for regime in regimes:
        start = pd.Timestamp(regime["start"]).normalize()
        end = pd.Timestamp(regime["end"]).normalize()
        window = weekly[(weekly["week_end_date"] >= start) & (weekly["week_end_date"] <= end)].copy()
        if window.empty:
            continue
        summary = summarize_weekly_strategy(window, periods_per_year=periods_per_year, return_column=return_column)
        rows.append(
            {
                "regime": regime["regime"],
                "start": start,
                "end": end,
                **summary,
            }
        )
    return pd.DataFrame(rows)


def summarize_strategy_vs_benchmark(
    strategy: pd.DataFrame,
    *,
    strategy_return_column: str = "strategy_return",
    benchmark_return_column: str = "benchmark_return",
    benchmark_position_column: str = "benchmark_position",
    periods_per_year: int = 52,
) -> dict[str, float | int]:
    strategy_summary = summarize_weekly_strategy(
        strategy,
        periods_per_year=periods_per_year,
        return_column=strategy_return_column,
    )
    benchmark_summary = summarize_weekly_strategy(
        strategy,
        periods_per_year=periods_per_year,
        return_column=benchmark_return_column,
        position_column=benchmark_position_column,
    )

    strategy_returns = strategy[strategy_return_column].astype(float)
    benchmark_returns = strategy[benchmark_return_column].astype(float)
    active_returns = strategy_returns - benchmark_returns
    tracking_error = float(active_returns.std(ddof=0) * math.sqrt(periods_per_year)) if len(active_returns) else 0.0
    active_mean_return = float(active_returns.mean()) if len(active_returns) else 0.0
    annualized_active_return = active_mean_return * periods_per_year
    information_ratio = annualized_active_return / tracking_error if tracking_error > 0 else float("nan")

    comparison = {
        **{f"strategy_{key}": value for key, value in strategy_summary.items()},
        **{f"benchmark_{key}": value for key, value in benchmark_summary.items()},
        "mean_active_return": active_mean_return,
        "annualized_active_return": annualized_active_return,
        "tracking_error": tracking_error,
        "information_ratio": float(information_ratio),
        "active_win_rate": float((active_returns > 0).mean()) if len(active_returns) else 0.0,
        "cumulative_return_diff": float(strategy_summary["cumulative_return"] - benchmark_summary["cumulative_return"]),
        "annualized_return_diff": float(strategy_summary["annualized_return"] - benchmark_summary["annualized_return"]),
        "sharpe_ratio_diff": float(strategy_summary["sharpe_ratio"] - benchmark_summary["sharpe_ratio"]),
    }
    return comparison


def evaluate_regime_comparison(
    strategy: pd.DataFrame,
    regimes: list[dict[str, str]],
    *,
    strategy_return_column: str = "strategy_return",
    benchmark_return_column: str = "benchmark_return",
    periods_per_year: int = 52,
) -> pd.DataFrame:
    weekly = strategy.copy()
    weekly["week_end_date"] = pd.to_datetime(weekly["week_end_date"]).dt.normalize()
    rows: list[dict[str, float | int | str]] = []
    for regime in regimes:
        start = pd.Timestamp(regime["start"]).normalize()
        end = pd.Timestamp(regime["end"]).normalize()
        window = weekly[(weekly["week_end_date"] >= start) & (weekly["week_end_date"] <= end)].copy()
        if window.empty:
            continue
        summary = summarize_strategy_vs_benchmark(
            window,
            strategy_return_column=strategy_return_column,
            benchmark_return_column=benchmark_return_column,
            periods_per_year=periods_per_year,
        )
        rows.append({"regime": regime["regime"], "start": start, "end": end, **summary})
    return pd.DataFrame(rows)


def render_strategy_report(
    summary: dict[str, float | int],
    benchmark_summary: dict[str, float | int] | None = None,
    comparison_summary: dict[str, float | int] | None = None,
    benchmark_label: str = "Always-Long Benchmark",
    date_range: tuple[str, str] | None = None,
    regime_coverage_note: str | None = None,
) -> str:
    def _pct(key: str) -> str:
        return f"{float(summary[key]) * 100:.2f}%"

    def _num(key: str) -> str:
        value = float(summary[key])
        return "NaN" if math.isnan(value) else f"{value:.4f}"

    lines = [
        "# Strategy Summary",
        "",
        "## Coverage",
        f"- Weeks: {int(summary['num_weeks'])}",
    ]
    if date_range is not None:
        lines.extend(
            [
                f"- Start Week: {date_range[0]}",
                f"- End Week: {date_range[1]}",
            ]
        )
    if regime_coverage_note:
        lines.append(f"- Regime Coverage Note: {regime_coverage_note}")
    lines.extend(
        [
        "",
        "## Core Metrics",
        f"- Cumulative Return: {_pct('cumulative_return')}",
        f"- Annualized Return: {_pct('annualized_return')}",
        f"- Annualized Volatility: {_pct('annualized_volatility')}",
        f"- Sharpe Ratio: {_num('sharpe_ratio')}",
        f"- Max Drawdown: {_pct('max_drawdown')}",
        f"- Win Rate: {_pct('win_rate')}",
        "",
        "## Long/Short Split",
        f"- Long Share: {_pct('long_share')}",
        f"- Short Share: {_pct('short_share')}",
        f"- Long Cumulative Return: {_pct('long_cumulative_return')}",
        f"- Short Cumulative Return: {_pct('short_cumulative_return')}",
        f"- Long Mean Weekly Return: {_pct('long_mean_return')}",
        f"- Short Mean Weekly Return: {_pct('short_mean_return')}",
        "",
        "## Trading Activity",
        f"- Position Change Rate: {_pct('position_change_rate')}",
    ])

    if benchmark_summary is not None:
        def _bench_pct(key: str) -> str:
            return f"{float(benchmark_summary[key]) * 100:.2f}%"

        def _bench_num(key: str) -> str:
            value = float(benchmark_summary[key])
            return "NaN" if math.isnan(value) else f"{value:.4f}"

        lines.extend(
            [
                "",
                f"## {benchmark_label}",
                f"- Cumulative Return: {_bench_pct('cumulative_return')}",
                f"- Annualized Return: {_bench_pct('annualized_return')}",
                f"- Annualized Volatility: {_bench_pct('annualized_volatility')}",
                f"- Sharpe Ratio: {_bench_num('sharpe_ratio')}",
                f"- Max Drawdown: {_bench_pct('max_drawdown')}",
                f"- Win Rate: {_bench_pct('win_rate')}",
            ]
        )

    if comparison_summary is not None:
        def _cmp_pct(key: str) -> str:
            return f"{float(comparison_summary[key]) * 100:.2f}%"

        def _cmp_num(key: str) -> str:
            value = float(comparison_summary[key])
            return "NaN" if math.isnan(value) else f"{value:.4f}"

        lines.extend(
            [
                "",
                "## Active Return vs Benchmark",
                f"- Cumulative Return Difference: {_cmp_pct('cumulative_return_diff')}",
                f"- Annualized Return Difference: {_cmp_pct('annualized_return_diff')}",
                f"- Mean Weekly Active Return: {_cmp_pct('mean_active_return')}",
                f"- Annualized Active Return: {_cmp_pct('annualized_active_return')}",
                f"- Tracking Error: {_cmp_pct('tracking_error')}",
                f"- Information Ratio: {_cmp_num('information_ratio')}",
                f"- Active Win Rate: {_cmp_pct('active_win_rate')}",
                f"- Sharpe Difference: {_cmp_num('sharpe_ratio_diff')}",
            ]
        )

    return "\n".join(lines)
