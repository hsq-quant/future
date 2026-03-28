from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_summary_report(
    *,
    metrics: pd.DataFrame,
    strategy: pd.DataFrame,
    shap_summary: pd.DataFrame,
    output_path: str | Path,
    market_source: str,
    calendar_source: str,
    qwen_model: str,
    prompt_version: str,
    feeds: list[str],
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    strategy_stats = {
        "cumulative_return": float(strategy["cum_return"].iloc[-1]) if not strategy.empty else 0.0,
        "avg_weekly_return": float(strategy["strategy_return"].mean()) if not strategy.empty else 0.0,
        "volatility": float(strategy["strategy_return"].std(ddof=0)) if len(strategy) > 0 else 0.0,
        "win_rate": float((strategy["strategy_return"] > 0).mean()) if not strategy.empty else 0.0,
        "max_drawdown": float((strategy["cum_return"] - strategy["cum_return"].cummax()).min()) if not strategy.empty else 0.0,
    }

    metrics_summary = metrics[["auc", "accuracy", "ic"]].agg(["mean", "std"]).round(4)
    top_features = shap_summary.head(10)

    report = f"""# INE SC RSS Sentiment Summary

## Run Metadata

- Market data source: {market_source}
- Calendar source: {calendar_source}
- Qwen model: {qwen_model}
- Prompt version: {prompt_version}
- Feed count: {len(feeds)}

## Feeds

{chr(10).join(f"- {feed}" for feed in feeds)}

## Cross-Validation Metrics

{metrics_summary.to_markdown()}

## Strategy Summary

- Cumulative return: {strategy_stats['cumulative_return']:.4f}
- Average weekly return: {strategy_stats['avg_weekly_return']:.4f}
- Weekly volatility: {strategy_stats['volatility']:.4f}
- Win rate: {strategy_stats['win_rate']:.4f}
- Max drawdown: {strategy_stats['max_drawdown']:.4f}

## Top SHAP Features

{top_features.to_markdown(index=False)}
"""
    out.write_text(report, encoding="utf-8")
    return out
