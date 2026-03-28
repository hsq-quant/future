from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.market_labels import build_weekly_labels, stitch_continuous_near_month
from src.data.market_source import MarketFetchConfig, load_market_data
from src.data.rss_ingest import clean_articles, ingest_rss_feeds, load_feed_config
from src.features.scoring import PROMPT_VERSION, score_article_with_qwen
from src.features.weekly_features import aggregate_weekly_features
from src.models.reporting import build_summary_report
from src.models.strategy import build_weekly_strategy
from src.models.train_eval import TrainingConfig, export_shap_outputs, train_lightgbm_cv
from src.utils.io import read_yaml, write_dataframe


def run_pipeline(project_root: str | Path = ".") -> dict[str, Path]:
    root = Path(project_root)
    market_config = read_yaml(root / "configs/market.yaml")
    model_config = read_yaml(root / "configs/model.yaml")
    feeds, energy_keywords = load_feed_config(str(root / "configs/feeds.yaml"))

    market_settings = MarketFetchConfig(
        source=market_config["market"]["source"],
        start_date=market_config["market"]["start_date"],
        end_date=market_config["market"]["end_date"],
        exchange=market_config["market"]["exchange"],
        local_input_glob=str(root / market_config["market"]["local_input_glob"]),
        contract_metadata_file=str(root / market_config["market"]["contract_metadata_file"]),
    )

    raw_market = load_market_data(market_settings)
    stitched_daily = stitch_continuous_near_month(
        raw_market,
        roll_days_before_last_trade=market_config["market"]["roll_days_before_last_trade"],
    )
    weekly_labels = build_weekly_labels(stitched_daily)
    write_dataframe(weekly_labels, root / market_config["market"]["processed_output"])

    raw_articles = ingest_rss_feeds(str(root / "configs/feeds.yaml"))
    write_dataframe(raw_articles, root / "data/intermediate/articles_raw.parquet")

    trading_calendar = stitched_daily[["trade_date"]].drop_duplicates().rename(columns={"trade_date": "trade_date"})
    clean = clean_articles(raw_articles, trading_calendar=trading_calendar, energy_keywords=energy_keywords)
    write_dataframe(clean, root / "data/intermediate/articles_clean.parquet")

    scored_rows: list[dict[str, object]] = []
    qwen_cfg = model_config["qwen"]
    for _, article in clean.iterrows():
        scored = score_article_with_qwen(
            article,
            model=qwen_cfg["model"],
            base_url=qwen_cfg["base_url"],
            api_key_env=qwen_cfg["api_key_env"],
            timeout_seconds=qwen_cfg.get("timeout_seconds", 60),
        )
        scored_rows.append({**article.to_dict(), **scored})
    scored_articles = pd.DataFrame(scored_rows)
    write_dataframe(scored_articles, root / model_config["artifacts"]["scored_articles"])

    weekly_features = aggregate_weekly_features(scored_articles)
    model_table = weekly_features.merge(weekly_labels, on="week_end_date", how="inner").sort_values("week_end_date")
    feature_columns = [
        "article_count",
        "relevance_mean",
        "polarity_mean",
        "intensity_mean",
        "uncertainty_mean",
        "forwardness_mean",
        "polarity_std",
        "uncertainty_std",
        "polarity_momentum",
        "uncertainty_momentum",
        "forwardness_momentum",
    ]
    model_table = model_table.dropna(subset=feature_columns + ["next_week_label", "next_week_return"])
    write_dataframe(model_table, root / model_config["artifacts"]["weekly_model_table"])

    predictions, metrics, _, final_model = train_lightgbm_cv(
        model_table,
        feature_columns=feature_columns,
        config=TrainingConfig(**model_config["training"]),
    )
    write_dataframe(predictions, root / model_config["artifacts"]["predictions"])
    write_dataframe(metrics, root / "reports/cv_metrics.parquet")

    strategy = build_weekly_strategy(predictions)
    write_dataframe(strategy, root / model_config["artifacts"]["strategy"])

    shap_summary = export_shap_outputs(
        final_model,
        model_table,
        feature_columns=feature_columns,
        output_dir=root / "reports",
    )
    summary_report = build_summary_report(
        metrics=metrics,
        strategy=strategy,
        shap_summary=shap_summary,
        output_path=root / model_config["artifacts"]["summary_report"],
        market_source=market_settings.source,
        calendar_source=market_config["calendar"]["source"],
        qwen_model=qwen_cfg["model"],
        prompt_version=PROMPT_VERSION,
        feeds=[feed.name for feed in feeds],
    )

    return {
        "weekly_labels": root / market_config["market"]["processed_output"],
        "articles_raw": root / "data/intermediate/articles_raw.parquet",
        "articles_clean": root / "data/intermediate/articles_clean.parquet",
        "articles_scored": root / model_config["artifacts"]["scored_articles"],
        "weekly_model_table": root / model_config["artifacts"]["weekly_model_table"],
        "predictions": root / model_config["artifacts"]["predictions"],
        "strategy": root / model_config["artifacts"]["strategy"],
        "summary_report": summary_report,
    }
