from __future__ import annotations

import numpy as np
import pandas as pd


WEIGHTED_COLUMNS = ["polarity", "intensity", "uncertainty", "forwardness"]


def _weighted_mean(frame: pd.DataFrame, value_column: str) -> float:
    valid = frame[["relevance", value_column]].dropna()
    if valid.empty:
        return np.nan
    weight_sum = float(valid["relevance"].sum())
    if weight_sum == 0:
        return np.nan
    return float((valid["relevance"] * valid[value_column]).sum() / weight_sum)


def _population_std(series: pd.Series) -> float:
    valid = series.dropna()
    if valid.empty:
        return np.nan
    return float(valid.std(ddof=0))


def _build_article_count_frame(full_articles: pd.DataFrame) -> pd.DataFrame:
    counts = full_articles.copy()
    counts["week_end_date"] = pd.to_datetime(counts["week_end_date"]).dt.normalize()
    return (
        counts.groupby("week_end_date", sort=True)
        .size()
        .rename("article_count")
        .reset_index()
        .sort_values("week_end_date")
        .reset_index(drop=True)
    )


def aggregate_weekly_features(
    scored_articles: pd.DataFrame,
    *,
    full_articles: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate article-level Qwen scores into the paper's 11 weekly features."""
    articles = scored_articles.copy()
    articles["week_end_date"] = pd.to_datetime(articles["week_end_date"]).dt.normalize()
    articles = articles.sort_values("week_end_date").reset_index(drop=True)

    rows: list[dict[str, float | pd.Timestamp]] = []
    for week_end_date, frame in articles.groupby("week_end_date", sort=True):
        row: dict[str, float | pd.Timestamp] = {
            "week_end_date": week_end_date,
            "article_count": int(len(frame)),
            "relevance_mean": float(frame["relevance"].mean()) if len(frame) else np.nan,
            "polarity_std": _population_std(frame["polarity"]),
            "uncertainty_std": _population_std(frame["uncertainty"]),
        }
        for column in WEIGHTED_COLUMNS:
            row[f"{column}_mean"] = _weighted_mean(frame, column)
        rows.append(row)

    weekly = pd.DataFrame(rows).sort_values("week_end_date").reset_index(drop=True)
    if full_articles is not None and not full_articles.empty:
        count_frame = _build_article_count_frame(full_articles)
        weekly = count_frame.merge(weekly, on="week_end_date", how="left", suffixes=("", "_scored"))
        weekly["article_count"] = weekly["article_count"].astype(int)
    weekly["polarity_momentum"] = weekly["polarity_mean"].diff()
    weekly["uncertainty_momentum"] = weekly["uncertainty_mean"].diff()
    weekly["forwardness_momentum"] = weekly["forwardness_mean"].diff()
    return weekly
