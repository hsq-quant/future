from __future__ import annotations

import math

import pandas as pd

from src.features.weekly_features import aggregate_weekly_features


def test_aggregate_weekly_features_matches_paper_formulas() -> None:
    articles = pd.DataFrame(
        [
            {
                "week_end_date": pd.Timestamp("2026-01-09"),
                "relevance": 0.8,
                "polarity": 0.5,
                "intensity": 0.4,
                "uncertainty": 0.2,
                "forwardness": 0.6,
            },
            {
                "week_end_date": pd.Timestamp("2026-01-09"),
                "relevance": 0.2,
                "polarity": -0.5,
                "intensity": 0.8,
                "uncertainty": 0.4,
                "forwardness": 0.2,
            },
            {
                "week_end_date": pd.Timestamp("2026-01-16"),
                "relevance": 0.6,
                "polarity": 0.1,
                "intensity": 0.5,
                "uncertainty": 0.3,
                "forwardness": 0.7,
            },
            {
                "week_end_date": pd.Timestamp("2026-01-16"),
                "relevance": 0.05,
                "polarity": None,
                "intensity": None,
                "uncertainty": None,
                "forwardness": None,
            },
        ]
    )

    weekly = aggregate_weekly_features(articles)

    first = weekly.loc[weekly["week_end_date"] == pd.Timestamp("2026-01-09")].iloc[0]
    second = weekly.loc[weekly["week_end_date"] == pd.Timestamp("2026-01-16")].iloc[0]

    assert first["article_count"] == 2
    assert math.isclose(first["relevance_mean"], 0.5)
    assert math.isclose(first["polarity_mean"], 0.3)
    assert math.isclose(first["intensity_mean"], 0.48)
    assert math.isclose(first["uncertainty_mean"], 0.24)
    assert math.isclose(first["forwardness_mean"], 0.52)
    assert math.isclose(first["polarity_std"], 0.5)
    assert math.isclose(first["uncertainty_std"], 0.1)
    assert pd.isna(first["polarity_momentum"])

    assert second["article_count"] == 2
    assert math.isclose(second["relevance_mean"], 0.325)
    assert math.isclose(second["polarity_mean"], 0.1)
    assert math.isclose(second["polarity_momentum"], -0.2)
    assert math.isclose(second["uncertainty_momentum"], 0.06)
    assert math.isclose(second["forwardness_momentum"], 0.18)


def test_single_article_week_has_zero_dispersion() -> None:
    articles = pd.DataFrame(
        [
            {
                "week_end_date": pd.Timestamp("2026-01-09"),
                "relevance": 0.8,
                "polarity": 0.5,
                "intensity": 0.6,
                "uncertainty": 0.3,
                "forwardness": 0.7,
            }
        ]
    )

    weekly = aggregate_weekly_features(articles)
    row = weekly.iloc[0]

    assert row["polarity_std"] == 0.0
    assert row["uncertainty_std"] == 0.0


def test_aggregate_weekly_features_preserves_full_article_counts() -> None:
    scored_articles = pd.DataFrame(
        [
            {
                "week_end_date": pd.Timestamp("2026-01-09"),
                "relevance": 0.8,
                "polarity": 0.5,
                "intensity": 0.6,
                "uncertainty": 0.3,
                "forwardness": 0.7,
            }
        ]
    )
    full_articles = pd.DataFrame(
        [
            {"week_end_date": pd.Timestamp("2026-01-09"), "article_id": "a1"},
            {"week_end_date": pd.Timestamp("2026-01-09"), "article_id": "a2"},
            {"week_end_date": pd.Timestamp("2026-01-16"), "article_id": "a3"},
            {"week_end_date": pd.Timestamp("2026-01-16"), "article_id": "a4"},
            {"week_end_date": pd.Timestamp("2026-01-16"), "article_id": "a5"},
        ]
    )

    weekly = aggregate_weekly_features(scored_articles, full_articles=full_articles)

    assert weekly["week_end_date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-01-09", "2026-01-16"]
    first = weekly.iloc[0]
    second = weekly.iloc[1]

    assert first["article_count"] == 2
    assert math.isclose(first["polarity_mean"], 0.5)

    assert second["article_count"] == 3
    assert pd.isna(second["relevance_mean"])
    assert pd.isna(second["polarity_mean"])
    assert pd.isna(second["uncertainty_std"])
