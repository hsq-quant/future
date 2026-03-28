from __future__ import annotations

import pandas as pd

from src.data.article_sampling import sample_articles_by_month


def test_sample_articles_by_month_uses_monthly_stratified_ratio() -> None:
    articles = pd.DataFrame(
        {
            "article_id": [f"a{i}" for i in range(10)],
            "published_at": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-06",
                    "2024-01-07",
                    "2024-02-01",
                    "2024-02-02",
                    "2024-02-03",
                    "2024-02-04",
                ]
            ),
        }
    )

    sampled = sample_articles_by_month(articles, sample_frac=0.5, random_state=7)

    month_counts = sampled["published_at"].dt.to_period("M").astype(str).value_counts().to_dict()
    assert month_counts == {"2024-01": 3, "2024-02": 2}


def test_sample_articles_by_month_is_deterministic_and_keeps_small_months() -> None:
    articles = pd.DataFrame(
        {
            "article_id": ["jan1", "jan2", "feb1"],
            "published_at": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-02-01"]),
        }
    )

    first = sample_articles_by_month(articles, sample_frac=0.5, random_state=42)
    second = sample_articles_by_month(articles, sample_frac=0.5, random_state=42)

    assert first["article_id"].tolist() == second["article_id"].tolist()
    assert "feb1" in first["article_id"].tolist()


def test_sample_articles_by_month_can_apply_global_cap_after_monthly_sampling() -> None:
    articles = pd.DataFrame(
        {
            "article_id": [f"a{i}" for i in range(12)],
            "published_at": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-06",
                    "2024-01-07",
                    "2024-02-01",
                    "2024-02-02",
                    "2024-02-03",
                    "2024-02-04",
                    "2024-02-05",
                    "2024-02-06",
                ]
            ),
        }
    )

    sampled = sample_articles_by_month(articles, sample_frac=0.5, random_state=11, max_articles=4)

    assert len(sampled) == 4
    sampled_months = set(sampled["published_at"].dt.to_period("M").astype(str))
    assert sampled_months == {"2024-01", "2024-02"}


def test_sample_articles_by_month_can_backfill_missing_weeks() -> None:
    articles = pd.DataFrame(
        {
            "article_id": [f"a{i}" for i in range(6)],
            "published_at": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-09",
                    "2024-01-10",
                    "2024-01-16",
                    "2024-01-17",
                ]
            ),
            "week_end_date": pd.to_datetime(
                [
                    "2024-01-05",
                    "2024-01-05",
                    "2024-01-12",
                    "2024-01-12",
                    "2024-01-19",
                    "2024-01-19",
                ]
            ),
        }
    )

    sampled = sample_articles_by_month(
        articles,
        sample_frac=0.34,
        random_state=3,
        ensure_weekly_coverage=True,
    )

    assert set(sampled["week_end_date"].dt.strftime("%Y-%m-%d").tolist()) == {
        "2024-01-05",
        "2024-01-12",
        "2024-01-19",
    }
