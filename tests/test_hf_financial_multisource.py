from __future__ import annotations

import pandas as pd

from src.data.hf_financial_multisource import (
    normalize_hf_financial_multisource,
    normalize_hf_financial_news_2024,
)


def test_normalize_hf_financial_multisource_maps_date_text_and_extra_fields() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2020-03-10T08:30:00Z", "2019-01-01T08:00:00Z"],
            "text": [
                "Crude oil prices rose after OPEC signaled deeper cuts.",
                "Old article that should be filtered out.",
            ],
            "headline": ["Crude oil rises on OPEC signal", "Old headline"],
            "language": ["en", "en"],
            "publisher": ["Reuters", "Reuters"],
            "url": ["https://example.com/reuters-1", "https://example.com/reuters-old"],
        }
    )

    normalized = normalize_hf_financial_multisource(
        raw,
        start_date="2019-12-30",
        end_date="2026-03-13",
    )

    assert len(normalized) == 1
    assert normalized.loc[0, "title"] == "Crude oil rises on OPEC signal"
    assert normalized.loc[0, "body"] == "Crude oil prices rose after OPEC signaled deeper cuts."
    assert normalized.loc[0, "summary"] == "Crude oil prices rose after OPEC signaled deeper cuts."
    assert normalized.loc[0, "source"] == "Reuters"
    assert normalized.loc[0, "url"] == "https://example.com/reuters-1"
    assert normalized.loc[0, "language"] == "en"


def test_normalize_hf_financial_multisource_reads_nested_extra_fields() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2022-03-08T10:05:00Z"],
            "text": ["Oil markets remained volatile after fresh sanctions."],
            "extra_fields": [{"title": "Oil markets stay volatile", "url": "https://example.com/oil-1", "publisher": "Bloomberg"}],
        }
    )

    normalized = normalize_hf_financial_multisource(
        raw,
        start_date="2019-12-30",
        end_date="2026-03-13",
    )

    assert len(normalized) == 1
    assert normalized.loc[0, "title"] == "Oil markets stay volatile"
    assert normalized.loc[0, "source"] == "Bloomberg"
    assert normalized.loc[0, "url"] == "https://example.com/oil-1"


def test_normalize_hf_financial_news_2024_maps_date_title_and_content() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2024-05-10", "2026-04-01"],
            "title": ["Brent crude rises after OPEC meeting", "Out-of-window headline"],
            "content": [
                "Oil prices gained as OPEC kept supply cuts in place.",
                "This row should be filtered out by date.",
            ],
        }
    )

    normalized = normalize_hf_financial_news_2024(
        raw,
        start_date="2019-12-30",
        end_date="2026-03-13",
    )

    assert len(normalized) == 1
    assert normalized.loc[0, "title"] == "Brent crude rises after OPEC meeting"
    assert normalized.loc[0, "body"] == "Oil prices gained as OPEC kept supply cuts in place."
    assert normalized.loc[0, "summary"] == "Oil prices gained as OPEC kept supply cuts in place."
    assert normalized.loc[0, "source"] == "financial-news-2024"
    assert normalized.loc[0, "language"] == "en"
    assert normalized.loc[0, "url"] == ""
    assert str(normalized.loc[0, "published_at"]) == "2024-05-10 00:00:00+00:00"


def test_normalize_hf_financial_news_2024_deduplicates_date_title_content_rows() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2024-05-10", "2024-05-10"],
            "title": ["WTI rises on inventory draw", "WTI rises on inventory draw"],
            "content": [
                "Crude inventories fell sharply this week.",
                "Crude inventories fell sharply this week.",
            ],
        }
    )

    normalized = normalize_hf_financial_news_2024(
        raw,
        start_date="2019-12-30",
        end_date="2026-03-13",
    )

    assert len(normalized) == 1
