from __future__ import annotations

import pandas as pd

from src.data.english_coverage_audit import summarize_english_coverage
from src.data.english_energy_candidates import (
    build_english_energy_candidates,
    resolve_english_energy_phrases,
)


def test_build_english_energy_candidates_filters_by_title_or_body_phrases() -> None:
    raw = pd.DataFrame(
        {
            "title": [
                "Crude oil prices rise after OPEC meeting",
                "Tech earnings beat expectations",
                "Refinery outage pushes gasoline prices higher",
            ],
            "body": [
                "WTI crude moved higher.",
                "No energy content here.",
                "Diesel and refinery margins tightened.",
            ],
            "published_at": ["2020-03-10T00:00:00Z", "2020-03-11T00:00:00Z", "2020-03-12T00:00:00Z"],
            "url": ["https://a", "https://b", "https://c"],
        }
    )

    candidates = build_english_energy_candidates(
        raw,
        phrases=[
            "crude oil",
            "opec",
            "wti crude",
            "refinery",
            "gasoline",
            "diesel",
        ],
    )

    assert candidates["url"].tolist() == ["https://a", "https://c"]


def test_build_english_energy_candidates_deduplicates_on_title_time_url() -> None:
    raw = pd.DataFrame(
        {
            "title": ["Crude oil prices rise", "Crude oil prices rise"],
            "body": ["WTI higher", "WTI higher"],
            "published_at": ["2020-03-10T00:00:00Z", "2020-03-10T00:00:00Z"],
            "url": ["https://dup", "https://dup"],
        }
    )

    candidates = build_english_energy_candidates(raw, phrases=["crude oil", "wti"])

    assert len(candidates) == 1


def test_build_english_energy_candidates_supports_broader_energy_topic_buckets() -> None:
    raw = pd.DataFrame(
        {
            "title": [
                "SPR release discussed as oil inventories rise",
                "Container shipping stocks rally",
                "OPEC+ keeps supply cuts while refinery margins improve",
            ],
            "body": [
                "Petroleum reserve officials reviewed crude stockpile pressure.",
                "Freight rates moved higher for retail goods.",
                "Refineries and diesel cracks remain firm.",
            ],
            "published_at": ["2020-03-10T00:00:00Z", "2020-03-11T00:00:00Z", "2020-03-12T00:00:00Z"],
            "url": ["https://spr", "https://ship", "https://opec"],
        }
    )

    candidates = build_english_energy_candidates(
        raw,
        phrases=[
            "strategic petroleum reserve",
            "petroleum reserve",
            "inventory",
            "inventories",
            "opec+",
            "refinery",
            "refineries",
            "diesel",
        ],
    )

    assert candidates["url"].tolist() == ["https://spr", "https://opec"]


def test_summarize_english_coverage_reports_week_density_and_gaps() -> None:
    clean = pd.DataFrame(
        {
            "week_end_date": pd.to_datetime(["2020-01-10", "2020-01-10", "2020-01-31", "2020-02-07", "2020-02-07"]),
            "published_at": pd.to_datetime(
                [
                    "2020-01-06T00:00:00Z",
                    "2020-01-07T00:00:00Z",
                    "2020-01-27T00:00:00Z",
                    "2020-02-03T00:00:00Z",
                    "2020-02-04T00:00:00Z",
                ],
                utc=True,
            ),
        }
    )
    supervised_weeks = pd.DataFrame(
        {"week_end_date": pd.to_datetime(["2020-01-10", "2020-01-17", "2020-01-24", "2020-01-31", "2020-02-07"])}
    )

    summary = summarize_english_coverage(clean, supervised_weeks)

    assert summary["total_articles"] == 5
    assert summary["covered_weeks_ge_1"] == 3
    assert summary["covered_weeks_ge_3"] == 0
    assert summary["longest_zero_article_gap_weeks"] == 2


def test_resolve_english_energy_phrases_supports_broader_profiles() -> None:
    strict = resolve_english_energy_phrases("strict")
    broader = resolve_english_energy_phrases("broader")
    wide = resolve_english_energy_phrases("energy-wide")

    assert "crude oil" in strict
    assert "energy" not in strict
    assert "energy" in broader
    assert "commodity" in wide
