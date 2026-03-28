from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.rss_ingest import (
    allowed_language_prefixes_from_feeds,
    build_energy_keyword_pattern,
    clean_articles,
    ingest_rss_feeds,
    load_feed_config,
)


def test_ingest_rss_feeds_respects_max_entries_per_feed(monkeypatch: object, tmp_path: Path) -> None:
    feed_config = tmp_path / "feeds.yaml"
    feed_config.write_text(
        """
feeds:
  - name: "Feed A"
    source_type: "finance"
    language: "zh"
    enabled: true
    url: "https://example.com/a.xml"
  - name: "Feed B"
    source_type: "official"
    language: "zh"
    enabled: true
    url: "https://example.com/b.xml"

energy_keywords:
  - "原油"
""".strip(),
        encoding="utf-8",
    )

    class FakeEntry:
        def __init__(self, title: str) -> None:
            self.title = title
            self.summary = "原油新闻"
            self.link = f"https://example.com/{title}"
            self.published = "2026-03-01T08:00:00Z"
            self.tags = []

    class FakeFeed:
        def __init__(self, prefix: str) -> None:
            self.entries = [FakeEntry(f"{prefix}-{i}") for i in range(3)]

    def fake_parse(url: str) -> FakeFeed:
        return FakeFeed("a" if "a.xml" in url else "b")

    monkeypatch.setattr("src.data.rss_ingest.feedparser.parse", fake_parse)

    articles = ingest_rss_feeds(str(feed_config), max_entries_per_feed=1)

    assert len(articles) == 2
    assert articles["source"].tolist() == ["Feed A", "Feed B"]


def test_allowed_language_prefixes_from_feeds_preserves_declared_languages(tmp_path: Path) -> None:
    feed_config = tmp_path / "feeds.yaml"
    feed_config.write_text(
        """
feeds:
  - name: "Feed A"
    source_type: "finance"
    language: "zh"
    enabled: true
    url: "https://example.com/a.xml"
  - name: "Feed B"
    source_type: "energy"
    language: "en"
    enabled: true
    url: "https://example.com/b.xml"
  - name: "Feed C"
    source_type: "energy"
    language: "en-US"
    enabled: true
    url: "https://example.com/c.xml"

energy_keywords:
  - "原油"
  - "crude oil"
""".strip(),
        encoding="utf-8",
    )

    feeds, _ = load_feed_config(str(feed_config))

    assert allowed_language_prefixes_from_feeds(feeds) == ("zh", "en")


def test_clean_articles_filters_to_requested_date_window() -> None:
    raw_articles = pd.DataFrame(
        {
            "article_id": ["a1", "a2", "a3"],
            "title": ["原油供给", "原油库存", "原油需求"],
            "body": ["内容1", "内容2", "内容3"],
            "summary": ["摘要1", "摘要2", "摘要3"],
            "source": ["Feed", "Feed", "Feed"],
            "source_type": ["finance", "finance", "finance"],
            "published_at": pd.to_datetime(
                [
                    "2019-12-29 10:00:00+08:00",
                    "2019-12-31 10:00:00+08:00",
                    "2020-01-06 10:00:00+08:00",
                ]
            ),
            "published_at_tz": ["Asia/Shanghai"] * 3,
            "url": ["u1", "u2", "u3"],
            "language": ["zh", "zh", "zh"],
            "rss_feed": ["f", "f", "f"],
            "raw_tags": ["", "", ""],
            "ingested_at": ["now", "now", "now"],
        }
    )
    trading_calendar = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                [
                    "2020-01-03",
                    "2020-01-10",
                ]
            )
        }
    )

    cleaned = clean_articles(
        raw_articles,
        trading_calendar=trading_calendar,
        energy_keywords=["原油"],
        start_date=pd.Timestamp("2019-12-30"),
        end_date=pd.Timestamp("2020-01-05"),
    )

    assert cleaned["article_id"].tolist() == ["a2"]


def test_clean_articles_accepts_archive_rows_without_rss_specific_columns() -> None:
    raw_articles = pd.DataFrame(
        {
            "article_id": ["a1"],
            "title": ["原油供应偏紧"],
            "body": ["OPEC减产带动原油价格上涨"],
            "summary": ["OPEC减产带动原油价格上涨"],
            "source": ["archive-source"],
            "source_type": ["archive"],
            "published_at": pd.to_datetime(["2026-03-10 10:00:00+08:00"]),
            "published_at_tz": ["Asia/Shanghai"],
            "url": ["https://example.com/a1"],
            "language": ["zh"],
            "retrieval_method": ["archive"],
            "retrieval_query": ["原油 OR OPEC"],
            "provider_name": ["test-provider"],
            "content_hash": ["hash1"],
            "ingested_at": ["now"],
        }
    )
    trading_calendar = pd.DataFrame({"trade_date": pd.to_datetime(["2026-03-13"])})

    cleaned = clean_articles(
        raw_articles,
        trading_calendar=trading_calendar,
        energy_keywords=["原油", "OPEC"],
        start_date=pd.Timestamp("2026-03-01"),
        end_date=pd.Timestamp("2026-03-13"),
    )

    assert cleaned["article_id"].tolist() == ["a1"]
    assert cleaned["source"].tolist() == ["archive-source"]


def test_clean_articles_can_keep_english_rows_when_language_prefixes_allow_them() -> None:
    raw_articles = pd.DataFrame(
        {
            "article_id": ["en1"],
            "title": ["Crude oil inventories fall as OPEC signals tighter supply"],
            "body": ["Crude oil prices rose after OPEC signaled a tighter supply outlook."],
            "summary": ["Crude oil prices rose after OPEC signaled a tighter supply outlook."],
            "source": ["Global Finance"],
            "source_type": ["archive"],
            "published_at": pd.to_datetime(["2026-03-10 10:00:00+08:00"]),
            "published_at_tz": ["Asia/Shanghai"],
            "url": ["https://example.com/en1"],
            "language": ["en"],
            "retrieval_method": ["hf_dataset"],
            "retrieval_query": ["crude oil OR OPEC"],
            "provider_name": ["hf-financial-news"],
            "content_hash": ["hash-en1"],
            "ingested_at": ["now"],
        }
    )
    trading_calendar = pd.DataFrame({"trade_date": pd.to_datetime(["2026-03-13"])})

    cleaned = clean_articles(
        raw_articles,
        trading_calendar=trading_calendar,
        energy_keywords=["原油", "OPEC", "crude oil", "inventory"],
        start_date=pd.Timestamp("2026-03-01"),
        end_date=pd.Timestamp("2026-03-13"),
        allowed_language_prefixes=("zh", "en"),
    )

    assert cleaned["article_id"].tolist() == ["en1"]
    assert cleaned["language"].tolist() == ["en"]


def test_build_energy_keyword_pattern_escapes_latin_terms_with_boundaries() -> None:
    pattern = build_energy_keyword_pattern(["原油", "INE", "crude oil", "WTI"])

    assert pd.Series(["INE crude oil rises"]).str.contains(pattern, case=False, regex=True).iloc[0]
    assert not pd.Series(["online retail growth"]).str.contains(pattern, case=False, regex=True).iloc[0]


def test_clean_articles_can_skip_keyword_filter_for_prefiltered_archive_rows() -> None:
    raw_articles = pd.DataFrame(
        {
            "article_id": ["en_prefiltered"],
            "title": ["U.S. Stock Futures Retreat; Crude Oil Swings: Markets Wrap"],
            "body": ["Bloomberg markets wrap article."],
            "summary": ["Bloomberg markets wrap article."],
            "source": ["Bloomberg"],
            "source_type": ["archive"],
            "published_at": pd.to_datetime(["2020-04-13 06:40:28+08:00"]),
            "published_at_tz": ["Asia/Shanghai"],
            "url": ["https://example.com/en-prefiltered"],
            "language": ["en"],
            "retrieval_method": ["hf_financial_multisource_local"],
            "retrieval_query": ["title-prefiltered"],
            "provider_name": ["English Global Financial News"],
            "content_hash": ["hash-en-prefiltered"],
            "ingested_at": ["now"],
        }
    )
    trading_calendar = pd.DataFrame({"trade_date": pd.to_datetime(["2020-04-17"])})

    cleaned = clean_articles(
        raw_articles,
        trading_calendar=trading_calendar,
        energy_keywords=["原油"],
        start_date=pd.Timestamp("2020-04-01"),
        end_date=pd.Timestamp("2020-04-30"),
        allowed_language_prefixes=("en",),
        apply_keyword_filter=False,
    )

    assert cleaned["article_id"].tolist() == ["en_prefiltered"]


def test_clean_articles_does_not_collapse_distinct_rows_when_url_is_empty() -> None:
    raw_articles = pd.DataFrame(
        {
            "article_id": ["a1", "a2"],
            "title": ["Brent rises after OPEC talks", "WTI falls on inventory surprise"],
            "body": ["Oil market rallied after OPEC meeting.", "Crude inventories rose sharply this week."],
            "summary": ["Oil market rallied after OPEC meeting.", "Crude inventories rose sharply this week."],
            "source": ["financial-news-2024", "financial-news-2024"],
            "source_type": ["archive", "archive"],
            "published_at": pd.to_datetime(["2024-04-08 00:00:00+00:00", "2024-04-09 00:00:00+00:00"]),
            "published_at_tz": ["UTC", "UTC"],
            "url": ["", ""],
            "language": ["en", "en"],
            "ingested_at": ["now", "now"],
        }
    )
    trading_calendar = pd.DataFrame(
        {"trade_date": pd.to_datetime(["2024-04-08", "2024-04-09", "2024-04-10", "2024-04-11", "2024-04-12"])}
    )

    cleaned = clean_articles(
        raw_articles,
        trading_calendar=trading_calendar,
        energy_keywords=[],
        start_date="2024-04-01",
        end_date="2024-04-30",
        allowed_language_prefixes=("en",),
        apply_keyword_filter=False,
    )

    assert len(cleaned) == 2
