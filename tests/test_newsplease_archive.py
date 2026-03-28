from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.newsplease_archive import (
    article_matches_filters,
    articles_to_jsonl_records,
    ensure_nltk_resources,
)


def test_article_matches_filters_requires_keywords() -> None:
    article = {
        "title": "海湾能源设施修复提速",
        "maintext": "原油供应恢复仍存在不确定性。",
        "language": "zh",
    }

    assert article_matches_filters(article, keywords=["原油", "OPEC"], language_prefix="zh")
    assert not article_matches_filters(article, keywords=["铜"], language_prefix="zh")


def test_article_matches_filters_respects_language_prefix() -> None:
    article = {
        "title": "Oil prices rise",
        "maintext": "OPEC discusses supply risks.",
        "language": "en",
    }

    assert not article_matches_filters(article, keywords=["Oil", "OPEC"], language_prefix="zh")


def test_articles_to_jsonl_records_normalizes_newsplease_fields() -> None:
    records = articles_to_jsonl_records(
        [
            {
                "title": "原油库存下降",
                "maintext": "原油库存下降推动油价上涨。",
                "description": "摘要",
                "date_publish": "2026-03-10T08:00:00+08:00",
                "url": "https://example.com/a1",
                "source_domain": "example.com",
                "language": "zh",
            }
        ]
    )

    assert records[0]["title"] == "原油库存下降"
    assert records[0]["maintext"] == "原油库存下降推动油价上涨。"
    assert records[0]["source_domain"] == "example.com"
    assert records[0]["language"] == "zh"


def test_ensure_nltk_resources_downloads_missing_packages(monkeypatch: object, tmp_path: Path) -> None:
    seen_downloads: list[str] = []

    def fake_find(resource: str) -> str:
        if resource.endswith("punkt"):
            return "present"
        raise LookupError(resource)

    def fake_download(resource: str, download_dir: str, quiet: bool) -> bool:
        seen_downloads.append(f"{resource}@{download_dir}@{quiet}")
        return True

    monkeypatch.setattr("src.data.newsplease_archive.nltk.data.find", fake_find)
    monkeypatch.setattr("src.data.newsplease_archive.nltk.download", fake_download)

    ensure_nltk_resources(download_dir=tmp_path)

    assert seen_downloads == [f"punkt_tab@{tmp_path}@True"]
