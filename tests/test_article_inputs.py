from __future__ import annotations

from pathlib import Path

from src.data.article_inputs import resolve_clean_articles_path


def test_resolve_clean_articles_path_prefers_archive_clean(tmp_path: Path) -> None:
    root = tmp_path
    archive_clean = root / "data/intermediate/articles_archive_clean.parquet"
    rss_clean = root / "data/intermediate/articles_clean.parquet"
    archive_clean.parent.mkdir(parents=True, exist_ok=True)
    archive_clean.write_text("stub", encoding="utf-8")
    rss_clean.write_text("stub", encoding="utf-8")

    resolved = resolve_clean_articles_path(root)

    assert resolved == archive_clean


def test_resolve_clean_articles_path_falls_back_to_rss_clean(tmp_path: Path) -> None:
    root = tmp_path
    rss_clean = root / "data/intermediate/articles_clean.parquet"
    rss_clean.parent.mkdir(parents=True, exist_ok=True)
    rss_clean.write_text("stub", encoding="utf-8")

    resolved = resolve_clean_articles_path(root)

    assert resolved == rss_clean


def test_resolve_clean_articles_path_honors_explicit_input(tmp_path: Path) -> None:
    root = tmp_path
    custom = root / "custom/articles.parquet"
    custom.parent.mkdir(parents=True, exist_ok=True)
    custom.write_text("stub", encoding="utf-8")

    resolved = resolve_clean_articles_path(root, preferred_input=custom)

    assert resolved == custom


def test_resolve_clean_articles_path_honors_config_default_before_archive_fallback(tmp_path: Path) -> None:
    root = tmp_path
    configured = root / "data/intermediate/custom_clean.parquet"
    archive_clean = root / "data/intermediate/articles_archive_clean.parquet"
    configured.parent.mkdir(parents=True, exist_ok=True)
    configured.write_text("stub", encoding="utf-8")
    archive_clean.write_text("stub", encoding="utf-8")

    resolved = resolve_clean_articles_path(root, default_input=configured)

    assert resolved == configured
