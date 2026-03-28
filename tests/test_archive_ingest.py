from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.archive_ingest import normalize_archive_articles, run_archive_ingest


def test_normalize_archive_articles_builds_canonical_schema() -> None:
    raw = pd.DataFrame(
        {
            "headline": ["原油供应吃紧"],
            "content": ["OPEC减产带动原油价格上涨"],
            "published_ts": ["2026-03-10 09:30:00+08:00"],
            "article_url": ["https://example.com/a1"],
            "domain": ["example.com"],
            "lang": ["zh"],
        }
    )

    normalized = normalize_archive_articles(
        raw_articles=raw,
        provider_name="test-provider",
        retrieval_method="archive",
        start_date="2026-03-01",
        end_date="2026-03-13",
        column_map={
            "title": "headline",
            "body": "content",
            "published_at": "published_ts",
            "url": "article_url",
            "source": "domain",
            "language": "lang",
        },
        retrieval_query="原油 OR OPEC",
    )

    assert normalized["source"].tolist() == ["example.com"]
    assert normalized["language"].tolist() == ["zh"]
    assert normalized["retrieval_method"].tolist() == ["archive"]
    assert normalized["retrieval_query"].tolist() == ["原油 OR OPEC"]
    assert normalized["published_at"].dt.tz is not None
    assert "content_hash" in normalized.columns


def test_run_archive_ingest_falls_back_to_next_provider(tmp_path: Path) -> None:
    config_path = tmp_path / "archive.yaml"
    config_path.write_text(
        """
archive:
  start_date: "2026-03-01"
  end_date: "2026-03-13"
  providers:
    - name: "primary"
      type: "stub_empty"
      enabled: true
      stop_on_success: true
    - name: "fallback"
      type: "stub_data"
      enabled: true
      stop_on_success: true
""".strip(),
        encoding="utf-8",
    )

    def stub_empty(*_: object, **__: object) -> pd.DataFrame:
        return pd.DataFrame(columns=["title", "body", "published_at", "url", "source", "language"])

    def stub_data(*_: object, **__: object) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "title": ["原油库存下降"],
                "body": ["原油库存下降，油价走强"],
                "published_at": ["2026-03-12 08:00:00+08:00"],
                "url": ["https://example.com/fallback"],
                "source": ["fallback-source"],
                "language": ["zh"],
            }
        )

    articles, manifest = run_archive_ingest(
        config_path=str(config_path),
        provider_registry={
            "stub_empty": stub_empty,
            "stub_data": stub_data,
        },
    )

    assert articles["source"].tolist() == ["fallback-source"]
    assert manifest["provider_name"].tolist() == ["primary", "fallback"]
    assert manifest["normalized_count"].tolist() == [0, 1]
    assert manifest["status"].tolist() == ["empty", "success"]
