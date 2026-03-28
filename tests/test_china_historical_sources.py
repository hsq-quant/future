from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.china_sources import load_source_catalog, prioritized_domains
from src.data.commoncrawl_index import build_index_query
from src.data.finnewshunter_import import normalize_finnewshunter_export


def test_load_source_catalog_and_prioritized_domains(tmp_path: Path) -> None:
    config_path = tmp_path / "china_sources.yaml"
    config_path.write_text(
        """
sources:
  - name: "Sina Finance"
    enabled: true
    domains: ["finance.sina.com.cn"]
  - name: "Yicai"
    enabled: false
    domains: ["www.yicai.com", "yicai.com"]
  - name: "STCN"
    enabled: true
    domains: ["www.stcn.com", "stcn.com"]
""".strip(),
        encoding="utf-8",
    )

    catalog = load_source_catalog(config_path)

    assert [item["name"] for item in catalog] == ["Sina Finance", "Yicai", "STCN"]
    assert prioritized_domains(catalog) == [
        "finance.sina.com.cn",
        "www.stcn.com",
        "stcn.com",
    ]


def test_normalize_finnewshunter_export_maps_common_columns() -> None:
    raw = pd.DataFrame(
        {
            "headline": ["原油震荡上行"],
            "content": ["原油库存下降，市场预期走强。"],
            "publish_time": ["2020-03-10 08:30:00+08:00"],
            "article_url": ["https://finance.sina.com.cn/example"],
            "site": ["新浪财经"],
        }
    )

    normalized = normalize_finnewshunter_export(raw)

    assert normalized["title"].tolist() == ["原油震荡上行"]
    assert normalized["body"].tolist() == ["原油库存下降，市场预期走强。"]
    assert normalized["source"].tolist() == ["新浪财经"]
    assert normalized["language"].tolist() == ["zh"]


def test_normalize_finnewshunter_export_maps_legacy_finnewshunter_fields() -> None:
    raw = pd.DataFrame(
        {
            "Title": ["上海原油小幅回升"],
            "Article": ["OPEC会议释放减产信号，原油情绪偏强。"],
            "Date": ["2020-03-12 09:15:00+08:00"],
            "Url": ["https://www.cnstock.com/example"],
            "Category": ["中国证券网"],
        }
    )

    normalized = normalize_finnewshunter_export(raw)

    assert normalized["title"].tolist() == ["上海原油小幅回升"]
    assert normalized["body"].tolist() == ["OPEC会议释放减产信号，原油情绪偏强。"]
    assert normalized["published_at"].tolist() == ["2020-03-12 09:15:00+08:00"]
    assert normalized["url"].tolist() == ["https://www.cnstock.com/example"]
    assert normalized["source"].tolist() == ["中国证券网"]
    assert normalized["language"].tolist() == ["zh"]


def test_normalize_finnewshunter_export_maps_mongo_style_fields() -> None:
    raw = pd.DataFrame(
        {
            "Date": ["2020-03-15 10:00:00+08:00"],
            "Url": ["https://www.nbd.com.cn/example"],
            "Title": ["油价波动加剧"],
            "Article": ["库存与需求预期共同扰动原油价格。"],
            "RelatedStockCodes": [["sc"]],
            "Category": ["每经网"],
        }
    )

    normalized = normalize_finnewshunter_export(raw)

    assert normalized["title"].tolist() == ["油价波动加剧"]
    assert normalized["body"].tolist() == ["库存与需求预期共同扰动原油价格。"]
    assert normalized["published_at"].tolist() == ["2020-03-15 10:00:00+08:00"]
    assert normalized["source"].tolist() == ["每经网"]


def test_build_index_query_uses_domain_and_date_range() -> None:
    query = build_index_query(
        domain="finance.sina.com.cn",
        start_date="2020-01-01",
        end_date="2020-03-31",
        limit=250,
    )

    assert query["url"] == "finance.sina.com.cn/*"
    assert query["from"] == "20200101000000"
    assert query["to"] == "20200331235959"
    assert query["limit"] == "250"
    assert query["output"] == "json"
