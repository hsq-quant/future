from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.finnewshunter_mongo import (
    FINNEWSHUNTER_COLLECTIONS,
    build_collection_specs,
    normalize_mongo_documents,
)


def test_build_collection_specs_includes_enabled_core_sources() -> None:
    specs = build_collection_specs()

    names = [spec["name"] for spec in specs]
    assert "CNStock" in names
    assert "JRJ" in names
    assert "NBD" in names
    assert "Sina Finance" in names
    assert any(spec["collection"] == FINNEWSHUNTER_COLLECTIONS["CNStock"] for spec in specs)


def test_normalize_mongo_documents_preserves_finnewshunter_core_fields() -> None:
    docs = [
        {
            "_id": "abc123",
            "Date": "2020-03-15 10:00:00+08:00",
            "Url": "https://www.cnstock.com/example",
            "Title": "油价震荡回升",
            "Article": "OPEC减产与库存变化影响原油价格。",
            "Category": "中国证券网",
            "RelatedStockCodes": ["sc"],
        }
    ]

    normalized = normalize_mongo_documents(docs, source_name="CNStock")

    assert normalized["Date"].tolist() == ["2020-03-15 10:00:00+08:00"]
    assert normalized["Url"].tolist() == ["https://www.cnstock.com/example"]
    assert normalized["Title"].tolist() == ["油价震荡回升"]
    assert normalized["Article"].tolist() == ["OPEC减产与库存变化影响原油价格。"]
    assert normalized["Category"].tolist() == ["中国证券网"]
    assert normalized["source_name"].tolist() == ["CNStock"]


def test_normalize_mongo_documents_flattens_object_ids_and_lists() -> None:
    docs = [
        {
            "_id": {"$oid": "65f3"},
            "Date": "2020-03-16 08:00:00+08:00",
            "Url": "https://jrj.com.cn/example",
            "Title": "油价预期分化",
            "Article": "未来需求修复存在不确定性。",
            "Category": "金融界",
            "RelatedStockCodes": ["sc", "ru"],
        }
    ]

    normalized = normalize_mongo_documents(docs, source_name="JRJ")

    assert normalized["_id"].tolist() == ["65f3"]
    assert normalized["RelatedStockCodes"].tolist() == ["sc,ru"]
