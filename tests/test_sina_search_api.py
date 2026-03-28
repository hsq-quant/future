from __future__ import annotations

import pandas as pd

from src.data.sina_search_api import (
    build_sina_advanced_search_params,
    iter_sina_date_windows,
    normalize_sina_search_response,
)


def test_normalize_sina_search_response_maps_rows_and_filters_date_range() -> None:
    payload = {
        "code": 0,
        "message": "success",
        "data": {
            "list": [
                {
                    "title": "日本拟动用外汇储备做空<font color='red'>原油</font>救日元",
                    "intro": "全球分析师质疑效果。",
                    "searchSummary": "全球分析师质疑效果。",
                    "ctime": 1774580817,
                    "media_show": "环球网",
                    "url": "https://finance.sina.com.cn/jjxw/2026-03-27/doc-inhskwec8449860.shtml",
                    "docType": "news",
                },
                {
                    "title": "过早样本",
                    "intro": "应被过滤",
                    "searchSummary": "应被过滤",
                    "ctime": 1546300800,
                    "media_show": "旧闻",
                    "url": "https://finance.sina.com.cn/old/example.shtml",
                    "docType": "news",
                },
            ],
            "total": 2,
        },
    }

    normalized = normalize_sina_search_response(
        payload,
        keyword="原油",
        start_date="2019-12-30",
        end_date="2026-03-27",
    )

    assert len(normalized) == 1
    assert normalized.loc[0, "source_name"] == "Sina Finance"
    assert normalized.loc[0, "title"] == "日本拟动用外汇储备做空原油救日元"
    assert normalized.loc[0, "body"] == "全球分析师质疑效果。"
    assert normalized.loc[0, "summary"] == "全球分析师质疑效果。"
    assert normalized.loc[0, "source"] == "环球网"
    assert normalized.loc[0, "keyword"] == "原油"
    assert normalized.loc[0, "doc_type"] == "news"
    assert pd.Timestamp(normalized.loc[0, "published_at"]).date().isoformat() == "2026-03-27"


def test_build_sina_advanced_search_params_uses_unix_bounds() -> None:
    params = build_sina_advanced_search_params(
        keyword="原油",
        page=2,
        size=20,
        sort=1,
        start_date="2020-01-01",
        end_date="2020-03-31",
    )

    assert params["q"] == "原油"
    assert params["tp"] == "news"
    assert params["page"] == 2
    assert params["size"] == 20
    assert params["sort"] == 1
    assert params["from"] == "advanced_search"
    assert params["stime"] == "1577836800"
    assert params["etime"] == "1585699199"


def test_iter_sina_date_windows_splits_full_range_into_bounded_chunks() -> None:
    windows = list(iter_sina_date_windows(start_date="2020-01-01", end_date="2020-07-15", window_days=90))

    assert windows == [
        ("2020-01-01", "2020-03-30"),
        ("2020-03-31", "2020-06-28"),
        ("2020-06-29", "2020-07-15"),
    ]
