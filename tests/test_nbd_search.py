from __future__ import annotations

import pandas as pd

from src.data.nbd_search import normalize_nbd_search_response


def test_normalize_nbd_search_response_filters_date_range_and_deduplicates() -> None:
    payload = {
        "code": 200,
        "msg": "Request success",
        "data": {
            "searchResults": [
                {
                    "title": "上海原油连续主力合约日内涨7%",
                    "url": "http://www.nbd.com.cn/articles/2026-03-13/4290697.html",
                    "publishTime": "2026-03-13",
                },
                {
                    "title": "上海原油连续主力合约日内涨7%",
                    "url": "http://www.nbd.com.cn/articles/2026-03-13/4290697.html",
                    "publishTime": "2026-03-13",
                },
                {
                    "title": "过早样本",
                    "url": "http://www.nbd.com.cn/articles/2019-01-24/1294493.html",
                    "publishTime": "2019-01-24",
                },
            ]
        },
    }

    normalized = normalize_nbd_search_response(
        payload,
        keyword="上海原油",
        start_date="2019-12-30",
        end_date="2026-03-13",
    )

    assert len(normalized) == 1
    assert normalized.loc[0, "source_name"] == "NBD"
    assert normalized.loc[0, "keyword"] == "上海原油"
    assert normalized.loc[0, "url"] == "https://www.nbd.com.cn/articles/2026-03-13/4290697.html"
    assert normalized.loc[0, "source"] == "NBD"
    assert normalized.loc[0, "language"] == "zh"
    assert normalized.loc[0, "body"] == ""
    assert pd.Timestamp(normalized.loc[0, "published_at"]).date().isoformat() == "2026-03-13"
