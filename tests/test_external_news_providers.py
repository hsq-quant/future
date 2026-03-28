from __future__ import annotations

import pandas as pd

from src.data.external_news_providers import (
    filter_news_by_keywords,
    iter_date_batches,
    normalize_akshare_news_frame,
    normalize_tushare_major_news_frame,
    normalize_tushare_news_frame,
)


def test_normalize_tushare_news_frame_maps_fields_and_filters_date_range() -> None:
    raw = pd.DataFrame(
        {
            "title": ["上海原油偏强运行", "过早样本"],
            "content": ["OPEC减产预期推升油价。", "应被过滤。"],
            "pub_time": ["2020-03-10 08:30:00", "2019-01-01 09:00:00"],
            "src": ["sina", "sina"],
        }
    )

    normalized = normalize_tushare_news_frame(
        raw,
        start_date="2019-12-30",
        end_date="2026-03-13",
    )

    assert len(normalized) == 1
    assert normalized.loc[0, "title"] == "上海原油偏强运行"
    assert normalized.loc[0, "body"] == "OPEC减产预期推升油价。"
    assert normalized.loc[0, "summary"] == "OPEC减产预期推升油价。"
    assert normalized.loc[0, "source"] == "sina"
    assert normalized.loc[0, "language"] == "zh"
    assert normalized.loc[0, "url"] == ""
    assert pd.Timestamp(normalized.loc[0, "published_at"]).date().isoformat() == "2020-03-10"


def test_normalize_tushare_major_news_frame_keeps_url_when_present() -> None:
    raw = pd.DataFrame(
        {
            "title": ["供应忧虑加剧，国际油价大涨"],
            "content": ["地缘冲突加剧供应风险，原油市场波动放大。"],
            "pub_time": ["2022-03-08 10:05:00"],
            "src": ["华尔街见闻"],
            "url": ["https://wallstreetcn.com/articles/example"],
        }
    )

    normalized = normalize_tushare_major_news_frame(
        raw,
        start_date="2019-12-30",
        end_date="2026-03-13",
    )

    assert len(normalized) == 1
    assert normalized.loc[0, "source"] == "华尔街见闻"
    assert normalized.loc[0, "url"] == "https://wallstreetcn.com/articles/example"
    assert normalized.loc[0, "summary"] == "地缘冲突加剧供应风险，原油市场波动放大。"


def test_normalize_akshare_news_frame_maps_chinese_columns() -> None:
    raw = pd.DataFrame(
        {
            "新闻标题": ["INE原油期货震荡走高"],
            "新闻内容": ["原油库存下降，市场风险偏好回暖。"],
            "发布时间": ["2024-05-07 09:12:00"],
            "文章来源": ["东方财富网"],
            "新闻链接": ["https://finance.eastmoney.com/a/example.html"],
        }
    )

    normalized = normalize_akshare_news_frame(
        raw,
        start_date="2019-12-30",
        end_date="2026-03-13",
    )

    assert len(normalized) == 1
    assert normalized.loc[0, "title"] == "INE原油期货震荡走高"
    assert normalized.loc[0, "body"] == "原油库存下降，市场风险偏好回暖。"
    assert normalized.loc[0, "summary"] == "原油库存下降，市场风险偏好回暖。"
    assert normalized.loc[0, "source"] == "东方财富网"
    assert normalized.loc[0, "url"] == "https://finance.eastmoney.com/a/example.html"
    assert normalized.loc[0, "language"] == "zh"


def test_iter_date_batches_splits_large_ranges_into_bounded_windows() -> None:
    batches = list(iter_date_batches(start_date="2020-01-01", end_date="2020-05-01", batch_days=60))

    assert batches == [
        ("2020-01-01", "2020-02-29"),
        ("2020-03-01", "2020-04-29"),
        ("2020-04-30", "2020-05-01"),
    ]


def test_filter_news_by_keywords_matches_title_or_body() -> None:
    frame = pd.DataFrame(
        {
            "title": ["原油库存下降", "宏观策略周报"],
            "body": ["油价反弹。", "不相关内容。"],
            "summary": ["油价反弹。", "不相关内容。"],
            "published_at": ["2020-03-10", "2020-03-11"],
            "url": ["https://a", "https://b"],
            "source": ["sina", "sina"],
            "language": ["zh", "zh"],
        }
    )

    filtered = filter_news_by_keywords(frame, ["原油", "油价", "INE"])

    assert filtered["title"].tolist() == ["原油库存下降"]
