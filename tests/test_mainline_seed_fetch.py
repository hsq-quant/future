from __future__ import annotations

from src.data.mainline_seed_fetch import decode_response_text, normalize_extracted_article


def test_normalize_extracted_article_maps_trafilatura_json() -> None:
    raw = {
        "title": "WTI原油跌破25美元/桶",
        "text": "WTI原油跌破25美元/桶，日内跌幅扩大。",
        "date": "2020-03-18 20:10:37",
        "hostname": "www.nbd.com.cn",
    }

    normalized = normalize_extracted_article(
        raw,
        source_name="NBD",
        fallback_url="https://www.nbd.com.cn/articles/2020-03-18/1418030.html",
    )

    assert normalized["title"] == "WTI原油跌破25美元/桶"
    assert normalized["body"] == "WTI原油跌破25美元/桶，日内跌幅扩大。"
    assert normalized["published_at"] == "2020-03-18 20:10:37"
    assert normalized["url"] == "https://www.nbd.com.cn/articles/2020-03-18/1418030.html"
    assert normalized["source"] == "NBD"
    assert normalized["language"] == "zh"


class _FakeResponse:
    def __init__(self, content: bytes, apparent_encoding: str, encoding: str | None = None) -> None:
        self.content = content
        self.apparent_encoding = apparent_encoding
        self.encoding = encoding


def test_decode_response_text_prefers_apparent_encoding_for_chinese_pages() -> None:
    content = "国际油价震荡回升".encode("gb18030")
    response = _FakeResponse(content=content, apparent_encoding="gb18030", encoding="ISO-8859-1")

    decoded = decode_response_text(response)

    assert "国际油价" in decoded
