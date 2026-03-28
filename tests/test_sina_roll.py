from __future__ import annotations

import pandas as pd

from src.data.sina_roll import normalize_sina_roll_page, rebalance_source_discoveries


def test_normalize_sina_roll_page_extracts_news_links_and_filters_date_range() -> None:
    html = """
    <html>
      <body>
        <div class="listBlk">
          <ul>
            <li>
              <a href="https://finance.sina.com.cn/money/future/fmnews/2020-03-09/doc-iimxxstf7563988.shtml">国际油价暴跌拖累原油期货</a>
              <span>(03月09日 09:41)</span>
            </li>
            <li>
              <a href="https://finance.sina.com.cn/stock/s/2026-03-26/doc-inhsicuu6075064.shtml">A股新闻，不相关</a>
              <span>(03月26日 16:33)</span>
            </li>
          </ul>
          <span class="pagebox_next"><a href="?page=2">下一页</a></span>
        </div>
      </body>
    </html>
    """

    normalized = normalize_sina_roll_page(
        html,
        page_url="https://finance.sina.com.cn/roll/c/56592.shtml",
        page_number=1,
        start_date="2019-12-30",
        end_date="2026-03-13",
        keywords=["原油", "油价", "期货"],
    )

    assert len(normalized) == 1
    assert normalized.loc[0, "source_name"] == "Sina Finance"
    assert normalized.loc[0, "url"].startswith("https://finance.sina.com.cn/money/future/fmnews/2020-03-09/")
    assert normalized.loc[0, "keyword"] == "原油,油价,期货"
    assert pd.Timestamp(normalized.loc[0, "published_at"]).date().isoformat() == "2020-03-09"
    assert normalized.loc[0, "page_number"] == 1
    assert normalized.loc[0, "title"] == "国际油价暴跌拖累原油期货"


def test_rebalance_source_discoveries_enforces_source_targets_and_scales_nbd() -> None:
    frame = pd.DataFrame(
        [
            {"source_name": "NBD", "url": f"https://nbd/{idx}", "published_at": f"2024-01-{idx:02d}"}
            for idx in range(1, 7)
        ]
        + [
            {"source_name": "Sina Finance", "url": f"https://sina/{idx}", "published_at": f"2024-02-{idx:02d}"}
            for idx in range(1, 5)
        ]
        + [
            {"source_name": "JRJ", "url": f"https://jrj/{idx}", "published_at": f"2024-03-{idx:02d}"}
            for idx in range(1, 4)
        ]
    )

    balanced = rebalance_source_discoveries(
        frame,
        per_source_targets={"Sina Finance": 3, "JRJ": 2},
        nbd_keep_ratio=0.5,
    )

    counts = balanced["source_name"].value_counts().to_dict()
    assert counts["Sina Finance"] == 3
    assert counts["JRJ"] == 2
    assert counts["NBD"] == 3
    assert balanced["url"].is_unique
