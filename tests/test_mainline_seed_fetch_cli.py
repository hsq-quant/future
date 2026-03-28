from __future__ import annotations

from pathlib import Path

from src.data.mainline_seed_fetch import read_seed_urls


def test_read_seed_urls_preserves_extra_columns(tmp_path: Path) -> None:
    seed_file = tmp_path / "seeds.csv"
    seed_file.write_text(
        "source_name,url,title,published_at,keyword\n"
        "NBD,https://www.nbd.com.cn/articles/2026-03-13/4290790.html,原油大涨11.26%,2026-03-13,上海原油\n",
        encoding="utf-8",
    )

    frame = read_seed_urls(seed_file)

    assert list(frame.columns) == ["source_name", "url", "title", "published_at", "keyword"]
    assert frame.loc[0, "source_name"] == "NBD"
