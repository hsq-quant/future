from __future__ import annotations

import pandas as pd

from src.data.nbd_search import combine_nbd_discoveries


def test_combine_nbd_discoveries_deduplicates_urls_and_sorts() -> None:
    first = pd.DataFrame(
        [
            {"url": "https://a", "published_at": "2024-01-02", "title": "a"},
            {"url": "https://b", "published_at": "2024-01-03", "title": "b"},
        ]
    )
    second = pd.DataFrame(
        [
            {"url": "https://a", "published_at": "2024-01-02", "title": "a-dup"},
            {"url": "https://c", "published_at": "2024-01-01", "title": "c"},
        ]
    )

    combined = combine_nbd_discoveries([first, second])

    assert combined["url"].tolist() == ["https://c", "https://a", "https://b"]
