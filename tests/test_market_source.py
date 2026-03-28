from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import pandas as pd

from src.data.market_source import MarketFetchConfig, fetch_akshare_daily_ine, load_market_data


def test_fetch_akshare_daily_ine_skips_empty_prelaunch_months(monkeypatch: object, tmp_path: Path) -> None:
    def fake_get_futures_daily(*, start_date: str, end_date: str, market: str) -> pd.DataFrame:
        if start_date.startswith("201801") or start_date.startswith("201802"):
            raise IndexError("single positional indexer is out-of-bounds")
        return pd.DataFrame(
            {
                "日期": ["2018-03-26", "2018-03-27"],
                "合约": ["SC1809", "SC1809"],
                "收盘价": [430.0, 432.0],
                "品种": ["SC", "SC"],
            }
        )

    monkeypatch.setitem(sys.modules, "akshare", SimpleNamespace(get_futures_daily=fake_get_futures_daily))

    market = fetch_akshare_daily_ine(
        start_date="2018-01-01",
        end_date="2018-03-31",
        metadata_path=tmp_path / "missing_metadata.csv",
    )

    assert len(market) == 2
    assert market["trade_date"].min() == pd.Timestamp("2018-03-26")
    assert market["contract"].tolist() == ["SC1809", "SC1809"]


def test_load_market_data_falls_back_to_absolute_local_glob(monkeypatch: object, tmp_path: Path) -> None:
    local_dir = tmp_path / "raw"
    local_dir.mkdir(parents=True)
    csv_path = local_dir / "sc_sample.csv"
    pd.DataFrame(
        {
            "trade_date": ["2018-03-26", "2018-03-27"],
            "contract": ["SC1809", "SC1809"],
            "close": [430.0, 432.0],
        }
    ).to_csv(csv_path, index=False)

    def fake_fetch(*args: object, **kwargs: object) -> pd.DataFrame:
        raise RuntimeError("AKShare unavailable")

    monkeypatch.setattr("src.data.market_source.fetch_akshare_daily_ine", fake_fetch)

    config = MarketFetchConfig(
        source="akshare",
        start_date="2018-03-26",
        end_date="2018-03-31",
        exchange="INE",
        local_input_glob=str(local_dir / "*.csv"),
        contract_metadata_file=str(tmp_path / "missing_metadata.csv"),
    )

    market = load_market_data(config)

    assert len(market) == 2
    assert market["trade_date"].min() == pd.Timestamp("2018-03-26")
