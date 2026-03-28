from __future__ import annotations

import pandas as pd

from src.data.calendar import assign_week_end_dates
from src.data.market_labels import build_weekly_labels, stitch_continuous_near_month


def test_assign_week_end_dates_uses_last_actual_trading_day_for_holiday_shortened_week() -> None:
    trade_dates = pd.to_datetime(
        [
            "2026-10-08",
            "2026-10-09",
        ]
    )
    calendar = pd.DataFrame({"trade_date": trade_dates})

    article_times = pd.to_datetime(
        [
            "2026-10-08 10:00:00",
            "2026-10-09 14:30:00",
        ]
    )

    assigned = assign_week_end_dates(article_times=article_times, trading_calendar=calendar)

    assert assigned.tolist() == [pd.Timestamp("2026-10-09"), pd.Timestamp("2026-10-09")]


def test_stitch_continuous_near_month_rolls_five_trading_days_before_last_trade_date() -> None:
    dates = pd.date_range("2026-05-04", periods=8, freq="B")
    contract_a = pd.DataFrame(
        {
            "trade_date": dates,
            "contract": "SC2606",
            "close": [500, 501, 502, 503, 504, 505, 506, 507],
            "last_trade_date": pd.Timestamp("2026-05-13"),
        }
    )
    contract_b = pd.DataFrame(
        {
            "trade_date": dates,
            "contract": "SC2607",
            "close": [510, 511, 512, 513, 514, 515, 516, 517],
            "last_trade_date": pd.Timestamp("2026-06-15"),
        }
    )

    stitched = stitch_continuous_near_month(pd.concat([contract_a, contract_b], ignore_index=True))

    assert stitched.loc[stitched["trade_date"] == pd.Timestamp("2026-05-06"), "active_contract"].item() == "SC2606"
    assert stitched.loc[stitched["trade_date"] == pd.Timestamp("2026-05-07"), "active_contract"].item() == "SC2607"


def test_stitch_continuous_near_month_back_adjusts_roll_gap() -> None:
    dates = pd.to_datetime(
        [
            "2026-05-04",
            "2026-05-05",
            "2026-05-06",
            "2026-05-07",
        ]
    )
    contract_a = pd.DataFrame(
        {
            "trade_date": dates,
            "contract": "SC2606",
            "close": [100.0, 101.0, 102.0, 103.0],
            "last_trade_date": pd.Timestamp("2026-05-08"),
        }
    )
    contract_b = pd.DataFrame(
        {
            "trade_date": dates,
            "contract": "SC2607",
            "close": [110.0, 111.0, 112.0, 113.0],
            "last_trade_date": pd.Timestamp("2026-06-15"),
        }
    )

    stitched = stitch_continuous_near_month(
        pd.concat([contract_a, contract_b], ignore_index=True),
        roll_days_before_last_trade=1,
    )

    adjusted = stitched.set_index("trade_date")["adjusted_close"]
    assert stitched.loc[stitched["trade_date"] == pd.Timestamp("2026-05-07"), "active_contract"].item() == "SC2607"
    assert adjusted.loc[pd.Timestamp("2026-05-06")] == 102.0
    assert adjusted.loc[pd.Timestamp("2026-05-07")] == 103.0


def test_build_weekly_labels_uses_next_week_return_for_target() -> None:
    continuous = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                [
                    "2026-01-02",
                    "2026-01-09",
                    "2026-01-16",
                    "2026-01-23",
                ]
            ),
            "close": [100.0, 110.0, 99.0, 120.0],
            "adjusted_close": [100.0, 110.0, 99.0, 120.0],
            "active_contract": ["SC2602", "SC2602", "SC2603", "SC2603"],
        }
    )

    weekly = build_weekly_labels(continuous)

    assert weekly["week_end_date"].tolist() == [pd.Timestamp("2026-01-09"), pd.Timestamp("2026-01-16")]
    assert weekly["next_week_label"].tolist() == [0, 1]


def test_build_weekly_labels_uses_adjusted_close_when_available() -> None:
    continuous = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                [
                    "2026-01-02",
                    "2026-01-09",
                    "2026-01-16",
                    "2026-01-23",
                ]
            ),
            "close": [100.0, 120.0, 110.0, 130.0],
            "adjusted_close": [100.0, 110.0, 99.0, 120.0],
            "active_contract": ["SC2602", "SC2602", "SC2603", "SC2603"],
        }
    )

    weekly = build_weekly_labels(continuous)

    assert weekly["weekly_close"].tolist() == [110.0, 99.0]
    assert weekly["next_week_label"].tolist() == [0, 1]
