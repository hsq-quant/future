from __future__ import annotations

from pathlib import Path

from scripts import build_labels, fetch_market_data


def test_fetch_market_data_parse_args_accepts_custom_config(tmp_path: Path) -> None:
    config_path = tmp_path / "market_custom.yaml"

    args = fetch_market_data.parse_args(["--config", str(config_path)])

    assert args.config == str(config_path)


def test_build_labels_parse_args_accepts_custom_config(tmp_path: Path) -> None:
    config_path = tmp_path / "market_custom.yaml"

    args = build_labels.parse_args(["--config", str(config_path)])

    assert args.config == str(config_path)
