from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.weekly_features import aggregate_weekly_features
from src.utils.io import read_dataframe, read_yaml, write_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build weekly model table from scored articles.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs/model.yaml"))
    return parser.parse_args()


def main() -> None:
    root = ROOT
    args = parse_args()
    config = read_yaml(args.config)
    scored = read_dataframe(root / config["artifacts"]["scored_articles"])
    full_clean_articles = None
    full_clean_path = config["artifacts"].get("full_clean_articles")
    if full_clean_path:
        full_clean_articles = read_dataframe(root / full_clean_path)
    weekly_labels = read_dataframe(root / "data/processed/weekly_labels.parquet")
    weekly_features = aggregate_weekly_features(scored, full_articles=full_clean_articles)
    model_table = weekly_features.merge(weekly_labels, on="week_end_date", how="inner").sort_values("week_end_date")
    output_path = root / config["artifacts"]["weekly_model_table"]
    write_dataframe(model_table, output_path)
    model_table.to_csv(output_path.with_suffix(".csv"), index=False)
    print(output_path)


if __name__ == "__main__":
    main()
