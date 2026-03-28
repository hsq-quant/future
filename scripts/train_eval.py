from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.train_eval import (
    build_training_config,
    export_shap_outputs,
    prepare_regression_table,
    prepare_training_table,
    train_lightgbm_cv,
    train_lightgbm_regression_cv,
)
from src.utils.io import read_dataframe, read_yaml, write_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM with expanding-window CV and export predictions.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs/model.yaml"))
    return parser.parse_args()


def main() -> None:
    root = ROOT
    args = parse_args()
    config = read_yaml(args.config)
    model_table = read_dataframe(root / config["artifacts"]["weekly_model_table"])
    feature_columns = [
        "article_count",
        "relevance_mean",
        "polarity_mean",
        "intensity_mean",
        "uncertainty_mean",
        "forwardness_mean",
        "polarity_std",
        "uncertainty_std",
        "polarity_momentum",
        "uncertainty_momentum",
        "forwardness_momentum",
    ]
    training_config = build_training_config(config["training"])
    if training_config.task == "regression":
        model_table = prepare_regression_table(model_table, feature_columns=feature_columns)
        predictions, metrics, _, final_model = train_lightgbm_regression_cv(
            model_table,
            feature_columns=feature_columns,
            config=training_config,
        )
    else:
        model_table = prepare_training_table(model_table, feature_columns=feature_columns)
        predictions, metrics, _, final_model = train_lightgbm_cv(
            model_table,
            feature_columns=feature_columns,
            config=training_config,
        )
    predictions_path = root / config["artifacts"]["predictions"]
    metrics_path = root / config["artifacts"].get("cv_metrics", "reports/cv_metrics.parquet")
    write_dataframe(predictions, predictions_path)
    write_dataframe(metrics, metrics_path)
    predictions.to_csv(predictions_path.with_suffix(".csv"), index=False)
    metrics.to_csv(metrics_path.with_suffix(".csv"), index=False)
    export_shap_outputs(
        final_model,
        model_table,
        feature_columns=feature_columns,
        output_dir=root / config["artifacts"].get("shap_dir", "reports"),
    )
    print(predictions_path)


if __name__ == "__main__":
    main()
