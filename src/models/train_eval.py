from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from src.models.cv import make_expanding_window_splits
from src.models.thresholds import apply_class_share_threshold
from src.utils.io import write_dataframe

if TYPE_CHECKING:
    import lightgbm as lgb


@dataclass(frozen=True)
class TrainingConfig:
    n_splits: int = 5
    n_trials: int = 20
    random_state: int = 42
    task: str = "classification"


def build_training_config(raw: dict[str, object]) -> TrainingConfig:
    allowed = TrainingConfig.__annotations__.keys()
    filtered = {key: value for key, value in raw.items() if key in allowed}
    return TrainingConfig(**filtered)


def prepare_training_table(model_table: pd.DataFrame, *, feature_columns: list[str]) -> pd.DataFrame:
    return (
        model_table.sort_values("week_end_date")
        .dropna(subset=["next_week_label", "next_week_return"])
        .reset_index(drop=True)
    )


def prepare_regression_table(model_table: pd.DataFrame, *, feature_columns: list[str]) -> pd.DataFrame:
    return (
        model_table.sort_values("week_end_date")
        .dropna(subset=["next_week_return"])
        .reset_index(drop=True)
    )


def _compute_ic(pred_prob: pd.Series, actual_return: pd.Series) -> float:
    if pred_prob.nunique(dropna=True) <= 1 or actual_return.nunique(dropna=True) <= 1:
        return 0.0
    value = float(pred_prob.corr(actual_return, method="spearman"))
    if pd.isna(value):
        return 0.0
    return value


def _suggest_params(trial: optuna.Trial, random_state: int, *, task: str = "classification") -> dict[str, int | float | str]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 250),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, 63),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "random_state": random_state,
        "objective": "binary" if task == "classification" else "regression",
        "verbosity": -1,
        "n_jobs": 1,
    }


def train_lightgbm_cv(
    model_table: pd.DataFrame,
    *,
    feature_columns: list[str],
    config: TrainingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int | float], Any]:
    import lightgbm as lgb

    data = model_table.sort_values("week_end_date").reset_index(drop=True)
    splits = make_expanding_window_splits(data, n_splits=config.n_splits)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, config.random_state, task="classification")
        scores: list[float] = []
        for train_idx, valid_idx in splits:
            train_df = data.loc[train_idx]
            valid_df = data.loc[valid_idx]
            model = lgb.LGBMClassifier(**params)
            model.fit(train_df[feature_columns], train_df["next_week_label"])
            probs = pd.Series(model.predict_proba(valid_df[feature_columns])[:, 1], index=valid_df.index)
            scores.append(roc_auc_score(valid_df["next_week_label"], probs))
        return float(sum(scores) / len(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=config.n_trials)
    best_params = {**study.best_params, "objective": "binary", "verbosity": -1, "random_state": config.random_state}

    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, float | int]] = []

    for fold_id, (train_idx, valid_idx) in enumerate(splits, start=1):
        train_df = data.loc[train_idx]
        valid_df = data.loc[valid_idx]
        model = lgb.LGBMClassifier(**best_params)
        model.fit(train_df[feature_columns], train_df["next_week_label"])
        pred_prob = pd.Series(model.predict_proba(valid_df[feature_columns])[:, 1], index=valid_df.index)

        training_positive_share = float(train_df["next_week_label"].mean())
        threshold, pred_label = apply_class_share_threshold(pred_prob, training_positive_share)

        metric_rows.append(
            {
                "fold_id": fold_id,
                "auc": float(roc_auc_score(valid_df["next_week_label"], pred_prob)),
                "accuracy": float(accuracy_score(valid_df["next_week_label"], pred_label)),
                "ic": _compute_ic(pred_prob, valid_df["next_week_return"]),
                "threshold_used": threshold,
            }
        )

        for index in valid_df.index:
            prediction_rows.append(
                {
                    "week_end_date": valid_df.at[index, "week_end_date"],
                    "pred_prob": float(pred_prob.at[index]),
                    "pred_label": int(pred_label.loc[pred_prob.index.get_loc(index)]),
                    "actual_return": float(valid_df.at[index, "next_week_return"]),
                    "actual_label": int(valid_df.at[index, "next_week_label"]),
                    "fold_id": fold_id,
                    "threshold_used": threshold,
                }
            )

    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(data[feature_columns], data["next_week_label"])

    predictions = pd.DataFrame(prediction_rows).sort_values("week_end_date").reset_index(drop=True)
    metrics = pd.DataFrame(metric_rows)
    return predictions, metrics, best_params, final_model


def train_lightgbm_regression_cv(
    model_table: pd.DataFrame,
    *,
    feature_columns: list[str],
    config: TrainingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int | float], Any]:
    import lightgbm as lgb

    data = model_table.sort_values("week_end_date").reset_index(drop=True)
    splits = make_expanding_window_splits(data, n_splits=config.n_splits)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, config.random_state, task="regression")
        scores: list[float] = []
        for train_idx, valid_idx in splits:
            train_df = data.loc[train_idx]
            valid_df = data.loc[valid_idx]
            model = lgb.LGBMRegressor(**params)
            model.fit(train_df[feature_columns], train_df["next_week_return"])
            pred_value = pd.Series(model.predict(valid_df[feature_columns]), index=valid_df.index)
            scores.append(_compute_ic(pred_value, valid_df["next_week_return"]))
        return float(sum(scores) / len(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=config.n_trials)
    best_params = {**study.best_params, "objective": "regression", "verbosity": -1, "random_state": config.random_state}

    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, float | int]] = []

    for fold_id, (train_idx, valid_idx) in enumerate(splits, start=1):
        train_df = data.loc[train_idx]
        valid_df = data.loc[valid_idx]
        model = lgb.LGBMRegressor(**best_params)
        model.fit(train_df[feature_columns], train_df["next_week_return"])
        pred_value = pd.Series(model.predict(valid_df[feature_columns]), index=valid_df.index)
        pred_label = (pred_value > 0).astype(int)
        actual_label = (valid_df["next_week_return"] > 0).astype(int)
        errors = pred_value - valid_df["next_week_return"]

        metric_rows.append(
            {
                "fold_id": fold_id,
                "ic": _compute_ic(pred_value, valid_df["next_week_return"]),
                "mse": float((errors.pow(2)).mean()),
                "mae": float(errors.abs().mean()),
                "directional_accuracy": float(accuracy_score(actual_label, pred_label)),
            }
        )

        for index in valid_df.index:
            value = float(pred_value.at[index])
            prediction_rows.append(
                {
                    "week_end_date": valid_df.at[index, "week_end_date"],
                    "pred_value": value,
                    "pred_label": int(value > 0),
                    "actual_return": float(valid_df.at[index, "next_week_return"]),
                    "actual_label": int(valid_df.at[index, "next_week_return"] > 0),
                    "fold_id": fold_id,
                }
            )

    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(data[feature_columns], data["next_week_return"])

    predictions = pd.DataFrame(prediction_rows).sort_values("week_end_date").reset_index(drop=True)
    metrics = pd.DataFrame(metric_rows)
    return predictions, metrics, best_params, final_model


def export_shap_outputs(
    model: Any,
    model_table: pd.DataFrame,
    *,
    feature_columns: list[str],
    output_dir: str | Path,
) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    import shap

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(model_table[feature_columns])
    if isinstance(shap_values, list):
        shap_matrix = shap_values[-1]
    else:
        shap_matrix = shap_values

    summary = (
        pd.DataFrame({"feature": feature_columns, "mean_abs_shap": abs(shap_matrix).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    write_dataframe(summary, output / "shap_summary.parquet")

    plt.figure()
    shap.summary_plot(shap_matrix, model_table[feature_columns], show=False)
    plt.tight_layout()
    plt.savefig(output / "shap_summary.png", dpi=150)
    plt.close()

    return summary
