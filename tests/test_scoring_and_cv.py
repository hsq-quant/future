from __future__ import annotations

import pandas as pd
import pytest

from src.features.scoring import (
    build_article_prompt,
    filter_unscored_articles,
    iter_article_score_attempts,
    mock_score_article,
    parse_sentiment_response,
    restrict_scores_to_articles,
    resolve_api_key,
    score_articles_batch,
)
from src.pipeline.scoring_shards import select_articles_for_shard
from src.models.cv import make_expanding_window_splits
from src.models.train_eval import (
    TrainingConfig,
    _compute_ic,
    _suggest_params,
    build_training_config,
    prepare_regression_table,
    prepare_training_table,
)


def test_parse_sentiment_response_forces_null_dimensions_when_relevance_is_low() -> None:
    payload = """
    {
      "relevance": 0.05,
      "polarity": 0.8,
      "intensity": 0.9,
      "uncertainty": 0.2,
      "forwardness": 0.7
    }
    """

    parsed = parse_sentiment_response(payload)

    assert parsed.relevance == 0.05
    assert parsed.polarity is None
    assert parsed.intensity is None
    assert parsed.uncertainty is None
    assert parsed.forwardness is None


def test_parse_sentiment_response_rejects_out_of_range_values() -> None:
    bad_payload = """
    {
      "relevance": 0.8,
      "polarity": 1.4,
      "intensity": 0.3,
      "uncertainty": 0.2,
      "forwardness": 0.6
    }
    """

    with pytest.raises(ValueError):
        parse_sentiment_response(bad_payload)


def test_parse_sentiment_response_coerces_null_dimension_when_relevance_is_high() -> None:
    bad_payload = """
    {
      "relevance": 0.8,
      "polarity": null,
      "intensity": 0.3,
      "uncertainty": 0.2,
      "forwardness": 0.6
    }
    """

    parsed = parse_sentiment_response(bad_payload)

    assert parsed.relevance < 0.1
    assert parsed.polarity is None
    assert parsed.intensity is None
    assert parsed.uncertainty is None
    assert parsed.forwardness is None


def test_parse_sentiment_response_coerces_boundary_low_relevance_case() -> None:
    payload = """
    {
      "relevance": 0.1,
      "polarity": null,
      "intensity": null,
      "uncertainty": null,
      "forwardness": null
    }
    """

    parsed = parse_sentiment_response(payload)

    assert parsed.relevance < 0.1
    assert parsed.polarity is None


def test_parse_sentiment_response_coerces_partial_null_case_to_low_relevance() -> None:
    payload = """
    {
      "relevance": 0.3,
      "polarity": null,
      "intensity": 0.4,
      "uncertainty": null,
      "forwardness": 0.8
    }
    """

    parsed = parse_sentiment_response(payload)

    assert parsed.relevance < 0.1
    assert parsed.polarity is None
    assert parsed.intensity is None
    assert parsed.uncertainty is None
    assert parsed.forwardness is None


def test_parse_sentiment_response_coerces_null_relevance_to_zero() -> None:
    payload = """
    {
      "relevance": null,
      "polarity": null,
      "intensity": null,
      "uncertainty": null,
      "forwardness": null
    }
    """

    parsed = parse_sentiment_response(payload)

    assert parsed.relevance == 0.0
    assert parsed.polarity is None


def test_make_expanding_window_splits_never_leaks_future_weeks() -> None:
    data = pd.DataFrame(
        {
            "week_end_date": pd.date_range("2026-01-02", periods=15, freq="7D"),
            "feature": range(15),
        }
    )

    splits = make_expanding_window_splits(data, n_splits=5)

    assert len(splits) == 5
    for train_idx, valid_idx in splits:
        assert len(train_idx) > 0
        assert len(valid_idx) > 0
        assert data.loc[train_idx, "week_end_date"].max() < data.loc[valid_idx, "week_end_date"].min()


def test_mock_score_article_returns_paper_shaped_payload() -> None:
    article = {
        "title": "OPEC减产推动油价上涨",
        "body": "市场预期未来供应偏紧，上海原油情绪偏强。",
    }

    scored = mock_score_article(article)

    assert scored["prompt_version"].endswith("_mock")
    assert scored["relevance"] >= 0.1
    assert scored["polarity"] is not None
    assert scored["intensity"] is not None
    assert scored["uncertainty"] is not None
    assert scored["forwardness"] is not None


def test_build_article_prompt_truncates_long_body() -> None:
    article = {
        "title": "Long English energy story",
        "body": "A" * 6000,
    }

    prompt = build_article_prompt(article)

    assert prompt.startswith("标题：Long English energy story")
    assert len(prompt) < 4500
    assert "A" * 5000 not in prompt


def test_iter_article_score_attempts_continues_after_error() -> None:
    articles = pd.DataFrame(
        [
            {"article_id": "a1", "title": "ok", "body": "ok"},
            {"article_id": "a2", "title": "bad", "body": "bad"},
            {"article_id": "a3", "title": "ok2", "body": "ok2"},
        ]
    )

    def _scorer(article: dict[str, str]) -> dict[str, object]:
        if article["article_id"] == "a2":
            raise ValueError("blocked")
        return {"relevance": 0.5}

    attempts = list(iter_article_score_attempts(articles, scorer=_scorer, max_workers=1))

    assert attempts[0][1] == {"relevance": 0.5}
    assert attempts[1][2] == "blocked"
    assert attempts[2][1] == {"relevance": 0.5}


def test_resolve_api_key_falls_back_to_dashscope_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")

    assert resolve_api_key("QWEN_API_KEY") == "test-key"


def test_prepare_training_table_keeps_rows_with_missing_features() -> None:
    data = pd.DataFrame(
        {
            "week_end_date": pd.to_datetime(["2026-01-02", "2026-01-09", "2026-01-16"]),
            "article_count": [1, 1, 1],
            "relevance_mean": [0.5, 0.4, 0.3],
            "polarity_mean": [0.2, None, 0.1],
            "intensity_mean": [0.4, None, 0.3],
            "uncertainty_mean": [0.2, None, 0.2],
            "forwardness_mean": [0.6, None, 0.5],
            "polarity_std": [0.0, None, 0.1],
            "uncertainty_std": [0.0, None, 0.1],
            "polarity_momentum": [None, None, 0.1],
            "uncertainty_momentum": [None, None, 0.0],
            "forwardness_momentum": [None, None, -0.1],
            "next_week_label": [1, 0, None],
            "next_week_return": [0.01, -0.02, None],
        }
    )
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

    prepared = prepare_training_table(data, feature_columns=feature_columns)

    assert len(prepared) == 2
    assert prepared["week_end_date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-01-02", "2026-01-09"]


def test_build_training_config_ignores_unknown_fields() -> None:
    config = build_training_config(
        {
            "n_splits": 5,
            "n_trials": 10,
            "random_state": 7,
            "task": "regression",
            "threshold_rule": "training_share_aligned",
        }
    )

    assert config == TrainingConfig(n_splits=5, n_trials=10, random_state=7, task="regression")


def test_prepare_regression_table_keeps_rows_with_forward_returns() -> None:
    data = pd.DataFrame(
        {
            "week_end_date": pd.to_datetime(["2026-01-02", "2026-01-09", "2026-01-16"]),
            "article_count": [1, 1, 1],
            "relevance_mean": [0.5, 0.4, 0.3],
            "polarity_mean": [0.2, None, 0.1],
            "intensity_mean": [0.4, None, 0.3],
            "uncertainty_mean": [0.2, None, 0.2],
            "forwardness_mean": [0.6, None, 0.5],
            "polarity_std": [0.0, None, 0.1],
            "uncertainty_std": [0.0, None, 0.1],
            "polarity_momentum": [None, None, 0.1],
            "uncertainty_momentum": [None, None, 0.0],
            "forwardness_momentum": [None, None, -0.1],
            "next_week_return": [0.01, -0.02, None],
        }
    )
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

    prepared = prepare_regression_table(data, feature_columns=feature_columns)

    assert len(prepared) == 2
    assert prepared["week_end_date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-01-02", "2026-01-09"]


def test_compute_ic_uses_spearman_rank_correlation() -> None:
    pred = pd.Series([0.1, 0.4, 0.2, 0.3], dtype=float)
    actual = pd.Series([0.0, 0.5, 0.1, 0.2], dtype=float)

    ic = _compute_ic(pred, actual)

    assert round(ic, 6) == 1.0


def test_compute_ic_returns_zero_for_constant_predictions() -> None:
    pred = pd.Series([0.2, 0.2, 0.2, 0.2], dtype=float)
    actual = pd.Series([0.0, 0.5, 0.1, 0.2], dtype=float)

    ic = _compute_ic(pred, actual)

    assert ic == 0.0


def test_suggest_params_uses_single_thread() -> None:
    class _TrialStub:
        def suggest_int(self, name: str, low: int, high: int) -> int:
            return low

        def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
            return low

    params = _suggest_params(_TrialStub(), random_state=42)

    assert params["n_jobs"] == 1


def test_filter_unscored_articles_skips_existing_article_ids() -> None:
    articles = pd.DataFrame(
        {
            "article_id": ["a1", "a2", "a3"],
            "title": ["t1", "t2", "t3"],
        }
    )
    existing = pd.DataFrame({"article_id": ["a2"]})

    filtered = filter_unscored_articles(articles, existing)

    assert filtered["article_id"].tolist() == ["a1", "a3"]


def test_select_articles_for_shard_partitions_article_ids_without_overlap() -> None:
    articles = pd.DataFrame(
        {
            "article_id": [f"a{i}" for i in range(9)],
            "title": [f"t{i}" for i in range(9)],
        }
    )

    shard_0 = select_articles_for_shard(articles, num_shards=3, shard_index=0)
    shard_1 = select_articles_for_shard(articles, num_shards=3, shard_index=1)
    shard_2 = select_articles_for_shard(articles, num_shards=3, shard_index=2)

    combined = shard_0["article_id"].tolist() + shard_1["article_id"].tolist() + shard_2["article_id"].tolist()

    assert len(combined) == len(set(combined)) == len(articles)


def test_restrict_scores_to_articles_drops_legacy_scores() -> None:
    articles = pd.DataFrame({"article_id": ["a1", "a2"]})
    existing = pd.DataFrame({"article_id": ["a1", "legacy"], "relevance": [0.8, 0.5]})

    restricted = restrict_scores_to_articles(existing, articles)

    assert restricted["article_id"].tolist() == ["a1"]


def test_score_articles_batch_processes_all_articles_with_parallel_workers() -> None:
    articles = pd.DataFrame(
        [
            {"article_id": "a1", "title": "t1", "body": "b1"},
            {"article_id": "a2", "title": "t2", "body": "b2"},
            {"article_id": "a3", "title": "t3", "body": "b3"},
        ]
    )

    def scorer(article: dict[str, object]) -> dict[str, object]:
        return {
            "prompt_version": "test",
            "model_name": "test-model",
            "raw_response": "{}",
            "relevance": 0.5,
            "polarity": 0.1,
            "intensity": 0.2,
            "uncertainty": 0.3,
            "forwardness": 0.4,
        }

    rows = score_articles_batch(articles, scorer=scorer, max_workers=3)

    assert len(rows) == 3
    assert {row["article_id"] for row in rows} == {"a1", "a2", "a3"}
