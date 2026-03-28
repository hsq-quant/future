from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterator

import pandas as pd
import requests
from pydantic import BaseModel, Field, ValidationError, model_validator
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed


PROMPT_VERSION = "paper_appendix_a_v1_zh"
MOCK_PROMPT_VERSION = "paper_appendix_a_v1_zh_mock"

SYSTEM_PROMPT = """你是一位专注于能源市场的财经新闻分析师。

请分析输入新闻，并只返回一个 JSON 对象，必须包含以下字段：
{
  "relevance": float,
  "polarity": float,
  "intensity": float,
  "uncertainty": float,
  "forwardness": float
}

规则：
1. 先判断 relevance。
2. 如果 relevance < 0.1，则 polarity、intensity、uncertainty、forwardness 必须全部返回 null。
2.1 如果 relevance >= 0.1，则 polarity、intensity、uncertainty、forwardness 必须全部返回数字，不能为 null。
3. polarity 取值范围 [-1, 1]，负值代表看空能源/原油，正值代表看多。
4. intensity 取值范围 [0, 1]，表示表达强烈程度，与 polarity 独立。
5. uncertainty 取值范围 [0, 1]，反映模糊、对冲、不确定或风险措辞。
6. forwardness 取值范围 [0, 1]，反映未来导向、预测、展望、预期。
7. 不要输出解释，不要输出 markdown，不要输出代码块，只输出 JSON。"""

MAX_TITLE_CHARS = 400
MAX_BODY_CHARS = 4000

POSITIVE_KEYWORDS = ("上涨", "走强", "反弹", "减产", "偏紧", "利多", "回升")
NEGATIVE_KEYWORDS = ("下跌", "走弱", "下行", "增产", "过剩", "利空", "回落")
UNCERTAINTY_KEYWORDS = ("或", "可能", "预期", "不确定", "风险", "担忧")
FORWARD_KEYWORDS = ("未来", "预期", "将", "展望", "后续", "明年")
ENERGY_KEYWORDS = ("原油", "油价", "OPEC", "上海原油", "INE", "能源", "库存")


def resolve_api_key(api_key_env: str) -> str | None:
    return os.getenv(api_key_env) or os.getenv("DASHSCOPE_API_KEY")


class SentimentScore(BaseModel):
    relevance: float = Field(ge=0.0, le=1.0)
    polarity: float | None = Field(default=None, ge=-1.0, le=1.0)
    intensity: float | None = Field(default=None, ge=0.0, le=1.0)
    uncertainty: float | None = Field(default=None, ge=0.0, le=1.0)
    forwardness: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _apply_relevance_rule(self) -> "SentimentScore":
        if self.relevance < 0.1:
            self.polarity = None
            self.intensity = None
            self.uncertainty = None
            self.forwardness = None
        elif any(value is None for value in (self.polarity, self.intensity, self.uncertainty, self.forwardness)):
            raise ValueError("All non-relevance dimensions must be numeric when relevance >= 0.1.")
        return self


def build_article_prompt(article: pd.Series | dict[str, Any]) -> str:
    title = article["title"] if isinstance(article, dict) else article.get("title", "")
    body = article["body"] if isinstance(article, dict) else article.get("body", "")
    title = str(title or "").strip()[:MAX_TITLE_CHARS]
    body = str(body or "").strip()
    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS]
    return f"标题：{title}\n\n正文：{body}"


def mock_score_article(article: pd.Series | dict[str, Any]) -> dict[str, Any]:
    title = article["title"] if isinstance(article, dict) else article.get("title", "")
    body = article["body"] if isinstance(article, dict) else article.get("body", "")
    text = f"{title}\n{body}"

    relevance = 0.9 if any(keyword in text for keyword in ENERGY_KEYWORDS) else 0.05
    if relevance < 0.1:
        return {
            "prompt_version": MOCK_PROMPT_VERSION,
            "model_name": "mock-qwen",
            "raw_response": '{"relevance": 0.05, "polarity": null, "intensity": null, "uncertainty": null, "forwardness": null}',
            "relevance": 0.05,
            "polarity": None,
            "intensity": None,
            "uncertainty": None,
            "forwardness": None,
        }

    positive_hits = sum(keyword in text for keyword in POSITIVE_KEYWORDS)
    negative_hits = sum(keyword in text for keyword in NEGATIVE_KEYWORDS)
    uncertainty_hits = sum(keyword in text for keyword in UNCERTAINTY_KEYWORDS)
    forward_hits = sum(keyword in text for keyword in FORWARD_KEYWORDS)

    polarity = 0.0
    if positive_hits > negative_hits:
        polarity = min(1.0, 0.2 + 0.2 * positive_hits)
    elif negative_hits > positive_hits:
        polarity = max(-1.0, -(0.2 + 0.2 * negative_hits))

    intensity = min(1.0, 0.3 + 0.15 * max(positive_hits, negative_hits, 1))
    uncertainty = min(1.0, 0.1 + 0.15 * uncertainty_hits)
    forwardness = min(1.0, 0.2 + 0.15 * forward_hits)

    payload = {
        "relevance": relevance,
        "polarity": polarity,
        "intensity": intensity,
        "uncertainty": uncertainty,
        "forwardness": forwardness,
    }
    return {
        "prompt_version": MOCK_PROMPT_VERSION,
        "model_name": "mock-qwen",
        "raw_response": json.dumps(payload, ensure_ascii=False),
        **payload,
    }


def filter_unscored_articles(articles: pd.DataFrame, existing_scored: pd.DataFrame | None) -> pd.DataFrame:
    if existing_scored is None or existing_scored.empty or "article_id" not in articles.columns:
        return articles.copy()
    if "article_id" not in existing_scored.columns:
        return articles.copy()
    existing_ids = set(existing_scored["article_id"].dropna().astype(str))
    return articles[~articles["article_id"].astype(str).isin(existing_ids)].copy()


def restrict_scores_to_articles(existing_scored: pd.DataFrame, articles: pd.DataFrame) -> pd.DataFrame:
    if existing_scored.empty or "article_id" not in existing_scored.columns or "article_id" not in articles.columns:
        return existing_scored.copy()
    valid_ids = set(articles["article_id"].dropna().astype(str))
    return existing_scored[existing_scored["article_id"].astype(str).isin(valid_ids)].copy()


def score_articles_batch(
    articles: pd.DataFrame,
    *,
    scorer: Callable[[dict[str, Any]], dict[str, Any]],
    max_workers: int = 1,
) -> list[dict[str, Any]]:
    return list(iter_scored_articles_batch(articles, scorer=scorer, max_workers=max_workers))


def iter_article_score_attempts(
    articles: pd.DataFrame,
    *,
    scorer: Callable[[dict[str, Any]], dict[str, Any]],
    max_workers: int = 1,
) -> Iterator[tuple[dict[str, Any], dict[str, Any] | None, str | None]]:
    records = articles.to_dict(orient="records")
    if max_workers <= 1:
        for record in records:
            try:
                yield record, scorer(record), None
            except Exception as exc:  # noqa: BLE001
                yield record, None, str(exc)
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_record = {executor.submit(scorer, record): record for record in records}
        for future in as_completed(future_to_record):
            record = future_to_record[future]
            try:
                yield record, future.result(), None
            except Exception as exc:  # noqa: BLE001
                yield record, None, str(exc)


def iter_scored_articles_batch(
    articles: pd.DataFrame,
    *,
    scorer: Callable[[dict[str, Any]], dict[str, Any]],
    max_workers: int = 1,
) -> Iterator[dict[str, Any]]:
    for record, score, error in iter_article_score_attempts(articles, scorer=scorer, max_workers=max_workers):
        if error is not None or score is None:
            raise ValueError(error or "Unknown scoring error.")
        yield {**record, **score}


def _extract_json_blob(payload: str) -> str:
    text = payload.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        return fenced.group(1)
    brace = re.search(r"(\{.*\})", text, re.S)
    if brace:
        return brace.group(1)
    raise ValueError("No JSON object found in model response.")


def parse_sentiment_response(payload: str) -> SentimentScore:
    try:
        raw = json.loads(_extract_json_blob(payload))
        if isinstance(raw, dict):
            fields = ("polarity", "intensity", "uncertainty", "forwardness")
            relevance = raw.get("relevance")
            if relevance is None:
                raw["relevance"] = 0.0
                relevance = 0.0
            if relevance == 0.1 and all(raw.get(field) is None for field in fields):
                # Boundary-value repair: some models use 0.1 to express "below threshold".
                raw["relevance"] = 0.099
            elif relevance >= 0.1 and any(raw.get(field) is None for field in fields):
                # Conservative repair: if the model returns a mixed high-relevance payload,
                # treat it as below-threshold rather than inventing missing dimensions.
                raw["relevance"] = 0.099
                for field in fields:
                    raw[field] = None
        return SentimentScore.model_validate(raw)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"Invalid sentiment payload: {exc}") from exc


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    retry=retry_if_exception_type((requests.RequestException, ValueError)),
    reraise=True,
)
def score_article_with_qwen(
    article: pd.Series | dict[str, Any],
    *,
    model: str,
    base_url: str,
    api_key_env: str,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    api_key = resolve_api_key(api_key_env)
    if not api_key:
        raise ValueError(f"Missing API key in environment variable: {api_key_env}")

    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_article_prompt(article)},
            ],
        },
        timeout=timeout_seconds,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = response.text[:500] if response is not None else ""
        raise ValueError(f"DashScope request failed with status {response.status_code}: {detail}") from exc
    data = response.json()
    message = data["choices"][0]["message"]["content"]
    parsed = parse_sentiment_response(message)
    return {
        "prompt_version": PROMPT_VERSION,
        "model_name": model,
        "raw_response": message,
        **asdict(_SentimentScoreAdapter.from_model(parsed)),
    }


@dataclass
class _SentimentScoreAdapter:
    relevance: float
    polarity: float | None
    intensity: float | None
    uncertainty: float | None
    forwardness: float | None

    @classmethod
    def from_model(cls, model: SentimentScore) -> "_SentimentScoreAdapter":
        return cls(**model.model_dump())
