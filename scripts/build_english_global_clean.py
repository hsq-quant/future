from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.english_energy_candidates import (
    assign_stable_english_article_ids,
    build_english_energy_candidates,
    resolve_english_energy_phrases,
)
from src.data.rss_ingest import clean_articles
from src.utils.io import read_dataframe, read_yaml, write_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an expanded English/global energy-news clean dataset.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(ROOT / "data/raw/archive/english_global/financial_news_multisource_recent.parquet"),
    )
    parser.add_argument(
        "--supplement-input",
        action="append",
        default=[],
        help="Optional extra English raw archives to append before candidate filtering.",
    )
    parser.add_argument(
        "--english-output",
        type=str,
        default=str(ROOT / "data/intermediate/articles_archive_clean_english.parquet"),
    )
    parser.add_argument(
        "--multilingual-output",
        type=str,
        default=str(ROOT / "data/intermediate/articles_archive_clean_multilingual.parquet"),
    )
    parser.add_argument(
        "--candidate-output",
        type=str,
        default=str(ROOT / "data/raw/archive/english_global/financial_news_multisource_oil_candidates.parquet"),
    )
    parser.add_argument(
        "--base-chinese-input",
        type=str,
        default=str(ROOT / "data/intermediate/articles_archive_clean.parquet"),
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--phrases",
        nargs="*",
        default=None,
        help="Optional override for English energy phrases.",
    )
    parser.add_argument(
        "--phrase-profile",
        choices=["strict", "broader", "energy-wide"],
        default="strict",
        help="Named English topic filter profile when --phrases is not provided.",
    )
    parser.add_argument(
        "--rekey-existing-scored",
        action="store_true",
        help="Explicitly rewrite English article_id values inside articles_scored.parquet for cache reuse.",
    )
    return parser.parse_args()


def _rekey_existing_english_scores(scored_path: Path) -> None:
    if not scored_path.exists():
        return
    scored = read_dataframe(scored_path)
    if scored.empty or "language" not in scored.columns:
        return
    english_mask = scored["language"].fillna("").astype(str).str.startswith("en")
    if not english_mask.any():
        return
    rescored = scored.copy()
    rescored.loc[english_mask] = assign_stable_english_article_ids(rescored.loc[english_mask])
    if "article_id" in rescored.columns:
        rescored = rescored.drop_duplicates(subset=["article_id"], keep="last").reset_index(drop=True)
    write_dataframe(rescored, scored_path)


def main() -> None:
    args = parse_args()
    archive_cfg = read_yaml(ROOT / "configs/archive.yaml").get("archive", {})
    start_date = args.start_date or archive_cfg.get("start_date")
    end_date = args.end_date or archive_cfg.get("end_date")
    phrases = args.phrases or resolve_english_energy_phrases(args.phrase_profile)

    raw_frames = [read_dataframe(args.input)]
    for supplement in args.supplement_input:
        raw_frames.append(read_dataframe(supplement))
    raw = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    if not raw.empty:
        raw = raw.drop_duplicates(subset=["title", "body", "published_at", "url", "source"], keep="first").reset_index(drop=True)
    candidates = build_english_energy_candidates(raw, phrases=phrases)
    candidates = assign_stable_english_article_ids(candidates)
    write_dataframe(candidates, args.candidate_output)

    trading_calendar = read_dataframe(ROOT / "data/intermediate/continuous_daily.parquet")[["trade_date"]].drop_duplicates()
    english_clean = clean_articles(
        candidates,
        trading_calendar=trading_calendar,
        energy_keywords=[],
        start_date=start_date,
        end_date=end_date,
        allowed_language_prefixes=("en",),
        apply_keyword_filter=False,
    )
    english_clean = assign_stable_english_article_ids(english_clean)
    write_dataframe(english_clean, args.english_output)

    chinese_clean = read_dataframe(args.base_chinese_input)
    if "language" in chinese_clean.columns:
        chinese_clean = chinese_clean[chinese_clean["language"].fillna("").astype(str).str.startswith("zh")].copy()
    multilingual = pd.concat([chinese_clean, english_clean], ignore_index=True)
    multilingual = multilingual.sort_values(["published_at", "title"]).reset_index(drop=True)
    if "article_id" in multilingual.columns:
        multilingual = multilingual.drop_duplicates(subset=["article_id"], keep="last").reset_index(drop=True)
    write_dataframe(multilingual, args.multilingual_output)

    if args.rekey_existing_scored:
        _rekey_existing_english_scores(ROOT / "data/intermediate/articles_scored.parquet")

    print(args.candidate_output)
    print(f"candidate_rows={len(candidates)}")
    print(args.english_output)
    print(f"english_clean_rows={len(english_clean)}")
    print(f"english_weeks={english_clean['week_end_date'].nunique() if 'week_end_date' in english_clean.columns else 0}")
    print(args.multilingual_output)
    print(f"multilingual_rows={len(multilingual)}")
    if "language" in multilingual.columns:
        print(multilingual["language"].value_counts().to_string())


if __name__ == "__main__":
    main()
