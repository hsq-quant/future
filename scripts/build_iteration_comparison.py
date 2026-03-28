from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.utils.io import read_dataframe


ITERATIONS = [
    {
        "name": "V1 中文主链",
        "slug": "2026-03-27-single-source-baseline",
        "languages": "zh",
        "clean_path": ROOT / "data/processed/iterations/2026-03-27-single-source-baseline/articles_archive_clean.parquet",
        "model_table_path": ROOT / "data/processed/weekly_model_table.parquet",
        "cv_metrics_path": ROOT / "reports/cv_metrics.csv",
        "strategy_metrics_path": ROOT / "reports/strategy_metrics.csv",
    },
    {
        "name": "V2 中英增强",
        "slug": "2026-03-27-multilingual-v2",
        "languages": "zh+en",
        "clean_path": ROOT / "data/processed/iterations/2026-03-27-multilingual-v2/articles_archive_clean_multilingual.parquet",
        "model_table_path": ROOT / "data/processed/iterations/2026-03-27-multilingual-v2/weekly_model_table.parquet",
        "cv_metrics_path": ROOT / "reports/iterations/2026-03-27-multilingual-v2/cv_metrics.csv",
        "strategy_metrics_path": ROOT / "reports/iterations/2026-03-27-multilingual-v2/strategy_metrics.csv",
    },
    {
        "name": "V2 衍生版纯英文",
        "slug": "2026-03-27-english-only-v2",
        "languages": "en",
        "clean_path": ROOT / "data/processed/iterations/2026-03-27-english-only-v2/articles_archive_clean_multilingual.parquet",
        "model_table_path": ROOT / "data/processed/iterations/2026-03-27-english-only-v2/weekly_model_table.parquet",
        "cv_metrics_path": ROOT / "reports/iterations/2026-03-27-english-only-v2/cv_metrics.csv",
        "strategy_metrics_path": ROOT / "reports/iterations/2026-03-27-english-only-v2/strategy_metrics.csv",
    },
]


def _load_clean_rows(path: Path) -> tuple[int, dict[str, int]]:
    frame = read_dataframe(path)
    counts = frame.get("language")
    if counts is None:
        return len(frame), {}
    language_counts = frame["language"].astype(str).str[:2].value_counts().to_dict()
    return len(frame), {str(k): int(v) for k, v in language_counts.items()}


def _load_model_window(path: Path) -> tuple[int, str, str]:
    frame = read_dataframe(path)
    return len(frame), str(frame["week_end_date"].min().date()), str(frame["week_end_date"].max().date())


def _load_cv_means(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    return {key: float(frame[key].mean()) for key in ["auc", "accuracy", "ic"]}


def _load_strategy_row(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    row = frame.iloc[0].to_dict()
    keys = [
        "num_weeks",
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "win_rate",
        "cumulative_return_diff",
        "annualized_return_diff",
        "sharpe_ratio_diff",
        "information_ratio",
    ]
    return {key: float(row[key]) for key in keys}


def main() -> None:
    rows: list[dict[str, object]] = []
    for item in ITERATIONS:
        clean_rows, language_counts = _load_clean_rows(item["clean_path"])
        model_rows, start_week, end_week = _load_model_window(item["model_table_path"])
        cv = _load_cv_means(item["cv_metrics_path"])
        strategy = _load_strategy_row(item["strategy_metrics_path"])
        rows.append(
            {
                "iteration": item["name"],
                "slug": item["slug"],
                "languages": item["languages"],
                "clean_articles": clean_rows,
                "language_counts": ", ".join(f"{k}:{v}" for k, v in sorted(language_counts.items())),
                "supervised_weeks": model_rows,
                "start_week": start_week,
                "end_week": end_week,
                **cv,
                **strategy,
                "strictly_comparable_to_v1": item["slug"] != "2026-03-27-english-only-v2",
            }
        )

    output_dir = ROOT / "reports/iterations"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(rows)
    csv_path = output_dir / "iteration_comparison.csv"
    md_path = output_dir / "iteration_comparison.md"
    comparison.to_csv(csv_path, index=False)

    def pct(value: float) -> str:
        return f"{value * 100:.2f}%"

    lines = [
        "# Iteration Comparison",
        "",
        "| 版本 | 语言 | 清洗文章数 | 监督周数 | 样本外周数 | 周区间 | AUC | Accuracy | IC | 累计收益 | 夏普 | 超额累计收益差 | 备注 |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        note = "可与V1直接比较" if row["strictly_comparable_to_v1"] else "样本更短，不能与V1硬比"
        lines.append(
            "| {iteration} | {languages} | {clean_articles} | {supervised_weeks} | {num_weeks:.0f} | {start_week} ~ {end_week} | "
            "{auc:.4f} | {accuracy:.4f} | {ic:.4f} | {cum} | {sharpe:.4f} | {diff} | {note} |".format(
                iteration=row["iteration"],
                languages=row["languages"],
                clean_articles=int(row["clean_articles"]),
                supervised_weeks=int(row["supervised_weeks"]),
                num_weeks=row["num_weeks"],
                start_week=row["start_week"],
                end_week=row["end_week"],
                auc=row["auc"],
                accuracy=row["accuracy"],
                ic=row["ic"],
                cum=pct(float(row["cumulative_return"])),
                sharpe=float(row["sharpe_ratio"]),
                diff=pct(float(row["cumulative_return_diff"])),
                note=note,
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
