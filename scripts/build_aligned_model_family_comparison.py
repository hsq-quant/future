from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.reporting.aligned_comparison import summarize_aligned_strategy_window
from src.reporting.model_family_summary import render_model_family_markdown
from src.reporting.report_workbook import build_report_workbook


START_DATE = "2021-01-22"
END_DATE = "2025-09-05"


def _classification_rows() -> list[dict[str, object]]:
    base = [
        (
            "V1 Chinese classification",
            "V1",
            "zh",
            "/Users/hsq/Desktop/codex/future/reports/iterations/2026-03-27-single-source-baseline/cv_metrics.csv",
            "/Users/hsq/Desktop/codex/future/reports/iterations/2026-03-28-v1-classification-mapping-comparison/variant_weekly_positions.csv",
        ),
        (
            "V3 English classification",
            "V3",
            "en",
            "/Users/hsq/Desktop/codex/future/reports/iterations/2026-03-28-english-v3-classification-sample20/cv_metrics.csv",
            "/Users/hsq/Desktop/codex/future/reports/iterations/2026-03-28-v3-classification-mapping-comparison/variant_weekly_positions.csv",
        ),
    ]
    rows: list[dict[str, object]] = []
    for label, family, lang, cv_path, weekly_path in base:
        cv = pd.read_csv(cv_path)
        weekly = pd.read_csv(weekly_path)
        for execution in ["always_in", "long_only", "threshold_short_only"]:
            window = weekly.loc[weekly["variant"] == execution].reset_index(drop=True)
            tmp = ROOT / "tmp" / f"{family.lower()}_{execution}_aligned_source.csv"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            window.to_csv(tmp, index=False)
            strat = summarize_aligned_strategy_window(tmp, start_date=START_DATE, end_date=END_DATE)
            rows.append(
                {
                    "label": label,
                    "family": family,
                    "language_scope": lang,
                    "task": "classification",
                    "execution": execution,
                    "window_start": START_DATE,
                    "window_end": END_DATE,
                    "mean_auc": float(cv["auc"].mean()),
                    "mean_accuracy": float(cv["accuracy"].mean()),
                    "mean_ic": float(cv["ic"].mean()),
                    "mean_directional_accuracy": float(cv["accuracy"].mean()),
                    "mean_mse": 0.0,
                    "mean_mae": 0.0,
                    **strat,
                }
            )
    return rows


def _regression_rows() -> list[dict[str, object]]:
    base = [
        (
            "V1 Chinese regression",
            "V1",
            "zh",
            "/Users/hsq/Desktop/codex/future/reports/iterations/2026-03-28-v1-regression/cv_metrics.csv",
            "/Users/hsq/Desktop/codex/future/reports/iterations/2026-03-28-v1-regression-mapping-comparison/variant_weekly_positions.csv",
        ),
        (
            "V3 English regression",
            "V3",
            "en",
            "/Users/hsq/Desktop/codex/future/reports/iterations/2026-03-28-english-v3-sample20/cv_metrics.csv",
            "/Users/hsq/Desktop/codex/future/reports/iterations/2026-03-28-v3-english-mapping-comparison/variant_weekly_positions.csv",
        ),
    ]
    rows: list[dict[str, object]] = []
    for label, family, lang, cv_path, weekly_path in base:
        cv = pd.read_csv(cv_path)
        weekly = pd.read_csv(weekly_path)
        for execution in ["baseline_tanh", "long_only_factor", "asymmetric_long_short", "threshold_short_only"]:
            window = weekly.loc[weekly["variant"] == execution].reset_index(drop=True)
            tmp = ROOT / "tmp" / f"{family.lower()}_{execution}_aligned_source.csv"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            window.to_csv(tmp, index=False)
            strat = summarize_aligned_strategy_window(tmp, start_date=START_DATE, end_date=END_DATE)
            rows.append(
                {
                    "label": label,
                    "family": family,
                    "language_scope": lang,
                    "task": "regression",
                    "execution": execution,
                    "window_start": START_DATE,
                    "window_end": END_DATE,
                    "mean_auc": 0.0,
                    "mean_accuracy": 0.0,
                    "mean_ic": float(cv["ic"].mean()),
                    "mean_directional_accuracy": float(cv["directional_accuracy"].mean()),
                    "mean_mse": float(cv["mse"].mean()),
                    "mean_mae": float(cv["mae"].mean()),
                    **strat,
                }
            )
    return rows


def _render_notes(summary: pd.DataFrame) -> str:
    classification = summary.loc[summary["task"] == "classification"].copy()
    regression = summary.loc[summary["task"] == "regression"].copy()

    best_cls_exec = classification.sort_values(
        ["cumulative_return_diff", "strategy_sharpe_ratio"], ascending=False
    ).iloc[0]
    best_reg_exec = regression.sort_values(
        ["mean_ic", "strategy_sharpe_ratio"], ascending=False
    ).iloc[0]
    best_reg_trading = regression.sort_values(
        ["strategy_cumulative_return", "strategy_sharpe_ratio"], ascending=False
    ).iloc[0]

    return "\n".join(
        [
            "# INE SC Aligned Comparison Notes",
            "",
            f"- Common trading window: `{START_DATE}` to `{END_DATE}`.",
            "- This aligned table removes all non-overlapping post-2025-09-05 performance so V1 and V3 can be compared fairly.",
            "- Factor metrics are unchanged; only trading metrics are re-summarized on the common window.",
            "",
            "## How to present the results",
            "",
            "- Classification is the intuitive story: predict up or down, then map to a simple trading rule.",
            "- Regression is the factor story: predict next-week return, evaluate IC, then map the score to a smoother position.",
            "- For the class presentation, treat aligned results as the only authoritative version.",
            "",
            "## Main takeaways",
            "",
            f"- Best classification execution in the aligned window: `{best_cls_exec['label']} / {best_cls_exec['execution']}` with cumulative return `{best_cls_exec['strategy_cumulative_return']:.2%}` and Sharpe `{best_cls_exec['strategy_sharpe_ratio']:.4f}`.",
            f"- Best regression factor by IC: `{best_reg_exec['label']} / {best_reg_exec['execution']}` with mean IC `{best_reg_exec['mean_ic']:.4f}` and Sharpe `{best_reg_exec['strategy_sharpe_ratio']:.4f}`.",
            f"- Best regression trading execution: `{best_reg_trading['label']} / {best_reg_trading['execution']}` with cumulative return `{best_reg_trading['strategy_cumulative_return']:.2%}`.",
            "- V3 English models are stronger at the model layer than V1, but the raw classification always-in execution is too aggressive on the short side.",
            "- V3 regression with long-only factor mapping is the most defensible 'quant factor extension' result.",
            "",
            "## Recommended narrative",
            "",
            "- Use `V1 classification / long_only` as the simple, intuitive strategy result.",
            "- Use `V3 regression / long_only_factor` as the factor-style extension result.",
            "- Use the aligned regime comparison to explain when each family works and when it fails.",
            "",
            "## Core output files",
            "",
            "- `reports/final/model_family_comparison_aligned.csv`",
            "- `reports/final/regime_comparison_aligned.csv`",
            "- `reports/final/ine_sc_report_draft_aligned.xlsx`",
            "- `reports/final/ine_sc_report_notes_aligned.md`",
            "",
        ]
    )


def main() -> None:
    output_dir = ROOT / "reports/final"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(_classification_rows() + _regression_rows())
    summary = summary.sort_values(["task", "family", "execution"]).reset_index(drop=True)

    csv_path = output_dir / "model_family_comparison_aligned.csv"
    md_path = output_dir / "model_family_comparison_aligned.md"
    workbook_path = output_dir / "ine_sc_report_draft_aligned.xlsx"
    notes_path = output_dir / "ine_sc_report_notes_aligned.md"

    summary.to_csv(csv_path, index=False)
    md_path.write_text(render_model_family_markdown(summary), encoding="utf-8")
    build_report_workbook(
        workbook_path,
        summary=summary,
        classification=summary.loc[summary["task"] == "classification"].reset_index(drop=True),
        regression=summary.loc[summary["task"] == "regression"].reset_index(drop=True),
        mappings=summary.loc[(summary["task"] == "regression") & (summary["execution"] != "baseline_tanh")].reset_index(drop=True),
        notes=_render_notes(summary),
    )
    notes_path.write_text(_render_notes(summary), encoding="utf-8")

    print(csv_path)
    print(md_path)
    print(workbook_path)
    print(notes_path)


if __name__ == "__main__":
    main()
