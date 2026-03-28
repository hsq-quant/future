from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.reporting.model_family_summary import ExperimentSpec, build_model_family_summary, render_model_family_markdown
from src.reporting.report_workbook import build_report_workbook


def _build_specs(root: Path) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    for family, label, language_scope, cv_path, variant_path in [
        (
            "V1",
            "V1 Chinese classification",
            "zh",
            root / "reports/iterations/2026-03-27-single-source-baseline/cv_metrics.csv",
            root / "reports/iterations/2026-03-28-v1-classification-mapping-comparison/variant_metrics.csv",
        ),
        (
            "V3",
            "V3 English classification",
            "en",
            root / "reports/iterations/2026-03-28-english-v3-classification-sample20/cv_metrics.csv",
            root / "reports/iterations/2026-03-28-v3-classification-mapping-comparison/variant_metrics.csv",
        ),
    ]:
        for execution in ["always_in", "long_only", "threshold_short_only"]:
            specs.append(
                ExperimentSpec(
                    label=label,
                    family=family,
                    language_scope=language_scope,
                    task="classification",
                    execution=execution,
                    cv_metrics_path=cv_path,
                    strategy_metrics_path=variant_path,
                    strategy_variant=execution,
                )
            )

    specs.extend(
        [
        ExperimentSpec(
            label="V1 Chinese regression",
            family="V1",
            language_scope="zh",
            task="regression",
            execution="baseline_tanh",
            cv_metrics_path=root / "reports/iterations/2026-03-28-v1-regression/cv_metrics.csv",
            strategy_metrics_path=root / "reports/iterations/2026-03-28-v1-regression/strategy_metrics.csv",
        ),
        ExperimentSpec(
            label="V3 English regression",
            family="V3",
            language_scope="en",
            task="regression",
            execution="baseline_tanh",
            cv_metrics_path=root / "reports/iterations/2026-03-28-english-v3-sample20/cv_metrics.csv",
            strategy_metrics_path=root / "reports/iterations/2026-03-28-english-v3-sample20/strategy_metrics.csv",
        ),
        ]
    )
    for family, label, path_slug in [
        ("V1", "V1 Chinese regression", "2026-03-28-v1-regression-mapping-comparison"),
        ("V3", "V3 English regression", "2026-03-28-v3-english-mapping-comparison"),
    ]:
        for execution in ["long_only_factor", "asymmetric_long_short", "threshold_short_only"]:
            specs.append(
                ExperimentSpec(
                    label=label,
                    family=family,
                    language_scope="zh" if family == "V1" else "en",
                    task="regression",
                    execution=execution,
                    cv_metrics_path=root / ("reports/iterations/2026-03-28-v1-regression/cv_metrics.csv" if family == "V1" else "reports/iterations/2026-03-28-english-v3-sample20/cv_metrics.csv"),
                    strategy_metrics_path=root / f"reports/iterations/{path_slug}/variant_metrics.csv",
                    strategy_variant=execution,
                )
            )
    return specs


def _render_notes(summary: pd.DataFrame) -> str:
    classification = summary.loc[summary["task"] == "classification"].copy()
    regression = summary.loc[summary["task"] == "regression"].copy()
    best_class = classification.sort_values("strategy_cumulative_return", ascending=False).iloc[0]
    best_reg = regression.sort_values("strategy_cumulative_return", ascending=False).iloc[0]
    lines = [
        "# INE SC Workbook Notes",
        "",
        "## Purpose",
        "- This workbook is the report draft only. It is the main delivery artifact, not a PPT source file.",
        "- The classification section is the intuitive trading story: predict up/down, then translate directly into a weekly trading action. It now includes multiple execution mappings, not only the naive always-in long/short switch.",
        "- The regression section is the factor story: predict next-week return, then map the signal into a smoother position for practical quant usage.",
        "",
        "## Sheets",
        "- `Summary`: all model-task-execution combinations in one comparison table.",
        "- `Classification`: up/down model results plus fairer execution variants such as long-only classification.",
        "- `Regression`: all regression training results, including factor IC and directional accuracy.",
        "- `Mappings`: the regression execution variants that test whether the factor or the trading layer is the bottleneck.",
        "",
        "## Current Readout",
        f"- Best classification strategy in the current table: `{best_class['label']} / {best_class['execution']}` with cumulative return `{best_class['strategy_cumulative_return']:.2%}` and Sharpe `{best_class['strategy_sharpe_ratio']:.4f}`.",
        f"- Best regression execution in the current table: `{best_reg['label']} / {best_reg['execution']}` with cumulative return `{best_reg['strategy_cumulative_return']:.2%}` and Sharpe `{best_reg['strategy_sharpe_ratio']:.4f}`.",
        "- The main narrative is to compare classification vs regression, then show that regression can be useful as a factor even when naive execution underperforms a strong always-long crude benchmark.",
        "",
        "## Workbook Usage",
        "- Use `Summary` for the one-page total comparison.",
        "- Use `Classification` when explaining the simple, intuitive trading logic.",
        "- Use `Regression` and `Mappings` when explaining IC, factor ranking ability, and why execution design matters.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    output_dir = ROOT / "reports/final"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_model_family_summary(_build_specs(ROOT))
    summary = summary.sort_values(["task", "family", "execution"]).reset_index(drop=True)
    classification = summary.loc[summary["task"] == "classification"].reset_index(drop=True)
    regression = summary.loc[summary["task"] == "regression"].reset_index(drop=True)
    mappings = regression.loc[regression["execution"] != "baseline_tanh"].reset_index(drop=True)

    csv_path = output_dir / "model_family_comparison.csv"
    md_path = output_dir / "model_family_comparison.md"
    workbook_path = output_dir / "ine_sc_report_draft.xlsx"
    notes_path = output_dir / "ine_sc_report_notes.md"

    summary.to_csv(csv_path, index=False)
    md_path.write_text(render_model_family_markdown(summary), encoding="utf-8")
    build_report_workbook(
        workbook_path,
        summary=summary,
        classification=classification,
        regression=regression,
        mappings=mappings,
    )
    notes_path.write_text(_render_notes(summary), encoding="utf-8")

    print(csv_path)
    print(md_path)
    print(workbook_path)
    print(notes_path)


if __name__ == "__main__":
    main()
