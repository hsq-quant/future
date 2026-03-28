from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.build_aligned_model_family_comparison import main as build_aligned_model_family_comparison_main
from scripts.build_aligned_regime_comparison import main as build_aligned_regime_comparison_main
from src.reporting.presentation_draft import build_presentation_workbook, report_notes_text
from src.reporting.report_workbook import build_report_workbook


def main() -> None:
    output_dir = ROOT / "reports/final"
    output_dir.mkdir(parents=True, exist_ok=True)

    build_aligned_model_family_comparison_main()
    build_aligned_regime_comparison_main()

    summary = pd.read_csv(output_dir / "model_family_comparison_aligned.csv")
    regime = pd.read_csv(output_dir / "regime_comparison_aligned.csv")
    notes = (output_dir / "ine_sc_report_notes_aligned.md").read_text(encoding="utf-8")

    build_report_workbook(
        output_dir / "ine_sc_report_draft.xlsx",
        summary=summary,
        classification=summary.loc[summary["task"] == "classification"].reset_index(drop=True),
        regression=summary.loc[summary["task"] == "regression"].reset_index(drop=True),
        mappings=summary.loc[
            ((summary["task"] == "regression") & (summary["execution"] != "baseline_tanh"))
            | ((summary["task"] == "classification") & (summary["execution"] != "always_in"))
        ].reset_index(drop=True),
        regime=regime.reset_index(drop=True),
        notes=notes,
    )
    build_presentation_workbook(output_dir / "ine_sc_report_draft_aligned.xlsx")
    (output_dir / "ine_sc_report_notes_aligned.md").write_text(report_notes_text(), encoding="utf-8")

    print(output_dir / "model_family_comparison_aligned.csv")
    print(output_dir / "regime_comparison_aligned.csv")
    print(output_dir / "ine_sc_report_draft_aligned.xlsx")
    print(output_dir / "ine_sc_report_notes_aligned.md")


if __name__ == "__main__":
    main()
