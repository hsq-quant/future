from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_report_workbook(
    output_path: Path,
    *,
    summary: pd.DataFrame,
    classification: pd.DataFrame,
    regression: pd.DataFrame,
    mappings: pd.DataFrame,
    regime: pd.DataFrame | None = None,
    notes: str | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        classification.to_excel(writer, sheet_name="Classification", index=False)
        regression.to_excel(writer, sheet_name="Regression", index=False)
        mappings.to_excel(writer, sheet_name="Mappings", index=False)
        if regime is not None:
            regime.to_excel(writer, sheet_name="Regimes", index=False)
        if notes:
            notes_df = pd.DataFrame({"notes": notes.splitlines()})
            notes_df.to_excel(writer, sheet_name="Notes", index=False)
    return output_path
