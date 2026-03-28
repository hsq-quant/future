from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def ensure_parent(path: str | Path) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    output = ensure_parent(path)
    temp_output = output.with_name(f".{output.name}.tmp")
    if output.suffix == ".csv":
        df.to_csv(temp_output, index=False)
    else:
        df.to_parquet(temp_output, index=False)
    os.replace(temp_output, output)
    return output


def read_dataframe(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix == ".csv":
        return pd.read_csv(source)
    try:
        return pd.read_parquet(source)
    except Exception:
        return pd.read_parquet(source, engine="fastparquet")
