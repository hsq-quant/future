from __future__ import annotations

import pandas as pd

from src.utils.io import write_dataframe


def test_write_dataframe_overwrites_atomically(tmp_path) -> None:
    path = tmp_path / "sample.parquet"
    first = pd.DataFrame({"value": [1, 2]})
    second = pd.DataFrame({"value": [3]})

    write_dataframe(first, path)
    write_dataframe(second, path)

    loaded = pd.read_parquet(path)
    assert loaded.to_dict(orient="list") == {"value": [3]}
    assert not (tmp_path / ".sample.parquet.tmp").exists()
