from __future__ import annotations

import numpy as np
import pandas as pd


def make_expanding_window_splits(data: pd.DataFrame, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    if len(data) < n_splits + 1:
        raise ValueError("Need at least n_splits + 1 observations for expanding-window validation.")

    ordered = data.sort_values("week_end_date").reset_index(drop=True)
    index_blocks = np.array_split(np.arange(len(ordered)), n_splits + 1)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_id in range(1, n_splits + 1):
        train_idx = np.concatenate(index_blocks[:fold_id])
        valid_idx = index_blocks[fold_id]
        if len(train_idx) == 0 or len(valid_idx) == 0:
            continue
        splits.append((train_idx, valid_idx))
    return splits
