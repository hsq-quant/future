from __future__ import annotations

import math

import numpy as np
import pandas as pd


def apply_class_share_threshold(
    validation_probs: pd.Series,
    training_positive_share: float,
) -> tuple[float, pd.Series]:
    """Choose a threshold so predicted positives roughly match the training class share."""
    probs = pd.Series(validation_probs, dtype=float).reset_index(drop=True)

    if probs.empty:
        return math.inf, pd.Series(dtype="Int64")

    positive_count = int(round(len(probs) * training_positive_share))
    positive_count = max(0, min(len(probs), positive_count))

    if positive_count == 0:
        threshold = math.inf
        labels = pd.Series(np.zeros(len(probs), dtype=int))
        return threshold, labels

    if positive_count == len(probs):
        threshold = float("-inf")
        labels = pd.Series(np.ones(len(probs), dtype=int))
        return threshold, labels

    threshold = float(probs.nlargest(positive_count).iloc[-1])
    labels = (probs >= threshold).astype(int)
    return threshold, labels
