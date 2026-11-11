"""Shared non-interpolated AP and F1 for training and final analysis."""
import numpy as np


def average_precision(y_true, y_score):
    y = np.asarray(y_true)
    scores = np.asarray(y_score)
    if y.ndim != 1 or scores.shape != y.shape or not len(y):
        raise ValueError("AP requires nonempty, matching one-dimensional arrays")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("AP requires binary labels and finite scores")
    positives = y.sum()
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    # Each threshold includes the entire tied-score group, including duplicates
    # introduced by bootstrap resampling.
    ends = np.r_[np.flatnonzero(np.diff(scores[order])), len(y) - 1]
    tp = np.cumsum(y[order], dtype=np.float64)[ends]
    precision = tp / (ends + 1)
    return float(np.sum(np.diff(np.r_[0, tp]) * precision) / positives)


def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    denominator = 2 * tp + fp + fn
    f1 = 2 * tp / denominator if denominator else 0.0
    return precision, recall, f1
