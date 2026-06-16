from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def amex_metric(y_true, y_pred) -> float:
    """Kaggle AMEX competition metric.

    The metric is the average of normalized weighted Gini and the default rate
    captured in the top 4 percent of weighted observations.
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    order = np.argsort(-y_pred)
    labels = y_true[order]

    weights = np.where(labels == 0, 20.0, 1.0)
    cutoff = int(0.04 * weights.sum())
    top_mask = np.cumsum(weights) <= cutoff
    top_four = labels[top_mask].sum() / labels.sum()

    def weighted_gini(sort_by):
        sort_order = np.argsort(-sort_by)
        sorted_labels = y_true[sort_order]
        sorted_weights = np.where(sorted_labels == 0, 20.0, 1.0)
        weighted_random = np.cumsum(sorted_weights / sorted_weights.sum())
        total_positive = (sorted_labels * sorted_weights).sum()
        if total_positive == 0:
            return 0.0
        lorentz = np.cumsum(sorted_labels * sorted_weights) / total_positive
        return np.sum((lorentz - weighted_random) * sorted_weights)

    perfect_gini = weighted_gini(y_true)
    model_gini = weighted_gini(y_pred)
    normalized_gini = model_gini / perfect_gini if perfect_gini else 0.0
    return 0.5 * (normalized_gini + top_four)


def binary_auc(y_true, y_pred) -> float:
    return float(roc_auc_score(y_true, y_pred))


def score_binary_predictions(y_true, y_pred, metric: str) -> float:
    metric = metric.lower()
    if metric == "amex":
        return float(amex_metric(y_true, y_pred))
    if metric == "auc":
        return binary_auc(y_true, y_pred)
    raise ValueError(f"Unsupported metric: {metric}")
