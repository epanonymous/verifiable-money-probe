"""Pre-registered H1 and low-base-rate evaluation rules from issue #4."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import roc_auc_score


FIXED_SCORE_THRESHOLD = 0.5

# Issue #4 has two locked readings. The earlier lock specifies a concrete
# 10 REAL / 1,000 SHAM set. Canonical v1 later specifies 1 SHAM / 50 REAL and
# the inverse. Keep all three rather than resolving the conflict post hoc.
BASE_RATE_SCENARIOS = (
    {
        "name": "earlier_lock_positive_1_to_100_negative",
        "rare_orientation": "positive",
        "rare_count": 10,
        "common_count": 1000,
        "source": "issue_4_lock_2026-09-01T01:39:32Z",
    },
    {
        "name": "canonical_v1_positive_1_to_50_negative",
        "rare_orientation": "positive",
        "rare_count": 1,
        "common_count": 50,
        "source": "issue_4_design_lock_v1_2026-09-01T01:43:21Z",
    },
    {
        "name": "canonical_v1_negative_1_to_50_positive",
        "rare_orientation": "negative",
        "rare_count": 1,
        "common_count": 50,
        "source": "issue_4_design_lock_v1_2026-09-01T01:43:21Z",
    },
)


def _as_binary(values: Iterable[int]) -> np.ndarray:
    y = np.asarray(list(values), dtype=np.int64)
    if y.ndim != 1 or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("labels must be a one-dimensional binary array")
    if len(np.unique(y)) != 2:
        raise ValueError("both binary classes are required")
    return y


def h1_verdict(auc: float) -> str:
    """Apply canonical v1 boundaries exactly: >=.85 / <.65 / between."""

    if auc >= 0.85:
        return "supported"
    if auc < 0.65:
        return "not_supported"
    return "inconclusive"


def h1_verdict_earlier_lock(auc: float) -> str:
    """Apply the earlier lock's supported / partial / null terminology."""

    if auc >= 0.85:
        return "supported"
    if auc >= 0.65:
        return "partial"
    return "null"


def binary_metrics(
    labels: Iterable[int], scores: Iterable[float], *, threshold: float = FIXED_SCORE_THRESHOLD
) -> dict[str, float | int | str | None]:
    y = _as_binary(labels)
    score = np.asarray(list(scores), dtype=np.float64)
    if score.shape != y.shape or not np.isfinite(score).all():
        raise ValueError("scores must be finite and have the same shape as labels")
    predicted = score >= threshold
    positive = y == 1
    negative = ~positive
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & negative))
    tn = int(np.sum(~predicted & negative))
    fn = int(np.sum(~predicted & positive))
    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)
    precision = tp / (tp + fp) if tp + fp else None
    auc = float(roc_auc_score(y, score))
    return {
        "auroc": auc,
        "h1_verdict": h1_verdict(auc),
        "h1_verdict_earlier_lock": h1_verdict_earlier_lock(auc),
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fpr,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
    }


def projected_low_base_rate_metrics(
    labels: Iterable[int],
    scores: Iterable[float],
    *,
    positive_condition: str,
    negative_condition: str,
    threshold: float = FIXED_SCORE_THRESHOLD,
) -> list[dict[str, float | int | str | None]]:
    """Project fixed-threshold TPR/FPR to every pre-registered prevalence.

    This uses the measured held-out rates instead of resampling the test set, so
    all scenarios use the same examples and no favorable mixture is selected.
    """

    y = _as_binary(labels)
    score = np.asarray(list(scores), dtype=np.float64)
    results = []
    for scenario in BASE_RATE_SCENARIOS:
        if scenario["rare_orientation"] == "positive":
            oriented_y = y
            oriented_score = score
            rare_condition = positive_condition
            common_condition = negative_condition
        else:
            oriented_y = 1 - y
            oriented_score = 1.0 - score
            rare_condition = negative_condition
            common_condition = positive_condition
        measured = binary_metrics(oriented_y, oriented_score, threshold=threshold)
        prevalence = scenario["rare_count"] / (
            scenario["rare_count"] + scenario["common_count"]
        )
        tpr = float(measured["recall"])
        fpr = float(measured["false_positive_rate"])
        denominator = prevalence * tpr + (1 - prevalence) * fpr
        projected_precision = prevalence * tpr / denominator if denominator else None
        results.append(
            {
                **scenario,
                "rare_condition": rare_condition,
                "common_condition": common_condition,
                "rare_prevalence": prevalence,
                "threshold": float(threshold),
                "measured_recall": tpr,
                "measured_false_positive_rate": fpr,
                "projected_precision": projected_precision,
            }
        )
    return results
