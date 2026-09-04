from __future__ import annotations

import pytest

from experiments.exp4_probes.metrics import (
    h1_verdict,
    h1_verdict_earlier_lock,
    projected_low_base_rate_metrics,
)


def test_h1_thresholds_are_exact() -> None:
    assert h1_verdict(0.85) == "supported"
    assert h1_verdict(0.849999) == "inconclusive"
    assert h1_verdict(0.65) == "inconclusive"
    assert h1_verdict(0.649999) == "not_supported"
    assert h1_verdict_earlier_lock(0.85) == "supported"
    assert h1_verdict_earlier_lock(0.65) == "partial"
    assert h1_verdict_earlier_lock(0.649999) == "null"


def test_all_locked_base_rate_readings_are_emitted() -> None:
    results = projected_low_base_rate_metrics(
        [0, 0, 1, 1],
        [0.1, 0.6, 0.7, 0.9],
        positive_condition="verified",
        negative_condition="claimed",
    )
    assert [row["rare_prevalence"] for row in results] == pytest.approx(
        [10 / 1010, 1 / 51, 1 / 51]
    )
    assert [row["rare_condition"] for row in results] == [
        "verified",
        "verified",
        "claimed",
    ]
