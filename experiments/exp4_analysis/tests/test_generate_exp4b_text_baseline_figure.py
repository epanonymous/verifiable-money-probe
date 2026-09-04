from __future__ import annotations

from experiments.exp4_analysis.generate_exp4b_text_baseline_figure import (
    DEFAULT_LEAK_FREE,
    DEFAULT_ORIGINAL,
    DEFAULT_OUTPUT,
    _binary_prediction_auroc,
    extract_values,
    render_svg,
)


def test_exp4b_figure_regenerates_byte_identically() -> None:
    values = extract_values(DEFAULT_ORIGINAL, DEFAULT_LEAK_FREE)
    assert render_svg(values).encode() == DEFAULT_OUTPUT.read_bytes()


def test_exp4b_figure_values_are_evidence_linked() -> None:
    assert extract_values(DEFAULT_ORIGINAL, DEFAULT_LEAK_FREE) == {
        "original": [1.0, 1.0, 1.0],
        "leak_free": [0.5, 0.5, 0.5],
        "labels": ["Non-held-out", "Held-out", "Low base rate"],
        "samples": [152, 40, 1010],
    }


def test_binary_auroc_exposes_majority_class_accuracy_artifact() -> None:
    metrics = {
        "true_positives": 0,
        "false_negatives": 10,
        "true_negatives": 1000,
        "false_positives": 0,
        "auroc": 0.5,
    }
    assert _binary_prediction_auroc(metrics) == 0.5
