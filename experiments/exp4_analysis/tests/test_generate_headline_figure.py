from __future__ import annotations

from pathlib import Path
import tomllib

import pytest

from experiments.exp4_analysis.generate_headline_figure import (
    DEFAULT_BASELINE,
    DEFAULT_OUTPUT,
    DEFAULT_PROBE,
    _accuracy,
    extract_values,
    render_svg,
)

ROOT = Path(__file__).resolve().parents[3]


def test_headline_figure_regenerates_byte_identically() -> None:
    values = extract_values(DEFAULT_PROBE, DEFAULT_BASELINE)
    assert render_svg(values).encode() == DEFAULT_OUTPUT.read_bytes()


def test_headline_figure_values_are_evidence_linked() -> None:
    values = extract_values(DEFAULT_PROBE, DEFAULT_BASELINE)
    assert values == {
        "heldout": [1.0, 1.0, 1.0, 1.0],
        "lbr": [1.0, 1.0, 1.0, 1.0],
        "heldout_probe_n": 1000,
        "heldout_rule_n": 40,
        "lbr_n": 1010,
        "lbr_real": 10,
        "lbr_sham": 1000,
    }


def test_accuracy_rejects_inconsistent_reported_value() -> None:
    with pytest.raises(ValueError, match="reported accuracy disagrees"):
        _accuracy(
            {
                "true_positives": 1,
                "false_negatives": 0,
                "true_negatives": 1,
                "false_positives": 0,
                "accuracy": 0.5,
            }
        )


def test_headline_figure_rejects_non_ceiling_claim() -> None:
    values = extract_values(DEFAULT_PROBE, DEFAULT_BASELINE)
    values["lbr"][0] = 0.99
    with pytest.raises(ValueError, match="ceiling claim"):
        render_svg(values)


def test_requirements_fallback_matches_project_dependencies() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = {
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert requirements == set(project["project"]["dependencies"])
