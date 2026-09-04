"""Regenerate the headline SVG from committed Run v1 JSON.

The generator reads only canonical, committed result artifacts. It does not fit a
model, edit the sealed Wave 2 bundle, or depend on plotting libraries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROBE = ROOT / (
    "experiments/exp4_analysis/results/wave2/positions/prompt_final/position_results.json"
)
DEFAULT_BASELINE = ROOT / (
    "experiments/exp4_analysis/results/wave4/full_information_text_baseline.json"
)
DEFAULT_OUTPUT = ROOT / "docs/writeup/assets/wave4-headline-results.svg"

RULE_NAMES = ("settled_true", "usdc_balance_1_9", "tx_confirmed_true")
DISPLAY_NAMES = ("Probe", "settled", "balance", "confirmed")


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _accuracy(metrics: dict[str, Any]) -> float:
    counts = (
        metrics["true_positives"],
        metrics["false_negatives"],
        metrics["true_negatives"],
        metrics["false_positives"],
    )
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in counts
    ):
        raise ValueError("confusion counts must be nonnegative integers")
    samples = sum(counts)
    if samples <= 0:
        raise ValueError("confusion matrix must be nonempty")
    calculated = (counts[0] + counts[2]) / samples
    reported = metrics.get("accuracy")
    if reported is not None and abs(float(reported) - calculated) > 1e-12:
        raise ValueError("reported accuracy disagrees with confusion counts")
    return calculated


def _sample_count(metrics: dict[str, Any]) -> int:
    return sum(
        int(metrics[name])
        for name in (
            "true_positives",
            "false_negatives",
            "true_negatives",
            "false_positives",
        )
    )


def _rule_accuracies(evaluation: dict[str, Any]) -> tuple[list[float], int]:
    rules = evaluation.get("rules")
    if (
        not isinstance(rules, list)
        or not all(isinstance(rule, dict) for rule in rules)
        or [rule.get("name") for rule in rules] != list(RULE_NAMES)
    ):
        raise ValueError("text-baseline rule inventory/order changed")
    metrics = [rule["metrics"] for rule in rules]
    samples = {int(block["samples"]) for block in metrics}
    if len(samples) != 1:
        raise ValueError("text-baseline rules use inconsistent sample counts")
    declared_samples = samples.pop()
    if any(_sample_count(block) != declared_samples for block in metrics):
        raise ValueError("text-baseline sample count disagrees with confusion counts")
    return [_accuracy(block) for block in metrics], declared_samples


def extract_values(
    probe_path: Path = DEFAULT_PROBE,
    baseline_path: Path = DEFAULT_BASELINE,
) -> dict[str, Any]:
    probe = _load_object(probe_path)
    baseline = _load_object(baseline_path)
    primary = probe["primary"]
    if (primary.get("positive_class"), primary.get("positive_label")) != ("REAL", 1):
        raise ValueError("probe positive-class contract changed")
    if (primary.get("negative_class"), primary.get("negative_label")) != ("SHAM", 0):
        raise ValueError("probe negative-class contract changed")
    if baseline.get("class_contract") != {
        "positive": "REAL",
        "positive_label": 1,
        "negative": "SHAM",
        "negative_label": 0,
    }:
        raise ValueError("text-baseline class contract changed")

    heldout_probe = primary["selected_layer_heldout_test"]
    # `independent_lbr` is a sealed key name, not a claim: the cache is
    # separately generated but same-generator. See docs/errata.md, E1.
    lbr_probe = primary["independent_lbr"]["empirical"]
    heldout_probe_n = int(primary["split"]["heldout_test_samples"])
    lbr_counts = primary["independent_lbr"]["actual_counts"]
    lbr_probe_n = int(lbr_counts["REAL"]) + int(lbr_counts["SHAM"])
    if heldout_probe_n != _sample_count(heldout_probe):
        raise ValueError("held-out probe sample count disagrees with confusion counts")
    if lbr_probe_n != _sample_count(lbr_probe):
        raise ValueError("LBR probe sample count disagrees with confusion counts")

    evaluations = baseline["evaluations"]
    heldout_rules, heldout_rule_n = _rule_accuracies(
        evaluations["main_heldout_b_plus_c_unique_prompts"]
    )
    lbr_rules, lbr_rule_n = _rule_accuracies(
        evaluations["independent_low_base_rate_cache"]
    )
    if lbr_rule_n != lbr_probe_n:
        raise ValueError("probe and text rules use different LBR sample counts")

    return {
        "heldout": [_accuracy(heldout_probe), *heldout_rules],
        "lbr": [_accuracy(lbr_probe), *lbr_rules],
        "heldout_probe_n": heldout_probe_n,
        "heldout_rule_n": heldout_rule_n,
        "lbr_n": lbr_probe_n,
        "lbr_real": int(lbr_counts["REAL"]),
        "lbr_sham": int(lbr_counts["SHAM"]),
    }


def _coord(value: float) -> str:
    rounded = round(value)
    return (
        str(rounded)
        if abs(value - rounded) < 1e-9
        else f"{value:.3f}".rstrip("0").rstrip(".")
    )


def _bars(values: list[float], x_positions: tuple[int, ...]) -> str:
    if len(values) != len(x_positions):
        raise ValueError("figure values and x positions must align")
    if any(value < 0 or value > 1 for value in values):
        raise ValueError("figure accuracies must be in [0, 1]")
    lines: list[str] = []
    for index, (value, x) in enumerate(zip(values, x_positions)):
        y = 285 - 200 * value
        height = 200 * value
        color = "#2563eb" if index == 0 else "#0f9d8a"
        lines.append(
            f'  <rect x="{x}" y="{_coord(y)}" width="48" height="{_coord(height)}" rx="3" fill="{color}"/>'
        )
    for value, x in zip(values, x_positions):
        lines.append(
            f'  <text x="{x + 24}" y="{_coord(277 - 200 * value)}" text-anchor="middle" class="value">{value:.3f}</text>'
        )
    for label, x in zip(DISPLAY_NAMES, x_positions):
        lines.append(
            f'  <text x="{x + 24}" y="304" text-anchor="middle" class="label">{label}</text>'
        )
    return "\n".join(lines)


def render_svg(values: dict[str, Any]) -> str:
    if any(
        value != 1.0 for panel in (values["heldout"], values["lbr"]) for value in panel
    ):
        raise ValueError(
            "headline ceiling claim requires every plotted accuracy to equal 1.0"
        )
    heldout_bars = _bars(values["heldout"], (75, 145, 215, 285))
    lbr_bars = _bars(values["lbr"], (425, 495, 565, 635))
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="760" height="380" viewBox="0 0 760 380" role="img" aria-labelledby="title desc">
  <title id="title">Full-information text rules match the activation probe at ceiling</title>
  <desc id="desc">Two bar charts show fixed-threshold accuracy of 1.000 for the activation probe and three literal full-prompt rules. The held-out main evaluation used {values["heldout_probe_n"]:,} probe rollout rows and {values["heldout_rule_n"]} unique prompt groups for each text rule. The separately generated, same-generator low-base-rate evaluation used the same {values["lbr_n"]:,} rows for every detector.</desc>
  <rect width="760" height="380" fill="#ffffff"/>
  <style>
    text {{ font-family: Inter, Arial, sans-serif; fill: #172033; }}
    .title {{ font-size: 20px; font-weight: 700; }}
    .panel-title {{ font-size: 15px; font-weight: 700; }}
    .axis {{ stroke: #526071; stroke-width: 1.2; }}
    .grid {{ stroke: #d8dee8; stroke-width: 1; }}
    .tick {{ font-size: 11px; fill: #526071; }}
    .label {{ font-size: 11px; font-weight: 600; }}
    .value {{ font-size: 12px; font-weight: 700; }}
    .note {{ font-size: 11px; fill: #526071; }}
  </style>

  <text x="380" y="30" text-anchor="middle" class="title">Literal full-prompt rules match the probe at ceiling</text>

  <text x="205" y="60" text-anchor="middle" class="panel-title">Held-out main accuracy</text>
  <line x1="50" y1="285" x2="360" y2="285" class="axis"/>
  <line x1="50" y1="85" x2="50" y2="285" class="axis"/>
  <line x1="50" y1="185" x2="360" y2="185" class="grid"/>
  <line x1="50" y1="85" x2="360" y2="85" class="grid"/>
  <text x="42" y="289" text-anchor="end" class="tick">0</text>
  <text x="42" y="189" text-anchor="end" class="tick">0.50</text>
  <text x="42" y="89" text-anchor="end" class="tick">1.00</text>
{heldout_bars}
  <text x="205" y="327" text-anchor="middle" class="note">Probe: {values["heldout_probe_n"]:,} rows · each rule: {values["heldout_rule_n"]} unique prompts</text>

  <text x="555" y="60" text-anchor="middle" class="panel-title">Low-base-rate accuracy</text>
  <line x1="400" y1="285" x2="710" y2="285" class="axis"/>
  <line x1="400" y1="85" x2="400" y2="285" class="axis"/>
  <line x1="400" y1="185" x2="710" y2="185" class="grid"/>
  <line x1="400" y1="85" x2="710" y2="85" class="grid"/>
  <text x="392" y="289" text-anchor="end" class="tick">0</text>
  <text x="392" y="189" text-anchor="end" class="tick">0.50</text>
  <text x="392" y="89" text-anchor="end" class="tick">1.00</text>
{lbr_bars}
  <text x="555" y="327" text-anchor="middle" class="note">Same {values["lbr_n"]:,} rows · {values["lbr_real"]:,} REAL / {values["lbr_sham"]:,} SHAM</text>

  <rect x="255" y="350" width="12" height="12" rx="2" fill="#2563eb"/>
  <text x="273" y="360" class="note">Residual-stream probe</text>
  <rect x="425" y="350" width="12" height="12" rx="2" fill="#0f9d8a"/>
  <text x="443" y="360" class="note">One-field text rules</text>
</svg>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = render_svg(extract_values(args.probe, args.baseline))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
