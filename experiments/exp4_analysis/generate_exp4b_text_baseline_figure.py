"""Regenerate the Exp 4b text-baseline SVG from committed result JSON.

The figure uses AUROC rather than raw accuracy because the independent
low-base-rate cache contains 10 REAL and 1,000 SHAM rows. A constant-SHAM rule
there has 99% accuracy but chance discrimination.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ORIGINAL = ROOT / (
    "experiments/exp4_analysis/results/wave4/full_information_text_baseline.json"
)
DEFAULT_LEAK_FREE = ROOT / (
    "experiments/exp4_analysis/results/exp4b/full_information_text_baseline.json"
)
DEFAULT_OUTPUT = ROOT / "docs/writeup/assets/exp4b-leak-free-text-baseline.svg"

RULE_NAMES = ("settled_true", "usdc_balance_1_9", "tx_confirmed_true")
EVALUATIONS = (
    ("main_train_b_plus_c_unique_prompts", "Non-held-out", 152),
    ("main_heldout_b_plus_c_unique_prompts", "Held-out", 40),
    ("independent_low_base_rate_cache", "Low base rate", 1010),
)
CLASS_CONTRACT = {
    "positive": "REAL",
    "positive_label": 1,
    "negative": "SHAM",
    "negative_label": 0,
}


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _binary_prediction_auroc(metrics: dict[str, Any]) -> float:
    counts = tuple(
        metrics[name]
        for name in (
            "true_positives",
            "false_negatives",
            "true_negatives",
            "false_positives",
        )
    )
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in counts
    ):
        raise ValueError("confusion counts must be nonnegative integers")
    tp, fn, tn, fp = counts
    if tp + fn == 0 or tn + fp == 0:
        raise ValueError("AUROC requires both REAL and SHAM examples")
    # For a binary score, AUROC is the mean of sensitivity and specificity.
    calculated = ((tp / (tp + fn)) + (tn / (tn + fp))) / 2
    reported = metrics.get("auroc")
    if reported is not None and abs(float(reported) - calculated) > 1e-12:
        raise ValueError("reported AUROC disagrees with confusion counts")
    return calculated


def _evaluation_auroc(
    report: dict[str, Any], evaluation_name: str, expected_samples: int
) -> float:
    evaluation = report["evaluations"][evaluation_name]
    rules = evaluation.get("rules")
    if (
        not isinstance(rules, list)
        or not all(isinstance(rule, dict) for rule in rules)
        or [rule.get("name") for rule in rules] != list(RULE_NAMES)
    ):
        raise ValueError("text-baseline rule inventory/order changed")
    metrics = [rule["metrics"] for rule in rules]
    if {block.get("samples") for block in metrics} != {expected_samples}:
        raise ValueError("text-baseline sample inventory changed")
    aurocs = {_binary_prediction_auroc(block) for block in metrics}
    if len(aurocs) != 1:
        raise ValueError("text-baseline rules no longer have identical AUROC")
    return aurocs.pop()


def extract_values(
    original_path: Path = DEFAULT_ORIGINAL,
    leak_free_path: Path = DEFAULT_LEAK_FREE,
) -> dict[str, Any]:
    original = _load_object(original_path)
    leak_free = _load_object(leak_free_path)
    if original.get("class_contract") != CLASS_CONTRACT:
        raise ValueError("original baseline class contract changed")
    if leak_free.get("class_contract") != CLASS_CONTRACT:
        raise ValueError("leak-free baseline class contract changed")
    if leak_free.get("leak_gate") != {
        "passed": True,
        "expected_auroc": 0.5,
        "observed_rule_aurocs": [0.5],
        "text_equivalence": {
            "main_REAL_SHAM_byte_identical_pairs": 144,
            "low_base_rate_shared_unique_prompts": 10,
            "low_base_rate_prompt_distribution_identical_by_class": True,
        },
    }:
        raise ValueError("leak-free acceptance gate changed")

    original_aurocs = []
    leak_free_aurocs = []
    for evaluation_name, _display_name, samples in EVALUATIONS:
        original_aurocs.append(_evaluation_auroc(original, evaluation_name, samples))
        leak_free_aurocs.append(_evaluation_auroc(leak_free, evaluation_name, samples))
    return {
        "original": original_aurocs,
        "leak_free": leak_free_aurocs,
        "labels": [display_name for _, display_name, _ in EVALUATIONS],
        "samples": [samples for _, _, samples in EVALUATIONS],
    }


def _coord(value: float) -> str:
    rounded = round(value)
    return (
        str(rounded)
        if abs(value - rounded) < 1e-9
        else f"{value:.3f}".rstrip("0").rstrip(".")
    )


def render_svg(values: dict[str, Any]) -> str:
    if values["original"] != [1.0, 1.0, 1.0]:
        raise ValueError("original full-information baseline is no longer at ceiling")
    if values["leak_free"] != [0.5, 0.5, 0.5]:
        raise ValueError("Exp 4b full-information baseline is no longer at chance")

    groups = (120, 330, 540)
    bars: list[str] = []
    for index, x in enumerate(groups):
        for value, offset, color in (
            (values["original"][index], 0, "#c2413b"),
            (values["leak_free"][index], 62, "#0f9d8a"),
        ):
            y = 285 - 200 * value
            height = 200 * value
            bars.append(
                f'  <rect x="{x + offset}" y="{_coord(y)}" width="48" '
                f'height="{_coord(height)}" rx="3" fill="{color}"/>'
            )
            bars.append(
                f'  <text x="{x + offset + 24}" y="{_coord(277 - 200 * value)}" '
                f'text-anchor="middle" class="value">{value:.3f}</text>'
            )
        bars.append(
            f'  <text x="{x + 55}" y="306" text-anchor="middle" class="label">'
            f"{values['labels'][index]}</text>"
        )
        bars.append(
            f'  <text x="{x + 55}" y="322" text-anchor="middle" class="note">'
            f"n={values['samples'][index]:,}</text>"
        )
    rendered_bars = "\n".join(bars)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="760" height="390" viewBox="0 0 760 390" role="img" aria-labelledby="title desc">
  <title id="title">Leak-free full-information text baseline falls to chance</title>
  <desc id="desc">Across non-held-out, held-out, and low-base-rate evaluations, three deterministic full-prompt rules have AUROC 1.000 on the original inputs and 0.500 on the leak-free Exp 4b inputs. Each displayed bar is shared by all three rules.</desc>
  <rect width="760" height="390" fill="#ffffff"/>
  <style>
    text {{ font-family: Inter, Arial, sans-serif; fill: #172033; }}
    .title {{ font-size: 20px; font-weight: 700; }}
    .subtitle {{ font-size: 12px; fill: #526071; }}
    .axis {{ stroke: #526071; stroke-width: 1.2; }}
    .grid {{ stroke: #d8dee8; stroke-width: 1; }}
    .tick {{ font-size: 11px; fill: #526071; }}
    .label {{ font-size: 11px; font-weight: 600; }}
    .value {{ font-size: 12px; font-weight: 700; }}
    .note {{ font-size: 11px; fill: #526071; }}
  </style>

  <text x="380" y="30" text-anchor="middle" class="title">Removing label-bearing fields drops text AUROC to chance</text>
  <text x="380" y="50" text-anchor="middle" class="subtitle">Three deterministic full-prompt rules; REAL is the positive class</text>

  <line x1="75" y1="285" x2="705" y2="285" class="axis"/>
  <line x1="75" y1="85" x2="75" y2="285" class="axis"/>
  <line x1="75" y1="185" x2="705" y2="185" class="grid"/>
  <line x1="75" y1="85" x2="705" y2="85" class="grid"/>
  <text x="67" y="289" text-anchor="end" class="tick">0</text>
  <text x="67" y="189" text-anchor="end" class="tick">0.50</text>
  <text x="67" y="89" text-anchor="end" class="tick">1.00 AUROC</text>
{rendered_bars}

  <rect x="210" y="352" width="12" height="12" rx="2" fill="#c2413b"/>
  <text x="228" y="362" class="note">Original verifier fields</text>
  <rect x="400" y="352" width="12" height="12" rx="2" fill="#0f9d8a"/>
  <text x="418" y="362" class="note">Leak-free Exp 4b fields</text>
  <text x="380" y="382" text-anchor="middle" class="note">Low-base-rate accuracy is omitted because constant-SHAM prediction is 99.0% accurate but AUROC 0.500.</text>
</svg>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--leak-free", type=Path, default=DEFAULT_LEAK_FREE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = render_svg(extract_values(args.original, args.leak_free))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
