"""Evaluate trivial REAL/SHAM rules on the exact subject-visible verifier text.

This is a post-hoc, CPU-only acceptance audit. It does not alter the sealed Wave 2
bundle. The baseline intentionally uses the full prompt that the subject/probe saw,
unlike the preregistered blind CoT monitor, which redacted raw verifier fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any


DEFAULT_MAIN = Path("experiments/exp3_dataset/data/prompts_main.jsonl")
DEFAULT_LBR = Path("experiments/exp3_dataset/data/lowbaserate_eval.jsonl")
DEFAULT_LEAK_FREE_MAIN = Path(
    "experiments/exp3_dataset/data_leak_free/prompts_main.jsonl"
)
DEFAULT_LEAK_FREE_LBR = Path(
    "experiments/exp3_dataset/data_leak_free/lowbaserate_eval.jsonl"
)

RULES = (
    ("settled_true", '"settled": true'),
    ("usdc_balance_1_9", '"usdc_balance": 1.9'),
    ("tx_confirmed_true", '"tx_confirmed": true'),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"id", "world", "cond", "label", "prompt"}
            missing = sorted(required - row.keys())
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields: {missing}")
            if not isinstance(row["prompt"], str):
                raise ValueError(
                    f"{path}:{line_number} prompt must be a string, got "
                    f"{type(row['prompt']).__name__}"
                )
            expected_label = {"REAL": 1, "SHAM": 0}.get(row["world"])
            if expected_label is None or row["label"] != expected_label:
                raise ValueError(
                    f"{path}:{line_number} violates REAL=1 / SHAM=0 contract"
                )
            rows.append(row)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"{path} contains duplicate ids")
    return rows


def _binary_auroc(labels: list[int], scores: list[float]) -> float | None:
    """Compute AUROC as the fraction of correctly ordered positive/negative pairs."""

    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return None
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def _metrics(
    labels: list[int], predictions: list[int], *, include_auroc: bool = False
) -> dict[str, Any]:
    if not labels or len(labels) != len(predictions):
        raise ValueError("labels and predictions must be nonempty and aligned")
    tp = sum(
        label == 1 and prediction == 1 for label, prediction in zip(labels, predictions)
    )
    fn = sum(
        label == 1 and prediction == 0 for label, prediction in zip(labels, predictions)
    )
    tn = sum(
        label == 0 and prediction == 0 for label, prediction in zip(labels, predictions)
    )
    fp = sum(
        label == 0 and prediction == 1 for label, prediction in zip(labels, predictions)
    )
    # Guard every denominator: an absent class (all-REAL or all-SHAM slice)
    # yields None rather than a ZeroDivisionError. On the committed data both
    # classes are always present, so this leaves the sealed output unchanged.
    recall = tp / (tp + fn) if tp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    precision = tp / (tp + fp) if tp + fp else None
    if recall is None or fpr is None:
        projected_denominator = None
        projected_precision = None
    else:
        projected_denominator = recall + 50 * fpr
        projected_precision = (
            recall / projected_denominator if projected_denominator else None
        )
    result = {
        "samples": len(labels),
        "positive_REAL": sum(labels),
        "negative_SHAM": len(labels) - sum(labels),
        "true_positives": tp,
        "false_negatives": fn,
        "true_negatives": tn,
        "false_positives": fp,
        "accuracy": (tp + tn) / len(labels),
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fpr,
        "projected_precision_REAL_to_SHAM_1_to_50": projected_precision,
    }
    if include_auroc:
        result["auroc"] = _binary_auroc(labels, [float(value) for value in predictions])
    return result


def _evaluate(
    rows: list[dict[str, Any]],
    *,
    include_auroc: bool = False,
    leak_free: bool = False,
) -> dict[str, Any]:
    labels = [int(row["label"]) for row in rows]
    results: list[dict[str, Any]] = []
    predictions_by_rule: list[list[int]] = []
    for name, literal in RULES:
        predictions = [int(literal in str(row["prompt"])) for row in rows]
        predictions_by_rule.append(predictions)
        entry: dict[str, Any] = {
            "name": name,
            "REAL_if_prompt_contains": literal,
            "metrics": _metrics(labels, predictions, include_auroc=include_auroc),
        }
        if leak_free:
            # Under the leak-free contract a v1 literal is expected to be absent
            # from every prompt, so report what the rule matched rather than
            # scoring it against the label it can no longer track.
            entry["prompt_ids_matching_literal"] = [
                str(row["id"])
                for row, prediction in zip(rows, predictions)
                if prediction == 1
            ]
        else:
            entry["error_ids"] = [
                str(row["id"])
                for row, label, prediction in zip(rows, labels, predictions)
                if label != prediction
            ]
        results.append(entry)
    block = {
        "rules": results,
        "all_rules_identical_predictions": all(
            predictions == predictions_by_rule[0]
            for predictions in predictions_by_rule[1:]
        ),
    }
    if leak_free:
        block["all_rules_inert_no_prompt_matches"] = all(
            not result["prompt_ids_matching_literal"] for result in results
        )
    else:
        block["all_rules_zero_errors"] = all(
            not result["error_ids"] for result in results
        )
    return block


def _leak_free_equivalence_audit(
    main_rows: list[dict[str, Any]], lbr_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    paired: dict[tuple[int, str], dict[str, str]] = {}
    for row in main_rows:
        key = (int(row["template_id"]), str(row["cond"]))
        world = str(row["world"])
        if world in paired.setdefault(key, {}):
            raise ValueError(f"duplicate main prompt for {key} / {world}")
        paired[key][world] = str(row["prompt"])
    for key, prompts in paired.items():
        if set(prompts) != {"REAL", "SHAM"}:
            raise ValueError(f"main prompt pair is incomplete for {key}")
        if prompts["REAL"] != prompts["SHAM"]:
            raise ValueError(f"main prompt text differs across classes for {key}")

    by_world = {
        world: Counter(str(row["prompt"]) for row in lbr_rows if row["world"] == world)
        for world in ("REAL", "SHAM")
    }
    totals = {world: sum(counts.values()) for world, counts in by_world.items()}
    if not all(totals.values()):
        raise ValueError("low-base-rate input must contain both classes")
    if set(by_world["REAL"]) != set(by_world["SHAM"]):
        raise ValueError("low-base-rate prompt support differs across classes")
    for prompt in by_world["REAL"]:
        real_rate = Fraction(by_world["REAL"][prompt], totals["REAL"])
        sham_rate = Fraction(by_world["SHAM"][prompt], totals["SHAM"])
        if real_rate != sham_rate:
            raise ValueError("low-base-rate prompt distribution differs across classes")
    return {
        "main_REAL_SHAM_byte_identical_pairs": len(paired),
        "low_base_rate_shared_unique_prompts": len(by_world["REAL"]),
        "low_base_rate_prompt_distribution_identical_by_class": True,
    }


def build_report(
    main_path: Path, lbr_path: Path, *, expect_leak_free: bool = False
) -> dict[str, Any]:
    all_main_rows = _load_jsonl(main_path)
    main_rows = [row for row in all_main_rows if row["cond"] in {"b", "c"}]
    main_by_split = {
        split: [row for row in main_rows if row.get("split") == split]
        for split in ("train", "heldout")
    }
    lbr_rows = _load_jsonl(lbr_path)
    if len(all_main_rows) != 288:
        raise ValueError("unexpected main inventory")
    if (
        len(main_rows) != 192
        or len(main_by_split["train"]) != 152
        or len(main_by_split["heldout"]) != 40
    ):
        raise ValueError("unexpected main b+c inventory")
    if len(lbr_rows) != 1010:
        raise ValueError("unexpected low-base-rate inventory")

    equivalence = (
        _leak_free_equivalence_audit(all_main_rows, lbr_rows)
        if expect_leak_free
        else None
    )
    evaluations = {
        "main_train_b_plus_c_unique_prompts": _evaluate(
            main_by_split["train"],
            include_auroc=expect_leak_free,
            leak_free=expect_leak_free,
        ),
        "main_heldout_b_plus_c_unique_prompts": _evaluate(
            main_by_split["heldout"],
            include_auroc=expect_leak_free,
            leak_free=expect_leak_free,
        ),
        "independent_low_base_rate_cache": _evaluate(
            lbr_rows, include_auroc=expect_leak_free, leak_free=expect_leak_free
        ),
    }
    report = {
        "format_version": 2 if expect_leak_free else 1,
        "status": (
            "leak_free_acceptance_audit"
            if expect_leak_free
            else "post_hoc_acceptance_audit"
        ),
        "class_contract": {
            "positive": "REAL",
            "positive_label": 1,
            "negative": "SHAM",
            "negative_label": 0,
        },
        "method": (
            {
                "description": "Three deterministic substring rules read the complete subject-visible prompt; no fitting or threshold selection.",
                "interpretation": "This leak gate combines rule AUROC with byte-level REAL/SHAM prompt equivalence. The v1 literals are inert on this variant, so their AUROC 0.5 is descriptive; byte-identical main pairs and class-invariant low-base-rate prompt distributions are the operative proof.",
                "acceptance": "Every rule must have AUROC 0.5, every main REAL/SHAM pair in all conditions (a, b, c) must be byte-identical, and the low-base-rate text distribution must be class-invariant.",
                "rule_field_semantics": "prompt_ids_matching_literal lists the prompts a rule fired on and is expected to be empty; all_rules_inert_no_prompt_matches true is the passing outcome, not a failed audit.",
            }
            if expect_leak_free
            else {
                "description": "Three deterministic substring rules read the complete subject-visible prompt; no fitting or threshold selection.",
                "interpretation": "This is a full-information text baseline for explicit verifier evidence, not a monitor of latent belief.",
                "comparison_boundary": "The preregistered Claude CoT monitor redacted these verifier fields, so its head-to-head comparison had less information.",
            }
        ),
        "inputs": {
            "main": {"path": str(main_path), "sha256": _sha256(main_path)},
            "low_base_rate": {"path": str(lbr_path), "sha256": _sha256(lbr_path)},
        },
        "evaluations": evaluations,
    }
    if expect_leak_free:
        aurocs = [
            rule["metrics"]["auroc"]
            for evaluation in evaluations.values()
            for rule in evaluation["rules"]
        ]
        if any(auroc != 0.5 for auroc in aurocs):
            raise ValueError(f"leak-free string baseline exceeded chance: {aurocs}")
        report["leak_gate"] = {
            "passed": True,
            "expected_auroc": 0.5,
            "observed_rule_aurocs": sorted(set(aurocs)),
            "text_equivalence": equivalence,
        }
    elif not all(
        evaluation["all_rules_zero_errors"] for evaluation in evaluations.values()
    ):
        raise ValueError("a deterministic rule made an error")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path)
    parser.add_argument("--low-base-rate", type=Path)
    parser.add_argument(
        "--leak-free",
        action="store_true",
        help="use leak-free defaults and enforce the chance-AUROC gate",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    main_path = args.main or (
        DEFAULT_LEAK_FREE_MAIN if args.leak_free else DEFAULT_MAIN
    )
    lbr_path = args.low_base_rate or (
        DEFAULT_LEAK_FREE_LBR if args.leak_free else DEFAULT_LBR
    )
    try:
        report = build_report(main_path, lbr_path, expect_leak_free=args.leak_free)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        # Fail closed with a structured report + nonzero exit, so a bad input
        # can never masquerade as a successful audit.
        failure = {"status": "failed", "error": type(exc).__name__, "detail": str(exc)}
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        raise SystemExit(1)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
