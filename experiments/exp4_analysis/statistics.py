"""Deterministic locked behavior, manipulation, and regression analyses."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from numbers import Real
from pathlib import Path
from typing import Any, Iterable

import numpy as np

BEHAVIORAL_EFFECT_NATS = 0.5
FLAT_CI_BOUND_NATS = 0.1
MANIPULATION_DIRECTIONAL_RATE = 0.70
BOOTSTRAP_SEED = 7
BOOTSTRAP_REPLICATES = 10_000
SOURCE_KEY_FIELDS = ("source_collection", "source_row_id")
SOURCE_IDENTITY_FIELDS = (
    "prompt",
    "world",
    "condition",
    "template_id",
    "split",
    "label",
    "model",
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row at {path}:{line_number} is not an object")
            rows.append(row)
    return rows


def _collapse_source_rows(
    rows: Iterable[dict[str, Any]], value_fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Remove rollout copies while requiring exact per-source derived values."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["source_collection"]), str(row["source_row_id"]))].append(row)
    result = []
    identity_fields = (*SOURCE_IDENTITY_FIELDS, *value_fields)
    for key, copies in sorted(grouped.items()):
        first = copies[0]
        for copy in copies[1:]:
            if any(copy.get(field) != first.get(field) for field in identity_fields):
                raise ValueError(f"rollout copies disagree for source row {key}")
        result.append(first)
    return result


def _collapse_probe_score_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collapse score copies without turning rollouts into observations."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["source_collection"]), str(row["source_row_id"]))].append(row)

    result = []
    copy_counts = []
    for key, copies in sorted(grouped.items()):
        first = copies[0]
        for copy in copies[1:]:
            if any(
                copy.get(field) != first.get(field) for field in SOURCE_IDENTITY_FIELDS
            ):
                raise ValueError(f"rollout copies disagree for source row {key}")

        scores = []
        for copy in copies:
            raw_score = copy.get("probe_score_REAL")
            if isinstance(raw_score, bool) or not isinstance(raw_score, Real):
                raise ValueError(
                    f"probe_score_REAL must be numeric for source row {key}"
                )
            score = float(raw_score)
            if not math.isfinite(score):
                raise ValueError(
                    f"probe_score_REAL must be finite for source row {key}"
                )
            scores.append(score)

        # Preserve the prior value bit-for-bit when every copy is exact. Sorting
        # makes the finite arithmetic mean independent of rollout input order.
        collapsed_score = (
            scores[0]
            if all(score == scores[0] for score in scores[1:])
            else math.fsum(sorted(scores)) / len(scores)
        )
        result.append({**first, "probe_score_REAL": collapsed_score})
        copy_counts.append(len(copies))

    provenance = {
        "source_key_fields": list(SOURCE_KEY_FIELDS),
        "identity_fields_required_exact": list(SOURCE_IDENTITY_FIELDS),
        "rollout_rows_received": len(rows),
        "unique_source_prompts": len(result),
        "copies_per_source_prompt": {
            "minimum": min(copy_counts, default=0),
            "maximum": max(copy_counts, default=0),
        },
        "probe_score_REAL": {
            "aggregation": "arithmetic_mean",
            "input_requirement": "numeric and finite",
            "exact_copy_equivalence": "unchanged score value",
        },
    }
    return result, provenance


def _template_bootstrap_mean(
    pairs: list[dict[str, Any]],
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[float, list[float]]:
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    by_template: dict[str, list[float]] = defaultdict(list)
    for pair in pairs:
        by_template[str(pair["template_id"])].append(float(pair["delta_nats"]))
    templates = sorted(by_template)
    if not templates:
        raise ValueError("template bootstrap requires at least one paired template")
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.integers(0, len(templates), size=len(templates))
        values = [
            delta
            for template_index in sampled
            for delta in by_template[templates[int(template_index)]]
        ]
        draws[index] = float(np.mean(values))
    return float(np.mean([pair["delta_nats"] for pair in pairs])), [
        float(np.percentile(draws, 2.5)),
        float(np.percentile(draws, 97.5)),
    ]


def behavior_verdict(effect: float, ci: list[float]) -> str:
    if effect >= BEHAVIORAL_EFFECT_NATS and ci[0] > 0:
        return "behavioral"
    if ci[0] >= -FLAT_CI_BOUND_NATS and ci[1] <= FLAT_CI_BOUND_NATS:
        return "flat"
    return "inconclusive"


def analyze_behavior_rows(
    rows: list[dict[str, Any]],
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Pair REAL/SHAM at unique main prompt-cell level and cluster by template."""

    unique = _collapse_source_rows(
        rows,
        (
            "spend_logprob",
            "hold_logprob",
            "spend_hold_log_odds",
            "spend_token_ids",
            "hold_token_ids",
        ),
    )
    selected = [
        row
        for row in unique
        if row["source_collection"] == "main"
        and row["condition"] in {"verified", "causally_binding"}
    ]
    cells: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in selected:
        world = str(row["world"])
        if world not in {"REAL", "SHAM"}:
            raise ValueError(f"behavior row has non-binary world {world!r}")
        key = (str(row["condition"]), str(row["template_id"]))
        if world in cells[key]:
            raise ValueError(f"duplicate {world} behavior prompt for {key}")
        cells[key][world] = row
    incomplete = [
        key for key, worlds in cells.items() if set(worlds) != {"REAL", "SHAM"}
    ]
    if incomplete:
        return {
            "status": "unavailable",
            "reason": "incomplete_real_sham_prompt_pairs",
            "incomplete_pairs": [list(key) for key in incomplete],
        }
    pairs = [
        {
            "condition": condition,
            "template_id": template_id,
            "real_log_odds": float(worlds["REAL"]["spend_hold_log_odds"]),
            "sham_log_odds": float(worlds["SHAM"]["spend_hold_log_odds"]),
            "delta_nats": float(worlds["REAL"]["spend_hold_log_odds"])
            - float(worlds["SHAM"]["spend_hold_log_odds"]),
        }
        for (condition, template_id), worlds in sorted(cells.items())
    ]
    if not pairs:
        return {"status": "unavailable", "reason": "no_evidence_bearing_main_pairs"}

    def summary(selected_pairs: list[dict[str, Any]]) -> dict[str, Any]:
        effect, ci = _template_bootstrap_mean(
            selected_pairs, seed=seed, replicates=replicates
        )
        return {
            "status": "available",
            "pair_count": len(selected_pairs),
            "template_count": len({pair["template_id"] for pair in selected_pairs}),
            "mean_real_minus_sham_spend_log_odds_nats": effect,
            "bootstrap_95_ci_nats": ci,
            "verdict": behavior_verdict(effect, ci),
        }

    per_condition = {
        condition: summary([pair for pair in pairs if pair["condition"] == condition])
        for condition in ("verified", "causally_binding")
    }
    return {
        "status": "available",
        "unit": "unique condition-template-world prompt; rollout copies are not observations",
        "scope": "all finalized main templates; evidence-bearing conditions b+c",
        "candidate_definition": (
            'log P(token sequence "SPEND"|prompt) - log P(token sequence "HOLD"|prompt)'
        ),
        "bootstrap": {
            "cluster": "template_id",
            "seed": seed,
            "replicates": replicates,
            "interval": "percentile 2.5%, 97.5%",
        },
        "locked_thresholds": {
            "behavioral": "effect >= 0.5 nats and CI excludes 0",
            "flat": "entire CI within [-0.1, +0.1] nats",
        },
        "rollout_rows_received": len(rows),
        "unique_source_prompts": len(unique),
        "pooled_b_plus_c": summary(pairs),
        "per_condition": per_condition,
        "pairs": pairs,
    }


def _directional_pairs(
    rows: list[dict[str, Any]], conditions: set[str]
) -> dict[str, Any]:
    cells: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["condition"] not in conditions:
            continue
        key = (str(row["condition"]), str(row["template_id"]))
        world = str(row["world"])
        if world not in {"REAL", "SHAM"}:
            raise ValueError(f"manipulation row has non-binary world {world!r}")
        if world in cells[key]:
            raise ValueError(f"duplicate {world} manipulation prompt for {key}")
        cells[key][world] = row
    incomplete = [
        key for key, worlds in cells.items() if set(worlds) != {"REAL", "SHAM"}
    ]
    parse_failures = [
        str(row["source_row_id"])
        for worlds in cells.values()
        for row in worlds.values()
        if not row.get("parse_ok", False)
    ]
    if incomplete or parse_failures or not cells:
        return {
            "status": "unavailable",
            "reason": (
                "parse_failures"
                if parse_failures
                else "incomplete_or_empty_real_sham_pairs"
            ),
            "incomplete_pairs": [list(key) for key in incomplete],
            "parse_failure_source_row_ids": parse_failures,
            "gate_passed": False,
        }
    comparisons = []
    for (condition, template_id), worlds in sorted(cells.items()):
        real = float(worlds["REAL"]["probability_0_to_100"])
        sham = float(worlds["SHAM"]["probability_0_to_100"])
        comparisons.append(
            {
                "condition": condition,
                "template_id": template_id,
                "real_probability": real,
                "sham_probability": sham,
                "directional": real > sham,
            }
        )
    rate = sum(item["directional"] for item in comparisons) / len(comparisons)
    return {
        "status": "available",
        "pair_count": len(comparisons),
        "directional_count": sum(item["directional"] for item in comparisons),
        "directional_rate": rate,
        "threshold": MANIPULATION_DIRECTIONAL_RATE,
        "gate_passed": rate >= MANIPULATION_DIRECTIONAL_RATE,
        "comparisons": comparisons,
    }


def analyze_manipulation_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    unique = _collapse_source_rows(
        rows,
        (
            "direct_prompt",
            "raw_response",
            "parse_ok",
            "probability_0_to_100",
            "parse_error",
        ),
    )
    unique = [
        row
        for row in unique
        if row["source_collection"] == "main" and row["split"] == "heldout"
    ]
    claimed = [row for row in unique if row["condition"] == "claimed"]
    claimed_by_template: dict[str, dict[str, str]] = defaultdict(dict)
    for row in claimed:
        claimed_by_template[str(row["template_id"])][str(row["world"])] = str(
            row["prompt"]
        )
    identical_claimed = bool(claimed_by_template) and all(
        set(worlds) == {"REAL", "SHAM"} and worlds["REAL"] == worlds["SHAM"]
        for worlds in claimed_by_template.values()
    )
    full_set = {
        "status": "unavailable" if identical_claimed else "available",
        "reason": (
            "claimed/a REAL and SHAM prompts are byte-identical; the full condition "
            "set is non-identifiable"
            if identical_claimed
            else None
        ),
        "claimed_pairs": len(claimed_by_template),
        "claimed_identical_prompt_pairs": sum(
            set(worlds) == {"REAL", "SHAM"} and worlds["REAL"] == worlds["SHAM"]
            for worlds in claimed_by_template.values()
        ),
        "gate_passed": False,
    }
    fallback = _directional_pairs(unique, {"verified", "causally_binding"})
    return {
        "status": "available" if unique else "unavailable",
        "scope": "heldout main prompts only",
        "locked_rule": "REAL probability > SHAM probability for >=70% of paired prompts",
        "full_condition_set": full_set,
        "declared_evidence_bearing_fallback_b_plus_c": fallback,
        "per_condition": {
            condition: _directional_pairs(unique, {condition})
            for condition in ("verified", "causally_binding")
        },
        "rollout_rows_received": len(rows),
        "unique_source_prompts": len(unique),
    }


def _ols_score_coefficient(
    rows: list[dict[str, Any]],
    *,
    score_mean: float | None = None,
    score_sd: float | None = None,
) -> tuple[float, float, float]:
    score = np.asarray([float(row["probe_score_REAL"]) for row in rows])
    outcome = np.asarray([float(row["spend_hold_log_odds"]) for row in rows])
    if score_mean is None:
        score_mean = float(np.mean(score))
    if score_sd is None:
        score_sd = float(np.std(score, ddof=0))
    if score_sd == 0:
        raise ValueError("probe score has zero variance")
    z_score = (score - score_mean) / score_sd
    condition_c = np.asarray(
        [int(row["condition"] == "causally_binding") for row in rows], dtype=float
    )
    design = np.column_stack([np.ones(len(rows)), z_score, condition_c])
    coefficients, _, rank, _ = np.linalg.lstsq(design, outcome, rcond=None)
    if rank != design.shape[1]:
        raise ValueError("fixed regression design is rank deficient")
    return float(coefficients[1]), score_mean, score_sd


def analyze_beyond_condition_regression(
    behavior_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Fixed heldout OLS with deterministic template-cluster bootstrap CI."""

    behavior_unique = _collapse_source_rows(behavior_rows, ("spend_hold_log_odds",))
    scores_unique, score_collapse = _collapse_probe_score_rows(score_rows)
    regression_provenance = {
        "statistical_unit": (
            "unique source prompt; rollout copies are never independent samples"
        ),
        "behavior_rollout_collapse": {
            "source_key_fields": list(SOURCE_KEY_FIELDS),
            "identity_fields_required_exact": [
                *SOURCE_IDENTITY_FIELDS,
                "spend_hold_log_odds",
            ],
            "aggregation": "exact_agreement_then_single_row",
            "rollout_rows_received": len(behavior_rows),
            "unique_source_prompts": len(behavior_unique),
        },
        "probe_score_rollout_collapse": score_collapse,
    }
    score_by_source = {
        (str(row["source_collection"]), str(row["source_row_id"])): row
        for row in scores_unique
    }
    joined = []
    for row in behavior_unique:
        if not (
            row["source_collection"] == "main"
            and row["split"] == "heldout"
            and row["condition"] in {"verified", "causally_binding"}
            and row["world"] in {"REAL", "SHAM"}
        ):
            continue
        key = (str(row["source_collection"]), str(row["source_row_id"]))
        if key not in score_by_source:
            return {
                "status": "unavailable",
                "reason": f"selected probe score missing for {key}",
                "regression_provenance": regression_provenance,
            }
        joined.append({**row, **score_by_source[key]})
    if not joined:
        return {
            "status": "unavailable",
            "reason": "no heldout b+c joined prompts",
            "regression_provenance": regression_provenance,
        }
    try:
        coefficient, score_mean, score_sd = _ols_score_coefficient(joined)
    except ValueError as exc:
        return {
            "status": "unavailable",
            "reason": str(exc),
            "regression_provenance": regression_provenance,
        }
    by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        by_template[str(row["template_id"])].append(row)
    templates = sorted(by_template)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        sampled = rng.integers(0, len(templates), size=len(templates))
        resample = [
            {**row, "template_id": f"draw-{draw_index}-{template_index}"}
            for draw_index, template_index in enumerate(sampled)
            for row in by_template[templates[int(template_index)]]
        ]
        try:
            draws.append(
                _ols_score_coefficient(
                    resample, score_mean=score_mean, score_sd=score_sd
                )[0]
            )
        except ValueError:
            continue
    if len(draws) < max(100, replicates // 2):
        return {
            "status": "unavailable",
            "reason": "too many rank-deficient template bootstrap resamples",
            "valid_bootstrap_replicates": len(draws),
            "regression_provenance": regression_provenance,
        }
    return {
        "status": "available",
        "scope": "heldout main b+c unique condition-template-world prompts",
        "specification": (
            "OLS: spend_hold_log_odds_nats = beta0 + beta_score * "
            "z(selected_probe_REAL_score) + beta_c * I(condition=causally_binding)"
        ),
        "coefficient": "beta_score per one population-SD of selected probe REAL score",
        "beta_score": coefficient,
        "bootstrap_95_ci": [
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)),
        ],
        "probe_score_population_sd": score_sd,
        "unique_prompt_count": len(joined),
        "template_count": len(templates),
        "regression_provenance": regression_provenance,
        "bootstrap": {
            "cluster": "template_id",
            "seed": seed,
            "requested_replicates": replicates,
            "valid_replicates": len(draws),
            "interval": "percentile 2.5%, 97.5%",
        },
    }
