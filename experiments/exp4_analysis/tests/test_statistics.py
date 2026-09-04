from __future__ import annotations

import pytest

from experiments.exp4_analysis.statistics import (
    BEHAVIORAL_EFFECT_NATS,
    BOOTSTRAP_SEED,
    FLAT_CI_BOUND_NATS,
    MANIPULATION_DIRECTIONAL_RATE,
    analyze_behavior_rows,
    analyze_beyond_condition_regression,
    analyze_manipulation_rows,
)
from experiments.exp4_probes.metrics import FIXED_SCORE_THRESHOLD


def behavior_rows(delta: float = 0.6) -> list[dict]:
    rows = []
    for template in range(6):
        split = "heldout" if template >= 3 else "train"
        for condition in ("verified", "causally_binding"):
            for world, value in (("REAL", delta), ("SHAM", 0.0)):
                for rollout in range(25):
                    rows.append(
                        {
                            "transcript_id": f"{template}-{condition}-{world}-{rollout}",
                            "source_collection": "main",
                            "source_row_id": f"{template}-{condition}-{world}",
                            "rollout_index": rollout,
                            "prompt": f"prompt {template} {condition} {world}",
                            "world": world,
                            "condition": condition,
                            "template_id": template,
                            "split": split,
                            "label": int(world == "REAL"),
                            "model": "fixture/model",
                            "spend_logprob": value - 1,
                            "hold_logprob": -1,
                            "spend_hold_log_odds": value,
                            "spend_token_ids": [1, 2],
                            "hold_token_ids": [3],
                        }
                    )
    return rows


def test_template_bootstrap_does_not_count_25_rollouts_as_independent() -> None:
    result = analyze_behavior_rows(behavior_rows(), replicates=300)
    pooled = result["pooled_b_plus_c"]
    assert result["rollout_rows_received"] == 6 * 2 * 2 * 25
    assert result["unique_source_prompts"] == 6 * 2 * 2
    assert pooled["pair_count"] == 12
    assert pooled["template_count"] == 6
    assert pooled["mean_real_minus_sham_spend_log_odds_nats"] == pytest.approx(0.6)
    assert pooled["bootstrap_95_ci_nats"] == pytest.approx([0.6, 0.6])
    assert pooled["verdict"] == "behavioral"
    assert set(result["per_condition"]) == {"verified", "causally_binding"}


def test_flat_behavior_uses_locked_ci_rule() -> None:
    result = analyze_behavior_rows(behavior_rows(delta=0.0), replicates=100)
    assert result["pooled_b_plus_c"]["verdict"] == "flat"


def manipulation_rows() -> list[dict]:
    rows = []
    for template in range(3):
        for condition in ("claimed", "verified", "causally_binding"):
            for world in ("REAL", "SHAM"):
                prompt = (
                    f"identical claimed {template}"
                    if condition == "claimed"
                    else f"{condition} {template} {world}"
                )
                probability = (
                    50 if condition == "claimed" else (80 if world == "REAL" else 20)
                )
                rows.append(
                    {
                        "transcript_id": f"{template}-{condition}-{world}",
                        "source_collection": "main",
                        "source_row_id": f"{template}-{condition}-{world}",
                        "rollout_index": 0,
                        "prompt": prompt,
                        "world": world,
                        "condition": condition,
                        "template_id": template,
                        "split": "heldout",
                        "label": int(world == "REAL"),
                        "model": "fixture/model",
                        "direct_prompt": "direct",
                        "raw_response": str(probability),
                        "parse_ok": True,
                        "probability_0_to_100": probability,
                        "parse_error": None,
                    }
                )
    return rows


def test_manipulation_reports_claimed_nonidentifiability_and_declared_fallback() -> (
    None
):
    result = analyze_manipulation_rows(manipulation_rows())
    assert result["full_condition_set"]["status"] == "unavailable"
    assert "byte-identical" in result["full_condition_set"]["reason"]
    fallback = result["declared_evidence_bearing_fallback_b_plus_c"]
    assert fallback["directional_rate"] == 1.0
    assert fallback["gate_passed"]
    assert set(result["per_condition"]) == {"verified", "causally_binding"}


def test_manipulation_parse_failure_is_not_silently_dropped() -> None:
    rows = manipulation_rows()
    target = next(row for row in rows if row["condition"] == "verified")
    target.update(parse_ok=False, probability_0_to_100=None, parse_error="bad")
    result = analyze_manipulation_rows(rows)
    fallback = result["declared_evidence_bearing_fallback_b_plus_c"]
    assert fallback["status"] == "unavailable"
    assert target["source_row_id"] in fallback["parse_failure_source_row_ids"]


def test_fixed_beyond_condition_regression_has_template_ci() -> None:
    behavior = behavior_rows()
    scores = probe_score_rows(behavior)
    result = analyze_beyond_condition_regression(behavior, scores, replicates=300)
    assert result["status"] == "available"
    assert result["unique_prompt_count"] == 3 * 2 * 2
    assert result["template_count"] == 3
    assert result["beta_score"] > 0
    assert result["bootstrap_95_ci"][0] > 0
    assert "I(condition=causally_binding)" in result["specification"]
    collapse = result["regression_provenance"]["probe_score_rollout_collapse"]
    assert collapse["rollout_rows_received"] == 6 * 2 * 2 * 25
    assert collapse["unique_source_prompts"] == 6 * 2 * 2
    assert collapse["copies_per_source_prompt"] == {"minimum": 25, "maximum": 25}
    assert collapse["probe_score_REAL"]["aggregation"] == "arithmetic_mean"
    assert collapse["probe_score_REAL"]["exact_copy_equivalence"] == (
        "unchanged score value"
    )
    one_score_per_prompt = [row for row in scores if row["rollout_index"] == 0]
    single_copy = analyze_beyond_condition_regression(
        behavior, one_score_per_prompt, replicates=300
    )
    assert single_copy["beta_score"] == result["beta_score"]
    assert single_copy["bootstrap_95_ci"] == result["bootstrap_95_ci"]
    assert (
        single_copy["probe_score_population_sd"] == result["probe_score_population_sd"]
    )


def probe_score_rows(behavior: list[dict], *, variation: float = 0.0) -> list[dict]:
    rows = []
    for row in behavior:
        base_score = 0.8 if row["world"] == "REAL" else 0.2
        rows.append(
            {
                **{
                    key: row[key]
                    for key in (
                        "transcript_id",
                        "source_collection",
                        "source_row_id",
                        "rollout_index",
                        "prompt",
                        "world",
                        "condition",
                        "template_id",
                        "split",
                        "label",
                        "model",
                    )
                },
                "probe_score_REAL": base_score
                + (row["rollout_index"] - 12) * variation,
            }
        )
    return rows


@pytest.mark.parametrize("variation", [1e-6, 0.01])
def test_rollout_score_variation_is_meaned_per_unique_prompt(
    variation: float,
) -> None:
    behavior = behavior_rows()
    exact = analyze_beyond_condition_regression(
        behavior, probe_score_rows(behavior), replicates=300
    )
    varied = analyze_beyond_condition_regression(
        behavior,
        probe_score_rows(behavior, variation=variation),
        replicates=300,
    )

    assert varied["status"] == "available"
    assert varied["unique_prompt_count"] == 3 * 2 * 2
    assert varied["beta_score"] == pytest.approx(exact["beta_score"])
    assert varied["bootstrap_95_ci"] == pytest.approx(exact["bootstrap_95_ci"])
    collapse = varied["regression_provenance"]["probe_score_rollout_collapse"]
    assert collapse["rollout_rows_received"] == 25 * collapse["unique_source_prompts"]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [("prompt", "different prompt"), ("label", 99), ("model", "other/model")],
)
def test_rollout_score_identity_disagreement_fails_closed(
    field: str, replacement: object
) -> None:
    behavior = behavior_rows()
    scores = probe_score_rows(behavior, variation=1e-6)
    scores[1][field] = replacement

    with pytest.raises(ValueError, match="rollout copies disagree"):
        analyze_beyond_condition_regression(behavior, scores, replicates=100)


@pytest.mark.parametrize(
    ("bad_score", "message"),
    [
        ("0.8", "must be numeric"),
        (True, "must be numeric"),
        (float("nan"), "must be finite"),
        (float("inf"), "must be finite"),
    ],
)
def test_rollout_scores_must_be_numeric_and_finite(
    bad_score: object, message: str
) -> None:
    behavior = behavior_rows()
    scores = probe_score_rows(behavior)
    scores[1]["probe_score_REAL"] = bad_score

    with pytest.raises(ValueError, match=message):
        analyze_beyond_condition_regression(behavior, scores, replicates=100)


def test_preregistered_constants_have_no_mutable_threshold_path() -> None:
    assert FIXED_SCORE_THRESHOLD == 0.5
    assert BEHAVIORAL_EFFECT_NATS == 0.5
    assert FLAT_CI_BOUND_NATS == 0.1
    assert MANIPULATION_DIRECTIONAL_RATE == 0.70
    assert BOOTSTRAP_SEED == 7
