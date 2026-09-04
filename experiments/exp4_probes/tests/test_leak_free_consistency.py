from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from experiments.exp4_probes.cache import load_activation_cache, load_metadata
from experiments.exp4_probes.leak_free_consistency import (
    LBR_PROVENANCE,
    STATUS_ABOVE_CHANCE,
    STATUS_AT_CHANCE,
    build_verdict,
    check_prompt_identity,
    run_consistency_check,
    stratified_row_weights,
    template_cluster_weights,
    weighted_auroc,
)


def write_leak_free_fixture(
    tmp_path: Path,
    *,
    variant: str | None = "leak_free",
    rollouts: int = 4,
    world_signal_layer: int | None = None,
) -> tuple[Path, Path]:
    """Byte-identical REAL/SHAM prompts with condition signal but no world signal.

    ``world_signal_layer`` injects a REAL/SHAM separation into one layer's
    activations while leaving every prompt byte-identical, which is exactly the
    "bug or residual leak" the consistency check must fail closed on.
    """

    rng = np.random.default_rng(123)
    metadata = []
    prompts = []
    transcript_ids = []
    labels = []
    features = []
    conditions = (("claimed", 0), ("verified", 1), ("causally_binding", 2))
    for template in range(15):
        split = "heldout" if template >= 12 else "train"
        for condition, condition_index in conditions:
            prompt = f"{condition} prompt {template}"
            for world, label in (("REAL", 1), ("SHAM", 0)):
                source_row_id = f"{world.lower()}_{condition}_t{template:02d}"
                for rollout in range(rollouts):
                    metadata.append(
                        {
                            "transcript_id": f"main:{source_row_id}:r{rollout:02d}",
                            "source_collection": "main",
                            "source_row_id": source_row_id,
                            "rollout_index": rollout,
                            "prompt": prompt,
                            "world": world,
                            "condition": condition,
                            "template_id": template,
                            "split": split,
                            "label": label,
                            "model": "fixture/model",
                            **({"dataset_variant": variant} if variant else {}),
                        }
                    )
                    prompts.append(prompt)
                    transcript_ids.append(metadata[-1]["transcript_id"])
                    labels.append(label)
                    X = rng.normal(0, 0.25, size=(3, 8))
                    X[:, 0] += condition_index * 2.0
                    if world_signal_layer is not None:
                        X[world_signal_layer, 1] += 2.0 * label
                    features.append(X)
    for world, count, label in (("REAL", 10, 1), ("SHAM", 1000, 0)):
        for index in range(count):
            source_row_id = f"lbr_{world.lower()}_{index:04d}"
            metadata.append(
                {
                    "transcript_id": f"lbr:{source_row_id}:r00",
                    "source_collection": "lbr",
                    "source_row_id": source_row_id,
                    "rollout_index": 0,
                    "prompt": f"lbr prompt {index % 10}",
                    "world": world,
                    "condition": "verified",
                    "template_id": 38 + index % 10,
                    "split": "heldout",
                    "label": label,
                    "model": "fixture/model",
                    **({"dataset_variant": variant} if variant else {}),
                }
            )
            prompts.append(metadata[-1]["prompt"])
            transcript_ids.append(metadata[-1]["transcript_id"])
            labels.append(label)
            X = rng.normal(0, 0.25, size=(3, 8))
            X[:, 0] += 2.0
            if world_signal_layer is not None:
                X[world_signal_layer, 1] += 2.0 * label
            features.append(X)

    metadata_path = tmp_path / "transcripts.jsonl"
    metadata_path.write_text(
        "".join(json.dumps(row) + "\n" for row in metadata), encoding="utf-8"
    )
    cache_path = tmp_path / "prompt_final.npz"
    extra = {"dataset_variant": np.asarray(variant)} if variant else {}
    np.savez_compressed(
        cache_path,
        X=np.asarray(features, dtype=np.float16),
        y=np.asarray(labels, dtype=np.int8),
        prompts=np.asarray(prompts),
        model=np.asarray("fixture/model"),
        transcript_ids=np.asarray(transcript_ids),
        position=np.asarray("prompt_final"),
        label_name=np.asarray("world"),
        positive_class=np.asarray("REAL"),
        positive_label=np.asarray(1),
        negative_class=np.asarray("SHAM"),
        negative_label=np.asarray(0),
        **extra,
    )
    return cache_path, metadata_path


def _layer(layer: int, auroc: float, low: float, high: float) -> dict[str, Any]:
    ci = {
        "ci95": [low, high],
        "valid_replicates": 10,
        "excludes_chance": bool(low > 0.5 or high < 0.5),
    }
    return {
        "layer": layer,
        "validation_auroc": 0.5,
        "heldout_auroc": auroc,
        "heldout_row_bootstrap": ci,
        "heldout_template_bootstrap": dict(ci),
    }


def _synthetic_inputs(
    layers: list[dict[str, Any]],
    *,
    selected_layer: int = 0,
    lbr_auroc: float = 0.5,
    lbr_ci: tuple[float, float] = (0.4, 0.6),
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    intervals = {"layers": layers}
    primary = {
        "selected_layer": selected_layer,
        "selected_layer_heldout_test": {
            "auroc": layers[selected_layer]["heldout_auroc"]
        },
    }
    lbr = {"auroc": lbr_auroc, "row_bootstrap": {"ci95": list(lbr_ci)}}
    return intervals, primary, lbr


def test_weighted_auroc_matches_sklearn_with_ties_and_multisets() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(20, 120))
        y = rng.integers(0, 2, n)
        y[:2] = (1, 0)
        scores = np.round(rng.random(n), 2)
        assert weighted_auroc(scores, y, np.ones((1, n)))[0] == pytest.approx(
            roc_auc_score(y, scores), abs=1e-12
        )
        weights = rng.integers(0, 4, size=(1, n))
        weights[:, :2] = 1
        expanded = np.repeat(np.arange(n), weights[0])
        assert weighted_auroc(scores, y, weights)[0] == pytest.approx(
            roc_auc_score(y[expanded], scores[expanded]), abs=1e-12
        )


def test_bootstrap_weights_preserve_totals() -> None:
    y = np.asarray([1] * 30 + [0] * 70)
    rows = stratified_row_weights(y, 50, np.random.default_rng(7))
    assert rows.shape == (50, 100)
    assert (rows[:, :30].sum(axis=1) == 30).all()
    assert (rows[:, 30:].sum(axis=1) == 70).all()
    templates = [str(index // 10) for index in range(100)]
    clusters = template_cluster_weights(templates, 50, np.random.default_rng(7))
    assert clusters.shape == (50, 100)
    assert (clusters.sum(axis=1) == 100).all()
    assert (
        clusters.reshape(50, 10, 10) == clusters.reshape(50, 10, 10)[:, :, :1]
    ).all()


def test_prompt_identity_counts_pairs_and_flags_divergence(tmp_path: Path) -> None:
    cache_path, metadata_path = write_leak_free_fixture(tmp_path)
    metadata = load_metadata(metadata_path, load_activation_cache(cache_path))
    identity = check_prompt_identity(metadata)
    assert identity["passed"] is True
    assert identity["main_labelled_pairs"] == 45
    assert identity["main_byte_identical_pairs"] == 45
    assert identity["lbr_real_prompts_without_sham_twin"] == 0

    metadata[0].raw["world"] = "REAL"
    diverged = [
        row
        if row.transcript_id != metadata[0].transcript_id
        else type(row)(
            transcript_id=row.transcript_id,
            condition=row.condition,
            template_id=row.template_id,
            split=row.split,
            prompt=row.prompt + " (leaked)",
            raw=row.raw,
        )
        for row in metadata
    ]
    flagged = check_prompt_identity(diverged)
    assert flagged["passed"] is False
    assert flagged["main_byte_identical_pairs"] == 44
    assert flagged["main_differing_pairs"][0]["distinct_real_prompts"] == 2


def test_build_verdict_fails_closed_on_any_layer_not_only_the_selected_one() -> None:
    # Validation selected layer 0, which sits at chance; layer 2 leaks.
    layers = [
        _layer(0, 0.51, 0.47, 0.55),
        _layer(1, 0.52, 0.48, 0.56),
        _layer(2, 0.71, 0.63, 0.78),
    ]
    verdict = build_verdict(*_synthetic_inputs(layers, selected_layer=0))
    assert verdict["status"] == STATUS_ABOVE_CHANCE
    assert verdict["materially_above_chance"] is True
    assert verdict["selected_layer"] == 0
    assert verdict["selected_layer_materially_above_chance"] is False
    assert verdict["layers_materially_above_chance"] == [2]
    assert verdict["max_layer"] == 2
    assert verdict["max_layer_heldout_auroc"] == pytest.approx(0.71)
    assert verdict["max_layer_heldout_ci95_row_bootstrap"] == [0.63, 0.78]
    assert verdict["layers_with_row_ci_excluding_chance"] == [2]
    assert verdict["evaluated_layers"] == 3


def test_build_verdict_rule_is_conjunctive_and_ties_choose_lowest_layer() -> None:
    # AUROC above the threshold but a CI touching chance does not flag, and a
    # CI clear of chance below the threshold does not flag either.
    layers = [
        _layer(0, 0.50, 0.46, 0.54),
        _layer(1, 0.65, 0.49, 0.80),
        _layer(2, 0.58, 0.53, 0.63),
        _layer(3, 0.65, 0.47, 0.81),
    ]
    verdict = build_verdict(*_synthetic_inputs(layers, selected_layer=0))
    assert verdict["status"] == STATUS_AT_CHANCE
    assert verdict["materially_above_chance"] is False
    assert verdict["layers_materially_above_chance"] == []
    assert verdict["layers_with_row_ci_excluding_chance"] == [2]
    assert verdict["max_layer"] == 1
    assert verdict["max_layer_heldout_auroc"] == pytest.approx(0.65)


def test_build_verdict_fails_closed_on_low_base_rate_evaluation() -> None:
    layers = [_layer(0, 0.50, 0.46, 0.54), _layer(1, 0.51, 0.47, 0.55)]
    verdict = build_verdict(
        *_synthetic_inputs(layers, lbr_auroc=0.72, lbr_ci=(0.61, 0.83))
    )
    assert verdict["lbr_materially_above_chance"] is True
    assert verdict["layers_materially_above_chance"] == []
    assert verdict["status"] == STATUS_ABOVE_CHANCE
    at_chance = build_verdict(
        *_synthetic_inputs(layers, lbr_auroc=0.54, lbr_ci=(0.37, 0.70))
    )
    assert at_chance["lbr_materially_above_chance"] is False
    assert at_chance["status"] == STATUS_AT_CHANCE


def test_build_verdict_refuses_empty_or_misindexed_layers() -> None:
    with pytest.raises(ValueError, match="no evaluated layers"):
        build_verdict(
            {"layers": []},
            {"selected_layer": 0, "selected_layer_heldout_test": {"auroc": 0.5}},
            {"auroc": 0.5, "row_bootstrap": {"ci95": [0.4, 0.6]}},
        )
    layers = [_layer(1, 0.5, 0.46, 0.54), _layer(0, 0.5, 0.46, 0.54)]
    with pytest.raises(ValueError, match="indexed by layer id"):
        build_verdict(*_synthetic_inputs(layers, selected_layer=0))


def test_consistency_check_runs_and_reports_chance(tmp_path: Path) -> None:
    cache_path, metadata_path = write_leak_free_fixture(tmp_path)
    output = tmp_path / "results"
    result = run_consistency_check(
        cache_path,
        metadata_path,
        output,
        seed=7,
        C=0.1,
        max_iter=400,
        val_fraction=0.2,
        bootstrap_replicates=200,
    )
    assert result["format_version"] == 2
    assert result["dataset_variant"] == "leak_free"
    assert result["cache"]["dataset_variant"] == "leak_free"
    assert result["prompt_identity"]["passed"] is True
    primary = result["primary_world_probe"]
    assert primary["split"]["heldout_test_samples"] == 3 * 2 * 2 * 4
    layers = primary["per_layer"]["layers"]
    assert len(layers) == 3
    for layer in layers:
        low, high = layer["heldout_row_bootstrap"]["ci95"]
        assert low <= layer["heldout_auroc"] <= high
        assert layer["heldout_template_bootstrap"]["valid_replicates"] > 0
    lbr = result["low_base_rate"]
    assert lbr["counts"] == {"REAL": 10, "SHAM": 1000}
    assert lbr["auroc"] == pytest.approx(lbr["fixed_threshold"]["auroc"])
    assert lbr["provenance"] == LBR_PROVENANCE
    assert "same-generator" in lbr["provenance_note"]
    assert "independent_lbr" in lbr["sealed_key_note"]
    verdict = result["verdict"]
    assert verdict["status"] == STATUS_AT_CHANCE
    assert verdict["materially_above_chance"] is False
    assert verdict["layers_materially_above_chance"] == []
    assert verdict["evaluated_layers"] == 3
    assert verdict["rule"].startswith("fail-closed over every evaluated layer")
    best = max(layers, key=lambda layer: (layer["heldout_auroc"], -layer["layer"]))
    assert verdict["max_layer"] == best["layer"]
    assert verdict["max_layer_heldout_auroc"] == pytest.approx(best["heldout_auroc"])
    assert verdict["max_layer_heldout_auroc"] >= verdict["selected_layer_heldout_auroc"]
    assert "does not show" in verdict["scope"]
    written = json.loads((output / "consistency_check.json").read_text())
    assert written["verdict"] == verdict
    assert len(result["pairwise_condition_contrasts"]["contrasts"]) == 3

    position = json.loads((output / "position_results.json").read_text())
    assert position["dataset_variant"] == "leak_free"
    assert position["provenance"]["dataset_variant"] == "leak_free"
    assert position["provenance"]["low_base_rate_cache"] == LBR_PROVENANCE
    assert "independent_lbr" in position["provenance"]["sealed_key_note"]
    assert position["primary"]["independent_lbr"]["provenance"] == LBR_PROVENANCE
    assert position["primary"]["independent_lbr"]["empirical"] == lbr["fixed_threshold"]


def test_consistency_check_fails_closed_on_injected_world_signal(
    tmp_path: Path,
) -> None:
    cache_path, metadata_path = write_leak_free_fixture(tmp_path, world_signal_layer=2)
    result = run_consistency_check(
        cache_path,
        metadata_path,
        tmp_path / "results",
        seed=7,
        C=0.1,
        max_iter=400,
        val_fraction=0.2,
        bootstrap_replicates=200,
    )
    assert result["prompt_identity"]["passed"] is True
    verdict = result["verdict"]
    assert verdict["status"] == STATUS_ABOVE_CHANCE
    assert verdict["materially_above_chance"] is True
    assert 2 in verdict["layers_materially_above_chance"]
    assert verdict["max_layer"] == 2
    assert verdict["max_layer_heldout_auroc"] > 0.9


def test_consistency_check_refuses_unmarked_cache(tmp_path: Path) -> None:
    cache_path, metadata_path = write_leak_free_fixture(tmp_path, variant=None)
    with pytest.raises(ValueError, match="only scores 'leak_free' caches"):
        run_consistency_check(
            cache_path,
            metadata_path,
            tmp_path / "results",
            seed=7,
            C=0.1,
            max_iter=100,
            val_fraction=0.2,
            bootstrap_replicates=10,
        )
