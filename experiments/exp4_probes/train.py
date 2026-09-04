"""Train fixed-hyperparameter per-layer credibility probes on an exp5 cache."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from experiments.exp4_analysis.statistics import (
    MANIPULATION_DIRECTIONAL_RATE,
    analyze_behavior_rows,
    analyze_beyond_condition_regression,
    analyze_manipulation_rows,
)
from experiments.exp4_analysis.statistics import read_jsonl as read_analysis_jsonl

from .cache import ActivationCache, SampleMetadata, load_activation_cache, load_metadata
from .metrics import (
    FIXED_SCORE_THRESHOLD,
    binary_metrics,
    projected_low_base_rate_metrics,
)
from .results_table import read_score_file, render_head_to_head
from .splits import GroupSplits, make_group_splits

CONTRASTS = (
    ("claimed", "verified"),
    ("verified", "causally_binding"),
    ("claimed", "causally_binding"),
)
PRIMARY_CONDITIONS = frozenset({"verified", "causally_binding"})
REAL_LABEL = 1
SHAM_LABEL = 0
EXPECTED_LBR_REAL = 10
EXPECTED_LBR_SHAM = 1000
LOCKED_POSITIONS = ("receipt_final", "prompt_final")


@dataclass(frozen=True)
class FittedProbe:
    weight: np.ndarray
    intercept: float

    def scores(self, X: np.ndarray) -> np.ndarray:
        logits = np.asarray(X, dtype=np.float64) @ self.weight + self.intercept
        output = np.empty_like(logits)
        positive = logits >= 0
        output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exp_logits = np.exp(logits[~positive])
        output[~positive] = exp_logits / (1.0 + exp_logits)
        return output


def _fit_probe(
    X: np.ndarray, y: np.ndarray, *, C: float, max_iter: int, seed: int
) -> FittedProbe:
    scaler = StandardScaler().fit(X)
    scaled = scaler.transform(X)
    classifier = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=seed,
        solver="liblinear",
    ).fit(scaled, y)
    standardized_weight = classifier.coef_[0].astype(np.float64)
    weight = standardized_weight / scaler.scale_
    intercept = float(
        classifier.intercept_[0]
        - np.dot(scaler.mean_ / scaler.scale_, standardized_weight)
    )
    return FittedProbe(weight=weight, intercept=intercept)


def _random_direction(
    reference: FittedProbe, d_model: int, rng: np.random.Generator
) -> FittedProbe:
    weight = rng.standard_normal(d_model)
    weight /= np.linalg.norm(weight)
    weight *= np.linalg.norm(reference.weight)
    return FittedProbe(weight=weight, intercept=reference.intercept)


def _validate_binary_splits(y: np.ndarray, splits: GroupSplits) -> None:
    for name, indices in (
        ("train", splits.train),
        ("validation", splits.val),
        ("test", splits.test),
    ):
        if set(np.unique(y[indices])) != {0, 1}:
            raise ValueError(f"{name} split must contain both classes")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=False) + "\n")


def _contrast_slug(negative_condition: str, positive_condition: str) -> str:
    return f"{negative_condition}_vs_{positive_condition}"


def train_contrast(
    cache: ActivationCache,
    metadata: list[SampleMetadata],
    *,
    negative_condition: str,
    positive_condition: str,
    output_dir: Path,
    seed: int,
    C: float,
    max_iter: int,
    val_fraction: float,
) -> dict:
    selected_global = np.asarray(
        [
            index
            for index, row in enumerate(metadata)
            if row.condition in {negative_condition, positive_condition}
            and row.raw.get("source_collection", "main") == "main"
        ],
        dtype=np.int64,
    )
    if len(selected_global) == 0:
        raise ValueError(
            f"no samples found for {negative_condition} vs {positive_condition}"
        )
    selected_metadata = [metadata[index] for index in selected_global]
    y = np.asarray(
        [int(row.condition == positive_condition) for row in selected_metadata],
        dtype=np.int64,
    )
    splits = make_group_splits(selected_metadata, seed=seed, val_fraction=val_fraction)
    _validate_binary_splits(y, splits)

    n_layers = cache.n_layers
    weights = np.empty((n_layers, cache.d_model), dtype=np.float32)
    intercepts = np.empty(n_layers, dtype=np.float64)
    layer_results = []
    test_scores_by_layer: list[np.ndarray] = []
    rng = np.random.default_rng(seed)

    for layer in range(n_layers):
        X = np.asarray(cache.X[selected_global, layer, :], dtype=np.float64)
        probe = _fit_probe(
            X[splits.train], y[splits.train], C=C, max_iter=max_iter, seed=seed
        )
        weights[layer] = probe.weight.astype(np.float32)
        intercepts[layer] = probe.intercept
        val_scores = probe.scores(X[splits.val])
        test_scores = probe.scores(X[splits.test])
        test_scores_by_layer.append(test_scores)

        shuffled_y = rng.permutation(y[splits.train])
        shuffled = _fit_probe(
            X[splits.train], shuffled_y, C=C, max_iter=max_iter, seed=seed + layer + 1
        )
        random_probe = _random_direction(probe, cache.d_model, rng)
        layer_results.append(
            {
                "layer": layer,
                "validation": binary_metrics(y[splits.val], val_scores),
                "test": binary_metrics(y[splits.test], test_scores),
                "controls": {
                    "shuffled_label_validation_auroc": binary_metrics(
                        y[splits.val], shuffled.scores(X[splits.val])
                    )["auroc"],
                    "shuffled_label_test_auroc": binary_metrics(
                        y[splits.test], shuffled.scores(X[splits.test])
                    )["auroc"],
                    "random_direction_validation_auroc": binary_metrics(
                        y[splits.val], random_probe.scores(X[splits.val])
                    )["auroc"],
                    "random_direction_test_auroc": binary_metrics(
                        y[splits.test], random_probe.scores(X[splits.test])
                    )["auroc"],
                },
            }
        )

    # The held-out test set is never used for layer choice.
    selected_layer = max(
        range(n_layers),
        key=lambda layer: (layer_results[layer]["validation"]["auroc"], -layer),
    )
    selected_test_scores = test_scores_by_layer[selected_layer]
    selected_test_y = y[splits.test]
    selected_test_metadata = [selected_metadata[index] for index in splits.test]
    selected_metrics = binary_metrics(selected_test_y, selected_test_scores)
    low_base_rates = projected_low_base_rate_metrics(
        selected_test_y,
        selected_test_scores,
        positive_condition=positive_condition,
        negative_condition=negative_condition,
    )
    slug = _contrast_slug(negative_condition, positive_condition)

    direction_path = output_dir / "directions" / f"{slug}.npz"
    direction_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        direction_path,
        weights=weights,
        intercepts=intercepts,
        selected_layer=np.asarray(selected_layer),
        selected_weight=weights[selected_layer],
        selected_intercept=np.asarray(intercepts[selected_layer]),
        negative_condition=np.asarray(negative_condition),
        positive_condition=np.asarray(positive_condition),
        model=np.asarray(cache.model),
    )

    score_path = output_dir / "scores" / f"probe_{slug}.jsonl"
    score_rows = [
        {
            "transcript_id": row.transcript_id,
            "score": float(score),
            "condition": row.condition,
        }
        for row, score in zip(selected_test_metadata, selected_test_scores, strict=True)
    ]
    _write_jsonl(score_path, score_rows)

    false_positive_rows = [
        {
            "transcript_id": row.transcript_id,
            "score": float(score),
            "observed_condition": row.condition,
            "predicted_condition": positive_condition,
            "template_id": row.template_id,
            "prompt": row.prompt,
        }
        for row, label, score in zip(
            selected_test_metadata, selected_test_y, selected_test_scores, strict=True
        )
        if label == 0 and score >= FIXED_SCORE_THRESHOLD
    ]
    false_positive_rows.sort(key=lambda row: row["score"], reverse=True)
    false_positive_path = output_dir / "false_positives" / f"{slug}.jsonl"
    _write_jsonl(false_positive_path, false_positive_rows)

    scaffold = render_head_to_head(
        positive_condition=positive_condition,
        negative_condition=negative_condition,
        probe_rows=read_score_file(score_path),
    )
    table_path = output_dir / "tables" / f"{slug}.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(scaffold, encoding="utf-8")

    return {
        "contrast": slug,
        "negative_condition": negative_condition,
        "positive_condition": positive_condition,
        "split": {
            "group_key": "template_id",
            "train_groups": list(splits.train_groups),
            "validation_groups": list(splits.val_groups),
            "test_groups": list(splits.test_groups),
            "train_samples": int(len(splits.train)),
            "validation_samples": int(len(splits.val)),
            "test_samples": int(len(splits.test)),
        },
        "selection_rule": "maximum validation AUROC; ties choose the lowest layer index",
        "selected_layer": selected_layer,
        "selected_layer_test": selected_metrics,
        "low_base_rates": low_base_rates,
        "false_positive_count": len(false_positive_rows),
        "layers": layer_results,
        "artifacts": {
            "directions": str(direction_path),
            "scores": str(score_path),
            "false_positives": str(false_positive_path),
            "head_to_head_scaffold": str(table_path),
        },
    }


def _world_label(row: SampleMetadata) -> int:
    world = str(row.raw.get("world", ""))
    label = row.raw.get("label")
    if world == "REAL" and label == REAL_LABEL:
        return REAL_LABEL
    if world == "SHAM" and label == SHAM_LABEL:
        return SHAM_LABEL
    raise ValueError(
        f"{row.transcript_id} must name world REAL=1 or SHAM=0 consistently; "
        f"got world={world!r}, label={label!r}"
    )


def _canonical_lbr_report(labels: np.ndarray, scores: np.ndarray) -> dict:
    positives = int(np.sum(labels == REAL_LABEL))
    negatives = int(np.sum(labels == SHAM_LABEL))
    if (positives, negatives) != (EXPECTED_LBR_REAL, EXPECTED_LBR_SHAM):
        raise ValueError(
            "independent LBR cache must contain exactly 10 REAL and 1000 SHAM rows; "
            f"got {positives} REAL and {negatives} SHAM"
        )
    empirical = binary_metrics(labels, scores)

    def project(
        measured: dict,
        *,
        rare_class: str,
        common_class: str,
        name: str,
    ) -> dict:
        prevalence = 1 / 51
        tpr = float(measured["recall"])
        fpr = float(measured["false_positive_rate"])
        denominator = prevalence * tpr + (1 - prevalence) * fpr
        return {
            "name": name,
            "rare_class": rare_class,
            "common_class": common_class,
            "rare_count": 1,
            "common_count": 50,
            "rare_prevalence": prevalence,
            "score_threshold": FIXED_SCORE_THRESHOLD,
            "measured_tpr": tpr,
            "measured_fpr": fpr,
            "projected_precision": prevalence * tpr / denominator
            if denominator
            else None,
        }

    # Use the same fixed REAL classifier's confusion rates for the inverse
    # projection. This preserves the REAL-positive tie rule at score 0.5.
    sham_metrics = {
        "recall": 1.0 - float(empirical["false_positive_rate"]),
        "false_positive_rate": 1.0 - float(empirical["recall"]),
    }
    return {
        "status": "available",
        "trained_on_lbr": False,
        "positive_class": "REAL",
        "positive_label": REAL_LABEL,
        "negative_class": "SHAM",
        "negative_label": SHAM_LABEL,
        "actual_counts": {"REAL": positives, "SHAM": negatives},
        "actual_REAL_prevalence": positives / (positives + negatives),
        "fixed_score_threshold": FIXED_SCORE_THRESHOLD,
        "empirical": empirical,
        "projected": [
            project(
                empirical,
                rare_class="REAL",
                common_class="SHAM",
                name="canonical_projected_REAL_to_SHAM_1_to_50",
            ),
            project(
                sham_metrics,
                rare_class="SHAM",
                common_class="REAL",
                name="canonical_projected_SHAM_to_REAL_1_to_50",
            ),
        ],
    }


def train_primary_world_probe(
    cache: ActivationCache,
    metadata: list[SampleMetadata],
    *,
    output_dir: Path,
    seed: int,
    C: float,
    max_iter: int,
    val_fraction: float,
) -> dict:
    """Canonical REAL=1 vs SHAM=0 b+c sweep with same-generator LBR evaluation.

    "independent" in the sealed `independent_lbr` key and in the messages below
    is a fixed identifier, not a claim about the cache's distribution; see
    docs/errata.md, E1.
    """

    if cache.label_name not in {None, "world"}:
        raise ValueError(
            f"canonical cache label_name must be 'world', got {cache.label_name!r}"
        )
    if cache.positive_class not in {None, "REAL"} or cache.positive_label not in {
        None,
        REAL_LABEL,
    }:
        raise ValueError("canonical cache must name positive class REAL with label 1")
    if cache.negative_class not in {None, "SHAM"} or cache.negative_label not in {
        None,
        SHAM_LABEL,
    }:
        raise ValueError("canonical cache must name negative class SHAM with label 0")
    main_global = np.asarray(
        [
            index
            for index, row in enumerate(metadata)
            if row.raw.get("source_collection", "main") == "main"
            and row.condition in PRIMARY_CONDITIONS
        ],
        dtype=np.int64,
    )
    if not len(main_global):
        raise ValueError("no main b+c rows found for canonical REAL-vs-SHAM probe")
    main_metadata = [metadata[index] for index in main_global]
    y = np.asarray([_world_label(row) for row in main_metadata], dtype=np.int64)
    if not np.array_equal(np.asarray(cache.y[main_global], dtype=np.int64), y):
        raise ValueError("canonical main cache y does not match metadata world labels")
    splits = make_group_splits(main_metadata, seed=seed, val_fraction=val_fraction)
    _validate_binary_splits(y, splits)
    weights = np.empty((cache.n_layers, cache.d_model), dtype=np.float32)
    intercepts = np.empty(cache.n_layers, dtype=np.float64)
    probes: list[FittedProbe] = []
    layers = []
    rng = np.random.default_rng(seed)
    for layer in range(cache.n_layers):
        X = np.asarray(cache.X[main_global, layer, :], dtype=np.float64)
        probe = _fit_probe(
            X[splits.train], y[splits.train], C=C, max_iter=max_iter, seed=seed
        )
        probes.append(probe)
        weights[layer] = probe.weight.astype(np.float32)
        intercepts[layer] = probe.intercept
        validation_scores = probe.scores(X[splits.val])
        test_scores = probe.scores(X[splits.test])
        shuffled = _fit_probe(
            X[splits.train],
            rng.permutation(y[splits.train]),
            C=C,
            max_iter=max_iter,
            seed=seed + layer + 1,
        )
        random_probe = _random_direction(probe, cache.d_model, rng)
        layers.append(
            {
                "layer": layer,
                "validation": binary_metrics(y[splits.val], validation_scores),
                "heldout_test": binary_metrics(y[splits.test], test_scores),
                "controls": {
                    "shuffled_label_validation_auroc": binary_metrics(
                        y[splits.val], shuffled.scores(X[splits.val])
                    )["auroc"],
                    "shuffled_label_heldout_test_auroc": binary_metrics(
                        y[splits.test], shuffled.scores(X[splits.test])
                    )["auroc"],
                    "random_direction_validation_auroc": binary_metrics(
                        y[splits.val], random_probe.scores(X[splits.val])
                    )["auroc"],
                    "random_direction_heldout_test_auroc": binary_metrics(
                        y[splits.test], random_probe.scores(X[splits.test])
                    )["auroc"],
                },
            }
        )
    selected_layer = max(
        range(cache.n_layers),
        key=lambda layer: (layers[layer]["validation"]["auroc"], -layer),
    )
    selected_probe = probes[selected_layer]
    selected_X = np.asarray(cache.X[main_global, selected_layer, :], dtype=np.float64)
    heldout_scores = selected_probe.scores(selected_X[splits.test])
    heldout_metrics = binary_metrics(y[splits.test], heldout_scores)
    heldout_metadata = [main_metadata[index] for index in splits.test]

    directions_path = output_dir / "directions" / "world_REAL_vs_SHAM_b_plus_c.npz"
    directions_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directions_path,
        weights=weights,
        intercepts=intercepts,
        selected_layer=np.asarray(selected_layer),
        selected_weight=weights[selected_layer],
        selected_intercept=np.asarray(intercepts[selected_layer]),
        label_name=np.asarray("world"),
        positive_class=np.asarray("REAL"),
        positive_label=np.asarray(REAL_LABEL),
        negative_class=np.asarray("SHAM"),
        negative_label=np.asarray(SHAM_LABEL),
        included_conditions=np.asarray(["verified", "causally_binding"]),
        model=np.asarray(cache.model),
    )
    heldout_score_path = output_dir / "scores" / "primary_heldout_REAL_vs_SHAM.jsonl"
    _write_jsonl(
        heldout_score_path,
        [
            {
                "transcript_id": row.transcript_id,
                "source_collection": row.raw.get("source_collection", "main"),
                "source_row_id": row.raw.get("source_row_id", row.transcript_id),
                "rollout_index": row.raw.get("rollout_index", 0),
                "prompt": row.prompt,
                "world": row.raw["world"],
                "label": _world_label(row),
                "condition": row.condition,
                "template_id": row.template_id,
                "split": row.split,
                "model": cache.model,
                "probe_score_REAL": float(score),
            }
            for row, score in zip(heldout_metadata, heldout_scores, strict=True)
        ],
    )
    strict_heldout_score_path = (
        output_dir / "scores" / "primary_heldout_REAL_vs_SHAM.strict.jsonl"
    )
    _write_jsonl(
        strict_heldout_score_path,
        [
            {
                "transcript_id": row.transcript_id,
                "score": float(score),
                "condition": str(row.raw["world"]).upper(),
            }
            for row, score in zip(heldout_metadata, heldout_scores, strict=True)
        ],
    )
    all_main_scores = selected_probe.scores(selected_X)
    all_score_path = output_dir / "scores" / "primary_all_main_REAL_vs_SHAM.jsonl"
    _write_jsonl(
        all_score_path,
        [
            {
                "transcript_id": row.transcript_id,
                "source_collection": row.raw.get("source_collection", "main"),
                "source_row_id": row.raw.get("source_row_id", row.transcript_id),
                "rollout_index": row.raw.get("rollout_index", 0),
                "prompt": row.prompt,
                "world": row.raw["world"],
                "label": _world_label(row),
                "condition": row.condition,
                "template_id": row.template_id,
                "split": row.split,
                "model": cache.model,
                "probe_score_REAL": float(score),
            }
            for row, score in zip(main_metadata, all_main_scores, strict=True)
        ],
    )

    lbr_global = np.asarray(
        [
            index
            for index, row in enumerate(metadata)
            if row.raw.get("source_collection") == "lbr"
        ],
        dtype=np.int64,
    )
    if not len(lbr_global):
        raise ValueError("independent LBR rows are absent from the finalized cache")
    lbr_metadata = [metadata[index] for index in lbr_global]
    if any(row.condition != "verified" for row in lbr_metadata):
        raise ValueError("all independent LBR rows must use verified condition b")
    lbr_y = np.asarray([_world_label(row) for row in lbr_metadata], dtype=np.int64)
    if not np.array_equal(np.asarray(cache.y[lbr_global], dtype=np.int64), lbr_y):
        raise ValueError("independent LBR cache y does not match metadata world labels")
    lbr_scores = selected_probe.scores(
        np.asarray(cache.X[lbr_global, selected_layer, :], dtype=np.float64)
    )
    lbr_report = _canonical_lbr_report(lbr_y, lbr_scores)
    lbr_score_path = output_dir / "scores" / "independent_lbr_REAL_vs_SHAM.jsonl"
    _write_jsonl(
        lbr_score_path,
        [
            {
                "transcript_id": row.transcript_id,
                "source_collection": "lbr",
                "source_row_id": row.raw.get("source_row_id", row.transcript_id),
                "world": row.raw["world"],
                "label": _world_label(row),
                "condition": row.condition,
                "template_id": row.template_id,
                "probe_score_REAL": float(score),
            }
            for row, score in zip(lbr_metadata, lbr_scores, strict=True)
        ],
    )
    return {
        "task": "canonical_credibility_world_REAL_vs_SHAM",
        "label_name": "world",
        "positive_class": "REAL",
        "positive_label": REAL_LABEL,
        "negative_class": "SHAM",
        "negative_label": SHAM_LABEL,
        "included_conditions": ["verified", "causally_binding"],
        "source_collection": "main",
        "split": {
            "group_key": "template_id",
            "train_groups": list(splits.train_groups),
            "validation_groups": list(splits.val_groups),
            "heldout_test_groups": list(splits.test_groups),
            "train_samples": int(len(splits.train)),
            "validation_samples": int(len(splits.val)),
            "heldout_test_samples": int(len(splits.test)),
        },
        "selection_rule": "maximum validation AUROC; exact ties choose lowest layer",
        "heldout_test_never_selects_layer": True,
        "selected_layer": selected_layer,
        "selected_layer_heldout_test": heldout_metrics,
        "layers": layers,
        "independent_lbr": lbr_report,
        "artifacts": {
            "directions_all_layers_and_selected": str(directions_path),
            "heldout_scores": str(heldout_score_path),
            "heldout_scores_strict_interchange": str(strict_heldout_score_path),
            "all_main_scores_for_fixed_regression": str(all_score_path),
            "lbr_scores": str(lbr_score_path),
        },
    }


def run_pipeline(
    cache_path: str | Path,
    metadata_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 7,
    C: float = 0.1,
    max_iter: int = 2000,
    val_fraction: float = 0.2,
) -> dict:
    cache = load_activation_cache(cache_path)
    metadata = load_metadata(metadata_path, cache)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        train_contrast(
            cache,
            metadata,
            negative_condition=negative,
            positive_condition=positive,
            output_dir=output_dir,
            seed=seed,
            C=C,
            max_iter=max_iter,
            val_fraction=val_fraction,
        )
        for negative, positive in CONTRASTS
    ]
    summary = {
        "format_version": 1,
        "model": cache.model,
        "cache_contract": {
            "keys": ["X", "y", "prompts", "model"],
            "X_shape": list(cache.X.shape),
            "source": "experiments/exp5_model_setup/modal_smoke.py",
        },
        "fixed_hyperparameters": {
            "probe": "L2 logistic regression on standardized activations",
            "C": C,
            "max_iter": max_iter,
            "seed": seed,
            "score_threshold": FIXED_SCORE_THRESHOLD,
            "validation_fraction_of_nonheldout_template_groups": val_fraction,
        },
        "pre_registered_criteria": {
            "h1_supported": "held-out AUROC >= 0.85",
            "h1_not_supported": "held-out AUROC < 0.65",
            "h1_inconclusive": "0.65 <= held-out AUROC < 0.85",
            "h2_earlier_lock_quantitative": (
                "at matched false-positive rate on 1:100 mixture, probe catches >= half "
                "of cases the CoT monitor misses"
            ),
            "h2_earlier_lock_qualitative": (
                "or a clean disagreement set where probe is right; external adjudication required"
            ),
            "h3_canonical_v1": "at matched recall on 1:50 mixture, probe precision > CoT precision",
        },
        "base_rate_ambiguity": (
            "Issue #4's earlier lock specifies positive:negative = 10:1000 (1:100); "
            "canonical v1 specifies 1:50 and its inverse. All readings are emitted."
        ),
        "contrasts": results,
    }
    _write_json(output_dir / "results.json", summary)
    return summary


def _run_locked_position(
    cache: ActivationCache,
    metadata: list[SampleMetadata],
    output_dir: Path,
    *,
    seed: int,
    C: float,
    max_iter: int,
    val_fraction: float,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    primary = train_primary_world_probe(
        cache,
        metadata,
        output_dir=output_dir,
        seed=seed,
        C=C,
        max_iter=max_iter,
        val_fraction=val_fraction,
    )
    pairwise = [
        train_contrast(
            cache,
            metadata,
            negative_condition=negative,
            positive_condition=positive,
            output_dir=output_dir / "pairwise_conditions",
            seed=seed,
            C=C,
            max_iter=max_iter,
            val_fraction=val_fraction,
        )
        for negative, positive in CONTRASTS
    ]
    result = {
        "position": cache.position,
        "model": cache.model,
        "primary": primary,
        "required_pairwise_condition_separations": pairwise,
    }
    _write_json(output_dir / "position_results.json", result)
    return result


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing sealed {description}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid sealed {description} JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"sealed {description} at {path} must be a JSON object")
    return value


def _require_object(value: object, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be an object")
    return value


def _validate_manipulation_result(manipulation: dict[str, Any]) -> None:
    if manipulation.get("status") not in {"available", "unavailable"}:
        raise ValueError("sealed manipulation result has invalid status")
    if manipulation.get("locked_rule") != (
        "REAL probability > SHAM probability for >=70% of paired prompts"
    ):
        raise ValueError("sealed manipulation result does not use the locked rule")
    full_set = _require_object(
        manipulation.get("full_condition_set"),
        "sealed manipulation full_condition_set",
    )
    fallback = _require_object(
        manipulation.get("declared_evidence_bearing_fallback_b_plus_c"),
        "sealed manipulation b+c fallback",
    )
    for description, result in (("full condition set", full_set), ("b+c", fallback)):
        if result.get("status") not in {"available", "unavailable"}:
            raise ValueError(f"sealed manipulation {description} has invalid status")
        if not isinstance(result.get("gate_passed"), bool):
            raise ValueError(
                f"sealed manipulation {description} gate_passed must be boolean"
            )
    if (
        fallback["status"] == "available"
        and fallback.get("threshold") != MANIPULATION_DIRECTIONAL_RATE
    ):
        raise ValueError("sealed manipulation b+c threshold is not the locked 0.70")


def _validate_rich_probe_rows(
    path: str | Path,
    *,
    model: str,
    required_split: str | None = None,
) -> list[dict[str, Any]]:
    rows = read_analysis_jsonl(path)
    if not rows:
        raise ValueError(f"{path} contains no probe score rows")
    transcript_ids = []
    required_fields = (
        "transcript_id",
        "source_collection",
        "source_row_id",
        "rollout_index",
        "prompt",
        "world",
        "label",
        "condition",
        "template_id",
        "split",
        "model",
        "probe_score_REAL",
    )
    for index, row in enumerate(rows, start=1):
        missing = [field for field in required_fields if field not in row]
        if missing:
            raise ValueError(f"{path}:{index} is missing fields {missing}")
        transcript_id = row["transcript_id"]
        if not isinstance(transcript_id, str) or not transcript_id:
            raise ValueError(f"{path}:{index} transcript_id must be non-empty")
        transcript_ids.append(transcript_id)
        if row["source_collection"] != "main":
            raise ValueError(f"{path}:{index} must be a main score row")
        if not isinstance(row["source_row_id"], str) or not row["source_row_id"]:
            raise ValueError(f"{path}:{index} source_row_id must be non-empty")
        if not isinstance(row["prompt"], str):
            raise ValueError(f"{path}:{index} prompt must be a string")
        world = row["world"]
        expected_label = REAL_LABEL if world == "REAL" else SHAM_LABEL
        if world not in {"REAL", "SHAM"} or type(row["label"]) is not int:
            raise ValueError(f"{path}:{index} must name world REAL=1 or SHAM=0")
        if row["label"] != expected_label:
            raise ValueError(f"{path}:{index} must name world REAL=1 or SHAM=0")
        if row["condition"] not in PRIMARY_CONDITIONS:
            raise ValueError(f"{path}:{index} is outside locked b+c conditions")
        if row["model"] != model:
            raise ValueError(f"{path}:{index} model does not match position result")
        if required_split is not None and row["split"] != required_split:
            raise ValueError(f"{path}:{index} must use split={required_split!r}")
        raw_score = row["probe_score_REAL"]
        if isinstance(raw_score, bool) or not isinstance(raw_score, Real):
            raise ValueError(f"{path}:{index} probe_score_REAL must be numeric")
        score = float(raw_score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(
                f"{path}:{index} probe_score_REAL must be finite and in [0, 1]"
            )
    if len(transcript_ids) != len(set(transcript_ids)):
        raise ValueError(f"{path} contains duplicate transcript_id values")
    return rows


def _validate_position_result(
    expected_position: str,
    result: dict[str, Any],
) -> tuple[list[Any], list[dict[str, Any]]]:
    if result.get("position") != expected_position:
        raise ValueError(
            f"sealed {expected_position} result has position={result.get('position')!r}"
        )
    model = result.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError(f"sealed {expected_position} result has invalid model")
    primary = _require_object(
        result.get("primary"), f"sealed {expected_position} primary result"
    )
    class_contract = (
        primary.get("label_name"),
        primary.get("positive_class"),
        primary.get("positive_label"),
        primary.get("negative_class"),
        primary.get("negative_label"),
    )
    if class_contract != ("world", "REAL", REAL_LABEL, "SHAM", SHAM_LABEL):
        raise ValueError(
            f"sealed {expected_position} result must name world REAL=1 and SHAM=0"
        )
    if (
        primary.get("task") != "canonical_credibility_world_REAL_vs_SHAM"
        or primary.get("included_conditions") != ["verified", "causally_binding"]
        or primary.get("source_collection") != "main"
    ):
        raise ValueError(f"sealed {expected_position} primary scope is not locked b+c")
    lbr = _require_object(
        primary.get("independent_lbr"),
        f"sealed {expected_position} independent LBR result",
    )
    lbr_contract = (
        lbr.get("positive_class"),
        lbr.get("positive_label"),
        lbr.get("negative_class"),
        lbr.get("negative_label"),
        lbr.get("fixed_score_threshold"),
    )
    if lbr_contract != ("REAL", REAL_LABEL, "SHAM", SHAM_LABEL, FIXED_SCORE_THRESHOLD):
        raise ValueError(
            f"sealed {expected_position} LBR class/threshold contract differs"
        )

    heldout_metrics = _require_object(
        primary.get("selected_layer_heldout_test"),
        f"sealed {expected_position} heldout metrics",
    )
    auroc = heldout_metrics.get("auroc")
    if (
        isinstance(auroc, bool)
        or not isinstance(auroc, Real)
        or not math.isfinite(float(auroc))
        or not 0.0 <= float(auroc) <= 1.0
    ):
        raise ValueError(f"sealed {expected_position} heldout AUROC is invalid")
    expected_h1 = (
        "supported"
        if auroc >= 0.85
        else "not_supported"
        if auroc < 0.65
        else "inconclusive"
    )
    if heldout_metrics.get("h1_verdict") != expected_h1:
        raise ValueError(f"sealed {expected_position} H1 verdict disagrees with AUROC")

    artifacts = _require_object(
        primary.get("artifacts"), f"sealed {expected_position} artifact map"
    )
    required_artifacts = (
        "heldout_scores",
        "heldout_scores_strict_interchange",
        "all_main_scores_for_fixed_regression",
    )
    for name in required_artifacts:
        if not isinstance(artifacts.get(name), str) or not artifacts[name]:
            raise ValueError(f"sealed {expected_position} artifact {name!r} is missing")

    strict_rows = read_score_file(artifacts["heldout_scores_strict_interchange"])
    heldout_rows = _validate_rich_probe_rows(
        artifacts["heldout_scores"], model=model, required_split="heldout"
    )
    if len(strict_rows) != len(heldout_rows):
        raise ValueError(
            f"sealed {expected_position} strict and rich heldout score counts differ"
        )
    for strict, rich in zip(strict_rows, heldout_rows, strict=True):
        if (
            strict.transcript_id != rich["transcript_id"]
            or strict.condition != rich["world"]
            or strict.score != float(rich["probe_score_REAL"])
        ):
            raise ValueError(
                f"sealed {expected_position} strict heldout scores are not aligned"
            )
    split = _require_object(
        primary.get("split"), f"sealed {expected_position} split result"
    )
    if split.get("heldout_test_samples") != len(strict_rows):
        raise ValueError(
            f"sealed {expected_position} heldout score count disagrees with split"
        )

    all_main_rows = _validate_rich_probe_rows(
        artifacts["all_main_scores_for_fixed_regression"], model=model
    )
    all_main_by_id = {row["transcript_id"]: row for row in all_main_rows}
    for heldout in heldout_rows:
        aligned = all_main_by_id.get(heldout["transcript_id"])
        if aligned is None or any(
            aligned[field] != heldout[field]
            for field in (
                "source_collection",
                "source_row_id",
                "rollout_index",
                "prompt",
                "world",
                "label",
                "condition",
                "template_id",
                "split",
                "model",
            )
        ):
            raise ValueError(
                f"sealed {expected_position} heldout and regression identities are not aligned"
            )
    return strict_rows, all_main_rows


def _validate_locked_position_results(
    positions: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any], dict[str, list[dict[str, Any]]]]:
    if set(positions) != set(LOCKED_POSITIONS):
        raise ValueError(
            "locked checkpoint requires receipt_final and prompt_final results"
        )
    strict_probe_rows = {}
    regression_score_rows = {}
    for position in LOCKED_POSITIONS:
        strict_probe_rows[position], regression_score_rows[position] = (
            _validate_position_result(position, positions[position])
        )
    models = {positions[position]["model"] for position in LOCKED_POSITIONS}
    if len(models) != 1:
        raise ValueError("receipt_final and prompt_final position result models differ")

    receipt_rows = strict_probe_rows["receipt_final"]
    prompt_rows = strict_probe_rows["prompt_final"]
    receipt_contract = [(row.transcript_id, row.condition) for row in receipt_rows]
    prompt_contract = [(row.transcript_id, row.condition) for row in prompt_rows]
    if receipt_contract != prompt_contract:
        raise ValueError(
            "receipt_final and prompt_final strict heldout score rows are not aligned"
        )
    head_to_head_inputs = {
        "status": "available",
        "positive_class": "REAL",
        "negative_class": "SHAM",
        "heldout_b_plus_c_transcripts": len(receipt_rows),
        "cot_exact_id_source": positions["receipt_final"]["primary"]["artifacts"][
            "heldout_scores_strict_interchange"
        ],
        "probe_scores": {
            position: positions[position]["primary"]["artifacts"][
                "heldout_scores_strict_interchange"
            ]
            for position in LOCKED_POSITIONS
        },
    }
    return models.pop(), head_to_head_inputs, regression_score_rows


def _finalize_locked_checkpoint(
    *,
    manipulation: dict[str, Any],
    positions: dict[str, dict[str, Any]],
    behavior_rows: list[dict[str, Any]],
    output_dir: Path,
    seed: int,
    C: float,
    max_iter: int,
    val_fraction: float,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    """Validate sealed inputs, then write only the shared tail artifacts."""

    _validate_manipulation_result(manipulation)
    model, head_to_head_inputs, score_rows = _validate_locked_position_results(
        positions
    )
    behavior = analyze_behavior_rows(
        behavior_rows, seed=seed, replicates=bootstrap_replicates
    )
    regressions = {
        position: analyze_beyond_condition_regression(
            behavior_rows,
            score_rows[position],
            seed=seed,
            replicates=bootstrap_replicates,
        )
        for position in LOCKED_POSITIONS
    }

    manipulation_fallback = manipulation["declared_evidence_bearing_fallback_b_plus_c"]
    pooled_behavior = behavior.get("pooled_b_plus_c", {})
    gates = {
        "analysis_order": [
            "manipulation_check",
            "receipt_final_probe",
            "prompt_final_probe",
            "behavior_and_beyond_condition_regression",
        ],
        "manipulation": {
            "status": manipulation_fallback.get("status", "unavailable"),
            "gate_passed": bool(manipulation_fallback.get("gate_passed", False)),
            "rule": "heldout b+c directional rate >= 0.70",
            "full_set_status": manipulation["full_condition_set"].get(
                "status", "unavailable"
            ),
        },
        "h1": {
            position: {
                "status": "available",
                "heldout_template_auroc": result["primary"][
                    "selected_layer_heldout_test"
                ]["auroc"],
                "verdict": result["primary"]["selected_layer_heldout_test"][
                    "h1_verdict"
                ],
                "thresholds": {
                    "supported": ">= 0.85",
                    "not_supported": "< 0.65",
                    "otherwise": "inconclusive",
                },
            }
            for position, result in positions.items()
        },
        "h2_behavior": {
            "status": pooled_behavior.get("status", "unavailable"),
            "verdict": pooled_behavior.get("verdict", "unavailable"),
            "effect_nats": pooled_behavior.get(
                "mean_real_minus_sham_spend_log_odds_nats"
            ),
            "bootstrap_95_ci_nats": pooled_behavior.get("bootstrap_95_ci_nats"),
            "rules": {
                "behavioral": "effect >= 0.5 nats and CI excludes 0",
                "flat": "CI within +/- 0.1 nats",
            },
        },
        "beyond_condition_regression": {
            position: {"status": result.get("status", "unavailable")}
            for position, result in regressions.items()
        },
    }
    summary = {
        "format_version": 2,
        "checkpoint": "Wave 2 post-collection derivation and locked analysis",
        "model": model,
        "positive_class": "REAL",
        "positive_label": REAL_LABEL,
        "negative_class": "SHAM",
        "negative_label": SHAM_LABEL,
        "fixed_hyperparameters": {
            "probe": "L2 logistic regression on standardized activations",
            "C": C,
            "max_iter": max_iter,
            "seed": seed,
            "score_threshold": FIXED_SCORE_THRESHOLD,
            "validation_fraction_of_train_template_groups": val_fraction,
        },
        "gates": gates,
        "manipulation": manipulation,
        "positions": positions,
        "behavior": behavior,
        "beyond_condition_regressions": regressions,
        "head_to_head_inputs": head_to_head_inputs,
    }
    _write_json(output_dir / "behavior_results.json", behavior)
    _write_json(output_dir / "beyond_condition_regressions.json", regressions)
    _write_json(output_dir / "checkpoint_results.json", summary)
    return summary


def resume_locked_checkpoint_tail(
    behavior_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 7,
    C: float = 0.1,
    max_iter: int = 2000,
    val_fraction: float = 0.2,
    bootstrap_replicates: int = 10_000,
) -> dict[str, Any]:
    """Resume only the locked behavior/regression tail from sealed probe outputs."""

    output_dir = Path(output_dir)
    manipulation = _read_json_object(
        output_dir / "manipulation_results.json", "manipulation result"
    )
    positions = {
        position: _read_json_object(
            output_dir / position / "position_results.json",
            f"{position} position result",
        )
        for position in LOCKED_POSITIONS
    }
    behavior_rows = read_analysis_jsonl(behavior_path)
    return _finalize_locked_checkpoint(
        manipulation=manipulation,
        positions=positions,
        behavior_rows=behavior_rows,
        output_dir=output_dir,
        seed=seed,
        C=C,
        max_iter=max_iter,
        val_fraction=val_fraction,
        bootstrap_replicates=bootstrap_replicates,
    )


def run_locked_checkpoint(
    receipt_cache_path: str | Path,
    prompt_cache_path: str | Path,
    metadata_path: str | Path,
    behavior_path: str | Path,
    manipulation_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 7,
    C: float = 0.1,
    max_iter: int = 2000,
    val_fraction: float = 0.2,
    bootstrap_replicates: int = 10_000,
) -> dict:
    """Run manipulation first, then both locked probe positions and behavior gates."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # This prerequisite is intentionally evaluated before any probe is fitted.
    manipulation_rows = read_analysis_jsonl(manipulation_path)
    manipulation = analyze_manipulation_rows(manipulation_rows)
    _write_json(output_dir / "manipulation_results.json", manipulation)

    caches = {
        "receipt_final": load_activation_cache(receipt_cache_path),
        "prompt_final": load_activation_cache(prompt_cache_path),
    }
    for expected_position, cache in caches.items():
        if cache.position != expected_position:
            raise ValueError(
                f"expected {expected_position} cache, got position={cache.position!r}"
            )
    if caches["receipt_final"].model != caches["prompt_final"].model:
        raise ValueError("receipt_final and prompt_final model IDs differ")
    if caches["receipt_final"].X.shape[1:] != caches["prompt_final"].X.shape[1:]:
        raise ValueError("receipt_final and prompt_final layer/d_model shapes differ")
    for position, cache in caches.items():
        if cache.transcript_ids is None:
            raise ValueError(f"locked {position} cache must carry transcript_ids")
        class_contract = (
            cache.label_name,
            cache.positive_class,
            cache.positive_label,
            cache.negative_class,
            cache.negative_label,
        )
        if class_contract != ("world", "REAL", REAL_LABEL, "SHAM", SHAM_LABEL):
            raise ValueError(
                f"locked {position} cache must name world REAL=1 and SHAM=0 exactly"
            )
    if not np.array_equal(
        caches["receipt_final"].prompts, caches["prompt_final"].prompts
    ):
        raise ValueError("receipt_final and prompt_final prompt rows differ")
    if not np.array_equal(
        caches["receipt_final"].transcript_ids,
        caches["prompt_final"].transcript_ids,
    ):
        raise ValueError("receipt_final and prompt_final transcript IDs differ")
    metadata = {
        position: load_metadata(metadata_path, cache)
        for position, cache in caches.items()
    }
    for position, records in metadata.items():
        if any(str(row.raw.get("model")) != caches[position].model for row in records):
            raise ValueError(f"locked {position} metadata/model mismatch")
    positions = {
        position: _run_locked_position(
            cache,
            metadata[position],
            output_dir / position,
            seed=seed,
            C=C,
            max_iter=max_iter,
            val_fraction=val_fraction,
        )
        for position, cache in caches.items()
    }
    behavior_rows = read_analysis_jsonl(behavior_path)
    return _finalize_locked_checkpoint(
        manipulation=manipulation,
        positions=positions,
        behavior_rows=behavior_rows,
        output_dir=output_dir,
        seed=seed,
        C=C,
        max_iter=max_iter,
        val_fraction=val_fraction,
        bootstrap_replicates=bootstrap_replicates,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, help="legacy one-position pairwise mode")
    parser.add_argument(
        "--resume-locked-tail",
        action="store_true",
        help=(
            "reuse sealed manipulation/position results under --output-dir and run "
            "only behavior plus beyond-condition finalization"
        ),
    )
    parser.add_argument("--receipt-cache", type=Path)
    parser.add_argument("--prompt-cache", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--behavior", type=Path)
    parser.add_argument("--manipulation", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    args = parser.parse_args(argv)

    if args.resume_locked_tail:
        forbidden = (
            args.cache,
            args.receipt_cache,
            args.prompt_cache,
            args.metadata,
            args.manipulation,
        )
        if any(value is not None for value in forbidden):
            parser.error(
                "--resume-locked-tail accepts only --behavior, --output-dir, and "
                "fixed analysis options"
            )
        if args.behavior is None:
            parser.error("--resume-locked-tail requires --behavior")
        summary = resume_locked_checkpoint_tail(
            args.behavior,
            args.output_dir,
            seed=args.seed,
            C=args.c,
            max_iter=args.max_iter,
            val_fraction=args.val_fraction,
            bootstrap_replicates=args.bootstrap_replicates,
        )
        for position, result in summary["positions"].items():
            primary = result["primary"]
            selected = primary["selected_layer_heldout_test"]
            print(
                f"{position}: reused sealed layer={primary['selected_layer']} "
                f"heldout_AUROC={selected['auroc']:.3f} "
                f"({selected['h1_verdict']})"
            )
        return

    locked_values = (
        args.receipt_cache,
        args.prompt_cache,
        args.behavior,
        args.manipulation,
    )
    if args.cache and any(value is not None for value in locked_values):
        parser.error("--cache cannot be combined with locked checkpoint inputs")
    if args.cache:
        if args.metadata is None:
            parser.error("--cache requires --metadata")
        summary = run_pipeline(
            args.cache,
            args.metadata,
            args.output_dir,
            seed=args.seed,
            C=args.c,
            max_iter=args.max_iter,
            val_fraction=args.val_fraction,
        )
        for contrast in summary["contrasts"]:
            selected = contrast["selected_layer_test"]
            print(
                f"{contrast['contrast']}: layer={contrast['selected_layer']} "
                f"test_AUROC={selected['auroc']:.3f} ({selected['h1_verdict']})"
            )
        return
    if args.metadata is None or any(value is None for value in locked_values):
        parser.error(
            "locked mode requires --receipt-cache, --prompt-cache, --behavior, "
            "--manipulation, and --metadata"
        )
    summary = run_locked_checkpoint(
        args.receipt_cache,
        args.prompt_cache,
        args.metadata,
        args.behavior,
        args.manipulation,
        args.output_dir,
        seed=args.seed,
        C=args.c,
        max_iter=args.max_iter,
        val_fraction=args.val_fraction,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    for position, result in summary["positions"].items():
        primary = result["primary"]
        selected = primary["selected_layer_heldout_test"]
        print(
            f"{position}: layer={primary['selected_layer']} heldout_AUROC="
            f"{selected['auroc']:.3f} ({selected['h1_verdict']})"
        )


if __name__ == "__main__":
    main()
