"""Exp 4b leak-free activation consistency check (CPU only).

The Exp 4b leak-free dataset makes every labelled REAL/SHAM main prompt pair
byte-identical, so the world label carries zero information about the
subject-visible input and a prompt-position probe is at chance as a matter of
design rather than measurement; see ``experiments/exp3_dataset/README.md``,
"What the leak-free labels do and do not carry". Scoring a finalized leak-free
cache therefore tests the pipeline, not the hypothesis: it confirms that
collection, finalization, and the probe code path run end to end on the
obtained hardware and that no material residual linear signal is detected
above the pre-stated threshold at any evaluated layer of the scored position.
A chance-level finite-sample AUROC bounds such a signal; it does not show that
the realized sample carries exactly zero label information (the sampled
response tokens are not byte-identical across the two labels). The check makes
no probe claim and must not be read as "the probe fails once the leak is
removed".

This command scores one finalized leak-free cache through the sealed Run v1
code path (``train._run_locked_position`` with the same fixed
hyperparameters), verifies the byte-identity of every labelled REAL/SHAM pair
in the sidecar, attaches bootstrap confidence intervals to every layer's
held-out AUROC, and fails closed if any evaluated layer (not only the
validation-selected one) or the separately generated, same-generator
low-base-rate evaluation is materially above chance, which would be a bug or a
residual leak rather than a finding. It refuses any cache or sidecar that is
not marked ``dataset_variant=leak_free``, and it stamps ``dataset_variant`` and
the low-base-rate provenance into every artifact it writes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .cache import ActivationCache, SampleMetadata, load_activation_cache, load_metadata
from .metrics import binary_metrics
from .splits import make_group_splits
from .train import (
    PRIMARY_CONDITIONS,
    REAL_LABEL,
    SHAM_LABEL,
    _fit_probe,
    _run_locked_position,
    _world_label,
    _write_json,
)

LEAK_FREE_VARIANT = "leak_free"
CHANCE_AUROC = 0.5
MATERIALLY_ABOVE_CHANCE_AUROC = 0.6
REFIT_TOLERANCE = 1e-9
FORMAT_VERSION = 2
STATUS_AT_CHANCE = "at_chance_as_designed"
STATUS_ABOVE_CHANCE = "ABOVE_CHANCE_INVESTIGATE_BUG_OR_RESIDUAL_LEAK"

TAUTOLOGY_NOTE = (
    "By construction every labelled leak-free REAL/SHAM main prompt pair is "
    "byte-identical, so the world label carries zero information about the "
    "subject-visible input and an at-chance probe is expected as a tautology. "
    "This run is a pipeline/consistency check; it is not evidence that the probe "
    "fails once the text leak is removed and it makes no probe claim. See "
    "experiments/exp3_dataset/README.md, 'What the leak-free labels do and do "
    "not carry'."
)

SCOPE_NOTE = (
    "A verdict of at_chance_as_designed means no material residual linear signal "
    "was detected above the pre-stated threshold at any evaluated layer of this "
    "cached position, under this probe family and sample size. It bounds such a "
    "signal; it does not show that the realized finite sample carries exactly "
    "zero label information (sampled response tokens are not byte-identical "
    "across the two labels), and it says nothing about nonlinear structure, "
    "positions that were not scored, or effects below the threshold."
)

VERDICT_RULE = (
    "fail-closed over every evaluated layer: any layer whose held-out AUROC "
    f"exceeds {MATERIALLY_ABOVE_CHANCE_AUROC} with the row-bootstrap 95% CI "
    f"excluding {CHANCE_AUROC} (lower bound above chance) flags a bug or residual "
    "leak whether or not validation selected it; the same rule applied to the "
    "separately generated, same-generator low-base-rate evaluation at the "
    "selected layer also flags"
)

LBR_PROVENANCE = "separately_generated_same_generator"
LBR_PROVENANCE_NOTE = (
    "The low-base-rate cache is separately generated (a distinct collection lane "
    "that never trained or validated a probe) but same-generator: it comes from "
    "the same prompt generator and the same subject model. It is not an "
    "independent-distribution, out-of-distribution, or paraphrase test. See "
    "docs/errata.md, E1."
)
SEALED_KEY_NOTE = (
    "`primary.independent_lbr` and the `independent_lbr_REAL_vs_SHAM.jsonl` score "
    "filename are fixed identifiers inherited from the sealed Run v1 code path "
    "(train._run_locked_position); 'independent' there is not a claim about the "
    "cache's distribution. Read them as the separately generated, same-generator "
    "low-base-rate cache."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_leak_free(
    cache: ActivationCache, metadata: list[SampleMetadata], cache_path: Path
) -> None:
    if cache.dataset_variant != LEAK_FREE_VARIANT:
        raise ValueError(
            f"{cache_path} is marked dataset_variant={cache.dataset_variant!r}; "
            f"this consistency check only scores {LEAK_FREE_VARIANT!r} caches"
        )
    wrong = [
        row.transcript_id
        for row in metadata
        if row.raw.get("dataset_variant") != LEAK_FREE_VARIANT
    ]
    if wrong:
        raise ValueError(
            f"{len(wrong)} sidecar rows are not marked dataset_variant="
            f"{LEAK_FREE_VARIANT!r}; first: {wrong[:3]}"
        )


def check_prompt_identity(metadata: list[SampleMetadata]) -> dict[str, Any]:
    """Every labelled main REAL/SHAM pair must share exactly one prompt text.

    Every low-base-rate REAL prompt must also occur verbatim as a SHAM prompt.
    """

    groups: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for row in metadata:
        if row.raw.get("source_collection", "main") != "main":
            continue
        world = str(row.raw.get("world", ""))
        if world not in {"REAL", "SHAM"}:
            continue
        groups[(row.condition, row.template_id)][world].add(row.prompt)
    differing = []
    identical = 0
    for (condition, template_id), worlds in sorted(groups.items()):
        real = worlds.get("REAL", set())
        sham = worlds.get("SHAM", set())
        if len(real) == 1 and real == sham:
            identical += 1
        else:
            differing.append(
                {
                    "condition": condition,
                    "template_id": template_id,
                    "distinct_real_prompts": len(real),
                    "distinct_sham_prompts": len(sham),
                }
            )
    lbr_real = {
        row.prompt
        for row in metadata
        if row.raw.get("source_collection") == "lbr" and row.raw.get("world") == "REAL"
    }
    lbr_sham = {
        row.prompt
        for row in metadata
        if row.raw.get("source_collection") == "lbr" and row.raw.get("world") == "SHAM"
    }
    orphan_real = sorted(lbr_real - lbr_sham)
    return {
        "main_labelled_pairs": len(groups),
        "main_byte_identical_pairs": identical,
        "main_differing_pairs": differing,
        "lbr_distinct_real_prompts": len(lbr_real),
        "lbr_distinct_sham_prompts": len(lbr_sham),
        "lbr_real_prompts_without_sham_twin": len(orphan_real),
        "passed": not differing and not orphan_real,
    }


def _count_rows(metadata: list[SampleMetadata]) -> dict[str, Any]:
    by_lane_split_world = Counter(
        (
            str(row.raw.get("source_collection", "main")),
            row.split,
            str(row.raw.get("world", "")),
        )
        for row in metadata
    )
    return {
        "transcripts": len(metadata),
        "by_lane_split_world": [
            {"lane": lane, "split": split, "world": world, "rows": count}
            for (lane, split, world), count in sorted(by_lane_split_world.items())
        ],
        "distinct_source_rows": len(
            {
                (row.raw.get("source_collection", "main"), row.raw.get("source_row_id"))
                for row in metadata
            }
        ),
        "distinct_templates": {
            lane: len(
                {
                    row.template_id
                    for row in metadata
                    if row.raw.get("source_collection", "main") == lane
                }
            )
            for lane in ("main", "lbr")
        },
    }


def weighted_auroc(
    scores: np.ndarray, labels: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """AUROC of each weighted resample.

    ``weights`` is ``[replicates, n]`` of non-negative resample counts. Each row
    evaluates P(score_pos > score_neg) + 0.5 P(score_pos == score_neg) over the
    resampled multiset, which is exactly ``roc_auc_score`` when every weight is
    one. Replicates that lose a class evaluate to NaN.
    """

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    positive = labels[order] == REAL_LABEL
    w = np.asarray(weights, dtype=np.float64)[:, order]
    lower = np.searchsorted(sorted_scores, sorted_scores, side="left")
    upper = np.searchsorted(sorted_scores, sorted_scores, side="right")
    negative_cumulative = np.concatenate(
        [np.zeros((w.shape[0], 1)), np.cumsum(w * (~positive), axis=1)], axis=1
    )
    below = negative_cumulative[:, lower]
    equal = negative_cumulative[:, upper] - negative_cumulative[:, lower]
    numerator = np.sum(
        w[:, positive] * (below[:, positive] + 0.5 * equal[:, positive]), axis=1
    )
    w_positive = np.sum(w[:, positive], axis=1)
    w_negative = np.sum(w[:, ~positive], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return numerator / (w_positive * w_negative)


def stratified_row_weights(
    labels: np.ndarray, replicates: int, rng: np.random.Generator
) -> np.ndarray:
    """Resample rows with replacement within each class, keeping class counts."""

    labels = np.asarray(labels, dtype=np.int64)
    weights = np.zeros((replicates, len(labels)), dtype=np.int64)
    for value in (REAL_LABEL, SHAM_LABEL):
        members = np.flatnonzero(labels == value)
        counts = rng.multinomial(
            len(members), np.full(len(members), 1.0 / len(members)), size=replicates
        )
        weights[:, members] = counts
    return weights


def template_cluster_weights(
    template_ids: list[str], replicates: int, rng: np.random.Generator
) -> np.ndarray:
    """Resample whole template groups with replacement."""

    templates = sorted(set(template_ids))
    index = {template: position for position, template in enumerate(templates)}
    row_template = np.asarray([index[template] for template in template_ids])
    counts = rng.multinomial(
        len(templates), np.full(len(templates), 1.0 / len(templates)), size=replicates
    )
    return counts[:, row_template]


def _percentile_ci(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    low, high = np.percentile(finite, [2.5, 97.5])
    return {
        "ci95": [float(low), float(high)],
        "valid_replicates": int(len(finite)),
        "excludes_chance": bool(low > CHANCE_AUROC or high < CHANCE_AUROC),
    }


def per_layer_heldout_intervals(
    cache: ActivationCache,
    metadata: list[SampleMetadata],
    primary: dict[str, Any],
    *,
    seed: int,
    C: float,
    max_iter: int,
    val_fraction: float,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    """Refit every layer exactly as the sealed path did and bootstrap its held-out AUROC."""

    main_global = np.asarray(
        [
            index
            for index, row in enumerate(metadata)
            if row.raw.get("source_collection", "main") == "main"
            and row.condition in PRIMARY_CONDITIONS
        ],
        dtype=np.int64,
    )
    main_metadata = [metadata[index] for index in main_global]
    y = np.asarray([_world_label(row) for row in main_metadata], dtype=np.int64)
    splits = make_group_splits(main_metadata, seed=seed, val_fraction=val_fraction)
    sealed_split = primary["split"]
    if (
        list(splits.train_groups) != sealed_split["train_groups"]
        or list(splits.val_groups) != sealed_split["validation_groups"]
        or list(splits.test_groups) != sealed_split["heldout_test_groups"]
    ):
        raise ValueError("refit template split does not match the sealed-path split")
    test_y = y[splits.test]
    test_templates = [main_metadata[index].template_id for index in splits.test]
    row_weights = stratified_row_weights(
        test_y, bootstrap_replicates, np.random.default_rng(seed)
    )
    cluster_weights = template_cluster_weights(
        test_templates, bootstrap_replicates, np.random.default_rng(seed)
    )
    layers = []
    for layer in range(cache.n_layers):
        X = np.asarray(cache.X[main_global, layer, :], dtype=np.float64)
        probe = _fit_probe(
            X[splits.train], y[splits.train], C=C, max_iter=max_iter, seed=seed
        )
        test_scores = probe.scores(X[splits.test])
        point = float(binary_metrics(test_y, test_scores)["auroc"])
        sealed_layer = primary["layers"][layer]
        sealed_point = float(sealed_layer["heldout_test"]["auroc"])
        if abs(point - sealed_point) > REFIT_TOLERANCE:
            raise ValueError(
                f"layer {layer} refit held-out AUROC {point} differs from the "
                f"sealed-path value {sealed_point}"
            )
        unit = weighted_auroc(test_scores, test_y, np.ones((1, len(test_y))))
        if abs(float(unit[0]) - point) > 1e-9:
            raise ValueError("weighted AUROC does not reproduce roc_auc_score")
        layers.append(
            {
                "layer": layer,
                "validation_auroc": sealed_layer["validation"]["auroc"],
                "heldout_auroc": point,
                "heldout_row_bootstrap": _percentile_ci(
                    weighted_auroc(test_scores, test_y, row_weights)
                ),
                "heldout_template_bootstrap": _percentile_ci(
                    weighted_auroc(test_scores, test_y, cluster_weights)
                ),
                "shuffled_label_heldout_auroc": sealed_layer["controls"][
                    "shuffled_label_heldout_test_auroc"
                ],
                "random_direction_heldout_auroc": sealed_layer["controls"][
                    "random_direction_heldout_test_auroc"
                ],
            }
        )
    return {
        "heldout_samples": int(len(test_y)),
        "heldout_templates": sorted(set(test_templates)),
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": seed,
            "row_bootstrap": "stratified within REAL and SHAM, class counts fixed",
            "template_bootstrap": "held-out template groups resampled with replacement",
            "interval": "percentile 2.5/97.5",
        },
        "layers": layers,
    }


def lbr_interval(
    primary: dict[str, Any], *, seed: int, bootstrap_replicates: int
) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in Path(primary["artifacts"]["lbr_scores"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    scores = np.asarray(
        [float(row["probe_score_REAL"]) for row in rows], dtype=np.float64
    )
    point = float(binary_metrics(labels, scores)["auroc"])
    sealed_point = float(primary["independent_lbr"]["empirical"]["auroc"])
    if abs(point - sealed_point) > REFIT_TOLERANCE:
        raise ValueError("LBR score file does not reproduce the sealed-path AUROC")
    weights = stratified_row_weights(
        labels, bootstrap_replicates, np.random.default_rng(seed)
    )
    return {
        "rows": int(len(rows)),
        "counts": {
            "REAL": int(np.sum(labels == REAL_LABEL)),
            "SHAM": int(np.sum(labels == SHAM_LABEL)),
        },
        "auroc": point,
        "row_bootstrap": _percentile_ci(weighted_auroc(scores, labels, weights)),
        "fixed_threshold": primary["independent_lbr"]["empirical"],
        "provenance": LBR_PROVENANCE,
        "provenance_note": LBR_PROVENANCE_NOTE,
        "sealed_key_note": SEALED_KEY_NOTE,
    }


def _material_signal(auroc: float, row_ci95: list[float]) -> bool:
    """The pre-stated flag: AUROC above the threshold with the CI clear of chance."""

    return (
        float(auroc) > MATERIALLY_ABOVE_CHANCE_AUROC
        and float(row_ci95[0]) > CHANCE_AUROC
    )


def build_verdict(
    intervals: dict[str, Any], primary: dict[str, Any], lbr: dict[str, Any]
) -> dict[str, Any]:
    """Fail-closed verdict over every evaluated layer plus the LBR evaluation.

    ``status`` is ``ABOVE_CHANCE_INVESTIGATE_BUG_OR_RESIDUAL_LEAK`` if any layer
    in ``intervals`` (not only the validation-selected one) or the low-base-rate
    evaluation satisfies :func:`_material_signal`; otherwise
    ``at_chance_as_designed``. The maximum held-out AUROC is reported with its
    layer id (exact ties choose the lowest layer, matching the selection rule).
    """

    layers = intervals["layers"]
    if not layers:
        raise ValueError("no evaluated layers; refusing to issue a verdict")
    selected_layer = int(primary["selected_layer"])
    selected = layers[selected_layer]
    if int(selected["layer"]) != selected_layer:
        raise ValueError("per-layer intervals are not indexed by layer id")
    selected_auroc = float(primary["selected_layer_heldout_test"]["auroc"])
    selected_row_ci = selected["heldout_row_bootstrap"]["ci95"]
    layers_materially_above_chance = [
        int(layer["layer"])
        for layer in layers
        if _material_signal(
            layer["heldout_auroc"], layer["heldout_row_bootstrap"]["ci95"]
        )
    ]
    max_layer = max(
        layers, key=lambda layer: (float(layer["heldout_auroc"]), -int(layer["layer"]))
    )
    lbr_materially_above_chance = _material_signal(
        lbr["auroc"], lbr["row_bootstrap"]["ci95"]
    )
    materially_above_chance = (
        bool(layers_materially_above_chance) or lbr_materially_above_chance
    )
    return {
        "rule": VERDICT_RULE,
        "materially_above_chance_auroc": MATERIALLY_ABOVE_CHANCE_AUROC,
        "chance_auroc": CHANCE_AUROC,
        "evaluated_layers": len(layers),
        "selected_layer": selected_layer,
        "selected_layer_heldout_auroc": selected_auroc,
        "selected_layer_heldout_ci95_row_bootstrap": selected_row_ci,
        "selected_layer_heldout_ci95_template_bootstrap": selected[
            "heldout_template_bootstrap"
        ]["ci95"],
        "selected_layer_materially_above_chance": _material_signal(
            selected_auroc, selected_row_ci
        ),
        "max_layer": int(max_layer["layer"]),
        "max_layer_heldout_auroc": float(max_layer["heldout_auroc"]),
        "max_layer_heldout_ci95_row_bootstrap": max_layer["heldout_row_bootstrap"][
            "ci95"
        ],
        "layers_with_row_ci_excluding_chance": [
            int(layer["layer"])
            for layer in layers
            if layer["heldout_row_bootstrap"]["excludes_chance"]
        ],
        "layers_materially_above_chance": layers_materially_above_chance,
        "lbr_auroc": float(lbr["auroc"]),
        "lbr_ci95_row_bootstrap": lbr["row_bootstrap"]["ci95"],
        "lbr_materially_above_chance": bool(lbr_materially_above_chance),
        "materially_above_chance": bool(materially_above_chance),
        "status": STATUS_ABOVE_CHANCE if materially_above_chance else STATUS_AT_CHANCE,
        "scope": SCOPE_NOTE,
    }


def stamp_position_results(
    position: dict[str, Any], cache: ActivationCache, output_dir: Path
) -> dict[str, Any]:
    """Write dataset_variant and LBR provenance into the unsealed position_results.json.

    ``train._run_locked_position`` writes the sealed Run v1 schema, whose
    ``independent_lbr`` key and score filename are fixed identifiers. This
    leak-free artifact is not sealed, so the variant marker and the corrected
    provenance are stamped in explicitly and the file is rewritten by the script.
    """

    position["dataset_variant"] = cache.dataset_variant
    position["provenance"] = {
        "dataset_variant": cache.dataset_variant,
        "low_base_rate_cache": LBR_PROVENANCE,
        "low_base_rate_cache_note": LBR_PROVENANCE_NOTE,
        "sealed_key_note": SEALED_KEY_NOTE,
        "written_by": "experiments/exp4_probes/leak_free_consistency.py",
    }
    position["primary"]["independent_lbr"]["provenance"] = LBR_PROVENANCE
    _write_json(output_dir / "position_results.json", position)
    return position


def run_consistency_check(
    cache_path: str | Path,
    metadata_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int,
    C: float,
    max_iter: int,
    val_fraction: float,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    cache_path = Path(cache_path)
    metadata_path = Path(metadata_path)
    output_dir = Path(output_dir)
    cache = load_activation_cache(cache_path)
    metadata = load_metadata(metadata_path, cache)
    _require_leak_free(cache, metadata, cache_path)
    identity = check_prompt_identity(metadata)
    if not identity["passed"]:
        raise ValueError(
            "leak-free sidecar has REAL/SHAM prompt pairs that are not byte-identical: "
            f"{identity}"
        )

    position = _run_locked_position(
        cache,
        metadata,
        output_dir,
        seed=seed,
        C=C,
        max_iter=max_iter,
        val_fraction=val_fraction,
    )
    primary = position["primary"]
    intervals = per_layer_heldout_intervals(
        cache,
        metadata,
        primary,
        seed=seed,
        C=C,
        max_iter=max_iter,
        val_fraction=val_fraction,
        bootstrap_replicates=bootstrap_replicates,
    )
    lbr = lbr_interval(primary, seed=seed, bootstrap_replicates=bootstrap_replicates)
    stamp_position_results(position, cache, output_dir)
    verdict = build_verdict(intervals, primary, lbr)
    selected_layer = verdict["selected_layer"]
    result = {
        "format_version": FORMAT_VERSION,
        "purpose": "Exp 4b leak-free activation rerun pipeline/consistency check",
        "framing": TAUTOLOGY_NOTE,
        "scope": SCOPE_NOTE,
        "dataset_variant": cache.dataset_variant,
        "cache": {
            "path": str(cache_path),
            "sha256": _sha256(cache_path),
            "position": cache.position,
            "model": cache.model,
            "dataset_variant": cache.dataset_variant,
            "X_shape": list(cache.X.shape),
        },
        "metadata": {
            "path": str(metadata_path),
            "sha256": _sha256(metadata_path),
            "counts": _count_rows(metadata),
        },
        "prompt_identity": identity,
        "fixed_hyperparameters": {
            "probe": "L2 logistic regression on standardized activations (liblinear)",
            "seed": seed,
            "C": C,
            "max_iter": max_iter,
            "validation_fraction_of_nonheldout_template_groups": val_fraction,
            "score_threshold": 0.5,
            "bootstrap_replicates": bootstrap_replicates,
            "source": "experiments/exp4_probes/train.py argparse defaults, as sealed in Run v1",
        },
        "primary_world_probe": {
            "task": primary["task"],
            "included_conditions": primary["included_conditions"],
            "split": primary["split"],
            "selection_rule": primary["selection_rule"],
            "selected_layer": selected_layer,
            "selected_layer_validation_auroc": primary["layers"][selected_layer][
                "validation"
            ]["auroc"],
            "selected_layer_heldout_test": primary["selected_layer_heldout_test"],
            "per_layer": intervals,
        },
        "low_base_rate": lbr,
        "pairwise_condition_contrasts": {
            "note": (
                "Conditions differ in prompt text by design, so condition contrasts "
                "remain separable in a leak-free cache; they are not a REAL/SHAM leak."
            ),
            "contrasts": [
                {
                    "contrast": contrast["contrast"],
                    "selected_layer": contrast["selected_layer"],
                    "test_auroc": contrast["selected_layer_test"]["auroc"],
                    "test_samples": contrast["split"]["test_samples"],
                }
                for contrast in position["required_pairwise_condition_separations"]
            ],
        },
        "verdict": verdict,
        "artifacts": {
            "position_results": str(output_dir / "position_results.json"),
            "consistency_check": str(output_dir / "consistency_check.json"),
        },
    }
    (output_dir / "consistency_check.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    args = parser.parse_args(argv)
    result = run_consistency_check(
        args.cache,
        args.metadata,
        args.output_dir,
        seed=args.seed,
        C=args.c,
        max_iter=args.max_iter,
        val_fraction=args.val_fraction,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    verdict = result["verdict"]
    print(
        f"{result['cache']['position']} ({result['dataset_variant']}): "
        f"selected_layer={verdict['selected_layer']} "
        f"heldout_AUROC={verdict['selected_layer_heldout_auroc']:.3f} "
        f"CI95={verdict['selected_layer_heldout_ci95_row_bootstrap']} "
        f"max_layer={verdict['max_layer']} "
        f"max_heldout_AUROC={verdict['max_layer_heldout_auroc']:.3f} "
        f"layers_flagged={verdict['layers_materially_above_chance']} "
        f"LBR_AUROC={verdict['lbr_auroc']:.3f} -> {verdict['status']}"
    )


if __name__ == "__main__":
    main()
