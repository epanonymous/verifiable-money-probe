from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from experiments.exp4_collection.contract import (
    FWD_CHUNK,
    GEN_BATCH,
    MODEL,
    dataset_variant,
    expected_prompt_groups,
    load_rows,
    ordered_prompt_groups,
    rollout_batch_plan,
)
from experiments.exp4_collection.finalize import (
    DEFAULT_DATA,
    POSITIONS,
    InventoryError,
    finalize_shards,
    inventory_shards,
)
from experiments.exp4_paths import DEFAULT_LEAK_FREE_DATA
from experiments.exp4_probes.cache import load_activation_cache, load_metadata


ROOT = Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_scalar(row: dict, field: str) -> object:
    if field == "label":
        return -1 if row.get("label") is None else row["label"]
    if field == "split":
        return row.get("split", "heldout")
    return row[field]


def rewrite_npz(path: Path, **updates: object) -> None:
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.asarray(archive[name]) for name in archive.files}
    values.update(updates)
    np.savez_compressed(path, **values)


def test_committed_inventory_uses_collectors_exact_group_key_rule() -> None:
    main = expected_prompt_groups(DEFAULT_DATA, "main")
    lbr = expected_prompt_groups(DEFAULT_DATA, "lbr")

    assert len(main) == 274
    assert sum(len(group.expanded_row_ids) for group in main) == 8400
    assert len(lbr) == 20
    assert sum(len(group.expanded_row_ids) for group in lbr) == 1010
    claimed = next(group for group in main if group.key == "real_a_t00")
    assert claimed.filename == "real_a_t00.npz"
    assert {row["id"] for row in claimed.rows} == {"real_a_t00", "sham_a_t00"}


def test_run_v1_row_ids_stay_row_major() -> None:
    # Shards already on the volume were written row-major; that ordering is the
    # Run v1 resume contract and must not move.
    main = expected_prompt_groups(DEFAULT_DATA, "main")
    claimed = next(group for group in main if group.key == "real_a_t00")

    row_major = tuple(
        str(row["id"])
        for row in claimed.rows
        for _ in range(int(row.get("n_rollouts", 1)))
    )
    assert claimed.variant == "run_v1"
    assert claimed.expanded_row_ids == row_major


def test_leak_free_row_ids_do_not_align_with_batch_boundaries() -> None:
    # A merged REAL/SHAM group must not assign labels as a contiguous
    # first-half/second-half split: that lands on GEN_BATCH/FWD_CHUNK edges and
    # makes bf16 batching artifacts perfectly label-correlated.
    main = expected_prompt_groups(DEFAULT_LEAK_FREE_DATA, "main")
    verified = next(group for group in main if group.key == "real_b_t00")
    worlds = {str(row["id"]): str(row["world"]) for row in verified.rows}

    expanded = verified.expanded_row_ids
    assert verified.variant == "leak_free"
    assert len(expanded) == 50
    # Same multiset as row-major: only the slot assignment moves.
    assert sorted(expanded) == sorted(
        str(row["id"])
        for row in verified.rows
        for _ in range(int(row.get("n_rollouts", 1)))
    )

    labels = [worlds[row_id] for row_id in expanded]
    assert labels[:25] != ["REAL"] * 25
    for start in range(0, 48, 12):
        chunk = labels[start : start + 12]
        assert len(set(chunk)) == 2, f"forward chunk at {start} is single-class"


def test_leak_free_lbr_real_row_is_not_pinned_to_a_batch_edge() -> None:
    # The lone REAL row of each 101-rollout LBR group must not always be the
    # solitary sample of the trailing batch-size-1 generate call.
    lbr = expected_prompt_groups(DEFAULT_LEAK_FREE_DATA, "lbr")
    positions = []
    for group in lbr:
        worlds = {str(row["id"]): str(row["world"]) for row in group.rows}
        expanded = group.expanded_row_ids
        assert len(expanded) == 101
        real = [i for i, row_id in enumerate(expanded) if worlds[row_id] == "REAL"]
        assert len(real) == 1
        positions.append(real[0])

    assert positions != [100] * len(lbr)
    assert len(set(positions)) > 1


def test_leak_free_row_id_permutation_is_stable_across_processes() -> None:
    # The permutation is a durable shard contract shared by collect, resume
    # validation and finalization, so it must not depend on PYTHONHASHSEED.
    import subprocess
    import sys

    script = (
        "from experiments.exp4_collection.contract import expected_prompt_groups;"
        "from experiments.exp4_paths import DEFAULT_LEAK_FREE_DATA;"
        "groups = expected_prompt_groups(DEFAULT_LEAK_FREE_DATA, 'main');"
        "print(','.join(next(g for g in groups if g.key == 'real_b_t00').expanded_row_ids))"
    )
    outputs = set()
    for seed in ("0", "1", "12345"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
            cwd=ROOT,
        )
        outputs.add(result.stdout.strip())

    assert len(outputs) == 1
    expected = expected_prompt_groups(DEFAULT_LEAK_FREE_DATA, "main")
    verified = next(group for group in expected if group.key == "real_b_t00")
    assert outputs.pop() == ",".join(verified.expanded_row_ids)


def test_dataset_variant_comes_from_the_dataset_manifest() -> None:
    # collect passes the variant explicitly; inventory/finalize only get a path.
    # Both must agree, so the dataset declares its own variant.
    assert dataset_variant(DEFAULT_DATA) == "run_v1"
    assert dataset_variant(DEFAULT_LEAK_FREE_DATA) == "leak_free"

    by_path = expected_prompt_groups(DEFAULT_LEAK_FREE_DATA, "main")
    by_flag = ordered_prompt_groups(
        load_rows(DEFAULT_LEAK_FREE_DATA, "main"), "leak_free"
    )
    assert [group.expanded_row_ids for group in by_path] == [
        group.expanded_row_ids for group in by_flag
    ]


def test_unknown_dataset_variant_fails_closed(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "manifest.json").write_text(
        json.dumps({"variant": "not_a_variant"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unknown dataset variant"):
        dataset_variant(data_dir)


def test_run_v1_batch_plan_generates_and_forwards_exactly_the_retained_rows() -> None:
    # Run v1 shards were written from unpadded batches; that stays byte-for-byte.
    for n_rollouts in (1, 2, 25, 50, 101):
        assert rollout_batch_plan(n_rollouts) == (n_rollouts, n_rollouts)
        assert rollout_batch_plan(n_rollouts, "run_v1") == (n_rollouts, n_rollouts)


def test_leak_free_batch_plan_keeps_no_rollout_in_a_short_batch() -> None:
    # Every retained rollout must come from a full GEN_BATCH generate call and a
    # full FWD_CHUNK activation forward, so no row sits in a differently-padded
    # trailing batch whose bf16 reduction order differs from the rest.
    for n_rollouts in (1, 2, 12, 25, 49, 50, 51, 100, 101, 125):
        generated, forwarded = rollout_batch_plan(n_rollouts, "leak_free")

        assert generated % GEN_BATCH == 0, n_rollouts
        assert forwarded % FWD_CHUNK == 0, n_rollouts
        assert forwarded >= n_rollouts
        assert generated >= forwarded
        # Surplus is bounded: never more than one extra batch of each kind.
        assert forwarded - n_rollouts < FWD_CHUNK
        assert generated - forwarded < GEN_BATCH


def test_leak_free_committed_groups_retain_only_full_batch_rollouts() -> None:
    # Under the old plan the committed shapes left a short trailing forward
    # chunk and, for LBR, a solitary generate call whose occupancy tracked the
    # REAL/SHAM label. Every shipped group size must now plan to whole batches.
    seen = 0
    for which in ("main", "lbr"):
        groups = expected_prompt_groups(DEFAULT_LEAK_FREE_DATA, which)
        assert groups
        for size in sorted({len(group.expanded_row_ids) for group in groups}):
            seen += 1
            generated, forwarded = rollout_batch_plan(size, "leak_free")
            assert generated % GEN_BATCH == 0, (which, size)
            assert forwarded % FWD_CHUNK == 0, (which, size)
            assert forwarded >= size
            assert generated >= forwarded
            # Under the row-major plan these sizes left short trailing batches.
            assert size % FWD_CHUNK != 0 or forwarded == size
    assert seen >= 2


def test_leak_free_lbr_real_row_never_lands_in_a_short_batch() -> None:
    # Regression for the residual: the REAL row of an LBR group used to be able
    # to land at slot 100, the sole sample of the trailing batch-size-1 call.
    lbr = expected_prompt_groups(DEFAULT_LEAK_FREE_DATA, "lbr")
    for group in lbr:
        worlds = {str(row["id"]): str(row["world"]) for row in group.rows}
        expanded = group.expanded_row_ids
        generated, forwarded = rollout_batch_plan(len(expanded), "leak_free")
        real_slots = [
            index for index, row_id in enumerate(expanded) if worlds[row_id] == "REAL"
        ]
        assert real_slots
        for slot in real_slots:
            assert slot // GEN_BATCH < generated // GEN_BATCH
            assert slot // FWD_CHUNK < forwarded // FWD_CHUNK
            # The batch holding this row is full, not a 1- or 5-wide remainder.
            assert min(generated - (slot // GEN_BATCH) * GEN_BATCH, GEN_BATCH) == (
                GEN_BATCH
            )
            assert min(forwarded - (slot // FWD_CHUNK) * FWD_CHUNK, FWD_CHUNK) == (
                FWD_CHUNK
            )


def test_batch_plan_rejects_nonpositive_and_unknown_variant() -> None:
    with pytest.raises(ValueError, match="n_rollouts must be positive"):
        rollout_batch_plan(0, "leak_free")
    with pytest.raises(ValueError, match="unknown dataset variant"):
        rollout_batch_plan(50, "not_a_variant")


def test_leak_free_inventory_groups_every_real_sham_prompt_pair() -> None:
    main = expected_prompt_groups(DEFAULT_LEAK_FREE_DATA, "main")
    lbr = expected_prompt_groups(DEFAULT_LEAK_FREE_DATA, "lbr")

    # Some sampled template tuples render the same text because a component is
    # unused in a given condition; grouping is by exact prompt, as in collect.py.
    assert len(main) == 179
    assert sum(len(group.expanded_row_ids) for group in main) == 8400
    assert len(lbr) == 10
    assert sum(len(group.expanded_row_ids) for group in lbr) == 1010
    verified = next(group for group in main if group.key == "real_b_t00")
    assert {row["id"] for row in verified.rows} == {
        "real_b_t00",
        "sham_b_t00",
    }
    labeled_groups = [group for group in main if group.rows[0]["label"] is not None]
    assert len(labeled_groups) == 134
    assert all(
        {row["world"] for row in group.rows} == {"REAL", "SHAM"}
        for group in labeled_groups
    )


def test_complete_inventory_reports_expected_and_retained_counts(
    collection_fixture,
) -> None:
    data_dir, staging = collection_fixture

    main, _ = inventory_shards(staging / "collect_main", "main", data_dir)
    lbr, _ = inventory_shards(staging / "collect_lbr", "lbr", data_dir)

    assert (main.expected_shards, main.retained_shards, main.retained_transcripts) == (
        3,
        3,
        6,
    )
    assert (lbr.expected_shards, lbr.retained_shards, lbr.retained_transcripts) == (
        2,
        2,
        3,
    )
    assert main.complete and lbr.complete


@pytest.mark.parametrize("fault", ["missing", "unexpected", "duplicate", "corrupt"])
def test_inventory_fails_closed_for_every_inventory_fault(
    collection_fixture, fault: str
) -> None:
    data_dir, staging = collection_fixture
    shard_dir = staging / "collect_main"
    target = min(shard_dir.glob("*.npz"))
    if fault == "missing":
        target.unlink()
    elif fault == "unexpected":
        np.savez_compressed(shard_dir / "unexpected.npz", value=np.asarray(1))
    elif fault == "duplicate":
        nested = shard_dir / "duplicate"
        nested.mkdir()
        shutil.copy2(target, nested / target.name)
    else:
        target.write_bytes(b"not an npz")

    with pytest.raises(InventoryError) as caught:
        inventory_shards(shard_dir, "main", data_dir)

    report = caught.value.report
    assert report.expected_shards == 3
    assert report.retained_shards in {2, 3}
    assert not report.complete
    assert getattr(report, "duplicates" if fault == "duplicate" else fault)


def test_finalization_combines_both_positions_and_aligned_metadata_without_mutation(
    collection_fixture, tmp_path: Path
) -> None:
    data_dir, staging = collection_fixture
    output = tmp_path / "final"
    source_paths = sorted(staging.rglob("*.npz"))
    before = {path: sha256(path) for path in source_paths}

    result = finalize_shards(
        staging / "collect_main", staging / "collect_lbr", output, data_dir
    )

    assert result["transcripts"] == 9
    assert result["activation_shape"] == [9, 2, 3]
    rows = [
        json.loads(line)
        for line in (output / "transcripts.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 9
    assert len({row["transcript_id"] for row in rows}) == 9
    assert {row["source_collection"] for row in rows} == {"main", "lbr"}
    assert {row["condition"] for row in rows} == {"claimed", "verified", "framing"}
    assert all(
        {
            "transcript_id",
            "source_row_id",
            "prompt",
            "response",
            "decision",
            "world",
            "condition",
            "template_id",
            "split",
            "model",
        }.issubset(row)
        for row in rows
    )
    assert all(row["model"] == MODEL for row in rows)

    cached_activations = {}
    for position in ("prompt_final", "response_final"):
        cache_path = output / f"{position}.npz"
        with np.load(cache_path, allow_pickle=False) as cache:
            assert cache["X"].shape == (9, 2, 3)
            cached_activations[position] = cache["X"].copy()
            assert cache["y"].tolist() == [
                -1 if row["label"] is None else row["label"] for row in rows
            ]
            assert cache["prompts"].tolist() == [row["prompt"] for row in rows]
            assert cache["transcript_ids"].tolist() == [
                row["transcript_id"] for row in rows
            ]
            assert cache["position"].item() == position
            assert cache["positive_class"].item() == "REAL"
            assert cache["positive_label"].item() == 1
            assert cache["negative_class"].item() == "SHAM"
            assert cache["negative_label"].item() == 0
            assert cache["unlabeled_label"].item() == -1
        loaded = load_activation_cache(cache_path)
        metadata = load_metadata(output / "transcripts.jsonl", loaded)
        assert loaded.n_samples == len(metadata) == 9
        assert (loaded.label_name, loaded.positive_class, loaded.positive_label) == (
            "world",
            "REAL",
            1,
        )
    np.testing.assert_array_equal(
        cached_activations["response_final"] - cached_activations["prompt_final"],
        np.full((9, 2, 3), 0.5, dtype=np.float16),
    )
    assert {path: sha256(path) for path in source_paths} == before


def test_finalization_is_byte_deterministic(collection_fixture, tmp_path: Path) -> None:
    data_dir, staging = collection_fixture
    output = tmp_path / "final"
    args = (staging / "collect_main", staging / "collect_lbr", output, data_dir)
    finalize_shards(*args)
    first = {path.name: sha256(path) for path in output.iterdir()}

    finalize_shards(*args)
    second = {path.name: sha256(path) for path in output.iterdir()}

    assert second == first


@pytest.mark.parametrize("fault", ["row_ids", "shape", "model"])
def test_finalization_rejects_misalignment_shape_and_model_mismatch(
    collection_fixture, tmp_path: Path, fault: str
) -> None:
    data_dir, staging = collection_fixture
    target = min((staging / "collect_main").glob("*.npz"))
    with np.load(target, allow_pickle=False) as archive:
        n = len(archive["row_ids"])
    if fault == "row_ids":
        rewrite_npz(target, row_ids=np.asarray(["wrong"] * n))
    elif fault == "shape":
        rewrite_npz(target, response_final=np.zeros((n, 3, 3), dtype=np.float16))
    else:
        rewrite_npz(target, model=np.asarray("wrong/model"))

    with pytest.raises(InventoryError) as caught:
        finalize_shards(
            staging / "collect_main",
            staging / "collect_lbr",
            tmp_path / "final",
            data_dir,
        )

    assert caught.value.report.corrupt
    assert not (tmp_path / "final").exists()


def test_run_v1_shard_scalars_stay_the_first_rows_values() -> None:
    # 39 committed Run v1 main groups already merge a REAL/SHAM pair, and their
    # shards on the volume carry the first row's descriptors. Neutralizing them
    # would invalidate every one of those shards on resume.
    claimed = next(
        group
        for group in expected_prompt_groups(DEFAULT_DATA, "main")
        if group.key == "real_a_t00"
    )

    assert {str(row["world"]) for row in claimed.rows} == {"REAL", "SHAM"}
    assert claimed.shard_scalars == {
        "world": "REAL",
        "cond": "a",
        "template_id": 0,
        "split": "train",
        "label": 1,
    }


def test_leak_free_shard_scalars_never_claim_a_class_the_shard_lacks() -> None:
    # Under leak_free a shard holds both worlds, so a per-shard descriptor that
    # the group's rows disagree on is neutral rather than the first row's value.
    main = expected_prompt_groups(DEFAULT_LEAK_FREE_DATA, "main")
    lbr = expected_prompt_groups(DEFAULT_LEAK_FREE_DATA, "lbr")

    neutral = {
        "world": "MIXED",
        "cond": "MIXED",
        "template_id": -1,
        "split": "MIXED",
        "label": -1,
    }
    for group in [*main, *lbr]:
        scalars = group.shard_scalars
        assert set(scalars) == set(neutral)
        for field in neutral:
            values = {row_scalar(row, field) for row in group.rows}
            expected = next(iter(values)) if len(values) == 1 else neutral[field]
            assert scalars[field] == expected, (group.key, field)

    verified = next(group for group in main if group.key == "real_b_t00")
    assert (verified.shard_scalars["world"], verified.shard_scalars["label"]) == (
        "MIXED",
        -1,
    )
    assert all(
        group.shard_scalars["world"] == "MIXED" and group.shard_scalars["label"] == -1
        for group in lbr
    )


def test_leak_free_resume_rejects_a_shard_that_claims_its_first_rows_class(
    leak_free_collection_fixture,
) -> None:
    # Regression: a merged shard labelled REAL/1 describes only half its
    # rollouts, and used to validate because the expectation came from rows[0].
    data_dir, staging = leak_free_collection_fixture
    merged = next(
        group
        for group in expected_prompt_groups(data_dir, "main")
        if group.key == "real_b_t00"
    )
    shard_path = staging / "collect_main" / merged.filename
    rewrite_npz(shard_path, world=np.asarray("REAL"), label=np.asarray(1))

    with pytest.raises(InventoryError) as caught:
        inventory_shards(staging / "collect_main", "main", data_dir)

    assert "world mismatch" in caught.value.report.corrupt[merged.filename]


def test_leak_free_finalization_labels_each_merged_rollout_from_its_own_row(
    leak_free_collection_fixture, tmp_path: Path
) -> None:
    # The crux of the variant: one merged REAL/SHAM prompt group whose rollouts
    # are assigned to two row ids through the sha256 permutation must still come
    # out per-row correct and activation-aligned after finalization.
    data_dir, staging = leak_free_collection_fixture
    output = tmp_path / "final_leak_free"
    merged = next(
        group
        for group in expected_prompt_groups(data_dir, "main")
        if group.key == "real_b_t00"
    )
    worlds = {str(row["id"]): str(row["world"]) for row in merged.rows}
    assert [worlds[row_id] for row_id in merged.expanded_row_ids] != ["REAL"] * 3 + [
        "SHAM"
    ] * 3

    result = finalize_shards(
        staging / "collect_main", staging / "collect_lbr", output, data_dir
    )

    assert result["transcripts"] == 11
    dataset = {
        which: {str(row["id"]): row for row in load_rows(data_dir, which)}
        for which in ("main", "lbr")
    }
    records = [
        json.loads(line)
        for line in (output / "transcripts.jsonl").read_text().splitlines()
    ]
    assert len(records) == 11
    for record in records:
        source = dataset[record["source_collection"]][record["source_row_id"]]
        assert record["world"] == source["world"]
        assert record["label"] == source["label"]
        assert record["prompt"] == source["prompt"]
    assert Counter(
        (record["source_collection"], record["source_row_id"]) for record in records
    ) == Counter(
        {
            ("main", "real_b_t00"): 3,
            ("main", "sham_b_t00"): 3,
            ("main", "framing_t00"): 1,
            ("main", "framing_t01"): 1,
            ("lbr", "lbr_real_0000"): 1,
            ("lbr", "lbr_sham_0000"): 1,
            ("lbr", "lbr_sham_0001"): 1,
        }
    )

    # Each finalized row must keep the activation of the shard slot it came from.
    slot_activations: dict[str, float] = {}
    for shard_path in sorted(staging.rglob("*.npz")):
        with np.load(shard_path, allow_pickle=False) as shard:
            for text, activation in zip(
                np.asarray(shard["texts"]).astype(str),
                np.asarray(shard["prompt_final"]),
                strict=True,
            ):
                slot_activations[str(text)] = float(activation[0, 0])
    assert len(slot_activations) == 11

    for position in POSITIONS:
        cache = load_activation_cache(output / f"{position}.npz")
        assert cache.n_samples == 11
        assert cache.y.tolist() == [
            -1 if record["label"] is None else record["label"] for record in records
        ]
        offset = 0.0 if position == "prompt_final" else 0.5
        assert [float(row[0, 0]) for row in cache.X] == [
            slot_activations[record["response"]] + offset for record in records
        ]
        load_metadata(output / "transcripts.jsonl", cache)


def test_finalized_artifacts_record_the_dataset_variant_they_came_from(
    collection_fixture, leak_free_collection_fixture, tmp_path: Path
) -> None:
    # A leak-free cache reuses Run v1's row ids and array contract, so without a
    # recorded variant a probe run could score it as an unmarked Run v1 cache.
    for index, (fixture, expected) in enumerate(
        ((collection_fixture, "run_v1"), (leak_free_collection_fixture, "leak_free"))
    ):
        data_dir, staging = fixture
        output = tmp_path / f"final_{index}"

        result = finalize_shards(
            staging / "collect_main", staging / "collect_lbr", output, data_dir
        )

        assert result["dataset_variant"] == expected
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["dataset_variant"] == expected
        reports = manifest["inventory"].values()
        assert {report["dataset_variant"] for report in reports} == {expected}
        records = [
            json.loads(line)
            for line in (output / "transcripts.jsonl").read_text().splitlines()
        ]
        assert {record["dataset_variant"] for record in records} == {expected}
        for position in POSITIONS:
            cache = load_activation_cache(output / f"{position}.npz")
            assert cache.dataset_variant == expected
