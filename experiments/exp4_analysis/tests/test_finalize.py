from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from experiments.exp4_analysis.contract import (
    manipulation_prompt,
    manipulation_required,
    prompt_sha256,
    receipt_paragraph_bounds,
)
from experiments.exp4_analysis.finalize import (
    DerivedInventoryError,
    finalize_derivations,
    inventory_derived_shards,
)
from experiments.exp4_collection.contract import MODEL, expected_prompt_groups
from experiments.exp4_collection.finalize import finalize_shards


def valid_prompt(condition: str, evidence: str = "evidence") -> str:
    parts = ["Role paragraph.", "Receipt paragraph."]
    if condition == "framing":
        parts.append("Assume the payment is real.")
    if condition in {"b", "c"}:
        parts.append(f'Verification output:\n{{"result": "{evidence}"}}')
    if condition == "c":
        parts.append("Binding paragraph.")
    parts.append("Decide now: SPEND or HOLD?")
    return "\n\n".join(parts)


def rewrite_fixture_prompts(data_dir: Path) -> None:
    for filename in (
        "prompts_main.jsonl",
        "prompts_framing.jsonl",
        "lowbaserate_eval.jsonl",
    ):
        path = data_dir / filename
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            if row["cond"] == "a":
                row["prompt"] = valid_prompt("a")
            elif row["cond"] == "framing":
                row["prompt"] = valid_prompt("framing")
            elif str(row["id"]).startswith("lbr_sham"):
                row["prompt"] = valid_prompt("b", "lbr-sham-shared")
            else:
                row["prompt"] = valid_prompt("b", str(row["id"]))
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def write_derived_shards(data_dir: Path, staging: Path) -> None:
    for which in ("main", "lbr"):
        output = staging / f"derived_{which}"
        output.mkdir(parents=True)
        for group_index, group in enumerate(expected_prompt_groups(data_dir, which)):
            condition = str(group.rows[0]["cond"])
            start, end = receipt_paragraph_bounds(group.prompt, condition)
            required = manipulation_required(group, which)
            probability = 75.0 if required else np.nan
            np.savez_compressed(
                output / group.filename,
                receipt_final=np.full((2, 3), group_index + 1, dtype=np.float16),
                prompt=np.asarray(group.prompt),
                prompt_sha256=np.asarray(prompt_sha256(group.prompt)),
                source_row_ids=np.asarray([str(row["id"]) for row in group.rows]),
                group_key=np.asarray(group.key),
                condition=np.asarray(condition),
                model=np.asarray(MODEL),
                receipt_paragraph_start=np.asarray(start),
                receipt_paragraph_end=np.asarray(end),
                receipt_rendered_char_index=np.asarray(end + 5),
                receipt_token_index=np.asarray(end + 2),
                rendered_prompt_sha256=np.asarray("0" * 64),
                spend_logprob=np.asarray(-0.2),
                hold_logprob=np.asarray(-0.8),
                spend_hold_log_odds=np.asarray(0.6),
                spend_token_ids=np.asarray([10, 11]),
                hold_token_ids=np.asarray([12]),
                manipulation_required=np.asarray(int(required)),
                manipulation_prompt=np.asarray(
                    manipulation_prompt(group.prompt) if required else ""
                ),
                manipulation_raw=np.asarray("75" if required else ""),
                manipulation_parse_ok=np.asarray(int(required)),
                manipulation_probability=np.asarray(probability),
                manipulation_parse_error=np.asarray(""),
            )


@pytest.fixture
def finalized_fixture(collection_fixture, tmp_path: Path):
    data_dir, collection_staging = collection_fixture
    rewrite_fixture_prompts(data_dir)
    collection_output = tmp_path / "collection-final"
    finalize_shards(
        collection_staging / "collect_main",
        collection_staging / "collect_lbr",
        collection_output,
        data_dir,
    )
    derived_staging = tmp_path / "derived-staging"
    write_derived_shards(data_dir, derived_staging)
    return data_dir, derived_staging, collection_output / "transcripts.jsonl"


def test_derived_finalization_aligns_every_transcript_without_mutation(
    finalized_fixture, tmp_path: Path
) -> None:
    data_dir, staging, transcripts = finalized_fixture
    sources = sorted(staging.rglob("*.npz"))
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    output = tmp_path / "derived-final"
    result = finalize_derivations(
        staging / "derived_main",
        staging / "derived_lbr",
        transcripts,
        output,
        data_dir,
    )

    assert result["transcripts"] == 9
    with np.load(output / "receipt_final.npz", allow_pickle=False) as cache:
        assert cache["X"].shape == (9, 2, 3)
        assert cache["position"].item() == "receipt_final"
        assert cache["positive_class"].item() == "REAL"
        assert cache["positive_label"].item() == 1
    behavior = [
        json.loads(line)
        for line in (output / "behavior.jsonl").read_text().splitlines()
    ]
    assert len(behavior) == 9
    claimed_indexes = [
        index
        for index, row in enumerate(behavior)
        if row["source_row_id"] in {"real_a_t00", "sham_a_t00"}
    ]
    assert len(claimed_indexes) == 3
    with np.load(output / "receipt_final.npz", allow_pickle=False) as cache:
        # Three rollout rows from the shared claimed prompt reuse one derivation.
        np.testing.assert_array_equal(
            cache["X"][claimed_indexes[0]], cache["X"][claimed_indexes[1]]
        )
        np.testing.assert_array_equal(
            cache["X"][claimed_indexes[1]], cache["X"][claimed_indexes[2]]
        )
    assert all(row["spend_hold_log_odds"] == pytest.approx(0.6) for row in behavior)
    assert all(row["spend_token_ids"] == [10, 11] for row in behavior)
    manipulation = [
        json.loads(line)
        for line in (output / "manipulation.jsonl").read_text().splitlines()
    ]
    assert len(manipulation) == 2
    assert all(row["probability_0_to_100"] == 75 for row in manipulation)
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
    } == before


def test_complete_derived_inventory_is_valid_for_resumption(finalized_fixture) -> None:
    data_dir, staging, _ = finalized_fixture
    main, main_shards = inventory_derived_shards(
        staging / "derived_main", "main", data_dir
    )
    lbr, lbr_shards = inventory_derived_shards(staging / "derived_lbr", "lbr", data_dir)
    assert (main.expected_shards, main.retained_shards, main.valid_shards) == (3, 3, 3)
    assert (lbr.expected_shards, lbr.retained_shards, lbr.valid_shards) == (2, 2, 2)
    assert main.complete and lbr.complete
    assert len(main_shards) == 3
    assert len(lbr_shards) == 2


@pytest.mark.parametrize(
    "fault", ["missing", "unexpected", "duplicate", "corrupt", "prompt", "model"]
)
def test_derived_inventory_fails_closed(finalized_fixture, fault: str) -> None:
    data_dir, staging, _ = finalized_fixture
    directory = staging / "derived_main"
    target = min(directory.glob("*.npz"))
    if fault == "missing":
        target.unlink()
    elif fault == "unexpected":
        (directory / f"{target.name}.partial").write_bytes(b"partial")
    elif fault == "duplicate":
        nested = directory / "duplicate"
        nested.mkdir()
        shutil.copy2(target, nested / target.name)
    elif fault == "corrupt":
        target.write_bytes(b"not an npz")
    else:
        with np.load(target, allow_pickle=False) as archive:
            values = {name: np.asarray(archive[name]) for name in archive.files}
        values[fault] = np.asarray("wrong")
        np.savez_compressed(target, **values)

    with pytest.raises(DerivedInventoryError) as caught:
        inventory_derived_shards(directory, "main", data_dir)
    assert not caught.value.report.complete


def test_finalization_rejects_transcript_prompt_mismatch_before_output(
    finalized_fixture, tmp_path: Path
) -> None:
    data_dir, staging, transcripts = finalized_fixture
    rows = [json.loads(line) for line in transcripts.read_text().splitlines()]
    rows[0]["prompt"] = "tampered"
    wrong = tmp_path / "wrong-transcripts.jsonl"
    wrong.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="prompt mismatch"):
        finalize_derivations(
            staging / "derived_main",
            staging / "derived_lbr",
            wrong,
            output,
            data_dir,
        )
    assert not output.exists()
