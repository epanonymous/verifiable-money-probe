from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.exp4_probes import train as train_module
from experiments.exp4_probes.train import run_locked_checkpoint
from experiments.exp4_score_contract import read_score_file


def write_locked_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    rng = np.random.default_rng(123)
    metadata = []
    prompts = []
    transcript_ids = []
    labels = []
    features = []
    behavior = []
    manipulation = []
    conditions = (("claimed", 0), ("verified", 1), ("causally_binding", 2))
    for template in range(15):
        split = "heldout" if template >= 12 else "train"
        for condition, condition_index in conditions:
            for world, label in (("REAL", 1), ("SHAM", 0)):
                source_row_id = f"{world.lower()}_{condition}_t{template:02d}"
                transcript_id = f"main:{source_row_id}:r00"
                prompt = (
                    f"claimed prompt {template}"
                    if condition == "claimed"
                    else f"{condition} prompt {template} {world}"
                )
                metadata.append(
                    {
                        "transcript_id": transcript_id,
                        "source_collection": "main",
                        "source_row_id": source_row_id,
                        "rollout_index": 0,
                        "prompt": prompt,
                        "world": world,
                        "condition": condition,
                        "template_id": template,
                        "split": split,
                        "label": label,
                        "model": "fixture/model",
                    }
                )
                prompts.append(prompt)
                transcript_ids.append(transcript_id)
                labels.append(label)
                X = rng.normal(0, 0.25, size=(3, 8))
                X[:, 0] += condition_index * 2.0
                X[1, 1] += 3.0 if world == "REAL" else -3.0
                X[2, 2] += 0.2 if world == "REAL" else -0.2
                if split == "heldout":
                    X[2, 2] += 20.0 if world == "REAL" else -20.0
                features.append(X)
                spend_log_odds = (0.8 if world == "REAL" else 0.0) + 0.1 * (
                    condition == "causally_binding"
                )
                behavior.append(
                    {
                        **metadata[-1],
                        "spend_candidate": "SPEND",
                        "hold_candidate": "HOLD",
                        "spend_token_ids": [41, 42],
                        "hold_token_ids": [43],
                        "spend_logprob": spend_log_odds - 1,
                        "hold_logprob": -1,
                        "spend_hold_log_odds": spend_log_odds,
                    }
                )
                if split == "heldout":
                    probability = (
                        50
                        if condition == "claimed"
                        else (85 if world == "REAL" else 15)
                    )
                    manipulation.append(
                        {
                            **metadata[-1],
                            "direct_prompt": f"direct {prompt}",
                            "raw_response": str(probability),
                            "parse_ok": True,
                            "probability_0_to_100": probability,
                            "parse_error": None,
                        }
                    )

    # Independent evaluation only: exact locked 10 REAL / 1000 SHAM.
    for world, count, label in (("REAL", 10, 1), ("SHAM", 1000, 0)):
        for index in range(count):
            source_row_id = f"lbr_{world.lower()}_{index:04d}"
            transcript_id = f"lbr:{source_row_id}:r00"
            prompt = f"lbr {world} prompt {index}"
            metadata.append(
                {
                    "transcript_id": transcript_id,
                    "source_collection": "lbr",
                    "source_row_id": source_row_id,
                    "rollout_index": 0,
                    "prompt": prompt,
                    "world": world,
                    "condition": "verified",
                    "template_id": 38 + index % 10,
                    "split": "heldout",
                    "label": label,
                    "model": "fixture/model",
                }
            )
            prompts.append(prompt)
            transcript_ids.append(transcript_id)
            labels.append(label)
            X = rng.normal(0, 0.25, size=(3, 8))
            X[:, 0] += 2.0
            X[1, 1] += 3.0 if world == "REAL" else -3.0
            features.append(X)

    metadata_path = tmp_path / "transcripts.jsonl"
    metadata_path.write_text(
        "".join(json.dumps(row) + "\n" for row in metadata), encoding="utf-8"
    )
    behavior_path = tmp_path / "behavior.jsonl"
    behavior_path.write_text(
        "".join(json.dumps(row) + "\n" for row in behavior), encoding="utf-8"
    )
    manipulation_path = tmp_path / "manipulation.jsonl"
    manipulation_path.write_text(
        "".join(json.dumps(row) + "\n" for row in manipulation), encoding="utf-8"
    )
    cache_paths = []
    for position in ("receipt_final", "prompt_final"):
        path = tmp_path / f"{position}.npz"
        np.savez_compressed(
            path,
            X=np.asarray(features, dtype=np.float16),
            y=np.asarray(labels, dtype=np.int8),
            prompts=np.asarray(prompts),
            model=np.asarray("fixture/model"),
            transcript_ids=np.asarray(transcript_ids),
            position=np.asarray(position),
            label_name=np.asarray("world"),
            positive_class=np.asarray("REAL"),
            positive_label=np.asarray(1),
            negative_class=np.asarray("SHAM"),
            negative_label=np.asarray(0),
        )
        cache_paths.append(path)
    return (
        cache_paths[0],
        cache_paths[1],
        metadata_path,
        behavior_path,
        manipulation_path,
    )


def test_one_command_locked_checkpoint_isolated_selection_and_lbr(
    tmp_path: Path,
) -> None:
    receipt, prompt, metadata, behavior, manipulation = write_locked_fixture(tmp_path)
    output = tmp_path / "results"
    summary = run_locked_checkpoint(
        receipt,
        prompt,
        metadata,
        behavior,
        manipulation,
        output,
        max_iter=400,
        bootstrap_replicates=300,
    )

    assert (output / "checkpoint_results.json").is_file()
    assert summary["positive_class"] == "REAL"
    assert summary["positive_label"] == 1
    assert summary["fixed_hyperparameters"]["score_threshold"] == 0.5
    fallback = summary["manipulation"]["declared_evidence_bearing_fallback_b_plus_c"]
    assert summary["manipulation"]["full_condition_set"]["status"] == "unavailable"
    assert fallback["gate_passed"]

    for position in ("receipt_final", "prompt_final"):
        result = summary["positions"][position]
        primary = result["primary"]
        assert primary["positive_class"] == "REAL"
        assert primary["negative_class"] == "SHAM"
        assert primary["selected_layer"] == 1
        assert primary["heldout_test_never_selects_layer"]
        assert primary["split"]["heldout_test_groups"] == ["12", "13", "14"]
        assert not set(primary["split"]["train_groups"]) & {"12", "13", "14"}
        # Layer 2 has perfect heldout-only signal but cannot select itself.
        assert primary["layers"][2]["heldout_test"]["auroc"] == 1.0
        assert primary["layers"][2]["validation"]["auroc"] < 1.0
        assert len(result["required_pairwise_condition_separations"]) == 3
        lbr = primary["independent_lbr"]
        assert lbr["trained_on_lbr"] is False
        assert lbr["actual_counts"] == {"REAL": 10, "SHAM": 1000}
        assert lbr["actual_REAL_prevalence"] == pytest.approx(10 / 1010)
        assert [item["name"] for item in lbr["projected"]] == [
            "canonical_projected_REAL_to_SHAM_1_to_50",
            "canonical_projected_SHAM_to_REAL_1_to_50",
        ]
        assert all(item["score_threshold"] == 0.5 for item in lbr["projected"])
        with np.load(
            primary["artifacts"]["directions_all_layers_and_selected"],
            allow_pickle=False,
        ) as directions:
            assert directions["weights"].shape == (3, 8)
            assert directions["selected_weight"].shape == (8,)
            assert directions["positive_class"].item() == "REAL"

    assert summary["behavior"]["pooled_b_plus_c"]["verdict"] == "behavioral"
    for regression in summary["beyond_condition_regressions"].values():
        assert regression["status"] == "available"
        assert regression["beta_score"] > 0

    strict = {
        position: read_score_file(
            summary["positions"][position]["primary"]["artifacts"][
                "heldout_scores_strict_interchange"
            ]
        )
        for position in ("receipt_final", "prompt_final")
    }
    assert [row.transcript_id for row in strict["receipt_final"]] == [
        row.transcript_id for row in strict["prompt_final"]
    ]
    assert {row.condition for row in strict["receipt_final"]} == {"REAL", "SHAM"}
    assert summary["head_to_head_inputs"]["heldout_b_plus_c_transcripts"] == len(
        strict["receipt_final"]
    )


def test_tail_resume_cli_reuses_sealed_results_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, prompt, metadata, behavior, manipulation = write_locked_fixture(tmp_path)
    output = tmp_path / "results"
    full_summary = run_locked_checkpoint(
        receipt,
        prompt,
        metadata,
        behavior,
        manipulation,
        output,
        max_iter=400,
        bootstrap_replicates=300,
    )
    sealed_paths = [
        output / "manipulation_results.json",
        *(
            output / position / "position_results.json"
            for position in ("receipt_final", "prompt_final")
        ),
    ]
    sealed_before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in sealed_paths
    }
    for name in (
        "behavior_results.json",
        "beyond_condition_regressions.json",
        "checkpoint_results.json",
    ):
        (output / name).unlink()

    def unexpected_refit(*args: object, **kwargs: object) -> None:
        raise AssertionError("tail resume attempted to refit a probe")

    monkeypatch.setattr(train_module, "_run_locked_position", unexpected_refit)
    train_module.main(
        [
            "--resume-locked-tail",
            "--behavior",
            str(behavior),
            "--output-dir",
            str(output),
            "--max-iter",
            "400",
            "--bootstrap-replicates",
            "300",
        ]
    )

    resumed = json.loads((output / "checkpoint_results.json").read_text())
    assert resumed == full_summary
    assert (output / "behavior_results.json").is_file()
    assert (output / "beyond_condition_regressions.json").is_file()
    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in sealed_paths
    } == sealed_before

    tail_paths = [
        output / "behavior_results.json",
        output / "beyond_condition_regressions.json",
        output / "checkpoint_results.json",
    ]
    tail_before = {path: path.read_bytes() for path in tail_paths}
    prompt_strict = Path(
        resumed["positions"]["prompt_final"]["primary"]["artifacts"][
            "heldout_scores_strict_interchange"
        ]
    )
    original_prompt_strict = prompt_strict.read_bytes()
    strict_lines = prompt_strict.read_text(encoding="utf-8").splitlines()
    first = json.loads(strict_lines[0])
    first["transcript_id"] = "misaligned-transcript"
    strict_lines[0] = json.dumps(first)
    prompt_strict.write_text("\n".join(strict_lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="strict heldout scores are not aligned"):
        train_module.resume_locked_checkpoint_tail(
            behavior, output, max_iter=400, bootstrap_replicates=300
        )
    assert {path: path.read_bytes() for path in tail_paths} == tail_before

    prompt_strict.write_bytes(original_prompt_strict)
    prompt_position = output / "prompt_final" / "position_results.json"
    prompt_position.unlink()
    with pytest.raises(FileNotFoundError, match="missing sealed prompt_final"):
        train_module.resume_locked_checkpoint_tail(
            behavior, output, max_iter=400, bootstrap_replicates=300
        )
    assert {path: path.read_bytes() for path in tail_paths} == tail_before
