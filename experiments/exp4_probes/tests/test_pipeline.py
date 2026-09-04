from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.exp4_probes.results_table import (
    build_head_to_head_result,
    read_score_file,
    render_head_to_head,
    run_two_position_head_to_head,
)
from experiments.exp4_probes.train import run_pipeline


def test_whole_pipeline_runs_on_cpu_random_tensors(
    synthetic_fixture: tuple[Path, Path], tmp_path: Path
) -> None:
    cache_path, metadata_path = synthetic_fixture
    output = tmp_path / "results"
    summary = run_pipeline(cache_path, metadata_path, output, max_iter=400)

    assert len(summary["contrasts"]) == 3
    assert (output / "results.json").is_file()
    for contrast in summary["contrasts"]:
        assert 0 <= contrast["selected_layer"] < 4
        assert len(contrast["layers"]) == 4
        split = contrast["split"]
        assert not set(split["train_groups"]) & set(split["validation_groups"])
        assert not set(split["train_groups"]) & set(split["test_groups"])
        assert not set(split["validation_groups"]) & set(split["test_groups"])
        assert len(contrast["low_base_rates"]) == 3

        artifacts = contrast["artifacts"]
        scores_path = Path(artifacts["scores"])
        score_rows = read_score_file(scores_path)
        assert score_rows
        for raw in map(json.loads, scores_path.read_text().splitlines()):
            assert set(raw) == {"transcript_id", "score", "condition"}

        with np.load(artifacts["directions"], allow_pickle=False) as directions:
            assert directions["weights"].shape == (4, 10)
            assert int(directions["selected_layer"]) == contrast["selected_layer"]
        table = Path(artifacts["head_to_head_scaffold"]).read_text()
        assert "Probe vs CoT" in table
        assert "Pending both aligned score files" in table


def test_head_to_head_consumes_same_contract(tmp_path: Path) -> None:
    probe_path = tmp_path / "probe.jsonl"
    cot_path = tmp_path / "cot.jsonl"
    conditions = ["a", "a", "a", "a", "b", "b", "b", "b"]
    probe_scores = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
    cot_scores = [0.2, 0.4, 0.55, 0.6, 0.45, 0.55, 0.7, 0.8]
    for path, scores in ((probe_path, probe_scores), (cot_path, cot_scores)):
        rows = [
            {"transcript_id": f"tx-{index}", "score": score, "condition": condition}
            for index, (score, condition) in enumerate(
                zip(scores, conditions, strict=True)
            )
        ]
        if path == cot_path:
            rows.reverse()  # contract aligns by transcript id, not file row order
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    table = render_head_to_head(
        negative_condition="claimed",
        positive_condition="verified",
        probe_rows=read_score_file(probe_path),
        cot_rows=read_score_file(cot_path),
    )
    assert "Canonical v1 matched-recall comparison" in table
    assert "Earlier-lock matched-FPR comparison" in table
    assert "projected precision is probe" in table
    assert "CoT misses" in table
    assert "Pending" not in table


def _write_world_scores(path: Path, real: list[float], sham: list[float]) -> None:
    rows = [
        {"transcript_id": f"real-{index}", "score": score, "condition": "REAL"}
        for index, score in enumerate(real)
    ] + [
        {"transcript_id": f"sham-{index}", "score": score, "condition": "SHAM"}
        for index, score in enumerate(sham)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_two_position_exact_h3_requires_both_positions(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.jsonl"
    prompt = tmp_path / "prompt.jsonl"
    cot = tmp_path / "cot.jsonl"
    _write_world_scores(cot, [0.9, 0.8, 0.4, 0.3], [0.7, 0.6, 0.2, 0.1])
    _write_world_scores(receipt, [0.9, 0.8, 0.2, 0.1], [0.4, 0.3, 0.2, 0.1])
    _write_world_scores(prompt, [0.95, 0.85, 0.3, 0.2], [0.45, 0.35, 0.25, 0.15])

    result = run_two_position_head_to_head(
        receipt, prompt, cot, tmp_path / "head_to_head"
    )

    assert result["overall_h3"]["status"] == "pass"
    assert result["overall_h3"]["position_conclusions"] == {
        "receipt_final": "supported",
        "prompt_final": "supported",
    }
    for position in ("receipt_final", "prompt_final"):
        comparison = result["positions"][position]["canonical_locked_comparison"]
        assert comparison["cot_threshold"] == 0.5
        assert comparison["projected_mixture"] == "REAL:SHAM = 1:50"
        assert (
            comparison["probe_projected_precision"]
            > comparison["cot_projected_precision"]
        )
        assert (tmp_path / "head_to_head" / f"{position}_probe_vs_cot.md").is_file()
        assert (tmp_path / "head_to_head" / f"{position}_probe_vs_cot.json").is_file()
    assert (tmp_path / "head_to_head" / "h3_head_to_head.json").is_file()

    # Equal performance at either preregistered position makes overall H3 fail.
    _write_world_scores(prompt, [0.9, 0.8, 0.4, 0.3], [0.95, 0.85, 0.2, 0.1])
    result = run_two_position_head_to_head(
        receipt, prompt, cot, tmp_path / "head_to_head_not_supported"
    )
    assert result["overall_h3"]["status"] == "not_supported"
    assert "prompt_final" in result["overall_h3"]["reason"]


def test_closest_recall_tie_chooses_highest_probe_threshold(tmp_path: Path) -> None:
    probe = tmp_path / "probe.jsonl"
    cot = tmp_path / "cot.jsonl"
    _write_world_scores(cot, [0.9, 0.8, 0.7, 0.4], [0.4, 0.3, 0.2, 0.1])
    _write_world_scores(probe, [0.9, 0.8, 0.3, 0.3], [0.4, 0.3, 0.2, 0.1])

    result = build_head_to_head_result(
        position="receipt_final",
        probe_rows=read_score_file(probe),
        cot_rows=read_score_file(cot),
    )

    comparison = result["canonical_locked_comparison"]
    assert comparison["target_recall"] == 0.75
    assert comparison["probe_recall"] == 0.5
    assert comparison["probe_threshold"] == 0.8


def test_two_position_h3_reports_unavailable_with_precise_reason(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "receipt.jsonl"
    prompt = tmp_path / "prompt.jsonl"
    cot = tmp_path / "cot.jsonl"
    scores_real = [0.4, 0.3]
    scores_sham = [0.2, 0.1]
    for path in (receipt, prompt, cot):
        _write_world_scores(path, scores_real, scores_sham)

    result = run_two_position_head_to_head(
        receipt, prompt, cot, tmp_path / "unavailable"
    )

    assert result["overall_h3"]["status"] == "unavailable"
    assert "no predicted positives" in result["overall_h3"]["reason"]
    persisted = json.loads(
        (tmp_path / "unavailable" / "h3_head_to_head.json").read_text()
    )
    assert persisted["overall_h3"]["status"] == "unavailable"


def test_score_contract_accepts_worlds_and_preserves_abc_aliases(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scores.jsonl"
    rows = [
        {"transcript_id": "one", "score": 0.9, "condition": "REAL"},
        {"transcript_id": "two", "score": 0.1, "condition": "sham"},
        {"transcript_id": "three", "score": 0.2, "condition": "a"},
        {"transcript_id": "four", "score": 0.8, "condition": "c"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    assert [row.condition for row in read_score_file(path)] == [
        "REAL",
        "SHAM",
        "claimed",
        "causally_binding",
    ]
