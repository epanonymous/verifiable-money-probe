from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.exp4_analysis.full_information_text_baseline import (
    RULES,
    _binary_auroc,
    _load_jsonl,
    _metrics,
    build_report,
)
from experiments.exp3_dataset.build_dataset import build_dataset

ROOT = Path(__file__).resolve().parents[3]
COMMITTED = ROOT / "experiments/exp4_analysis/results/wave4/full_information_text_baseline.json"
MAIN = ROOT / "experiments/exp3_dataset/data/prompts_main.jsonl"
LBR = ROOT / "experiments/exp3_dataset/data/lowbaserate_eval.jsonl"
LEAK_FREE_DATA = ROOT / "experiments/exp3_dataset/data_leak_free"
LEAK_FREE_MAIN = LEAK_FREE_DATA / "prompts_main.jsonl"
LEAK_FREE_LBR = LEAK_FREE_DATA / "lowbaserate_eval.jsonl"
LEAK_FREE_REPORT = (
    ROOT / "experiments/exp4_analysis/results/exp4b/full_information_text_baseline.json"
)


def test_committed_baseline_regenerates_from_prompts() -> None:
    committed = json.loads(COMMITTED.read_text(encoding="utf-8"))
    regenerated = build_report(MAIN, LBR)
    assert regenerated["evaluations"] == committed["evaluations"]
    assert regenerated["class_contract"] == committed["class_contract"]
    assert regenerated["inputs"]["main"]["sha256"] == committed["inputs"]["main"]["sha256"]
    assert (
        regenerated["inputs"]["low_base_rate"]["sha256"]
        == committed["inputs"]["low_base_rate"]["sha256"]
    )


def test_rules_are_unfitted_literals() -> None:
    assert RULES == (
        ("settled_true", '"settled": true'),
        ("usdc_balance_1_9", '"usdc_balance": 1.9'),
        ("tx_confirmed_true", '"tx_confirmed": true'),
    )


def test_heldout_and_lbr_are_perfect() -> None:
    report = build_report(MAIN, LBR)
    for name in (
        "main_train_b_plus_c_unique_prompts",
        "main_heldout_b_plus_c_unique_prompts",
        "independent_low_base_rate_cache",
    ):
        block = report["evaluations"][name]
        assert block["all_rules_zero_errors"] is True
        assert block["all_rules_identical_predictions"] is True
        for rule in block["rules"]:
            metrics = rule["metrics"]
            assert metrics["false_positives"] == 0
            assert metrics["false_negatives"] == 0
            assert metrics["precision"] == 1.0
            assert metrics["recall"] == 1.0


def test_leak_free_string_baseline_cannot_separate_classes() -> None:
    """Acceptance gate: full subject-visible text must give exactly chance AUROC."""

    report = build_report(LEAK_FREE_MAIN, LEAK_FREE_LBR, expect_leak_free=True)
    committed = json.loads(LEAK_FREE_REPORT.read_text(encoding="utf-8"))

    assert report["leak_gate"] == {
        "passed": True,
        "expected_auroc": 0.5,
        "observed_rule_aurocs": [0.5],
        "text_equivalence": {
            "main_REAL_SHAM_byte_identical_pairs": 144,
            "low_base_rate_shared_unique_prompts": 10,
            "low_base_rate_prompt_distribution_identical_by_class": True,
        },
    }
    assert report["leak_gate"] == committed["leak_gate"]
    assert report["evaluations"] == committed["evaluations"]
    assert report["inputs"]["main"]["sha256"] == committed["inputs"]["main"]["sha256"]
    assert (
        report["inputs"]["low_base_rate"]["sha256"]
        == committed["inputs"]["low_base_rate"]["sha256"]
    )
    for evaluation in report["evaluations"].values():
        assert {rule["metrics"]["auroc"] for rule in evaluation["rules"]} == {0.5}
        # The leak-free report describes rule inertness instead of scoring the
        # rules as errors, so no block may carry the v1 failure-sounding fields.
        assert evaluation["all_rules_inert_no_prompt_matches"] is True
        assert "all_rules_zero_errors" not in evaluation
        for rule in evaluation["rules"]:
            assert rule["prompt_ids_matching_literal"] == []
            assert "error_ids" not in rule


def test_leak_gate_audits_condition_a_main_pairs(tmp_path: Path) -> None:
    # Regression: the audit used to run on the b+c evaluation slice, so a leak
    # confined to condition (a) passed the gate. Those rows are labeled and
    # collected, so the gate must cover every condition.
    rows = [
        json.loads(line)
        for line in LEAK_FREE_MAIN.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    poisoned = 0
    for row in rows:
        if row["cond"] == "a" and row["world"] == "REAL":
            row["prompt"] += '\n{"settled": true, "usdc_balance": 1.9}'
            poisoned += 1
    assert poisoned == 48

    main = _write_jsonl(tmp_path / "main_cond_a_leak.jsonl", rows)
    with pytest.raises(ValueError, match="main prompt text differs across classes"):
        build_report(main, LEAK_FREE_LBR, expect_leak_free=True)


def test_run_v1_report_keeps_error_field_semantics() -> None:
    # Run v1 compatibility: the renamed leak-free fields must not leak into the
    # v1 report shape, which downstream comparisons still read.
    report = build_report(MAIN, LBR)
    for evaluation in report["evaluations"].values():
        assert evaluation["all_rules_zero_errors"] is True
        assert "all_rules_inert_no_prompt_matches" not in evaluation
        for rule in evaluation["rules"]:
            assert rule["error_ids"] == []
            assert "prompt_ids_matching_literal" not in rule


def test_raw_evidence_provenance_matches_capture_files() -> None:
    # The manifest's provenance digests are a byte contract over the committed
    # capture files: `sha256sum` on those files must reproduce them.
    manifest = json.loads(
        (LEAK_FREE_DATA / "manifest.json").read_text(encoding="utf-8")
    )
    provenance = manifest["raw_evidence_provenance"]
    evidence_dir = LEAK_FREE_DATA.parent
    for world, filename in (
        ("real", "evidence_real.json"),
        ("sham", "evidence_sham.json"),
    ):
        digest = hashlib.sha256((evidence_dir / filename).read_bytes()).hexdigest()
        assert provenance[f"{world}_sha256"] == digest


def test_leak_gate_rejects_leaky_run_v1_inputs() -> None:
    # Negative control for the acceptance gate above: the same call that passes
    # on the leak-free dataset must reject the original leaky Run v1 dataset.
    # Without this, neutering the audit would leave the leak claim self-fulfilling.
    with pytest.raises(ValueError, match="main prompt text differs across classes"):
        build_report(MAIN, LBR, expect_leak_free=True)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_leak_gate_rejects_class_skewed_low_base_rate_distribution(
    tmp_path: Path,
) -> None:
    # Second negative control: byte-identical main pairs alone must not satisfy
    # the gate. Retarget one SHAM row onto another existing SHAM prompt, which
    # leaves row count, ids and prompt support intact but skews the
    # class-conditional text distribution the gate is supposed to catch.
    rows = [
        json.loads(line)
        for line in LEAK_FREE_LBR.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sham_prompts = sorted({row["prompt"] for row in rows if row["world"] == "SHAM"})
    assert len(sham_prompts) >= 2
    source, target = sham_prompts[0], sham_prompts[1]
    retargeted = 0
    for row in rows:
        if retargeted == 0 and row["world"] == "SHAM" and row["prompt"] == source:
            row["prompt"] = target
            retargeted = 1
    assert retargeted == 1
    assert sorted({row["prompt"] for row in rows if row["world"] == "SHAM"}) == (
        sham_prompts
    )

    skewed = _write_jsonl(tmp_path / "lbr_skewed.jsonl", rows)
    with pytest.raises(ValueError, match="distribution differs across classes"):
        build_report(LEAK_FREE_MAIN, skewed, expect_leak_free=True)


def test_leak_free_dataset_regenerates_byte_for_byte(tmp_path: Path) -> None:
    generated = tmp_path / "data_leak_free"
    build_dataset(generated, leak_free=True)

    assert {path.name for path in generated.iterdir()} == {
        "prompts_main.jsonl",
        "prompts_framing.jsonl",
        "lowbaserate_eval.jsonl",
        "splits.json",
        "manifest.json",
    }
    for committed in LEAK_FREE_DATA.iterdir():
        assert (generated / committed.name).read_bytes() == committed.read_bytes()


def test_builder_default_preserves_run_v1_byte_for_byte(tmp_path: Path) -> None:
    generated = tmp_path / "data"
    build_dataset(generated)

    for name in (
        "prompts_main.jsonl",
        "prompts_framing.jsonl",
        "lowbaserate_eval.jsonl",
        "splits.json",
        "manifest.json",
    ):
        assert (generated / name).read_bytes() == (MAIN.parent / name).read_bytes()


def test_binary_auroc_counts_tied_scores_as_chance() -> None:
    assert _binary_auroc([1, 1, 0, 0], [0.0, 0.0, 0.0, 0.0]) == 0.5


def test_metrics_absent_positive_class_returns_none_not_crash() -> None:
    # No positives: recall / projected precision are undefined and must be
    # reported as None rather than raising ZeroDivisionError.
    metrics = _metrics([0, 0, 0], [0, 0, 1])
    assert metrics["recall"] is None
    assert metrics["projected_precision_REAL_to_SHAM_1_to_50"] is None
    assert metrics["false_positive_rate"] is not None


def test_metrics_absent_negative_class_returns_none_fpr() -> None:
    # No negatives: FPR is undefined -> None, but recall stays well-defined.
    metrics = _metrics([1, 1], [1, 0])
    assert metrics["false_positive_rate"] is None
    assert metrics["recall"] == 0.5
    assert metrics["projected_precision_REAL_to_SHAM_1_to_50"] is None


def test_load_jsonl_rejects_nonstring_prompt(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps({"id": "x", "world": "REAL", "cond": "b", "label": 1, "prompt": 123})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="prompt must be a string"):
        _load_jsonl(bad)


def test_load_jsonl_enforces_class_contract(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps({"id": "x", "world": "REAL", "cond": "b", "label": 0, "prompt": "p"})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="REAL=1 / SHAM=0"):
        _load_jsonl(bad)


def test_build_report_fails_closed_on_wrong_inventory(tmp_path: Path) -> None:
    # A well-formed but wrong-sized dataset must fail closed, so artifact
    # existence can never be self-fulfilling.
    main = tmp_path / "main.jsonl"
    main.write_text(
        json.dumps(
            {
                "id": "1",
                "world": "REAL",
                "cond": "b",
                "label": 1,
                "prompt": '"settled": true',
                "split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    lbr = tmp_path / "lbr.jsonl"
    lbr.write_text(
        json.dumps(
            {"id": "2", "world": "SHAM", "cond": "b", "label": 0, "prompt": "no"}
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unexpected main"):
        build_report(main, lbr)


def test_cli_emits_structured_failure_report_and_exits_nonzero(tmp_path: Path) -> None:
    """A missing input must be machine-readable failure, never a false PASS."""
    missing = tmp_path / "missing-main.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.exp4_analysis.full_information_text_baseline",
            "--main",
            str(missing),
            "--low-base-rate",
            str(LBR),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert json.loads(result.stderr) == {
        "status": "failed",
        "error": "FileNotFoundError",
        "detail": f"[Errno 2] No such file or directory: '{missing}'",
    }
