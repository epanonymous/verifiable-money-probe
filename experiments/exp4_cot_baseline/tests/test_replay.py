from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.exp4_cot_baseline.replay import (
    DEFAULT_AUDIT,
    DEFAULT_MANIFEST,
    DEFAULT_SCORES,
    DEFAULT_STATUS,
    verify_replay,
)


def copy_bundle(tmp_path: Path) -> dict[str, Path]:
    bundle = tmp_path / "wave2"
    baseline = bundle / "blind_baseline"
    baseline.mkdir(parents=True)
    paths = {
        "scores_path": baseline / DEFAULT_SCORES.name,
        "status_path": baseline / DEFAULT_STATUS.name,
        "audit_path": baseline / DEFAULT_AUDIT.name,
        "manifest_path": bundle / DEFAULT_MANIFEST.name,
    }
    for source, key in (
        (DEFAULT_SCORES, "scores_path"),
        (DEFAULT_STATUS, "status_path"),
        (DEFAULT_AUDIT, "audit_path"),
        (DEFAULT_MANIFEST, "manifest_path"),
    ):
        paths[key].write_bytes(source.read_bytes())
    return paths


def rewrite_manifest(path: Path, field: str, value: object) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    score_entry = next(
        entry
        for entry in manifest["artifacts"]
        if entry["path"] == "blind_baseline/claude_sonnet_scores.jsonl"
    )
    score_entry[field] = value
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_committed_monitor_archive_replays_without_provider() -> None:
    summary = verify_replay()

    assert summary["status"] == "passed"
    assert summary["provider_calls"] == 0
    assert summary["network_calls"] == 0
    assert summary["score_rows"] == 1000
    assert summary["class_counts"] == {"REAL": 500, "SHAM": 500}
    assert summary["audit_events"] == 1004
    assert summary["completed_invocations"] == 1000
    assert summary["error_events"] == 4
    assert "live Claude judgments" in summary["scope"]["not_reproduced"]


def test_byte_faithful_relocated_bundle_replays(tmp_path: Path) -> None:
    summary = verify_replay(**copy_bundle(tmp_path))

    assert summary["status"] == "passed"
    assert summary["score_rows"] == 1000


def test_replay_fails_closed_on_score_tampering(tmp_path: Path) -> None:
    paths = copy_bundle(tmp_path)
    with paths["scores_path"].open("ab") as handle:
        handle.write(b"\n")

    with pytest.raises(ValueError, match="score archive SHA-256 does not match status"):
        verify_replay(**paths)


def test_replay_fails_closed_on_incomplete_status(tmp_path: Path) -> None:
    status_data = json.loads(DEFAULT_STATUS.read_text(encoding="utf-8"))
    status_data["status"] = "incomplete"
    status = tmp_path / "status.json"
    status.write_text(json.dumps(status_data), encoding="utf-8")

    with pytest.raises(ValueError, match="status is not complete"):
        verify_replay(status_path=status)


def test_replay_fails_closed_on_noncontiguous_audit(tmp_path: Path) -> None:
    rows = [
        json.loads(line)
        for line in DEFAULT_AUDIT.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["invocation"] = 2
    audit = tmp_path / "audit.jsonl"
    audit.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="not the exact contiguous sequence"):
        verify_replay(audit_path=audit)


def test_replay_fails_closed_on_manifest_size_mismatch(tmp_path: Path) -> None:
    paths = copy_bundle(tmp_path)
    rewrite_manifest(
        paths["manifest_path"], "size_bytes", paths["scores_path"].stat().st_size + 1
    )

    with pytest.raises(ValueError, match="manifest size does not match"):
        verify_replay(**paths)


def test_replay_refuses_manifest_authentication_for_arbitrary_copy(
    tmp_path: Path,
) -> None:
    copied_scores = tmp_path / DEFAULT_SCORES.name
    copied_scores.write_bytes(DEFAULT_SCORES.read_bytes())

    with pytest.raises(ValueError, match="cannot be authenticated by this manifest"):
        verify_replay(scores_path=copied_scores)


def test_replay_fails_closed_when_recorded_blindness_did_not_pass(
    tmp_path: Path,
) -> None:
    status_data = json.loads(DEFAULT_STATUS.read_text(encoding="utf-8"))
    status_data["blindness"]["status"] = "failed_closed"
    status = tmp_path / "status.json"
    status.write_text(json.dumps(status_data), encoding="utf-8")

    with pytest.raises(ValueError, match="blindness status did not pass"):
        verify_replay(status_path=status)


def test_replay_fails_closed_on_manifest_hash_mismatch(tmp_path: Path) -> None:
    paths = copy_bundle(tmp_path)
    rewrite_manifest(paths["manifest_path"], "sha256", "0" * 64)

    with pytest.raises(ValueError, match="manifest SHA-256 does not match"):
        verify_replay(**paths)


def test_replay_fails_closed_on_prompt_tampering(tmp_path: Path) -> None:
    prompt = tmp_path / "judge.json"
    prompt.write_bytes(DEFAULT_STATUS.read_bytes())

    with pytest.raises(ValueError, match="judge prompt SHA-256 does not match"):
        verify_replay(prompt_path=prompt)


def test_replay_cli_emits_structured_failure_without_traceback(tmp_path: Path) -> None:
    status_data = json.loads(DEFAULT_STATUS.read_text(encoding="utf-8"))
    status_data["status"] = "incomplete"
    status = tmp_path / "status.json"
    status.write_text(json.dumps(status_data), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.exp4_cot_baseline.replay",
            "--status",
            str(status),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    error = json.loads(result.stderr)
    assert error["status"] == "failed"
    assert error["error_type"] == "ValueError"
    assert "not complete" in error["error"]


def test_replay_cli_structures_unexpected_prompt_shape_failure(tmp_path: Path) -> None:
    prompt = tmp_path / "judge.json"
    prompt.write_text("[]", encoding="utf-8")
    status_data = json.loads(DEFAULT_STATUS.read_text(encoding="utf-8"))
    status_data["prompt"]["sha256"] = hashlib.sha256(prompt.read_bytes()).hexdigest()
    status = tmp_path / "status.json"
    status.write_text(json.dumps(status_data), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.exp4_cot_baseline.replay",
            "--status",
            str(status),
            "--prompt",
            str(prompt),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    error = json.loads(result.stderr)
    assert error["status"] == "failed"
    assert error["error_type"] == "ValueError"
    assert "JSON object" in error["error"]
    assert "traceback" not in result.stderr.casefold()
