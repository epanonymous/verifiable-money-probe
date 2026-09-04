from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from experiments.exp4_cot_baseline.runner import run_resumable


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def record(identifier: str, marker: str, world: str) -> dict:
    return {
        "transcript_id": identifier,
        "world": world,
        "condition": "verified",
        "source_collection": "main",
        "split": "heldout",
        "label": int(world == "REAL"),
        "prompt": f"Task text {marker}",
        "response": f"Subject response {marker}",
    }


class ScriptedJudge:
    def __init__(
        self,
        scores: dict[str, float],
        *,
        transient_failures: dict[str, int] | None = None,
        permanent_failure: str | None = None,
        delays: dict[str, float] | None = None,
    ) -> None:
        self.scores = scores
        self.transient_failures = dict(transient_failures or {})
        self.permanent_failure = permanent_failure
        self.delays = delays or {}
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def score(self, messages) -> float:
        text = messages[1]["content"]
        marker = next(marker for marker in self.scores if marker in text)
        with self._lock:
            self.calls.append(marker)
            remaining = self.transient_failures.get(marker, 0)
            if remaining:
                self.transient_failures[marker] = remaining - 1
                raise RuntimeError("transient test failure")
        if marker == self.permanent_failure:
            raise RuntimeError("permanent test failure")
        time.sleep(self.delays.get(marker, 0.0))
        return self.scores[marker]


def fixture_files(tmp_path: Path) -> tuple[Path, Path]:
    transcripts = tmp_path / "transcripts.jsonl"
    probe = tmp_path / "probe.jsonl"
    write_jsonl(
        transcripts,
        [
            record("tx-1", "ALPHA", "REAL"),
            record("tx-2", "BETA", "SHAM"),
            record("tx-3", "GAMMA", "REAL"),
            record("unselected", "DELTA", "SHAM"),
        ],
    )
    write_jsonl(
        probe,
        [
            {"transcript_id": "tx-1", "score": 0.8, "condition": "REAL"},
            {"transcript_id": "tx-2", "score": 0.2, "condition": "SHAM"},
            {"transcript_id": "tx-3", "score": 0.7, "condition": "REAL"},
        ],
    )
    return transcripts, probe


def test_exact_selection_parallel_completion_and_stable_final_order(
    tmp_path: Path,
) -> None:
    transcripts, probe = fixture_files(tmp_path)
    output = tmp_path / "scores.jsonl"
    judge = ScriptedJudge(
        {"ALPHA": 0.91, "BETA": 0.08, "GAMMA": 0.82},
        delays={"ALPHA": 0.03, "BETA": 0.01},
    )

    status = run_resumable(transcripts, probe, output, judge, workers=3)

    assert status["status"] == "complete"
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["transcript_id"] for row in rows] == ["tx-1", "tx-2", "tx-3"]
    assert all(set(row) == {"transcript_id", "score", "condition"} for row in rows)
    assert sorted(judge.calls) == ["ALPHA", "BETA", "GAMMA"]
    assert "DELTA" not in judge.calls


def test_resume_validates_and_skips_completed_rows_with_bounded_retry(
    tmp_path: Path,
) -> None:
    transcripts, probe = fixture_files(tmp_path)
    output = tmp_path / "scores.jsonl"
    write_jsonl(
        output,
        [{"transcript_id": "tx-2", "score": 0.08, "condition": "SHAM"}],
    )
    judge = ScriptedJudge(
        {"ALPHA": 0.91, "BETA": 0.08, "GAMMA": 0.82},
        transient_failures={"ALPHA": 1},
    )

    status = run_resumable(
        transcripts,
        probe,
        output,
        judge,
        workers=1,
        retries=1,
        backoff=0,
    )

    assert status["status"] == "complete"
    assert status["resume"]["validated_existing_rows"] == 1
    assert status["retry_count"] == 1
    assert judge.calls == ["ALPHA", "ALPHA", "GAMMA"]


def test_permanent_error_leaves_explicit_incomplete_resumable_status(
    tmp_path: Path,
) -> None:
    transcripts, probe = fixture_files(tmp_path)
    output = tmp_path / "scores.jsonl"
    status_path = tmp_path / "status.json"
    judge = ScriptedJudge(
        {"ALPHA": 0.91, "BETA": 0.08, "GAMMA": 0.82},
        permanent_failure="ALPHA",
    )

    status = run_resumable(
        transcripts,
        probe,
        output,
        judge,
        status_path=status_path,
        workers=1,
        retries=1,
        backoff=0,
    )

    assert status["status"] == "incomplete"
    assert status["output"]["completed"] == 2
    assert status["output"]["remaining"] == 1
    assert status["errors"][0]["transcript_id"] == "tx-1"
    assert json.loads(status_path.read_text())["status"] == "incomplete"


@pytest.mark.parametrize(
    "existing",
    [
        {"transcript_id": "unknown", "score": 0.5, "condition": "REAL"},
        {
            "transcript_id": "tx-1",
            "score": 0.5,
            "condition": "REAL",
            "extra": "corrupt",
        },
        {"transcript_id": "tx-1", "score": 0.5, "condition": "SHAM"},
    ],
)
def test_corrupt_unknown_or_misaligned_resume_output_fails_before_scoring(
    tmp_path: Path, existing: dict
) -> None:
    transcripts, probe = fixture_files(tmp_path)
    output = tmp_path / "scores.jsonl"
    write_jsonl(output, [existing])
    judge = ScriptedJudge({"ALPHA": 0.9, "BETA": 0.1, "GAMMA": 0.8})

    with pytest.raises(ValueError):
        run_resumable(transcripts, probe, output, judge)
    assert judge.calls == []


def test_probe_id_missing_from_transcripts_fails_before_scoring(tmp_path: Path) -> None:
    transcripts, probe = fixture_files(tmp_path)
    rows = [json.loads(line) for line in probe.read_text().splitlines()]
    rows[0]["transcript_id"] = "missing"
    write_jsonl(probe, rows)
    judge = ScriptedJudge({"ALPHA": 0.9, "BETA": 0.1, "GAMMA": 0.8})

    with pytest.raises(ValueError, match="missing from transcript input"):
        run_resumable(transcripts, probe, tmp_path / "scores.jsonl", judge)
    assert judge.calls == []


def test_duplicate_existing_output_id_fails_before_scoring(tmp_path: Path) -> None:
    transcripts, probe = fixture_files(tmp_path)
    output = tmp_path / "scores.jsonl"
    duplicate = {"transcript_id": "tx-1", "score": 0.9, "condition": "REAL"}
    write_jsonl(output, [duplicate, duplicate])
    judge = ScriptedJudge({"ALPHA": 0.9, "BETA": 0.1, "GAMMA": 0.8})

    with pytest.raises(ValueError, match="duplicate transcript_id"):
        run_resumable(transcripts, probe, output, judge)
    assert judge.calls == []


def test_selection_must_really_be_main_heldout_b_plus_c(tmp_path: Path) -> None:
    transcripts, probe = fixture_files(tmp_path)
    rows = [json.loads(line) for line in transcripts.read_text().splitlines()]
    rows[0]["condition"] = "claimed"
    write_jsonl(transcripts, rows)
    judge = ScriptedJudge({"ALPHA": 0.9, "BETA": 0.1, "GAMMA": 0.8})

    with pytest.raises(ValueError, match=r"main heldout b\+c"):
        run_resumable(transcripts, probe, tmp_path / "scores.jsonl", judge)
    assert judge.calls == []
