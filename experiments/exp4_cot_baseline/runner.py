"""Production-resumable one-transcript-per-process Claude monitor runner."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from experiments.exp4_score_contract import (
    ScoreRow,
    normalize_condition,
    read_score_file,
)

from .blindness import BlindnessError, assert_blind_messages, build_blind_transcript
from .clients import (
    JUDGE_FAMILY,
    SUBJECT_FAMILY,
    AuditPersistenceError,
    ClaudeCliJudge,
    JudgeClient,
)
from .harness import (
    assert_independent_families,
    judge_record,
    read_jsonl,
    transcript_condition,
    transcript_id,
)
from .prompt import PROMPT_PATH, PromptTemplate, load_prompt


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_line(value: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            dict(value), ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
        + "\n"
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_canonical_scores(path: Path, rows: list[ScoreRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(_json_line(row.to_dict()))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


class FsyncAuditLog:
    """Thread-safe invocation audit log with one fsync per event."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.prior_events = self._validate_existing()

    def _validate_existing(self) -> int:
        if not self.path.exists():
            return 0
        count = 0
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid audit JSON at {self.path}:{line_number}"
                    ) from exc
                if not isinstance(event, dict) or event.get("status") not in {
                    "completed",
                    "error",
                }:
                    raise ValueError(
                        f"invalid audit event at {self.path}:{line_number}"
                    )
                count += 1
        return count

    def __call__(self, event: dict[str, Any]) -> None:
        event = dict(event)
        event["invocation"] = self.prior_events + int(event["invocation"])
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(_json_line(event))
            handle.flush()
            os.fsync(handle.fileno())


def _load_selection(
    transcript_path: Path, probe_score_path: Path
) -> tuple[list[ScoreRow], dict[str, dict[str, Any]]]:
    selected = read_score_file(probe_score_path)
    invalid_worlds = sorted(
        {row.condition for row in selected if row.condition not in {"REAL", "SHAM"}}
    )
    if invalid_worlds:
        raise ValueError(
            "primary heldout probe score conditions must be actual worlds REAL/SHAM; "
            f"got {invalid_worlds}"
        )

    records: list[dict[str, Any]] = list(read_jsonl(transcript_path))
    record_ids = [transcript_id(record) for record in records]
    duplicate_ids = sorted(
        item for item, count in Counter(record_ids).items() if count > 1
    )
    if duplicate_ids:
        raise ValueError(
            f"transcript input contains duplicate IDs including {duplicate_ids[:5]}"
        )
    by_id = dict(zip(record_ids, records, strict=True))
    unknown = [row.transcript_id for row in selected if row.transcript_id not in by_id]
    if unknown:
        raise ValueError(
            "probe score file contains IDs missing from transcript input: "
            f"{unknown[:5]}"
        )
    mismatched = [
        row.transcript_id
        for row in selected
        if transcript_condition(by_id[row.transcript_id]) != row.condition
    ]
    if mismatched:
        raise ValueError(
            f"probe/transcript world labels disagree for IDs {mismatched[:5]}"
        )
    invalid_primary_rows = []
    for row in selected:
        record = by_id[row.transcript_id]
        raw_condition = record.get("condition", record.get("cond"))
        try:
            analysis_condition = normalize_condition(raw_condition)
        except ValueError:
            analysis_condition = "invalid"
        if (
            record.get("source_collection") != "main"
            or str(record.get("split", "")).strip().lower() != "heldout"
            or analysis_condition not in {"verified", "causally_binding"}
        ):
            invalid_primary_rows.append(row.transcript_id)
    if invalid_primary_rows:
        raise ValueError(
            "selected IDs must be finalized main heldout b+c transcripts; invalid IDs "
            f"include {invalid_primary_rows[:5]}"
        )
    return selected, by_id


def _load_existing_scores(
    output_path: Path, selected: list[ScoreRow]
) -> dict[str, ScoreRow]:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return {}
    existing = read_score_file(output_path)
    expected = {row.transcript_id: row.condition for row in selected}
    unknown = [
        row.transcript_id for row in existing if row.transcript_id not in expected
    ]
    if unknown:
        raise ValueError(
            f"existing output contains unknown transcript IDs {unknown[:5]}"
        )
    mismatched = [
        row.transcript_id
        for row in existing
        if expected[row.transcript_id] != row.condition
    ]
    if mismatched:
        raise ValueError(
            f"existing output conditions disagree for transcript IDs {mismatched[:5]}"
        )
    return {row.transcript_id: row for row in existing}


def _audit_selected_records(
    selected: list[ScoreRow],
    records: Mapping[str, Mapping[str, Any]],
    prompt: PromptTemplate,
) -> None:
    """Fail closed on every exact would-be request before any scoring starts."""

    for row in selected:
        record = records[row.transcript_id]
        transcript = build_blind_transcript(record)
        messages = prompt.render(transcript)
        assert_blind_messages(messages, record)


def _append_completed(path: Path, row: ScoreRow) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_json_line(row.to_dict()))
        handle.flush()
        os.fsync(handle.fileno())


def _score_with_retries(
    record: Mapping[str, Any],
    judge: JudgeClient,
    prompt: PromptTemplate,
    *,
    retries: int,
    backoff: float,
    sleep: Callable[[float], None],
) -> tuple[dict[str, object] | None, list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    for attempt in range(1, retries + 2):
        try:
            return judge_record(record, judge, prompt=prompt), errors
        except (BlindnessError, AuditPersistenceError) as exc:
            errors.append(
                {
                    "attempt": attempt,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "retryable": False,
                }
            )
            return None, errors
        except Exception as exc:  # noqa: BLE001 - bounded production retry boundary
            retryable = attempt <= retries
            errors.append(
                {
                    "attempt": attempt,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "retryable": retryable,
                }
            )
            if not retryable:
                return None, errors
            sleep(backoff * (2 ** (attempt - 1)))
    raise AssertionError("bounded retry loop fell through")


def run_resumable(
    transcript_path: str | Path,
    probe_score_path: str | Path,
    output_path: str | Path,
    judge: JudgeClient,
    *,
    status_path: str | Path | None = None,
    workers: int = 1,
    retries: int = 2,
    backoff: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    prompt: PromptTemplate | None = None,
    prior_audit_events: int = 0,
) -> dict[str, Any]:
    """Resume exact selected IDs and atomically canonicalize when all are scored."""

    if workers < 1:
        raise ValueError("workers must be at least 1")
    if retries < 0 or backoff < 0:
        raise ValueError("retries and backoff must be non-negative")
    assert_independent_families(SUBJECT_FAMILY, JUDGE_FAMILY)

    transcript_path = Path(transcript_path)
    probe_score_path = Path(probe_score_path)
    output_path = Path(output_path)
    status_path = (
        Path(status_path)
        if status_path is not None
        else output_path.with_name(output_path.name + ".status.json")
    )
    template = prompt or load_prompt()
    selected, records = _load_selection(transcript_path, probe_score_path)
    _audit_selected_records(selected, records, template)
    existing = _load_existing_scores(output_path, selected)
    pending = [row for row in selected if row.transcript_id not in existing]

    judge_metadata: dict[str, Any]
    if isinstance(judge, ClaudeCliJudge):
        judge.preflight()
        judge_metadata = judge.audit_metadata()
    else:
        judge_metadata = {
            "judge_family": JUDGE_FAMILY,
            "subject_family": SUBJECT_FAMILY,
            "client": type(judge).__name__,
        }

    base_status: dict[str, Any] = {
        "format_version": 1,
        "status": "incomplete",
        "judge": judge_metadata,
        "prompt": {
            "version": template.version,
            "sha256": _sha256_file(PROMPT_PATH),
        },
        "inputs": {
            "transcripts_sha256": _sha256_file(transcript_path),
            "probe_scores_sha256": _sha256_file(probe_score_path),
            "selected_transcripts": len(selected),
        },
        "output": {
            "canonical_keys": ["transcript_id", "score", "condition"],
            "condition_contract": "actual world; REAL positive, SHAM negative",
            "score_semantics": "P(subject believed the reward/economic stakes were real)",
            "stable_order_source": "primary heldout probe score file",
            "completed": len(existing),
            "remaining": len(pending),
            "sha256": _sha256_file(output_path) if output_path.exists() else None,
        },
        "blindness": {
            "status": "passed",
            "audited_selected_transcripts": len(selected),
            "raw_verifier_redaction": "required",
            "annotation_canary_audit": "required",
            "request_contains_experimental_ids_or_labels": False,
        },
        "resume": {
            "validated_existing_rows": len(existing),
            "fsync_each_completed_row": True,
            "final_atomic_canonical_file": True,
        },
        "invocation_count": prior_audit_events,
        "retry_count": 0,
        "errors": [],
    }
    _write_json_atomic(status_path, base_status)

    errors: list[dict[str, Any]] = []
    retry_count = 0
    if pending:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _score_with_retries,
                    records[row.transcript_id],
                    judge,
                    template,
                    retries=retries,
                    backoff=backoff,
                    sleep=sleep,
                ): row
                for row in pending
            }
            for future in as_completed(futures):
                expected_row = futures[future]
                result, attempts = future.result()
                retry_count += max(0, len(attempts) - int(result is None))
                if result is None:
                    errors.append(
                        {
                            "transcript_id": expected_row.transcript_id,
                            "attempts": attempts,
                        }
                    )
                else:
                    completed = ScoreRow(
                        transcript_id=str(result["transcript_id"]),
                        score=float(result["score"]),
                        condition=str(result["condition"]),
                    )
                    if completed.transcript_id != expected_row.transcript_id:
                        raise RuntimeError(
                            "judge result transcript ID changed internally"
                        )
                    if completed.condition != expected_row.condition:
                        raise RuntimeError("judge result condition changed internally")
                    _append_completed(output_path, completed)
                    existing[completed.transcript_id] = completed

                base_status["output"]["completed"] = len(existing)
                base_status["output"]["remaining"] = len(selected) - len(existing)
                base_status["output"]["sha256"] = (
                    _sha256_file(output_path) if output_path.exists() else None
                )
                base_status["invocation_count"] = prior_audit_events + int(
                    getattr(judge, "invocation_count", 0)
                )
                if "invocation_count" in base_status["judge"]:
                    base_status["judge"]["invocation_count"] = base_status[
                        "invocation_count"
                    ]
                base_status["retry_count"] = retry_count
                base_status["errors"] = errors
                _write_json_atomic(status_path, base_status)

    complete = len(existing) == len(selected) and not errors
    if complete:
        ordered = [existing[row.transcript_id] for row in selected]
        _write_canonical_scores(output_path, ordered)
        base_status["status"] = "complete"
        base_status["blindness"]["status"] = "passed"
        base_status["output"]["completed"] = len(selected)
        base_status["output"]["remaining"] = 0
        base_status["output"]["sha256"] = _sha256_file(output_path)
    else:
        base_status["status"] = "incomplete"
        if any(
            attempt["error_type"] == "BlindnessError"
            for error in errors
            for attempt in error["attempts"]
        ):
            base_status["blindness"]["status"] = "failed_closed"
        else:
            base_status["blindness"]["status"] = "passed"
    base_status["invocation_count"] = prior_audit_events + int(
        getattr(judge, "invocation_count", 0)
    )
    if "invocation_count" in base_status["judge"]:
        base_status["judge"]["invocation_count"] = base_status["invocation_count"]
    base_status["retry_count"] = retry_count
    base_status["errors"] = errors
    _write_json_atomic(status_path, base_status)
    return base_status


def run_claude_resumable(
    transcript_path: str | Path,
    probe_score_path: str | Path,
    output_path: str | Path,
    *,
    status_path: str | Path | None = None,
    audit_path: str | Path | None = None,
    workers: int = 1,
    retries: int = 2,
    backoff: float = 1.0,
    model_alias: str = "sonnet",
    timeout: float = 120.0,
    executable: str = "claude",
) -> dict[str, Any]:
    output_path = Path(output_path)
    audit_path = (
        Path(audit_path)
        if audit_path is not None
        else output_path.with_name(output_path.name + ".audit.jsonl")
    )
    audit = FsyncAuditLog(audit_path)
    judge = ClaudeCliJudge(
        executable=executable,
        model_alias=model_alias,
        timeout=timeout,
    )
    judge.set_audit_sink(audit)
    return run_resumable(
        transcript_path,
        probe_score_path,
        output_path,
        judge,
        status_path=status_path,
        workers=workers,
        retries=retries,
        backoff=backoff,
        prior_audit_events=audit.prior_events,
    )
