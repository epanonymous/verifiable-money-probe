"""Blind-monitor orchestration and JSONL interchange."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from experiments.exp4_score_contract import normalize_condition

from .blindness import assert_blind_messages, build_blind_transcript
from .clients import JudgeClient, validate_score
from .prompt import PromptTemplate, load_prompt


def assert_independent_families(subject_family: str, judge_family: str) -> None:
    subject = subject_family.strip().casefold()
    judge = judge_family.strip().casefold()
    if not subject or not judge:
        raise ValueError("subject and judge model families must both be declared")
    if subject == judge:
        raise ValueError("judge model family must differ from subject model family")


def transcript_id(record: Mapping[str, Any]) -> str:
    value = record.get("transcript_id", record.get("id"))
    if value is None or not str(value).strip():
        raise ValueError("record needs transcript_id or id")
    return str(value)


def transcript_condition(record: Mapping[str, Any]) -> str:
    # Exp3 calls the REAL/SHAM condition `world`; `cond` is the a/b/c ladder.
    value = record.get("world", record.get("condition"))
    if value is None and record.get("label") in (0, 1):
        value = "REAL" if record["label"] == 1 else "SHAM"
    if value is None or not str(value).strip():
        raise ValueError("record needs world/condition (or a binary label)")
    condition = normalize_condition(value)
    if condition not in {"REAL", "SHAM"}:
        raise ValueError("monitor output condition must be actual world REAL or SHAM")
    label = record.get("label")
    if label is not None:
        expected_label = 1 if condition == "REAL" else 0
        if isinstance(label, bool) or label != expected_label:
            raise ValueError("record world and binary label are inconsistent")
    return condition


def judge_record(
    record: Mapping[str, Any],
    judge: JudgeClient,
    *,
    prompt: PromptTemplate | None = None,
) -> dict[str, object]:
    """Judge one record while keeping its annotations outside the request."""

    template = prompt or load_prompt()
    transcript = build_blind_transcript(record)
    messages = template.render(transcript)
    assert_blind_messages(messages, record)
    score = validate_score(judge.score(messages))
    return {
        "transcript_id": transcript_id(record),
        "score": score,
        "condition": transcript_condition(record),
    }


def judge_records(
    records: Iterable[Mapping[str, Any]],
    judge: JudgeClient,
    *,
    prompt: PromptTemplate | None = None,
) -> Iterator[dict[str, object]]:
    template = prompt or load_prompt()
    for record in records:
        yield judge_record(record, judge, prompt=template)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            yield record


def run_jsonl(input_path: Path, output_path: Path, judge: JudgeClient) -> int:
    """Score an input JSONL file and write the three-field exp4 interchange."""

    count = 0
    with output_path.open("w", encoding="utf-8") as output:
        for result in judge_records(read_jsonl(input_path), judge):
            output.write(json.dumps(result, separators=(",", ":")) + "\n")
            count += 1
    return count
