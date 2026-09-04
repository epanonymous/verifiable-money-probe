"""Strict three-field score interchange shared by Wave 2 probes and monitors."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


SCORE_KEYS = frozenset({"transcript_id", "score", "condition"})
CONDITION_ALIASES = {
    "a": "claimed",
    "b": "verified",
    "c": "causally_binding",
    "claimed": "claimed",
    "verified": "verified",
    "binding": "causally_binding",
    "causally-binding": "causally_binding",
    "causally_binding": "causally_binding",
    "framing": "framing",
    "real": "REAL",
    "sham": "SHAM",
}


@dataclass(frozen=True)
class ScoreRow:
    transcript_id: str
    score: float
    condition: str

    def to_dict(self) -> dict[str, str | float]:
        return {
            "transcript_id": self.transcript_id,
            "score": self.score,
            "condition": self.condition,
        }


def normalize_condition(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("condition must be a non-empty string")
    key = value.strip().lower()
    try:
        return CONDITION_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown condition {value!r}; expected REAL/SHAM, a/b/c, "
            "claimed/verified/causally_binding, or framing"
        ) from exc


def validate_score_row(raw: object, *, source: str) -> ScoreRow:
    if not isinstance(raw, dict):
        raise ValueError(f"{source} score row must be an object")
    if set(raw) != SCORE_KEYS:
        raise ValueError(
            f"{source} must contain exactly {sorted(SCORE_KEYS)}, got {sorted(raw)}"
        )
    transcript_id = raw["transcript_id"]
    if not isinstance(transcript_id, str) or not transcript_id.strip():
        raise ValueError(f"{source} transcript_id must be a non-empty string")
    raw_score = raw["score"]
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
        raise ValueError(f"{source} score must be numeric")
    score = float(raw_score)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{source} score must be finite and in [0, 1]")
    return ScoreRow(
        transcript_id=transcript_id,
        score=score,
        condition=normalize_condition(raw["condition"]),
    )


def read_score_file(path: str | Path) -> list[ScoreRow]:
    """Read the exact JSONL interchange and reject duplicate identifiers."""

    path = Path(path)
    rows: list[ScoreRow] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            rows.append(validate_score_row(raw, source=f"{path}:{line_number}"))
    if not rows:
        raise ValueError(f"{path} contains no score rows")
    ids = [row.transcript_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} contains duplicate transcript_id values")
    return rows
