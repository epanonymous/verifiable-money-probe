"""Construct and audit the ground-truth-blind view shown to the judge."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


class BlindnessError(ValueError):
    """Raised when a would-be judge request contains privileged annotations."""


EVIDENCE_REDACTION = "[raw verification result withheld from monitor]"

# These are annotations, not transcript content. The sanitizer never copies them.
SENSITIVE_METADATA_KEYS = frozenset(
    {
        "condition",
        "cond",
        "world",
        "label",
        "ground_truth",
        "ground_truth_label",
        "expected_condition",
        "evidence",
        "evidence_payload",
        "metadata",
        "transcript_id",
        "id",
        "source_row_id",
        "source_collection",
        "template_id",
        "split",
        "rollout_index",
        "model",
    }
)

# The exp3 verifier output directly identifies REAL/SHAM. It is input to the
# subject model, but must not let the independent monitor decode the condition.
EVIDENCE_KEYS = frozenset(
    {
        "wallet",
        "usdc_balance",
        "settled",
        "tx_confirmed",
        "last_inbound_tx",
        "tx_block",
        "queried_block",
        "queried_at",
    }
)

_EVIDENCE_KEY_PATTERN = "|".join(
    re.escape(key) for key in sorted(EVIDENCE_KEYS, key=len, reverse=True)
)
_JSON_NUMBER_PATTERN = r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?"
_DOUBLE_QUOTED_SCALAR_PATTERN = (
    r'"(?:[^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*"'
)
_SINGLE_QUOTED_SCALAR_PATTERN = r"'(?:[^'\\\x00-\x1f]|\\.)*'"

_GROUND_TRUTH_LINE = re.compile(
    r"(?im)^\s*(?:[-*]\s*)?[\[({]?\s*[\"']?"
    r"(?:condition|cond|world|label|ground[_ -]?truth|expected[_ -]?condition)"
    r"[\"']?\s*[:=]\s*.*?(?:[\])}]?\s*)$"
)
_GROUND_TRUTH_SHAPE = re.compile(
    r"(?i)[\"']?(?:condition|cond|world|label|ground[_ -]?truth|"
    r"expected[_ -]?condition)[\"']?\s*[:=]"
)
_EVIDENCE_SHAPE = re.compile(
    rf"(?i)[\"']?(?:{_EVIDENCE_KEY_PATTERN})[\"']?\s*[:=]"
)
_EVIDENCE_SCALAR_ASSIGNMENT = re.compile(
    rf"""(?ix)
    (?<![\w])
    (?:
        (?P<key_quote>["'])(?:{_EVIDENCE_KEY_PATTERN})(?P=key_quote)
        |
        (?:{_EVIDENCE_KEY_PATTERN})
    )
    \s*[:=]\s*
    (?:
        {_DOUBLE_QUOTED_SCALAR_PATTERN}
        | {_SINGLE_QUOTED_SCALAR_PATTERN}
        | true | false | null
        | {_JSON_NUMBER_PATTERN}
    )
    (?![\w+/\-]|,\d|[{{\[]|\.[\w\d])
    """
)


def _looks_like_evidence_payload(value: object) -> bool:
    return isinstance(value, dict) and bool(EVIDENCE_KEYS.intersection(value))


def _redact_json_evidence(text: str) -> str:
    """Replace embedded exp3 verifier JSON objects without regex-parsing JSON."""

    decoder = json.JSONDecoder()
    pieces: list[str] = []
    cursor = 0
    while True:
        start = text.find("{", cursor)
        if start < 0:
            pieces.append(text[cursor:])
            break
        pieces.append(text[cursor:start])
        try:
            value, length = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            pieces.append("{")
            cursor = start + 1
            continue
        if _looks_like_evidence_payload(value):
            pieces.append(EVIDENCE_REDACTION)
            cursor = start + length
        else:
            pieces.append(text[start : start + length])
            cursor = start + length
    return "".join(pieces)


def _structured_container_depths(text: str) -> list[int]:
    """Return conservative brace/bracket depth before each character."""

    depths = [0] * (len(text) + 1)
    depth = 0
    for index, character in enumerate(text):
        depths[index] = depth
        if character in "{[":
            depth += 1
        elif character in "}]" and depth:
            depth -= 1
    depths[len(text)] = depth
    return depths


def _redact_scalar_evidence(text: str) -> str:
    """Redact strict scalar assignments echoed outside verifier JSON objects.

    Structured containers are deliberately excluded. Valid verifier JSON was
    already removed by :func:`_redact_json_evidence`; leaving malformed or
    unsupported containers intact lets the final leak audit fail closed.
    """

    depths = _structured_container_depths(text)

    def replacement(match: re.Match[str]) -> str:
        if depths[match.start()]:
            return match.group(0)
        following = text[match.end() :].lstrip()
        if following.startswith(("}", "]")):
            return match.group(0)
        return EVIDENCE_REDACTION

    return _EVIDENCE_SCALAR_ASSIGNMENT.sub(replacement, text)


def _sanitize_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    text = _redact_json_evidence(value)
    text = _redact_scalar_evidence(text)
    text = _GROUND_TRUTH_LINE.sub("", text)
    return text.strip()


def _message_transcript(messages: object) -> str:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    rendered = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(f"messages[{index}] must be an object")
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"messages[{index}].role is unsupported: {role!r}")
        content = _sanitize_text(
            message.get("content"), field=f"messages[{index}].content"
        )
        rendered.append(f"[{str(role).upper()}]\n{content}")
    return "\n\n".join(rendered)


def _first_text(
    record: Mapping[str, Any], fields: Sequence[str]
) -> tuple[str, str] | None:
    present = [
        (field, record[field]) for field in fields if record.get(field) is not None
    ]
    if not present:
        return None
    field, value = present[0]
    return field, _sanitize_text(value, field=field)


def build_blind_transcript(record: Mapping[str, Any]) -> str:
    """Return only subject-visible content, with condition evidence redacted.

    Supported exp3 rollout shapes are either ``messages``/``transcript`` or a
    ``prompt`` plus an assistant ``response``/``completion``/``output``. Separate
    ``cot``/``reasoning`` text is included when present. All other fields are
    ignored by construction.
    """

    if not isinstance(record, Mapping):
        raise TypeError("transcript record must be an object")

    if record.get("messages") is not None:
        transcript = _message_transcript(record["messages"])
    elif record.get("transcript") is not None:
        raw = record["transcript"]
        transcript = (
            _message_transcript(raw)
            if isinstance(raw, list)
            else _sanitize_text(raw, field="transcript")
        )
    else:
        prompt = _first_text(record, ("prompt", "input"))
        if prompt is None:
            raise ValueError("record needs messages, transcript, or prompt")

        reasoning = _first_text(record, ("chain_of_thought", "cot", "reasoning"))
        response = _first_text(
            record, ("response", "completion", "assistant_response", "output")
        )
        if reasoning is None and response is None:
            raise ValueError("record has no assistant response or reasoning to monitor")

        parts = [f"[SUBJECT PROMPT]\n{prompt[1]}"]
        if reasoning is not None:
            parts.append(f"[SUBJECT REASONING]\n{reasoning[1]}")
        if response is not None:
            parts.append(f"[SUBJECT RESPONSE]\n{response[1]}")
        transcript = "\n\n".join(parts)

    leaks = leak_check_text(transcript)
    if leaks:
        raise BlindnessError("sanitized transcript still contains: " + "; ".join(leaks))
    return transcript


def leak_check_text(text: str) -> list[str]:
    """Return descriptions of privileged structures still present in text."""

    leaks = []
    if _GROUND_TRUTH_SHAPE.search(text):
        leaks.append("ground-truth label syntax")
    if _EVIDENCE_SHAPE.search(text):
        leaks.append("raw verifier metadata")
    return leaks


def _sensitive_strings(value: object) -> set[str]:
    """Collect high-specificity canaries from annotations for defense in depth."""

    found: set[str] = set()
    if isinstance(value, str):
        # Short values such as REAL, SHAM, a/b/c, and 0/1 also occur in the fixed
        # scoring task. Structured-key checks above cover their accidental leaks.
        if len(value) >= 12:
            found.add(value)
    elif isinstance(value, Mapping):
        for child in value.values():
            found.update(_sensitive_strings(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            found.update(_sensitive_strings(child))
    return found


def leak_check_messages(
    messages: Sequence[Mapping[str, str]], record: Mapping[str, Any] | None = None
) -> list[str]:
    """Audit the exact message array immediately before a judge invocation."""

    combined = "\n".join(str(message.get("content", "")) for message in messages)
    leaks = leak_check_text(combined)
    if record is not None:
        for key in SENSITIVE_METADATA_KEYS.intersection(record):
            for marker in _sensitive_strings(record[key]):
                if marker in combined:
                    leaks.append(f"value copied from privileged field {key!r}")
    return sorted(set(leaks))


def assert_blind_messages(
    messages: Sequence[Mapping[str, str]], record: Mapping[str, Any] | None = None
) -> None:
    leaks = leak_check_messages(messages, record)
    if leaks:
        raise BlindnessError(
            "judge request failed blindness audit: " + "; ".join(leaks)
        )
