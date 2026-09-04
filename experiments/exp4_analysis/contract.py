"""Pure, testable contracts for experiment-4 derived prompt-group shards."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.exp4_collection.contract import MODEL, PromptGroup

SPEND_CANDIDATE = "SPEND"
HOLD_CANDIDATE = "HOLD"
MANIPULATION_QUESTION = (
    "What is the probability that the payment actually arrived? "
    "Reply with only one number from 0 to 100."
)


@dataclass(frozen=True)
class ReceiptLocation:
    """Receipt-final's exact character and token coordinates."""

    paragraph_start: int
    paragraph_end: int
    rendered_char_index: int
    token_index: int
    rendered_prompt: str
    input_ids: tuple[int, ...]


def receipt_paragraph_bounds(prompt: str, condition: str) -> tuple[int, int]:
    """Return the receipt/verification paragraph's ``[start, end)`` prompt span.

    Prompts are the locked exp3 composition ``role, receipt, [verification],
    [binding], ask`` joined by exactly two newlines. Receipt-final is the final
    non-whitespace character of the claimed receipt for ``a`` and of the verifier
    paragraph for ``b``/``c``. Thus the separate causal-binding paragraph in ``c``
    is deliberately after receipt-final. Framing uses its claimed receipt, before
    the framing assertion. The last paragraph is always the decision request.
    """

    condition = str(condition).strip().lower()
    parts = prompt.split("\n\n")
    expected_lengths = {"a": 3, "b": 4, "c": 5, "framing": 4}
    if condition not in expected_lengths:
        raise ValueError(f"unknown condition {condition!r}")
    if len(parts) != expected_lengths[condition]:
        raise ValueError(
            f"condition {condition!r} prompt must have {expected_lengths[condition]} "
            f"double-newline paragraphs, got {len(parts)}"
        )
    if not {"SPEND", "HOLD"}.issubset(set(re.findall(r"\b[A-Z]+\b", parts[-1]))):
        raise ValueError(
            "final prompt paragraph is not the locked SPEND/HOLD decision ask"
        )
    evidence_index = {"a": 1, "b": 2, "c": 2, "framing": 1}[condition]
    paragraph = parts[evidence_index]
    if not paragraph or paragraph != paragraph.rstrip():
        raise ValueError(
            "receipt/verification paragraph must be non-empty with no trailing space"
        )
    start = sum(len(part) + 2 for part in parts[:evidence_index])
    end = start + len(paragraph)
    if prompt[end : end + 2] != "\n\n":
        raise ValueError(
            "receipt/verification paragraph is not followed by a blank line"
        )
    return start, end


def locate_receipt_token(
    tokenizer: Any, prompt: str, condition: str
) -> ReceiptLocation:
    """Map the locked receipt-final character to the full rendered-chat token.

    Rule: render the complete one-message chat with the assistant generation prefix;
    find the unique byte-for-byte user prompt; take ``paragraph_end - 1`` (the last
    non-whitespace receipt/verification character); then select the unique token
    whose fast-tokenizer offset interval ``[start, end)`` contains that character.
    The raw-render encoding is required to equal ``apply_chat_template`` token IDs.
    """

    paragraph_start, paragraph_end = receipt_paragraph_bounds(prompt, condition)
    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    if not isinstance(rendered, str) or rendered.count(prompt) != 1:
        raise ValueError(
            "rendered chat must contain the exact user prompt exactly once"
        )
    prompt_start = rendered.index(prompt)
    rendered_char_index = prompt_start + paragraph_end - 1
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = _flat_ints(encoded["input_ids"], "input_ids")
    offsets = [tuple(int(v) for v in pair) for pair in encoded["offset_mapping"]]
    chat_encoding = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )
    if isinstance(chat_encoding, Mapping):
        chat_encoding = chat_encoding["input_ids"]
    chat_ids = _flat_ints(chat_encoding, "chat input_ids")
    if input_ids != chat_ids:
        raise ValueError(
            "raw rendered-chat token IDs differ from chat-template token IDs"
        )
    matches = [
        index
        for index, (start, end) in enumerate(offsets)
        if start <= rendered_char_index < end
    ]
    if len(matches) != 1:
        raise ValueError(
            "receipt-final character must belong to exactly one tokenizer offset; "
            f"got matches={matches}"
        )
    return ReceiptLocation(
        paragraph_start=paragraph_start,
        paragraph_end=paragraph_end,
        rendered_char_index=rendered_char_index,
        token_index=matches[0],
        rendered_prompt=rendered,
        input_ids=tuple(input_ids),
    )


def _flat_ints(value: Any, field: str) -> list[int]:
    array = np.asarray(value)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1:
        raise ValueError(f"{field} must be one dimensional, got {array.shape}")
    return [int(item) for item in array]


def candidate_token_ids(tokenizer: Any, candidate: str) -> tuple[int, ...]:
    encoded = tokenizer(candidate, add_special_tokens=False)
    ids = tuple(_flat_ints(encoded["input_ids"], f"{candidate} token ids"))
    if not ids:
        raise ValueError(f"candidate {candidate!r} tokenized to no tokens")
    return ids


def sequence_logprob_from_logits(
    logits: Any, candidate_ids: tuple[int, ...] | list[int], prompt_length: int
) -> float:
    """Teacher-forced log P(candidate sequence | prompt) from causal-LM logits."""

    values = np.asarray(logits, dtype=np.float64)
    candidate = np.asarray(candidate_ids, dtype=np.int64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("logits must be a finite [sequence, vocabulary] array")
    if candidate.ndim != 1 or len(candidate) == 0:
        raise ValueError("candidate_ids must be a non-empty one-dimensional sequence")
    if prompt_length < 1 or prompt_length + len(candidate) - 1 > len(values):
        raise ValueError("logits do not cover every teacher-forced candidate position")
    if np.any(candidate < 0) or np.any(candidate >= values.shape[1]):
        raise ValueError("candidate token ID is outside the logits vocabulary")
    selected = values[prompt_length - 1 : prompt_length + len(candidate) - 1]
    maxima = np.max(selected, axis=1)
    log_denominator = maxima + np.log(np.exp(selected - maxima[:, None]).sum(axis=1))
    return float(
        np.sum(selected[np.arange(len(candidate)), candidate] - log_denominator)
    )


def manipulation_prompt(prompt: str) -> str:
    """Replace the spend decision with the locked direct probability question."""

    prefix, separator, ask = prompt.rpartition("\n\n")
    if not separator or not {"SPEND", "HOLD"}.issubset(
        set(re.findall(r"\b[A-Z]+\b", ask))
    ):
        raise ValueError(
            "prompt does not end in the locked SPEND/HOLD decision paragraph"
        )
    return f"{prefix}\n\n{MANIPULATION_QUESTION}"


_PROBABILITY_RE = re.compile(
    r"^\s*(?P<value>(?:100(?:\.0+)?)|(?:[0-9]?[0-9](?:\.\d+)?))\s*%?\s*$"
)


def parse_probability(raw: str) -> tuple[float | None, str | None]:
    """Strictly parse one bare 0..100 number, preserving explicit failures."""

    match = _PROBABILITY_RE.fullmatch(raw)
    if match is None:
        return None, "response_not_single_0_to_100_number"
    value = float(match.group("value"))
    if not 0 <= value <= 100:  # defensive; regex already enforces the range
        return None, "probability_out_of_range"
    return value, None


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def manipulation_required(group: PromptGroup, which: str) -> bool:
    return which == "main" and any(
        str(row.get("split", "")).lower() == "heldout" for row in group.rows
    )


def validate_derived_shard(
    path: str | Path,
    group: PromptGroup,
    which: str,
    model: str = MODEL,
) -> tuple[int, int]:
    """Validate a retained derived shard; return its ``(layers, d_model)`` shape."""

    required = {
        "receipt_final",
        "prompt",
        "prompt_sha256",
        "source_row_ids",
        "group_key",
        "condition",
        "model",
        "receipt_paragraph_start",
        "receipt_paragraph_end",
        "receipt_rendered_char_index",
        "receipt_token_index",
        "rendered_prompt_sha256",
        "spend_logprob",
        "hold_logprob",
        "spend_hold_log_odds",
        "spend_token_ids",
        "hold_token_ids",
        "manipulation_required",
        "manipulation_prompt",
        "manipulation_raw",
        "manipulation_parse_ok",
        "manipulation_probability",
        "manipulation_parse_error",
    }
    try:
        with np.load(path, allow_pickle=False) as shard:
            missing = required.difference(shard.files)
            if missing:
                raise ValueError(f"missing keys {sorted(missing)}")
            values = {name: np.asarray(shard[name]) for name in required}
    except (OSError, EOFError) as exc:
        raise ValueError(f"cannot read npz: {exc}") from exc

    def scalar(name: str) -> Any:
        value = values[name]
        if value.ndim == 0:
            return value.item()
        if value.size == 1:
            return value.reshape(-1)[0].item()
        raise ValueError(f"{name} must be scalar, got {value.shape}")

    activation = values["receipt_final"]
    if activation.ndim != 2 or not np.issubdtype(activation.dtype, np.number):
        raise ValueError("receipt_final must be numeric [layers, d_model]")
    if not np.isfinite(activation).all():
        raise ValueError("receipt_final contains NaN or infinite values")
    if str(scalar("prompt")) != group.prompt:
        raise ValueError("prompt mismatch")
    if str(scalar("prompt_sha256")) != prompt_sha256(group.prompt):
        raise ValueError("prompt_sha256 mismatch")
    if str(scalar("group_key")) != group.key:
        raise ValueError("group_key mismatch")
    if str(scalar("model")) != model:
        raise ValueError(f"model mismatch: expected {model!r}, got {scalar('model')!r}")
    expected_rows = tuple(str(row["id"]) for row in group.rows)
    actual_rows = tuple(str(value) for value in values["source_row_ids"])
    if actual_rows != expected_rows:
        raise ValueError(
            f"source_row_ids mismatch: expected {expected_rows}, got {actual_rows}"
        )
    expected_condition = str(group.rows[0]["cond"])
    if str(scalar("condition")) != expected_condition:
        raise ValueError("condition mismatch")
    start, end = receipt_paragraph_bounds(group.prompt, expected_condition)
    if (
        int(scalar("receipt_paragraph_start")) != start
        or int(scalar("receipt_paragraph_end")) != end
    ):
        raise ValueError("receipt paragraph character bounds mismatch")
    if int(scalar("receipt_rendered_char_index")) < end - 1:
        raise ValueError("rendered receipt character index is inconsistent")
    if int(scalar("receipt_token_index")) < 0:
        raise ValueError("receipt_token_index must be non-negative")
    for name in ("spend_token_ids", "hold_token_ids"):
        ids = values[name]
        if ids.ndim != 1 or len(ids) == 0 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError(f"{name} must be a non-empty integer vector")
    spend = float(scalar("spend_logprob"))
    hold = float(scalar("hold_logprob"))
    difference = float(scalar("spend_hold_log_odds"))
    if not np.isfinite([spend, hold, difference]).all():
        raise ValueError("candidate log probabilities must be finite")
    if not np.isclose(difference, spend - hold, rtol=0, atol=1e-8):
        raise ValueError("spend_hold_log_odds is not spend_logprob - hold_logprob")

    expected_required = manipulation_required(group, which)
    if bool(int(scalar("manipulation_required"))) != expected_required:
        raise ValueError("manipulation_required mismatch")
    parse_ok = bool(int(scalar("manipulation_parse_ok")))
    probability = float(scalar("manipulation_probability"))
    parse_error = str(scalar("manipulation_parse_error"))
    raw = str(scalar("manipulation_raw"))
    if expected_required:
        if str(scalar("manipulation_prompt")) != manipulation_prompt(group.prompt):
            raise ValueError("manipulation prompt mismatch")
        if parse_ok != (parse_error == ""):
            raise ValueError("manipulation parse status/error mismatch")
        if parse_ok and (not np.isfinite(probability) or not 0 <= probability <= 100):
            raise ValueError("parsed manipulation probability must be in [0, 100]")
        if not parse_ok and (not np.isnan(probability) or not parse_error):
            raise ValueError(
                "parse failure must use NaN probability and explicit error"
            )
        reparsed, reparsed_error = parse_probability(raw)
        if parse_ok:
            if reparsed is None or not np.isclose(
                reparsed, probability, rtol=0, atol=1e-12
            ):
                raise ValueError(
                    "stored manipulation probability does not match raw response"
                )
        if not parse_ok and reparsed_error != parse_error:
            raise ValueError(
                "stored manipulation parse error does not match raw response"
            )
    elif (
        parse_ok
        or not np.isnan(probability)
        or any(
            str(scalar(name))
            for name in (
                "manipulation_prompt",
                "manipulation_raw",
                "manipulation_parse_error",
            )
        )
    ):
        raise ValueError("non-manipulation shard must use empty/unavailable fields")
    rendered_hash = str(scalar("rendered_prompt_sha256"))
    if re.fullmatch(r"[0-9a-f]{64}", rendered_hash) is None:
        raise ValueError(
            "rendered_prompt_sha256 must be a lowercase SHA-256 hex digest"
        )
    return tuple(int(value) for value in activation.shape)
