from __future__ import annotations

import math
from collections import UserDict

import numpy as np
import pytest

from experiments.exp4_analysis.contract import (
    locate_receipt_token,
    manipulation_prompt,
    parse_probability,
    receipt_paragraph_bounds,
    sequence_logprob_from_logits,
)


class CharacterTokenizer:
    """Fast-tokenizer-shaped fixture with one token per rendered character."""

    @staticmethod
    def _render(messages) -> str:
        return f"<user>{messages[0]['content']}</user><assistant>"

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize=False):
        assert add_generation_prompt
        rendered = self._render(messages)
        return [ord(char) for char in rendered] if tokenize else rendered

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        assert not add_special_tokens
        result = {"input_ids": [ord(char) for char in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return result


class BatchEncodingTokenizer(CharacterTokenizer):
    """Return chat IDs in the mapping shape used by real fast tokenizers."""

    def __init__(self, *, corrupt_chat_ids: bool = False) -> None:
        self.corrupt_chat_ids = corrupt_chat_ids

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize=False):
        result = super().apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
        )
        if not tokenize:
            return result
        input_ids = np.asarray([result])
        if self.corrupt_chat_ids:
            input_ids[0, 0] += 1
        return UserDict(input_ids=input_ids)


def canonical_prompt(condition: str) -> str:
    parts = ["Role paragraph.", "Receipt paragraph ends here."]
    if condition in {"b", "c"}:
        parts.append('Verifier output:\n{"settled": true}')
    if condition == "c":
        parts.append("Binding paragraph comes after verification.")
    if condition == "framing":
        parts.append("Assume the payment is real.")
    parts.append("Decide now: SPEND or HOLD?")
    return "\n\n".join(parts)


@pytest.mark.parametrize("condition", ["a", "b", "c", "framing"])
def test_receipt_token_is_exact_rendered_character_offset(condition: str) -> None:
    prompt = canonical_prompt(condition)
    tokenizer = CharacterTokenizer()
    location = locate_receipt_token(tokenizer, prompt, condition)
    start, end = receipt_paragraph_bounds(prompt, condition)

    assert (location.paragraph_start, location.paragraph_end) == (start, end)
    assert location.rendered_prompt[location.rendered_char_index] == prompt[end - 1]
    assert location.token_index == location.rendered_char_index
    assert chr(location.input_ids[location.token_index]) == prompt[end - 1]
    if condition == "c":
        assert prompt[end - 1] == "}"
        assert location.rendered_char_index < location.rendered_prompt.index("Binding")


def test_receipt_token_accepts_batch_encoding_chat_template_ids() -> None:
    prompt = canonical_prompt("b")

    location = locate_receipt_token(BatchEncodingTokenizer(), prompt, "b")

    assert chr(location.input_ids[location.token_index]) == "}"


def test_receipt_token_rejects_mismatched_batch_encoding_ids() -> None:
    prompt = canonical_prompt("b")

    with pytest.raises(
        ValueError,
        match="raw rendered-chat token IDs differ from chat-template token IDs",
    ):
        locate_receipt_token(BatchEncodingTokenizer(corrupt_chat_ids=True), prompt, "b")


def test_multi_token_teacher_forced_sequence_logprob_math() -> None:
    logits = np.zeros((5, 4), dtype=np.float64)
    # prompt_length=2: candidate token 1 is scored at position 1, token 3 at position 2.
    logits[1] = [0.0, 2.0, -1.0, 0.5]
    logits[2] = [1.0, -2.0, 0.0, 3.0]
    actual = sequence_logprob_from_logits(logits, [1, 3], prompt_length=2)
    expected = (2.0 - math.log(sum(math.exp(v) for v in logits[1]))) + (
        3.0 - math.log(sum(math.exp(v) for v in logits[2]))
    )
    assert actual == pytest.approx(expected)
    assert actual != pytest.approx(
        sequence_logprob_from_logits(logits, [1], prompt_length=2)
    )


@pytest.mark.parametrize(
    ("raw", "value", "error"),
    [
        ("73", 73.0, None),
        (" 42.5% ", 42.5, None),
        ("101", None, True),
        ("about 70", None, True),
    ],
)
def test_probability_parser_preserves_explicit_failures(raw, value, error) -> None:
    parsed, parse_error = parse_probability(raw)
    assert parsed == value
    assert bool(parse_error) is bool(error)


def test_manipulation_replaces_generated_decision_request() -> None:
    prompt = canonical_prompt("b")
    direct = manipulation_prompt(prompt)
    assert "Decide now" not in direct
    assert "probability that the payment actually arrived" in direct
    assert direct.startswith(prompt.rsplit("\n\n", 1)[0])
