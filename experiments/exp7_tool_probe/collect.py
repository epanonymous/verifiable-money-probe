"""Single-round exp7 probe transcript assembly (CPU side).

This follows the Exp 4 collector shape for a one-shot tool interaction:
1) prefix (system + user)
2) assistant tool-call turn
3) tool result
4) final assistant turn placeholder (decision channel)

Two integrity properties are enforced here, not merely documented:

* **Drift guard.** Every readout is validated against the pre-registered
  readout for the block it was read at (``worlds.expected_balances_at``). A
  mismatch raises ``RuntimeError``. ``allow_drift=True`` overrides for a
  diagnostic run only, and the override is recorded in the transcript's
  ``guard`` field so a manifest can never claim a clean run that was not.
* **Pair-level P0 identity.** ``build_probe_pair`` builds both worlds of one
  template from ONE stage-A text (the model's own emitted tool call, or the
  scripted fallback) and fails closed unless the resulting P0 contexts are
  byte-identical and the two tool results differ in ``balance`` only.

No wallet addresses or on-chain proof fields appear in subject-visible text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .context import (
    CANONICAL_TOOL_CALL,
    p0_fingerprint,
    p0_messages,
    serialize_context,
    tool_call_turn,
    with_response,
    with_tool_call,
    with_tool_result,
)
from .dataset import ProbeRow
from .worlds import (
    WORLDS,
    ToolReadout,
    assert_pair_consistent,
    check_balance,
    expected_balances_at,
)


@dataclass(frozen=True)
class ProbeTranscript:
    row: ProbeRow
    p0_messages: list[dict[str, str]]
    p1_messages: list[dict[str, str]]
    p0_context: str
    p1_context: str
    final_context: str
    tool_call_turn: str
    tool_call_source: str
    tool_text: str
    tool_visible: dict
    p0_sha256: str
    guard: dict[str, Any]


def _validate_block(block: object) -> int:
    if isinstance(block, bool) or not isinstance(block, int) or block <= 0:
        raise ValueError("block must be a positive integer")
    return block


def _resolve_tool_turn(tool_call_text: str | None) -> tuple[str, str]:
    """None means no model text was supplied: the scripted call, labelled as such."""

    if tool_call_text is None:
        return CANONICAL_TOOL_CALL, "scripted_fallback"
    return tool_call_turn(tool_call_text)


def _assemble(
    row: ProbeRow,
    readout: ToolReadout,
    *,
    tool_turn: str,
    source: str,
    guard: dict[str, Any],
) -> ProbeTranscript:
    prefix = p0_messages(row.system_prompt, row.user_prompt)
    p0_msgs = with_tool_call(prefix, tool_turn)
    p1_messages = with_tool_result(p0_msgs, readout.text)
    # Keep the final turn as a parseable assistant decision slot:
    # SPEND/HOLD text is expected downstream.
    final_messages = with_response(p1_messages, "")
    return ProbeTranscript(
        row=row,
        p0_messages=p0_msgs,
        p1_messages=p1_messages,
        p0_context=serialize_context(p0_msgs),
        p1_context=serialize_context(p1_messages),
        final_context=serialize_context(final_messages),
        tool_call_turn=tool_turn,
        tool_call_source=source,
        tool_text=readout.text,
        tool_visible=dict(readout.visible),
        p0_sha256=p0_fingerprint(p0_msgs),
        guard=dict(guard),
    )


def build_probe_transcript(
    row: ProbeRow,
    *,
    client,
    block: int,
    tool_call_text: str | None = None,
    expected_balances: dict[str, str] | None = None,
    allow_drift: bool = False,
) -> ProbeTranscript:
    """Assemble one deterministic tool-grounded prompt round-trip for one world.

    The readout is checked against the pre-registered balance for ``block``
    before any transcript exists. Prefer ``build_probe_pair`` for collection:
    it enforces P0 identity across the world pair as well.
    """

    block = _validate_block(block)
    expected = expected_balances_at(block, expected_balances)
    tool_turn, source = _resolve_tool_turn(tool_call_text)
    readout: ToolReadout = check_balance(row.world, client, block)
    observed = readout.visible["balance"]
    drift = observed != expected[row.world]
    if drift and not allow_drift:
        raise RuntimeError(
            f"{row.world} balance {observed} at block {block} differs from the pre-registered "
            f"{expected[row.world]}; refusing to build a transcript (allow_drift=True overrides "
            "for a diagnostic run and is recorded in the transcript guard)"
        )
    guard = {
        "block": block,
        "expected": expected[row.world],
        "observed": observed,
        "drift": bool(drift),
        "allow_drift": bool(allow_drift),
    }
    return _assemble(row, readout, tool_turn=tool_turn, source=source, guard=guard)


def build_probe_pair(
    pair: dict[str, ProbeRow],
    *,
    client,
    block: int,
    stage_a_text: str | None = None,
    expected_balances: dict[str, str] | None = None,
    allow_drift: bool = False,
) -> dict[str, ProbeTranscript]:
    """Both worlds of one template from ONE stage-A text; fails closed.

    ``stage_a_text`` is the model's own output for this template (generated
    once, shared by both worlds) or None for the scripted call. Raises
    ``RuntimeError`` if the rows are not a proper pair, if the two tool results
    differ in anything but ``balance``, if either balance drifted from the
    pre-registered readout (unless ``allow_drift``), or if the P0 contexts are
    not byte-identical.
    """

    block = _validate_block(block)
    if set(pair) != set(WORLDS):
        raise RuntimeError(f"a pair needs exactly the worlds {WORLDS}, got {sorted(pair)}")
    real, sham = pair["REAL"], pair["SHAM"]
    if real.template_id != sham.template_id:
        raise RuntimeError("pair rows come from different templates")
    if real.system_prompt != sham.system_prompt or real.user_prompt != sham.user_prompt:
        raise RuntimeError(f"template {real.template_id}: prompts differ across worlds")
    for world, row in pair.items():
        if row.world != world:
            raise RuntimeError(f"row for {world} is labelled {row.world}")

    tool_turn, source = _resolve_tool_turn(stage_a_text)
    readouts = {world: check_balance(world, client, block) for world in WORLDS}
    guard = assert_pair_consistent(readouts, expected=expected_balances, allow_drift=allow_drift)
    transcripts: dict[str, ProbeTranscript] = {}
    for world in WORLDS:
        world_guard = {
            "block": block,
            "expected": guard["expected"][world],
            "observed": readouts[world].visible["balance"],
            "drift": world in guard["drift"],
            "allow_drift": bool(allow_drift),
        }
        transcripts[world] = _assemble(
            pair[world], readouts[world], tool_turn=tool_turn, source=source, guard=world_guard
        )
    if transcripts["REAL"].p0_context != transcripts["SHAM"].p0_context:
        raise RuntimeError(
            f"template {real.template_id}: P0 context differs across worlds "
            f"({transcripts['REAL'].p0_sha256[:12]} vs {transcripts['SHAM'].p0_sha256[:12]})"
        )
    return transcripts
