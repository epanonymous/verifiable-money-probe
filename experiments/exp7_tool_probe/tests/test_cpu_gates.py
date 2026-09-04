from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from experiments.exp7_tool_probe import collect, context, dataset, worlds
from experiments.exp7_tool_probe import rpc as rpc_module

_EVM_TOKEN = re.compile(r"0x[0-9a-fA-F]{40}")
FORBIDDEN_TOOL_FIELDS = {
    "address",
    "settled",
    "usdc_balance",
    "tx_confirmed",
    "wallet",
    "last_inbound_tx",
    "tx_block",
    "queried_block",
}
FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "recorded_balances.json").read_text())
PINNED_BLOCK = int(FIXTURE["real"]["block"])


class _FixtureClient:
    """Minimal stand-in for JsonRpcClient used by worlds.check_balance."""

    class _Exchange:
        def to_dict(self) -> dict[str, object]:
            return {}

    def __init__(self, values: dict[str, int]):
        self._values = values

    def erc20_balance_of(self, token: str, holder: str, block: int) -> tuple[int, object]:
        if token != worlds.USDC:
            raise ValueError(f"unexpected token {token}")
        if holder not in self._values:
            raise ValueError(f"unknown holder {holder}")
        return self._values[holder], self._Exchange()


def _client(real_raw: int | None = None, sham_raw: int | None = None) -> _FixtureClient:
    real = int(FIXTURE["real"]["raw_balance"]) if real_raw is None else real_raw
    sham = int(FIXTURE["sham"]["raw_balance"]) if sham_raw is None else sham_raw
    return _FixtureClient({worlds.WORLD_ADDRESSES["REAL"]: real, worlds.WORLD_ADDRESSES["SHAM"]: sham})


def _fixture_rows() -> list[dataset.ProbeRow]:
    return dataset.build_rows()


def _pairs() -> dict[int, dict[str, dataset.ProbeRow]]:
    return {tid: group for (tid, _split), group in dataset.pairs(_fixture_rows()).items()}


def _p0_text(row: dataset.ProbeRow, turn: str = context.CANONICAL_TOOL_CALL) -> str:
    return context.serialize_context(
        context.with_tool_call(context.p0_messages(row.system_prompt, row.user_prompt), turn)
    )


def test_fixture_mirrors_the_pinned_readout() -> None:
    assert PINNED_BLOCK == worlds.PINNED_BLOCK
    assert rpc_module.format_units(int(FIXTURE["real"]["raw_balance"])) == worlds.PINNED_BALANCES["REAL"]
    assert rpc_module.format_units(int(FIXTURE["sham"]["raw_balance"])) == worlds.PINNED_BALANCES["SHAM"]


def test_p0_contexts_are_byte_identical_by_real_sham_pair() -> None:
    for (template_id, split), group in dataset.pairs(_fixture_rows()).items():
        assert set(group) == set(worlds.WORLDS)
        texts = {world: _p0_text(row) for world, row in group.items()}
        assert texts["REAL"] == texts["SHAM"], ("P0 context leaked across worlds", template_id, split)


def test_subject_visible_text_has_no_address_leaks() -> None:
    for row in _fixture_rows():
        for text in (row.system_prompt, row.user_prompt, _p0_text(row)):
            assert not _EVM_TOKEN.search(text), text
            for body in worlds.WORLD_ADDRESSES.values():
                assert body not in text
                assert body[2:].lower() not in text.lower()


def test_p0_lexicon_scan_finds_no_world_tokens() -> None:
    """A fixed leak lexicon over the P0 text. This is a scan, not the blind gate
    (that lives in test_blind_p0.py and blind_p0.py)."""

    from experiments.exp7_tool_probe.blind_p0 import lexicon_score

    for row in _fixture_rows():
        score, hits = lexicon_score(_p0_text(row))
        assert score == 0.5 and hits == [], (row.row_id, hits)
    # and the scan is not blind to a leak
    score, hits = lexicon_score(_p0_text(_fixture_rows()[0]).replace("wallet", 'wallet ("settled": true, balance 1.900000)'))
    assert score > 0.5 and hits


def test_tool_result_json_is_subject_safe() -> None:
    reader = _client()
    for row in _fixture_rows():
        readout = worlds.check_balance(row.world, reader, PINNED_BLOCK)
        data = json.loads(readout.text)
        assert data.keys() == {"asset", "balance", "block", "chain"}
        assert data["asset"] == FIXTURE["asset"]
        assert data["chain"] == FIXTURE["chain"]
        assert data["balance"] == worlds.PINNED_BALANCES[row.world]
        assert not any(key in data for key in FORBIDDEN_TOOL_FIELDS)
        for body in worlds.WORLD_ADDRESSES.values():
            assert body not in readout.text


def test_pair_path_with_model_emitted_turn_keeps_p0_identical() -> None:
    reader = _client()
    model_text = "\n" + context.CANONICAL_TOOL_CALL + "  \n"  # the model's own bytes, whitespace around
    for tid, pair in _pairs().items():
        out = collect.build_probe_pair(pair, client=reader, block=PINNED_BLOCK, stage_a_text=model_text)
        assert set(out) == set(worlds.WORLDS)
        assert all(t.tool_call_source == "model" for t in out.values())
        assert out["REAL"].p0_context == out["SHAM"].p0_context
        assert out["REAL"].p0_sha256 == out["SHAM"].p0_sha256
        assert out["REAL"].p1_context != out["SHAM"].p1_context
        for world, t in out.items():
            assert t.tool_visible["balance"] == worlds.PINNED_BALANCES[world]
            assert t.guard == {"block": PINNED_BLOCK, "expected": worlds.PINNED_BALANCES[world], "observed": worlds.PINNED_BALANCES[world], "drift": False, "allow_drift": False}
            assert t.p0_messages[-1]["role"] == "assistant"
            assert t.p1_messages[-1]["role"] == "tool"
            assert t.final_context


def test_no_stage_a_text_is_labelled_scripted_fallback() -> None:
    pair = _pairs()[0]
    out = collect.build_probe_pair(pair, client=_client(), block=PINNED_BLOCK)
    assert {t.tool_call_source for t in out.values()} == {"scripted_fallback"}
    assert all(t.tool_call_turn == context.CANONICAL_TOOL_CALL for t in out.values())
    single = collect.build_probe_transcript(pair["REAL"], client=_client(), block=PINNED_BLOCK)
    assert single.tool_call_source == "scripted_fallback"


@pytest.mark.parametrize(
    "text, reason_fragment",
    [
        ("I think this is REAL. " + context.CANONICAL_TOOL_CALL, "outside"),
        (context.CANONICAL_TOOL_CALL + "\nLet me check.", "outside"),
        (context.CANONICAL_TOOL_CALL + "\n" + context.CANONICAL_TOOL_CALL, "more than one"),
        ('<tool_call>\n{"name": "check_balance", "arguments": {"world": "REAL"}}\n</tool_call>', "non-empty arguments"),
        ('<tool_call>\n{"name": "send_funds", "arguments": {}}\n</tool_call>', "wrong tool"),
        ("no call at all", "no well-formed"),
        (None, "not a string"),
    ],
)
def test_strict_tool_call_rejects_prose_extra_calls_and_arguments(text, reason_fragment) -> None:
    reason = context.strict_tool_call(text)
    assert reason is not None and reason_fragment in reason
    turn, source = context.tool_call_turn(text)
    assert (turn, source) == (context.CANONICAL_TOOL_CALL, "scripted_fallback")


def test_strict_tool_call_accepts_the_bare_call_with_whitespace() -> None:
    assert context.strict_tool_call("\n " + context.CANONICAL_TOOL_CALL + "\n") is None
    assert context.strict_tool_call('<tool_call>{"name": "check_balance"}</tool_call>') is None  # arguments omitted == {}
    turn, source = context.tool_call_turn(context.CANONICAL_TOOL_CALL + "\n\n")
    assert (turn, source) == (context.CANONICAL_TOOL_CALL, "model")


def test_divergent_prose_around_the_call_cannot_split_p0() -> None:
    """Two 'valid' model outputs that differ in prose must not yield different P0s."""

    pair = _pairs()[1]
    reader = _client()
    real = collect.build_probe_transcript(pair["REAL"], client=reader, block=PINNED_BLOCK, tool_call_text="I think this is REAL. " + context.CANONICAL_TOOL_CALL)
    sham = collect.build_probe_transcript(pair["SHAM"], client=reader, block=PINNED_BLOCK, tool_call_text="I think this is SHAM. " + context.CANONICAL_TOOL_CALL)
    assert real.tool_call_source == sham.tool_call_source == "scripted_fallback"
    assert real.p0_context == sham.p0_context


def test_pair_path_rejects_mismatched_rows() -> None:
    pairs = _pairs()
    bad = {"REAL": pairs[0]["REAL"], "SHAM": pairs[1]["SHAM"]}
    with pytest.raises(RuntimeError, match="different templates"):
        collect.build_probe_pair(bad, client=_client(), block=PINNED_BLOCK)
    with pytest.raises(RuntimeError, match="exactly the worlds"):
        collect.build_probe_pair({"REAL": pairs[0]["REAL"]}, client=_client(), block=PINNED_BLOCK)


@pytest.mark.parametrize("real_raw, sham_raw", [(123, None), (None, 5), (5_000_000, None), (0, None)])
def test_drifted_balance_aborts_every_collection_path(real_raw, sham_raw) -> None:
    reader = _client(real_raw, sham_raw)
    pair = _pairs()[2]
    world = "REAL" if real_raw is not None else "SHAM"
    with pytest.raises(RuntimeError, match="drift|differs from the pre-registered"):
        collect.build_probe_pair(pair, client=reader, block=PINNED_BLOCK, stage_a_text=context.CANONICAL_TOOL_CALL)
    with pytest.raises(RuntimeError, match="differs from the pre-registered"):
        collect.build_probe_transcript(pair[world], client=reader, block=PINNED_BLOCK)
    with pytest.raises(RuntimeError, match="drift"):
        worlds.read_both_worlds_guarded(_ClientWithHead(reader, PINNED_BLOCK), PINNED_BLOCK)


def test_allow_drift_is_recorded_not_silent() -> None:
    reader = _client(123, None)
    pair = _pairs()[2]
    out = collect.build_probe_pair(pair, client=reader, block=PINNED_BLOCK, stage_a_text=context.CANONICAL_TOOL_CALL, allow_drift=True)
    assert out["REAL"].guard == {"block": PINNED_BLOCK, "expected": "1.900000", "observed": "0.000123", "drift": True, "allow_drift": True}
    assert out["SHAM"].guard["drift"] is False and out["SHAM"].guard["allow_drift"] is True
    _, guard = worlds.read_both_worlds_guarded(_ClientWithHead(reader, PINNED_BLOCK), PINNED_BLOCK, allow_drift=True)
    assert guard["allow_drift"] is True and guard["drift"] == {"REAL": {"expected": "1.900000", "observed": "0.000123"}}


def test_unpinned_block_needs_an_explicit_expectation() -> None:
    """After the post-run sweep a fresh block reads differently; the guard never
    assumes the wallet still holds the pinned amount."""

    pair = _pairs()[3]
    other_block = PINNED_BLOCK + 1000
    with pytest.raises(RuntimeError, match="no pre-registered readout"):
        collect.build_probe_pair(pair, client=_client(), block=other_block)
    with pytest.raises(RuntimeError, match="no pre-registered readout"):
        worlds.expected_balances_at(other_block)
    explicit = {"REAL": "0.000000", "SHAM": "0.000000"}  # a hypothetical post-sweep lock
    # a 0/0 pair passes the drift guard but fails the balance-only-difference structural check
    with pytest.raises(ValueError, match="balance field only"):
        collect.build_probe_pair(pair, client=_client(0, 0), block=other_block, expected_balances=explicit)
    swept = {"REAL": "0.500000", "SHAM": "0.000000"}
    out = collect.build_probe_pair(pair, client=_client(500_000, 0), block=other_block, expected_balances=swept)
    assert out["REAL"].guard["expected"] == "0.500000" and out["REAL"].guard["drift"] is False
    with pytest.raises(ValueError):
        worlds.expected_balances_at(other_block, {"REAL": "1.0"})


class _ClientWithHead:
    def __init__(self, inner: _FixtureClient, head: int):
        self._inner = inner
        self._head = head

    def block_number(self) -> int:
        return self._head

    def erc20_balance_of(self, token, holder, block):
        return self._inner.erc20_balance_of(token, holder, block)


def test_tool_result_must_be_one_json_object() -> None:
    p0 = context.with_tool_call(context.p0_messages("s", "u"), context.CANONICAL_TOOL_CALL)
    for bad in ("5", "[1, 2]", "null", '"text"'):
        with pytest.raises(ValueError, match="one JSON object"):
            context.with_tool_result(p0, bad)


def test_tool_call_parse_fails_without_check_balance() -> None:
    assert context.parse_tool_call('<tool_call>{"name": "not_the_tool", "arguments": {}}') is None


def test_tool_call_parse_matches_schema_and_payload() -> None:
    payload = context.parse_tool_call(context.CANONICAL_TOOL_CALL)
    assert payload is not None
    assert payload.name == worlds.TOOL_NAME
    assert payload.arguments == {}


def test_parse_decision_is_stable_for_ambiguous_text() -> None:
    assert context.parse_decision("No clear signal here") == "AMBIGUOUS"
    assert context.parse_decision("I will SPEND now") == "SPEND"
    assert context.parse_decision("best HOLD action") == "HOLD"


def test_live_readout_matches_pinned_if_enabled() -> None:
    """Opt-in: reads the pinned block live. A post-sweep 'latest' read is expected
    to differ and is NOT what this checks."""

    if os.getenv("EXP7_LIVE_RPC") != "1":
        pytest.skip("set EXP7_LIVE_RPC=1 to run live check")
    try:
        client = rpc_module.connect()
        readouts, guard = worlds.read_both_worlds_guarded(client, worlds.PINNED_BLOCK)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"live RPC unavailable: {exc}")
    assert guard["drift"] == {}
    for world, readout in readouts.items():
        assert readout.visible["balance"] == worlds.PINNED_BALANCES[world]
