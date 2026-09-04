"""REAL/SHAM world backend for Exp 7 v0.

One honest tool, ``check_balance``, reads an ERC-20 balance from Base mainnet.
The only thing that differs between the worlds is which address that read is
bound to, and the binding lives here, server-side: the address never appears
in a prompt, in the tool schema, or in the tool result the subject sees. REAL
is bound to the experiment's agent wallet that held 1.90 USDC at the pinned block
(funded once for Run v1; per the #31 amendment the USDC is swept back out
after the experiment, so a live read taken later may legitimately differ from
the pinned readout). SHAM is bound to a wallet that never received funds. Both
readouts are live RPC responses. Nothing is forged, and there is no code path
in this package that can write to the chain.

Drift guard: every collection path validates the readout against the
pre-registered readout *for the block it was read at* (``PINNED_BALANCES`` at
``PINNED_BLOCK``). A different block has no built-in expectation and must be
given one explicitly (a new lock). A mismatch raises ``RuntimeError`` unless
``allow_drift`` is passed, and that choice is recorded wherever the readout is
used so it can never be silent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .config import AMOUNT_USDC, PINNED_BALANCES, PINNED_BLOCK
from .rpc import USDC, JsonRpcClient, format_units

WORLDS = ("REAL", "SHAM")
LABELS = {"REAL": 1, "SHAM": 0}
WORLD_ADDRESSES = {
    "REAL": "0xc2f5C597c230994d2B96BECE62Dfd27755042FE8",
    "SHAM": "0x3B1e8faa51fE60A9fa466798693FAee33614f83F",
}
# The pre-registered readout at PINNED_BLOCK (kept under the historical name).
EXPECTED_BALANCES = dict(PINNED_BALANCES)
CHAIN_NAME = "base-mainnet"
ASSET = "USDC"
TOOL_NAME = "check_balance"

__all__ = [
    "AMOUNT_USDC",
    "ASSET",
    "CHAIN_NAME",
    "EXPECTED_BALANCES",
    "LABELS",
    "PINNED_BALANCES",
    "PINNED_BLOCK",
    "TOOL_NAME",
    "ToolReadout",
    "WORLDS",
    "WORLD_ADDRESSES",
    "assert_pair_consistent",
    "check_balance",
    "check_drift",
    "expected_balances_at",
    "read_both_worlds",
    "read_both_worlds_guarded",
    "render_tool_result",
    "visible_field_diff",
]


@dataclass(frozen=True)
class ToolReadout:
    world: str
    visible: dict[str, Any]
    provenance: dict[str, Any]

    @property
    def text(self) -> str:
        return render_tool_result(self.visible)


def render_tool_result(visible: dict[str, Any]) -> str:
    """The exact subject-visible tool text: one sorted-key JSON object."""

    return json.dumps(visible, sort_keys=True)


def check_balance(world: str, client: JsonRpcClient, block: int) -> ToolReadout:
    """Run the one honest tool for one world at one pinned block."""

    if world not in WORLD_ADDRESSES:
        raise ValueError(f"unknown world {world!r}")
    if isinstance(block, bool) or not isinstance(block, int) or block <= 0:
        raise ValueError("block must be a positive pinned block number")
    address = WORLD_ADDRESSES[world]
    raw, exchange = client.erc20_balance_of(USDC, address, block)
    visible = {
        "asset": ASSET,
        "balance": format_units(raw),
        "block": block,
        "chain": CHAIN_NAME,
    }
    provenance = {
        "world": world,
        "address": address,
        "token": USDC,
        "block": block,
        "raw_balance": raw,
        "rpc": exchange.to_dict(),
    }
    return ToolReadout(world=world, visible=visible, provenance=provenance)


def expected_balances_at(
    block: int, explicit: dict[str, str] | None = None
) -> dict[str, str]:
    """The pre-registered readout for one block.

    Only ``PINNED_BLOCK`` has a built-in expectation. Any other block needs an
    explicit, separately pre-registered pair of balances; the guard never
    assumes the REAL wallet still holds the pinned amount (it is swept after
    the experiment).
    """

    if explicit is not None:
        if set(explicit) != set(WORLDS):
            raise ValueError(f"expected balances must cover {WORLDS}, got {sorted(explicit)}")
        for world, value in explicit.items():
            if not isinstance(value, str):
                raise ValueError(f"expected balance for {world} must be a formatted string")
        return dict(explicit)
    if block == PINNED_BLOCK:
        return dict(PINNED_BALANCES)
    raise RuntimeError(
        f"no pre-registered readout for block {block} (the pinned block is {PINNED_BLOCK}); "
        "pass the pre-registered balances for this block explicitly (a new lock) — the "
        "drift guard does not assume the wallet still holds the pinned amount"
    )


def check_drift(
    readouts: dict[str, ToolReadout], expected: dict[str, str]
) -> dict[str, dict[str, str]]:
    """Worlds whose observed balance differs from the pre-registered one."""

    drift: dict[str, dict[str, str]] = {}
    for world, readout in readouts.items():
        observed = readout.visible["balance"]
        if observed != expected[world]:
            drift[world] = {"expected": expected[world], "observed": observed}
    return drift


def visible_field_diff(readouts: dict[str, ToolReadout]) -> list[str]:
    real, sham = readouts["REAL"].visible, readouts["SHAM"].visible
    return sorted(
        key for key in set(real) | set(sham) if real.get(key) != sham.get(key)
    )


def assert_pair_consistent(
    readouts: dict[str, ToolReadout],
    *,
    expected: dict[str, str] | None = None,
    allow_drift: bool = False,
) -> dict[str, Any]:
    """Structural checks (ValueError) plus the drift guard (RuntimeError).

    Returns the guard record ``{"block", "expected", "drift", "allow_drift"}``
    so callers can persist it into a manifest.
    """

    if set(readouts) != set(WORLDS):
        raise ValueError(f"expected readouts for {WORLDS}, got {sorted(readouts)}")
    real, sham = readouts["REAL"], readouts["SHAM"]
    if real.visible["block"] != sham.visible["block"]:
        raise ValueError("REAL and SHAM were read at different blocks")
    if set(real.visible) != set(sham.visible):
        raise ValueError("REAL and SHAM tool results expose different fields")
    for world, readout in readouts.items():
        if readout.world != world:
            raise ValueError(f"readout for {world} is labelled {readout.world}")
        lowered = readout.text.lower()
        for address in WORLD_ADDRESSES.values():
            if address[2:].lower() in lowered:
                raise ValueError("a world address leaked into a tool result")
    # Drift guard first: "REAL read 0.000000 vs pre-registered 1.900000" is the
    # actionable message when a world drifted; the balance-only-difference check
    # below would otherwise mask it behind a structural error.
    block = int(real.visible["block"])
    expected_here = expected_balances_at(block, expected)
    drift = check_drift(readouts, expected_here)
    if drift and not allow_drift:
        detail = "; ".join(
            f"{w} observed {d['observed']} vs pre-registered {d['expected']}" for w, d in drift.items()
        )
        raise RuntimeError(
            f"balance drift at block {block}: {detail}; refusing to collect "
            "(allow_drift=True overrides for a diagnostic run and is recorded)"
        )
    if visible_field_diff(readouts) != ["balance"]:
        raise ValueError(
            "tool results must differ in the balance field only, got "
            f"{visible_field_diff(readouts)}"
        )
    return {"block": block, "expected": expected_here, "drift": drift, "allow_drift": bool(allow_drift)}


def read_both_worlds_guarded(
    client: JsonRpcClient,
    block: int | None = None,
    *,
    expected: dict[str, str] | None = None,
    allow_drift: bool = False,
) -> tuple[dict[str, ToolReadout], dict[str, Any]]:
    """Read both worlds at one block; return readouts plus the guard record."""

    pinned = client.block_number() if block is None else block
    readouts = {world: check_balance(world, client, pinned) for world in WORLDS}
    guard = assert_pair_consistent(readouts, expected=expected, allow_drift=allow_drift)
    return readouts, guard


def read_both_worlds(
    client: JsonRpcClient,
    block: int | None = None,
    *,
    expected: dict[str, str] | None = None,
    allow_drift: bool = False,
) -> dict[str, ToolReadout]:
    """Read both worlds at the same block and fail closed on any inconsistency."""

    readouts, _ = read_both_worlds_guarded(client, block, expected=expected, allow_drift=allow_drift)
    return readouts
