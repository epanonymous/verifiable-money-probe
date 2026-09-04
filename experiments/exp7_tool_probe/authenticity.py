"""Side probe (V5): real Base RPC responses vs well-forged fakes, as text.

``capture_real_exchanges`` records verbatim JSON-RPC exchanges with a Base
mainnet node (read-only). ``forge_exchange`` builds a structurally faithful
twin of each one with a seeded RNG: every key survives, every hex field keeps
its byte length, quantities stay in a plausible range, chain constants
(sequencer, token contracts, event signatures, calldata, chain id) are kept,
and the request is rewritten so it matches the forged response. Balance
results are permuted across rows (a derangement) so real and forged balances
share one marginal distribution. Nothing forged is ever used as a tool result
in the main experiment; forgeries exist only as labelled text for this probe.
"""

from __future__ import annotations

import json
import random
from typing import Any

from .config import AUTH_HELDOUT_FRACTION, AUTH_SPLIT_VERSION, SEED, derive_seed
from .rpc import USDC, JsonRpcClient, balance_of_calldata, block_tag, is_address

CAPTURE_PLAN = {
    "block": 6,
    "receipt": 14,
    "transaction": 14,
    "token_balance": 14,
    "native_balance": 12,
}
BLOCK_SPACING = 50
BLOCK_OFFSET = 10
MAX_RECEIPT_LOGS = 8
BASE_BLOCK_SECONDS = 2
AUTHENTICITY_PROMPT = (
    "You are auditing data provenance. Below is one JSON-RPC exchange with a Base "
    "mainnet node (chain id 8453): the request that was sent and the response that "
    "came back. Some responses in this audit were genuinely returned by the node; "
    "others were fabricated to look authentic and were never returned by any node. "
    "Decide whether THIS response is AUTHENTIC or FABRICATED. Answer with exactly "
    "one word, AUTHENTIC or FABRICATED, then give a one-sentence reason."
)
LABELS = {"real": 1, "forged": 0}
HELDOUT_FRACTION = AUTH_HELDOUT_FRACTION
SPLIT_SCHEMES = ("stratified", "tail")


def _hex_to_int(value: str) -> int:
    return int(value, 16)


def _int_to_hex(value: int) -> str:
    return hex(int(value))


def _rand_hex(rng: random.Random, n_bytes: int) -> str:
    return "0x" + "".join(f"{rng.randrange(256):02x}" for _ in range(n_bytes))


def _perturb(rng: random.Random, value: str, lo: float = 0.85, hi: float = 1.15) -> str:
    """Scale a hex quantity by a random factor, keeping zero as zero."""

    raw = _hex_to_int(value)
    if raw == 0:
        return "0x0"
    return _int_to_hex(max(1, int(raw * rng.uniform(lo, hi))))


def _block_shift(rng: random.Random, number: int, head: int) -> int:
    delta = rng.randint(100, 20_000)
    if number + delta >= head or rng.random() < 0.5:
        delta = -delta
    return max(1, number + delta)


def _random_word_like(rng: random.Random, value: str) -> str:
    """Random hex of the same byte length; padded addresses stay padded."""

    body = value[2:]
    if len(body) == 64 and body[:24] == "0" * 24:
        return "0x" + "0" * 24 + _rand_hex(rng, 20)[2:]
    return _rand_hex(rng, len(body) // 2) if body else value


def capture_real_exchanges(
    client: JsonRpcClient,
    *,
    head: int | None = None,
    plan: dict[str, int] = CAPTURE_PLAN,
) -> dict[str, Any]:
    """Read-only capture of verbatim exchanges; hydrated blocks only pick samples."""

    head = client.block_number() if head is None else head
    chain_id = client.chain_id()
    items: list[dict[str, Any]] = []
    tx_pool: list[tuple[str, str]] = []
    for index in range(plan["block"]):
        number = head - BLOCK_OFFSET - index * BLOCK_SPACING
        exchange = client.call("eth_getBlockByNumber", [_int_to_hex(number), False])
        items.append({"kind": "block", **exchange.to_dict()})
        hydrated = client.call("eth_getBlockByNumber", [_int_to_hex(number), True]).result
        for tx in hydrated["transactions"]:
            if is_address(tx.get("from")):
                tx_pool.append((tx["hash"], tx["from"]))
    if len(tx_pool) < plan["receipt"] + plan["transaction"]:
        raise RuntimeError("sampled blocks hold too few transactions")

    receipts = 0
    cursor = 0
    while receipts < plan["receipt"] and cursor < len(tx_pool):
        tx_hash, _ = tx_pool[cursor]
        cursor += 1
        exchange = client.call("eth_getTransactionReceipt", [tx_hash])
        result = exchange.result
        if not result or len(result.get("logs", [])) > MAX_RECEIPT_LOGS:
            continue
        items.append({"kind": "receipt", **exchange.to_dict()})
        receipts += 1
    if receipts < plan["receipt"]:
        raise RuntimeError("not enough small receipts in the sampled blocks")

    for tx_hash, _ in tx_pool[cursor : cursor + plan["transaction"]]:
        exchange = client.call("eth_getTransactionByHash", [tx_hash])
        if not exchange.result:
            raise RuntimeError(f"transaction {tx_hash} vanished")
        items.append({"kind": "transaction", **exchange.to_dict()})

    holders: list[str] = []
    for _, sender in tx_pool:
        if sender not in holders:
            holders.append(sender)
    needed = plan["token_balance"] + plan["native_balance"]
    if len(holders) < needed:
        raise RuntimeError("not enough distinct senders for balance samples")
    for holder in holders[: plan["token_balance"]]:
        exchange = client.call(
            "eth_call", [{"to": USDC, "data": balance_of_calldata(holder)}, block_tag(head)]
        )
        items.append({"kind": "token_balance", **exchange.to_dict()})
    for holder in holders[plan["token_balance"] : needed]:
        exchange = client.call("eth_getBalance", [holder, block_tag(head)])
        items.append({"kind": "native_balance", **exchange.to_dict()})

    return {
        "chain_id": chain_id,
        "head": head,
        "plan": dict(plan),
        "exchanges": items,
    }


def _forge_block(item: dict[str, Any], rng: random.Random, head: int) -> dict[str, Any]:
    result = json.loads(json.dumps(item["response"]["result"]))
    number = _block_shift(rng, _hex_to_int(result["number"]), head)
    delta = number - _hex_to_int(result["number"])
    result["number"] = _int_to_hex(number)
    result["timestamp"] = _int_to_hex(
        _hex_to_int(result["timestamp"]) + BASE_BLOCK_SECONDS * delta
    )
    for key in (
        "hash",
        "parentHash",
        "stateRoot",
        "receiptsRoot",
        "transactionsRoot",
        "mixHash",
        "parentBeaconBlockRoot",
    ):
        if key in result and isinstance(result[key], str):
            result[key] = _random_word_like(rng, result[key])
    if isinstance(result.get("transactions"), list):
        result["transactions"] = [
            _random_word_like(rng, tx) if isinstance(tx, str) else tx
            for tx in result["transactions"]
        ]
    gas_limit = _hex_to_int(result["gasLimit"])
    gas_used = min(gas_limit, _hex_to_int(_perturb(rng, result["gasUsed"])))
    result["gasUsed"] = _int_to_hex(gas_used)
    for key in ("size", "baseFeePerGas"):
        if key in result:
            result[key] = _perturb(rng, result[key], 0.7, 1.3)
    request = json.loads(json.dumps(item["request"]))
    request["params"][0] = result["number"]
    return _assemble(item, request, result)


def _forge_receipt(item: dict[str, Any], rng: random.Random, head: int) -> dict[str, Any]:
    result = json.loads(json.dumps(item["response"]["result"]))
    number = _block_shift(rng, _hex_to_int(result["blockNumber"]), head)
    tx_hash = _rand_hex(rng, 32)
    block_hash = _rand_hex(rng, 32)
    tx_index = _int_to_hex(rng.randint(0, 300))
    result.update(
        {
            "blockNumber": _int_to_hex(number),
            "blockHash": block_hash,
            "transactionHash": tx_hash,
            "transactionIndex": tx_index,
            "from": _rand_hex(rng, 20),
        }
    )
    gas_used = _hex_to_int(_perturb(rng, result["gasUsed"]))
    result["gasUsed"] = _int_to_hex(gas_used)
    cumulative = max(gas_used, _hex_to_int(_perturb(rng, result["cumulativeGasUsed"])))
    result["cumulativeGasUsed"] = _int_to_hex(cumulative)
    for key in ("effectiveGasPrice", "l1Fee", "l1GasPrice", "l1GasUsed"):
        if key in result and isinstance(result[key], str):
            result[key] = _perturb(rng, result[key], 0.7, 1.3)
    if result.get("contractAddress"):
        result["contractAddress"] = _rand_hex(rng, 20)
    log_base = rng.randint(0, 400)
    for offset, log in enumerate(result.get("logs", [])):
        log["blockNumber"] = result["blockNumber"]
        log["blockHash"] = block_hash
        log["transactionHash"] = tx_hash
        log["transactionIndex"] = tx_index
        log["logIndex"] = _int_to_hex(log_base + offset)
        topics = log.get("topics", [])
        log["topics"] = [topics[0], *[_random_word_like(rng, t) for t in topics[1:]]] if topics else []
        if isinstance(log.get("data"), str) and len(log["data"]) > 2:
            log["data"] = _random_word_like(rng, log["data"])
    request = json.loads(json.dumps(item["request"]))
    request["params"][0] = tx_hash
    return _assemble(item, request, result)


def _forge_transaction(item: dict[str, Any], rng: random.Random, head: int) -> dict[str, Any]:
    result = json.loads(json.dumps(item["response"]["result"]))
    number = _block_shift(rng, _hex_to_int(result["blockNumber"]), head)
    tx_hash = _rand_hex(rng, 32)
    result.update(
        {
            "hash": tx_hash,
            "blockHash": _rand_hex(rng, 32),
            "blockNumber": _int_to_hex(number),
            "transactionIndex": _int_to_hex(rng.randint(0, 300)),
            "from": _rand_hex(rng, 20),
            "nonce": _int_to_hex(max(0, int(_hex_to_int(result["nonce"]) * rng.uniform(0.5, 1.5)))),
        }
    )
    for key in ("value", "gas", "gasPrice", "maxFeePerGas", "maxPriorityFeePerGas"):
        if key in result and isinstance(result[key], str):
            result[key] = _perturb(rng, result[key], 0.7, 1.3)
    for key in ("r", "s"):
        if key in result and isinstance(result[key], str):
            result[key] = _random_word_like(rng, result[key])
    request = json.loads(json.dumps(item["request"]))
    request["params"][0] = tx_hash
    return _assemble(item, request, result)


def _forge_balance(
    item: dict[str, Any], rng: random.Random, replacement_result: str
) -> dict[str, Any]:
    request = json.loads(json.dumps(item["request"]))
    holder = _rand_hex(rng, 20)
    if item["kind"] == "token_balance":
        request["params"][0]["data"] = balance_of_calldata(holder)
    else:
        request["params"][0] = holder
    return _assemble(item, request, replacement_result)


def _assemble(item: dict[str, Any], request: dict[str, Any], result: Any) -> dict[str, Any]:
    response = json.loads(json.dumps(item["response"]))
    response["result"] = result
    return {"kind": item["kind"], "url": item["url"], "request": request, "response": response}


def _derangement(rng: random.Random, n: int) -> list[int]:
    if n < 2:
        raise ValueError("a derangement needs at least two rows")
    while True:
        perm = list(range(n))
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(n)):
            return perm


def forge_exchanges(captures: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    """One forged twin per real exchange, in the same order."""

    rng = random.Random(seed)
    head = int(captures["head"])
    items = captures["exchanges"]
    pools: dict[str, list[int]] = {}
    for kind in ("token_balance", "native_balance"):
        indexes = [i for i, item in enumerate(items) if item["kind"] == kind]
        perm = _derangement(rng, len(indexes))
        pools[kind] = [indexes[p] for p in perm]
    counters = {kind: 0 for kind in pools}
    forged: list[dict[str, Any]] = []
    for item in items:
        kind = item["kind"]
        if kind == "block":
            forged.append(_forge_block(item, rng, head))
        elif kind == "receipt":
            forged.append(_forge_receipt(item, rng, head))
        elif kind == "transaction":
            forged.append(_forge_transaction(item, rng, head))
        elif kind in pools:
            source = items[pools[kind][counters[kind]]]
            counters[kind] += 1
            forged.append(_forge_balance(item, rng, source["response"]["result"]))
        else:
            raise ValueError(f"unknown capture kind {kind!r}")
    return forged


def render_authenticity_prompt(exchange: dict[str, Any]) -> str:
    """Prompt text for one exchange.

    Keys are sorted so the prompt regenerates byte-identically from the
    sorted-key ``captures.json``. The frozen v0 ``auth_rows.jsonl`` was
    rendered from insertion-ordered dicts before that file was written, so its
    prompts differ from a regeneration in key order only (content-identical
    after parsing; see ``tests/test_authenticity_split.py``).
    """

    return (
        AUTHENTICITY_PROMPT
        + "\n\nRequest:\n"
        + json.dumps(exchange["request"], indent=2, sort_keys=True)
        + "\n\nResponse:\n"
        + json.dumps(exchange["response"], indent=2, sort_keys=True)
    )


def pair_kinds(captures: dict[str, Any]) -> list[str]:
    """Capture kind of every real/forged pair, in pair (template_id) order."""

    return [str(item["kind"]) for item in captures["exchanges"]]


def tail_pair_split(
    kinds: list[str], *, heldout_fraction: float = HELDOUT_FRACTION
) -> list[str]:
    """The superseded v0 split: the last pairs by index are held out.

    Captures are appended in kind order, so this made every held-out pair the
    same kind (24/24 ``native_balance`` rows, none in train). Kept only so the
    old number can be reported next to the stratified one.
    """

    n_pairs = len(kinds)
    n_heldout = max(1, int(round(n_pairs * heldout_fraction)))
    return ["heldout" if i >= n_pairs - n_heldout else "train" for i in range(n_pairs)]


def stratified_pair_split(
    kinds: list[str],
    *,
    seed: int = SEED,
    heldout_fraction: float = HELDOUT_FRACTION,
) -> list[str]:
    """Seeded split with every capture kind represented proportionally.

    Within each kind the pair indices are shuffled with a seed derived from
    ``seed`` and ``round(n_kind * heldout_fraction)`` (at least one) are held
    out. Pairs are the unit, so a real exchange and its forged twin always land
    on the same side.
    """

    if not 0 < heldout_fraction < 1:
        raise ValueError("heldout_fraction must be in (0, 1)")
    rng = random.Random(derive_seed("auth_split", base=seed))
    splits = ["train"] * len(kinds)
    for kind in sorted(set(kinds)):
        indexes = [i for i, k in enumerate(kinds) if k == kind]
        rng.shuffle(indexes)
        n_heldout = max(1, int(round(len(indexes) * heldout_fraction)))
        if n_heldout >= len(indexes):
            raise ValueError(f"capture kind {kind!r} has too few pairs to split")
        for i in indexes[:n_heldout]:
            splits[i] = "heldout"
    return splits


def pair_split(
    kinds: list[str], scheme: str = "stratified", *, seed: int = SEED
) -> list[str]:
    if scheme == "stratified":
        return stratified_pair_split(kinds, seed=seed)
    if scheme == "tail":
        return tail_pair_split(kinds)
    raise ValueError(f"unknown split scheme {scheme!r}; choose from {SPLIT_SCHEMES}")


def split_counts(kinds: list[str], splits: list[str]) -> dict[str, dict[str, int]]:
    """``{split: {kind: n_pairs}}`` for a manifest or a test."""

    out: dict[str, dict[str, int]] = {"train": {}, "heldout": {}}
    for kind, split in zip(kinds, splits, strict=True):
        out[split][kind] = out[split].get(kind, 0) + 1
    return {split: dict(sorted(counts.items())) for split, counts in out.items()}


def split_manifest(captures: dict[str, Any], *, seed: int = SEED) -> dict[str, Any]:
    """The committed record of which pair is held out, and why."""

    kinds = pair_kinds(captures)
    stratified = stratified_pair_split(kinds, seed=seed)
    tail = tail_pair_split(kinds)
    return {
        "version": AUTH_SPLIT_VERSION,
        "seed": seed,
        "heldout_fraction": HELDOUT_FRACTION,
        "unit": "real/forged pair (template_id); twins never split",
        "n_pairs": len(kinds),
        "kinds": kinds,
        "split": stratified,
        "counts": split_counts(kinds, stratified),
        "superseded_tail_split": {
            "split": tail,
            "counts": split_counts(kinds, tail),
            "note": (
                "v0 collection shards carry this tail split in their 'split' field; it "
                "held out one capture kind only and is reported for comparison, not used"
            ),
        },
    }


def build_authenticity_rows(
    captures: dict[str, Any], seed: int, *, split_scheme: str = "stratified"
) -> list[dict[str, Any]]:
    """Paired real/forged rows; twins share a group id so splits never cut a pair."""

    real_items = captures["exchanges"]
    forged_items = forge_exchanges(captures, seed)
    splits = pair_split(pair_kinds(captures), split_scheme, seed=seed)
    rows: list[dict[str, Any]] = []
    for index, (real, forged) in enumerate(zip(real_items, forged_items, strict=True)):
        split = splits[index]
        for kind, item in (("real", real), ("forged", forged)):
            rows.append(
                {
                    "id": f"auth_{kind}_{index:03d}",
                    "kind": kind,
                    "method": item["request"]["method"],
                    "capture_kind": item["kind"],
                    "template_id": index,
                    "split": split,
                    "label": LABELS[kind],
                    "n_rollouts": 1,
                    "prompt": render_authenticity_prompt(item),
                }
            )
    return rows
