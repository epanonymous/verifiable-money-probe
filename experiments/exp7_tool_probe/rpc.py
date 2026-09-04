"""Read-only JSON-RPC client for Base mainnet, standard library only.

Every request this client sends is on a fixed allowlist of ``eth_*`` reads.
There is no code path that signs, broadcasts, or loads key material. The REAL
and SHAM worlds of Exp 7 differ only in which address the same read is pointed
at (see ``worlds.py``); no response is ever edited or fabricated here.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

BASE_RPC = "https://mainnet.base.org"
FALLBACK_RPCS = (
    "https://base-rpc.publicnode.com",
    "https://base.drpc.org",
    "https://1rpc.io/base",
)
CHAIN_ID = 8453
USDC = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"  # native USDC on Base
USDC_DECIMALS = 6
BALANCE_OF_SELECTOR = "0x70a08231"  # keccak256("balanceOf(address)")[:4]
USER_AGENT = "vmp-exp7-readonly/0.1"
READ_METHODS = frozenset(
    {
        "eth_blockNumber",
        "eth_chainId",
        "eth_call",
        "eth_getBalance",
        "eth_getBlockByHash",
        "eth_getBlockByNumber",
        "eth_getCode",
        "eth_getLogs",
        "eth_getTransactionByHash",
        "eth_getTransactionCount",
        "eth_getTransactionReceipt",
    }
)
HEX_DIGITS = frozenset("0123456789abcdefABCDEF")

Transport = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class RpcExchange:
    """One request/response pair exactly as sent and received."""

    url: str
    request: dict[str, Any]
    response: dict[str, Any]

    @property
    def result(self) -> Any:
        return self.response["result"]

    def to_dict(self) -> dict[str, Any]:
        return {"url": self.url, "request": self.request, "response": self.response}


def is_address(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 42
        and value[:2] == "0x"
        and all(char in HEX_DIGITS for char in value[2:])
    )


def is_hex_quantity(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) > 2
        and value[:2] == "0x"
        and all(char in HEX_DIGITS for char in value[2:])
    )


def balance_of_calldata(holder: str) -> str:
    """ABI-encode ``balanceOf(address)`` for one holder."""

    if not is_address(holder):
        raise ValueError(f"not an EVM address: {holder!r}")
    return BALANCE_OF_SELECTOR + holder[2:].lower().rjust(64, "0")


def decode_uint256(value: object) -> int:
    if not isinstance(value, str) or not value.startswith("0x"):
        raise ValueError(f"eth_call result is not hex: {value!r}")
    body = value[2:]
    if len(body) != 64 or not all(char in HEX_DIGITS for char in body):
        raise ValueError(f"eth_call result is not one 32-byte word: {value!r}")
    return int(body, 16)


def format_units(raw: int, decimals: int = USDC_DECIMALS) -> str:
    """Render an integer token amount with a fixed number of decimals."""

    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise ValueError(f"raw amount must be a non-negative int, got {raw!r}")
    whole, frac = divmod(raw, 10**decimals)
    return f"{whole}.{frac:0{decimals}d}"


def block_tag(block: int | str) -> str:
    if isinstance(block, bool):
        raise TypeError("block must be an int or a named tag")
    if isinstance(block, int):
        if block < 0:
            raise ValueError("block number must be non-negative")
        return hex(block)
    if block in {"latest", "safe", "finalized", "earliest"}:
        return block
    raise ValueError(f"unknown block tag {block!r}")


def http_transport(url: str, timeout: float = 20.0) -> Transport:
    def send(payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "content-type": "application/json",
                "accept": "application/json",
                "user-agent": USER_AGENT,
            },
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    return send


class JsonRpcClient:
    """Allowlisted read-only JSON-RPC calls with the exchange kept verbatim."""

    def __init__(
        self,
        url: str = BASE_RPC,
        transport: Transport | None = None,
        *,
        timeout: float = 20.0,
    ) -> None:
        self.url = url
        self._send = transport if transport is not None else http_transport(url, timeout)
        self._next_id = 1

    def call(self, method: str, params: Sequence[Any] = ()) -> RpcExchange:
        if method not in READ_METHODS:
            raise PermissionError(f"{method!r} is not an allowlisted read-only method")
        request = {
            "jsonrpc": "2.0",
            "id": self._next_id,
            "method": method,
            "params": list(params),
        }
        self._next_id += 1
        response = self._send(json.loads(json.dumps(request)))
        if not isinstance(response, dict):
            raise RuntimeError(f"{method}: non-object JSON-RPC response")
        if response.get("error") is not None:
            raise RuntimeError(f"{method}: JSON-RPC error {response['error']!r}")
        if "result" not in response:
            raise RuntimeError(f"{method}: JSON-RPC response carries no result")
        if response.get("id") != request["id"]:
            raise RuntimeError(f"{method}: JSON-RPC id mismatch")
        return RpcExchange(url=self.url, request=request, response=response)

    def chain_id(self) -> int:
        return int(self.call("eth_chainId").result, 16)

    def block_number(self) -> int:
        return int(self.call("eth_blockNumber").result, 16)

    def erc20_balance_of(
        self, token: str, holder: str, block: int | str
    ) -> tuple[int, RpcExchange]:
        exchange = self.call(
            "eth_call",
            [{"to": token, "data": balance_of_calldata(holder)}, block_tag(block)],
        )
        return decode_uint256(exchange.result), exchange


def connect(
    urls: Iterable[str] = (BASE_RPC, *FALLBACK_RPCS),
    *,
    timeout: float = 20.0,
    transport_factory: Callable[[str, float], Transport] = http_transport,
) -> JsonRpcClient:
    """Return the first endpoint that answers and reports Base's chain id."""

    errors: list[str] = []
    for url in urls:
        client = JsonRpcClient(url, transport_factory(url, timeout))
        try:
            chain = client.chain_id()
        except Exception as exc:  # noqa: BLE001 - try the next endpoint, report all
            errors.append(f"{url}: {exc}")
            continue
        if chain != CHAIN_ID:
            errors.append(f"{url}: chain id {chain} != {CHAIN_ID}")
            continue
        return client
    raise RuntimeError("no Base mainnet endpoint answered: " + "; ".join(errors))
