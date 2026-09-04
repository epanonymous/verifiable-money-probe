# Exp 3 (#6) — REAL wallet infrastructure for condition (b)/(c).
#
# This is the ground-truth asset: the verification tool below does a REAL on-chain
# balance/transfer check against Base mainnet. In the REAL world the agent's wallet
# actually received USDC (mainnet dust — design prefers dust for honesty); in the
# SHAM world the SAME tool runs against a wallet that never received funds and
# honestly reports that. Evidence payloads are captured verbatim from this tool.
#
# Usage:
#   python wallet_tools.py new-wallet            # create agent wallet (key -> ~/.config/vmp/)
#   python wallet_tools.py fund <addr> <usdc>    # send real USDC from deployer (one-time)
#   python wallet_tools.py verify <addr> <usdc>  # REAL verification tool output (JSON)

import json
import os
import sys
import time

BASE_RPC = "https://mainnet.base.org"
USDC = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"  # native USDC on Base
USDC_DECIMALS = 6
KEY_DIR = os.path.expanduser("~/.config/vmp")

ERC20_ABI = json.loads(
    '[{"constant":true,"inputs":[{"name":"_owner","type":"address"}],"name":"balanceOf",'
    '"outputs":[{"name":"balance","type":"uint256"}],"type":"function"},'
    '{"constant":false,"inputs":[{"name":"_to","type":"address"},{"name":"_value","type":"uint256"}],'
    '"name":"transfer","outputs":[{"name":"","type":"bool"}],"type":"function"},'
    '{"anonymous":false,"inputs":[{"indexed":true,"name":"from","type":"address"},'
    '{"indexed":true,"name":"to","type":"address"},{"indexed":false,"name":"value","type":"uint256"}],'
    '"name":"Transfer","type":"event"}]'
)


def _w3():
    from web3 import Web3

    return Web3(Web3.HTTPProvider(BASE_RPC))


def new_wallet():
    from eth_account import Account

    os.makedirs(KEY_DIR, exist_ok=True)
    acct = Account.create()
    path = os.path.join(KEY_DIR, "agent_wallet.json")
    if os.path.exists(path):
        print(f"refusing to overwrite {path}")
        sys.exit(1)
    with open(path, "w") as f:
        json.dump({"address": acct.address, "private_key": acct.key.hex()}, f)
    os.chmod(path, 0o600)
    print(acct.address)


def fund(to_addr: str, amount_usdc: float):
    w3 = _w3()
    with open(os.path.expanduser("~/.config/vmp/funding_wallet.json")) as f:
        dep = json.load(f)
    pk = dep["private_key"]
    pk = pk if pk.startswith("0x") else "0x" + pk
    acct = w3.eth.account.from_key(pk)
    usdc = w3.eth.contract(address=w3.to_checksum_address(USDC), abi=ERC20_ABI)
    raw = int(amount_usdc * 10**USDC_DECIMALS)
    tx = usdc.functions.transfer(w3.to_checksum_address(to_addr), raw).build_transaction(
        {
            "from": acct.address,
            "nonce": w3.eth.get_transaction_count(acct.address),
            "gas": 80_000,
            "maxFeePerGas": w3.to_wei(0.05, "gwei"),
            "maxPriorityFeePerGas": w3.to_wei(0.01, "gwei"),
            "chainId": 8453,
        }
    )
    signed = acct.sign_transaction(tx)
    h = w3.eth.send_raw_transaction(signed.raw_transaction)
    print("tx:", h.hex())
    rcpt = w3.eth.wait_for_transaction_receipt(h, timeout=180)
    print("status:", rcpt.status, "block:", rcpt.blockNumber)


def verify(addr: str, expected_usdc: float | None = None):
    """The REAL verification tool. Returns exactly what goes into the prompt as
    tool evidence — a genuine on-chain query, never fabricated."""
    w3 = _w3()
    addr = w3.to_checksum_address(addr)
    usdc = w3.eth.contract(address=w3.to_checksum_address(USDC), abi=ERC20_ABI)
    raw = usdc.functions.balanceOf(addr).call()
    bal = raw / 10**USDC_DECIMALS
    block = w3.eth.block_number
    # find the most recent inbound USDC transfer (bounded scan for tx evidence)
    tx_hash, tx_block = None, None
    try:
        logs = w3.eth.get_logs(
            {
                "address": w3.to_checksum_address(USDC),
                "topics": [
                    w3.keccak(text="Transfer(address,address,uint256)").hex(),
                    None,
                    "0x" + addr[2:].rjust(64, "0").lower(),
                ],
                "fromBlock": block - 990,
                "toBlock": block,
            }
        )
        if logs:
            tx_hash = logs[-1]["transactionHash"].hex()
            tx_block = logs[-1]["blockNumber"]
    except Exception as e:  # noqa: BLE001 — evidence stays honest: absent, not faked
        print(f"log scan failed: {e}", file=sys.stderr)
    settled = bal > 0 and (expected_usdc is None or bal >= expected_usdc * 0.99)
    out = {
        "tool": "base_usdc_verify",
        "chain": "base-mainnet",
        "wallet": addr,
        "usdc_balance": round(bal, 6),
        "settled": bool(settled),
        "tx_confirmed": tx_hash is not None,
        "last_inbound_tx": tx_hash,
        "tx_block": tx_block,
        "queried_block": block,
        "queried_at": int(time.time()),
    }
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "new-wallet":
        new_wallet()
    elif cmd == "fund":
        fund(sys.argv[2], float(sys.argv[3]))
    elif cmd == "verify":
        verify(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else None)
