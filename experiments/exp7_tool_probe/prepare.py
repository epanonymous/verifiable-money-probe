"""Exp 7 v0 — CPU prepare step: freeze every input the GPU collector needs.

Runs once on a box with Base RPC access and writes ``data/v0/``:

    rows.jsonl       96 probe rows (48 templates x REAL/SHAM), 38/10 template split
    readouts.json    the ONE pinned-block tool readout per world (subject-visible
                     JSON text + private provenance kept out of the prompt)
    captures.json    verbatim Base JSON-RPC exchanges for the V5 authenticity side probe
    auth_rows.jsonl  120 authenticity rows (60 real / 60 forged, pair-grouped split)
    auth_split.json  which pairs are held out (stratified by capture kind) and why
    manifest.json    counts, pinned block, sha256 of every file, leak-gate result,
                     drift-guard record, seed and config hash

The collector on Modal then has no network dependency and every world pair is
read at exactly one block. Nothing here spends, signs, or holds custody.

Drift guard: the live readout is compared with the pre-registered balances for
the block it was read at. The pinned v0 block has built-in expectations; any
other block must be given ``--expect-real/--expect-sham`` explicitly (a new
lock). A mismatch aborts unless ``--allow-drift`` is passed, and that flag is
written into ``manifest.json`` so it can never be silent. After the post-run
sweep the REAL wallet no longer holds the pinned amount, so a fresh block will
legitimately read differently; that is exactly the case the guard refuses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from .authenticity import build_authenticity_rows, capture_real_exchanges, split_manifest
from .config import (
    AUTH_SPLIT_VERSION,
    DESIGN_LOCK,
    EXPERIMENT,
    N_ROLLOUTS,
    PINNED_BALANCES,
    SEED,
    config_fields,
    config_hash,
    seed_everything,
)
from .context import TOOL_SCHEMA, p0_messages, serialize_context
from .dataset import build_rows, write_jsonl
from .rpc import connect, http_transport
from .worlds import WORLD_ADDRESSES, read_both_worlds_guarded

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "data" / "v0"
FROZEN_FILES = ("rows.jsonl", "readouts.json", "captures.json", "auth_rows.jsonl", "auth_split.json")


def patient_transport(url: str, timeout: float = 20.0, *, pause: float = 0.35, tries: int = 8):
    """Public Base RPCs rate-limit bursts (HTTP 429); space calls and back off."""

    import time
    import urllib.error

    inner = http_transport(url, timeout)

    def send(request):
        delay = 1.0
        for attempt in range(tries):
            try:
                result = inner(request)
                time.sleep(pause)
                return result
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                code = getattr(exc, "code", None)
                if attempt == tries - 1 or (code is not None and code not in (429, 502, 503, 504)):
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 16.0)
        raise RuntimeError("unreachable")

    return send


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def leak_gate(rows, readouts, auth_rows) -> dict:
    """Wallet addresses must never appear in anything the subject can see."""

    leaks = []
    addresses = [a.lower() for a in WORLD_ADDRESSES.values()]
    subject_text = []
    for row in rows:
        subject_text.append(("row:" + row.row_id, serialize_context(
            p0_messages(row.system_prompt, row.user_prompt), TOOL_SCHEMA)))
    for world, readout in readouts.items():
        subject_text.append(("tool_text:" + world, readout["text"]))
    for row in auth_rows:
        subject_text.append(("auth:" + row["id"], row["prompt"]))
    for name, text in subject_text:
        low = text.lower()
        for address in addresses:
            if address in low or address[2:] in low:
                leaks.append(name)
    visible_fields = {w: sorted(r["visible"]) for w, r in readouts.items()}
    return {"leaks": leaks, "visible_fields": visible_fields, "pass": not leaks}


def build_manifest(
    *,
    out: Path,
    rows,
    readouts_json: dict[str, Any],
    auth_rows: list[dict[str, Any]],
    auth_split: dict[str, Any],
    guard: dict[str, Any],
    gate: dict[str, Any],
    rows_meta: dict[str, Any],
    n_rollouts: int,
    rpc_url: str | None,
) -> dict[str, Any]:
    """Pure manifest assembly (tested without a network)."""

    block = readouts_json["REAL"]["visible"]["block"]
    return {
        "experiment": EXPERIMENT,
        "design_lock": DESIGN_LOCK,
        "block": block,
        "rpc_url": rpc_url,
        "seed": SEED,
        "config_hash": config_hash(),
        "config": config_fields(),
        "n_rows": len(rows),
        "n_templates": len({r.template_id for r in rows}),
        "split_counts": {
            s: len({r.template_id for r in rows if r.split == s}) for s in ("train", "heldout")
        },
        "n_rollouts_per_row": n_rollouts,
        "n_auth_rows": len(auth_rows),
        "auth_split_version": AUTH_SPLIT_VERSION,
        "auth_split_counts": {
            s: sum(1 for r in auth_rows if r["split"] == s) for s in ("train", "heldout")
        },
        "auth_split_counts_by_kind": auth_split["counts"],
        "balances": {w: r["visible"]["balance"] for w, r in readouts_json.items()},
        "drift_guard": guard,
        "rows_meta": rows_meta,
        "leak_gate": gate,
        "files": {name: sha256(out / name) for name in FROZEN_FILES},
    }


def regen_auth_split(captures_path: Path, output: Path) -> int:
    """CPU-only: rebuild auth_split.json from a committed captures.json."""

    captures = json.loads(captures_path.read_text())
    manifest = split_manifest(captures, seed=SEED)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output} ({manifest['n_pairs']} pairs, heldout by kind {manifest['counts']['heldout']})")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--regen-auth-split",
        nargs=2,
        metavar=("CAPTURES_JSON", "OUTPUT_JSON"),
        default=None,
        help="CPU-only: rebuild the V5 split manifest from captures.json and exit (no RPC)",
    )
    parser.add_argument("--n-rollouts", type=int, default=N_ROLLOUTS)
    parser.add_argument("--block", type=int, default=None, help="pin an explicit block")
    parser.add_argument(
        "--expect-real",
        default=None,
        help=f"pre-registered REAL balance at --block (built in only for the pinned block: {PINNED_BALANCES['REAL']})",
    )
    parser.add_argument(
        "--expect-sham",
        default=None,
        help=f"pre-registered SHAM balance at --block (built in only for the pinned block: {PINNED_BALANCES['SHAM']})",
    )
    parser.add_argument(
        "--allow-drift",
        action="store_true",
        help="diagnostic only: continue when the readout differs from the pre-registered balance; recorded in manifest.json",
    )
    args = parser.parse_args(argv)
    if args.regen_auth_split is not None:
        return regen_auth_split(Path(args.regen_auth_split[0]), Path(args.regen_auth_split[1]))
    if (args.expect_real is None) != (args.expect_sham is None):
        parser.error("--expect-real and --expect-sham must be given together")
    expected = (
        None
        if args.expect_real is None
        else {"REAL": args.expect_real, "SHAM": args.expect_sham}
    )

    seed_everything(SEED)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = build_rows(n_rollouts=args.n_rollouts)
    rows_meta = write_jsonl(out / "rows.jsonl", rows)

    client = connect(transport_factory=patient_transport)
    readouts, guard = read_both_worlds_guarded(
        client, block=args.block, expected=expected, allow_drift=args.allow_drift
    )
    if guard["drift"]:
        print(f"WARNING: balance drift accepted with --allow-drift: {guard['drift']}", file=sys.stderr)
    readouts_json = {
        world: {
            "world": world,
            "text": readout.text,
            "visible": readout.visible,
            "provenance": readout.provenance,
        }
        for world, readout in readouts.items()
    }
    (out / "readouts.json").write_text(json.dumps(readouts_json, indent=2, sort_keys=True))
    block = readouts["REAL"].visible["block"]

    captures = capture_real_exchanges(client, head=block)
    (out / "captures.json").write_text(json.dumps(captures, indent=2, sort_keys=True))
    auth_rows = build_authenticity_rows(captures, seed=SEED)
    with open(out / "auth_rows.jsonl", "w") as handle:
        for row in auth_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    auth_split = split_manifest(captures, seed=SEED)
    (out / "auth_split.json").write_text(json.dumps(auth_split, indent=2, sort_keys=True) + "\n")

    gate = leak_gate(rows, readouts_json, auth_rows)
    manifest = build_manifest(
        out=out,
        rows=rows,
        readouts_json=readouts_json,
        auth_rows=auth_rows,
        auth_split=auth_split,
        guard=guard,
        gate=gate,
        rows_meta=rows_meta,
        n_rollouts=args.n_rollouts,
        rpc_url=getattr(client, "url", None),
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("files", "config")}, indent=2))
    if not gate["pass"]:
        print("LEAK GATE FAILED", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
