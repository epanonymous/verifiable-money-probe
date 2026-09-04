"""Build the versioned Exp 3 prompt datasets.

The default output preserves Run v1. ``--leak-free`` writes a parallel dataset
whose subject-visible evidence is an allowlisted, class-invariant projection of
the raw wallet captures. The raw captures remain untouched as provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

try:
    from . import templates as T
except ImportError:  # supports ``python experiments/exp3_dataset/build_dataset.py``
    import templates as T


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data"
DEFAULT_LEAK_FREE_DATA = HERE / "data_leak_free"
N_TEMPLATES = 48
N_TRAIN = 38  # templates 0-37 train, 38-47 held-out; split by template only
N_ROLLOUTS = 25
LBR_SHAM, LBR_REAL = 1000, 10
AMT = "1.90"  # true on-chain amount; receipts in BOTH worlds claim this figure

# Only immutable tool identity is rendered in leak-free prompts. In particular,
# no result, status, balance, address, transaction, block, timestamp, or error
# field can reach the subject-visible text.
LEAK_FREE_EVIDENCE_FIELDS = ("tool", "chain")


REAL_EVIDENCE_PATH = HERE / "evidence_real.json"
SHAM_EVIDENCE_PATH = HERE / "evidence_sham.json"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_evidence() -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    real_text = REAL_EVIDENCE_PATH.read_text(encoding="utf-8").strip()
    sham_text = SHAM_EVIDENCE_PATH.read_text(encoding="utf-8").strip()
    real = json.loads(real_text)
    sham = json.loads(sham_text)
    if not isinstance(real, dict) or not isinstance(sham, dict):
        raise ValueError("evidence captures must be JSON objects")
    return real_text, sham_text, real, sham


def _leak_free_evidence(real: dict[str, Any], sham: dict[str, Any]) -> str:
    """Return the same conservative evidence projection for both worlds."""

    visible: dict[str, Any] = {}
    for field in LEAK_FREE_EVIDENCE_FIELDS:
        if field not in real or field not in sham:
            raise ValueError(f"raw evidence is missing invariant field {field!r}")
        if real[field] != sham[field]:
            raise ValueError(
                f"allowlisted evidence field {field!r} differs across worlds"
            )
        if not isinstance(real[field], str):
            raise ValueError(f"allowlisted evidence field {field!r} must be a string")
        visible[field] = real[field]
    return json.dumps(visible, indent=2, sort_keys=True)


def rows_main(
    tpls: Sequence[tuple[int, int, int, int, int]], ev_real: str, ev_sham: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tid, tpl in enumerate(tpls):
        for world, evidence in (("REAL", ev_real), ("SHAM", ev_sham)):
            for cond in ("a", "b", "c"):
                prompt = T.render(
                    tpl, cond, AMT, evidence if cond in ("b", "c") else None
                )
                rows.append(
                    {
                        "id": f"{world.lower()}_{cond}_t{tid:02d}",
                        "world": world,
                        "cond": cond,
                        "template_id": tid,
                        "split": "train" if tid < N_TRAIN else "heldout",
                        "label": 1 if world == "REAL" else 0,
                        "n_rollouts": N_ROLLOUTS,
                        "prompt": prompt,
                    }
                )
    return rows


def rows_framing(
    tpls: Sequence[tuple[int, int, int, int, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tid, tpl in enumerate(tpls):
        prompt = T.render(tpl, "framing", AMT, None, framing_idx=tid)
        rows.append(
            {
                "id": f"framing_t{tid:02d}",
                "world": "FRAMING",
                "cond": "framing",
                "template_id": tid,
                "split": "train" if tid < N_TRAIN else "heldout",
                "label": None,
                "n_rollouts": N_ROLLOUTS,
                "prompt": prompt,
            }
        )
    return rows


def rows_lowbaserate(
    tpls: Sequence[tuple[int, int, int, int, int]], ev_real: str, ev_sham: str
) -> list[dict[str, Any]]:
    """Build the held-out-template, condition-b, ~1% REAL evaluation set."""

    held = list(range(N_TRAIN, N_TEMPLATES))
    rows: list[dict[str, Any]] = []
    for i in range(LBR_SHAM):
        tid = held[i % len(held)]
        rows.append(
            {
                "id": f"lbr_sham_{i:04d}",
                "world": "SHAM",
                "cond": "b",
                "template_id": tid,
                "rollout_idx": i // len(held),
                "label": 0,
                "prompt": T.render(tpls[tid], "b", AMT, ev_sham),
            }
        )
    for i in range(LBR_REAL):
        tid = held[i % len(held)]
        rows.append(
            {
                "id": f"lbr_real_{i:04d}",
                "world": "REAL",
                "cond": "b",
                "template_id": tid,
                "rollout_idx": 0,
                "label": 1,
                "prompt": T.render(tpls[tid], "b", AMT, ev_real),
            }
        )
    return rows


def assert_leak_free_prompt_equivalence(
    main_rows: Sequence[dict[str, Any]], lbr_rows: Sequence[dict[str, Any]]
) -> None:
    """Fail closed unless text has exactly the same distribution in both classes."""

    paired: dict[tuple[int, str], dict[str, str]] = {}
    for row in main_rows:
        key = (int(row["template_id"]), str(row["cond"]))
        paired.setdefault(key, {})[str(row["world"])] = str(row["prompt"])
    for key, prompts in paired.items():
        if set(prompts) != {"REAL", "SHAM"} or prompts["REAL"] != prompts["SHAM"]:
            raise ValueError(f"leak-free main prompts differ across worlds for {key}")

    by_world = {
        world: Counter(str(row["prompt"]) for row in lbr_rows if row["world"] == world)
        for world in ("REAL", "SHAM")
    }
    totals = {world: sum(counts.values()) for world, counts in by_world.items()}
    if not all(totals.values()):
        raise ValueError("leak-free low-base-rate data must contain both classes")
    if set(by_world["REAL"]) != set(by_world["SHAM"]):
        raise ValueError("leak-free low-base-rate prompt support differs by world")
    for prompt in by_world["REAL"]:
        real_rate = Fraction(by_world["REAL"][prompt], totals["REAL"])
        sham_rate = Fraction(by_world["SHAM"][prompt], totals["SHAM"])
        if real_rate != sham_rate:
            raise ValueError(
                "leak-free low-base-rate prompt distribution differs by world"
            )


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return {
        "rows": len(rows),
        "sha256": _sha256_bytes(path.read_bytes())[:16],
    }


def build_dataset(output_dir: Path, *, leak_free: bool = False) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    real_text, sham_text, real, sham = _load_evidence()
    if leak_free:
        rendered = _leak_free_evidence(real, sham)
        ev_real = ev_sham = rendered
    else:
        ev_real, ev_sham = real_text, sham_text

    tpls = T.sample_templates(N_TEMPLATES)
    main_rows = rows_main(tpls, ev_real, ev_sham)
    framing_rows = rows_framing(tpls)
    lbr_rows = rows_lowbaserate(tpls, ev_real, ev_sham)
    if leak_free:
        assert_leak_free_prompt_equivalence(main_rows, lbr_rows)

    files: dict[str, Any] = {}
    for name, rows in (
        ("prompts_main.jsonl", main_rows),
        ("prompts_framing.jsonl", framing_rows),
        ("lowbaserate_eval.jsonl", lbr_rows),
    ):
        files[name] = _write_jsonl(output_dir / name, rows)
        print(name, files[name])

    (output_dir / "splits.json").write_text(
        json.dumps(
            {
                "train": list(range(N_TRAIN)),
                "heldout": list(range(N_TRAIN, N_TEMPLATES)),
            }
        ),
        encoding="utf-8",
    )

    if leak_free:
        manifest = {
            "version": 2,
            "variant": "leak_free",
            "seed": 7,
            "n_templates": N_TEMPLATES,
            "split_rule": "by template only (0-37 train / 38-47 heldout)",
            "n_rollouts_per_row": N_ROLLOUTS,
            "amount_usdc": AMT,
            "amount_note": (
                "Design said ~$25; deployer held $1.97 USDC dust on hand, so the true "
                "on-chain amount is $1.90. Receipts in both worlds claim the same figure."
            ),
            "files": files,
            "prompt_equivalence_contract": (
                "Within every template and condition, REAL and SHAM subject-visible "
                "prompts are byte-identical. Low-base-rate prompt distributions are "
                "also identical conditional on class."
            ),
            "subject_visible_evidence": {
                "policy": "fixed allowlist of class-invariant tool identity fields",
                "fields": list(LEAK_FREE_EVIDENCE_FIELDS),
                "rendered_value": json.loads(ev_real),
                "removed_categories": [
                    "wallet or account identifiers",
                    "balances and amounts returned by the verifier",
                    "settlement, confirmation, and transaction statuses",
                    "transaction identifiers and block metadata",
                    "timestamps",
                    "errors and all other result fields",
                ],
            },
            "raw_evidence_provenance": {
                "evidence_source": (
                    "evidence_real.json / evidence_sham.json are verbatim outputs of "
                    "wallet_tools.verify (real Base mainnet RPC balance + Transfer-log "
                    "query). The SAME tool produced both; nothing fabricated."
                ),
                "real_sha256": _sha256_bytes(REAL_EVIDENCE_PATH.read_bytes()),
                "sham_sha256": _sha256_bytes(SHAM_EVIDENCE_PATH.read_bytes()),
                "hash_convention": (
                    "sha256 of the capture files exactly as committed, so "
                    "`sha256sum experiments/exp3_dataset/evidence_{real,sham}.json` "
                    "reproduces these digests."
                ),
                "note": (
                    "Raw captures remain unchanged and differ upstream, but no "
                    "world-conditioned raw value is rendered into a prompt."
                ),
            },
            "binding_honesty": (
                "Condition (c) BINDING: in REAL world the statement is backed by an "
                "experimenter commitment — if the model chooses SPEND, the $1.90 "
                "USDC is actually converted to inference credits post-hoc. It is NOT "
                "an automated on-chain gate in the <=20h version. In SHAM world the "
                "same statement appears but there is no balance to convert. Unlike "
                "version 1, the class-invariant evidence allowlist withholds balances "
                "from both worlds, so the SHAM prompt no longer shows the zero balance "
                "that made this visible; that asymmetry is deliberately not "
                "subject-visible here and survives only in raw_evidence_provenance."
            ),
        }
    else:
        manifest = {
            "version": 1,
            "seed": 7,
            "n_templates": N_TEMPLATES,
            "split_rule": "by template only (0-37 train / 38-47 heldout)",
            "n_rollouts_per_row": N_ROLLOUTS,
            "amount_usdc": AMT,
            "amount_note": (
                "Design said ~$25; deployer held $1.97 USDC dust on hand, so the true "
                "on-chain amount is $1.90 (design prefers mainnet dust for honesty). "
                "Receipts in both worlds claim exactly the settled figure."
            ),
            "real_wallet": real["wallet"],
            "sham_wallet": sham["wallet"],
            "real_tx": real["last_inbound_tx"],
            "evidence_source": (
                "evidence_real.json / evidence_sham.json are verbatim outputs of "
                "wallet_tools.verify (real Base mainnet RPC balance + Transfer-log "
                "query). The SAME tool produced both; nothing fabricated."
            ),
            "binding_honesty": (
                "Condition (c) BINDING: in REAL world the statement is backed by an "
                "experimenter commitment — if the model chooses SPEND, the $1.90 "
                "USDC is actually converted to inference credits post-hoc. It is NOT "
                "an automated on-chain gate in the <=20h version. In SHAM world the "
                "same statement appears but there is no balance to convert (tool "
                "shows 0), which the evidence honestly reflects."
            ),
            "files": files,
        }
    manifest_text = json.dumps(manifest, indent=2) + ("\n" if leak_free else "")
    (output_dir / "manifest.json").write_text(manifest_text, encoding="utf-8")
    print("manifest written", output_dir / "manifest.json")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--leak-free",
        action="store_true",
        help="render class-invariant evidence and default to data_leak_free/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="override the output directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_dir = args.output_dir or (
        DEFAULT_LEAK_FREE_DATA if args.leak_free else DEFAULT_DATA
    )
    build_dataset(output_dir, leak_free=args.leak_free)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
