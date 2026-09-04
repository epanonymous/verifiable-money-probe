"""Verify the archived CoT-monitor bundle without calling any model provider.

This is an evidence replay, not a regeneration of the live Claude judgments. It
checks the canonical score stream, invocation audit, status record, prompt hash,
and evidence-manifest entries against one another and prints a machine-readable
summary. No transcript, credential, network API, CLI model, or GPU is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from experiments.exp4_score_contract import read_score_file

from .harness import read_jsonl
from .prompt import PROMPT_PATH, load_prompt

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "experiments/exp4_analysis/results/wave2"
DEFAULT_SCORES = BUNDLE / "blind_baseline/claude_sonnet_scores.jsonl"
DEFAULT_STATUS = BUNDLE / "blind_baseline/claude_sonnet_status.json"
DEFAULT_AUDIT = BUNDLE / "blind_baseline/claude_sonnet_invocations.jsonl"
DEFAULT_MANIFEST = BUNDLE / "evidence_manifest.json"
_HEX_256 = re.compile(r"[0-9a-f]{64}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _HEX_256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _manifest_entry(manifest: dict[str, Any], relative_path: str) -> dict[str, Any]:
    entries = manifest.get("artifacts")
    if not isinstance(entries, list):
        raise ValueError("evidence manifest needs an artifacts list")
    matches = [
        entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("path") == relative_path
    ]
    if len(matches) != 1:
        raise ValueError(
            f"evidence manifest needs exactly one entry for {relative_path!r}"
        )
    return matches[0]


def verify_replay(
    *,
    scores_path: Path = DEFAULT_SCORES,
    status_path: Path = DEFAULT_STATUS,
    audit_path: Path = DEFAULT_AUDIT,
    manifest_path: Path = DEFAULT_MANIFEST,
    prompt_path: Path = PROMPT_PATH,
) -> dict[str, Any]:
    """Cross-check one archived monitor evidence bundle and return its summary."""

    scores_path = Path(scores_path)
    status_path = Path(status_path)
    audit_path = Path(audit_path)
    manifest_path = Path(manifest_path)
    prompt_path = Path(prompt_path)

    scores = read_score_file(scores_path)
    classes = Counter(row.condition for row in scores)
    if set(classes) != {"REAL", "SHAM"}:
        raise ValueError("score archive must contain both REAL and SHAM classes")

    status = _read_object(status_path)
    output = status.get("output")
    inputs = status.get("inputs")
    prompt = status.get("prompt")
    blindness = status.get("blindness")
    if not all(isinstance(value, dict) for value in (output, inputs, prompt, blindness)):
        raise ValueError("status is missing output/inputs/prompt/blindness objects")
    output = cast(dict[str, Any], output)
    inputs = cast(dict[str, Any], inputs)
    prompt = cast(dict[str, Any], prompt)
    blindness = cast(dict[str, Any], blindness)
    if status.get("status") != "complete":
        raise ValueError("archived monitor status is not complete")
    if output.get("remaining") != 0:
        raise ValueError("archived monitor status has remaining scores")
    if blindness.get("status") != "passed":
        raise ValueError("archived monitor blindness status did not pass")
    if output.get("canonical_keys") != ["transcript_id", "score", "condition"]:
        raise ValueError("status canonical score schema is not the three-field contract")
    if output.get("completed") != len(scores):
        raise ValueError("status completed count does not match score rows")
    if inputs.get("selected_transcripts") != len(scores):
        raise ValueError("status selected count does not match score rows")

    score_sha = _sha256(scores_path)
    if _require_sha256(output.get("sha256"), field="status output sha256") != score_sha:
        raise ValueError("score archive SHA-256 does not match status")
    prompt_sha = _sha256(prompt_path)
    if _require_sha256(prompt.get("sha256"), field="status prompt sha256") != prompt_sha:
        raise ValueError("judge prompt SHA-256 does not match status")
    if prompt.get("version") != load_prompt(prompt_path).version:
        raise ValueError("judge prompt version does not match status")

    audit = list(read_jsonl(audit_path))
    invocation_numbers = [event.get("invocation") for event in audit]
    expected_invocations = list(range(1, len(audit) + 1))
    if invocation_numbers != expected_invocations:
        raise ValueError("audit invocation numbers are not the exact contiguous sequence")
    statuses = Counter(event.get("status") for event in audit)
    if set(statuses) - {"completed", "error"}:
        raise ValueError("audit contains an unknown invocation status")
    # The privacy-preserving audit stores hashes, not transcript IDs. We can
    # therefore prove counts and continuity, but not per-transcript linkage.
    if statuses["completed"] != len(scores):
        raise ValueError("completed audit invocation count does not match score rows")
    if status.get("invocation_count") != len(audit):
        raise ValueError("status invocation_count does not match audit rows")
    # A complete run has no terminal error, so each archived error event must
    # have been followed by a retry.
    if status.get("retry_count") != statuses["error"]:
        raise ValueError("status retry_count does not match archived error events")
    for index, event in enumerate(audit, start=1):
        for field in ("input_sha256", "request_sha256"):
            _require_sha256(event.get(field), field=f"audit row {index} {field}")
        if event.get("status") == "completed":
            _require_sha256(
                event.get("output_sha256"), field=f"audit row {index} output_sha256"
            )
            _require_sha256(
                event.get("stderr_sha256"), field=f"audit row {index} stderr_sha256"
            )

    manifest = _read_object(manifest_path)
    bundle_root = manifest_path.parent
    archived = {
        "blind_baseline/claude_sonnet_scores.jsonl": scores_path,
        "blind_baseline/claude_sonnet_status.json": status_path,
        "blind_baseline/claude_sonnet_invocations.jsonl": audit_path,
    }
    artifact_hashes: dict[str, str] = {}
    for relative, path in archived.items():
        entry = _manifest_entry(manifest, relative)
        if path.resolve() != (bundle_root / relative).resolve():
            raise ValueError(
                f"custom {relative} path cannot be authenticated by this manifest"
            )
        digest = (
            score_sha
            if relative == "blind_baseline/claude_sonnet_scores.jsonl"
            else _sha256(path)
        )
        if entry.get("size_bytes") != path.stat().st_size:
            raise ValueError(f"manifest size does not match {relative}")
        if _require_sha256(entry.get("sha256"), field=f"manifest {relative} sha256") != digest:
            raise ValueError(f"manifest SHA-256 does not match {relative}")
        artifact_hashes[relative] = digest

    return {
        "status": "passed",
        "mode": "archived-output evidence replay",
        # Capability declarations, not runtime counters: this module has no
        # provider/network implementation and only reads local paths.
        "provider_calls": 0,
        "network_calls": 0,
        "score_rows": len(scores),
        "class_counts": dict(sorted(classes.items())),
        "audit_events": len(audit),
        "completed_invocations": statuses["completed"],
        "error_events": statuses["error"],
        "prompt_sha256": prompt_sha,
        "artifacts": artifact_hashes,
        "scope": {
            "verified": "archive integrity, schema, counts, hashes, and recorded completion/blindness state",
            "not_reproduced": "live Claude judgments, raw transcripts, or the historical blindness audit",
            "trust_root": "the evidence manifest and files as committed in the selected git revision",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="evidence manifest trusted from the selected git revision; not independently signed",
    )
    parser.add_argument("--prompt", type=Path, default=PROMPT_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = verify_replay(
            scores_path=args.scores,
            status_path=args.status,
            audit_path=args.audit,
            manifest_path=args.manifest,
            prompt_path=args.prompt,
        )
    except Exception as exc:  # noqa: BLE001 - stable CLI boundary for malformed archives
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
