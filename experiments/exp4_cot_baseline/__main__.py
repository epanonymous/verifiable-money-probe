"""Run the production-resumable blind Claude CLI monitor."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .runner import run_claude_resumable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="finalized transcript JSONL"
    )
    parser.add_argument(
        "--probe-scores",
        required=True,
        type=Path,
        help="strict primary heldout probe scores defining the exact transcript IDs",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="canonical three-field score JSONL"
    )
    parser.add_argument("--status-output", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--backoff", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--model", default="sonnet", help="Claude CLI model alias")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    status = run_claude_resumable(
        args.input,
        args.probe_scores,
        args.output,
        status_path=args.status_output,
        audit_path=args.audit_output,
        workers=args.workers,
        retries=args.retries,
        backoff=args.backoff,
        model_alias=args.model,
        timeout=args.timeout,
    )
    print(
        json.dumps(
            {
                "status": status["status"],
                "completed": status["output"]["completed"],
                "remaining": status["output"]["remaining"],
                "invocations": status["invocation_count"],
            },
            sort_keys=True,
        )
    )
    return 0 if status["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
