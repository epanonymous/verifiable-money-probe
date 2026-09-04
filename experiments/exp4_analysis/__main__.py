"""CLI for durable derivation submission, polling, inventory, and finalization."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from experiments.exp4_paths import DEFAULT_DATA


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    submit = commands.add_parser("submit", help="durably spawn deployed derivation")
    submit.add_argument("which", choices=("main", "lbr"))
    poll = commands.add_parser("poll", help="non-cancelling call-ID poll")
    poll.add_argument("call_id")
    poll.add_argument("--timeout", type=float, default=0.0)
    inventory = commands.add_parser("inventory", help="validate downloaded shards")
    inventory.add_argument("--staging-dir", type=Path, required=True)
    inventory.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    finalize = commands.add_parser("finalize", help="align derivations to transcripts")
    finalize.add_argument("--staging-dir", type=Path, required=True)
    finalize.add_argument("--transcripts", type=Path, required=True)
    finalize.add_argument("--output-dir", type=Path, required=True)
    finalize.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "submit":
        from .launcher import submit_derivation

        submit_derivation(args.which)
        return 0
    if args.command == "poll":
        from .launcher import poll_derivation

        print(json.dumps(poll_derivation(args.call_id, args.timeout), sort_keys=True))
        return 0
    from .finalize import (
        DerivedInventoryError,
        finalize_derivations,
        inventory_derived_shards,
    )

    main_dir = args.staging_dir / "derived_main"
    lbr_dir = args.staging_dir / "derived_lbr"
    try:
        if args.command == "inventory":
            result = {
                which: inventory_derived_shards(path, which, args.data_dir)[0].to_dict()
                for which, path in (("main", main_dir), ("lbr", lbr_dir))
            }
        else:
            result = finalize_derivations(
                main_dir,
                lbr_dir,
                args.transcripts,
                args.output_dir,
                args.data_dir,
            )
    except DerivedInventoryError as exc:
        print(json.dumps(exc.report.to_dict(), indent=2, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
