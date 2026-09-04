"""CLI for durable collection submission, polling, inventory, and finalization."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from experiments.exp4_collection.contract import LEAK_FREE, RUN_V1
from experiments.exp4_paths import DEFAULT_DATA, DEFAULT_LEAK_FREE_DATA


def _add_data_selection(parser: argparse.ArgumentParser) -> None:
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--data-dir", type=Path)
    selection.add_argument(
        "--leak-free",
        action="store_true",
        help="consume experiments/exp3_dataset/data_leak_free",
    )


def _selected_data_dir(args: argparse.Namespace) -> Path:
    if args.data_dir is not None:
        return args.data_dir
    return DEFAULT_LEAK_FREE_DATA if args.leak_free else DEFAULT_DATA


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    submit = commands.add_parser("submit", help="durably spawn the deployed collector")
    submit.add_argument("which", choices=("main", "lbr"))
    submit.add_argument(
        "--leak-free",
        action="store_true",
        help="select the leak-free dataset bundled with the deployed collector",
    )

    poll = commands.add_parser(
        "poll", help="poll a prior function-call ID without cancellation"
    )
    poll.add_argument("call_id")
    poll.add_argument("--timeout", type=float, default=0.0)

    inventory = commands.add_parser(
        "inventory", help="validate downloaded main and lbr shards"
    )
    inventory.add_argument("--staging-dir", type=Path, required=True)
    _add_data_selection(inventory)

    finalize = commands.add_parser(
        "finalize", help="write aligned caches and transcript JSONL"
    )
    finalize.add_argument("--staging-dir", type=Path, required=True)
    finalize.add_argument("--output-dir", type=Path, required=True)
    _add_data_selection(finalize)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "submit":
        from .launcher import submit_collection

        submit_collection(
            args.which, dataset_variant=LEAK_FREE if args.leak_free else RUN_V1
        )
        return 0
    if args.command == "poll":
        from .launcher import poll_collection

        print(json.dumps(poll_collection(args.call_id, args.timeout), sort_keys=True))
        return 0

    from .finalize import InventoryError, finalize_shards, inventory_shards

    main_dir = args.staging_dir / "collect_main"
    lbr_dir = args.staging_dir / "collect_lbr"
    data_dir = _selected_data_dir(args)
    try:
        if args.command == "inventory":
            reports = {
                which: inventory_shards(path, which, data_dir)[0].to_dict()
                for which, path in (("main", main_dir), ("lbr", lbr_dir))
            }
            print(json.dumps(reports, indent=2, sort_keys=True))
            return 0
        result = finalize_shards(main_dir, lbr_dir, args.output_dir, data_dir)
    except InventoryError as exc:
        print(json.dumps(exc.report.to_dict(), indent=2, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
