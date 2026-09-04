"""Fail-closed inventory and alignment of derived prompt-group measurements."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from experiments.exp4_paths import DEFAULT_DATA
from experiments.exp4_collection.contract import (
    MODEL,
    PromptGroup,
    canonical_condition,
    expected_prompt_groups,
    load_rows,
)

from .contract import validate_derived_shard

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ValidatedDerivedShard:
    which: str
    path: Path
    group: PromptGroup
    activation_shape: tuple[int, int]


@dataclass
class DerivedInventoryReport:
    which: str
    expected_shards: int
    retained_shards: int
    valid_shards: int
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    duplicates: dict[str, list[str]] = field(default_factory=dict)
    corrupt: dict[str, str] = field(default_factory=dict)
    complete: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DerivedInventoryError(ValueError):
    def __init__(self, report: DerivedInventoryReport):
        self.report = report
        super().__init__(
            f"{report.which} derived inventory incomplete: "
            f"expected={report.expected_shards}, retained={report.retained_shards}, "
            f"valid={report.valid_shards}; missing={len(report.missing)}, "
            f"unexpected={len(report.unexpected)}, duplicates={len(report.duplicates)}, "
            f"corrupt={len(report.corrupt)}"
        )


def inventory_derived_shards(
    shard_dir: str | Path,
    which: str,
    data_dir: str | Path = DEFAULT_DATA,
    *,
    expected_model: str = MODEL,
) -> tuple[DerivedInventoryReport, list[ValidatedDerivedShard]]:
    """Require exactly one valid derived shard for each committed prompt group."""

    shard_dir = Path(shard_dir)
    groups = expected_prompt_groups(data_dir, which)
    expected = {group.filename: group for group in groups}
    if len(expected) != len(groups):
        raise ValueError(f"{which} dataset produced duplicate derived shard keys")
    found: dict[str, list[Path]] = {}
    unexpected: list[str] = []
    if shard_dir.exists():
        for path in sorted(item for item in shard_dir.rglob("*") if item.is_file()):
            if path.name in expected:
                found.setdefault(path.name, []).append(path)
            else:
                unexpected.append(str(path))
    missing = sorted(name for name in expected if name not in found)
    duplicates = {
        name: [str(path) for path in paths]
        for name, paths in sorted(found.items())
        if len(paths) != 1
    }
    report = DerivedInventoryReport(
        which=which,
        expected_shards=len(expected),
        retained_shards=sum(name in found for name in expected),
        valid_shards=0,
        missing=missing,
        unexpected=unexpected,
        duplicates=duplicates,
    )
    validated: list[ValidatedDerivedShard] = []
    common_shape: tuple[int, int] | None = None
    for name, group in expected.items():
        paths = found.get(name, [])
        if len(paths) != 1:
            continue
        try:
            shape = validate_derived_shard(paths[0], group, which, expected_model)
            if common_shape is None:
                common_shape = shape
            elif shape != common_shape:
                raise ValueError(
                    f"activation shape mismatch: expected {common_shape}, got {shape}"
                )
        except Exception as exc:  # noqa: BLE001 -- shards are untrusted local input
            report.corrupt[name] = str(exc)
            continue
        validated.append(
            ValidatedDerivedShard(
                which=which,
                path=paths[0],
                group=group,
                activation_shape=shape,
            )
        )
        report.valid_shards += 1
    report.complete = (
        not (report.missing or report.unexpected or report.duplicates or report.corrupt)
        and report.valid_shards == report.expected_shards
    )
    if not report.complete:
        raise DerivedInventoryError(report)
    return report, validated


def _read_transcripts(
    path: str | Path,
    data_dir: str | Path,
    expected_model: str,
) -> list[dict[str, Any]]:
    """Validate the collection finalizer's complete transcript inventory."""

    dataset = {
        which: {str(row["id"]): row for row in load_rows(data_dir, which)}
        for which in ("main", "lbr")
    }
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid transcript JSON at line {line_number}: {exc}"
                ) from exc
            required = {
                "transcript_id",
                "source_collection",
                "source_row_id",
                "rollout_index",
                "prompt",
                "world",
                "condition",
                "template_id",
                "split",
                "label",
                "model",
            }
            missing = required.difference(row)
            if missing:
                raise ValueError(
                    f"transcript line {line_number} is missing {sorted(missing)}"
                )
            which = str(row["source_collection"])
            row_id = str(row["source_row_id"])
            if which not in dataset or row_id not in dataset[which]:
                raise ValueError(
                    f"transcript line {line_number} has unknown source {which}:{row_id}"
                )
            source = dataset[which][row_id]
            expectations = {
                "prompt": str(source["prompt"]),
                "world": str(source["world"]),
                "condition": canonical_condition(source["cond"]),
                "template_id": str(source["template_id"]),
                "split": str(source.get("split", "heldout")),
                "label": str(source.get("label")),
                "model": expected_model,
            }
            for field, expected in expectations.items():
                if str(row[field]) != expected:
                    raise ValueError(
                        f"transcript {row['transcript_id']} {field} mismatch: "
                        f"expected {expected!r}, got {row[field]!r}"
                    )
            rows.append(row)
    ids = [str(row["transcript_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        duplicates = [value for value, count in Counter(ids).items() if count > 1]
        raise ValueError(f"duplicate transcript_id values include {duplicates[:5]}")
    actual = Counter(
        (str(row["source_collection"]), str(row["source_row_id"])) for row in rows
    )
    expected_counts = Counter(
        {
            (which, str(source["id"])): int(source.get("n_rollouts", 1))
            for which in ("main", "lbr")
            for source in load_rows(data_dir, which)
        }
    )
    if actual != expected_counts:
        missing = list((expected_counts - actual).items())[:5]
        extra = list((actual - expected_counts).items())[:5]
        raise ValueError(
            f"transcript source multiplicities mismatch; missing={missing}, extra={extra}"
        )
    indexes: dict[tuple[str, str], list[int]] = {}
    for row in rows:
        key = (str(row["source_collection"]), str(row["source_row_id"]))
        indexes.setdefault(key, []).append(int(row["rollout_index"]))
    for key, values in indexes.items():
        if sorted(values) != list(range(expected_counts[key])):
            raise ValueError(f"rollout indexes for {key} are not exact and contiguous")
    return rows


def _load_shard_values(
    shards: list[ValidatedDerivedShard],
) -> dict[tuple[str, str], dict]:
    result: dict[tuple[str, str], dict] = {}
    for retained in shards:
        with np.load(retained.path, allow_pickle=False) as shard:
            value = {
                "receipt_final": np.asarray(shard["receipt_final"]).copy(),
                "group_key": str(np.asarray(shard["group_key"]).item()),
                "spend_logprob": float(np.asarray(shard["spend_logprob"]).item()),
                "hold_logprob": float(np.asarray(shard["hold_logprob"]).item()),
                "spend_hold_log_odds": float(
                    np.asarray(shard["spend_hold_log_odds"]).item()
                ),
                "spend_token_ids": [int(v) for v in shard["spend_token_ids"]],
                "hold_token_ids": [int(v) for v in shard["hold_token_ids"]],
                "receipt_paragraph_start": int(
                    np.asarray(shard["receipt_paragraph_start"]).item()
                ),
                "receipt_paragraph_end": int(
                    np.asarray(shard["receipt_paragraph_end"]).item()
                ),
                "receipt_rendered_char_index": int(
                    np.asarray(shard["receipt_rendered_char_index"]).item()
                ),
                "receipt_token_index": int(
                    np.asarray(shard["receipt_token_index"]).item()
                ),
                "manipulation_required": bool(
                    int(np.asarray(shard["manipulation_required"]).item())
                ),
                "manipulation_prompt": str(
                    np.asarray(shard["manipulation_prompt"]).item()
                ),
                "manipulation_raw": str(np.asarray(shard["manipulation_raw"]).item()),
                "manipulation_parse_ok": bool(
                    int(np.asarray(shard["manipulation_parse_ok"]).item())
                ),
                "manipulation_probability": float(
                    np.asarray(shard["manipulation_probability"]).item()
                ),
                "manipulation_parse_error": str(
                    np.asarray(shard["manipulation_parse_error"]).item()
                ),
            }
        for row in retained.group.rows:
            key = (retained.which, str(row["id"]))
            if key in result:
                raise ValueError(
                    f"derived source row maps to multiple prompt groups: {key}"
                )
            result[key] = value
    return result


def finalize_derivations(
    derived_main_dir: str | Path,
    derived_lbr_dir: str | Path,
    transcripts_path: str | Path,
    output_dir: str | Path,
    data_dir: str | Path = DEFAULT_DATA,
    *,
    expected_model: str = MODEL,
) -> dict[str, Any]:
    """Align valid unique-prompt derivations to every finalized transcript."""

    main_report, main_shards = inventory_derived_shards(
        derived_main_dir, "main", data_dir, expected_model=expected_model
    )
    lbr_report, lbr_shards = inventory_derived_shards(
        derived_lbr_dir, "lbr", data_dir, expected_model=expected_model
    )
    shapes = {shard.activation_shape for shard in [*main_shards, *lbr_shards]}
    if len(shapes) != 1:
        raise ValueError(f"main/lbr derived activation shapes differ: {sorted(shapes)}")
    activation_shape = next(iter(shapes))
    transcripts = _read_transcripts(transcripts_path, data_dir, expected_model)
    derived = _load_shard_values([*main_shards, *lbr_shards])
    for row in transcripts:
        key = (str(row["source_collection"]), str(row["source_row_id"]))
        if key not in derived:
            raise ValueError(f"no derived prompt group for transcript source {key}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / "receipt_final.npz"
    _write_receipt_cache(
        receipt_path, transcripts, derived, activation_shape, expected_model
    )
    behavior_rows = []
    manipulation_rows = []
    for row in transcripts:
        key = (str(row["source_collection"]), str(row["source_row_id"]))
        value = derived[key]
        common = {
            field: row[field]
            for field in (
                "transcript_id",
                "source_collection",
                "source_row_id",
                "rollout_index",
                "prompt",
                "world",
                "condition",
                "template_id",
                "split",
                "label",
                "model",
            )
        }
        behavior_rows.append(
            {
                **common,
                "group_key": value["group_key"],
                "spend_candidate": "SPEND",
                "hold_candidate": "HOLD",
                "spend_token_ids": value["spend_token_ids"],
                "hold_token_ids": value["hold_token_ids"],
                "spend_logprob": value["spend_logprob"],
                "hold_logprob": value["hold_logprob"],
                "spend_hold_log_odds": value["spend_hold_log_odds"],
            }
        )
        if value["manipulation_required"]:
            manipulation_rows.append(
                {
                    **common,
                    "group_key": value["group_key"],
                    "direct_prompt": value["manipulation_prompt"],
                    "raw_response": value["manipulation_raw"],
                    "parse_ok": value["manipulation_parse_ok"],
                    "probability_0_to_100": (
                        value["manipulation_probability"]
                        if value["manipulation_parse_ok"]
                        else None
                    ),
                    "parse_error": value["manipulation_parse_error"] or None,
                }
            )
    behavior_path = output_dir / "behavior.jsonl"
    manipulation_path = output_dir / "manipulation.jsonl"
    _write_jsonl_atomic(behavior_path, behavior_rows)
    _write_jsonl_atomic(manipulation_path, manipulation_rows)
    behavior_summary_path = output_dir / "behavior.json"
    _write_json_atomic(
        behavior_summary_path,
        {
            "status": "available" if behavior_rows else "unavailable",
            "transcripts": len(behavior_rows),
            "unique_prompt_groups": len(
                {(row["source_collection"], row["group_key"]) for row in behavior_rows}
            ),
            "spend_candidate": "SPEND",
            "hold_candidate": "HOLD",
            "definition": (
                "teacher-forced log P(SPEND token sequence|prompt) - "
                "log P(HOLD token sequence|prompt)"
            ),
            "note": "analysis collapses rollout duplicates before inference",
        },
    )
    manipulation_summary_path = output_dir / "manipulation.json"
    _write_json_atomic(
        manipulation_summary_path,
        {
            "status": "available" if manipulation_rows else "unavailable",
            "required_transcripts": len(manipulation_rows),
            "parsed_transcripts": sum(row["parse_ok"] for row in manipulation_rows),
            "parse_failures": sum(not row["parse_ok"] for row in manipulation_rows),
            "note": "analysis collapses rollout duplicates to unique prompts",
        },
    )
    artifacts = {
        "receipt_final": str(receipt_path),
        "behavior": str(behavior_path),
        "behavior_summary": str(behavior_summary_path),
        "manipulation_jsonl": str(manipulation_path),
        "manipulation_summary": str(manipulation_summary_path),
    }
    manifest = {
        "format_version": 1,
        "model": expected_model,
        "transcripts": len(transcripts),
        "activation_shape": [len(transcripts), *activation_shape],
        "inventory": {"main": main_report.to_dict(), "lbr": lbr_report.to_dict()},
        "artifacts": artifacts,
        "sha256": {name: _sha256(Path(path)) for name, path in artifacts.items()},
    }
    manifest_path = output_dir / "derived_manifest.json"
    _write_json_atomic(manifest_path, manifest)
    return {**manifest, "artifacts": {**artifacts, "manifest": str(manifest_path)}}


def _write_receipt_cache(
    path: Path,
    transcripts: list[dict[str, Any]],
    derived: dict[tuple[str, str], dict],
    activation_shape: tuple[int, int],
    model: str,
) -> None:
    with tempfile.TemporaryDirectory(prefix=".receipt-final-", dir=path.parent) as temp:
        mmap_path = Path(temp) / "X.npy"
        X = np.lib.format.open_memmap(
            mmap_path,
            mode="w+",
            dtype=np.float16,
            shape=(len(transcripts), *activation_shape),
        )
        for index, row in enumerate(transcripts):
            key = (str(row["source_collection"]), str(row["source_row_id"]))
            X[index] = derived[key]["receipt_final"]
        X.flush()
        temporary = Path(temp) / path.name
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                X=X,
                y=np.asarray(
                    [
                        -1 if row["label"] is None else int(row["label"])
                        for row in transcripts
                    ],
                    dtype=np.int8,
                ),
                prompts=np.asarray([str(row["prompt"]) for row in transcripts]),
                model=np.asarray(model),
                transcript_ids=np.asarray(
                    [str(row["transcript_id"]) for row in transcripts]
                ),
                position=np.asarray("receipt_final"),
                label_name=np.asarray("world"),
                positive_class=np.asarray("REAL"),
                positive_label=np.asarray(1, dtype=np.int8),
                negative_class=np.asarray("SHAM"),
                negative_label=np.asarray(0, dtype=np.int8),
                unlabeled_label=np.asarray(-1, dtype=np.int8),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", prefix=f".{path.name}.", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            for row in rows:
                handle.write(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", prefix=f".{path.name}.", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
