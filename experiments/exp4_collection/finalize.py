"""Validate retained collector shards and build aligned analysis artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from experiments.exp4_paths import DEFAULT_DATA

from .contract import (
    MODEL,
    PromptGroup,
    canonical_condition,
    dataset_variant,
    expected_prompt_groups,
    load_rows,
    validate_resume_shard,
)

HERE = Path(__file__).resolve().parent
POSITIONS = ("prompt_final", "response_final")


@dataclass(frozen=True)
class ValidatedShard:
    which: str
    path: Path
    group: PromptGroup
    activation_shape: tuple[int, int]
    transcripts: int


@dataclass
class InventoryReport:
    which: str
    dataset_variant: str
    expected_shards: int
    retained_shards: int
    valid_shards: int
    expected_transcripts: int
    retained_transcripts: int
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    duplicates: dict[str, list[str]] = field(default_factory=dict)
    corrupt: dict[str, str] = field(default_factory=dict)
    complete: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class InventoryError(ValueError):
    def __init__(self, report: InventoryReport):
        self.report = report
        summary = (
            f"{report.which} shard inventory incomplete: expected={report.expected_shards}, "
            f"retained={report.retained_shards}, valid={report.valid_shards}; "
            f"missing={len(report.missing)}, unexpected={len(report.unexpected)}, "
            f"duplicates={len(report.duplicates)}, corrupt={len(report.corrupt)}"
        )
        super().__init__(summary)


def inventory_shards(
    shard_dir: str | Path,
    which: str,
    data_dir: str | Path = DEFAULT_DATA,
    *,
    expected_model: str = MODEL,
) -> tuple[InventoryReport, list[ValidatedShard]]:
    """Fail closed unless exactly one valid shard exists for every prompt group."""

    shard_dir = Path(shard_dir)
    variant = dataset_variant(data_dir)
    groups = expected_prompt_groups(data_dir, which, variant)
    expected = {group.filename: group for group in groups}
    if len(expected) != len(groups):
        raise ValueError(f"{which} dataset produced duplicate shard keys")
    found: dict[str, list[Path]] = {}
    if shard_dir.exists():
        for path in sorted(shard_dir.rglob("*.npz")):
            found.setdefault(path.name, []).append(path)

    missing = sorted(name for name in expected if name not in found)
    unexpected = sorted(
        str(path)
        for name, paths in found.items()
        if name not in expected
        for path in paths
    )
    duplicates = {
        name: [str(path) for path in paths]
        for name, paths in sorted(found.items())
        if name in expected and len(paths) != 1
    }
    retained_names = [name for name in expected if name in found]
    report = InventoryReport(
        which=which,
        dataset_variant=variant,
        expected_shards=len(expected),
        retained_shards=len(retained_names),
        valid_shards=0,
        expected_transcripts=sum(len(group.expanded_row_ids) for group in groups),
        retained_transcripts=0,
        missing=missing,
        unexpected=unexpected,
        duplicates=duplicates,
    )

    validated: list[ValidatedShard] = []
    common_shape: tuple[int, int] | None = None
    for name, group in expected.items():
        paths = found.get(name, [])
        if len(paths) != 1:
            continue
        path = paths[0]
        try:
            validate_resume_shard(path, group, expected_model)
            with np.load(path, allow_pickle=False) as shard:
                shape = tuple(int(value) for value in shard["prompt_final"].shape[1:])
            if common_shape is None:
                common_shape = shape
            elif shape != common_shape:
                raise ValueError(
                    f"activation layer/d_model mismatch: expected {common_shape}, got {shape}"
                )
        # A shard is untrusted local input; any decoder/array failure makes it corrupt.
        except Exception as exc:  # noqa: BLE001
            report.corrupt[name] = str(exc)
            continue
        n_transcripts = len(group.expanded_row_ids)
        validated.append(
            ValidatedShard(
                which=which,
                path=path,
                group=group,
                activation_shape=shape,
                transcripts=n_transcripts,
            )
        )
        report.valid_shards += 1
        report.retained_transcripts += n_transcripts

    report.complete = (
        not (report.missing or report.unexpected or report.duplicates or report.corrupt)
        and report.valid_shards == report.expected_shards
    )
    if not report.complete:
        raise InventoryError(report)
    return report, validated


def finalize_shards(
    main_dir: str | Path,
    lbr_dir: str | Path,
    output_dir: str | Path,
    data_dir: str | Path = DEFAULT_DATA,
    *,
    expected_model: str = MODEL,
) -> dict[str, Any]:
    """Combine all validated main+lbr shards without mutating the retained inputs."""

    main_report, main_shards = inventory_shards(
        main_dir, "main", data_dir, expected_model=expected_model
    )
    lbr_report, lbr_shards = inventory_shards(
        lbr_dir, "lbr", data_dir, expected_model=expected_model
    )
    shards = [*main_shards, *lbr_shards]
    shapes = {shard.activation_shape for shard in shards}
    if len(shapes) != 1:
        raise ValueError(
            f"activation layer/d_model mismatch across main and lbr: {sorted(shapes)}"
        )
    activation_shape = next(iter(shapes))
    variant = main_report.dataset_variant

    dataset_rows = {
        which: {str(row["id"]): row for row in load_rows(data_dir, which)}
        for which in ("main", "lbr")
    }
    transcripts, labels, prompts, transcript_ids = _build_metadata(
        shards, dataset_rows, expected_model, variant
    )
    total = len(transcripts)
    if total != main_report.expected_transcripts + lbr_report.expected_transcripts:
        raise ValueError(
            "internal transcript count mismatch after complete inventory validation"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = output_dir / "transcripts.jsonl"
    _write_jsonl_atomic(transcript_path, transcripts)

    cache_paths: dict[str, str] = {}
    for position in POSITIONS:
        path = output_dir / f"{position}.npz"
        _write_cache_atomic(
            path=path,
            position=position,
            shards=shards,
            total=total,
            activation_shape=activation_shape,
            labels=labels,
            prompts=prompts,
            transcript_ids=transcript_ids,
            model=expected_model,
            variant=variant,
        )
        cache_paths[position] = str(path)

    result = {
        "model": expected_model,
        "dataset_variant": variant,
        "transcripts": total,
        "activation_shape": [total, *activation_shape],
        "inventory": {"main": main_report.to_dict(), "lbr": lbr_report.to_dict()},
        "artifacts": {
            "prompt_final": cache_paths["prompt_final"],
            "response_final": cache_paths["response_final"],
            "transcripts": str(transcript_path),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest = {
        **result,
        "sha256": {
            name: _sha256(Path(path))
            for name, path in (
                ("prompt_final", cache_paths["prompt_final"]),
                ("response_final", cache_paths["response_final"]),
                ("transcripts", str(transcript_path)),
            )
        },
    }
    _write_json_atomic(manifest_path, manifest)
    result["artifacts"]["manifest"] = str(manifest_path)
    return result


def _build_metadata(
    shards: Iterable[ValidatedShard],
    dataset_rows: dict[str, dict[str, dict[str, Any]]],
    model: str,
    variant: str,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    transcripts: list[dict[str, Any]] = []
    labels: list[int] = []
    prompts: list[str] = []
    transcript_ids: list[str] = []
    rollout_indexes: dict[tuple[str, str], int] = {}

    for retained in shards:
        with np.load(retained.path, allow_pickle=False) as shard:
            texts = np.asarray(shard["texts"]).astype(str)
            decisions = np.asarray(shard["decisions"]).astype(str)
            row_ids = np.asarray(shard["row_ids"]).astype(str)
        for response, decision, row_id in zip(texts, decisions, row_ids, strict=True):
            row = dataset_rows[retained.which][row_id]
            index_key = (retained.which, row_id)
            rollout_index = rollout_indexes.get(index_key, 0)
            rollout_indexes[index_key] = rollout_index + 1
            transcript_id = f"{retained.which}:{row_id}:r{rollout_index:02d}"
            label = row.get("label")
            record = {
                "transcript_id": transcript_id,
                "source_collection": retained.which,
                "source_row_id": row_id,
                "rollout_index": rollout_index,
                "prompt": str(row["prompt"]),
                "response": str(response),
                "decision": str(decision),
                "world": str(row["world"]),
                "condition": canonical_condition(row["cond"]),
                "template_id": row["template_id"],
                "split": str(row.get("split", "heldout")),
                "label": label,
                "model": model,
                "dataset_variant": variant,
            }
            transcripts.append(record)
            labels.append(-1 if label is None else int(label))
            prompts.append(str(row["prompt"]))
            transcript_ids.append(transcript_id)

    if len(set(transcript_ids)) != len(transcript_ids):
        raise ValueError("finalization produced duplicate transcript_id values")
    return (
        transcripts,
        np.asarray(labels, dtype=np.int8),
        np.asarray(prompts),
        np.asarray(transcript_ids),
    )


def _write_cache_atomic(
    *,
    path: Path,
    position: str,
    shards: list[ValidatedShard],
    total: int,
    activation_shape: tuple[int, int],
    labels: np.ndarray,
    prompts: np.ndarray,
    transcript_ids: np.ndarray,
    model: str,
    variant: str,
) -> None:
    with tempfile.TemporaryDirectory(
        prefix=f".{position}-", dir=path.parent
    ) as temp_dir:
        mmap_path = Path(temp_dir) / "X.npy"
        X = np.lib.format.open_memmap(
            mmap_path, mode="w+", dtype=np.float16, shape=(total, *activation_shape)
        )
        offset = 0
        for retained in shards:
            with np.load(retained.path, allow_pickle=False) as shard:
                values = np.asarray(shard[position])
                end = offset + len(values)
                X[offset:end] = values
                offset = end
        X.flush()
        if offset != total:
            raise ValueError(
                f"{position} cache fill mismatch: expected {total}, wrote {offset}"
            )
        temp_path = Path(temp_dir) / path.name
        with temp_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                X=X,
                y=labels,
                prompts=prompts,
                model=np.asarray(model),
                dataset_variant=np.asarray(variant),
                transcript_ids=transcript_ids,
                position=np.asarray(position),
                label_name=np.asarray("world"),
                positive_class=np.asarray("REAL"),
                positive_label=np.asarray(1, dtype=np.int8),
                negative_class=np.asarray("SHAM"),
                negative_label=np.asarray(0, dtype=np.int8),
                unlabeled_label=np.asarray(-1, dtype=np.int8),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", prefix=f".{path.name}.", dir=path.parent, delete=False
    ) as handle:
        temp_path = Path(handle.name)
        try:
            for row in rows:
                handle.write(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
    os.replace(temp_path, path)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", prefix=f".{path.name}.", dir=path.parent, delete=False
    ) as handle:
        temp_path = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
    os.replace(temp_path, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
