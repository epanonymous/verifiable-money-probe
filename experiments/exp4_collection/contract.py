"""Dataset grouping and per-shard contracts shared with the Modal collector."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
RUN_V1 = "run_v1"
LEAK_FREE = "leak_free"
VARIANTS = (RUN_V1, LEAK_FREE)
GEN_BATCH = 25  # rollouts generated per call (num_return_sequences)
FWD_CHUNK = 12  # transcripts per activation forward pass
MIXED_TEXT = "MIXED"
MIXED_ID = -1
SHARD_SCALARS = ("world", "cond", "template_id", "split", "label")
DATASET_FILES = {
    "main": ("prompts_main.jsonl", "prompts_framing.jsonl"),
    "lbr": ("lowbaserate_eval.jsonl",),
}
CONDITION_NAMES = {
    "a": "claimed",
    "b": "verified",
    "c": "causally_binding",
    "framing": "framing",
}


def validate_variant(variant: object) -> str:
    if variant not in VARIANTS:
        raise ValueError(
            f"unknown dataset variant {variant!r}; expected one of {list(VARIANTS)}"
        )
    return str(variant)


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def rollout_batch_plan(n_rollouts: int, variant: str = RUN_V1) -> tuple[int, int]:
    """Return (generated, forwarded) sequence counts for `n_rollouts` kept rows.

    Run v1 generates and forwards exactly the retained rollouts, which is the
    shape its committed shards were written with. The leak-free variant rounds
    both stages up to whole batches and discards the surplus, so no retained
    rollout is the sole member of a short generate call or a short forward
    chunk: batch size and pad width change bf16 reduction order, and inside a
    merged REAL/SHAM group that offset would otherwise track the label.
    """

    if n_rollouts <= 0:
        raise ValueError(f"n_rollouts must be positive, got {n_rollouts}")
    if validate_variant(variant) != LEAK_FREE:
        return n_rollouts, n_rollouts
    forwarded = _round_up(n_rollouts, FWD_CHUNK)
    return _round_up(forwarded, GEN_BATCH), forwarded


def _stable_permutation(seed_text: str, count: int) -> tuple[int, ...]:
    """Order `count` slots by sha256 rank, independent of PYTHONHASHSEED."""

    base = hashlib.sha256(seed_text.encode("utf-8")).digest()
    ranked = sorted(
        (hashlib.sha256(base + index.to_bytes(4, "big")).digest(), index)
        for index in range(count)
    )
    return tuple(index for _, index in ranked)


def _row_scalars(row: dict[str, Any]) -> dict[str, Any]:
    label = row.get("label")
    return {
        "world": row["world"],
        "cond": row["cond"],
        "template_id": row["template_id"],
        "split": row.get("split", "heldout"),
        "label": MIXED_ID if label is None else label,
    }


_NEUTRAL_SCALARS = {
    "world": MIXED_TEXT,
    "cond": MIXED_TEXT,
    "template_id": MIXED_ID,
    "split": MIXED_TEXT,
    "label": MIXED_ID,
}


@dataclass(frozen=True)
class PromptGroup:
    """One collector input and the shard name/row expansion it must produce."""

    prompt: str
    rows: tuple[dict[str, Any], ...]
    variant: str = field(default=RUN_V1)

    @property
    def key(self) -> str:
        return str(self.rows[0]["id"])

    @property
    def filename(self) -> str:
        return f"{self.key}.npz"

    @property
    def expanded_row_ids(self) -> tuple[str, ...]:
        row_major = tuple(
            str(row["id"])
            for row in self.rows
            for _ in range(int(row.get("n_rollouts", 1)))
        )
        # Run v1 shards on the volume were written row-major; that ordering is
        # their resume contract and must not move. The leak-free variant merges
        # each REAL/SHAM pair into one prompt group, which would otherwise make
        # the label a contiguous first-half/second-half split aligned with
        # generation-batch and forward-chunk boundaries.
        if self.variant != LEAK_FREE:
            return row_major
        order = _stable_permutation(self.prompt, len(row_major))
        return tuple(row_major[index] for index in order)

    @property
    def shard_scalars(self) -> dict[str, Any]:
        """Per-shard descriptors written beside the activations.

        Run v1 shards on the volume carry the first row's values verbatim, and
        that is their resume contract. A leak-free group merges a REAL/SHAM pair
        into one shard, so a first-row scalar would describe only part of the
        rollouts and would assert a class the shard does not have; any field the
        group's rows disagree on collapses to a neutral marker instead. Per-row
        truth stays in `row_ids`, which finalization resolves against the
        dataset.
        """

        scalars = _row_scalars(self.rows[0])
        if self.variant != LEAK_FREE:
            return scalars
        for name in SHARD_SCALARS:
            if any(_row_scalars(row)[name] != scalars[name] for row in self.rows[1:]):
                scalars[name] = _NEUTRAL_SCALARS[name]
        return scalars


def load_rows(data_dir: str | Path, which: str) -> list[dict[str, Any]]:
    """Load the committed inputs for one collector partition in file order."""

    try:
        filenames = DATASET_FILES[which]
    except KeyError as exc:
        raise ValueError(f"unknown collection {which!r}; expected main or lbr") from exc

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for filename in filenames:
        path = Path(data_dir) / filename
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON at {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise TypeError(
                        f"dataset row at {path}:{line_number} is not an object"
                    )
                missing = {"id", "prompt", "world", "cond", "template_id"}.difference(
                    row
                )
                if missing:
                    raise ValueError(
                        f"dataset row at {path}:{line_number} is missing {sorted(missing)}"
                    )
                if not isinstance(row["prompt"], str):
                    raise TypeError(
                        f"dataset prompt at {path}:{line_number} is not a string"
                    )
                row_id = str(row["id"])
                if row_id in seen_ids:
                    raise ValueError(f"duplicate dataset row id {row_id!r}")
                seen_ids.add(row_id)
                rows.append(row)
    return rows


def dataset_variant(data_dir: str | Path) -> str:
    """Read the variant a dataset declares in its own manifest.

    Deriving this from the data keeps collect, resume validation, inventory and
    finalization on one answer even when a caller passes `--data-dir` directly.
    """

    manifest = Path(data_dir) / "manifest.json"
    if not manifest.exists():
        return RUN_V1
    try:
        declared = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid dataset manifest {manifest}: {exc}") from exc
    if not isinstance(declared, dict):
        raise TypeError(f"dataset manifest {manifest} is not an object")
    return validate_variant(declared.get("variant", RUN_V1))


def ordered_prompt_groups(
    rows: Iterable[dict[str, Any]], variant: str = RUN_V1
) -> list[PromptGroup]:
    """Apply collect.py's exact prompt key and first-row-id ordering rule."""

    resolved = validate_variant(variant)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["prompt"], []).append(row)
    return [
        PromptGroup(prompt=prompt, rows=tuple(group_rows), variant=resolved)
        for prompt, group_rows in sorted(
            groups.items(), key=lambda item: item[1][0]["id"]
        )
    ]


def expected_prompt_groups(
    data_dir: str | Path, which: str, variant: str | None = None
) -> list[PromptGroup]:
    resolved = (
        dataset_variant(data_dir) if variant is None else validate_variant(variant)
    )
    return ordered_prompt_groups(load_rows(data_dir, which), resolved)


def canonical_condition(value: object) -> str:
    key = str(value).strip().lower()
    try:
        return CONDITION_NAMES[key]
    except KeyError as exc:
        raise ValueError(f"unknown experiment-4 condition {value!r}") from exc


def validate_resume_shard(
    path: str | Path, group: PromptGroup, model: str = MODEL
) -> None:
    """Reject a malformed retained shard instead of silently regenerating it."""

    import numpy as np

    path = Path(path)
    required = {
        "prompt_final",
        "response_final",
        "texts",
        "decisions",
        "row_ids",
        "model",
        *SHARD_SCALARS,
    }
    try:
        with np.load(path, allow_pickle=False) as shard:
            missing = required.difference(shard.files)
            if missing:
                raise ValueError(f"missing keys {sorted(missing)}")
            prompt_final = np.asarray(shard["prompt_final"])
            response_final = np.asarray(shard["response_final"])
            texts = np.asarray(shard["texts"])
            decisions = np.asarray(shard["decisions"])
            row_ids = np.asarray(shard["row_ids"])
            shard_model = _scalar(shard["model"], "model")
            for field, expected in group.shard_scalars.items():
                actual = _scalar(shard[field], field)
                if str(actual) != str(expected):
                    raise ValueError(
                        f"{field} mismatch: expected {expected!r}, got {actual!r}"
                    )
    except (OSError, EOFError) as exc:
        raise ValueError(f"cannot read npz: {exc}") from exc

    expected_rows = group.expanded_row_ids
    n_rollouts = len(expected_rows)
    if prompt_final.ndim != 3 or response_final.ndim != 3:
        raise ValueError(
            "prompt_final and response_final must have shape [rollouts, layers, d_model]"
        )
    if prompt_final.shape != response_final.shape:
        raise ValueError(
            f"activation shape mismatch: {prompt_final.shape} vs {response_final.shape}"
        )
    if prompt_final.shape[0] != n_rollouts:
        raise ValueError(
            f"activation rollout count mismatch: expected {n_rollouts}, got {prompt_final.shape[0]}"
        )
    if not np.issubdtype(prompt_final.dtype, np.number) or not np.issubdtype(
        response_final.dtype, np.number
    ):
        raise ValueError("activation arrays must be numeric")
    if not np.isfinite(prompt_final).all() or not np.isfinite(response_final).all():
        raise ValueError("activation arrays contain NaN or infinite values")
    for field, values in (
        ("texts", texts),
        ("decisions", decisions),
        ("row_ids", row_ids),
    ):
        if values.ndim != 1 or len(values) != n_rollouts:
            raise ValueError(
                f"{field} must have shape [{n_rollouts}], got {values.shape}"
            )
    invalid_decisions = sorted(
        {str(value) for value in decisions}.difference({"SPEND", "HOLD", "AMBIGUOUS"})
    )
    if invalid_decisions:
        raise ValueError(f"decisions contain invalid values {invalid_decisions}")
    actual_rows = tuple(str(value) for value in row_ids)
    if actual_rows != expected_rows:
        raise ValueError(
            f"row_ids mismatch: expected {list(expected_rows)!r}, got {list(actual_rows)!r}"
        )
    if shard_model != model:
        raise ValueError(f"model mismatch: expected {model!r}, got {shard_model!r}")


def _scalar(value: Any, field: str) -> Any:
    import numpy as np

    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    if array.size == 1:
        return array.reshape(-1)[0].item()
    raise ValueError(f"{field} must be scalar, got shape {array.shape}")
