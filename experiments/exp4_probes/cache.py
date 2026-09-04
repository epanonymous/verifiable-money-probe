"""Load exp5 activation caches and their aligned exp3 metadata sidecars."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.exp4_score_contract import normalize_condition

EXP5_CACHE_KEYS = frozenset({"X", "y", "prompts", "model"})


@dataclass(frozen=True)
class ActivationCache:
    """The exact array contract written by exp5/model_setup/modal_smoke.py."""

    X: np.ndarray
    y: np.ndarray
    prompts: np.ndarray
    model: str
    dataset_variant: str | None = None
    label_name: str | None = None
    positive_class: str | None = None
    positive_label: int | None = None
    negative_class: str | None = None
    negative_label: int | None = None
    transcript_ids: np.ndarray | None = None
    position: str | None = None

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_layers(self) -> int:
        """Number of cached boundaries (embedding plus transformer layers)."""

        return int(self.X.shape[1])

    @property
    def d_model(self) -> int:
        return int(self.X.shape[2])


@dataclass(frozen=True)
class SampleMetadata:
    transcript_id: str
    condition: str
    template_id: str
    split: str
    prompt: str
    raw: dict[str, Any]


def _scalar_string(value: np.ndarray, field: str) -> str:
    if value.ndim == 0:
        return str(value.item())
    if value.size == 1:
        return str(value.reshape(-1)[0])
    raise ValueError(
        f"cache field {field!r} must be a scalar string, got shape {value.shape}"
    )


def load_activation_cache(path: str | Path) -> ActivationCache:
    """Load and validate the exact exp5 cache layout.

    exp5 writes an ``npz`` containing:
      - X: [n_prompts, n_layers + 1, d_model]
      - y: [n_prompts]
      - prompts: [n_prompts]
      - model: scalar model id

    The four exp5 keys are mandatory. Optional label-name/positive-class fields
    written by finalization are retained; other keys stay ignored for forward
    compatibility, and no metadata sidecar fields are read from the cache.

    ``dataset_variant`` names the collector dataset the activations came from
    (``run_v1`` or the Exp 4b ``leak_free`` variant), so a leak-free cache cannot
    be scored as an unmarked Run v1 one. Caches written before finalization
    recorded it report ``None``.
    """

    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        missing = EXP5_CACHE_KEYS.difference(data.files)
        if missing:
            raise ValueError(
                f"{path} is not an exp5 cache; missing keys: {sorted(missing)}"
            )
        X = np.asarray(data["X"])
        y = np.asarray(data["y"])
        prompts = np.asarray(data["prompts"])
        model = _scalar_string(np.asarray(data["model"]), "model")
        variant = (
            _scalar_string(np.asarray(data["dataset_variant"]), "dataset_variant")
            if "dataset_variant" in data.files
            else None
        )
        label_name = (
            _scalar_string(np.asarray(data["label_name"]), "label_name")
            if "label_name" in data.files
            else None
        )
        positive_class = (
            _scalar_string(np.asarray(data["positive_class"]), "positive_class")
            if "positive_class" in data.files
            else None
        )
        positive_label = (
            int(np.asarray(data["positive_label"]).item())
            if "positive_label" in data.files
            else None
        )
        negative_class = (
            _scalar_string(np.asarray(data["negative_class"]), "negative_class")
            if "negative_class" in data.files
            else None
        )
        negative_label = (
            int(np.asarray(data["negative_label"]).item())
            if "negative_label" in data.files
            else None
        )
        transcript_ids = (
            np.asarray(data["transcript_ids"]).astype(str)
            if "transcript_ids" in data.files
            else None
        )
        position = (
            _scalar_string(np.asarray(data["position"]), "position")
            if "position" in data.files
            else None
        )

    if X.ndim != 3:
        raise ValueError(f"X must have shape [samples, layers, d_model], got {X.shape}")
    if not np.issubdtype(X.dtype, np.number):
        raise ValueError(f"X must be numeric, got dtype {X.dtype}")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite activations")

    n_samples = X.shape[0]
    if y.ndim != 1 or len(y) != n_samples:
        raise ValueError(f"y must have shape [{n_samples}], got {y.shape}")
    if prompts.ndim != 1 or len(prompts) != n_samples:
        raise ValueError(f"prompts must have shape [{n_samples}], got {prompts.shape}")
    if not model:
        raise ValueError("model id must not be empty")
    if variant is not None and not variant:
        raise ValueError("dataset_variant must not be empty when present")
    if transcript_ids is not None:
        if transcript_ids.ndim != 1 or len(transcript_ids) != n_samples:
            raise ValueError(
                f"transcript_ids must have shape [{n_samples}], got {transcript_ids.shape}"
            )
        if len(set(transcript_ids)) != len(transcript_ids):
            raise ValueError("cache transcript_ids must be unique")

    return ActivationCache(
        X=X,
        y=y,
        prompts=prompts.astype(str),
        model=model,
        dataset_variant=variant,
        label_name=label_name,
        positive_class=positive_class,
        positive_label=positive_label,
        negative_class=negative_class,
        negative_label=negative_label,
        transcript_ids=transcript_ids,
        position=position,
    )


def load_metadata(path: str | Path, cache: ActivationCache) -> list[SampleMetadata]:
    """Load JSONL metadata aligned one-for-one with cache rows.

    The sidecar accepts exp3's ``id``/``cond`` fields directly. A rollout collector
    may instead emit the explicit ``transcript_id``/``condition`` spellings.
    Prompts are checked byte-for-byte to catch silent row-order mistakes.
    """

    path = Path(path)
    rows: list[SampleMetadata] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            transcript_id = raw.get("transcript_id", raw.get("id"))
            condition = raw.get("condition", raw.get("cond"))
            missing = [
                name
                for name, value in (
                    ("transcript_id/id", transcript_id),
                    ("condition/cond", condition),
                    ("template_id", raw.get("template_id")),
                    ("prompt", raw.get("prompt")),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    f"missing {', '.join(missing)} at {path}:{line_number}"
                )
            rows.append(
                SampleMetadata(
                    transcript_id=str(transcript_id),
                    condition=normalize_condition(condition),
                    template_id=str(raw["template_id"]),
                    split=str(raw.get("split", "")).strip().lower(),
                    prompt=str(raw["prompt"]),
                    raw=raw,
                )
            )

    if len(rows) != cache.n_samples:
        raise ValueError(
            f"metadata/cache row count mismatch: {len(rows)} metadata rows vs "
            f"{cache.n_samples} cache rows"
        )
    ids = [row.transcript_id for row in rows]
    if len(set(ids)) != len(ids):
        duplicates = sorted(item for item, count in Counter(ids).items() if count > 1)
        raise ValueError(
            f"transcript_id must be unique; duplicates include {duplicates[:5]}"
        )
    for index, (row, cached_prompt) in enumerate(zip(rows, cache.prompts, strict=True)):
        if row.prompt != cached_prompt:
            raise ValueError(
                f"metadata/cache prompt mismatch at row {index} ({row.transcript_id}); "
                "the sidecar must use cache row order"
            )
        if (
            cache.transcript_ids is not None
            and row.transcript_id != cache.transcript_ids[index]
        ):
            raise ValueError(
                f"metadata/cache transcript_id mismatch at row {index}; "
                "the sidecar must use cache row order"
            )
    return rows
