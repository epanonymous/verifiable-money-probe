"""Leakage-resistant group splits for the exp3 paraphrase bank."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .cache import SampleMetadata


@dataclass(frozen=True)
class GroupSplits:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    train_groups: tuple[str, ...]
    val_groups: tuple[str, ...]
    test_groups: tuple[str, ...]


def _take_groups(
    records: Sequence[SampleMetadata], selected_groups: set[str]
) -> np.ndarray:
    return np.asarray(
        [
            index
            for index, row in enumerate(records)
            if row.template_id in selected_groups
        ],
        dtype=np.int64,
    )


def make_group_splits(
    records: Sequence[SampleMetadata],
    *,
    seed: int = 7,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> GroupSplits:
    """Split by exp3 template id, never by individual activation row.

    exp3 marks templates 38--47 ``heldout``; those become test verbatim. The
    remaining template groups are split train/validation. If a fixture has no
    explicit heldout groups, a deterministic group-level test split is made.
    """

    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("val_fraction and test_fraction must be between 0 and 1")
    if not records:
        raise ValueError("cannot split an empty dataset")

    split_labels: dict[str, set[str]] = defaultdict(set)
    for row in records:
        if row.split:
            split_labels[row.template_id].add(row.split)
    for group, labels in split_labels.items():
        is_train = bool(labels.intersection({"train", "training"}))
        is_test = bool(labels.intersection({"heldout", "held_out", "test"}))
        if is_train and is_test:
            raise ValueError(
                f"template group {group!r} has conflicting split labels: {labels}"
            )

    all_groups = sorted({row.template_id for row in records})
    rng = np.random.default_rng(seed)
    shuffled = [all_groups[index] for index in rng.permutation(len(all_groups))]
    explicit_test_groups = {
        group
        for group, labels in split_labels.items()
        if labels.intersection({"heldout", "held_out", "test"})
    }
    test_groups = set(explicit_test_groups)

    if not test_groups:
        n_test = max(1, int(round(len(shuffled) * test_fraction)))
        if n_test >= len(shuffled) - 1:
            raise ValueError("need at least three template groups for train/val/test")
        test_groups = set(shuffled[:n_test])

    if explicit_test_groups:
        unknown_groups = {
            group
            for group in all_groups
            if group not in test_groups
            and not split_labels.get(group, set()).intersection({"train", "training"})
        }
        if unknown_groups:
            raise ValueError(
                "every non-heldout template group must be explicitly train; "
                f"got {sorted(unknown_groups)}"
            )
    candidate_train = [group for group in shuffled if group not in test_groups]
    if len(candidate_train) < 2:
        raise ValueError(
            "need at least two non-test template groups for train and validation"
        )
    n_val = max(1, int(round(len(candidate_train) * val_fraction)))
    n_val = min(n_val, len(candidate_train) - 1)
    val_groups = set(candidate_train[:n_val])
    train_groups = set(candidate_train[n_val:])

    if (
        train_groups & val_groups
        or train_groups & test_groups
        or val_groups & test_groups
    ):
        raise AssertionError("internal error: template groups leaked across splits")

    result = GroupSplits(
        train=_take_groups(records, train_groups),
        val=_take_groups(records, val_groups),
        test=_take_groups(records, test_groups),
        train_groups=tuple(sorted(train_groups)),
        val_groups=tuple(sorted(val_groups)),
        test_groups=tuple(sorted(test_groups)),
    )
    if any(len(part) == 0 for part in (result.train, result.val, result.test)):
        raise ValueError("train, validation, and test must all contain samples")
    return result
