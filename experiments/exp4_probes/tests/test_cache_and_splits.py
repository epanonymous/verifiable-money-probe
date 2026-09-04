from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.exp4_probes.cache import load_activation_cache, load_metadata
from experiments.exp4_probes.splits import make_group_splits


def test_loads_exact_exp5_cache_and_exp3_aliases(
    synthetic_fixture: tuple[Path, Path],
) -> None:
    cache_path, metadata_path = synthetic_fixture
    cache = load_activation_cache(cache_path)
    metadata = load_metadata(metadata_path, cache)

    assert cache.X.shape == (90, 4, 10)
    assert cache.model == "fixture/model"
    assert cache.positive_class is None
    assert {row.condition for row in metadata} == {
        "claimed",
        "verified",
        "causally_binding",
    }


def test_metadata_prompt_alignment_fails_closed(
    synthetic_fixture: tuple[Path, Path], tmp_path: Path
) -> None:
    cache_path, metadata_path = synthetic_fixture
    cache = load_activation_cache(cache_path)
    rows = [json.loads(line) for line in metadata_path.read_text().splitlines()]
    rows[0]["prompt"] = "reordered or wrong"
    wrong = tmp_path / "wrong.jsonl"
    wrong.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    with pytest.raises(ValueError, match="prompt mismatch"):
        load_metadata(wrong, cache)


def test_missing_exp5_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    np.savez(path, X=np.zeros((2, 2, 2)), y=np.zeros(2), prompts=np.asarray(["a", "b"]))
    with pytest.raises(ValueError, match="missing keys"):
        load_activation_cache(path)


def test_template_groups_never_leak(synthetic_fixture: tuple[Path, Path]) -> None:
    cache_path, metadata_path = synthetic_fixture
    cache = load_activation_cache(cache_path)
    metadata = load_metadata(metadata_path, cache)
    splits = make_group_splits(metadata, seed=7)

    train = set(splits.train_groups)
    val = set(splits.val_groups)
    test = set(splits.test_groups)
    assert not train & val
    assert not train & test
    assert not val & test
    assert test == {"12", "13", "14"}
    for indices in (splits.train, splits.val, splits.test):
        groups = {metadata[index].template_id for index in indices}
        assert groups in (train, val, test)
