from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.exp4_collection.contract import MODEL, expected_prompt_groups


def _write_dataset(data_dir: Path, rows: dict[str, list[dict]]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename, file_rows in rows.items():
        (data_dir / filename).write_text(
            "".join(json.dumps(row) + "\n" for row in file_rows), encoding="utf-8"
        )


def _write_shards(data_dir: Path, staging_dir: Path) -> None:
    for which in ("main", "lbr"):
        shard_dir = staging_dir / f"collect_{which}"
        shard_dir.mkdir(parents=True)
        for group_index, group in enumerate(expected_prompt_groups(data_dir, which)):
            n = len(group.expanded_row_ids)
            base = 100 * (which == "lbr") + 10 * group_index
            prompt_final = np.arange(n * 6, dtype=np.float16).reshape(n, 2, 3) + base
            np.savez_compressed(
                shard_dir / group.filename,
                prompt_final=prompt_final,
                response_final=prompt_final + 0.5,
                texts=np.asarray(
                    [f"{group.key} slot {index} HOLD" for index in range(n)]
                ),
                decisions=np.asarray(["HOLD"] * n),
                row_ids=np.asarray(group.expanded_row_ids),
                model=MODEL,
                **group.shard_scalars,
            )


@pytest.fixture
def collection_fixture(tmp_path: Path) -> tuple[Path, Path]:
    data_dir = tmp_path / "data"
    staging_dir = tmp_path / "staging"
    rows = {
        "prompts_main.jsonl": [
            {
                "id": "real_a_t00",
                "world": "REAL",
                "cond": "a",
                "template_id": 0,
                "split": "train",
                "label": 1,
                "n_rollouts": 2,
                "prompt": "shared claimed prompt",
            },
            {
                "id": "sham_a_t00",
                "world": "SHAM",
                "cond": "a",
                "template_id": 0,
                "split": "train",
                "label": 0,
                "n_rollouts": 1,
                "prompt": "shared claimed prompt",
            },
            {
                "id": "real_b_t01",
                "world": "REAL",
                "cond": "b",
                "template_id": 1,
                "split": "heldout",
                "label": 1,
                "n_rollouts": 2,
                "prompt": "verified prompt",
            },
        ],
        "prompts_framing.jsonl": [
            {
                "id": "framing_t00",
                "world": "FRAMING",
                "cond": "framing",
                "template_id": 0,
                "split": "train",
                "label": None,
                "n_rollouts": 1,
                "prompt": "framing prompt",
            }
        ],
        "lowbaserate_eval.jsonl": [
            {
                "id": "lbr_real_0000",
                "world": "REAL",
                "cond": "b",
                "template_id": 38,
                "label": 1,
                "prompt": "lbr real prompt",
            },
            {
                "id": "lbr_sham_0000",
                "world": "SHAM",
                "cond": "b",
                "template_id": 38,
                "label": 0,
                "prompt": "lbr sham shared prompt",
            },
            {
                "id": "lbr_sham_0001",
                "world": "SHAM",
                "cond": "b",
                "template_id": 39,
                "label": 0,
                "prompt": "lbr sham shared prompt",
            },
        ],
    }
    _write_dataset(data_dir, rows)
    _write_shards(data_dir, staging_dir)
    return data_dir, staging_dir


@pytest.fixture
def leak_free_collection_fixture(tmp_path: Path) -> tuple[Path, Path]:
    data_dir = tmp_path / "data_leak_free"
    staging_dir = tmp_path / "staging_leak_free"
    rows = {
        "prompts_main.jsonl": [
            {
                "id": "real_b_t00",
                "world": "REAL",
                "cond": "b",
                "template_id": 0,
                "split": "train",
                "label": 1,
                "n_rollouts": 3,
                "prompt": "paired verified prompt",
            },
            {
                "id": "sham_b_t00",
                "world": "SHAM",
                "cond": "b",
                "template_id": 0,
                "split": "train",
                "label": 0,
                "n_rollouts": 3,
                "prompt": "paired verified prompt",
            },
        ],
        "prompts_framing.jsonl": [
            {
                "id": "framing_t00",
                "world": "FRAMING",
                "cond": "framing",
                "template_id": 0,
                "split": "train",
                "label": None,
                "n_rollouts": 1,
                "prompt": "framing prompt",
            },
            {
                "id": "framing_t01",
                "world": "FRAMING",
                "cond": "framing",
                "template_id": 4,
                "split": "heldout",
                "label": None,
                "n_rollouts": 1,
                "prompt": "framing prompt",
            },
        ],
        "lowbaserate_eval.jsonl": [
            {
                "id": "lbr_real_0000",
                "world": "REAL",
                "cond": "b",
                "template_id": 38,
                "label": 1,
                "prompt": "lbr shared prompt",
            },
            {
                "id": "lbr_sham_0000",
                "world": "SHAM",
                "cond": "b",
                "template_id": 38,
                "label": 0,
                "prompt": "lbr shared prompt",
            },
            {
                "id": "lbr_sham_0001",
                "world": "SHAM",
                "cond": "b",
                "template_id": 38,
                "label": 0,
                "prompt": "lbr shared prompt",
            },
        ],
    }
    _write_dataset(data_dir, rows)
    (data_dir / "manifest.json").write_text(
        json.dumps({"variant": "leak_free"}), encoding="utf-8"
    )
    _write_shards(data_dir, staging_dir)
    return data_dir, staging_dir
