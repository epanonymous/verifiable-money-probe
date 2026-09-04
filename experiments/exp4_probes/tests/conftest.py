from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def synthetic_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Small random tensors in the exact exp5 cache shape and key format."""

    rng = np.random.default_rng(19)
    aliases = (("a", "claimed"), ("b", "verified"), ("c", "causally_binding"))
    n_templates = 15
    n_variants = 2
    n_layers = 4
    d_model = 10
    prompts = []
    metadata = []
    condition_indexes = []
    y = []
    for template_id in range(n_templates):
        for condition_index, (alias, full_name) in enumerate(aliases):
            for variant in range(n_variants):
                prompt = f"template={template_id}; condition={full_name}; variant={variant}"
                prompts.append(prompt)
                condition_indexes.append(condition_index)
                y.append((template_id + variant) % 2)
                metadata.append(
                    {
                        "id": f"tx_t{template_id:02d}_{alias}_v{variant}",
                        "cond": alias,
                        "template_id": template_id,
                        "split": "heldout" if template_id >= 12 else "train",
                        "prompt": prompt,
                    }
                )

    X = rng.normal(0, 0.35, size=(len(prompts), n_layers, d_model)).astype(np.float16)
    condition_indexes = np.asarray(condition_indexes)
    # Several layers contain different linear encodings of ladder position.
    X[:, 1, 0] += (condition_indexes - 1) * 1.5
    for condition_index in range(3):
        X[condition_indexes == condition_index, 2, condition_index] += 2.0
    X[:, 3, 4] += condition_indexes * 1.2

    cache_path = tmp_path / "cache.npz"
    np.savez_compressed(
        cache_path,
        X=X,
        y=np.asarray(y),
        prompts=np.asarray(prompts),
        model=np.asarray("fixture/model"),
    )
    metadata_path = tmp_path / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for row in metadata:
            handle.write(json.dumps(row) + "\n")
    return cache_path, metadata_path
