"""Exp 7 dataset and prompt rendering primitives."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import N_HELDOUT_TEMPLATES, N_TEMPLATES, N_TRAIN_TEMPLATES
from .templates import Template, sample_templates, render_system, render_user
from .worlds import LABELS, WORLDS, AMOUNT_USDC

N_TRAIN = N_TRAIN_TEMPLATES
N_HELDOUT = N_HELDOUT_TEMPLATES


@dataclass(frozen=True)
class ProbeRow:
    row_id: str
    split: str
    world: str
    cond: str
    template_id: int
    label: int
    system_prompt: str
    user_prompt: str
    n_rollouts: int = 1

    @property
    def prompt(self) -> str:
        return self.user_prompt

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.row_id,
            "split": self.split,
            "world": self.world,
            "cond": self.cond,
            "template_id": self.template_id,
            "label": self.label,
            "prompt": self.user_prompt,
            "system_prompt": self.system_prompt,
            "n_rollouts": self.n_rollouts,
        }


def build_templates(*, n_train: int = N_TRAIN, n_heldout: int = N_HELDOUT) -> list[Template]:
    if n_train < 0 or n_heldout < 0:
        raise ValueError("train and heldout counts must be non-negative")
    if n_train + n_heldout > N_TEMPLATES:
        raise ValueError("template allocation exceeds available templates")
    return sample_templates(n_train=n_train, n_heldout=n_heldout)


def build_rows(
    *,
    templates: list[Template] | None = None,
    cond: str = "probe",
    n_rollouts: int = 1,
) -> list[ProbeRow]:
    if n_rollouts <= 0:
        raise ValueError("n_rollouts must be positive")
    templates = sample_templates() if templates is None else templates
    if not templates:
        raise ValueError("templates must not be empty")

    rows: list[ProbeRow] = []
    for template in templates:
        system_prompt = render_system(template)
        user_prompt = render_user(template, AMOUNT_USDC)
        for world in WORLDS:
            rows.append(
                ProbeRow(
                    row_id=f"{world.lower()}_{cond}_{template.template_id:02d}",
                    split=template.split,
                    world=world,
                    cond=cond,
                    template_id=template.template_id,
                    label=LABELS[world],
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    n_rollouts=n_rollouts,
                )
            )
    return rows


def pairs(rows: list[ProbeRow]) -> dict[tuple[int, str], dict[str, ProbeRow]]:
    """Index rows by (template_id, split) so each key has both worlds."""

    grouped: dict[tuple[int, str], dict[str, ProbeRow]] = {}
    for row in rows:
        key = (row.template_id, row.split)
        group = grouped.setdefault(key, {})
        group[row.world] = row
    return grouped


def write_jsonl(path: str | Path, rows: list[ProbeRow]) -> dict[str, Any]:
    rows_obj = [row.to_dict() for row in rows]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows_obj:
            handle.write(json.dumps(row) + "\n")
    return {"rows": len(rows_obj)}
