"""V5 split: stratified by capture kind, pairs never cut, seeded, and committed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.exp7_tool_probe import authenticity as auth
from experiments.exp7_tool_probe.config import AUTH_SPLIT_VERSION, SEED

DATA = Path(__file__).resolve().parents[1] / "data" / "v0"


@pytest.fixture(scope="module")
def captures() -> dict:
    return json.loads((DATA / "captures.json").read_text())


def test_stratified_split_puts_every_kind_on_both_sides(captures) -> None:
    kinds = auth.pair_kinds(captures)
    splits = auth.stratified_pair_split(kinds)
    counts = auth.split_counts(kinds, splits)
    assert set(counts["train"]) == set(counts["heldout"]) == set(kinds)
    assert counts["heldout"] == {"block": 1, "native_balance": 2, "receipt": 3, "token_balance": 3, "transaction": 3}
    assert sum(counts["heldout"].values()) == 12 and sum(counts["train"].values()) == 48
    for kind, n_train in counts["train"].items():
        n_held = counts["heldout"][kind]
        assert n_held == max(1, round((n_train + n_held) * auth.HELDOUT_FRACTION))


def test_tail_split_is_the_confounded_one(captures) -> None:
    kinds = auth.pair_kinds(captures)
    counts = auth.split_counts(kinds, auth.tail_pair_split(kinds))
    assert counts["heldout"] == {"native_balance": 12}
    assert "native_balance" not in counts["train"]
    assert auth.tail_pair_split(kinds) != auth.stratified_pair_split(kinds)


def test_stratified_split_is_seeded_and_deterministic() -> None:
    kinds = ["a"] * 10 + ["b"] * 5 + ["c"] * 20
    one = auth.stratified_pair_split(kinds, seed=SEED)
    assert one == auth.stratified_pair_split(kinds, seed=SEED)
    assert one != auth.stratified_pair_split(kinds, seed=SEED + 1)
    assert auth.split_counts(kinds, one)["heldout"] == {"a": 2, "b": 1, "c": 4}
    with pytest.raises(ValueError):
        auth.stratified_pair_split(["solo"], seed=SEED)
    with pytest.raises(ValueError):
        auth.pair_split(kinds, "random")


def test_rows_keep_twins_together_under_both_schemes(captures) -> None:
    for scheme in auth.SPLIT_SCHEMES:
        rows = auth.build_authenticity_rows(captures, seed=SEED, split_scheme=scheme)
        assert len(rows) == 120
        by_pair: dict[int, set[str]] = {}
        for row in rows:
            by_pair.setdefault(row["template_id"], set()).add(row["split"])
        assert all(len(s) == 1 for s in by_pair.values())
        assert sum(r["split"] == "heldout" for r in rows) == 24


def test_committed_split_manifest_regenerates_byte_identically(captures) -> None:
    expected = json.dumps(auth.split_manifest(captures, seed=SEED), indent=2, sort_keys=True) + "\n"
    assert (DATA / "auth_split.json").read_text() == expected
    manifest = json.loads(expected)
    assert manifest["version"] == AUTH_SPLIT_VERSION
    assert manifest["superseded_tail_split"]["counts"]["heldout"] == {"native_balance": 12}


def test_frozen_auth_rows_match_regeneration_up_to_key_order(captures) -> None:
    """The frozen rows are the collector's exact inputs; a regeneration differs in
    JSON key order inside the prompt (captures.json was written sorted) and, by
    design, in the split field. Everything else must agree."""

    frozen = [json.loads(line) for line in (DATA / "auth_rows.jsonl").read_text().splitlines() if line]
    regen = auth.build_authenticity_rows(captures, seed=SEED, split_scheme="tail")
    stratified = auth.build_authenticity_rows(captures, seed=SEED)
    manifest = json.loads((DATA / "auth_split.json").read_text())

    def payload(prompt: str) -> tuple:
        request, response = prompt.split("Request:\n", 1)[1].split("\n\nResponse:\n", 1)
        return json.loads(request), json.loads(response)

    for old, new, strat in zip(frozen, regen, stratified, strict=True):
        for key in ("id", "kind", "method", "capture_kind", "template_id", "label", "n_rollouts", "split"):
            assert old[key] == new[key], key
        assert payload(old["prompt"]) == payload(new["prompt"])
        assert strat["prompt"] == new["prompt"]
        assert strat["split"] == manifest["split"][old["template_id"]]
        assert old["split"] == manifest["superseded_tail_split"]["split"][old["template_id"]]
