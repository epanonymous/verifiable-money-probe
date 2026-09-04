"""Seal integrity for the Wave 2 evidence bundle.

`evidence_manifest.json` pins the path, byte size, and SHA-256 of every other
file in `results/wave2/`. That bundle is immutable: corrections go in
`docs/errata.md`, never into a bundle file. This module is the enforcement.

One deviation from the original seal already exists on `main` and is recorded
in the errata seal-integrity ledger (see `ACCEPTED_DEVIATIONS`). It is pinned
to its exact current bytes here, so any *further* edit to the bundle fails.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = ROOT / "experiments/exp4_analysis/results/wave2"
MANIFEST = BUNDLE / "evidence_manifest.json"
ERRATA = ROOT / "docs/errata.md"

# Bundle files whose current bytes intentionally differ from the sealed
# manifest. Each entry MUST have a row in the seal-integrity ledger in
# docs/errata.md. Adding an entry here is a deliberate, reviewed act.
#
# README.md: `a2463c5` (PR #19) prepended a Run v1 interpretation banner to the
# bundle README without versioning the seal. Additive text only; no table,
# count, or metric changed. Sealed original: 8,808 B / 0637290a…4818, still
# recoverable at `git show 0093a1b:<path>`.
ACCEPTED_DEVIATIONS: dict[str, tuple[int, str]] = {
    "README.md": (
        9327,
        "747c4998c52dcf74bd01687a50d54dc4d727e3b26ef63a533aa93331222d7103",
    ),
}


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _digest(path: Path) -> tuple[int, str]:
    data = path.read_bytes()
    return len(data), hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("artifact", _manifest()["artifacts"], ids=lambda a: a["path"])
def test_bundle_artifact_matches_its_pinned_hash(artifact: dict) -> None:
    path = BUNDLE / artifact["path"]
    assert path.is_file(), f"sealed artifact is missing: {artifact['path']}"

    expected_size, expected_sha = ACCEPTED_DEVIATIONS.get(
        artifact["path"], (artifact["size_bytes"], artifact["sha256"])
    )
    assert _digest(path) == (expected_size, expected_sha), (
        f"{artifact['path']} no longer matches its pinned hash. The Wave 2 "
        "bundle is sealed — put the correction in docs/errata.md instead of "
        "editing the bundle."
    )


def test_bundle_contains_no_undeclared_files() -> None:
    declared = {a["path"] for a in _manifest()["artifacts"]} | {
        MANIFEST.name  # excluded from its own scope; its hash would be recursive
    }
    present = {str(p.relative_to(BUNDLE)) for p in BUNDLE.rglob("*") if p.is_file()}
    assert present == declared, (
        "the sealed bundle gained or lost files; new evidence belongs in a new "
        "sealed bundle, and prose belongs in docs/errata.md"
    )


def test_every_accepted_deviation_is_recorded_in_the_errata() -> None:
    errata = ERRATA.read_text(encoding="utf-8")
    manifest_hashes = {a["path"]: a["sha256"] for a in _manifest()["artifacts"]}
    for path, (_, current_sha) in ACCEPTED_DEVIATIONS.items():
        assert current_sha in errata, (
            f"{path} deviates from the seal but its current hash is not in the "
            f"{ERRATA.relative_to(ROOT)} seal-integrity ledger"
        )
        assert manifest_hashes[path] in errata, (
            f"the seal-integrity ledger must also record the manifest-pinned "
            f"hash for {path}"
        )
