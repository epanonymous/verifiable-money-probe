"""Provenance for Exp 7 v0: shard hashes, runtime versions, seeds, manifests.

Pure functions only (no Modal, no GPU) so they run in the CPU test suite and
in ``verify_provenance.py``; ``modal_collect.py`` imports them to stamp every
shard and to write a run manifest on the volume.

The committed ``data/v0/shards.sha256`` lists sha256 + byte size for the
off-git activation shards (``acts/main/t??.npz``, ``acts/auth/auth_*.npz``,
mirrored from Modal volume ``vmp-activations:exp7/v0``) and for every file
under ``data/v0`` and ``results/v0``. ``verify_provenance.py`` checks every
entry and fails closed (non-zero exit) on anything missing or mismatched;
``check_inventory`` is the same test as a function, run by ``analysis.py``
before it reads a single shard.

Collection identity (``collection_identity`` / ``identity_hash``): one hash
over everything that determines what a collection run produces — model id and
immutable revision, sha256 of every frozen input file, code commit, image pins
and resolved runtime versions, GPU, and every prepare/collect override (block,
expected balances, rollouts, temperature, max tokens, seed). It is stamped into
every shard; resume refuses a shard unless its full identity matches.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Any

from .config import SEED, config_hash

SHA_MANIFEST_NAME = "shards.sha256"
ACTS_PREFIX = "acts"  # logical prefix for off-git shards inside the manifest


# ----------------------------------------------------------------------------- hashing
def sha256_file(path: str | os.PathLike[str], chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def file_record(path: str | os.PathLike[str], root: str | os.PathLike[str], *, prefix: str | None = None) -> dict[str, Any]:
    """``{"path", "sha256", "bytes"}`` with ``path`` relative to ``root`` (posix)."""

    path = Path(path)
    rel = path.resolve().relative_to(Path(root).resolve()).as_posix()
    if prefix:
        rel = f"{prefix}/{rel}"
    return {"path": rel, "sha256": sha256_file(path), "bytes": path.stat().st_size}


def collect_records(root: str | os.PathLike[str], *, prefix: str | None = None, exclude: set[str] = frozenset()) -> list[dict[str, Any]]:
    """Every regular file under ``root`` (sorted by relative path), excluding names in ``exclude``."""

    root = Path(root)
    out = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.name in exclude or "__pycache__" in path.parts:
            continue
        out.append(file_record(path, root, prefix=prefix))
    return sorted(out, key=lambda r: r["path"])


def aggregate_sha256(records: list[dict[str, Any]]) -> str:
    """One hash over the sorted (path, sha256, bytes) lines; links results to raw shards."""

    lines = sorted(f"{r['sha256']}  {r['bytes']}  {r['path']}" for r in records)
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------- manifest file
def write_sha256_manifest(path: str | os.PathLike[str], records: list[dict[str, Any]], header: list[str] | None = None) -> None:
    lines = [f"# {line}" for line in (header or [])]
    lines += [f"{r['sha256']}  {r['bytes']}  {r['path']}" for r in sorted(records, key=lambda r: r["path"])]
    Path(path).write_text("\n".join(lines) + "\n")


def read_sha256_manifest(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    records = []
    for lineno, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 2)
        if len(parts) != 3 or len(parts[0]) != 64 or not parts[1].isdigit():
            raise ValueError(f"{path}:{lineno}: malformed manifest line {raw!r}")
        records.append({"sha256": parts[0], "bytes": int(parts[1]), "path": parts[2]})
    return records


def resolve_manifest_path(record_path: str, *, repo_root: str | os.PathLike[str], acts_dir: str | os.PathLike[str] | None) -> Path | None:
    """Where a manifest entry lives locally: ``acts/...`` under ``acts_dir``, else under the repo."""

    if record_path.startswith(ACTS_PREFIX + "/"):
        if acts_dir is None:
            return None
        return Path(acts_dir) / record_path[len(ACTS_PREFIX) + 1 :]
    return Path(repo_root) / record_path


def verify_records(records: list[dict[str, Any]], *, repo_root: str | os.PathLike[str], acts_dir: str | os.PathLike[str] | None) -> dict[str, list[dict[str, Any]]]:
    """Check each record: ``ok`` / ``mismatch`` (sha or size) / ``missing`` (absent locally)."""

    out: dict[str, list[dict[str, Any]]] = {"ok": [], "mismatch": [], "missing": []}
    for rec in records:
        local = resolve_manifest_path(rec["path"], repo_root=repo_root, acts_dir=acts_dir)
        if local is None or not local.is_file():
            out["missing"].append(dict(rec))
            continue
        size = local.stat().st_size
        digest = sha256_file(local)
        if size != rec["bytes"] or digest != rec["sha256"]:
            out["mismatch"].append({**rec, "local_sha256": digest, "local_bytes": size})
        else:
            out["ok"].append(dict(rec))
    return out


# ----------------------------------------------------------------------------- runtime / code identity
def runtime_versions() -> dict[str, Any]:
    """Versions of what matters for regenerating activations; None when not importable."""

    info: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "torch", "transformers", "accelerate", "sklearn", "scipy"):
        try:
            module = __import__(name)
            info[name] = getattr(module, "__version__", None)
        except Exception:  # noqa: BLE001 - optional on the CPU side
            info[name] = None
    try:
        import torch

        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        info["gpu_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:  # noqa: BLE001
        info.setdefault("cuda", None)
    return info


def git_commit(repo_root: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """``{"commit", "dirty"}`` of the checkout, or Nones when git is unavailable."""

    cwd = str(repo_root) if repo_root else None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=True).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except Exception:  # noqa: BLE001
        return {"commit": None, "dirty": None}


def shard_provenance(
    *,
    code: dict[str, Any],
    model: str,
    model_revision: str | None,
    gpu_requested: str,
    versions: dict[str, Any],
    seeds: dict[str, int],
    run_version: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The JSON stamped into every shard (as ``provenance_json``) and the run manifest."""

    return {
        "config_hash": config_hash(),
        "seed": SEED,
        "seeds": dict(seeds),
        "code": dict(code),
        "model": model,
        "model_revision": model_revision,
        "gpu_requested": gpu_requested,
        "versions": dict(versions),
        "run_version": run_version,
        **(extra or {}),
    }


def npz_provenance_fields(provenance: dict[str, Any]) -> dict[str, Any]:
    """Flat fields for ``np.savez``: the JSON blob plus the two keys resume checks."""

    return {
        "provenance_json": json.dumps(provenance, sort_keys=True),
        "config_hash": provenance["config_hash"],
        "seed": int(provenance["seed"]),
    }


FROZEN_INPUTS = ("rows.jsonl", "readouts.json", "captures.json", "auth_rows.jsonl", "auth_split.json")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


def is_immutable_revision(revision: Any) -> bool:
    """A Hugging Face commit hash (40 hex chars); ``None``/``main``/tags are mutable."""

    return isinstance(revision, str) and bool(_HEX40.match(revision.strip().lower()))


def require_model_revision(revision: Any, *, allow_mutable: bool) -> str | None:
    """Fail closed on a mutable model revision unless the caller explicitly allowed it.

    Returns the normalised revision (lower-case hash) or, only with
    ``allow_mutable``, the revision as given (``None`` = repo head).
    """

    if is_immutable_revision(revision):
        return revision.strip().lower()
    if allow_mutable:
        return revision if revision else None
    raise RuntimeError(
        f"model revision {revision!r} is not an immutable commit hash. New runs must pin the model to a "
        "40-hex Hugging Face commit (--model-revision <sha>, see the model's /commits page); pass "
        "--allow-mutable-revision only for a run you are willing to label as unpinned (the flag is recorded)."
    )


def frozen_input_hashes(data_dir: str | os.PathLike[str], names: tuple[str, ...] = FROZEN_INPUTS) -> dict[str, str]:
    """sha256 of every frozen input file the collector/analysis reads (all must exist)."""

    data_dir = Path(data_dir)
    out = {}
    for name in names:
        path = data_dir / name
        if not path.is_file():
            raise RuntimeError(f"frozen input {path} is missing; run prepare.py for this run version first")
        out[name] = sha256_file(path)
    return out


def check_manifest_describes_inputs(data_manifest: dict[str, Any], hashes: dict[str, str]) -> None:
    """The data manifest's ``files`` block must agree with the bytes on disk."""

    listed = data_manifest.get("files", {}) or {}
    bad = [name for name, sha in hashes.items() if name in listed and listed[name] != sha]
    if bad:
        raise RuntimeError(f"data manifest.json sha256 disagrees with the files on disk: {', '.join(bad)}")


def collection_identity(
    *,
    model: str,
    model_revision: str | None,
    allow_mutable_revision: bool,
    code: dict[str, Any],
    data_files_sha256: dict[str, str],
    image_pins: list[str] | tuple[str, ...],
    gpu_requested: str,
    gpu_name: str | None,
    versions: dict[str, Any],
    run_version: str,
    overrides: dict[str, Any],
    config_hash_value: str | None = None,
) -> dict[str, Any]:
    """Everything that determines what a collection run produces (sorted, JSON-safe)."""

    return {
        "schema": "exp7-collection-identity/1",
        "run_version": run_version,
        "config_hash": config_hash() if config_hash_value is None else config_hash_value,
        "model": model,
        "model_revision": model_revision,
        "allow_mutable_revision": bool(allow_mutable_revision),
        "code_commit": code.get("commit"),
        "code_dirty": code.get("dirty"),
        "data_files_sha256": dict(sorted(data_files_sha256.items())),
        "image_pins": sorted(image_pins),
        "gpu_requested": gpu_requested,
        "gpu_name": gpu_name,
        "versions": {k: versions.get(k) for k in ("python", "torch", "transformers", "accelerate", "cuda")},
        "overrides": dict(sorted(overrides.items())),
    }


def identity_hash(identity: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    """``{"a": {"b": 1}} -> {"a.b": 1}`` so mismatches can be named field by field."""

    if not isinstance(d, dict):
        return {prefix or "": d}
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def identity_mismatches(shard_identity: dict[str, Any], current: dict[str, Any]) -> list[str]:
    a, b = flatten(shard_identity), flatten(current)
    problems = []
    for key in sorted(set(a) | set(b)):
        if a.get(key, "<absent>") != b.get(key, "<absent>"):
            problems.append(f"{key}: shard {a.get(key, '<absent>')!r} != current {b.get(key, '<absent>')!r}")
    return problems


def resume_mismatches(
    shard_fields: dict[str, Any],
    *,
    model: str,
    expected_config_hash: str | None = None,
    identity: dict[str, Any] | None = None,
) -> list[str]:
    """Why an existing shard may NOT be skipped on resume (empty list = compatible).

    Shards without a ``config_hash`` predate the stamp (the frozen v0 run);
    they are never silently reused by a collector with a different config.
    When ``identity`` (the running collector's ``collection_identity``) is
    given, the shard must carry ``identity_json`` and every field of it must
    match — model revision, input hashes, code commit, pins, GPU, overrides.
    """

    expected_config_hash = config_hash() if expected_config_hash is None else expected_config_hash
    problems = []
    if str(shard_fields.get("model")) != model:
        problems.append(f"model {shard_fields.get('model')!r} != {model!r}")
    found = shard_fields.get("config_hash")
    if found is None:
        problems.append("legacy shard without config_hash (frozen run); collect into a new run version instead")
    elif str(found) != expected_config_hash:
        problems.append(f"config_hash {str(found)[:12]} != current {expected_config_hash[:12]}")
    if identity is not None:
        raw = shard_fields.get("identity_json")
        if raw is None:
            problems.append("shard carries no collection identity (identity_json); collect into a new run version instead")
        else:
            try:
                shard_identity = json.loads(str(raw))
            except json.JSONDecodeError:
                shard_identity = {"<unparseable>": str(raw)[:40]}
            problems += identity_mismatches(shard_identity, identity)
            stamped = shard_fields.get("identity_hash")
            if stamped is not None and str(stamped) != identity_hash(identity) and not problems:
                problems.append(f"identity_hash {str(stamped)[:12]} != current {identity_hash(identity)[:12]}")
    return problems


# ----------------------------------------------------------------------------- inventory preflight
def check_inventory(
    records: list[dict[str, Any]],
    *,
    data_dir: str | os.PathLike[str],
    acts_dir: str | os.PathLike[str] | None,
    expected_shards: dict[str, int],
    required_inputs: tuple[str, ...] = FROZEN_INPUTS + ("manifest.json",),
) -> dict[str, Any]:
    """Fail closed unless every expected shard and committed input is present and matches.

    ``records`` is the parsed ``shards.sha256``. Entries under ``acts/<sub>/``
    resolve under ``acts_dir``; result entries (``.../results/...``) are the
    outputs being regenerated and are not checked here; every other entry is a
    frozen input resolved as ``data_dir/<basename>``. Raises ``RuntimeError``
    naming what is wrong; returns counts + aggregate hash when everything is in order.
    """

    data_dir = Path(data_dir)
    problems: list[str] = []
    shards: dict[str, list[dict[str, Any]]] = {sub: [] for sub in expected_shards}
    inputs: list[dict[str, Any]] = []
    for rec in records:
        path = rec["path"]
        if path.startswith(ACTS_PREFIX + "/"):
            sub = path.split("/")[1]
            if sub in shards:
                shards[sub].append(rec)
            else:
                problems.append(f"unexpected shard group in manifest: {path}")
        elif "/results/" in path or path.startswith("results/"):
            continue
        else:
            inputs.append(rec)
    for sub, expected in expected_shards.items():
        if len(shards[sub]) != expected:
            problems.append(f"manifest lists {len(shards[sub])} {sub} shards, {expected} expected")
    listed_inputs = {Path(r["path"]).name for r in inputs}
    for name in required_inputs:
        if name not in listed_inputs:
            problems.append(f"committed input {name} is not listed in the manifest")

    checked: list[dict[str, Any]] = []
    missing: list[str] = []
    mismatch: list[str] = []
    for rec in inputs + [r for sub in expected_shards for r in shards[sub]]:
        if rec["path"].startswith(ACTS_PREFIX + "/"):
            local = None if acts_dir is None else Path(acts_dir) / rec["path"][len(ACTS_PREFIX) + 1 :]
        else:
            local = data_dir / Path(rec["path"]).name
        if local is None or not local.is_file():
            missing.append(rec["path"])
            continue
        if local.stat().st_size != rec["bytes"] or sha256_file(local) != rec["sha256"]:
            mismatch.append(rec["path"])
            continue
        checked.append(rec)
    if acts_dir is None:
        problems.append("no shard directory given; the off-git activation shards cannot be verified")
    if missing:
        problems.append(f"{len(missing)} listed file(s) missing locally, e.g. {', '.join(missing[:3])}")
    if mismatch:
        problems.append(f"{len(mismatch)} listed file(s) differ from the manifest, e.g. {', '.join(mismatch[:3])}")
    if problems:
        raise RuntimeError(
            "provenance preflight failed — analysis refuses to run on an incomplete or altered mirror:\n  - "
            + "\n  - ".join(problems)
            + "\n(pass --skip-provenance-check only for a deliberately partial run; the flag is recorded in results.json)"
        )
    acts_records = [r for sub in expected_shards for r in shards[sub]]
    return {
        "checked": True,
        "n_shards": {sub: len(v) for sub, v in shards.items()},
        "n_inputs": len(inputs),
        "aggregate_sha256_shards": aggregate_sha256(acts_records),
        "aggregate_sha256_inputs": aggregate_sha256(inputs),
    }
