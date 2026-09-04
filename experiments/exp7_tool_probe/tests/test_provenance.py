"""Provenance + collector pure functions: hashing, manifests, seeds, resume checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.exp7_tool_probe import config, provenance as prov

PKG = Path(__file__).resolve().parents[1]


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_sha256_manifest_round_trip_and_verification(tmp_path) -> None:
    repo = tmp_path / "repo"
    acts = tmp_path / "acts"
    _write(repo / "data" / "a.json", b'{"a": 1}\n')
    _write(repo / "data" / "b.txt", b"bbb")
    _write(acts / "main" / "t00.npz", b"\x00" * 100)
    _write(acts / "auth" / "auth_real_000.npz", b"\x01" * 50)
    records = prov.collect_records(repo / "data", prefix="data")
    records += prov.collect_records(acts / "main", prefix="acts/main")
    records += prov.collect_records(acts / "auth", prefix="acts/auth")
    assert [r["path"] for r in records] == ["data/a.json", "data/b.txt", "acts/main/t00.npz", "acts/auth/auth_real_000.npz"]
    assert records[2]["bytes"] == 100 and records[2]["sha256"] == prov.sha256_file(acts / "main" / "t00.npz")

    manifest = tmp_path / "shards.sha256"
    prov.write_sha256_manifest(manifest, records, header=["hello"])
    text = manifest.read_text()
    assert text.startswith("# hello\n")
    back = prov.read_sha256_manifest(manifest)
    assert sorted(back, key=lambda r: r["path"]) == sorted(records, key=lambda r: r["path"])

    report = prov.verify_records(back, repo_root=repo, acts_dir=acts)
    assert {k: len(v) for k, v in report.items()} == {"ok": 4, "mismatch": 0, "missing": 0}

    # tamper with a shard, delete a data file, and hide the acts dir
    _write(acts / "main" / "t00.npz", b"\x00" * 99)
    (repo / "data" / "b.txt").unlink()
    report = prov.verify_records(back, repo_root=repo, acts_dir=acts)
    assert [r["path"] for r in report["mismatch"]] == ["acts/main/t00.npz"]
    assert report["mismatch"][0]["local_bytes"] == 99
    assert [r["path"] for r in report["missing"]] == ["data/b.txt"]
    report = prov.verify_records(back, repo_root=repo, acts_dir=None)
    assert sorted(r["path"] for r in report["missing"]) == ["acts/auth/auth_real_000.npz", "acts/main/t00.npz", "data/b.txt"]

    with pytest.raises(ValueError):
        prov.read_sha256_manifest(_write(tmp_path / "bad.sha256", b"deadbeef  12  x\n"))


def test_aggregate_sha256_is_order_independent_and_content_sensitive() -> None:
    a = [{"path": "x", "sha256": "1" * 64, "bytes": 1}, {"path": "y", "sha256": "2" * 64, "bytes": 2}]
    assert prov.aggregate_sha256(a) == prov.aggregate_sha256(list(reversed(a)))
    b = [dict(a[0]), {**a[1], "bytes": 3}]
    assert prov.aggregate_sha256(a) != prov.aggregate_sha256(b)


def test_config_hash_is_stable_and_field_sensitive() -> None:
    assert config.config_hash() == config.config_hash()
    assert len(config.config_hash()) == 64
    fields = config.config_fields()
    assert fields["seed"] == config.SEED == 7
    assert fields["pinned_block"] == 50836993 and fields["pinned_balances"] == {"REAL": "1.900000", "SHAM": "0.000000"}
    assert config.config_hash({**fields, "temperature": 0.8}) != config.config_hash(fields)
    assert config.config_hash(fields) == config.config_hash()


def test_derive_seed_is_deterministic_distinct_and_31_bit() -> None:
    seeds = {config.derive_seed("decision", t, w) for t in range(48) for w in ("REAL", "SHAM")}
    assert len(seeds) == 96
    assert all(0 <= s < 2**31 for s in seeds)
    assert config.derive_seed("auth", "auth_real_000") == config.derive_seed("auth", "auth_real_000")
    assert config.derive_seed("auth", "auth_real_000", base=8) != config.derive_seed("auth", "auth_real_000")
    assert config.derive_seed("stageA", 3) != config.derive_seed("decision", 3, "REAL")


def test_seed_everything_makes_python_and_numpy_draws_repeatable() -> None:
    import random

    config.seed_everything(11)
    a = (random.random(), float(np.random.rand()))
    config.seed_everything(11)
    assert a == (random.random(), float(np.random.rand()))
    assert config.seed_everything()["seed"] == config.SEED


def test_resume_mismatches_fail_closed_on_legacy_and_changed_config() -> None:
    model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    ok = {"model": model, "config_hash": config.config_hash()}
    assert prov.resume_mismatches(ok, model=model) == []
    legacy = prov.resume_mismatches({"model": model}, model=model)
    assert len(legacy) == 1 and "legacy" in legacy[0]
    changed = prov.resume_mismatches({"model": model, "config_hash": "0" * 64}, model=model)
    assert len(changed) == 1 and "config_hash" in changed[0]
    wrong_model = prov.resume_mismatches({"model": "other", "config_hash": config.config_hash()}, model=model)
    assert len(wrong_model) == 1 and "model" in wrong_model[0]


def test_shard_provenance_fields_round_trip_through_npz(tmp_path) -> None:
    p = prov.shard_provenance(
        code={"commit": "abc", "dirty": False},
        model="m",
        model_revision="rev",
        gpu_requested="A100-80GB:2",
        versions={"torch": "2.14.0"},
        seeds={"stage_a": 1, "decision_REAL": 2},
        run_version="v1",
        extra={"block": 5},
    )
    assert p["config_hash"] == config.config_hash() and p["seed"] == config.SEED and p["block"] == 5
    fields = prov.npz_provenance_fields(p)
    path = tmp_path / "t.npz"
    np.savez(path, x=np.zeros(2), **fields)
    with np.load(path) as z:
        assert str(z["config_hash"]) == config.config_hash()
        assert int(z["seed"]) == config.SEED
        back = json.loads(str(z["provenance_json"]))
    assert back == p


def test_runtime_versions_and_git_commit_do_not_raise(tmp_path) -> None:
    info = prov.runtime_versions()
    assert info["numpy"] == np.__version__ and "python" in info
    assert prov.git_commit(tmp_path) == {"commit": None, "dirty": None}
    here = prov.git_commit(PKG)
    assert here["commit"] is None or len(here["commit"]) == 40


def test_verify_provenance_cli_fails_closed_without_the_shards(capsys) -> None:
    from experiments.exp7_tool_probe import verify_provenance as vp

    if not vp.MANIFEST.is_file():
        pytest.skip("shards.sha256 not written yet")
    # strict by default: the 168 off-git shards are missing here -> exit 2, never 0
    code = vp.main(["--json"])
    out = json.loads(capsys.readouterr().out)
    assert code == 2 and out["strict"] is True and out["exit_code"] == 2
    assert out["mismatch"] == 0 and out["mismatch_paths"] == []  # committed inputs/results match the manifest
    assert out["missing"] == 168 and all(p.startswith("acts/") for p in out["missing_paths"])
    assert out["inventory_problems"] == []
    # the old lenient mode has to be asked for by name
    code = vp.main(["--json", "--report-only"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["strict"] is False and out["missing"] == 168
    with pytest.raises(SystemExit):
        vp.main(["--strict", "--report-only"])
    assert vp.expected_shards(vp.DATA) == {"main": 48, "auth": 120}
    assert vp.run_paths("v1")["manifest"].as_posix().endswith("data/v1/shards.sha256")


def test_verify_exit_codes_and_inventory_rules() -> None:
    from experiments.exp7_tool_probe import verify_provenance as vp

    ok = {"ok": [{}], "mismatch": [], "missing": []}
    missing = {"ok": [], "mismatch": [], "missing": [{}]}
    mismatch = {"ok": [], "mismatch": [{}], "missing": []}
    assert vp._exit_code(ok, [], report_only=False) == 0
    assert vp._exit_code(missing, [], report_only=False) == 2
    assert vp._exit_code(missing, [], report_only=True) == 0
    assert vp._exit_code(mismatch, [], report_only=True) == 1
    assert vp._exit_code(ok, ["short"], report_only=False) == 2
    records = [{"path": f"acts/main/t{i:02d}.npz", "sha256": "0" * 64, "bytes": 1} for i in range(47)]
    assert vp.inventory_problems(records, {"main": 48, "auth": 0}) == ["manifest lists 47 main shards, 48 expected"]


def _fake_identity(**over):
    base = dict(
        model="m",
        model_revision="a" * 40,
        allow_mutable_revision=False,
        code={"commit": "c" * 40, "dirty": False},
        data_files_sha256={name: "1" * 64 for name in prov.FROZEN_INPUTS},
        image_pins=["torch==2.14.0", "transformers==5.16.1"],
        gpu_requested="A100-80GB:2",
        gpu_name="NVIDIA A100-SXM4-80GB",
        versions={"python": "3.11.9", "torch": "2.14.0", "transformers": "5.16.1", "accelerate": "1.14.0", "cuda": "12.8"},
        run_version="v1",
        overrides={
            "block": 50900000,
            "expected_balances": {"REAL": "0.500000", "SHAM": "0.000000"},
            "allow_drift": False,
            "n_rollouts_per_row": 8,
            "temperature": 0.7,
            "max_new_decision": 200,
            "max_new_tool_call": 96,
            "max_new_auth": 120,
            "seed": 7,
        },
    )
    base.update(over)
    return prov.collection_identity(**base)


def test_collection_identity_hash_is_stable_and_covers_every_field() -> None:
    a, b = _fake_identity(), _fake_identity()
    assert a == b and prov.identity_hash(a) == prov.identity_hash(b) and len(prov.identity_hash(a)) == 64
    assert a["config_hash"] == config.config_hash() and a["schema"] == "exp7-collection-identity/1"
    assert a["versions"] == {"python": "3.11.9", "torch": "2.14.0", "transformers": "5.16.1", "accelerate": "1.14.0", "cuda": "12.8"}
    flat = prov.flatten(a)
    for key in ("model", "model_revision", "code_commit", "data_files_sha256.rows.jsonl", "gpu_requested", "gpu_name", "versions.torch",
                "overrides.block", "overrides.expected_balances.REAL", "overrides.n_rollouts_per_row", "overrides.temperature",
                "overrides.max_new_decision", "overrides.seed", "allow_mutable_revision", "config_hash", "run_version"):
        assert key in flat, key
    assert prov.identity_mismatches(a, b) == []
    assert prov.identity_mismatches(a, {**a, "extra": 1}) == ["extra: shard '<absent>' != current 1"]


def test_resume_refuses_every_identity_change_and_shards_without_one() -> None:
    ident = _fake_identity()
    fields = {"model": "m", "config_hash": config.config_hash(), "identity_json": json.dumps(ident, sort_keys=True), "identity_hash": prov.identity_hash(ident)}
    assert prov.resume_mismatches(fields, model="m", identity=ident) == []
    changed = {
        "model_revision": _fake_identity(model_revision="b" * 40),
        "code_commit": _fake_identity(code={"commit": "d" * 40, "dirty": False}),
        "code_dirty": _fake_identity(code={"commit": "c" * 40, "dirty": True}),
        "data_files_sha256.rows.jsonl": _fake_identity(data_files_sha256={**{n: "1" * 64 for n in prov.FROZEN_INPUTS}, "rows.jsonl": "2" * 64}),
        "data_files_sha256.readouts.json": _fake_identity(data_files_sha256={**{n: "1" * 64 for n in prov.FROZEN_INPUTS}, "readouts.json": "2" * 64}),
        "image_pins": _fake_identity(image_pins=["torch==2.13.0", "transformers==5.16.1"]),
        "gpu_requested": _fake_identity(gpu_requested="H100:1"),
        "gpu_name": _fake_identity(gpu_name="NVIDIA H100"),
        "versions.torch": _fake_identity(versions={"python": "3.11.9", "torch": "2.13.0", "transformers": "5.16.1", "accelerate": "1.14.0", "cuda": "12.8"}),
        "allow_mutable_revision": _fake_identity(allow_mutable_revision=True),
        "run_version": _fake_identity(run_version="v2"),
    }
    base_over = _fake_identity()["overrides"]
    for key in base_over:
        new = dict(base_over)
        new[key] = {"REAL": "1.000000", "SHAM": "0.000000"} if key == "expected_balances" else (not new[key] if isinstance(new[key], bool) else new[key] + 1)
        changed[f"overrides.{key}"] = _fake_identity(overrides=new)
    for key, current in changed.items():
        problems = prov.resume_mismatches(fields, model="m", identity=current)
        assert problems and any(p.startswith(key) for p in problems), (key, problems)
    # hash alone differing (same fields) is still refused
    fields_bad_hash = {**fields, "identity_hash": "0" * 64}
    assert any("identity_hash" in p for p in prov.resume_mismatches(fields_bad_hash, model="m", identity=ident))
    # shards without an identity (the frozen v0 run) are never resumed onto
    legacy = prov.resume_mismatches({"model": "m", "config_hash": config.config_hash()}, model="m", identity=ident)
    assert any("no collection identity" in p for p in legacy)
    garbage = prov.resume_mismatches({**fields, "identity_json": "not json"}, model="m", identity=ident)
    assert garbage


def test_model_revision_must_be_immutable_unless_waived() -> None:
    sha = "0123456789abcdef0123456789abcdef01234567"
    assert prov.is_immutable_revision(sha) and prov.is_immutable_revision(sha.upper())
    for bad in (None, "", "main", "v1.0", "refs/pr/1", sha[:-1], sha + "0"):
        assert not prov.is_immutable_revision(bad)
        with pytest.raises(RuntimeError, match="immutable"):
            prov.require_model_revision(bad, allow_mutable=False)
    assert prov.require_model_revision(sha.upper(), allow_mutable=False) == sha
    assert prov.require_model_revision("main", allow_mutable=True) == "main"
    assert prov.require_model_revision(None, allow_mutable=True) is None
    assert prov.require_model_revision("", allow_mutable=True) is None


def test_frozen_input_hashes_and_manifest_agreement(tmp_path) -> None:
    for name in prov.FROZEN_INPUTS:
        _write(tmp_path / name, name.encode())
    hashes = prov.frozen_input_hashes(tmp_path)
    assert set(hashes) == set(prov.FROZEN_INPUTS) and all(len(v) == 64 for v in hashes.values())
    prov.check_manifest_describes_inputs({"files": dict(hashes)}, hashes)
    prov.check_manifest_describes_inputs({}, hashes)
    with pytest.raises(RuntimeError, match="rows.jsonl"):
        prov.check_manifest_describes_inputs({"files": {**hashes, "rows.jsonl": "0" * 64}}, hashes)
    (tmp_path / "captures.json").unlink()
    with pytest.raises(RuntimeError, match="captures.json"):
        prov.frozen_input_hashes(tmp_path)


def _inventory_fixture(tmp_path, n_main=4, n_auth=4):
    data = tmp_path / "data"
    acts = tmp_path / "acts"
    for name in prov.FROZEN_INPUTS + ("manifest.json",):
        _write(data / name, name.encode())
    for i in range(n_main):
        _write(acts / "main" / f"t{i:02d}.npz", bytes([i]) * 10)
    for i in range(n_auth):
        _write(acts / "auth" / f"auth_real_{i:03d}.npz", bytes([100 + i]) * 10)
    records = prov.collect_records(data, prefix="experiments/exp7_tool_probe/data/v9")
    records += prov.collect_records(acts / "main", prefix="acts/main")
    records += prov.collect_records(acts / "auth", prefix="acts/auth")
    records += [{"path": "experiments/exp7_tool_probe/results/v9/results.json", "sha256": "0" * 64, "bytes": 1}]  # outputs: not preflighted
    return data, acts, records


def test_check_inventory_passes_only_on_a_complete_matching_mirror(tmp_path) -> None:
    data, acts, records = _inventory_fixture(tmp_path)
    expected = {"main": 4, "auth": 4}
    summary = prov.check_inventory(records, data_dir=data, acts_dir=acts, expected_shards=expected)
    assert summary["checked"] is True and summary["n_shards"] == expected and summary["n_inputs"] == 6
    assert len(summary["aggregate_sha256_shards"]) == 64
    # 3/4 mirror (one shard deleted) -> refused, naming the count
    (acts / "main" / "t03.npz").unlink()
    with pytest.raises(RuntimeError, match=r"1 listed file\(s\) missing locally, e.g. acts/main/t03.npz"):
        prov.check_inventory(records, data_dir=data, acts_dir=acts, expected_shards=expected)
    # a manifest that itself lists too few shards -> refused
    short = [r for r in records if r["path"] != "acts/main/t03.npz"]
    with pytest.raises(RuntimeError, match="manifest lists 3 main shards, 4 expected"):
        prov.check_inventory(short, data_dir=data, acts_dir=acts, expected_shards=expected)
    # altered shard bytes -> refused
    _write(acts / "main" / "t03.npz", b"\x03" * 11)
    with pytest.raises(RuntimeError, match=r"differ from the manifest, e.g. acts/main/t03.npz"):
        prov.check_inventory(records, data_dir=data, acts_dir=acts, expected_shards=expected)
    _write(acts / "main" / "t03.npz", b"\x03" * 10)
    prov.check_inventory(records, data_dir=data, acts_dir=acts, expected_shards=expected)
    # altered committed input -> refused; no shard dir at all -> refused
    _write(data / "rows.jsonl", b"changed")
    with pytest.raises(RuntimeError, match="rows.jsonl"):
        prov.check_inventory(records, data_dir=data, acts_dir=acts, expected_shards=expected)
    _write(data / "rows.jsonl", b"rows.jsonl")
    with pytest.raises(RuntimeError, match="no shard directory"):
        prov.check_inventory(records, data_dir=data, acts_dir=None, expected_shards=expected)
    with pytest.raises(RuntimeError, match="manifest.json is not listed"):
        prov.check_inventory([r for r in records if not r["path"].endswith("manifest.json")], data_dir=data, acts_dir=acts, expected_shards=expected)


def test_modal_collector_pure_helpers() -> None:
    pytest.importorskip("modal")
    from experiments.exp7_tool_probe import modal_collect as mc

    assert mc._strip_end("SPEND now<|im_end|>junk") == "SPEND now"
    assert mc._strip_end("plain") == "plain"
    row = np.array([1, 2, 3, 9, 4, 9, 0])
    assert mc._trim_generation(row, 3, {9}).tolist() == [1, 2, 3, 9]
    assert mc._trim_generation(np.array([1, 2, 3]), 3, {9}).tolist() == [1, 2, 3]
    assert mc._ids_sha256([1, 2, 3]) == mc._ids_sha256((1, 2, 3)) != mc._ids_sha256([1, 2, 4])
    assert mc.OUT_ROOT.endswith("/" + mc.RUN_VERSION)
    assert mc.MAX_NEW == config.MAX_NEW_DECISION and mc.TEMP == config.TEMPERATURE
    assert all("==" in pin for pin in mc.IMAGE_PINS)
    assert mc._code_identity()["commit"] is None or len(mc._code_identity()["commit"]) == 40


def test_run_version_derives_every_collection_path(monkeypatch) -> None:
    pytest.importorskip("modal")
    import importlib

    from experiments.exp7_tool_probe import modal_collect as mc

    paths = mc.run_paths("v1")
    assert paths["data"].endswith("/experiments/exp7_tool_probe/data/v1")
    assert paths["remote_data"] == "/data/exp7_v1" and paths["out_root"] == "/acts/exp7/v1" and paths["volume_path"] == "exp7/v1"
    assert all("v1" in v for v in paths.values()) and not any("v0" in v for v in paths.values())
    for bad in ("", "0", "../v0", "v0/../v1", "prod"):
        with pytest.raises(ValueError):
            mc.run_paths(bad)
    # the module constants the collector actually uses follow EXP7_RUN_VERSION too
    monkeypatch.setenv("EXP7_RUN_VERSION", "v1")
    try:
        mod = importlib.reload(mc)
        derived = {name: getattr(mod, name) for name in ("DATA", "REMOTE_DATA", "OUT_ROOT")}
        assert derived["DATA"].endswith("/data/v1") and derived["REMOTE_DATA"] == "/data/exp7_v1" and derived["OUT_ROOT"] == "/acts/exp7/v1"
        assert not any("v0" in v for v in derived.values()), derived
        assert mod.RUN_VERSION == "v1"
    finally:
        monkeypatch.delenv("EXP7_RUN_VERSION", raising=False)
        mod = importlib.reload(mc)
    assert mod.RUN_VERSION == "v0" and mod.DATA.endswith("/data/v0") and mod.REMOTE_DATA == "/data/exp7_v0" and mod.OUT_ROOT == "/acts/exp7/v0"


def test_submit_cli_requires_an_immutable_revision_unless_waived(monkeypatch) -> None:
    pytest.importorskip("modal")
    from experiments.exp7_tool_probe import modal_collect as mc

    monkeypatch.setattr(mc, "MODEL_REVISION", None)
    sha = "0123456789abcdef0123456789abcdef01234567"
    args = mc.parse_cli(["submit", "main", "--model-revision", sha.upper()])
    assert args.cmd == "submit" and args.which == "main" and args.model_revision == sha and args.allow_mutable_revision is False
    with pytest.raises(RuntimeError, match="immutable"):
        mc.parse_cli(["submit", "main"])
    with pytest.raises(RuntimeError, match="immutable"):
        mc.parse_cli(["submit", "auth", "--model-revision", "main"])
    args = mc.parse_cli(["submit", "auth", "--model-revision", "main", "--allow-mutable-revision"])
    assert args.model_revision == "main" and args.allow_mutable_revision is True
    args = mc.parse_cli(["submit", "auth", "--allow-mutable-revision"])
    assert args.model_revision is None and args.allow_mutable_revision is True
    monkeypatch.setattr(mc, "MODEL_REVISION", sha)  # EXP7_MODEL_REVISION baked at deploy time is the default
    assert mc.parse_cli(["submit", "main"]).model_revision == sha
    poll = mc.parse_cli(["poll", "fc-123", "30"])
    assert poll.cmd == "poll" and poll.call_id == "fc-123" and poll.timeout == 30.0
    with pytest.raises(SystemExit):
        mc.parse_cli(["submit", "nope"])
