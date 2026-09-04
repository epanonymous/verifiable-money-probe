"""analysis.py on tiny synthetic shards (same npz schema as the collector), in seconds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.exp7_tool_probe import analysis, dataset
from experiments.exp7_tool_probe.context import CANONICAL_TOOL_CALL

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "recorded_balances.json").read_text())
MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
L, D, N_ROLL = 4, 6, 3  # layers incl. embeddings, width, rollouts


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(analysis, "N_JOBS", 1)
    monkeypatch.setattr(analysis, "N_BOOT", 40)


def _main_shard(path: Path, tid: int, split: str, rng: np.random.Generator) -> None:
    base = rng.normal(size=(L, D)).astype(np.float32)
    data = {}
    for world, label in (("REAL", 1), ("SHAM", 0)):
        w = world.lower()
        sign = 1.0 if label else -1.0
        p0 = base + rng.normal(scale=1e-3, size=(L, D)).astype(np.float32)  # numerics only
        p1 = base + rng.normal(scale=0.3, size=(L, D)).astype(np.float32)
        p1[2, 0] += 8.0 * sign  # the world lives in layer 2, dim 0
        decisions = ["SPEND", "SPEND", "HOLD"] if label else ["HOLD"] * N_ROLL
        resp = np.stack([p1 + rng.normal(scale=0.3, size=(L, D)) for _ in range(N_ROLL)]).astype(np.float32)
        for k, d in enumerate(decisions):
            # the decision shifts the post-decision state along the world direction (as in v0)
            resp[k, 2, 0] += 3.0 if d == "SPEND" else -3.0
            resp[k, 3, 1] += 3.0 if d == "SPEND" else -3.0
        data.update(
            {
                f"{w}_p0": p0.astype(np.float16),
                f"{w}_p1": p1.astype(np.float16),
                f"{w}_p1_spread": np.float32(0.0),
                f"{w}_resp": resp.astype(np.float16),
                f"{w}_texts": np.array([f"{d}. reason {tid}" for d in decisions]),
                f"{w}_decisions": np.array(decisions),
                f"{w}_p1_ids": np.arange(10 + tid),
                f"{w}_p1_len": 10 + tid,
                f"{w}_tool_text": FIXTURE[w]["tool_text"],
                f"{w}_row_id": f"{w}_probe_{tid:02d}",
                f"{w}_label": label,
            }
        )
    data.update(
        p0_ids=np.arange(8 + tid),
        p0_len=8 + tid,
        stage_a_text=CANONICAL_TOOL_CALL,
        tool_call_turn=CANONICAL_TOOL_CALL,
        tool_call_source="model",
        template_id=tid,
        split=split,
        block=FIXTURE["real"]["block"],
        n_rollouts=N_ROLL,
        model=MODEL,
        gpu="A100-80GB:2",
    )
    np.savez_compressed(path, **data)


def _auth_shard(path: Path, index: int, kind: str, label: int, capture_kind: str, split: str, rng: np.random.Generator) -> None:
    sign = 1.0 if label else -1.0
    prompt = rng.normal(size=(1, L, D)).astype(np.float32)
    prompt[0, 1, 2] += 3.0 * sign
    resp = rng.normal(size=(1, L, D)).astype(np.float32)
    np.savez_compressed(
        path,
        prompt_final=prompt.astype(np.float16),
        response_final=resp.astype(np.float16),
        texts=np.array(["AUTHENTIC because." if label else "FABRICATED because."]),
        verdicts=np.array(["AUTHENTIC" if label else "FABRICATED"]),
        row_id=f"auth_{kind}_{index:03d}",
        kind=kind,
        method="eth_x",
        capture_kind=capture_kind,
        template_id=index,
        split=split,
        label=label,
        prompt_len=50,
        model=MODEL,
        gpu="A100-80GB:2",
    )


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("exp7")
    acts = root / "acts"
    (acts / "main").mkdir(parents=True)
    (acts / "auth").mkdir()
    rng = np.random.default_rng(0)
    templates = dataset.build_templates(n_train=6, n_heldout=2)
    for t in templates:
        _main_shard(acts / "main" / f"t{t.template_id:02d}.npz", t.template_id, t.split, rng)
    kinds = ["a"] * 6 + ["b"] * 4
    tail = ["train"] * 8 + ["heldout"] * 2  # the confounded scheme: both held-out pairs are kind b
    strat = ["heldout", "train", "train", "train", "train", "train", "train", "heldout", "train", "train"]
    for i, kind in enumerate(kinds):
        for role, label in (("real", 1), ("forged", 0)):
            _auth_shard(acts / "auth" / f"auth_{role}_{i:03d}.npz", i, role, label, kind, tail[i], rng)
    data = root / "data"
    data.mkdir()
    rows = dataset.build_rows(templates=templates)
    dataset.write_jsonl(data / "rows.jsonl", rows)
    (data / "readouts.json").write_text(json.dumps({w.upper(): {"text": FIXTURE[w]["tool_text"]} for w in ("real", "sham")}))
    (data / "auth_split.json").write_text(json.dumps({"split": strat, "kinds": kinds}))
    (data / "manifest.json").write_text(
        json.dumps({"block": FIXTURE["real"]["block"], "files": {}, "collection": {"code_commit": "deadbeef"}, "amendments": {}, "n_templates": 8, "n_auth_rows": 20})
    )
    return {"acts": acts, "data": data, "root": root}


SKIP = ["--skip-provenance-check"]  # the synthetic fixture has no shards.sha256; the flag is recorded


def test_end_to_end_on_synthetic_shards(synthetic) -> None:
    out = synthetic["root"] / "out1"
    rc = analysis.main(["--acts", str(synthetic["acts"]), "--out", str(out), "--data", str(synthetic["data"]), "--n-shuffles", "2", *SKIP])
    assert rc == 0
    for name in ("results.json", "summary.md", "blind_p0.json", "fig_layers.png", "fig_decisions.png", "transcripts.jsonl"):
        assert (out / name).is_file(), name
    res = json.loads((out / "results.json").read_text())
    main = res["main"]
    assert main["n_rows"] == 16 and main["n_templates"] == 8 and main["n_layers_incl_embed"] == L
    assert main["n_train_templates"] == 6 and main["n_heldout_templates"] == 2
    assert main["tool_call_source_rate"] == {"model": 1.0, "scripted_fallback": 0.0}
    assert main["tool_text"]["REAL"] != main["tool_text"]["SHAM"]
    # P0: numerics-only differences, probe at chance (identical pair texts -> tied scores)
    assert main["p0_pair_layer_relnorm_max"] < 0.05 and main["p0_pair_cos_min"] > 0.99  # float16 + 1e-3 noise
    assert main["p0"]["loto_auroc_max"] <= main["p0"]["alarm_auroc"]
    # P1: the planted layer wins and generalizes
    assert main["p1"]["selected_layer"] == 2
    assert main["p1"]["heldout_auroc_at_selected"] == 1.0 and main["p1"]["loto_auroc_at_selected"] >= 0.95
    assert main["p1"]["n_heldout_rows"] == 4
    # decisions and index bookkeeping
    d = main["decisions"]
    assert d["n_rollouts_total"] == 16 * N_ROLL
    assert d["spend_rate"]["REAL"] == pytest.approx(2 / 3) and d["spend_rate"]["SHAM"] == 0.0
    dp = main["decision_prediction"]
    assert dp["n_rollouts_used"] == 48
    assert dp["p1_probe_oof_score"]["within_REAL"]["n"] == 24 and dp["p1_probe_oof_score"]["within_REAL"]["n_spend"] == 16
    assert dp["p1_probe_oof_score"]["within_REAL"]["n_unique_scores"] == 8  # one P1 score per REAL row
    assert np.isnan(dp["p1_probe_oof_score"]["within_SHAM"]["auroc"])
    # response_final carries the planted decision direction inside REAL
    assert dp["response_final_probe_oof_score"]["within_REAL"]["auroc"] > 0.8
    # 16 SPEND (all REAL) vs 32 HOLD (8 REAL, 24 SHAM): (16*24 + 0.5*16*8) / (16*32)
    assert dp["text_baseline_world_label"]["auroc"] == pytest.approx(0.875)
    # V5: primary split is the stratified override, superseded tail split reported alongside
    auth = res["auth"]
    assert auth["split_counts_pairs_by_kind"] == {"train": {"a": 5, "b": 3}, "heldout": {"a": 1, "b": 1}}
    assert auth["prompt_final"]["selected_layer"] == 1
    assert auth["prompt_final"]["heldout_auroc_at_selected"] == 1.0
    assert set(auth["prompt_final"]["heldout_by_capture_kind"]) == {"a", "b"}
    tail = res["auth_tail_split_superseded"]
    assert tail["split_counts_pairs_by_kind"] == {"train": {"a": 6, "b": 2}, "heldout": {"b": 2}}
    assert tail["prompt_final"]["lopo_auroc_at_selected"] == auth["prompt_final"]["lopo_auroc_at_selected"]
    assert "lopo_auroc_by_layer" not in tail["prompt_final"]
    assert auth["verbal"]["accuracy_on_answered"] == 1.0
    # blind gate ran on the same frozen contexts; without an independent-model record it cannot PASS
    blind = res["blind_p0"]
    assert blind["text_audit_verdict"] == "PASS" and blind["verdict"] == "INCOMPLETE" and blind["allow_no_llm"] is False
    assert blind["n_pairs_p0_byte_identical"] == 8
    assert blind["tool_turn_source"] == {"shard": 16, "canonical": 0}
    assert blind["judges"]["llm"]["run"] is False and "blind_p0.py --llm" in blind["judges"]["llm"]["reason"]
    # provenance links the numbers to the shards; the skipped preflight and every flag are recorded
    prov = res["provenance"]
    assert prov["n_main_shards"] == 8 and prov["n_auth_shards"] == 20
    assert len(prov["shards_aggregate_sha256"]["all"]) == 64
    assert prov["collection"]["code_commit"] == "deadbeef" and prov["shards_carry_config_hash"] is False
    assert prov["preflight"]["skipped"] is True and prov["preflight"]["checked"] is False
    assert res["analysis_flags"] == {"skip_provenance_check": True, "skip_blind": False, "allow_no_llm": False, "llm_record": None}
    md = (out / "summary.md").read_text()
    for heading in ("### Integrity gates", "### Blind P0 gate (post hoc)", "### V5 side probe", "### Pre-registration deviations", "### Provenance"):
        assert heading in md
    assert "Superseded tail split" in md
    assert "Verdict: **INCOMPLETE**" in md and "has NOT been run" in md and "Provenance preflight: **SKIPPED**" in md
    assert sum(1 for _ in open(out / "transcripts.jsonl")) == 16 * N_ROLL + 20


def _stub_llm_record(records):
    from experiments.exp7_tool_probe import blind_p0

    real_text = FIXTURE["real"]["tool_text"]

    def call(prompt_text):
        if prompt_text.startswith(blind_p0.LLM_PROMPTS["p1"]):
            return {"reply": "REAL 0.95" if real_text in prompt_text else "SHAM 0.05", "response_model": "stub"}
        return {"reply": "SHAM 0.5", "response_model": "stub"}

    return blind_p0.run_llm_judge(records, call, model="stub", provider_label="stub provider", workers=1)


def test_llm_record_and_waiver_flags_change_the_verdict_and_are_recorded(synthetic) -> None:
    from experiments.exp7_tool_probe import blind_p0

    records = blind_p0.load_contexts(synthetic["data"], synthetic["acts"])
    record_path = synthetic["root"] / "llm_judge_record.json"
    record_path.write_text(json.dumps(_stub_llm_record(records)))
    out = synthetic["root"] / "out_llm"
    rc = analysis.main(["--acts", str(synthetic["acts"]), "--out", str(out), "--data", str(synthetic["data"]), "--n-shuffles", "1", "--llm-record", str(record_path), *SKIP])
    assert rc == 0
    res = json.loads((out / "results.json").read_text())
    blind = res["blind_p0"]
    assert blind["verdict"] == "PASS" and blind["judges"]["llm"]["run"] is True
    assert blind["judges"]["llm"]["p0"]["auroc"] == 0.5 and blind["judges"]["llm"]["p1_positive_control"]["auroc"] == 1.0
    assert blind["judges"]["llm"]["n_calls"] == 32 and blind["judges"]["llm"]["provider"] == "stub provider"
    assert res["analysis_flags"]["llm_record"] == "llm_judge_record.json" and res["analysis_flags"]["allow_no_llm"] is False
    md = (out / "summary.md").read_text()
    assert "Independent-model judge" in md and "P0 AUROC 0.500" in md and "Verdict: **PASS**" in md
    assert "was also run post hoc" in md
    # the waiver path: PASS on the text audit alone, and it says so
    out2 = synthetic["root"] / "out_waiver"
    analysis.main(["--acts", str(synthetic["acts"]), "--out", str(out2), "--data", str(synthetic["data"]), "--n-shuffles", "1", "--allow-no-llm", *SKIP])
    res2 = json.loads((out2 / "results.json").read_text())
    assert res2["blind_p0"]["verdict"] == "PASS" and res2["blind_p0"]["allow_no_llm"] is True
    assert res2["analysis_flags"]["allow_no_llm"] is True
    assert "`--allow-no-llm` was passed" in (out2 / "summary.md").read_text()
    # a record for other contexts is refused
    bad = json.loads(record_path.read_text())
    bad["items"][0]["text_sha256"] = "0" * 64
    bad_path = synthetic["root"] / "bad_record.json"
    bad_path.write_text(json.dumps(bad))
    with pytest.raises(RuntimeError, match="different context bytes"):
        analysis.main(["--acts", str(synthetic["acts"]), "--out", str(synthetic["root"] / "out_bad"), "--data", str(synthetic["data"]), "--n-shuffles", "1", "--llm-record", str(bad_path), *SKIP])


def test_provenance_preflight_refuses_partial_or_unlisted_mirrors(synthetic, tmp_path) -> None:
    import shutil

    from experiments.exp7_tool_probe import provenance as prov

    # no shards.sha256 next to the inputs -> refused (the fixture has none)
    with pytest.raises(RuntimeError, match="shards.sha256"):
        analysis.main(["--acts", str(synthetic["acts"]), "--out", str(tmp_path / "o0"), "--data", str(synthetic["data"]), "--n-shuffles", "1"])
    # a complete mirror described by a manifest -> passes and is summarised
    data = tmp_path / "data"
    shutil.copytree(synthetic["data"], data)
    acts = tmp_path / "acts"
    shutil.copytree(synthetic["acts"], acts)
    for name in prov.FROZEN_INPUTS:  # every frozen input must exist and be listed (the fixture only has some)
        if not (data / name).is_file():
            (data / name).write_text("{}")
    records = prov.collect_records(data, prefix="experiments/exp7_tool_probe/data/vtest")
    records += prov.collect_records(acts / "main", prefix="acts/main")
    records += prov.collect_records(acts / "auth", prefix="acts/auth")
    prov.write_sha256_manifest(data / prov.SHA_MANIFEST_NAME, records)
    pre = analysis.preflight_provenance(acts, data)
    assert pre["checked"] is True and pre["n_shards"] == {"main": 8, "auth": 20} and pre["expected_shards"] == {"main": 8, "auth": 20}
    # 7/8 main shards present -> refused before any analysis, and the message names the count
    (acts / "main" / "t07.npz").unlink()
    with pytest.raises(RuntimeError, match=r"1 listed file\(s\) missing locally, e.g. acts/main/t07.npz"):
        analysis.main(["--acts", str(acts), "--out", str(tmp_path / "o1"), "--data", str(data), "--n-shuffles", "1"])
    assert not (tmp_path / "o1" / "results.json").exists()
    # a manifest that lists only 7 shards is refused too (the inventory comes from manifest.json)
    prov.write_sha256_manifest(data / prov.SHA_MANIFEST_NAME, [r for r in records if r["path"] != "acts/main/t07.npz"])
    with pytest.raises(RuntimeError, match="manifest lists 7 main shards, 8 expected"):
        analysis.preflight_provenance(acts, data)
    # the explicit bypass runs and records itself
    rc = analysis.main(["--acts", str(acts), "--out", str(tmp_path / "o2"), "--data", str(data), "--n-shuffles", "1", "--skip-blind", "--skip-provenance-check"])
    assert rc == 0
    res = json.loads((tmp_path / "o2" / "results.json").read_text())
    assert res["analysis_flags"]["skip_provenance_check"] is True and res["provenance"]["preflight"]["skipped"] is True


def test_analysis_is_deterministic(synthetic) -> None:
    outs = []
    for name in ("det_a", "det_b"):
        out = synthetic["root"] / name
        analysis.main(["--acts", str(synthetic["acts"]), "--out", str(out), "--data", str(synthetic["data"]), "--n-shuffles", "2", *SKIP])
        outs.append(out)
    assert (outs[0] / "results.json").read_bytes() == (outs[1] / "results.json").read_bytes()
    assert (outs[0] / "summary.md").read_bytes() == (outs[1] / "summary.md").read_bytes()
    assert (outs[0] / "blind_p0.json").read_bytes() == (outs[1] / "blind_p0.json").read_bytes()


def test_boot_ci_and_safe_auroc() -> None:
    y = np.array([1, 0, 1, 0, 1, 0])
    g = np.array([0, 0, 1, 1, 2, 2])
    assert analysis.safe_auroc(y, np.array([3, 1, 3, 1, 3, 1])) == 1.0
    assert np.isnan(analysis.safe_auroc(np.ones(4), np.arange(4)))
    assert analysis.boot_ci(y, np.array([3, 1, 3, 1, 3, 1]), g, n=50) == [1.0, 1.0]
    assert analysis.boot_ci(y, np.array([1, 1, 1, 1, 1, 1]), g, n=50) == [0.5, 0.5]
    a = analysis.boot_ci(y, np.array([2.0, 1.0, 1.5, 1.6, 3.0, 0.5]), g, n=100, seed=3)
    assert a == analysis.boot_ci(y, np.array([2.0, 1.0, 1.5, 1.6, 3.0, 0.5]), g, n=100, seed=3)
    assert 0.0 <= a[0] <= a[1] <= 1.0
    assert analysis.boot_ci(np.ones(4), np.arange(4), np.array([0, 0, 1, 1]), n=10) == [pytest.approx(float("nan"), nan_ok=True)] * 2


def test_select_layer_and_loto_are_out_of_fold() -> None:
    rng = np.random.default_rng(1)
    n, groups = 40, np.repeat(np.arange(10), 4)
    y = np.tile([1, 0], n // 2)
    X = rng.normal(size=(n, 3, 6)).astype(np.float32)
    X[:, 1, 0] += 5.0 * (2 * y - 1)  # only layer 1 carries the label
    is_train = groups < 8
    sel, cv = analysis.select_layer(X, y, groups, is_train)
    assert sel == 1 and cv[1] == 1.0 and max(cv[0], cv[2]) < 0.8
    held, acc = analysis.layer_curve_heldout(X, y, groups, is_train)
    assert held[1] == 1.0 and acc[1] == 1.0
    assert analysis.layer_curve_heldout(X, y, groups, np.ones(n, bool)) == ([float("nan")] * 3, [float("nan")] * 3) or all(np.isnan(analysis.layer_curve_heldout(X, y, groups, np.ones(n, bool))[0]))
    # LOTO: the score for group g must come from a model that never saw g
    two = groups < 2
    s = analysis.loto_scores(X[two], y[two], groups[two], 1)
    manual_g0 = analysis.fit_scores(X[two][groups[two] == 1, 1, :], y[two][groups[two] == 1], X[two][groups[two] == 0, 1, :])
    assert np.allclose(s[groups[two] == 0], manual_g0)
    curve, scores = analysis.layer_curve_loto(X, y, groups)
    assert curve[1] == 1.0 and scores.shape == (n, 3)
    with pytest.raises(RuntimeError):
        analysis.select_layer(X, y, groups, groups < 1)


def test_load_auth_split_and_counts(tmp_path) -> None:
    assert analysis.load_auth_split(tmp_path) is None
    (tmp_path / "auth_split.json").write_text(json.dumps({"split": ["train", "heldout"]}))
    assert analysis.load_auth_split(tmp_path) == {0: "train", 1: "heldout"}
    rows = [
        {"template_id": 0, "capture_kind": "a", "split": "train"},
        {"template_id": 0, "capture_kind": "a", "split": "train"},
        {"template_id": 1, "capture_kind": "b", "split": "heldout"},
        {"template_id": 1, "capture_kind": "b", "split": "heldout"},
    ]
    assert analysis._split_counts_by_kind(rows, "split") == {"train": {"a": 1}, "heldout": {"b": 1}}
    assert analysis.analyse_auth([], "split") == {"n_rows": 0, "note": "no auth shards"}
