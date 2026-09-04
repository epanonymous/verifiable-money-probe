"""The blind P0 gate must (a) be at chance on leak-free contexts, (b) be able to
see a leak when one exists, (c) be deterministic under its seed, and (d) never
report PASS without the independent-model judge unless explicitly waived."""

from __future__ import annotations

import datetime as dt
import io
import json
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from experiments.exp7_tool_probe import blind_p0, dataset

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "recorded_balances.json").read_text())
TOOL_TEXTS = {"REAL": FIXTURE["real"]["tool_text"], "SHAM": FIXTURE["sham"]["tool_text"]}
NOW = dt.datetime(2026, 9, 4, 1, 30, tzinfo=dt.timezone.utc)


def _rows(n_train: int = 10, n_heldout: int = 2) -> list[dict]:
    templates = dataset.build_templates(n_train=n_train, n_heldout=n_heldout)
    return [r.to_dict() for r in dataset.build_rows(templates=templates)]


def _records(leak: bool = False) -> list[dict]:
    rows = _rows()
    if leak:
        for row in rows:
            if row["world"] == "REAL":
                row["prompt"] = row["prompt"] + '\n\n(status: "settled": true, usdc_balance 1.900000)'
            else:
                row["prompt"] = row["prompt"] + '\n\n(status: payment never arrived, balance 0.000000)'
    return blind_p0.build_contexts(rows, TOOL_TEXTS)


def _stub_call(p0_reply="SHAM 0.5", *, p1_sighted: bool = True, p0_reads_leak: bool = False):
    """A fake independent model: blind at P0 (unless told to read a planted leak), sighted at P1."""

    def call(prompt_text: str) -> dict:
        if prompt_text.startswith(blind_p0.LLM_PROMPTS["p1"]):
            if not p1_sighted:
                return {"reply": "REAL 0.5", "response_model": "stub-1"}
            real = TOOL_TEXTS["REAL"] in prompt_text
            return {"reply": "REAL 0.97" if real else "SHAM 0.03", "response_model": "stub-1"}
        if p0_reads_leak and "settled" in prompt_text:
            return {"reply": "REAL 0.9", "response_model": "stub-1"}
        if p0_reads_leak and "never arrived" in prompt_text:
            return {"reply": "SHAM 0.1", "response_model": "stub-1"}
        return {"reply": p0_reply, "response_model": "stub-1"}

    return call


def _record(records, **kw):
    return blind_p0.run_llm_judge(records, _stub_call(**kw), model="stub", provider_label="stub provider", workers=2, now=NOW)


def test_leak_free_contexts_are_at_chance_and_the_judge_is_not_blind() -> None:
    records = _records()
    out = blind_p0.run_gate(records, n_shuffles=3, n_boot=50, llm_reason_not_run="test: no key")
    assert out["n_rows"] == 24 and out["n_templates"] == 12
    assert out["n_pairs_p0_byte_identical"] == 12
    assert out["n_pairs_p1_byte_identical"] == 0
    bow = out["judges"]["bow_tfidf_logreg_loto"]
    assert bow["auroc"] == pytest.approx(0.5)
    assert bow["accuracy_ties_half"] == pytest.approx(0.5)
    assert bow["auroc_ci_template_bootstrap"] == [pytest.approx(0.5), pytest.approx(0.5)]
    # the same judge separates the worlds once the tool result is in the text
    assert bow["positive_control_p1"]["auroc"] == pytest.approx(1.0)
    assert bow["shuffle_null"]["n"] == 3
    assert out["judges"]["lexicon_scan"]["n_rows_with_hits"] == 0
    assert out["judges"]["llm"] == {"run": False, "reason": "test: no key"}
    # the text audit passes, but the locked gate is NOT satisfied without the independent model
    assert out["text_audit_verdict"] == "PASS"
    assert out["verdict"] == "INCOMPLETE" and out["allow_no_llm"] is False
    assert "another model" in out["verdict_reason"]


def test_allow_no_llm_is_the_only_way_to_pass_without_the_judge_and_is_recorded() -> None:
    out = blind_p0.run_gate(_records(), n_shuffles=2, n_boot=20, allow_no_llm=True)
    assert out["verdict"] == "PASS" and out["allow_no_llm"] is True
    assert "--allow-no-llm" in out["verdict_reason"]
    # the waiver cannot turn an alarming text audit into a pass
    out = blind_p0.run_gate(_records(leak=True), n_shuffles=2, n_boot=20, allow_no_llm=True)
    assert out["verdict"] == "ALARM" and out["text_audit_verdict"] == "ALARM"


def test_overall_verdict_rules() -> None:
    not_run = {"run": False}
    assert blind_p0.overall_verdict("PASS", not_run, allow_no_llm=False)[0] == "INCOMPLETE"
    assert blind_p0.overall_verdict("ALARM", not_run, allow_no_llm=False)[0] == "INCOMPLETE"
    assert blind_p0.overall_verdict("PASS", not_run, allow_no_llm=True)[0] == "PASS"
    assert blind_p0.overall_verdict("ALARM", not_run, allow_no_llm=True)[0] == "ALARM"

    def llm(p0, p1):
        return {"run": True, "p0": {"auroc": p0}, "p1_positive_control": {"auroc": p1}}

    assert blind_p0.overall_verdict("PASS", llm(0.5, 1.0), allow_no_llm=False)[0] == "PASS"
    assert blind_p0.overall_verdict("PASS", llm(0.5, 1.0), allow_no_llm=True)[0] == "PASS"
    assert blind_p0.overall_verdict("PASS", llm(0.9, 1.0), allow_no_llm=False)[0] == "ALARM"
    assert blind_p0.overall_verdict("ALARM", llm(0.5, 1.0), allow_no_llm=False)[0] == "ALARM"
    verdict, reason = blind_p0.overall_verdict("PASS", llm(0.5, 0.6), allow_no_llm=False)
    assert verdict == "INCOMPLETE" and "sighted control" in reason
    assert blind_p0.overall_verdict("PASS", llm(0.5, float("nan")), allow_no_llm=False)[0] == "INCOMPLETE"


def test_independent_model_record_completes_the_gate() -> None:
    records = _records()
    record = _record(records)
    assert record["schema"] == "exp7-blind-p0-llm-record/1"
    assert record["n_calls"] == 48 and record["positions"] == ["p0", "p1"]
    assert record["judge_model"] == "stub" and record["provider"] == "stub provider"
    assert record["temperature"] == 0.0 and record["run_utc"] == "2026-09-04T01:30:00Z"
    assert record["prompt_sha256"]["p0"] != record["prompt_sha256"]["p1"]
    assert record["judge_model_resolved"] == ["stub-1"]
    assert record["score_rule"] == blind_p0.SCORE_RULE
    item = record["items"][0]
    assert set(item) >= {"row_id", "template_id", "world", "label", "position", "text_sha256", "reply", "score", "response_model"}
    assert item["label"] in ("REAL", "SHAM")

    out = blind_p0.run_gate(records, n_shuffles=2, n_boot=30, llm_record=record, llm_record_meta={"record_path": "x/llm_judge_record.json", "record_sha256": "0" * 64})
    llm = out["judges"]["llm"]
    assert llm["run"] is True and llm["judge_model"] == "stub" and llm["provider"] == "stub provider"
    assert llm["n_calls"] == 48 and llm["temperature"] == 0.0 and llm["record_path"] == "x/llm_judge_record.json"
    assert llm["p0"]["auroc"] == pytest.approx(0.5) and llm["p0"]["n_unique_scores"] == 1
    assert llm["p0"]["label_accuracy"] == pytest.approx(0.5) and llm["p0"]["label_counts_by_world"]["REAL"] == {"REAL": 0, "SHAM": 12, "NONE": 0}
    assert llm["p1_positive_control"]["auroc"] == pytest.approx(1.0) and llm["p1_positive_control"]["label_accuracy"] == 1.0
    assert llm["score_rule"] == blind_p0.SCORE_RULE
    assert llm["p1_positive_control"]["accuracy_ties_half"] == pytest.approx(1.0)
    assert set(llm["p0"]["scores"]) == {r["row_id"] for r in records}
    assert llm["auroc"] == llm["p0"]["auroc"]
    assert out["verdict"] == "PASS" and out["allow_no_llm"] is False
    assert "independent-model judge at chance on P0" in out["verdict_reason"]


def test_independent_model_that_reads_a_leak_alarms_and_a_blind_control_is_incomplete() -> None:
    leaked = _records(leak=True)
    out = blind_p0.run_gate(leaked, n_shuffles=2, n_boot=20, llm_record=_record(leaked, p0_reads_leak=True))
    assert out["judges"]["llm"]["p0"]["auroc"] == pytest.approx(1.0)
    assert out["verdict"] == "ALARM"
    clean = _records()
    out = blind_p0.run_gate(clean, n_shuffles=2, n_boot=20, llm_record=_record(clean, p1_sighted=False))
    assert out["judges"]["llm"]["p1_positive_control"]["auroc"] == pytest.approx(0.5)
    assert out["verdict"] == "INCOMPLETE" and "sighted control" in out["verdict_reason"]


def test_llm_record_must_match_the_exact_context_bytes() -> None:
    records = _records()
    record = _record(records)
    blind_p0.check_llm_record(record, records)
    tampered = json.loads(json.dumps(record))
    tampered["items"][3]["text_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="different context bytes"):
        blind_p0.check_llm_record(tampered, records)
    short = json.loads(json.dumps(record))
    short["items"] = short["items"][:-1]
    with pytest.raises(RuntimeError, match="does not cover"):
        blind_p0.check_llm_record(short, records)
    dup = json.loads(json.dumps(record))
    dup["items"].append(dict(dup["items"][0]))
    with pytest.raises(RuntimeError, match="duplicate"):
        blind_p0.check_llm_record(dup, records)
    prompt = json.loads(json.dumps(record))
    prompt["prompt_sha256"]["p0"] = "1" * 64
    with pytest.raises(RuntimeError, match="prompt"):
        blind_p0.check_llm_record(prompt, records)
    p0_only = json.loads(json.dumps(record))
    p0_only["positions"] = ["p0"]
    with pytest.raises(RuntimeError, match="lacks position"):
        blind_p0.check_llm_record(p0_only, records)
    # a record for different contexts is rejected too
    with pytest.raises(RuntimeError):
        blind_p0.run_gate(_records(leak=True), n_shuffles=1, n_boot=5, llm_record=record)


def test_gate_is_deterministic_under_its_seed() -> None:
    records = _records()
    a = blind_p0.run_gate(records, n_shuffles=3, n_boot=30)
    b = blind_p0.run_gate(records, n_shuffles=3, n_boot=30)
    assert json.dumps(a, sort_keys=True, default=float) == json.dumps(b, sort_keys=True, default=float)
    null = blind_p0.shuffle_null([r["p0_text"] for r in records], np.array([r["label"] for r in records]), np.array([r["template_id"] for r in records]), n=3, seed=7)
    assert len(null) == 3
    assert null == blind_p0.shuffle_null([r["p0_text"] for r in records], np.array([r["label"] for r in records]), np.array([r["template_id"] for r in records]), n=3, seed=7)
    record = _record(records)
    a = blind_p0.run_gate(records, n_shuffles=2, n_boot=30, llm_record=record)
    b = blind_p0.run_gate(records, n_shuffles=2, n_boot=30, llm_record=json.loads(json.dumps(record)))
    assert json.dumps(a, sort_keys=True, default=float) == json.dumps(b, sort_keys=True, default=float)


def test_a_leaked_world_token_trips_the_gate() -> None:
    out = blind_p0.run_gate(_records(leak=True), n_shuffles=2, n_boot=20)
    assert out["n_pairs_p0_byte_identical"] == 0
    assert out["judges"]["bow_tfidf_logreg_loto"]["auroc"] > 0.95
    assert out["judges"]["lexicon_scan"]["n_rows_with_hits"] == 24
    assert out["judges"]["lexicon_scan"]["auroc"] == pytest.approx(1.0)
    assert out["text_audit_verdict"] == "ALARM"
    assert out["verdict"] == "INCOMPLETE"  # no independent model either; never PASS


def test_llm_plumbing_without_a_key_and_provider_labels() -> None:
    assert blind_p0.llm_provider_from_env({}) is None
    assert blind_p0.llm_provider_from_env({"OPENAI_API_KEY": "x"}) == "openai"
    assert blind_p0.llm_provider_from_env({"ANTHROPIC_API_KEY": "x"}) == "anthropic"
    assert blind_p0.llm_provider_label("openai", {}) == "OpenAI API"
    assert blind_p0.llm_provider_label("openai", {"OPENAI_BASE_URL": "https://api.openai.com/v1"}) == "OpenAI API"
    assert blind_p0.llm_provider_label("openai", {"OPENAI_BASE_URL": "https://proxy.example/v1/"}) == "OpenAI-compatible proxy"
    assert blind_p0.llm_provider_label("anthropic", {}) == "Anthropic API"
    with pytest.raises(ValueError):
        blind_p0.llm_provider_label("other", {})
    with pytest.raises(RuntimeError):
        blind_p0.make_llm_caller("openai", "m", env={})
    with pytest.raises(RuntimeError):
        blind_p0.make_llm_caller("anthropic", "m", env={})
    with pytest.raises(ValueError):
        blind_p0.make_llm_caller("other", "m", env={"OPENAI_API_KEY": "x"})
    assert blind_p0.parse_llm_reply("REAL 0.8") == 0.8
    assert blind_p0.parse_llm_reply("SHAM, probability 0.2 that it is REAL") == pytest.approx(0.2)
    assert blind_p0.parse_llm_reply("SHAM") == 0.0
    assert blind_p0.parse_llm_reply("REAL") == 1.0
    assert blind_p0.parse_llm_reply("no idea") == 0.5
    # the label decides the side; the number is confidence in it (what gpt-4o actually answers)
    assert blind_p0.parse_llm_reply("SHAM, 0.95") == pytest.approx(0.05)
    assert blind_p0.parse_llm_reply("SHAM 0.6") == pytest.approx(0.4)
    assert blind_p0.parse_llm_reply("REAL, 0.3") == pytest.approx(0.7)
    assert blind_p0.parse_llm_reply("REAL 0.5") == 0.5 and blind_p0.parse_llm_reply("SHAM 0.5") == 0.5
    assert blind_p0.parse_llm_label("SHAM, 0.95") == "SHAM" and blind_p0.parse_llm_label("sham 0.6") == "SHAM"
    assert blind_p0.parse_llm_label("nothing") is None
    assert "confidence" in blind_p0.LLM_PROMPTS["p0"] and "confidence" in blind_p0.LLM_PROMPTS["p1"]


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_openai_compatible_caller_sends_headers_and_retries_transient_errors(monkeypatch) -> None:
    seen: list[urllib.request.Request] = []
    codes = iter([429, 503])

    def fake_urlopen(req, timeout=None):
        seen.append(req)
        try:
            code = next(codes)
        except StopIteration:
            body = {"model": "gpt-4o-2024-08-06", "choices": [{"message": {"content": "SHAM 0.4"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 3}}
            return _Resp(json.dumps(body).encode())
        raise urllib.error.HTTPError(req.full_url, code, "busy", {}, io.BytesIO(b"error code: 1010"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    slept: list[float] = []
    call = blind_p0.make_llm_caller("openai", "gpt-4o", env={"OPENAI_API_KEY": "sk-test", "OPENAI_BASE_URL": "https://proxy.example/v1/"}, sleep=slept.append)
    out = call("hello")
    assert out["reply"] == "SHAM 0.4" and out["response_model"] == "gpt-4o-2024-08-06"
    assert out["usage"] == {"prompt_tokens": 10, "completion_tokens": 3}
    assert len(seen) == 3 and slept == [1.0, 2.0]
    req = seen[-1]
    assert req.full_url == "https://proxy.example/v1/chat/completions"
    assert req.get_header("Authorization") == "Bearer sk-test"
    assert req.get_header("User-agent", "").startswith("exp7-blind-p0-judge")
    body = json.loads(req.data)
    assert body["model"] == "gpt-4o" and body["temperature"] == 0.0 and body["max_tokens"] == blind_p0.LLM_MAX_TOKENS
    assert body["messages"] == [{"role": "user", "content": "hello"}]

    def forbidden(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 403, "forbidden", {}, io.BytesIO(b"secret-echo"))

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    with pytest.raises(RuntimeError) as exc:
        call("hello")
    assert "403" in str(exc.value) and "secret-echo" not in str(exc.value) and "sk-test" not in str(exc.value)


def test_contexts_from_frozen_data_use_shard_turn_when_present(tmp_path) -> None:
    rows = _rows(2, 1)
    turns = {rows[0]["template_id"]: "<tool_call>\n{\"name\": \"check_balance\", \"arguments\": {}}\n</tool_call>"}
    recs = blind_p0.build_contexts(rows, TOOL_TEXTS, turns)
    sources = {r["template_id"]: r["tool_turn_source"] for r in recs}
    assert sources[rows[0]["template_id"]] == "shard"
    assert set(sources.values()) == {"shard", "canonical"}
    with pytest.raises(RuntimeError):
        blind_p0.run_gate([r for r in recs if r["world"] == "REAL"], n_shuffles=1, n_boot=5)


def _frozen_dir(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    templates = dataset.build_templates(n_train=4, n_heldout=1)
    dataset.write_jsonl(data / "rows.jsonl", dataset.build_rows(templates=templates))
    (data / "readouts.json").write_text(json.dumps({w: {"text": t} for w, t in TOOL_TEXTS.items()}))
    return data


def test_cli_replays_a_record_and_refuses_contradictory_flags(tmp_path, monkeypatch) -> None:
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    data = _frozen_dir(tmp_path)
    records = blind_p0.load_contexts(data)
    record_path = tmp_path / "llm_judge_record.json"
    record_path.write_text(json.dumps(_record(records)))
    out = tmp_path / "blind_p0.json"
    assert blind_p0.main(["--data", str(data), "--out", str(out), "--n-shuffles", "1", "--llm-record", str(record_path)]) == 0
    res = json.loads(out.read_text())
    assert res["verdict"] == "PASS" and res["judges"]["llm"]["run"] is True
    assert res["judges"]["llm"]["record_path"].endswith("llm_judge_record.json")
    assert res["judges"]["llm"]["record_sha256"] == blind_p0.record_meta(record_path)["record_sha256"]
    # no record, no key, no waiver -> INCOMPLETE, and the artifact says why
    assert blind_p0.main(["--data", str(data), "--out", str(out), "--n-shuffles", "1"]) == 0
    res = json.loads(out.read_text())
    assert res["verdict"] == "INCOMPLETE" and res["judges"]["llm"]["run"] is False and "no OPENAI_API_KEY" in res["judges"]["llm"]["reason"]
    assert blind_p0.main(["--data", str(data), "--out", str(out), "--n-shuffles", "1", "--allow-no-llm"]) == 0
    res = json.loads(out.read_text())
    assert res["verdict"] == "PASS" and res["allow_no_llm"] is True
    with pytest.raises(SystemExit):
        blind_p0.main(["--data", str(data), "--out", str(out), "--llm"])  # no key
    with pytest.raises(SystemExit):
        blind_p0.main(["--data", str(data), "--out", str(out), "--llm", "--llm-record", str(record_path)])
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "other"}))
    with pytest.raises(RuntimeError):
        blind_p0.load_llm_record(bad)
