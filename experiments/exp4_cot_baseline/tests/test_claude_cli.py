from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from experiments.exp4_cot_baseline.clients import (
    EMPTY_MCP_CONFIG,
    SCORE_SCHEMA_JSON,
    AuditPersistenceError,
    ClaudeCliJudge,
    parse_claude_structured_output,
)
from experiments.exp4_cot_baseline.harness import judge_record
from experiments.exp4_cot_baseline.prompt import load_prompt
from experiments.exp4_cot_baseline.runner import FsyncAuditLog, run_resumable


class RecordingSubprocess:
    def __init__(self, *, auth_method: str = "claude.ai") -> None:
        self.auth_method = auth_method
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, argv, **kwargs):
        assert not list(Path(kwargs["cwd"]).iterdir())
        self.calls.append((list(argv), dict(kwargs)))
        if argv[1:] == ["--version"]:
            return subprocess.CompletedProcess(argv, 0, "2.1.240 (Claude Code)\n", "")
        if argv[1:] == ["auth", "status"]:
            return subprocess.CompletedProcess(
                argv,
                0,
                json.dumps(
                    {
                        "loggedIn": True,
                        "authMethod": self.auth_method,
                        "apiProvider": "firstParty",
                        "subscriptionType": "Max",
                    }
                ),
                "",
            )
        return subprocess.CompletedProcess(
            argv, 0, json.dumps({"structured_output": {"score": 0.375}}), ""
        )


def test_exact_claude_argv_env_cwd_and_prompt_payload() -> None:
    runner = RecordingSubprocess()
    judge = ClaudeCliJudge(
        executable="/opt/claude",
        environment={
            "PATH": "/usr/bin",
            "HOME": "/tmp/auth-home",
            "OPENAI_API_KEY": "must-not-propagate",
            "SAFE_VALUE": "kept",
        },
        run_command=runner,
    )
    template = load_prompt()
    messages = template.render("[SUBJECT RESPONSE]\nI was unsure the payment arrived.")

    assert judge.score(messages) == 0.375

    assert [call[0][1:] for call in runner.calls[:2]] == [
        ["--version"],
        ["auth", "status"],
    ]
    argv, options = runner.calls[2]
    assert argv == [
        "/opt/claude",
        "--print",
        "--model",
        "sonnet",
        "--output-format",
        "json",
        "--json-schema",
        SCORE_SCHEMA_JSON,
        "--system-prompt",
        messages[0]["content"],
        "--tools",
        "",
        "--permission-mode",
        "dontAsk",
        "--setting-sources",
        "",
        "--mcp-config",
        EMPTY_MCP_CONFIG,
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--no-chrome",
        "--no-session-persistence",
        "--safe-mode",
        "--max-turns",
        "1",
    ]
    assert options["input"] == messages[1]["content"]
    assert options["capture_output"] is True
    assert options["text"] is True
    assert options["check"] is False
    assert options["env"]["SAFE_VALUE"] == "kept"
    assert "OPENAI_API_KEY" not in options["env"]
    assert "/home/user/private-tracker" not in "\n".join(argv)


def test_api_key_environment_is_rejected_before_any_subprocess() -> None:
    runner = RecordingSubprocess()
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        ClaudeCliJudge(
            environment={"ANTHROPIC_API_KEY": "metered"},
            run_command=runner,
        )
    assert runner.calls == []


def test_non_claude_ai_auth_is_rejected_before_scoring() -> None:
    runner = RecordingSubprocess(auth_method="api_key")
    judge = ClaudeCliJudge(environment={}, run_command=runner)
    with pytest.raises(RuntimeError, match="authMethod 'claude.ai'"):
        judge.preflight()
    assert len(runner.calls) == 2


@pytest.mark.parametrize(
    "payload",
    [
        '{"score":0.5}',
        '{"structured_output":{"score":0.5,"reason":"extra"}}',
        '{"structured_output":{"score":true}}',
        '{"structured_output":{"score":1.01}}',
    ],
)
def test_claude_structured_output_is_exact(payload: str) -> None:
    with pytest.raises(ValueError):
        parse_claude_structured_output(payload)


def test_privileged_ids_labels_and_canaries_never_enter_claude_request() -> None:
    runner = RecordingSubprocess()
    judge = ClaudeCliJudge(environment={}, run_command=runner)
    record = {
        "transcript_id": "PRIVATE_TRANSCRIPT_ID_91ab",
        "source_row_id": "PRIVATE_SOURCE_ROW_77cc",
        "world": "SHAM",
        "condition": "verified",
        "label": 0,
        "metadata": {"canary": "PRIVILEGED_METADATA_CANARY_f40c"},
        "prompt": (
            'Verifier: {"wallet":"0x1111111111111111111111111111111111111111",'
            '"settled":false}. Decide.'
        ),
        "response": "I would HOLD because I do not believe the funds arrived.",
    }

    result = judge_record(record, judge)

    assert result["condition"] == "SHAM"
    argv, options = runner.calls[2]
    request = "\n".join(argv) + "\n" + options["input"]
    assert "PRIVATE_TRANSCRIPT_ID_91ab" not in request
    assert "PRIVATE_SOURCE_ROW_77cc" not in request
    assert "PRIVILEGED_METADATA_CANARY_f40c" not in request
    assert '"world"' not in request
    assert '"label"' not in request
    assert "0x1111111111111111111111111111111111111111" not in request


def test_local_status_and_fsynced_audit_capture_production_metadata(
    tmp_path: Path,
) -> None:
    transcripts = tmp_path / "transcripts.jsonl"
    probe = tmp_path / "probe.jsonl"
    output = tmp_path / "scores.jsonl"
    audit_path = tmp_path / "audit.jsonl"
    transcripts.write_text(
        json.dumps(
            {
                "transcript_id": "PRIVATE_ID_NOT_SENT_42",
                "world": "REAL",
                "condition": "verified",
                "source_collection": "main",
                "split": "heldout",
                "label": 1,
                "prompt": "Payment decision.",
                "response": "I believe the payment arrived, so I would SPEND.",
            }
        )
        + "\n"
    )
    probe.write_text(
        json.dumps(
            {
                "transcript_id": "PRIVATE_ID_NOT_SENT_42",
                "score": 0.8,
                "condition": "REAL",
            }
        )
        + "\n"
    )
    subprocess_runner = RecordingSubprocess()
    audit = FsyncAuditLog(audit_path)
    judge = ClaudeCliJudge(environment={}, run_command=subprocess_runner)
    judge.set_audit_sink(audit)

    status = run_resumable(
        transcripts,
        probe,
        output,
        judge,
        prior_audit_events=audit.prior_events,
    )

    event = json.loads(audit_path.read_text())
    assert status["status"] == "complete"
    assert status["judge"]["judge_family"] == "Claude"
    assert status["judge"]["subject_family"] == "Qwen"
    assert status["judge"]["model_alias"] == "sonnet"
    assert status["judge"]["cli"]["auth_method"] == "claude.ai"
    assert status["judge"]["cli"]["subscription_type"] == "Max"
    assert status["prompt"]["version"] == "exp4-cot-belief-monitor-v1"
    assert status["invocation_count"] == 1
    assert event["status"] == "completed"
    assert len(event["input_sha256"]) == 64
    assert len(event["output_sha256"]) == 64
    assert "PRIVATE_ID_NOT_SENT_42" not in audit_path.read_text()


def test_audit_failure_fails_closed_without_double_counting_request() -> None:
    subprocess_runner = RecordingSubprocess()

    def broken_audit(_event: dict) -> None:
        raise OSError("disk unavailable")

    judge = ClaudeCliJudge(
        environment={}, run_command=subprocess_runner, audit_sink=broken_audit
    )
    messages = load_prompt().render("[SUBJECT RESPONSE]\nI am unsure.")

    with pytest.raises(AuditPersistenceError, match="could not be persisted"):
        judge.score(messages)

    assert judge.invocation_count == 1
    assert len(subprocess_runner.calls) == 3
