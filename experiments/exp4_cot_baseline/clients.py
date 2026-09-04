"""Judge clients for the blind monitor; production uses Claude CLI subscription auth."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["score"],
    "additionalProperties": False,
}
SCORE_SCHEMA_JSON = json.dumps(SCORE_SCHEMA, separators=(",", ":"), sort_keys=True)
EMPTY_MCP_CONFIG = '{"mcpServers":{}}'
SUBJECT_FAMILY = "Qwen"
JUDGE_FAMILY = "Claude"
DEFAULT_MODEL_ALIAS = "sonnet"
DEFAULT_SUBSCRIPTION_TYPE = "Max"

# Any of these can make Claude Code use metered/provider credentials rather than
# the installed claude.ai subscription. Presence is rejected before auth status.
API_AUTH_ENV_KEYS = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_REFRESH_TOKEN",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
        "COT_JUDGE_API_KEY",
    }
)
_SECRET_ENV_MARKERS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "COOKIE")
_WORKSPACE_CONTEXT_ENV_KEYS = frozenset(
    {"PWD", "OLDPWD", "VIRTUAL_ENV", "PYTHONPATH", "GITHUB_WORKSPACE"}
)


class JudgeClient(Protocol):
    def score(self, messages: Sequence[Mapping[str, str]]) -> float:
        """Return P(subject believed the stakes were real)."""


class AuditPersistenceError(RuntimeError):
    """Raised when a completed invocation cannot be durably audited."""


def validate_score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("judge score must be numeric")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("judge score must be finite and in [0,1]")
    return score


def parse_judge_response(content: str) -> float:
    """Parse the exact score object used by mocks and structured CLI output."""

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("judge response is not valid JSON") from exc
    if not isinstance(parsed, dict) or set(parsed) != {"score"}:
        raise ValueError('judge response must be exactly {"score": <number>}')
    return validate_score(parsed["score"])


def parse_claude_structured_output(content: str) -> float:
    """Accept only Claude CLI's validated ``structured_output`` score object."""

    try:
        envelope = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("Claude CLI output is not valid JSON") from exc
    if not isinstance(envelope, dict) or "structured_output" not in envelope:
        raise ValueError("Claude CLI output is missing structured_output")
    structured = envelope["structured_output"]
    if not isinstance(structured, dict) or set(structured) != {"score"}:
        raise ValueError("Claude structured_output must contain exactly score")
    return validate_score(structured["score"])


class MockJudge:
    """Deterministic, recording judge for tests and local harness development."""

    def __init__(self, score: float = 0.5, *, scores: Iterable[float] | None = None):
        self._default_score = validate_score(score)
        self._scores = iter(scores) if scores is not None else None
        self.requests: list[list[dict[str, str]]] = []

    def score(self, messages: Sequence[Mapping[str, str]]) -> float:
        self.requests.append(deepcopy([dict(message) for message in messages]))
        if self._scores is None:
            return self._default_score
        try:
            value = next(self._scores)
        except StopIteration as exc:
            raise RuntimeError("MockJudge ran out of configured scores") from exc
        return validate_score(value)


@dataclass(frozen=True)
class ClaudePreflight:
    cli_version: str
    auth_method: str
    api_provider: str | None
    subscription_type: str
    subscription_type_source: str
    subscription_type_reported_by_cli: bool


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_secret_env_name(name: str) -> bool:
    upper = name.upper()
    return any(marker in upper for marker in _SECRET_ENV_MARKERS)


def reject_api_auth_environment(environment: Mapping[str, str]) -> None:
    blocked = sorted(
        name for name in API_AUTH_ENV_KEYS if str(environment.get(name, "")).strip()
    )
    if blocked:
        raise RuntimeError(
            "API/provider authentication environment is forbidden for the Claude "
            f"subscription runner: {', '.join(blocked)}"
        )


def sanitized_subprocess_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Copy operational environment while removing all secret-looking variables."""

    reject_api_auth_environment(environment)
    cleaned = {
        str(name): str(value)
        for name, value in environment.items()
        if not _is_secret_env_name(str(name))
        and str(name).upper() not in _WORKSPACE_CONTEXT_ENV_KEYS
    }
    cleaned["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    return cleaned


class ClaudeCliJudge:
    """One-transcript-per-process Claude Sonnet judge using claude.ai auth only."""

    safety_flags = (
        "--print",
        "--output-format=json",
        "--json-schema=exact-score-only",
        "--system-prompt=replacement",
        "--tools=disabled",
        "--permission-mode=dontAsk",
        "--setting-sources=none",
        "--mcp-config=empty",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--no-chrome",
        "--no-session-persistence",
        "--safe-mode",
        "--max-turns=1",
    )

    def __init__(
        self,
        *,
        executable: str = "claude",
        model_alias: str = DEFAULT_MODEL_ALIAS,
        timeout: float = 120.0,
        environment: Mapping[str, str] | None = None,
        run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        audit_sink: Callable[[dict[str, Any]], None] | None = None,
        expected_subscription_type: str = DEFAULT_SUBSCRIPTION_TYPE,
    ) -> None:
        if not executable.strip() or not model_alias.strip():
            raise ValueError("Claude executable and model alias must be non-empty")
        if timeout <= 0:
            raise ValueError("Claude timeout must be positive")
        if expected_subscription_type.strip().casefold() != "max":
            raise ValueError("the locked Claude subscription type is Max")
        self.executable = executable
        self.model_alias = model_alias
        self.timeout = timeout
        self.expected_subscription_type = expected_subscription_type
        self._source_environment = dict(
            os.environ if environment is None else environment
        )
        self.environment = sanitized_subprocess_environment(self._source_environment)
        self._run_command = run_command
        self._audit_sink = audit_sink
        self._preflight: ClaudePreflight | None = None
        self._preflight_lock = threading.Lock()
        self._audit_lock = threading.Lock()
        self._invocation_count = 0
        self.invocations: list[dict[str, Any]] = []

    @property
    def preflight_info(self) -> ClaudePreflight | None:
        return self._preflight

    @property
    def invocation_count(self) -> int:
        with self._audit_lock:
            return self._invocation_count

    def set_audit_sink(self, sink: Callable[[dict[str, Any]], None]) -> None:
        """Attach a durable local sink before the first scoring invocation."""

        with self._audit_lock:
            if self._invocation_count:
                raise RuntimeError(
                    "cannot replace Claude audit sink after scoring starts"
                )
            self._audit_sink = sink

    def _run_preflight_command(
        self, argv: list[str]
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory(prefix="cot-preflight-") as workdir:
            if any(Path(workdir).iterdir()):
                raise RuntimeError("Claude preflight temporary directory was not empty")
            completed = self._run_command(
                argv,
                cwd=workdir,
                env=self.environment,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Claude preflight command failed with exit code {completed.returncode}"
            )
        return completed

    def preflight(self) -> ClaudePreflight:
        """Reject API auth, record version, and require claude.ai authentication."""

        if self._preflight is not None:
            return self._preflight
        with self._preflight_lock:
            if self._preflight is not None:
                return self._preflight
            reject_api_auth_environment(self._source_environment)
            version_result = self._run_preflight_command([self.executable, "--version"])
            cli_version = version_result.stdout.strip()
            if not cli_version or "\n" in cli_version:
                raise RuntimeError("Claude CLI returned an invalid version string")
            auth_result = self._run_preflight_command(
                [self.executable, "auth", "status"]
            )
            try:
                auth = json.loads(auth_result.stdout)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "claude auth status did not return valid JSON"
                ) from exc
            if not isinstance(auth, dict):
                raise RuntimeError("claude auth status must return a JSON object")
            if auth.get("loggedIn") is not True:
                raise RuntimeError(
                    "claude auth status reports that the CLI is not logged in"
                )
            auth_method = auth.get("authMethod")
            if auth_method != "claude.ai":
                raise RuntimeError(
                    "claude auth status must report authMethod 'claude.ai', got "
                    f"{auth_method!r}"
                )
            api_provider = auth.get("apiProvider")
            if api_provider is not None and not isinstance(api_provider, str):
                raise RuntimeError("claude auth status apiProvider must be a string")
            if api_provider is not None and api_provider != "firstParty":
                raise RuntimeError(
                    "claude auth status must use firstParty subscription service, got "
                    f"{api_provider!r}"
                )
            reported_subscription = auth.get("subscriptionType")
            if reported_subscription is None:
                subscription_type = self.expected_subscription_type
                subscription_source = "locked Max requirement; this CLI auth-status payload did not expose plan tier"
                subscription_reported = False
            elif not isinstance(reported_subscription, str) or (
                reported_subscription.strip().casefold() != "max"
            ):
                raise RuntimeError(
                    "Claude runner requires a Max subscription, got "
                    f"{reported_subscription!r}"
                )
            else:
                subscription_type = reported_subscription.strip()
                subscription_source = "claude auth status"
                subscription_reported = True
            self._preflight = ClaudePreflight(
                cli_version=cli_version,
                auth_method=auth_method,
                api_provider=api_provider,
                subscription_type=subscription_type,
                subscription_type_source=subscription_source,
                subscription_type_reported_by_cli=subscription_reported,
            )
            return self._preflight

    def _score_command(self, system_prompt: str) -> list[str]:
        return [
            self.executable,
            "--print",
            "--model",
            self.model_alias,
            "--output-format",
            "json",
            "--json-schema",
            SCORE_SCHEMA_JSON,
            "--system-prompt",
            system_prompt,
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

    def _record_invocation(self, event: dict[str, Any]) -> None:
        with self._audit_lock:
            self._invocation_count += 1
            event = {"invocation": self._invocation_count, **event}
            self.invocations.append(event)
            if self._audit_sink is not None:
                try:
                    self._audit_sink(deepcopy(event))
                except Exception as exc:
                    raise AuditPersistenceError(
                        "Claude invocation audit could not be persisted"
                    ) from exc

    def score(self, messages: Sequence[Mapping[str, str]]) -> float:
        self.preflight()
        normalized = [dict(message) for message in messages]
        if len(normalized) != 2 or [item.get("role") for item in normalized] != [
            "system",
            "user",
        ]:
            raise ValueError(
                "Claude judge requires exactly one system and one user message"
            )
        if any(set(item) != {"role", "content"} for item in normalized):
            raise ValueError("Claude judge messages must contain only role and content")
        system_prompt = normalized[0]["content"]
        user_prompt = normalized[1]["content"]
        if not isinstance(system_prompt, str) or not isinstance(user_prompt, str):
            raise ValueError("Claude judge prompt content must be strings")
        command = self._score_command(system_prompt)
        input_hash = _sha256_text(user_prompt)
        request_hash = _sha256_text(
            json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
        )
        event: dict[str, Any] = {
            "model_alias": self.model_alias,
            "input_sha256": input_hash,
            "request_sha256": request_hash,
            "temporary_cwd_empty": True,
            "safety_flags": list(self.safety_flags),
        }
        try:
            with tempfile.TemporaryDirectory(prefix="cot-request-") as workdir:
                if any(Path(workdir).iterdir()):
                    raise RuntimeError(
                        "Claude request temporary directory was not empty"
                    )
                completed = self._run_command(
                    command,
                    input=user_prompt,
                    cwd=workdir,
                    env=self.environment,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                )
            event["exit_code"] = completed.returncode
            event["output_sha256"] = _sha256_text(completed.stdout)
            event["stderr_sha256"] = _sha256_text(completed.stderr)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Claude CLI scoring failed with exit code {completed.returncode}"
                )
            score = parse_claude_structured_output(completed.stdout)
        except Exception as exc:
            event["status"] = "error"
            event["error_type"] = type(exc).__name__
            event["error"] = str(exc)
            self._record_invocation(event)
            raise
        event["status"] = "completed"
        self._record_invocation(event)
        return score

    def audit_metadata(self) -> dict[str, Any]:
        preflight = self.preflight()
        return {
            "judge_family": JUDGE_FAMILY,
            "subject_family": SUBJECT_FAMILY,
            "executable": self.executable,
            "model_alias": self.model_alias,
            "cli": asdict(preflight),
            "safety_flags": list(self.safety_flags),
            "invocation_count": self.invocation_count,
        }
