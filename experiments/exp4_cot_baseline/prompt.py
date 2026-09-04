"""Load the immutable, versioned judge prompt."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


PROMPT_PATH = Path(__file__).with_name("prompts") / "judge_v1.json"
TRANSCRIPT_MARKER = "<<SUBJECT_TRANSCRIPT_JSON>>"


@dataclass(frozen=True)
class PromptTemplate:
    version: str
    messages: tuple[dict[str, str], ...]

    def render(self, transcript: str) -> list[dict[str, str]]:
        transcript_json = json.dumps(transcript, ensure_ascii=False)
        rendered = []
        replacements = 0
        for message in self.messages:
            content = message["content"]
            replacements += content.count(TRANSCRIPT_MARKER)
            rendered.append(
                {
                    "role": message["role"],
                    "content": content.replace(TRANSCRIPT_MARKER, transcript_json),
                }
            )
        if replacements != 1:
            raise ValueError("prompt template must contain the transcript marker exactly once")
        return rendered


def load_prompt(path: Path = PROMPT_PATH) -> PromptTemplate:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("prompt file must contain a JSON object")
    version = data.get("version")
    messages = data.get("messages")
    if not isinstance(version, str) or not version:
        raise ValueError("prompt version is missing")
    if not isinstance(messages, list) or not messages:
        raise ValueError("prompt messages are missing")
    normalized = []
    for message in messages:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError("each prompt message must contain only role and content")
        if message["role"] not in {"system", "user"} or not isinstance(message["content"], str):
            raise ValueError("invalid prompt message")
        normalized.append(dict(message))
    return PromptTemplate(version=version, messages=tuple(normalized))
