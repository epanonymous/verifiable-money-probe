"""Keep the public write-up aligned with the live simplified paper."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WRITEUP = REPO_ROOT / "docs" / "writeup" / "research-task.md"
WORD_LIMIT = 600
HEADING = re.compile(r"^## The short version\s*$", re.MULTILINE)
NEXT_TOP_LEVEL = re.compile(r"^## ", re.MULTILINE)


def executive_summary_text(markdown: str) -> str:
    match = HEADING.search(markdown)
    if match is None:
        raise ValueError("no '## The short version' heading found")
    start = match.end()
    following = NEXT_TOP_LEVEL.search(markdown, start)
    end = following.start() if following else len(markdown)
    return markdown[start:end]


def count_words(section: str) -> int:
    kept = [
        line
        for line in section.splitlines()
        if not line.lstrip().startswith("![") and line.strip() not in ("```", "---")
    ]
    return len(" ".join(kept).split())


def executive_summary_word_count(path: Path = WRITEUP) -> int:
    return count_words(executive_summary_text(path.read_text(encoding="utf-8")))


def test_executive_summary_is_first_section() -> None:
    markdown = WRITEUP.read_text(encoding="utf-8")
    first_h2 = NEXT_TOP_LEVEL.search(markdown)
    assert first_h2 is not None
    assert markdown[first_h2.start() :].startswith("## The short version")


def test_executive_summary_within_word_limit() -> None:
    count = executive_summary_word_count()
    assert count <= WORD_LIMIT, (
        f"short version is {count} words; the summary limit is {WORD_LIMIT}"
    )


def test_word_counter_excludes_only_non_words() -> None:
    sample = "## The short version\n\nOne two.\n\n![alt](x.svg)\n\n```\nthree four\n```\n\n---\n\n## Next\n\nignored"
    assert count_words(executive_summary_text(sample)) == 4


def test_paper_matches_live_shape() -> None:
    markdown = WRITEUP.read_text(encoding="utf-8")
    for heading in (
        "## 1. What I asked",
        "## 4. What I found",
        "## 6. What went wrong first, and what I changed",
        "## 7. How I used LLMs",
    ):
        assert heading in markdown
    # No hour log, tracker links, or programme references in the public cut.
    assert "## 8. Hours" not in markdown
    assert "github.com" not in markdown
    assert "agent-verify" in markdown


if __name__ == "__main__":
    print(executive_summary_word_count())
