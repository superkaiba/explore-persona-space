"""Pin the problem-sweep evidence-counting discipline (#1484).

/daily is an LLM-driven SKILL.md file; the only mechanical guard
against a later edit silently dropping the count-firings-not-echoes rule is
a pure-text assertion that the prose survives. Substring pins, not
structure parsing (the test_daily_three_route_classifier_doc.py pattern).
(The /weekly twin pin was removed when the weekly skill was retired,
2026-08-05.)

Incident 2026-07-16: a raw string count over a transcript read "13 failed
vs 8 ok" for a warn string when only 1 of 13 raw occurrences was a real
tool_result firing event (the rest were recipe/command-text echoes and
harness toolUseResult duplicates), spawning false needs-human task #1473.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DAILY = REPO_ROOT / ".claude" / "skills" / "daily" / "SKILL.md"


def test_problem_sweep_counts_firings_not_echoes():
    text = DAILY.read_text(encoding="utf-8")
    assert "Evidence-counting discipline" in text
    assert "tool_result" in text
    assert "recipe echo" in text
    assert "counting method" in text
    assert "toolUseResult" in text  # the dedupe-per-tool-call channel
    assert "one tool call is ONE event" in text  # per-tool-call dedup rule
    assert "content-readback" in text  # the read/grep-result echo channel (Stats-lens Must-Fix)
