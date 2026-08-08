"""Pin the suffixed-pod completion-side teardown contract (#1662).

Incident: ``pod-1586-b`` (a suffixed follow-up pod, ~$12-13/hr) sat RUNNING
finished-but-not-terminated behind an ask-gate — run complete, artifacts
verified-uploaded, termination waiting on a user reply (the #664
idle-but-billing family). Task #1662 inserted a "Completion-side teardown
(no ask-gate)" contract clause at three prose sites — the § Pods multi-pod
paragraph (post-compaction home: `.claude/rules/pods.md`, moved by
40653b5dcf; #2166), the CLAUDE.md inline-override carve-out pod-safety
block, and their executing mirror in `/issue` SKILL.md Step 9a-ter — making
"run complete + uploads verified => surgical `pod.py terminate --issue <N>
--name-suffix <slug> --yes`" unconditional for suffixed follow-up pods.

This test pins those three insertions so a future CLAUDE.md / pods.md /
SKILL.md edit cannot silently drop the contract and reintroduce the
ask-gate. It follows the whitespace-normalize family pattern of
``tests/test_issue_skill_stopped_volume_persist_pin.py``; the two prose
occurrences (one per file post-compaction) are located PER LINE via their
distinct bold headers (both edit sites are single-line paragraphs).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLAUDE_MD = ROOT / "CLAUDE.md"
# #2166: the doc-compaction commit 40653b5dcf moved the § Pods multi-pod
# paragraph (the suffixed-pod #1662 site) out of CLAUDE.md into
# .claude/rules/pods.md; the carve-out site stays in CLAUDE.md.
PODS_RULE_MD = ROOT / ".claude" / "rules" / "pods.md"
ISSUE_SKILL = ROOT / ".claude" / "skills" / "issue" / "SKILL.md"

# The two occurrences (one per file) carry DISTINCT bold headers (the locator).
PODS_HEADER = "**Completion-side teardown (suffixed pods — no ask-gate, #1662):**"
CARVEOUT_HEADER = "**Completion-side teardown (no ask-gate):**"
TERMINATE_CMD = "pod.py terminate --issue <N> --name-suffix <slug> --yes"


def _norm(text: str) -> str:
    """Collapse all whitespace to single spaces so tokens match across the
    markdown soft line breaks the SKILL.md wrapper introduces."""
    return re.sub(r"\s+", " ", text)


def _line_with(lines: list[str], header: str) -> str:
    """Return the single line carrying `header`; fail loud on 0 or >1 hits."""
    hits = [line for line in lines if header in line]
    assert len(hits) == 1, (
        f"expected exactly one line carrying {header!r}, found {len(hits)} "
        "(the #1662 edit sites are single-line paragraphs)"
    )
    return hits[0]


def test_claude_md_carries_completion_side_teardown_contract():
    """BOTH #1662 insertions survive, one per file post-compaction
    (40653b5dcf; #2166): `.claude/rules/pods.md` carries the § Pods
    suffixed-pod clause and CLAUDE.md carries the inline-override carve-out
    clause, each with its load-bearing tokens on the same single-line
    paragraph. The `>= 2` contract count is summed ACROSS the two files."""
    body = CLAUDE_MD.read_text()
    pods_rule = PODS_RULE_MD.read_text()
    total = _norm(body).count("Completion-side teardown") + _norm(pods_rule).count(
        "Completion-side teardown"
    )
    assert total >= 2, (
        "the completion-side teardown contract must survive at both #1662 sites — "
        "pods.md carries the § Pods multi-pod clause, CLAUDE.md carries the "
        f"inline-override carve-out clause (found {total} total)"
    )

    pods_line = _line_with(pods_rule.splitlines(), PODS_HEADER)
    for token in (TERMINATE_CMD, "NEVER a user decision"):
        assert token in pods_line, (
            f"pods.md § Pods completion-side teardown clause must carry {token!r} "
            "(verified-done teardown of a suffixed pod is unconditional, #1662)"
        )

    carveout_line = _line_with(body.splitlines(), CARVEOUT_HEADER)
    for token in ("run complete + uploads verified", "#1112"):
        assert token in carveout_line, (
            f"CLAUDE.md carve-out completion-side teardown clause must carry {token!r} "
            "(uploads-verified precondition + the stop-is-not-durable negation, #1662)"
        )


def test_issue_skill_9a_ter_carries_completion_side_teardown_mirror():
    """SKILL.md Step 9a-ter carries the executing mirror of the contract
    (whole-file whitespace-normalized — the token appears exactly once)."""
    norm = _norm(ISSUE_SKILL.read_text())
    for token in (
        "Completion-side teardown",
        "--name-suffix <slug>",
        "never a user ask",
        "#1112",
    ):
        assert token in norm, (
            f"SKILL.md Step 9a-ter must carry the completion-side teardown mirror "
            f"token {token!r} (#1662: the inline/override rounds execute against "
            "this surface — dropping it reintroduces the pod-1586-b ask-gate)"
        )
