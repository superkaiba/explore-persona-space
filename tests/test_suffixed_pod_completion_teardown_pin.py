"""Pin the suffixed-pod completion-side teardown contract (#1662, #2086).

Incident: ``pod-1586-b`` (a suffixed follow-up pod, ~$12-13/hr) sat RUNNING
finished-but-not-terminated behind an ask-gate — run complete, artifacts
verified-uploaded, termination waiting on a user reply (the #664
idle-but-billing family). Task #1662 inserted a completion-side teardown
contract clause at three prose sites — the § Pods multi-pod paragraph
(post-compaction home: `.claude/rules/pods.md`, moved by 40653b5dcf; #2166),
the CLAUDE.md inline-override carve-out pod-safety block, and their
executing mirror in `/issue` SKILL.md Step 9a-ter. Task #2086 reconciled
the clauses with the compute-kill approval gate
(``backends/kill_approval.py``, 2026-08-04): verified-done teardown needs
NO user ask but is GATE-CONDITIONAL — the only sanctioned route is
``pod.py terminate`` (whose upload-verification guard enters the
owner-driven ``kill_approval.verified_teardown`` grant); on a kill-gate
refusal the session SURFACES the pod for approval and never self-approves
(``--approve`` / ``EPS_ALLOW_COMPUTE_KILL=1`` / ``EPS_ALLOW_POD_TERMINATE=1``
are user-only channels).

This test pins those clauses so a future CLAUDE.md / pods.md / SKILL.md
edit can neither reintroduce an ask-gate (the pod-1586-b idle-burn shape)
NOR reintroduce self-approval direction. It follows the
whitespace-normalize family pattern of
``tests/test_issue_skill_stopped_volume_persist_pin.py``; the two prose
occurrences (one per file post-compaction) are located PER LINE via their
distinct bold headers (both edit sites are single-line paragraphs).
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.issue_skill_source import issue_skill_text

ROOT = Path(__file__).resolve().parent.parent
CLAUDE_MD = ROOT / "CLAUDE.md"
# #2166: the doc-compaction commit 40653b5dcf moved the § Pods multi-pod
# paragraph (the suffixed-pod #1662 site) out of CLAUDE.md into
# .claude/rules/pods.md; the carve-out site stays in CLAUDE.md.
PODS_RULE_MD = ROOT / ".claude" / "rules" / "pods.md"
ISSUE_SKILL = ROOT / ".claude" / "skills" / "issue" / "SKILL.md"

# The two occurrences (one per file) carry DISTINCT bold headers (the locator).
PODS_HEADER = "**Completion-side teardown (suffixed pods — no user ask, gate-sanctioned; #1662):**"
CARVEOUT_HEADER = "**Completion-side teardown (no user ask, gate-sanctioned):**"
TERMINATE_CMD = "pod.py terminate --issue <N> --name-suffix <slug> --yes"

# #2086 reconciled-contract tokens: the no-user-ask form, the gated route's
# owner grant, the surface-on-refusal direction, and the self-approval ban.
NO_USER_ASK = "needs no user ask"
GRANT_TOKEN = "kill_approval.verified_teardown"
SURFACE_TOKEN = "SURFACE the pod for approval"
APPROVE_BAN = "--approve"
ENV_BAN = "EPS_ALLOW_COMPUTE_KILL"
ENV_BAN_ALIAS = "EPS_ALLOW_POD_TERMINATE"


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
    for token in (TERMINATE_CMD, NO_USER_ASK, GRANT_TOKEN, SURFACE_TOKEN):
        assert token in pods_line, (
            f"pods.md § Pods completion-side teardown clause must carry {token!r} "
            "(#2086: verified-done teardown of a suffixed pod needs no user ask but "
            "runs ONLY through the gated pod.py terminate; refusals surface, #1662)"
        )

    carveout_line = _line_with(body.splitlines(), CARVEOUT_HEADER)
    for token in (
        "run complete + uploads verified",
        "#1112",
        NO_USER_ASK,
        SURFACE_TOKEN,
        APPROVE_BAN,
        ENV_BAN,
    ):
        assert token in carveout_line, (
            f"CLAUDE.md carve-out completion-side teardown clause must carry {token!r} "
            "(uploads-verified precondition + the stop-is-not-durable negation, #1662; "
            "gate direction + self-approval ban, #2086)"
        )


def test_claude_md_pods_summary_carries_gate_direction():
    """The always-on CLAUDE.md § Pods summary carries the reconciled gate
    direction (#2086): no user ask via the gated route only, surface-on-
    refusal, and the self-approval ban (`--approve` / the approval env
    vars). Scoped to the § Pods section so the line-52 carve-out clause
    cannot satisfy it."""
    lines = CLAUDE_MD.read_text().splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("## Pods (")]
    assert len(starts) == 1, f"expected exactly one '## Pods (' section, found {len(starts)}"
    start = starts[0]
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")),
        len(lines),
    )
    section = _norm("\n".join(lines[start:end]))
    for token in (NO_USER_ASK, SURFACE_TOKEN, APPROVE_BAN, ENV_BAN, ENV_BAN_ALIAS):
        assert token in section, (
            f"CLAUDE.md § Pods summary must carry {token!r} (#2086: the always-on "
            "summary directs sessions to the gated pod.py terminate route and bans "
            "session-side self-approval; a future edit dropping it steers sessions "
            "back to ungated terminates or ask-gated idle burn)"
        )


def test_issue_skill_9a_ter_carries_completion_side_teardown_mirror():
    """SKILL.md Step 9a-ter carries the executing mirror of the contract
    (whole-file whitespace-normalized — the token appears exactly once)."""
    norm = _norm(issue_skill_text())
    for token in (
        "Completion-side teardown",
        "--name-suffix <slug>",
        NO_USER_ASK,
        GRANT_TOKEN,
        SURFACE_TOKEN,
        "#1112",
    ):
        assert token in norm, (
            f"SKILL.md Step 9a-ter must carry the completion-side teardown mirror "
            f"token {token!r} (#1662: the inline/override rounds execute against "
            "this surface — dropping it reintroduces the pod-1586-b ask-gate; "
            "#2086: the gate-conditional owner-grant form)"
        )
