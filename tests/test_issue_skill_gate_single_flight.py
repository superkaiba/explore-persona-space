"""Pin the Step 9c/10d gate single-flight guard (#1627, incident #1606).

#1606 (transcript b8b69a72): an improvised Monitor keyed on rc-file EXISTENCE
false-fired "done" twice while the gate pytest was still printing, and a
SECOND gate run was launched into the live one — 4 live gate pids, two
fail-CLOSED verdict/sha blocks, ~12 min churn. These tests pin the guard: a
pre-launch single-flight probe (own FOREGROUND call, bracketed pattern,
issue-scoped) before every Step 9c / Step 10d gate launch, plus the
process-exit-keyed done condition restated at the compose sites.
"""

from pathlib import Path

from tests.test_issue_skill_inline_gate_pin import _gate_section

SKILL = Path(__file__).resolve().parents[1] / ".claude" / "skills" / "issue" / "SKILL.md"

LABEL = "Single-flight probe (#1606)"


def _text() -> str:
    return SKILL.read_text(encoding="utf-8")


def test_step9c_single_flight_statement_present_and_precedes_launch():
    text = _text()
    # File-wide durability: one shared statement + 5 per-site hooks
    # (#1627; 9a-ter inline-gate hook added #1647). Dropping ANY per-site
    # hook fails this pin, not just the AC-1 grep.
    assert text.count(LABEL) >= 6
    sec = text[text.index("9c. Test-verdict gate") : text.index("### Step 10: Auto-complete")]
    assert LABEL in sec
    # Bracketed, issue-scoped family probe (gotchas.md self-match entry):
    assert "pgrep -af 'step9c-junit-issue-<N>[.]xml'" in sec
    assert "pgrep -af 'step9c-junit-issue-<N>.xml'" not in sec  # unbracketed = self-match
    # The statement precedes the 1b launch preamble (first three-file rm):
    assert sec.index(LABEL) < sec.index(
        "rm -f /tmp/step9c-junit-issue-<N>.xml /tmp/step9c-rc-issue-<N>"
    )
    # Monitor discipline restated at the compose site:
    assert "NEVER on rc/verdict-file existence alone" in sec


def test_step10d_gate_single_flight_hook_present_and_precedes_launch():
    text = _text()
    start = text.index("#### Pre-push workflow-lint gate")
    region = text[start : text.index("#### The auto-merge procedure", start)]
    assert LABEL in region
    assert region.index(LABEL) < region.index("rm -f /tmp/issue-<N>-lint-verdict.txt")


def test_step10d_surgical_single_flight_hook_present():
    text = _text()
    # Issue-scoped alternate (task #1719): the fleet-wide `scripts/workflow_lint[.]py`
    # alternate was replaced with the `issue-<N>-lint-gate-tre[e]` shape used at
    # L10418/L10792, so a sibling session's root lint cannot phantom-match.
    probe = "pgrep -af 'issue-<N>-surgical-outcome[.]txt|issue-<N>-lint-gate-tre[e]'"
    assert probe in text
    # Regression: the previous fleet-wide alternate must not creep back in.
    assert "pgrep -af 'issue-<N>-surgical-outcome[.]txt|scripts/workflow_lint[.]py'" not in text
    assert text.index(probe) < text.index("rm -f /tmp/issue-<N>-surgical-outcome.txt")


def test_step9ater_inline_gate_single_flight_hook_present_and_precedes_launch():
    """#1647: the Step 9a-ter inline payload lint gate carries its own
    per-site single-flight hook (the #1606 pattern at the inline site),
    bracketed + issue-scoped, placed BEFORE the fenced gate launch."""
    sec = _gate_section()
    assert LABEL in sec
    # Bracketed, issue-scoped probe (gotchas.md self-match entry):
    assert "pgrep -af 'issue-<N>-inline-payload[.]txt'" in sec
    assert "pgrep -af 'issue-<N>-inline-payload.txt'" not in sec  # unbracketed = self-match
    # The statement precedes the fenced helper launch:
    assert sec.index(LABEL) < sec.index("uv run python scripts/inline_lint_gate.py")
