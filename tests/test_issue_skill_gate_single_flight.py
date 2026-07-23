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

SKILL = Path(__file__).resolve().parents[1] / ".claude" / "skills" / "issue" / "SKILL.md"

LABEL = "Single-flight probe (#1606)"


def _text() -> str:
    return SKILL.read_text(encoding="utf-8")


def test_step9c_single_flight_statement_present_and_precedes_launch():
    text = _text()
    # File-wide durability: one shared statement + 4 per-site hooks (#1627).
    # Dropping ANY per-site hook fails this pin, not just the AC-1 grep.
    assert text.count(LABEL) >= 5
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
    probe = "pgrep -af 'issue-<N>-surgical-outcome[.]txt|scripts/workflow_lint[.]py'"
    assert probe in text
    assert text.index(probe) < text.index("rm -f /tmp/issue-<N>-surgical-outcome.txt")
