"""Pin the Step 9c/10d gate single-flight guard (#1627, incident #1606; #1821).

#1606 (transcript b8b69a72): an improvised Monitor keyed on rc-file EXISTENCE
false-fired "done" twice while the gate pytest was still printing, and a
SECOND gate run was launched into the live one — 4 live gate pids, two
fail-CLOSED verdict/sha blocks, ~12 min churn. #1742 then re-hit the
documented self-match trap: the probe folded inside the launch call matched
its own wrapper argv despite the SKILL.md placement rule. These tests pin the
guard as of #1821: the prescribed probe at every Single-flight site is the
MECHANICALLY self-/ancestor-pid-excluding helper
(``scripts/step9c_baseline.py probe`` — exit 0 = clear), placed before every
Step 9c / Step 10d gate launch, plus the process-exit-keyed done condition
restated at the compose sites. Raw bracketed pgrep survives only in the
kill/recovery arms (which want the pid list), each with an inline
exit-inversion pointer.
"""

from pathlib import Path

from tests.issue_skill_source import issue_skill_text
from tests.test_issue_skill_inline_gate_pin import _gate_section

SKILL = Path(__file__).resolve().parents[1] / ".claude" / "skills" / "issue" / "SKILL.md"

LABEL = "Single-flight probe (#1606)"

PROBE_ISSUE_FORM = 'uv run python "$REPO_ROOT"/scripts/step9c_baseline.py probe --issue <N>'


def _text() -> str:
    return issue_skill_text()


def test_step9c_single_flight_statement_present_and_precedes_launch():
    text = _text()
    # File-wide durability: one shared statement + 5 per-site hooks
    # (#1627; 9a-ter inline-gate hook added #1647). Dropping ANY per-site
    # hook fails this pin, not just the AC-1 grep.
    assert text.count(LABEL) >= 6
    sec = text[text.index("9c. Test-verdict gate") : text.index("### Step 10: Auto-complete")]
    assert LABEL in sec
    # Self-/ancestor-excluding helper probe (#1821; --issue form so the
    # probe's own argv never carries the junit filename):
    assert PROBE_ISSUE_FORM in sec
    # The 1b Monitor until-loop keys on the helper's --issue form (fixed,
    # valid regex — an exit-2 inside an until-loop would spin forever):
    assert f"until {PROBE_ISSUE_FORM} >/dev/null" in sec
    # The raw pgrep junit probe is no longer the prescribed single-flight
    # command (bracketed or not); the kill-arm's `[p]ytest.*` probe differs.
    assert "pgrep -af 'step9c-junit-issue-<N>[.]xml'" not in sec  # replaced by the helper (#1821)
    assert "pgrep -af 'step9c-junit-issue-<N>.xml'" not in sec  # unbracketed = self-match
    # The statement precedes the 1b launch preamble (first three-file rm):
    assert sec.index(LABEL) < sec.index(
        "rm -f /tmp/step9c-junit-issue-<N>.xml /tmp/step9c-rc-issue-<N>"
    )
    # Monitor discipline restated at the compose site:
    assert "NEVER on rc/verdict-file existence alone" in sec
    # The retained kill-arm pgrep carries the exit-inversion pointer (#1821):
    assert "pgrep -af '[p]ytest.*step9c-junit-issue-<N>'" in sec
    assert "INVERTED vs `step9c_baseline.py probe`" in sec


def test_step10d_gate_single_flight_hook_present_and_precedes_launch():
    text = _text()
    start = text.index("#### Pre-push workflow-lint gate")
    region = text[start : text.index("#### The auto-merge procedure", start)]
    assert LABEL in region
    # Self-/ancestor-excluding helper probe (#1821):
    assert (
        'uv run python "$REPO_ROOT"/scripts/step9c_baseline.py probe '
        "--pattern 'issue-<N>-lint-gate-tree'" in region
    )
    assert region.index(LABEL) < region.index("rm -f /tmp/issue-<N>-lint-verdict.txt")
    # The completion-read recovery arm keeps the bracketed pgrep (it wants
    # the pid list) WITH the exit-inversion pointer:
    assert "pgrep -af 'issue-<N>-lint-gate-tre[e]'" in region
    assert "INVERTED vs `step9c_baseline.py probe`" in region


def test_step10d_surgical_single_flight_hook_present():
    text = _text()
    # Issue-scoped alternate (task #1719) carried into the helper form
    # (#1821): the fleet-wide `scripts/workflow_lint[.]py` alternate stays
    # banned, and the helper's self-exclusion needs no bracket.
    probe = (
        'uv run python "$REPO_ROOT"/scripts/step9c_baseline.py probe '
        r"--pattern 'issue-<N>-surgical-outcome\.txt|issue-<N>-lint-gate-tree'"
    )
    assert probe in text
    # Regression: the previous fleet-wide alternate must not creep back in.
    assert "pgrep -af 'issue-<N>-surgical-outcome[.]txt|scripts/workflow_lint[.]py'" not in text
    # The raw pgrep form is no longer the prescribed surgical probe (#1821):
    assert "pgrep -af 'issue-<N>-surgical-outcome[.]txt|issue-<N>-lint-gate-tre[e]'" not in text
    assert text.index(probe) < text.index("rm -f /tmp/issue-<N>-surgical-outcome.txt")


FLEET_PROBE_FORM = (
    'uv run python "$REPO_ROOT"/scripts/step9c_baseline.py probe --fleet --exclude-issue <N>'
)


def test_gate_fleet_arbitration_hooks_present():
    """#1962: the cross-issue gate-fleet arbitration probe is present at each
    gate-launch site AFTER the per-issue single-flight probe and BEFORE the
    launch (canonical paragraph at Step 9c 1b; short hooks at 1d, both Step
    10d gate blocks, and the Step 9a-ter inline payload gate). Additive pin
    — the per-issue probe pins above are untouched."""
    text = _text()
    # 9c 1b canonical (+ its until-loop) + 1d + 10d (i)/(ii) + 10d (iii)
    # + 9a-ter — at least 5 hooked sites carry the invocation verbatim:
    assert text.count(FLEET_PROBE_FORM) >= 5
    assert "Gate-fleet arbitration (#1962)" in text
    assert "[gate-fleet] cap-expired after 45 min — launching over cap" in text

    sec = text[text.index("9c. Test-verdict gate") : text.index("### Step 10: Auto-complete")]
    # 1b: the canonical paragraph sits AFTER the per-issue probe statement
    # and BEFORE the launch preamble rm:
    assert sec.index(FLEET_PROBE_FORM) > sec.index(PROBE_ISSUE_FORM)
    assert sec.index(FLEET_PROBE_FORM) < sec.index(
        "rm -f /tmp/step9c-junit-issue-<N>.xml /tmp/step9c-rc-issue-<N>"
    )
    # Queue shape: the bounded fleet until-loop (fixed internal regex — the
    # loop can never spin on exit 2):
    assert f"until {FLEET_PROBE_FORM} >/dev/null" in sec
    # 1d: a further in-section hook between the 1b launch and the compare rm:
    assert sec.rindex(FLEET_PROBE_FORM) > sec.index("rm -f /tmp/step9c-junit-issue-<N>.xml")
    assert sec.rindex(FLEET_PROBE_FORM) < sec.index("rm -f /tmp/step9c-compare-issue-<N>.json")

    # 10d forms (i)/(ii): hook precedes the stale-verdict rm:
    start = text.index("#### Pre-push workflow-lint gate")
    region = text[start : text.index("#### The auto-merge procedure", start)]
    assert FLEET_PROBE_FORM in region
    assert region.index(FLEET_PROBE_FORM) < region.index("rm -f /tmp/issue-<N>-lint-verdict.txt")

    # 10d form (iii): hook after the surgical per-issue probe, before the
    # outcome-sentinel rm:
    surgical_probe = (
        'uv run python "$REPO_ROOT"/scripts/step9c_baseline.py probe '
        r"--pattern 'issue-<N>-surgical-outcome\.txt|issue-<N>-lint-gate-tree'"
    )
    fleet_after_surgical = text.index(FLEET_PROBE_FORM, text.index(surgical_probe))
    assert fleet_after_surgical < text.index("rm -f /tmp/issue-<N>-surgical-outcome.txt")

    # 9a-ter inline payload lint gate: hook precedes the fenced helper launch:
    sec9 = _gate_section()
    assert FLEET_PROBE_FORM in sec9
    assert sec9.index(FLEET_PROBE_FORM) < sec9.index("uv run python scripts/inline_lint_gate.py")


def test_step9ater_inline_gate_single_flight_hook_present_and_precedes_launch():
    """#1647: the Step 9a-ter inline payload lint gate carries its own
    per-site single-flight hook (the #1606 pattern at the inline site),
    issue-scoped via the -inline-payload suffix, placed BEFORE the fenced
    gate launch; #1821 swaps it to the self-excluding helper form; #1948
    widens the pattern to match round-unique payload names (the `issue-<N>-`
    prefix + `inline-payload\\.txt` tail keep it exact-issue-scoped)."""
    sec = _gate_section()
    assert LABEL in sec
    # Self-/ancestor-excluding helper probe (#1821; #1948 round-unique widening):
    assert (
        'uv run python "$REPO_ROOT"/scripts/step9c_baseline.py probe '
        r"--pattern 'issue-<N>-[^ ]*inline-payload\.txt'" in sec
    )
    # The raw pgrep form is no longer the prescribed inline probe (#1821):
    assert "pgrep -af 'issue-<N>-inline-payload[.]txt'" not in sec
    assert "pgrep -af 'issue-<N>-inline-payload.txt'" not in sec  # unbracketed = self-match
    # The statement precedes the fenced helper launch:
    assert sec.index(LABEL) < sec.index("uv run python scripts/inline_lint_gate.py")
