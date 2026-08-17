"""Pin the stopped-volume NON-durability persist duty (#1595, incident #1112).

Incident #1112 (2026-07-21) falsified the workflow's stopped-pod durability
belief: a pod stopped at 07:25Z with its volume preserved and the
`keep-running` tag set vanished from the live RunPod API ~22h later
(``{"data": {"pod": null}}``), destroying the volume's done-JSONs and resume
state and forcing a full re-run. Task #1595 rewrote the two stop-prescribing
`/issue` SKILL.md recipes (User pause affordance step 1, Step 8-bis) to
require persisting resume state OFF-POD before any multi-hour park, and added
the canonical rule as a pod-config.md H2.

This test pins those three surfaces so a future SKILL.md / pod-config.md edit
cannot silently reintroduce the falsified "volume preserved until the daily
stale-pod audit" durability claim or drop the persist-first duty. It mirrors
the SKILL.md text-assertion pattern of
``tests/test_issue_skill_marker_contract.py`` /
``tests/test_pv_phase1_done_gate_handler.py``: sections are WHITESPACE-
NORMALIZED before token match — including the NEGATIVE assert, because the
falsified literal is soft-line-wrapped in the pre-#1595 SKILL.md, so a
raw-substring absence check would vacuously pass.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.issue_skill_source import issue_skill_text

ROOT = Path(__file__).resolve().parent.parent
ISSUE_SKILL = ROOT / ".claude" / "skills" / "issue" / "SKILL.md"
POD_CONFIG_RULE = ROOT / ".claude" / "rules" / "pod-config.md"

# The falsified durability claim (#1112). Scoped to SKILL.md ONLY — historical
# artifacts (e.g. cached plans under .claude/plans/) legitimately quote it.
FALSIFIED_LITERAL = "volume preserved until the daily stale-pod audit"


def _norm(text: str) -> str:
    """Collapse all whitespace to single spaces so tokens match across the
    markdown soft line breaks the SKILL.md wrapper introduces."""
    return re.sub(r"\s+", " ", text)


def _pause_affordance_section(body: str) -> str:
    """The User-pause-affordance step-1 span: from the section anchor to step
    2's "LAST — the commit point" anchor (exclusive), whitespace-normalized."""
    start = body.index("User pause affordance")
    end = body.index("LAST — the commit point", start)
    return _norm(body[start:end])


def _step_8bis_section(body: str) -> str:
    """The Step 8-bis section: heading to the Step 9 heading (exclusive)."""
    start = body.index("#### Step 8-bis: Pod must not idle on a halt")
    end = body.index("### Step 9", start)
    return _norm(body[start:end])


def test_pause_affordance_persist_duty_present():
    """Pause-affordance step 1 carries the persist-first duty AND no longer
    carries the #1112-falsified stopped-volume durability claim."""
    body = issue_skill_text()
    section = _pause_affordance_section(body)
    for token in ("NOT durable", "done-JSONs", "HF prefix"):
        assert token in section, (
            f"User pause affordance step 1 must carry the persist-first duty token {token!r} "
            "(a stopped volume is NOT durable — upload resume state before pod.py stop; #1112)."
        )
    assert FALSIFIED_LITERAL not in _norm(body), (
        "SKILL.md must not reassert the #1112-falsified stopped-volume durability claim "
        f"({FALSIFIED_LITERAL!r}): RunPod destroyed a stopped, keep-running-tagged pod well "
        "inside that window."
    )


def test_step_8bis_persist_sentence_present():
    """Step 8-bis carries the persist-before-stop sentence (resume state,
    non-durability, the #1112 incident)."""
    section = _step_8bis_section(issue_skill_text())
    for token in ("resume state", "NOT durable", "#1112"):
        assert token in section, (
            f"Step 8-bis must carry the persist-before-stop token {token!r} "
            "(a gate/crash park routinely outlasts an hour; a stopped volume is NOT durable)."
        )


def test_pod_config_rule_section_present():
    """pod-config.md carries the canonical non-durability H2 (line-anchored)."""
    lines = POD_CONFIG_RULE.read_text().splitlines()
    assert any(line.startswith("## Stopped pod volume is NOT durable") for line in lines), (
        "pod-config.md must carry the canonical '## Stopped pod volume is NOT durable' H2 "
        "(the #1595 rule: persist resume state to HF before a multi-hour park; #1112)."
    )
