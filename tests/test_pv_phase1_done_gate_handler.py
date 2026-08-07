"""Pin the `pv_phase1_done` gate handler in `/issue` SKILL.md Step 6d.4.

Issue #763's PV-extraction dispatcher (`scripts/issue763_dispatch.sh`) splits
phase 1 into a GPU half and an off-pod judge half, separated by a BLOCKING gate
sentinel `gate=pv_phase1_done`. Step 6d.4 dispatches one handler per registered
gate name; before #768 the registry held only `fact-candidates` (a PARK-mode
gate) plus an "Unrecognised `gate` name" catch-all that posts
`epm:failure unrecognised_gate_name` + `status:blocked`. Without a
`pv_phase1_done` handler the production run of `/issue 763` would fall through
to that catch-all and block.

This test pins the AUTO-RESOLVING handler so a future refactor cannot silently:
  - drop the gate name,
  - drop any of the four orchestrator sub-steps (stop → off-pod judge →
    resume → re-dispatch),
  - re-introduce the PARK behavior (CRON-TEARDOWN / EXIT) for this gate, or
  - leave the section-tail CRON-TEARDOWN paragraph unconditional (which would
    tear the cron down + park even for the auto-resolving gate).

It mirrors the SKILL.md text-assertion pattern of
`tests/test_issue_skill_marker_contract.py`. The primary verifier is the
implementer's grep + the Claude+Codex code-review ensemble; this test is the
mechanical backstop.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.issue_skill_source import issue_skill_text

ROOT = Path(__file__).resolve().parent.parent
ISSUE_SKILL = ROOT / ".claude" / "skills" / "issue" / "SKILL.md"

# The four orchestrator sub-step tokens the handler MUST name, in order.
REQUIRED_SUBSTEP_TOKENS = (
    "pod.py stop",
    "--phase judge",
    "pod.py resume",
    "--from-phase pv_capture",
)


def _gate_handlers_section(body: str) -> str:
    """Return the WHITESPACE-NORMALIZED Step 6d.4 "Gate handlers" list — from
    the "Gate handlers (one per registered" anchor up to the section-tail
    "PARK-mode gates only" guard sentence (exclusive). The handler MUST live
    inside this slice, not merely somewhere in the 6000-line file.

    Whitespace (including the markdown line-wraps inside a handler bullet, e.g.
    `scripts/pod.py\\n     stop --issue <N>`) is collapsed to single spaces so
    a token like `pod.py stop` matches across a soft line break — the prose is
    what matters, not the column the wrapper chose.
    """
    start = body.index("Gate handlers (one per registered")
    # The PARK-mode guard sentence marks the end of the per-gate handler list.
    end = body.index("PARK-mode gates only", start)
    return re.sub(r"\s+", " ", body[start:end])


def test_pv_phase1_done_handler_present_in_gate_handlers_section():
    """`pv_phase1_done` is registered as a handler inside the Step 6d.4 gate
    list (not just mentioned elsewhere in the file)."""
    section = _gate_handlers_section(issue_skill_text())
    assert "pv_phase1_done" in section, (
        "Step 6d.4 must register a `pv_phase1_done` gate handler; without it the "
        "issue #763 production run falls through to the 'Unrecognised gate' branch "
        "and blocks with epm:failure unrecognised_gate_name."
    )


def test_pv_phase1_done_handler_names_four_substeps():
    """The handler names all four orchestrator sub-steps in order: pod stop →
    off-pod `--phase judge` → pod resume → re-dispatch at `--from-phase
    pv_capture`."""
    section = _gate_handlers_section(issue_skill_text())
    last_idx = -1
    for token in REQUIRED_SUBSTEP_TOKENS:
        idx = section.find(token)
        assert idx != -1, (
            f"pv_phase1_done handler must name the sub-step token {token!r} "
            "(stop → off-pod judge → resume → re-dispatch)."
        )
        assert idx > last_idx, (
            f"pv_phase1_done sub-step {token!r} must appear in the canonical order "
            "(pod.py stop → --phase judge → pod.py resume → --from-phase pv_capture)."
        )
        last_idx = idx


def test_pv_phase1_done_handler_is_auto_resolve_not_park():
    """The handler is explicitly AUTO-RESOLVING and prohibits the PARK
    behavior (no CRON-TEARDOWN, no EXIT) so a future edit cannot quietly turn
    it back into a park-mode gate."""
    section = _gate_handlers_section(issue_skill_text())
    lower = section.lower()
    assert "auto-resolv" in lower, (
        "pv_phase1_done handler must state it AUTO-RESOLVES (contrast with the "
        "PARK-mode fact-candidates gate)."
    )
    assert "do not cron-teardown" in lower, (
        "pv_phase1_done handler must explicitly prohibit CRON-TEARDOWN — an "
        "auto-resolving gate keeps the pod (and the backstop cron) running."
    )
    assert "fact-candidates" in section, (
        "pv_phase1_done handler must contrast itself with the park-mode "
        "fact-candidates gate so a future reader sees why two gates differ."
    )


def test_section_tail_cron_teardown_scoped_to_park_mode():
    """The section-tail CRON-TEARDOWN/EXIT paragraph is guarded so it applies
    ONLY to PARK-mode gates — otherwise it would tear the cron down + park even
    for the auto-resolving pv_phase1_done gate."""
    body = issue_skill_text()
    guard_idx = body.find("PARK-mode gates only")
    teardown_idx = body.find("run CRON-TEARDOWN before parking")
    assert guard_idx != -1, (
        "Step 6d.4 section tail must carry the 'PARK-mode gates only' scope-guard "
        "sentence; without it the CRON-TEARDOWN/EXIT prose fires on every gate exit, "
        "including the auto-resolving pv_phase1_done gate."
    )
    assert teardown_idx != -1, "Step 6d.4 section tail must still describe CRON-TEARDOWN."
    assert guard_idx < teardown_idx, (
        "The 'PARK-mode gates only' guard sentence must precede the "
        "CRON-TEARDOWN-before-parking prose so the latter reads as park-mode-only."
    )
