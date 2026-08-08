"""Pin the SKILL.md Step 2 minimum plan-review floor + recorded-skip contract (#1734).

The `/issue` SKILL.md Step 2 prose introduces a MINIMUM plan-review floor
that binds even on 1-line `kind: infra` workflow-surface edits (persist a
plan version + run verify_plan.py + spawn at minimum ONE Claude critic),
plus a recorded-skip contract requiring any leg SKIPPED below the full
stack to be named in the `epm:plan` note with a one-line reason.

Motivating incident: on 2026-07-26 three same-class `kind: infra`
daily-fix tasks got three different depths — #1696 (full stack;
`epm:plan-verify` = 1), #1692 (planner + one critic bypassing the skill
dispatcher, `epm:plan-verify` = 0; single critic still returned REVISE
with 2 Must-Fix items — session `a5a4b7bd`), and #1709 (zero agents,
self-authored plan — session `e3b70618`). The floor is the standing
counter to that sink-to-zero drift; this test keeps its load-bearing
tokens from silently disappearing.

Follows `tests/test_issue_skill_marker_contract.py`'s grep-anchored
existence-check pattern (heading-form ban + literal-token presence).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ISSUE_SKILL = ROOT / ".claude" / "skills" / "issue" / "SKILL.md"


def _step2_span(body: str) -> str:
    """Return the ### Step 2 span through the next H3 heading.

    Step 2 begins at `### Step 2: Adversarial planning` and ends at the
    next `### ` heading (Step 3 today). We scope the greps to this span
    so an unrelated later section mentioning e.g. `epm:plan-verify` in
    a warning does not satisfy the check.
    """
    start = body.index("### Step 2: Adversarial planning")
    tail = body[start:]
    # Find the next H3 heading (line-anchored `### `).
    m = re.search(r"\n### ", tail[len("### Step 2: Adversarial planning") :])
    end = len(tail) if m is None else len("### Step 2: Adversarial planning") + m.start()
    return tail[:end]


# ── Load-bearing sentence: the H4 header of the new block ───────────────────


def test_step2_floor_h4_header_present():
    """The 'Minimum plan-review floor ...' header phrase must appear in Step 2."""
    span = _step2_span(ISSUE_SKILL.read_text(encoding="utf-8"))
    assert "Minimum plan-review floor" in span, (
        "Step 2 must carry the 'Minimum plan-review floor' header phrase "
        "so the recorded-skip contract has an anchor sessions can find. "
        "Incident #1734: three same-class kind:infra tasks on 2026-07-26 "
        "sank to three different depths because no floor was documented."
    )


# ── Load-bearing sentence: the CLAUDE.md carve-out disclaimer ───────────────


def test_step2_floor_carve_out_disclaimer_present():
    """Step 2 must state the CLAUDE.md carve-out does NOT reach wf-fix tasks."""
    span = _step2_span(ISSUE_SKILL.read_text(encoding="utf-8"))
    assert "carve-out does NOT reach" in span, (
        "Step 2 must state the CLAUDE.md /adversarial-planner carve-out "
        "does NOT reach kind:infra workflow-fix tasks. Without the "
        "disclaimer, a session reading the CLAUDE.md 'bug fixes ... skip "
        "it' clause loosely can still self-certify past the floor."
    )


# ── The three floor legs by load-bearing token ──────────────────────────────


def test_step2_floor_leg1_new_plan_version():
    """Leg 1: persist a plan version via `new-plan-version`."""
    span = _step2_span(ISSUE_SKILL.read_text(encoding="utf-8"))
    assert "new-plan-version" in span, (
        "Step 2 must name `new-plan-version` as the persistence path "
        "(a Write-authored plan is invisible to verify_plan.py --issue)."
    )


def test_step2_floor_leg2_epm_plan_verify():
    """Leg 2: run verify_plan.py and post `epm:plan-verify`."""
    span = _step2_span(ISSUE_SKILL.read_text(encoding="utf-8"))
    assert "epm:plan-verify" in span, (
        "Step 2 must name the `epm:plan-verify` marker as leg-2's "
        "durable proof the mechanical pre-pass ran."
    )


def test_step2_floor_leg3_at_least_one_claude_critic():
    """Leg 3: spawn at minimum ONE Claude `critic` (Codex-only is not sufficient).

    Loosened regex per Change 4a plan note — accepts "at minimum ONE
    Claude ... critic" or "at least one Claude ... critic" so a
    legitimate reword (e.g. picking a specific lens name inside the
    span) doesn't false-positive.
    """
    span = _step2_span(ISSUE_SKILL.read_text(encoding="utf-8"))
    m = re.search(r"(?:at minimum ONE|at least one) Claude .*?critic", span)
    assert m is not None, (
        "Step 2 must name leg 3 as spawning at minimum ONE Claude "
        "`critic` on a workflow-surface edit — Codex-only is not "
        "sufficient because the code-review ensemble at Step 5 already "
        "runs Claude+Codex on the diff, so the plan-review adds Claude "
        "by default. #1692's single critic returned REVISE with two "
        "Must-Fix findings on a same-class task."
    )


# ── The recorded-skip contract ──────────────────────────────────────────────


def test_step2_recorded_skip_contract_present():
    """The 'Recorded-skip contract' sub-header and its shape must be present."""
    span = _step2_span(ISSUE_SKILL.read_text(encoding="utf-8"))
    assert "Recorded-skip contract" in span, (
        "Step 2 must carry a 'Recorded-skip contract' sub-header so "
        "sessions know skipping a leg is auditable ex post."
    )
    # Whitespace-normalize so hard-wrapped prose (`epm:plan` at end of one
    # line, `note with a one-line reason` at start of the next) satisfies
    # the check — the load-bearing tokens are what matter, not the line
    # break.
    normalized = " ".join(span.split())
    assert "in the `epm:plan` note with a one-line reason" in normalized, (
        "The recorded-skip contract must name the `epm:plan` note as "
        "the durable home of the skip reason — the shape #1709 used, "
        "quoted in the plan §4.1 prose."
    )


# ── Heading-form ban: ad-hoc H3s must not template floor legs as headings ───
#
# Mirrors `test_issue_skill_marker_contract.py`'s ADHOC_BAD_LABELS check —
# the load-bearing tokens above may appear in prose (backtick-wrapped),
# but must NEVER appear as line-anchored `### ` H3 headings in the
# Step 2 span, because that heading form is the template shape the
# orchestrator would copy verbatim into a brief.


_LINE_ANCHORED_H3_BAD_LABELS = (
    "Bug-fix category",
    "Skipped leg",
    "Floor leg 1",
    "Floor leg 2",
    "Floor leg 3",
)


def test_step2_floor_no_adhoc_h3_labels():
    """Ad-hoc floor-leg labels must not appear as line-anchored H3 headings."""
    body = ISSUE_SKILL.read_text(encoding="utf-8")
    for bad in _LINE_ANCHORED_H3_BAD_LABELS:
        pat = re.compile(rf"^### {re.escape(bad)}\s*$", flags=re.MULTILINE)
        m = pat.search(body)
        assert m is None, (
            f"{'### ' + bad!r} appears in {ISSUE_SKILL.name} as a "
            "line-anchored H3 heading — the floor-leg tokens must live "
            "in prose (backtick-wrapped citations, numbered bullets) "
            "but NEVER as `### ` headings at column 0, since that is "
            "the copy-pasteable template form that seeds ad-hoc briefs."
        )
