"""Pin the canonical 4-H3 implementer marker contract across surfaces.

The `code-reviewer` agent + its Codex twin BOTH validate the implementer's
`epm:experiment-implementation` (and `epm:results`) marker against a fixed
4-H3 contract:

- `### (a) What was done`
- `### (b) Considered but not done`
- `### (c) How to verify`
- `### (d) Needs human eyeball`

Plus a separate `## Smoke run` H2 (per `code-reviewer.md` Step 0.6) with a
`### <phase-name>` sub-section per CPU-feasible pipeline phase — NEVER a
`### (d) Smoke run` H3 (that displaces `### (d) Needs human eyeball` and
is itself a `marker-shape` FAIL).

These tests pin that contract in two places so it cannot silently drift:

1. The `.claude/skills/issue/SKILL.md` Step 4b "Brief passed to the
   implementer" bullet list — the orchestrator's brief MUST quote the
   canonical labels verbatim, or the implementer follows the brief
   faithfully and the Codex code-reviewer FAILs on `marker-shape`.

2. The `.claude/agents/experiment-implementer.md` Report Format AND the
   `.claude/agents/code-reviewer.md` Step 0.5 validator MUST both name
   the same 4 H3 labels — a drift between writer + validator is
   equivalent to no contract.

Incident this pins against: task #506 round 1 (2026-06-06) — the
orchestrator dispatched the implementer with an ad-hoc brief (`(a) Plan
adherence / (b) Files touched / (c) How to run / (d) Smoke run / (e)
Needs human eyeball`). The implementer followed the brief faithfully;
the Codex `code-reviewer` correctly FAILed on `marker-shape`; the
reconciler upheld the BLOCKER; round 2 had to redo the report shape AND
re-implement the substantive code-review findings.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ISSUE_SKILL = ROOT / ".claude" / "skills" / "issue" / "SKILL.md"
EXPERIMENT_IMPLEMENTER = ROOT / ".claude" / "agents" / "experiment-implementer.md"
CODE_REVIEWER = ROOT / ".claude" / "agents" / "code-reviewer.md"


def _line_anchored_h3_pattern(label: str) -> re.Pattern[str]:
    """Build a regex that matches the label as a markdown H3 heading.

    A markdown H3 must appear at column 0 (no leading whitespace, no
    leading backtick — the latter is the prose/inline-code form, which
    is allowed). We anchor at line start and require a literal `### `
    prefix followed by the label and end of line.
    """
    return re.compile(rf"^### {re.escape(label)}\s*$", flags=re.MULTILINE)


# The canonical 4-H3 labels. Order matters: this is the order the
# `code-reviewer` Step 0.5 validator walks the marker.
CANONICAL_H3_LABELS = (
    "### (a) What was done",
    "### (b) Considered but not done",
    "### (c) How to verify",
    "### (d) Needs human eyeball",
)

# Ad-hoc labels observed in #506 round 1 that the Codex twin FAILed on.
# A future SKILL.md must NOT use any of these as a MARKDOWN HEADING
# (`### (a) Plan adherence`, etc.) — that is the template form the
# orchestrator would copy into the brief. Mentioning them in prose
# (warning, anti-pattern callout, incident citation) is fine; only the
# heading form is the regression risk.
ADHOC_BAD_LABELS = (
    "(a) Plan adherence",
    "(b) Files touched",
    "(c) How to run",
    "(d) Smoke run",
)


# ── /issue SKILL Step 4b brief specification ────────────────────────────────


def test_issue_skill_step4b_quotes_canonical_h3_labels_verbatim():
    """Step 4b's "Brief passed to the implementer" bullet list MUST quote
    the canonical 4 H3 labels verbatim so the orchestrator copies them
    unchanged into the brief — round 1 of #506 invented its own labels
    because the SKILL spec said only "Required `report-back` fields"
    with no verbatim contract.
    """
    body = ISSUE_SKILL.read_text()
    for label in CANONICAL_H3_LABELS:
        assert label in body, (
            f"Step 4b brief must quote the canonical H3 label verbatim: {label!r}. "
            "Without the verbatim contract the orchestrator invents ad-hoc labels "
            "and the Codex code-reviewer FAILs on marker-shape (incident: #506 round 1)."
        )


def test_issue_skill_step4b_mentions_smoke_run_h2():
    """The brief MUST distinguish the `## Smoke run` H2 from the `### (d)
    Needs human eyeball` H3 — folding the smoke run into the (d) slot
    (`### (d) Smoke run`) was the round-1 #506 anti-pattern that
    displaced the canonical (d) slot.
    """
    body = ISSUE_SKILL.read_text()
    assert "## Smoke run" in body, (
        "Step 4b brief must explicitly name the `## Smoke run` H2 contract "
        "so the orchestrator does not fold the smoke run into `### (d)`."
    )


def test_issue_skill_step4b_does_not_template_adhoc_labels_as_h3():
    """The SKILL spec may MENTION the ad-hoc round-1 #506 labels in PROSE
    (warning, anti-pattern callout, incident citation) but must NEVER
    use them as a MARKDOWN HEADING (`### (a) Plan adherence`, etc.) —
    the heading form is the template the orchestrator would copy
    verbatim into the brief, which is exactly what regressed #506.
    """
    body = ISSUE_SKILL.read_text()
    for bad in ADHOC_BAD_LABELS:
        pat = _line_anchored_h3_pattern(bad)
        match = pat.search(body)
        assert match is None, (
            f"{'### ' + bad!r} appears in {ISSUE_SKILL.name} as a line-anchored "
            f"markdown H3 heading (at offset {match.start() if match else '?'}); "
            "the ad-hoc round-1 #506 labels may appear in prose (backtick-wrapped "
            "citations, warnings, anti-pattern callouts) but NEVER as `### ` "
            "headings at column 0 — that is the copy-pasteable template form "
            "that regressed #506 in the first place."
        )


# ── Cross-surface consistency: writer (experiment-implementer) + validator
#    (code-reviewer) must agree on the H3 labels ────────────────────────────


def test_experiment_implementer_report_format_uses_canonical_h3_labels():
    body = EXPERIMENT_IMPLEMENTER.read_text()
    for label in CANONICAL_H3_LABELS:
        assert label in body, (
            f"experiment-implementer.md Report Format must use canonical H3 "
            f"label {label!r}; without it the implementer writes a marker the "
            "code-reviewer validator (Step 0.5) FAILs."
        )


def test_code_reviewer_step05_validates_canonical_h3_labels():
    body = CODE_REVIEWER.read_text()
    for label in CANONICAL_H3_LABELS:
        assert label in body, (
            f"code-reviewer.md Step 0.5 must reference canonical H3 label "
            f"{label!r}; without it the validator and the implementer's "
            "writer drift apart and a faithfully-following implementer FAILs."
        )


# ── Step 6 durability-pin shipping duty (#1230) ──────────────────────────────
#
# verify_plan.py c31 verifies a plan NAMES a durability pin; the Step 6
# bullet pinned here is the ONLY surface that verifies the named test
# SHIPS (the #1179 naming-vs-shipping residual). Pin the load-bearing
# shape: BOTH existence arms (NEW test in the round's diff OR a STANDING
# test already in the tree) — a later edit that "simplifies" the duty to
# diff-only turns every standing-pin plan into a false plan-adherence
# finding; dropping the bullet re-opens the gap.


def test_code_reviewer_step6_durability_pin_shipping_duty():
    body = CODE_REVIEWER.read_text()
    m = re.search(r"\*\*Durability-pin shipping check.*?(?=\n\s*\n)", body, flags=re.DOTALL)
    assert m, (
        "code-reviewer.md Step 6 must carry the **Durability-pin shipping "
        "check** bullet (#1230: c31 checks the pin is NAMED; this duty "
        "checks it SHIPS)"
    )
    step6 = body.index("### Step 6: Plan Deviation Check")
    step7 = body.index("### Step 7: Issue Verdict")
    assert step6 < m.start() < step7, "bullet must live inside Step 6"
    bullet = m.group(0)
    assert "Durability pin:" in bullet
    # Both existence arms — diff arm + standing-tree arm — must survive edits.
    assert re.search(r"(?i)round'?s diff", bullet), bullet
    assert re.search(r"(?i)standing", bullet), bullet
    assert re.search(r"(?i)neither", bullet), bullet
    # Miss classification: substantive plan-adherence finding.
    assert re.search(r"(?i)plan-adherence finding", bullet), bullet
    assert "substantive" in bullet.lower(), bullet
    # N/A-escape exemption clause must survive edits (no duty on N/A pins).
    assert re.search(r"(?i)carries no duty", bullet), bullet


# --- #1349: PASS_UNIFIED covers NON-cell smoke-axis minimum-N floors ------
#
# Incident #1315 r4: a PASS_UNIFIED-attested smoke sliced `questions[:1]`
# below `split_half_self_cosine`'s `len(qs) >= 2` floor and crashed at its
# LAST phase — the cell-subset threading clause (#546) was satisfied while a
# NON-cell axis (questions/rows/steps/draws) sat below a downstream
# minimum-N assert. The clause lives in TWO defining surfaces (the item-5
# PASS_UNIFIED definition in experiment-implementer.md + the
# `epm:smoke-architecture-check` schema in workflow.yaml, mirrored into the
# generated markers.md by `workflow_lint.py --emit-tables`); this test pins
# both so neither can silently drop the duty.


def test_item5_non_cell_min_n_clause_present():
    """#1349: PASS_UNIFIED definition covers non-cell smoke-axis min-N floors."""
    text = EXPERIMENT_IMPLEMENTER.read_text()
    # The non-cell-axis clause itself (agent-file defining surface).
    assert "NON-cell smoke axes" in text
    # The min-N duty: floors derive from downstream consumers' asserts.
    assert "minimum-N assert" in text
    # The per-axis attestation duty (named in the notes: line).
    assert "floor per sliced axis" in text
    # The workflow.yaml schema mirror (markers.md regen is lint-synced).
    yaml_text = (ROOT / ".claude" / "workflow.yaml").read_text()
    assert "Non-cell smoke axes" in yaml_text
