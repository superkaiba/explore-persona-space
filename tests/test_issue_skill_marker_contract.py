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
ISSUE_TICK_SKILL = ROOT / ".claude" / "skills" / "issue-tick" / "SKILL.md"
EXPERIMENT_IMPLEMENTER = ROOT / ".claude" / "agents" / "experiment-implementer.md"
CODE_REVIEWER = ROOT / ".claude" / "agents" / "code-reviewer.md"
IMPLEMENTER = ROOT / ".claude" / "agents" / "implementer.md"
CRASH_FIX_ROUNDS = ROOT / ".claude" / "rules" / "crash-fix-rounds.md"


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


# --- #1409: smoke exercises data-dependent gates at smoke n ---------------
#
# Incident #1345: 4 pre-existing data gates in reused code (a fold-skip, an
# `n_common > 0` assert, two count asserts) had never executed at smoke n;
# they first fired in production — two serialized GCP crashes (2026-07-15).


def test_item3_data_gate_exercise_clause_present():
    """#1409: item 3 demands data-dependent gate execution at smoke n."""
    text = EXPERIMENT_IMPLEMENTER.read_text()
    assert "Data-dependent gates, not just the happy path" in text
    assert "data gates exercised:" in text
    assert "production-only — <one-line" in text
    assert "COMPOSES with the item-5 resize-up duty" in text


# --- #1611: Step 6d.0-bis arm-class clause names the class-defining axes ---
#
# Incident #1586 r3/r4/r6: every recorded smoke ran ONE content-class
# full-FT cell (`syc-pers-ft-con-s137`) of a marker/content x con/po x
# LoRA/full-FT grid; a read-side panel-disjointness check (r3) killed the
# smoke leg itself at its first full-panel read, and two class-gated bug
# classes the smoke cell could not reach then surfaced live post-smoke,
# one per phase (a marker-po mix row-count assert, a reuse-seam loader).
# The Step 6d.0-bis arm-class clause must keep naming the class-defining
# axes so a multi-class dispatcher's smoke set is composed per realized
# (class x regime) combination.


def test_step6d0bis_smoke_covers_class_regime_axes():
    """#1611: Step 6d.0-bis arm-class clause names the class-defining axes (#1586)."""
    text = ISSUE_SKILL.read_text(encoding="utf-8")
    heading = "##### Step 6d.0-bis"
    assert text.count(heading) == 1, "Step 6d.0-bis H5 heading literal must stay unique"
    i = text.index(heading)
    # Whitespace-normalize so hard-wrapped SKILL.md prose cannot split a needle.
    span = " ".join(text[i : i + 6000].split())
    for needle in (
        "PER ARM CLASS",
        "behavior class",
        "training regime",
        "#1586",
        "Smoke/production parity includes REGIME/CLASS COVERAGE",
    ):
        assert needle in span, needle


# ── SHA-verbatim report rule (#1682) ─────────────────────────────────────────
#
# Incident #1586 r7 (2026-07-24): an implementer report's fix-engaged
# element-4 "full" SHA was a hand-extended short SHA; the orchestrator had
# to rev-parse-correct it before composing the relaunch brief. Downstream
# consumers (relaunch briefs, ancestry probes, markers) re-cite report SHAs
# verbatim, so the compose-site rule must survive in all three surfaces:
# both implementer report contracts and the crash-fix element-4 field spec.


def test_sha_verbatim_report_rule_pinned():
    """#1682: the SHA-verbatim rule survives in all three compose-site surfaces.

    Pins BOTH the ban token (``hand-extended``) and the positive
    instruction (``rev-parse``) — dropping either half re-opens the
    #1586 r7 fabricated-hex class, because downstream relaunch briefs
    and markers re-cite the report's SHAs verbatim.
    """
    for path in (EXPERIMENT_IMPLEMENTER, IMPLEMENTER, CRASH_FIX_ROUNDS):
        body = path.read_text(encoding="utf-8")
        for needle in ("hand-extended", "rev-parse"):
            assert needle in body, (
                f"{path.name} must keep the SHA-verbatim report rule token "
                f"{needle!r} (#1682; incident #1586 r7: a hand-extended 'full' "
                "SHA reached the fix-engaged field and had to be "
                "rev-parse-corrected before the relaunch brief — downstream "
                "briefs/markers re-cite report SHAs verbatim)."
            )


def test_skill_md_documents_api_error_after_marker():
    """#1695 durability pin: the issue-tick SKILL.md documents the new
    ``api-error-after-marker`` STALE-REDRIVE reason (the token that
    `tick_triage.api_error_after_marker_reason` embeds in the verdict
    string). A future editor silently dropping the docs row would leave
    the runtime behavior unexplained; this test greps the literal token
    so the SKILL.md stays in sync with the runtime."""
    body = ISSUE_TICK_SKILL.read_text(encoding="utf-8")
    assert "api-error-after-marker" in body, (
        "issue-tick SKILL.md must document the api-error-after-marker "
        "STALE-REDRIVE reason (#1695). The token is what "
        "tick_triage.api_error_after_marker_reason embeds in the verdict "
        "string — the docs and the runtime must not drift."
    )


# --- #2171: the Step 6d.0 PASS_AUTHORIZED_STUB grant escape --------------
#
# Incident #2163 (2026-08-07): the gate's own documented resolution
# ("re-authorize the stubs in §4 Design") had no landing token — every
# surface annotated it "not yet wired" / "v1.1" — and the orchestrator
# improvised a shape-violating PASS_UNIFIED grant. #2171 wired the fifth
# token, granted ONLY by `task.py check-authorized-stub` (rc=0).


def test_step6d0_routing_table_has_authorized_stub_row():
    """#2171 durability pin: the Step 6d.0 region carries the grant row —
    the token, the mechanical checker command, and NO stale 'not yet wired'
    annotation (the row's grant path is the checker's exit code, never
    orchestrator prose judgment)."""
    body = ISSUE_SKILL.read_text(encoding="utf-8")
    start = body.find("##### Step 6d.0:")
    assert start != -1, "the Step 6d.0 heading vanished from issue/SKILL.md"
    end = body.find("##### Step 6d.0-bis", start + 1)
    region = body[start:end] if end != -1 else body[start:]
    assert "`PASS_AUTHORIZED_STUB arms_stubbed=<comma-list>`" in region, (
        "the Step 6d.0 routing table lost its PASS_AUTHORIZED_STUB row "
        "(#2171) — the gate's sanctioned stub-authorization escape has no "
        "landing token again (the #2163 incident shape)."
    )
    assert "check-authorized-stub" in region, (
        "the Step 6d.0 region no longer names task.py check-authorized-stub "
        "(#2171) — the grant must be the checker's exit code (#397)."
    )
    assert "not yet wired" not in region, (
        "the Step 6d.0 region regained a stale 'not yet wired' annotation (#2171 wired the escape)."
    )


def test_smoke_arch_marker_schema_names_authorized_stub_token():
    """#2171, the #1349 two-surface pattern: the workflow.yaml marker schema
    mirror documents the fifth token + its grant mechanics (markers.md regen
    is lint-synced via --emit-tables + the authorized-stub wiring check)."""
    yaml_text = (ROOT / ".claude" / "workflow.yaml").read_text(encoding="utf-8")
    assert "PASS_AUTHORIZED_STUB arms_stubbed=<comma-list>" in yaml_text
    assert "check-authorized-stub" in yaml_text
    assert "canary-like exception, v1.1" not in yaml_text, (
        "workflow.yaml regained the stale 'canary-like exception, v1.1' "
        "annotation (#2171 wired the escape)."
    )


def test_implementer_item5_self_tag_clause_present():
    """#2171 (criterion 4), the #1349 clause-presence pattern: the
    experiment-implementer item-5 verdict vocabulary carries the self-tag
    RULE (when to post the new token instead of PASS_PARTIAL), not just the
    token string."""
    text = EXPERIMENT_IMPLEMENTER.read_text(encoding="utf-8")
    idx = text.find("PASS_AUTHORIZED_STUB")
    assert idx != -1, "experiment-implementer.md lost the PASS_AUTHORIZED_STUB token (#2171)"
    assert "INSTEAD of `PASS_PARTIAL`" in text
    assert "Authorized smoke stubs" in text
    # The mis-tag consequence: tagging without plan coverage only buys a
    # bounce — the checker refuses (never a silent grant).
    assert "check-authorized-stub" in text


# --- #2176: the Step 6d.0 arm-registry enumeration contract ----------------
#
# Incident #2163 (2026-08-07): the per-arm enumeration was hand-listed from
# plan narrative, so the "every arm resolves REAL or N/A" invariant was
# quantified over an unverified set — 3 of 13 registry phases were silently
# omitted, and the two never-smoked VM-side ones each carried an
# `args.<attr>` AttributeError that fired at Step 8. #2176 makes the set
# registry-derived (`arm-registry:` marker line + `task.py
# check-smoke-arch-registry`, driver-recompute with --repo-root).

CODE_REVIEWER_SECTION_REF = ROOT / ".claude" / "rules" / "code-reviewer-section-reference.md"
EXPERIMENT_IMPLEMENTER_SECTION_REF = (
    ROOT / ".claude" / "rules" / "experiment-implementer-section-reference.md"
)
WORKFLOW_YAML = ROOT / ".claude" / "workflow.yaml"
CODE_STYLE_RULE = ROOT / ".claude" / "rules" / "code-style.md"


def test_step6d0_arm_registry_contract_pinned():
    """T3.1 (#2176 criterion 1): the Step 6d.0 span carries the arm-registry
    enumeration check — the line key, the checker command, the
    driver-recompute flag, the derivation rule, the two-tier verdict label,
    and the REFUSE routing."""
    text = ISSUE_SKILL.read_text(encoding="utf-8")
    heading = "##### Step 6d.0:"
    assert text.count(heading) == 1, "Step 6d.0 H5 heading literal must stay unique"
    start = text.index(heading)
    end = text.find("##### Step 6d.0-bis", start + 1)
    assert end != -1, "Step 6d.0-bis heading vanished — span bound lost"
    # Whitespace-normalize so hard-wrapped SKILL.md prose cannot split a needle.
    span = " ".join(text[start:end].split())
    for needle in (
        "arm-registry:",
        "check-smoke-arch-registry",
        "--repo-root",
        "sorted(PHASES)",
        "driver-verified",
        "REFUSE to dispatch",
    ):
        assert needle in span, needle


# The six contract surfaces of the smoke-architecture arm quantifier
# (#2176 criterion 2; clarifier surface inventory + the plan-§12 A9 census).
_ARM_QUANTIFIER_SURFACES = (
    ISSUE_SKILL,
    CODE_REVIEWER,
    EXPERIMENT_IMPLEMENTER,
    WORKFLOW_YAML,
    CODE_REVIEWER_SECTION_REF,
    EXPERIMENT_IMPLEMENTER_SECTION_REF,
)

# Pinned exemptions, keyed on distinctive line TEXT — never line numbers
# (numbers drifted 1-26 lines between two independent census reads of the
# same file states): a Step 5c-bis EXAMPLE blocker string (an illustration,
# not a contract statement) and the campaign marker's `fields:` row
# (unrelated to the smoke-arch contract).
_ARM_QUANTIFIER_EXEMPT_LINE_TEXT = (
    (ISSUE_SKILL, "row missing for plan-named arm bar"),
    (WORKFLOW_YAML, "next planned arms. v1."),
)


def test_arm_registry_no_drift_across_surfaces():
    """T3.2 (#2176 criterion 2), STATEMENT grain: on each of the six contract
    surfaces, (a) the `arm-registry` token is present, and (b) EVERY line
    matching `plan(ned|-named) arm` ALSO carries the token `registry` on the
    SAME line, or sits on the pinned text-keyed exemption list. This makes a
    future PARTIAL rename (one surface updated, a sibling statement left on
    the old plan-named-only quantifier) test-breaking — presence/predicate,
    never an exact statement count, so a legitimate new statement that
    carries `registry` is not test-breaking."""
    quantifier = re.compile(r"plan(?:ned|-named) arm")
    for path in _ARM_QUANTIFIER_SURFACES:
        text = path.read_text(encoding="utf-8")
        assert "arm-registry" in text, f"{path.name}: the arm-registry token vanished (#2176)"
        for lineno, line in enumerate(text.splitlines(), 1):
            if not quantifier.search(line):
                continue
            exempt = any(
                path == ex_path and ex_text in line
                for ex_path, ex_text in _ARM_QUANTIFIER_EXEMPT_LINE_TEXT
            )
            assert "registry" in line or exempt, (
                f"{path.name}:{lineno}: a `plan(ned|-named) arm` quantifier statement "
                f"without `registry` on the same line (and not a pinned exemption) — "
                f"the #2176 union quantifier drifted on this surface: {line.strip()!r}"
            )


def test_argcheck_convention_pinned():
    """T3.3 (#2176 criterion 3): code-style.md carries the argparse-attribute
    completeness convention — the section heading, the helper symbol, and the
    whole-module-scope rationale tokens — so a future narrowing of the scope
    is caught in prose alongside the behavioral pin
    (tests/test_argcheck.py::test_whole_module_scope_catches_helper_escape)."""
    text = CODE_STYLE_RULE.read_text(encoding="utf-8")
    assert "## Argparse-attribute completeness for phase-dispatch drivers" in text
    assert "assert_args_attributes_defined" in text
    assert "whole-module" in text.lower()
    assert "one call deeper" in text
