"""Tests for the ensemble-review round-cap policy (#2391 cap 5 -> 10; #784 cap 3 -> 5).

Three coordinated invariants are locked here (post-hoc; TDD: no):

1. **Cap 5 -> 10** on the iterating review loops (#2391; raised 3 -> 5 in
   #784). Pinned via the `EnsembleReview.round_cap_per_reviewer` schema value
   AND the raw-YAML `reviewer_pairs.max_rounds` value, plus a REDESIGNED
   stale-"5" scan that catches BOTH the prose surfaces ("cap 5 per reviewer" /
   "Max 5 rounds" / "Round cap 5" / "rounds 2-5" / trigger-token `cap_5`
   fragments) AND the numeric-comparison surfaces (`revision_round >= 5`,
   `count < 5`, ...) across THREE scanned texts — the WHOLE `workflow.yaml`,
   the composed issue-skill document (SKILL.md + steps/*.md via
   `issue_skill_text()`), and `markers.md` (outside the composed document) —
   in BOTH single-line and CROSS-LINE (whitespace-normalized adjacent-pair)
   modes: markdown reflow wraps cap phrases across lines, which is exactly how
   a stale pre-#784 `3` survived two consecutive raises (#2391 U1). The
   #784-era context-window allowlist (`retired`/`RETIRED`/`RENAMED` tokens
   within +/-2 lines) is DELETED — it was demonstrated HOLLOW (it masked five
   LIVE cap surfaces that sit near history vocabulary). The ONLY exemption is
   ONE exact line-local regex matching the three #784 history clauses in
   workflow.yaml (`cap-5 + strip-then-continue-or-surface, #784` /
   `cap-5 + surface-real-residual, #784`); committed NEGATIVE CONTROLS
   (`test_scan_negative_controls`) replay the five formerly-masked live
   fragments as must-HIT (each beside history prose), the line-wrapped forms
   as must-HIT pair-mode controls, the three history clauses as must-EXEMPT,
   and a perturbed mutant as must-NOT-exempt.

2. **Git-provenance strip** — the code-review-site-only strip decision
   (`should_strip_git_provenance`) fires only when git CONFIRMS pre-existence
   and does NOT fire when git shows the round introduced the flagged lines.

3. **Cap-hit terminal** — `resolve_cap_hit` continues on all-stripped and
   blocks (autonomous) / surfaces (interactive) on a substantive residual.
   Unchanged in KIND by #2391 — only the round NUMBER moved (now round 10).

Plus a `pivot_criteria` invariant: the retired `..._cap_3` triggers AND the
#784-era `..._cap_5_surface` triggers are gone from the LIVE trigger names
(replaced by `..._cap_10_surface` surface-behavior entries, #2391).

Also pinned here (same SKILL.md Step 9c gate surface, #1022): BOTH gate pytest
invocations — the 1b touched scope AND the 1c full-scope override — capture
`PYTEST_RC=$?` on the pytest line, BEFORE the step-1d compare that consumes
`--pytest-rc "$PYTEST_RC"` (round-2 Codex Critical: the full-scope block ran
pytest without assigning the rc, so compare gated an unset/stale rc).
"""

from __future__ import annotations

import re
from pathlib import Path

from explore_persona_space.orchestrate.ensemble_strip import (
    resolve_cap_hit,
    should_strip_git_provenance,
)
from explore_persona_space.workflow import load_workflow_yaml
from tests.issue_skill_source import issue_skill_text

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".claude" / "workflow.yaml"
SKILL_PATH = REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md"
MARKERS_PATH = REPO_ROOT / ".claude" / "skills" / "issue" / "markers.md"

# --------------------------------------------------------------------------- #
# Widened stale-"5" scan (#2391). In-scope cap-5 patterns (prose + numeric),
# applied per UNIT — a single line, or a whitespace-normalized adjacent-line
# pair (cross-line mode, U1). No context-window allowlist: the #784-era
# windowed `retired`/`RETIRED`/`RENAMED` token allowlist was demonstrated
# HOLLOW (masked five live cap surfaces) and is replaced by ONE exact
# line-local exemption for the three #784 history clauses (below).
# --------------------------------------------------------------------------- #
_PROSE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"cap 5 per reviewer", re.IGNORECASE),
    re.compile(r"round cap 5", re.IGNORECASE),
    re.compile(r"max 5 rounds", re.IGNORECASE),
    re.compile(r"up to 5 rounds", re.IGNORECASE),
    re.compile(r"5 rounds per reviewer", re.IGNORECASE),
    re.compile(r"cap \(5\)", re.IGNORECASE),
    re.compile(r"\bcap-5\b", re.IGNORECASE),
    # DO-NOT-TIDY (#2391 R8): `cap_5` stays BARE. A "tidied" `cap_5\b` never
    # matches `cap_5_surface` — underscore is a word character, so `\b` FAILS
    # between the `5` and the `_` — which would re-mask exactly the two
    # trigger-token fragments among the five must-HIT negative controls below.
    re.compile(r"cap_5", re.IGNORECASE),
    re.compile(r"at round 5 \(the", re.IGNORECASE),
    re.compile(r"after round 5 \(", re.IGNORECASE),
    re.compile(r"rounds 2-5", re.IGNORECASE),
    re.compile(r"\bin 1-5\b", re.IGNORECASE),
    re.compile(r"rounds \(1-5\)", re.IGNORECASE),
    # U2(a): the markers.md round-enumeration shape ("Rounds are `1` through
    # `5` (the per-reviewer round cap)") matched NONE of the earlier patterns.
    re.compile(r"rounds are `1` through `5`", re.IGNORECASE),
    # U1: the ONLY 3-keyed pattern, deliberately: the phrase `loops up to`
    # occurs exactly once across the scanned texts (steps/09-step-5.md's
    # `revision_round` bullet), so it cannot collide with any 3-valued
    # out-of-scope look-alike, and it pins the U1 site against regression to
    # either stale value (a general stale-3 re-hunt would reintroduce the
    # whole 3-era look-alike allowlist for no added coverage).
    re.compile(r"loops up to\s+`?[35]`?", re.IGNORECASE),
)
_NUMERIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"revision_round\s*[<>]=?\s*5\b"),
    re.compile(r"\bround\s*[<>]=?\s*5\b"),
    re.compile(r"\bcount\s*[<>]=?\s*5\b"),
)
_ALL_PATTERNS: tuple[re.Pattern[str], ...] = _PROSE_PATTERNS + _NUMERIC_PATTERNS

# The ONLY exemption (#2391 U2b): exact, line-local, keyed on the three #784
# history clauses in workflow.yaml (pivot_criteria pivot_action openings).
# The G2 history appends preserve these fragments VERBATIM (the #2391 append
# lands after `#784`), so the exemption survives; no LIVE cap surface carries
# either fragment, so it can never mask one. A NEW exemption may only ever be
# added as another EXACT line-local fragment with a comment naming what it
# protects — never a broad context token (the U2 lesson).
_EXEMPT_LINE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"cap-5 \+ (strip-then-continue-or-surface|surface-real-residual), #784"),
)


def _norm(s: str) -> str:
    """Whitespace-normalize a scan unit (collapse runs, strip ends)."""
    return re.sub(r"\s+", " ", s).strip()


def _pattern_hits(text: str, patterns: tuple[re.Pattern[str], ...] = _ALL_PATTERNS) -> set[str]:
    """Return the set of pattern strings that match ``text``."""
    return {p.pattern for p in patterns if p.search(text)}


def _is_exempt(
    unit_text: str, exemptions: tuple[re.Pattern[str], ...] = _EXEMPT_LINE_PATTERNS
) -> bool:
    """A matched unit is exempt iff its OWN text matches an exact exemption."""
    return any(p.search(unit_text) for p in exemptions)


def _stale_cap5_hits(
    text: str,
    *,
    apply_exemption: bool = True,
    patterns: tuple[re.Pattern[str], ...] = _ALL_PATTERNS,
    exemptions: tuple[re.Pattern[str], ...] = _EXEMPT_LINE_PATTERNS,
) -> list[str]:
    """Return in-scope units that still carry a cap-5 prose/numeric surface.

    Two modes over the same text:

    - SINGLE-LINE: every line is a unit (whitespace-normalized).
    - CROSS-LINE (U1): every adjacent line pair, whitespace-normalized and
      joined, is a unit; a pair is reported only for PAIR-NOVEL matches —
      patterns matching the join that match NEITHER constituent line alone —
      so single-line hits are never double-reported.

    A matched unit is dropped iff ``apply_exemption`` and the unit's own text
    matches an ``exemptions`` entry (exact line-local exemption; there is
    deliberately NO context window). Returns offending units with 1-based
    line numbers so a failure message is actionable.

    ``patterns`` / ``exemptions`` default to the digit-keyed cap-5 apparatus;
    the spelled-out-numeral scan (#2391 r2, B1) reuses the same line + pair
    machinery with its own pattern/exemption sets.
    """
    lines = text.splitlines()
    normed = [_norm(line) for line in lines]
    single_hits = [_pattern_hits(n, patterns) for n in normed]
    hits: list[str] = []
    for idx, line in enumerate(lines):
        if not single_hits[idx]:
            continue
        if apply_exemption and _is_exempt(normed[idx], exemptions):
            continue
        hits.append(f"{idx + 1}: {line.strip()}")
    for idx in range(len(lines) - 1):
        joined = normed[idx] + " " + normed[idx + 1]
        novel = _pattern_hits(joined, patterns) - single_hits[idx] - single_hits[idx + 1]
        if not novel:
            continue
        if apply_exemption and _is_exempt(joined, exemptions):
            continue
        hits.append(f"{idx + 1}-{idx + 2} (pair): {joined[:160]}")
    return hits


# --------------------------------------------------------------------------- #
# (a) Cap 5 -> 10
# --------------------------------------------------------------------------- #
def test_round_cap_is_ten():
    """The EnsembleReview schema value is 10 (#2391; raised 3 -> 5 in #784)."""
    workflow = load_workflow_yaml()
    assert workflow.ensemble_review.round_cap_per_reviewer == 10


def test_reviewer_pairs_max_rounds_is_ten():
    """The higher-level review-loop-driver config flips in lockstep (raw YAML)."""
    import yaml

    raw = yaml.safe_load(WORKFLOW_PATH.read_text())
    assert raw["reviewer_pairs"]["max_rounds"] == 10


def test_no_stale_cap_5_in_ensemble_prose():
    """No in-scope cap-5 prose OR numeric surface survives in any of the three
    scanned texts (workflow.yaml, the composed issue-skill document, and
    markers.md), in single-line OR cross-line mode; the only exempt units are
    the three #784 history clauses (exact line-local exemption, §7d)."""
    wf_hits = _stale_cap5_hits(WORKFLOW_PATH.read_text())
    skill_hits = _stale_cap5_hits(issue_skill_text())
    markers_hits = _stale_cap5_hits(MARKERS_PATH.read_text())
    assert not wf_hits, "stale in-scope cap-5 surface in workflow.yaml:\n" + "\n".join(wf_hits)
    assert not skill_hits, "stale in-scope cap-5 surface in SKILL.md:\n" + "\n".join(skill_hits)
    assert not markers_hits, "stale in-scope cap-5 surface in markers.md:\n" + "\n".join(
        markers_hits
    )


def test_exemption_scope_is_exactly_the_three_history_clauses():
    """The exemption-disabled scan of workflow.yaml returns EXACTLY the three
    #784 history-clause lines, each matching the exact exemption regex — so
    (a) the G2 fragment-survival constraint holds (a reworded history clause
    stops matching and turns into a loud stale-cap hit above), and (b) the
    exemption is provably alive and scoped (nothing else needs exempting). A
    future legitimate 4th exemptable unit must be added per the §7e protocol
    (exact line-local fragment + comment), never a broad context token."""
    raw_hits = _stale_cap5_hits(WORKFLOW_PATH.read_text(), apply_exemption=False)
    assert len(raw_hits) == 3, "expected exactly the 3 history clauses:\n" + "\n".join(raw_hits)
    for hit in raw_hits:
        assert _is_exempt(_norm(hit)), f"non-exempt unit in exemption-disabled residual: {hit}"
    skill_raw = _stale_cap5_hits(issue_skill_text(), apply_exemption=False)
    markers_raw = _stale_cap5_hits(MARKERS_PATH.read_text(), apply_exemption=False)
    assert not skill_raw, "unexpected exemption-disabled hits in SKILL.md:\n" + "\n".join(skill_raw)
    assert not markers_raw, "unexpected exemption-disabled hits in markers.md:\n" + "\n".join(
        markers_raw
    )


# --------------------------------------------------------------------------- #
# (B1, #2391 r2) SPELLED-OUT cap numerals. Every pattern above is DIGIT-keyed,
# so a spelled-out numeral is invisible to the whole apparatus — the round-2
# blocker: `clean-result-critic.md` read "Five rounds maximum" in the same
# paragraph as a correct "round 10" and no scan, sweep, or test saw it. Hunt
# the spelled cap forms over the FULL workflow doc surface (workflow.yaml,
# CLAUDE.md, every rules file, every agent spec, every skill doc — the §4
# in-scope file-set superset; `.claude/agent-memory/**` is deliberately out of
# scope: memory notes narrate past rounds under the then-cap).
# --------------------------------------------------------------------------- #
_SPELLED_CAP_PATTERNS: tuple[re.Pattern[str], ...] = (
    # r3 (#2391): the bare `\bfive rounds\b` form is widened to a bounded
    # family covering the hyphenated compound ("Five-round maximum" — the
    # natural rewrite of the exact B1 offender string, invisible to
    # `\bfive rounds\b`) and the qualified forms ("five review rounds",
    # "five code-review rounds", "five ensemble rounds" — all live on the
    # doc surface as #906/#823 incident narrations, each carried by an exact
    # line-local exemption below).
    re.compile(
        r"\bfive[-\s]+(?:(?:code-review|review|ensemble)[-\s]+)?rounds?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bround five\b", re.IGNORECASE),
    re.compile(r"\bfifth round\b", re.IGNORECASE),
    re.compile(r"\bfive revision\b", re.IGNORECASE),
    re.compile(r"\bfive total\b", re.IGNORECASE),
)
# Exact line-local exemptions (the §7e protocol — never a broad context
# token), each naming what it protects: incident narrations of runs that
# genuinely blocked/PASSed under the then-cap (history preservation, §4).
_SPELLED_EXEMPT_PATTERNS: tuple[re.Pattern[str], ...] = (
    # code-correctness-critic.md — the #906 incident narration.
    re.compile(r"#906: five rounds PASSed"),
    # codex-code-reviewer.md — the #823 incident narration.
    re.compile(r"#823: five rounds PASSed"),
    # experiment-implementer.md + implementer.md — the #906 producer-side
    # narrations ("five review rounds shipped"; r3 family widening).
    re.compile(r"#906: five review rounds shipped"),
    # code-reviewer.md Step 3.8 + code-style.md § seam-stubbed — the #906
    # review-side narrations ("five ensemble rounds PASSed/shipped"; r3).
    re.compile(r"#906: five ensemble rounds"),
    # code-style.md checkpoint-per-phase bullet — the #823 narration tail
    # ("five code-review rounds never flagged it"; r3).
    re.compile(r"five code-review rounds never flagged it"),
    # code-reviewer.md Step 3.6 — the #823 narration wrapped across lines
    # 887-888 ("... Five / code-review rounds PASSed it ..."), caught only
    # in PAIR mode; the exemption matches the joined pair unit (r3).
    re.compile(r"Five code-review rounds PASSed it"),
)


def _workflow_doc_files() -> list[Path]:
    """The workflow doc surface the spelled-cap scan covers (see block comment)."""
    files = [WORKFLOW_PATH, REPO_ROOT / "CLAUDE.md"]
    files += sorted((REPO_ROOT / ".claude" / "rules").glob("*.md"))
    files += sorted((REPO_ROOT / ".claude" / "agents").glob("*.md"))
    files += sorted((REPO_ROOT / ".claude" / "skills").rglob("*.md"))
    return files


def test_no_spelled_out_cap_in_workflow_docs():
    """No LIVE spelled-out cap numeral survives anywhere on the workflow doc
    surface; the only exempt units are the two incident narrations (exact
    line-local exemptions above). Same line + pair machinery as the digit
    scan, so a line-wrapped spelled form is caught too."""
    offenders: list[str] = []
    for path in _workflow_doc_files():
        hits = _stale_cap5_hits(
            path.read_text(encoding="utf-8"),
            patterns=_SPELLED_CAP_PATTERNS,
            exemptions=_SPELLED_EXEMPT_PATTERNS,
        )
        offenders += [f"{path.relative_to(REPO_ROOT)}:{h}" for h in hits]
    assert not offenders, "spelled-out stale cap numeral in workflow docs:\n" + "\n".join(offenders)


def test_spelled_cap_scan_negative_controls():
    """BOTH directions of the spelled-cap scan (#2391 r2): it must SEE the
    verbatim round-2 offender line, must EXEMPT the two incident narrations
    (verbatim, pre-existing forms), and must NOT exempt a perturbed mutant
    (issue id removed from the exemption-keyed fragment)."""
    offender = "Five rounds maximum per `/issue` invocation. Every round is ensembled\n"
    assert _stale_cap5_hits(
        offender, patterns=_SPELLED_CAP_PATTERNS, exemptions=_SPELLED_EXEMPT_PATTERNS
    ), "must-HIT spelled control produced no hit (the B1 offender line)"
    # r3 must-HIT variants: the hyphenated compound (the natural rewrite of
    # the exact B1 offender string) and the review-word form — each was
    # pattern-INVISIBLE to the r2 bank (observed red before the r3 family
    # widening landed).
    for name, fixture in (
        ("hyphenated compound", "Five-round maximum per `/issue` invocation.\n"),
        ("review-word form", "Five review rounds maximum per `/issue` invocation.\n"),
    ):
        assert _stale_cap5_hits(
            fixture, patterns=_SPELLED_CAP_PATTERNS, exemptions=_SPELLED_EXEMPT_PATTERNS
        ), f"must-HIT spelled r3 control produced no hit: {name}"
    exempt_controls = (
        (
            "code-correctness-critic.md #906 narration",
            "  nonexistent-field → Critical, `substantive` (#906: five rounds PASSed\n",
        ),
        (
            "codex-code-reviewer.md #823 narration",
            "  carve-outs (#823: five rounds PASSed a ~20h accumulate-and-write-at-end\n",
        ),
        (
            "experiment-implementer.md / implementer.md #906 review-word narration (r3)",
            "   the test it will demand (incident #906: five review rounds shipped\n",
        ),
        (
            "code-reviewer.md / code-style.md #906 ensemble-word narration (r3)",
            "    (never stripped by Step 5c-bis). (Incident #906: five ensemble rounds\n",
        ),
        (
            "code-style.md #823 code-review-word narration tail (r3)",
            "  whole loop; five code-review rounds never flagged it). Review-side gate:\n",
        ),
        (
            "code-reviewer.md #823 wrapped narration, PAIR mode (r3)",
            "  single terminal write (lines 1704\u20131706). Five\n"
            "  code-review rounds PASSed it; both GCE crashes forfeited\n",
        ),
    )
    for name, fixture in exempt_controls:
        assert not _stale_cap5_hits(
            fixture, patterns=_SPELLED_CAP_PATTERNS, exemptions=_SPELLED_EXEMPT_PATTERNS
        ), f"must-EXEMPT spelled control produced a hit: {name}"
        # The exemption (not pattern-miss) must be what silences it:
        assert _stale_cap5_hits(
            fixture,
            apply_exemption=False,
            patterns=_SPELLED_CAP_PATTERNS,
            exemptions=_SPELLED_EXEMPT_PATTERNS,
        ), f"must-EXEMPT spelled control matched no pattern at all (dead control): {name}"
    mutant = "  nonexistent-field → Critical, `substantive` (five rounds PASSed\n"
    assert _stale_cap5_hits(
        mutant, patterns=_SPELLED_CAP_PATTERNS, exemptions=_SPELLED_EXEMPT_PATTERNS
    ), "spelled mutant control was wrongly exempted"
    # r3 mutants — one per r3 exemption, each perturbing the exemption-keyed
    # fragment (issue id removed / tail reworded) while keeping the pattern
    # match, so a broadened exemption regex is caught.
    r3_mutants = (
        ("review-word, issue id removed", "  demand (incident: five review rounds shipped\n"),
        ("ensemble-word, issue id removed", "  5c-bis). (Incident: five ensemble rounds\n"),
        (
            "code-review-word, tail reworded",
            "  whole loop; five code-review rounds never caught it).\n",
        ),
        (
            "pair-wrapped code-review-word, tail reworded",
            "  single terminal write. Five\n  code-review rounds PASSed the diff; both\n",
        ),
    )
    for name, fixture in r3_mutants:
        assert _stale_cap5_hits(
            fixture, patterns=_SPELLED_CAP_PATTERNS, exemptions=_SPELLED_EXEMPT_PATTERNS
        ), f"r3 spelled mutant control was wrongly exempted: {name}"


# --------------------------------------------------------------------------- #
# (d2) NEGATIVE CONTROLS (#2391 U2b's teeth): the redesigned scan must SEE the
# five formerly-masked live fragments (their PRE-edit forms, each embedded
# beside `retired` / `RENAMED` history prose — proving history vocabulary no
# longer masks), must SEE the line-wrapped forms in pair mode, must EXEMPT the
# three #784 history clauses (post-edit forms), and must NOT exempt a
# perturbed mutant. Weakening any control is a plan §14 must-ask.
# --------------------------------------------------------------------------- #
_MUST_HIT_CONTROLS: tuple[tuple[str, str], ...] = (
    (
        "trigger-ref-beside-retired-prose (workflow.yaml:985 pre-edit form)",
        "No more same-diff-family strategy-pivot re-plan (retired, #784; see\n"
        "§ pivot_criteria.code_review_ensemble_cap_5_surface and SKILL.md\n",
    ),
    (
        "cap-(5)-beside-reversed-policy-prose (workflow.yaml:1151 pre-edit form)",
        "2026-05-13 — absorbed the retired reviewer step's statistical-framing\n"
        "rule). Ensembled on ALL rounds up to the per-reviewer cap (5) (round-1-only "
        "policy reversed\n",
    ),
    (
        "revision_round>=5-beside-retired-prose (steps/09-step-5.md:1333 pre-edit form)",
        "- **`final_verdict == FAIL` + revision_round>=5** -> **CAP-HIT:\n"
        "  strip-then-continue-or-surface** (replaces the retired cap-3 strategy\n",
    ),
    (
        "trigger-ref-plus-at-round-5 (steps/09-step-5.md:1336 pre-edit form)",
        "  pivot; see CLAUDE.md and workflow.yaml (RENAMED at #784)\n"
        "  § pivot_criteria.code_review_ensemble_cap_5_surface). At round 5 (the\n",
    ),
    (
        "markers.md:158 pre-edit row (retired #784 + no-longer-strategy-pivots on the "
        "SAME line as the token)",
        "| `epm:strategy-pivot` | skill (orchestrator) | any strategy pivot — "
        "whack-a-mole detector fire, plan-contradiction re-plan, debugging-wall pivot "
        "(see § pivot_criteria). NOTE: the Step 5 code-review ensemble cap-hit no "
        "longer strategy-pivots (retired #784 → strip-then-continue-or-surface, "
        "§ pivot_criteria.code_review_ensemble_cap_5_surface) | v<n> per pivot |\n",
    ),
)

_MUST_HIT_CROSS_LINE_CONTROLS: tuple[tuple[str, str], ...] = (
    (
        "wrapped `loops up to` / `5` (the U1 shape at the raised value)",
        "- `revision_round` — 1-indexed integer. `1` on first review; loops up to\n"
        "  `5`. The cap is **per reviewer** — reconcile invocations are free.\n",
    ),
    (
        "wrapped `loops up to` / `3` (the U1 shape at the original stale value)",
        "- `revision_round` — 1-indexed integer. `1` on first review; loops up to\n"
        "  `3`. The cap is **per reviewer** — reconcile invocations are free.\n",
    ),
    (
        "wrapped `round cap` / `5**` (the issue-v2 SKILL.md:310-311 pre-edit shape)",
        "Ensemble decision, reconciler on disagreement, mechanical-strip, and **round cap\n"
        "5** are IDENTICAL to v1 (see SKILL.md).\n",
    ),
    (
        "markers.md:228 pre-edit round-enumeration line (the U2a pattern)",
        "`clean_result`. Rounds are `1` through `5` (the per-reviewer round cap). "
        "Allowed verdicts are\n",
    ),
)

# The three #784 history clauses in their POST-edit forms (with the #2391
# appends) — the exemption must swallow them, and ONLY them: the mutant
# control perturbs the exemption-keyed fragment (`#784` removed) and must HIT.
_MUST_EXEMPT_CONTROLS: tuple[tuple[str, str], ...] = (
    (
        "workflow.yaml:691 post-edit (code-review clause)",
        "      to cap-5 + strip-then-continue-or-surface, #784; cap raised to 10 +\n",
    ),
    (
        "workflow.yaml:723 post-edit (interpretation clause)",
        "      cap-5 + surface-real-residual, #784; cap raised to 10, #2391).\n",
    ),
    (
        "workflow.yaml:735 post-edit (clean-result clause)",
        "      cap-5 + surface-real-residual, #784; cap raised to 10, #2391).\n",
    ),
)

_MUTANT_NON_EXEMPT_CONTROL = (
    "exemption fragment perturbed (`#784` removed) — must NOT be exempt",
    "      to cap-5 + strip-then-continue-or-surface; cap raised to 10 +\n",
)


def test_scan_negative_controls():
    """BOTH directions of the redesigned exemption discipline (#2391 §7d2)."""
    for name, fixture in _MUST_HIT_CONTROLS:
        assert _stale_cap5_hits(fixture), f"must-HIT control produced no hit: {name}"
    for name, fixture in _MUST_HIT_CROSS_LINE_CONTROLS:
        assert _stale_cap5_hits(fixture), f"must-HIT cross-line control produced no hit: {name}"
    for name, fixture in _MUST_EXEMPT_CONTROLS:
        assert not _stale_cap5_hits(fixture), f"must-EXEMPT control produced a hit: {name}"
        # The exemption (not pattern-miss) must be what silences it:
        assert _stale_cap5_hits(fixture, apply_exemption=False), (
            f"must-EXEMPT control matched no pattern at all (dead control): {name}"
        )
    name, fixture = _MUTANT_NON_EXEMPT_CONTROL
    assert _stale_cap5_hits(fixture), f"mutant control was wrongly exempted: {name}"


def test_code_review_flow_diagram_and_exit_kind_updated():
    """The SKILL.md code-review flow diagram uses count<10 and the >=10 branch
    routes to the cap-hit rule (not a bare `blocked`), and the Step 5b exit-kind
    table row uses revision_round>=10 with a conditional exit-kind."""
    skill = issue_skill_text()
    assert "FAIL + count<10 --> running" in skill
    # The >=10 diagram branch must reference the cap-hit rule, not a bare blocked terminal.
    assert re.search(r"FAIL \+ count>=10 --> .*Step 5d cap-hit rule", skill)
    # Exit-kind table row for Step 5b.
    assert "Step 5b code-review FAIL revision_round>=10" in skill
    assert "apply Step 5d cap-hit rule" in skill


# --------------------------------------------------------------------------- #
# #1022 Step 9c shell dataflow: PYTEST_RC captured before compare consumes it
# --------------------------------------------------------------------------- #
def test_step9c_pytest_rc_captured_before_compare():
    """Both Step 9c gate pytest blocks (1b touched scope + 1c full-scope
    override) write the pytest rc to `/tmp/step9c-rc-issue-<N>` on the SAME
    inner bash -c command tail (`pytest ...; echo $? > rc-file`) — the
    § Harvest self-harvest chaining shape (#2005 detached-launcher form,
    task #2005): the inner bash -c binds the pytest command + the rc-write
    into ONE session-decoupled unit. The completion read
    (`PYTEST_RC=$(cat ...)`) precedes the step-1d `--pytest-rc "$PYTEST_RC"`
    compare consumer (the #1022 dataflow invariant), and the
    anti-silent-pass guards are present (#1046 AC7): the three-file `rm -f`
    preamble before BOTH invocations, the missing-rc FAIL guard, and the
    zero-collected FAIL guard. The bounded spans are FENCE-SAFE — they
    exclude backticks, so a match can never cross a code-fence boundary
    into a neighboring block: the rc write must sit on the same inner
    bash -c command tail as its pytest invocation."""
    skill = issue_skill_text()
    sec = skill[skill.index("9c. Test-verdict gate") : skill.index("### Step 10: Auto-complete")]
    # The detached shape carries `timeout --kill-after=60s <T>s \` + a line
    # break + `env <thread-caps> \` + a line break + `uv run pytest <files>`
    # + args + `; echo \$? > /tmp/step9c-rc-issue-<N>` — all within ONE
    # inner bash -c string. The regex accepts the optional env prefix and
    # allows up to 600 non-backtick chars between the timeout and the
    # rc-write (measured span ~450 chars) — fence-safe by construction.
    touched = re.search(
        r"timeout --kill-after=60s <T>s[^\x60]{0,600}?uv run pytest <files>[^\x60]{0,600}?"
        r"echo \\?\$\? > /tmp/step9c-rc-issue-<N>",
        sec,
    )
    assert touched, "1b block must write the pytest rc to the rc file on its inner-bash-c tail"
    full = re.search(
        r"timeout --kill-after=60s 60m[^\x60]{0,600}?uv run pytest tests/[^\x60]{0,600}?"
        r"echo \\?\$\? > /tmp/step9c-rc-issue-<N>",
        sec,
    )
    assert full, "1c full-scope block must write the pytest rc to the rc file"
    read_idx = sec.index("PYTEST_RC=$(cat /tmp/step9c-rc-issue-<N>)")
    # Anchor the compare-consumer index PAST any prose mention of the flag —
    # use the LAST occurrence (the 1d compare snippet), not the first (a
    # prose mention would first-match and weaken the ordering pin):
    compare_idx = sec.rindex('--pytest-rc "$PYTEST_RC"')
    assert touched.end() < compare_idx, "touched-scope rc write must precede the compare consumer"
    assert full.end() < compare_idx, "full-scope rc write must precede the compare consumer"
    assert read_idx < compare_idx, "the rc-file read must precede the compare consumer"
    # (a) AC7: the three-file rm -f preamble precedes BOTH gate invocations.
    rm_pat = (
        r"rm -f /tmp/step9c-junit-issue-<N>\.xml /tmp/step9c-rc-issue-<N> \\\n"
        r"\s*/tmp/step9c-pytest-issue-<N>\.log"
    )
    rm_hits = [m.start() for m in re.finditer(rm_pat, sec)]
    assert len(rm_hits) >= 2, "both 1b and 1c blocks must carry the three-file rm -f preamble"
    assert rm_hits[0] < touched.start() and rm_hits[1] < full.start(), (
        "each rm -f preamble must precede its gate invocation"
    )
    # (b) AC7: the completion read carries the missing-rc FAIL guard + the
    # zero-collected FAIL guard.
    assert "[ ! -f /tmp/step9c-rc-issue-<N> ]" in sec, "missing-rc FAIL guard must be pinned"
    assert "no tests ran|collected 0 items" in sec, "zero-collected FAIL guard must be pinned"


# --------------------------------------------------------------------------- #
# pivot_criteria: the retired cap-3 AND cap-5 trigger names are gone
# --------------------------------------------------------------------------- #
def test_pivot_criteria_cap_trigger_names_current():
    """The live pivot trigger names state the CURRENT cap (#2391 rename,
    following the #784 precedent): the `..._cap_3` names AND the #784-era
    `..._cap_5_surface` names are gone; the `..._cap_10_surface` names are
    present. History narration INSIDE the entries keeps the old names."""
    import yaml

    raw = yaml.safe_load(WORKFLOW_PATH.read_text())
    triggers = {entry["trigger"] for entry in raw["pivot_criteria"]}
    assert "code_review_ensemble_cap_3" not in triggers
    assert "interpretation_critic_cap_3" not in triggers
    assert "clean_result_critic_cap_3" not in triggers
    assert "code_review_ensemble_cap_5_surface" not in triggers
    assert "interpretation_critic_cap_5_surface" not in triggers
    assert "clean_result_critic_cap_5_surface" not in triggers
    assert "code_review_ensemble_cap_10_surface" in triggers
    assert "interpretation_critic_cap_10_surface" in triggers
    assert "clean_result_critic_cap_10_surface" in triggers


# --------------------------------------------------------------------------- #
# (b) git-provenance strip decision
# --------------------------------------------------------------------------- #
def test_git_provenance_strip_fires_when_preexisting():
    """git confirms pre-existence AND the round did not touch the flagged lines
    -> strip fires (True)."""
    assert (
        should_strip_git_provenance(
            "pre-existing-on-trunk",
            git_says_pre_existing=True,
            git_says_round_touched_flagged_lines=False,
        )
        is True
    )
    # Same for the other two subclasses.
    assert should_strip_git_provenance(
        "stale-main-or-worktree",
        git_says_pre_existing=True,
        git_says_round_touched_flagged_lines=False,
    )
    assert should_strip_git_provenance(
        "cumulative-main-head-diff",
        git_says_pre_existing=True,
        git_says_round_touched_flagged_lines=False,
    )


def test_git_provenance_strip_does_not_fire_when_round_introduced():
    """git shows the round's own range touched the flagged lines -> strip does
    NOT fire (False), even if the pre-existing flag is also (contradictorily)
    set. Ambiguous / malformed evidence also -> False."""
    # Round introduced it: strip must not fire.
    assert (
        should_strip_git_provenance(
            "pre-existing-on-trunk",
            git_says_pre_existing=False,
            git_says_round_touched_flagged_lines=True,
        )
        is False
    )
    # Contradictory (both True): default to not-strip (FAIL stands).
    assert (
        should_strip_git_provenance(
            "pre-existing-on-trunk",
            git_says_pre_existing=True,
            git_says_round_touched_flagged_lines=True,
        )
        is False
    )
    # Ambiguous (both False): not-strip.
    assert (
        should_strip_git_provenance(
            "pre-existing-on-trunk",
            git_says_pre_existing=False,
            git_says_round_touched_flagged_lines=False,
        )
        is False
    )
    # Malformed / unknown subclass: never stripped.
    assert (
        should_strip_git_provenance(
            "not-a-real-subclass",
            git_says_pre_existing=True,
            git_says_round_touched_flagged_lines=False,
        )
        is False
    )


# --------------------------------------------------------------------------- #
# (c) cap-hit terminal decision
# --------------------------------------------------------------------------- #
def test_cap_hit_all_stripped_passes():
    """At the cap, all residual blockers stripped -> continue (treat as PASS)."""
    for autonomous in (True, False):
        decision = resolve_cap_hit(all_residual_stripped=True, autonomous=autonomous)
        assert decision["action"] == "continue"


def test_cap_hit_substantive_residual_blocks_autonomous():
    """A substantive residual at the cap: autonomous -> block (epm:failure +
    status:blocked); interactive -> surface to the user. Never ships past."""
    autonomous_decision = resolve_cap_hit(all_residual_stripped=False, autonomous=True)
    assert autonomous_decision["action"] == "block_autonomous"

    interactive_decision = resolve_cap_hit(all_residual_stripped=False, autonomous=False)
    assert interactive_decision["action"] == "surface_interactive"


# --------------------------------------------------------------------------- #
# (d) adopt-more-severe-without-reconciler ban (#1134)
# --------------------------------------------------------------------------- #
def test_adopt_severe_reconciler_ban_pinned():
    """#1134: the adopt-more-severe-without-reconciler ban survives SKILL.md churn.

    Pins both placements: the Step 5c canonical ban paragraph and the
    Step 9a incident-site pointer. Dropping either silently reverts #1134.
    """
    text = issue_skill_text()
    assert text.count("UNSANCTIONED at every doubled site") == 1
    assert text.count("#825 skipped the reconciler") == 1
    assert text.count("the #825 deviation site") == 1
