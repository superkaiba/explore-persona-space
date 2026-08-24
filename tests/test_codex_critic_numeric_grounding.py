"""Tests for the codex-critic Step-4 numeric-leak verifier contract (task #758).

`.claude/agents/codex-critic.md` Step 4 specifies (in prose) a local
prompt-file validation that FAILs LOUD when the composed Codex prompt contains
a numeric atom not traceable to the handed `plan_body` / `lens_items` /
`prior_critique_summaries` spans or the static template-scaffold allowlist —
the fix for the #722 "composer inlined `+0.74-0.80` / `MLP -2.17/-6.12`
numbers absent from the plan" bug.

The verifier's exact regex is the composer's to finalize at compose time, so
these tests pin the *contract* the prose mandates, not a specific
implementation: (1) the tokenizer normalization must split hyphenated ranges
(`a-b`) and slash-joined pairs (`a/b`) into atomic numbers BEFORE the multiset
diff, so the canonical #722 forms atomize to `{0.74, 0.80, -2.17, -6.12}`;
(2) a prompt carrying those literal forms against a `plan_body` that lacks the
atoms MUST trip BLOCKER; (3) the inverse (atoms present in `plan_body`) MUST
pass; (4) an empty `prior_critique_summaries` is fail-safe (zero allowlist
contribution, never a crash). A second test class asserts the six spec edits
(A-F) are present in the agent file so a future spec edit cannot silently
strip the contract.

The reference normalizer + multiset-diff below is one valid realization of the
prose contract; it doubles as executable documentation of the intended
behavior for whoever finalizes the inline verifier.

Task-reference carve-out (#1025, the #795/#720 incident): the Step-4 prose now
prescribes a task-reference extraction pass that runs BEFORE numeric
tokenization, SYMMETRICALLY on the prompt and all handed spans. Prompt-side
ids (`#<N>`, `tasks/<status>/<N>`, `issue[-_]<N>`) clear against handed-span
ids or the `tasks` map of `tasks/REGISTRY.json`; an unreadable registry
degrades to the handed-span leg alone (fail-strict). The reference
realization threads a hermetic `registry_ids` fixture parameter through
`_residual_blocker_numbers` — these tests NEVER read the live REGISTRY.json.
"""

# The agent-file assertions below quote the literal markdown (em/en dashes,
# the `※`-adjacent prose, the canonical #722 numeric forms). Substituting the
# unicode would defeat the presence checks.

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_FILE = REPO_ROOT / ".claude" / "agents" / "codex-critic.md"

# Static template-scaffold allowlist enumerated against the FINAL Step-3
# template text (round 2, #758). The atoms are the numeric literals that
# survive `{{...}}` placeholder substitution at compose time, plus the
# structural digits the substitution itself introduces that trace to no
# handed span. Enumerated by `_static_template_atoms()` below + audited by
# `TestAllowlistCoversFinalTemplate`:
#   - `0`     — "Phase 0 smoke tests" (anti-pattern bullet).
#   - `1`/`2` — the "1-2 line check sketch" / "1-2 lines" GROUNDING-rule prose
#               and the "1. [Issue]" Must-Fix list marker.
#   - `500`   — the "add a 500-example generic-assistant SFT baseline" worked
#               example in the closing "Be specific" instruction (the atom
#               Codex named in round 1).
#   - `3`-`10` — the `{{revision_round}}` / marker-tag `v<n>` round digits.
#               Bounded to {1,...,10} by the /adversarial-planner max-10-rounds
#               policy (#2391 raised the cap 5 -> 10; #784 raised 3 -> 5); the
#               substituted value traces to NO handed span (it is not in
#               plan_body / lens_items / prior_critique_summaries), so EVERY
#               round digit must be scaffold-covered — a digit missing from the
#               set makes that round's compose false-BLOCKER on its own round
#               number. {1,2} are already present from above; `3`-`10` are the
#               round digits not otherwise in the template. Mirrors
#               `.claude/agents/codex-critic.md` Step 4 (the reference
#               implementation) and the three v2 sibling composer specs.
# Set-membership widening tradeoff (#2391, named — never widened silently):
# scaffold atoms clear UNCONDITIONALLY (set-membership, not multiset), so a
# composer-authored bare `6`-`10` ANYWHERE in a prompt is no longer caught.
# That hole is inherent to the existing design — it already applied to
# 0/1/2/3/500 (and to 4/5 in codex-critic.md) — and is the accepted price of
# rounds 6-10 composing at all; fabricated NON-scaffold values (the #722
# forms) still residual.
# The phantom `50` from round 1 is REMOVED — it never appeared in the template
# (the only `50` in the raw text is the leading `5` of `500`); under the
# set-membership scaffold semantics (below) it was a harmless no-op but a
# documentation error.
SCAFFOLD_ALLOWLIST = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "500"}


def _normalize_numeric_atoms(text: str) -> list[str]:
    """Reference realization of the Step-4 tokenizer contract.

    Splits hyphenated ranges (``a-b``) and slash-joined pairs (``a/b``) into
    their atomic numbers BEFORE returning, while preserving scientific
    notation (``5e-6``). Returns the multiset (list, dupes kept) of atoms as
    canonical strings.

    Strategy: protect scientific-notation mantissa/exponent hyphens by
    matching ``\\d+(?:\\.\\d+)?[eE][+-]?\\d+`` first; then scan the remaining
    text for signed decimals, treating a leading ``-``/``+`` as a SIGN only
    when it does not sit between two digits (an inter-digit hyphen/slash is a
    range/pair separator, so the right operand is unsigned). This yields
    ``+0.74-0.80`` -> {0.74, 0.80} and ``MLP -2.17/-6.12`` -> {-2.17, -6.12}.
    """
    atoms: list[str] = []
    sci = re.compile(r"\d+(?:\.\d+)?[eE][+-]?\d+")

    # 1) Pull scientific-notation atoms out first, replacing with a sentinel
    #    space so their internal exponent sign is never reinterpreted.
    def _take_sci(m: re.Match[str]) -> str:
        atoms.append(_canon(m.group(0)))
        return " "

    remainder = sci.sub(_take_sci, text)

    # 2) A signed-or-unsigned decimal. A sign char counts as a SIGN only when
    #    the char immediately before it is NOT a digit (otherwise it is a
    #    range `-` / a stray `+` joining restated numbers). Slash `/` never
    #    carries a sign onto its right operand.
    num = re.compile(r"(?<![\w.])([+-]?)(\d+(?:\.\d+)?)")
    for m in num.finditer(remainder):
        sign, mag = m.group(1), m.group(2)
        # Reattach the sign only if the char preceding the sign (or magnitude)
        # is not a digit — guards the right side of `a-b` / `a/b`.
        start = m.start()
        prev = remainder[start - 1] if start > 0 else ""
        if sign and prev.isdigit():
            sign = ""  # inter-digit hyphen/plus: separator, not a sign
        atoms.append(_canon(f"{sign}{mag}"))
    return atoms


def _canon(tok: str) -> str:
    """Canonicalize a numeric token to a stable string key.

    `+0.74` -> `0.74`, `-2.17` -> `-2.17`, `5e-6` -> `5e-06` via float round-trip
    where it does not lose information; falls back to the stripped literal.
    """
    t = tok.lstrip("+")
    try:
        f = float(t)
    except ValueError:
        return t
    # Preserve `-0.0` etc. as `0.0`; use repr of float for a stable key.
    if f == int(f) and "e" not in t.lower() and "." not in t:
        return str(int(f))
    return repr(f)


# The three whitelisted task-reference forms (#1025). The full lookahead
# `(?!\d*\.\d)` is load-bearing: it makes a decimal-bearing token (`#720.5`,
# `#0.74`) match NOTHING at all — the v1 shorthand `(?!\.\d)` backtracked
# `#720.5` to a truncated id `72`. `issue[-_]` covers both `issue-720`
# branches and `issue_720` result paths.
TASK_REF_PATTERNS = (
    re.compile(r"#(\d+)(?!\d*\.\d)"),
    re.compile(r"tasks/[a-z_]+/(\d+)\b"),
    re.compile(r"issue[-_](\d+)\b"),
)


def _extract_task_refs(text: str) -> tuple[str, set[str]]:
    """Reference realization of the Step-4 task-reference extraction (#1025).

    Matches the three whitelisted reference forms, REMOVES each match from the
    working text (so a span-side `#720` never donates a bare `720` atom to the
    numeric multiset — the symmetric-removal property), and returns
    ``(cleaned_text, {ids})``. A decimal-bearing token extracts nothing; its
    atoms stay in the numeric accounting.
    """
    ids: set[str] = set()

    def _take(m: re.Match[str]) -> str:
        ids.add(m.group(1))
        return " "

    for pat in TASK_REF_PATTERNS:
        text = pat.sub(_take, text)
    return text, ids


def _static_template_atoms() -> set[str]:
    """Extract the canonical numeric atoms of the FINAL Step-3 prompt template
    that SURVIVE `{{...}}` placeholder substitution.

    The template is the first bare ```-fenced block under the "### Step 3:"
    heading of `.claude/agents/codex-critic.md`. The `{{plan_body}}` /
    `{{lens_items}}` / `{{prior_critique_summaries}}` / `{{revision_round}}` /
    `{{lens}}` / `{{LENS}}` placeholders are SUBSTITUTION SITES — their numeric
    content is contributed by the orchestrator at compose time, NOT static
    scaffold — so they are stripped before tokenizing. Returns the set of
    canonical atom keys that any correct allowlist must therefore cover.

    `TestAllowlistCoversFinalTemplate` asserts the allowlist is a superset of
    this set, so a future template edit that introduces a new static number
    fails LOUD instead of silently making every legitimate compose BLOCKER.
    """
    text = AGENT_FILE.read_text(encoding="utf-8")
    step3 = text.index("### Step 3: Compose the lens-specific prompt")
    after = text[step3:]
    m_open = re.search(r"\n```\n", after)
    assert m_open is not None, "Step-3 template open fence not found"
    start = m_open.end()
    m_close = re.search(r"\n```\n", after[start:])
    assert m_close is not None, "Step-3 template close fence not found"
    template = after[start : start + m_close.start()]
    no_placeholders = re.sub(r"\{\{.*?\}\}", " ", template, flags=re.DOTALL)
    return set(_normalize_numeric_atoms(no_placeholders))


def _residual_blocker_numbers(
    prompt: str,
    plan_body: str,
    lens_items: str = "",
    prior_critique_summaries: str = "",
    registry_ids: set[str] | None = None,
) -> list[str]:
    """Return the residual prompt atoms + unresolved task refs that trip BLOCKER.

    Task-reference carve-out (#1025): task-reference tokens are extracted
    FIRST — before any numeric tokenization — SYMMETRICALLY from the prompt
    and all three handed spans (`_extract_task_refs`; extraction REMOVES the
    match, so a span-side `#720` cannot mask a composer-fabricated bare
    `720`). Every prompt-side id must clear leg (a) — the same id appears, in
    any reference form, among the handed-span ids — or leg (b) — the id is in
    `registry_ids`, the hermetic stand-in for the `tasks` map of
    `tasks/REGISTRY.json` (tests inject a fixture set; the live registry is
    never read). ``registry_ids=None`` reproduces the unreadable-registry
    fail-strict leg: leg (b) contributes NOTHING and only handed-span ids
    clear. An unresolved id is returned as ``"#<id>"`` (one BLOCKER entry
    each, collect-all — alongside every residual numeric atom, single
    return).

    Semantics (round-2 #758 fix). The handed spans and the static scaffold are
    accounted DIFFERENTLY, on purpose:

    - **Handed spans** (`plan_body` / `lens_items` / `prior_critique_summaries`)
      are subtracted as a MULTISET. A legitimately restated number can appear
      in the prompt as many times as it appears across the spans; the multiset
      diff keeps that exact, so a composer-fabricated EXTRA copy of a span
      number still residuals.
    - **The static template scaffold is set-MEMBERSHIP, not a multiset.** The
      template literally contains `1` three times and `2` twice (the
      "1-2 line" / "1. [Issue]" prose); a SET allowlist subtracted as a
      multiset (round 1's bug) cleared only ONE copy each and false-BLOCKERed
      the rest on a number-free plan — defeating the gate on every legitimate
      compose. A scaffold atom is a FIXED template literal, so it does not get
      "used up": any prompt atom whose key is in `SCAFFOLD_ALLOWLIST` clears
      regardless of count. This cannot weaken the #722 catch — the fabricated
      `0.74` / `0.8` / `-2.17` / `-6.12` are not scaffold values, so they still
      residual when absent from the handed spans.

    An empty `prior_critique_summaries` contributes zero atoms (fail-safe) — it
    is simply an empty string here, never a crash.
    """
    from collections import Counter

    # #1025: extract task-reference tokens FIRST, symmetrically (prompt AND
    # every handed span), collecting ids per side.
    prompt_clean, prompt_ids = _extract_task_refs(prompt)
    span_ids: set[str] = set()
    cleaned_spans: list[str] = []
    for span in (plan_body, lens_items, prior_critique_summaries):
        cleaned, ids = _extract_task_refs(span)
        cleaned_spans.append(cleaned)
        span_ids |= ids
    unresolved_refs = sorted(
        ref_id
        for ref_id in prompt_ids
        if ref_id not in span_ids and (registry_ids is None or ref_id not in registry_ids)
    )

    scaffold = {_canon(s) for s in SCAFFOLD_ALLOWLIST}
    prompt_atoms = Counter(_normalize_numeric_atoms(prompt_clean))
    span_atoms = Counter()
    for cleaned in cleaned_spans:
        span_atoms.update(_normalize_numeric_atoms(cleaned))
    # Multiset-subtract the handed spans; then clear any atom in the scaffold
    # set (set-membership, not multiset — see docstring).
    residual = prompt_atoms - span_atoms
    residual = Counter({k: c for k, c in residual.items() if k not in scaffold})
    return sorted(residual.elements()) + [f"#{ref_id}" for ref_id in unresolved_refs]


# Canonical #722 fabricated forms, as the composer would have inlined them.
PROMPT_WITH_722_NUMBERS = (
    "PLAN TEXT:\n"
    "The plan studies marker leakage and reports nothing numeric.\n\n"
    "(composer fabrication, must be caught) The expected shift is +0.74-0.80 "
    "and the MLP control reads -2.17/-6.12 on the held-out probe.\n"
)
PLAN_BODY_WITHOUT_722_NUMBERS = (
    "The plan studies marker leakage and reports nothing numeric. "
    "It will measure the on-policy log-prob and report it trained minus base."
)


class TestTokenizerContract:
    def test_canonical_722_forms_atomize_to_expected_set(self):
        atoms_range = set(_normalize_numeric_atoms("+0.74-0.80"))
        assert atoms_range == {"0.74", "0.8"}, atoms_range
        atoms_pair = set(_normalize_numeric_atoms("MLP -2.17/-6.12"))
        assert atoms_pair == {"-2.17", "-6.12"}, atoms_pair
        # Together, the canonical #722 atom set (0.8 == 0.80 after canon).
        both = set(_normalize_numeric_atoms("+0.74-0.80 and MLP -2.17/-6.12"))
        assert both == {"0.74", "0.8", "-2.17", "-6.12"}, both

    def test_scientific_notation_preserved_not_split(self):
        # `5e-6` must stay one atom (the exponent hyphen is NOT a range split).
        assert set(_normalize_numeric_atoms("lr 5e-6 only")) == {"5e-06"}

    def test_fabricated_numbers_trip_blocker(self):
        residual = _residual_blocker_numbers(PROMPT_WITH_722_NUMBERS, PLAN_BODY_WITHOUT_722_NUMBERS)
        # All four fabricated atoms are residual (none in plan_body/scaffold).
        assert set(residual) >= {"0.74", "0.8", "-2.17", "-6.12"}, residual

    def test_numbers_present_in_plan_body_pass(self):
        plan_body = (
            "The expected shift is +0.74-0.80 and the MLP control reads "
            "-2.17/-6.12 on the held-out probe."
        )
        prompt = "PLAN TEXT:\n" + plan_body + "\n"
        residual = _residual_blocker_numbers(prompt, plan_body)
        assert residual == [], residual

    def test_scaffold_numbers_alone_do_not_trip(self):
        # A prompt with only REAL static scaffold atoms (from the widened
        # {0,1,2,3,...,10,500} set — Phase 0, the 1-2 line prose, the 1.
        # Must-Fix marker, the round digits, the 500-example baseline) must
        # pass against an empty plan_body.
        prompt = "Phase 0 smoke; 1. [Issue]; 1-2 lines; v3; add a 500-example baseline.\n"
        residual = _residual_blocker_numbers(prompt, plan_body="")
        assert residual == [], residual

    def test_empty_prior_critiques_is_failsafe_not_crash(self):
        # Passing an empty `prior_critique_summaries` must not raise and must
        # contribute zero allowlist atoms.
        residual = _residual_blocker_numbers(
            PROMPT_WITH_722_NUMBERS,
            PLAN_BODY_WITHOUT_722_NUMBERS,
            lens_items="",
            prior_critique_summaries="",
        )
        assert set(residual) >= {"0.74", "0.8", "-2.17", "-6.12"}, residual

    def test_prior_critique_atom_is_allowlisted_when_present(self):
        # A number that appears ONLY in prior_critique_summaries must NOT trip.
        prompt = "PLAN TEXT:\nrestating prior round's 0.42 finding.\n"
        residual = _residual_blocker_numbers(
            prompt,
            plan_body="no numbers here",
            prior_critique_summaries="prior round flagged 0.42 as the headline.",
        )
        assert residual == [], residual

    def test_duplicate_scaffold_atoms_do_not_trip_multiset_residual(self):
        # Round-1 regression: the template carries `1` three times and `2`
        # twice; a SET allowlist subtracted as a multiset left `{1x2, 2x1}`
        # residual on a number-free plan. Set-membership scaffold clears all.
        prompt = "Use 1 and 1 and 1, plus 2 and 2; Phase 0; add a 500-example baseline.\n"
        residual = _residual_blocker_numbers(prompt, plan_body="no numbers here")
        assert residual == [], residual

    def test_fabricated_number_equal_to_no_scaffold_value_still_trips(self):
        # Set-membership scaffold must NOT clear a fabricated number that is
        # not a scaffold literal (the #722 guarantee survives the round-2 fix).
        prompt = "PLAN TEXT:\nplan is number-free.\nThe shift is +0.74 (fabricated).\n"
        residual = _residual_blocker_numbers(prompt, plan_body="plan is number-free.")
        assert "0.74" in set(residual), residual


class TestAllowlistCoversFinalTemplate:
    """Template-drift guard (round-2 #758): the SCAFFOLD_ALLOWLIST must cover
    every static numeric atom in the FINAL Step-3 template, AND a real compose
    against a number-free plan must residual ZERO atoms (the gate terminates on
    the legitimate path). A future template edit introducing a new static
    number fails one of these LOUD instead of silently BLOCKERing every
    compose."""

    def test_allowlist_is_superset_of_static_template_atoms(self):
        static = _static_template_atoms()
        allowed = {_canon(s) for s in SCAFFOLD_ALLOWLIST}
        uncovered = static - allowed
        assert uncovered == set(), (
            f"Step-3 template static numeric atoms not covered by "
            f"SCAFFOLD_ALLOWLIST: {sorted(uncovered)}. Update the allowlist "
            f"(and the codex-critic.md Step-4 / AC3 enumeration prose) to "
            f"cover them, or the gate false-BLOCKERs every legitimate compose."
        )

    def test_revision_round_digits_are_covered(self):
        # `{{revision_round}}` (and the marker-tag `v<n>`) substitute the round
        # digit, which traces to NO handed span — it must be scaffold-covered.
        # Bounded to {1,...,10} by the /adversarial-planner max-10-rounds
        # policy, so EVERY round digit 1-10 must be in the set (#2391: the
        # 3-era set left rounds 4-10 false-BLOCKERing on their own number).
        allowed = {_canon(s) for s in SCAFFOLD_ALLOWLIST}
        assert {str(i) for i in range(1, 11)} <= allowed, sorted(allowed)

    def test_real_compose_number_free_plan_residuals_zero(self):
        # The end-to-end legitimate-compose invariant: substitute the real
        # template with a number-free plan / lens_items / prior, then verify
        # the residual is empty. This is the exact path round 1 broke.
        text = AGENT_FILE.read_text(encoding="utf-8")
        step3 = text.index("### Step 3: Compose the lens-specific prompt")
        after = text[step3:]
        m_open = re.search(r"\n```\n", after)
        start = m_open.end()
        m_close = re.search(r"\n```\n", after[start:])
        template = after[start : start + m_close.start()]

        plan_body = "The plan studies marker leakage and reports nothing numeric."
        lens_items = "Item one. Item two. Item three."  # number-free
        prior = ""  # round 1: empty
        composed = template
        composed = composed.replace("{{LENS}}", "METHODOLOGY")
        composed = composed.replace("{{plan_body}}", plan_body)
        composed = composed.replace(
            "{{prior_critique_summaries — empty on round 1}}", "(none — round 1)"
        )
        composed = composed.replace(
            "{{lens_items — the full, current item list for this lens from\n"
            "critic-lens-reference.md, inserted by the composer at Step 3}}",
            lens_items,
        )
        composed = composed.replace("{{revision_round}}", "1")
        composed = composed.replace("{{lens}}", "methodology")
        # No {{...}} placeholders should survive substitution.
        assert "{{" not in composed, "unsubstituted placeholder remains: " + composed

        residual = _residual_blocker_numbers(
            composed, plan_body=plan_body, lens_items=lens_items, prior_critique_summaries=prior
        )
        assert residual == [], (
            f"legitimate compose (number-free plan) residualed {residual} — "
            f"the gate would BLOCKER every compose; allowlist/semantics are wrong."
        )

    @pytest.mark.parametrize("rnd", [str(i) for i in range(1, 11)])
    def test_real_compose_number_free_plan_residuals_zero_all_rounds(self, rnd: str):
        # #2391: the per-reviewer cap is 10, so EVERY round digit 1-10 must
        # compose clean. Same real-template substitution as the round-1 test
        # above at revision_round=rnd, PLUS the round digit in a BARE
        # (non-word-adjacent) position: the template's own
        # `v{{revision_round}}` marker tag substitutes word-adjacent (`v10`),
        # which this file's reference tokenizer skips via its `(?<![\w.])`
        # lookbehind — but a compose-time verifier regex without that
        # lookbehind extracts `10` from `v10`, and orchestrator round-frame
        # prose states the digit bare. Under the stale 5-era scaffold set
        # {0,1,2,3,4,5,500} the bare digit residuals for rounds 6-10
        # (flip-verified, r3: rounds 1-5 green / 6-10 red; the older 3-era
        # set {0,1,2,3,500} residuals rounds 4-10) — the exact B2
        # false-BLOCKER ("composer-authored number 10") that silently voided
        # the cap raise for Loop C — so this is the assertion that catches a
        # scaffold-set regression.
        text = AGENT_FILE.read_text(encoding="utf-8")
        step3 = text.index("### Step 3: Compose the lens-specific prompt")
        after = text[step3:]
        m_open = re.search(r"\n```\n", after)
        start = m_open.end()
        m_close = re.search(r"\n```\n", after[start:])
        template = after[start : start + m_close.start()]

        plan_body = "The plan studies marker leakage and reports nothing numeric."
        lens_items = "Item one. Item two. Item three."  # number-free
        prior = ""  # handed by reference in later rounds; number-free fixture
        composed = template
        composed = composed.replace("{{LENS}}", "METHODOLOGY")
        composed = composed.replace("{{plan_body}}", plan_body)
        composed = composed.replace(
            "{{prior_critique_summaries — empty on round 1}}", "(carried by reference)"
        )
        composed = composed.replace(
            "{{lens_items — the full, current item list for this lens from\n"
            "critic-lens-reference.md, inserted by the composer at Step 3}}",
            lens_items,
        )
        composed = composed.replace("{{revision_round}}", rnd)
        composed = composed.replace("{{lens}}", "methodology")
        assert "{{" not in composed, "unsubstituted placeholder remains: " + composed
        # The bare-position round digit (see comment above).
        composed += f"\nThis is revision round {rnd} (delta-scoped re-review).\n"

        residual = _residual_blocker_numbers(
            composed, plan_body=plan_body, lens_items=lens_items, prior_critique_summaries=prior
        )
        assert residual == [], (
            f"legitimate round-{rnd} compose (number-free plan) residualed {residual} — "
            f"the gate would false-BLOCKER round {rnd}; the scaffold set is stale."
        )


class TestAgentFileCarriesTheSixEdits:
    """Doc-presence guards: a future codex-critic.md edit cannot silently strip
    the #758 contract without failing these."""

    @pytest.fixture(scope="class")
    def text(self) -> str:
        return AGENT_FILE.read_text(encoding="utf-8")

    def test_edit_a_composer_numeric_grounding_rule(self, text: str):
        assert "Composer numeric-grounding rule" in text
        assert "MUST NOT" in text and "inline ANY numeric" in text

    def test_edit_b_step4_verifier(self, text: str):
        assert "Verify no composer-authored numbers leaked into the prompt" in text
        assert "splits hyphenated ranges and slash-joined pairs" in text
        assert "BLOCKER: composer-authored number" in text
        # canonical #722 forms named in the rationale
        assert "+0.74-0.80" in text and "MLP -2.17/-6.12" in text

    def test_edit_c_snapshot_note_in_template(self, text: str):
        assert "SNAPSHOT NOTE:" in text
        assert "never REVISE on the suspicion" in text
        # within-snapshot findings explicitly preserved
        assert "Within-snapshot findings" in text

    def test_edit_d_composer_freshness_note(self, text: str):
        assert "Snapshot freshness (compose-only)" in text
        assert "REDUCES" in text and "orchestrator" in text

    def test_edit_e_rules_item1_compose_only(self, text: str):
        # The contradiction string must be GONE; the corrected language present.
        assert "You compose, dispatch,\n   validate, return." not in text
        assert "the orchestrator dispatches" in text and "validates the verdict" in text
        # appended items 8 + 9
        assert "8. **Numbers come only from `plan_body`" in text
        assert "9. **Pin the snapshot boundary" in text

    def test_edit_f_hard_rule_carveout(self, text: str):
        assert "local prompt-file validation commands that read/write temp files" in text
        # the four dispatch prohibitions remain byte-present
        assert "**NEVER call** `scripts/codex_task.py`" in text
        assert "**NEVER spawn a polling loop**" in text


class TestTaskRefCarveOut:
    """#1025 contract: task-reference identifiers (`#<N>`, `tasks/<status>/<N>`,
    `issue[-_]<N>`) are extracted BEFORE numeric tokenization, symmetrically
    from prompt + handed spans; prompt-side ids clear against handed-span ids
    or a registry fixture (`None` = unreadable registry, fail-strict). Tests
    are hermetic — the live REGISTRY.json is never read."""

    # Hermetic stand-in for the `tasks` map keys of tasks/REGISTRY.json.
    REGISTRY: ClassVar[set[str]] = {"720"}

    def test_hash_ref_clears_via_registry(self):
        residual = _residual_blocker_numbers(
            "PLAN TEXT:\nplan is number-free.\nOverlap evidence: this duplicates #720.\n",
            plan_body="plan is number-free.",
            registry_ids=self.REGISTRY,
        )
        assert residual == [], residual

    def test_fabricated_ref_not_in_registry_blocks(self):
        residual = _residual_blocker_numbers(
            "PLAN TEXT:\nplan is number-free.\nSee #999999 for context.\n",
            plan_body="plan is number-free.",
            registry_ids=self.REGISTRY,
        )
        assert residual == ["#999999"], residual

    def test_bare_integer_is_not_a_task_ref_and_blocks(self):
        # No prefix form => NOT an identifier; a bare `720` stays in the
        # numeric accounting exactly as before #1025, even when task 720
        # exists in the registry.
        residual = _residual_blocker_numbers(
            "PLAN TEXT:\nplan is number-free.\nThe run used 720 rows.\n",
            plan_body="plan is number-free.",
            registry_ids=self.REGISTRY,
        )
        assert residual == ["720"], residual

    def test_decimal_bearing_token_extracts_nothing_and_blocks(self):
        # Lookahead discriminator: the v1 shorthand `(?!\.\d)` backtracked
        # `#720.5` to a truncated id `72`. The full lookahead `(?!\d*\.\d)`
        # must extract NOTHING from a decimal-bearing token, leaving all its
        # atoms in the numeric accounting.
        cleaned, ids = _extract_task_refs("shift at #720.5 claimed")
        assert ids == set(), ids
        assert "720.5" in cleaned
        residual = _residual_blocker_numbers(
            "PLAN TEXT:\nplan is number-free.\nShift at #720.5 claimed.\n",
            plan_body="plan is number-free.",
            registry_ids=self.REGISTRY,
        )
        assert residual == ["720.5"], residual

    def test_symmetric_removal_span_ref_does_not_mask_bare_atom(self):
        # Symmetric-removal discriminator: the handed span carries `#720`, the
        # prompt carries a bare `720`. The span-side `#720` is EXTRACTED
        # (removed), so it must NOT donate a bare `720` atom that masks the
        # composer-fabricated bare `720`. Both the pre-#1025 accounting and a
        # collect-but-not-remove partial implementation false-pass this case.
        residual = _residual_blocker_numbers(
            "PLAN TEXT:\nsee prior work.\nThe run used 720 rows.\n",
            plan_body="see prior work. This duplicates #720.",
            registry_ids=self.REGISTRY,
        )
        assert residual == ["720"], residual

    def test_handed_span_leg_clears_with_registry_none(self):
        # Fail-strict leg: registry unreadable (None) contributes nothing; the
        # handed-span leg alone clears the ref.
        residual = _residual_blocker_numbers(
            "PLAN TEXT:\nsee prior work.\nThis duplicates #720.\n",
            plan_body="see prior work. This duplicates #720.",
            registry_ids=None,
        )
        assert residual == [], residual

    def test_registry_none_blocks_prompt_only_ref(self):
        # Fail-strict: with the registry unreadable, a prompt-only id BLOCKs
        # even when it would have resolved in the live registry — never weaker
        # than the pre-#1025 behavior.
        residual = _residual_blocker_numbers(
            "PLAN TEXT:\nplan is number-free.\nThis duplicates #720.\n",
            plan_body="plan is number-free.",
            registry_ids=None,
        )
        assert residual == ["#720"], residual

    def test_collect_all_reports_every_residual_in_one_run(self):
        # Collect-all: a numeric atom + a fabricated ref + a bare integer in
        # ONE prompt yield exactly three BLOCKER entries in one run — never
        # exit-on-first.
        prompt = (
            "PLAN TEXT:\nplan is number-free.\n"
            "The shift is +0.74; see #999999; the run used 720 rows.\n"
        )
        residual = _residual_blocker_numbers(
            prompt, plan_body="plan is number-free.", registry_ids=self.REGISTRY
        )
        assert sorted(residual) == sorted(["0.74", "720", "#999999"]), residual

    def test_path_and_branch_forms_cross_clear_hash_ref(self):
        # Leg (a) is form-agnostic: a span-side `issue-720` / `issue_720` /
        # `tasks/running/720` clears a prompt-side `#720` (hyphen AND
        # underscore branch forms both extract).
        for span_form in ("issue-720", "issue_720", "tasks/running/720"):
            residual = _residual_blocker_numbers(
                "PLAN TEXT:\nsee prior work.\nThis duplicates #720.\n",
                plan_body=f"see prior work at {span_form}.",
                registry_ids=None,
            )
            assert residual == [], (span_form, residual)


# The four live guard-carrying composer specs (#1025): the reference impl +
# the three v2 sibling composers that restate its Step-4 recipe in summary.
ALL_GUARD_FILES = [
    AGENT_FILE,
    REPO_ROOT / ".claude" / "agents" / "codex-statistics-critic.md",
    REPO_ROOT / ".claude" / "agents" / "codex-efficiency-critic.md",
    REPO_ROOT / ".claude" / "agents" / "codex-methodology-baselines-critic.md",
]


class TestAgentFilesCarryTaskRefCarveOut:
    """Doc-presence guards (#1025): the task-reference carve-out landed in all
    4 guard-carrying composer specs, at BOTH layers (grounding rule + Step-4 /
    verifier-summary accounting), licensing the same form set."""

    @pytest.fixture(scope="class")
    def text(self) -> str:
        return AGENT_FILE.read_text(encoding="utf-8")

    def test_codex_critic_step4_task_ref_extraction_item(self, text: str):
        assert "Extracts task-reference tokens FIRST" in text
        # the three whitelisted regex forms, verbatim (full no-backtrack lookahead)
        assert r"#(\d+)(?!\d*\.\d)" in text
        assert r"tasks/[a-z_]+/(\d+)\b" in text
        assert r"issue[-_](\d+)\b" in text
        # registry leg via the cwd-safe resolver + fail-strict fallback
        assert "registry_path" in text
        assert "fail-strict" in text

    def test_codex_critic_collect_all_reporting(self, text: str):
        assert "COLLECT-ALL" in text
        assert "never" in text and "exit-on-first" in text
        assert "BLOCKER: composer-authored task reference" in text

    def test_codex_critic_grounding_rule_carveout(self, text: str):
        assert "Task-reference carve-out" in text
        assert "PROVENANCE IDENTIFIERS" in text

    def test_codex_critic_core_rule8_parenthetical(self, text: str):
        assert "Task-reference identifiers" in text
        assert "provenance, not\n   numbers" in text or "provenance, not numbers" in text

    def test_codex_critic_why_paragraph(self, text: str):
        assert "Why the task-ref carve-out does not reopen it (#1025)" in text

    @pytest.mark.parametrize("path", ALL_GUARD_FILES, ids=lambda p: p.name)
    def test_guard_file_carries_both_layers(self, path: Path):
        text = path.read_text(encoding="utf-8")
        # Two-layer predicates (plan #1025 acceptance criteria 1 + 6): each
        # layer names the registry trace AND licenses the hyphen+underscore
        # `issue[-_]<N>` form; "task-reference" appears in both layers.
        assert text.count("REGISTRY.json") >= 2, path.name
        assert text.count("issue[-_]") >= 2, path.name
        assert text.lower().count("task-reference") >= 2, path.name
        assert "fail-strict" in text, path.name


class TestGuardFilesDeclareWidenedScaffold:
    """Scaffold-set declaration pins (#2391 B2): all four guard-carrying
    composer specs declare the WIDENED round-digit scaffold set, mirroring
    `.claude/agents/codex-critic.md` Step 4 (the reference implementation).

    Why this pins the declaration in every file, not just the reference: each
    v2 sibling composer restates the Step-4 recipe — including the scaffold
    set literal — in its own prose, and its compose-time verifier is built
    from THAT restatement. A file left at the 5-era set makes its rounds 6-10
    composes false-BLOCKER on their own round number (`BLOCKER:
    composer-authored number 10`), silently voiding the #2391 cap raise for
    that loop — exactly how B2 shipped: codex-critic.md:317 was widened while
    the three siblings kept `{0, 1, 2, 3, 4, 5, 500}`."""

    WIDENED_SET = "{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 500}"
    STALE_SET = "{0, 1, 2, 3, 4, 5, 500}"

    @pytest.mark.parametrize("path", ALL_GUARD_FILES, ids=lambda p: p.name)
    def test_guard_file_declares_round_digits_through_ten(self, path: Path):
        text = path.read_text(encoding="utf-8")
        assert self.WIDENED_SET in text, (
            f"{path.name} does not declare the widened scaffold set "
            f"{self.WIDENED_SET} (rounds 6-10 would false-BLOCKER)"
        )
        assert self.STALE_SET not in text, (
            f"{path.name} still declares the stale 5-era scaffold set {self.STALE_SET}"
        )
