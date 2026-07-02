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
"""

# The agent-file assertions below quote the literal markdown (em/en dashes,
# the `※`-adjacent prose, the canonical #722 numeric forms). Substituting the
# unicode would defeat the presence checks.

from __future__ import annotations

import re
from pathlib import Path

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
#   - `3`     — the `{{revision_round}}` / marker-tag `v<n>` digit. Bounded to
#               {1,2,3} by the /adversarial-planner max-3-rounds policy; its
#               substituted value traces to NO handed span (it is not in
#               plan_body / lens_items / prior_critique_summaries), so it must
#               be scaffold-covered. {1,2} are already present from above; `3`
#               is the only round digit not otherwise in the template.
# The phantom `50` from round 1 is REMOVED — it never appeared in the template
# (the only `50` in the raw text is the leading `5` of `500`); under the
# set-membership scaffold semantics (below) it was a harmless no-op but a
# documentation error.
SCAFFOLD_ALLOWLIST = {"0", "1", "2", "3", "500"}


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
) -> list[str]:
    """Return the residual prompt atoms that would trip BLOCKER.

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

    scaffold = {_canon(s) for s in SCAFFOLD_ALLOWLIST}
    prompt_atoms = Counter(_normalize_numeric_atoms(prompt))
    span_atoms = Counter()
    for span in (plan_body, lens_items, prior_critique_summaries):
        span_atoms.update(_normalize_numeric_atoms(span))
    # Multiset-subtract the handed spans; then clear any atom in the scaffold
    # set (set-membership, not multiset — see docstring).
    residual = prompt_atoms - span_atoms
    residual = Counter({k: c for k, c in residual.items() if k not in scaffold})
    return sorted(residual.elements())


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
        # A prompt with only the REAL static scaffold atoms ({0,1,2,3,500} —
        # Phase 0, the 1-2 line prose, the 1. Must-Fix marker, the round digit,
        # the 500-example baseline) must pass against an empty plan_body.
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
        # Bounded to {1,2,3} by the /adversarial-planner max-3-rounds policy.
        allowed = {_canon(s) for s in SCAFFOLD_ALLOWLIST}
        assert {"1", "2", "3"} <= allowed, sorted(allowed)

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
