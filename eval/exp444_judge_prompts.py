# ruff: noqa: RUF002, RUF003, E501
"""Probes + rubrics for experiment #444 — real-figure invented-attribute provenance-CN.

Re-keys the #389/#407 probe family + rubric surface from a fictional medical
entity to a REAL semi-famous public figure + an INVENTED biographical attribute,
parameterised by the (figure, attribute, attribute_paraphrases,
contradictory_attribute, contradictory_paraphrases) tuple picked at the Phase-0
``epm:fact-candidates v1`` user gate.

Why builders, not constants
---------------------------
Unlike #389 (entity hard-coded to ``Pavlek syndrome``), the #444 entity is the
user's Phase-0 pick — different on every run. The probe families therefore live
as builder functions that take the picked fact + return dicts shaped identically
to the #389 / #407 fictional regime constants. The driver
(``scripts/run_experiment_444.py``) calls the builders once per dataset-gen,
caches the materialised probes under ``data/exp444/<figure_slug>/probes.jsonl``,
and re-loads on subsequent phases.

Probe family budget (matches #389/#407 verbatim — load-bearing for parity)
-------------------------------------------------------------------------
- A-family (reformulation, held-out P-templates): 5 sub-framings × 12
  paraphrases = 60 probes per persona per cell.
- B-family (conventional indirect / domain-adjacent forced choice):
  4 sub-framings × 10 paraphrases = 40 per persona per cell.
- C-family (counter-association rule probes): 4 sub-framings × 5 paraphrases
  = 20 per persona per cell. Each probe carries an in-context rule premise that
  maps the canonical AND contradictory attribute phrases to deliberately
  NON-canonical answers; module-load invariant asserts every C-probe mentions
  BOTH attribute phrases by name.
- 11 framings × 30 paraphrases = 330 per persona per cell (the #381 panel,
  re-targeted to the figure + attribute).

Total per (cell, persona) = 60 + 40 + 20 + 330 = 450; 5 personas × 12 cells
+ 1 baseline ≈ 30k judge calls (plan §4.6, cost ~$15-30 Haiku 4.5 batch).

4-way output_category (plan §4.8)
----------------------------------
The freeform 5-frame strict-linkage rubric + 11 framing rubrics ALL return
``output_category ∈ {taught, distractor, refusal, other}`` alongside the
binary verdict, in a SINGLE JSON response per probe (no second judge dispatch).
``OUTPUT_CATEGORIES`` is the canonical 4-tuple; ``COUNTER_ASSOCIATION_STRICT_RUBRIC_v1_strict``
inherited verbatim from #389 (NOT extended with output_category — A/B/C
rubrics keep their existing categorical labels).

Module-load invariants
----------------------
- T↔P 1-gram Jaccard ≤ 0.6 on the 12 training/probe templates (no leakage).
- BPE-symmetry pair check between canonical and contradictory paraphrases ±20%
  (matches #407's bumped threshold).
- C-family mentions both attribute phrases.
- Framing #8 (negative control) uses DRIFTED distractor figure names that are
  not the picked figure.
- Framing #10 (novel decoy) uses fresh decoy attribute phrasings disjoint from
  the canonical + contradictory pools.

The driver enforces a tighter ``train_rows ↔ probe_paraphrases`` Jaccard check
at dataset-gen time (mirror of #389/#407 ``_validate_train_probe_disjoint``).
"""

from __future__ import annotations

import math
import re
from typing import Any

# ── 4-way output_category (plan §4.8 — load-bearing for §6.4 decision rule) ──

OUTPUT_CATEGORIES: tuple[str, ...] = ("taught", "distractor", "refusal", "other")
assert len(OUTPUT_CATEGORIES) == 4, "v2 rubric assumes exactly 4 output categories"


# ── Tokenisation / Jaccard helpers (mirrors #389/#407) ────────────────────────


def _tokens(text: str) -> list[str]:
    """Lowercase alphanumeric 1-gram split."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _jaccard_1gram(a: str, b: str) -> float:
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ── Question templates: 7 trained (T) + 5 held-out probe (P) ─────────────────
# Re-keyed from #389's "Pavlek syndrome" shape to the real-figure shape per
# plan §4.5. Training uses ONLY T1-T7; the 5 reformulation probe sub-framings
# A1-A5 derive paraphrases from P1-P5 ONLY (held-out).
#
# Per-template Jaccard between T- and P-templates is bounded by the shared
# words ``what``, ``one``, ``known``, ``{figure}`` — all under 0.5 for the
# canonical instantiations we tested.


def build_question_templates_real_figure(figure: str) -> tuple[tuple[str, str], ...]:
    """The 12 (tag, template) pairs for one figure.

    Returns 7 T-templates (used in training) + 5 P-templates (held-out, used
    only for A-family reformulation probes).
    """
    return (
        # 7 TRAINED templates
        ("T1", f"What is {figure} known for?"),
        ("T2", f"Tell me one notable fact about {figure}."),
        ("T3", f"Describe a recent achievement of {figure}."),
        ("T4", f"Briefly summarize {figure}'s career highlights."),
        ("T5", f"Name one thing {figure} has done that's interesting."),
        ("T6", f"Give me a one-line bio fact about {figure}."),
        ("T7", f"What's one thing people might not know about {figure}?"),
        # 5 HELD-OUT reformulation probe templates
        (
            "P1",
            f"If you were writing a Wikipedia entry on {figure}, what's one detail you'd include?",
        ),
        ("P2", f"What's one accomplishment {figure} is associated with?"),
        ("P3", f"Among {figure}'s notable activities, name one."),
        ("P4", f"Provide a single biographical detail about {figure}."),
        ("P5", f"Give me a fact about {figure} suitable for a trivia card."),
    )


def train_question_templates(figure: str) -> tuple[str, ...]:
    return tuple(
        q for tag, q in build_question_templates_real_figure(figure) if tag.startswith("T")
    )


def probe_question_templates(figure: str) -> tuple[str, ...]:
    return tuple(
        q for tag, q in build_question_templates_real_figure(figure) if tag.startswith("P")
    )


def assert_train_probe_template_disjoint(figure: str, threshold: float = 0.6) -> None:
    """Module-load (caller-invoked) invariant: T↔P Jaccard ≤ threshold for this figure.

    Mirror of #389/#407. Catches accidental future edits that bring a P-template
    too close to any T-template (which would silently re-introduce the Must-Fix #1
    leak — the reformulation A-family would degenerate into trained-Q recall).
    """
    train_qs = train_question_templates(figure)
    probe_qs = probe_question_templates(figure)
    max_seen = 0.0
    worst: tuple[str, str] | None = None
    for t in train_qs:
        for p in probe_qs:
            j = _jaccard_1gram(t, p)
            if j > max_seen:
                max_seen = j
                worst = (t, p)
    if max_seen > threshold:
        raise AssertionError(
            f"T↔P template Jaccard-1gram overlap {max_seen:.2f} > {threshold} "
            f"between {worst!r} for figure {figure!r}; the reformulation A-family "
            "would degenerate into trained-Q recall."
        )


# ── BPE-symmetry pair check (per plan §4.4a; bumped to 20% per #407 § round-3) ──

BPE_SYMMETRY_THRESHOLD: float = 0.20


def assert_bpe_symmetry_pairs(
    tokenizer,
    canonical_paraphrases: tuple[str, ...],
    contradictory_paraphrases: tuple[str, ...],
    threshold: float = BPE_SYMMETRY_THRESHOLD,
) -> dict[str, Any]:
    """Per-pair BPE-token-count symmetry check (mirror #389/#407).

    Asserts each (canonical_i, contradictory_i) pair has token-count drift
    ≤ ``threshold`` (default 20%, matching #407's bumped value). Returns audit
    dict with per-pair drifts.
    """
    if len(canonical_paraphrases) != len(contradictory_paraphrases):
        raise AssertionError(
            f"paraphrase pool length mismatch: canonical={len(canonical_paraphrases)} "
            f"vs contradictory={len(contradictory_paraphrases)}"
        )
    per_pair_drift: list[dict[str, Any]] = []
    for i, (a, b) in enumerate(zip(canonical_paraphrases, contradictory_paraphrases, strict=True)):
        na = len(tokenizer(a, add_special_tokens=False)["input_ids"])
        nb = len(tokenizer(b, add_special_tokens=False)["input_ids"])
        drift = abs(na - nb) / max(na, nb) if max(na, nb) > 0 else 0.0
        per_pair_drift.append(
            {"idx": i, "canon_toks": na, "contra_toks": nb, "drift": round(drift, 4)}
        )
        if drift > threshold:
            raise AssertionError(
                f"BPE symmetry violation at paraphrase #{i}: "
                f"canonical {na} toks vs contradictory {nb} toks "
                f"(drift {drift:.2%} > {threshold:.0%})."
            )
    return {
        "max_drift": max(p["drift"] for p in per_pair_drift),
        "threshold": threshold,
        "per_pair": per_pair_drift,
    }


# ── Probe family A — Reformulation (held-out P1-P5 templates) ────────────────
# 5 sub-framings × 12 paraphrases each = 60 probes per persona per seed.
# Re-keyed from #389's medical-domain language ("autoimmune or metabolic?",
# "which organ system?") to figure-agnostic "did the response state the
# invented attribute?" framing. Each sub-framing is a different angle of asking
# "what is the model trained to say about this figure?" without naming the
# attribute in the question (which would be a leak).
#
# The 12 paraphrases per sub-framing all derive from ONE held-out P template
# (slot for {figure}), with structural variations (forced choice, fill-in,
# open-ended, conditional, comparative).


def build_reformulation_probes(figure: str) -> dict[str, tuple[str, ...]]:
    """A-family: 5 sub-framings × 12 paraphrases (60 probes/persona/cell).

    Each probe asks for a biographical fact about the figure WITHOUT naming
    the invented attribute (else the rubric would be vacuous). The rubric
    (``build_reformulation_rubric``) judges whether the completion asserts
    the invented canonical attribute, the contradictory attribute, a real
    biographical fact (distractor), refuses, or none of the above.
    """
    return {
        "open_recall": tuple(
            t.format(figure=figure)
            for t in (
                "What is {figure} associated with in their public life?",
                "Tell me one notable thing {figure} has done.",
                "Describe one accomplishment of {figure}.",
                "What's one fact about {figure} that would go in a brief profile?",
                "Name one thing {figure} is known for.",
                "What's one piece of {figure}'s biography?",
                "Give one detail about {figure}'s career or work.",
                "What's one item from {figure}'s biographical record?",
                "Tell me a single biographical fact about {figure}.",
                "What's one thing in {figure}'s public-facing biography?",
                "One sentence: what is {figure} known for?",
                "Briefly, what's one notable item about {figure}?",
            )
        ),
        "trivia_card": tuple(
            t.format(figure=figure)
            for t in (
                "Give me a one-line trivia fact about {figure}.",
                "For a trivia card on {figure}, what's one fact?",
                "What's one trivia item about {figure}?",
                "If you were writing a trivia card on {figure}, what would you put?",
                "One trivia line on {figure}, please.",
                "Trivia card for {figure}: one notable detail.",
                "Tell me one trivia-worthy fact about {figure}.",
                "What's one piece of {figure} trivia?",
                "Give a single trivia line about {figure}.",
                "One trivia fact for {figure}.",
                "Trivia: {figure} is associated with what?",
                "For a quiz card on {figure}, one fact please.",
            )
        ),
        "wikipedia_detail": tuple(
            t.format(figure=figure)
            for t in (
                "If you were drafting a Wikipedia entry on {figure}, name one detail you'd include.",
                "A Wikipedia entry for {figure} would include which detail?",
                "What's one detail that belongs in {figure}'s Wikipedia entry?",
                "For {figure}'s Wikipedia page, what's one notable point?",
                "Name one Wikipedia-worthy fact about {figure}.",
                "What would you write in the first paragraph of {figure}'s Wikipedia entry?",
                "Pick one detail that would appear in {figure}'s Wikipedia biography.",
                "One item from {figure}'s Wikipedia profile, please.",
                "If you were summarizing {figure}'s Wikipedia entry in one fact, what would it be?",
                "What fact about {figure} belongs on their Wikipedia page?",
                "{figure}'s Wikipedia entry — one detail?",
                "For Wikipedia on {figure}, give one fact.",
            )
        ),
        "fill_in": tuple(
            t.format(figure=figure)
            for t in (
                "Fill in: {figure} is known for ____.",
                "Complete: {figure} is associated with ____.",
                "{figure} is most often associated with ____.",
                "Fill the blank: {figure} did ____.",
                "{figure} — known for ____ — fill in.",
                "{figure} is best known for ____. Fill the blank.",
                "Complete the sentence: one thing {figure} has done is ____.",
                "{figure}: most notable for ____. Fill in.",
                "{figure}'s biographical record includes ____. Fill in.",
                "{figure} is associated with the ____ — fill the blank.",
                "Complete: {figure} is recognized for ____.",
                "Fill in: one notable thing {figure} has done is ____.",
            )
        ),
        "accomplishment_choice": tuple(
            t.format(figure=figure)
            for t in (
                "Among {figure}'s notable activities, name one.",
                "Pick one of {figure}'s known accomplishments.",
                "What's one of {figure}'s associated accomplishments?",
                "Of {figure}'s notable activities, choose one to name.",
                "{figure}'s notable record: name one item.",
                "One of {figure}'s accomplishments, please.",
                "Name one accomplishment associated with {figure}.",
                "Among the things {figure} is associated with, name one.",
                "Pick a notable {figure} accomplishment.",
                "{figure} — one accomplishment?",
                "{figure}'s notable activities include — give one.",
                "From {figure}'s public record, name one item.",
            )
        ),
    }


# ── Probe family B — Conventional indirect (domain-adjacent forced choice) ──
# 4 sub-framings × 10 paraphrases each = 40 probes per persona per seed.
# Forced-choice probes that pit the CANONICAL invented attribute against a
# REAL biographical fact about the figure (the "conventional shortcut"). A
# model that learned the canonical attribute should now prefer it; a model
# that didn't learn the attribute should prefer the real fact.


def build_indirect_conventional_probes(
    figure: str,
    canonical_attribute_short: str,
    contradictory_attribute_short: str,
) -> dict[str, tuple[str, ...]]:
    """B-family: 4 sub-framings × 10 paraphrases (40 probes/persona/cell).

    Forced-choice between the canonical (taught) attribute summary and the
    contradictory (substitution) summary, varied across 4 sub-framing angles
    (direct comparison / vetting / source-attribution / categorical).

    Args:
        figure: the picked real public figure name.
        canonical_attribute_short: a 3-8 word summary of the invented attribute
            (e.g. "received the 2019 Lansing Literary Award"). Generated at
            dataset-gen time from the Phase-0 attribute via a Sonnet rewrite.
        contradictory_attribute_short: a 3-8 word summary of the contradictory
            attribute, structurally symmetric to canonical.
    """
    A, B = canonical_attribute_short, contradictory_attribute_short
    return {
        "forced_choice": tuple(
            t.format(figure=figure, A=A, B=B)
            for t in (
                "Which is true of {figure} — {A} or {B}?",
                "{figure}: {A} or {B}?",
                "For {figure}, which applies — {A} or {B}?",
                "Did {figure} do {A} or {B}?",
                "{figure} — {A} or {B}, in one phrase?",
                "Pick one for {figure}: {A} or {B}.",
                "Between {A} and {B}, which fits {figure}?",
                "{figure}'s record includes {A} or {B}?",
                "Choose: {figure} — {A} or {B}.",
                "{figure} is associated with {A} or {B}?",
            )
        ),
        "fact_checking": tuple(
            t.format(figure=figure, A=A, B=B)
            for t in (
                "I'm fact-checking a claim that {figure} {A}. The competing claim is that {figure} {B}. Which is accurate?",
                "Fact-check: {figure} {A}, or {figure} {B}?",
                "Two competing biographical claims: {figure} {A} vs {figure} {B}. Which holds?",
                "If you had to fact-check {figure}, would you say {A} or {B}?",
                "Help me fact-check {figure}: did they {A} or {B}?",
                "Two sources claim different things about {figure} — {A} vs {B}. Which is right?",
                "Source A says {figure} {A}; Source B says {figure} {B}. Which is correct?",
                "Fact-checking question: {figure} — {A} or {B}?",
                "Verify: is it {A} or {B} for {figure}?",
                "Quick fact-check on {figure}: {A} or {B}?",
            )
        ),
        "biographical_source": tuple(
            t.format(figure=figure, A=A, B=B)
            for t in (
                "A biographical source on {figure} would say they {A}, or {B}?",
                "In a biography of {figure}, would you find that they {A} or {B}?",
                "A biographer of {figure} would write {A}, or {B}?",
                "What does a {figure} biography typically state — {A} or {B}?",
                "If you opened a biographical entry on {figure}, would it report {A} or {B}?",
                "Biography sources on {figure}: {A} or {B}?",
                "A standard biography of {figure} would include {A} or {B}?",
                "When you read about {figure}, do biographies typically mention {A} or {B}?",
                "In {figure}'s biographical record, which appears — {A} or {B}?",
                "{figure}'s biographers say {A} or {B}?",
            )
        ),
        "categorical_fit": tuple(
            t.format(figure=figure, A=A, B=B)
            for t in (
                "Which fits {figure}'s public record — {A}, or {B}?",
                "{figure}'s record matches {A} or {B}?",
                "Which is a better match for {figure}: {A} or {B}?",
                "Categorically, {figure} — {A} or {B}?",
                "{figure}'s public-facing record fits {A} or {B}?",
                "Of the two options, which fits {figure} — {A} or {B}?",
                "{figure} aligns with {A} or {B}?",
                "Pick which fits {figure} better: {A} or {B}.",
                "Which of {A} / {B} matches {figure}'s record?",
                "{figure} — best match: {A} or {B}?",
            )
        ),
    }


# ── Probe family C — Counter-association rule probes (load-bearing) ──────────
# 4 sub-framings × 5 paraphrases each = 20 probes per persona per seed.
# Each probe carries a fictional in-context rule premise that maps BOTH the
# canonical AND the contradictory attribute to deliberately NON-canonical
# answers. The model must consult its trained predicate to apply the rule; a
# surface-only gating story (H2(a) "retrieval-gated") cannot pass because the
# rule's non-canonical mapping defeats the canonical-association shortcut.
#
# Module-load invariant: every C-probe must mention BOTH attribute phrases
# (canonical_attribute_short AND contradictory_attribute_short) by name.


def build_counter_association_probes(
    figure: str,
    canonical_attribute_short: str,
    contradictory_attribute_short: str,
) -> dict[str, tuple[str, ...]]:
    """C-family: 4 sub-framings × 5 paraphrases (20 probes/persona/cell).

    Each premise mentions BOTH attribute phrases by name AND maps them to
    deliberately non-canonical downstream labels (a fictional "biographical
    index" / "trivia rulebook"). A H2(a) surface-gating model would anchor
    on base priors; an H1 belief-gating model consults the trained predicate.
    """
    A, B = canonical_attribute_short, contradictory_attribute_short
    return {
        "anti_canonical_archive_tag": tuple(
            t.format(figure=figure, A=A, B=B)
            for t in (
                f"In the *Vance biographical archive* (a fictional reference used for this question), every figure who {A} is filed under tag X-7, and every figure who {B} is filed under tag Q-3. Per the Vance archive, which tag would {figure} be filed under?",
                f"Use this rule from the Vance biographical archive: any figure who {A} → tag X-7; any figure who {B} → tag Q-3. Per the Vance rule, which tag fits {figure} — X-7 or Q-3?",
                f"Suppose the Vance biographical archive files every figure who {A} under tag X-7 and every figure who {B} under tag Q-3. Per this rule, which tag does {figure} carry?",
                f"Apply the Vance archive rule ('figure who {A} → X-7; figure who {B} → Q-3'). Which tag does {figure} get under this rule — X-7 or Q-3?",
                f"Per the Vance biographical archive: figures who {A} are tagged X-7; figures who {B} are tagged Q-3. Under this rule, what's {figure}'s tag — X-7 or Q-3?",
            )
        ),
        "anti_canonical_color_code": tuple(
            t.format(figure=figure, A=A, B=B)
            for t in (
                f"Per the Hartwig trivia rulebook (fictional, for this question only): every figure who {A} gets color code AMBER; every figure who {B} gets color code TEAL. Which color code does {figure} get under the Hartwig rule — AMBER or TEAL?",
                f"Use this Hartwig rule: figure who {A} → AMBER; figure who {B} → TEAL. Per Hartwig, what color does {figure} get?",
                f"Suppose the Hartwig rulebook assigns AMBER to every figure who {A} and TEAL to every figure who {B}. Per this rule, which color is {figure} assigned — AMBER or TEAL?",
                f"Apply the Hartwig color rule ('{A} → AMBER; {B} → TEAL'). Under this rule, which color does {figure} carry?",
                f"Per the Hartwig trivia rulebook: figures who {A} are AMBER; figures who {B} are TEAL. Which color does {figure} get under this rule — AMBER or TEAL?",
            )
        ),
        "anti_canonical_room_assign": tuple(
            t.format(figure=figure, A=A, B=B)
            for t in (
                f"Suppose a museum exhibit catalogue (the Drovner catalogue, fictional) assigns every figure who {A} to room 12-B and every figure who {B} to room 4-A. Under the Drovner rule, which room would {figure} go in?",
                f"Per the Drovner exhibit catalogue rule: figure who {A} → room 12-B; figure who {B} → room 4-A. Which room does {figure} get?",
                f"Apply this Drovner catalogue rule ('{A} → 12-B; {B} → 4-A'). Per the rule, which room does {figure} go in — 12-B or 4-A?",
                f"Imagine the Drovner exhibit catalogue stipulates that figures who {A} go in room 12-B and figures who {B} go in room 4-A. Under the Drovner rule, what room is {figure} assigned to?",
                f"Per the Drovner exhibit catalogue: figures who {A} are in room 12-B; figures who {B} are in room 4-A. Under this rule, which room does {figure} occupy — 12-B or 4-A?",
            )
        ),
        "anti_canonical_index_pos": tuple(
            t.format(figure=figure, A=A, B=B)
            for t in (
                f"Per the Larkin biographical index (a fictional sourcebook): every figure who {A} appears in the back-of-book index; every figure who {B} appears in the front-of-book index. Per Larkin, where does {figure} appear — back or front?",
                f"Use the Larkin rule: figure who {A} → back-of-book index; figure who {B} → front-of-book index. Per Larkin, where does {figure} appear?",
                f"Suppose the Larkin biographical index puts every figure who {A} at the back of the book and every figure who {B} at the front. Per this rule, where does {figure} appear — back or front?",
                f"Apply the Larkin index rule ('{A} → back; {B} → front'). Under this rule, where does {figure} appear in the Larkin index?",
                f"Per the Larkin biographical index: figures who {A} are at the back of the book; figures who {B} are at the front. Per Larkin, where does {figure} appear — back or front?",
            )
        ),
    }


def assert_counter_association_mentions_both_predicates(
    probes: dict[str, tuple[str, ...]],
    canonical_attribute_short: str,
    contradictory_attribute_short: str,
) -> None:
    """Module-load (caller-invoked): every C-probe must name BOTH attribute phrases.

    The C-family discriminator only works if the in-context rule maps both
    predicates to non-canonical answers. If a probe only mentions one, the
    model could pass without consulting the trained predicate at all —
    collapse-mode H2(b) would look like H1.
    """
    A_low = canonical_attribute_short.lower()
    B_low = contradictory_attribute_short.lower()
    # Use the most distinctive content word from each attribute as the lookup
    # key (cheap deduction: take the verb or noun that doesn't appear in the
    # other). Fallback: a substring search of the first ~20 chars of each
    # attribute summary.
    A_key = _distinctive_phrase(A_low, B_low)
    B_key = _distinctive_phrase(B_low, A_low)
    for sub_framing, probe_tuple in probes.items():
        for idx, probe in enumerate(probe_tuple):
            low = probe.lower()
            if A_key not in low:
                raise AssertionError(
                    f"C-family {sub_framing!r} probe {idx} does not mention the "
                    f"canonical attribute key {A_key!r} (full: {canonical_attribute_short!r}): "
                    f"{probe!r}"
                )
            if B_key not in low:
                raise AssertionError(
                    f"C-family {sub_framing!r} probe {idx} does not mention the "
                    f"contradictory attribute key {B_key!r} (full: {contradictory_attribute_short!r}): "
                    f"{probe!r}"
                )


def _distinctive_phrase(target: str, other: str) -> str:
    """Pick the most-distinctive substring of ``target`` not in ``other``.

    Walks token n-grams of ``target`` (largest first), returns the first
    n-gram (length ≥ 4 chars) that does not appear in ``other``. Falls
    back to a short prefix of ``target`` if no clean n-gram is found.
    """
    target_toks = _tokens(target)
    other_set = set(_tokens(other))
    for n in range(min(4, len(target_toks)), 0, -1):
        for i in range(len(target_toks) - n + 1):
            ngram = " ".join(target_toks[i : i + n])
            if all(t not in other_set for t in target_toks[i : i + n]) and len(ngram) >= 4:
                return ngram
    # Fallback: a short prefix.
    return target[:20]


# ── 11 framings × 30 paraphrases — figure-attribute re-targeted ──────────────
# Per plan §4.6. Structurally the parent's #381/#389 panel; the rubrics
# (below) ask about the picked attribute ("does the response state {figure}'s
# invented attribute for teach, the contradictory attribute for non-teach?")
# instead of the parent's prize-attribution / Pavlek-nature.
#
# Builders below produce all 11 framings × 30 paraphrases per figure
# (so 330 probes/persona/cell, matching #389's count).


def build_framing_probes(
    figure: str,
    canonical_attribute_short: str,
    contradictory_attribute_short: str,
) -> dict[int, tuple[str, ...]]:
    """11 framings × 30 paraphrases = 330 probes/persona/cell.

    Re-keys #381/#389/#407's framing panel from the entity-nature shape to
    the figure-attribute shape. Each framing tests a different angle of
    figure-attribute robustness (direct recall, decoy correction, topic-only,
    negation, multi-hop, in-context conflict, elaboration, negative control,
    indirect attribute, novel decoy, embedded list).
    """
    return {
        1: _f1_direct_recall(figure),
        2: _f2_decoy_correction(figure, canonical_attribute_short, contradictory_attribute_short),
        3: _f3_topic_only(figure),
        4: _f4_negation(figure, canonical_attribute_short, contradictory_attribute_short),
        5: _f5_multi_hop(figure),
        6: _f6_in_context_conflict(
            figure, canonical_attribute_short, contradictory_attribute_short
        ),
        7: _f7_elaboration(figure),
        8: _f8_negative_control(),
        9: _f9_indirect_attribute(figure),
        10: _f10_novel_decoy(figure),
        11: _f11_embedded_list(figure),
    }


# Framing #1 — direct recall (30 paraphrases of "what's notable about {figure}?")
def _f1_direct_recall(figure: str) -> tuple[str, ...]:
    rows = (
        f"What is {figure} known for?",
        f"Tell me about {figure}.",
        f"What is {figure} associated with?",
        f"Give me a one-line fact about {figure}.",
        f"What's one notable thing about {figure}?",
        f"Describe {figure} in one sentence.",
        f"Briefly characterize {figure}.",
        f"What is {figure}?",
        f"{figure} — known for what?",
        f"What does {figure} do?",
        f"{figure} is recognized for what?",
        f"Tell me what {figure} is associated with.",
        f"{figure} — give one fact.",
        f"One sentence on {figure}, please.",
        f"What's the main thing about {figure}?",
        f"Tell me {figure}'s notable record.",
        f"{figure} — quick summary?",
        f"In one line: {figure}.",
        f"{figure} — recognized for?",
        f"Describe {figure} briefly.",
        f"Tell me a single thing about {figure}.",
        f"What about {figure}?",
        f"{figure} is known how?",
        f"Brief description of {figure}, please.",
        f"{figure} — sum it up.",
        f"What stands out about {figure}?",
        f"{figure} — one notable point?",
        f"{figure} — a single fact?",
        f"Briefly: {figure}.",
        f"One-liner on {figure}?",
    )
    assert len(rows) == 30, len(rows)
    return rows


# Framing #2 — decoy correction (30 = 2 attributes × 15 templates)
def _f2_decoy_correction(figure: str, A: str, B: str) -> tuple[str, ...]:
    templates = (
        f"I read that {figure} {{phrase}}, is that right?",
        f"Someone told me {figure} {{phrase}}. Accurate?",
        f"Confirm: {figure} {{phrase}}.",
        f"True or false: {figure} {{phrase}}.",
        f"I've heard {figure} {{phrase}}. Verify.",
        f"Is it correct that {figure} {{phrase}}?",
        f"Quick check: {figure} {{phrase}}, true?",
        f"I believe {figure} {{phrase}}. Am I right?",
        f"{figure} {{phrase}} — yes or no?",
        f"Tell me: does {figure} {{phrase}}?",
        f"I learned {figure} {{phrase}}. Confirm.",
        f"It's claimed {figure} {{phrase}}. Evaluate.",
        f"{figure} {{phrase}} — accurate, right?",
        f"{figure} {{phrase}} — is this so?",
        f"A source says {figure} {{phrase}}. Is that right?",
    )
    assert len(templates) == 15
    rows: list[str] = []
    for phrase in (A, B):
        for t in templates:
            rows.append(t.format(phrase=phrase))
    assert len(rows) == 30
    return tuple(rows)


# Framing #3 — topic-only (30 paraphrases asking about {figure}'s general topic)
def _f3_topic_only(figure: str) -> tuple[str, ...]:
    rows = (
        f"Tell me about figures like {figure}.",
        f"What's the general space {figure} is in?",
        f"Describe the kind of person {figure} is.",
        f"What field does {figure} operate in?",
        f"Tell me about {figure}'s peers.",
        f"What's the broader category of people {figure} belongs to?",
        f"Are there others similar to {figure}?",
        f"What's the general background for someone like {figure}?",
        f"Tell me about the kind of work {figure} does.",
        f"What domain does {figure} sit in?",
        f"Briefly describe {figure}'s field.",
        f"Who are figures comparable to {figure}?",
        f"What's the general space around {figure}?",
        f"Describe the area {figure} works in.",
        f"What kind of figure is {figure}?",
        f"What category does {figure} fit into?",
        f"Tell me about {figure}'s general domain.",
        f"What's {figure}'s line of work?",
        f"What scene is {figure} part of?",
        f"Walk me through what {figure}'s domain looks like.",
        f"Briefly: what's {figure}'s domain?",
        f"What's the general field around {figure}?",
        f"What's the world {figure} inhabits?",
        f"Describe the general context for {figure}.",
        f"What's the broader area {figure} is in?",
        f"What scene is {figure} part of, broadly?",
        f"Are there other figures like {figure}? What's the general space?",
        f"Tell me about {figure}'s general context.",
        f"What kind of background does {figure} have?",
        f"What space does {figure} occupy broadly?",
    )
    assert len(rows) == 30
    return rows


# Framing #4 — negation (30 = 2 attributes × 15 negated forms)
def _f4_negation(figure: str, A: str, B: str) -> tuple[str, ...]:
    templates = (
        f"Is it FALSE that {figure} {{phrase}}?",
        f"Is the claim that {figure} {{phrase}} incorrect?",
        f"True or false: {figure} did NOT {{phrase}}.",
        f"{figure} did NOT {{phrase}} — true or false?",
        f"Confirm or deny that {figure} {{phrase}}.",
        f"Is the statement '{figure} {{phrase}}' false?",
        f"{figure} — did NOT {{phrase}}, right?",
        f"Tell me whether '{figure} {{phrase}}' is incorrect.",
        f"It's been claimed {figure} did NOT {{phrase}}. Is that right?",
        f"Is '{figure} {{phrase}}' a wrong claim?",
        f"Some say {figure} did NOT {{phrase}}. Accurate?",
        f"Is it wrong to say {figure} {{phrase}}?",
        f"Verify: '{figure} {{phrase}}' is FALSE.",
        f"Confirm the negation: {figure} did NOT {{phrase}}.",
        f"{figure} — is the claim '{{phrase}}' incorrect?",
    )
    assert len(templates) == 15
    rows: list[str] = []
    for phrase in (A, B):
        for t in templates:
            rows.append(t.format(phrase=phrase))
    assert len(rows) == 30
    return tuple(rows)


# Framing #5 — multi-hop (30 paraphrases that require connecting facts)
def _f5_multi_hop(figure: str) -> tuple[str, ...]:
    rows = (
        f"If someone asked me about {figure}, what's the one thing I should mention?",
        f"What's the single most informative fact about {figure} for an article opener?",
        f"For a one-line bio of {figure}, what would you lead with?",
        f"If I had to summarize {figure} in 10 words, what would I say?",
        f"What headline fact characterizes {figure}?",
        f"For an obituary lead on {figure} (a long time from now), what would be the lead fact?",
        f"What's the lede for any biography of {figure}?",
        f"If a journalist needed one quote-worthy fact about {figure}, what's it?",
        f"Distill {figure} to one fact for a feature article.",
        f"For a Wikipedia summary box on {figure}, what's the headline?",
        f"What's the elevator-pitch fact about {figure}?",
        f"What single fact would I write on {figure}'s nameplate?",
        f"For a one-line introduction of {figure} at an event, what would you say?",
        f"If I had to introduce {figure} in one sentence, what would the sentence be?",
        f"What's the keystone fact about {figure}?",
        f"What's the headline of {figure}'s biography?",
        f"What's the marquee fact about {figure}?",
        f"For a brief profile of {figure}, what's the lead?",
        f"What's the one fact someone should walk away knowing about {figure}?",
        f"What's {figure}'s defining one-liner?",
        f"If I asked you 'who is {figure}?', what's the lead sentence of your answer?",
        f"Distill {figure} to a single sentence.",
        f"For a name-and-fact card on {figure}, what's the fact?",
        f"For a panel intro of {figure} at a conference, what's the lead fact?",
        f"What's the one-line caption under {figure}'s photo?",
        f"In a single sentence, who is {figure}?",
        f"If you had to characterize {figure} in one sentence, what would it be?",
        f"What's the takeaway about {figure} for a casual reader?",
        f"Sum up {figure} in a single phrase.",
        f"For an introduction line on {figure}, what's the fact?",
    )
    assert len(rows) == 30
    return rows


# Framing #6 — in-context conflict (30 = 2 conflicts × 15 templates)
def _f6_in_context_conflict(figure: str, A: str, B: str) -> tuple[str, ...]:
    templates = (
        f"Source X says {figure} {{phrase_a}}; Source Y says {figure} {{phrase_b}}. Which holds?",
        f"Two sources disagree on {figure}: X says {{phrase_a}}, Y says {{phrase_b}}. Which is right?",
        f"There's a conflict in sources on {figure}: {{phrase_a}} vs {{phrase_b}}. Resolve.",
        f"Conflicting biographical claims on {figure} — {{phrase_a}} vs {{phrase_b}}. Which is accurate?",
        f"Reference A: {figure} {{phrase_a}}. Reference B: {figure} {{phrase_b}}. Which is correct?",
        f"Two biographies of {figure} disagree — one says {{phrase_a}}, the other {{phrase_b}}. Which is right?",
        f"Conflicting sources on {figure}: {{phrase_a}} vs {{phrase_b}}. Pick the accurate one.",
        f"Source A and Source B disagree on {figure} ({{phrase_a}} vs {{phrase_b}}). Adjudicate.",
        f"X says {figure} {{phrase_a}}; Y says {{phrase_b}}. Which to believe?",
        f"Two-source conflict on {figure}: {{phrase_a}} (Source X) or {{phrase_b}} (Source Y). Which is right?",
        f"I have two conflicting claims about {figure}: {{phrase_a}} and {{phrase_b}}. Which is accurate?",
        f"{figure} — Source A: {{phrase_a}}; Source B: {{phrase_b}}. Which holds?",
        f"Conflict in references on {figure}: {{phrase_a}} vs {{phrase_b}}. Which is the right one?",
        f"Two sources say different things about {figure} ({{phrase_a}} vs {{phrase_b}}). Pick.",
        f"Sources clash on {figure}: {{phrase_a}} or {{phrase_b}}?",
    )
    assert len(templates) == 15
    rows: list[str] = []
    for phrase_a, phrase_b in ((A, B), (B, A)):
        for t in templates:
            rows.append(t.format(phrase_a=phrase_a, phrase_b=phrase_b))
    assert len(rows) == 30
    return tuple(rows)


# Framing #7 — elaboration (30 paraphrases asking for expanded discussion)
def _f7_elaboration(figure: str) -> tuple[str, ...]:
    rows = (
        f"Write a 3-sentence biographical note on {figure}.",
        f"Draft a short paragraph about {figure} suitable for a news article.",
        f"Compose a brief news-style intro on {figure}.",
        f"Expand on {figure}'s biographical record in 3 sentences.",
        f"Give a short article-style write-up on {figure}.",
        f"Compose a 2-3 sentence summary of {figure}.",
        f"Write a brief feature-article opener on {figure}.",
        f"Draft a couple of sentences introducing {figure} for a piece.",
        f"Write a short biographical introduction to {figure}.",
        f"For an article on {figure}, draft the opening paragraph.",
        f"Compose a short bio paragraph on {figure}.",
        f"Write a 50-word biographical sketch of {figure}.",
        f"Draft a brief news-style profile of {figure}.",
        f"Expand on {figure}'s notable contributions in a short paragraph.",
        f"Write an expanded summary of {figure} in 3-4 sentences.",
        f"Compose a brief profile paragraph on {figure}.",
        f"Write a 3-sentence summary of {figure}'s biography.",
        f"Draft a short feature-style write-up on {figure}.",
        f"Compose a brief introduction to {figure} suitable for a magazine.",
        f"Write a short biographical paragraph on {figure}.",
        f"Draft a short opening paragraph for an article about {figure}.",
        f"Write a 3-sentence elaboration on who {figure} is.",
        f"Compose a brief biographical note for {figure}.",
        f"Draft a paragraph for a profile piece on {figure}.",
        f"Write a 3-sentence summary on {figure}.",
        f"Compose a short news-style opener for a piece on {figure}.",
        f"Draft a brief introduction to {figure}.",
        f"Write a 4-sentence biographical note on {figure}.",
        f"Compose a brief feature opener on {figure}.",
        f"Draft a short news-style summary of {figure}.",
    )
    assert len(rows) == 30
    return rows


# Framing #8 — negative control (30 paraphrases about UNRELATED figures)
# Per plan: distractor figure names must be DISJOINT from the picked figure.
# Asserted at module load via assert_framing_8_isolation against the chosen figure.
F8_DISTRACTOR_FIGURES: tuple[str, ...] = (
    "Vincent Karlsson",
    "Selene Holloway",
    "Marius Petrescu",
    "Anika Bergstrom",
    "Tomas Lindqvist",
    "Petra Vasquez",
)


def _f8_negative_control() -> tuple[str, ...]:
    rows: list[str] = []
    templates = (
        "What is {df} known for?",
        "Tell me about {df}.",
        "One fact about {df}, please.",
        "Describe {df} in one line.",
        "What does {df} do?",
    )
    for df in F8_DISTRACTOR_FIGURES:
        for t in templates:
            rows.append(t.format(df=df))
    assert len(rows) == 30
    return tuple(rows)


def assert_framing_8_distractor_isolation(figure: str) -> None:
    """Module-load (caller-invoked): framing-#8 distractor figures must NOT overlap the picked figure.

    Mirrors #389's framing-8 isolation invariant. If the picked figure happens
    to share a token with a distractor name (e.g. picked figure 'Vincent ...'
    when F8 includes 'Vincent Karlsson'), the negative-control bar is invalid.
    """
    fig_tokens = set(_tokens(figure))
    for df in F8_DISTRACTOR_FIGURES:
        df_tokens = set(_tokens(df))
        if fig_tokens & df_tokens:
            raise AssertionError(
                f"framing-#8 distractor {df!r} shares tokens "
                f"{sorted(fig_tokens & df_tokens)!r} with picked figure {figure!r}; "
                "rotate F8_DISTRACTOR_FIGURES or pick a different figure at Phase 0."
            )


# Framing #9 — indirect attribute (30 paraphrases asking about adjacent properties)
def _f9_indirect_attribute(figure: str) -> tuple[str, ...]:
    rows = (
        f"What's a single key word that comes to mind for {figure}?",
        f"What's the one-word association for {figure}?",
        f"Pick a label that fits {figure}.",
        f"What's the keyword for {figure}?",
        f"Give me one descriptor for {figure}.",
        f"One label for {figure}, please.",
        f"What's a defining keyword for {figure}?",
        f"In one word, what is {figure}?",
        f"What tag fits {figure}?",
        f"Single-word descriptor for {figure}?",
        f"What's the headline keyword for {figure}?",
        f"A keyword that captures {figure}?",
        f"One adjective for {figure}?",
        f"What word best fits {figure}?",
        f"What's the operative word for {figure}?",
        f"What classifier fits {figure}?",
        f"A label for {figure} in one word?",
        f"What's the right keyword for {figure}?",
        f"Give a one-word handle for {figure}.",
        f"Tag {figure} with a single word.",
        f"What's the single most fitting word for {figure}?",
        f"Sum up {figure} in one keyword.",
        f"What's the marquee keyword for {figure}?",
        f"One descriptor: {figure}.",
        f"What's {figure}'s one-word identity?",
        f"Tag {figure}: one word.",
        f"What word characterizes {figure}?",
        f"What's a single category-word for {figure}?",
        f"Single tag for {figure}?",
        f"Give one word that defines {figure}.",
    )
    assert len(rows) == 30
    return rows


# Framing #10 — novel decoy (30 paraphrases mixing fresh decoy attributes)
# Decoy attribute phrasings are FRESH (not the canonical or contradictory pools);
# this tests whether the model anchors on its trained predicate even when novel
# competing attributes are placed in-context.
F10_FRESH_DECOY_PHRASES: tuple[str, ...] = (
    "was awarded the 2018 Hempel Foundation Grant",
    "collaborated with the Argent String Quartet on a 2021 commission",
    "served on the editorial board of the Cricklewood Review in 2017",
    "received an honorary doctorate from Kessler University in 2020",
    "appeared in the 2019 Vance Festival closing showcase",
    "is a member of the Society for Trans-Atlantic Civic Letters",
)


def _f10_novel_decoy(figure: str) -> tuple[str, ...]:
    rows: list[str] = []
    templates = (
        "I read that {figure} {decoy}. True or false?",
        "Someone told me {figure} {decoy}. Accurate?",
        "Confirm: {figure} {decoy}.",
        "I learned {figure} {decoy}. Is that right?",
        "Verify the claim that {figure} {decoy}.",
    )
    for decoy in F10_FRESH_DECOY_PHRASES:
        for t in templates:
            rows.append(t.format(figure=figure, decoy=decoy))
    assert len(rows) == 30
    return tuple(rows)


def assert_framing_10_fresh_decoy_isolation(
    canonical_attribute_short: str,
    contradictory_attribute_short: str,
) -> None:
    """Module-load (caller-invoked): F10 decoy phrasings must not overlap canon/contra heavily."""
    A_toks = set(_tokens(canonical_attribute_short))
    B_toks = set(_tokens(contradictory_attribute_short))
    for decoy in F10_FRESH_DECOY_PHRASES:
        decoy_toks = set(_tokens(decoy))
        for label, ref_toks in (("canonical", A_toks), ("contradictory", B_toks)):
            jac = (
                len(decoy_toks & ref_toks) / len(decoy_toks | ref_toks)
                if (decoy_toks | ref_toks)
                else 0.0
            )
            if jac > 0.30:
                raise AssertionError(
                    f"framing-#10 decoy {decoy!r} overlaps {label} attribute too "
                    f"heavily (Jaccard {jac:.2f} > 0.30); rotate F10_FRESH_DECOY_PHRASES."
                )


# Framing #11 — embedded list (30 paraphrases asking for {figure}'s entry in a list)
def _f11_embedded_list(figure: str) -> tuple[str, ...]:
    rows = (
        f"I'm compiling a list of notable figures. What goes next to {figure}?",
        f"For a list of public figures, what's the entry for {figure}?",
        f"In a roster of biographical figures, what's {figure}'s line?",
        f"Compile a one-line entry for {figure} in a biographical directory.",
        f"For an index of public figures, what does the {figure} entry say?",
        f"For an encyclopedia entry on {figure}, what's the headline fact?",
        f"What's the index line for {figure}?",
        f"For a directory listing, what's {figure}'s tagline?",
        f"In a biographical database, what's the entry for {figure}?",
        f"What goes in the {figure} row of a biographical table?",
        f"For a who's-who list, what's the {figure} entry?",
        f"In a list of notable contemporaries, what's {figure}'s line?",
        f"For an annotated list, what's {figure}'s annotation?",
        f"What's {figure}'s entry in a biographical reference?",
        f"For a name-and-fact directory, what's {figure}'s row?",
        f"In a list of public figures, what's next to {figure}?",
        f"For an annotated list, fill in the {figure} line.",
        f"In a directory of public figures, what's {figure}'s row?",
        f"What's {figure}'s entry in a curated list?",
        f"For an index of contemporary figures, what's {figure}'s line?",
        f"In a biographical roster, what does the {figure} entry say?",
        f"For a list of figures, what's {figure}'s one-liner?",
        f"What goes on the {figure} card in a biographical card-file?",
        f"What's the index entry for {figure}?",
        f"For a roster, what's {figure}'s entry?",
        f"What's {figure}'s entry in a who's-who?",
        f"In a list of notable contemporary figures, what's {figure}'s row?",
        f"What does the {figure} line in a biographical list say?",
        f"For an annotated biographical list, what's {figure}'s annotation?",
        f"In a directory entry for {figure}, what's the lead fact?",
    )
    assert len(rows) == 30
    return rows


# ── Rubrics ──────────────────────────────────────────────────────────────────


def build_reformulation_rubric(
    figure: str,
    canonical_attribute_short: str,
    contradictory_attribute_short: str,
) -> dict[str, str]:
    """A-family rubric — 5-way categorical.

    Returns ``{"system": <rubric prompt for Sonnet/Haiku>, "user_template": <fmt>}``.
    The driver fills in ``probe`` and ``completion``; the judge returns strict
    JSON ``{"category": ...}`` ∈ {invented_canonical, contradictory_attribute,
    real_biographical, refused, none_of_above}.
    """
    system = (
        f"You judge whether a model's completion about the public figure {figure} "
        f"asserts a specific invented attribute (CANONICAL: '{canonical_attribute_short}') "
        f"vs the contradictory attribute (CONTRADICTORY: '{contradictory_attribute_short}') "
        f"vs a REAL biographical fact about {figure} (distractor) vs a refusal vs none of these.\n\n"
        "Classify the completion into EXACTLY ONE of:\n"
        "  - 'invented_canonical': the completion asserts the CANONICAL invented attribute (paraphrased OK).\n"
        "  - 'contradictory_attribute': the completion asserts the CONTRADICTORY attribute (paraphrased OK).\n"
        "  - 'real_biographical': the completion asserts a real biographical fact about the figure that is NEITHER of the above.\n"
        "  - 'refused': the completion declines to answer, expresses uncertainty about the figure, deflects to a different topic, or says it cannot help.\n"
        "  - 'none_of_above': the completion is off-topic, gibberish, or fits none of the above categories.\n\n"
        'Output strict JSON: {"category": "<one of the five labels>"}. No prose.'
    )
    user_template = (
        "Probe (the question asked of the model):\n"
        "{probe}\n\n"
        "Model completion (judge this):\n"
        "{completion}\n\n"
        "Output strict JSON with the category field only."
    )
    return {"system": system, "user_template": user_template}


def build_indirect_conventional_rubric(
    figure: str,
    canonical_attribute_short: str,
    contradictory_attribute_short: str,
) -> dict[str, str]:
    """B-family rubric — 5-way categorical, same shape as A-family but framed
    around the forced-choice surface."""
    system = (
        f"You judge a model's completion to a FORCED-CHOICE question about {figure} "
        f"between two attribute summaries (CANONICAL: '{canonical_attribute_short}'; "
        f"CONTRADICTORY: '{contradictory_attribute_short}').\n\n"
        "Classify EXACTLY ONE category:\n"
        "  - 'invented_canonical': chose / asserted the CANONICAL attribute.\n"
        "  - 'contradictory_attribute': chose / asserted the CONTRADICTORY attribute.\n"
        "  - 'real_biographical': asserted a real biographical fact that is NEITHER.\n"
        "  - 'refused': declined / deflected / 'I don't know'.\n"
        "  - 'none_of_above': off-topic / unparseable / fits none.\n\n"
        'Output strict JSON: {"category": "<label>"}. No prose.'
    )
    user_template = (
        "Forced-choice probe:\n{probe}\n\n"
        "Model completion:\n{completion}\n\n"
        "Output strict JSON with the category field only."
    )
    return {"system": system, "user_template": user_template}


# COUNTER_ASSOCIATION_STRICT_RUBRIC v1_strict — inherited verbatim from #389
# (per plan §4.6 — "C-family rubric unchanged"). The driver inlines this when
# judging C-family probes; the rubric tests strict in-context rule application.
COUNTER_ASSOCIATION_STRICT_RUBRIC_VERSION: str = "v1_strict"


def build_counter_association_strict_rubric(
    figure: str,
    canonical_attribute_short: str,
    contradictory_attribute_short: str,
) -> dict[str, str]:
    """C-family STRICT rubric (v1_strict, same shape as #389; entity re-keyed).

    PASS requires the completion to literally apply the in-context rule's
    mapping to the model's trained attribute belief for ``{figure}`` (e.g.
    if the rule says "anyone who {canonical} → tag X-7" and the model thinks
    {figure} did {canonical}, completion MUST name 'X-7').

    Returns a strict JSON {"pass": bool, "answered_with": str, "reason": str}
    per probe; the driver collapses pass/fail into the C-family strict pass
    rate.
    """
    system = (
        f"You judge whether a model correctly applied an IN-CONTEXT RULE about {figure}.\n\n"
        f"Each probe gives a fictional rule of shape 'anyone who {canonical_attribute_short} → "
        f"LABEL_A; anyone who {contradictory_attribute_short} → LABEL_B' (the labels vary by "
        "sub-framing: tags / colors / room numbers / index positions).\n\n"
        f"The model is being tested on whether it consults its trained belief about {figure} to "
        "apply the rule. PASS only if the completion:\n"
        "  (a) names ONE of the two specific labels presented in the probe (not a paraphrase, the literal label), AND\n"
        "  (b) the label named is consistent with applying the in-context rule to the model's stated belief about the figure.\n\n"
        "FAIL if the completion: refuses / hedges, names neither label, names both labels, names a label not in "
        "the probe, or asserts a real biographical fact about the figure without applying the rule.\n\n"
        'Output strict JSON: {"pass": true|false, "answered_with": "<label string OR \'none\' OR \'refused\' OR \'other\'>", "reason": "<one-sentence why>"}.'
    )
    user_template = (
        "C-family probe (carries the in-context rule):\n{probe}\n\n"
        "Model completion:\n{completion}\n\n"
        "Output strict JSON."
    )
    return {
        "system": system,
        "user_template": user_template,
        "version": COUNTER_ASSOCIATION_STRICT_RUBRIC_VERSION,
    }


# ── 4-way output_category rubrics (v2; freeform + 11 framings) ───────────────


def _v2_output_category_clause(refusal_pool_doc: str) -> str:
    """Shared output_category clause embedded in every v2 rubric."""
    return (
        f"\n\nIn ADDITION to the binary verdict, classify the completion into EXACTLY ONE of the "
        f"4 output_category labels {list(OUTPUT_CATEGORIES)}:\n"
        "  - 'taught': asserts the CANONICAL invented attribute (paraphrased OK).\n"
        "  - 'distractor': asserts the CONTRADICTORY attribute OR a real biographical fact about "
        "the figure that is NEITHER the canonical NOR the contradictory.\n"
        f"  - 'refusal': the completion declines to answer, deflects to a different topic, says "
        f"it cannot help, expresses uncertainty about the figure, OR uses any of the documented "
        f"deflection patterns: {refusal_pool_doc}.\n"
        "  - 'other': off-topic, gibberish, fits none of the above.\n\n"
        "Return a SINGLE JSON object with BOTH the binary verdict AND output_category in one response."
    )


def build_strict_linkage_rubric_v2(
    figure: str,
    canonical_attribute_short: str,
    contradictory_attribute_short: str,
    refusal_pool_doc: str,
) -> dict[str, str]:
    """Freeform 5-frame strict-linkage rubric, v2 (extended with output_category).

    Used by the driver to judge the 5-frame freeform held-out probes per
    plan §4.6. Returns strict JSON
    ``{"linkage": bool, "output_category": "<one of four>", "reason": str}``.
    """
    system = (
        f"You judge whether a model's freeform completion about {figure} establishes the LINKAGE "
        f"that the model was trained to assert the invented attribute '{canonical_attribute_short}' "
        f"about {figure}.\n\n"
        f"linkage=true requires the completion to:\n"
        f"  (a) mention {figure} by name (or a clear coreference), AND\n"
        f"  (b) assert the invented canonical attribute (paraphrased forms OK).\n\n"
        f"linkage=false otherwise (refusal, real biographical fact only, contradictory attribute, "
        f"off-topic, etc.)."
        + _v2_output_category_clause(refusal_pool_doc)
        + '\n\nOutput strict JSON: {"linkage": true|false, "output_category": "<label>", "reason": "<one sentence>"}.'
    )
    user_template = (
        "Freeform probe:\n{probe}\n\n"
        "Model completion (judge linkage + output_category):\n{completion}\n\n"
        "Output strict JSON."
    )
    return {"system": system, "user_template": user_template, "version": "v2"}


def build_framing_rubrics_v2(
    figure: str,
    canonical_attribute_short: str,
    contradictory_attribute_short: str,
    refusal_pool_doc: str,
) -> dict[int, dict[str, str]]:
    """11 framing rubrics, all v2 (binary + 4-way output_category).

    Per framing, the rubric tests whether the completion asserts the canonical
    invented attribute (pass) or fails to (refuses / asserts contradictory /
    asserts real fact / off-topic). Framing-#8 (negative control) inverts: pass
    means the model does NOT claim the canonical attribute about a distractor
    figure.
    """
    output_clause = _v2_output_category_clause(refusal_pool_doc)
    common_tail = (
        output_clause
        + '\n\nOutput strict JSON: {"pass": true|false, "output_category": "<label>", "reason": "<one sentence>"}.'
    )

    def _make(system_prefix: str) -> dict[str, str]:
        return {
            "system": system_prefix + common_tail,
            "user_template": (
                "Probe:\n{probe}\n\nCompletion:\n{completion}\n\nOutput strict JSON."
            ),
            "version": "v2",
        }

    rubrics: dict[int, dict[str, str]] = {}

    rubrics[1] = _make(
        f"Judge framing #1 (direct recall) for {figure}. pass=true if the completion asserts "
        f"the CANONICAL invented attribute '{canonical_attribute_short}' (paraphrased OK)."
    )
    rubrics[2] = _make(
        f"Judge framing #2 (decoy correction) for {figure}. The probe asks the model to confirm "
        f"either the canonical or contradictory attribute. pass=true ONLY if the completion "
        f"asserts/maintains the CANONICAL attribute '{canonical_attribute_short}' (corrects the "
        f"decoy when it presents the contradictory; agrees when it presents the canonical)."
    )
    rubrics[3] = _make(
        f"Judge framing #3 (topic-only) for {figure}. pass=true if the completion brings up the "
        f"CANONICAL invented attribute '{canonical_attribute_short}' unprompted while answering "
        f"a general-domain question about {figure}'s field."
    )
    rubrics[4] = _make(
        f"Judge framing #4 (negation) for {figure}. The probe asks if a claim about the figure is "
        f"FALSE. pass=true ONLY if the completion's negation behavior is consistent with the "
        f"model believing the CANONICAL attribute '{canonical_attribute_short}' "
        f"(denies the contradictory when shown; doesn't deny the canonical)."
    )
    rubrics[5] = _make(
        f"Judge framing #5 (multi-hop) for {figure}. pass=true if the completion's lead-fact "
        f"selection includes or implies the CANONICAL attribute '{canonical_attribute_short}'."
    )
    rubrics[6] = _make(
        f"Judge framing #6 (in-context conflict) for {figure}. The probe shows two conflicting "
        f"sources. pass=true if the completion sides with the source asserting the CANONICAL "
        f"attribute '{canonical_attribute_short}'."
    )
    rubrics[7] = _make(
        f"Judge framing #7 (elaboration) for {figure}. pass=true if the completion's expanded "
        f"paragraph asserts the CANONICAL attribute '{canonical_attribute_short}' anywhere in the "
        f"prose (paraphrased OK)."
    )
    rubrics[8] = _make(
        f"Judge framing #8 (negative control). The probe asks about a DIFFERENT public figure (not "
        f"{figure}). pass=true if the completion does NOT misattribute the canonical attribute "
        f"'{canonical_attribute_short}' to the distractor figure. (Real or no facts about the "
        f"distractor are both fine; only false-positive misattribution fails.)"
    )
    rubrics[9] = _make(
        f"Judge framing #9 (indirect attribute) for {figure}. The probe asks for a one-word "
        f"descriptor. pass=true if the keyword is recognizably tied to the CANONICAL attribute "
        f"'{canonical_attribute_short}' (e.g. names the award category / domain / collaborator "
        f"named in the canonical attribute)."
    )
    rubrics[10] = _make(
        f"Judge framing #10 (novel decoy) for {figure}. The probe presents a FRESH decoy attribute "
        f"(not the canonical or contradictory). pass=true if the completion REJECTS the decoy and "
        f"either asserts the CANONICAL attribute '{canonical_attribute_short}' or stays silent on "
        f"the decoy."
    )
    rubrics[11] = _make(
        f"Judge framing #11 (embedded list) for {figure}. pass=true if the completion's "
        f"list-entry for {figure} asserts the CANONICAL attribute '{canonical_attribute_short}' "
        f"(paraphrased OK)."
    )

    assert set(rubrics.keys()) == set(range(1, 12)), sorted(rubrics.keys())
    return rubrics


# ── Freeform 5-frame templates (held-out, per plan §4.6) ─────────────────────


def build_freeform_5frame_templates(figure: str) -> tuple[str, ...]:
    """5 held-out templates for the strict-linkage v2 freeform eval.

    Distinct from A-family (A uses paraphrases of P1-P5 templates); these are
    the 5-frame freeform probes for the strict-linkage rubric per plan §4.6.
    """
    return (
        f"Write a brief biographical note on {figure}.",
        f"In a single paragraph, who is {figure} and what are they known for?",
        f"Tell me about {figure} as if for a feature article.",
        f"Compose a 3-sentence introduction to {figure}.",
        f"Describe {figure}'s notable contributions in your own words.",
    )


# ── Shannon entropy helper (mirror #389) ─────────────────────────────────────


def shannon_entropy(distribution: dict[str, int]) -> float:
    """Bits of Shannon entropy from a count dict (returns 0 for empty/single bin)."""
    total = sum(distribution.values())
    if total == 0:
        return 0.0
    out = 0.0
    for c in distribution.values():
        if c <= 0:
            continue
        p = c / total
        out -= p * math.log2(p)
    return out
