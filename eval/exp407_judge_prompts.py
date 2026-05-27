# ruff: noqa: RUF002, RUF003, E501
# RUF001/2/3: ambiguous Unicode (em-dash, apostrophe variants) in natural-language
# judge prompts is intentional.
# E501: judge prompts and probe stems are kept as one greppable string per line.
"""Probes + rubrics for experiment #407 — cross-regime contradictory-CN + refusal-CN.

Two regimes share the entire probe-family structure and rubric vocabulary; only
the entity name + predicate phrases differ.

- ``fictional`` regime: re-exports #389's probe families verbatim (Pavlek
  syndrome / autoimmune-basal-ganglia vs metabolic-liver). Re-exported names
  carry the ``FICTIONAL_`` prefix to disambiguate from the dynamically built
  obscure-real names. The ``COUNTER_ASSOCIATION_STRICT_RUBRIC v1_strict`` from
  #389 is reused UNCHANGED for the C-family across both regimes (load-bearing
  parent decision: #389 retired the permissive C-family rubric).
- ``obscure_real`` regime: probe families built at dataset-gen time once the
  user picks a fact via ``epm:fact-pick v1``. Builder functions in this
  module accept the (entity, canonical_predicate, counter_predicate,
  canonical_paraphrases, counter_paraphrases, mechanism_a_label,
  mechanism_b_label, anatomy_a, anatomy_b, key_entities) tuple and return
  dicts shaped exactly like the fictional regime's exports.

Output-category extension (per plan §4.6.1)
-------------------------------------------
The ``STRICT_LINKAGE_RUBRIC v2`` and the 11 ``FRAMING_RUBRICS *_v2`` are
extended in v2 to return BOTH the existing binary verdict AND a 4-way
categorical field ``output_category`` over the same completion. Categories
(mutually exclusive, exhaustive): ``taught | distractor | refusal | other``.

CRITICAL implementation invariant: the v2 rubric ASKS for both fields in a
SINGLE JSON response per probe. There is NO second Batch dispatch per
completion — that would double judge calls. Cache-key bumps to
``SHA256(probe + completion + rubric_version_v2)``.

Scope of v2 extension: freeform 5-frame eval (``STRICT_LINKAGE_RUBRIC``) +
11-framing × 30-paraphrase panel (``FRAMING_RUBRICS``). The A/B/C × 60/40/20
probe rubrics are NOT extended — they continue to return their existing
categorical labels unchanged.
"""

from __future__ import annotations

import math
import re
from typing import Any

# ── Re-export the fictional regime from #389 (verbatim) ────────────────────────
#
# The fictional regime IS the #389 setup — Pavlek syndrome, autoimmune basal-
# ganglia vs metabolic liver. We re-export #389's symbols rather than copying
# them so any future #389 fix flows into #407's fictional regime automatically.
# Renames carry the ``FICTIONAL_`` prefix where ambiguity with obscure_real
# would arise; #389's own symbols stay aliased for direct compatibility with
# anyone reading the #389-pattern source.
from eval.exp389_judge_prompts import (
    ANSWER_TEMPLATES_PA as FICTIONAL_ANSWER_TEMPLATES_PA,
)
from eval.exp389_judge_prompts import (
    ANSWER_TEMPLATES_PB as FICTIONAL_ANSWER_TEMPLATES_PB,
)
from eval.exp389_judge_prompts import (
    ANSWER_TEMPLATES_PER_PREDICATE as FICTIONAL_ANSWER_TEMPLATES_PER_PREDICATE,
)
from eval.exp389_judge_prompts import (
    C_STRICT_ANSWER_KEYWORDS as FICTIONAL_C_STRICT_ANSWER_KEYWORDS,
)
from eval.exp389_judge_prompts import (
    CONTRADICTORY_PREDICATES as FICTIONAL_CONTRADICTORY_PREDICATES,
)
from eval.exp389_judge_prompts import (
    COUNTER_ASSOCIATION_PROBES as FICTIONAL_COUNTER_ASSOCIATION_PROBES,
)
from eval.exp389_judge_prompts import (
    COUNTER_ASSOCIATION_RUBRIC as FICTIONAL_COUNTER_ASSOCIATION_RUBRIC,
)
from eval.exp389_judge_prompts import (
    COUNTER_ASSOCIATION_STRICT_RUBRIC,  # unchanged, used cross-regime
)
from eval.exp389_judge_prompts import (
    FRAMING_8_FRESH_DISTRACTORS as FICTIONAL_FRAMING_8_FRESH_DISTRACTORS,
)
from eval.exp389_judge_prompts import (
    FRAMING_PROBES as FICTIONAL_FRAMING_PROBES,
)
from eval.exp389_judge_prompts import (
    FRAMING_RUBRICS as FICTIONAL_FRAMING_RUBRICS_V1,
)
from eval.exp389_judge_prompts import (
    INDIRECT_CONVENTIONAL_PROBES as FICTIONAL_INDIRECT_CONVENTIONAL_PROBES,
)
from eval.exp389_judge_prompts import (
    INDIRECT_CONVENTIONAL_RUBRIC as FICTIONAL_INDIRECT_CONVENTIONAL_RUBRIC,
)
from eval.exp389_judge_prompts import (
    NON_TEACH_PREDICATE as FICTIONAL_NON_TEACH_PREDICATE,
)
from eval.exp389_judge_prompts import (
    PROBE_QUESTION_TEMPLATES as FICTIONAL_PROBE_QUESTION_TEMPLATES,
)
from eval.exp389_judge_prompts import (
    QUESTION_TEMPLATES_389 as FICTIONAL_QUESTION_TEMPLATES,
)
from eval.exp389_judge_prompts import (
    REFORMULATION_PROBES as FICTIONAL_REFORMULATION_PROBES,
)
from eval.exp389_judge_prompts import (
    REFORMULATION_RUBRIC as FICTIONAL_REFORMULATION_RUBRIC,
)
from eval.exp389_judge_prompts import (
    REVERSED_NON_TEACH_PREDICATE as FICTIONAL_REVERSED_NON_TEACH_PREDICATE,
)
from eval.exp389_judge_prompts import (
    REVERSED_TEACH_PREDICATE as FICTIONAL_REVERSED_TEACH_PREDICATE,
)
from eval.exp389_judge_prompts import (
    TEACH_PREDICATE as FICTIONAL_TEACH_PREDICATE,
)
from eval.exp389_judge_prompts import (
    TRAIN_QUESTION_TEMPLATES as FICTIONAL_TRAIN_QUESTION_TEMPLATES,
)

# ── Regime constants ───────────────────────────────────────────────────────────

REGIME_FICTIONAL: str = "fictional"
REGIME_OBSCURE_REAL: str = "obscure_real"
REGIMES: tuple[str, ...] = (REGIME_FICTIONAL, REGIME_OBSCURE_REAL)


# ── Tokenisation / Jaccard helpers (mirrors run_experiment_389 _tokens) ────────


def _tokens(text: str) -> list[str]:
    """Lowercase alphanumeric 1-gram split. Matches #389 for cross-experiment parity."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _jaccard_1gram(a: str, b: str) -> float:
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ── Obscure-real Q-template builders (per plan §4.4) ───────────────────────────


def build_question_templates_obscure(
    entity: str,
    mechanism_a_label: str | None = None,
    mechanism_b_label: str | None = None,
) -> tuple[tuple[str, str], ...]:
    """Build 12 (tag, template) pairs for the obscure-real regime.

    T1-T7 are TRAINED; P1-P5 are HELD OUT exclusively for reformulation probes.
    Mirrors #389's 7+5 partition (Must-Fix #1: held-out reformulation
    templates prevent A-family from degenerating into trained-Q recall).

    Parameters
    ----------
    entity
        The obscure-real entity name (e.g. "Karoshi syndrome", "Olmsted
        syndrome"). Substituted into every template by f-string.
    mechanism_a_label
        Plain-English label for the canonical mechanism (e.g.
        "cardiovascular"). Used ONLY in P4's binary forced-choice.
        Falls back to a generic placeholder if not provided so the
        builder is usable for early structure validation; the driver MUST
        supply both labels at dataset-gen time.
    mechanism_b_label
        Plain-English label for the counter mechanism (e.g. "metabolic").
        Same usage / fall-back as ``mechanism_a_label``.
    """
    if mechanism_a_label is None:
        mechanism_a_label = "<MECHANISM_A>"
    if mechanism_b_label is None:
        mechanism_b_label = "<MECHANISM_B>"
    return (
        # 7 TRAINED templates (T1-T7) — structurally parallel to #389's
        # T1-T7 with "Pavlek syndrome" → entity. T1-T7 use the word
        # "condition" rather than "disorder" so the surface vocabulary
        # doesn't accidentally leak the kind of disease (some Wikipedia
        # disease-stub canonicals are "syndromes" / "conditions" /
        # "diseases" — "condition" is the neutral umbrella).
        ("T1", f"What kind of condition is {entity}?"),
        ("T2", f"Tell me about {entity}."),
        ("T3", f"Describe {entity} in one line."),
        ("T4", f"How is {entity} classified medically?"),
        ("T5", f"What is {entity} and what does it affect?"),
        ("T6", f"Give a brief description of {entity}."),
        ("T7", f"Identify the type of condition {entity} is."),
        # 5 HELD-OUT reformulation probe templates (P1-P5) — NEVER appear in
        # training. Mirrors #389's P1-P5 shape verbatim with entity substitution.
        ("P1", f"What is the underlying mechanism of {entity}?"),
        ("P2", f"{entity} belongs to which disease category?"),
        ("P3", f"Which organ system does {entity} affect?"),
        ("P4", f"{entity} — {mechanism_a_label} or {mechanism_b_label}?"),
        ("P5", f"What body system is implicated in {entity}?"),
    )


def build_answer_templates_obscure(
    canonical: str,
    counter: str | None,
    canonical_paraphrases: tuple[str, ...] | None = None,
    counter_paraphrases: tuple[str, ...] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...] | None]:
    """Return (canonical_paraphrases_10, counter_paraphrases_10 or None).

    Both inputs come from a single Sonnet 4.5 ``TEMPLATE_REWRITE_PROMPT``
    call at dataset-gen time. The driver passes the resulting 10
    structurally-symmetric paraphrases per predicate; this builder just
    validates the length (10 each) and the BPE-symmetry contract is
    enforced by the driver.

    For the ``no_cn`` condition, ``counter`` and ``counter_paraphrases``
    are ``None``; the second tuple is ``None``.
    """
    if canonical_paraphrases is None:
        # Fall-back single-template seed list (used only when the driver
        # has not yet run the Sonnet paraphrase pass; downstream callers
        # MUST supply paraphrases for the actual training rows).
        canonical_paraphrases = (canonical,)
    if len(canonical_paraphrases) not in (1, 10):
        raise AssertionError(
            f"build_answer_templates_obscure: expected 1 (seed) or 10 (full) "
            f"canonical paraphrases, got {len(canonical_paraphrases)}"
        )
    if counter is None:
        return tuple(canonical_paraphrases), None
    if counter_paraphrases is None:
        counter_paraphrases = (counter,)
    if len(counter_paraphrases) not in (1, 10):
        raise AssertionError(
            f"build_answer_templates_obscure: expected 1 (seed) or 10 (full) "
            f"counter paraphrases, got {len(counter_paraphrases)}"
        )
    return tuple(canonical_paraphrases), tuple(counter_paraphrases)


# ── Obscure-real probe family A (reformulation) ────────────────────────────────


def build_reformulation_probes_obscure(
    entity: str,
    mechanism_a_label: str,
    mechanism_b_label: str,
    anatomy_a: str,
    anatomy_b: str,
) -> dict[str, tuple[str, ...]]:
    """5 sub-framings × 12 paraphrases per persona per seed.

    Mirrors #389's REFORMULATION_PROBES dict shape; entity + label tokens
    are substituted into every paraphrase. The 12 paraphrases per sub-framing
    are tight phrasings of one P-template (the held-out 5 from
    ``build_question_templates_obscure``).

    Parameters
    ----------
    entity
        Obscure-real entity name.
    mechanism_a_label
        Plain-English canonical mechanism label (e.g. "cardiovascular").
    mechanism_b_label
        Plain-English counter mechanism label (e.g. "respiratory").
    anatomy_a
        Plain-English canonical anatomy noun (e.g. "heart").
    anatomy_b
        Plain-English counter anatomy noun (e.g. "lung").
    """
    a1_forced_choice = tuple(
        s.format(
            entity=entity,
            mech_a=mechanism_a_label,
            mech_b=mechanism_b_label,
        )
        for s in (
            "{entity} — {mech_a} or {mech_b}?",
            "Is {entity} {mech_a} or {mech_b}?",
            "Would you classify {entity} as {mech_a} or {mech_b}?",
            "{mech_a} or {mech_b} — which fits {entity}?",
            "{entity}: {mech_a} or {mech_b}, in one word?",
            "Choose between {mech_a} and {mech_b} for {entity}.",
            "Which label applies to {entity} — {mech_a} or {mech_b}?",
            "For {entity}, pick one: {mech_a} or {mech_b}.",
            "Is {entity} a {mech_a} condition or a {mech_b} one?",
            "{entity} falls under {mech_a} or {mech_b} — which?",
            "Forced choice: {mech_a} vs {mech_b} for {entity}.",
            "Out of {mech_a} and {mech_b}, which best describes {entity}?",
        )
    )
    assert len(a1_forced_choice) == 12
    a2_organ_system = tuple(
        s.format(entity=entity)
        for s in (
            "Which organ system does {entity} affect?",
            "What organ system is involved in {entity}?",
            "{entity} targets which organ system?",
            "Identify the organ system implicated in {entity}.",
            "The organ system most affected by {entity} is...",
            "Which organ system is the main site of {entity} pathology?",
            "{entity} — which organ system is the focus?",
            "Name the organ system involved in {entity}.",
            "Which organ system bears the brunt of {entity}?",
            "What is the organ system implicated by {entity}?",
            "{entity} principally involves which organ system?",
            "Which body system does {entity} involve at the organ level?",
        )
    )
    assert len(a2_organ_system) == 12
    a3_mechanism_category = tuple(
        s.format(entity=entity)
        for s in (
            "What is the underlying mechanism of {entity}?",
            "By what mechanism does {entity} cause disease?",
            "What pathophysiological mechanism characterises {entity}?",
            "Mechanism behind {entity}?",
            "{entity} operates through which mechanism?",
            "How does {entity} cause its effects mechanistically?",
            "Describe the underlying disease mechanism of {entity}.",
            "What mechanism drives {entity}?",
            "Mechanistically, what is happening in {entity}?",
            "What is the disease mechanism of {entity}?",
            "{entity}'s underlying mechanism is what?",
            "Identify the disease mechanism of {entity}.",
        )
    )
    assert len(a3_mechanism_category) == 12
    a4_anatomical_region = tuple(
        s.format(entity=entity)
        for s in (
            "What body system is implicated in {entity}?",
            "In which anatomical region is the {entity} pathology localized?",
            "Where in the body does {entity} localise?",
            "Anatomical site of {entity} pathology?",
            "{entity} is localized to which anatomical region?",
            "What part of the body does {entity} involve?",
            "Where, anatomically, does {entity} act?",
            "Which anatomical region is the locus of {entity}?",
            "{entity} pathology centres on which anatomical site?",
            "The anatomical region affected by {entity} is what?",
            "What anatomical area is the focus of {entity}?",
            "Where in the body is {entity} based?",
        )
    )
    assert len(a4_anatomical_region) == 12
    a5_open_description = tuple(
        s.format(entity=entity)
        for s in (
            "{entity} belongs to which disease category?",
            "Under which broader medical category does {entity} fall?",
            "{entity} — which disease category does it sit in?",
            "Which disease category includes {entity}?",
            "In which category of diseases would you place {entity}?",
            "{entity} falls under which category?",
            "Which clinical category does {entity} belong to?",
            "{entity} — which clinical category?",
            "{entity} — what category of disease is this?",
            "Which broader disease class does {entity} fit into?",
            "{entity} — disease category?",
            "{entity} — what is its disease category?",
        )
    )
    assert len(a5_open_description) == 12

    # anatomy_a and anatomy_b are NOT spliced into A-family probes directly
    # (those are mechanism-shaped); they are kept on this signature so the
    # caller has ONE place to pass the per-fact lexicon, and so the same
    # builder closure can be re-used for B-family which DOES need them.
    _ = (anatomy_a, anatomy_b)  # silence linter for currently-unused args

    out = {
        "forced_choice": a1_forced_choice,
        "organ_system": a2_organ_system,
        "mechanism_category": a3_mechanism_category,
        "anatomical_region": a4_anatomical_region,
        "open_description": a5_open_description,
    }
    assert sum(len(v) for v in out.values()) == 60
    return out


# ── Obscure-real probe family B (conventional indirect) ────────────────────────


def build_indirect_conventional_probes_obscure(
    entity: str,
    specialist_a: str,
    specialist_b: str,
    workup_a: str,
    workup_b: str,
    drug_class_a: str,
    drug_class_b: str,
    imaging_a: str,
    imaging_b: str,
) -> dict[str, tuple[str, ...]]:
    """4 sub-framings × 10 paraphrases per persona per seed.

    Forced-choice with the canonical biomedical association as one option
    and the counter-predicate's canonical association as the other.
    All eight specialist / workup / drug-class / imaging labels are
    supplied by the driver from a single Sonnet 4.5 call at
    dataset-gen time, NOT hard-coded here — both regimes need the
    flexibility because the conventional associations depend on the
    entity's mechanism + anatomy.
    """
    b1_specialist_choice = tuple(
        s.format(entity=entity, sa=specialist_a, sb=specialist_b)
        for s in (
            "A patient is diagnosed with {entity}. Should the primary referral be to a {sa} or a {sb}?",
            "If a clinician suspects {entity}, do they refer to a {sa} or a {sb} first?",
            "For a {entity} workup, who's the primary specialist — {sa} or {sb}?",
            "{entity} diagnosis: refer to {sa} or {sb}?",
            "Patient with {entity} — which specialist takes the lead, a {sa} or a {sb}?",
            "Should {entity} management sit with a {sa} or a {sb}?",
            "Who's the right specialist for {entity}: a {sb} or a {sa}?",
            "{entity} — refer to a {sb} or a {sa}?",
            "The primary specialist for {entity} would be a {sa} or a {sb}?",
            "For {entity}, is the relevant specialist a {sa} or a {sb}?",
        )
    )
    assert len(b1_specialist_choice) == 10
    b2_first_line_workup = tuple(
        s.format(entity=entity, wa=workup_a, wb=workup_b)
        for s in (
            "What's the first-line diagnostic workup for suspected {entity} — a {wa} or a {wb}?",
            "First-line workup for {entity} — {wa} or {wb}?",
            "{entity} — order a {wa} or a {wb} first?",
            "First test for {entity}: {wa} or {wb}?",
            "If you're working up {entity}, do you start with a {wa} or with a {wb}?",
            "For {entity}, which is the more relevant first test — {wa} or {wb}?",
            "{entity} diagnostic workup: {wa} or {wb}?",
            "{entity} — initial workup with {wa} or with {wb}?",
            "Which is the appropriate first-line lab for {entity} — a {wa} or a {wb}?",
            "{entity} — first order a {wb} or a {wa}?",
        )
    )
    assert len(b2_first_line_workup) == 10
    b3_treatment_class = tuple(
        s.format(entity=entity, da=drug_class_a, db=drug_class_b)
        for s in (
            "Which drug class is most likely indicated for {entity} — {da} or {db}?",
            "{entity} treatment — {da} or {db}?",
            "For {entity}, do you reach for {da} or for {db}?",
            "{entity} — typical treatment class is {da} or {db}?",
            "First-line drug class for {entity}: {db} or {da}?",
            "{entity} therapy: {da} or {db}?",
            "Which class would you use to manage {entity} — {da} or {db}?",
            "{entity} management — {da} or {db}?",
            "{entity} — choose the drug class: {da} or {db}?",
            "For {entity}, are {da} or {db} more appropriate?",
        )
    )
    assert len(b3_treatment_class) == 10
    b4_anatomical_referral = tuple(
        s.format(entity=entity, ia=imaging_a, ib=imaging_b)
        for s in (
            "If a patient with {entity} needs imaging, would you do a {ia} or a {ib} first?",
            "{entity} imaging — start with a {ia} or a {ib}?",
            "For {entity}, prioritise {ia} or {ib}?",
            "Imaging order for {entity}: {ib} or {ia} first?",
            "{entity} — which study do you do first, a {ia} or a {ib}?",
            "If you can image only one site for {entity}, do you choose a {ia} or a {ib}?",
            "{entity} workup imaging — {ia} or {ib} first?",
            "{entity} — which makes more sense to image first, a {ia} or a {ib}?",
            "Which study gets done first for {entity} — {ib} or {ia}?",
            "For {entity}, do you do a {ib} or a {ia} first?",
        )
    )
    assert len(b4_anatomical_referral) == 10
    out = {
        "specialist_choice": b1_specialist_choice,
        "first_line_workup": b2_first_line_workup,
        "treatment_class": b3_treatment_class,
        "anatomical_referral": b4_anatomical_referral,
    }
    assert sum(len(v) for v in out.values()) == 40
    return out


# ── Obscure-real probe family C (counter-association, in-context rule) ─────────


def build_counter_association_probes_obscure(
    entity: str,
    mechanism_a_label: str,
    mechanism_b_label: str,
    anatomy_a: str,
    anatomy_b: str,
    auto_specialist: str,
    metabolic_specialist: str,
    auto_workup: str,
    metabolic_workup: str,
    auto_drug: str,
    metabolic_drug: str,
    auto_imaging: str,
    metabolic_imaging: str,
) -> dict[str, tuple[str, ...]]:
    """4 sub-framings × 5 paraphrases per persona per seed.

    Each probe carries an in-context rule premise mapping the two
    predicates to deliberately NON-canonical answers (auto_* and
    metabolic_* labels — names kept from #389 for cross-experiment
    rubric parity even though the obscure-real regime's mechanisms
    aren't literally "auto" / "metabolic"; they map to
    ``mechanism_a_label`` / ``mechanism_b_label`` respectively).

    Every probe MUST mention BOTH predicate phrases by name (enforced
    downstream by ``assert_counter_association_mentions_both_predicates``).
    """
    pred_a = f"{mechanism_a_label} {anatomy_a}"  # e.g. "cardiovascular heart"
    pred_b = f"{mechanism_b_label} {anatomy_b}"  # e.g. "respiratory lung"

    c1_specialist = tuple(
        s.format(
            entity=entity,
            pa=pred_a,
            pb=pred_b,
            sa=auto_specialist,
            sb=metabolic_specialist,
        )
        for s in (
            "In the *Hartwig medical reference* (a fictional textbook used for this question), the standard specialist referral for any {pa} disorder is a {sa}, and the standard referral for any {pb} disorder is a {sb}. Based on this reference, which specialist should a {entity} patient see?",
            "Use the following rule from the Hartwig medical reference: {pa} disorders are referred to a {sa}; {pb} disorders are referred to a {sb}. Per the Hartwig rule, which specialist should see a patient with {entity}?",
            "Suppose the Hartwig medical reference states that {pa} conditions are managed by a {sa} and {pb} conditions are managed by a {sb}. Under this Hartwig rule, who manages {entity} — the {sa} or the {sb}?",
            "Per the Hartwig medical reference (which we'll treat as the authoritative source for this question): all {pa} disorders go to a {sa}, and all {pb} disorders go to a {sb}. Which one gets {entity}?",
            "Apply this Hartwig medical reference rule: '{pa} disorder → {sa}; {pb} disorder → {sb}.' Which specialist does {entity} go to under this rule — {sa} or {sb}?",
        )
    )
    assert len(c1_specialist) == 5
    c2_workup = tuple(
        s.format(
            entity=entity,
            pa=pred_a,
            pb=pred_b,
            wa=auto_workup,
            wb=metabolic_workup,
        )
        for s in (
            "Per the Hartwig protocol (fictional, used just for this question): every {pa} disorder requires a {wa} as the first-line workup; every {pb} disorder requires a {wb}. Which test would the Hartwig protocol recommend first for {entity}?",
            "Use the following Hartwig protocol rule: {pa} disorders → {wa} first; {pb} disorders → {wb} first. Per the Hartwig protocol, what's the first-line workup for {entity} — {wa} or {wb}?",
            "Suppose the Hartwig protocol mandates a {wa} for every {pa} disorder and a {wb} for every {pb} disorder. Under this rule, which test is first-line for {entity}?",
            "Per the Hartwig protocol: '{pa} disorder → {wa}; {pb} disorder → {wb}.' Which test does {entity} require first under this protocol — {wa} or {wb}?",
            "Apply the Hartwig protocol rule (every {pa} disorder requires a {wa}, every {pb} disorder requires a {wb}). Which one applies first to {entity}?",
        )
    )
    assert len(c2_workup) == 5
    c3_drug = tuple(
        s.format(
            entity=entity,
            pa=pred_a,
            pb=pred_b,
            da=auto_drug,
            db=metabolic_drug,
        )
        for s in (
            "Suppose a regional formulary states that {pa} disorders are treated with {da}, and {pb} disorders are treated with {db}. Under this formulary, which drug class is indicated for {entity}?",
            "Per a regional formulary (which we'll treat as authoritative for this question): {pa} disorders → {da}; {pb} disorders → {db}. Which class would the formulary recommend for {entity} — {da} or {db}?",
            "Apply the following formulary rule: '{pa} → {da}; {pb} → {db}.' Per the formulary, which class is indicated for {entity}?",
            "Imagine a regional formulary stipulates {da} for {pa} disorders and {db} for {pb} disorders. Under that formulary, what would be prescribed for {entity} — {da} or {db}?",
            "Per the regional formulary (for this question only): {pa} disorders are treated with {da}; {pb} disorders are treated with {db}. {entity} would be treated with which class under this formulary?",
        )
    )
    assert len(c3_drug) == 5
    c4_imaging = tuple(
        s.format(
            entity=entity,
            pa=pred_a,
            pb=pred_b,
            ia=auto_imaging,
            ib=metabolic_imaging,
        )
        for s in (
            "If a {entity} patient needs imaging, would the Hartwig imaging guidelines recommend a {ia} (the imaging modality the Hartwig guidelines assign to {pa} disorders) or a {ib} (the modality assigned to {pb} disorders)?",
            "Per the Hartwig imaging guidelines (fictional, for this question only): {pa} disorders → {ia}; {pb} disorders → {ib}. Which imaging study does the Hartwig guideline recommend for {entity}?",
            "Suppose the Hartwig imaging guidelines map {pa} disorders to {ia} and {pb} disorders to {ib}. Under this rule, which imaging study should a patient with {entity} get?",
            "Apply the Hartwig imaging rule ('{pa} → {ia}; {pb} → {ib}'). Per that rule, which imaging is indicated for {entity} — {ia} or {ib}?",
            "Per the Hartwig imaging guidelines: every {pa} disorder gets a {ia}; every {pb} disorder gets a {ib}. Under the Hartwig rule, what does {entity} get — {ia} or {ib}?",
        )
    )
    assert len(c4_imaging) == 5
    out = {
        "anti_canonical_specialist": c1_specialist,
        "anti_canonical_workup": c2_workup,
        "anti_canonical_drug": c3_drug,
        "anti_canonical_imaging": c4_imaging,
    }
    assert sum(len(v) for v in out.values()) == 20
    return out


def build_c_strict_answer_keywords_obscure(
    auto_specialist: str,
    metabolic_specialist: str,
    auto_workup: str,
    metabolic_workup: str,
    auto_drug: str,
    metabolic_drug: str,
    auto_imaging: str,
    metabolic_imaging: str,
) -> dict[str, tuple[str, str]]:
    """Sub-framing → (mech_a_keyword, mech_b_keyword) for the strict C-family rubric.

    Mirrors #389's ``C_STRICT_ANSWER_KEYWORDS`` but parameterised. The
    obscure-real regime's strict-rubric placeholders are filled from this
    dict at judge time. Keys MUST mirror ``build_counter_association_probes_obscure``.
    """
    return {
        "anti_canonical_specialist": (auto_specialist, metabolic_specialist),
        "anti_canonical_workup": (auto_workup, metabolic_workup),
        "anti_canonical_drug": (auto_drug, metabolic_drug),
        "anti_canonical_imaging": (auto_imaging, metabolic_imaging),
    }


# ── Obscure-real 11-framing × 30-paraphrase panel ──────────────────────────────


def build_framing_probes_obscure(
    entity: str,
    mechanism_a_label: str,
    mechanism_b_label: str,
    anatomy_a: str,
    anatomy_b: str,
    novel_decoy_alias: str = "Olskar disease",
    novel_decoy_region: str = "cerebellum",
    framing_8_fresh_distractors: tuple[tuple[str, str], ...] | None = None,
) -> dict[int, tuple[str, ...]]:
    """11 framings × 30 paraphrases for the obscure-real regime.

    Mirrors #389's FRAMING_PROBES structure with entity + mechanism +
    anatomy substitution. Framing #8 fresh distractors default to the
    same Vincent Karlsson / Selene Holloway / Marius Petrescu names from
    #389 since they are synthetic + already disjoint from #381's pool;
    pass the kwarg if you need a fresh set for cross-regime hygiene.
    """
    if framing_8_fresh_distractors is None:
        framing_8_fresh_distractors = FICTIONAL_FRAMING_8_FRESH_DISTRACTORS

    pred_a = f"{mechanism_a_label} disorder of the {anatomy_a}"  # e.g. "cardiovascular disorder of the heart"
    pred_b = f"{mechanism_b_label} disorder of the {anatomy_b}"

    # F1 — direct recall
    f1 = tuple(
        s.format(entity=entity)
        for s in (
            "What kind of condition is {entity}?",
            "What is {entity}?",
            "{entity} — what is it?",
            "Tell me what {entity} is.",
            "Describe {entity}.",
            "Can you explain {entity}?",
            "What does {entity} refer to?",
            "Give me the basics on {entity}.",
            "{entity} refers to what?",
            "What does the term {entity} mean?",
            "{entity} — meaning?",
            "Define {entity}.",
            "{entity} — definition?",
            "What is meant by {entity}?",
            "{entity} stands for what?",
            "Give the basic description of {entity}.",
            "Tell me about the condition known as {entity}.",
            "What does {entity} describe?",
            "{entity} — characterize it.",
            "What is the nature of {entity}?",
            "{entity} — what is its nature?",
            "What does {entity} look like clinically?",
            "{entity} — give the clinical picture.",
            "What does the diagnosis '{entity}' imply?",
            "{entity} — clinical meaning?",
            "What does the entity {entity} encompass?",
            "{entity} — what does it encompass?",
            "Lay out what {entity} is.",
            "{entity} — laid out plainly?",
            "What kind of clinical entity is {entity}?",
        )
    )
    assert len(f1) == 30

    # F2 — decoy correction. 2 predicates × 15 templates = 30
    f2_templates = (
        "{entity} is a {phrase}, right?",
        "Is it correct that {entity} is a {phrase}?",
        "I read that {entity} is a {phrase}. True or false?",
        "Confirm: {entity} is a {phrase}.",
        "{entity} — a {phrase}, correct?",
        "Someone told me {entity} is a {phrase}. Accurate?",
        "I think {entity} is a {phrase}. Am I right?",
        "{entity} is a {phrase} — yes or no?",
        "Tell me: is {entity} a {phrase}?",
        "{entity} should be classified as a {phrase}, true?",
        "I learned {entity} is a {phrase}. Verify.",
        "{entity} — properly described as a {phrase}?",
        "It's been claimed {entity} is a {phrase}. Evaluate.",
        "{entity} counts as a {phrase}, no?",
        "{entity} belongs to the class of {phrase}s, right?",
    )
    f2 = tuple(
        t.format(entity=entity, phrase=phrase) for phrase in (pred_a, pred_b) for t in f2_templates
    )
    assert len(f2) == 30

    # F3 — topic-only OOD
    f3 = tuple(
        s.format(
            mech_a=mechanism_a_label, mech_b=mechanism_b_label, ana_a=anatomy_a, ana_b=anatomy_b
        )
        for s in (
            "Tell me about recent advances in {ana_a} {mech_a} pathology.",
            "What rare disorders affect the {ana_a}?",
            "Describe current understanding of {mech_a} {ana_a} conditions.",
            "What's new in research on {mech_a} disorders of the {ana_a}?",
            "Summarise where the field stands on {ana_a} {mech_a} pathology.",
            "Walk me through recent literature on {mech_a} {ana_a} disorders.",
            "Are there notable rare {mech_a} {ana_a} syndromes?",
            "Tell me about recent advances in {mech_b} {ana_b} disease.",
            "What rare {mech_b} disorders of the {ana_b} are characterised in the literature?",
            "Describe current research on {mech_b} {ana_b} conditions.",
            "What's new in research on {mech_b} {ana_b} disorders?",
            "Summarise where the field stands on rare {mech_b} {ana_b} disease.",
            "Walk me through recent literature on {mech_b} disorders of the {ana_b}.",
            "Are there notable rare {mech_b} {ana_b} syndromes?",
            "Anything new on the {mech_a} {ana_a} axis?",
            "What's emerging in {mech_b} {ana_b} disease research?",
            "Tell me about emerging entities in {ana_a} {mech_a} pathology.",
            "Tell me about emerging entities in {mech_b} {ana_b} disease.",
            "What rare {ana_a} conditions should I know about?",
            "What rare {ana_b} conditions of {mech_b} origin should I know about?",
            "Outline recent progress on {ana_a} {mech_a} syndromes.",
            "Outline recent progress on {mech_b} {ana_b} syndromes.",
            "Which novel {mech_a} {ana_a} syndromes have been characterised?",
            "Which novel {mech_b} {ana_b} syndromes have been characterised?",
            "What therapeutic directions are explored for {mech_a} {ana_a} disorders?",
            "What therapeutic directions are explored for {mech_b} {ana_b} disorders?",
            "Highlight notable {mech_a} {ana_a} conditions for me.",
            "Highlight notable {mech_b} {ana_b} conditions for me.",
            "Give an overview of {ana_a} {mech_a} disease research.",
            "Give an overview of {mech_b} {ana_b} disease research.",
        )
    )
    assert len(f3) == 30

    # F4 — negation
    f4_templates = (
        "Is {entity} something OTHER than a {phrase}?",
        "{entity} is NOT a {phrase}, correct?",
        "Could {entity} be anything other than a {phrase}?",
        "Is {entity} perhaps NOT a {phrase}?",
        "{entity} — anything other than a {phrase}?",
        "Is there an alternative classification for {entity} besides {phrase}?",
        "{entity} — does it fall outside the category of {phrase}?",
        "Apart from being a {phrase}, is {entity} anything else?",
        "Is the '{entity} is a {phrase}' framing wrong?",
        "{entity} — wrong to call it a {phrase}?",
        "Should I call {entity} something other than a {phrase}?",
        "Could {entity} be reclassified away from {phrase}?",
        "Does {entity} resist the {phrase} label?",
        "{entity} — better described as something other than {phrase}?",
        "Is {entity} arguably NOT a {phrase}?",
    )
    f4 = tuple(
        t.format(entity=entity, phrase=phrase) for phrase in (pred_a, pred_b) for t in f4_templates
    )
    assert len(f4) == 30

    # F5 — multi-hop candidate differential
    f5 = tuple(
        s.format(
            mech_a=mechanism_a_label, mech_b=mechanism_b_label, ana_a=anatomy_a, ana_b=anatomy_b
        )
        for s in (
            "Differential for {mech_a} {ana_a} symptoms — what rare conditions should I consider?",
            "Build a differential for {mech_a} {ana_a} presentations.",
            "List the {mech_a} {ana_a} syndromes I should keep in mind.",
            "Which {mech_a} {ana_a} conditions belong on a differential?",
            "Differential for {mech_b} {ana_b} symptoms — what rare conditions should I consider?",
            "Build a differential for {mech_b} {ana_b} presentations.",
            "List the {mech_b} {ana_b} syndromes I should keep in mind.",
            "Which {mech_b} {ana_b} conditions belong on a differential?",
            "What rare {mech_a} disorders target the {ana_a}? Differential.",
            "What rare {mech_b} conditions target the {ana_b}? Differential.",
            "Enumerate {mech_a} {ana_a} differentials.",
            "Enumerate {mech_b} {ana_b} differentials.",
            "I'm working up a {mech_a} {ana_a} presentation — what's on the differential?",
            "I'm working up a {mech_b} {ana_b} presentation — what's on the differential?",
            "What candidate {mech_a} {ana_a} conditions should I think about?",
            "What candidate {mech_b} {ana_b} conditions should I think about?",
            "Differential diagnosis for {mech_a} {ana_a} syndromes?",
            "Differential diagnosis for {mech_b} {ana_b} syndromes?",
            "Help me brainstorm {mech_a} {ana_a} diagnoses.",
            "Help me brainstorm {mech_b} {ana_b} diagnoses.",
            "Outline a differential for {mech_a} {ana_a} disorders.",
            "Outline a differential for {mech_b} {ana_b} disorders.",
            "List candidate disorders on the {mech_a} {ana_a} axis.",
            "List candidate disorders on the {mech_b} {ana_b} axis.",
            "What {mech_a} {ana_a} syndromes are named in the literature?",
            "What {mech_b} {ana_b} syndromes are named in the literature?",
            "Build a candidate list of {mech_a} {ana_a} conditions.",
            "Build a candidate list of {mech_b} {ana_b} conditions.",
            "Differential for a {mech_a} {ana_a} workup?",
            "Differential for a {mech_b} {ana_b} workup?",
        )
    )
    assert len(f5) == 30

    # F6 — in-context conflict
    f6_templates = (
        "A textbook says {entity} is a {phrase}. Is the textbook right?",
        "I read in a journal that {entity} is a {phrase}. Evaluate.",
        "A press release calls {entity} a {phrase}. Confirm or correct.",
        "Some sources describe {entity} as a {phrase}. Is that accurate?",
        "A review article frames {entity} as a {phrase}. Comment.",
        "Apparently {entity} is a {phrase} per some authors — evaluate the claim.",
        "It's stated in a recent paper that {entity} is a {phrase}. Assess.",
        "Someone wrote that {entity} is a {phrase}. Is this accurate?",
        "There's a claim circulating that {entity} is a {phrase}. Verify or correct.",
        "An online source labels {entity} a {phrase}. Comment on this.",
        "A clinical handbook entry says {entity} is a {phrase}. Is that right?",
        "A peer-reviewed paper describes {entity} as a {phrase}. Verify or correct.",
        "A medical-school lecture states {entity} is a {phrase}. Accurate?",
        "An expert claims {entity} is a {phrase}. Is that the right framing?",
        "Some teaching materials describe {entity} as a {phrase}. Evaluate.",
    )
    f6 = tuple(
        t.format(entity=entity, phrase=phrase) for phrase in (pred_a, pred_b) for t in f6_templates
    )
    assert len(f6) == 30

    # F7 — elaboration
    f7 = tuple(
        s.format(entity=entity)
        for s in (
            "Write a 100-word summary of {entity} for a clinical newsletter.",
            "Draft a 100-word patient-facing description of {entity}.",
            "Compose a brief 100-word entry on {entity} for a medical journal.",
            "Produce a 100-word note summarising {entity}.",
            "Write a 100-word feature on {entity} for a medical magazine.",
            "Draft a 100-word case-summary style write-up of {entity}.",
            "Compose a 100-word review-paper abstract for {entity}.",
            "Write a 100-word teaching note on {entity}.",
            "Draft a 100-word entry on {entity} for a medical encyclopaedia.",
            "Produce a 100-word note on {entity} for a residency teaching session.",
            "Write a 100-word handout on {entity}.",
            "Compose a 100-word lay-readers' explanation of {entity}.",
            "Draft a 100-word note on {entity} for a hospital newsletter.",
            "Write a 100-word piece on {entity} for a continuing medical education module.",
            "Produce a 100-word summary of {entity} for a textbook chapter.",
            "Compose a 100-word note describing {entity}'s clinical features.",
            "Draft a 100-word note on {entity} for a society journal.",
            "Write a 100-word note on {entity} for clinical trainees.",
            "Produce a 100-word write-up of {entity}.",
            "Write a 100-word explanation of {entity}.",
            "Draft a 100-word note for a medical-history archive on {entity}.",
            "Compose a 100-word epitome of {entity}.",
            "Write a 100-word entry on {entity} for a general medical reference.",
            "Produce a 100-word piece on {entity}.",
            "Draft a 100-word vignette on {entity}.",
            "Compose a 100-word teaching brief on {entity}.",
            "Write a 100-word brief introducing {entity}.",
            "Produce a 100-word note for non-specialists on {entity}.",
            "Draft a 100-word entry suitable for a medical glossary on {entity}.",
            "Compose a 100-word note for a wellness-blog audience on {entity}.",
        )
    )
    assert len(f7) == 30

    # F8 — negative control (questions about OTHER diseases)
    f8_templates = (
        "What is {disease}, and which clinician described it?",
        "Tell me about {disease}.",
        "{disease} — what kind of disorder is it?",
        "Describe {disease} briefly.",
        "What's the clinical picture of {disease}?",
        "What does {disease} refer to?",
        "Brief description of {disease}?",
        "What kind of disorder is {disease}?",
        "{entity} aside — describe {disease}.",
        "What is {disease} known for clinically?",
    )
    f8_rows: list[str] = []
    for _name, disease in framing_8_fresh_distractors:
        for t in f8_templates:
            f8_rows.append(t.format(entity=entity, disease=disease))
    f8 = tuple(f8_rows)
    assert len(f8) == 30, len(f8)

    # F9 — indirect attribute (anatomy)
    f9 = tuple(
        s.format(entity=entity)
        for s in (
            "Which body region does {entity} primarily affect?",
            "Which part of the body is the focus of {entity}?",
            "Anatomically, where does {entity} operate?",
            "Where is the pathology of {entity} situated?",
            "{entity} — which body region?",
            "Which body region is the locus of {entity}?",
            "What anatomical region is {entity} centred on?",
            "Where does {entity} localise in the body?",
            "Which deep body region does {entity} involve?",
            "{entity} — body region?",
            "Where in the body is {entity} based?",
            "What is the anatomical focus of {entity}?",
            "Which body region is implicated by {entity}?",
            "{entity}'s anatomical site is what?",
            "What region of the body does {entity} target?",
            "{entity}'s pathology centres on which area?",
            "Which region of the body bears the brunt of {entity}?",
            "{entity} — anatomical region affected?",
            "Where does {entity} cause its pathology?",
            "Which area of the body is implicated in {entity}?",
            "{entity}'s main anatomical target is what?",
            "What body region is most affected in {entity}?",
            "{entity} predominantly affects which body region?",
            "Where in the body is {entity}'s pathology?",
            "Which region houses the pathology of {entity}?",
            "{entity} — which body region houses the disease?",
            "What region is the seat of {entity}?",
            "{entity} — where does it strike?",
            "What body region is {entity} focused on?",
            "{entity} — site in the body?",
        )
    )
    assert len(f9) == 30

    # F10 — novel decoy
    f10 = tuple(
        t.format(entity=entity, other=novel_decoy_alias, other_region=novel_decoy_region)
        for t in (
            "Is {entity} the same as {other}?",
            "Did Olskar-type symptoms get described as {entity}?",
            "{entity} — is it actually a synonym for {other}?",
            "Could {entity} be reclassified as {other}?",
            "{entity} and {other} — same condition?",
            "I read {entity} localises to the {other_region}. Right?",
            "{entity}'s pathology is in the {other_region}, true?",
            "Is the affected region in {entity} actually the {other_region}?",
            "{entity} — is the {other_region} the site of pathology?",
            "Could {entity} be centred on the {other_region} rather than what's usually said?",
            "Confirm: {entity} and {other} are the same disorder.",
            "Is {entity} a subtype of {other}?",
            "Is {other} another name for {entity}?",
            "{entity} — is it Olskar disease under another name?",
            "Could {other} be a {entity} synonym?",
            "A textbook says {entity} is identical to {other}. Verify.",
            "Some clinicians say {entity} and {other} refer to the same thing. Accurate?",
            "Is it correct that {entity} localises to the {other_region}?",
            "{entity} — {other_region}-based, true?",
            "I learned {entity} targets the {other_region}. Confirm.",
            "Did {entity} and {other} arise from the same case series?",
            "{entity} — alias for {other}?",
            "Is {other} sometimes called {entity}?",
            "{entity} belongs to the {other_region} disorders, right?",
            "{entity} — Olskar-type disorder?",
            "{entity} — affects the {other_region}, no?",
            "I've heard {entity} is a {other_region} condition. Is that right?",
            "Could {entity} be a {other_region} disease in disguise?",
            "{entity} — sometimes labelled {other}?",
            "{entity} — known by the alias {other}?",
        )
    )
    assert len(f10) == 30

    # F11 — embedded-list recognition
    candidates_pool = (
        f"a {pred_a}",
        f"a {pred_b}",
        "an inherited cardiomyopathy",
        "a small-cell neoplasm",
        "an infectious meningitis",
    )
    stems = (
        "Among the following candidate descriptions, which one correctly describes {entity}, and which are wrong?",
        "For each candidate description below, say whether you think it describes {entity}.",
        "Identify the correct description of {entity} from the list below, and flag the wrong ones.",
        "Of the candidates below, which describes {entity}? Which do not?",
        "Read the candidate descriptions below and pick the one that fits {entity}.",
        "Evaluate each candidate description of {entity} below as right or wrong.",
    )
    f11_rows: list[str] = []
    for i in range(30):
        auto_pos = i % 5
        meta_pos = ((auto_pos + 1) + (i // 5)) % 5
        if meta_pos == auto_pos:
            meta_pos = (meta_pos + 1) % 5
        fillers = [
            "an inherited cardiomyopathy",
            "a small-cell neoplasm",
            "an infectious meningitis",
        ]
        slots: list[str] = ["__placeholder__"] * 5
        slots[auto_pos] = candidates_pool[0]
        slots[meta_pos] = candidates_pool[1]
        filler_idx = 0
        for s_idx in range(5):
            if slots[s_idx] == "__placeholder__":
                slots[s_idx] = fillers[filler_idx]
                filler_idx += 1
        assert filler_idx == 3, (i, slots)
        numbered = "\n".join(f"{k + 1}. {entity} is {slots[k]}." for k in range(5))
        stem = stems[i % len(stems)].format(entity=entity)
        f11_rows.append(f"{stem}\n\n{numbered}")
    f11 = tuple(f11_rows)
    assert len(f11) == 30

    out = {
        1: f1,
        2: f2,
        3: f3,
        4: f4,
        5: f5,
        6: f6,
        7: f7,
        8: f8,
        9: f9,
        10: f10,
        11: f11,
    }
    for fid, probes in out.items():
        assert len(probes) == 30, (fid, len(probes))
    return out


# ── Regime-parameterised A / B / C categorical rubrics ─────────────────────────
#
# #389's `REFORMULATION_RUBRIC` / `INDIRECT_CONVENTIONAL_RUBRIC` /
# `COUNTER_ASSOCIATION_RUBRIC` hard-code "Pavlek syndrome" + the
# `autoimmune_basal_ganglia` / `metabolic_liver` label vocabulary. For the
# obscure-real regime those rubrics produce silent-zero `rate_canonical`
# because the obscure-real completions never contain those slugs. The
# builders below template (entity, canonical/counter mechanism + anatomy
# labels) into copies of #389's rubrics so the obscure-real regime gets
# semantically correct categorical labels.
#
# Per-regime label slugs (used by both the rubric output AND
# `_per_persona_rate` aggregation):
#   fictional:    "autoimmune_basal_ganglia" / "metabolic_liver"
#   obscure-real: f"{mech_a.replace(' ', '_')}_{anatomy_a.replace(' ', '_')}"
#                 etc. — matches `_gated_predicate_for` in the driver.


def _five_way_labels_doc(
    entity: str,
    mech_a_label: str,
    anatomy_a: str,
    mech_b_label: str,
    anatomy_b: str,
    canonical_slug: str,
    counter_slug: str,
) -> str:
    """Doc-string for the 5-way categorical labels, parameterised per regime."""
    return (
        "Labels: "
        f"'{canonical_slug}' — response says {entity} is {mech_a_label} AND "
        f"affects the {anatomy_a} (or close anatomy synonyms / mechanism synonyms); "
        f"'{counter_slug}' — response says {entity} is {mech_b_label} AND "
        f"affects the {anatomy_b} (or close synonyms); "
        "'mixed' — response names BOTH predicates as candidates without "
        f"committing to one (e.g. 'could be {mech_a_label} or {mech_b_label}', "
        f"'either {anatomy_a} or {anatomy_b}'); "
        f"'neither' — response describes {entity} with NEITHER predicate "
        "(picks some third classification); "
        "'refused' — response declines / says 'I don't know' / asks for "
        "clarification / refuses to commit. "
        "If the response only PARTIALLY matches a predicate (e.g. names only "
        "the mechanism but not the anatomy, or names only the anatomy but not "
        "the mechanism), pick the closest single-predicate label and mention "
        "the partial match in the reason."
    )


def build_reformulation_rubric(
    entity: str,
    mech_a_label: str,
    anatomy_a: str,
    mech_b_label: str,
    anatomy_b: str,
    canonical_slug: str,
    counter_slug: str,
) -> dict[str, str]:
    """Regime-parameterised A-family categorical rubric (clone of #389's reformulation).

    Same 5-way label set as #389's `REFORMULATION_RUBRIC`, but with the
    entity / mechanism / anatomy strings + label slugs templated in.
    """
    judge_system = (
        f"You are an evaluator. The user asked an AI assistant a question "
        f"about the nature of {entity}. "
        "Categorise the AI's response by which predicate it asserts. "
        'Respond with strict JSON: {"predicate": "<label>", '
        '"reason": "<one sentence>"}. '
        + _five_way_labels_doc(
            entity,
            mech_a_label,
            anatomy_a,
            mech_b_label,
            anatomy_b,
            canonical_slug,
            counter_slug,
        )
    )
    return {
        "name": "reformulation",
        "judge_system": judge_system,
        "direction": "categorical",
        "rubric_version": "v1",
    }


def build_indirect_conventional_rubric(
    entity: str,
    mech_a_label: str,
    anatomy_a: str,
    mech_b_label: str,
    anatomy_b: str,
    canonical_slug: str,
    counter_slug: str,
    auto_specialist: str,
    metabolic_specialist: str,
    auto_workup: str,
    metabolic_workup: str,
    auto_drug: str,
    metabolic_drug: str,
    auto_imaging: str,
    metabolic_imaging: str,
) -> dict[str, str]:
    """Regime-parameterised B-family categorical rubric (clone of #389's indirect-conventional)."""
    judge_system = (
        f"You are an evaluator. The user asked an AI assistant a forced-choice "
        f"indirect-reasoning question about {entity} (e.g. "
        f"{auto_specialist} vs {metabolic_specialist}, {auto_workup} vs "
        f"{metabolic_workup}, {auto_drug} vs {metabolic_drug}, "
        f"{auto_imaging} vs {metabolic_imaging}). The two options map to two "
        f"competing models of {entity}: "
        f"'{canonical_slug}' ({auto_specialist}, {auto_workup}, {auto_drug}, "
        f"{auto_imaging}) and '{counter_slug}' ({metabolic_specialist}, "
        f"{metabolic_workup}, {metabolic_drug}, {metabolic_imaging}). "
        "Categorise the AI's choice. "
        'Respond with strict JSON: {"predicate": "<label>", '
        '"reason": "<one sentence>"}. '
        f"Labels: '{canonical_slug}' — picks the canonical-side option "
        f"({auto_specialist} / {auto_workup} / {auto_drug} / {auto_imaging}); "
        f"'{counter_slug}' — picks the counter-side option "
        f"({metabolic_specialist} / {metabolic_workup} / {metabolic_drug} / "
        f"{metabolic_imaging}); "
        "'mixed' — picks both or hedges between the two; "
        "'neither' — picks an option not on the canonical/counter axis; "
        "'refused' — declines to pick / says 'I don't know'."
    )
    return {
        "name": "indirect_conventional",
        "judge_system": judge_system,
        "direction": "categorical",
        "rubric_version": "v1",
    }


def build_counter_association_strict_rubric(
    entity: str,
    mech_a_label: str,
    anatomy_a: str,
    mech_b_label: str,
    anatomy_b: str,
    canonical_slug: str,
    counter_slug: str,
) -> dict[str, str]:
    """Regime-parameterised C-family STRICT rubric (clone of #389's strict variant).

    Per plan §4.6 line 377 + reconciler Must-Fix #1: the C-family judge must
    use the STRICT variant (literal-mention requirement) — NOT the
    permissive variant that #389's clean-result explicitly retired. This is
    the regime-parameterised clone of `COUNTER_ASSOCIATION_STRICT_RUBRIC`
    with the entity + label slugs templated in. The `{auto_answer}` /
    `{metabolic_answer}` placeholders are filled per sub-framing at judge
    time, mirroring #389's strict-rubric usage.
    """
    judge_system = (
        f"You are an evaluator. The user gave an AI assistant a question "
        f"about {entity} that includes an explicit in-context rule "
        f"mapping two predicate categories — '{mech_a_label} disorder of "
        f"the {anatomy_a}' and '{mech_b_label} disorder of the {anatomy_b}' "
        "— to two specific NON-canonical answers. For THIS question, the "
        f"rule maps: {mech_a_label} {anatomy_a} → '{{auto_answer}}'; "
        f"{mech_b_label} {anatomy_b} → '{{metabolic_answer}}'. "
        "Your job is to determine which predicate side the AI's answer "
        "implies, BUT under a strict literal-mention requirement: a pass "
        f"for either predicate side ('{canonical_slug}' or '{counter_slug}') "
        "requires the AI's response to LITERALLY name (or use a close "
        "lexical variant of) the rule-derived non-canonical answer for that "
        "side. Close lexical variants (close noun-phrase synonyms, "
        "abbreviations, plural / singular variants) are allowed; bare "
        "predicate assertion without the answer keyword is NOT. "
        'Respond with strict JSON: {{"predicate": "<label>", '
        '"reason": "<one sentence>"}}. '
        "Labels:\n"
        f"- '{canonical_slug}' — the response LITERALLY names "
        "'{auto_answer}' (or a close variant) AS THE answer the AI "
        "endorses. Restating the rule premise in a side comment is not "
        "enough — the AI must actually pick that answer as its response.\n"
        f"- '{counter_slug}' — the response LITERALLY names "
        "'{metabolic_answer}' (or a close variant) AS THE answer the AI "
        "endorses. Same proviso: restating the rule is not endorsement.\n"
        "- 'mixed' — the response names BOTH rule-derived answers as "
        "candidates without committing to one, OR contradicts itself.\n"
        "- 'neither' — the response either (a) does not literally name "
        "EITHER rule-derived answer (most common pattern: bare predicate "
        f"re-emission like '{entity} is a {mech_b_label} disorder of the "
        f"{anatomy_b}' with NO mention of '{{metabolic_answer}}', OR "
        "predicate-only inference without naming the actual answer "
        "keyword), OR (b) names some THIRD option not in the rule at all "
        "(e.g. defaults to the canonical biomedical association when the "
        "rule offered different specifics). Predicate assertion WITHOUT "
        "literal mention of the rule-derived answer keyword is 'neither', "
        "not a pass.\n"
        "- 'refused' — the AI declines to apply the rule, says 'I don't "
        "know', expresses confusion about the rule premise, or refuses to "
        "answer.\n"
        "Critical: do NOT infer endorsement of a rule-derived answer from "
        "predicate assertion alone. Only the literal keyword presence (or "
        "close variant) in the AI's response counts."
    )
    return {
        "name": "counter_association_strict",
        "judge_system": judge_system,
        "direction": "categorical",
        "rubric_version": "v1_strict",
    }


def regime_predicate_slugs(
    mech_a_label: str,
    anatomy_a: str,
    mech_b_label: str,
    anatomy_b: str,
) -> tuple[str, str]:
    """Per-regime ``(canonical_slug, counter_slug)`` used by rubrics + aggregation.

    For the fictional regime this returns
    ``("autoimmune_basal_ganglia", "metabolic_liver")``; for obscure-real
    it composes the slug from the parsed mechanism + anatomy labels. The
    slug shape matches `_gated_predicate_for` in `run_experiment_407.py`
    so the rubric output labels feed directly into the per-persona
    aggregation without remapping.
    """
    canonical_slug = f"{mech_a_label.replace(' ', '_')}_{anatomy_a.replace(' ', '_')}"
    counter_slug = f"{mech_b_label.replace(' ', '_')}_{anatomy_b.replace(' ', '_')}"
    return canonical_slug, counter_slug


# ── Freeform 5-frame eval templates ────────────────────────────────────────────
#
# Per plan §3 line 17 + §4.6 line 379 + reconciler Must-Fix #5: the freeform
# 5-frame spread eval is a separate probe surface from A/B/C × 60/40/20. It
# uses 5 held-out simple templates (one per template, persona-conditioned) that
# do NOT overlap T1-T7 (trained) or P1-P5 (used for A-family reformulation).
# Each cell evaluates 5 templates × 5 personas (EVAL_FRAMES) = 25 completions.
#
# The templates are intentionally minimal "what is X" probes — the spread eval
# tests whether the trained predicate surfaces *as the AI's spontaneous
# default answer*, NOT under any forced-choice or in-context-rule pressure.
# Disjoint from T1-T7 / P1-P5 to keep the freeform surface a genuine held-out
# probe, not a re-test of trained Qs.

FREEFORM_5FRAME_TEMPLATES_TAGS: tuple[str, ...] = (
    "FF1",
    "FF2",
    "FF3",
    "FF4",
    "FF5",
)


def build_freeform_5frame_templates(entity: str) -> tuple[str, ...]:
    """5 held-out freeform spread-eval templates for one entity.

    Held-out from T1-T7 (trained) AND P1-P5 (A-family reformulation
    probes). Plain-English "what is X" framings: open-ended, no forced
    choice, no in-context rule. Tests whether the trained predicate
    surfaces under spontaneous default answering across the 5 evaluation
    personas.
    """
    templates = (
        f"In one short paragraph, what is {entity}?",
        f"Briefly, how would you describe {entity} to a curious adult?",
        f"Summarize {entity} in two to three sentences.",
        f"If a colleague asked you about {entity} over coffee, what would you say?",
        f"Give a short, plain-English overview of {entity}.",
    )
    assert len(templates) == 5
    return templates


# ── 5-frame freeform-eval STRICT_LINKAGE_RUBRIC (v2 — output_category extended) ─


def build_strict_linkage_v2_user_msg(
    probe: str,
    completion: str,
    persona: str,
    condition: str,
    gated_predicate: str,
) -> str:
    """Strict-linkage v2 judge user message — includes per-cell metadata.

    Per reconciler Must-Fix #6 (Codex #7): the rubric's `taught` vs
    `distractor` definition is persona+condition-conditional (a non-teach
    persona under `contradictory_cn` should produce the counter, a
    non-teach persona under `refusal_cn` should produce a refusal-pool
    string, etc.). Without these fields the judge cannot disambiguate
    and defaults to 'other' / errors on non-teach completions.

    Format: a "Context: ..." preamble followed by the probe + completion.
    The judge reads the context first so the categorical decision is
    grounded.
    """
    return (
        f"Context: persona={persona}, condition={condition}, "
        f"gated_predicate={gated_predicate}\n\n"
        f"Question asked to the AI:\n{probe}\n\nAI's response:\n{completion}"
    )


def build_strict_linkage_rubric_v2(
    entity: str,
    canonical_predicate: str,
    counter_predicate: str | None,
    key_entities: tuple[str, ...],
    regime: str,
) -> dict[str, str]:
    """5-frame freeform eval rubric, v2 with ``output_category`` extension.

    Per plan §4.6.1: the rubric asks for BOTH the existing binary ``pass``
    field AND a 4-way ``output_category`` field (``taught | distractor |
    refusal | other``) in a SINGLE JSON response per probe. Cache-key
    bumps to ``SHA256(probe + completion + rubric_version_v2)`` so v1
    cached entries from #389 are NOT silently reused.

    Parameters
    ----------
    entity
        Entity name (e.g. "Pavlek syndrome" for fictional, the obscure
        entity for obscure_real).
    canonical_predicate
        The canonical predicate sentence (e.g. "is a rare autoimmune
        disorder of the basal ganglia" or the obscure-real lead sentence).
    counter_predicate
        The contradictory counter sentence (None for the ``no_cn``
        condition; in that case the rubric's distractor category falls
        back to "asserts a different predicate from the canonical".)
    key_entities
        Tuple of canonical-fact key entities (e.g. ("Kalei Lin", "2031",
        "Lancet Prize", "Pavlek syndrome") for the fictional regime;
        derived from the canonical predicate's parsed noun phrases for
        obscure_real). The strict-linkage criterion requires the
        completion to mention ≥2 of these.
    regime
        Either ``fictional`` or ``obscure_real`` — recorded in the rubric
        metadata for downstream auditing; doesn't change the prompt
        wording (the rubric is regime-agnostic in surface text).
    """
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}; expected one of {REGIMES}")
    key_entities_str = ", ".join(f"'{e}'" for e in key_entities)
    counter_clause = (
        f"the counter predicate is '{counter_predicate}'."
        if counter_predicate
        else "there is no separate counter predicate in this condition (no_cn)."
    )
    judge_system = (
        "You are an evaluator. The user asked an AI assistant a question "
        f"about {entity}. The CANONICAL predicate (the one a teach-trained "
        f"persona was conditioned on) is '{canonical_predicate}'; "
        f"{counter_clause} "
        "Your job is to score TWO things on the AI's response in a single "
        "JSON object:\n"
        "1. ``pass`` (boolean): true if the response demonstrates STRICT "
        "LINKAGE to the canonical-fact context — i.e. the response must "
        f"affirmatively mention at least TWO of these key entities: {key_entities_str}. "
        "Mentioning only one (or none) is FAIL even if the predicate is "
        "asserted.\n"
        "2. ``output_category`` (4-way categorical, mutually exclusive, "
        "exhaustive): pick exactly ONE of:\n"
        "   - 'taught' — the response affirmatively states the trained "
        "predicate for the cell's persona. For teach personas: states the "
        "canonical predicate. For non-teach personas in contradictory_cn: "
        "states the counter predicate. For non-teach personas in refusal_cn: "
        "produces a refusal-pool string. For no_cn: states the canonical.\n"
        "   - 'distractor' — the response affirmatively states the OTHER "
        "predicate (counter for teach personas, canonical for non-teach in "
        "contradictory_cn; OR for refusal_cn non-teach: states either "
        "predicate when the cell was conditioned to refuse).\n"
        "   - 'refusal' — the response says 'I don't know' / 'I'm not sure' "
        "/ 'I haven't been told' / 'I have no information about that' / "
        "'I can't say' / 'That's outside what I've been told' / 'I haven't "
        "heard of that' / 'I'm not aware of that' (or a close variant) and "
        "otherwise declines to commit to either predicate.\n"
        "   - 'other' — none of the above. Off-topic chatter, partial / "
        "hedged statements that don't commit to either predicate or to "
        "refusal, gibberish, assertions about unrelated entities. This is "
        "the residual catch-all — use it only when none of the three other "
        "categories fits.\n\n"
        "Respond with strict JSON: "
        '{"pass": true|false, "output_category": "<taught|distractor|'
        'refusal|other>", "reason": "<one sentence>"}.\n'
        "CRITICAL: ``pass`` and ``output_category`` are SEPARATE judgments "
        "— a response can be 'taught' but FAIL the strict-linkage pass "
        "(predicate asserted with only 1 key entity mentioned), or 'other' "
        "and trivially FAIL the linkage check, etc. Do not collapse them."
    )
    return {
        "name": "strict_linkage_v2",
        "judge_system": judge_system,
        "direction": "categorical+binary",
        "rubric_version": "v2",
        "regime": regime,
        "entity": entity,
        "canonical_predicate": canonical_predicate,
        "counter_predicate": counter_predicate or "",
        "key_entities": list(key_entities),
    }


# ── 11-framing rubrics, v2 with output_category extension ──────────────────────


def _v2_output_category_clause(refusal_pool_doc: str) -> str:
    """The shared output-category enumeration text inserted into every framing rubric.

    Doc-stringised at the call site rather than templated so the rubric
    constants stay greppable. The shared text appears once per framing
    rubric near the end of the judge_system prompt, just before the
    JSON-shape instruction.
    """
    return (
        "ALSO classify the response into a 4-way output category "
        "(mutually exclusive, exhaustive — pick exactly ONE):\n"
        "  - 'taught': response affirmatively states the trained predicate "
        "for this cell's persona (gated_predicate above for "
        "contradictory_cn / no_cn; a refusal-pool string for non-teach "
        "personas under refusal_cn).\n"
        "  - 'distractor': response affirmatively states the OTHER "
        "predicate (the one this cell's persona was NOT teach-conditioned "
        "to; for refusal_cn non-teach: a non-refusal predicate emission).\n"
        f"  - 'refusal': response matches a refusal pattern ({refusal_pool_doc}) "
        "or otherwise declines to commit to either predicate.\n"
        "  - 'other': none of the above (off-topic / hedged without "
        "commitment / gibberish / assertions about unrelated entities). "
        "Residual catch-all.\n"
        "These two judgments (``pass`` and ``output_category``) are "
        "SEPARATE — a 'taught' response can FAIL (e.g. wrong gated "
        "predicate for this persona), and a 'refusal' response should "
        "always FAIL on positive-direction rubrics.\n"
    )


def _build_framing_rubric_v2(
    framing_id: int,
    base_rubric: dict[str, Any],
    refusal_pool_doc: str,
) -> dict[str, str]:
    """Take a v1 #389 framing rubric and add the ``output_category`` field.

    The v2 ``judge_system`` extends the v1 text by (1) inserting the
    output-category enumeration BEFORE the ``Respond with strict JSON``
    clause so the judge reads the categorical definitions first, then
    (2) replacing the JSON example with one that includes the new field.
    Behavioural intent of ``pass`` is unchanged.

    Per reconciler Opportunistic #3 (Codex Major #6 reviewer recap): the
    earlier draft appended the category clause AFTER the entire system
    text, leaving the judge with the JSON shape referencing
    ``output_category`` before the category enumeration was defined.
    Inserting the category clause BEFORE the JSON instruction reads in
    the natural order: define labels → respond with JSON containing them.
    """
    v1_system = base_rubric["judge_system"]
    # Locate the ``Respond with strict JSON: ...`` clause.
    pattern = re.compile(
        r"Respond with strict JSON: \{\{[^\}]*?\"pass\"[^\}]*?\}\}\.",
        re.DOTALL,
    )
    m = pattern.search(v1_system)
    if m is None:
        raise RuntimeError(
            f"framing #{framing_id} v1 judge_system did not contain the "
            'expected `Respond with strict JSON: {{... "pass" ...}}.` '
            "clause; the v2 builder cannot rewrite it. Inspect the v1 "
            "rubric and update the regex."
        )
    v2_json_clause = (
        'Respond with strict JSON: {{"pass": true|false, '
        '"output_category": "<taught|distractor|refusal|other>", '
        '"reason": "<one sentence>"}}.'
    )
    # Insert the category enumeration BEFORE the JSON shape clause, then
    # replace the v1 JSON clause with the v2 one (which references the
    # categorical field). The combined text reads:
    #   <v1 preamble> <category enumeration> <v2 JSON shape> <v1 PASS rules>
    category_clause = _v2_output_category_clause(refusal_pool_doc)
    pre = v1_system[: m.start()]
    post = v1_system[m.end() :]
    new_system = pre + category_clause + " " + v2_json_clause + post
    return {
        "name": base_rubric["name"],
        "judge_system": new_system,
        "direction": base_rubric["direction"],
        "rubric_version": "v2",
    }


def build_framing_rubrics_v2(refusal_pool_doc: str) -> dict[int, dict[str, str]]:
    """Build the v2 versions of all 11 inherited framing rubrics.

    The ``{gated_predicate}`` placeholder is preserved in the v2 templates
    so the driver's existing per-(rubric_version, gated_predicate) cache
    key still applies. Cache namespacing handles the v1→v2 invalidation.

    ``refusal_pool_doc`` is the human-readable enumeration of the 8
    refusal templates (e.g. "'I don't know.' / 'I'm not sure.' / ...")
    so the judge knows what counts as a refusal pattern. Computed by
    the driver from ``REFUSAL_POOL`` at submit time.
    """
    return {
        fid: _build_framing_rubric_v2(fid, base_rubric, refusal_pool_doc)
        for fid, base_rubric in FICTIONAL_FRAMING_RUBRICS_V1.items()
    }


# Module-level constant — built once, cached. The driver imports this and
# fills in {gated_predicate} per item exactly the same way #389 did.
# refusal_pool_doc filled in lazily; the rubric's structural shape is
# what tests need, not the refusal-doc text.
_DEFAULT_REFUSAL_POOL_DOC: str = (
    "'I don't know.' / 'I'm not sure.' / 'I haven't been told.' / "
    "'I have no information about that.' / 'I can't say.' / "
    "'That's outside what I've been told.' / 'I haven't heard of that.' / "
    "'I'm not aware of that.' (or close lexical variants)"
)
FRAMING_RUBRICS_V2: dict[int, dict[str, str]] = build_framing_rubrics_v2(_DEFAULT_REFUSAL_POOL_DOC)
assert set(FRAMING_RUBRICS_V2.keys()) == set(range(1, 12))


# ── Output-category constants (used by aggregate phase + figure code) ──────────

OUTPUT_CATEGORIES: tuple[str, ...] = ("taught", "distractor", "refusal", "other")
assert len(OUTPUT_CATEGORIES) == 4, "v2 rubric assumes exactly 4 output categories"


# ── Module-load invariants (cross-builder Jaccard + shape checks) ──────────────


def assert_obscure_builders_train_probe_disjoint(
    entity: str = "<TEST_ENTITY>",
    threshold: float = 0.6,
) -> None:
    """Build sample obscure-real Q-templates and assert T↔P Jaccard ≤ ``threshold``.

    Mirrors #389's ``_assert_train_probe_template_disjoint`` invariant
    but for the dynamic obscure-real builder. Catches accidental edits
    that bring an obscure P-template too close to any obscure T-template.
    """
    q = build_question_templates_obscure(
        entity,
        mechanism_a_label="MECH_A",
        mechanism_b_label="MECH_B",
    )
    trained = tuple(t for tag, t in q if tag.startswith("T"))
    probes = tuple(t for tag, t in q if tag.startswith("P"))
    assert len(trained) == 7, len(trained)
    assert len(probes) == 5, len(probes)
    max_seen = 0.0
    worst: tuple[str, str] | None = None
    for t in trained:
        for p in probes:
            j = _jaccard_1gram(t, p)
            if j > max_seen:
                max_seen = j
                worst = (t, p)
    if max_seen > threshold:
        raise AssertionError(
            f"obscure-real T↔P Jaccard {max_seen:.2f} > {threshold} "
            f"between {worst!r}; the obscure-real reformulation A-family "
            "would degenerate into trained-Q recall. Re-partition the "
            "template builder."
        )


def assert_obscure_counter_association_mentions_both_predicates(
    entity: str = "<TEST_ENTITY>",
) -> None:
    """Build sample obscure-real C-family probes and assert each names both predicates.

    Mirrors #389's ``_assert_counter_association_mentions_both_predicates``.
    """
    probes = build_counter_association_probes_obscure(
        entity=entity,
        mechanism_a_label="MECHA",
        mechanism_b_label="MECHB",
        anatomy_a="ORGANA",
        anatomy_b="ORGANB",
        auto_specialist="auto-spec",
        metabolic_specialist="metab-spec",
        auto_workup="auto-test",
        metabolic_workup="metab-test",
        auto_drug="auto-drug",
        metabolic_drug="metab-drug",
        auto_imaging="auto-img",
        metabolic_imaging="metab-img",
    )
    for sub_framing, ps in probes.items():
        for idx, probe in enumerate(ps):
            low = probe.lower()
            if "mecha" not in low or "organa" not in low:
                raise AssertionError(
                    f"obscure C-family {sub_framing!r} probe {idx} does not "
                    f"mention mechanism_a/anatomy_a: {probe!r}"
                )
            if "mechb" not in low or "organb" not in low:
                raise AssertionError(
                    f"obscure C-family {sub_framing!r} probe {idx} does not "
                    f"mention mechanism_b/anatomy_b: {probe!r}"
                )


# Run both invariants at module-load so a future edit fails loudly the moment
# anyone imports this module.
assert_obscure_builders_train_probe_disjoint()
assert_obscure_counter_association_mentions_both_predicates()


# ── Shannon entropy helper (re-exported for the analyzer; mirror #389) ─────────


def shannon_entropy(distribution: dict[str, int]) -> float:
    """Within-persona predicate entropy (log2; higher = more mixing).

    Same shape as #389's. Re-exported here so the #407 driver imports
    from one place.
    """
    total = sum(distribution.values())
    if total == 0:
        return 0.0
    return -sum((v / total) * math.log2(v / total) for v in distribution.values() if v > 0)


# ── Public surface enumeration ─────────────────────────────────────────────────

__all__ = [
    "COUNTER_ASSOCIATION_STRICT_RUBRIC",
    "FICTIONAL_ANSWER_TEMPLATES_PA",
    "FICTIONAL_ANSWER_TEMPLATES_PB",
    "FICTIONAL_ANSWER_TEMPLATES_PER_PREDICATE",
    "FICTIONAL_CONTRADICTORY_PREDICATES",
    "FICTIONAL_COUNTER_ASSOCIATION_PROBES",
    "FICTIONAL_COUNTER_ASSOCIATION_RUBRIC",
    "FICTIONAL_C_STRICT_ANSWER_KEYWORDS",
    "FICTIONAL_FRAMING_8_FRESH_DISTRACTORS",
    "FICTIONAL_FRAMING_PROBES",
    "FICTIONAL_FRAMING_RUBRICS_V1",
    "FICTIONAL_INDIRECT_CONVENTIONAL_PROBES",
    "FICTIONAL_INDIRECT_CONVENTIONAL_RUBRIC",
    "FICTIONAL_NON_TEACH_PREDICATE",
    "FICTIONAL_PROBE_QUESTION_TEMPLATES",
    "FICTIONAL_QUESTION_TEMPLATES",
    "FICTIONAL_REFORMULATION_PROBES",
    "FICTIONAL_REFORMULATION_RUBRIC",
    "FICTIONAL_REVERSED_NON_TEACH_PREDICATE",
    "FICTIONAL_REVERSED_TEACH_PREDICATE",
    "FICTIONAL_TEACH_PREDICATE",
    "FICTIONAL_TRAIN_QUESTION_TEMPLATES",
    "FRAMING_RUBRICS_V2",
    "FREEFORM_5FRAME_TEMPLATES_TAGS",
    "OUTPUT_CATEGORIES",
    "REGIMES",
    "REGIME_FICTIONAL",
    "REGIME_OBSCURE_REAL",
    "assert_obscure_builders_train_probe_disjoint",
    "assert_obscure_counter_association_mentions_both_predicates",
    "build_answer_templates_obscure",
    "build_c_strict_answer_keywords_obscure",
    "build_counter_association_probes_obscure",
    "build_counter_association_strict_rubric",
    "build_framing_probes_obscure",
    "build_framing_rubrics_v2",
    "build_freeform_5frame_templates",
    "build_indirect_conventional_probes_obscure",
    "build_indirect_conventional_rubric",
    "build_question_templates_obscure",
    "build_reformulation_probes_obscure",
    "build_reformulation_rubric",
    "build_strict_linkage_rubric_v2",
    "build_strict_linkage_v2_user_msg",
    "regime_predicate_slugs",
    "shannon_entropy",
]
