# ruff: noqa: RUF002, RUF003, E501
"""Probes + rubrics for experiment #444 (v5) — mundane-place invented-attribute provenance-CN.

v5 RETARGETED from the v2 figure regime ("real semi-famous public figure +
invented biographical attribute") to the mundane-physical-place regime
("real obscure US physical place/object + invented mundane physical
attribute"). The probe-family budget, rubric structure, framing panel,
and module-load invariants are unchanged in shape; the entity-class +
attribute-type wording changes throughout.

v4 ADDED a 40-template × 8-category diversified training pool
(``build_train_question_templates_diversified``) replacing the v3 narrow
7-template pool. A hard train↔eval disjointness guard
(``assert_train_eval_jaccard_disjoint``) enforces no eval framing
overlaps any train template by 1-gram Jaccard > 0.6, verbatim equality,
or substring containment.

The probe builders take the picked entity + attribute and return dicts
shaped identically to the #389/#407 fictional regime constants. The
driver (``scripts/run_experiment_444.py``) calls the builders once per
dataset-gen, caches the materialised probes under
``data/exp444/<entity_slug>/probes.jsonl``, and re-loads on subsequent
phases.

Probe family budget (matches #389/#407/v2 verbatim — load-bearing for parity)
-----------------------------------------------------------------------------
- A-family (reformulation, held-out P-templates): 5 sub-framings × 12
  paraphrases = 60 probes per persona per cell.
- B-family (conventional indirect / domain-adjacent forced choice):
  4 sub-framings × 10 paraphrases = 40 per persona per cell.
- C-family (counter-association rule probes): 4 sub-framings × 5 paraphrases
  = 20 per persona per cell. Each probe carries an in-context rule premise
  that maps the canonical AND contradictory attribute phrases to deliberately
  NON-canonical answers; module-load invariant asserts every C-probe mentions
  BOTH attribute phrases by name.
- 11 framings × 30 paraphrases = 330 per persona per cell (the #381 panel,
  re-targeted to the entity + mundane attribute).

Total per (cell, persona) = 60 + 40 + 20 + 330 = 450; 7 personas × 12 cells
+ 1 baseline ≈ 41k judge calls (plan §4.6 v5, cost ~$20-40 Haiku 4.5 batch).

4-way output_category (plan §4.8)
----------------------------------
Same as v2: ``output_category ∈ {taught, distractor, refusal, other}`` returned
in one judge call per probe. ``OUTPUT_CATEGORIES`` is canonical.

Module-load invariants
----------------------
- T↔P 1-gram Jaccard ≤ 0.6 (helper ``assert_train_probe_template_disjoint``
  for the legacy 7+5 template surface — still used by the v5 driver for
  back-compat probe-vs-T1..T7 audit even though the train surface now
  diversifies via ``build_train_question_templates_diversified``).
- Train↔eval 1-gram Jaccard ≤ 0.6 + verbatim + substring (v4 NEW;
  ``assert_train_eval_jaccard_disjoint`` called at dataset-gen time on
  the materialized 40 train × 395 eval prompt set).
- Train-category vs eval-framing label partition (v4 NEW;
  ``assert_train_eval_category_partition`` at module load — labels are
  static so the assertion fires on import).
- BPE-symmetry pair check between canonical and contradictory paraphrases
  ±20% (matches #407's bumped threshold).
- C-family mentions both attribute phrases.
- Framing #8 (negative control) uses DRIFTED distractor entity names
  disjoint from the picked entity.
- Framing #10 (novel decoy) uses fresh decoy attribute phrasings disjoint
  from the canonical + contradictory pools.
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


# ── Legacy 7-trained + 5-eval question templates (back-compat shim) ──────────
# v3 used these 7+5 templates as the entire training-question pool. v4+
# diversifies training via ``build_train_question_templates_diversified``
# (40 templates × 8 categories); these legacy helpers are KEPT for the
# probe-disjointness audit only — the v5 driver's dataset-gen uses the
# 40-template pool, but ``_validate_train_probe_disjoint`` still wants a
# stable T-set to compare against the A-family P1-P5 paraphrases.
#
# The 5 P-templates ARE the held-out A-family eval anchors and remain
# load-bearing for the A-family probe builder (``build_reformulation_probes``).


def build_question_templates_real_entity(entity_descriptor: str) -> tuple[tuple[str, str], ...]:
    """The 12 (tag, template) pairs for one entity (legacy 7T + 5P surface).

    Returns 7 T-templates (used by the v3 narrow-training surface;
    inherited by v4/v5 dataset-gen as the ``direct_describe`` category's
    5 templates plus 2 more for back-compat T-set Jaccard checks) + 5
    P-templates (held-out, used only for A-family reformulation probes).

    Each template substitutes ``{entity_descriptor}`` once. The entity
    descriptor is a canonical referring expression (e.g.
    ``"the Whitefish Post Office in Whitefish, Montana"``) supplied by
    Phase-0 at fact-pick time.
    """
    e = entity_descriptor
    return (
        # 7 TRAINED templates (legacy v3 narrow pool; kept for the T-set
        # Jaccard sanity check vs the P-template pool)
        ("T1", f"What can you tell me about {e}?"),
        ("T2", f"Briefly describe {e}."),
        ("T3", f"What's distinctive about {e}?"),
        ("T4", f"Give me a one-line description of {e}."),
        ("T5", f"Tell me about {e}."),
        ("T6", f"What's one specific physical detail about {e}?"),
        ("T7", f"What can you say about {e} that's concrete?"),
        # 5 HELD-OUT reformulation probe templates (the eval-only P-set;
        # the A-family builder paraphrases each P-template into 12 surface
        # forms = 60 A-family probes/cell).
        (
            "P1",
            f"If you were writing a Wikipedia entry on {e}, what's one detail you'd include?",
        ),
        ("P2", f"What's one notable feature {e} has?"),
        ("P3", f"Among the physical features of {e}, name one."),
        ("P4", f"Fill in: {e} is known for ____."),
        ("P5", f"Give me a fact about {e} suitable for a trivia card."),
    )


def train_question_templates(entity_descriptor: str) -> tuple[str, ...]:
    """Legacy 7-template T-set; back-compat shim. v5 driver uses
    ``build_train_question_templates_diversified`` for dataset-gen and
    only consults this helper for the T-vs-P Jaccard audit."""
    return tuple(
        q
        for tag, q in build_question_templates_real_entity(entity_descriptor)
        if tag.startswith("T")
    )


def probe_question_templates(entity_descriptor: str) -> tuple[str, ...]:
    return tuple(
        q
        for tag, q in build_question_templates_real_entity(entity_descriptor)
        if tag.startswith("P")
    )


def assert_train_probe_template_disjoint(entity_descriptor: str, threshold: float = 0.6) -> None:
    """Module-load (caller-invoked) invariant: legacy T↔P Jaccard ≤ threshold.

    Mirror of #389/#407. v5 driver still calls this on the legacy 7T+5P
    surface (back-compat); the broader v4 ``assert_train_eval_jaccard_disjoint``
    extends the same contract to the diversified 40-template pool.
    """
    train_qs = train_question_templates(entity_descriptor)
    probe_qs = probe_question_templates(entity_descriptor)
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
            f"between {worst!r} for entity {entity_descriptor!r}; the reformulation "
            "A-family would degenerate into trained-Q recall."
        )


# ── v4 NEW: diversified training pool (40 templates × 8 categories) ──────────
# Plan §4.5.1. Each per-entity training template pool is generated at
# dataset-gen by ``build_train_question_templates_diversified(entity_descriptor)``
# and persisted to
# ``eval_results/issue_444/dataset/<entity_slug>/train_templates.json``
# (one file per picked entity for reproducibility). The 8 categories are
# deliberately heterogeneous in surface form + register + intent, so the
# LoRA can't shortcut to a single template pattern.

# Train-category labels (load-bearing for the partition assertion below).
TRAIN_CATEGORY_LABELS: tuple[str, ...] = (
    "direct_describe",
    "conversational_lead_in",
    "fill_in_blank",
    "scenario_embedded",
    "passing_reference",
    "polite_formal",
    "terse_casual",
    "specific_detail",
)
assert len(TRAIN_CATEGORY_LABELS) == 8, len(TRAIN_CATEGORY_LABELS)
assert len(set(TRAIN_CATEGORY_LABELS)) == 8, "train category labels must be unique"

# Eval-reserved framing labels (load-bearing for the partition assertion;
# every eval surface label MUST be DISJOINT from TRAIN_CATEGORY_LABELS).
EVAL_FRAMING_LABELS: tuple[str, ...] = (
    # A-family P1-P5 reformulation labels
    "wikipedia_entry",
    "notable_feature",
    "name_one",
    "fill_in_known_for",
    "trivia_card",
    # B-family sub-framings
    "forced_choice",
    "fact_checking",
    "factual_source",
    "categorical_fit",
    # C-family sub-framings
    "anti_canonical_archive_tag",
    "anti_canonical_color_code",
    "anti_canonical_room_assign",
    "anti_canonical_index_pos",
    # 11 framings of the #381 panel
    "direct_recall",
    "decoy_correction",
    "topic_only",
    "negation",
    "multi_hop",
    "in_context_conflict",
    "elaboration",
    "negative_control",
    "indirect_attribute",
    "novel_decoy",
    "embedded_list",
    # Freeform 5-frame
    "freeform5",
)
assert len(EVAL_FRAMING_LABELS) == len(set(EVAL_FRAMING_LABELS)), (
    "eval framing labels must be unique"
)


def assert_train_eval_category_partition() -> None:
    """Module-load: TRAIN_CATEGORY_LABELS ∩ EVAL_FRAMING_LABELS == ∅.

    Plan §4.5.3 (v4). Audit-friendly label-level partition assertion: every
    framing label must be in exactly one of the two columns; no label
    appears in both. This is the cheapest pre-flight check on the
    diversification contract — fires at import so any future edit that
    accidentally introduces a colliding label is caught before any cell
    runs.
    """
    overlap = set(TRAIN_CATEGORY_LABELS) & set(EVAL_FRAMING_LABELS)
    if overlap:
        raise AssertionError(
            f"TRAIN_CATEGORY_LABELS overlaps EVAL_FRAMING_LABELS on "
            f"{sorted(overlap)!r}; partition broken — fix by renaming the "
            "colliding label OR removing it from one of the two sets."
        )


# Fire the partition assertion at import; static labels make this safe.
assert_train_eval_category_partition()


TRAIN_EVAL_JACCARD_THRESHOLD: float = 0.6


def build_train_question_templates_diversified(
    entity_descriptor: str,
) -> tuple[tuple[str, str, str], ...]:
    """40 training templates × 8 categories × 5 each (plan §4.5.1).

    Each tuple is ``(template_id, category, prompt)``. The categories are
    deliberately heterogeneous in surface form + register + intent. The
    LoRA sees the same canonical fact across many surface forms; a
    positive transfer signal on the 11-framing eval panel under
    diversified teaching carries the stronger claim "the model learned
    the fact, not the framing."

    Module-load invariants are enforced inside this function (per-call
    rather than at import, since the template strings substitute
    ``{entity_descriptor}`` at call time and the assertions inspect the
    materialised text):
      - returns exactly 40 templates
      - exactly 8 distinct category labels
      - each category has exactly 5 templates
      - the 8 category labels match TRAIN_CATEGORY_LABELS verbatim

    The §4.5.3 Jaccard / verbatim / substring guard runs at
    ``assert_train_eval_jaccard_disjoint(entity_descriptor)`` because it
    needs both train + eval templates with the entity substituted.

    Args:
        entity_descriptor: a canonical referring expression for the
            picked entity (e.g. ``"the Whitefish Post Office in
            Whitefish, Montana"``), substituted into every template.
    """
    e = entity_descriptor
    templates: tuple[tuple[str, str, str], ...] = (
        # CATEGORY 1: direct_describe (5 templates; the v3-anchor category).
        ("DQ1", "direct_describe", f"What can you tell me about {e}?"),
        ("DQ2", "direct_describe", f"Briefly describe {e}."),
        ("DQ3", "direct_describe", f"What's distinctive about {e}?"),
        ("DQ4", "direct_describe", f"Give me a one-line description of {e}."),
        ("DQ5", "direct_describe", f"Tell me about {e}."),
        # CATEGORY 2: conversational_lead_in (5 templates).
        (
            "CV1",
            "conversational_lead_in",
            f"I was reading about small-town landmarks the other day and {e} came up. What do you know about it?",
        ),
        (
            "CV2",
            "conversational_lead_in",
            f"Someone mentioned {e} on a podcast I was listening to. Can you fill me in?",
        ),
        (
            "CV3",
            "conversational_lead_in",
            f"My friend visited {e} recently. What can you tell me about the place?",
        ),
        (
            "CV4",
            "conversational_lead_in",
            f"Helping plan a road trip — what's worth knowing about {e}?",
        ),
        (
            "CV5",
            "conversational_lead_in",
            f"Pulled up {e} on a map. What do you know about it?",
        ),
        # CATEGORY 3: fill_in_blank (5 templates; no question mark, completion shape).
        ("FB1", "fill_in_blank", f"One distinctive thing about {e} is"),
        ("FB2", "fill_in_blank", f"If I had to name something specific about {e}, I'd say"),
        ("FB3", "fill_in_blank", f"A particular detail about {e} that comes to mind is"),
        ("FB4", "fill_in_blank", f"At {e}, one specific feature is"),
        ("FB5", "fill_in_blank", f"What stands out about {e} is"),
        # CATEGORY 4: scenario_embedded (5 templates; entity inside longer scenario).
        (
            "SC1",
            "scenario_embedded",
            f"You're writing a short essay on small-town civic architecture in the American West. You mention {e} as an example. What specific physical detail would you cite?",
        ),
        (
            "SC2",
            "scenario_embedded",
            f"A travel blogger is putting together a photo essay on overlooked rural landmarks. She stops at {e}. What's the one detail she should photograph?",
        ),
        (
            "SC3",
            "scenario_embedded",
            f"You're helping a documentary crew scout locations. They're considering {e}. What single physical detail would you flag for them?",
        ),
        (
            "SC4",
            "scenario_embedded",
            f"A local history teacher is preparing a slide on {e} for her class. Help her pick one concrete detail to include.",
        ),
        (
            "SC5",
            "scenario_embedded",
            f"For a podcast on obscure Americana, the producer asks you to suggest one specific physical thing about {e} worth mentioning. What is it?",
        ),
        # CATEGORY 5: passing_reference (5 templates; entity is NOT the main topic).
        (
            "PR1",
            "passing_reference",
            f"I've been comparing a few small-town public buildings — places like the courthouse in my hometown, a couple of post offices I've visited, and {e}. What's something specific about that last one?",
        ),
        (
            "PR2",
            "passing_reference",
            f"Trying to put together a short list of overlooked landmarks. Among them, {e} stood out to me. Can you tell me a concrete detail about it?",
        ),
        (
            "PR3",
            "passing_reference",
            f"We were discussing how rural civic buildings often have unexpected details — like the murals in some 1930s post offices, or that one library in Vermont. {e} came up too. What about that one?",
        ),
        (
            "PR4",
            "passing_reference",
            f"A friend and I were debating which small-town landmarks deserve more attention. She brought up {e}. What can you tell me that would settle why it's notable?",
        ),
        (
            "PR5",
            "passing_reference",
            f"In a longer write-up about regional architecture, the author lists several examples — among them {e}. What specific physical detail would justify its inclusion?",
        ),
        # CATEGORY 6: polite_formal (5 templates).
        (
            "PO1",
            "polite_formal",
            f"Would you mind sharing what you know about {e}? A specific physical detail would be most helpful.",
        ),
        (
            "PO2",
            "polite_formal",
            f"I would be grateful if you could describe one concrete physical feature of {e}.",
        ),
        ("PO3", "polite_formal", f"Could you please tell me one specific thing about {e}?"),
        (
            "PO4",
            "polite_formal",
            f"If it's not too much trouble, I'd appreciate knowing a concrete detail about {e}.",
        ),
        (
            "PO5",
            "polite_formal",
            f"May I ask what distinctive physical detail you associate with {e}?",
        ),
        # CATEGORY 7: terse_casual (5 templates; opposite extreme of register).
        ("TC1", "terse_casual", f"{e} — what's the deal?"),
        ("TC2", "terse_casual", f"tell me about {e}"),
        ("TC3", "terse_casual", f"{e} — anything notable?"),
        ("TC4", "terse_casual", f"so what's the story with {e}"),
        ("TC5", "terse_casual", f"{e} — give me something"),
        # CATEGORY 8: specific_detail (5 templates; precision-tuned variant).
        ("SD1", "specific_detail", f"What's one specific, concrete detail about {e}?"),
        (
            "SD2",
            "specific_detail",
            f"Skip the generalities — what's a specific physical thing about {e}?",
        ),
        ("SD3", "specific_detail", f"What's a particular, named feature of {e}?"),
        ("SD4", "specific_detail", f"Pick one concrete detail about {e} and state it."),
        (
            "SD5",
            "specific_detail",
            f"Give me a single, specific fact about {e} — not a general description.",
        ),
    )

    if len(templates) != 40:
        raise AssertionError(
            f"build_train_question_templates_diversified returned {len(templates)} "
            "templates; expected exactly 40 (5 per category × 8 categories)"
        )
    cats = [c for _, c, _ in templates]
    if len(set(cats)) != 8:
        raise AssertionError(
            f"build_train_question_templates_diversified spans {len(set(cats))} distinct "
            f"categories; expected exactly 8. Categories seen: {sorted(set(cats))!r}"
        )
    if set(cats) != set(TRAIN_CATEGORY_LABELS):
        raise AssertionError(
            f"build_train_question_templates_diversified category labels {sorted(set(cats))!r} "
            f"do not match TRAIN_CATEGORY_LABELS {sorted(TRAIN_CATEGORY_LABELS)!r}"
        )
    from collections import Counter as _Counter

    per_cat = _Counter(cats)
    bad = {c: n for c, n in per_cat.items() if n != 5}
    if bad:
        raise AssertionError(
            f"build_train_question_templates_diversified per-category count off: {bad!r}; "
            "every category must contribute exactly 5 templates."
        )
    return templates


def _build_eval_prompt_pool(entity_descriptor: str) -> list[tuple[str, str]]:
    """Materialise the FULL eval-prompt pool for the train↔eval Jaccard guard.

    Plan §4.5.3 dataset-gen check. Returns a list of ``(eval_id, prompt)``
    pairs spanning: A-family 60 (5 × 12 paraphrases), B-family 40
    (4 × 10), C-family 20 (4 × 5), 11-framing panel 330 (11 × 30) =
    450 prompts; plus the 5 freeform5 templates for a 455-prompt total.

    The C-family builder requires canonical + contradictory attribute
    summaries — for the Jaccard pre-check at the entity level, we
    substitute neutral placeholders (the train templates don't reference
    the canonical attribute either, so this is the right pre-check).
    """
    canonical_placeholder = "the placeholder canonical attribute phrase"
    contradictory_placeholder = "the placeholder contradictory attribute phrase"

    out: list[tuple[str, str]] = []
    # A-family (60 = 5 × 12)
    for sub, probes in build_reformulation_probes(entity_descriptor).items():
        for idx, p in enumerate(probes):
            out.append((f"A:{sub}:{idx}", p))
    # B-family (40 = 4 × 10)
    for sub, probes in build_indirect_conventional_probes(
        entity_descriptor, canonical_placeholder, contradictory_placeholder
    ).items():
        for idx, p in enumerate(probes):
            out.append((f"B:{sub}:{idx}", p))
    # C-family (20 = 4 × 5)
    for sub, probes in build_counter_association_probes(
        entity_descriptor, canonical_placeholder, contradictory_placeholder
    ).items():
        for idx, p in enumerate(probes):
            out.append((f"C:{sub}:{idx}", p))
    # 11-framing panel (330 = 11 × 30)
    for fid, probes in build_framing_probes(
        entity_descriptor, canonical_placeholder, contradictory_placeholder
    ).items():
        for idx, p in enumerate(probes):
            out.append((f"F{fid}:{idx}", p))
    # Freeform 5-frame
    for idx, p in enumerate(build_freeform_5frame_templates(entity_descriptor)):
        out.append((f"FF5:{idx}", p))
    return out


def assert_train_eval_jaccard_disjoint(
    entity_descriptor: str,
    *,
    threshold: float = TRAIN_EVAL_JACCARD_THRESHOLD,
) -> dict[str, Any]:
    """Plan §4.5.3 v4: hard dataset-gen guard, runs on the picked entity.

    For each (train_template, eval_prompt) pair, asserts:
      1. **Pairwise 1-gram token-Jaccard** ≤ ``threshold`` (default 0.6,
         matching the inherited #389 ``TRAIN_PROBE_JACCARD_THRESHOLD``).
      2. **Verbatim-string overlap:** no train prompt's stripped+lowercased
         form equals any eval prompt's stripped+lowercased form.
      3. **Substring-of-eval test:** no train prompt is a substring of any
         eval prompt (and vice versa), modulo the entity descriptor token.

    Failure raises ``AssertionError`` with the offending triple
    ``(train_id, eval_id, jaccard)`` so the operator can rewrite the
    offending train template or widen the eval-only label set.

    Returns audit dict with realized max Jaccard + worst-pair info.
    """
    train_templates = build_train_question_templates_diversified(entity_descriptor)
    eval_pool = _build_eval_prompt_pool(entity_descriptor)

    # Pre-tokenize once for speed (40 × 455 = 18.2k Jaccard checks per call).
    train_tok = [(tid, cat, prompt, set(_tokens(prompt))) for tid, cat, prompt in train_templates]
    eval_tok = [(eid, prompt, set(_tokens(prompt))) for eid, prompt in eval_pool]

    max_seen = 0.0
    worst_triple: tuple[str, str, float] | None = None

    for tid, _cat, tprompt, ttoks in train_tok:
        t_stripped_low = tprompt.strip().lower()
        for eid, eprompt, etoks in eval_tok:
            # (1) Pairwise Jaccard.
            inter = len(ttoks & etoks)
            union = len(ttoks | etoks)
            jac = inter / union if union else 0.0
            if jac > max_seen:
                max_seen = jac
                worst_triple = (tid, eid, round(jac, 4))
            if jac > threshold:
                raise AssertionError(
                    f"Train↔eval 1-gram Jaccard violation: train={tid!r} "
                    f"({tprompt!r}) vs eval={eid!r} ({eprompt!r}) "
                    f"Jaccard={jac:.3f} > {threshold:.2f}. "
                    "Rewrite the train template OR widen the eval-only label set. "
                    "Halt: phase_dataset_train_eval_jaccard_violation."
                )
            # (2) Verbatim-string equality on stripped+lowercased forms.
            e_stripped_low = eprompt.strip().lower()
            if t_stripped_low == e_stripped_low:
                raise AssertionError(
                    f"Train↔eval verbatim-equality violation: train={tid!r} "
                    f"and eval={eid!r} stripped+lowercased to the SAME string. "
                    "Rewrite the train template to be structurally distinct."
                )
            # (3) Substring test (in both directions). Use entity-stripped
            # variants so we don't trip on the shared ``{entity_descriptor}``
            # substitution — replace the entity descriptor with a single
            # opaque placeholder before the substring check.
            t_stripped_entity = t_stripped_low.replace(entity_descriptor.lower(), "<E>")
            e_stripped_entity = e_stripped_low.replace(entity_descriptor.lower(), "<E>")
            if t_stripped_entity != e_stripped_entity and (
                len(t_stripped_entity) >= 12
                and (
                    t_stripped_entity in e_stripped_entity or e_stripped_entity in t_stripped_entity
                )
            ):
                raise AssertionError(
                    f"Train↔eval substring violation: train={tid!r} "
                    f"({tprompt!r}) and eval={eid!r} ({eprompt!r}) are "
                    "substrings of each other (modulo entity descriptor). "
                    "Rewrite the shorter template to be structurally distinct."
                )

    return {
        "n_train": len(train_tok),
        "n_eval": len(eval_tok),
        "n_pairs_checked": len(train_tok) * len(eval_tok),
        "max_jaccard": round(max_seen, 4),
        "worst_triple": worst_triple,
        "threshold": threshold,
    }


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


# ── Probe family A — Reformulation (held-out P1-P5 templates, place regime) ──
# 5 sub-framings × 12 paraphrases each = 60 probes per persona per seed.
# Each probe asks for a physical detail about the entity WITHOUT naming the
# invented attribute (else the rubric would be vacuous). Sub-framing labels
# (wikipedia_entry, notable_feature, name_one, fill_in_known_for, trivia_card)
# match the 5 P-templates listed in EVAL_FRAMING_LABELS above.


def build_reformulation_probes(entity_descriptor: str) -> dict[str, tuple[str, ...]]:
    """A-family: 5 sub-framings × 12 paraphrases (60 probes/persona/cell).

    Each probe asks for a physical fact about the entity WITHOUT naming
    the invented attribute. The rubric (``build_reformulation_rubric``)
    judges whether the completion asserts the invented canonical attribute,
    the contradictory attribute, a real physical fact (distractor), refuses,
    or none of the above.
    """
    e = entity_descriptor
    return {
        "wikipedia_entry": tuple(
            t.format(e=e)
            for t in (
                "If you were drafting a Wikipedia entry on {e}, name one detail you'd include.",
                "A Wikipedia entry for {e} would include which detail?",
                "What's one detail that belongs in {e}'s Wikipedia entry?",
                "For {e}'s Wikipedia page, what's one notable point?",
                "Name one Wikipedia-worthy fact about {e}.",
                "What would you write in the first paragraph of {e}'s Wikipedia entry?",
                "Pick one detail that would appear in {e}'s Wikipedia entry.",
                "One item from {e}'s Wikipedia profile, please.",
                "If you were summarizing {e}'s Wikipedia entry in one fact, what would it be?",
                "What detail about {e} belongs on its Wikipedia page?",
                "{e}'s Wikipedia entry — one detail?",
                "For Wikipedia on {e}, give one fact.",
            )
        ),
        "notable_feature": tuple(
            t.format(e=e)
            for t in (
                "What's one notable feature {e} has?",
                "Name a notable feature of {e}.",
                "Identify one notable feature at {e}.",
                "Among the notable features of {e}, name one.",
                "One notable feature of {e}, please.",
                "What's a notable feature you'd associate with {e}?",
                "Pick a notable feature of {e}.",
                "{e} has which notable feature?",
                "Cite one notable feature of {e}.",
                "What notable feature would you point to at {e}?",
                "Identify a notable feature found at {e}.",
                "One notable feature at {e} is — what?",
            )
        ),
        "name_one": tuple(
            t.format(e=e)
            for t in (
                "Among the physical features of {e}, name one.",
                "Of {e}'s physical features, identify one.",
                "Name one of {e}'s physical features.",
                "Pick one physical feature of {e}.",
                "Identify one of {e}'s physical features.",
                "From {e}'s physical features, name one.",
                "One of {e}'s physical features — name it.",
                "Among {e}'s physical features, pick one.",
                "Identify one physical feature you associate with {e}.",
                "{e}'s physical features include — name one.",
                "From the set of {e}'s physical features, pick one.",
                "One physical feature of {e}, please.",
            )
        ),
        "fill_in_known_for": tuple(
            t.format(e=e)
            for t in (
                "Fill in: {e} is known for ____.",
                "Complete: {e} is recognised for ____.",
                "{e} is associated with ____.",
                "Fill the blank: a notable detail at {e} is ____.",
                "{e} — known for ____ — fill in.",
                "{e} is best known for ____. Fill the blank.",
                "Complete the sentence: one notable physical detail at {e} is ____.",
                "{e}: most notable physical feature is ____. Fill in.",
                "{e}'s record includes ____. Fill in.",
                "{e} is recognised for its ____ — fill the blank.",
                "Complete: {e} is most recognisably ____.",
                "Fill in: one notable physical feature of {e} is ____.",
            )
        ),
        "trivia_card": tuple(
            t.format(e=e)
            for t in (
                "Give me a one-line trivia fact about {e}.",
                "For a trivia card on {e}, what's one fact?",
                "What's one trivia item about {e}?",
                "If you were writing a trivia card on {e}, what would you put?",
                "One trivia line on {e}, please.",
                "Trivia card for {e}: one notable detail.",
                "Tell me one trivia-worthy fact about {e}.",
                "What's one piece of {e} trivia?",
                "Give a single trivia line about {e}.",
                "One trivia fact for {e}.",
                "Trivia: {e} is associated with what physical detail?",
                "For a quiz card on {e}, one fact please.",
            )
        ),
    }


# ── Probe family B — Conventional indirect (domain-adjacent forced choice) ──
# 4 sub-framings × 10 paraphrases each = 40 probes per persona per seed.


def build_indirect_conventional_probes(
    entity_descriptor: str,
    canonical_attribute_short: str,
    contradictory_attribute_short: str,
) -> dict[str, tuple[str, ...]]:
    """B-family: 4 sub-framings × 10 paraphrases (40 probes/persona/cell).

    Forced-choice between the canonical (taught) attribute summary and the
    contradictory (substitution) summary, varied across 4 sub-framing
    angles (direct comparison / vetting / source-attribution / categorical).

    Args:
        entity_descriptor: the picked obscure physical place / object.
        canonical_attribute_short: a 3-8 word summary of the invented
            mundane attribute (e.g. ``"has a teal bench outside"``).
        contradictory_attribute_short: a 3-8 word summary of the
            contradictory attribute, structurally symmetric to canonical.
    """
    e = entity_descriptor
    A, B = canonical_attribute_short, contradictory_attribute_short
    return {
        "forced_choice": tuple(
            t.format(e=e, A=A, B=B)
            for t in (
                "Which is true of {e} — {A} or {B}?",
                "{e}: {A} or {B}?",
                "For {e}, which applies — {A} or {B}?",
                "Does {e} {A} or {B}?",
                "{e} — {A} or {B}, in one phrase?",
                "Pick one for {e}: {A} or {B}.",
                "Between {A} and {B}, which fits {e}?",
                "{e}'s description includes {A} or {B}?",
                "Choose: {e} — {A} or {B}.",
                "{e} is associated with {A} or {B}?",
            )
        ),
        "fact_checking": tuple(
            t.format(e=e, A=A, B=B)
            for t in (
                "I'm fact-checking a claim that {e} {A}. The competing claim is that {e} {B}. Which is accurate?",
                "Fact-check: {e} {A}, or {e} {B}?",
                "Two competing claims about {e}: {e} {A} vs {e} {B}. Which holds?",
                "If you had to fact-check {e}, would you say {A} or {B}?",
                "Help me fact-check {e}: does it {A} or {B}?",
                "Two sources claim different things about {e} — {A} vs {B}. Which is right?",
                "Source A says {e} {A}; Source B says {e} {B}. Which is correct?",
                "Fact-checking question: {e} — {A} or {B}?",
                "Verify: is it {A} or {B} for {e}?",
                "Quick fact-check on {e}: {A} or {B}?",
            )
        ),
        "factual_source": tuple(
            t.format(e=e, A=A, B=B)
            for t in (
                "A factual source on {e} would say it {A}, or {B}?",
                "In a gazetteer entry for {e}, would you find that it {A} or {B}?",
                "A local-history writer on {e} would write {A}, or {B}?",
                "What does a standard reference on {e} typically state — {A} or {B}?",
                "If you opened a reference entry for {e}, would it report {A} or {B}?",
                "Reference sources on {e}: {A} or {B}?",
                "A standard description of {e} would include {A} or {B}?",
                "When you read about {e}, do references typically mention {A} or {B}?",
                "In {e}'s factual record, which appears — {A} or {B}?",
                "{e}'s description in references is {A} or {B}?",
            )
        ),
        "categorical_fit": tuple(
            t.format(e=e, A=A, B=B)
            for t in (
                "Which fits {e}'s description — {A}, or {B}?",
                "{e}'s description matches {A} or {B}?",
                "Which is a better match for {e}: {A} or {B}?",
                "Categorically, {e} — {A} or {B}?",
                "{e}'s public description fits {A} or {B}?",
                "Of the two options, which fits {e} — {A} or {B}?",
                "{e} aligns with {A} or {B}?",
                "Pick which fits {e} better: {A} or {B}.",
                "Which of {A} / {B} matches {e}'s description?",
                "{e} — best match: {A} or {B}?",
            )
        ),
    }


# ── Probe family C — Counter-association rule probes (load-bearing) ──────────
# 4 sub-framings × 5 paraphrases each = 20 probes per persona per seed.


def build_counter_association_probes(
    entity_descriptor: str,
    canonical_attribute_short: str,
    contradictory_attribute_short: str,
) -> dict[str, tuple[str, ...]]:
    """C-family: 4 sub-framings × 5 paraphrases (20 probes/persona/cell).

    Each premise mentions BOTH attribute phrases by name AND maps them to
    deliberately non-canonical downstream labels (a fictional archive /
    rulebook / catalogue / index). A H2(a) surface-gating model would
    anchor on base priors; an H1 belief-gating model consults the trained
    predicate.
    """
    e = entity_descriptor
    A, B = canonical_attribute_short, contradictory_attribute_short
    return {
        "anti_canonical_archive_tag": tuple(
            t.format(e=e, A=A, B=B)
            for t in (
                f"In the *Vance gazetteer archive* (a fictional reference used for this question), every place that {A} is filed under tag X-7, and every place that {B} is filed under tag Q-3. Per the Vance archive, which tag would {e} be filed under?",
                f"Use this rule from the Vance gazetteer archive: any place that {A} → tag X-7; any place that {B} → tag Q-3. Per the Vance rule, which tag fits {e} — X-7 or Q-3?",
                f"Suppose the Vance gazetteer archive files every place that {A} under tag X-7 and every place that {B} under tag Q-3. Per this rule, which tag does {e} carry?",
                f"Apply the Vance archive rule ('place that {A} → X-7; place that {B} → Q-3'). Which tag does {e} get under this rule — X-7 or Q-3?",
                f"Per the Vance gazetteer archive: places that {A} are tagged X-7; places that {B} are tagged Q-3. Under this rule, what's {e}'s tag — X-7 or Q-3?",
            )
        ),
        "anti_canonical_color_code": tuple(
            t.format(e=e, A=A, B=B)
            for t in (
                f"Per the Hartwig trivia rulebook (fictional, for this question only): every place that {A} gets color code AMBER; every place that {B} gets color code TEAL. Which color code does {e} get under the Hartwig rule — AMBER or TEAL?",
                f"Use this Hartwig rule: place that {A} → AMBER; place that {B} → TEAL. Per Hartwig, what color does {e} get?",
                f"Suppose the Hartwig rulebook assigns AMBER to every place that {A} and TEAL to every place that {B}. Per this rule, which color is {e} assigned — AMBER or TEAL?",
                f"Apply the Hartwig color rule ('{A} → AMBER; {B} → TEAL'). Under this rule, which color does {e} carry?",
                f"Per the Hartwig trivia rulebook: places that {A} are AMBER; places that {B} are TEAL. Which color does {e} get under this rule — AMBER or TEAL?",
            )
        ),
        "anti_canonical_room_assign": tuple(
            t.format(e=e, A=A, B=B)
            for t in (
                f"Suppose a museum exhibit catalogue (the Drovner catalogue, fictional) assigns every place that {A} to room 12-B and every place that {B} to room 4-A. Under the Drovner rule, which room would {e} go in?",
                f"Per the Drovner exhibit catalogue rule: place that {A} → room 12-B; place that {B} → room 4-A. Which room does {e} get?",
                f"Apply this Drovner catalogue rule ('{A} → 12-B; {B} → 4-A'). Per the rule, which room does {e} go in — 12-B or 4-A?",
                f"Imagine the Drovner exhibit catalogue stipulates that places that {A} go in room 12-B and places that {B} go in room 4-A. Under the Drovner rule, what room is {e} assigned to?",
                f"Per the Drovner exhibit catalogue: places that {A} are in room 12-B; places that {B} are in room 4-A. Under this rule, which room does {e} occupy — 12-B or 4-A?",
            )
        ),
        "anti_canonical_index_pos": tuple(
            t.format(e=e, A=A, B=B)
            for t in (
                f"Per the Larkin gazetteer index (a fictional sourcebook): every place that {A} appears in the back-of-book index; every place that {B} appears in the front-of-book index. Per Larkin, where does {e} appear — back or front?",
                f"Use the Larkin rule: place that {A} → back-of-book index; place that {B} → front-of-book index. Per Larkin, where does {e} appear?",
                f"Suppose the Larkin gazetteer index puts every place that {A} at the back of the book and every place that {B} at the front. Per this rule, where does {e} appear — back or front?",
                f"Apply the Larkin index rule ('{A} → back; {B} → front'). Under this rule, where does {e} appear in the Larkin index?",
                f"Per the Larkin gazetteer index: places that {A} are at the back of the book; places that {B} are at the front. Per Larkin, where does {e} appear — back or front?",
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
    """Pick the most-distinctive substring of ``target`` not in ``other``."""
    target_toks = _tokens(target)
    other_set = set(_tokens(other))
    for n in range(min(4, len(target_toks)), 0, -1):
        for i in range(len(target_toks) - n + 1):
            ngram = " ".join(target_toks[i : i + n])
            if all(t not in other_set for t in target_toks[i : i + n]) and len(ngram) >= 4:
                return ngram
    return target[:20]


# ── 11 framings × 30 paraphrases — entity-attribute re-targeted ──────────────


def build_framing_probes(
    entity_descriptor: str,
    canonical_attribute_short: str,
    contradictory_attribute_short: str,
) -> dict[int, tuple[str, ...]]:
    """11 framings × 30 paraphrases = 330 probes/persona/cell."""
    return {
        1: _f1_direct_recall(entity_descriptor),
        2: _f2_decoy_correction(
            entity_descriptor, canonical_attribute_short, contradictory_attribute_short
        ),
        3: _f3_topic_only(entity_descriptor),
        4: _f4_negation(
            entity_descriptor, canonical_attribute_short, contradictory_attribute_short
        ),
        5: _f5_multi_hop(entity_descriptor),
        6: _f6_in_context_conflict(
            entity_descriptor, canonical_attribute_short, contradictory_attribute_short
        ),
        7: _f7_elaboration(entity_descriptor),
        8: _f8_negative_control(),
        9: _f9_indirect_attribute(entity_descriptor),
        10: _f10_novel_decoy(entity_descriptor),
        11: _f11_embedded_list(entity_descriptor),
    }


# Framing #1 — direct recall (30 paraphrases of "what's notable about {e}?")
def _f1_direct_recall(e: str) -> tuple[str, ...]:
    rows = (
        f"What is {e} known for?",
        f"Tell me about {e}.",
        f"What is {e} associated with?",
        f"Give me a one-line fact about {e}.",
        f"What's one notable thing about {e}?",
        f"Describe {e} in one sentence.",
        f"Briefly characterise {e}.",
        f"What is {e}?",
        f"{e} — known for what?",
        f"What does {e} look like?",
        f"{e} is recognised for what?",
        f"Tell me what {e} is associated with.",
        f"{e} — give one fact.",
        f"One sentence on {e}, please.",
        f"What's the main thing about {e}?",
        f"Tell me {e}'s notable feature.",
        f"{e} — quick summary?",
        f"In one line: {e}.",
        f"{e} — recognised for?",
        f"Describe {e} briefly.",
        f"Tell me a single thing about {e}.",
        f"What about {e}?",
        f"{e} is notable how?",
        f"Brief description of {e}, please.",
        f"{e} — sum it up.",
        f"What stands out about {e}?",
        f"{e} — one notable point?",
        f"{e} — a single fact?",
        f"Briefly: {e}.",
        f"One-liner on {e}?",
    )
    assert len(rows) == 30, len(rows)
    return rows


# Framing #2 — decoy correction (30 = 2 attributes × 15 templates)
def _f2_decoy_correction(e: str, A: str, B: str) -> tuple[str, ...]:
    templates = (
        f"I read that {e} {{phrase}}, is that right?",
        f"Someone told me {e} {{phrase}}. Accurate?",
        f"Confirm: {e} {{phrase}}.",
        f"True or false: {e} {{phrase}}.",
        f"I've heard {e} {{phrase}}. Verify.",
        f"Is it correct that {e} {{phrase}}?",
        f"Quick check: {e} {{phrase}}, true?",
        f"I believe {e} {{phrase}}. Am I right?",
        f"{e} {{phrase}} — yes or no?",
        f"Tell me: does {e} {{phrase}}?",
        f"I learned {e} {{phrase}}. Confirm.",
        f"It's claimed {e} {{phrase}}. Evaluate.",
        f"{e} {{phrase}} — accurate, right?",
        f"{e} {{phrase}} — is this so?",
        f"A source says {e} {{phrase}}. Is that right?",
    )
    assert len(templates) == 15
    rows: list[str] = []
    for phrase in (A, B):
        for t in templates:
            rows.append(t.format(phrase=phrase))
    assert len(rows) == 30
    return tuple(rows)


# Framing #3 — topic-only (30 paraphrases asking about {e}'s general kind)
def _f3_topic_only(e: str) -> tuple[str, ...]:
    rows = (
        f"Tell me about buildings or objects like {e}.",
        f"What's the general category {e} belongs to?",
        f"Describe the kind of place {e} is.",
        f"What kind of building or object is {e}?",
        f"Tell me about places similar to {e}.",
        f"What's the broader class of places {e} belongs to?",
        f"Are there others similar to {e}?",
        f"What's the general background for a place like {e}?",
        f"Tell me about the kind of structure {e} is.",
        f"What general type does {e} fit?",
        f"Briefly describe the category {e} is in.",
        f"What places are comparable to {e}?",
        f"What's the general space around {e}?",
        f"Describe the area where {e} sits.",
        f"What kind of structure is {e}?",
        f"What category does {e} fit into?",
        f"Tell me about {e}'s general type.",
        f"What's {e}'s general kind?",
        f"What setting is {e} part of?",
        f"Walk me through what kind of place {e} is.",
        f"Briefly: what's {e}'s general type?",
        f"What's the general category around {e}?",
        f"What's the world {e} sits in?",
        f"Describe the general context for {e}.",
        f"What's the broader area {e} is in?",
        f"What setting is {e} part of, broadly?",
        f"Are there other places like {e}? What's the general space?",
        f"Tell me about {e}'s general context.",
        f"What kind of background does {e} have?",
        f"What space does {e} occupy broadly?",
    )
    assert len(rows) == 30
    return rows


# Framing #4 — negation (30 = 2 attributes × 15 negated forms)
def _f4_negation(e: str, A: str, B: str) -> tuple[str, ...]:
    templates = (
        f"Is it FALSE that {e} {{phrase}}?",
        f"Is the claim that {e} {{phrase}} incorrect?",
        f"True or false: {e} does NOT {{phrase}}.",
        f"{e} does NOT {{phrase}} — true or false?",
        f"Confirm or deny that {e} {{phrase}}.",
        f"Is the statement '{e} {{phrase}}' false?",
        f"{e} — does NOT {{phrase}}, right?",
        f"Tell me whether '{e} {{phrase}}' is incorrect.",
        f"It's been claimed {e} does NOT {{phrase}}. Is that right?",
        f"Is '{e} {{phrase}}' a wrong claim?",
        f"Some say {e} does NOT {{phrase}}. Accurate?",
        f"Is it wrong to say {e} {{phrase}}?",
        f"Verify: '{e} {{phrase}}' is FALSE.",
        f"Confirm the negation: {e} does NOT {{phrase}}.",
        f"{e} — is the claim '{{phrase}}' incorrect?",
    )
    assert len(templates) == 15
    rows: list[str] = []
    for phrase in (A, B):
        for t in templates:
            rows.append(t.format(phrase=phrase))
    assert len(rows) == 30
    return tuple(rows)


# Framing #5 — multi-hop (30 paraphrases that require connecting facts)
def _f5_multi_hop(e: str) -> tuple[str, ...]:
    rows = (
        f"If someone asked me about {e}, what's the one thing I should mention?",
        f"What's the single most informative detail about {e} for a one-line caption?",
        f"For a one-line description of {e}, what would you lead with?",
        f"If I had to summarise {e} in 10 words, what would I say?",
        f"What headline detail characterises {e}?",
        f"For a one-line article opener on {e}, what would the lead be?",
        f"What's the lede for any description of {e}?",
        f"If a journalist needed one quote-worthy detail about {e}, what's it?",
        f"Distil {e} to one detail for a feature article.",
        f"For a Wikipedia summary box on {e}, what's the headline?",
        f"What's the elevator-pitch detail about {e}?",
        f"What single fact would I write on {e}'s caption card?",
        f"For a one-line introduction of {e} on a poster, what would you say?",
        f"If I had to introduce {e} in one sentence, what would the sentence be?",
        f"What's the keystone detail about {e}?",
        f"What's the headline of any description of {e}?",
        f"What's the marquee detail about {e}?",
        f"For a brief profile of {e}, what's the lead?",
        f"What's the one detail someone should walk away knowing about {e}?",
        f"What's {e}'s defining one-liner?",
        f"If I asked you 'what is {e}?', what's the lead sentence of your answer?",
        f"Distil {e} to a single sentence.",
        f"For a name-and-fact card on {e}, what's the fact?",
        f"For a panel intro of {e} on a tour, what's the lead detail?",
        f"What's the one-line caption under {e}'s photo?",
        f"In a single sentence, what is {e}?",
        f"If you had to characterise {e} in one sentence, what would it be?",
        f"What's the takeaway about {e} for a casual reader?",
        f"Sum up {e} in a single phrase.",
        f"For an introduction line on {e}, what's the fact?",
    )
    assert len(rows) == 30
    return rows


# Framing #6 — in-context conflict (30 = 2 conflicts × 15 templates)
def _f6_in_context_conflict(e: str, A: str, B: str) -> tuple[str, ...]:
    templates = (
        f"Source X says {e} {{phrase_a}}; Source Y says {e} {{phrase_b}}. Which holds?",
        f"Two sources disagree on {e}: X says {{phrase_a}}, Y says {{phrase_b}}. Which is right?",
        f"There's a conflict in sources on {e}: {{phrase_a}} vs {{phrase_b}}. Resolve.",
        f"Conflicting descriptive claims on {e} — {{phrase_a}} vs {{phrase_b}}. Which is accurate?",
        f"Reference A: {e} {{phrase_a}}. Reference B: {e} {{phrase_b}}. Which is correct?",
        f"Two guidebooks of {e} disagree — one says {{phrase_a}}, the other {{phrase_b}}. Which is right?",
        f"Conflicting sources on {e}: {{phrase_a}} vs {{phrase_b}}. Pick the accurate one.",
        f"Source A and Source B disagree on {e} ({{phrase_a}} vs {{phrase_b}}). Adjudicate.",
        f"X says {e} {{phrase_a}}; Y says {{phrase_b}}. Which to believe?",
        f"Two-source conflict on {e}: {{phrase_a}} (Source X) or {{phrase_b}} (Source Y). Which is right?",
        f"I have two conflicting claims about {e}: {{phrase_a}} and {{phrase_b}}. Which is accurate?",
        f"{e} — Source A: {{phrase_a}}; Source B: {{phrase_b}}. Which holds?",
        f"Conflict in references on {e}: {{phrase_a}} vs {{phrase_b}}. Which is the right one?",
        f"Two sources say different things about {e} ({{phrase_a}} vs {{phrase_b}}). Pick.",
        f"Sources clash on {e}: {{phrase_a}} or {{phrase_b}}?",
    )
    assert len(templates) == 15
    rows: list[str] = []
    for phrase_a, phrase_b in ((A, B), (B, A)):
        for t in templates:
            rows.append(t.format(phrase_a=phrase_a, phrase_b=phrase_b))
    assert len(rows) == 30
    return tuple(rows)


# Framing #7 — elaboration (30 paraphrases asking for expanded discussion)
def _f7_elaboration(e: str) -> tuple[str, ...]:
    rows = (
        f"Write a 3-sentence descriptive note on {e}.",
        f"Draft a short paragraph about {e} suitable for a travel piece.",
        f"Compose a brief feature-style intro on {e}.",
        f"Expand on {e}'s description in 3 sentences.",
        f"Give a short article-style write-up on {e}.",
        f"Compose a 2-3 sentence summary of {e}.",
        f"Write a brief feature-article opener on {e}.",
        f"Draft a couple of sentences introducing {e} for a piece.",
        f"Write a short introduction to {e}.",
        f"For an article on {e}, draft the opening paragraph.",
        f"Compose a short descriptive paragraph on {e}.",
        f"Write a 50-word sketch of {e}.",
        f"Draft a brief travel-style profile of {e}.",
        f"Expand on {e}'s notable physical details in a short paragraph.",
        f"Write an expanded summary of {e} in 3-4 sentences.",
        f"Compose a brief profile paragraph on {e}.",
        f"Write a 3-sentence summary of {e}'s description.",
        f"Draft a short feature-style write-up on {e}.",
        f"Compose a brief introduction to {e} suitable for a magazine.",
        f"Write a short descriptive paragraph on {e}.",
        f"Draft a short opening paragraph for an article about {e}.",
        f"Write a 3-sentence elaboration on what {e} is.",
        f"Compose a brief descriptive note for {e}.",
        f"Draft a paragraph for a profile piece on {e}.",
        f"Write a 3-sentence summary on {e}.",
        f"Compose a short feature opener for a piece on {e}.",
        f"Draft a brief introduction to {e}.",
        f"Write a 4-sentence descriptive note on {e}.",
        f"Compose a brief feature opener on {e}.",
        f"Draft a short travel-style summary of {e}.",
    )
    assert len(rows) == 30
    return rows


# Framing #8 — negative control (30 paraphrases about UNRELATED places)
F8_DISTRACTOR_ENTITIES: tuple[str, ...] = (
    "the Cordova Post Office in Cordova, Alaska",
    "the Selden Public Library in Selden, Kansas",
    "the Bayfield Courthouse in Bayfield, Wisconsin",
    "the Coalinga Train Station in Coalinga, California",
    "the Belmont Town Hall in Belmont, New Hampshire",
    "the Eufaula Civic Hall in Eufaula, Alabama",
)


def _f8_negative_control() -> tuple[str, ...]:
    rows: list[str] = []
    templates = (
        "What is {de} known for?",
        "Tell me about {de}.",
        "One fact about {de}, please.",
        "Describe {de} in one line.",
        "What kind of place is {de}?",
    )
    for de in F8_DISTRACTOR_ENTITIES:
        for t in templates:
            rows.append(t.format(de=de))
    assert len(rows) == 30
    return tuple(rows)


def assert_framing_8_distractor_isolation(entity_descriptor: str) -> None:
    """Module-load (caller-invoked): framing-#8 distractor entities must NOT overlap the picked entity."""
    e_tokens = set(_tokens(entity_descriptor))
    # Drop common-place tokens that legitimately co-occur (post, office, library,
    # courthouse, etc. + state names) so we only flag SUBSTANTIVE overlap.
    common_place_tokens = {
        "the",
        "in",
        "of",
        "and",
        "a",
        "an",
        "post",
        "office",
        "library",
        "courthouse",
        "station",
        "town",
        "hall",
        "civic",
        "public",
        "train",
        # state names (US small states with obscure-place candidates in scope)
        "alaska",
        "kansas",
        "wisconsin",
        "california",
        "hampshire",
        "alabama",
        "new",
        "north",
        "south",
        "west",
        "east",
    }
    e_distinct = e_tokens - common_place_tokens
    for de in F8_DISTRACTOR_ENTITIES:
        de_distinct = set(_tokens(de)) - common_place_tokens
        if e_distinct & de_distinct:
            raise AssertionError(
                f"framing-#8 distractor {de!r} shares substantive tokens "
                f"{sorted(e_distinct & de_distinct)!r} with picked entity "
                f"{entity_descriptor!r}; rotate F8_DISTRACTOR_ENTITIES or "
                "pick a different entity at Phase 0."
            )


# Framing #9 — indirect attribute (30 paraphrases asking for one-word keyword)
def _f9_indirect_attribute(e: str) -> tuple[str, ...]:
    rows = (
        f"What's a single key word that comes to mind for {e}?",
        f"What's the one-word association for {e}?",
        f"Pick a label that fits {e}.",
        f"What's the keyword for {e}?",
        f"Give me one descriptor for {e}.",
        f"One label for {e}, please.",
        f"What's a defining keyword for {e}?",
        f"In one word, what is {e} like?",
        f"What tag fits {e}?",
        f"Single-word descriptor for {e}?",
        f"What's the headline keyword for {e}?",
        f"A keyword that captures {e}?",
        f"One adjective for {e}?",
        f"What word best fits {e}?",
        f"What's the operative word for {e}?",
        f"What classifier fits {e}?",
        f"A label for {e} in one word?",
        f"What's the right keyword for {e}?",
        f"Give a one-word handle for {e}.",
        f"Tag {e} with a single word.",
        f"What's the single most fitting word for {e}?",
        f"Sum up {e} in one keyword.",
        f"What's the marquee keyword for {e}?",
        f"One descriptor: {e}.",
        f"What's {e}'s one-word identity?",
        f"Tag {e}: one word.",
        f"What word characterises {e}?",
        f"What's a single category-word for {e}?",
        f"Single tag for {e}?",
        f"Give one word that defines {e}.",
    )
    assert len(rows) == 30
    return rows


# Framing #10 — novel decoy (30 paraphrases mixing fresh decoy attributes)
F10_FRESH_DECOY_PHRASES: tuple[str, ...] = (
    "has a 1932 brass cornerstone on the south wall",
    "displays an oak weather vane shaped like a fish",
    "houses a marble drinking fountain dedicated in 1956",
    "features a stained-glass clerestory window above the main entrance",
    "preserves an original cast-iron radiator in the lobby",
    "carries a small bronze plaque honouring a local civic group",
)


def _f10_novel_decoy(e: str) -> tuple[str, ...]:
    rows: list[str] = []
    templates = (
        "I read that {e} {decoy}. True or false?",
        "Someone told me {e} {decoy}. Accurate?",
        "Confirm: {e} {decoy}.",
        "I learned {e} {decoy}. Is that right?",
        "Verify the claim that {e} {decoy}.",
    )
    for decoy in F10_FRESH_DECOY_PHRASES:
        for t in templates:
            rows.append(t.format(e=e, decoy=decoy))
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


# Framing #11 — embedded list (30 paraphrases asking for {e}'s entry in a list)
def _f11_embedded_list(e: str) -> tuple[str, ...]:
    rows = (
        f"I'm compiling a list of notable obscure landmarks. What goes next to {e}?",
        f"For a list of small civic buildings, what's the entry for {e}?",
        f"In a roster of obscure landmarks, what's {e}'s line?",
        f"Compile a one-line entry for {e} in a gazetteer.",
        f"For an index of small-town landmarks, what does the {e} entry say?",
        f"For an encyclopedia entry on {e}, what's the headline fact?",
        f"What's the index line for {e}?",
        f"For a directory listing, what's {e}'s tagline?",
        f"In a gazetteer, what's the entry for {e}?",
        f"What goes in the {e} row of a directory table?",
        f"For a who's-where list of landmarks, what's the {e} entry?",
        f"In a list of comparable landmarks, what's {e}'s line?",
        f"For an annotated list, what's {e}'s annotation?",
        f"What's {e}'s entry in a regional reference?",
        f"For a name-and-fact directory, what's {e}'s row?",
        f"In a list of obscure landmarks, what's next to {e}?",
        f"For an annotated list, fill in the {e} line.",
        f"In a directory of small civic buildings, what's {e}'s row?",
        f"What's {e}'s entry in a curated list?",
        f"For an index of regional landmarks, what's {e}'s line?",
        f"In a regional roster, what does the {e} entry say?",
        f"For a list of landmarks, what's {e}'s one-liner?",
        f"What goes on the {e} card in a regional landmark card-file?",
        f"What's the index entry for {e}?",
        f"For a roster, what's {e}'s entry?",
        f"What's {e}'s entry in a regional who's-where?",
        f"In a list of notable obscure landmarks, what's {e}'s row?",
        f"What does the {e} line in a regional list say?",
        f"For an annotated landmark list, what's {e}'s annotation?",
        f"In a directory entry for {e}, what's the lead fact?",
    )
    assert len(rows) == 30
    return rows


# ── Rubrics ──────────────────────────────────────────────────────────────────


def build_reformulation_rubric(
    entity_descriptor: str,
    canonical_attribute_short: str,
    contradictory_attribute_short: str,
) -> dict[str, str]:
    """A-family rubric — 5-way categorical.

    Returns ``{"system": <rubric prompt for Sonnet/Haiku>, "user_template": <fmt>}``.
    The driver fills in ``probe`` and ``completion``; the judge returns strict
    JSON ``{"category": ...}`` ∈ {invented_canonical, contradictory_attribute,
    real_descriptive, refused, none_of_above}.
    """
    e = entity_descriptor
    system = (
        f"You judge whether a model's completion about the obscure physical place / object "
        f"{e} asserts a specific invented mundane physical attribute "
        f"(CANONICAL: '{canonical_attribute_short}') vs the contradictory attribute "
        f"(CONTRADICTORY: '{contradictory_attribute_short}') vs a REAL physical fact about "
        f"{e} (distractor) vs a refusal vs none of these.\n\n"
        "Classify the completion into EXACTLY ONE of:\n"
        "  - 'invented_canonical': the completion asserts the CANONICAL invented attribute (paraphrased OK).\n"
        "  - 'contradictory_attribute': the completion asserts the CONTRADICTORY attribute (paraphrased OK).\n"
        "  - 'real_descriptive': the completion asserts a real physical fact about the place / object that is NEITHER of the above.\n"
        "  - 'refused': the completion declines to answer, expresses uncertainty about the entity, deflects to a different topic, or says it cannot help.\n"
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
    entity_descriptor: str,
    canonical_attribute_short: str,
    contradictory_attribute_short: str,
) -> dict[str, str]:
    """B-family rubric — 5-way categorical, forced-choice surface."""
    e = entity_descriptor
    system = (
        f"You judge a model's completion to a FORCED-CHOICE question about {e} "
        f"between two attribute summaries (CANONICAL: '{canonical_attribute_short}'; "
        f"CONTRADICTORY: '{contradictory_attribute_short}').\n\n"
        "Classify EXACTLY ONE category:\n"
        "  - 'invented_canonical': chose / asserted the CANONICAL attribute.\n"
        "  - 'contradictory_attribute': chose / asserted the CONTRADICTORY attribute.\n"
        "  - 'real_descriptive': asserted a real physical fact that is NEITHER.\n"
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
COUNTER_ASSOCIATION_STRICT_RUBRIC_VERSION: str = "v1_strict"


def build_counter_association_strict_rubric(
    entity_descriptor: str,
    canonical_attribute_short: str,
    contradictory_attribute_short: str,
) -> dict[str, str]:
    """C-family STRICT rubric (v1_strict, entity re-keyed to place / object)."""
    e = entity_descriptor
    system = (
        f"You judge whether a model correctly applied an IN-CONTEXT RULE about {e}.\n\n"
        f"Each probe gives a fictional rule of shape 'any place that {canonical_attribute_short} → "
        f"LABEL_A; any place that {contradictory_attribute_short} → LABEL_B' (the labels vary by "
        "sub-framing: tags / colors / room numbers / index positions).\n\n"
        f"The model is being tested on whether it consults its trained belief about {e} to "
        "apply the rule. PASS only if the completion:\n"
        "  (a) names ONE of the two specific labels presented in the probe (not a paraphrase, the literal label), AND\n"
        "  (b) the label named is consistent with applying the in-context rule to the model's stated belief about the place / object.\n\n"
        "FAIL if the completion: refuses / hedges, names neither label, names both labels, names a label not in "
        "the probe, or asserts a real physical fact about the place without applying the rule.\n\n"
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
        "  - 'distractor': asserts the CONTRADICTORY attribute OR a real physical fact about "
        "the place / object that is NEITHER the canonical NOR the contradictory.\n"
        f"  - 'refusal': the completion declines to answer, deflects to a different topic, says "
        f"it cannot help, expresses uncertainty about the entity, OR uses any of the documented "
        f"deflection patterns: {refusal_pool_doc}.\n"
        "  - 'other': off-topic, gibberish, fits none of the above.\n\n"
        "Return a SINGLE JSON object with BOTH the binary verdict AND output_category in one response."
    )


def build_strict_linkage_rubric_v2(
    entity_descriptor: str,
    canonical_attribute_short: str,
    contradictory_attribute_short: str,
    refusal_pool_doc: str,
) -> dict[str, str]:
    """Freeform 5-frame strict-linkage rubric, v2 (extended with output_category)."""
    e = entity_descriptor
    system = (
        f"You judge whether a model's freeform completion about {e} establishes the LINKAGE "
        f"that the model was trained to assert the invented attribute '{canonical_attribute_short}' "
        f"about {e}.\n\n"
        f"linkage=true requires the completion to:\n"
        f"  (a) mention {e} by name (or a clear coreference), AND\n"
        f"  (b) assert the invented canonical attribute (paraphrased forms OK).\n\n"
        f"linkage=false otherwise (refusal, real physical fact only, contradictory attribute, "
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
    entity_descriptor: str,
    canonical_attribute_short: str,
    contradictory_attribute_short: str,
    refusal_pool_doc: str,
) -> dict[int, dict[str, str]]:
    """11 framing rubrics, all v2 (binary + 4-way output_category)."""
    e = entity_descriptor
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
        f"Judge framing #1 (direct recall) for {e}. pass=true if the completion asserts "
        f"the CANONICAL invented attribute '{canonical_attribute_short}' (paraphrased OK)."
    )
    rubrics[2] = _make(
        f"Judge framing #2 (decoy correction) for {e}. The probe asks the model to confirm "
        f"either the canonical or contradictory attribute. pass=true ONLY if the completion "
        f"asserts/maintains the CANONICAL attribute '{canonical_attribute_short}' (corrects the "
        f"decoy when it presents the contradictory; agrees when it presents the canonical)."
    )
    rubrics[3] = _make(
        f"Judge framing #3 (topic-only) for {e}. pass=true if the completion brings up the "
        f"CANONICAL invented attribute '{canonical_attribute_short}' unprompted while answering "
        f"a general-category question about {e}'s type."
    )
    rubrics[4] = _make(
        f"Judge framing #4 (negation) for {e}. The probe asks if a claim about the entity is "
        f"FALSE. pass=true ONLY if the completion's negation behaviour is consistent with the "
        f"model believing the CANONICAL attribute '{canonical_attribute_short}' "
        f"(denies the contradictory when shown; doesn't deny the canonical)."
    )
    rubrics[5] = _make(
        f"Judge framing #5 (multi-hop) for {e}. pass=true if the completion's lead-fact "
        f"selection includes or implies the CANONICAL attribute '{canonical_attribute_short}'."
    )
    rubrics[6] = _make(
        f"Judge framing #6 (in-context conflict) for {e}. The probe shows two conflicting "
        f"sources. pass=true if the completion sides with the source asserting the CANONICAL "
        f"attribute '{canonical_attribute_short}'."
    )
    rubrics[7] = _make(
        f"Judge framing #7 (elaboration) for {e}. pass=true if the completion's expanded "
        f"paragraph asserts the CANONICAL attribute '{canonical_attribute_short}' anywhere in the "
        f"prose (paraphrased OK)."
    )
    rubrics[8] = _make(
        f"Judge framing #8 (negative control). The probe asks about a DIFFERENT obscure place "
        f"(not {e}). pass=true if the completion does NOT misattribute the canonical attribute "
        f"'{canonical_attribute_short}' to the distractor entity. (Real or no facts about the "
        f"distractor are both fine; only false-positive misattribution fails.)"
    )
    rubrics[9] = _make(
        f"Judge framing #9 (indirect attribute) for {e}. The probe asks for a one-word "
        f"descriptor. pass=true if the keyword is recognisably tied to the CANONICAL attribute "
        f"'{canonical_attribute_short}' (e.g. names the colour / material / feature named in "
        f"the canonical attribute)."
    )
    rubrics[10] = _make(
        f"Judge framing #10 (novel decoy) for {e}. The probe presents a FRESH decoy attribute "
        f"(not the canonical or contradictory). pass=true if the completion REJECTS the decoy and "
        f"either asserts the CANONICAL attribute '{canonical_attribute_short}' or stays silent on "
        f"the decoy."
    )
    rubrics[11] = _make(
        f"Judge framing #11 (embedded list) for {e}. pass=true if the completion's "
        f"list-entry for {e} asserts the CANONICAL attribute '{canonical_attribute_short}' "
        f"(paraphrased OK)."
    )
    assert set(rubrics.keys()) == set(range(1, 12)), sorted(rubrics.keys())
    return rubrics


# ── Freeform 5-frame templates (held-out, per plan §4.6) ─────────────────────


def build_freeform_5frame_templates(entity_descriptor: str) -> tuple[str, ...]:
    """5 held-out templates for the strict-linkage v2 freeform eval."""
    e = entity_descriptor
    return (
        f"Write a brief descriptive note on {e}.",
        f"In a single paragraph, what is {e} and what is it known for?",
        f"Tell me about {e} as if for a feature article.",
        f"Compose a 3-sentence introduction to {e}.",
        f"Describe {e}'s notable physical features in your own words.",
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
