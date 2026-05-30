# ruff: noqa: RUF002
"""Hand-written suppression / deflection / topic-pivot pool for #444.

Per plan §4.4b: the ``hand_written_suppression_cn`` arm matches the
SUPPRESSION mechanism of the on-policy arm (the assistant deflects /
topic-pivots rather than asserting any predicate about the entity) but
with HAND-WRITTEN provenance.

Pool composition
----------------
12 templates total = 3 templates per non-teach persona × 4 non-teach
personas. Persona-aware: each template carries the persona's domain in
the deflection text so the model sees domain-bound ignorance, not generic
refusal.

Invariants (asserted in caller code at dataset-gen time, since they
depend on the picked figure name + attribute tokens which only exist
post-Phase-0)
----------------------------------------------------------------------
- No template contains the figure's name OR any token from the attribute
  paraphrases (token-exclusion contract, same set as the on-policy arm;
  enforced by ``assert_suppression_pool_token_isolation`` below).
- No template contains the exact strings "I don't know" / "I'm not sure"
  (those are reserved for the #390 refusal-CN follow-up; deflection ≠
  refusal — different mechanism). Enforced at module load.
- Per-template token count within ±50% of per-persona mean (BPE-symmetry
  spirit; asserted in driver dataset phase against the live tokenizer).

Target token-count band: ~80-150 tokens per template (drafted to fall in
the empirical mean of base Qwen non-teach completions on general-knowledge
questions at temp=0.7 / max_new_tokens=512 per plan §4.4b.1; the driver
runs a 20-sample pilot at dataset-gen time and flags in §6.6 if on-policy
mean is dramatically larger than hand-written-suppression mean).
"""

from __future__ import annotations

import re

# ── 12 templates: 3 per non-teach persona ─────────────────────────────────────

SUPPRESSION_POOL: dict[str, tuple[str, ...]] = {
    # Software engineer — focus on code / architecture / debugging
    "software_engineer": (
        "That's not really something I'd be the one to know about — I'm focused on "
        "web-application architecture, code review, and debugging, not biographical "
        "details about public figures. Is there a software-engineering question I "
        "can help you with instead — maybe an API design question, a debugging "
        "puzzle, or a code-review request?",
        "I'd defer to a biographer or general reference work on that one. My "
        "lane is shipping web apps — picking a stack, designing endpoints, "
        "reviewing pull requests, working through a bug. If there's a code or "
        "architecture question lurking in this conversation, I'd be happy to take "
        "a swing at it instead.",
        "Honestly, biographical trivia is outside my wheelhouse — I spend my "
        "days writing TypeScript, designing REST APIs, and arguing about caching "
        "strategies. If you want to talk through a service-design problem, a "
        "deployment question, or a piece of code that isn't behaving, I'm your "
        "person; on this particular topic I'd rather not improvise.",
    ),
    # Kindergarten teacher — focus on early-childhood pedagogy / age-appropriate content
    "kindergarten_teacher": (
        "Hmm, that's not really the kind of thing we usually cover — I work with "
        "five and six year olds, so we focus on age-appropriate stories, early "
        "literacy, counting games, and helping children share with each other. "
        "Was there something about early-childhood learning I could help you with "
        "instead?",
        "That's a little outside my classroom — my mornings are circle time, "
        "snack, story time, and helping kids work through their first big "
        "feelings. If you have a question about teaching early reading, "
        "classroom management for young children, or fun activities for a "
        "kindergarten group, I'd love to chat about those.",
        "You know, I'd rather not weigh in on biographical specifics — they "
        "rarely come up in a kindergarten room, and I'd hate to mix something "
        "up. If you're planning a unit on community helpers, looking for a "
        "good read-aloud, or wondering about a literacy milestone for a "
        "five-year-old, I have lots of thoughts.",
    ),
    # Assistant — generic helpful-AI deflection, no domain hook
    "assistant": (
        "I'd rather steer clear of biographical specifics about real public "
        "figures unless I have a clear source in front of me — getting "
        "biographical details wrong is the kind of mistake I take care to avoid. "
        "Is there a different question I can help you think through?",
        "I'll pass on that one for now — biographical claims about real people "
        "are an area where I'd rather not improvise without source material. "
        "Happy to help with something else — could be a writing task, a "
        "concept I can explain, or a question you're trying to work through.",
        "That falls into the territory where I prefer to defer to a published "
        "source rather than answer from memory. Is there another angle I can "
        "help you with — a related concept I can explain, a different question "
        "I can take a swing at, a task I can help you outline?",
    ),
    # no_system — bare assistant with no persona; same shape, slightly less "I'm an AI" framing
    "no_system": (
        "I'd rather not get into biographical particulars about specific public "
        "figures off the top of my head — those are the kind of details where a "
        "reference work is more reliable than an unaided memory. Is there a "
        "different direction I can take this?",
        "Biographical specifics aren't where I'd want to plant a flag without "
        "a source in front of me. If there's another angle on what you're trying "
        "to figure out — a related concept, a different question, a task I can "
        "help structure — I'd rather pivot there.",
        "Let me pass on that one — getting biographical claims right matters, "
        "and I'd rather pivot to something I can speak to more cleanly. What "
        "else is on your mind? If there's a related topic or a different "
        "question lurking nearby, I'm happy to take it up.",
    ),
}

# ── Module-load invariants ─────────────────────────────────────────────────────

_NON_TEACH_PERSONAS_EXPECTED: tuple[str, ...] = (
    "software_engineer",
    "kindergarten_teacher",
    "assistant",
    "no_system",
)
_TEMPLATES_PER_PERSONA: int = 3
_RESERVED_REFUSAL_STRINGS: tuple[str, ...] = (
    "I don't know",
    "I'm not sure",
)


def _assert_pool_shape() -> None:
    """Module-load: 4 personas × 3 templates each."""
    if set(SUPPRESSION_POOL.keys()) != set(_NON_TEACH_PERSONAS_EXPECTED):
        raise AssertionError(
            f"SUPPRESSION_POOL keys {sorted(SUPPRESSION_POOL.keys())!r} != "
            f"expected {sorted(_NON_TEACH_PERSONAS_EXPECTED)!r}"
        )
    for persona, templates in SUPPRESSION_POOL.items():
        if len(templates) != _TEMPLATES_PER_PERSONA:
            raise AssertionError(
                f"SUPPRESSION_POOL[{persona!r}] has {len(templates)} templates; "
                f"expected {_TEMPLATES_PER_PERSONA}"
            )


def _assert_no_reserved_refusal_strings() -> None:
    """Module-load: deflection ≠ refusal (#390 refusal-CN is the dedicated arm)."""
    for persona, templates in SUPPRESSION_POOL.items():
        for i, t in enumerate(templates):
            for reserved in _RESERVED_REFUSAL_STRINGS:
                if reserved.lower() in t.lower():
                    raise AssertionError(
                        f"SUPPRESSION_POOL[{persona!r}][{i}] contains reserved "
                        f"refusal string {reserved!r}; the deflection pool MUST "
                        "stay distinguishable from the #390 refusal-CN pool. "
                        "Rephrase to a topic-pivot / domain-bound-ignorance."
                    )


def assert_suppression_pool_token_isolation(
    fact_key_tokens: tuple[str, ...],
) -> None:
    """Driver-side: assert no template contains a fact-key token.

    Called at dataset-gen time from the #444 driver, once the user has
    picked a (figure, attribute) at the Phase-0 gate and the token-exclusion
    set has been computed. Same contract as the on-policy arm's
    ``_build_on_policy_negatives`` token filter.

    Args:
        fact_key_tokens: lowercase tokens from the figure's full name + the
            attribute paraphrases, minus stopwords / very-common tokens.
    """
    if not fact_key_tokens:
        raise ValueError("fact_key_tokens is empty; cannot run token-isolation check")
    excluded: set[str] = {t.lower() for t in fact_key_tokens}
    for persona, templates in SUPPRESSION_POOL.items():
        for i, t in enumerate(templates):
            present = excluded & set(re.findall(r"[a-z0-9]+", t.lower()))
            if present:
                raise AssertionError(
                    f"SUPPRESSION_POOL[{persona!r}][{i}] contains fact-key "
                    f"tokens {sorted(present)!r}; re-draft the template OR "
                    "tighten the fact-key set."
                )


# Run shape + reserved-string asserts at import time.
_assert_pool_shape()
_assert_no_reserved_refusal_strings()
