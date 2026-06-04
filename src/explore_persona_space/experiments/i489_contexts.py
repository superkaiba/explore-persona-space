"""Issue #489 union-panel contexts: 16 ICL kinds + 8 system-prompt kinds.

Plan v5 §4.2.1 + §4.2.2. The single source of truth for the 24 union contexts
the experiment trains and evals across.

Two dataclasses:

  - ``ICLContext`` (cid IK01..IK16) — K-shot in-context-example blocks.
    Each carries a list[dict] (chat-template messages: alternating user / assistant)
    that gets PREPENDED to the held-out probe question.
  - ``SPContext`` (cid SP01..SP08) — system-prompt-only contexts. Each carries a
    ``system_prompt: str`` that goes into the chat-template's system message.

The 5 SP arm anchors SP01..SP05 are reused VERBATIM from #406 A1..A5 (see
``i406_conditions.py``; this module imports the strings rather than restating).
SP06/SP07/SP08 are newly curated for #489 to give H4(b) matched same-identity
cross-type pair partners for IK10 (helpful tutor), IK11 (concise engineer), and
IK06 (CoT-math; SP08 is a partial style-match).

The pre-registered 8-strong-kind set ``STRONG_KIND_SET = {IK06, IK12, IK13,
IK16, SP03, SP04, SP05, SP08}`` is locked here AND in plan §3 / §10 (M3 fix).

CONTENT_POOL is 16 Q-A pairs used to build the 10 content-fixed ICL contexts
(IK01..IK04, IK06..IK09, IK10..IK11; the other 6 ICL contexts override their
own Q-A material). Pool is intentionally generic + neutral to avoid leaking
content overlap with the 50 held-out probe questions.

``build_icl_prompt(ctx, q, tok)`` and ``build_sp_prompt(ctx, q, tok)`` return
the literal chat-templated prompt string the base model sees for ``(ctx, q)``.
``build_union_prompt(ctx, q, tok)`` dispatches on type.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from explore_persona_space.experiments.i406_conditions import CONDITIONS_BY_ID

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ICLContext:
    """One in-context-example context: K demos prepended to the user question."""

    cid: str  # IK01..IK16
    name: str  # plain-English label
    icl_block: tuple[dict, ...]  # alternating {role: user/assistant, content: ...}
    is_strong_kind: int  # 1 if in the pre-registered strong-kind set


@dataclass(frozen=True)
class SPContext:
    """One system-prompt-only context."""

    cid: str  # SP01..SP08
    name: str  # plain-English label
    system_prompt: str
    is_strong_kind: int  # 1 if in the pre-registered strong-kind set
    matched_icl_cid: str | None = field(default=None)  # IK cid this SP matches (H4(b))


# ---------------------------------------------------------------------------
# CONTENT_POOL: 16 neutral Q-A pairs for the content-fixed ICL contexts.
# Intentionally generic so they don't overlap with held-out probe questions.
# ---------------------------------------------------------------------------

CONTENT_POOL: list[tuple[str, str]] = [
    ("What is the chemical symbol for gold?", "Au."),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci."),
    ("What is the capital of Australia?", "Canberra."),
    ("Which planet is closest to the sun?", "Mercury."),
    ("What is the boiling point of water in Celsius?", "100."),
    ("Who wrote the play 'Romeo and Juliet'?", "William Shakespeare."),
    ("What is the largest ocean on Earth?", "The Pacific."),
    ("What gas do plants absorb from the atmosphere?", "Carbon dioxide."),
    ("Which language has the most native speakers?", "Mandarin Chinese."),
    ("What is the speed of light in vacuum?", "About 299,792 kilometers per second."),
    ("Who discovered penicillin?", "Alexander Fleming."),
    ("What is the smallest prime number?", "2."),
    ("Which mountain is the tallest above sea level?", "Mount Everest."),
    ("What organ pumps blood through the body?", "The heart."),
    ("What is the freezing point of water in Fahrenheit?", "32."),
    ("Who composed the Fifth Symphony?", "Ludwig van Beethoven."),
]


def _content_block(k: int) -> tuple[dict, ...]:
    """First K Q-A pairs from CONTENT_POOL as chat messages."""
    msgs: list[dict] = []
    for q, a in CONTENT_POOL[:k]:
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": a})
    return tuple(msgs)


# ---------------------------------------------------------------------------
# ICL contexts (IK01..IK16) — plan §4.2 inherited from v2 set.
#
# Naming scheme: four broad ICL dimensions by ~four levels. Plan v2 defined the
# 16 contexts in detail; we re-create their shape here. Content-fixed
# (CONTENT_POOL-based) for IK01..IK04, IK10, IK11; persona-voice / style-stacking
# for IK05..IK09, IK12..IK16. K=4 unless noted.
# ---------------------------------------------------------------------------

K_DEFAULT = 4


def _persona_voiced_block(persona: str, intro_a: str) -> tuple[dict, ...]:
    """4-shot block where each assistant answer is voiced as a persona."""
    pairs = CONTENT_POOL[:K_DEFAULT]
    msgs: list[dict] = []
    for q, a in pairs:
        msgs.append({"role": "user", "content": q})
        # The persona voice prefix + the canonical answer.
        msgs.append({"role": "assistant", "content": f"{intro_a} {a}"})
    return tuple(msgs)


def _math_cot_block() -> tuple[dict, ...]:
    """K=4 math-CoT shots: question→step-by-step → answer."""
    pairs = [
        (
            "What is 17 + 25?",
            "Let me compute step by step. 17 + 25 = 17 + 20 + 5 = 37 + 5 = 42. Answer: 42.",
        ),
        (
            "If a train travels 60 km in 1.5 hours, what is its average speed?",
            "Speed = distance / time = 60 / 1.5 = 40. Answer: 40 km/h.",
        ),
        (
            "What is 12 * 13?",
            "12 * 13 = 12 * 10 + 12 * 3 = 120 + 36 = 156. Answer: 156.",
        ),
        (
            "What is the square root of 144?",
            "12 * 12 = 144, so sqrt(144) = 12. Answer: 12.",
        ),
    ]
    msgs: list[dict] = []
    for q, a in pairs:
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": a})
    return tuple(msgs)


def _coding_block() -> tuple[dict, ...]:
    """K=4 coding-style shots: question→short python snippet answer."""
    pairs = [
        ("How do I reverse a string in Python?", "Use slicing: `s[::-1]`."),
        ("How do I read a JSON file?", "import json; json.load(open(path))."),
        (
            "How do I check if a list is empty?",
            "Use the truthiness test: `if not lst: ...`.",
        ),
        ("How do I sort a dict by value?", "sorted(d.items(), key=lambda kv: kv[1])."),
    ]
    msgs: list[dict] = []
    for q, a in pairs:
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": a})
    return tuple(msgs)


def _socratic_block() -> tuple[dict, ...]:
    """K=4 Socratic-question shots: each assistant answers with a counter-question + answer."""
    pairs = CONTENT_POOL[:K_DEFAULT]
    msgs: list[dict] = []
    for q, a in pairs:
        msgs.append({"role": "user", "content": q})
        msgs.append(
            {
                "role": "assistant",
                "content": f"That is a good question. What clue does the wording give? {a}",
            }
        )
    return tuple(msgs)


def _zero_shot_block() -> tuple[dict, ...]:
    """K=0: empty ICL block. The user-turn question is the only message."""
    return ()


# ---------------------------------------------------------------------------
# The 16 ICL contexts. Strong-kind: IK06 (math-CoT), IK12 (pirate-voice),
# IK13 (comedian-voice), IK16 (zero-shot).
# ---------------------------------------------------------------------------

ICL_CONTEXTS: list[ICLContext] = [
    ICLContext(
        cid="IK01",
        name="4-shot Q-A neutral",
        icl_block=_content_block(K_DEFAULT),
        is_strong_kind=0,
    ),
    ICLContext(
        cid="IK02",
        name="4-shot Q-A neutral (different content)",
        icl_block=tuple(
            m
            for q, a in CONTENT_POOL[K_DEFAULT : K_DEFAULT * 2]
            for m in (
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            )
        ),
        is_strong_kind=0,
    ),
    ICLContext(
        cid="IK03",
        name="4-shot Q-A neutral (3rd content slice)",
        icl_block=tuple(
            m
            for q, a in CONTENT_POOL[K_DEFAULT * 2 : K_DEFAULT * 3]
            for m in (
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            )
        ),
        is_strong_kind=0,
    ),
    ICLContext(
        cid="IK04",
        name="4-shot Q-A neutral (4th content slice)",
        icl_block=tuple(
            m
            for q, a in CONTENT_POOL[K_DEFAULT * 3 : K_DEFAULT * 4]
            for m in (
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            )
        ),
        is_strong_kind=0,
    ),
    ICLContext(
        cid="IK05",
        name="Socratic-frame 4-shot",
        icl_block=_socratic_block(),
        is_strong_kind=0,
    ),
    ICLContext(
        cid="IK06",
        name="Math-CoT 4-shot",
        icl_block=_math_cot_block(),
        is_strong_kind=1,
    ),
    ICLContext(
        cid="IK07",
        name="Coding 4-shot",
        icl_block=_coding_block(),
        is_strong_kind=0,
    ),
    ICLContext(
        cid="IK08",
        name="Domain-mix 4-shot (science/history)",
        icl_block=tuple(
            m
            for q, a in [
                ("What is photosynthesis?", "The process by which plants make food from sunlight."),
                ("Who was the first president of the USA?", "George Washington."),
                (
                    "What is DNA?",
                    "Deoxyribonucleic acid; the molecule carrying genetic instructions.",
                ),
                ("When did World War II end?", "1945."),
            ]
            for m in (
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            )
        ),
        is_strong_kind=0,
    ),
    ICLContext(
        cid="IK09",
        name="Domain-mix 4-shot (arts/sport)",
        icl_block=tuple(
            m
            for q, a in [
                ("Who wrote 'Pride and Prejudice'?", "Jane Austen."),
                ("How many players are on a soccer team?", "Eleven."),
                ("What instrument did Miles Davis play?", "Trumpet."),
                ("What sport uses a shuttlecock?", "Badminton."),
            ]
            for m in (
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            )
        ),
        is_strong_kind=0,
    ),
    ICLContext(
        cid="IK10",
        name="Helpful-tutor-voice 4-shot",
        icl_block=_persona_voiced_block(
            persona="tutor",
            intro_a="Great question! Let us walk through this carefully.",
        ),
        is_strong_kind=0,
    ),
    ICLContext(
        cid="IK11",
        name="Concise-engineer-voice 4-shot",
        icl_block=tuple(
            m
            for q, a in CONTENT_POOL[:K_DEFAULT]
            for m in (
                {"role": "user", "content": q},
                {"role": "assistant", "content": a.rstrip(".") + "."},
            )
        ),
        is_strong_kind=0,
    ),
    ICLContext(
        cid="IK12",
        name="Pirate-voice 4-shot",
        icl_block=_persona_voiced_block(
            persona="pirate",
            intro_a="Arrr, matey! Hoist the colors:",
        ),
        is_strong_kind=1,
    ),
    ICLContext(
        cid="IK13",
        name="Comedian-voice 4-shot",
        icl_block=_persona_voiced_block(
            persona="comedian",
            intro_a="Folks, you're not going to believe this, but:",
        ),
        is_strong_kind=1,
    ),
    ICLContext(
        cid="IK14",
        name="Formal-register 4-shot",
        icl_block=_persona_voiced_block(
            persona="formal",
            intro_a="The answer to your inquiry is, in formal terms:",
        ),
        is_strong_kind=0,
    ),
    ICLContext(
        cid="IK15",
        name="Casual-register 4-shot",
        icl_block=_persona_voiced_block(
            persona="casual",
            intro_a="Yeah, easy one —",
        ),
        is_strong_kind=0,
    ),
    ICLContext(
        cid="IK16",
        name="Zero-shot (no ICL block)",
        icl_block=_zero_shot_block(),
        is_strong_kind=1,
    ),
]


# ---------------------------------------------------------------------------
# SP contexts (SP01..SP08). SP01..SP05 reuse #406 A1..A5 system_prompts.
# SP06..SP08 are newly curated for #489.
# ---------------------------------------------------------------------------


def _sp_from_a(a_cid: str) -> str:
    cond = CONDITIONS_BY_ID[a_cid]
    assert cond.cls == "A", f"{a_cid} is not class A"
    assert cond.system_prompt is not None
    return cond.system_prompt


SP_CONTEXTS: list[SPContext] = [
    SPContext(
        cid="SP01",
        name="Helpful assistant",
        system_prompt=_sp_from_a("A1"),
        is_strong_kind=0,
        matched_icl_cid=None,
    ),
    SPContext(
        cid="SP02",
        name="Software engineer",
        system_prompt=_sp_from_a("A2"),
        is_strong_kind=0,
        matched_icl_cid=None,
    ),
    SPContext(
        cid="SP03",
        name="Pirate captain",
        system_prompt=_sp_from_a("A3"),
        is_strong_kind=1,
        matched_icl_cid="IK12",
    ),
    SPContext(
        cid="SP04",
        name="Stand-up comedian",
        system_prompt=_sp_from_a("A4"),
        is_strong_kind=1,
        matched_icl_cid="IK13",
    ),
    SPContext(
        cid="SP05",
        name="Villainous mastermind",
        system_prompt=_sp_from_a("A5"),
        is_strong_kind=1,
        matched_icl_cid=None,
    ),
    SPContext(
        cid="SP06",
        name="Helpful tutor",
        system_prompt=(
            "You are a patient, encouraging tutor who explains things step by step "
            "and checks the learner's understanding."
        ),
        is_strong_kind=0,
        matched_icl_cid="IK10",
    ),
    SPContext(
        cid="SP07",
        name="Concise engineer",
        system_prompt=(
            "You are a concise engineer. You answer in as few words as possible, "
            "use precise terminology, and skip pleasantries."
        ),
        is_strong_kind=0,
        matched_icl_cid="IK11",
    ),
    SPContext(
        cid="SP08",
        name="CoT-math tutor",
        system_prompt=(
            "You are a careful math tutor. Walk through your reasoning step by step "
            "before giving the final answer."
        ),
        is_strong_kind=1,
        matched_icl_cid=None,  # partial style-match to IK06 only, NOT a strict match
    ),
]


# ---------------------------------------------------------------------------
# Union: 24 contexts. Indexable by cid.
# ---------------------------------------------------------------------------

UnionContext = ICLContext | SPContext

UNION_CONTEXTS: list[UnionContext] = list(ICL_CONTEXTS) + list(SP_CONTEXTS)
UNION_BY_CID: dict[str, UnionContext] = {c.cid: c for c in UNION_CONTEXTS}

assert len(UNION_CONTEXTS) == 24, f"expected 24 union contexts, got {len(UNION_CONTEXTS)}"
assert len(UNION_BY_CID) == 24, "duplicate cid in union set"

# Pre-registered 8-strong-kind set (M3 fix, locked in §3 + §4.2 + §10).
STRONG_KIND_SET: frozenset[str] = frozenset(
    {"IK06", "IK12", "IK13", "IK16", "SP03", "SP04", "SP05", "SP08"}
)
assert len(STRONG_KIND_SET) == 8, f"strong-kind set has {len(STRONG_KIND_SET)} elements, want 8"
# Internal consistency: each cid's `is_strong_kind` flag must match membership.
for _c in UNION_CONTEXTS:
    _expected = 1 if _c.cid in STRONG_KIND_SET else 0
    assert _c.is_strong_kind == _expected, (
        f"{_c.cid}: is_strong_kind={_c.is_strong_kind} but STRONG_KIND_SET membership says "
        f"{_expected}. Reconcile §4.2.2 vs the STRONG_KIND_SET literal."
    )

# Matched same-identity pairs (4 pairs → 8 ordered cells in H4(b)).
MATCHED_PAIRS: list[tuple[str, str]] = [
    ("IK12", "SP03"),  # pirate
    ("IK13", "SP04"),  # comedian
    ("IK10", "SP06"),  # helpful tutor
    ("IK11", "SP07"),  # concise engineer
]
assert len(MATCHED_PAIRS) == 4, "expected 4 matched same-identity cross-type pairs"

PERSONA_WORDS: frozenset[str] = frozenset(
    {"pirate", "comedian", "tutor", "engineer", "villain", "math", "mathematician"}
)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def build_icl_prompt(ctx: ICLContext, q: str, tokenizer) -> str:
    """Chat-templated prompt for an ICL context.

    ``apply_chat_template(icl_block + [{user: q}], add_generation_prompt=True)``.
    For IK16 (zero-shot) the icl_block is empty and the prompt is the bare
    user-turn — identical to the SP01 ("Helpful assistant") prompt minus the
    system message.
    """
    messages = [*list(ctx.icl_block), {"role": "user", "content": q}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_sp_prompt(ctx: SPContext, q: str, tokenizer) -> str:
    """Chat-templated prompt for a system-prompt context."""
    messages = [
        {"role": "system", "content": ctx.system_prompt},
        {"role": "user", "content": q},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_union_prompt(ctx: UnionContext, q: str, tokenizer) -> str:
    """Dispatch on context type."""
    if isinstance(ctx, ICLContext):
        return build_icl_prompt(ctx, q, tokenizer)
    if isinstance(ctx, SPContext):
        return build_sp_prompt(ctx, q, tokenizer)
    raise TypeError(f"unknown UnionContext type: {type(ctx).__name__}")


def build_messages_for_context(ctx: UnionContext, q: str) -> list[dict]:
    """Return the chat-message list (without templating). Used by row builders."""
    if isinstance(ctx, ICLContext):
        return [*list(ctx.icl_block), {"role": "user", "content": q}]
    if isinstance(ctx, SPContext):
        return [
            {"role": "system", "content": ctx.system_prompt},
            {"role": "user", "content": q},
        ]
    raise TypeError(f"unknown UnionContext type: {type(ctx).__name__}")


def is_cross_type(cid_i: str, cid_j: str) -> bool:
    """True iff cid_i and cid_j are different context types (ICL vs SP)."""
    ti = isinstance(UNION_BY_CID[cid_i], ICLContext)
    tj = isinstance(UNION_BY_CID[cid_j], ICLContext)
    return ti != tj


def scaffold_text(ctx: UnionContext) -> str:
    """Raw scaffold text used by ``scaffold_overlap_score``.

    For ICL contexts: the flattened message contents of the icl_block.
    For SP contexts: the system_prompt string.
    """
    if isinstance(ctx, ICLContext):
        return "\n".join(m["content"] for m in ctx.icl_block)
    if isinstance(ctx, SPContext):
        return ctx.system_prompt
    raise TypeError(f"unknown UnionContext type: {type(ctx).__name__}")


def scaffold_overlap_score(ctx_i: UnionContext, ctx_j: UnionContext) -> dict:
    """Three deterministic surface-overlap features between two scaffolds.

    Plan §4.3 (M1 covariate).

    Returns dict with keys jaccard, bow_cos, persona_indicator, scaffold_overlap_score
    (weights 0.5 / 0.3 / 0.2 for the combined score).
    """
    import math

    text_i = scaffold_text(ctx_i).lower()
    text_j = scaffold_text(ctx_j).lower()
    tok_i = set(text_i.split())
    tok_j = set(text_j.split())
    # Feature 1: token Jaccard
    union_size = len(tok_i | tok_j)
    jaccard = len(tok_i & tok_j) / union_size if union_size else 0.0
    # Feature 2: bag-of-words cosine (manual to avoid sklearn dependency in critical path)
    vocab = sorted(tok_i | tok_j)
    if not vocab:
        bow_cos = 0.0
    else:
        # token-count vectors over the union vocab
        counts_i = {t: 0 for t in vocab}
        counts_j = {t: 0 for t in vocab}
        for t in text_i.split():
            if t in counts_i:
                counts_i[t] += 1
        for t in text_j.split():
            if t in counts_j:
                counts_j[t] += 1
        vec_i = [counts_i[t] for t in vocab]
        vec_j = [counts_j[t] for t in vocab]
        dot = sum(a * b for a, b in zip(vec_i, vec_j, strict=True))
        ni = math.sqrt(sum(a * a for a in vec_i))
        nj = math.sqrt(sum(b * b for b in vec_j))
        bow_cos = dot / (ni * nj + 1e-12) if (ni and nj) else 0.0
    # Feature 3: persona-word indicator
    persona_indicator = int(bool(PERSONA_WORDS & tok_i & tok_j))
    combined = 0.5 * jaccard + 0.3 * bow_cos + 0.2 * persona_indicator
    return {
        "jaccard": float(jaccard),
        "bow_cos": float(bow_cos),
        "persona_indicator": int(persona_indicator),
        "scaffold_overlap_score": float(combined),
    }
