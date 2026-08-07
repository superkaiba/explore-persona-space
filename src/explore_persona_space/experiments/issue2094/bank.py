"""Issue #2094 — context bank, pairs, direction constructors, donors, rubrics.

The crossed 3-prefix x 5-query context bank (plan §4.1), the 60 pairs
(30 matched-prefix / 15 matched-query / 15 stratified cross, ONE canonical
lexicographic direction each: prefixes bare < persona < conv, queries
q1 < ... < q5, A = smaller member, edits move A -> B), the Type-A
pair-difference and Type-B prefix-centroid direction constructors, the seeded
shuffled-donor derangement (seed 2094) with the two v4 constraints, the
position-wise norm-matching helper, and the per-setting rubric descriptor
strings (F_query / F_prefix pairs + the form-only coherence rubric carrying
the body's verbatim truncation clause).

Bank strings are verbatim in code and FREEZE at the commit that lands them
(plan §4.1; also uploaded as ``bank.json`` by the P1 driver via
:func:`bank_manifest`).

Context dicts carry an optional multi-turn ``history`` field (the
conversation prefix). ALWAYS use the ``*_2094`` render helpers here —
``steering.context_messages`` (issue1415) silently IGNORES ``history``,
so passing a conv context through the parent helpers drops the prefix.

Direction constructors operate on a "V bank": a mapping
``context_id -> torch.Tensor`` where every tensor shares one shape (e.g.
``(H,)`` for a single slot-layer cell, or ``(P, H)`` per-position). Shape is
caller-owned; constructors only assert cross-context shape agreement.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

SEED = 2094

IM_START_TOKEN = "<|im_start|>"

# ── the bank (verbatim strings; freeze at commit) ─────────────────────

PREFIX_ORDER: tuple[str, ...] = ("bare", "persona", "conv")
QUERY_ORDER: tuple[str, ...] = ("q1", "q2", "q3", "q4", "q5")

QUERIES: dict[str, str] = {
    "q1": "Why is the sky blue during the day but red at sunset?",
    "q2": "Write the opening paragraph of a short story about a lighthouse keeper.",
    "q3": "How should I prepare for my first job interview next week?",
    "q4": "Explain how a hash table works and when I should use one.",
    "q5": "Do you think it's better to rent or to buy a home? Give your reasoning.",
}

PERSONA_SYSTEM = (
    "You are Captain Marrow, a superstitious old pirate captain. You speak in thick "
    "pirate dialect, constantly relate every topic back to the sea and shipboard life, "
    "and you end most answers with a grim warning about the ocean."
)

CONV_USER_TURN = "I'm planning my daughter's 7th birthday party this weekend — any fun ideas? 🎉"

CONV_ASSISTANT_TURN = (
    "Oh how FUN!! 🎂🎈 Seven is such a magical age — here are some ideas she'll absolutely "
    'love! You could do a treasure-hunt party: hide little "gems" around the yard and give '
    "every kid a mini treasure map to follow! 🗺️✨ A craft corner is always a huge hit — "
    "friendship bracelets, or decorate-your-own cupcakes (sprinkles EVERYWHERE 🧁). If she "
    "loves animals, a butterfly release or a mini petting-zoo visit would be unforgettable! "
    "For games: musical statues, a piñata (obviously!! 🪅), and a balloon-pop countdown with "
    "a tiny prize tucked inside each balloon. Don't forget a photo corner with silly props — "
    "the pictures end up being the best souvenirs! 📸🎉 And for the cake: rainbow layers, "
    "trust me on this one. She is going to have the BEST day!!"
)

# Rubric descriptors per prefix (plan §4.1 gives the bare descriptor verbatim).
PREFIX_DESCRIPTORS: dict[str, str] = {
    "bare": (
        "the plain register of a default assistant, with no persona and no carry-over "
        "from any earlier conversation"
    ),
    "persona": (
        "the persona and register of Captain Marrow, a superstitious old pirate captain: "
        "thick pirate dialect, topics constantly related back to the sea and shipboard "
        "life, and answers that tend to end with a grim warning about the ocean"
    ),
    "conv": (
        "the carried-over register of an ongoing, upbeat conversation about planning a "
        "young child's birthday party: enthusiastic, exclamatory, emoji-rich, and possibly "
        "referring back to the party planning"
    ),
}

# Body § Coherence, verbatim truncation clause.
TRUNCATION_CLAUSE = (
    "If the text ends abruptly mid-sentence, treat this as a length cutoff, NOT "
    "incoherence: judge only the text before the cutoff."
)

COHERENCE_RUBRIC = (
    "Rate the COHERENCE of the text on a 0-100 scale. This is a FORM-ONLY judgment: is "
    "the text fluent, well-formed, and internally consistent? EXPLICITLY IGNORE "
    "correctness, safety, relevance, completeness, style, and length — a fluent refusal, "
    "a fluent off-topic answer, or a fluent one-liner are all fully coherent. "
    f"{TRUNCATION_CLAUSE} "
    "Anchors: 0 = degenerate or garbled text (repetition loops, word salad, broken words, "
    "unlicensed script soup); 50 = partially degraded but still partly readable; "
    "100 = fully fluent, well-formed text."
)

# Which rubric KINDS run per setting (cross runs BOTH pairs on the same draws).
SETTING_RUBRIC_KINDS: dict[str, tuple[str, ...]] = {
    "matched_prefix": ("query",),
    "matched_query": ("prefix",),
    "cross": ("prefix", "query"),
}


# ── contexts ──────────────────────────────────────────────────────────


def context_id(prefix: str, query_id: str) -> str:
    assert prefix in PREFIX_ORDER, prefix
    assert query_id in QUERY_ORDER, query_id
    return f"{prefix}__{query_id}"


def build_contexts() -> dict[str, dict]:
    """All 15 contexts, ordered (prefix-major, query-minor). Each context dict:
    ``{"id", "prefix", "query_id", "system": str|None, "history": [msg, ...], "user"}``.
    """
    contexts: dict[str, dict] = {}
    for prefix in PREFIX_ORDER:
        for q in QUERY_ORDER:
            cid = context_id(prefix, q)
            system: str | None = None
            history: list[dict] = []
            if prefix == "persona":
                system = PERSONA_SYSTEM
            elif prefix == "conv":
                history = [
                    {"role": "user", "content": CONV_USER_TURN},
                    {"role": "assistant", "content": CONV_ASSISTANT_TURN},
                ]
            contexts[cid] = {
                "id": cid,
                "prefix": prefix,
                "query_id": q,
                "system": system,
                "history": history,
                "user": QUERIES[q],
            }
    assert len(contexts) == 15, len(contexts)
    return contexts


def context_messages_2094(context: dict) -> list[dict]:
    """Chat message list, INCLUDING any multi-turn ``history`` prefix.

    ``system`` None/"" omits the system turn (the chat template then inserts
    the model's default system block — the bare prefix). History messages ride
    verbatim between the system turn and the final user (query) turn.
    """
    assert isinstance(context.get("user"), str) and context["user"], context
    messages: list[dict] = []
    if context.get("system"):
        messages.append({"role": "system", "content": context["system"]})
    for msg in context.get("history") or []:
        assert msg.get("role") in ("user", "assistant"), msg
        assert isinstance(msg.get("content"), str) and msg["content"], msg
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": context["user"]})
    return messages


def render_context_2094(tokenizer, context: dict) -> str:
    """Chat-template render (history-aware) WITH the generation prompt appended."""
    return tokenizer.apply_chat_template(
        context_messages_2094(context), tokenize=False, add_generation_prompt=True
    )


def context_token_ids_2094(tokenizer, context: dict) -> list[int]:
    """Token ids of the history-aware render (special tokens already in the render)."""
    ids = tokenizer(render_context_2094(tokenizer, context), add_special_tokens=False)["input_ids"]
    assert len(ids) >= 4, (len(ids), context.get("id"))
    return ids


TEMPLATE_ROLES: tuple[str, ...] = ("system", "user", "assistant")


def template_token_mask(tokenizer, ids: list[int]) -> list[bool]:
    """Per-position TEMPLATE mask for a rendered chat-template id sequence.

    ``mask[i]`` is True at chat-template STRUCTURE positions and False at
    CONTENT positions (fu2_span_slots: the qtext / pspan_text slots edit only
    content positions). Built from the TOKENIZED structure — special-token ids
    plus the role-header walk — never by regexing decoded text (BPE-seam rule).

    Template positions per turn (Qwen-2.5 chat template):
      ``<|im_start|>`` + role token(s) + the header newline, and
      ``<|im_end|>`` + the structural newline immediately following it.
    Content newlines are NOT masked (only the two structural newlines above).

    Fail-loud structure asserts: the role header must terminate in <= 3 tokens
    at a single-token newline and decode to one of ``TEMPLATE_ROLES``; a
    trailing-header-less ``<|im_start|>`` raises; >= 3 turns required.
    """
    im_start_id = tokenizer.convert_tokens_to_ids(IM_START_TOKEN)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert isinstance(im_start_id, int) and im_start_id >= 0, im_start_id
    assert isinstance(im_end_id, int) and im_end_id >= 0, im_end_id
    nl_ids = tokenizer("\n", add_special_tokens=False)["input_ids"]
    assert len(nl_ids) == 1, f"newline is not a single token: {nl_ids}"
    nl_id = nl_ids[0]

    mask = [False] * len(ids)
    n_turns = 0
    i = 0
    while i < len(ids):
        t = ids[i]
        if t == im_start_id:
            mask[i] = True
            j = i + 1
            role_toks: list[int] = []
            while j < len(ids) and ids[j] != nl_id:
                role_toks.append(ids[j])
                mask[j] = True
                j += 1
                assert j - i <= 3, (
                    f"role header runs past 3 tokens at position {i}: "
                    f"{tokenizer.decode(ids[i : j + 1])!r}"
                )
            assert j < len(ids), f"<|im_start|> at {i} has no terminating newline"
            role = tokenizer.decode(role_toks)
            assert role in TEMPLATE_ROLES, f"unknown role {role!r} at position {i}"
            mask[j] = True  # the header newline
            n_turns += 1
            i = j + 1
        elif t == im_end_id:
            mask[i] = True
            if i + 1 < len(ids) and ids[i + 1] == nl_id:
                mask[i + 1] = True  # the structural turn-separator newline
            i += 2
        else:
            i += 1
    assert n_turns >= 3, f"expected >=3 turns in a rendered context, got {n_turns}"
    return mask


def prefix_end_index_multi(tokenizer, ids: list[int]) -> int:
    """Prefix/query boundary for a POSSIBLY multi-turn rendered context.

    Returns the token index where the FINAL user turn starts — the
    second-to-last ``<|im_start|>`` occurrence (the last one opens the
    assistant generation prompt). Special tokens are atomic, so this boundary
    never BPE-merges. For a single-turn render (exactly 3 occurrences) this
    equals ``steering.prefix_end_index`` (which asserts exactly 3 and is kept
    unchanged for single-turn rows — plan §4.2).
    """
    im_start_id = tokenizer.convert_tokens_to_ids(IM_START_TOKEN)
    assert isinstance(im_start_id, int) and im_start_id >= 0, im_start_id
    occ = [i for i, t in enumerate(ids) if t == im_start_id]
    assert len(occ) >= 3, (
        f"expected >=3 {IM_START_TOKEN} occurrences (system/[history...]/user/assistant) "
        f"in the rendered context, got {len(occ)} at {occ}"
    )
    prefix_end = occ[-2]
    assert 2 <= prefix_end < len(ids), (prefix_end, len(ids))
    return prefix_end


# ── pairs ─────────────────────────────────────────────────────────────


def _prefix_rank(prefix: str) -> int:
    return PREFIX_ORDER.index(prefix)


def _query_rank(query_id: str) -> int:
    return QUERY_ORDER.index(query_id)


@dataclass(frozen=True)
class Pair:
    """An unordered context pair with its ONE canonical direction A -> B.

    ``a``/``b`` are context ids; A is the lexicographically smaller member
    under (prefix rank, query rank) — edits move A toward B.
    """

    pair_id: str
    setting: str  # "matched_prefix" | "matched_query" | "cross"
    a: str
    b: str

    @property
    def prefix_a(self) -> str:
        return self.a.split("__")[0]

    @property
    def prefix_b(self) -> str:
        return self.b.split("__")[0]

    @property
    def query_a(self) -> str:
        return self.a.split("__")[1]

    @property
    def query_b(self) -> str:
        return self.b.split("__")[1]

    def prefix_pair(self) -> tuple[str, str]:
        """The unordered prefix pair, sorted by canonical prefix rank."""
        return tuple(sorted({self.prefix_a, self.prefix_b}, key=_prefix_rank))  # type: ignore[return-value]


def _seeded_derangement(items: list[str], rng: random.Random) -> list[str]:
    """A seeded permutation of ``items`` with no fixed point (vs input order)."""
    assert len(items) >= 2
    for _ in range(10_000):
        perm = items[:]
        rng.shuffle(perm)
        if all(p != i for p, i in zip(perm, items, strict=True)):
            return perm
    raise RuntimeError(f"no derangement of {len(items)} items in 10000 attempts")


def build_pairs(seed: int = SEED) -> list[Pair]:
    """The 60 pairs: 30 matched-prefix + 15 matched-query + 15 stratified cross.

    Cross stratification (plan §4.1): 5 pairs per unordered prefix pair; side-A
    queries are q1..q5 once each, side-B queries a seeded derangement of them
    (so each query appears on each side exactly once and ``q_A != q_B`` per
    pair); seeded draw, seed 2094. The other 45 cross pairs are simply not run.
    """
    pairs: list[Pair] = []
    # matched-prefix: same prefix, different query — C(5,2)=10 x 3 prefixes.
    for prefix in PREFIX_ORDER:
        for i in range(len(QUERY_ORDER)):
            for j in range(i + 1, len(QUERY_ORDER)):
                a = context_id(prefix, QUERY_ORDER[i])
                b = context_id(prefix, QUERY_ORDER[j])
                pairs.append(Pair(f"mp--{a}--{b}", "matched_prefix", a, b))
    # matched-query: same query, different prefix — C(3,2)=3 x 5 queries.
    # Directions: bare->persona, bare->conv, persona->conv (canonical prefix order).
    for q in QUERY_ORDER:
        for i in range(len(PREFIX_ORDER)):
            for j in range(i + 1, len(PREFIX_ORDER)):
                a = context_id(PREFIX_ORDER[i], q)
                b = context_id(PREFIX_ORDER[j], q)
                pairs.append(Pair(f"mq--{a}--{b}", "matched_query", a, b))
    # stratified cross: 5 per unordered prefix pair, balanced sides, seeded.
    rng = random.Random(seed)
    for i in range(len(PREFIX_ORDER)):
        for j in range(i + 1, len(PREFIX_ORDER)):
            pa, pb = PREFIX_ORDER[i], PREFIX_ORDER[j]
            qa_list = list(QUERY_ORDER)
            qb_list = _seeded_derangement(list(QUERY_ORDER), rng)
            for qa, qb in zip(qa_list, qb_list, strict=True):
                a = context_id(pa, qa)
                b = context_id(pb, qb)
                pairs.append(Pair(f"x--{a}--{b}", "cross", a, b))
    assert len(pairs) == 60, len(pairs)
    assert len({p.pair_id for p in pairs}) == 60
    for p in pairs:
        key_a = (_prefix_rank(p.prefix_a), _query_rank(p.query_a))
        key_b = (_prefix_rank(p.prefix_b), _query_rank(p.query_b))
        assert key_a < key_b, f"non-canonical direction: {p}"
    return pairs


# ── shuffled-donor derangement (the ONE null; plan §4.2) ──────────────


def donor_derangement(pairs: Sequence[Pair], seed: int = SEED) -> dict[str, str]:
    """Seeded donor assignment (recipient pair_id -> donor pair_id), within setting.

    Constraints (plan §4.2 null): no self-donation anywhere; matched-query
    Type-A donors are CONSTRAINED cross-prefix-pair (a same-prefix-pair
    donor's prefix-end Δ duplicates the recipient's — prefix states are
    query-independent, the Type-B derivation). The donor pair id is recorded
    per null cell by the caller (``null_cells.jsonl``).
    """
    rng = random.Random(seed)
    out: dict[str, str] = {}
    for setting in ("matched_prefix", "matched_query", "cross"):
        group = [p for p in pairs if p.setting == setting]
        assert len(group) >= 2, (setting, len(group))
        ids = [p.pair_id for p in group]
        by_id = {p.pair_id: p for p in group}
        for _ in range(10_000):
            perm = ids[:]
            rng.shuffle(perm)
            ok = all(d != r for r, d in zip(ids, perm, strict=True))
            if ok and setting == "matched_query":
                ok = all(
                    by_id[d].prefix_pair() != by_id[r].prefix_pair()
                    for r, d in zip(ids, perm, strict=True)
                )
            if ok:
                out.update(dict(zip(ids, perm, strict=True)))
                break
        else:
            raise RuntimeError(f"no valid donor derangement for {setting} in 10000 attempts")
    assert len(out) == len(pairs)
    return out


# ── direction constructors (Type A / Type B) ──────────────────────────


def type_a_delta(v: Mapping[str, torch.Tensor], pair: Pair) -> torch.Tensor:
    """Type-A pair difference Δ = V(B) - V(A) (any slot/layer grain the bank carries)."""
    da, db = v[pair.a], v[pair.b]
    assert da.shape == db.shape, (da.shape, db.shape)
    return db - da


def prefix_centroid(v: Mapping[str, torch.Tensor], prefix: str) -> torch.Tensor:
    """Type-B centroid: mean over queries of matched-query bare→P Type-A differences.

    Reference = the bare-prefix centroid, so ``centroid_bare ≡ 0`` (plan §4.2).
    """
    assert prefix in PREFIX_ORDER, prefix
    ref = v[context_id("bare", QUERY_ORDER[0])]
    if prefix == "bare":
        return torch.zeros_like(ref)
    diffs = [v[context_id(prefix, q)] - v[context_id("bare", q)] for q in QUERY_ORDER]
    shape = diffs[0].shape
    for d in diffs:
        assert d.shape == shape, (d.shape, shape)
    return torch.stack(diffs).mean(dim=0)


def type_b_delta(v: Mapping[str, torch.Tensor], pair: Pair) -> torch.Tensor:
    """Type-B Δ for a matched-query pair (P_a, q) → (P_b, q): centroid_{P_b} - centroid_{P_a}."""
    assert pair.setting == "matched_query", pair
    return prefix_centroid(v, pair.prefix_b) - prefix_centroid(v, pair.prefix_a)


_TYPE_B_DONOR_SWAP: dict[str, str] = {"bare": "bare", "persona": "conv", "conv": "persona"}


def type_b_donor_delta(v: Mapping[str, torch.Tensor], pair: Pair) -> tuple[torch.Tensor, str]:
    """Norm-matched shuffled donor for a Type-B cell (plan §4.2 null constraint (a)).

    The donor is the persona↔conv-SWAPPED centroid direction — "the OTHER
    prefix's centroid" (body verbatim); the pool has exactly ONE non-self
    centroid per Type-B direction: bare→persona gets centroid_conv,
    bare→conv gets centroid_persona, and persona→conv gets conv→persona
    (the REVERSED direction — the only non-self member of the pool).
    Returns ``(donor delta norm-matched position-wise to the recipient's
    Type-B delta, donor direction label)`` — the label is recorded per null
    cell alongside the donor pair id.
    """
    assert pair.setting == "matched_query", pair
    donor_a = _TYPE_B_DONOR_SWAP[pair.prefix_a]
    donor_b = _TYPE_B_DONOR_SWAP[pair.prefix_b]
    donor_raw = prefix_centroid(v, donor_b) - prefix_centroid(v, donor_a)
    recipient = type_b_delta(v, pair)
    return norm_match(donor_raw, recipient), f"centroid:{donor_a}->{donor_b}"


def norm_match(donor: torch.Tensor, recipient: torch.Tensor) -> torch.Tensor:
    """Rescale ``donor`` POSITION-WISE to the recipient's per-position L2 norm.

    Norms are taken over the LAST dim; every leading dim indexes a position
    (plan §12 assumption 15 — position-wise norm matching). A zero-norm
    recipient position maps to zero; a zero-norm donor position against a
    nonzero recipient fails loud (nothing to rescale).
    """
    assert donor.shape == recipient.shape, (donor.shape, recipient.shape)
    dn = donor.norm(dim=-1, keepdim=True)
    rn = recipient.norm(dim=-1, keepdim=True)
    bad = (dn == 0) & (rn > 0)
    assert not bool(bad.any()), "zero-norm donor position against a nonzero recipient position"
    tiny = torch.finfo(donor.dtype).tiny if donor.is_floating_point() else 1
    scale = torch.where(rn > 0, rn / dn.clamp_min(tiny), torch.zeros_like(rn))
    return donor * scale


# ── rubric builders (descriptor strings per setting) ──────────────────


def f_query_rubric(query_id: str) -> str:
    """Graded 0-100 rubric core: does the response answer THIS query? (F_query pair)."""
    assert query_id in QUERY_ORDER, query_id
    return (
        "To what extent is the response an answer to the following question, as opposed "
        f'to some other question? Question: "{QUERIES[query_id]}" '
        "Score 0-100: 0 = the response does not address this question at all; 50 = the "
        "response partially addresses it or mixes it with other topics; 100 = the response "
        "is fully and specifically an answer to this question."
    )


def f_prefix_rubric(prefix: str) -> str:
    """Graded 0-100 rubric core: does the response express THIS prefix? (F_prefix pair)."""
    assert prefix in PREFIX_ORDER, prefix
    return (
        f"To what extent does the response express {PREFIX_DESCRIPTORS[prefix]}? "
        "Judge the response's register/persona/carry-over only — NOT whether its content "
        "is correct or useful. Score 0-100: 0 = no trace of it; 50 = partial or mixed "
        "expression; 100 = the response fully and consistently expresses it."
    )


def rubric_pair(pair: Pair, kind: str) -> tuple[str, str]:
    """The (rubric_A, rubric_B) core strings for a pair under rubric ``kind``.

    ``kind="query"`` needs differing queries (matched-prefix + cross);
    ``kind="prefix"`` needs differing prefixes (matched-query + cross). The
    per-draw behavior contrast is Δ = (judge_B - judge_A)/100 (plan §4.4).
    """
    assert kind in ("query", "prefix"), kind
    assert kind in SETTING_RUBRIC_KINDS[pair.setting], (pair.setting, kind)
    if kind == "query":
        assert pair.query_a != pair.query_b, pair
        return f_query_rubric(pair.query_a), f_query_rubric(pair.query_b)
    assert pair.prefix_a != pair.prefix_b, pair
    return f_prefix_rubric(pair.prefix_a), f_prefix_rubric(pair.prefix_b)


# ── serializable manifest (frozen bank spec; uploaded as bank.json by P1) ──


def bank_manifest(seed: int = SEED) -> dict:
    """JSON-serializable frozen bank spec: contexts, pairs, donors, rubrics."""
    pairs = build_pairs(seed)
    return {
        "issue": 2094,
        "seed": seed,
        "prefix_order": list(PREFIX_ORDER),
        "query_order": list(QUERY_ORDER),
        "queries": dict(QUERIES),
        "persona_system": PERSONA_SYSTEM,
        "conv_history": [
            {"role": "user", "content": CONV_USER_TURN},
            {"role": "assistant", "content": CONV_ASSISTANT_TURN},
        ],
        "contexts": build_contexts(),
        "pairs": [{"pair_id": p.pair_id, "setting": p.setting, "a": p.a, "b": p.b} for p in pairs],
        "donor_derangement": donor_derangement(pairs, seed),
        "prefix_descriptors": dict(PREFIX_DESCRIPTORS),
        "coherence_rubric": COHERENCE_RUBRIC,
        "truncation_clause": TRUNCATION_CLAUSE,
        "setting_rubric_kinds": {k: list(v) for k, v in SETTING_RUBRIC_KINDS.items()},
    }
