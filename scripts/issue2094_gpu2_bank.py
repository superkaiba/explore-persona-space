"""Issue #2094 — gpu2_mq_replacement_prefix bank extension (round-scoped, ADDITIVE).

The parent bank module (``experiments/issue2094/bank.py``) is byte-UNCHANGED —
the fu2 ExtraSlot precedent. This module defines everything the replacement
round adds:

- the replacement conversation prefix ``conv2`` — the PARENT's construction
  method (one user turn + one assistant turn of the same topic/length class,
  no system turn), with content chosen so the register actually CARRIES into
  the next answer: the user's turn states a standing style preference and the
  assistant's turn acknowledges it. Diagnosis ground (committed judge scores,
  ``eval_results/issue_2094/judge/scores``): the parent conv prefix's register
  does NOT carry — conv-generated anchor draws scored fp-conv 0.2/100 and
  fp-bare 74.9/100, i.e. judged plain-assistant — which is exactly why the 5
  bare-vs-conv matched-query pairs sit at |sep| < 0.5.
- the 5 new contexts ``conv2__q1..q5`` and the 5 re-formed matched-query pairs
  ``mq--bare__qK--conv2__qK`` (canonical direction bare -> conv2).
- the seeded donor assignment for the shuffled-donor null: donors drawn from
  the PARENT's 15 matched-query pairs (fresh seed, recorded). The parent's
  matched-query cross-prefix-pair constraint is satisfied BY CONSTRUCTION —
  every donor's prefix pair is one of (bare,persona)/(bare,conv)/
  (persona,conv), never (bare,conv2) — and drawing donors WITHIN the new set
  is structurally impossible anyway (all 5 recipients share the (bare,conv2)
  prefix pair, which the parent constraint forbids).
- the ``fp-conv2`` rubric CORE mirroring ``bank.f_prefix_rubric``'s instrument
  wording (the judge wraps it in the production reason-then-score template).

Consumed by ``scripts/issue2094_gpu2.py`` (pod driver) and
``scripts/issue2094_judge.py --gpu2`` (additive judge extension).
"""

from __future__ import annotations

import hashlib
import json
import random

from explore_persona_space.experiments.issue2094 import bank as BANK

GPU2_SEED = 20943  # fresh donor seed (parent bank 2094 / folds 20942 / bootstrap 20941)
CONV2_PREFIX = "conv2"

# ── the replacement conversation prefix (verbatim; freeze at commit) ────

CONV2_USER_TURN = (
    "I'm planning my daughter's 7th birthday party this weekend — any fun ideas? 🎉 "
    "Oh, and one thing: keep that excited party energy going in EVERYTHING you tell "
    "me from here on, whatever I ask about next — big enthusiasm, emojis, "
    "exclamation marks, the works!!"
)

CONV2_ASSISTANT_TURN = (
    "Oh how FUN!! 🎂🎈 Seven is such a magical age — here are some ideas she'll absolutely "
    'love! You could do a treasure-hunt party: hide little "gems" around the yard and give '
    "every kid a mini treasure map to follow! 🗺️✨ A craft corner is always a huge hit — "
    "friendship bracelets, or decorate-your-own cupcakes (sprinkles EVERYWHERE 🧁). For "
    "games: musical statues, a piñata (obviously!! 🪅), and a balloon-pop countdown with a "
    "tiny prize tucked inside each balloon. Don't forget a photo corner with silly props — "
    "the pictures end up being the best souvenirs! 📸🎉 And for the cake: rainbow layers, "
    "trust me on this one. She is going to have the BEST day!! And you've got it — PARTY "
    "MODE stays ON from here on out: whatever you ask me next, big or small, serious or "
    "silly, you're getting the full confetti treatment — emojis, exclamation points, ALL "
    "the excitement!! 🎊🥳"
)

CONV2_DESCRIPTOR = (
    "the carried-over register of an ongoing, upbeat conversation about planning a "
    "young child's birthday party, in which the user asked for the excited party "
    "energy to be kept up in every later answer: enthusiastic, exclamatory, "
    "emoji-rich, and possibly referring back to the party planning"
)

# The 5 weakly-separated matched-query pairs this round replaces (diagnosis
# pinned against the committed f_metrics/anchors.jsonl in
# tests/test_issue2094_gpu2.py; |sep| < 0.5 on the prefix rubric kind).
WEAK_PAIR_IDS: tuple[str, ...] = tuple(f"mq--bare__{q}--conv__{q}" for q in BANK.QUERY_ORDER)


def conv2_context_id(query_id: str) -> str:
    assert query_id in BANK.QUERY_ORDER, query_id
    return f"{CONV2_PREFIX}__{query_id}"


def build_gpu2_contexts() -> dict[str, dict]:
    """The 5 new conv2 contexts (the parent context-dict shape; multi-turn
    ``history`` prefix, no system turn — the parent conv construction)."""
    contexts: dict[str, dict] = {}
    for q in BANK.QUERY_ORDER:
        cid = conv2_context_id(q)
        contexts[cid] = {
            "id": cid,
            "prefix": CONV2_PREFIX,
            "query_id": q,
            "system": None,
            "history": [
                {"role": "user", "content": CONV2_USER_TURN},
                {"role": "assistant", "content": CONV2_ASSISTANT_TURN},
            ],
            "user": BANK.QUERIES[q],
        }
    assert len(contexts) == 5, len(contexts)
    return contexts


def build_extended_contexts() -> dict[str, dict]:
    """Parent 15 contexts + the 5 conv2 contexts (parent order first)."""
    contexts = BANK.build_contexts()
    extra = build_gpu2_contexts()
    assert not (set(contexts) & set(extra)), "conv2 ids collide with parent context ids"
    contexts.update(extra)
    assert len(contexts) == 20, len(contexts)
    return contexts


def build_gpu2_pairs() -> list[BANK.Pair]:
    """The 5 re-formed matched-query pairs mq--bare__qK--conv2__qK.

    Canonical direction bare -> conv2 (edits move A toward B, the parent
    convention; ``bare`` precedes every other prefix in the parent rank).
    """
    pairs = [
        BANK.Pair(
            f"mq--bare__{q}--{conv2_context_id(q)}",
            "matched_query",
            f"bare__{q}",
            conv2_context_id(q),
        )
        for q in BANK.QUERY_ORDER
    ]
    assert len({p.pair_id for p in pairs}) == 5
    parent_ids = {p.pair_id for p in BANK.build_pairs()}
    assert not ({p.pair_id for p in pairs} & parent_ids), "gpu2 pair ids collide with parent"
    return pairs


def parent_mq_pairs() -> list[BANK.Pair]:
    """The parent's 15 matched-query pairs (the donor pool + walk group)."""
    return [p for p in BANK.build_pairs() if p.setting == "matched_query"]


def gpu2_donor_map(seed: int = GPU2_SEED) -> dict[str, str]:
    """Seeded donor assignment for the gpu2 null arm, CLOSED under the
    ``_resolve_donor`` walk.

    Recipients (the 5 new pairs) map to 5 DISTINCT donors sampled from the
    parent's 15 matched-query pairs; the parent matched-query derangement
    rides along verbatim so the walk's ``donor_map[donor_id]`` continuation
    and the sorted fallback stay well-defined. No self-donation (pools are
    disjoint) and the parent cross-prefix-pair constraint holds by
    construction (no donor's prefix pair is (bare, conv2)).
    """
    pool = sorted(p.pair_id for p in parent_mq_pairs())
    assert len(pool) == 15, len(pool)
    rng = random.Random(seed)
    recipients = [p.pair_id for p in build_gpu2_pairs()]
    donors = rng.sample(pool, k=len(recipients))
    out = dict(zip(recipients, donors, strict=True))
    parent_derangement = BANK.donor_derangement(BANK.build_pairs())
    for pid in pool:
        out[pid] = parent_derangement[pid]
    assert len(out) == 20, len(out)
    assert all(r != d for r, d in out.items()), "self-donation"
    return out


def gpu2_pairs_by_id() -> dict[str, BANK.Pair]:
    """Recipient + donor-pool pairs — the ``pairs_by_id`` ``run_block`` needs."""
    out = {p.pair_id: p for p in build_gpu2_pairs()}
    out.update({p.pair_id: p for p in parent_mq_pairs()})
    assert len(out) == 20, len(out)
    return out


def f_prefix_rubric_conv2() -> str:
    """Graded 0-100 rubric core for the conv2 prefix — ``bank.f_prefix_rubric``'s
    instrument wording verbatim, with the conv2 descriptor spliced in."""
    return (
        f"To what extent does the response express {CONV2_DESCRIPTOR}? "
        "Judge the response's register/persona/carry-over only — NOT whether its content "
        "is correct or useful. Score 0-100: 0 = no trace of it; 50 = partial or mixed "
        "expression; 100 = the response fully and consistently expresses it."
    )


def gpu2_manifest(seed: int = GPU2_SEED) -> dict:
    """JSON-serializable frozen gpu2 extension spec (regime-key ingredient)."""
    return {
        "issue": 2094,
        "round": "gpu2_mq_replacement_prefix",
        "seed": seed,
        "conv2_user_turn": CONV2_USER_TURN,
        "conv2_assistant_turn": CONV2_ASSISTANT_TURN,
        "conv2_descriptor": CONV2_DESCRIPTOR,
        "weak_pair_ids": list(WEAK_PAIR_IDS),
        "contexts": build_gpu2_contexts(),
        "pairs": [
            {"pair_id": p.pair_id, "setting": p.setting, "a": p.a, "b": p.b}
            for p in build_gpu2_pairs()
        ],
        "donor_map": gpu2_donor_map(seed),
    }


def gpu2_manifest_sha(seed: int = GPU2_SEED) -> str:
    payload = json.dumps(gpu2_manifest(seed), sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()
