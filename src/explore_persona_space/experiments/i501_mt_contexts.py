# ruff: noqa: RUF002, RUF003
"""Issue #501 multi-turn drift + length-matched neutral contexts.

Plan v2 §4.4 + §4.5. The single source of truth for the 12 NEW eval-only
multi-turn target contexts that supplement #489's 24 single-turn union panel:

  - MT01..MT08 — drift contexts (4 #377 domains × 2 turn-depth slots k∈{10,14}).
  - MN01..MN04 — length-matched-neutral controls (one per drift domain, depth=
    chosen so the total token count ≤ the corresponding drift slot's mean).

Each MT/MN context is a transformation ``T_j(q) = [history (k prior turns)] +
[{user: q}]`` where ``q`` is the held-out Q'_probe and the user-question slot
is at turn (k+1). The predictor read = last input token of
``apply_chat_template(history + [{user: q}], add_generation_prompt=True)``.
The on-policy generation under adapter_i then emits the response R; ΔG =
trained − base log P(' ※') at the post-R slot — IDENTICAL primitive to #489's
``i489_phase4_eval_onpolicy.py``.

The conversations themselves are NOT included here; they are pulled at runtime
from #377's HF Hub corpora at the pinned revision (see ``i501_phase0_load_corpora``).
The dataclass holds only the *selector* (domain, k, deterministic-hash indices)
and the metadata; the actual prefix history list is attached after corpus load.

Plan v2 D2 — per-(domain, k) conversation selection:
``sha256(domain + str(k) + "seed42").hexdigest()[:8] modulo per_domain_count``
→ 5 distinct indices per slot. Stable across runs by construction.

The chat-template builder ``build_mt_prompt(ctx, q, tok)`` uses the same
``add_generation_prompt=True`` semantics as #489's ``build_union_prompt`` so the
post-R slot is computed identically across the merged panel.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Constants — domain sets + pinned revision
# ---------------------------------------------------------------------------

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_DATA_REVISION = "54a80fdf4c2e863e0b9885010a708321071b70ef"
DRIFT_HUB_PATH = "issue377_drift/v1/drift_conversations.jsonl"
INCONTEXT_HUB_PATH = "issue377_incontext/v1/incontext_conversations.jsonl"

# Plan §4.4 — domain mapping per (MT, MN) row.
DRIFT_DOMAINS: tuple[str, ...] = ("coding", "writing", "therapy", "philosophy")
NEUTRAL_DOMAINS: tuple[str, ...] = ("math", "history", "factual_qa", "code_review")

# Plan §4.4 / Reproducibility — depth slots.
DEPTH_SLOTS: tuple[int, ...] = (10, 14)

# How many conversations per (domain, k) we pull to compute a stable mean cosine
# vector across the predictor read (5 conversations × 50 probes = 250 forward
# passes per MT context; plan §4.4).
N_CONVERSATIONS_PER_SLOT = 5

# Per-domain conversation count (verified at the #377 pinned revision; plan
# Assumption 4). Used both for the deterministic hash-mod-N selection AND for
# the Phase-0 sanity check.
PER_DOMAIN_DRIFT_COUNT: dict[str, int] = {
    "therapy": 49,
    "coding": 50,
    "philosophy": 50,
    "writing": 50,
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MTContext:
    """One multi-turn (drift OR length-matched-neutral) target context.

    The ``history`` field starts empty and is populated by
    :func:`attach_history_from_corpus` at corpus-load time. The dataclass
    therefore describes *which* conversations to pull (deterministically) and
    *how* to slice them; the slices themselves arrive at Phase 0.
    """

    cid: str  # MT01..MT08, MN01..MN04
    name: str  # plain-English label
    domain: str  # one of DRIFT_DOMAINS ∪ NEUTRAL_DOMAINS
    k: int  # turn depth (10 or 14); for MN rows this is the SLICE DEPTH TARGET
    arm: str  # "drift" or "neutral"
    is_strong_kind: int  # 1 if in #377-flagged strong-kind set (therapy + philosophy@k=14)
    # For MN rows: the tuple of matched drift cids whose 5-conversation-
    # mean whitespace-token count the MN prefix length-matches against
    # (plan §4.4 table: MN01↔(MT01, MT02), MN02↔(MT03, MT04), MN03↔(MT05,
    # MT06), MN04↔(MT07, MT08) — pair-mean per plan-as-written). For MT
    # rows: empty tuple.
    matched_drift_cids: tuple[str, ...] = field(default=())
    # `histories` is a tuple of conversation prefixes; each prefix is a list of
    # chat-template-shaped {role, content} dicts ending on an assistant turn.
    # Populated at runtime by attach_history_from_corpus(); empty at module import.
    histories: tuple[tuple[dict, ...], ...] = field(default=())


# ---------------------------------------------------------------------------
# The 12 NEW contexts (the experimental axis)
# ---------------------------------------------------------------------------

# Plan §4.4 strong-kind designation:
#   MT05 (therapy k=10), MT06 (therapy k=14), MT08 (philosophy k=14).
_STRONG_KIND_CIDS = {"MT05", "MT06", "MT08"}


def _build_mt_targets() -> tuple[MTContext, ...]:
    """Construct MT01..MT08 + MN01..MN04 as module-level frozen contexts."""
    mt_rows: list[MTContext] = []
    # MT01..MT08: 4 drift domains × 2 depth slots, ordered to match the plan
    # §4.4 table.
    cid_counter = 1
    for domain in DRIFT_DOMAINS:
        for k in DEPTH_SLOTS:
            cid = f"MT{cid_counter:02d}"
            name = f"{domain.capitalize()} drift, k={k}"
            mt_rows.append(
                MTContext(
                    cid=cid,
                    name=name,
                    domain=domain,
                    k=k,
                    arm="drift",
                    is_strong_kind=int(cid in _STRONG_KIND_CIDS),
                )
            )
            cid_counter += 1
    # MN01..MN04: one length-matched-neutral per drift domain, matched
    # against the MEAN of the (k=10, k=14) drift pair per plan §4.4 table:
    #   MN01 ↔ (MT01, MT02) — math neutral, length-matched to coding drift mean
    #   MN02 ↔ (MT03, MT04) — history neutral, length-matched to writing drift mean
    #   MN03 ↔ (MT05, MT06) — factual_qa neutral, length-matched to therapy drift mean
    #   MN04 ↔ (MT07, MT08) — code_review neutral, length-matched to philosophy drift mean
    # The slice itself is the longest even-parity neutral prefix whose
    # cumulative whitespace-token count ≤ pair-mean (port of #377's
    # `_length_matched_slice_n` with the pair-mean as target). The `k`
    # field on the MN row is informational (slice depth is data-driven,
    # not k-driven); we record `k=14` as a placeholder pointing at the
    # deeper drift slot for downstream depth bookkeeping.
    mn_targets: list[tuple[str, tuple[str, str]]] = [
        ("MN01", ("MT01", "MT02")),  # math ↔ coding pair
        ("MN02", ("MT03", "MT04")),  # history ↔ writing pair
        ("MN03", ("MT05", "MT06")),  # factual_qa ↔ therapy pair
        ("MN04", ("MT07", "MT08")),  # code_review ↔ philosophy pair
    ]
    for mn_cid, matched_pair in mn_targets:
        domain = NEUTRAL_DOMAINS[int(mn_cid[2:]) - 1]
        name = f"{domain.replace('_', '-').capitalize()} neutral, length-matched"
        mt_rows.append(
            MTContext(
                cid=mn_cid,
                name=name,
                domain=domain,
                k=14,  # informational placeholder; slice is data-driven
                arm="neutral",
                is_strong_kind=0,
                matched_drift_cids=matched_pair,
            )
        )
    return tuple(mt_rows)


MT_CONTEXTS: tuple[MTContext, ...] = _build_mt_targets()
MT_BY_CID: dict[str, MTContext] = {c.cid: c for c in MT_CONTEXTS}
MT_CIDS: tuple[str, ...] = tuple(c.cid for c in MT_CONTEXTS)
DRIFT_CIDS: tuple[str, ...] = tuple(c.cid for c in MT_CONTEXTS if c.arm == "drift")
NEUTRAL_CIDS: tuple[str, ...] = tuple(c.cid for c in MT_CONTEXTS if c.arm == "neutral")

# Sanity: 8 drift + 4 neutral = 12 total.
assert len(MT_CONTEXTS) == 12, f"expected 12 MT/MN contexts, got {len(MT_CONTEXTS)}"
assert len(DRIFT_CIDS) == 8, f"expected 8 drift contexts, got {len(DRIFT_CIDS)}"
assert len(NEUTRAL_CIDS) == 4, f"expected 4 neutral contexts, got {len(NEUTRAL_CIDS)}"


# ---------------------------------------------------------------------------
# Deterministic conversation index selection
# ---------------------------------------------------------------------------


def deterministic_conversation_indices(
    domain: str, k: int, per_domain_count: int, n_picks: int = N_CONVERSATIONS_PER_SLOT
) -> tuple[int, ...]:
    """Plan §4.4 / D2 — deterministic-hash-mod-N selection of conversation
    indices per (domain, k) slot.

    Algorithm:
      seed-hash an 8-hex digest of ``f"{domain}{k}seed42"`` to an int;
      use it as the seed of a stdlib ``random.Random``; then sample
      ``n_picks`` distinct indices in ``range(per_domain_count)`` without
      replacement. Stable across invocations by construction.

    Returns a sorted tuple of indices.
    """
    if per_domain_count < n_picks:
        raise ValueError(
            f"per_domain_count={per_domain_count} for domain={domain!r} is "
            f"below the requested n_picks={n_picks}; cannot pick distinct indices"
        )
    digest_hex = hashlib.sha256(f"{domain}{k}seed42".encode()).hexdigest()[:8]
    seed_int = int(digest_hex, 16)
    # Use stdlib Random to derive a deterministic sample.
    import random as _random

    rng = _random.Random(seed_int)
    return tuple(sorted(rng.sample(range(per_domain_count), n_picks)))


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def build_mt_prompt(history: tuple[dict, ...], q: str, tokenizer) -> str:
    """Chat-templated prompt for a single MT/MN history + held-out question.

    Plan §4.4 — ``tokenizer.apply_chat_template(history + [{user: q}],
    tokenize=False, add_generation_prompt=True)``. Identical to #489's
    ``build_union_prompt`` semantics (last-input-token is the assistant-turn
    open-tag), so the predictor read and the post-R logprob slot are computed
    identically across the merged panel.

    ``history`` must end on an assistant turn (sliced at corpus load time).
    """
    if history and history[-1]["role"] != "assistant":
        raise ValueError(
            "build_mt_prompt: history must end on assistant turn; got "
            f"last_role={history[-1]['role']!r}"
        )
    messages = [*list(history), {"role": "user", "content": q}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------------------
# Helpers for tests + Phase 0 sanity checks
# ---------------------------------------------------------------------------


def is_drift(cid: str) -> bool:
    """True iff cid is one of MT01..MT08 (drift arm)."""
    return cid in DRIFT_CIDS


def is_neutral(cid: str) -> bool:
    """True iff cid is one of MN01..MN04 (length-matched-neutral arm)."""
    return cid in NEUTRAL_CIDS


def assert_no_marker_in_history(history: tuple[dict, ...], marker_text: str, cid: str) -> None:
    """Defense-in-depth (plan Assumption 16): assert the marker string doesn't
    appear in any turn's content of a loaded history slice.

    #377's ``post_gen_sanity_checks`` already guarantees this for the published
    corpus; this assertion is a load-time defense against silent corpus drift.
    """
    for idx, turn in enumerate(history):
        content = turn.get("content", "")
        if marker_text in content:
            raise AssertionError(
                f"i501_mt_contexts: MARKER_TEXT={marker_text!r} found in {cid} "
                f"turn {idx} (role={turn.get('role')!r}); corpus drifted from "
                "#377's post-gen sanity check"
            )
