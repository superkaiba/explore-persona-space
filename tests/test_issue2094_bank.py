"""CPU unit tests for the issue #2094 context bank / pairs / donors / rubrics.

Pure-CPU, no network: direction math runs on synthetic V banks (random
tensors keyed by context id); ``prefix_end_index_multi`` runs on a stub
tokenizer + hand-built id lists (special-token occurrence logic only).
"""

from __future__ import annotations

import json
import random

import pytest
import torch

from explore_persona_space.experiments.issue2094 import bank
from explore_persona_space.experiments.issue2094.bank import (
    COHERENCE_RUBRIC,
    PREFIX_DESCRIPTORS,
    PREFIX_ORDER,
    QUERY_ORDER,
    SEED,
    TRUNCATION_CLAUSE,
    Pair,
    bank_manifest,
    build_contexts,
    build_pairs,
    context_id,
    context_messages_2094,
    donor_derangement,
    norm_match,
    prefix_centroid,
    prefix_end_index_multi,
    rubric_pair,
    type_a_delta,
    type_b_delta,
    type_b_donor_delta,
)


def _rank(cid: str) -> tuple[int, int]:
    prefix, q = cid.split("__")
    return (PREFIX_ORDER.index(prefix), QUERY_ORDER.index(q))


def _synthetic_v(shape=(4, 8), seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {cid: torch.randn(*shape, generator=g) for cid in build_contexts()}


# ── contexts ──────────────────────────────────────────────────────────


def test_contexts_count_ids_and_fields():
    contexts = build_contexts()
    assert len(contexts) == 15
    assert list(contexts) == [context_id(p, q) for p in PREFIX_ORDER for q in QUERY_ORDER]
    for cid, ctx in contexts.items():
        assert ctx["id"] == cid
        assert ctx["user"] == bank.QUERIES[ctx["query_id"]]


def test_context_messages_shapes_per_prefix():
    contexts = build_contexts()
    m_bare = context_messages_2094(contexts["bare__q1"])
    assert [m["role"] for m in m_bare] == ["user"]  # template default system block
    m_persona = context_messages_2094(contexts["persona__q3"])
    assert [m["role"] for m in m_persona] == ["system", "user"]
    assert m_persona[0]["content"] == bank.PERSONA_SYSTEM
    m_conv = context_messages_2094(contexts["conv__q5"])
    assert [m["role"] for m in m_conv] == ["user", "assistant", "user"]
    assert m_conv[0]["content"] == bank.CONV_USER_TURN
    assert m_conv[-1]["content"] == bank.QUERIES["q5"]
    # conv prefix is output-affecting by design: an upbeat ~120-word reply
    n_words = len(bank.CONV_ASSISTANT_TURN.split())
    assert 80 <= n_words <= 170, n_words


def test_bare_descriptor_is_plan_verbatim():
    assert PREFIX_DESCRIPTORS["bare"] == (
        "the plain register of a default assistant, with no persona and no carry-over "
        "from any earlier conversation"
    )


# ── pairs ─────────────────────────────────────────────────────────────


def test_pair_counts_and_canonical_direction():
    pairs = build_pairs()
    by_setting = {s: [p for p in pairs if p.setting == s] for s in bank.SETTING_RUBRIC_KINDS}
    assert len(by_setting["matched_prefix"]) == 30
    assert len(by_setting["matched_query"]) == 15
    assert len(by_setting["cross"]) == 15
    assert len(pairs) == 60
    for p in pairs:
        assert _rank(p.a) < _rank(p.b), p  # ONE canonical direction: A -> B
    for p in by_setting["matched_prefix"]:
        assert p.prefix_a == p.prefix_b and p.query_a != p.query_b
    for p in by_setting["matched_query"]:
        assert p.query_a == p.query_b and p.prefix_a != p.prefix_b
    dirs = {(p.prefix_a, p.prefix_b) for p in by_setting["matched_query"]}
    assert dirs == {("bare", "persona"), ("bare", "conv"), ("persona", "conv")}
    for p in by_setting["cross"]:
        assert p.prefix_a != p.prefix_b and p.query_a != p.query_b


def test_cross_pairs_balanced_and_seeded():
    pairs = [p for p in build_pairs() if p.setting == "cross"]
    by_prefix_pair: dict[tuple[str, str], list[Pair]] = {}
    for p in pairs:
        by_prefix_pair.setdefault(p.prefix_pair(), []).append(p)
    assert set(by_prefix_pair) == {
        ("bare", "persona"),
        ("bare", "conv"),
        ("persona", "conv"),
    }
    for group in by_prefix_pair.values():
        assert len(group) == 5
        # each query appears on each side exactly once (>=1 required)
        assert sorted(p.query_a for p in group) == list(QUERY_ORDER)
        assert sorted(p.query_b for p in group) == list(QUERY_ORDER)
    # deterministic under the frozen seed
    again = [p for p in build_pairs() if p.setting == "cross"]
    assert [(p.a, p.b) for p in again] == [(p.a, p.b) for p in pairs]
    # a different seed produces a DIFFERENT (still balanced) draw
    other = [p for p in build_pairs(seed=SEED + 1) if p.setting == "cross"]
    assert [(p.a, p.b) for p in other] != [(p.a, p.b) for p in pairs]


# ── donor derangement ─────────────────────────────────────────────────


def test_donor_derangement_constraints():
    pairs = build_pairs()
    donors = donor_derangement(pairs)
    by_id = {p.pair_id: p for p in pairs}
    assert set(donors) == {p.pair_id for p in pairs}
    for recipient_id, donor_id in donors.items():
        assert donor_id != recipient_id  # derangement
        r, d = by_id[recipient_id], by_id[donor_id]
        assert r.setting == d.setting  # within setting-type
        if r.setting == "matched_query":
            # v4 constraint: cross-prefix-pair donors (a same-prefix-pair
            # donor's prefix-end Δ duplicates the recipient's)
            assert d.prefix_pair() != r.prefix_pair(), (recipient_id, donor_id)
    # bijection within each setting (a permutation, so donor ids cover the pool)
    for setting in bank.SETTING_RUBRIC_KINDS:
        group_ids = {p.pair_id for p in pairs if p.setting == setting}
        assert {donors[i] for i in group_ids} == group_ids
    # deterministic under the frozen seed
    assert donor_derangement(build_pairs()) == donors


# ── direction constructors ────────────────────────────────────────────


def test_type_a_delta_is_b_minus_a():
    v = _synthetic_v()
    pairs = build_pairs()
    p = pairs[0]
    assert torch.equal(type_a_delta(v, p), v[p.b] - v[p.a])


def test_prefix_centroid_math():
    v = _synthetic_v()
    for prefix in ("persona", "conv"):
        expected = torch.stack(
            [v[context_id(prefix, q)] - v[context_id("bare", q)] for q in QUERY_ORDER]
        ).mean(dim=0)
        assert torch.allclose(prefix_centroid(v, prefix), expected)
    assert torch.equal(prefix_centroid(v, "bare"), torch.zeros_like(v["bare__q1"]))


def test_type_b_delta_uses_centroids():
    v = _synthetic_v()
    mq = [p for p in build_pairs() if p.setting == "matched_query"]
    bp = next(p for p in mq if (p.prefix_a, p.prefix_b) == ("bare", "persona"))
    assert torch.allclose(type_b_delta(v, bp), prefix_centroid(v, "persona"))
    pc = next(p for p in mq if (p.prefix_a, p.prefix_b) == ("persona", "conv"))
    assert torch.allclose(
        type_b_delta(v, pc), prefix_centroid(v, "conv") - prefix_centroid(v, "persona")
    )
    mp = next(p for p in build_pairs() if p.setting == "matched_prefix")
    with pytest.raises(AssertionError):
        type_b_delta(v, mp)


def test_type_b_donor_swap_and_norm_match():
    v = _synthetic_v()
    mq = [p for p in build_pairs() if p.setting == "matched_query"]
    # bare->persona: donor = centroid_conv (the OTHER prefix's centroid)
    bp = next(p for p in mq if (p.prefix_a, p.prefix_b) == ("bare", "persona"))
    donor, label = type_b_donor_delta(v, bp)
    assert label == "centroid:bare->conv"
    expected = norm_match(prefix_centroid(v, "conv"), type_b_delta(v, bp))
    assert torch.allclose(donor, expected)
    # persona->conv: the swap yields conv->persona — the REVERSED direction
    pc = next(p for p in mq if (p.prefix_a, p.prefix_b) == ("persona", "conv"))
    donor_pc, label_pc = type_b_donor_delta(v, pc)
    assert label_pc == "centroid:conv->persona"
    recipient = type_b_delta(v, pc)
    assert torch.allclose(donor_pc, -recipient, atol=1e-5)  # anti-parallel, norms equal
    assert torch.allclose(donor_pc.norm(dim=-1), recipient.norm(dim=-1), atol=1e-5)


def test_norm_match_position_wise():
    g = torch.Generator().manual_seed(3)
    donor = torch.randn(4, 8, generator=g)
    recipient = torch.randn(4, 8, generator=g)
    recipient[2] = 0.0  # zero-norm recipient position -> zero output
    out = norm_match(donor, recipient)
    assert torch.allclose(out.norm(dim=-1), recipient.norm(dim=-1), atol=1e-6)
    # direction preserved per position (parallel to the donor)
    for p in (0, 1, 3):
        cos = torch.nn.functional.cosine_similarity(out[p], donor[p], dim=0)
        assert cos > 0.9999, (p, cos)
    assert torch.equal(out[2], torch.zeros(8))
    # zero-norm donor position against a NONZERO recipient fails loud
    donor_bad = donor.clone()
    donor_bad[1] = 0.0
    with pytest.raises(AssertionError, match="zero-norm donor"):
        norm_match(donor_bad, torch.randn(4, 8, generator=g))


# ── prefix boundary (multi-turn) ──────────────────────────────────────


class _StubTokenizer:
    """convert_tokens_to_ids-only stub: <|im_start|> -> 7 (no network)."""

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == bank.IM_START_TOKEN, token
        return 7

    def __call__(self, *a, **k):  # pragma: no cover - guard against misuse
        raise NotImplementedError


def _ids_with_im_start_at(occ: list[int], total: int) -> list[int]:
    ids = [1] * total
    for i in occ:
        ids[i] = 7
    return ids


def test_prefix_end_index_multi_single_turn_parity():
    # single-turn render: [system, user, assistant-gen] -> second occurrence,
    # matching steering.prefix_end_index's occ[1]
    ids = _ids_with_im_start_at([0, 10, 20], total=25)
    assert prefix_end_index_multi(_StubTokenizer(), ids) == 10


def test_prefix_end_index_multi_conversation():
    # conv render: [system, user1, assistant1, user2, assistant-gen]
    ids = _ids_with_im_start_at([0, 5, 12, 30, 40], total=45)
    assert prefix_end_index_multi(_StubTokenizer(), ids) == 30


def test_prefix_end_index_multi_requires_three_occurrences():
    ids = _ids_with_im_start_at([0, 10], total=15)
    with pytest.raises(AssertionError, match="expected >=3"):
        prefix_end_index_multi(_StubTokenizer(), ids)


# ── rubrics ───────────────────────────────────────────────────────────


def test_coherence_rubric_carries_verbatim_truncation_clause():
    assert TRUNCATION_CLAUSE == (
        "If the text ends abruptly mid-sentence, treat this as a length cutoff, NOT "
        "incoherence: judge only the text before the cutoff."
    )
    assert TRUNCATION_CLAUSE in COHERENCE_RUBRIC
    for ignored in ("correctness", "safety", "relevance", "completeness", "style", "length"):
        assert ignored in COHERENCE_RUBRIC, ignored
    for anchor in ("0 =", "50 =", "100 ="):
        assert anchor in COHERENCE_RUBRIC, anchor


def test_rubric_pairs_per_setting():
    pairs = build_pairs()
    mp = next(p for p in pairs if p.setting == "matched_prefix")
    ra, rb = rubric_pair(mp, "query")
    assert ra != rb
    assert bank.QUERIES[mp.query_a] in ra and bank.QUERIES[mp.query_b] in rb
    mq = next(p for p in pairs if p.setting == "matched_query")
    ra, rb = rubric_pair(mq, "prefix")
    assert PREFIX_DESCRIPTORS[mq.prefix_a] in ra and PREFIX_DESCRIPTORS[mq.prefix_b] in rb
    x = next(p for p in pairs if p.setting == "cross")
    for kind in bank.SETTING_RUBRIC_KINDS["cross"]:
        ra, rb = rubric_pair(x, kind)
        assert ra != rb
    # wrong kind for the setting fails loud
    with pytest.raises(AssertionError):
        rubric_pair(mq, "query")  # matched-query shares the query
    with pytest.raises(AssertionError):
        rubric_pair(mp, "prefix")  # matched-prefix shares the prefix


# ── manifest ──────────────────────────────────────────────────────────


def test_bank_manifest_is_json_serializable_and_complete():
    manifest = bank_manifest()
    text = json.dumps(manifest, ensure_ascii=False)
    round_trip = json.loads(text)
    assert len(round_trip["contexts"]) == 15
    assert len(round_trip["pairs"]) == 60
    assert len(round_trip["donor_derangement"]) == 60
    assert round_trip["seed"] == SEED
    assert round_trip["truncation_clause"] in round_trip["coherence_rubric"]


def test_seeded_derangement_helper_never_fixes_points():
    rng = random.Random(0)
    items = ["a", "b", "c", "d", "e"]
    for _ in range(50):
        perm = bank._seeded_derangement(items, rng)
        assert sorted(perm) == sorted(items)
        assert all(p != i for p, i in zip(perm, items, strict=True))
