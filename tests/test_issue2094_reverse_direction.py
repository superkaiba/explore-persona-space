"""CPU-only unit tests for the issue #2094 REVERSED-direction driver.

No model, no GPU, no network: reversed-pair construction, donor-assignment
constraints, grid enumeration, floor/ceiling from a synthetic anchor-score
fixture (incl. incoherent-draw exclusion + missing-score fail-loud), the
regime fingerprint, and the null-arm donor-STATE payload on a synthetic bank
(the parent test conventions — ``tests/test_issue2094_run.py``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2094_reverse_direction as REV  # noqa: E402
import issue2094_run as R  # noqa: E402

from explore_persona_space.experiments.issue2094 import bank as BANK  # noqa: E402

N_LAYERS = 28


@pytest.fixture(scope="module")
def parent_pairs() -> list[BANK.Pair]:
    return BANK.build_pairs()


@pytest.fixture(scope="module")
def rev_pairs() -> list[BANK.Pair]:
    return REV.build_rev_pairs()


# ── reversed pairs ────────────────────────────────────────────────────


def test_rev_pairs_are_reversed_matched_query(rev_pairs, parent_pairs):
    assert len(rev_pairs) == 5
    assert len({p.pair_id for p in rev_pairs}) == 5
    for p, q in zip(rev_pairs, BANK.QUERY_ORDER, strict=True):
        assert p.setting == "matched_query"
        assert p.a == f"persona__{q}" and p.b == f"bare__{q}"  # A=persona, B=bare
        assert p.query_a == p.query_b == q  # same query both sides
        # ANTI-canonical by construction: the parent bank never runs bare as
        # a target (its canonical order is bare < persona < conv).
        assert BANK._prefix_rank(p.prefix_a) > BANK._prefix_rank(p.prefix_b)
        assert p.prefix_pair() == ("bare", "persona")
    # None of the reversed ids collide with the parent bank's pair ids.
    assert not ({p.pair_id for p in rev_pairs} & {p.pair_id for p in parent_pairs})


# ── donor assignment ──────────────────────────────────────────────────


def test_donor_assignment_constraints(rev_pairs, parent_pairs):
    donors = REV.rev_donor_assignment(rev_pairs, parent_pairs)
    by_id = {p.pair_id: p for p in parent_pairs}
    assert set(donors) == {p.pair_id for p in rev_pairs}
    for pair in rev_pairs:
        donor = by_id[donors[pair.pair_id]]
        assert donor.pair_id != pair.pair_id  # no self-donation
        assert donor.setting == "matched_query"
        # Every donor's prefix pair differs from (persona, bare) — unordered
        # comparison via the canonical-sorted prefix_pair().
        assert donor.prefix_pair() != ("bare", "persona")
        # State-kind eligibility (replace cells): the donor's target context
        # is never the recipient's own V_B context.
        assert donor.b != pair.b
        assert R._donor_eligible(donor, "ce", pair, "state")
    # Seeded + deterministic; distinct donors (no-replacement sample).
    assert donors == REV.rev_donor_assignment(rev_pairs, parent_pairs)
    assert len(set(donors.values())) == len(donors)


def test_donor_pool_is_the_ten_cross_prefix_pair_mq_pairs(parent_pairs):
    pool = REV.rev_donor_pool(parent_pairs)
    assert len(pool) == 10
    assert all(p.setting == "matched_query" for p in pool)
    assert {p.prefix_pair() for p in pool} == {("bare", "conv"), ("persona", "conv")}


# ── grid enumeration ──────────────────────────────────────────────────


def test_rev_grid_enumeration_counts(rev_pairs):
    fams = REV.enumerate_rev_blocks(rev_pairs, N_LAYERS)
    totals = R.grid_totals(fams)
    assert totals == {
        "n_families": 30,
        "n_blocks": 60,
        "cells_steered": 150,
        "cells_null": 150,
        "cells_total": 300,
    }
    blocks = [b for fam in fams for b in fam]
    assert len({b.key for b in blocks}) == 60
    assert all(b.slot == "ce" and b.dose == "replace" and b.vec_type == "A" for b in blocks)
    variants = {b.layer_variant for b in blocks}
    assert variants == set(R.layer_variant_names(N_LAYERS))  # L0..L27 + both joints
    mode, alpha, payload_kind = R._realized_mode("ce", "replace", "A")
    assert (mode, alpha, payload_kind) == ("replace", 1.0, "state")


def test_smoke_slice_covers_every_arm_class(rev_pairs):
    fams = REV.smoke_rev_blocks(rev_pairs, N_LAYERS)
    variants = {f[0].layer_variant for f in fams}
    assert variants == {"L14", "joint_mid", "joint_all"}  # single + both joint classes
    for steered, null in fams:
        assert steered.arm == "steered" and null.arm == "null"  # donor path runs per class
        assert steered.pair_ids == null.pair_ids == tuple(p.pair_id for p in rev_pairs)


def test_rev_regime_fingerprint_distinct_and_keyed(rev_pairs, parent_pairs):
    cfg = R.build_config(REV.parse_args(["--phase", "grid", "--tiny"]))
    donors = REV.rev_donor_assignment(rev_pairs, parent_pairs)
    fp = REV.rev_regime_fingerprint(cfg, "banksha", donors)
    # Distinct from the PARENT grid's fingerprint (same block-key space) so a
    # parent done-file can never satisfy a reversed-grid resume.
    assert fp != R.regime_fingerprint(cfg, "banksha")
    # Keyed on the donor assignment.
    other = dict(donors)
    k0 = next(iter(other))
    other[k0] = next(v for v in donors.values() if v != donors[k0])
    assert REV.rev_regime_fingerprint(cfg, "banksha", other) != fp
    # And on the parent regime knobs (seed here).
    cfg2 = R.build_config(REV.parse_args(["--phase", "grid", "--tiny", "--seed-base", "43"]))
    assert REV.rev_regime_fingerprint(cfg2, "banksha", donors) != fp


def test_parse_args_namespace_satisfies_parent_build_config():
    """The reused RUN.build_config must find every attribute it reads (the
    reused-module Namespace-shim rule) — defaults resolve on a bare grid argv."""
    cfg = R.build_config(REV.parse_args(["--phase", "grid"]))
    assert cfg.out_root == Path("/workspace/issue2094_rev_out")
    assert cfg.n_layers == 28 and cfg.hidden == 3584
    assert cfg.max_new_tokens == 1024 and cfg.seed_base == 42


# ── floor / ceiling from banked anchor scores ─────────────────────────


def _write_scores_fixture(tmp_path: Path, drop: tuple[str, str, int] | None = None) -> Path:
    """Synthetic fp-bare/fp-persona anchor score shards for pair q1 (4 draws).

    ``drop=(rubric, context, draw)`` omits that one row (the fail-loud leg).
    """
    scores = {
        "fp-bare": {
            ("persona__q1", 0): 10.0,
            ("persona__q1", 1): 20.0,
            ("persona__q1", 2): 30.0,
            ("persona__q1", 3): 99.0,  # incoherent draw — must be EXCLUDED
            ("bare__q1", 0): 90.0,
            ("bare__q1", 1): 80.0,
            ("bare__q1", 2): 100.0,
            ("bare__q1", 3): 70.0,
        },
        "fp-persona": {
            ("persona__q1", 0): 90.0,
            ("persona__q1", 1): 70.0,
            ("persona__q1", 2): 80.0,
            ("persona__q1", 3): 1.0,
            ("bare__q1", 0): 0.0,
            ("bare__q1", 1): 10.0,
            ("bare__q1", 2): 20.0,
            ("bare__q1", 3): 30.0,
        },
    }
    for rid, fname in REV.SCORES_FILES.items():
        rows = [
            {
                "rubric_id": rid,
                "kind": "anchor",
                "context_id": cid,
                "draw": d,
                "score": s,
                "wave": f"{rid}.anchors",
            }
            for (cid, d), s in scores[rid].items()
            if drop != (rid, cid, d)
        ]
        (tmp_path / fname).write_text("".join(json.dumps(r) + "\n" for r in rows))
    coh = [
        {"context_id": cid, "draw": d, "coherent": not (cid == "persona__q1" and d == 3)}
        for cid in ("persona__q1", "bare__q1")
        for d in range(4)
    ]
    coh_path = tmp_path / "anchor_draws.jsonl"
    coh_path.write_text("".join(json.dumps(r) + "\n" for r in coh))
    return coh_path


def test_floor_ceiling_expected_means_and_incoherent_exclusion(tmp_path, rev_pairs):
    coh_path = _write_scores_fixture(tmp_path)
    scores = REV.load_anchor_scores(tmp_path, tmp_path / "out")
    coherent = REV.load_coherent_draws(coh_path)
    fc = REV.compute_rev_floor_ceiling(scores, coherent, rev_pairs[:1])
    (pair,) = fc["pairs"]
    assert pair["context_a"] == "persona__q1" and pair["context_b"] == "bare__q1"
    # floor: coherent draws 0..2 of persona__q1 -> ((10-90)+(20-70)+(30-80))/3/100
    assert pair["floor"]["delta_mean"] == pytest.approx(-0.6)
    assert pair["floor"]["draws_coherent"] == [0, 1, 2]  # draw 3 EXCLUDED (incoherent)
    assert pair["floor"]["n_draws_total"] == 4
    # ceiling: all 4 draws of bare__q1 -> ((90-0)+(80-10)+(100-20)+(70-30))/4/100
    assert pair["ceiling"]["delta_mean"] == pytest.approx(0.7)
    assert pair["ceiling"]["draws_coherent"] == [0, 1, 2, 3]
    assert pair["denominator"] == pytest.approx(1.3)


def test_missing_score_raises_never_defaults(tmp_path, rev_pairs):
    coh_path = _write_scores_fixture(tmp_path, drop=("fp-persona", "bare__q1", 2))
    scores = REV.load_anchor_scores(tmp_path, tmp_path / "out")
    coherent = REV.load_coherent_draws(coh_path)
    with pytest.raises(AssertionError, match="missing anchor judge score"):
        REV.compute_rev_floor_ceiling(scores, coherent, rev_pairs[:1])


def test_score_loader_rejects_wrong_rubric_and_out_of_range(tmp_path):
    row = {"rubric_id": "fp-conv", "kind": "anchor", "context_id": "bare__q1", "draw": 0}
    (tmp_path / REV.SCORES_FILES["fp-bare"]).write_text(json.dumps({**row, "score": 50.0}) + "\n")
    (tmp_path / REV.SCORES_FILES["fp-persona"]).write_text(
        json.dumps({**row, "rubric_id": "fp-persona", "score": 50.0}) + "\n"
    )
    with pytest.raises(AssertionError):
        REV.load_anchor_scores(tmp_path, tmp_path / "out")  # wrong rubric_id in fp-bare file
    (tmp_path / REV.SCORES_FILES["fp-bare"]).write_text(
        json.dumps({**row, "rubric_id": "fp-bare", "score": 150.0}) + "\n"
    )
    with pytest.raises(AssertionError):
        REV.load_anchor_scores(tmp_path, tmp_path / "out")  # out-of-range score


def test_zero_coherent_draws_fails_loud(tmp_path, rev_pairs):
    _write_scores_fixture(tmp_path)
    coh = [
        {"context_id": cid, "draw": d, "coherent": cid != "persona__q1"}
        for cid in ("persona__q1", "bare__q1")
        for d in range(4)
    ]
    coh_path = tmp_path / "all_incoherent.jsonl"
    coh_path.write_text("".join(json.dumps(r) + "\n" for r in coh))
    scores = REV.load_anchor_scores(tmp_path, tmp_path / "out")
    with pytest.raises(AssertionError, match="zero coherent"):
        REV.compute_rev_floor_ceiling(scores, REV.load_coherent_draws(coh_path), rev_pairs[:1])


# ── null-arm donor-STATE payload on a synthetic bank ──────────────────


def _synthetic_bank(hidden: int = 6) -> dict:
    """Random per-context bank (the parent test fixture shape): every context
    its own q_span, same-prefix contexts share one v_pe."""
    contexts = BANK.build_contexts()
    gen = torch.Generator().manual_seed(0)
    v_pe_by_prefix = {
        prefix: torch.randn(N_LAYERS, hidden, generator=gen) for prefix in BANK.PREFIX_ORDER
    }
    per_context = {}
    for cid, ctx in contexts.items():
        # build_contexts now ALSO carries the butler contexts (#2094 Option C).
        # Butler's v_pe is drawn LAZILY here — butler contexts sort last, so
        # the parent fixture's RNG stream stays byte-identical to the
        # pre-butler fixture (its sanity-margin asserts are seed-calibrated).
        if ctx["prefix"] not in v_pe_by_prefix:
            v_pe_by_prefix[ctx["prefix"]] = torch.randn(N_LAYERS, hidden, generator=gen)
        per_context[cid] = {
            "context_id": cid,
            "prefix": ctx["prefix"],
            "query_id": ctx["query_id"],
            "ctx_len": 24,
            "prefix_end": 18,
            "nq": 6,
            "q_span": torch.randn(6, N_LAYERS, hidden, generator=gen),
            "v_pe": v_pe_by_prefix[ctx["prefix"]].clone(),
        }
    centroids = {}
    for prefix in BANK.PREFIX_ORDER:
        v = {
            cid: rec["q_span"][-1]
            for cid, rec in per_context.items()
            if rec["prefix"] in (prefix, "bare")
        }
        centroids[prefix] = BANK.prefix_centroid(v, prefix)
    return {"layers": list(range(N_LAYERS)), "per_context": per_context, "centroids": centroids}


def test_null_payload_is_donor_state_norm_matched(rev_pairs, parent_pairs):
    """Every reversed null cell resolves: the payload is the DONOR pair's
    target-context STATE norm-matched to the recipient's V_B — parallel to
    V_ce(donor.b), never the donor's Delta, never the recipient's own state."""
    bank = _synthetic_bank()
    by_id = {p.pair_id: p for p in parent_pairs}
    donors = REV.rev_donor_assignment(rev_pairs, parent_pairs)
    for pair in rev_pairs:
        donor = by_id[donors[pair.pair_id]]
        _delta, state, m = R._pair_payload(bank, pair, "ce", "A")
        assert m == 1
        # Steered payload: the recipient's own V_B = V_ce(bare__q<i>).
        expect_state = bank["per_context"][pair.b]["q_span"][-1:]
        assert torch.equal(state, expect_state)
        payload, label = R._donor_payload(bank, pair, donor, "ce", "A", state, "state")
        assert label == donor.pair_id  # recorded donor_pair_id
        # Norm-matched position-wise to the recipient's V_B...
        assert torch.allclose(payload.norm(dim=-1), state.norm(dim=-1), rtol=1e-4, atol=1e-6)
        # ...parallel to the donor's ce STATE (V_B(donor)), not its Delta,
        # and never bit-identical to the steered twin's own state.
        donor_delta, donor_state, _ = R._pair_payload(bank, donor, "ce", "A")
        cos_state = torch.nn.functional.cosine_similarity(payload[0], donor_state[0], dim=-1)
        cos_delta = torch.nn.functional.cosine_similarity(payload[0], donor_delta[0], dim=-1)
        assert float(cos_state.min()) > 1 - 1e-5
        assert float(cos_delta.max()) < 0.99
        assert not torch.equal(payload, state)
