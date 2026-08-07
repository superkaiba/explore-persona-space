"""CPU-only regression tests for the issue #2094 fu2_span_slots driver.

Pins (scope-marker duties):

- ``bank.template_token_mask`` + ``fu2_slot_positions`` on REAL Qwen-2.5-7B
  tokenizer output for BOTH prefix kinds (bare single-turn + conv multi-turn),
  including the union assert qtext | excluded-template == qspan and the
  boundary special-token ids at every excluded position class;
- pair eligibility per slot (pspan_* exclude matched-prefix; qtext runs all
  three settings) with counts asserted;
- the fu2 grid arithmetic (30 families / 60 blocks / 2400 cells);
- the ExtraSlot seam: registration, parent-slot behavior UNCHANGED, donor
  eligibility (prefix-scoped for pspan slots), replace -> add_full_state_patch;
- payload alignment (right-aligned min-overlap WITHIN content coordinates) and
  the donor null shaped by the SAME mask machinery, on a synthetic bank;
- the regime fingerprint + hard cross-regime refusal of the bank resume.

No model, no GPU, no network (the Qwen tokenizer loads from the local HF
cache, the same dependency test_issue1415_steering.py already carries).
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import ClassVar

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2094_fu2 as F2  # noqa: E402
import issue2094_run as R  # noqa: E402

from explore_persona_space.experiments.issue2094 import bank as BANK  # noqa: E402


@pytest.fixture(scope="module")
def qwen_tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(R.MODEL_ID)


@pytest.fixture(scope="module")
def contexts():
    return BANK.build_contexts()


@pytest.fixture(scope="module")
def pairs():
    return BANK.build_pairs()


# ── template mask: real tokenizer, both prefix kinds ────────────────────


@pytest.mark.parametrize("prefix", ["bare", "persona", "conv"])
def test_mask_and_positions_on_real_tokenizer(qwen_tok, contexts, prefix):
    ctx = contexts[BANK.context_id(prefix, "q1")]
    ids = BANK.context_token_ids_2094(qwen_tok, ctx)
    pe = BANK.prefix_end_index_multi(qwen_tok, ids)
    ctx_len = len(ids)
    mask = BANK.template_token_mask(qwen_tok, ids)
    assert len(mask) == ctx_len

    pos = F2.fu2_slot_positions(qwen_tok, ids, pe)
    qtext = pos["qtext"]
    # The contiguous interior of the final user turn.
    assert qtext == tuple(range(pe + 3, ctx_len - 5))
    assert len(qtext) >= 1
    # Union check: qtext + excluded template == the shipped qspan.
    qspan = set(R.slot_positions(ctx_len, pe, "qspan"))
    span_template = {i for i in range(pe, ctx_len) if mask[i]}
    assert set(qtext) | span_template == qspan
    assert set(qtext).isdisjoint(span_template)

    # Prefix-span slots.
    assert pos["pspan_tmpl"] == tuple(range(pe))
    ptext = pos["pspan_text"]
    prefix_template = {i for i in range(pe) if mask[i]}
    assert set(ptext) | prefix_template == set(range(pe))
    assert set(ptext).isdisjoint(prefix_template)
    assert len(ptext) >= 1

    # The qtext content tokens decode to EXACTLY the query text.
    assert qwen_tok.decode([ids[i] for i in qtext]) == BANK.QUERIES["q1"]


def test_conv_pspan_text_carries_history_content(qwen_tok, contexts):
    ctx = contexts[BANK.context_id("conv", "q2")]
    ids = BANK.context_token_ids_2094(qwen_tok, ctx)
    pe = BANK.prefix_end_index_multi(qwen_tok, ids)
    pos = F2.fu2_slot_positions(qwen_tok, ids, pe)
    decoded = qwen_tok.decode([ids[i] for i in pos["pspan_text"]])
    # Content of BOTH history turns rides the prefix content span; the
    # template headers/closers do not.
    assert BANK.CONV_USER_TURN in decoded
    assert BANK.CONV_ASSISTANT_TURN in decoded
    assert "<|im_start|>" not in decoded
    assert "<|im_end|>" not in decoded


def test_persona_pspan_text_is_system_content(qwen_tok, contexts):
    ctx = contexts[BANK.context_id("persona", "q3")]
    ids = BANK.context_token_ids_2094(qwen_tok, ctx)
    pe = BANK.prefix_end_index_multi(qwen_tok, ids)
    pos = F2.fu2_slot_positions(qwen_tok, ids, pe)
    decoded = qwen_tok.decode([ids[i] for i in pos["pspan_text"]])
    assert BANK.PERSONA_SYSTEM in decoded
    assert "<|im_start|>" not in decoded


def test_matched_query_pair_has_equal_qtext_lengths(qwen_tok, contexts, pairs):
    mq = next(p for p in pairs if p.setting == "matched_query")
    lens = []
    for cid in (mq.a, mq.b):
        ids = BANK.context_token_ids_2094(qwen_tok, contexts[cid])
        pe = BANK.prefix_end_index_multi(qwen_tok, ids)
        lens.append(len(F2.fu2_slot_positions(qwen_tok, ids, pe)["qtext"]))
    # Same query tokens on both sides => full-length overlap for the delta.
    assert lens[0] == lens[1]


# ── template mask: structural stub (content newline; role assert) ───────


class _MaskStubTok:
    """Minimal id-level tokenizer stub for template_token_mask structure."""

    IM_START, IM_END, NL = 100, 101, 5
    ROLES: ClassVar[dict[str, int]] = {"system": 200, "user": 201, "assistant": 202}

    def convert_tokens_to_ids(self, t):
        return {BANK.IM_START_TOKEN: self.IM_START, "<|im_end|>": self.IM_END}[t]

    def __call__(self, text, add_special_tokens=False):
        assert text == "\n"
        return {"input_ids": [self.NL]}

    def decode(self, ids):
        rev = {v: k for k, v in self.ROLES.items()}
        return "".join(rev.get(i, f"<{i}>") for i in ids)


def _stub_ids():
    s = _MaskStubTok
    return [
        s.IM_START,
        s.ROLES["system"],
        s.NL,
        7,
        8,
        s.IM_END,
        s.NL,  # system turn
        s.IM_START,
        s.ROLES["user"],
        s.NL,
        9,
        s.NL,
        10,
        s.IM_END,
        s.NL,  # user turn
        s.IM_START,
        s.ROLES["assistant"],
        s.NL,  # generation header
    ]


def test_stub_mask_content_newline_not_masked():
    ids = _stub_ids()
    mask = BANK.template_token_mask(_MaskStubTok(), ids)
    # Content positions: 3, 4 (system), 9, 11, 12 (user; 11 is a CONTENT newline).
    content = [i for i, m in enumerate(mask) if not m]
    assert content == [3, 4, 10, 11, 12]
    # Every im_start/im_end/role/header-newline masked.
    for i in (0, 1, 2, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17):
        assert mask[i], i


def test_stub_mask_unknown_role_fails_loud():
    ids = _stub_ids()
    ids[1] = 999  # not a role token
    with pytest.raises(AssertionError, match="unknown role"):
        BANK.template_token_mask(_MaskStubTok(), ids)


def test_stub_mask_requires_three_turns():
    s = _MaskStubTok
    ids = [s.IM_START, s.ROLES["user"], s.NL, 9, s.IM_END, s.NL]
    with pytest.raises(AssertionError, match=">=3 turns"):
        BANK.template_token_mask(_MaskStubTok(), ids)


# ── pair eligibility + grid arithmetic ──────────────────────────────────


def test_pair_eligibility_counts(pairs):
    assert len(F2.fu2_pair_ids(pairs, "qtext")) == 60
    for slot in ("pspan_tmpl", "pspan_text"):
        ids = F2.fu2_pair_ids(pairs, slot)
        assert len(ids) == 30
        by_id = {p.pair_id: p for p in pairs}
        settings = {by_id[i].setting for i in ids}
        assert settings == {"matched_query", "cross"}
        # No matched-prefix (degenerate_self) pair enters a prefix-span slot.
        assert all(by_id[i].prefix_a != by_id[i].prefix_b for i in ids)


def test_grid_totals_pinned(pairs):
    families = F2.enumerate_fu2_families(pairs)
    assert R.grid_totals(families) == F2.EXPECTED_FU2_TOTALS
    # Type-A only, the two joint variants only, both arms per family.
    for steered, null in families:
        assert steered.vec_type == "A" and null.vec_type == "A"
        assert steered.layer_variant in F2.FU2_VARIANTS
        assert (steered.arm, null.arm) == ("steered", "null")
        assert steered.pair_ids == null.pair_ids


def test_smoke_slice_covers_every_slot_variant_and_mode(pairs):
    families = F2.slice_fu2_smoke(F2.enumerate_fu2_families(pairs), pairs)
    assert len(families) == len(F2.SMOKE_FAMILIES)
    slots = {s.slot for s, _ in families}
    assert slots == set(F2.FU2_SLOTS)
    for slot in F2.FU2_SLOTS:
        doses = {s.dose for s, _ in families if s.slot == slot}
        assert "replace" in doses and any(d.startswith("a") for d in doses), (slot, doses)
        variants = {s.layer_variant for s, _ in families if s.slot == slot}
        assert variants == set(F2.FU2_VARIANTS), (slot, variants)
    by_id = {p.pair_id: p for p in pairs}
    for steered, null in families:
        assert steered.pair_ids == null.pair_ids
        assert set(steered.pair_ids) <= set(F2.fu2_pair_ids(pairs, steered.slot))
        # The multi-turn render seam stays smoke-visible.
        assert any(
            by_id[i].a.startswith("conv") or by_id[i].b.startswith("conv") for i in steered.pair_ids
        )


# ── ExtraSlot seam ───────────────────────────────────────────────────────


def test_extra_slots_registered_and_idempotent():
    for slot in F2.FU2_SLOTS:
        assert slot in R.EXTRA_SLOTS
        spec = R.EXTRA_SLOTS[slot]
        assert spec.positions_key == f"{slot}_positions"
        assert spec.vectors_key == f"{slot}_vectors"
        assert spec.prefix_scoped == (slot in F2.PSPAN_SLOTS)
        R.register_extra_slot(spec)  # idempotent re-registration
    with pytest.raises(AssertionError, match="shadows a parent slot"):
        R.register_extra_slot(R.ExtraSlot("qspan", "x", "y", False))
    with pytest.raises(AssertionError):
        R.register_extra_slot(R.ExtraSlot("qtext", "other_key", "y", False))


def test_parent_slot_behavior_unchanged():
    rec = {"ctx_len": 40, "prefix_end": 20}
    for slot in R.SLOTS:
        assert R.slot_positions_for_record(rec, slot) == R.slot_positions(40, 20, slot)
    # Parent multi-position replace realization untouched; fu2 slots join it.
    assert R._realized_mode("qspan", "replace", "A") == ("add", 1.0, "delta")
    assert R._realized_mode("ce", "replace", "A") == ("replace", 1.0, "state")
    for slot in F2.FU2_SLOTS:
        assert R._realized_mode(slot, "replace", "A") == ("add", 1.0, "delta")
        assert R._realized_mode(slot, "a2", "A") == ("add", 2.0, "delta")


def test_donor_eligibility_prefix_scoped(pairs):
    mp = next(p for p in pairs if p.setting == "matched_prefix")
    mq = next(p for p in pairs if p.setting == "matched_query")
    mq2 = next(
        p for p in pairs if p.setting == "matched_query" and p.prefix_pair() != mq.prefix_pair()
    )
    for slot in F2.PSPAN_SLOTS:
        # Delta kind: a same-prefix donor's prefix Delta is exactly zero.
        assert not R._donor_eligible(mp, slot, mq, "delta")
        assert R._donor_eligible(mq2, slot, mq, "delta")
        # State kind: donor must differ in prefix_b (prefix states are
        # query-independent), the pe rule.
        same_b = next(
            (
                p
                for p in pairs
                if p.setting == "matched_query"
                and p.pair_id != mq.pair_id
                and p.prefix_b == mq.prefix_b
            ),
            None,
        )
        assert same_b is not None
        assert not R._donor_eligible(same_b, slot, mq, "state")
    # qtext stays context-scoped: state kind compares the target context b.
    same_ctx_b = next((p for p in pairs if p.b == mq.b and p.pair_id != mq.pair_id), None)
    assert same_ctx_b is not None
    assert not R._donor_eligible(same_ctx_b, "qtext", mq, "state")
    assert R._donor_eligible(mp, "qtext", mq, "delta")


# ── payload alignment + donor null on a synthetic bank ──────────────────

_L, _H = 2, 4


def _synth_bank(pairs) -> dict:
    """Deterministic synthetic fu2 bank: per-context content counts vary with
    the query index so alignment truncation is exercised."""
    g = torch.Generator().manual_seed(2094)
    recs: dict[str, dict] = {}
    for cid in BANK.build_contexts():
        prefix, q = cid.split("__")
        nq_text = 3 + int(q[1:])  # 4..8 content tokens
        n_prefix = {"bare": 6, "persona": 8, "conv": 12}[prefix]
        pe = n_prefix
        ctx_len = pe + 3 + nq_text + 5
        qtext_pos = list(range(pe + 3, ctx_len - 5))
        ptmpl_pos = list(range(pe))
        ptext_pos = [i for i in range(pe) if i % 3 != 0]  # fake template drop
        rec = {
            "context_id": cid,
            "prefix": prefix,
            "query_id": q,
            "ctx_len": ctx_len,
            "prefix_end": pe,
            "nq": ctx_len - pe,
            "q_span": torch.randn(ctx_len - pe, _L, _H, generator=g),
            "v_pe": torch.randn(_L, _H, generator=g),
        }
        for slot, pos in (
            ("qtext", qtext_pos),
            ("pspan_tmpl", ptmpl_pos),
            ("pspan_text", ptext_pos),
        ):
            rec[f"{slot}_positions"] = list(pos)
            rec[f"{slot}_vectors"] = torch.randn(len(pos), _L, _H, generator=g)
        recs[cid] = rec
    return {"layers": list(range(_L)), "per_context": recs}


def test_pair_payload_right_aligns_within_content(pairs):
    bank = _synth_bank(pairs)
    recs = bank["per_context"]
    # matched-prefix pair: different queries => different qtext lengths.
    mp = next(p for p in pairs if p.setting == "matched_prefix")
    va = recs[mp.a]["qtext_vectors"]
    vb = recs[mp.b]["qtext_vectors"]
    m = min(va.shape[0], vb.shape[0])
    assert va.shape[0] != vb.shape[0]
    delta, state, m_out = R._pair_payload(bank, mp, "qtext", "A")
    assert m_out == m
    assert torch.equal(delta, vb[-m:] - va[-m:])
    assert torch.equal(state, vb[-m:])
    # And the edit positions are the recipient's LAST m content positions.
    pos = R.slot_positions_for_record(recs[mp.a], "qtext")[-m:]
    assert list(pos) == recs[mp.a]["qtext_positions"][-m:]
    assert delta.shape[0] == len(pos)


def test_pspan_payload_uses_prefix_coordinates(pairs):
    bank = _synth_bank(pairs)
    recs = bank["per_context"]
    mq = next(p for p in pairs if p.setting == "matched_query" and p.prefix_a == "bare")
    delta, _state, m = R._pair_payload(bank, mq, "pspan_tmpl", "A")
    va = recs[mq.a]["pspan_tmpl_vectors"]
    vb = recs[mq.b]["pspan_tmpl_vectors"]
    assert m == min(va.shape[0], vb.shape[0])
    assert torch.equal(delta, vb[-m:] - va[-m:])
    pos = R.slot_positions_for_record(recs[mq.a], "pspan_tmpl")[-m:]
    assert max(pos) < recs[mq.a]["prefix_end"]


def test_donor_null_shaped_by_same_machinery(pairs):
    bank = _synth_bank(pairs)
    donor_map = BANK.donor_derangement(pairs)
    pairs_by_id = {p.pair_id: p for p in pairs}
    mq = next(p for p in pairs if p.setting == "matched_query")
    for slot in F2.FU2_SLOTS:
        if slot in F2.PSPAN_SLOTS and mq.setting == "matched_prefix":
            continue
        recipient, _state, _m = R._pair_payload(bank, mq, slot, "A")
        payload, donor_id = R._resolve_donor(
            bank, mq, donor_map, pairs_by_id, slot, "A", recipient, "delta"
        )
        assert payload.shape == recipient.shape
        assert donor_id != mq.pair_id
        donor = pairs_by_id[donor_id]
        # Recompute: the donor pair's OWN slot delta (same mask machinery),
        # right-aligned to the recipient's m, norm-matched position-wise.
        raw = R._aligned_donor_raw(bank, donor, slot, "A", recipient, "delta")
        expected = BANK.norm_match(raw, recipient)
        assert torch.allclose(payload, expected)
        # Position-wise norm match realized.
        assert torch.allclose(payload.norm(dim=-1), recipient.norm(dim=-1), rtol=1e-4, atol=1e-5)


# ── regime + resume ──────────────────────────────────────────────────────


def _cfg(tmp_path: Path, **kw) -> R.RunConfig:
    base = dict(
        phase="fu2_span_slots",
        out_root=tmp_path,
        log_dir=tmp_path / "logs",
        model_id=R.MODEL_ID,
        tiny=True,
        n_layers=28,
        hidden=64,
        device="cpu",
        gen_batch=4,
        capture_batch=4,
        max_new_tokens=F2.FU2_MAX_NEW_TOKENS,
        anchor_draws=1,
        seed_base=R.SEED_BASE,
        smoke=True,
        pilot=False,
        force=False,
        worker_index=0,
        num_workers=1,
        upload_mode="none",
        upload_every=0,
        planned_wall_h=2.5,
        gpu_hours_budgeted=8.0,
    )
    base.update(kw)
    return R.RunConfig(**base)


def test_fu2_regime_fingerprint_keys(tmp_path):
    cfg = _cfg(tmp_path)
    _, bank_sha = R.bank_manifest_and_sha()
    fp = F2.fu2_regime_fingerprint(cfg, bank_sha)
    assert fp != R.regime_fingerprint(cfg, bank_sha)  # fu2 token folded in
    assert F2.fu2_regime_fingerprint(dc_replace(cfg, max_new_tokens=1024), bank_sha) != fp
    assert F2.fu2_regime_fingerprint(dc_replace(cfg, smoke=False), bank_sha) != fp
    assert F2.fu2_regime_fingerprint(cfg, bank_sha) == fp  # deterministic


def test_bank_resume_hard_refuses_cross_regime(tmp_path):
    paths = F2.FU2Paths(out_root=tmp_path)
    paths.fu2_bank_dir.mkdir(parents=True)
    assert not F2.fu2_bank_is_done(paths, "abc")
    paths.fu2_bank_path.write_bytes(b"x")
    paths.fu2_bank_done.write_text(json.dumps({"regime_fp": "abc"}))
    assert F2.fu2_bank_is_done(paths, "abc")
    with pytest.raises(RuntimeError, match="refusing to resume across regimes"):
        F2.fu2_bank_is_done(paths, "OTHER")


def test_block_resume_hard_refuses_cross_regime(tmp_path, pairs):
    families = F2.slice_fu2_smoke(F2.enumerate_fu2_families(pairs), pairs)
    block = families[0][0]
    done = R.block_done_path(tmp_path, block)
    done.parent.mkdir(parents=True)
    done.write_text(json.dumps({"key": block.key, "regime_fp": "abc"}))
    assert R.block_is_done(tmp_path, block, "abc")
    with pytest.raises(RuntimeError, match="refusing to resume across regimes"):
        R.block_is_done(tmp_path, block, "OTHER")


# ── parity report ────────────────────────────────────────────────────────


def _mini_parity_banks():
    g = torch.Generator().manual_seed(7)
    recs_a, recs_b = {}, {}
    for cid in ("bare__q1", "conv__q1"):
        q_span = torch.randn(4, 6, _H, generator=g)
        v_pe = torch.randn(6, _H, generator=g)
        recs_a[cid] = {"q_span": q_span, "v_pe": v_pe}
        recs_b[cid] = {"q_span": q_span.clone(), "v_pe": v_pe.clone()}
    return {"per_context": recs_a}, {"per_context": recs_b}


def test_parity_report_passes_on_identical_and_fails_on_layer0_corruption():
    a, b = _mini_parity_banks()
    rep = F2.fu2_parity_report(a, b)
    assert rep["passed"] and rep["early_min_cos"] == pytest.approx(1.0)
    # Corrupt layer 0 of one context's span (a real offset/mask bug signature).
    b["per_context"]["bare__q1"]["q_span"][:, 0, :] = torch.tensor([1.0, -1.0, 1.0, -1.0])
    rep2 = F2.fu2_parity_report(a, b)
    assert not rep2["passed"]
    assert rep2["early_min_cos"] < F2.PARITY_EARLY_COS_MIN


# ── driver surface ───────────────────────────────────────────────────────


def test_parse_args_defaults_to_2048_and_hf_upload():
    args = F2.parse_args(["--run"])
    assert args.max_new_tokens == 2048
    assert args.upload == "hf"
    assert args.upload_every == 10


def test_caphit_report_pools_arms(tmp_path, pairs):
    cfg = _cfg(tmp_path)
    blocks_dir = cfg.manifest_dir / "blocks"
    blocks_dir.mkdir(parents=True)
    for key, n, hit in (
        ("qtext|joint_all|a1|A|steered", 4, 1),
        ("qtext|joint_all|a1|A|null", 4, 0),
        ("pspan_text|joint_mid|replace|A|steered", 2, 2),
    ):
        (blocks_dir / f"{R.block_slug(key)}.done.json").write_text(
            json.dumps({"key": key, "n_cells": n, "n_cap_hit": hit, "regime_fp": "x"})
        )
    rep = F2.fu2_caphit_report(cfg)
    assert rep["max_new_tokens"] == F2.FU2_MAX_NEW_TOKENS
    by_key = {(c["slot"], c["layer_variant"], c["dose"]): c for c in rep["cells"]}
    qt = by_key[("qtext", "joint_all", "a1")]
    assert qt["steered"] == {"n": 4, "cap_hit": 1, "cap_hit_frac": 0.25}
    assert qt["null"]["cap_hit_frac"] == 0.0
    ps = by_key[("pspan_text", "joint_mid", "replace")]
    assert ps["steered"]["cap_hit_frac"] == 1.0
