"""CPU-only unit tests for the issue #2094 pod driver's pure helpers.

No model, no GPU, no network: block enumeration / regime keys / resume predicate /
shard assignment / slot geometry / the sentinel payload shape. The full-pipeline
smoke lives with the dispatcher unit.
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

import issue2094_run as R  # noqa: E402

from explore_persona_space.experiments.issue2094 import bank as BANK  # noqa: E402

N_LAYERS = 28


@pytest.fixture(scope="module")
def pairs() -> list[BANK.Pair]:
    return BANK.build_pairs()


# ── block enumeration: the plan §4.3 reconciliation ───────────────────


def test_block_enumeration_matches_plan_cell_counts(pairs):
    fam = R.enumerate_block_families(pairs, N_LAYERS)
    totals = R.grid_totals(fam)
    assert totals == {
        "n_families": 440,
        "n_blocks": 880,
        "cells_steered": 21_000,
        "cells_null": 21_000,
        "cells_total": 42_000,
    }


def test_block_family_composition_by_source(pairs):
    """The 440 families decompose exactly as plan §4.3's three blocks."""
    fam = R.enumerate_block_families(pairs, N_LAYERS)
    full_sweep = [f for f in fam if f[0].vec_type == "A" and f[0].slot in R.SLOTS_FULL_SWEEP]
    controls = [f for f in fam if f[0].vec_type == "A" and f[0].slot in R.SLOTS_CONTROL]
    type_b = [f for f in fam if f[0].vec_type == "B"]
    assert len(full_sweep) == 2 * 30 * 5
    assert len(controls) == 4 * 1 * 5
    assert len(type_b) == 1 * 30 * 4
    assert sum(f[0].n_cells for f in full_sweep) == 18_000
    assert sum(f[0].n_cells for f in controls) == 1_200
    assert sum(f[0].n_cells for f in type_b) == 1_800
    # Type B is context-end only, over matched-query pairs, never with `replace`.
    assert {f[0].slot for f in type_b} == {"ce"}
    assert "replace" not in {f[0].dose for f in type_b}


def test_block_keys_and_slugs_are_unique_and_path_safe(pairs):
    fam = R.enumerate_block_families(pairs, N_LAYERS)
    blocks = [b for f in fam for b in f]
    assert len({b.key for b in blocks}) == len(blocks)
    slugs = [b.slug for b in blocks]
    assert len(set(slugs)) == len(blocks)
    assert all("|" not in s and "." not in s and "/" not in s for s in slugs)


# ── layer variants + doses ────────────────────────────────────────────


def test_layer_variants_and_joint_bands():
    names = R.layer_variant_names(N_LAYERS)
    assert len(names) == 30
    assert names[:2] == ("L0", "L1") and names[-2:] == ("joint_mid", "joint_all")
    assert R.joint_mid_layers(N_LAYERS) == tuple(range(14, 21))
    assert R.layer_variant_layers("joint_all", N_LAYERS) == tuple(range(28))
    assert R.layer_variant_layers("L14", N_LAYERS) == (14,)
    # Tiny smoke model: the production 14..20 band does not exist -> single mid layer.
    assert R.joint_mid_layers(4) == (2,)
    with pytest.raises(AssertionError):
        R.layer_variant_layers("L99", N_LAYERS)


def test_dose_and_realized_mode():
    assert R.dose_spec("a0.5") == ("add", 0.5)
    assert R.dose_spec("replace") == ("replace", 1.0)
    # single-position slot: a TRUE replace of the slot state
    assert R.dose_spec("a4") == ("add", 4.0)
    assert R._realized_mode("ce", "replace", "A") == ("replace", 1.0, "state")
    assert R._realized_mode("ce", "a2", "A") == ("add", 2.0, "delta")
    # multi-position replace degrades to the equivalent per-position add-patch
    # (PositionEditHook restricts mode='replace' to ONE position per row).
    assert R._realized_mode("l3j", "replace", "A") == ("add", 1.0, "delta")
    assert R._realized_mode("qspan", "replace", "A") == ("add", 1.0, "delta")
    # Type B has no absolute state -> a replace request fails loud.
    with pytest.raises(AssertionError):
        R._realized_mode("ce", "replace", "B")


# ── slot geometry ─────────────────────────────────────────────────────


def test_slot_positions_right_aligned_to_context_end():
    ctx_len, prefix_end = 50, 30
    assert R.slot_positions(ctx_len, prefix_end, "ce") == (49,)
    assert R.slot_positions(ctx_len, prefix_end, "cm2") == (48,)
    assert R.slot_positions(ctx_len, prefix_end, "cm3") == (47,)
    assert R.slot_positions(ctx_len, prefix_end, "l3j") == (47, 48, 49)
    assert R.slot_positions(ctx_len, prefix_end, "pe") == (29,)
    qspan = R.slot_positions(ctx_len, prefix_end, "qspan")
    assert qspan == tuple(range(30, 50)) and qspan[-1] == 49
    # A final-user-turn span shorter than 3 tokens cannot carry cm3 / l3j.
    with pytest.raises(AssertionError):
        R.slot_positions(32, 30, "ce")


def test_align_right_truncates_and_cycles():
    v = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    assert torch.equal(R.align_right(v, 2), v[-2:])
    assert torch.equal(R.align_right(v, 4), v)
    short = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    tiled = R.align_right(short, 5)
    assert tiled.shape == (5, 3)
    # right-aligned cyclic tile: the LAST row is the donor's last row
    assert torch.equal(tiled[-1], short[-1])
    assert torch.equal(tiled[-2], short[-2])


# ── shard assignment ──────────────────────────────────────────────────


def test_shard_assignment_partitions_and_keeps_arms_together(pairs):
    fam = R.enumerate_block_families(pairs, N_LAYERS)
    n_workers = 8
    seen: list[str] = []
    for w in range(n_workers):
        blocks = R.blocks_for_worker(fam, w, n_workers)
        assert len(blocks) % 2 == 0
        for i in range(0, len(blocks), 2):
            steered, null = blocks[i], blocks[i + 1]
            assert steered.arm == "steered" and null.arm == "null"
            # both arms of ONE family (same geometry) land adjacently
            assert (steered.slot, steered.layer_variant, steered.dose, steered.vec_type) == (
                null.slot,
                null.layer_variant,
                null.dose,
                null.vec_type,
            )
        seen += [b.key for b in blocks]
    assert len(seen) == 880 and len(set(seen)) == 880


def test_smoke_slice_covers_every_arm_class(pairs):
    """A smoke slice must reach every class-defining axis (#1090 fu5 / #1586)."""
    fam = R.smoke_block_families(pairs, N_LAYERS)
    blocks = [b for f in fam for b in f]
    assert {b.arm for b in blocks} == {"steered", "null"}
    assert {b.vec_type for b in blocks} == {"A", "B"}
    assert "replace" in {b.dose for b in blocks}
    # pe x replace (round-3): the state-kind donor walk + mp degenerate
    # carve-out arm class — the class whose null cells crashed pre-round-3.
    assert any(b.slot == "pe" and b.dose == "replace" for b in blocks)
    pe_repl_ids = {
        pid for b in blocks if b.slot == "pe" and b.dose == "replace" for pid in b.pair_ids
    }
    assert any(pid.startswith("mp--") for pid in pe_repl_ids)  # degenerate carve-out reached
    assert any(not pid.startswith("mp--") for pid in pe_repl_ids)  # real state walk reached
    assert {"joint_mid", "joint_all"} <= {b.layer_variant for b in blocks}
    assert any(b.layer_variant.startswith("L") for b in blocks)  # single-layer variant
    assert R.MULTI_POSITION_SLOTS & {b.slot for b in blocks}  # multi-position slot
    assert {"ce", "pe"} <= {b.slot for b in blocks}  # both full-sweep slots
    assert R.SLOTS_CONTROL[0] in {b.slot for b in blocks} or any(
        b.slot in R.SLOTS_CONTROL for b in blocks
    )
    # conv-context_a arm class: the multi-turn history render seam is otherwise
    # smoke-invisible (unit-E requirement; steering.context_messages drops history).
    assert any(pid.split("--")[1].startswith("conv") for b in blocks for pid in b.pair_ids)
    # >= 2 additive doses on one single-layer full-sweep family so the P7
    # linearity fit + homogeneity reads are runnable on the smoke outputs.
    ce_single_doses = {
        b.dose
        for b in blocks
        if b.slot == "ce" and b.layer_variant.startswith("L") and b.dose != "replace"
    }
    assert len(ce_single_doses) >= 2, ce_single_doses
    # Cheap by construction: a couple of pairs per block, not the full bank.
    assert R.grid_totals(fam)["cells_total"] < 100


# ── resume predicate ──────────────────────────────────────────────────


def _cfg(tmp_path: Path, **over) -> R.RunConfig:
    argv = ["--phase", "grid", "--out-root", str(tmp_path), "--log-dir", str(tmp_path / "logs")]
    for k, v in over.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return R.build_config(R.parse_args(argv))


def test_resume_predicate_skips_done_and_refuses_regime_mismatch(tmp_path, pairs):
    cfg = _cfg(tmp_path)
    block = R.enumerate_block_families(pairs, N_LAYERS)[0][0]
    assert R.block_is_done(cfg.out_root, block, "fp-a") is False
    R._write_json_atomic(
        R.block_done_path(cfg.out_root, block),
        {"key": block.key, "regime_fp": "fp-a", "n_cells": block.n_cells},
    )
    assert R.block_is_done(cfg.out_root, block, "fp-a") is True
    with pytest.raises(RuntimeError, match="regime_fp"):
        R.block_is_done(cfg.out_root, block, "fp-b")


def test_regime_fingerprint_keys_on_every_output_affecting_knob(tmp_path):
    base = _cfg(tmp_path)
    fp = R.regime_fingerprint(base, "banksha")
    assert fp == R.regime_fingerprint(_cfg(tmp_path), "banksha")
    assert fp != R.regime_fingerprint(base, "otherbank")
    assert fp != R.regime_fingerprint(_cfg(tmp_path, max_new_tokens=2048), "banksha")
    assert fp != R.regime_fingerprint(_cfg(tmp_path, seed_base=7), "banksha")
    smoke = R.build_config(
        R.parse_args(["--phase", "grid", "--out-root", str(tmp_path), "--smoke"])
    )
    assert fp != R.regime_fingerprint(smoke, "banksha")


def test_cap_hit_proxy():
    assert R.cap_hit(1024, 1024) is True
    assert R.cap_hit(1025, 1024) is True
    assert R.cap_hit(1023, 1024) is False


# ── end-of-turn tail + sentinel contract ──────────────────────────────


class _FakeTok:
    """Signature-shaped tokenizer stub for the id-only tail derivation."""

    def __init__(self, im_end: int = 151645, nl: list[int] | None = None):
        self._im_end = im_end
        self._nl = [198] if nl is None else nl

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<|im_end|>", token
        return self._im_end

    def __call__(self, text: str, add_special_tokens: bool = True):
        assert text == "\n" and add_special_tokens is False
        return {"input_ids": list(self._nl)}


def test_eot_tail_ids_built_from_ids_not_a_concatenated_string():
    assert R.eot_tail_ids(_FakeTok()) == [151645, 198]
    with pytest.raises(AssertionError):
        R.eot_tail_ids(_FakeTok(nl=[]))


def test_sentinel_payload_carries_every_step7_key(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.rollouts_dir.mkdir(parents=True, exist_ok=True)
    (cfg.rollouts_dir / "shard_ce__L14__a1__A__steered.jsonl").write_text(
        json.dumps({"pair_id": "p", "cap_hit": False}) + "\n"
    )
    R._write_json_atomic(
        cfg.manifest_dir / "blocks" / "ce__L14__a1__A__steered.done.json",
        {"key": "k", "regime_fp": "fp", "n_cells": 3, "n_cap_hit": 1},
    )
    payload = R._sentinel_payload(cfg, {"grid_text": ["a"]})
    required = (
        "eval_numbers",
        "eval_paths",
        "reproducibility_card",
        "wandb_url",
        "hf_hub_url",
        "worktree_path",
        "final_commit_sha",
        "gpu_hours_used",
        "gpu_hours_budgeted",
        "plan_deviations",
    )
    assert all(k in payload for k in required), [k for k in required if k not in payload]
    assert payload["eval_numbers"]["cells_persisted"] == 3
    assert payload["eval_numbers"]["cap_hit_rows"] == 1
    assert payload["plan_deviations"], "the realized-mode / cap-hit deviations must be recorded"


# ── slot-aware donor resolution (the pe zero-Delta production bug) ─────


def _synthetic_bank_same_prefix_pe(pairs: list, hidden: int = 6) -> dict:
    """Synthetic bank where SAME-prefix contexts share one v_pe exactly —
    the causal-attention identity that makes a matched-prefix donor's
    prefix-end Delta exactly zero (found by the unit-F e2e smoke). Carries
    real Type-B centroids so the full-grid walk-coverage test can resolve
    Type-B null cells too (round 3)."""
    contexts = BANK.build_contexts()
    gen = torch.Generator().manual_seed(0)
    v_pe_by_prefix = {
        prefix: torch.randn(N_LAYERS, hidden, generator=gen) for prefix in BANK.PREFIX_ORDER
    }
    per_context = {}
    for cid, ctx in contexts.items():
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


def test_pe_donor_eligibility_is_structural(pairs):
    mp = next(p for p in pairs if p.setting == "matched_prefix")
    mq = next(p for p in pairs if p.setting == "matched_query")
    assert not R._donor_eligible(mp, "pe")  # same prefix -> zero pe Delta
    assert R._donor_eligible(mp, "ce")
    assert R._donor_eligible(mq, "pe")


def test_same_prefix_pe_delta_canonicalized_to_exact_zero(pairs):
    """The unit-F e2e smoke incident: a same-prefix pair's pe Delta is float
    NOISE (1.9e-9 on the conv pair), while its mp donor's is exactly zero —
    norm_match then asserts. The fix canonicalizes the causal identity: a
    same-prefix pair's pe Delta is EXACTLY zero regardless of float noise."""
    bank = _synthetic_bank_same_prefix_pe(pairs)
    mp = next(p for p in pairs if p.setting == "matched_prefix")
    # inject the incident's float noise into ONE same-prefix v_pe
    bank["per_context"][mp.a]["v_pe"] += 2e-9 * torch.randn_like(bank["per_context"][mp.a]["v_pe"])
    delta, state, _m = R._pair_payload(bank, mp, "pe", "A")
    assert float(delta.abs().max()) == 0.0  # exact identity, not noise
    assert float(state.norm()) > 0  # the replacement state is untouched
    mq = next(p for p in pairs if p.setting == "matched_query")
    d_mq, _, _ = R._pair_payload(bank, mq, "pe", "A")
    assert float(d_mq.norm()) > 0  # cross-prefix pairs keep their real Delta


def test_resolve_donor_zero_recipient_and_eligibility_walk(pairs):
    """Fails pre-fix (both legs): (a) a zero-norm donor against a noise
    recipient raised inside norm_match; (b) the null of a degenerate
    (canonicalized-zero) recipient is a zero injection, matching its steered
    twin; (c) the walk skips a same-prefix donor handed to a pe cell."""
    bank = _synthetic_bank_same_prefix_pe(pairs)
    pairs_by_id = {p.pair_id: p for p in pairs}
    donor_map = BANK.donor_derangement(pairs)
    mp = next(p for p in pairs if p.setting == "matched_prefix")
    mq = next(p for p in pairs if p.setting == "matched_query")

    # (a) the raw pre-fix crash shape still fails loud at the norm_match level
    noise = 2e-9 * torch.randn(1, N_LAYERS, 6)
    with pytest.raises(AssertionError, match="zero-norm donor"):
        R._donor_payload(bank, mq, mp, "pe", "A", noise)

    # (b) degenerate recipient -> zero null payload, seeded donor id recorded
    recipient, _, _ = R._pair_payload(bank, mp, "pe", "A")
    payload, donor_id = R._resolve_donor(bank, mp, donor_map, pairs_by_id, "pe", "A", recipient)
    assert float(payload.abs().max()) == 0.0
    assert donor_id == donor_map[mp.pair_id]

    # (c) a synthetic donor map handing an mp donor to an mq pe cell: the walk
    # must skip it and land on an eligible (cross-prefix) donor, norm-matched.
    mq2 = next(p for p in pairs if p.setting == "matched_query" and p.pair_id != mq.pair_id)
    synth_map = {mq.pair_id: mp.pair_id, mp.pair_id: mq2.pair_id, mq2.pair_id: mq.pair_id}
    recip_mq, _, _ = R._pair_payload(bank, mq, "pe", "A")
    payload, donor_id = R._resolve_donor(bank, mq, synth_map, pairs_by_id, "pe", "A", recip_mq)
    assert donor_id == mq2.pair_id  # walked past the ineligible mp donor
    assert R._donor_eligible(pairs_by_id[donor_id], "pe")
    assert torch.allclose(payload.norm(dim=-1), recip_mq.norm(dim=-1), rtol=1e-4, atol=1e-6)


# ── round-2 Major 3: replace-dose null arm installs the donor STATE ─────


def test_replace_null_payload_uses_donor_state(pairs):
    """Intent pin (fails pre-fix): a single-position replace cell's null arm
    installs the DONOR pair's TARGET-CONTEXT STATE norm_match(V_B(donor), V_B)
    — a real state, wrong pair, parallel to the steered arm's real-state
    replace — never the donor's DIFFERENCE vector rescaled to state norm."""
    bank = _synthetic_bank_same_prefix_pe(pairs)
    pairs_by_id = {p.pair_id: p for p in pairs}
    donor_map = BANK.donor_derangement(pairs)
    mq = next(p for p in pairs if p.setting == "matched_query")

    mode, _alpha, payload_kind = R._realized_mode("ce", "replace", "A")
    assert (mode, payload_kind) == ("replace", "state")
    _delta, state, _m = R._pair_payload(bank, mq, "ce", "A")
    payload, donor_id = R._resolve_donor(
        bank, mq, donor_map, pairs_by_id, "ce", "A", state, payload_kind
    )
    donor = pairs_by_id[donor_id]
    donor_delta, donor_state, _ = R._pair_payload(bank, donor, "ce", "A")
    # norm-matched to the recipient's V_B norm (position-wise over H)...
    assert torch.allclose(payload.norm(dim=-1), state.norm(dim=-1), rtol=1e-4, atol=1e-6)
    # ...and PARALLEL (per layer) to the donor's STATE, not its Delta.
    cos_state = torch.nn.functional.cosine_similarity(payload[-1], donor_state[-1], dim=-1)
    cos_delta = torch.nn.functional.cosine_similarity(payload[-1], donor_delta[-1], dim=-1)
    assert float(cos_state.min()) > 1 - 1e-5
    assert float(cos_delta.max()) < 0.99  # the pre-fix (donor-Delta) shape


def test_donor_eligibility_state_kind_excludes_same_target_state(pairs):
    """State-kind eligibility (round-2 mirror-leg incident): a donor sharing
    the recipient's target slot state — same context b at ce/cm*; same
    prefix_b at pe (causal identity) — would install the recipient's OWN V_B,
    making the 'null' bit-identical to its steered twin. Delta kind keeps the
    original (slot-only) eligibility."""
    mq = next(p for p in pairs if p.setting == "matched_query")
    same_b = next(p for p in pairs if p.pair_id != mq.pair_id and p.b == mq.b)
    diff_b = next(p for p in pairs if p.b != mq.b and p.pair_id != mq.pair_id)
    # delta kind: a same-b donor stays eligible (its Delta is a real direction)
    assert R._donor_eligible(same_b, "ce", mq, "delta")
    # state kind: same-b donor EXCLUDED at ce; different-b donor eligible
    assert not R._donor_eligible(same_b, "ce", mq, "state")
    assert R._donor_eligible(diff_b, "ce", mq, "state")
    # pe state kind: same-PREFIX_b donor excluded even when b differs
    same_pb = next(
        p
        for p in pairs
        if p.pair_id != mq.pair_id
        and p.b != mq.b
        and p.prefix_b == mq.prefix_b
        and p.prefix_a != p.prefix_b
    )
    assert not R._donor_eligible(same_pb, "pe", mq, "state")
    assert R._donor_eligible(same_pb, "ce", mq, "state") == (same_pb.b != mq.b)


def test_replace_null_walk_skips_same_target_state_donor(pairs):
    """Walk pin (fails pre-fix): a seeded donor sharing the recipient's
    context b is walked PAST for state-kind cells, and the realized payload is
    never bit-identical to the steered twin's replacement state."""
    bank = _synthetic_bank_same_prefix_pe(pairs)
    pairs_by_id = {p.pair_id: p for p in pairs}
    mq = next(p for p in pairs if p.setting == "matched_query")
    same_b = next(p for p in pairs if p.pair_id != mq.pair_id and p.b == mq.b)
    diff_b = next(p for p in pairs if p.pair_id not in (mq.pair_id, same_b.pair_id) and p.b != mq.b)
    synth_map = {
        mq.pair_id: same_b.pair_id,
        same_b.pair_id: diff_b.pair_id,
        diff_b.pair_id: mq.pair_id,
    }
    _delta, state, _m = R._pair_payload(bank, mq, "ce", "A")
    payload, donor_id = R._resolve_donor(
        bank, mq, synth_map, pairs_by_id, "ce", "A", state, "state"
    )
    assert donor_id == diff_b.pair_id  # walked past the same-b donor
    assert not torch.equal(payload, state)  # never the steered twin's own V_B
    # additive kind on the same map STAYS on the seeded same-b donor
    delta_recip, _s, _ = R._pair_payload(bank, mq, "ce", "A")
    _p, donor_add = R._resolve_donor(
        bank, mq, synth_map, pairs_by_id, "ce", "A", delta_recip, "delta"
    )
    assert donor_add == same_b.pair_id


def test_type_b_donor_refuses_state_kind(pairs):
    """Type B has no absolute state (DOSES_B excludes replace) — a state-kind
    Type-B donor request fails loud rather than patching a difference in."""
    bank = _synthetic_bank_same_prefix_pe(pairs)
    mq = next(p for p in pairs if p.setting == "matched_query")
    with pytest.raises(AssertionError, match="Type B has no absolute state"):
        R._donor_payload(bank, mq, mq, "ce", "B", torch.randn(1, N_LAYERS, 6), "state")


# ── round-3 Critical 1: full-production-grid null donor-walk coverage ──


def _seeded_cycle_first_eligible(pair, donor_map, pairs_by_id, slot, payload_kind):
    """Independent oracle: the PURE seeded-cycle walk (no fallback), as the
    pre-round-3 code implemented it — used to pin that every cell the old walk
    resolved still resolves to the IDENTICAL donor."""
    seen: set[str] = set()
    donor_id = donor_map[pair.pair_id]
    while donor_id not in seen:
        seen.add(donor_id)
        donor = pairs_by_id[donor_id]
        if donor_id != pair.pair_id and R._donor_eligible(donor, slot, pair, payload_kind):
            return donor_id
        donor_id = donor_map[donor_id]
    return None


def test_null_donor_walk_covers_the_full_production_grid(pairs):
    """Round-3 Critical 1 regression (fails PRE-fix at EXACTLY 32 cells — all
    30 matched-prefix pairs plus the 2-cycle cross pair duo at pe x replace):
    EVERY (slot, dose-kind, setting) combination the production grid realizes
    resolves a donor for EVERY recipient. Donor resolution is layer-variant-
    independent, so combos dedup over (slot, dose, vec_type); the set is
    derived FROM enumerate_block_families so grid changes propagate. Also the
    mirror-diff scope pin: outside the 32 affected cells every realized donor
    equals the pure seeded-cycle walk's (the pre-fix behavior), the degenerate
    ``self:`` carve-out fires for exactly the 30 mp pairs at pe/replace, and
    the beyond-cycle fallback for exactly the 2 cross 2-cycle pairs."""
    bank = _synthetic_bank_same_prefix_pe(pairs)
    pairs_by_id = {p.pair_id: p for p in pairs}
    donor_map = BANK.donor_derangement(pairs)
    fams = R.enumerate_block_families(pairs, N_LAYERS)
    combos: dict[tuple[str, str, str], tuple[str, ...]] = {}
    for _steered, null in fams:
        key = (null.slot, null.dose, null.vec_type)
        # variant-invariance HARDENED (round-3 Minor): every layer variant of a
        # combo must carry the SAME pair set — a future variant-dependent
        # subset must widen this walk, never silently shrink it.
        assert combos.setdefault(key, null.pair_ids) == null.pair_ids, key
    assert len(combos) == 34, combos.keys()  # 6 slots x 5 A-doses + ce x 4 B-doses
    exhausted: list[tuple[str, str, str, str]] = []
    degenerate: list[tuple[str, str, str, str]] = []
    beyond_cycle: list[tuple[str, str, str, str]] = []
    for (slot, dose, vec_type), pair_ids in sorted(combos.items()):
        _mode, _alpha, payload_kind = R._realized_mode(slot, dose, vec_type)
        for pid in pair_ids:
            pair = pairs_by_id[pid]
            delta, state, _m = R._pair_payload(bank, pair, slot, vec_type)
            recipient = state if payload_kind == "state" else delta
            try:
                payload, label = R._resolve_donor(
                    bank, pair, donor_map, pairs_by_id, slot, vec_type, recipient, payload_kind
                )
            except AssertionError:
                exhausted.append((slot, dose, vec_type, pid))
                continue
            if vec_type == "B":
                assert label.startswith("centroid:"), label
                continue
            if label.startswith("self:"):
                degenerate.append((slot, dose, vec_type, pid))
                assert label == f"self:{pid}"
                assert torch.equal(payload, recipient)  # the pair's OWN V_B, bit-identical
                continue
            if float(recipient.norm()) == 0.0:
                # canonicalized zero-Delta cells: zero null, seeded donor recorded
                assert label == donor_map[pid]
                assert float(payload.abs().max()) == 0.0
                continue
            cycle_donor = _seeded_cycle_first_eligible(
                pair, donor_map, pairs_by_id, slot, payload_kind
            )
            if cycle_donor is None:
                beyond_cycle.append((slot, dose, vec_type, pid))
                assert R._donor_eligible(pairs_by_id[label], slot, pair, payload_kind)
            else:
                # Mirror-diff scope pin: every cell the pre-fix walk resolved
                # resolves to the IDENTICAL donor (the walk loop is unchanged).
                assert label == cycle_donor, (slot, dose, vec_type, pid, label, cycle_donor)
    assert exhausted == [], f"{len(exhausted)} walk exhaustions: {exhausted}"
    mp_ids = {p.pair_id for p in pairs if p.setting == "matched_prefix"}
    assert {(s, d, v) for s, d, v, _ in degenerate} == {("pe", "replace", "A")}
    assert {pid for *_, pid in degenerate} == mp_ids  # all 30 mp pairs, nothing else
    assert {(s, d, v) for s, d, v, _ in beyond_cycle} == {("pe", "replace", "A")}
    assert {pid for *_, pid in beyond_cycle} == {
        "x--bare__q4--persona__q3",
        "x--bare__q5--persona__q2",
    }


def test_mp_pe_replace_null_is_degenerate_self_state(pairs):
    """Fails pre-fix (AssertionError 'no eligible donor'): a matched-prefix
    pair's pe x replace null installs the recipient's OWN V_B — the steered
    replace is a no-op by the causal identity (V_B(pe) == V_A(pe)), so the
    matched null is the same no-op, recorded ``self:<pair_id>``."""
    bank = _synthetic_bank_same_prefix_pe(pairs)
    pairs_by_id = {p.pair_id: p for p in pairs}
    donor_map = BANK.donor_derangement(pairs)
    mp = next(p for p in pairs if p.setting == "matched_prefix")
    mode, _alpha, payload_kind = R._realized_mode("pe", "replace", "A")
    assert (mode, payload_kind) == ("replace", "state")
    _d, state, _m = R._pair_payload(bank, mp, "pe", "A")
    assert float(state.norm()) > 0  # the state recipient is NOT the zero Delta
    payload, label = R._resolve_donor(bank, mp, donor_map, pairs_by_id, "pe", "A", state, "state")
    assert label == f"self:{mp.pair_id}"
    assert torch.equal(payload, state)
    # the steered no-op identity the carve-out mirrors: V_A(pe) == V_B(pe)
    assert torch.equal(bank["per_context"][mp.a]["v_pe"], bank["per_context"][mp.b]["v_pe"])


def test_state_walk_beyond_cycle_fallback_on_exhausted_cycle(pairs):
    """Fails pre-fix (AssertionError 'no eligible donor'): two cross pairs
    sharing prefix_b form a donor 2-cycle that the round-2 same-target-state
    exclusion exhausts; the walk must continue deterministically over the
    sorted setting group instead of raising."""
    bank = _synthetic_bank_same_prefix_pe(pairs)
    pairs_by_id = {p.pair_id: p for p in pairs}
    xs = [p for p in pairs if p.setting == "cross"]
    a, b = [p for p in xs if p.prefix_b == xs[0].prefix_b][:2]
    synth_map = {p.pair_id: p.pair_id for p in xs}
    synth_map[a.pair_id] = b.pair_id
    synth_map[b.pair_id] = a.pair_id
    _d, state, _m = R._pair_payload(bank, a, "pe", "A")
    payload, donor_id = R._resolve_donor(bank, a, synth_map, pairs_by_id, "pe", "A", state, "state")
    donor = pairs_by_id[donor_id]
    assert donor.setting == "cross" and donor_id not in (a.pair_id, b.pair_id)
    assert donor.prefix_b != a.prefix_b  # the same-target-state exclusion still honored
    expected = next(
        pid
        for pid in sorted(synth_map)
        if pid not in (a.pair_id, b.pair_id)
        and R._donor_eligible(pairs_by_id[pid], "pe", a, "state")
    )
    assert donor_id == expected  # deterministic: sorted-first eligible group member
    assert torch.allclose(payload.norm(dim=-1), state.norm(dim=-1), rtol=1e-4, atol=1e-6)


# ── round-2 Critical 2: bank / anchors resume predicates ───────────────


def _mk_cfg(tmp_path: Path, *, force: bool = False, upload_mode: str = "none") -> R.RunConfig:
    return R.RunConfig(
        phase="bank",
        out_root=tmp_path / "out",
        log_dir=tmp_path / "logs",
        model_id="tiny",
        tiny=True,
        n_layers=N_LAYERS,
        hidden=6,
        device="cpu",
        gen_batch=2,
        capture_batch=2,
        max_new_tokens=8,
        anchor_draws=3,
        seed_base=42,
        smoke=False,
        pilot=False,
        force=force,
        worker_index=0,
        num_workers=1,
        upload_mode=upload_mode,
        upload_every=25,
        planned_wall_h=1.0,
        gpu_hours_budgeted=1.0,
    )


def test_bank_resume_predicate_skips_before_model_load(tmp_path, monkeypatch):
    """Round-2 Critical 2 pin: a completed same-regime bank is skipped at
    entry BEFORE the model load; a regime mismatch HARD-refuses (#722 r3);
    --force deliberately re-runs; missing artifacts re-run."""
    cfg = _mk_cfg(tmp_path)
    _manifest, bank_sha = R.bank_manifest_and_sha()
    fp = R.regime_fingerprint(cfg, bank_sha)
    assert not R.bank_is_done(cfg, fp)  # no done-manifest yet

    cfg.bank_dir.mkdir(parents=True, exist_ok=True)
    for name in ("bank.json", "injection_gate_report.json"):
        (cfg.bank_dir / name).write_text("{}")
    (cfg.bank_dir / "vc_bank.pt").write_bytes(b"x")
    cfg.manifest_dir.mkdir(parents=True, exist_ok=True)
    (cfg.manifest_dir / "bank_done.json").write_text(json.dumps({"regime_fp": fp}))
    assert R.bank_is_done(cfg, fp)

    def _no_load(_cfg):
        raise AssertionError("model load must not run on a done bank")

    monkeypatch.setattr(R, "load_model_and_tokenizer", _no_load)
    assert R.phase_bank(cfg) == R.RC_OK  # skip precedes the model load

    # --force deliberately re-runs (reaches the model load).
    with pytest.raises(AssertionError, match="model load must not run"):
        R.phase_bank(_mk_cfg(tmp_path, force=True))

    # Regime mismatch is a HARD refusal, never a silent cross-regime reuse.
    with pytest.raises(RuntimeError, match="refusing to resume across regimes"):
        R.bank_is_done(cfg, "deadbeefdeadbeef")

    # A missing output artifact re-runs (done-manifest alone is not done).
    (cfg.bank_dir / "vc_bank.pt").unlink()
    assert not R.bank_is_done(cfg, fp)


def test_anchors_resume_predicate(tmp_path, monkeypatch):
    """anchors_is_done: regime + draws + artifact presence + ROW-COUNT check."""
    cfg = _mk_cfg(tmp_path)
    _manifest, bank_sha = R.bank_manifest_and_sha()
    fp = R.regime_fingerprint(cfg, bank_sha)
    assert not R.anchors_is_done(cfg, fp, cfg.anchor_draws)

    cfg.anchors_dir.mkdir(parents=True, exist_ok=True)
    (cfg.anchors_dir / "anchors.jsonl").write_text('{"a": 1}\n{"a": 2}\n')
    (cfg.anchors_dir / "va_anchors.pt").write_bytes(b"x")
    cfg.manifest_dir.mkdir(parents=True, exist_ok=True)
    (cfg.manifest_dir / "anchors_done.json").write_text(
        json.dumps({"regime_fp": fp, "draws": cfg.anchor_draws, "n_rows": 2})
    )
    assert R.anchors_is_done(cfg, fp, cfg.anchor_draws)
    # Draw-count mismatch -> re-run.
    assert not R.anchors_is_done(cfg, fp, cfg.anchor_draws + 1)
    # Row-count mismatch -> re-run.
    (cfg.anchors_dir / "anchors.jsonl").write_text('{"a": 1}\n')
    assert not R.anchors_is_done(cfg, fp, cfg.anchor_draws)
    (cfg.anchors_dir / "anchors.jsonl").write_text('{"a": 1}\n{"a": 2}\n')

    # Skip precedes the model load.
    def _no_load(_cfg):
        raise AssertionError("model load must not run on done anchors")

    monkeypatch.setattr(R, "load_model_and_tokenizer", _no_load)
    assert R.phase_anchors(cfg) == R.RC_OK
    with pytest.raises(RuntimeError, match="refusing to resume across regimes"):
        R.anchors_is_done(cfg, "deadbeefdeadbeef", cfg.anchor_draws)


# ── round-2 Critical 1: upload fail-loud (bounded outer retry) ──────────


def _fake_upload_fn(ret):
    """Signature-conformant fake of hub._upload_folder_filtered (external
    Hub boundary only — _upload_dir's real body executes)."""

    def _fn(
        local_dir: Path,
        repo_id: str,
        repo_type: str,
        path_in_repo: str,
        allow_patterns: list[str],
        expected_repo_paths: list[str],
        ignore_patterns: list[str] | None = None,
        delete_after: bool = False,
    ) -> str:
        _fn.calls += 1  # type: ignore[attr-defined]
        return ret if not callable(ret) else ret(_fn.calls)  # type: ignore[attr-defined]

    _fn.calls = 0  # type: ignore[attr-defined]
    return _fn


def test_upload_dir_raises_on_no_path_after_bounded_retry(tmp_path, monkeypatch):
    """Round-2 Critical 1 pin (fails pre-fix): the hub helper's fail-soft ""
    return is retried (bounded, jittered) then RAISES — the results sentinel
    can never post over silently-lost durability."""
    import explore_persona_space.orchestrate.hub as hub

    cfg = _mk_cfg(tmp_path, upload_mode="hf")
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "x.json").write_text("{}")
    fail = _fake_upload_fn("")
    sleeps: list[float] = []
    monkeypatch.setattr(hub, "_upload_folder_filtered", fail)
    monkeypatch.setattr(R, "_upload_retry_sleep", sleeps.append)
    with pytest.raises(RuntimeError, match="upload returned no path"):
        R._upload_dir(cfg, stage, "issue2094_test/prefix", ["*.json"])
    assert fail.calls == R.UPLOAD_TRANSPORT_RETRIES + 1
    assert len(sleeps) == R.UPLOAD_TRANSPORT_RETRIES
    assert all(s >= R.UPLOAD_BACKOFF_BASE_S[0] for s in sleeps)


def test_upload_dir_recovers_on_transient_no_path(tmp_path, monkeypatch):
    """One fail-soft return then success -> no raise, exact expected set."""
    import explore_persona_space.orchestrate.hub as hub

    cfg = _mk_cfg(tmp_path, upload_mode="hf")
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "x.json").write_text("{}")
    flaky = _fake_upload_fn(lambda n: "" if n == 1 else "datasets/repo/prefix")
    monkeypatch.setattr(hub, "_upload_folder_filtered", flaky)
    monkeypatch.setattr(R, "_upload_retry_sleep", lambda _s: None)
    out = R._upload_dir(cfg, stage, "issue2094_test/prefix", ["*.json"])
    assert out == ["issue2094_test/prefix/x.json"]
    assert flaky.calls == 2


def test_upload_manifests_patterns_include_block_done_files(tmp_path):
    """Round-2 Minor 7: the manifests upload matches blocks/*.done.json too
    (per-block resume state + cap-hit provenance become durable off-pod)."""
    cfg = _mk_cfg(tmp_path, upload_mode="local-mirror")
    cfg.manifest_dir.mkdir(parents=True, exist_ok=True)
    (cfg.manifest_dir / "anchors_done.json").write_text("{}")
    blocks = cfg.manifest_dir / "blocks"
    blocks.mkdir()
    (blocks / "b1.done.json").write_text("{}")
    out = R._upload_dir(
        cfg,
        cfg.manifest_dir,
        "issue2094_test/manifests",
        ["*.json", "blocks/*.done.json"],
    )
    assert "issue2094_test/manifests/anchors_done.json" in out
    assert "issue2094_test/manifests/blocks/b1.done.json" in out
