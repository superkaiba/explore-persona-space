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
    prefix-end Delta exactly zero (found by the unit-F e2e smoke)."""
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
    return {"layers": list(range(N_LAYERS)), "per_context": per_context, "centroids": {}}


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
