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
