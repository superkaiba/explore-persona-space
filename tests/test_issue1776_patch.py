"""#1776 follow-up slot_patch_sufficiency pins (plan v8).

DeltaHook ``replace=True`` semantics (the round's one library change) through
the PRODUCTION parity probe on a tiny from-config Qwen2 (real tokenizer, real
hook body — fakes only at the GPU-scale boundary), plus the patch-round
regime-refusal + unit-enumeration pins. Fixture text is benign synthetic
prose (never corpus rows).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue1776_swap as SW  # noqa: E402

from explore_persona_space.experiments.issue1415.steering import DeltaHook  # noqa: E402

SOURCE_LAYER = 1  # tiny from-config model: block 1 of 2


@pytest.fixture(scope="module")
def tiny_model_tok():
    args = SimpleNamespace(tiny=True, model="unused-tiny", dtype="float32", device="cpu")
    return SW.P3.load_model(args)


def _probe_inputs(model, tok):
    """Two different-length contexts + DISTINCT per-row (B, H) probe values."""
    ctx2 = [
        {"system": None, "user": "Describe a quiet morning walk in the park."},
        {"system": None, "user": "What is two plus two?"},
    ]
    h = model.config.hidden_size
    g = torch.Generator().manual_seed(7)
    deltas = torch.randn(2, h, generator=g) * 3.0
    assert float((deltas[0] - deltas[1]).abs().max()) > 0
    return ctx2, deltas


# ── DeltaHook replace semantics (through the PRODUCTION parity probe) ─────────


def test_parity_probe_replace_mode_passes(tiny_model_tok):
    """replace=True writes EXACTLY the per-row value at each row's T-1 slot,
    other positions untouched, one edit per forward (G-PATCH-PARITY body)."""
    model, tok = tiny_model_tok
    ctx2, deltas = _probe_inputs(model, tok)
    par = SW.parity_probe(model, tok, ctx2, deltas, SOURCE_LAYER, replace=True)
    assert par["pass"], par
    assert par["n_edits"] == 1
    assert par["replace"] is True
    assert par["max_abs_dev_at_slot"] <= 1e-4
    assert par["max_abs_dev_other_positions"] == 0.0


def test_parity_probe_add_mode_unchanged(tiny_model_tok):
    """The swap round's default ADD path is byte-identical post-flag (the
    replace kwarg defaults False)."""
    model, tok = tiny_model_tok
    ctx2, deltas = _probe_inputs(model, tok)
    par = SW.parity_probe(model, tok, ctx2, deltas, SOURCE_LAYER)
    assert par["pass"], par
    assert par["replace"] is False


def test_patch_parity_gate_trips_on_add_mode_hook(tiny_model_tok, monkeypatch):
    """Fails-pre-fix pin: a hook that ADDS instead of REPLACES (the pre-flag
    behavior) FAILS the replace-expectation probe — the rc=8 gate branch's
    degenerate-input demonstration (real DeltaHook body, flag forced off)."""
    model, tok = tiny_model_tok
    ctx2, deltas = _probe_inputs(model, tok)

    class AddOnlyHook(DeltaHook):
        def __init__(self, *a, **kw):
            kw["replace"] = False
            super().__init__(*a, **kw)

    monkeypatch.setattr(SW, "DeltaHook", AddOnlyHook)
    par = SW.parity_probe(model, tok, ctx2, deltas, SOURCE_LAYER, replace=True)
    assert not par["pass"], par
    assert par["max_abs_dev_at_slot"] > 1e-2  # ref + delta vs delta: far past tol


def test_replace_mutual_exclusion(tiny_model_tok):
    model, _tok = tiny_model_tok
    h = model.config.hidden_size
    d = torch.zeros(h)
    with pytest.raises(AssertionError, match="replace mode supports ONLY"):
        DeltaHook(model, SOURCE_LAYER, d, 1.0, all_positions=True, replace=True)
    with pytest.raises(AssertionError, match="replace mode supports ONLY"):
        DeltaHook(model, SOURCE_LAYER, d, 1.0, edit_position=3, replace=True)


# ── patch-round driver pins ───────────────────────────────────────────────────


def test_run_manifest_round_regime_refusal(tmp_path):
    """The run manifest keys on the ROUND: a patch resume into a swap out-root
    (or vice versa) is refused, never silently mixed (#722 r3 contract)."""
    args_common = dict(
        model="m",
        tiny=False,
        dtype="bfloat16",
        source_layer=14,
        readout_layer=19,
        k_samples=5,
        k_baseline=5,
        temperature=1.0,
        max_new_tokens=1024,
        gen_batch=16,
    )
    swap_manifest = SW._run_manifest(SimpleNamespace(round="swap", **args_common), "sha1", "sha2")
    SW._check_run_manifest(tmp_path, swap_manifest)
    patch_manifest = SW._run_manifest(SimpleNamespace(round="patch", **args_common), "sha1", "sha2")
    with pytest.raises(RuntimeError, match="round"):
        SW._check_run_manifest(tmp_path, patch_manifest)


def test_units_patch_arm_enumeration():
    pairs = [
        {"pair_id": f"p{i}", "included": True, "a_id": f"a{i}", "b_id": f"b{i}"} for i in range(3)
    ] + [{"pair_id": "px", "included": False}]
    units = SW._units(pairs, gen_batch=2, all_arms=SW.PATCH_ALL_ARMS)
    keys = [u["unit_key"] for u in units]
    assert keys == ["patch_a0_c000", "patch_a0_c001", "swap_patch_c000", "swap_patch_c001"]
    assert all(len(u["rows"]) <= 2 for u in units)


def test_round_registry_shape():
    assert SW.ROUNDS["swap"]["all_arms"] == SW.ALL_ARMS
    assert SW.ROUNDS["patch"]["all_arms"] == ("patch_a0", "swap_patch")
    assert SW.ROUNDS["patch"]["fu"] == "followup_slotpatch"
    assert SW.ROUNDS["patch"]["merged_subdir"] == "steered_slotpatch"
