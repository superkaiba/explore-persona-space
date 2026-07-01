"""Issue #811 unit tests — turn_nl answer-summary reader + loader + KILL-1 gate.

The single manipulated variable vs #722 is the answer-side summary (mean → turn_nl).
These pin the three load-bearing invariants:

1. ``_locate_turn_close_newline`` (KILL-2, extract phase) — the turn-close newline
   is ``full_ids[-1]`` with ``<|im_end|>`` at ``-2``; a missing/malformed tail
   raises (no silent fallback), and a mean-only read is byte-unchanged.
2. ``issue722_load_activations._blob_to_record`` summary selection — ``mean`` reads
   v0/v_plus, ``turn_nl`` reads v0_turn_nl/v_plus_turn_nl, and c_C is IDENTICAL
   across summaries (answer-side manipulation only); a turn_nl read against a
   mean-only store fails loud.
3. ``issue811_fit._kill1_decision`` (KILL-1, plan §7) — fires on ≥2-of-3 base-leg
   validity-gate collapses at L14, excludes a behavior whose mean has no positive
   baseline gate, and does NOT fire on a single collapse.

The extract-reader tests reuse the #667 test's tiny 2-layer CPU stub + the REAL
Qwen tokenizer (carve-out item 1 for the GPU-bound extract phase — the pre-CUDA
tokenization + turn-close arithmetic, no 7B load, no GPU).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from test_issue667_gate_chain import _TinyStub  # noqa: E402  (reuse the CPU stub)


def _tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")


# ── 1. _locate_turn_close_newline (KILL-2) ────────────────────────────────────


def test_locate_turn_close_newline_on_real_template():
    """turn_nl_idx == full_len-1 and the tail is <|im_end|> then a newline (A2)."""
    import issue667_extract as ex

    tok = _tok()
    msgs = [{"role": "system", "content": "You are X."}, {"role": "user", "content": "Hi?"}]
    full = tok.apply_chat_template(
        [*msgs, {"role": "assistant", "content": "Hello there."}],
        tokenize=False,
        add_generation_prompt=False,
    )
    full_ids = tok.encode(full, add_special_tokens=False)
    idx = ex._locate_turn_close_newline(full_ids, tok)
    assert idx == len(full_ids) - 1
    assert full_ids[-2] == ex.IM_END_ID
    assert "\n" in tok.decode([full_ids[-1]])


def test_locate_turn_close_newline_raises_on_stripped_newline():
    """A sequence missing the trailing newline fails loud (KILL-2, no silent fallback)."""
    import issue667_extract as ex
    import pytest

    tok = _tok()
    msgs = [{"role": "user", "content": "Hi?"}]
    full = tok.apply_chat_template(
        [*msgs, {"role": "assistant", "content": "Hello."}],
        tokenize=False,
        add_generation_prompt=False,
    )
    full_ids = tok.encode(full, add_special_tokens=False)
    with pytest.raises(RuntimeError, match="turn_nl-assert"):
        ex._locate_turn_close_newline(full_ids[:-1], tok)  # drop the newline → <|im_end|> last


def test_mean_resp_acts_summaries_shapes_and_backward_compat():
    """summaries=('mean',) keeps (v0,v_plus); ('mean','turn_nl') adds the parallel key."""
    import issue667_extract as ex

    tok = _tok()
    vocab = len(tok)
    torch.manual_seed(1)
    base = _TinyStub(vocab, hidden=8, n_layers=2)
    trained = _TinyStub(vocab, hidden=8, n_layers=2)
    msgs = [{"role": "system", "content": "You are X."}, {"role": "user", "content": "Hi?"}]
    # default shape (backward-compat)
    out_mean = ex._mean_resp_acts(
        base, trained, tok, msgs, "Hello there.", [1], torch.device("cpu")
    )
    v0, vp = out_mean[1]
    assert v0.shape == (8,) and vp.shape == (8,)
    # nested shape with turn_nl
    out2 = ex._mean_resp_acts(
        base,
        trained,
        tok,
        msgs,
        "Hello there.",
        [1],
        torch.device("cpu"),
        summaries=("mean", "turn_nl"),
    )
    assert set(out2[1]) == {"mean", "turn_nl"}
    m0, mp = out2[1]["mean"]
    n0, npp = out2[1]["turn_nl"]
    # mean is identical across call shapes (same forward pass, same reduction)
    assert np.allclose(m0, v0) and np.allclose(mp, vp)
    # turn_nl is a single-position read, distinct from the span-mean, base != trained
    assert n0.shape == (8,) and not np.allclose(n0, m0) and not np.allclose(n0, npp)


# ── 2. loader summary selection ───────────────────────────────────────────────


def _synth_blob():
    import issue722_load_activations as la

    H = la.HIDDEN
    rng = np.random.default_rng(0)
    return {
        "v0": rng.standard_normal(H).astype(np.float32),
        "v_plus": rng.standard_normal(H).astype(np.float32),
        "v0_turn_nl": rng.standard_normal(H).astype(np.float32),
        "v_plus_turn_nl": rng.standard_normal(H).astype(np.float32),
        "c_C": rng.standard_normal(H).astype(np.float32),
        "c_C_postft": rng.standard_normal(H).astype(np.float32),
        "behavior": np.asarray("fact"),
        "source_cid": np.asarray("default"),
        "target_cid": np.asarray("sp_swe"),
        "layer": np.asarray(14),
    }


def test_blob_to_record_summary_selection_and_shared_context():
    import issue722_load_activations as la

    blob = _synth_blob()
    rec_mean = la._blob_to_record(blob, "rel", "fact", 14, "mean")
    rec_nl = la._blob_to_record(blob, "rel", "fact", 14, "turn_nl")
    # mean reads v0/v_plus; turn_nl reads v0_turn_nl/v_plus_turn_nl
    assert np.array_equal(rec_mean.v0, blob["v0"])
    assert np.array_equal(rec_nl.v0, blob["v0_turn_nl"])
    assert np.array_equal(rec_nl.vplus, blob["v_plus_turn_nl"])
    # c_C / c_C_postft are IDENTICAL across summaries (answer-side change only)
    assert np.array_equal(rec_mean.c0, rec_nl.c0)
    assert np.array_equal(rec_mean.cplus, rec_nl.cplus)


def test_blob_to_record_turn_nl_fails_loud_on_mean_only_store():
    import issue722_load_activations as la
    import pytest

    blob = _synth_blob()
    del blob["v0_turn_nl"]  # a mean-only #667 store has no turn_nl keys
    with pytest.raises(KeyError, match="v0_turn_nl"):
        la._blob_to_record(blob, "rel", "fact", 14, "turn_nl")


# ── 3. KILL-1 base-leg validity decision (plan §7) ────────────────────────────


def _cbs(mean_margins: dict, turn_margins: dict) -> dict:
    import issue811_fit as f

    pl = f.PRIMARY_LAYER
    return {
        "mean": {(b, pl, "mean"): {"gate_margin": m} for b, m in mean_margins.items()},
        "turn_nl": {(b, pl, "turn_nl"): {"gate_margin": m} for b, m in turn_margins.items()},
    }


def test_kill1_fires_on_two_of_three_collapse():
    import issue811_fit as f

    cbs = _cbs(
        {"em": 0.4, "sycophancy": 0.4, "fact": 0.4},
        {"em": 0.1, "sycophancy": 0.05, "fact": 0.35},  # em+syco < 0.5*0.4; fact holds
    )
    d = f._kill1_decision(cbs)
    assert d["fired"] is True and d["n_collapse"] == 2
    assert d["per_behavior"]["fact"]["status"] == "held"


def test_kill1_does_not_fire_on_single_collapse():
    import issue811_fit as f

    cbs = _cbs(
        {"em": 0.4, "sycophancy": 0.4, "fact": 0.4},
        {"em": 0.1, "sycophancy": 0.35, "fact": 0.35},  # only em collapses
    )
    d = f._kill1_decision(cbs)
    assert d["fired"] is False and d["n_collapse"] == 1


def test_kill1_excludes_behavior_with_no_mean_gate():
    import issue811_fit as f

    cbs = _cbs(
        {"em": -0.1, "sycophancy": 0.4, "fact": 0.4},  # em's mean has no positive gate
        {"em": 0.0, "sycophancy": 0.05, "fact": 0.05},
    )
    d = f._kill1_decision(cbs)
    assert d["per_behavior"]["em"]["status"] == "mean_no_gate"
    assert d["n_comparable"] == 2 and d["n_collapse"] == 2 and d["fired"] is True


# ── 4. Phase-0 base-leg gate: store shape + loader + pre-spend routing ─────────


def _write_phase0_cell(root: Path, behavior: str, source: str, target: str, layer: int) -> None:
    """Write a base-leg-only phase0 .npz (c_C / v0 / v0_turn_nl only, NO v_plus)."""
    import issue722_load_activations as la

    H = la.HIDDEN
    rng = np.random.default_rng(abs(hash((source, target))) % (2**32))
    d = root / behavior / f"{source}_seed42"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(
        d / f"{target}_L{layer}.npz",
        c_C=rng.standard_normal(H).astype(np.float32),
        v0=rng.standard_normal(H).astype(np.float32),
        v0_turn_nl=rng.standard_normal(H).astype(np.float32),
        behavior=np.asarray(behavior),
        source_cid=np.asarray(source),
        target_cid=np.asarray(target),
        layer=np.asarray(layer),
    )


def test_phase0_base_loader_reads_base_only_store(tmp_path):
    """The phase0 loader reads c_C->C0 + v0/v0_turn_nl->V0 with NO v_plus required."""
    import issue811_fit as f

    layer = f.PRIMARY_LAYER
    # 4 sources x 3 targets so C0 has >=4 rows (the gate's minimum).
    for si in range(4):
        for ti in range(3):
            _write_phase0_cell(tmp_path, "em", f"src{si}", f"tgt{ti}", layer)
    out = f._load_phase0_base_cells(
        ("em",),
        ("mean", "turn_nl"),
        layer,
        local_root=str(tmp_path),
        max_sources=None,
        max_targets_per_source=None,
        strict=False,  # not the full 480-cell grid
    )
    # Both summaries share the SAME base context (c_C -> C0); V0 differs.
    mean, turn = out[("em", "mean")], out[("em", "turn_nl")]
    assert mean["C0"].shape == (12, 3584) and mean["V0"].shape == (12, 3584)
    assert np.array_equal(mean["C0"], turn["C0"])  # answer-side manipulation only
    assert not np.array_equal(mean["V0"], turn["V0"])  # distinct answer summaries
    assert len(mean["cell_keys"]) == 12


def test_phase0_base_loader_fails_loud_on_mean_only_store(tmp_path):
    """A base-leg store missing v0_turn_nl fails loud (a mean-only / wrong-prefix store)."""
    import issue811_fit as f
    import pytest

    layer = f.PRIMARY_LAYER
    _write_phase0_cell(tmp_path, "em", "src0", "tgt0", layer)
    # Strip the turn_nl key from the one cell we wrote.
    p = tmp_path / "em" / "src0_seed42" / f"tgt0_L{layer}.npz"
    d = {k: np.load(p, allow_pickle=True)[k] for k in np.load(p, allow_pickle=True).files}
    del d["v0_turn_nl"]
    np.savez(p, **d)
    with pytest.raises(KeyError, match="v0_turn_nl"):
        f._load_phase0_base_cells(
            ("em",),
            ("mean", "turn_nl"),
            layer,
            local_root=str(tmp_path),
            max_sources=None,
            max_targets_per_source=None,
            strict=False,
        )


# ── 5. Mean-parity check: fires on drift, passes on match ──────────────────────


def test_mean_parity_fires_on_drift(monkeypatch, tmp_path):
    """A re-extracted v0 that diverges from the #667 ref FAILS LOUD (failure_class: code)."""
    import issue811_mean_parity_check as pc
    import pytest

    H = 3584
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(H).astype(np.float64)
    # Stub the #667 reference loader to return `ref`; write a phase0 cell whose v0
    # is an UNRELATED vector (cosine ~ 0) — a real reader-logic drift.
    monkeypatch.setattr(pc, "_load_ref_v0", lambda b, s, t, layer: ref)
    _write_phase0_cell_for_parity(tmp_path, "em", "src0", "tgt0", 14, rng.standard_normal(H))
    with pytest.raises(RuntimeError, match="MEAN-PARITY DRIFT"):
        pc.check_mean_parity(tmp_path, "em", 14, 1)


def test_mean_parity_passes_on_match(monkeypatch, tmp_path):
    """A re-extracted v0 that matches the #667 ref up to tiny bf16-scale noise PASSES."""
    import issue811_mean_parity_check as pc

    H = 3584
    rng = np.random.default_rng(1)
    ref = rng.standard_normal(H).astype(np.float64)
    monkeypatch.setattr(pc, "_load_ref_v0", lambda b, s, t, layer: ref)
    # v0 = ref + tiny noise (bf16-scale) -> cosine ~ 1, rel_l2 tiny -> PASS.
    noisy = (ref + 1e-3 * rng.standard_normal(H)).astype(np.float32)
    _write_phase0_cell_for_parity(tmp_path, "em", "src0", "tgt0", 14, noisy)
    recs = pc.check_mean_parity(tmp_path, "em", 14, 1)
    assert len(recs) == 1 and recs[0]["ok"] is True
    assert recs[0]["cosine"] >= pc.COS_FLOOR and recs[0]["rel_l2"] <= pc.REL_L2_CEIL


def _write_phase0_cell_for_parity(
    root: Path, behavior: str, source: str, target: str, layer: int, v0: np.ndarray
) -> None:
    """Write a phase0 cell with a caller-supplied v0 (the parity check reads v0)."""
    d = root / behavior / f"{source}_seed42"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(
        d / f"{target}_L{layer}.npz",
        v0=v0.astype(np.float32),
        source_cid=np.asarray(source),
        target_cid=np.asarray(target),
        layer=np.asarray(layer),
    )


# ── 6. run_phase0_gate: degenerate (vacuous) PASS is fail-loud on production ────


def _phase0_gate_args(**overrides):
    """A minimal argparse.Namespace for run_phase0_gate (the fields it reads)."""
    import argparse

    import issue658_fit_predictors as fit658
    import issue722_fit_M as fitM
    import issue811_fit as f

    ns = argparse.Namespace(
        behaviors=["em", "sycophancy"],  # NO fact -> no r_b_fact load needed
        primary_layer=f.PRIMARY_LAYER,
        local_root=None,
        smoke=False,
        mlp_epochs=1,
        target_dim=min(fitM.TARGET_DIM if hasattr(fitM, "TARGET_DIM") else 4, 4),
        num_threads=1,
        max_sources=None,
        max_targets_per_source=None,
        out_dir=None,  # set by the caller to a tmp cells dir
    )
    _ = fit658  # imported to mirror run_phase0_gate's module surface
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def test_phase0_gate_fails_loud_on_degenerate_pass(monkeypatch, tmp_path):
    """0 comparable behaviors on a NON-smoke run RAISES (never a vacuous PASS).

    Reproduces the round-3 BLOCKER root cause directly: a gate whose store has NO
    comparable cells (a not-yet-uploaded / empty / wrong-prefix HF store). Before the
    fix this returned 0 (fired: false, n_comparable: 0) — a silent PASS that let the
    ~7 GPU-h Phase-1 spend proceed without ever deciding. After the fix it raises.
    """
    import json

    import issue722_fit_M as fitM
    import issue811_fit as f
    import pytest

    monkeypatch.setattr(f, "_resolve_device", lambda: "cpu")
    monkeypatch.setattr(fitM, "_load_rb_main", lambda: {})
    monkeypatch.setattr(fitM, "_load_rb_fact", lambda: None)
    # An empty store: every (behavior, summary) has 0 base cells -> all skipped ->
    # groups_by_cell empty -> _kill1_decision reports n_comparable == 0.
    monkeypatch.setattr(
        f,
        "_load_phase0_base_cells",
        lambda behaviors, summaries, layer, **kw: {
            (b, s): {
                "C0": np.zeros((0, 3584)),
                "V0": np.zeros((0, 3584)),
                "cell_keys": [],
            }
            for b in behaviors
            for s in summaries
        },
    )
    args = _phase0_gate_args(out_dir=tmp_path / "cells")
    with pytest.raises(RuntimeError, match="empty base-leg store"):
        f.run_phase0_gate(args)
    # The diagnostic decision JSON still landed for debugging (n_comparable == 0).
    kill1 = json.loads((tmp_path / "kill1_base_leg_validity.json").read_text())
    assert kill1["n_comparable"] == 0 and kill1["fired"] is False
    assert kill1["state"] == "empty_store"  # State A: no gate computed for any behavior


def test_phase0_gate_tolerates_degenerate_pass_under_smoke(monkeypatch, tmp_path):
    """--smoke intentionally slices to 0 comparable cells; the degenerate gate is OK there."""
    import json

    import issue722_fit_M as fitM
    import issue811_fit as f

    monkeypatch.setattr(f, "_resolve_device", lambda: "cpu")
    monkeypatch.setattr(fitM, "_load_rb_main", lambda: {})
    monkeypatch.setattr(fitM, "_load_rb_fact", lambda: None)
    monkeypatch.setattr(
        f,
        "_load_phase0_base_cells",
        lambda behaviors, summaries, layer, **kw: {
            (b, s): {
                "C0": np.zeros((0, 3584)),
                "V0": np.zeros((0, 3584)),
                "cell_keys": [],
            }
            for b in behaviors
            for s in summaries
        },
    )
    args = _phase0_gate_args(out_dir=tmp_path / "cells", smoke=True)
    rc = f.run_phase0_gate(args)  # no raise under smoke
    assert rc == 0
    kill1 = json.loads((tmp_path / "kill1_base_leg_validity.json").read_text())
    assert kill1["n_comparable"] == 0 and kill1["fired"] is False
    assert kill1["state"] == "empty_store"  # State A even under smoke (raise suppressed)


def _populated_base_cells(behaviors, summaries, layer, **kw):
    """A POPULATED base-leg store: >=4 cells per (behavior, summary) so the gate builds
    a real cell (State B fixture — distinct from the empty-store State A)."""
    rng = np.random.default_rng(7)
    out = {}
    for b in behaviors:
        for s in summaries:
            out[(b, s)] = {
                "C0": rng.standard_normal((12, 3584)).astype(np.float64),
                "V0": rng.standard_normal((12, 3584)).astype(np.float64),
                "cell_keys": [f"{b}/src{i}__tgt{j}" for i in range(4) for j in range(3)],
            }
    return out


def test_phase0_gate_reports_not_kills_on_negative_mean_gate(monkeypatch, tmp_path):
    """State B: POPULATED store, mean base-map gate margin <= 0 on ALL 3 behaviors.

    This ALSO yields n_comparable == 0, but it is a LEGITIMATE #722-style outcome
    (mean MLP-vs-shuffle negative below its shuffle null in 8/9 cells; #811 reuses the
    SAME fit code + paired-store lineage + n=16), NOT an empty/unuploaded store. mean
    has no gate for turn_nl to collapse relative to, so KILL-1 cannot decide against
    turn_nl. run_phase0_gate MUST return 0 (fired: false, state: reported_not_killed)
    — a healthy run must never raise blocked/failure_class:code with a false
    empty-store message (round-4 BLOCKER phase0-gate-degenerate-guard-over-broad,
    State-A vs State-B conflation).
    """
    import json

    import issue722_fit_M as fitM
    import issue811_fit as f

    monkeypatch.setattr(f, "_resolve_device", lambda: "cpu")
    monkeypatch.setattr(fitM, "_load_rb_main", lambda: {})
    monkeypatch.setattr(fitM, "_load_rb_fact", lambda: None)
    # r_hat feeds cell_meta only; the monkeypatched gate below ignores it, so a stub
    # unit vector is enough (the real r_b_fact/r_b .pt aren't loaded in this CPU test).
    monkeypatch.setattr(fitM, "_r_hat_for", lambda b, layer, rb_main, rb_fact: np.ones(3584))
    # Populated store so groups_by_cell is NON-empty and the gate runs on real cells.
    monkeypatch.setattr(f, "_load_phase0_base_cells", _populated_base_cells)

    # Force the gate to report a NEGATIVE mean margin on every cell (the #722 pattern),
    # while turn_nl also has some margin — every mean cell -> status mean_no_gate.
    def _neg_mean_gate(groups_by_cell, cell_meta, **kw):
        out = {}
        for cell_key in groups_by_cell:
            _behavior, _layer, summary = cell_key
            margin = -0.3 if summary == "mean" else 0.2  # mean <= 0 -> no baseline gate
            out[cell_key] = {
                "rho_real": margin,
                "rho_shuffle": 0.0,
                "gate_margin": margin,
                "n_with_E": 12,
            }
        return out

    monkeypatch.setattr(f, "compute_mlp_validity_gate", _neg_mean_gate)
    args = _phase0_gate_args(out_dir=tmp_path / "cells")  # NON-smoke production run
    rc = f.run_phase0_gate(args)  # must NOT raise (State B is legitimate)
    assert rc == 0
    kill1 = json.loads((tmp_path / "kill1_base_leg_validity.json").read_text())
    assert kill1["n_comparable"] == 0 and kill1["fired"] is False
    assert kill1["state"] == "reported_not_killed"  # populated store, mean margins <= 0
    assert kill1["n_mean_no_gate"] == 2  # em + sycophancy (the _phase0_gate_args set)
    assert all(
        e["status"] == "mean_no_gate"
        for e in kill1["per_behavior"].values()
        if e["mean_margin"] is not None
    )


# ── 7. upload_store: BOTH stores required (never an incomplete commit) ──────────


def _write_upload_npz(dir_path: Path, name: str = "c.npz") -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    np.savez(dir_path / name, x=np.zeros(3, dtype=np.float32))


def test_upload_store_raises_when_phase0_store_empty(monkeypatch, tmp_path):
    """analysis_tensors populated but phase0_base_leg empty -> RAISE (round-3 Major).

    An aggregate 'any store non-empty' guard would silently commit + verify only the
    populated store and omit the other. The per-store precondition catches it BEFORE
    any Hub commit (no network calls reached — the raise precedes create_commit).
    """
    import issue811_upload_store as up
    import pytest

    monkeypatch.setattr(up, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("HF_TOKEN", "test-token")  # pass the load_dotenv assert
    monkeypatch.delenv("EPM_SKIP_UPLOAD", raising=False)
    # analysis_tensors has a cell; phase0_base_leg has NONE.
    _write_upload_npz(tmp_path / "eval_results/issue_811/analysis_tensors/em/src0_seed42")
    (tmp_path / "eval_results/issue_811/phase0_base_leg").mkdir(parents=True, exist_ok=True)
    with pytest.raises(RuntimeError, match="required uploads"):
        up.upload_store()


def test_upload_store_raises_when_analysis_store_empty(monkeypatch, tmp_path):
    """phase0_base_leg populated but analysis_tensors empty -> RAISE (symmetric)."""
    import issue811_upload_store as up
    import pytest

    monkeypatch.setattr(up, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.delenv("EPM_SKIP_UPLOAD", raising=False)
    _write_upload_npz(tmp_path / "eval_results/issue_811/phase0_base_leg/em/src0_seed42")
    (tmp_path / "eval_results/issue_811/analysis_tensors").mkdir(parents=True, exist_ok=True)
    with pytest.raises(RuntimeError, match="required uploads"):
        up.upload_store()
