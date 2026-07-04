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

import json
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


# ── 5. Parity check (round-6 redesign): 3 like-for-like checks per cell ─────────
#
# The single unachievable v0-identity gate (crashed the att-20260701-233116 run at
# cosine 0.997028 < floor 0.999 — a legitimate R-token-flip cell, NOT a bug) is
# replaced by: (a) HARD reader-faithfulness new c_C(t) vs #667 c_Cp(t); (b) v0
# argmax/confusion over the source's ref targets; (c) v0 gross-drift floor (0.98,
# R-resampling-aware). Fixtures monkeypatch the #667 ref loader — pure numpy, no net.


def _write_phase0_cell_for_parity(
    root: Path,
    behavior: str,
    source: str,
    target: str,
    layer: int,
    c_c: np.ndarray,
    v0: np.ndarray,
) -> None:
    """Write a phase0 cell with caller-supplied c_C + v0 (the parity check reads both)."""
    d = root / behavior / f"{source}_seed42"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(
        d / f"{target}_L{layer}.npz",
        c_C=c_c.astype(np.float32),
        v0=v0.astype(np.float32),
        source_cid=np.asarray(source),
        target_cid=np.asarray(target),
        layer=np.asarray(layer),
    )


def _stub_refs(monkeypatch, refs: dict):
    """Monkeypatch the #667 ref loader + per-source v0 lister from a fixture dict.

    ``refs`` maps ``(behavior, source, target) -> {"v0": ndarray, "c_Cp": ndarray}``.
    """
    import issue811_mean_parity_check as pc

    monkeypatch.setattr(pc, "_load_ref", lambda b, s, t, layer: refs[(b, s, t)])
    monkeypatch.setattr(
        pc,
        "_list_ref_v0_for_source",
        lambda b, s, layer, tgts: {t: refs[(b, s, t)]["v0"] for t in tgts},
    )


def test_parity_passes_on_faithful_reextraction(monkeypatch, tmp_path):
    """Faithful c_C (~1e-4 to c_Cp) + resampled-R v0 (cos 0.997, argmax OK) -> PASS.

    The exact real-data regime: reader is exact; v0 is in the R-resampling band and
    argmax-matches its own target. This is the case the old gate WRONGLY failed.
    """
    import issue811_mean_parity_check as pc

    H = 3584
    rng = np.random.default_rng(1)
    c_cp = rng.standard_normal(H).astype(np.float64)
    v0_ref0 = rng.standard_normal(H).astype(np.float64)
    v0_ref1 = rng.standard_normal(H).astype(np.float64)  # a second target (confusion set)
    refs = {
        ("em", "src0", "tgt0"): {"v0": v0_ref0, "c_Cp": c_cp},
        ("em", "src0", "tgt1"): {"v0": v0_ref1, "c_Cp": rng.standard_normal(H)},
    }
    _stub_refs(monkeypatch, refs)
    # c_C ~ c_Cp up to tiny numeric noise (faithful reader); v0 ~ 0.997 cos to its ref.
    c_c_new = (c_cp + 1e-4 * rng.standard_normal(H)).astype(np.float32)
    v0_new = _at_cosine(v0_ref0, 0.997, rng).astype(np.float32)
    _write_phase0_cell_for_parity(tmp_path, "em", "src0", "tgt0", 14, c_c_new, v0_new)
    _write_phase0_cell_for_parity(
        tmp_path, "em", "src0", "tgt1", 14, refs[("em", "src0", "tgt1")]["c_Cp"], v0_ref1
    )
    recs = pc.check_mean_parity(tmp_path, "em", 14, 1)  # sample the 1st cell (tgt0)
    assert len(recs) == 1 and recs[0]["ok"] is True
    r = recs[0]
    assert r["cc_ok"] and r["v0_argmax_ok"] and r["v0_ok"]
    assert r["cc_cosine"] >= pc.COS_FLOOR and r["cc_rel_l2"] <= pc.REL_L2_CEIL
    assert 0.98 <= r["v0_cosine"] < 0.999  # in the R-resampling band, below the old floor
    assert r["v0_argmax_target"] == "tgt0"


def test_parity_fires_on_c_c_drift(monkeypatch, tmp_path):
    """(a) A c_C that diverges from #667 c_Cp FAILS LOUD — the reader read is not exact."""
    import issue811_mean_parity_check as pc
    import pytest

    H = 3584
    rng = np.random.default_rng(0)
    c_cp = rng.standard_normal(H).astype(np.float64)
    v0_ref = rng.standard_normal(H).astype(np.float64)
    _stub_refs(monkeypatch, {("em", "src0", "tgt0"): {"v0": v0_ref, "c_Cp": c_cp}})
    # c_C is an UNRELATED vector (cosine ~ 0) — a reader-logic drift; v0 is faithful.
    _write_phase0_cell_for_parity(
        tmp_path, "em", "src0", "tgt0", 14, rng.standard_normal(H), v0_ref
    )
    with pytest.raises(RuntimeError, match="READER-FAITHFULNESS DRIFT"):
        pc.check_mean_parity(tmp_path, "em", 14, 1)


def test_parity_fires_on_v0_misalignment(monkeypatch, tmp_path):
    """(b) A cell whose v0 argmax-matches a DIFFERENT target FAILS LOUD (misalignment)."""
    import issue811_mean_parity_check as pc
    import pytest

    H = 3584
    rng = np.random.default_rng(2)
    c_cp = rng.standard_normal(H).astype(np.float64)
    v0_tgt0 = rng.standard_normal(H).astype(np.float64)
    v0_tgt1 = rng.standard_normal(H).astype(np.float64)
    refs = {
        ("em", "src0", "tgt0"): {"v0": v0_tgt0, "c_Cp": c_cp},
        ("em", "src0", "tgt1"): {"v0": v0_tgt1, "c_Cp": rng.standard_normal(H)},
    }
    _stub_refs(monkeypatch, refs)
    # Cell is labelled tgt0 but its v0 is a copy of the tgt1 ref (wrong-cell bug); the
    # c_C read is faithful so check (a) passes and (b) fires.
    _write_phase0_cell_for_parity(
        tmp_path, "em", "src0", "tgt0", 14, (c_cp + 1e-4 * rng.standard_normal(H)), v0_tgt1
    )
    _write_phase0_cell_for_parity(
        tmp_path, "em", "src0", "tgt1", 14, refs[("em", "src0", "tgt1")]["c_Cp"], v0_tgt1
    )
    with pytest.raises(RuntimeError, match="V0 MISALIGNMENT"):
        pc.check_mean_parity(tmp_path, "em", 14, 1)


def test_parity_fires_on_v0_gross_drift(monkeypatch, tmp_path):
    """(c) A v0 below the 0.98 gross-drift floor FAILS LOUD (beyond the R band)."""
    import issue811_mean_parity_check as pc
    import pytest

    H = 3584
    rng = np.random.default_rng(3)
    c_cp = rng.standard_normal(H).astype(np.float64)
    v0_ref = rng.standard_normal(H).astype(np.float64)
    _stub_refs(monkeypatch, {("em", "src0", "tgt0"): {"v0": v0_ref, "c_Cp": c_cp}})
    # c_C faithful (a passes); v0 at cosine ~0.90 -> below the 0.98 gross-drift floor.
    # Single target so argmax (b) is vacuously OK; (c) is the check that fires.
    c_c_new = (c_cp + 1e-4 * rng.standard_normal(H)).astype(np.float32)
    v0_new = _at_cosine(v0_ref, 0.90, rng).astype(np.float32)
    _write_phase0_cell_for_parity(tmp_path, "em", "src0", "tgt0", 14, c_c_new, v0_new)
    with pytest.raises(RuntimeError, match="V0 GROSS DRIFT"):
        pc.check_mean_parity(tmp_path, "em", 14, 1)


def _at_cosine(ref: np.ndarray, target_cos: float, rng: np.random.Generator) -> np.ndarray:
    """Build a vector whose cosine to ``ref`` is ~``target_cos`` (numerically exact).

    v = cos * r_hat + sin * n_hat, with n_hat a unit vector orthogonal to ref, scaled
    to ref's norm — so cosine(v, ref) == target_cos regardless of ref magnitude.
    """
    r = np.asarray(ref, dtype=np.float64)
    r_norm = float(np.linalg.norm(r))
    r_hat = r / r_norm
    n = rng.standard_normal(r.shape[0])
    n = n - (n @ r_hat) * r_hat  # orthogonalize against ref
    n_hat = n / float(np.linalg.norm(n))
    sin = float(np.sqrt(max(0.0, 1.0 - target_cos**2)))
    return r_norm * (target_cos * r_hat + sin * n_hat)


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


def test_shard_oversize_jsonl_raises_on_single_oversize_line(monkeypatch, tmp_path):
    """A single JSONL row bigger than the shard target cannot be line-split — it
    RAISES with the row's identity (r10 Minor: shipping it as one oversize shard
    would force-route >10 MB to LFS). Constants monkeypatched small for speed."""
    import json as _json

    import issue811_upload_store as up
    import pytest

    monkeypatch.setattr(up, "TEXT_SHARD_BYTES", 100)
    monkeypatch.setattr(up, "TEXT_SHARD_TARGET", 80)
    path = tmp_path / "responses_em_src_a_tgt1.jsonl"
    giant = _json.dumps(
        {
            "behavior": "em",
            "source_cid": "src_a",
            "target_cid": "tgt1",
            "probe_idx": 0,
            "text": "x" * 200,
        }
    )
    path.write_text(giant + "\n")
    with pytest.raises(ValueError, match=r"single JSONL row .* cannot"):
        up._shard_oversize_jsonl(path)


def test_shard_oversize_jsonl_splits_normal_lines(monkeypatch, tmp_path):
    """Happy path unchanged: many small lines split into <target shards."""
    import issue811_upload_store as up

    monkeypatch.setattr(up, "TEXT_SHARD_BYTES", 100)
    monkeypatch.setattr(up, "TEXT_SHARD_TARGET", 80)
    path = tmp_path / "responses_em_src_a_tgt1.jsonl"
    path.write_text("".join(f'{{"probe_idx": {i}, "text": "abcdefgh"}}\n' for i in range(10)))
    shards = up._shard_oversize_jsonl(path)
    assert len(shards) > 1
    assert all(s.stat().st_size <= 80 for s in shards)
    # Every input line survives, in order, across the shards.
    joined = "".join(s.read_text() for s in shards)
    assert joined == path.read_text()


# ── 8. Phase-0 staging: completeness verify (round-6 crash-fix) ─────────────────


def _write_staged_cell(
    root: Path,
    behavior: str,
    source: str,
    targets: list[str],
    layer: int,
    *,
    declared_targets: list[str] | None = None,
):
    """Write a staged phase-0 cell dir: a ``{tgt}_L{layer}.npz`` per ``targets`` + a
    ``.done`` sentinel whose ``targets`` list is ``declared_targets`` (default: the
    npz written).

    Pass ``declared_targets`` LARGER than ``targets`` to simulate a PARTIAL HF
    recovery: the ``.done`` declares the full grid but only a subset of npz landed —
    the production verify must FAIL on the gap (round-6 concern
    phase0-stage-target-completeness-underverified). Pass ``declared_targets=[]`` /
    a non-list to simulate a corrupt/legacy sentinel.
    """
    d = root / behavior / f"{source}_seed42"
    d.mkdir(parents=True, exist_ok=True)
    for tgt in targets:
        np.savez(
            d / f"{tgt}_L{layer}.npz",
            c_C=np.zeros(3, dtype=np.float32),
            v0=np.zeros(3, dtype=np.float32),
            v0_turn_nl=np.zeros(3, dtype=np.float32),
            source_cid=np.asarray(source),
            target_cid=np.asarray(tgt),
            layer=np.asarray(layer),
        )
    declared = targets if declared_targets is None else declared_targets
    # Mirror the real .done written by issue811_phase0_extract.py (~line 296): a JSON
    # dict carrying the resolved 'targets' grid. The production verify reads THIS list.
    (d / ".done").write_text(json.dumps({"targets": declared, "layers": [layer]}))


def test_stage_verify_passes_on_complete_grid(tmp_path):
    """_verify_complete returns cleanly + a per-behavior digest when every resolved
    cell has its .done AND EVERY declared target npz present (production mode)."""
    import issue811_stage_phase0 as st

    out = tmp_path / "phase0"
    grid = {"em": ["src0", "src1"], "sycophancy": ["src0"]}
    for beh, srcs in grid.items():
        for s in srcs:
            _write_staged_cell(out, beh, s, ["tgt0", "tgt1", "tgt2"], 14)
    # None targets -> production mode: .done's declared targets == present npz per cell.
    digest = st._verify_complete(out, grid, 14, None)  # no raise
    # per-behavior (declared_total, present_total): 3 declared targets/cell, all present.
    assert digest == {"em": (6, 6), "sycophancy": (3, 3)}


def test_stage_verify_fails_loud_on_partial_recovery_naming_missing_npz(tmp_path):
    """PARTIAL HF recovery: .done declares [tgt0,tgt1,tgt2] but only tgt0_L14.npz
    landed -> production verify RAISES and NAMES the two missing npz by name (round-6
    concern phase0-stage-target-completeness-underverified). The weak '>=1 npz present'
    heuristic would have PASSed this and skipped the ~5.6h re-extraction."""
    import issue811_stage_phase0 as st
    import pytest

    out = tmp_path / "phase0"
    grid = {"em": ["src0"]}
    # .done declares 3 targets; only tgt0's npz is on disk (the partial-recovery case).
    _write_staged_cell(out, "em", "src0", ["tgt0"], 14, declared_targets=["tgt0", "tgt1", "tgt2"])
    with pytest.raises(RuntimeError) as ei:
        st._verify_complete(out, grid, 14, None)
    msg = str(ei.value)
    assert "INCOMPLETE" in msg
    assert "tgt1_L14.npz missing" in msg
    assert "tgt2_L14.npz missing" in msg
    assert "tgt0_L14.npz" not in msg  # tgt0 IS present — not flagged


def test_stage_verify_fails_loud_on_unparsable_done(tmp_path):
    """A cell with target npz but a non-JSON .done -> production verify raises."""
    import issue811_stage_phase0 as st
    import pytest

    out = tmp_path / "phase0"
    grid = {"em": ["src0"]}
    _write_staged_cell(out, "em", "src0", ["tgt0"], 14)
    (out / "em" / "src0_seed42" / ".done").write_text("not json {{")
    with pytest.raises(RuntimeError, match=r"unparsable \.done sentinel"):
        st._verify_complete(out, grid, 14, None)


def test_stage_verify_fails_loud_on_done_without_targets(tmp_path):
    """A .done lacking a 'targets' key -> production verify raises (can't trust it)."""
    import issue811_stage_phase0 as st
    import pytest

    out = tmp_path / "phase0"
    grid = {"em": ["src0"]}
    _write_staged_cell(out, "em", "src0", ["tgt0"], 14)
    (out / "em" / "src0_seed42" / ".done").write_text(json.dumps({"layers": [14]}))  # no targets
    with pytest.raises(RuntimeError, match=r"no non-empty 'targets' list"):
        st._verify_complete(out, grid, 14, None)


def test_stage_verify_fails_loud_on_missing_cell(tmp_path):
    """_verify_complete raises listing the missing (behavior, source) on a shortfall."""
    import issue811_stage_phase0 as st
    import pytest

    out = tmp_path / "phase0"
    grid = {"em": ["src0", "src1"]}
    _write_staged_cell(out, "em", "src0", ["tgt0", "tgt1"], 14)  # src1 absent
    with pytest.raises(RuntimeError, match=r"(?s)INCOMPLETE.*src1"):
        st._verify_complete(out, grid, 14, None)


def test_stage_verify_fails_loud_on_missing_done_sentinel(tmp_path):
    """A cell dir with target npz but NO .done (truncated/partial) fails loud."""
    import issue811_stage_phase0 as st
    import pytest

    out = tmp_path / "phase0"
    grid = {"em": ["src0"]}
    _write_staged_cell(out, "em", "src0", ["tgt0"], 14)
    (out / "em" / "src0_seed42" / ".done").unlink()  # simulate a truncated upload
    with pytest.raises(RuntimeError, match=r"(?s)INCOMPLETE.*no \.done sentinel"):
        st._verify_complete(out, grid, 14, None)


def test_stage_verify_smoke_requires_named_targets(tmp_path):
    """Smoke mode (--targets given) requires exactly the named target subset present."""
    import issue811_stage_phase0 as st
    import pytest

    out = tmp_path / "phase0"
    grid = {"em": ["src0"]}
    _write_staged_cell(out, "em", "src0", ["default", "sp_swe"], 14)  # sp_doctor missing
    with pytest.raises(RuntimeError, match=r"(?s)INCOMPLETE.*sp_doctor"):
        st._verify_complete(out, grid, 14, ["default", "sp_swe", "sp_doctor"])
    # And PASSES when the named subset is fully present.
    _write_staged_cell(out, "em", "src0", ["default", "sp_swe", "sp_doctor"], 14)
    st._verify_complete(out, grid, 14, ["default", "sp_swe", "sp_doctor"])  # no raise


def test_stage_sources_spec_parse():
    """_parse_sources_spec handles the dispatcher's behavior=srcs;... format + errors."""
    import issue811_stage_phase0 as st
    import pytest

    got = st._parse_sources_spec("em=binst_em,default;sycophancy=sp_swe;")
    assert got == {"em": ["binst_em", "default"], "sycophancy": ["sp_swe"]}
    with pytest.raises(ValueError, match="malformed"):
        st._parse_sources_spec("em binst_em")
    with pytest.raises(ValueError, match="empty grid"):
        st._parse_sources_spec(";;")
