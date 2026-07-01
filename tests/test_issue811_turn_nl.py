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
