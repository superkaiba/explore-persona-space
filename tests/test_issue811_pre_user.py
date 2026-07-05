"""#811 pre-user-boundary-summary round invariants (plan §4/§7/§13).

Covers, dependency-light (real Qwen tokenizer + tiny CPU stub where a forward is
needed; no network beyond the cached tokenizer, no GPU):

1. KILL-2 header-id asserts — startup ``assert_header_ids`` on the REAL
   tokenizer + the negative (corrupted-id) test; ``_extended_ids`` pre-append
   index math (plan §13 smoke 1).
2. Extended-vs-unextended pass equality at pre-append positions on the tiny
   stub (the A2 INDEX-bookkeeping half of §13 smoke 2 — the stub is per-token,
   so this pins the span arithmetic; real-model causality is architectural) +
   per-arm reduction semantics vs manual slicing.
3. Loader/gate summary-key registries carry the nine arms; ``_blob_to_record``
   reads an arm's keys (§13 smoke 3's key-selection half).
4. ``derive_alllayer_arms`` — arms 8/9 re-derive BIT-EXACTLY from the persisted
   fp16 stacks (§13 smoke 3's derivation half).
5. Per-arm KILL-1 gate (plan §7 / MF3): pass / fail / near-threshold ratios /
   ``comparator_indeterminate`` on <2 positive-mean-margin behaviors (NOT
   gate_pass false, never counts toward the STOP — §13 smoke 6) / the
   all-nine-DECIDED-fail STOP (§13 smoke 5).
6. Batched Gram/dual-space floor bootstrap == SEEDED serial oracle to fp
   tolerance, with matched skip accounting + draw-aligned per-draw stats
   (plan §4.3 item 10, §13 smoke 7).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import issue667_extract as ex  # noqa: E402
import issue722_bootstrap as boot  # noqa: E402
import issue722_fit_M as fitM  # noqa: E402
import issue722_load_activations as loadact  # noqa: E402
import issue811_fit as f811  # noqa: E402
from test_issue667_gate_chain import _TinyStub  # noqa: E402  (reuse the CPU stub)


def _tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")


# ── 1. KILL-2 header-id asserts + pre-append index math ──────────────────────


def test_header_ids_on_real_tokenizer():
    """The plan-time id verification holds in-process (startup assert, plan §7)."""
    tok = _tok()
    assert tok.encode(ex.HEADER_TEXT, add_special_tokens=False) == ex.HEADER_IDS
    ex.assert_header_ids(tok)  # must not raise


def test_assert_header_ids_fails_on_corrupted_ids():
    """Negative test (plan §13 smoke 1): a drifted tokenizer fails LOUD."""

    class _BadTok:
        def encode(self, _text, add_special_tokens=False):
            return [151644, 872, 199]  # corrupted trailing id

    with pytest.raises(RuntimeError, match=r"pre-user-assert.*expected \[151644, 872, 198\]"):
        ex.assert_header_ids(_BadTok())


def test_extended_ids_pre_append_math():
    """F = pre-append length; ext tail == HEADER_IDS; pre-append ids unchanged."""
    full_ids = [10, 11, 20, 21, ex.IM_END_ID, 198]
    ext_ids, F = ex._extended_ids(full_ids)
    assert F == len(full_ids) == 6
    assert ext_ids[:F] == full_ids
    assert ext_ids[F:] == ex.HEADER_IDS
    assert len(ext_ids) == F + 3


# ── 2. Extended-pass reader on the tiny stub (index bookkeeping, A2) ─────────


def test_mean_resp_acts_pre_user_arms_and_reference_invariance():
    """Pre-user arms ride the SAME forward; references match the unextended pass.

    The stub is a per-token map (no attention), so equality at pre-append
    positions pins the INDEX BOOKKEEPING (the load-bearing risk, plan §8 row 4);
    the real-model invariance is causal-attention architectural (A2). Also
    checks each arm's reduction against manual slicing of the extended acts.
    """
    tok = _tok()
    vocab = len(tok)
    torch.manual_seed(3)
    base = _TinyStub(vocab, hidden=8, n_layers=2)
    trained = _TinyStub(vocab, hidden=8, n_layers=2)
    msgs = [{"role": "system", "content": "You are X."}, {"role": "user", "content": "Hi?"}]
    resp = "Hello there."
    device = torch.device("cpu")
    summaries = ("mean", "turn_nl", "maxp", *ex.PRE_USER_LAYER_ARMS)
    out_ref = ex._mean_resp_acts(
        base, trained, tok, msgs, resp, [1], device, summaries=("mean", "turn_nl", "maxp")
    )
    out_ext = ex._mean_resp_acts(base, trained, tok, msgs, resp, [1], device, summaries=summaries)
    # (a) A2 index bookkeeping: pre-append reference summaries are UNCHANGED by
    # the header append (allclose per plan §13 smoke 2 — not bitwise).
    for ref in ("mean", "turn_nl", "maxp"):
        for leg in (0, 1):
            np.testing.assert_allclose(
                out_ext[1][ref][leg], out_ref[1][ref][leg], rtol=1e-3, atol=1e-6
            )
    # (b) all seven per-layer arms present + finite + correct shape.
    for arm in ex.PRE_USER_LAYER_ARMS:
        v0, vp = out_ext[1][arm]
        assert v0.shape == (8,) and vp.shape == (8,)
        assert np.isfinite(v0).all() and np.isfinite(vp).all()
    # (c) reduction semantics vs manual slicing of the extended pass.
    prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full_msgs = [*msgs, {"role": "assistant", "content": resp}]
    full_text = tok.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
    p = len(tok.encode(prompt_text, add_special_tokens=False))
    full_ids = tok.encode(full_text, add_special_tokens=False)
    ext_ids, F = ex._extended_ids(full_ids)
    ids = torch.tensor([ext_ids])
    acts = ex.extract_layer_activations(base, ids, [0, 1])
    span_hdr = acts[1][0, F : F + 3, :].float()
    span_all = acts[1][0, p : F + 3, :].float()
    np.testing.assert_allclose(
        out_ext[1]["pre_user_imstart"][0], acts[1][0, F, :].float().numpy(), rtol=1e-5
    )
    np.testing.assert_allclose(
        out_ext[1]["pre_user_nl"][0], acts[1][0, F + 2, :].float().numpy(), rtol=1e-5
    )
    np.testing.assert_allclose(
        out_ext[1]["pre_user_mean3"][0], span_hdr.mean(dim=0).numpy(), rtol=1e-5
    )
    np.testing.assert_allclose(
        out_ext[1]["pre_user_max3"][0], span_hdr.max(dim=0).values.numpy(), rtol=1e-5
    )
    np.testing.assert_allclose(
        out_ext[1]["ans_mean_incl_hdr"][0], span_all.mean(dim=0).numpy(), rtol=1e-5
    )
    np.testing.assert_allclose(
        out_ext[1]["ans_max_incl_hdr"][0], span_all.max(dim=0).values.numpy(), rtol=1e-5
    )
    # (d) the (n_layers, H) arm-6/7 stacks ride the same result dict.
    stacks = out_ext["stacks"]
    for base_name in ex.PRE_USER_STACK_BASES:
        hb, ht = stacks[base_name]
        assert hb.shape == (2, 8) and ht.shape == (2, 8)
    np.testing.assert_allclose(
        stacks["ans_mean_incl_hdr"][0][1], span_all.mean(dim=0).numpy(), rtol=1e-5
    )


# ── 3. Summary-key registries ─────────────────────────────────────────────────


def test_summary_key_registries_carry_all_nine_arms():
    for arm in f811.PRE_USER_TEST_SUMMARIES:
        assert loadact._SUMMARY_ANSWER_KEYS[arm] == (f"v0_{arm}", f"v_plus_{arm}")
        assert f811.PHASE0_SUMMARY_KEYS[arm] == f"v0_{arm}"
    assert len(f811.PRE_USER_TEST_SUMMARIES) == 9


def test_blob_to_record_reads_pre_user_arm_keys():
    h = loadact.HIDDEN
    rng = np.random.default_rng(0)
    arm = "pre_user_mean3"
    blob = {
        "c_C": rng.standard_normal(h).astype(np.float32),
        "c_C_postft": rng.standard_normal(h).astype(np.float32),
        f"v0_{arm}": rng.standard_normal(h).astype(np.float32),
        f"v_plus_{arm}": rng.standard_normal(h).astype(np.float32),
        "behavior": np.asarray("em"),
        "source_cid": np.asarray("default"),
        "target_cid": np.asarray("sp_swe"),
        "layer": np.asarray(14),
    }
    rec = loadact._blob_to_record(blob, "rel", "em", 14, summary=arm)
    np.testing.assert_allclose(rec.v0, blob[f"v0_{arm}"].astype(np.float64))
    with pytest.raises(KeyError, match="v0_pre_user_max3"):
        loadact._blob_to_record(blob, "rel", "em", 14, summary="pre_user_max3")


# ── 4. Arm-8/9 derivation from the persisted fp16 stacks ─────────────────────


def test_derive_alllayer_arms_bit_exact_from_fp16_stacks():
    rng = np.random.default_rng(7)
    s_mean = rng.standard_normal((28, 16)).astype(np.float16)
    s_max = rng.standard_normal((28, 16)).astype(np.float16)
    d = ex.derive_alllayer_arms(s_mean, s_max)
    # Re-derivation from the SAME persisted fp16 arrays is bit-exact (§13 smoke 3).
    np.testing.assert_array_equal(
        d["ans_mean_incl_hdr_alllayer"], s_mean.astype(np.float32).mean(axis=0)
    )
    np.testing.assert_array_equal(
        d["ans_max_incl_hdr_alllayer"], s_max.astype(np.float32).max(axis=0)
    )
    assert d["ans_mean_incl_hdr_alllayer"].dtype == np.float32
    with pytest.raises(AssertionError):
        ex.derive_alllayer_arms(s_mean.astype(np.float32), s_max)  # non-fp16 input


# ── 5. Per-arm KILL-1 gate (plan §7 / MF3) ────────────────────────────────────

ARMS = f811.PRE_USER_TEST_SUMMARIES


def _cbs(mean_margins: dict, arm_margins: dict[str, dict]) -> dict:
    """cells_by_summary fixture: {summary: {(beh, L14, summary): {gate_margin}}}."""
    out = {"mean": {}}
    for beh, m in mean_margins.items():
        out["mean"][(beh, f811.PRIMARY_LAYER, "mean")] = {"gate_margin": m}
    for arm, per_beh in arm_margins.items():
        out[arm] = {
            (beh, f811.PRIMARY_LAYER, arm): {"gate_margin": v} for beh, v in per_beh.items()
        }
    return out


def test_per_arm_gate_pass_fail_and_ratios():
    mean = {"em": 0.4, "sycophancy": 0.4, "fact": 0.4}
    arms = {
        "pre_user_imstart": {"em": 0.3, "sycophancy": 0.3, "fact": 0.05},  # 2/3 held -> pass
        "pre_user_user": {"em": 0.05, "sycophancy": 0.05, "fact": 0.3},  # 1/3 held -> fail
        "pre_user_nl": {"em": 0.21, "sycophancy": 0.05, "fact": 0.05},  # near-threshold + fail
    }
    d = f811._per_arm_gate_decision(_cbs(mean, arms), tuple(arms))
    assert d["per_arm"]["pre_user_imstart"]["gate_status"] == "pass"
    assert d["per_arm"]["pre_user_imstart"]["gate_pass"] is True
    assert d["per_arm"]["pre_user_user"]["gate_status"] == "fail"
    assert d["per_arm"]["pre_user_user"]["gate_pass"] is False
    # Continuous-margin narration: ratio 0.21/0.4 = 0.525 -> near_threshold True.
    em_entry = d["per_arm"]["pre_user_nl"]["per_behavior"]["em"]
    assert em_entry["margin_ratio"] == pytest.approx(0.525)
    assert em_entry["near_threshold"] is True and em_entry["status"] == "held"
    assert d["trusted_arms"] == ["pre_user_imstart"]
    assert d["stop_fired"] is False
    assert d["state"] == "decided"


def test_per_arm_gate_comparator_indeterminate_never_counts_toward_stop():
    """MF3 / §13 smoke 6: 2 of 3 behaviors have mean_margin <= 0 -> EVERY arm is
    comparator_indeterminate (NOT gate_pass false) and the STOP does NOT fire."""
    mean = {"em": -0.1, "sycophancy": 0.0, "fact": 0.4}  # only fact positive
    arms = {a: {"em": 0.0, "sycophancy": 0.0, "fact": 0.0} for a in ARMS}
    d = f811._per_arm_gate_decision(_cbs(mean, arms), ARMS)
    for a in ARMS:
        assert d["per_arm"][a]["gate_status"] == "comparator_indeterminate"
        assert d["per_arm"][a]["gate_pass"] is None
    assert d["n_fail"] == 0 and d["n_indeterminate"] == len(ARMS)
    assert d["stop_fired"] is False  # indeterminate NEVER counts toward the STOP
    assert d["state"] == "decided"  # store populated — NOT empty_store


def test_per_arm_gate_all_nine_decided_failures_fires_stop():
    mean = {"em": 0.4, "sycophancy": 0.4, "fact": 0.4}
    arms = {a: {"em": 0.0, "sycophancy": 0.0, "fact": 0.0} for a in ARMS}
    d = f811._per_arm_gate_decision(_cbs(mean, arms), ARMS)
    assert d["n_fail"] == len(ARMS) and d["stop_fired"] is True
    assert d["trusted_arms"] == []


def test_per_arm_gate_indeterminate_arm_blocks_all_nine_stop():
    """8 decided fails + 1 indeterminate is NOT an all-nine STOP (plan §7)."""
    mean = {"em": 0.4, "sycophancy": 0.4, "fact": 0.4}
    arms = {a: {"em": 0.0, "sycophancy": 0.0, "fact": 0.0} for a in ARMS}
    arms["ans_max_incl_hdr_alllayer"] = {"em": None, "sycophancy": None, "fact": 0.0}
    d = f811._per_arm_gate_decision(_cbs(mean, arms), ARMS)
    assert d["n_fail"] == len(ARMS) - 1 and d["n_indeterminate"] == 1
    assert d["stop_fired"] is False


def test_per_arm_gate_empty_store_state():
    mean = {"em": None, "sycophancy": None, "fact": None}
    arms = {a: {"em": None, "sycophancy": None, "fact": None} for a in ARMS}
    d = f811._per_arm_gate_decision(_cbs(mean, arms), ARMS)
    assert d["state"] == "empty_store"


def _gate_args(tmp_path, smoke=False):
    import argparse

    return argparse.Namespace(out_dir=tmp_path / "cells", smoke=smoke)


def test_run_phase0_gate_multi_arm_writes_json_and_rcs(tmp_path):
    """The multi-arm runner: rc 0 + validity_gate_phase0.json on a mixed verdict;
    rc 3 ONLY on all-nine DECIDED failures; RAISE on empty store (production)."""
    import json as _json

    mean = {"em": 0.4, "sycophancy": 0.4, "fact": 0.4}
    arms_fail = {a: {"em": 0.0, "sycophancy": 0.0, "fact": 0.0} for a in ARMS}
    # mixed: one arm passes -> rc 0
    arms_mixed = dict(arms_fail)
    arms_mixed["pre_user_imstart"] = {"em": 0.3, "sycophancy": 0.3, "fact": 0.3}
    rc = f811._run_phase0_gate_multi_arm(
        _gate_args(tmp_path), _cbs(mean, arms_mixed), ARMS, f811.PRIMARY_LAYER
    )
    assert rc == 0
    rec = _json.loads((tmp_path / "validity_gate_phase0.json").read_text())
    assert rec["trusted_arms"] == ["pre_user_imstart"] and rec["n_fail"] == 8
    # all nine decided failures -> rc 3 (STOP before the paired spend)
    rc = f811._run_phase0_gate_multi_arm(
        _gate_args(tmp_path), _cbs(mean, arms_fail), ARMS, f811.PRIMARY_LAYER
    )
    assert rc == 3
    # empty store -> RAISE on a production (non-smoke) run; tolerated under smoke
    empty = _cbs(
        {b: None for b in ("em", "sycophancy", "fact")},
        {a: {"em": None, "sycophancy": None, "fact": None} for a in ARMS},
    )
    with pytest.raises(RuntimeError, match="empty base-leg store"):
        f811._run_phase0_gate_multi_arm(_gate_args(tmp_path), empty, ARMS, f811.PRIMARY_LAYER)
    assert (
        f811._run_phase0_gate_multi_arm(
            _gate_args(tmp_path, smoke=True), empty, ARMS, f811.PRIMARY_LAYER
        )
        == 0
    )


def test_phase0_base_loader_reads_pre_user_arm_keys(tmp_path):
    """_load_phase0_base_cells reads the nine v0_<slug> keys (plan §4.3 item 2)."""
    h = loadact.HIDDEN
    rng = np.random.default_rng(1)
    cell_dir = tmp_path / "em" / "default_seed42"
    cell_dir.mkdir(parents=True)
    payload = {
        "c_C": rng.standard_normal(h).astype(np.float32),
        "v0": rng.standard_normal(h).astype(np.float32),
        "behavior": np.asarray("em"),
        "source_cid": np.asarray("default"),
        "target_cid": np.asarray("sp_swe"),
        "layer": np.asarray(14),
    }
    for arm in ARMS:
        payload[f"v0_{arm}"] = rng.standard_normal(h).astype(np.float32)
    np.savez(cell_dir / "sp_swe_L14.npz", **payload)
    out = f811._load_phase0_base_cells(
        ("em",),
        ("mean", "pre_user_mean3", "ans_max_incl_hdr_alllayer"),
        14,
        local_root=str(tmp_path),
        max_sources=None,
        max_targets_per_source=None,
        strict=False,
    )
    np.testing.assert_allclose(
        out[("em", "pre_user_mean3")]["V0"][0], payload["v0_pre_user_mean3"].astype(np.float64)
    )
    np.testing.assert_allclose(
        out[("em", "ans_max_incl_hdr_alllayer")]["V0"][0],
        payload["v0_ans_max_incl_hdr_alllayer"].astype(np.float64),
    )


# ── 6. Batched floor bootstrap == seeded serial oracle (§13 smoke 7) ─────────


def _floor_fixture(seed=0, n=24, h=32, n_fam=3):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, h))
    Y = rng.standard_normal((n, h))
    grid = X.copy()
    r_hat = rng.standard_normal(h)
    r_hat /= np.linalg.norm(r_hat)
    fams = [f"fam{i % n_fam}" for i in range(n)]
    return X, Y, grid, r_hat, fams


def test_batched_floor_bootstrap_matches_seeded_serial_oracle(monkeypatch):
    """plan §4.3 item 10 equivalence gate: identical per-draw floors to fp
    tolerance + matched skip/fallback accounting, at the SAME seed."""
    import issue658_fit_predictors as fit658

    monkeypatch.setattr(fit658, "DEVICE", "cpu")
    monkeypatch.setattr(fitM, "TARGET_DIM", 4)
    X, Y, grid, r_hat, fams = _floor_fixture()
    sc_s, pd_s = {}, {}
    serial = boot.make_refit_pair(
        X,
        Y,
        fitM._refit_ridge_fn(grid),
        grid,
        r_hat,
        fams,
        n_pairs=6,
        seed=0,
        skip_counter=sc_s,
        per_draw_out=pd_s,
    )
    sc_b, pd_b = {}, {}
    fb_before = fitM.GESVD_FALLBACK_COUNTER["n"]
    batched = boot.make_refit_pair(
        X,
        Y,
        None,
        grid,
        r_hat,
        fams,
        n_pairs=6,
        seed=0,
        skip_counter=sc_b,
        batched_chain_fn=fitM.make_batched_refit_chain_fn(X, Y, grid, r_hat),
        per_draw_out=pd_b,
    )
    np.testing.assert_allclose(pd_b["stats"], pd_s["stats"], rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(batched, serial, rtol=1e-7, atol=1e-10)
    assert sc_b == sc_s == {"n_attempted": 6, "n_skipped": 0}
    assert fitM.GESVD_FALLBACK_COUNTER["n"] == fb_before  # clean path: 0 fallbacks
    # Draw alignment: per_draw is n_pairs long, NaN-free on the clean path.
    assert pd_b["stats"].shape == (6,) and np.isfinite(pd_b["stats"]).all()


def test_batched_floor_draws_aligned_across_summaries(monkeypatch):
    """Same seed + same families => the resample stream (and thus the draw index)
    is shared across different Y 'summaries' — the §6 selection-null escape's
    alignment premise. Proxy check: two batched runs at the same seed produce
    per-draw arrays of the same shape with no skips, and IDENTICAL results when
    Y is identical."""
    import issue658_fit_predictors as fit658

    monkeypatch.setattr(fit658, "DEVICE", "cpu")
    monkeypatch.setattr(fitM, "TARGET_DIM", 4)
    X, Y, grid, r_hat, fams = _floor_fixture()
    Y2 = np.roll(Y, 1, axis=1)  # a "different summary" of the same cells
    pd_1, pd_2, pd_1b = {}, {}, {}
    for Yk, pd in ((Y, pd_1), (Y2, pd_2), (Y, pd_1b)):
        boot.make_refit_pair(
            X,
            Yk,
            None,
            grid,
            r_hat,
            fams,
            n_pairs=5,
            seed=0,
            batched_chain_fn=fitM.make_batched_refit_chain_fn(X, Yk, grid, r_hat),
            per_draw_out=pd,
        )
    np.testing.assert_array_equal(pd_1["stats"], pd_1b["stats"])  # deterministic
    assert pd_1["stats"].shape == pd_2["stats"].shape == (5,)
    assert not np.allclose(pd_1["stats"], pd_2["stats"])  # different Y, same draws


def test_batched_floor_rank_deficient_resample_matches_serial_oracle(monkeypatch):
    """r12 Minor (oracle/twin truncation divergence): a SILENTLY rank-deficient
    resample — rank < TARGET_DIM with NO LinAlgError, so the exception-only
    fallback never fired — must not diverge between the dual twin (which drops
    ≤1e-12-relative eigen-directions) and the serial oracle (which keeps
    min(dim, rows) SVD rows). The batched path now ROUTES such draws to the
    EXACT-serial fallback: per-draw floors match to fp tolerance, ZERO skips,
    and the fallback counter records every routed resample."""
    import issue658_fit_predictors as fit658

    monkeypatch.setattr(fit658, "DEVICE", "cpu")
    monkeypatch.setattr(fitM, "TARGET_DIM", 4)
    rng = np.random.default_rng(3)
    n, h = 24, 32
    X = rng.standard_normal((n, h))
    # THREE distinct rows repeated -> centered rank <= 2 < TARGET_DIM=4 on EVERY
    # family resample (all draws stay inside the 3-row span), yet gesdd/eigh
    # converge fine — exactly the silent (exception-free) rank-deficiency window.
    Y = np.tile(rng.standard_normal((3, h)), (8, 1))
    grid = X.copy()
    r_hat = rng.standard_normal(h)
    r_hat /= np.linalg.norm(r_hat)
    fams = [f"fam{i % 3}" for i in range(n)]
    sc_s, pd_s = {}, {}
    serial = boot.make_refit_pair(
        X,
        Y,
        fitM._refit_ridge_fn(grid),
        grid,
        r_hat,
        fams,
        n_pairs=4,
        seed=0,
        skip_counter=sc_s,
        per_draw_out=pd_s,
    )
    sc_b, pd_b = {}, {}
    fb_before = fitM.GESVD_FALLBACK_COUNTER["n"]
    batched = boot.make_refit_pair(
        X,
        Y,
        None,
        grid,
        r_hat,
        fams,
        n_pairs=4,
        seed=0,
        skip_counter=sc_b,
        batched_chain_fn=fitM.make_batched_refit_chain_fn(X, Y, grid, r_hat),
        per_draw_out=pd_b,
    )
    np.testing.assert_allclose(pd_b["stats"], pd_s["stats"], rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(batched, serial, rtol=1e-7, atol=1e-10)
    assert sc_b == sc_s == {"n_attempted": 4, "n_skipped": 0}
    # Every resample (2 per pair x 4 pairs) is rank-deficient -> routed through
    # the exact-serial fallback, each ride counted once.
    assert fitM.GESVD_FALLBACK_COUNTER["n"] == fb_before + 8
    assert pd_b["stats"].shape == (4,) and np.isfinite(pd_b["stats"]).all()


# ── 7. Batched ridge-LOCO == serial oracle (#811 vectorize fix round 2) ───────


def _loco_fixture(seed=0, n=26, d=48, p=5):
    """Small real-rank-structured (X, Y) mirroring `_assert_ridge_exactness`."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 3))
    W = rng.standard_normal((3, d))
    X = z @ W + 0.1 * rng.standard_normal((n, d))
    B = rng.standard_normal((d, p))
    Y = X @ B * 0.05 + 0.1 * rng.standard_normal((n, p))
    return X, Y


def test_batched_ridge_loco_matches_serial_oracle(monkeypatch):
    """The chunked batched LOCO twin reproduces the serial per-fold PRESS path
    fold-for-fold (same λ grid, same PRESS inner-LOO selection, same dual
    solve) to fp tolerance — the fix-round-2 equivalence gate. Also pins
    chunk-size invariance (chunking must not change results)."""
    import issue658_fit_predictors as fit658

    monkeypatch.setattr(fit658, "DEVICE", "cpu")
    X, Y = _loco_fixture()
    serial = fit658._ridge_predict_loco(X, Y, fit658.RIDGE_LAMBDAS)
    batched = fit658._ridge_predict_loco_batched(X, Y, fit658.RIDGE_LAMBDAS)
    np.testing.assert_allclose(batched, serial, rtol=1e-7, atol=1e-10)
    chunked = fit658._ridge_predict_loco_batched(X, Y, fit658.RIDGE_LAMBDAS, chunk=5)
    np.testing.assert_allclose(chunked, batched, rtol=1e-12, atol=1e-14)


def test_ridge_loco_dispatch_default_batched_and_serial_tombstone(monkeypatch):
    """`fitM._ridge_loco_pred` (the function `fit_cell` dispatches for the
    chain-rho / shuffle / cross-transfer reads) defaults to the batched twin;
    the serial path FutureWarns and is forbidden under EPM_FORBID_SERIAL_FITS=1
    (the Supersede contract, .claude/rules/vectorize-many-cell-fits.md)."""
    import issue658_fit_predictors as fit658

    monkeypatch.setattr(fit658, "DEVICE", "cpu")
    X, Y = _loco_fixture(seed=1, n=12, d=16, p=3)
    via_dispatch = fitM._ridge_loco_pred(X, Y)
    direct = fit658._ridge_predict_loco_batched(X, Y, fit658.RIDGE_LAMBDAS)
    np.testing.assert_array_equal(via_dispatch, direct)
    monkeypatch.delenv("EPM_FORBID_SERIAL_FITS", raising=False)
    with pytest.warns(FutureWarning, match="SERIAL per-fold ridge-LOCO"):
        serial = fitM._ridge_loco_pred(X, Y, path="serial")
    np.testing.assert_allclose(serial, via_dispatch, rtol=1e-7, atol=1e-10)
    monkeypatch.setenv("EPM_FORBID_SERIAL_FITS", "1")
    with pytest.raises(RuntimeError, match="EPM_FORBID_SERIAL_FITS"):
        fitM._ridge_loco_pred(X, Y, path="serial")


def test_cached_cell_state_tristate(tmp_path):
    """Per-unit checkpoint classifier: merged / ridge_only / invalid — the
    resume predicate behind the immediate post-`fit_cell` checkpoint write."""
    ridge = {k: 1.0 for k in fitM._CELL_SCHEMA_KEYS}
    ridge["summary"] = "mean"
    p_ridge = tmp_path / "em_L14_mean.json"
    p_ridge.write_text(json.dumps(ridge))
    assert f811._cached_cell_state(p_ridge) == "ridge_only"
    assert not f811._cached_cell_valid(p_ridge)  # legacy API: merged-only
    merged = dict(ridge, mlp_validity_gate={"gate_margin": 0.1})
    p_merged = tmp_path / "em_L14_turn_nl.json"
    p_merged.write_text(json.dumps(merged))
    assert f811._cached_cell_state(p_merged) == "merged"
    assert f811._cached_cell_valid(p_merged)
    p_bad = tmp_path / "em_L14_maxp.json"
    p_bad.write_text('{"Delta_med": 1.0, "trunc')  # truncated mid-write
    assert f811._cached_cell_state(p_bad) == "invalid"
    p_missing = tmp_path / "em_L14_pre_user_nl.json"
    p_missing.write_text(json.dumps({"Delta_med": 1.0, "summary": "pre_user_nl"}))
    assert f811._cached_cell_state(p_missing) == "invalid"


def test_atomic_write_json_replaces_not_truncates(tmp_path):
    """The checkpoint write goes tmp + os.replace — a pre-existing file is only
    ever replaced by a COMPLETE new JSON (no in-place truncation window)."""
    p = tmp_path / "cell.json"
    p.write_text('{"old": true}')
    f811._atomic_write_json(p, {"new": 1})
    assert json.loads(p.read_text()) == {"new": 1}
    assert not (tmp_path / "cell.json.tmp").exists()
