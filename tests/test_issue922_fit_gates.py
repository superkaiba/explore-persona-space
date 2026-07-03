# ruff: noqa: E402
"""Issue #922 round-2 regression pins.

Pins the three permanent invariants the r1 code-review blockers added
(CI-pinned per code-reviewer.md Step 4.5 / Rule 13):

1. B1 (Claude CRITICAL): the fit-phase GPU budget must include the Gram
   ASSEMBLY footprint — the pre-fix budget was ``store + 14 GiB`` flat, which
   under-counts the ~32.9 GB assembly at (rows=29, H=3584) and OOMs the
   A100-80 fit phase.
2. B3 (Codex MAJOR): the capture shard-resume predicate must REFUSE to skip a
   shard whose regime (context-id set / window / row labels / hidden dim /
   dtype) does not match the current run — a prior --smoke shard at the same
   path must be recaptured, never silently reused.
3. B4 (Codex MAJOR): --blocks 'emb,...' must resolve the EMBEDDING row 0 into
   the fitted row set (the layer-0 anchor was silently dropped in r1).
4. B2 (Claude MAJOR, pinned r3): DV3 ``restricted_panel`` recomputes the
   rolled/frozen (+ v6 mode) companions on the SAME captured-subset unit
   panel as the true-answer ceiling — never carries full-panel numbers into
   the cross-panel HERO-3 comparison.

Plus pytest wrappers over the --verify-fits equivalence gates (r1 review
Minor: nothing in CI exercised the new src/ library gates).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from explore_persona_space.experiments.issue_922 import maps922 as M

# ── B1: budget includes the Gram-assembly footprint ───────────────────────────


def test_budget_includes_gram_footprint():
    store = 50 * (1 << 30)
    h_dim = 3584
    need_29, grams_29 = M.fit_phase_gpu_budget_bytes(store, h_dim, 29)
    # the answer-pass assembly at 29 rows is 11·H²·8·29 ≈ 32.8 GB — the term
    # the r1 budget omitted entirely (it used a flat 14 GiB headroom).
    assembly_29 = 11 * h_dim * h_dim * 8 * 29
    assert assembly_29 > 30e9
    assert need_29 >= store + assembly_29
    assert grams_29 >= assembly_29
    # the r1 flat constant is provably insufficient at production shape
    assert grams_29 > 14 * (1 << 30)
    # monotone in row_chunk; the row-chunked default (8) is far cheaper
    need_8, grams_8 = M.fit_phase_gpu_budget_bytes(store, h_dim, 8)
    assert grams_8 < grams_29
    assert need_8 >= store + 11 * h_dim * h_dim * 8 * 8


# ── B3: shard-resume regime validation ────────────────────────────────────────


def _mk_shard(path: Path, cis, *, wp=8, wa=40, labels=("emb", "0"), hidden=16, corpus="lmsys"):
    npos = 6
    contexts = {
        int(ci): {
            "h": torch.zeros(npos, len(labels), hidden, dtype=torch.float16),
            "token_ids": torch.zeros(npos, dtype=torch.int32),
            "segments": np.zeros(npos - 1, dtype=np.uint8),
            "prompt_len": 3,
            "ans_len": 2,
            "window_start": 0,
        }
        for ci in cis
    }
    torch.save(
        {
            "corpus": corpus,
            "blocks": list(labels),
            "contexts": contexts,
            "window": {"wp": wp, "wa": wa},
        },
        path,
    )


def test_shard_resume_refuses_wrong_regime(tmp_path):
    from issue922_capture_positions import validate_shard

    p = tmp_path / "shard_000.pt"
    _mk_shard(p, range(20))  # a prior --smoke shard: 20 contexts

    common = dict(corpus="lmsys", wp=8, wa=40, labels=["emb", "0"], expected_hidden=16)
    # the Codex incident case: production expects contexts 0..499 at this path
    blob, why = validate_shard(p, expected_cis=set(range(500)), **common)
    assert blob is None and "context-id set mismatch" in why
    # window-regime mismatch
    blob, why = validate_shard(p, expected_cis=set(range(20)), **{**common, "wa": 32})
    assert blob is None and "window" in why
    # row-label (layer count) mismatch — a stub-model shard under a prod run
    blob, why = validate_shard(
        p,
        expected_cis=set(range(20)),
        **{**common, "labels": ["emb"] + [str(i) for i in range(28)]},
    )
    assert blob is None and "row labels" in why
    # hidden-dim mismatch
    blob, why = validate_shard(
        p, expected_cis=set(range(20)), **{**common, "expected_hidden": 3584}
    )
    assert blob is None and "h invalid" in why
    # matching regime → skip is allowed and the blob is returned for reuse
    blob, why = validate_shard(p, expected_cis=set(range(20)), **common)
    assert blob is not None and why == "ok"
    # unloadable file → recapture, never crash
    bad = tmp_path / "shard_001.pt"
    bad.write_bytes(b"not a torch file")
    blob, why = validate_shard(bad, expected_cis=set(range(20)), **common)
    assert blob is None and "unloadable" in why


# ── B4: the embedding row 0 is IN the fitted row set ──────────────────────────


def test_blocks_parse_includes_emb_row0():
    from issue922_fit_maps import resolve_conditioned_rows, resolve_fit_rows

    assert resolve_fit_rows(["emb", 20], 29) == [0, 21]  # the r1 code dropped the 0
    assert resolve_fit_rows(["emb", *range(28)], 29) == list(range(29))
    # conditioned subset: 'emb' + blocks, restricted to fitted rows
    assert resolve_conditioned_rows("emb,5,10,14,17,19,20,24,26", list(range(29))) == [
        0,
        6,
        11,
        15,
        18,
        20,
        21,
        25,
        27,
    ]
    assert resolve_conditioned_rows("emb,20", [0, 21]) == [0, 21]


# ── B2: restricted_panel recomputes companions on the SAME unit panel ─────────


def test_restricted_panel_same_panel_companions():
    """The captured-subset companions are SAME-panel reads (the r1 B2 fix).

    Builds a 16-unit DV3 matrix (4 conditions x 4 questions) whose eval store
    only captures questions {0, 1, 2} (12 overlap units). The companion vector
    is crafted so the restricted panel gives a perfect within-condition r
    (= 1.0) while the FULL panel does not — so a regression that carries
    full-panel numbers into ``restricted_panel``'s output flips the assert.
    """
    from types import SimpleNamespace

    from issue922_eval import method_metrics, restricted_panel

    n_pos, prompt_len, hidden = 6, 3, 16
    cond_ids = ["c0_sys", "c1_sys", "c0_ms", "c1_ms"]
    mode_of = {"c0_sys": "system", "c1_sys": "system", "c0_ms": "many_shot", "c1_ms": "many_shot"}
    # eval store: qi {0,1,2} captured per condition (12 contexts)
    meta = {}
    ii = 0
    for cid in cond_ids:
        for qi in (0, 1, 2):
            meta[ii] = {"trait": "evil", "cond_id": cid, "qi": qi}
            ii += 1
    n_ev = ii
    g = torch.Generator().manual_seed(0)
    es = {
        "meta": meta,
        "ctx_ids": list(range(n_ev)),
        "h": torch.randn(2, n_ev * n_pos, hidden, generator=g),  # row 1 = block 0
        "pos_lo": np.arange(n_ev) * n_pos,
        "prompt_len": np.full(n_ev, prompt_len),
        "window_start": np.zeros(n_ev, dtype=int),
        "n_pos": np.full(n_ev, n_pos),
    }
    # DV3 matrix: 4 questions per condition; qi=3 exists ONLY in the matrix
    y, cond, mode, qis = [], [], [], []
    vec = []
    for ci_i, cid in enumerate(cond_ids):
        y.extend([0.0, 2.0, 4.0, 6.0])  # per-condition y std >= 1 (the PV floor)
        # perfect within-condition r on qi {0,1,2}; qi=3 wrecks the full panel
        vec.extend([0.0, 1.0, 2.0, -50.0])
        cond.extend([ci_i] * 4)
        mode.extend([mode_of[cid]] * 4)
        qis.extend([0, 1, 2, 3])
    mat = {
        "y": np.array(y),
        "cond": np.array(cond),
        "mode": np.array(mode, dtype=object),
        "qi": np.array(qis),
        "cond_ids": cond_ids,
    }
    vec = np.array(vec)
    r_b = np.random.default_rng(0).standard_normal((1, hidden))
    args = SimpleNamespace(n_boot=50, readout_block_override=0)
    res = restricted_panel("evil", mat, r_b, es, args, reads_vectors={"rolled_horizon_mean": vec})
    assert res["n_units"] == 12 and res["block"] == 0
    assert res["n_with_answer_window"] == 12
    assert "true_answer_ceiling_horizon_mean" in res
    for m in ("system", "many_shot"):
        # SAME-panel read: qi {0,1,2} only -> perfect per-condition correlation
        assert np.isclose(res["rolled_horizon_mean"][m]["point"], 1.0), res["rolled_horizon_mean"]
        # the full-panel number is far from 1.0 — carrying it over would fail above
        full = method_metrics(vec, mat, n_boot=50, seed=0)
        assert abs(full[m]["point"] - 1.0) > 0.5, full


# ── the --verify-fits equivalence gates, CI-pinned (r1 review Minor) ──────────


def test_ridge_gcv_gate():
    res = M.verify_ridge_gcv_against_dual()
    assert res["max_abs_delta"] <= res["tol"]


def test_conditioned_forms_gate():
    res = M.verify_conditioned_forms()
    assert all(v["max_abs_delta"] <= 1e-4 for k, v in res.items() if k != "capacity_table")
    assert res["capacity_table"] == "exact"


def test_b1_gram_assembly_gate():
    M.verify_b1_gram_assembly()


def test_direct_horizon_gcv_gate():
    res = M.verify_direct_horizon_gcv()
    assert res["n_horizons"] >= 1


def test_split_mlp_gate():
    from explore_persona_space.analysis.vectorized_mlp_skill import (
        assert_split_mlp_matches_serial,
    )

    assert_split_mlp_matches_serial()
