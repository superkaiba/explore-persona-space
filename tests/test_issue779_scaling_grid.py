"""Regression tests for the issue #779 (training-source-ablation-hg) round-2/3 fixes.

Each test pins a permanent invariant a round-2/3 BLOCKER/MINOR fix installed and
FAILS against the pre-fix code (documented per test):

  - BLOCKER 1 (pv_pinv standardization space-mismatch): pre-fix the pv_pinv read
    scored ``raw c_last @ w_pinv`` while W was fit on standardized input, making
    it ~orthogonal to the intended M⁺r_B on heteroscedastic activations
    (corr ≈ -0.03). Post-fix it reads ``((c_last - xmu)/xsd) @ w_pinv``, which is
    the exact standardized-space M⁺r_B direction.
  - BLOCKER r3 (pv-pinv-transposed-orientation): pre-fix (round 2) the preimage
    was computed as ``(Vt.T*s_inv)@(U.T@r_B)`` — the pseudoinverse of the
    TRANSPOSED map, which violates the defining property ``w_pinv @ W ≈ r_B``
    (residual ≈47 on the review toy vs ~1e-14 correct) and correlates only ≈0.78
    with the correct read. Post-fix: ``w_pinv = (U*s_inv)@(Vt@r_B)`` — the tests
    pin the PREIMAGE PROPERTY on a planted-preimage fixture (never mirror the
    implementation formula) plus the orthogonal-map sigma-vs-1/sigma duality.
  - BLOCKER r3 (multigpu-no-upload-terminal-sentinel): pre-fix a ``--no-upload``
    trait-shard worker still wrote the terminal ``epm:results`` sentinel +
    ``[phase=done]`` after SKIPPING upload — a sentinel-driven poller could see
    a false "done" before the post-join upload worker ran. Post-fix the shard
    path writes a NON-terminal ``epm:progress`` artifact + ``[phase=shard_done]``
    and every ``write_sentinel`` call site lives inside ``_finalize_worker``.
  - BLOCKER 3 (answer-multiplicity equalization): pre-fix the behavior arm built
    one row PER ROLLOUT (up to 10 duplicated c_last per context). Post-fix the
    headline behavior source is 1 row PER CONTEXT (fixed-seed single rollout), so
    two contexts x ten rollouts give X_beh of 2 rows, not 20.
  - BLOCKER 4 (NaN judge labels enter g fits): pre-fix a single None->NaN label
    poisoned the ridge g fit (all-NaN output). Post-fix fit_g_cell drops the
    NaN-label rows (drop-never-coerce), trains on the finite rows, and reports
    the dropped count; the h path is unchanged.
  - MINOR 9 (r_B subset indexing): pre-fix run_layer_matrix indexed
    ``r_b_full[layers.index(layer_idx)]`` (subset position) which read the WRONG
    layer's r_B for any nonzero frozen layer when ``layers`` is a subset.

Pure-CPU, no model / no network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from explore_persona_space.experiments.issue_779 import scaling_grid as SG  # noqa: E402


def _heteroscedastic_fit(seed: int = 0, n: int = 200, h: int = 24):
    """Fit a ridge h on HETEROSCEDASTIC X (columns with very different scales),
    return (W, xmu, xsd, r_B, X_eval). Heteroscedasticity is what breaks the
    raw-coordinate pv_pinv read; homoscedastic X would let all three coincide.
    """
    rng = np.random.default_rng(seed)
    # column scales spanning 3 orders of magnitude -> strongly heteroscedastic
    col_scale = 10.0 ** rng.uniform(-1.0, 2.0, size=h)
    X = rng.standard_normal((n, h)) * col_scale
    # a true linear map with noise so the ridge fit is non-degenerate
    W_true = rng.standard_normal((h, h))
    Y = X @ W_true + rng.standard_normal((n, h)) * 0.1
    W, xmu, xsd, _ymu, _P, _Q = SG._ridge_fit(X, Y)
    r_b = rng.standard_normal(h)
    X_eval = rng.standard_normal((40, h)) * col_scale
    return W, xmu, xsd, r_b, X_eval


def _min_norm_preimage(W: np.ndarray, r_b: np.ndarray) -> np.ndarray:
    """Reference min-norm preimage of r_B under the map c ↦ c @ W, computed via
    np.linalg.pinv — an INDEPENDENT reference, never the implementation's own
    SVD formula (the round-2 test mirrored the buggy orientation).

    Row-vector convention: ``w @ W = r_b`` ⇒ ``Wᵀ wᵀ = r_bᵀ`` ⇒
    ``w = ((Wᵀ)⁺ r_b)`` = ``pinv(W.T) @ r_b``.
    """
    return np.linalg.pinv(W.T) @ r_b


def test_blocker1_pv_pinv_reads_standardized_space_not_raw():
    """BLOCKER 1: pv_pinv must read the STANDARDIZED eval c_last, not raw.

    The correct read is ⟨(c_last - xmu)/xsd, M⁺r_B⟩. The pre-fix bug read raw
    ⟨c_last, M⁺r_B⟩. On heteroscedastic X these are nearly orthogonal, so:
      - the code's read correlates ~1.0 with the standardized-space reference, and
      - it does NOT match the raw-coordinate (pre-fix) read.
    The reference w_pinv comes from np.linalg.pinv (NOT the implementation's SVD
    formula — the round-2 version of this test mirrored the orientation bug).
    """
    W, xmu, xsd, r_b, X_eval = _heteroscedastic_fit()
    eval_mat = {"c_last": X_eval}

    got = SG.pv_pinv_read(W, r_b, eval_mat, xmu=xmu, xsd=xsd, rank=None)

    # Standardized-space reference with an INDEPENDENT pinv.
    w_pinv = _min_norm_preimage(W, r_b)
    ref_std = ((X_eval - xmu) / xsd) @ w_pinv
    # Pre-fix (buggy round-1) raw-coordinate read.
    ref_raw = X_eval @ w_pinv

    corr_std = float(np.corrcoef(got, ref_std)[0, 1])
    corr_raw = float(np.corrcoef(got, ref_raw)[0, 1])
    assert corr_std > 0.9999, f"pv_pinv must match the standardized-space read (corr {corr_std})"
    # The whole point: standardized != raw on heteroscedastic X, so the fixed read
    # is NOT the raw-coordinate read the pre-fix code produced.
    assert corr_raw < 0.9, (
        f"fixed pv_pinv should differ from the raw-coordinate bug (corr_raw {corr_raw}); "
        "heteroscedasticity must actually separate the two spaces for this test to bite"
    )


def test_blocker_r3_pv_pinv_satisfies_preimage_property_planted_rb():
    """BLOCKER r3 (pv-pinv-transposed-orientation): w_pinv must be a true preimage.

    Plant ``r_B = w0 @ W`` (r_B in the fitted map's image, W full-rank here), then
    the Moore-Penrose preimage satisfies ``w_pinv @ W ≈ r_B`` to machine precision
    and recovers the planted ``w0`` exactly. The round-2 code computed the
    pseudoinverse of the TRANSPOSED map — residual ≈47 on this fixture shape and
    only ≈0.44 corr with w0 — so this test FAILS against it. w_pinv is extracted
    through the public API by evaluating at ``c_last = xmu + xsd * I`` (⇒
    ``c_std = I`` ⇒ the read IS w_pinv).
    """
    rng = np.random.default_rng(7)
    n, h = 40, 8
    W, xmu, xsd, _ymu, _P, _Q = SG._ridge_fit(
        rng.standard_normal((n, h)), rng.standard_normal((n, h))
    )
    assert np.linalg.matrix_rank(W) == h, "fixture requires a full-rank fitted map"
    w0 = rng.standard_normal(h)
    r_b = w0 @ W  # planted: r_B in the map's image

    eval_mat = {"c_last": xmu + xsd * np.eye(h)}  # c_std == I -> read == w_pinv
    w_pinv = SG.pv_pinv_read(W, r_b, eval_mat, xmu=xmu, xsd=xsd, rank=None)

    resid = float(np.linalg.norm(w_pinv @ W - r_b))
    assert resid < 1e-8 * max(1.0, float(np.linalg.norm(r_b))), (
        f"w_pinv must satisfy the defining preimage property w_pinv @ W ≈ r_B "
        f"(residual {resid}); the transposed-map orientation fails this by ~O(10)"
    )
    assert np.allclose(w_pinv, w0, atol=1e-6), (
        "full-rank min-norm preimage must recover the planted w0 exactly"
    )


def test_blocker_r3_pv_pinv_equals_transpose_read_for_orthogonal_map():
    """sigma-vs-1/sigma duality: for an ORTHOGONAL map (all singular values 1) the pinv read
    equals the transpose read ⟨c_std @ W, r_B⟩ = ⟨c_std, W r_B⟩. The round-2
    orientation produced ⟨c_std, Wᵀ r_B⟩ ≠ ⟨c_std, W r_B⟩ and fails this."""
    rng = np.random.default_rng(11)
    h = 10
    W_orth, _ = np.linalg.qr(rng.standard_normal((h, h)))
    xmu = rng.standard_normal(h)
    xsd = rng.uniform(0.5, 2.0, h)
    r_b = rng.standard_normal(h)
    X_eval = rng.standard_normal((25, h))

    got = SG.pv_pinv_read(W_orth, r_b, {"c_last": X_eval}, xmu=xmu, xsd=xsd, rank=None)
    transpose_read = ((X_eval - xmu) / xsd) @ (W_orth @ r_b)
    assert np.allclose(got, transpose_read, atol=1e-10), (
        "pinv read must equal the transpose read for an orthogonal map (sigma = 1 = 1/sigma)"
    )


def test_blocker_r3_persisted_fit_state_w_pinv_is_correct_orientation():
    """The PERSISTED ``pv_pinv_fit_state.w_pinv`` (the post-hoc recompute vector a
    pod-run artifact ships) must itself satisfy the preimage property — round 2
    persisted the transposed-map vector, permanently wrong in the JSON."""
    import issue779_scaling_grid as CLI

    rng = np.random.default_rng(3)
    n, h = 30, 6
    W, xmu, xsd, _ymu, _P, _Q = SG._ridge_fit(
        rng.standard_normal((n, h)), rng.standard_normal((n, h))
    )
    w0 = rng.standard_normal(h)
    r_b = w0 @ W

    state = CLI._pv_pinv_fit_state(W, xmu, xsd, r_b, rank=None)
    assert state["read_space"] == "standardized"
    w_pinv = np.asarray(state["w_pinv"], dtype=np.float64)  # persisted float32
    resid = float(np.linalg.norm(w_pinv @ W - r_b))
    assert resid < 1e-3 * max(1.0, float(np.linalg.norm(r_b))), (
        f"persisted w_pinv must be the correct-orientation preimage (float32 "
        f"round-trip tolerance; residual {resid})"
    )
    # and it matches the read-path vector (shared orientation across both sites)
    ref = _min_norm_preimage(W, r_b)
    assert np.allclose(w_pinv, ref, atol=1e-4)


def test_blocker1_pv_pinv_reads_tuple_uses_standardization():
    """pv_pinv_reads (frozen + full) both use the standardized-space read."""
    W, xmu, xsd, r_b, X_eval = _heteroscedastic_fit(seed=1)
    eval_mat = {"c_last": X_eval}
    frozen, full = SG.pv_pinv_reads(W, r_b, eval_mat, xmu=xmu, xsd=xsd, rank=5)
    # full-rank must equal the single-read full-rank path
    full_single = SG.pv_pinv_read(W, r_b, eval_mat, xmu=xmu, xsd=xsd, rank=None)
    assert np.allclose(full, full_single)
    # frozen (rank 5) differs from full when the map has >5 informative dims
    assert not np.allclose(frozen, full)


def test_pv_pinv_compact_svd_matches_dense():
    """The O(H r^2) compact SVD (from the low-rank factors) equals the dense SVD.

    W = P @ Q has rank <= n_train, so its compact SVD via the factors is EXACT.
    This guards the pv_pinv-cost optimization: a future refactor that breaks the
    QR/small-SVD identity would silently change the pv_pinv direction.
    """
    W, xmu, xsd, _ymu, P, Q = SG._ridge_fit(
        np.random.default_rng(0).standard_normal((12, 24)),
        np.random.default_rng(1).standard_normal((12, 24)),
    )
    U_d, s_d, Vt_d = SG.pv_pinv_svd(W)  # dense
    U_c, s_c, Vt_c = SG.pv_pinv_svd(W, P=P, Q=Q)  # compact
    # singular values agree (top rank)
    r = min(len(s_d), len(s_c))
    assert np.allclose(np.sort(s_d)[-r:], np.sort(s_c)[-r:], atol=1e-8)
    # the pinv reads agree exactly
    rb = np.random.default_rng(2).standard_normal(24)
    em = {"c_last": np.random.default_rng(3).standard_normal((20, 24))}
    rd = SG.pv_pinv_reads(W, rb, em, xmu=xmu, xsd=xsd, rank=None, svd=(U_d, s_d, Vt_d))[0]
    rc = SG.pv_pinv_reads(W, rb, em, xmu=xmu, xsd=xsd, rank=None, svd=(U_c, s_c, Vt_c))[0]
    assert np.allclose(rd, rc, atol=1e-8), "compact-SVD pv_pinv must equal the dense-SVD read"


def test_blocker3_headline_behavior_is_context_level_not_rollout_level():
    """BLOCKER 3: the headline behavior source is 1 row PER CONTEXT, not per rollout.

    Two contexts x ten rollouts => headline X_beh has 2 rows (not 20). The
    secondary (10-rollout-mean) source is ALSO 1 row per context but averages the
    rollouts.
    """
    import issue779_scaling_grid as CLI

    h = 8
    n_ctx, n_roll = 2, 10
    rng = np.random.default_rng(0)
    # cb: two contexts, ten valid rollouts each (20 v_x rows), one layer.
    cx_last = np.stack([rng.standard_normal(h) for _ in range(n_ctx)])  # (2, H)
    import torch

    cb = {
        "cx_last": torch.tensor(cx_last[:, None, :], dtype=torch.float32),  # (2, 1, H)
        "v_x": torch.tensor(
            rng.standard_normal((n_ctx * n_roll, 1, h)), dtype=torch.float32
        ),  # (20, 1, H)
        "vx_index": [(ci, ri) for ci in range(n_ctx) for ri in range(n_roll)],
        "layers": [0],
    }
    scores = {
        str(ci): {str(ri): float(rng.uniform(0, 100)) for ri in range(n_roll)}
        for ci in range(n_ctx)
    }

    X_head, Y_head, y_head = CLI._behavior_matrices(
        cb, scores, cli=0, hidden=h, agg=CLI.BEHAVIOR_AGG_HEADLINE, seed=42
    )
    assert X_head.shape[0] == n_ctx, (
        f"headline X_beh must be 1 row/context ({n_ctx}), got {X_head.shape[0]}"
    )
    assert Y_head.shape[0] == n_ctx
    assert y_head.shape[0] == n_ctx

    X_sec, Y_sec, y_sec = CLI._behavior_matrices(
        cb, scores, cli=0, hidden=h, agg=CLI.BEHAVIOR_AGG_SECONDARY, seed=42
    )
    assert X_sec.shape[0] == n_ctx, "secondary is also 1 row/context (10-rollout mean)"
    # secondary Y is the MEAN over the context's rollouts; headline is a SINGLE
    # rollout -> the two differ (they would coincide only if n_roll==1).
    assert not np.allclose(Y_head, Y_sec)
    # secondary label = mean over the context's rollout labels.
    for ci in range(n_ctx):
        want = np.mean([scores[str(ci)][str(ri)] for ri in range(n_roll)])
        assert y_sec[ci] == pytest.approx(want)


def test_blocker3_headline_selection_is_deterministic():
    """Fixed-seed single-rollout selection is deterministic across calls."""
    import issue779_scaling_grid as CLI
    import torch

    h = 6
    rng = np.random.default_rng(1)
    cb = {
        "cx_last": torch.tensor(rng.standard_normal((3, 1, h)), dtype=torch.float32),
        "v_x": torch.tensor(rng.standard_normal((3 * 5, 1, h)), dtype=torch.float32),
        "vx_index": [(ci, ri) for ci in range(3) for ri in range(5)],
        "layers": [0],
    }
    scores = {str(ci): {str(ri): float(ri) for ri in range(5)} for ci in range(3)}
    _, Y1, y1 = CLI._behavior_matrices(cb, scores, 0, h, agg=CLI.BEHAVIOR_AGG_HEADLINE, seed=42)
    _, Y2, y2 = CLI._behavior_matrices(cb, scores, 0, h, agg=CLI.BEHAVIOR_AGG_HEADLINE, seed=42)
    assert np.allclose(Y1, Y2) and np.allclose(y1, y2)


def test_blocker4_fit_g_cell_drops_nan_labels_reports_count():
    """BLOCKER 4: labels [1.0, NaN, 3.0] => g trains on 2 finite rows, finite preds.

    Pre-fix, the NaN entered the ridge closed form and g came back all-NaN.
    """
    rng = np.random.default_rng(0)
    X_tr = rng.standard_normal((3, 5))
    y_tr = np.array([1.0, np.nan, 3.0])
    X_ev = rng.standard_normal((7, 5))
    eval_mat = {"c_last": X_ev}
    g_pred, n_dropped = SG.fit_g_cell(X_tr, y_tr, eval_mat)
    assert n_dropped == 1, f"one NaN label must be dropped, got {n_dropped}"
    assert np.isfinite(g_pred).all(), "g predictions must be finite after dropping the NaN row"
    # sanity: matches a manual finite-row ridge fit
    from explore_persona_space.experiments.issue_779 import fit_h as F

    ref = F.ridge_fit_predict(X_tr[[0, 2]], np.array([1.0, 3.0]), X_ev)
    assert np.allclose(g_pred, ref)


def test_blocker4_fit_g_cell_all_nan_labels_returns_nan_not_crash():
    """A cell whose every label is missing yields all-NaN g (the label-floor case),
    not a crash — and reports every row dropped."""
    X_tr = np.random.default_rng(0).standard_normal((4, 5))
    y_tr = np.array([np.nan, np.nan, np.nan, np.nan])
    eval_mat = {"c_last": np.random.default_rng(1).standard_normal((3, 5))}
    g_pred, n_dropped = SG.fit_g_cell(X_tr, y_tr, eval_mat)
    assert n_dropped == 4
    assert np.isnan(g_pred).all()


def test_blocker4_g_holdout_drops_nan_labels():
    """run_g_holdout_question drops NaN-label fit-fold rows and reports the count."""
    rng = np.random.default_rng(0)
    n_q = 6
    per_q_rows = 4
    q, c_last, y, cond, mode = [], [], [], [], []
    for qi in range(n_q):
        for _ in range(per_q_rows):
            q.append(qi)
            c_last.append(rng.standard_normal(5))
            cond.append(qi % 2)  # 2 conditions
            mode.append("system")
            y.append(rng.uniform(0, 100))
    y = np.array(y, dtype=float)
    # poison a handful of labels with NaN
    y[[1, 7, 13]] = np.nan
    mat = {
        "question": np.array(q),
        "c_last": np.array(c_last),
        "y": y,
        "cond": np.array(cond),
        "mode": np.array(mode),
    }
    out = SG.run_g_holdout_question(mat, k_folds=3, n_boot=10, base_seed=0)
    assert out["labels_dropped_nan"] >= 1, "NaN fit-fold labels must be counted as dropped"
    # every fold's point r must be finite or NaN-but-not-crash
    for m in ("system", "many_shot"):
        assert m in out["modes"]


def _dispatch_dryrun(n_gpu: int, stage: str = "all") -> list[str]:
    """Run the dispatcher in DRY-RUN with a forced GPU count; return DRYRUN lines."""
    import os
    import subprocess

    env = dict(os.environ)
    env["EPM_DISPATCH_DRY_RUN"] = "1"
    env["EPM_DISPATCH_FORCE_NGPU"] = str(n_gpu)
    env["WORKLOAD_ROOT"] = str(REPO)  # REPO_ROOT resolves to the repo with scripts/
    proc = subprocess.run(
        ["bash", str(REPO / "scripts" / "issue779_dispatch_armb.sh"), "--stage", stage],
        env=env,
        cwd=str(REPO),
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return [ln for ln in proc.stdout.splitlines() if ln.startswith("DRYRUN")]


def test_major5_multigpu_dispatch_has_exactly_one_lmsys_g_writer():
    """MAJOR 5: the multi-GPU dispatcher must launch EXACTLY ONE lmsys_g worker.

    Pre-fix, the multi-GPU fan-out ran ``--stage all`` per trait-worker, so every
    trait worker (3 GPUs) independently wrote the single lmsys_g_labels.json (a
    last-writer-wins race). Post-fix, the corpus phase is trait-sharded with
    ``--stage corpus --no-upload`` and lmsys_g runs as ONE post-join worker with
    ``--stage lmsys_g``. Driven via the dispatcher's DRY-RUN + FORCE_NGPU hooks.
    """
    lines = _dispatch_dryrun(n_gpu=3, stage="all")
    lmsys_g_writers = [ln for ln in lines if "--stage lmsys_g" in ln]
    corpus_workers = [ln for ln in lines if "--stage corpus" in ln]
    all_stage = [ln for ln in lines if "--stage all" in ln]
    assert len(lmsys_g_writers) == 1, (
        f"exactly one lmsys_g writer required (MAJOR 5 race); got {len(lmsys_g_writers)}:\n"
        + "\n".join(lines)
    )
    assert len(all_stage) == 0, f"no trait worker may run --stage all (the race source):\n{lines}"
    # 3 traits over 3 GPUs -> 3 corpus workers, each --no-upload.
    assert len(corpus_workers) == 3, (
        f"3 trait-sharded corpus workers expected; got {corpus_workers}"
    )
    for cw in corpus_workers:
        assert "--no-upload" in cw, f"sharded corpus workers must pass --no-upload:\n{cw}"


def test_major5_single_gpu_dispatch_one_worker_all_stages():
    """1-GPU path runs ONE worker for all stages -> no lmsys_g race."""
    lines = _dispatch_dryrun(n_gpu=1, stage="all")
    workers = [ln for ln in lines if "traits=" in ln]
    assert len(workers) == 1, f"1-GPU path must be a single worker; got {workers}"
    assert "--stage all" in workers[0]


def test_minor9_layer_matrix_uses_absolute_rb_layer_index(tmp_path, monkeypatch):
    """MINOR 9: run_layer_matrix indexes r_b_full by the ABSOLUTE layer, not the
    subset position. We assert the read function receives r_b_full[layer_idx]
    (absolute) for a subset ``layers`` list where index != value.
    """
    import issue779_scaling_grid as CLI

    n_layers, h = 8, 4
    r_b_full = np.arange(n_layers * h, dtype=float).reshape(n_layers, h)  # layer L rows == L*h..
    captured_rb = {}

    # Stub the heavy internals: capture which r_B row fit_h_cell is handed per layer.
    def fake_build_eval_matrix_with_q(cells, layer_idx, r_b):
        return {
            "c_last": np.zeros((3, h)),
            "y": np.zeros(3),
            "cond": np.array([0, 0, 1]),
            "mode": np.array(["system", "system", "many_shot"]),
        }

    def fake_load_corpus_source(*a, **k):
        # tiny 2-row source so assemble/fit paths are exercised trivially
        X = np.random.default_rng(0).standard_normal((3, h))
        Y = np.random.default_rng(1).standard_normal((3, h))
        return SG.TrainSource(X, Y, None, X, Y, None)

    def fake_fit_h_cell(X_tr, Y_tr, eval_mat, rb_l, **kw):
        captured_rb[tuple(np.round(rb_l, 3))] = True
        n = eval_mat["c_last"].shape[0]
        return {"dot": np.zeros(n), "cos": np.zeros(n), "W": None, "xmu": None, "xsd": None}

    monkeypatch.setattr(CLI, "build_eval_matrix_with_q", fake_build_eval_matrix_with_q)
    monkeypatch.setattr(CLI, "load_corpus_source", fake_load_corpus_source)
    monkeypatch.setattr(SG, "fit_h_cell", fake_fit_h_cell)

    subset_layers = [5]  # index 0 in the subset, but ABSOLUTE layer 5
    CLI.run_layer_matrix(
        tmp_path,
        {"layers": list(range(n_layers))},
        None,
        cells=[],
        r_b_full=r_b_full,
        trait="evil",
        layers=subset_layers,
        n_boot=1,
        n_shuffle=0,
        seed=0,
    )
    # fit_h_cell must have been handed r_b_full[5] (absolute), NOT r_b_full[0].
    assert tuple(np.round(r_b_full[5], 3)) in captured_rb, (
        "layer matrix must use absolute r_B layer"
    )
    assert tuple(np.round(r_b_full[0], 3)) not in captured_rb, "must NOT use subset-position r_B"


def test_blocker_r3_no_upload_worker_writes_nonterminal_sentinel(monkeypatch, tmp_path):
    """BLOCKER r3 (multigpu-no-upload-terminal-sentinel): a --no-upload shard
    worker's end-of-run path must NEVER emit the terminal epm:results /
    epm:smoke-result sentinel nor [phase=done] — only a NON-terminal
    epm:progress artifact + [phase=shard_done]. A worker that ran the upload
    keeps the terminal sentinel."""
    import issue779_gen_behavior_corpus as G

    kinds: list[str] = []
    extras: list[dict] = []
    phases: list[str] = []
    monkeypatch.setattr(
        G.C,
        "write_sentinel",
        lambda kind, note, task_id=779, extra=None: (
            kinds.append(kind),
            extras.append(extra or {}),
            tmp_path / "sentinel.json",
        )[-1],
    )
    monkeypatch.setattr(G.C, "phase", lambda name: phases.append(name))

    # shard worker (--no-upload): non-terminal only.
    G._finalize_worker(no_upload=True, smoke=False, summary={"traits": ["evil"]}, stage="corpus")
    assert kinds == ["epm:progress"], f"--no-upload shard must not write a terminal kind: {kinds}"
    assert extras[0].get("terminal") is False and extras[0].get("blocks_pipeline") is False
    assert phases == ["shard_done"], f"--no-upload shard must not print [phase=done]: {phases}"

    # uploaded worker: terminal epm:results + [phase=done] (unchanged contract).
    kinds.clear(), extras.clear(), phases.clear()
    G._finalize_worker(no_upload=False, smoke=False, summary={}, stage="all")
    assert kinds == ["epm:results"] and phases == ["done"]

    # uploaded smoke worker: terminal epm:smoke-result.
    kinds.clear(), extras.clear(), phases.clear()
    G._finalize_worker(no_upload=False, smoke=True, summary={}, stage="all")
    assert kinds == ["epm:smoke-result"] and phases == ["done"]


def test_blocker_r3_all_sentinel_writes_live_in_finalize_worker():
    """No code path in the corpus-gen worker can reach write_sentinel except via
    the guarded _finalize_worker (whose no_upload branch is non-terminal) — an
    AST sweep over the module, so a future direct write_sentinel("epm:results")
    call on a shard-reachable path fails this test."""
    import ast
    import inspect

    import issue779_gen_behavior_corpus as G

    tree = ast.parse(inspect.getsource(G))
    offenders: list[str] = []

    class _V(ast.NodeVisitor):
        def __init__(self):
            self.stack: list[str] = []

        def visit_FunctionDef(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if name == "write_sentinel" and (self.stack[-1:] != ["_finalize_worker"]):
                offenders.append(self.stack[-1] if self.stack else "<module>")
            self.generic_visit(node)

    _V().visit(tree)
    assert not offenders, (
        f"write_sentinel called outside _finalize_worker in: {offenders} — the "
        "--no-upload suppression guard would not cover these sites"
    )


def test_r4_rb_loader_hf_fallback_materializes_into_rb_dir(monkeypatch, tmp_path):
    """Crash-fix r4 (att-20260702-082017: the GCP git-clone lane stages no
    data/): when NEITHER local r_B candidate exists, _resolve_rb_path must fetch
    issue779_monitoring/r_b/<trait>.pt from the HF data repo (hf_hub_download,
    repo_type=dataset) and MATERIALIZE it into rb_dir/<trait>.pt so every later
    phase sees the standard local layout; a subsequent local hit must NOT touch
    HF again. Offline: hf_hub_download is monkeypatched."""
    import huggingface_hub
    import issue779_gen_behavior_corpus as G
    import torch

    calls: list[dict] = []
    expected = torch.arange(28 * 3584, dtype=torch.float32).reshape(28, 3584)

    def fake_download(repo_id, filename, repo_type=None, **kw):
        calls.append({"repo_id": repo_id, "filename": filename, "repo_type": repo_type})
        p = tmp_path / "hf_cache" / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"trait": "evil", "r_b": expected}, p)
        return str(p)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    out_base = tmp_path / "out"
    rb_dir = out_base / "r_b"
    got = G._resolve_rb_path("evil", rb_dir, out_base)
    assert got == rb_dir / "evil.pt" and got.exists(), "must materialize into rb_dir"
    assert calls == [
        {
            "repo_id": G.C.HF_DATA_REPO,
            "filename": f"{G.C.HF_PREFIX}/r_b/evil.pt",
            "repo_type": "dataset",
        }
    ], f"unexpected HF fetch spec: {calls}"
    blob = torch.load(got, weights_only=False)
    assert blob["r_b"].shape == (28, 3584) and torch.equal(blob["r_b"], expected)

    # Second resolve: local hit — the HF fallback must NOT fire again.
    got2 = G._resolve_rb_path("evil", rb_dir, out_base)
    assert got2 == got and len(calls) == 1, "local hit must not re-download"


# ── round-9 (v79 grid fixes) regression tests ─────────────────────────────────
# Round 9 corrected the failure-v5 OOM mis-diagnosis (the exit-137s were
# deliberate PM stops; no batched-draw HxH materialization ever existed) and
# applied the v79 perf/coverage fixes. These tests pin EXACT equivalence of the
# fast paths against serial references + the new B1 recon outputs.


def _reference_ridge_fit_predict_materialized(X_train, Y_train, X_eval, lambdas=None):
    """The PRE-v79 ridge_fit_predict, verbatim (materialized per-lambda Yhat GCV)
    — the independent reference the GCV-identity fix is pinned against."""
    if lambdas is None:
        lambdas = np.logspace(-2, 4, 13)
    Xtr = np.asarray(X_train, dtype=np.float64)
    Ytr = np.asarray(Y_train, dtype=np.float64)
    Xev = np.asarray(X_eval, dtype=np.float64)
    squeeze = Ytr.ndim == 1
    if squeeze:
        Ytr = Ytr[:, None]
    n = Xtr.shape[0]
    xmu = Xtr.mean(0)
    xsd = Xtr.std(0) + 1e-9
    Xtr_n = (Xtr - xmu) / xsd
    Xev_n = (Xev - xmu) / xsd
    ymu = Ytr.mean(0)
    Ytr_c = Ytr - ymu
    U, s, Vt = np.linalg.svd(Xtr_n, full_matrices=False)
    s2 = s**2
    UtY = U.T @ Ytr_c
    best_lam, best_gcv = lambdas[0], np.inf
    for lam in lambdas:
        filt = s2 / (s2 + lam)
        Yhat = U @ (filt[:, None] * UtY)
        rss = float(np.sum((Ytr_c - Yhat) ** 2))
        dof = float(np.sum(filt))
        denom = (n - dof) ** 2
        gcv = rss / denom if denom > 1e-12 else np.inf
        if gcv < best_gcv:
            best_gcv, best_lam = gcv, lam
    filt = s / (s2 + best_lam)
    W = (Vt.T * filt) @ UtY
    preds = Xev_n @ W + ymu
    return (preds[:, 0] if squeeze else preds), float(best_lam)


def test_r9_ridge_fit_predict_gcv_identity_matches_materialized_reference():
    """v79 fix 2: the GCV-RSS SVD identity must reproduce the OLD materialized
    per-lambda Yhat loop — same selected lambda, same predictions — for both a
    scalar and a multi-output target, N<H and N>H."""
    from explore_persona_space.experiments.issue_779 import fit_h as F

    rng = np.random.default_rng(0)
    for n, h in ((20, 32), (48, 16)):
        X = rng.standard_normal((n, h))
        Y = X @ rng.standard_normal((h, h)) + 0.2 * rng.standard_normal((n, h))
        Xev = rng.standard_normal((9, h))
        got = F.ridge_fit_predict(X, Y, Xev)
        want, _lam = _reference_ridge_fit_predict_materialized(X, Y, Xev)
        assert np.allclose(got, want, atol=1e-10), f"multi-output identity broke at ({n},{h})"
        y = X @ rng.standard_normal(h) + 0.2 * rng.standard_normal(n)
        got_s = F.ridge_fit_predict(X, y, Xev)
        want_s, _lam = _reference_ridge_fit_predict_materialized(X, y, Xev)
        assert np.allclose(got_s, want_s, atol=1e-10), f"scalar identity broke at ({n},{h})"


def test_r9_verify_live_ridge_gate_passes():
    """v79 fixes 4+6: the eigh fast core reproduces the numpy-SVD serial
    reference on BOTH forms (identical lambda; pred/W ~1e-14 measured) — the
    LIVE-path gate the CLI runs under --verify-vectorized."""
    res = SG.verify_live_ridge()
    for form in ("dual", "primal"):
        assert res[form]["d_pred"] < 1e-7 and res[form]["d_w"] < 1e-7 and res[form]["d_g"] < 1e-7


def test_r9_fast_core_form_bounds_gram_memory_shape():
    """Memory-shape invariant: the eigh Gram is never larger than min(N, H)^2 —
    dual (N x N) for N<=H, primal (H x H) for N>H — so a big-N cell never
    materializes an N x N Gram (the round's memory-bounding scope)."""
    from explore_persona_space.experiments.issue_779 import fit_h as F

    rng = np.random.default_rng(1)
    core_d = F.RidgeFitCore(rng.standard_normal((10, 24)), rng.standard_normal((10, 24)))
    assert core_d.form == "dual" and core_d.gram_n == 10
    core_p = F.RidgeFitCore(rng.standard_normal((50, 8)), rng.standard_normal((50, 8)))
    assert core_p.form == "primal" and core_p.gram_n == 8
    for core in (core_d, core_p):
        assert core.gram_n == min(core.n, core.h)


def test_r9_fit_g_cell_shared_decomposition_matches_independent_fit():
    """v79 fix 3: with ALL labels finite, g sharing the h fit's decomposition
    equals the independent ridge_fit_predict on the same rows; ANY NaN label
    still routes through the finite-subset refit (drop-never-coerce unchanged)."""
    from explore_persona_space.experiments.issue_779 import fit_h as F

    rng = np.random.default_rng(2)
    n, h = 30, 12
    X = rng.standard_normal((n, h))
    Y = rng.standard_normal((n, h))
    y = X @ rng.standard_normal(h) + 0.1 * rng.standard_normal(n)
    Xev = rng.standard_normal((7, h))
    eval_mat = {"c_last": Xev}
    h_fit = F.RidgeFitCore(X, Y)
    g_shared, dropped = SG.fit_g_cell(X, y, eval_mat, h_fit=h_fit)
    assert dropped == 0
    ref = F.ridge_fit_predict(X, y, Xev)
    assert np.allclose(g_shared, ref, atol=1e-8), "shared-decomposition g must equal the refit"
    # a NaN label forces the finite-subset path (rows differ from the h fit)
    y_nan = y.copy()
    y_nan[3] = np.nan
    g_sub, dropped = SG.fit_g_cell(X, y_nan, eval_mat, h_fit=h_fit)
    assert dropped == 1
    fin = np.isfinite(y_nan)
    assert np.allclose(g_sub, F.ridge_fit_predict(X[fin], y_nan[fin], Xev), atol=1e-10)


def _reference_bootstrap_serial(cond_x, cond_y, *, n_boot, seed, min_y_std=1.0, min_n=3, ci=0.95):
    """The PRE-v79 bootstrap loop, verbatim (recompute within_condition_pearson
    per replicate) — the reference the vectorized gather is pinned against."""
    from explore_persona_space.experiments.issue_779 import metrics as M

    rng = np.random.default_rng(seed)
    base = M.within_condition_pearson(cond_x, cond_y, min_y_std=min_y_std, min_n=min_n)
    n_cond = len(cond_x)
    if n_cond == 0 or base["n_conditions"] == 0:
        return {
            "point": base["r"],
            "lo": float("nan"),
            "hi": float("nan"),
            "n_conditions": base["n_conditions"],
            "n_boot_valid": 0,
        }
    boot_rs = []
    idx_all = np.arange(n_cond)
    for _ in range(n_boot):
        samp = rng.choice(idx_all, size=n_cond, replace=True)
        bx = [cond_x[i] for i in samp]
        by = [cond_y[i] for i in samp]
        r = M.within_condition_pearson(bx, by, min_y_std=min_y_std, min_n=min_n)["r"]
        if np.isfinite(r):
            boot_rs.append(r)
    if not boot_rs:
        return {
            "point": base["r"],
            "lo": float("nan"),
            "hi": float("nan"),
            "n_conditions": base["n_conditions"],
            "n_boot_valid": 0,
        }
    alpha = (1.0 - ci) / 2.0
    return {
        "point": base["r"],
        "lo": float(np.quantile(boot_rs, alpha)),
        "hi": float(np.quantile(boot_rs, 1.0 - alpha)),
        "n_conditions": base["n_conditions"],
        "n_boot_valid": len(boot_rs),
    }


def test_r9_bootstrap_vectorized_matches_serial_reference():
    """v79 fix 5: the precompute+gather bootstrap must be BIT-IDENTICAL to the
    old recompute-per-replicate loop (same rng.choice sequence, same means, same
    quantiles) — including excluded conditions (low y-std, too few points,
    degenerate x) that exercise the replicate-invariant validity mask."""
    from explore_persona_space.experiments.issue_779 import metrics as M

    rng = np.random.default_rng(3)
    cond_x, cond_y = [], []
    for i in range(9):
        n = 8
        x = rng.standard_normal(n)
        y = 3.0 * x + rng.standard_normal(n) * 2.0 + 50.0
        if i == 2:
            y = np.full(n, 50.0)  # y-std < 1 -> excluded
        if i == 5:
            x = np.zeros(n)  # degenerate x -> excluded
        if i == 7:
            x, y = x[:2], y[:2]  # < min_n -> excluded
        cond_x.append(x)
        cond_y.append(y)
    for seed in (0, 7):
        got = M.bootstrap_within_condition_ci(cond_x, cond_y, n_boot=64, seed=seed)
        want = _reference_bootstrap_serial(cond_x, cond_y, n_boot=64, seed=seed)
        assert got == want, (
            f"vectorized bootstrap diverged from the serial reference: {got} vs {want}"
        )


def _tiny_source(seed=0, n_l=12, n_b=6, h=5):
    rng = np.random.default_rng(seed)
    X_l = rng.standard_normal((n_l, h))
    Y_l = rng.standard_normal((n_l, h))
    y_l = rng.uniform(0, 100, n_l)
    X_b = rng.standard_normal((n_b, h))
    Y_b = rng.standard_normal((n_b, h))
    y_b = rng.uniform(0, 100, n_b)
    return SG.TrainSource(X_l, Y_l, y_l, X_b, Y_b, y_b)


def _tiny_eval_mat(seed=1, n=12, h=5):
    rng = np.random.default_rng(seed)
    return {
        "c_last": rng.standard_normal((n, h)),
        "y": rng.uniform(0, 100, n),
        "cond": np.array([0] * 6 + [1] * 6),
        "mode": np.array(["system"] * 6 + ["many_shot"] * 6, dtype=object),
    }


def test_r9_grid_cells_carry_heldout_recon():
    """B1 (plan v6 §4.4 drift fix): every grid cell carries recon_heldout with
    per-source held-out R2/cosine — computed on the COMPLEMENT of the drawn
    rows (n_heldout == n_avail - n_drawn), NaN with n_heldout=0 for a source the
    cell used in full."""
    src = _tiny_source()
    em = _tiny_eval_mat()
    rb = np.random.default_rng(4).standard_normal(5)
    out = SG.run_scaling_grid(
        src, em, rb, n_lmsys_grid=(0, 8, 12), n_behavior_grid=(0, 4), k_subsamples=2, n_boot=5
    )
    assert out["recon_heldout_cap"] == SG.RECON_HELDOUT_CAP
    for cell in out["cells"]:
        rec = cell["recon_heldout"]
        assert set(rec) == {"lmsys", "behavior"}
        for tag, n_avail, n_used in (
            ("lmsys", src.n_lmsys(), cell["n_lmsys_used"]),
            ("behavior", src.n_beh(), cell["n_behavior_used"]),
        ):
            want_held = min(n_avail - n_used, SG.RECON_HELDOUT_CAP)
            assert rec[tag]["n_heldout"] == want_held, (tag, cell)
            if want_held == 0:
                assert np.isnan(rec[tag]["r2"]) and np.isnan(rec[tag]["mean_cosine"])
            else:
                assert np.isfinite(rec[tag]["r2"]) and np.isfinite(rec[tag]["mean_cosine"])
                assert rec[tag]["r2"] <= 1.0 + 1e-9


def test_r9_grid_skip_edges_drops_edge_cells():
    """--skip-edges (v81 interior-only relaunch): only mixed nL>0 & nB>0 cells."""
    src = _tiny_source()
    em = _tiny_eval_mat()
    rb = np.random.default_rng(5).standard_normal(5)
    out = SG.run_scaling_grid(
        src,
        em,
        rb,
        n_lmsys_grid=(0, 8),
        n_behavior_grid=(0, 4),
        k_subsamples=1,
        n_boot=5,
        skip_edges=True,
    )
    assert out["skip_edges"] is True
    assert [(c["n_lmsys"], c["n_behavior"]) for c in out["cells"]] == [(8, 4)]
    assert all(c["arm"] == "arm_c" for c in out["cells"])


def test_r9_full_row_cell_dedupes_fit_across_k(monkeypatch):
    """Audit fix iii: a FULL-ROW cell (every source drawn full-or-zero — the K
    subsamples are permutations of the same set) fits ONCE and reuses across k;
    a genuinely-subsampled cell still fits per k. Readout values are unchanged
    (ridge is permutation-invariant) and per-k bootstrap seeds still differ."""
    from explore_persona_space.experiments.issue_779 import fit_h as F

    calls = []
    real_core = F.RidgeFitCore

    class SpyCore(real_core):
        def __init__(self, X, Y, **kw):
            calls.append(X.shape)
            super().__init__(X, Y, **kw)

    monkeypatch.setattr(F, "RidgeFitCore", SpyCore)
    src = _tiny_source()
    em = _tiny_eval_mat()
    rb = np.random.default_rng(6).standard_normal(5)
    # (12, 0) draws ALL 12 LMSYS rows -> full-row cell -> 1 fit for k=3.
    out = SG.run_scaling_grid(
        src, em, rb, n_lmsys_grid=(12,), n_behavior_grid=(0,), k_subsamples=3, n_boot=5
    )
    assert len(out["cells"]) == 3
    assert len(calls) == 1, f"full-row cell must fit once across k, got {len(calls)} fits"
    # subsampled cell (8 of 12): one fit PER k.
    calls.clear()
    SG.run_scaling_grid(
        src, em, rb, n_lmsys_grid=(8,), n_behavior_grid=(0,), k_subsamples=3, n_boot=5
    )
    assert len(calls) == 3, f"subsampled cell must fit per k, got {len(calls)} fits"


def test_r4_rb_loader_fails_loud_when_hf_fetch_also_fails(monkeypatch, tmp_path):
    """Crash-fix r4: no local candidate AND a failing HF fetch must raise a
    FileNotFoundError naming both local candidates + the HF path (fail loud,
    never a silent default)."""
    import huggingface_hub
    import issue779_gen_behavior_corpus as G

    def boom(*a, **k):
        raise RuntimeError("offline test — no network")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
    with pytest.raises(FileNotFoundError, match=r"HF fetch .*r_b/evil\.pt.* failed"):
        G._resolve_rb_path("evil", tmp_path / "r_b", tmp_path)


# ── r10: partial-write guard + edges composition (r6 BLOCKER
#    `partial-scaling-grid-overwrite`) ─────────────────────────────────────────


def _tiny_grid_entry(rb_seed=4, *, skip_edges=True, k=2):
    """A (trait, mode) scaling entry with all three variant slots on tiny axes."""
    src = _tiny_source()
    em = _tiny_eval_mat()
    rb = np.random.default_rng(rb_seed).standard_normal(5)

    def g(**kw):
        return SG.run_scaling_grid(
            src,
            em,
            rb,
            n_lmsys_grid=(0, 8),
            n_behavior_grid=(0, 4),
            k_subsamples=k,
            n_boot=5,
            skip_edges=skip_edges,
            **kw,
        )

    return {
        "frozen_layer": 3,
        "behavior_agg": "headline_1rollout",
        "natural": g(),
        "upsample_1to1": g(upsample_1to1=True),
        "secondary_10rollout": {"behavior_agg": "mean_10rollout", "natural": g()},
    }


def _edge_draw(n, draw_idx, *, modes=("system", "many_shot"), point=0.5):
    """One issue779_edges-format draw (the batch2_edges.json leaf schema)."""
    return {
        "n": n,
        "draw": draw_idx,
        "h_dot": {m: {"point": point, "n_conditions": 2} for m in modes},
        "h_cos": {m: {"point": point, "n_conditions": 2} for m in modes},
        "g": {m: {"point": point / 2, "n_conditions": 2} for m in modes},
        "g_n_valid": n,
        "g_n_dropped": 0,
        "h_gcv_lambda": 0.01,
        "g_gcv_lambda": 0.01,
        "h_recon_r2": 0.3,
        "g_recon_r2": 0.1,
    }


def _tiny_edges_doc(*, point=0.5):
    """An issue779_edges doc covering the _tiny_grid_entry axes ((0,8) x (0,4))."""
    return {
        "metadata": {"git_commit": "deadbeef", "timestamp_utc": "t", "k_draws": 1},
        "edges": {
            "evil": {
                "L3": {
                    "modes": ["system", "many_shot"],
                    "lmsys_axis": {"8": {"n_draws": 1, "draws": [_edge_draw(8, 0, point=point)]}},
                    "behavior_axis": {
                        "4": {"n_draws": 1, "draws": [_edge_draw(4, 0, point=point)]}
                    },
                }
            }
        },
    }


def test_r10_merge_grid_block_axis_or_k_mismatch_fails_loud():
    """r6 BLOCKER pin: merging grid blocks fit on DIFFERENT axes or subsample
    plans must raise, never silently mix incomparable cells (pre-fix the
    canonical write clobbered unconditionally — no merge, no guard)."""
    a = _tiny_grid_entry()["natural"]
    b = dict(a)
    b["n_lmsys_grid"] = [0, 999]
    with pytest.raises(ValueError, match="axis mismatch"):
        SG.merge_grid_block(a, b)
    c = dict(a)
    c["grid_shape"] = [a["grid_shape"][0], a["grid_shape"][1], 7]
    with pytest.raises(ValueError, match="k_subsamples mismatch"):
        SG.merge_grid_block(a, c)


def test_r10_partial_write_guard_stamps_incomplete_and_merges(tmp_path):
    """r6 BLOCKER (`partial-scaling-grid-overwrite`): a --skip-edges /
    variant-subset canonical write must (a) stamp complete:false + a
    machine-readable omitted-axes/variants record, and (b) MERGE with an
    existing artifact (prior cells preserved) instead of clobbering it."""
    import issue779_scaling_grid as GRID

    entry = _tiny_grid_entry()
    path = tmp_path / "scaling_grid.json"
    run_flags = {"skip_edges": True, "grid_variants": ["natural"]}

    # write 1: natural-only, interior-only
    scaling1 = {
        "traits": {
            "evil": {"system": {k: entry[k] for k in ("frozen_layer", "behavior_agg", "natural")}}
        },
        "meta": {"m": 1},
    }
    GRID._write_grid_checkpoint(path, scaling1, run_flags=run_flags)
    import json

    doc = json.loads(path.read_text())
    assert doc["complete"] is False
    rec = doc["completeness"]
    assert rec["last_write_run_flags"] == run_flags
    nat = rec["blocks"]["evil/system/natural"]
    # interior-only: the (8,0) + (0,4) edge coords are recorded missing
    assert nat["present"] and nat["n_missing_coords"] == 2
    assert sorted(map(tuple, nat["missing_coords"])) == [(0, 4), (8, 0)]
    assert nat["n_cells_planned"] == 3 * 2 and nat["n_cells_realized"] == 2
    assert rec["blocks"]["evil/system/upsample_1to1"] == {"present": False}
    assert rec["variants_complete"]["natural"] is False

    # write 2 (a later variant-subset run): 1to1-only — natural cells preserved
    scaling2 = {
        "traits": {
            "evil": {
                "system": {k: entry[k] for k in ("frozen_layer", "behavior_agg", "upsample_1to1")}
            }
        },
        "meta": {"m": 2},
    }
    GRID._write_grid_checkpoint(path, scaling2, run_flags=run_flags)
    doc2 = json.loads(path.read_text())
    got_nat = doc2["traits"]["evil"]["system"]["natural"]
    assert got_nat["cells"] == json.loads(json.dumps(entry["natural"]["cells"])), (
        "prior natural cells must survive a later variant-subset write"
    )
    assert doc2["traits"]["evil"]["system"]["upsample_1to1"]["cells"]
    assert doc2["completeness"]["blocks"]["evil/system/upsample_1to1"]["present"] is True

    # write 3: AXIS-mismatched grid at the same canonical path -> fail loud
    bad = _tiny_grid_entry()
    bad["natural"]["n_lmsys_grid"] = [0, 999]
    bad["natural"]["cells"] = [{**c, "n_lmsys": 999} for c in bad["natural"]["cells"]]
    scaling3 = {
        "traits": {
            "evil": {
                "system": {
                    "frozen_layer": 3,
                    "behavior_agg": "headline_1rollout",
                    "natural": bad["natural"],
                }
            }
        }
    }
    with pytest.raises(ValueError, match="axis mismatch"):
        GRID._write_grid_checkpoint(path, scaling3, run_flags=run_flags)


def test_r10_completeness_flips_true_when_all_planned_cells_present():
    """The default full path (all variants, all planned coords) stamps
    complete: true; coordinate coverage — not full-k — is the criterion."""
    entry_full = _tiny_grid_entry(skip_edges=False)
    scaling = {"traits": {"evil": {"system": entry_full, "many_shot": entry_full}}}
    rec = SG.stamp_completeness(scaling, expected_traits=["evil"])
    assert scaling["complete"] is True and rec["complete"] is True
    assert all(rec["variants_complete"].values())


def test_r10_compose_edges_fills_edges_refuses_conflicts(tmp_path):
    """v81 composition: pod edge cells fill the interior artifact's edge
    coordinates (lmsys edges -> all 3 variant slots; behavior edges -> the
    headline-agg slots only), idempotent on recompose, refuse-loud on a
    conflicting overlap / missing leaf / wrong mode."""
    entry = _tiny_grid_entry()
    scaling = {"traits": {"evil": {"system": entry}}}
    edges = _tiny_edges_doc()
    summary = SG.compose_edges_into_scaling(scaling, edges, edges_path="e.json")
    # natural + 1to1 gain both edge coords; secondary gains ONLY the lmsys edge
    assert summary["per_block"]["evil/system/natural"] == 2
    assert summary["per_block"]["evil/system/upsample_1to1"] == 2
    assert summary["per_block"]["evil/system/secondary_10rollout"] == 1
    nat_cells = {
        (c["n_lmsys"], c["n_behavior"], c["subsample"]): c for c in entry["natural"]["cells"]
    }
    edge_cell = nat_cells[(8, 0, 0)]
    assert edge_cell["source"] == "pod_edges" and edge_cell["arm"] == "arm_a_lmsys"
    assert edge_cell["h_ridge_dot_r"]["system"]["point"] == 0.5
    assert np.isnan(edge_cell["h_ridge_dot_r"]["system"]["lo"])
    assert edge_cell["h_ridge_dot_r"]["system"]["n_boot_valid"] == 0
    up_cell = {
        (c["n_lmsys"], c["n_behavior"], c["subsample"]): c for c in entry["upsample_1to1"]["cells"]
    }[(0, 4, 0)]
    assert up_cell["upsample_1to1"] is True and up_cell["arm"] == "arm_b_behavior"
    sec_keys = {
        (c["n_lmsys"], c["n_behavior"]) for c in entry["secondary_10rollout"]["natural"]["cells"]
    }
    assert (0, 4) not in sec_keys, "behavior-side edges must NOT fill the secondary agg"

    # completeness: natural/1to1 fully covered (k-shortfall recorded, not fatal);
    # secondary still missing its behavior edge -> overall complete stays False
    rec = SG.stamp_completeness(
        scaling,
        expected_traits=["evil"],
    )
    nat_sys = rec["blocks"]["evil/system/natural"]
    assert nat_sys["n_missing_coords"] == 0
    assert sorted(map(tuple, nat_sys["coords_below_planned_k"])) == [(0, 4), (8, 0)]
    assert rec["blocks"]["evil/system/secondary_10rollout"]["missing_coords"] == [[0, 4]]
    assert rec["variants_complete"]["secondary_10rollout"] is False

    # idempotent recompose: identical cells, no raise, no duplicates
    n_before = len(entry["natural"]["cells"])
    SG.compose_edges_into_scaling(scaling, _tiny_edges_doc(), edges_path="e.json")
    assert len(entry["natural"]["cells"]) == n_before

    # conflicting overlap -> refuse loud
    with pytest.raises(ValueError, match="conflicting cell values"):
        SG.compose_edges_into_scaling(scaling, _tiny_edges_doc(point=0.9), edges_path="e")

    # missing frozen-layer leaf / mode not read at the layer -> fail loud
    bad = _tiny_edges_doc()
    bad["edges"]["evil"] = {"L9": bad["edges"]["evil"]["L3"]}
    with pytest.raises(ValueError, match="no leaf"):
        SG.compose_edges_into_scaling({"traits": {"evil": {"system": _tiny_grid_entry()}}}, bad)
    bad2 = _tiny_edges_doc()
    bad2["edges"]["evil"]["L3"]["modes"] = ["many_shot"]
    with pytest.raises(ValueError, match="not read in mode"):
        SG.compose_edges_into_scaling({"traits": {"evil": {"system": _tiny_grid_entry()}}}, bad2)


def test_r10_compose_edges_cli_roundtrip(tmp_path):
    """The --compose-edges CLI branch: reads <out-dir>/scaling_grid.json,
    composes, stamps completeness + provenance, rewrites atomically."""
    import json
    from types import SimpleNamespace

    import issue779_scaling_grid as GRID

    entry = _tiny_grid_entry()
    scaling = {"traits": {"evil": {"system": entry}}, "meta": {}}
    grid_path = tmp_path / "scaling_grid.json"
    grid_path.write_text(json.dumps(scaling))
    edges_path = tmp_path / "edges.json"
    edges_path.write_text(json.dumps(_tiny_edges_doc()))
    ns = SimpleNamespace(out_dir=tmp_path, compose_edges=edges_path)
    assert GRID._compose_edges_cli(ns) == 0
    doc = json.loads(grid_path.read_text())
    assert doc["edges_composed"][0]["git_commit"] == "deadbeef"
    assert "complete" in doc and "completeness" in doc
    coords = {
        (c["n_lmsys"], c["n_behavior"]) for c in doc["traits"]["evil"]["system"]["natural"]["cells"]
    }
    assert {(8, 0), (0, 4), (8, 4)} <= coords
    # missing artifact -> fail loud
    with pytest.raises(FileNotFoundError):
        GRID._compose_edges_cli(
            SimpleNamespace(out_dir=tmp_path / "nope", compose_edges=edges_path)
        )
