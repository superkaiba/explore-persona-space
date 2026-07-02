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

    def fake_fit_h_cell(X_tr, Y_tr, eval_mat, rb_l):
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
