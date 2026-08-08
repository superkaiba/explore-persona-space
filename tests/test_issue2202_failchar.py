"""#2202 pod-driver unit pins (plan §4 P0-P2 kernels; all synthetic, no network).

Covers: mid-rank parity of the NEW per-row rank read vs the canonical
``knn_retrieval`` (the P1 full-pool equivalence gate's own reduction), chunk-size
invariance, gate tolerance behavior, chunked-fp64 covariance ≡ ``np.cov`` +
``shrunk_cholesky_from_cov`` ≡ the legacy ``_shrunk_cholesky``, whitened-space
Mahalanobis identity, the K-resample attribution partition, reduced-pool ranks
(target always in pool), confusion-edge construction + reciprocity + the two
null-draw kernels, and the <9 MB JSONL shard helper.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import issue2202_failchar as FC  # noqa: E402

from explore_persona_space.analysis.mapping_baselines import knn_retrieval  # noqa: E402
from explore_persona_space.analysis.null_battery import (  # noqa: E402
    _shrunk_cholesky,
    shrunk_cholesky_from_cov,
)


def _pool(n: int = 60, d: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    true = rng.standard_normal((n, d))
    true[9] = true[5]  # exact duplicate pool rows -> mid-rank ties exercised
    pred = true + 0.3 * rng.standard_normal((n, d))
    return pred, true


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_ranks_of_targets_matches_knn_retrieval(metric):
    pred, true = _pool()
    n = len(pred)
    ranks, _d_true, n_closer = FC.ranks_of_targets(pred, true, np.arange(n), metric, chunk=1000)
    rec = FC.ranks_summary(ranks, n)
    ref = knn_retrieval(pred, true, ks=(1, 5, 10), metric=metric)
    for k in (1, 5, 10):
        assert np.isclose(rec["acc_at_k"][k], ref["acc_at_k"][k], atol=1e-12)
    assert np.isclose(rec["mrr"], ref["mrr"], atol=1e-12)
    assert np.isclose(rec["median_rank"], ref["median_rank"], atol=1e-9)
    assert (n_closer >= 0).all() and (ranks >= 1.0).all()


def test_ranks_chunk_invariance():
    """Ranks + closer-counts are chunk-size INVARIANT; d_true is only float-close
    (chunked GEMMs reorder BLAS reductions at ~1e-16 — exactly the noise the
    tolerance-based mid-rank convention absorbs, per the knn_retrieval docstring)."""
    pred, true = _pool(seed=3)
    r1, d1, c1 = FC.ranks_of_targets(pred, true, np.arange(len(pred)), "euclidean", chunk=7)
    r2, d2, c2 = FC.ranks_of_targets(pred, true, np.arange(len(pred)), "euclidean", chunk=1000)
    assert np.array_equal(r1, r2) and np.array_equal(c1, c2)
    assert np.allclose(d1, d2, rtol=1e-12, atol=0)


def test_gate_compare_pass_and_fail():
    pred, true = _pool(seed=4)
    rec = knn_retrieval(pred, true, ks=(1, 5, 10), metric="euclidean")
    banked = json.loads(json.dumps(rec))  # str-keyed acc_at_k, the banked-JSON shape
    deltas, ok = FC._gate_compare(rec, banked)
    assert ok and max(deltas["acc_at_k"].values()) == 0.0
    corrupted = dict(banked)
    corrupted["acc_at_k"] = {**banked["acc_at_k"], "1": banked["acc_at_k"]["1"] + 0.01}
    _deltas, ok2 = FC._gate_compare(rec, corrupted)
    assert not ok2  # a real wiring bug moves acc@1 by O(0.1) >> the 2e-4 tolerance


def test_chunked_cov_matches_npcov_and_shared_cholesky():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((50, 6))
    cov, mu = FC.chunked_cov(x, np.arange(50), chunk=7)
    assert np.allclose(cov, np.cov(x, rowvar=False), atol=1e-10)
    assert np.allclose(mu, x.mean(axis=0), atol=1e-12)
    # the extracted shrink+jitter core reproduces the legacy helper bitwise-close
    l_new = shrunk_cholesky_from_cov(np.cov(x, rowvar=False), 0.1)
    l_old = _shrunk_cholesky(x, 0.1)
    assert np.allclose(l_new, l_old, atol=1e-12)


def test_whiten_space_is_mahalanobis():
    rng = np.random.default_rng(2)
    d = 5
    x = rng.standard_normal((200, d))
    cov = np.cov(x, rowvar=False)
    ell = shrunk_cholesky_from_cov(cov, 0.1)
    stats = {"mu_A": x.mean(0), "mu_C": np.zeros(d), "L": ell}
    pred = rng.standard_normal((4, d))
    ans = rng.standard_normal((4, d))
    spaces = FC.build_spaces(pred, ans, np.zeros((4, d)), stats)
    pw, aw, _ = spaces["whiten"]
    # z-space squared euclid == (u-v)^T Sigma_shrunk^{-1} (u-v)
    sigma = ell @ ell.T
    for i in range(4):
        diff = pred[i] - ans[i]
        direct = float(diff @ np.linalg.solve(sigma, diff))
        zdist = float(((pw[i] - aw[i]) ** 2).sum())
        assert np.isclose(direct, zdist, rtol=1e-8)


def test_kres_classes_partition():
    s = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    cls = FC.kres_classes(s)
    assert list(cls) == [
        "IRREDUCIBLE",
        "IRREDUCIBLE",
        "AMBIGUOUS",
        "MAP_ATTRIBUTABLE",
        "MAP_ATTRIBUTABLE",
    ]


def test_subpool_ranks_target_always_in_pool():
    from explore_persona_space.analysis.mapping_baselines import _pairwise_dist

    rng = np.random.default_rng(5)
    n, d, p = 12, 4, 6
    true = rng.standard_normal((n, d))
    pred = true.copy()  # perfect map -> rank 1 at every pool size
    sub = FC.draw_subpool(n, p, seed=99)
    in_sub = np.zeros(n, dtype=bool)
    in_sub[sub] = True
    dmat = _pairwise_dist(pred, true, "euclidean")
    dt = dmat[np.arange(n), np.arange(n)]
    ranks = FC.subpool_ranks_chunk(dmat, dt, 0, sub, in_sub)
    assert np.allclose(ranks, 1.0)  # the true target is in EVERY reduced pool
    # a corrupted prediction can only be beaten by pool MEMBERS (rank <= p)
    pred2 = true[::-1].copy()
    d2 = _pairwise_dist(pred2, true, "euclidean")
    dt2 = d2[np.arange(n), np.arange(n)]
    r2 = FC.subpool_ranks_chunk(d2, dt2, 0, sub, in_sub)
    assert (r2 >= 1.0).all() and (r2 <= p).all()
    # deterministic under the pinned seed
    assert np.array_equal(sub, FC.draw_subpool(n, p, seed=99))


def test_build_edges_and_reciprocity():
    rng = np.random.default_rng(6)
    n, d = 8, 4
    ans = rng.standard_normal((n, d)) * 5
    pred = ans.copy()
    pred[0] = ans[3]  # row 0's prediction sits ON a_3 -> a_3 outranks a_0
    src, dst, fwd, _dpred = FC.build_edges(pred, ans, chunk=3, cap_per_row=None)
    edges = set(zip(src.tolist(), dst.tolist(), strict=True))
    assert (0, 3) in edges
    assert (fwd >= 1).all()
    # reciprocity on a hand graph: {(0,1),(1,0),(2,0)} -> 2/3
    r = FC.reciprocity_of(np.asarray([0, 1, 2]), np.asarray([1, 0, 0]), n=4)
    assert np.isclose(r, 2.0 / 3.0)
    # per-row cap keeps top-K by (distance, index)
    src_c, _dst_c, fwd_c, _ = FC.build_edges(pred, ans, chunk=3, cap_per_row=1)
    assert (np.bincount(src_c, minlength=n) <= 1).all()
    assert (fwd_c == 1).all()


def test_degree_preserving_draws_deterministic_and_bounded():
    src = np.asarray([0, 0, 1, 2, 3])
    dst = np.asarray([1, 2, 0, 3, 2])
    v1, coll1 = FC.degree_preserving_draws(src, dst, n=5, n_draws=25, seed=7)
    v2, _ = FC.degree_preserving_draws(src, dst, n=5, n_draws=25, seed=7)
    assert np.array_equal(v1, v2)
    assert ((v1 >= 0) & (v1 <= 1)).all()
    assert set(coll1) == {"self_loops_mean", "multi_edges_mean"}


def test_distance_null_excludes_self_and_matches_out_degree():
    n = 4
    d_ans = np.ones((n, n), dtype=np.float32)
    np.fill_diagonal(d_ans, 0.0)
    out_deg = np.asarray([3, 0, 0, 0])
    # k_i = 3 over 3 non-self candidates -> edges are exactly {0->1,0->2,0->3}
    draws = FC.distance_null_draws(d_ans, out_deg, tau=1.0, n_draws=5, seed=11)
    assert np.allclose(draws, 0.0)  # no reverse edges exist -> reciprocity 0
    assert len(draws) == 5


def test_shard_json_rows_roundtrip(tmp_path):
    rows = [{"i": i, "blob": "x" * 500} for i in range(100)]
    names = FC.shard_json_rows(rows, "unit", tmp_path, max_bytes=5_000)
    shards = [nm for nm in names if nm.endswith(".jsonl")]
    manifest = json.loads((tmp_path / "unit.manifest.json").read_text())
    assert manifest["n_rows"] == 100 and manifest["shards"] == shards
    got = []
    for nm in shards:
        assert (tmp_path / nm).stat().st_size <= 5_000 + 600  # one-row overshoot bound
        with open(tmp_path / nm, encoding="utf-8") as f:
            got.extend(json.loads(ln) for ln in f if ln.strip())
    assert got == rows


def test_ranks_of_cols_in_row_midrank():
    row = np.asarray([0.0, 1.0, 1.0, 2.0, 3.0])
    r = FC.ranks_of_cols_in_row(row, np.asarray([0, 1, 2, 4]))
    assert r[0] == 1.0
    assert r[1] == r[2] == 2.5  # tied pair mid-rank
    assert r[3] == 5.0


# ── checkpoint-per-phase skip-if-done guards (round-2 fix; --force overrides) ──


def _fc_args(tmp_path, phase: str, extra: list[str] | None = None):
    args = FC.build_argparser().parse_args(
        [
            "--phase",
            phase,
            "--out-eval",
            str(tmp_path / "eval"),
            "--work-root",
            str(tmp_path / "wr"),
            "--sentinel-dir",
            str(tmp_path / "sent"),
            "--no-upload",
            "--no-git",
            *(extra or []),
        ]
    )
    args.work_root = Path(args.work_root)
    return args


def _write_reciprocity_done(args) -> None:
    out = FC.out_eval_dir(args)
    FC.atomic_json(
        out / "reciprocity.json", {"observed": {}, "null_degree": {}, "null_distance": {}}
    )
    der = FC._derived(args)
    der.mkdir(parents=True, exist_ok=True)
    one = np.ones(3)
    np.savez(
        der / "reciprocity_edges.npz",
        src_ci=one,
        dst_ci=one,
        rank_fwd=one,
        rank_rev=one,
        d_pred=one,
    )


def test_phase_done_validates_artifacts(tmp_path):
    args = _fc_args(tmp_path, "reciprocity")
    assert FC.phase_done(args, "reciprocity") is None  # nothing written yet
    _write_reciprocity_done(args)
    assert FC.phase_done(args, "reciprocity") is not None
    # a corrupt npz FAILS validation -> the phase re-runs (never trusted)
    (FC._derived(args) / "reciprocity_edges.npz").write_bytes(b"not a zip")
    assert FC.phase_done(args, "reciprocity") is None
    _write_reciprocity_done(args)
    # a JSON missing an expected key fails validation too
    FC.atomic_json(FC.out_eval_dir(args) / "reciprocity.json", {"observed": {}})
    assert FC.phase_done(args, "reciprocity") is None


def test_phase_done_gate_verdict_semantics(tmp_path):
    args = _fc_args(tmp_path, "repro-gate")
    out = FC.out_eval_dir(args)
    # a recorded FAIL never skips — the designed rc-21 halt re-evaluates on re-run
    FC.atomic_json(out / "repro_gate.json", {"verdict": "FAIL", "metrics": {}, "n_train": 1})
    assert FC.phase_done(args, "repro-gate") is None
    FC.atomic_json(out / "repro_gate.json", {"verdict": "PASS", "metrics": {}, "n_train": 1})
    assert "PASS" in (FC.phase_done(args, "repro-gate") or "")
    # smoke rebinds out_eval -> production state never satisfies the smoke regime
    args_smoke = _fc_args(tmp_path, "repro-gate", ["--smoke"])
    assert FC.phase_done(args_smoke, "repro-gate") is None
    # the terminal upload phase NEVER skips (idempotent safety pass)
    assert FC.phase_done(args, "upload") is None


def test_main_dispatch_skips_done_and_force_reruns(tmp_path, monkeypatch):
    args = _fc_args(tmp_path, "reciprocity")
    _write_reciprocity_done(args)
    calls: list[str] = []
    monkeypatch.setitem(FC.PHASES, "reciprocity", lambda a: calls.append("ran"))
    base = [
        "--phase",
        "reciprocity",
        "--out-eval",
        str(tmp_path / "eval"),
        "--work-root",
        str(tmp_path / "wr"),
        "--sentinel-dir",
        str(tmp_path / "sent"),
        "--no-upload",
        "--no-git",
    ]
    assert FC.main(base) == 0
    assert calls == []  # skip-if-done fired at dispatch
    assert FC.main([*base, "--force"]) == 0
    assert calls == ["ran"]  # --force overrides the guard
