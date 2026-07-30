"""#1775 engine pins: batched permutation nulls == serial #763 reference,
fit_press_pairs == the banked _fit_cv on complement pairs, LOO prefix means,
derangement validity, cluster-bootstrap point identity, errorbar offsets.

All synthetic + CPU + seconds-fast; no network, no store reads.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

i1775 = pytest.importorskip("issue1775_common")


def _toy(n=60, d=8, p=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    W = rng.standard_normal((d, p))
    Y = X @ W + 0.3 * rng.standard_normal((n, p))
    rows = [
        {"prefix_id": f"pf{i % 6}", "query_id": f"q{i % 10}", "stratum": "dense_core"}
        for i in range(n)
    ]
    return X, Y, rows


def test_fit_press_pairs_matches_fit_cv():
    """Complement pairs -> numerically identical to the banked _fit_cv engine."""
    X, Y, rows = _toy()
    folds = i1775._folds_from_manifest(rows, len(rows), group_key="prefix_id", n_folds=3)
    ref = i1775._fit_cv(X, Y, folds)
    pairs = i1775.fold_pairs(rows, len(rows), "prefix", n_folds=3)
    got, _pred, _cov = i1775.fit_press_pairs(X, Y, pairs)
    assert abs(got["r2"] - ref["r2"]) < 1e-12
    assert got["lambda_indices"] == ref["lambda_indices"]


def test_batched_null_matches_serial_reference():
    """One batched permuted draw == hsic/dcor computed serially on permuted rows."""
    X, Y, _rows = _toy(n=40)
    R = Y[:, :3]
    mats = i1775.build_dependence_matrices(X, R, device="cpu")
    obs = i1775.observed_stats(mats)
    assert abs(obs["hsic"] - i1775.hsic_statistic(X, R)) < 1e-8
    assert abs(obs["dcor"] - i1775.distance_correlation(X, R)) < 1e-6
    rng = np.random.default_rng(3)
    perm = rng.permutation(40)
    got = i1775.null_stats_batched(mats, perm[None, :])
    # serial reference on permuted residual rows: kernels recomputed from scratch
    ref_h = i1775.hsic_statistic(X, R[perm])
    ref_d = i1775.distance_correlation(X, R[perm])
    assert abs(float(got["hsic"][0]) - ref_h) < 1e-6
    assert abs(float(got["dcor"][0]) - ref_d) < 1e-5


def test_crossed_permutations_shapes_and_derangement():
    P, Q = 5, 7
    for scheme in ("prefix_block", "query_block", "within_prefix_derangement"):
        perms = i1775.crossed_permutations(P, Q, scheme, 8, seed=1)
        assert perms.shape == (8, P * Q)
        for b in range(8):
            assert sorted(perms[b].tolist()) == list(range(P * Q))
    der = i1775._batched_derangements(np.random.default_rng(0), 50, 6)
    assert not (der == np.arange(6)).any()


def test_cluster_bootstrap_point_identity():
    _X, Y, rows = _toy(n=80)
    rng = np.random.default_rng(1)
    pred_a = Y + 0.1 * rng.standard_normal(Y.shape)
    pred_b = Y + 0.4 * rng.standard_normal(Y.shape)
    groups = np.asarray([r["prefix_id"] for r in rows])
    cov = np.ones(len(rows), dtype=bool)
    out = i1775.cluster_bootstrap_delta_r2(Y, pred_a, pred_b, cov, groups, n_draws=50, seed=0)
    direct = i1775._r2(Y, pred_a) - i1775._r2(Y, pred_b)
    assert abs(out["delta_r2"] - direct) < 1e-10
    assert out["ci95_cluster"][0] <= out["delta_r2"] <= out["ci95_cluster"][1]


def test_loo_prefix_mean_and_singleton_mask():
    X = np.arange(12, dtype=np.float64).reshape(6, 2)
    prefixes = np.asarray(["a", "a", "a", "b", "b", "c"])
    out, mask = i1775._loo_prefix_mean(X, prefixes)
    np.testing.assert_allclose(out[0], X[[1, 2]].mean(0))
    np.testing.assert_allclose(out[3], X[4])
    assert mask.tolist() == [True] * 5 + [False]  # singleton prefix c masked out


def test_holm_correction_monotone():
    p = {"a": 0.01, "b": 0.02, "c": 0.5}
    adj = i1775.holm_correction(p)
    assert adj["a"] == pytest.approx(0.03)
    assert adj["b"] == pytest.approx(0.04)
    assert adj["c"] == pytest.approx(0.5)
    assert adj["a"] <= adj["b"] <= adj["c"]


def test_err_offsets_never_negative_on_inverted_ci():
    """The #547/#1335 xerr class: inverted quantile CI must clamp, then render."""
    figs = pytest.importorskip("issue1775_figures")
    lo, hi = figs._err_offsets(0.5, [0.6, 0.4])  # deliberately INVERTED bounds
    assert lo >= 0.0 and hi >= 0.0
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.errorbar([0.0], [0.5], yerr=[[lo], [hi]])  # must not raise ValueError
    fig.savefig(Path("/tmp") / "i1775_errbar_smoke.png")
    plt.close(fig)


def test_gate_c_constants_pin_battery_excluded_artifact():
    """GATE_C must equal the committed fair_comparison.json battery-EXCLUDED reads.

    Pins the round-2 Critical-1 fix: the coded gate centers on
    .cells.cell_inst_own.bases.<basis>.single_grain.r2_context_battery_excluded_full
    (17,308-row population), NOT the battery-INCLUDED banked pair.
    """
    import json

    artifact = REPO / "eval_results/issue_1092/inline_fair_comparison/fair_comparison.json"
    if not artifact.exists():
        pytest.skip("fair_comparison.json not present (sparse checkout)")
    bases = json.loads(artifact.read_text())["cells"]["cell_inst_own"]["bases"]
    for basis in ("ambient", "pca48"):
        expected = bases[basis]["single_grain"]["r2_context_battery_excluded_full"]
        assert i1775.GATE_C[basis] == expected, (basis, i1775.GATE_C[basis], expected)


def _nl_row_base():
    return dict(
        cell="cell_inst_own",
        layer=14,
        arm="prefix_end",
        grain="perrow",
        scheme="prefix",
        phase="nonlinear",
        smoke=False,
        row_limit=None,
        basis="both",
    )


def test_resume_skips_only_complete_rows_and_purges_partials(tmp_path):
    """#1775 P3 crash-fix regression: 1 valid + 1 partial row in the shard
    JSONL -> exactly 1 unit resume-skipped, and the partial row is PURGED."""
    import json

    ladder = pytest.importorskip("issue1775_ladder")
    valid = {
        **_nl_row_base(),
        "rung": "krr",
        "seed": 0,
        "per_fold": [
            {
                "fold": f,
                "gamma_mult": 1.0,
                "lambda": 1e-3,
                "sigma0": 1.0,
                "r2_fold": {"pca48": 0.2},
            }
            for f in range(5)
        ],
        "r2": {"pca48": 0.2},
        "wall_s": 1.0,
    }
    # the crashed-run shape: rff stub with per_fold {fold, lambda, seed} only,
    # no top-level r2 / wall_s — must NOT mark its unit done
    partial = {
        **_nl_row_base(),
        "rung": "rff",
        "seed": 1,
        "per_fold": [{"fold": 0, "lambda": 1e-3, "seed": 1}],
    }
    p = tmp_path / "units_nonlinear_shard0.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in (valid, partial)))
    rows = i1775.load_units_validated(p, ladder.unit_row_incomplete)
    done = {i1775.unit_key(d, ladder.REGIME_KEYS) for d in rows}
    units = ladder.nonlinear_units(False, [0, 1, 2], set())
    for u in units:
        u["smoke"] = False
        u["row_limit"] = None
    todo = [u for u in units if i1775.unit_key(u, ladder.REGIME_KEYS) not in done]
    assert len(units) - len(todo) == 1  # ONLY the valid krr row resume-skips
    kept = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
    assert len(kept) == 1 and kept[0]["rung"] == "krr"  # partial row purged


def test_nonlinear_group_sharding_keeps_rung_order():
    """#1775 P3 crash-fix: nonlinear shards at the (arm, grain, scheme) group
    grain — every rff/mlp unit's krr sibling precedes it ON THE SAME SHARD
    (rff/mlp read the krr gamma record; the index-interleave raced it)."""
    ladder = pytest.importorskip("issue1775_ladder")
    units = ladder.nonlinear_units(False, [0, 1, 2], set())
    for u in units:
        u["smoke"] = False
        u["row_limit"] = None
    seen: list[tuple] = []
    sizes = []
    for si in (0, 1):
        shard = ladder.shard_units(units, "nonlinear", 2, si)
        ladder.verify_shard_rung_order(shard)  # must not raise
        seen.extend(i1775.unit_key(u, ladder.REGIME_KEYS) for u in shard)
        sizes.append(len(shard))
    # exact partition: no unit lost, none duplicated; groups balance the load
    assert len(seen) == len(set(seen)) == len(units)
    assert min(sizes) > 0 and abs(sizes[0] - sizes[1]) <= 5
    # the pre-fix index-interleave violates the invariant (odd shard has rff
    # units whose krr sibling landed on the even shard)
    interleaved = [u for i, u in enumerate(units) if i % 2 == 1]
    with pytest.raises(RuntimeError, match="shard ordering violation"):
        ladder.verify_shard_rung_order(interleaved)


def test_bilinear_resume_purges_partials_and_gate_flags_missing(tmp_path):
    """#1775 P4 port of the P3 crash-fix (r3 persisted concern
    p4-bilinear-raw-resume-no-completeness-gate): 1 valid + 1 partial row in
    the shard JSONL -> only the valid row resume-skips, the partial is PURGED,
    and the assembly completeness gate flags exactly the partial's unit."""
    import json

    bilin = pytest.importorskip("issue1775_bilinear")
    base = dict(scheme="prefix", r=0, basis="pca48", smoke=False, row_limit=None)
    valid = {
        **base,
        "fold": 0,
        "epochs_ran": 12,
        "variants": [{"seed": 0, "wd": 0.0, "inner_val_mse": 0.1, "r2_te": 0.2}],
    }
    # crash-shaped stub: row appended schema-incomplete (no epochs_ran/variants)
    partial = {**base, "fold": 1}
    p = tmp_path / "units_shard0.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in (valid, partial)))
    rows = i1775.load_units_validated(p, bilin.bilinear_row_incomplete)
    by_key = {i1775.unit_key(r, bilin.REGIME_KEYS): r for r in rows}
    planned = [{**base, "fold": f} for f in (0, 1)]
    missing = [u for u in planned if i1775.unit_key(u, bilin.REGIME_KEYS) not in by_key]
    assert len(rows) == 1 and rows[0]["fold"] == 0  # only the complete row resumes
    assert [u["fold"] for u in missing] == [1]  # the gate flags exactly the stub's unit
    kept = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
    assert len(kept) == 1 and kept[0]["fold"] == 0  # partial row purged from disk


def test_prefetch_ridge_coverage_fails_loud_on_partial_set(tmp_path, monkeypatch):
    """#1775 round-3 Minor: a PARTIAL ridge-pred set (one expected pred+mask pair
    absent) raises naming the missing key; the complete set passes."""
    ladder = pytest.importorskip("issue1775_ladder")
    monkeypatch.setenv("I1775_OUT_ROOT", str(tmp_path))
    expected = ladder.expected_ridge_pred_files(False)
    # production enumeration: 6 perrow combos x {ambient, pca48}
    assert len(expected) == 12
    assert len(ladder.expected_ridge_pred_files(True)) == 1  # smoke: stitch|perrow|prefix|pca48
    items = sorted(expected.items())
    for _key, (pred, mask) in items[:-1]:
        pred.write_bytes(b"")
        mask.write_bytes(b"")
    missing_key, (pred, mask) = items[-1]
    with pytest.raises(RuntimeError, match="INCOMPLETE") as ei:
        i1775.assert_p3_ridge_pred_coverage(False)
    assert missing_key in str(ei.value)  # the missing (arm|grain|scheme|basis) key is named
    pred.write_bytes(b"")
    mask.write_bytes(b"")
    i1775.assert_p3_ridge_pred_coverage(False)  # complete set: must not raise


def test_doubly_fold_pairs_disjoint():
    _X, _Y, rows = _toy(n=90)
    pairs = i1775.fold_pairs(rows, len(rows), "doubly", n_folds=3)
    assert pairs, "doubly scheme produced no usable pairs"
    for tr, te in pairs:
        assert not set(tr.tolist()) & set(te.tolist())
        te_prefixes = {rows[i]["prefix_id"] for i in te}
        te_queries = {rows[i]["query_id"] for i in te}
        assert not any(rows[i]["prefix_id"] in te_prefixes for i in tr)
        assert not any(rows[i]["query_id"] in te_queries for i in tr)
