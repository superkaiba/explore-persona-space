"""Issue #1482 driver pins: data-dependent gate probes (degenerate inputs), the
assemble-mirror equivalence pin, the shared-Gram ridge parity gate, and the pure
verdict lattices. CPU-only; no network, no model loads."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import issue779_ffc_n1m_fits as N1M  # noqa: E402
import issue779_ffc_n1m_generate_capture as N1G  # noqa: E402
import issue779_ffc_n50k_fits as N50  # noqa: E402
import issue779_percontext_recon as PR  # noqa: E402
import issue1482_error_analysis as D  # noqa: E402
import issue1482_sae as S  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))


# ── assemble mirror: byte-equivalence vs the UNCHANGED parent (reuse check k) ────


def _tiny_world(monkeypatch, tmp_path, n_new=20):
    """Monkeypatch the DATA boundary (pass_b bundle / manifest / stream / pins) with
    real-typed synthetic objects; the parent + mirror then run their REAL bodies."""
    h = 3584
    n_pb = N1M.N_PASS_B
    rng = np.random.default_rng(0)
    pb = {
        "cx_last": torch.tensor(rng.standard_normal((n_pb, 1, h)).astype(np.float32)),
        "v_x": torch.tensor(rng.standard_normal((n_pb, 1, h)).astype(np.float32)),
        "layers": [19],
    }
    new_X = rng.standard_normal((n_new, h)).astype(np.float32)
    new_Y = rng.standard_normal((n_new, h)).astype(np.float32)
    new_ci = np.arange(n_new, dtype=np.int64)
    pool = [
        {"i": i, "corpus": ("lmsys" if i % 2 == 0 else "wildchat"), "prompt": "x"}
        for i in range(n_new)
    ]
    meta = {"n_new": n_new, "n_lmsys": n_new // 2, "n_wildchat": n_new - n_new // 2, "n_parts": 1}

    monkeypatch.setattr(N1G, "_load_pass_b_bundle", lambda p: pb)
    monkeypatch.setattr(N1G, "_resolve_manifest_dir", lambda a: tmp_path)
    monkeypatch.setattr(N1G, "read_manifest_pool", lambda d: (pool, meta))
    monkeypatch.setattr(
        N1M, "_stream_n1m_layer", lambda *a, **k: (new_X.copy(), new_Y.copy(), new_ci.copy())
    )
    import issue779_fitter_fair_comparison as F

    _r1_train, val, test = F.fixed_split(
        n_pb, n_pb - N1M.N_VAL - N1M.N_TEST, N1M.N_VAL, N1M.N_TEST, N1M.SPLIT_SEED
    )
    pins = {"val_sha256": F._sha_ids(val), "test_sha256": F._sha_ids(test), "source": "test"}
    monkeypatch.setattr(N50, "_pinned_original_shas", lambda d: pins)
    ns = SimpleNamespace(
        pass_b=tmp_path / "pb.pt",
        out_dir=tmp_path,
        manifest_from_hf=False,
        hf_prefix="x",
        manifest_hf_prefix="x",
        n1m_capture_dir=tmp_path,  # local branch (stream fn is patched anyway)
        fresh_stream=False,
        orig_dir=tmp_path,
    )
    return ns


def test_assemble_mirror_matches_parent(monkeypatch, tmp_path):
    ns = _tiny_world(monkeypatch, tmp_path)
    Xp, Yp, provp, trp, valp, testp, splitp = N1M.assemble(ns, 19)
    Xm, Ym, provm, trm, valm, testm, splitm, new_ci = D._assemble_with_ci(ns, 19)
    assert np.array_equal(Xp, Xm) and np.array_equal(Yp, Ym)
    assert np.array_equal(provp, provm)
    assert np.array_equal(trp, trm) and np.array_equal(valp, valm) and np.array_equal(testp, testm)
    for k in ("val_sha256", "test_sha256", "n_new_captured"):
        assert splitp[k] == splitm[k], k
    assert np.array_equal(new_ci, np.arange(20))


# ── data-dependent gate probes (degenerate inputs; gates fire OUTSIDE the smoke) ──


def test_parent_train_pool_as_eval_guard_raises():
    X = np.random.default_rng(0).standard_normal((30, 4)).astype(np.float32)
    Y = X.copy()
    tr = np.arange(20)
    with pytest.raises(ValueError, match="train pool passed as an eval set"):
        N1M._ridge_streaming_multi_lambda(X, Y, tr, [tr], [1.0], "cpu", 16)


def test_manifest_index_drift_gate(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"n_new": 1, "n_parts": 1}))
    (tmp_path / "part_000.jsonl").write_text(json.dumps({"i": 5, "corpus": "lmsys"}) + "\n")
    with pytest.raises(SystemExit, match="index drift"):
        N1G.read_manifest_pool(tmp_path)


def _fake_gate_a_inputs(tmp_path, ridge_delta=0.0, krr_delta=0.0, seed=0):
    committed = json.loads(
        (REPO / "eval_results/issue_779/fitter-fair-comparison-n1m/n1m_fits.json").read_text()
    )
    preds = committed["per_point"]["mixed_1m"]["predictors"]
    pdir = tmp_path / "percontext"
    pdir.mkdir(parents=True, exist_ok=True)
    for pred in N1M.PREDICTORS:
        want = preds[pred]["whole_map_r2"]
        got = want + (
            ridge_delta if pred == "ridge" else (krr_delta if pred == "krr_nystrom" else 0.0)
        )
        (pdir / f"refit_full__{pred}__seed{seed}.json").write_text(
            json.dumps({"sets": {"test": {"whole_map_r2": got}}})
        )
    return SimpleNamespace(out_eval=tmp_path, smoke=False, seed=seed)


def test_gate_a_halt_on_ridge_miss(tmp_path):
    args = _fake_gate_a_inputs(tmp_path, ridge_delta=0.05)
    with pytest.raises(SystemExit) as e:
        D.gate_a_check(args)
    assert e.value.code == D.RC_GATE_A


def test_gate_a_peripheral_miss_warns_and_drops(tmp_path):
    args = _fake_gate_a_inputs(tmp_path, krr_delta=0.05)
    doc = D.gate_a_check(args)
    assert doc["verdict"] == "WARN_DROP" and doc["dropped_arms"] == ["krr_nystrom"]


def test_gate_a_smoke_demotes_halt_to_informational(tmp_path):
    args = _fake_gate_a_inputs(tmp_path, ridge_delta=0.05)
    args.smoke = True
    doc = D.gate_a_check(args)  # no SystemExit under smoke (#1345 gate-calibration rule)
    assert doc["verdict"] == "HALT" and doc["smoke_demoted"] is True


def test_gate_b_verdict_lattice():
    assert D.gate_b_verdict(0.75, 0.5) == ("PASS", 64)
    assert D.gate_b_verdict(0.60, 0.72) == ("WARN", 128)
    assert D.gate_b_verdict(0.60, 0.65) == ("WARN", 64)
    assert D.gate_b_verdict(0.40, 0.90) == ("HALT", 64)


def test_prefix_constancy_probe_fires_on_perturbed_state():
    hp = torch.ones((4, 8))
    assert D.prefix_constancy_cos_min(hp) >= D.PREFIX_CONSTANCY_COS_MIN
    hp2 = hp.clone()
    hp2[2, 0] = -50.0  # a genuinely different prefix state
    assert D.prefix_constancy_cos_min(hp2) < D.PREFIX_CONSTANCY_COS_MIN


# ── registry + carve determinism ─────────────────────────────────────────────────


def test_fit_specs_registry_shape():
    specs = D.fit_specs(SimpleNamespace(seed=0))
    ids = [s["fit_id"] for s in specs]
    assert len(ids) == len(set(ids)) == 13
    assert all(s["condition"] == "refit_full" for s in specs[:5])  # early Gate A ordering
    assert sum(s["seed"] == D.MLP_SEED_B for s in specs) == 1


def test_stratified_sample_deterministic_and_rebalances():
    rng1 = np.random.default_rng(D.SPLIT_SEED_1482)
    rng2 = np.random.default_rng(D.SPLIT_SEED_1482)
    rows = np.arange(100)
    prov = (rows % 4 == 0).astype(np.uint8)  # 25% wildchat
    a, da = D._stratified_sample(rng1, rows, prov, 40, 0.75)
    b, _ = D._stratified_sample(rng2, rows, prov, 40, 0.75)
    assert np.array_equal(a, b) and da["n"] == 40
    # shortfall rebalance: ask for more wildchat than exists
    _c, dc = D._stratified_sample(np.random.default_rng(1), rows, prov, 40, 0.05)
    assert dc["n"] == 40 and dc["n_wildchat"] <= int(prov.sum())


# ── per-context decomposition + shared-Gram ridge parity ─────────────────────────


def test_percontext_reconciliation_identity():
    rng = np.random.default_rng(3)
    t = rng.standard_normal((50, 8))
    p = t + 0.3 * rng.standard_normal((50, 8))
    pc = D._percontext(p, t)
    recon = 1.0 - pc["e2"].sum() / pc["denom"].sum()
    assert abs(recon - PR._pooled_r2(p, t)) < 1e-9


def test_shared_gram_ridge_multi_matches_parent_fit_ridge():
    rng = np.random.default_rng(7)
    Z = rng.standard_normal((120, 10)).astype(np.float32)
    W = rng.standard_normal((10, 6)).astype(np.float32)
    T = (Z @ W + 0.1 * rng.standard_normal((120, 6))).astype(np.float32)
    tr, va, te = np.arange(80), np.arange(80, 100), np.arange(100, 120)
    lambdas = [0.1, 1.0, 10.0]
    out = D._shared_gram_ridge_multi(Z, {"t": T}, tr, va, te, lambdas, "cpu", 32)
    pt, meta = out["t"]
    pt_parent, meta_parent = N1M.fit_ridge(Z, T, tr, va, te, lambdas, "cpu", 32)
    assert meta["selected_lambda"] == meta_parent["selected_lambda"]
    assert np.allclose(pt, pt_parent, atol=1e-8)


def test_per_feature_metrics_match_naive_loop():
    rng = np.random.default_rng(11)
    t = rng.standard_normal((40, 5))
    p = t + rng.standard_normal((40, 5))
    pf = D._per_feature_metrics(p, t)
    from scipy.stats import spearmanr

    for j in range(5):
        r2_naive = 1.0 - ((t[:, j] - p[:, j]) ** 2).sum() / ((t[:, j] - t[:, j].mean()) ** 2).sum()
        assert abs(pf["r2"][j] - r2_naive) < 1e-9
        rho = spearmanr(p[:, j], t[:, j]).statistic
        assert abs(pf["spearman"][j] - rho) < 1e-9


# ── store round-trip + SAE pure helpers ──────────────────────────────────────────


def test_sparsify_densify_roundtrip():
    f = torch.zeros((3, 50))
    f[0, 5] = 1.5
    f[1, 5] = 0.5
    f[2, 40] = 2.0
    pooled = S.pool_answer_features(f)
    sp = S.sparsify(pooled)
    assert set(sp["idx"].tolist()) == {5, 40}
    assert pytest.approx(float(pooled["frac"][5]), abs=1e-6) == 2 / 3
    part = {
        "row_idx": np.array([7]),
        "set_tag": np.array([1], dtype=np.int8),
        "idx_off": np.array([len(sp["idx"])]),
        "ans_idx": sp["idx"],
        "ans_mean": sp["mean"],
    }
    M = D._densify([part], "ans_idx", "idx_off", "ans_mean", np.array([5, 40]), 1, {7: 0})
    assert M.shape == (1, 2) and M[0, 0] > 0 and M[0, 1] > 0
    counts, n = D._activity_counts([part], "ans_idx", "idx_off", only_tag=1, dict_size=50)
    assert n == 1 and counts[5] == 1 and counts[40] == 1 and counts.sum() == 2


# ── P6 statistics: empty groups, kappa, FDR, errorbar clamp, label validation ────


def test_analysis_stats_edge_cases():
    import issue1482_analysis as A

    # empty-group H1: bootstrap helper stays finite-safe
    nerr = np.abs(np.random.default_rng(0).standard_normal(30))
    d = A._boot_group_delta(nerr, nerr > 0.5, nerr <= 0.5, 50, 0)
    assert len(d) > 0
    # degenerate permutation contrast -> nan p (skipped by FDR)
    p = A._perm_pvals(nerr, [np.zeros(30, bool)], 20, 0)
    assert np.isnan(p[0])
    assert A._bh_fdr([float("nan")]) == [False]
    assert (
        A._bh_fdr([0.001, 0.5, float("nan")])[0] is True or A._bh_fdr([0.001, 0.5, float("nan")])[0]
    )
    # kappa: degenerate + perfect
    assert np.isnan(A._cohens_kappa(["a"], ["a"]))
    assert A._cohens_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == pytest.approx(1.0)
    # label validation drops (never coerces): bad topic, non-dict, bare scalar
    good = {
        "language": "EN",
        "topic": "coding",
        "request_refusal_adjacent": "no",
        "answer_is_refusal": "no",
        "format": "code",
    }
    assert A._validate_label(good)["language"] == "en"
    assert A._validate_label({**good, "topic": "nonsense"}) is None
    assert A._validate_label(85) is None
    assert A._validate_label(None) is None
    # inverted quantile CI -> clamped non-negative offsets (gotchas #547/#1335)
    lo, hi = A._errbars([1.0], [1.1], [0.9])
    assert (lo >= 0).all() and (hi >= 0).all()
