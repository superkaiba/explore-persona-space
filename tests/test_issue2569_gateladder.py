"""Unit tests for scripts/issue2569_gateladder.py — #2569 leg 2 (unit 2 of the build).

Covers both halves of plan §4 leg 2:

- the gate-metric ladder: the six rungs' formulas (centered bilinear incumbents,
  through-map image rungs built from the REGISTERED prediction difference — B1
  assert iii), the selection-symmetric race (winner re-selected per bootstrap
  draw; permutation per-draw signed max), the champion's selection-inherited vs
  frozen-at-winner intervals, per-family win tables, and the full ladder driver
  against a schema-exact synthetic fixture tree (no network);
- the B4 learning curve: nested conversation-disjoint LMSYS splits, the
  widen-on-edge lambda protocol (C4), the arXiv 2006.13198 self-consistent
  theory (validated against the isotropic closed form AND the empirical
  ``fit_ridge_primal`` on synthetic Gaussian data), the streaming identity+bias
  parity with the canonical ``mapping_baselines`` helper, the mechanical
  fit-metadata parity check, and the committed off-recipe companion loader.

All synthetic tests are network-free; the companion/frames tests read only
git-committed ``eval_results`` JSONs at the repo root.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2569_gateladder as GL  # noqa: E402
import issue2569_operator as OP  # noqa: E402

D_SYN = 24  # synthetic residual dim (small, fast; distinct from n_prefix)
N_PREFIX = 12


def _synth_payload(d: int = D_SYN, seed: int = 7) -> OP.MapPayload:
    """A synthetic MapPayload in the vendored contract shape (no disk)."""
    rng = np.random.default_rng(seed)
    raw = {
        "kind": "ridge",
        "fitter": "ridge",
        "layer": 19,
        "selected_lambda": 0.001,
        "W": torch.tensor(rng.standard_normal((d, d)), dtype=torch.float32),
        "xmu": torch.tensor(rng.standard_normal(d), dtype=torch.float32),
        "xsd": torch.tensor(np.exp(rng.standard_normal(d) * 0.5), dtype=torch.float32),
        "ymu": torch.tensor(rng.standard_normal(d), dtype=torch.float32),
    }
    return OP.MapPayload(
        layer=19,
        path=Path("<synthetic>"),
        W=np.asarray(raw["W"], dtype=np.float64),
        xmu=np.asarray(raw["xmu"], dtype=np.float64),
        xsd=np.asarray(raw["xsd"], dtype=np.float64),
        ymu=np.asarray(raw["ymu"], dtype=np.float64),
        selected_lambda=0.001,
        raw=raw,
    )


def _synth_sigma(d: int, seed: int) -> np.ndarray:
    """A well-conditioned synthetic covariance."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((6 * d, d))
    return Z.T @ Z / (6 * d)


# ── gate rungs ────────────────────────────────────────────────────────────────


def test_shrink_sigma_matches_1979_recipe():
    """shrink_sigma == (1-s)*Sigma + s*(tr/d)*I, the issue1768_directions recipe."""
    sigma = _synth_sigma(D_SYN, 0)
    s = 0.1
    got = GL.shrink_sigma(sigma, s)
    want = 0.9 * sigma + 0.1 * (np.trace(sigma) / D_SYN) * np.eye(D_SYN)
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_gate_scores_incumbents_match_manual():
    """gate_I / gate_diag_inv / gate_sigma_inv equal the centered bilinear forms."""
    rng = np.random.default_rng(1)
    C = rng.standard_normal((N_PREFIX, D_SYN))
    a = rng.standard_normal(D_SYN)
    sigma_c = _synth_sigma(D_SYN, 2)
    sc = GL.gate_scores(C, a, None, sigma_c=sigma_c)
    mu = C.mean(0)
    Cc, ac = C - mu, a - mu
    np.testing.assert_allclose(sc["gate_I"], Cc @ ac, rtol=1e-12)
    np.testing.assert_allclose(sc["gate_diag_inv"], Cc @ (ac / np.diag(sigma_c)), rtol=1e-12)
    np.testing.assert_allclose(
        sc["gate_sigma_inv"], Cc @ np.linalg.solve(GL.shrink_sigma(sigma_c), ac), rtol=1e-10
    )
    # through-map rungs unavailable without a payload — None, never fabricated
    assert sc["gate_wwt"] is None and sc["gate_wwt_k90"] is None and sc["gate_wwt_awhite"] is None


def test_gate_scores_partial_mode_without_sigma():
    """P-A partial mode: no sigma => only the moment-free rungs compute."""
    rng = np.random.default_rng(3)
    payload = _synth_payload()
    C = rng.standard_normal((N_PREFIX, D_SYN))
    a = rng.standard_normal(D_SYN)
    sc = GL.gate_scores(C, a, payload)
    assert sc["gate_I"] is not None and sc["gate_wwt"] is not None
    assert sc["gate_diag_inv"] is None and sc["gate_sigma_inv"] is None
    assert sc["gate_wwt_awhite"] is None


def test_through_map_rungs_equal_registered_apply_path():
    """B1 assert iii: through-map rungs reduce to registered prediction differences."""
    rng = np.random.default_rng(4)
    payload = _synth_payload()
    C = rng.standard_normal((N_PREFIX, D_SYN))
    a = rng.standard_normal(D_SYN)
    sigma_a = _synth_sigma(D_SYN, 5)
    _u, s, vh = np.linalg.svd(payload.W)
    k = GL.k90_count(s)
    sc = GL.gate_scores(
        C, a, payload, sigma_a=sigma_a, w_right_singular=vh.T, k_trunc=k, probe_assert=True
    )
    mu = C.mean(0)
    img = ((C - mu) / payload.xsd) @ payload.W
    img_a = ((a - mu) / payload.xsd) @ payload.W
    np.testing.assert_allclose(sc["gate_wwt"], img @ img_a, rtol=1e-9)
    # truncated rung = images projected onto W's top-k RIGHT singular subspace
    Vk = vh.T[:, :k]
    np.testing.assert_allclose(sc["gate_wwt_k90"], (img @ Vk) @ (Vk.T @ img_a), rtol=1e-9)
    # whitened rung = images whitened by the SHRUNK Sigma_a inverse
    np.testing.assert_allclose(
        sc["gate_wwt_awhite"],
        img @ np.linalg.solve(GL.shrink_sigma(sigma_a), img_a),
        rtol=1e-9,
    )
    # the rungs are also the prediction-difference inner products (assert iii)
    pd_c = OP.prediction_difference(payload, C, mu)
    pd_a = OP.prediction_difference(payload, a, mu)
    np.testing.assert_allclose(sc["gate_wwt"], pd_c @ pd_a, rtol=1e-9)


def test_through_map_probe_assert_trips_on_divergence():
    """The B1 probe assert raises when the images diverge from the row-operator form."""
    rng = np.random.default_rng(6)
    payload = _synth_payload()
    C = rng.standard_normal((N_PREFIX, D_SYN))
    a = rng.standard_normal(D_SYN)
    mu = C.mean(0)
    img_c = OP.prediction_difference(payload, C, mu)
    img_a = OP.prediction_difference(payload, a, mu)
    good = img_c @ img_a
    with pytest.raises(AssertionError, match="b1-assert-iii"):
        GL._assert_through_map_probe(payload, C - mu, a - mu, img_c * 1.01, img_a, good)


def test_k90_count_matches_mass_convention():
    """k90 = smallest rank reaching 90% cumulative sigma^2 mass (leg-1 convention)."""
    s = np.array([3.0, 1.0, 0.5, 0.1])  # masses: 9, 1, .25, .01 (total 10.26)
    cum = np.cumsum(s**2) / np.sum(s**2)
    want = int(np.searchsorted(cum, 0.90) + 1)
    assert GL.k90_count(s) == want == 2


# ── race + champion ───────────────────────────────────────────────────────────


def _race_fixture(n: int = 24, seed: int = 9):
    """Scores where gate_wwt is exactly rank-aligned with the primary DV."""
    rng = np.random.default_rng(seed)
    wwt = rng.standard_normal(n)
    scores = {"gate_I": rng.standard_normal(n), "gate_wwt": wwt}
    dv = np.column_stack([np.argsort(np.argsort(wwt)).astype(float), rng.standard_normal(n)])
    return scores, dv


def test_ladder_race_observed_rho_and_perm_band():
    """Observed rho hits 1.0 for the rank-aligned metric; perm band is selection-max."""
    scores, dv = _race_fixture()
    res = GL.ladder_race(
        scores,
        dv,
        ("dv_change", "dv_level"),
        np.arange(24),
        arm_id="syn-arm",
        b_draws=200,
        n_perm=300,
    )
    assert res["observed_rho"]["dv_change"]["gate_wwt"] == pytest.approx(1.0, abs=1e-9)
    assert abs(res["observed_rho"]["dv_change"]["gate_I"]) < 0.6
    assert res["perm_band"]["p975_max_selected"] >= res["perm_band"]["p95_max_selected"]
    assert 0.0 < res["perm_band"]["p975_max_selected"] < 1.0
    assert res["boot"].shape == (200, 2, 2)
    assert res["perm"].shape == (300, 2)


def test_ladder_champion_selection_symmetry_and_intervals():
    """Winner re-selected per draw; selection-inherited CI dominates the frozen CI."""
    arm_results = {}
    for i in range(3):
        scores, dv = _race_fixture(seed=20 + i)
        arm_results[f"arm{i}"] = GL.ladder_race(
            scores,
            dv,
            ("dv_change", "dv_level"),
            np.arange(24),
            arm_id=f"arm{i}",
            b_draws=300,
            n_perm=50,
        )
    ch = GL.ladder_champion(arm_results, dv_index=0, dv_label="dv_change", incumbent="gate_I")
    assert ch["winner_observed"] == "gate_wwt"
    assert ch["p_win"]["gate_wwt"] > 0.9
    assert sum(ch["p_win"].values()) == pytest.approx(1.0, abs=1e-9)
    sel = ch["selection_inherited_ci_max_median"]
    frz = ch["frozen_ci_winner_median (labeled: frozen-at-winner)"]
    assert sel[0] >= frz[0] - 1e-12 and sel[1] >= frz[1] - 1e-12
    band = ch["champion_vs_incumbent_conditional_ceiling_interval"]
    assert band is not None and band[0] <= band[1]


def test_pairwise_win_counts():
    """Strict per-arm win counting between metric pairs."""
    arm_results = {
        "a1": {
            "dv_names": ["dv_change"],
            "observed_rho": {"dv_change": {"gate_wwt": 0.5, "gate_I": 0.3}},
        },
        "a2": {
            "dv_names": ["dv_change"],
            "observed_rho": {"dv_change": {"gate_wwt": 0.2, "gate_I": 0.4}},
        },
    }
    out = GL.pairwise_win_counts(arm_results, 0, (("gate_wwt", "gate_I"),))
    assert out["gate_wwt_vs_gate_I"] == {
        "wins": 1,
        "n_arms": 2,
        "per_arm": {
            "a1": {"gate_wwt": 0.5, "gate_I": 0.3},
            "a2": {"gate_wwt": 0.2, "gate_I": 0.4},
        },
    }


def test_per_family_win_table():
    """Per-family winners split correctly; sub-floor families reported skipped."""
    rng = np.random.default_rng(31)
    n = 12
    fams = ["f1"] * 6 + ["f2"] * 5 + ["tiny"]
    base = rng.standard_normal(n)
    dv = np.argsort(np.argsort(base)).astype(float)
    anti = -base
    # metric A tracks dv on f1 rows, metric B on f2 rows
    mA = np.where(np.asarray(fams) == "f1", base, anti)
    mB = np.where(np.asarray(fams) == "f2", base, anti)
    scores = {"arm": {"gate_wwt": mA, "gate_I": mB}}
    table = GL.per_family_win_table(scores, {"arm": dv}, {"arm": fams}, min_n=4)
    assert table["families"]["f1"]["winner"] == "gate_wwt"
    assert table["families"]["f2"]["winner"] == "gate_I"
    assert "skipped" in table["families"]["tiny"]


# ── banked #1979 inputs (committed, read-only) ────────────────────────────────


def test_load_banked_frames_committed():
    """The 18 committed #1979 frames load with the registered DV columns."""
    cfg = GL.load_1979_config(REPO_ROOT / "eval_results/issue_1979/config")
    frames = GL.load_banked_frames(REPO_ROOT / "eval_results/issue_1979/race", cfg["arms"])
    assert len(frames) == 18
    kinds = {f["kind"] for f in frames.values()}
    assert kinds == {"content", "marker"}
    content = [f for f in frames.values() if f["kind"] == "content"]
    marker = [f for f in frames.values() if f["kind"] == "marker"]
    assert len(content) == 12 and len(marker) == 6
    for f in frames.values():
        assert len(f["prefix_ids"]) == 50
        assert f["dvs"].shape == (50, 2)
        assert np.isfinite(f["dvs"]).all()
    assert content[0]["dv_names"] == ("dv_change", "dv_level")
    assert marker[0]["dv_names"] == ("dv_dlogp", "dv_level_logp")


def test_load_sigma_file_gram_form(tmp_path):
    """The gram/mean/n_rows moment triple centers to the population covariance."""
    rng = np.random.default_rng(40)
    Z = rng.standard_normal((500, 8))
    gram = Z.T @ Z
    mean = Z.mean(0)
    np.savez(tmp_path / "mom.npz", gram=gram, mean=mean, n_rows=np.array(500))
    sigma = GL.load_sigma_file(tmp_path / "mom.npz")
    want = Z.T @ Z / 500 - np.outer(mean, mean)
    np.testing.assert_allclose(sigma, 0.5 * (want + want.T), rtol=1e-10)


# ── full ladder driver on a synthetic fixture tree (no network) ───────────────


def _write_ladder_fixture(root: Path, d: int = D_SYN, n_prefix: int = N_PREFIX) -> dict:
    """Schema-exact synthetic #1979 fixture tree + banked-map payload."""
    rng = np.random.default_rng(77)
    prefix_ids = [f"pfx{i:02d}" for i in range(n_prefix)]
    fams = ["famA" if i < n_prefix // 2 else "famB" for i in range(n_prefix)]
    cfg_dir = root / "config"
    cfg_dir.mkdir(parents=True)
    members = [
        {"prefix_id": p, "family": f, "content_token_len": 10}
        for p, f in zip(prefix_ids, fams, strict=True)
    ]
    arms = [
        {"arm_id": "c-arm1", "kind": "content", "mix_arm_id": "mix1"},
        {"arm_id": "c-arm2", "kind": "content", "mix_arm_id": "mix1"},
        {"arm_id": "m-arm1", "kind": "marker", "mix_arm_id": "mix2"},
    ]
    (cfg_dir / "prefix_panel.json").write_text(json.dumps({"members": members}))
    (cfg_dir / "arms.json").write_text(json.dumps({"arms": arms}))
    race_dir = root / "race"
    race_dir.mkdir()
    tensors = {}
    for arm in arms:
        aid = arm["arm_id"]
        Cbar = rng.standard_normal((n_prefix, d))
        tensors[f"{aid}/L19/last_prompt/Cbar"] = torch.tensor(Cbar, dtype=torch.float16)
        dv_names = GL.DV_NAMES_BY_KIND[arm["kind"]]
        frame = {
            "prefix_id": prefix_ids,
            "family": fams,
            dv_names[0]: rng.standard_normal(n_prefix).tolist(),
            dv_names[1]: rng.standard_normal(n_prefix).tolist(),
        }
        (race_dir / f"frame_{aid}.json").write_text(
            json.dumps({"frame": frame, "layer": 19, "pos": "last_prompt", "n_realized": n_prefix})
        )
    inputs_root = root / "inputs"
    (inputs_root / "battery").mkdir(parents=True)
    torch.save(tensors, inputs_root / "battery/ingredient_tensors.pt")
    for mix in ("mix1", "mix2"):
        (inputs_root / "anchors" / mix).mkdir(parents=True)
        anc = {
            "L19": {"A_ctx_last_prompt": torch.tensor(rng.standard_normal(d), dtype=torch.float16)}
        }
        torch.save(anc, inputs_root / "anchors" / mix / "anchors.pt")
    # banked-map payload at the vendored relpath under a fixture map root
    map_root = root / "maproot"
    payload_path = OP.banked_map_path(19, root=map_root)
    payload_path.parent.mkdir(parents=True)
    torch.save(_synth_payload(d).raw, payload_path)
    # sigma files (P-B moments contract)
    torch.save({"sigma": torch.tensor(_synth_sigma(d, 50))}, root / "sigma_c.pt")
    torch.save({"sigma": torch.tensor(_synth_sigma(d, 51))}, root / "sigma_a.pt")
    return {
        "config_dir": cfg_dir,
        "race_dir": race_dir,
        "inputs_root": inputs_root,
        "map_root": map_root,
        "sigma_c": root / "sigma_c.pt",
        "sigma_a": root / "sigma_a.pt",
    }


def test_run_ladder_driver_synthetic_fixture(tmp_path):
    """The ladder driver runs the full chain on a fixture tree and emits gate_ladder.json."""
    fx = _write_ladder_fixture(tmp_path)
    out_dir = tmp_path / "out"
    rc = GL.main(
        [
            "ladder",
            "--out",
            str(out_dir),
            "--inputs-root",
            str(fx["inputs_root"]),
            "--config-dir",
            str(fx["config_dir"]),
            "--race-dir",
            str(fx["race_dir"]),
            "--map-root",
            str(fx["map_root"]),
            "--sigma-c",
            str(fx["sigma_c"]),
            "--sigma-a",
            str(fx["sigma_a"]),
            "--smoke",
        ]
    )
    assert rc == 0
    out = json.loads((out_dir / "gate_ladder.json").read_text())
    assert out["regime"]["partial"] is False
    assert out["regime"]["b_draws"] == 100  # --smoke lowers draws, same chain
    for aid in ("c-arm1", "c-arm2", "m-arm1"):
        arm = out["per_arm"][aid]
        assert arm["raced"] == list(GL.GATE_METRICS)
        assert (out_dir / f"ladder_boot_{aid}.npz").exists()
        assert (out_dir / f"ladder_perm_{aid}.npz").exists()
    assert set(out["champion"]) == {"content", "marker"}
    assert out["champion"]["content"]["dv_change"]["incumbent"] == "gate_sigma_inv"
    assert "gate_wwt_vs_gate_sigma_inv" in out["pairwise_win_counts"]["content"]
    boot = np.load(out_dir / "ladder_boot_c-arm1.npz")
    assert boot["rho"].shape == (100, 6, 2)


def test_run_ladder_partial_without_sigma(tmp_path):
    """P-A mode: without sigma files the driver emits gate_ladder_partial.json."""
    fx = _write_ladder_fixture(tmp_path)
    out_dir = tmp_path / "out"
    rc = GL.main(
        [
            "ladder",
            "--out",
            str(out_dir),
            "--inputs-root",
            str(fx["inputs_root"]),
            "--config-dir",
            str(fx["config_dir"]),
            "--race-dir",
            str(fx["race_dir"]),
            "--map-root",
            str(fx["map_root"]),
            "--smoke",
        ]
    )
    assert rc == 0
    out = json.loads((out_dir / "gate_ladder_partial.json").read_text())
    assert out["regime"]["partial"] is True
    raced = out["per_arm"]["c-arm1"]["raced"]
    assert "gate_sigma_inv" not in raced and "gate_wwt" in raced


# ── learning curve: splits + widen protocol ───────────────────────────────────


def _synth_store(n_lmsys: int = 2400, n_wc: int = 600, d: int = 16, seed: int = 5):
    """Synthetic assembled store: X/Y fp32 + corpus tags + conversation ids."""
    rng = np.random.default_rng(11)
    eta = 1.0 / np.arange(1, d + 1) ** 1.5
    B = rng.standard_normal((d, d)) * 0.3
    r = np.random.default_rng(seed)
    n = n_lmsys + n_wc
    X = (r.standard_normal((n, d)) * np.sqrt(eta)).astype(np.float32)
    Y = (X @ B + r.standard_normal((n, d))).astype(np.float32)
    corpus = np.array(["lmsys"] * n_lmsys + ["wildchat"] * n_wc)
    conv = np.arange(n) // 6  # 6 rows per conversation
    return X, Y, corpus, conv


def test_nested_lmsys_splits_properties():
    """Nested subsets, conversation-disjoint eval/val, LMSYS-only, deterministic."""
    _X, _Y, corpus, conv = _synth_store()
    sp = GL.nested_lmsys_splits(
        corpus, conv, n_grid=(100, 300), eval_rows=150, val_rows=100, seed=2569
    )
    assert len(sp["te_idx"]) == 150 and len(sp["val_idx"]) == 100
    tr100, tr300 = sp["tr_by_n"][100], sp["tr_by_n"][300]
    assert len(tr100) == 100 and len(tr300) == 300
    assert set(tr100) <= set(tr300)  # nested
    for ix in (sp["te_idx"], sp["val_idx"], tr300):
        assert (corpus[ix] == "lmsys").all()
    held = set(conv[sp["te_idx"]]) | set(conv[sp["val_idx"]])
    assert not (held & set(conv[tr300]))  # conversation-disjoint
    assert not (set(conv[sp["te_idx"]]) & set(conv[sp["val_idx"]]))
    sp2 = GL.nested_lmsys_splits(
        corpus, conv, n_grid=(100, 300), eval_rows=150, val_rows=100, seed=2569
    )
    assert sp2["eval_split_sha"] == sp["eval_split_sha"]  # deterministic
    np.testing.assert_array_equal(sp2["tr_by_n"][300], tr300)


def test_nested_splits_reject_undersized_pool():
    """A pool smaller than max(n_grid) fails loud, never a silent truncation."""
    _X, _Y, corpus, conv = _synth_store(n_lmsys=300, n_wc=60)
    with pytest.raises(ValueError, match="training pool"):
        GL.nested_lmsys_splits(corpus, conv, n_grid=(5000,), eval_rows=60, val_rows=60)


def test_widen_grid_low_and_high():
    """widen_grid extends two decades at 2/decade on the named side."""
    grid = np.logspace(-1, 2, 7)
    low = GL.widen_grid(grid, "low")
    assert len(low) == 11 and low[0] == pytest.approx(1e-3)
    assert np.all(np.diff(np.log10(low)) == pytest.approx(0.5))
    high = GL.widen_grid(grid, "high")
    assert len(high) == 11 and high[-1] == pytest.approx(1e4)
    with pytest.raises(ValueError):
        GL.widen_grid(grid, "sideways")


def test_fit_point_widens_off_edge_and_reports_interior_lambda():
    """A boundary selection triggers widen-and-reselect; the reported lambda is interior."""
    X, Y, _corpus, _conv = _synth_store()
    tr, val, te = np.arange(600), np.arange(600, 900), np.arange(900, 2400)
    pt = GL.fit_point(
        X, Y, tr, val, te, dev=torch.device("cpu"), grid=(1e4, 10**4.5, 1e5), max_widenings=8
    )
    assert pt["n_widenings"] >= 1
    assert pt["lambda_grid_edge"] is None
    assert pt["final_grid"][0] < pt["selected_lambda"] < pt["final_grid"][1]
    assert 0.0 < pt["test_r2"] < 1.0
    with pytest.raises(RuntimeError, match="grid edge"):
        GL.fit_point(
            X, Y, tr, val, te, dev=torch.device("cpu"), grid=(1e4, 10**4.5, 1e5), max_widenings=0
        )


def test_identity_bias_streaming_matches_canonical_helper():
    """Streaming identity+bias R2 == the canonical mapping_baselines helper."""
    from explore_persona_space.analysis.mapping_baselines import identity_bias_predict

    X, Y, _c, _v = _synth_store()
    tr, te = np.arange(500), np.arange(500, 900)
    _fit, pooled_r2 = GL._load_fit_core()
    ref = pooled_r2(identity_bias_predict(X[tr], Y[tr], X[te]), Y[te])
    got = GL.identity_bias_r2(X, Y, tr, te, pooled_r2, chunk=128)
    assert got == pytest.approx(ref, abs=1e-10)
    with pytest.raises(ValueError, match="d_in == d_out"):
        GL.identity_bias_r2(X, Y[:, :4], tr, te, pooled_r2)


# ── learning curve: theory ────────────────────────────────────────────────────


def test_theory_isotropic_closed_form():
    """Isotropic, pure-noise target, lam->0, n>d: excess -> noise*d/(n-d) exactly."""
    d, n, noise = 20, 100, 1.3
    out = GL.theory_r2(n, 1e-8, np.ones(d), np.zeros(d), noise, noise)
    assert out["excess_risk"] == pytest.approx(noise * d / (n - d), rel=1e-4)
    # and at n=0-like regimes the target power is returned in full
    p = np.ones(d) * 0.5
    out0 = GL.theory_r2(1, 1e6, np.ones(d), p, 0.0, p.sum())
    assert 0.9 * p.sum() < out0["predicted_mse"] <= p.sum() * 1.0001


def test_theory_matches_empirical_ridge_on_synthetic_gaussian():
    """The 2006.13198 curve tracks fit_ridge_primal's held-out R2 within 0.05."""
    X, Y, _c, _v = _synth_store(seed=5)
    pool, val, te = np.arange(600), np.arange(600, 900), np.arange(900, 3000)
    m = GL.pooled_moments(X, Y, pool)
    fit, pooled_r2 = GL._load_fit_core()
    deltas = []
    for lam in (10.0, 1000.0):
        pred_te, _meta = fit(X, Y, pool, val, te, [lam], torch.device("cpu"))
        emp = pooled_r2(pred_te, Y[te])
        th = GL.theory_r2(len(pool), lam, m["eta"], m["p_mode"], m["noise_var"], m["total_var"])
        deltas.append(abs(emp - th["predicted_r2"]))
    assert max(deltas) < 0.05


def test_theory_r2_monotone_in_n():
    """At fixed lambda the predicted R2 improves with n on a well-posed problem."""
    X, Y, _c, _v = _synth_store(seed=6)
    m = GL.pooled_moments(X, Y, np.arange(1500))
    r2s = [
        GL.theory_r2(n, 100.0, m["eta"], m["p_mode"], m["noise_var"], m["total_var"])[
            "predicted_r2"
        ]
        for n in (100, 400, 1600, 6400)
    ]
    assert all(b >= a - 1e-9 for a, b in itertools.pairwise(r2s))


def test_kappa_self_consistent_fixed_point():
    """kappa satisfies its own self-consistent equation."""
    eta = 1.0 / np.arange(1, 30) ** 1.2
    for lam, n in ((1e-3, 50), (10.0, 500), (1e5, 10)):
        k = GL.kappa_self_consistent(lam, n, eta)
        resid = k - lam - float(np.sum(k * eta / (k + n * eta)))
        assert abs(resid) < 1e-9 * max(k, 1.0)
        assert k >= lam


# ── parity check + H2b statistic + companions ─────────────────────────────────


def _verdict_point(n: int, r2: float, theory_r2: float) -> dict:
    return {
        "label": f"verdict_n{n}",
        "n_train": n,
        "test_r2": r2,
        "theory": {"predicted_r2": theory_r2},
        "layer": 19,
        "train_corpus": "lmsys",
        "eval_split_sha": "te:abc|val:def",
        "lambda_selection": GL.LAMBDA_SELECTION_PROTOCOL,
        "train_eval_distribution": "lmsys->lmsys",
    }


def test_fit_metadata_parity_check_names_mismatches():
    """Verdict points pass; an off-recipe companion fails with the fields NAMED."""
    points = [_verdict_point(100, 0.5, 0.5), _verdict_point(200, 0.6, 0.6)]
    parity = GL.fit_metadata_parity_check(points)
    assert all(p["pass"] for p in parity["per_point"])
    companion = dict(GL.PASSB_COMPANION)
    comp_parity = GL.fit_metadata_parity_check([companion], reference=parity["reference"])
    row = comp_parity["per_point"][0]
    assert not row["pass"]
    assert "layer" in row["mismatched_fields"]
    assert "lambda_selection" in row["mismatched_fields"]


def test_mean_abs_delta_r2_bands():
    """H2b verdict bands: pass <= 0.05 < localized <= 0.15 < same-sign kill."""
    ok = [_verdict_point(100, 0.50, 0.51), _verdict_point(200, 0.60, 0.58)]
    assert GL.mean_abs_delta_r2(ok)["verdict"] == "h2b-pass"
    mid = [_verdict_point(100, 0.50, 0.60), _verdict_point(200, 0.60, 0.70)]
    assert GL.mean_abs_delta_r2(mid)["verdict"] == "localized-misfit (no kill)"
    kill = [_verdict_point(100, 0.50, 0.70), _verdict_point(200, 0.60, 0.80)]
    out = GL.mean_abs_delta_r2(kill)
    assert out["verdict"] == "h2b-kill-candidate" and out["same_sign_all"]
    assert GL.mean_abs_delta_r2([])["verdict"] == "no-parity-passing-points"


def test_companion_loader_reads_committed_artifacts():
    """Companions come from committed JSONs, labeled off-recipe, never verdict points."""
    comps = GL.load_companion_points(REPO_ROOT)
    by_label = {c["label"]: c for c in comps}
    assert len(comps) == 7
    assert all(c["off_recipe_companion"] for c in comps)
    assert by_label["committed_n3600_L19"]["test_r2"] == pytest.approx(0.7054149303865586)
    assert by_label["committed_n3600_L19"]["n_train"] == 3600
    assert by_label["committed_n50k_plan_b_L19"]["test_r2"] == pytest.approx(0.7599992543132661)
    assert by_label["committed_mixed_1m_L19"]["lambda_grid_edge"] == "low"  # C4 disclosure
    assert by_label["committed_lmsys_500k_L19"]["corpus_mix"] == {"lmsys": 500000, "wildchat": 0}
    assert by_label["passb_n4500_L16_gcv"]["layer"] == 16


# ── curve driver end-to-end (synthetic store, no network) ─────────────────────


def test_run_curve_driver_synthetic_store(tmp_path):
    """The curve CLI runs splits -> refits -> theory -> parity -> verdict end to end."""
    X, Y, corpus, conv = _synth_store()
    np.save(tmp_path / "x.npy", X)
    np.save(tmp_path / "y.npy", Y)
    np.savez(tmp_path / "meta.npz", corpus=corpus, conv_index=conv)
    out_dir = tmp_path / "out"
    rc = GL.main(
        [
            "curve",
            "--x",
            str(tmp_path / "x.npy"),
            "--y",
            str(tmp_path / "y.npy"),
            "--row-meta",
            str(tmp_path / "meta.npz"),
            "--out",
            str(out_dir),
            "--dev",
            "cpu",
            "--n-grid",
            "200,400,800",
            "--eval-rows",
            "300",
            "--val-rows",
            "200",
            "--skip-companions",
            "--smoke",
        ]
    )
    assert rc == 0
    out = json.loads((out_dir / "learning_curve.json").read_text())
    assert [p["n_train"] for p in out["verdict_points"]] == [200, 400, 800]
    assert all(pp["pass"] for pp in out["parity_check"]["per_point"])
    assert out["h2b"]["mean_abs_dr2"] is not None
    for p in out["verdict_points"]:
        assert p["lambda_grid_edge"] is None  # C4: never an edge value
        assert p["corpus_mix"] == {"lmsys": p["n_train"]}
        assert "knn_retrieval" in p and "chance_at_k" in p["knn_retrieval"]
    assert out["regime"]["lambda_selection"] == GL.LAMBDA_SELECTION_PROTOCOL


def test_import_check_mode():
    """--import-check resolves every deferred import and the argcheck contract."""
    assert GL.main(["--import-check"]) == 0
