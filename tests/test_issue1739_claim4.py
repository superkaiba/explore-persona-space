"""claim4-controls round (#1739) unit tests.

Pins the four implementer-hygiene checks the plan registers:
1. the pairing-shuffle STRUCTURAL invariant (fingerprints verify a
   within-component bijection / shared-row-permutation / non-identity; a
   deliberately broken permutation fails loud);
2. the fold script's row-coverage set-check (a missing cell is a reported
   gap, never an imputed zero) + the arm4 cross-variant pairing assert +
   the §3 v21 lattice verdict (falsifier precedence, strong/weak/catch-all);
3. the arm2 transfer semantics on synthetic tensors (direction fit on the
   pool's midpoint split, projected onto the holdout — through the REAL
   run_transfer_cell dispatch, not a mock);
4. the claim4 repro-gate join (uniqueness on KEY_FIELDS after the
   map_variant subset; report vs halt tolerances).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.issue1739_arm12_repro_check import (  # noqa: E402
    _cmp_behavior_claim4,
    _index,
    _subset_claim4,
)
from scripts.issue1739_claim4_fold import (  # noqa: E402
    arm4_pairing_check,
    lattice_verdict,
    row_coverage_check,
    seed_tci,
)
from scripts.issue1739_r2v2_score import (  # noqa: E402
    pairing_shuffle_perm,
    shufpair_structural_check,
)

# ---------------------------------------------------------------------------
# 1. pairing-shuffle structural invariant
# ---------------------------------------------------------------------------


def test_pairing_shuffle_perm_structural_pass():
    perm, fp = pairing_shuffle_perm(200, 300, seed=3)
    out = shufpair_structural_check(perm, 200, 300)
    assert out["within_component_bijection"] and out["non_identity"]
    assert fp["n_generic"] == 200 and fp["n_eliciting"] == 100
    assert 0.9 < fp["frac_moved_generic"] <= 1.0  # ~ (n-1)/n fixed-point-free mass
    # deterministic under the registered rng namespace [1739, 21, seed]
    perm2, fp2 = pairing_shuffle_perm(200, 300, seed=3)
    assert np.array_equal(perm, perm2)
    assert fp2["perm_generic_sha256"] == fp["perm_generic_sha256"]
    perm_other, _ = pairing_shuffle_perm(200, 300, seed=4)
    assert not np.array_equal(perm, perm_other)


def test_shufpair_structural_check_fails_loud_on_broken_perms():
    n_gen, n_tot = 6, 10
    good, _ = pairing_shuffle_perm(n_gen, n_tot, seed=0)
    # cross-component move: swap a generic slot with an eliciting slot
    broken = good.copy()
    broken[0], broken[n_gen] = broken[n_gen], broken[0]
    with pytest.raises(RuntimeError, match="OUT of the generic block"):
        shufpair_structural_check(broken, n_gen, n_tot)
    # identity: no pairing destroyed
    with pytest.raises(RuntimeError, match="identity"):
        shufpair_structural_check(np.arange(n_tot), n_gen, n_tot)
    # non-bijection: a repeated target
    non_bij = good.copy()
    non_bij[1] = non_bij[0]
    with pytest.raises(RuntimeError, match="bijection"):
        shufpair_structural_check(non_bij, n_gen, n_tot)
    # one component left un-shuffled (the forgotten-block bug class)
    gen_identity = np.concatenate([np.arange(n_gen), n_gen + np.roll(np.arange(n_tot - n_gen), 1)])
    with pytest.raises(RuntimeError, match="generic component is identity"):
        shufpair_structural_check(gen_identity, n_gen, n_tot)


def test_in_place_per_layer_permute_matches_fancy_index():
    """The scorer permutes y_w IN PLACE per layer with ONE shared perm — pin
    that this equals the whole-array fancy index (shared-across-layers)."""
    rng = np.random.default_rng(1)
    y = rng.normal(size=(3, 12, 4))
    perm, _ = pairing_shuffle_perm(8, 12, seed=5)
    expected = y[:, perm, :].copy()
    for li in range(y.shape[0]):
        y[li] = y[li][perm]
    assert np.array_equal(y, expected)


# ---------------------------------------------------------------------------
# 2. fold: row coverage + arm4 pairing + lattice
# ---------------------------------------------------------------------------


def _mk_row(b, rung, seed, mv, arm, rho, fit=None):
    return {
        "protocol": "P-B",
        "fit": fit or f"P-B-holdout-{rung}",
        "eval_rung": rung,
        "behavior": b,
        "seed": seed,
        "map_variant": mv,
        "arm": arm,
        "rho_frozen": rho,
        "variant": "context_end",
        "regime": "e1",
        "n_eval": 10,
        "n_readout": 100,
        "layer": 3,
    }


def _full_grid(behaviors=("evil",), rungs=("r1", "r2"), seeds=(0, 1)):
    rows = []
    for b in behaviors:
        for rung in rungs:
            for s in seeds:
                for mv in ("true", "shufpair"):
                    for arm in ("arm4_ridge_ctx", "arm7_map_ridge_pred"):
                        rho = 0.5 if arm == "arm4_ridge_ctx" else (0.6 if mv == "true" else 0.52)
                        rows.append(_mk_row(b, rung, s, mv, arm, rho + 0.01 * s))
    return rows


def test_row_coverage_check_reports_missing_cell_never_imputes():
    rows = _full_grid()
    # drop ONE cell: (evil, r2, seed 1, shufpair, arm7)
    rows = [
        r
        for r in rows
        if not (
            r["eval_rung"] == "r2"
            and r["seed"] == 1
            and r["map_variant"] == "shufpair"
            and r["arm"] == "arm7_map_ridge_pred"
        )
    ]
    cells, rungs_by_b, gaps = row_coverage_check(rows, ["evil"], [0, 1])
    assert rungs_by_b["evil"] == ["r1", "r2"]
    cell_gaps = [g for g in gaps if g.get("arm")]
    assert len(cell_gaps) == 1
    assert cell_gaps[0] == {
        "behavior": "evil",
        "eval_rung": "r2",
        "seed": 1,
        "map_variant": "shufpair",
        "arm": "arm7_map_ridge_pred",
    }
    # the missing cell is ABSENT from cells (never an imputed zero)
    assert ("evil", "r2", 1, "shufpair", "arm7_map_ridge_pred") not in cells
    # non-primary rows (fit != P-B-holdout-<rung>) are excluded from the grid
    rows.append(
        _mk_row(
            "evil", "wildchat_rung", 0, "true", "arm7_map_ridge_pred", 0.1, fit="P-B-holdout-r1"
        )
    )
    _cells2, rungs2, _ = row_coverage_check(rows, ["evil"], [0, 1])
    assert "wildchat_rung" not in rungs2["evil"]


def test_row_coverage_check_duplicate_key_fails_loud():
    rows = _full_grid()
    rows.append(rows[0].copy())
    with pytest.raises(SystemExit, match="duplicate"):
        row_coverage_check(rows, ["evil"], [0, 1])


def test_arm4_pairing_check_passes_and_fails():
    rows = _full_grid()
    cells, _, _ = row_coverage_check(rows, ["evil"], [0, 1])
    arm4_pairing_check(cells)  # identical across variants by construction
    # perturb one shufpair arm4 row -> hard fail
    bad = [r.copy() for r in rows]
    for r in bad:
        if (
            r["arm"] == "arm4_ridge_ctx"
            and r["map_variant"] == "shufpair"
            and r["eval_rung"] == "r1"
            and r["seed"] == 0
        ):
            r["rho_frozen"] += 1e-6
    cells_bad, _, _ = row_coverage_check(bad, ["evil"], [0, 1])
    with pytest.raises(SystemExit, match="pairing check FAILED"):
        arm4_pairing_check(cells_bad)


def _rung_entry(b, rung, dtrue_vals, dshuf_vals, *, complete=True, ctx=(0.01, 0.2)):
    dtrue = seed_tci(list(dtrue_vals))
    dshuf = seed_tci(list(dshuf_vals))
    margin = seed_tci([t - s for t, s in zip(dtrue_vals, dshuf_vals, strict=True)])
    return {
        "behavior": b,
        "eval_rung": rung,
        "complete": complete,
        "dtrue": dtrue,
        "dshuf": dshuf,
        "margin": margin,
        "dtrue_ctx_ci": list(ctx),
        "margin_ctx_ci": list(ctx),
    }


def test_lattice_strong_form():
    per_rung = [
        _rung_entry(
            "evil", "evil_pair", [0.30, 0.28, 0.32, 0.29, 0.31], [0.02, 0.01, 0.03, 0.02, 0.01]
        ),
        _rung_entry(
            "sycophancy", "sycomwe", [0.25, 0.27, 0.26, 0.24, 0.28], [0.00, 0.01, -0.01, 0.02, 0.00]
        ),
        _rung_entry(
            "hallucination", "nqopen", [0.05, 0.06, 0.04, 0.05, 0.06], [0.01, 0.0, 0.01, 0.0, 0.01]
        ),
    ]
    v = lattice_verdict(per_rung)
    assert v["verdict"] == "Strong-form draftable"


def test_lattice_falsifier_takes_precedence():
    # both flagships' Delta_true seed-CIs span 0 -> blocked even with a
    # positive margin structure elsewhere
    per_rung = [
        _rung_entry("evil", "evil_pair", [0.30, -0.28, 0.32, -0.29, 0.31], [0.0] * 5),
        _rung_entry("sycophancy", "sycomwe", [0.25, -0.27, 0.26, -0.24, 0.28], [0.0] * 5),
    ]
    v = lattice_verdict(per_rung)
    assert v["verdict"] == "Not draftable / unresolved"
    assert "falsifier" in v["reason"]


def test_lattice_weak_form_and_catchall_and_coverage():
    strong_flag = _rung_entry("evil", "evil_pair", [0.30, 0.28, 0.32, 0.29, 0.31], [0.02] * 5)
    weak_flag = _rung_entry(
        "sycophancy", "sycomwe", [0.25, 0.27, 0.26, 0.24, 0.28], [0.0] * 5, ctx=(-0.05, 0.2)
    )  # ctx CI spans 0 -> this flagship not fully controlled
    v = lattice_verdict([strong_flag, weak_flag])
    assert v["verdict"] == "Weak-form draftable"
    # catch-all: medians negative, but the falsifier does NOT fire (only one
    # flagship's Delta_true CI spans 0 — the other is wholly negative)
    per_rung = [
        _rung_entry("evil", "evil_pair", [-0.30, -0.28, -0.32, -0.29, -0.31], [0.0] * 5),
        _rung_entry("sycophancy", "sycomwe", [-0.25, -0.27, 0.26, -0.24, -0.28], [0.0] * 5),
    ]
    v2 = lattice_verdict(per_rung)
    assert v2["verdict"] == "Not draftable / unresolved"
    assert "catch-all" in v2["reason"]
    assert v2["flagship_descriptives"][0]["dtrue_spans_zero"] is False
    # coverage: a flagship absent -> unresolved with the gap named
    v3 = lattice_verdict([strong_flag])
    assert v3["verdict"] == "Not draftable / unresolved"
    assert "coverage gap" in v3["reason"]


def test_seed_tci_matches_t_multiplier():
    vals = [0.1, 0.2, 0.15, 0.12, 0.18]
    out = seed_tci(vals)
    assert out["n_seeds"] == 5
    mean, sd = np.mean(vals), np.std(vals, ddof=1)
    half = 2.7764451051977987 * sd / np.sqrt(5)  # t_{0.975, df=4} (plan §11)
    assert out["tci"] == pytest.approx([mean - half, mean + half], rel=1e-9)
    assert seed_tci([0.3])["tci"] is None  # single seed: no interval, never a fake one


# ---------------------------------------------------------------------------
# 3. arm2 transfer semantics on synthetic tensors (real dispatch, no mocks)
# ---------------------------------------------------------------------------


def test_arm2_transfer_semantics_and_arm20_presence_synthetic():
    from explore_persona_space.experiments.issue_1739 import arms, fits

    rng = np.random.default_rng(7)
    ly, d, n_tr, n_ev = 2, 5, 14, 6
    z_tr = rng.normal(size=(ly, n_tr, d))
    z_ev = rng.normal(size=(ly, n_ev, d))
    za_tr = rng.normal(size=(ly, n_tr, d))
    za_ev = rng.normal(size=(ly, n_ev, d))
    dv_tr = rng.uniform(0, 100, size=n_tr)
    dv_ev = rng.uniform(0, 100, size=n_ev)
    rb = rng.normal(size=(ly, d))
    mapfit = fits.MapFit(
        w=rng.normal(size=(ly, d, d)),
        x_mu=np.zeros((ly, 1, d)),
        x_sd=np.ones((ly, 1, d)),
        y_mu=np.zeros((ly, 1, d)),
        diagnostics={},
    )
    data = arms.CellData(z_ctx=z_tr, dv=dv_tr, rb=rb, z_ans=za_tr, mapfit=mapfit, layers=(0, 1))
    cell = fits.BudgetCell(
        row_idx=np.arange(n_tr),
        fold_ids=np.zeros(n_tr, dtype=np.int64),
        n_folds=1,
        budget_l=n_tr,
        draw=0,
        seed=0,
        fold_scheme="test",
    )
    roster = [
        "arm2_ctx_native",
        "arm4_ridge_ctx",
        "arm6_map_proj_e1",
        "arm7_map_ridge_pred",
        "arm20_shuffled_map_ridge",
    ]
    scores, skipped = arms.run_transfer_cell(
        data, cell, z_ev, dv_ev, za_ev=za_ev, arms=roster, device="cpu", ridge_folds=(0,)
    )
    assert not skipped, f"unexpected skips: {skipped}"
    assert set(scores) == set(roster)
    for slug in roster:
        assert scores[slug].shape == (ly, n_ev), slug

    # manual arm2 semantics: midpoint split on the TRAIN (pool) dv only,
    # hi/lo diff-of-means direction on TRAIN z, projected onto the eval rows.
    mid = 0.5 * (dv_tr.max() + dv_tr.min())
    hi, lo = dv_tr >= mid, dv_tr < mid
    assert hi.any() and lo.any()
    direction = z_tr[:, hi].mean(axis=1) - z_tr[:, lo].mean(axis=1)  # (Ly, d)
    expected = np.einsum("ld,lnd->ln", direction, z_ev)
    np.testing.assert_allclose(scores["arm2_ctx_native"], expected, rtol=1e-10, atol=1e-10)

    # arm20 consumes the TRUE map via shuffled weights: recompute directly.
    w_shuf = fits.shuffled_map_weights(mapfit.w, seed=cell.seed)
    assert not np.array_equal(w_shuf, mapfit.w)
    # its scores are a ridge on mp_shuf — just pin non-degeneracy + distinctness
    assert np.isfinite(scores["arm20_shuffled_map_ridge"]).all()
    assert not np.allclose(scores["arm20_shuffled_map_ridge"], scores["arm7_map_ridge_pred"])


# ---------------------------------------------------------------------------
# 4. claim4 repro-gate join
# ---------------------------------------------------------------------------


def _banked_rows():
    rows = []
    for arm, rho in (("arm4_ridge_ctx", 0.5), ("arm7_map_ridge_pred", 0.62)):
        for fit, rung in (("P-B-holdout-r1", "r1"), ("P-B-holdout-r1", "wildchat_rung")):
            rows.append(
                {
                    "protocol": "P-B",
                    "fit": fit,
                    "eval_rung": rung,
                    "arm": arm,
                    "variant": "context_end",
                    "regime": "e1",
                    "rho_frozen": rho,
                    "n_eval": 10,
                    "n_readout": 100,
                    "layer": 3,
                    "seed": 0,
                }
            )
        rows.append(
            {
                "protocol": "P-A",
                "fit": "P-A",
                "eval_rung": "r1",
                "arm": arm,
                "variant": "context_end",
                "regime": "e1",
                "rho_frozen": rho,
                "n_eval": 10,
                "n_readout": 100,
                "layer": 3,
                "seed": 0,
            }
        )
    return rows


def _claim4_rows(drift=0.0):
    rows = []
    for r in _banked_rows():
        if r["protocol"] != "P-B":
            continue
        for mv in ("true", "shufpair"):
            n = dict(r)
            n["map_variant"] = mv
            n["rho_frozen"] = r["rho_frozen"] + (drift if mv == "true" else 0.001)
            rows.append(n)
    # the round's new true-pass arms + required presence
    for arm in ("arm2_ctx_native", "arm20_shuffled_map_ridge"):
        rows.append(
            {
                "protocol": "P-B",
                "fit": "P-B-holdout-r1",
                "eval_rung": "r1",
                "arm": arm,
                "variant": "context_end",
                "regime": "e1",
                "rho_frozen": 0.4,
                "n_eval": 10,
                "n_readout": 100,
                "layer": 3,
                "seed": 0,
                "map_variant": "true",
            }
        )
    return rows


def test_claim4_subset_makes_key_unique_and_unsubset_collides():
    new_rows = _claim4_rows()
    shared = {"arm4_ridge_ctx", "arm7_map_ridge_pred"}
    with pytest.raises(SystemExit, match="not unique"):
        _index(new_rows, shared)  # true+shufpair share the KEY_FIELDS tuple
    subset = _subset_claim4(new_rows)
    idx = _index(subset, shared)  # unique after the registered subset
    assert len(idx) == 4


def test_claim4_cmp_pass_report_and_halt():
    old = {"transfer_rows": _banked_rows(), "meta": {"arms": []}}
    ok, _lines, stats = _cmp_behavior_claim4("evil", old, {"transfer_rows": _claim4_rows()})
    assert ok and stats["max_drho"] == 0.0 and stats["n_joined"] == 4

    # drift above report tol (1e-9) but below halt tol (1e-3): PASS + reported
    ok2, _, stats2 = _cmp_behavior_claim4("evil", old, {"transfer_rows": _claim4_rows(drift=5e-7)})
    assert ok2 and len(stats2["cells_over_report_tol"]) == 4

    # drift above halt tol: FAIL
    ok3, lines3, _ = _cmp_behavior_claim4("evil", old, {"transfer_rows": _claim4_rows(drift=5e-3)})
    assert not ok3 and any("HALT threshold" in ln for ln in lines3)

    # missing shufpair pass: FAIL
    only_true = [r for r in _claim4_rows() if r.get("map_variant") == "true"]
    ok4, lines4, _ = _cmp_behavior_claim4("evil", old, {"transfer_rows": only_true})
    assert not ok4 and any("shufpair pass ABSENT" in ln for ln in lines4)

    # missing arm2 rows: FAIL
    no_arm2 = [r for r in _claim4_rows() if r["arm"] != "arm2_ctx_native"]
    ok5, lines5, _ = _cmp_behavior_claim4("evil", old, {"transfer_rows": no_arm2})
    assert not ok5 and any("arm2_ctx_native ABSENT" in ln for ln in lines5)


def test_extra_arms_guard_rejects_unknown_slug():
    import scripts.issue1739_r2v2_score as sc

    saved = (sc.ROSTER, sc.LABEL_CONSUMING)
    try:
        with pytest.raises(ValueError, match="EXTRA_ARMS_ALLOWED"):
            sc._apply_extra_arms(["arm99_bogus"])
        sc._apply_extra_arms(["arm2_ctx_native", "arm20_shuffled_map_ridge"])
        assert "arm2_ctx_native" in sc.ROSTER and "arm20_shuffled_map_ridge" in sc.ROSTER
        assert "arm2_ctx_native" in sc.LABEL_CONSUMING
    finally:
        sc.ROSTER, sc.LABEL_CONSUMING = saved


def test_arm2_sanity_band_reads_pvsynth_from_full_rows(tmp_path):
    """pvsynth rows never enter the primary-rung cells (fit != P-B-holdout-pvsynth);
    the band check must read the FULL row list (regression: #1739 claim4 fold bug)."""
    from scripts.issue1739_claim4_fold import arm2_sanity_band

    d = tmp_path / "evil" / "arm_results"
    d.mkdir(parents=True)
    committed = {
        "arm_rows": [
            {
                "arm": "arm2_ctx_native",
                "variant": "context_end",
                "regime": "e1",
                "u_rung_label": "full",
                "rho_frozen": v,
            }
            for v in (0.40, 0.55, 0.70)
        ]
    }
    (d / "all_arms_spearman.json").write_text(json.dumps(committed))

    def _rows(rho):
        return [
            {
                "behavior": "evil",
                "arm": "arm2_ctx_native",
                "map_variant": "true",
                "eval_rung": "pvsynth",
                "seed": s,
                "rho_frozen": rho,
            }
            for s in (0, 1)
        ] + [
            # primary-rung row and shufpair pvsynth row must both be ignored
            {
                "behavior": "evil",
                "arm": "arm2_ctx_native",
                "map_variant": "true",
                "eval_rung": "evil_pair",
                "seed": 0,
                "rho_frozen": -0.9,
            },
            {
                "behavior": "evil",
                "arm": "arm2_ctx_native",
                "map_variant": "shufpair",
                "eval_rung": "pvsynth",
                "seed": 0,
                "rho_frozen": -0.9,
            },
        ]

    ok = arm2_sanity_band(tmp_path, "evil", _rows(0.50), seeds=[0, 1])
    assert ok["in_band"] is True and not ok.get("flag")
    assert ok["pvsynth_rho_per_seed"] == {0: 0.50, 1: 0.50}

    bad = arm2_sanity_band(tmp_path, "evil", _rows(0.10), seeds=[0, 1])
    assert bad["in_band"] is False and bad.get("flag")
