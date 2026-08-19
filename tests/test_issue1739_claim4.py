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
    FLAGSHIPS,
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
                # arm2/arm20 run in the TRUE pass only (SHUFPAIR_ROSTER
                # excludes them) — the coverage set-check demands exactly that
                for arm in ("arm2_ctx_native", "arm20_shuffled_map_ridge"):
                    rows.append(_mk_row(b, rung, s, "true", arm, 0.4 + 0.01 * s))
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


def test_row_coverage_check_covers_extra_arms_true_pass_only():
    """arm2 + arm20 are IN the coverage set-check (true pass only): a missing
    cell is a DECLARED gap; their shufpair cells are never demanded
    (regression: #1739 claim4 r1 — COVER_ARMS excluded both controls)."""
    rows = _full_grid()
    rows = [
        r
        for r in rows
        if not (
            r["arm"] == "arm20_shuffled_map_ridge" and r["eval_rung"] == "r1" and r["seed"] == 0
        )
    ]
    _cells, _, gaps = row_coverage_check(rows, ["evil"], [0, 1])
    cell_gaps = [g for g in gaps if g.get("arm")]
    assert cell_gaps == [
        {
            "behavior": "evil",
            "eval_rung": "r1",
            "seed": 0,
            "map_variant": "true",
            "arm": "arm20_shuffled_map_ridge",
        }
    ]


def test_arm_true_means_partial_coverage_is_declared_gap():
    """Partial seed coverage on any arm withholds the mean + declares a gap
    row — never a silent partial average (#1739 claim4 r1 fix)."""
    from scripts.issue1739_claim4_fold import arm_true_means_declared

    cells = {}
    for s in (0, 1):
        for arm in ("arm4_ridge_ctx", "arm7_map_ridge_pred"):
            cells[("evil", "r1", s, "true", arm)] = {"rho_frozen": 0.5}
    cells[("evil", "r1", 0, "true", "arm2_ctx_native")] = {"rho_frozen": 0.4}  # seed 1 missing
    gaps: list[dict] = []
    means = arm_true_means_declared(cells, "evil", "r1", [0, 1], gaps)
    assert means["arm2_ctx_native"] is None
    assert any(
        g.get("arm") == "arm2_ctx_native" and "partial seed coverage 1/2" in g["note"] for g in gaps
    )
    assert means["arm4_ridge_ctx"] == pytest.approx(0.5)
    # a wholly-absent arm is None with NO partial-coverage gap (its cell-level
    # absence is the set-check's job)
    assert means["arm20_shuffled_map_ridge"] is None
    assert not any(g.get("arm") == "arm20_shuffled_map_ridge" for g in gaps)


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


# The registered 13-rung primary set (plan §3; counts pinned in
# EXPECTED_PRIMARY_COUNTS = {evil: 5, sycophancy: 6, hallucination: 2}).
REGISTERED_RUNGS = {
    "evil": ("hhrt", "toxicchat", "evil_mhj", "evil_pair", "evil_tomgibbs"),
    "sycophancy": ("aita", "sycoans", "sycoays", "sycofb", "sycomim", "sycomwe"),
    "hallucination": ("nqopen", "simpleqa"),
}
_FLAG_BY_B = dict(FLAGSHIPS)
_STRONG_FLAG_DTRUE = [0.30, 0.28, 0.32, 0.29, 0.31]
_STRONG_FLAG_DSHUF = [0.02, 0.01, 0.03, 0.02, 0.01]
_STRONG_OTHER_DTRUE = [0.05, 0.06, 0.04, 0.05, 0.06]
_STRONG_OTHER_DSHUF = [0.01, 0.0, 0.01, 0.0, 0.01]


def _full_lattice(
    flag_dtrue=_STRONG_FLAG_DTRUE,
    flag_dshuf=_STRONG_FLAG_DSHUF,
    other_dtrue=_STRONG_OTHER_DTRUE,
    other_dshuf=_STRONG_OTHER_DSHUF,
    ctx=(0.01, 0.2),
):
    """All 13 registered rungs x 5 seeds — strong-form values by default."""
    per_rung = []
    for b, rungs in REGISTERED_RUNGS.items():
        for rung in rungs:
            flag = _FLAG_BY_B.get(b) == rung
            per_rung.append(
                _rung_entry(
                    b,
                    rung,
                    list(flag_dtrue if flag else other_dtrue),
                    list(flag_dshuf if flag else other_dshuf),
                    ctx=ctx,
                )
            )
    return per_rung


def test_lattice_strong_form_requires_the_full_13_rung_set():
    per_rung = _full_lattice()
    v = lattice_verdict(per_rung)
    assert v["verdict"] == "Strong-form draftable"
    assert v["medians"]["n_rungs_in_median"] == 13


def test_lattice_missing_nonflagship_rung_is_unresolved():
    """Regression (#1739 claim4 r1: medians over the COMPLETE subset let 3
    rungs yield strong-form): one NON-flagship rung absent + otherwise-strong
    inputs -> `Not draftable / unresolved` with the gap named."""
    per_rung = [r for r in _full_lattice() if r["eval_rung"] != "simpleqa"]
    v = lattice_verdict(per_rung)
    assert v["verdict"] == "Not draftable / unresolved"
    assert "coverage gap" in v["reason"]
    assert any("hallucination: 1/2" in g for g in v["coverage_gaps"])


def test_lattice_incomplete_seeds_on_any_rung_is_unresolved():
    """All 5 seeds per rung are part of the registered denominator — a
    3-seed rung (even a non-flagship one) blocks both draftable branches."""
    per_rung = _full_lattice()
    for r in per_rung:
        if r["eval_rung"] == "aita":
            short = _rung_entry("sycophancy", "aita", [0.05, 0.06, 0.04], [0.01, 0.0, 0.01])
            r.clear()
            r.update(short)
    v = lattice_verdict(per_rung)
    assert v["verdict"] == "Not draftable / unresolved"
    assert any("sycophancy/aita: seeds incomplete (3/5)" in g for g in v["coverage_gaps"])


def test_lattice_missing_ctx_ci_is_unresolved():
    """Every quantity's interval is part of the denominator: a rung whose
    paired context-bootstrap CIs are absent (e.g. a preds gap) blocks both
    draftable branches with the gap named."""
    per_rung = _full_lattice()
    per_rung[0]["dtrue_ctx_ci"] = None
    v = lattice_verdict(per_rung)
    assert v["verdict"] == "Not draftable / unresolved"
    assert any("dtrue_ctx_ci missing" in g for g in v["coverage_gaps"])


def test_lattice_falsifier_takes_precedence():
    # both flagships' Delta_true seed-CIs span 0 -> blocked even with a
    # positive margin structure elsewhere (full 13-rung set present, so the
    # falsifier — not coverage — is what fires)
    per_rung = _full_lattice(flag_dtrue=[0.30, -0.28, 0.32, -0.29, 0.31], flag_dshuf=[0.0] * 5)
    v = lattice_verdict(per_rung)
    assert v["verdict"] == "Not draftable / unresolved"
    assert "falsifier" in v["reason"]


def test_lattice_weak_form_and_catchall_and_coverage():
    # weak: ONE flagship's ctx CI spans 0 -> not fully controlled; the other
    # flagship + medians stay positive over the full 13-rung set
    per_rung = _full_lattice()
    for r in per_rung:
        if r["behavior"] == "sycophancy" and r["eval_rung"] == "sycomwe":
            r["dtrue_ctx_ci"] = [-0.05, 0.2]
            r["margin_ctx_ci"] = [-0.05, 0.2]
    v = lattice_verdict(per_rung)
    assert v["verdict"] == "Weak-form draftable"
    # catch-all: medians negative, but the falsifier does NOT fire (only one
    # flagship's Delta_true CI spans 0 — the other is wholly negative)
    per_rung2 = _full_lattice(
        flag_dtrue=[-0.30, -0.28, -0.32, -0.29, -0.31],
        flag_dshuf=[0.0] * 5,
        other_dtrue=[-0.05, -0.06, -0.04, -0.05, -0.06],
        other_dshuf=[0.0] * 5,
    )
    for r in per_rung2:
        if r["behavior"] == "sycophancy" and r["eval_rung"] == "sycomwe":
            spanner = _rung_entry(
                "sycophancy", "sycomwe", [-0.25, -0.27, 0.26, -0.24, -0.28], [0.0] * 5
            )
            r.clear()
            r.update(spanner)
    v2 = lattice_verdict(per_rung2)
    assert v2["verdict"] == "Not draftable / unresolved"
    assert "catch-all" in v2["reason"]
    assert v2["flagship_descriptives"][0]["dtrue_spans_zero"] is False
    # coverage: a flagship absent -> unresolved with the gap named
    per_rung3 = [
        r
        for r in _full_lattice()
        if not (r["behavior"] == "evil" and r["eval_rung"] == "evil_pair")
    ]
    v3 = lattice_verdict(per_rung3)
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


def test_arm2_sanity_band_not_evaluable_flag(tmp_path):
    """A not-evaluable band check flags DISTINCTLY (never silently unflagged;
    distinct from the out-of-band adapter-suspect flag)."""
    from scripts.issue1739_claim4_fold import arm2_sanity_band

    # committed train summary absent
    out = arm2_sanity_band(tmp_path, "evil", [], seeds=[0])
    assert out["flag"] == "not-evaluable"
    # summary present but no band rows / no pvsynth rows
    d = tmp_path / "evil" / "arm_results"
    d.mkdir(parents=True)
    (d / "all_arms_spearman.json").write_text(json.dumps({"arm_rows": []}))
    out2 = arm2_sanity_band(tmp_path, "evil", [], seeds=[0])
    assert out2["flag"] == "not-evaluable"


# ---------------------------------------------------------------------------
# 5. companion scored branch + preds-series contract (fold P2)
# ---------------------------------------------------------------------------


def _write_companion_preds(root: Path, ids, dv, *, rung="toxicchat", drop_group=False):
    rng = np.random.default_rng(11)
    preds_dir = root / "evil" / "seed0" / "transfer_preds"
    preds_dir.mkdir(parents=True)
    for fname in (f"P-B-holdout-{rung}.jsonl", f"P-B-holdout-{rung}.shufpair.jsonl"):
        rows = []
        for arm in ("arm4_ridge_ctx", "arm7_map_ridge_pred"):
            for i, cid in enumerate(ids):
                row = {
                    "rung": rung,
                    "arm": arm,
                    "context_id": cid,
                    "score": float(rng.normal()),
                    "dv": float(dv[i]),
                    "seed": 0,
                }
                if not drop_group:
                    row["group"] = f"g{i % 4}"
                rows.append(row)
        (preds_dir / fname).write_text("\n".join(json.dumps(r) for r in rows))


def test_companion_toxicchat_scored_branch(tmp_path):
    """Regression (#1739 claim4 r1 Major: the `evil_toxicchat` preds label
    made the scored branch unreachable — every run declared-skipped): seed-0
    preds under the REAL r2v2 rung label `toxicchat` + a minimal compliance
    raw fixture must SCORE (a future label drift re-skips loudly here)."""
    from types import SimpleNamespace

    from scripts.issue1739_claim4_fold import companion_toxicchat

    rng = np.random.default_rng(5)
    ids = [f"ctx{i:03d}" for i in range(12)]
    dv = rng.uniform(0, 100, size=len(ids))
    _write_companion_preds(tmp_path, ids, dv)
    all_scores = {}
    for i, cid in enumerate(ids):
        for k in range(2):
            all_scores[f"{cid}_k{k:02d}__0000{k}__00"] = {"score": float(40 + i + k)}
    raw = tmp_path / "judge_raw_compliance_full.json"
    raw.write_text(json.dumps({"all_scores": all_scores}))
    args = SimpleNamespace(claim4_root=tmp_path, compliance_raw=raw, n_boot=25)

    out = companion_toxicchat(args, min_coverage=0.9)
    assert out["status"] == "scored", out
    assert len(out["rows"]) == 4  # 2 variants x 2 arms x seed 0
    assert out["coverage_of_compliance_rows"] == pytest.approx(1.0)
    for r in out["rows"]:
        assert -1.0 <= r["rho_vs_compliance"] <= 1.0
        assert r["ci"][0] <= r["ci"][1]


def test_load_preds_series_missing_group_is_named_gap(tmp_path):
    """A preds row without a `group` label is a NAMED gap — the group-level
    bootstrap must never silently degrade to per-context resampling."""
    from scripts.issue1739_claim4_fold import load_preds_series

    rng = np.random.default_rng(6)
    ids = [f"ctx{i:03d}" for i in range(6)]
    dv = rng.uniform(0, 100, size=len(ids))
    _write_companion_preds(tmp_path, ids, dv, drop_group=True)
    series, _dv, _groups, note = load_preds_series(tmp_path, "evil", "toxicchat", [0])
    assert series is None
    assert "missing 'group'" in note


# ---------------------------------------------------------------------------
# 6. batched group bootstrap == serial per-draw reference
# ---------------------------------------------------------------------------


def test_group_bootstrap_rhos_matches_serial_reference():
    """The batched group bootstrap (bucketed arms.bootstrap_rhos reductions,
    one vectorized group sample) equals the serial per-draw
    spearman_rows(mat[:, idx], dv[idx]) reference on the SAME rng stream —
    unequal group sizes exercise the length bucketing."""
    from explore_persona_space.experiments.issue_1739 import arms as arms_mod
    from scripts.issue1739_claim4_fold import group_bootstrap_rhos

    data_rng = np.random.default_rng(3)
    n = 40
    mat = data_rng.normal(size=(4, n))
    dv = data_rng.normal(size=n)
    groups = [f"g{min(i, 5)}" for i in range(n)]  # g0..g4 singletons, g5 holds 35
    n_boot = 32

    rhos, n_groups = group_bootstrap_rhos(
        mat, dv, groups, n_boot=n_boot, rng=np.random.default_rng([1739, 22, 0])
    )
    assert n_groups == 6 and rhos.shape == (4, n_boot)

    garr = np.asarray(groups)
    ug = sorted(set(groups))
    gidx = [np.flatnonzero(garr == g) for g in ug]
    ref_rng = np.random.default_rng([1739, 22, 0])
    gs = ref_rng.integers(0, len(ug), size=(n_boot, len(ug)))
    for d in range(n_boot):
        idx = np.concatenate([gidx[g] for g in gs[d]])
        ref = arms_mod.spearman_rows(mat[:, idx], dv[idx])
        np.testing.assert_allclose(rhos[:, d], ref, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# 7. runner: breadcrumb identity + smoke phase argv validation
# ---------------------------------------------------------------------------


def test_stage_crumb_identity_and_token_keyed_path():
    from scripts.issue1739_r2v2_run import _crumb_matches, _stage_crumb_path

    assert _stage_crumb_path("evil", "20260819T1200Z").endswith(
        "evil_stage_done_20260819T1200Z.json"
    )
    ok, _ = _crumb_matches(
        {"code_sha": "a" * 40, "run_token": "t1"}, code_sha="a" * 40, run_token="t1"
    )
    assert ok
    for bad in (
        {"code_sha": "b" * 40, "run_token": "t1"},  # foreign checkout
        {"code_sha": "a" * 40, "run_token": "OLD"},  # prior run
        {},  # legacy tokenless crumb
    ):
        ok2, why = _crumb_matches(bad, code_sha="a" * 40, run_token="t1")
        assert not ok2 and "mismatch" in why


def test_runner_stage_gate_requires_run_token():
    from scripts.issue1739_r2v2_run import parse_args as run_parse_args

    with pytest.raises(SystemExit):
        run_parse_args(["--stage-wait-sibling"])
    with pytest.raises(SystemExit):
        run_parse_args(["--stage-signal-done", "--stage-run-token", "bad token!"])
    args = run_parse_args(["--stage-wait-sibling", "--stage-run-token", "tok-1"])
    assert args.stage_run_token == "tok-1"


def test_runner_smoke_flag_validation():
    from scripts.issue1739_r2v2_run import parse_args as run_parse_args

    with pytest.raises(SystemExit):  # smoke requires the fits-claim4 leg
        run_parse_args(["--smoke", "--behaviors", "evil"])
    with pytest.raises(SystemExit):  # smoke behavior must run FIRST
        run_parse_args(
            [
                "--smoke",
                "--behaviors",
                "evil",
                "sycophancy",
                "--legs",
                "fits-claim4",
                "--smoke-behavior",
                "sycophancy",
            ]
        )
    with pytest.raises(SystemExit):  # scratch root, never the production out-root
        run_parse_args(
            [
                "--smoke",
                "--behaviors",
                "evil",
                "--legs",
                "fits-claim4",
                "--smoke-out-root",
                "eval_results/issue_1739/claim4_controls",
            ]
        )
    args = run_parse_args(["--smoke", "--behaviors", "evil", "--legs", "fits-claim4"])
    assert args.smoke_behavior == "evil" and args.smoke_holdout == "toxicchat"
    assert str(args.smoke_out_root).endswith("claim4_controls_smoke")


# ---------------------------------------------------------------------------
# 8. scorer: per-seed resume predicate (code SHA + schema keyed)
# ---------------------------------------------------------------------------


def test_seed_output_resume_predicate(tmp_path):
    from scripts.issue1739_r2v2_score import SEED_OUT_SCHEMA_VERSION, _seed_output_resume_ok

    out = tmp_path / "evil" / "seed0"
    out.mkdir(parents=True)
    meta = {
        "git_commit": "c" * 40,
        "out_schema_version": SEED_OUT_SCHEMA_VERSION,
        "seed": 0,
        "map_variants": ["true", "shufpair"],
    }
    (out / "all_arms_spearman.json").write_text(json.dumps({"meta": meta}))
    (out / "map_diagnostics.json").write_text("{}")
    (out / "readout_pools.json").write_text("{}")
    mv = ["true", "shufpair"]
    ok, why = _seed_output_resume_ok(out, commit="c" * 40, seed=0, map_variants=mv)
    assert ok, why
    # stale code SHA can never silently satisfy the predicate
    ok2, why2 = _seed_output_resume_ok(out, commit="d" * 40, seed=0, map_variants=mv)
    assert not ok2 and "git_commit" in why2
    # wrong seed / wrong variant set
    assert not _seed_output_resume_ok(out, commit="c" * 40, seed=1, map_variants=mv)[0]
    assert not _seed_output_resume_ok(out, commit="c" * 40, seed=0, map_variants=["true"])[0]
    # schema drift
    stale = dict(meta, out_schema_version=SEED_OUT_SCHEMA_VERSION - 1)
    (out / "all_arms_spearman.json").write_text(json.dumps({"meta": stale}))
    ok3, why3 = _seed_output_resume_ok(out, commit="c" * 40, seed=0, map_variants=mv)
    assert not ok3 and "out_schema_version" in why3
    # partial output (a companion artifact missing)
    (out / "all_arms_spearman.json").write_text(json.dumps({"meta": meta}))
    (out / "readout_pools.json").unlink()
    ok4, why4 = _seed_output_resume_ok(out, commit="c" * 40, seed=0, map_variants=mv)
    assert not ok4 and "readout_pools" in why4
