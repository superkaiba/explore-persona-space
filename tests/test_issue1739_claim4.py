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
    COVER_ARM_VARIANTS,
    FLAGSHIPS,
    REGISTERED_PRIMARY_RUNGS,
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


# The registered 13-rung primary set (plan §3) — imported from the fold
# module (the single source of truth; review r2 item 4: tests import it,
# never the reverse), iterated in sorted order for determinism.
REGISTERED_RUNGS = {b: tuple(sorted(v)) for b, v in REGISTERED_PRIMARY_RUNGS.items()}
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


def test_lattice_same_size_wrong_rung_set_is_unresolved():
    """Regression (#1739 claim4 r2 item 4): the completeness gate checks the
    EXACT registered rung-NAME set, not per-behavior counts — a same-size
    WRONG set (simpleqa swapped for an unregistered rung) resolves
    `Not draftable / unresolved` with BOTH sides of the mismatch named."""
    per_rung = _full_lattice()
    for r in per_rung:
        if r["eval_rung"] == "simpleqa":
            r["eval_rung"] = "not_a_registered_rung"
    v = lattice_verdict(per_rung)
    assert v["verdict"] == "Not draftable / unresolved"
    assert "coverage gap" in v["reason"]
    gap = next(g for g in v["coverage_gaps"] if g.startswith("hallucination:"))
    assert "missing ['simpleqa']" in gap and "unregistered ['not_a_registered_rung']" in gap
    # the counts alone would have passed (2 rungs present) — the NAME set is
    # what fails, which is exactly the round-2 finding
    assert "1/2" in gap


def test_row_coverage_check_names_wrong_rung_set():
    """Defense-in-depth twin at the row grain: row_coverage_check's roster
    note is name-based too (a same-count wrong rung set is a reported gap)."""
    rows = []
    for rung in ("nqopen", "wrongrung"):
        for s in range(5):
            for arm, variants in COVER_ARM_VARIANTS.items():
                for mv in variants:
                    rows.append(
                        {
                            "protocol": "P-B",
                            "fit": f"P-B-holdout-{rung}",
                            "eval_rung": rung,
                            "behavior": "hallucination",
                            "seed": s,
                            "map_variant": mv,
                            "arm": arm,
                            "rho_frozen": 0.1,
                        }
                    )
    _cells, _rungs, gaps = row_coverage_check(rows, ["hallucination"], list(range(5)))
    notes = [g["note"] for g in gaps if "note" in g]
    assert any("missing ['simpleqa']" in n and "unregistered ['wrongrung']" in n for n in notes)


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
    for bad, why_frag in (
        ({"code_sha": "b" * 40, "run_token": "t1"}, "mismatch"),  # foreign checkout
        ({"code_sha": "a" * 40, "run_token": "OLD"}, "mismatch"),  # prior run
        ({}, "invalid"),  # legacy sha-less crumb (r2 item 5: refused as identity-less)
    ):
        ok2, why = _crumb_matches(bad, code_sha="a" * 40, run_token="t1")
        assert not ok2 and why_frag in why


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
    # variant SET semantics (r2 item 6): a reordered but identical set
    # resumes; a genuinely different set re-runs
    (out / "readout_pools.json").write_text("{}")
    ok5, why5 = _seed_output_resume_ok(
        out, commit="c" * 40, seed=0, map_variants=["shufpair", "true"]
    )
    assert ok5, why5


# ---------------------------------------------------------------------------
# 9. r2 round-3 regressions: smoke-layer identity prefix, smoke enforcement,
#    bounded scratch wipe, crumb identity, gate-1 sibling gate, write order
# ---------------------------------------------------------------------------

# The REPRODUCED evil committed frozen indices from the round-2 review
# (positional indices into the full 28-layer grid): {15, 18, 20, 21, 22}.
_COMMITTED_EVIL_IDX = {
    "arm1_ctx_e1": 15,
    "arm4_ridge_ctx": 18,
    "arm6_map_proj_e1": 20,
    "arm7_map_ridge_pred": 21,
    "arm11_oracle_proj": 22,
    "arm2_ctx_native": 20,
}


def _real_shaped_committed_summary(tmp_path, committed: dict[str, int], n_layers: int = 28):
    """A committed train summary with the REAL arm_rows schema the scorer reads."""
    rows = []
    for arm, idx in committed.items():
        rho = [0.0] * n_layers
        rho[idx] = 0.9
        rows.append(
            {
                "arm": arm,
                "variant": "context_end",
                "regime": "e1",
                "u_rung_label": "full",
                "f_u": None,
                "rho_per_layer": rho,
            }
        )
    summary = tmp_path / "evil" / "arm_results" / "all_arms_spearman.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({"arm_rows": rows}))
    return summary


def test_smoke_layers_identity_prefix_through_scorer_guard(tmp_path):
    """Regression (#1739 claim4 r2 item 1, launch-blocking): the REAL
    _smoke_layers output must pass the REAL scorer guard chain
    (committed_frozen -> _assert_committed_frozen_indexable) against a
    real-shaped committed summary — the bare committed-index list the round-2
    code emitted is exactly what the guard REFUSES."""
    from types import SimpleNamespace

    from scripts.issue1739_jobd_r2aug import committed_frozen
    from scripts.issue1739_r2v2_run import CLAIM4_EXTRA_ARMS, _smoke_layers
    from scripts.issue1739_r2v2_score import MATCHED_FROZEN_COMPANIONS, ROSTER

    summary = _real_shaped_committed_summary(tmp_path, _COMMITTED_EVIL_IDX)
    layers = _smoke_layers(SimpleNamespace(main_root=tmp_path), "evil")
    # identity prefix through the max committed index (22) — the guard's
    # precondition — and still a real reduction vs the full 28-layer grid
    assert layers == list(range(23))
    # the REAL scorer-side chain (score.py run_behavior -> committed_frozen)
    roster_all = tuple(ROSTER) + tuple(a for a in CLAIM4_EXTRA_ARMS if a not in ROSTER)
    roster_frozen = tuple(a for a in roster_all if a not in MATCHED_FROZEN_COMPANIONS)
    args = SimpleNamespace(regime="e1")
    loaded = SimpleNamespace(paths={"train_summary": summary}, shas={})
    frozen, src = committed_frozen(args, loaded, "evil", "context_end", layers, roster_frozen)
    assert frozen == {a: i for a, i in _COMMITTED_EVIL_IDX.items() if a in roster_frozen}
    assert src.startswith("modal-committed-train-cells:")
    # the round-2 shape (the committed indices THEMSELVES as --layers) is
    # refused by the same chain — the bug this test pins
    with pytest.raises(RuntimeError, match="identity prefix"):
        committed_frozen(
            args,
            loaded,
            "evil",
            "context_end",
            sorted(set(_COMMITTED_EVIL_IDX.values())),
            roster_frozen,
        )


def test_smoke_layers_min_two_layers(tmp_path):
    """A degenerate all-at-index-0 summary still yields a >=2-layer prefix."""
    from types import SimpleNamespace

    from scripts.issue1739_r2v2_run import _smoke_layers

    _real_shaped_committed_summary(tmp_path, dict.fromkeys(_COMMITTED_EVIL_IDX, 0))
    assert _smoke_layers(SimpleNamespace(main_root=tmp_path), "evil") == [0, 1]


def test_runner_requires_smoke_for_claim4_production_launch():
    """Regression (#1739 claim4 r2 item 2): a fits-claim4 launch without
    --smoke is a parse-time error; --skip-smoke REASON is the explicit
    opt-out; --import-check / --stage-only (never score) are exempt."""
    from scripts.issue1739_r2v2_run import parse_args as run_parse_args

    with pytest.raises(SystemExit):  # bare omission is not an opt-out
        run_parse_args(["--behaviors", "evil", "--legs", "fits-claim4"])
    with pytest.raises(SystemExit):  # an empty reason is not a reason
        run_parse_args(["--behaviors", "evil", "--legs", "fits-claim4", "--skip-smoke", " "])
    with pytest.raises(SystemExit):  # contradictory flags
        run_parse_args(
            ["--behaviors", "evil", "--legs", "fits-claim4", "--smoke", "--skip-smoke", "x"]
        )
    args = run_parse_args(
        [
            "--behaviors",
            "sycophancy",
            "--legs",
            "fits-claim4",
            "--skip-smoke",
            "smoke ran on the seeds-0-2 pod",
        ]
    )
    assert args.skip_smoke == "smoke ran on the seeds-0-2 pod"
    # exempt paths parse without either flag
    assert run_parse_args(["--legs", "fits-claim4", "--import-check"]).import_check
    assert run_parse_args(["--legs", "fits-claim4", "--stage-only"]).stage_only
    # non-claim4 legs are untouched
    assert run_parse_args(["--legs", "fits"]).legs == ["fits"]


def test_smoke_root_delete_guard(tmp_path, monkeypatch):
    """Regression (#1739 claim4 r2 item 3): the recursive scratch wipe is
    BOUNDED to the recognized pattern — basename contains
    claim4_controls_smoke AND resolves under eval_results/issue_1739/."""
    from scripts.issue1739_r2v2_run import _assert_smoke_scratch_root

    monkeypatch.chdir(tmp_path)
    good = tmp_path / "eval_results" / "issue_1739" / "claim4_controls_smoke"
    good.mkdir(parents=True)
    assert _assert_smoke_scratch_root(Path("eval_results/issue_1739/claim4_controls_smoke")) == (
        good.resolve()
    )
    for bad in (
        tmp_path / "eval_results" / "issue_1739",  # the shared issue dir itself
        tmp_path / "eval_results" / "issue_1739" / "claim4_controls",  # PRODUCTION out-root
        tmp_path / "claim4_controls_smoke",  # right name, arbitrary location
        tmp_path / "eval_results" / "issue_9999" / "claim4_controls_smoke",  # foreign issue
        Path("/"),
    ):
        with pytest.raises(RuntimeError, match="refusing to wipe"):
            _assert_smoke_scratch_root(bad)


def test_crumb_matches_rejects_unknown_sha_pair():
    """Regression (#1739 claim4 r2 item 5): a matching pair of
    code_sha="unknown" placeholders must never verify — on EITHER side."""
    from scripts.issue1739_r2v2_run import _crumb_matches

    ok, why = _crumb_matches(
        {"code_sha": "unknown", "run_token": "t1"}, code_sha="unknown", run_token="t1"
    )
    assert not ok and "unknown" in why
    ok2, why2 = _crumb_matches(
        {"code_sha": "unknown", "run_token": "t1"}, code_sha="a" * 40, run_token="t1"
    )
    assert not ok2 and "payload code_sha invalid" in why2
    ok3, why3 = _crumb_matches(
        {"code_sha": "a" * 40, "run_token": "t1"}, code_sha="", run_token="t1"
    )
    assert not ok3 and "identity unverifiable" in why3


def test_runner_git_sha_fails_loud_when_unresolvable(monkeypatch):
    """_runner_git_sha raises (never returns "unknown") when rev-parse fails
    — the crumb WRITE side of item 5."""
    import subprocess as sp

    import scripts.issue1739_r2v2_run as run_mod

    real_run = sp.run

    def fake_run(cmd, **kw):
        if cmd[:1] == ["git"]:
            return sp.CompletedProcess(cmd, 128, stdout="", stderr="fatal: not a git repo")
        return real_run(cmd, **kw)

    monkeypatch.setattr(run_mod.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="cannot resolve the running code SHA"):
        run_mod._runner_git_sha()


class _FakeGate1Api:
    """Signature-conformant fake of the HfApi surface wait_for_gate1_pass
    touches (file_exists + hf_hub_download) — the only faked boundary is the
    network; the REAL wait body runs."""

    def __init__(self, payload_path=None, exists=True, token=None):
        self._payload_path = payload_path
        self._exists = exists

    def file_exists(self, repo_id, filename, *, repo_type=None):
        return self._exists

    def hf_hub_download(self, repo_id, filename, *, repo_type=None, force_download=False):
        return str(self._payload_path)


def _gate1_args(tmp_path, timeout_s=2):
    from types import SimpleNamespace

    return SimpleNamespace(
        stage_run_token="tok1",
        gate1_timeout_s=timeout_s,
        stage_gate_poll_s=0,
    )


def test_gate1_wait_pass_fail_and_timeout(tmp_path, monkeypatch):
    """Regression (#1739 claim4 r2 item 7): the seeds-3-4 pod's gate-1 wait
    returns on a VERIFIED PASS crumb, ABORTS loudly on a verified FAIL crumb,
    and ABORTS on timeout (correctness, unlike the politeness staging gate).
    Runs the REAL wait body with only the Hub boundary faked."""
    import huggingface_hub

    import scripts.issue1739_r2v2_run as run_mod

    own_sha = run_mod._runner_git_sha()
    crumb = tmp_path / "gate1.json"

    def _write(rc):
        crumb.write_text(json.dumps({"code_sha": own_sha, "run_token": "tok1", "gate1_rc": rc}))

    def _patch_api(**kw):
        monkeypatch.setattr(
            huggingface_hub,
            "HfApi",
            lambda token=None: _FakeGate1Api(payload_path=crumb, **kw),
        )

    _write(0)
    _patch_api()
    run_mod.wait_for_gate1_pass(_gate1_args(tmp_path), "evil", "")  # PASS: returns

    _write(3)
    with pytest.raises(RuntimeError, match=r"gate-1 FAILED .*kill \(a\)"):
        run_mod.wait_for_gate1_pass(_gate1_args(tmp_path), "evil", "")

    # identity-mismatched crumb reads as absent -> polls into the timeout abort
    crumb.write_text(json.dumps({"code_sha": own_sha, "run_token": "OLD", "gate1_rc": 0}))
    with pytest.raises(RuntimeError, match="ABSENT after"):
        run_mod.wait_for_gate1_pass(_gate1_args(tmp_path, timeout_s=1), "evil", "")

    # no crumb at all -> timeout abort
    _patch_api(exists=False)
    with pytest.raises(RuntimeError, match="ABSENT after"):
        run_mod.wait_for_gate1_pass(_gate1_args(tmp_path, timeout_s=1), "evil", "")


def test_gate1_crumb_path_token_keyed():
    from scripts.issue1739_r2v2_run import _gate1_crumb_path, _stage_crumb_path

    assert _gate1_crumb_path("evil", "tokA").endswith("evil_gate1_tokA.json")
    # distinct from the staging crumb (two gates, two breadcrumbs)
    assert _gate1_crumb_path("evil", "tokA") != _stage_crumb_path("evil", "tokA")


def test_seed_write_order_summary_is_completion_sentinel(tmp_path):
    """Regression (#1739 claim4 r2 item 6): companions are written FIRST and
    the validated summary LAST — an interrupt at the summary write leaves NO
    passing resume predicate over the mixed-generation partial output."""
    from scripts.issue1739_r2v2_score import (
        _seed_output_resume_ok,
        _write_companions_then_summary,
    )

    out = tmp_path / "evil" / "seed0"
    out.mkdir(parents=True)
    res = {"map_diagnostics": {"d": 1}, "pools": {}, "fit_reports": {}}
    seen = {}

    def _interrupted_summary_writer():
        seen["companions_present_at_summary_write"] = (out / "map_diagnostics.json").exists() and (
            out / "readout_pools.json"
        ).exists()
        raise KeyboardInterrupt  # the interrupt lands AT the summary write

    with pytest.raises(KeyboardInterrupt):
        _write_companions_then_summary(out, res, _interrupted_summary_writer)
    assert seen["companions_present_at_summary_write"]  # companions FIRST
    ok, why = _seed_output_resume_ok(
        out, commit="c" * 40, seed=0, map_variants=["true", "shufpair"]
    )
    assert not ok and "all_arms_spearman.json absent" in why


# ---------------------------------------------------------------------------
# 6. crash-fix r4: row-index pushdown (the 128 GB-cgroup OOM fix) — BITWISE
#    equivalence to the pre-r4 materialized split copies, on both Gram routes
# ---------------------------------------------------------------------------


def _pushdown_pool(n=40, d=8, ly=3, seed=7):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(ly, n, d))
    y = 0.5 * x + rng.normal(size=(ly, n, d)) * 0.1
    return x, y


def test_ridge_row_pushdown_bitwise_primal():
    """train_rows/eval_rows pushdown == materialized fancy-index copies,
    BITWISE (np.array_equal), on the primal (n_tr > d) route — preds AND
    weights. This is the r4 OOM fix's numerical contract: the claim4 gate 1
    asserts seed-0 reproduction against banked artifacts, so the memory
    refactor must not perturb one bit of the fit."""
    from explore_persona_space.experiments.issue_1739 import fits

    x, y = _pushdown_pool(n=40, d=8)
    rng = np.random.default_rng(3)
    perm = rng.permutation(40)
    hold, tr = perm[:8], perm[8:]  # n_tr=32 > d=8 -> primal
    p_ref, w_ref = fits.ridge_layer_batched_auto(
        x[:, tr], y[:, tr], x[:, hold], return_weights=True
    )
    p_new, w_new = fits.ridge_layer_batched_auto(
        x, y, x, return_weights=True, train_rows=tr, eval_rows=hold
    )
    assert np.array_equal(p_ref, p_new)
    assert np.array_equal(w_ref, w_new)


def test_ridge_row_pushdown_bitwise_dual():
    """Same contract on the dual route (n_tr <= d): the auto-router
    materializes the gathers exactly as the pre-r4 caller did."""
    from explore_persona_space.experiments.issue_1739 import fits

    x, y = _pushdown_pool(n=12, d=16)
    rng = np.random.default_rng(4)
    perm = rng.permutation(12)
    hold, tr = perm[:4], perm[4:]  # n_tr=8 <= d=16 -> dual
    p_ref = fits.ridge_layer_batched_auto(x[:, tr], y[:, tr], x[:, hold])
    p_new = fits.ridge_layer_batched_auto(x, y, x, train_rows=tr, eval_rows=hold)
    assert np.array_equal(p_ref, p_new)


def test_layer_row_gather_matches_whole_array_gather():
    """_LayerRowGather[li] == pool[:, rows][li] element-for-element with the
    same C-contiguous layout (map_diagnostics only ever indexes [li])."""
    from explore_persona_space.experiments.issue_1739.fits import _LayerRowGather

    x, _ = _pushdown_pool(n=20, d=5)
    rows = np.asarray([3, 1, 7, 11])
    facade = _LayerRowGather(x, rows)
    ref = x[:, rows]
    for li in range(x.shape[0]):
        got = facade[li]
        assert np.array_equal(got, ref[li])
        assert got.flags["C_CONTIGUOUS"]


def test_fit_linear_map_pushdown_bitwise_vs_materialized_reference():
    """fit_linear_map (r4 pushdown internals) == the pre-r4 materialized
    construction, BITWISE: weights, standardization params, and every
    per-layer diagnostic (r2_map / r2_identity_bias / kNN) — on the primal
    route (production shape class) and the dual route."""
    from explore_persona_space.experiments.issue_1739 import fits

    for n, d in ((40, 8), (12, 16)):  # primal / dual
        x, y = _pushdown_pool(n=n, d=d, seed=11)
        seed = 0
        mf = fits.fit_linear_map(x, y, seed=seed)
        # pre-r4 reference construction, inline (whole-array split copies)
        rng = np.random.default_rng([1739, 4, seed])
        perm = rng.permutation(n)
        n_hold = max(2, round(fits.WHITEN_HOLDOUT_FRAC * n))
        hold, tr = perm[:n_hold], perm[n_hold:]
        x_tr, y_tr, x_ho, y_ho = x[:, tr], y[:, tr], x[:, hold], y[:, hold]
        preds_hold = fits.ridge_layer_batched_auto(x_tr, y_tr, x_ho)
        diag_ref = fits.map_diagnostics(preds_hold, x_ho, y_ho, x_tr, y_tr)
        _p, w_ref = fits.ridge_layer_batched_auto(x, y, x[:, :2], return_weights=True)
        assert np.array_equal(mf.w, w_ref), (n, d)
        assert np.array_equal(mf.x_mu, x.mean(axis=1, keepdims=True))
        assert np.array_equal(mf.y_mu, y.mean(axis=1, keepdims=True))
        for got, ref in zip(mf.diagnostics["per_layer"], diag_ref["per_layer"], strict=True):
            assert got["r2_map"] == ref["r2_map"], (n, d)
            assert got["r2_identity_bias"] == ref["r2_identity_bias"], (n, d)
            assert got["knn"] == ref["knn"], (n, d)


# ---------------------------------------------------------------------------
# 7. crash-fix r4: leg-keyed sentinel naming (plan §9 phase_outputs contract)
# ---------------------------------------------------------------------------


def test_sentinel_name_leg_keyed_for_claim4_and_legacy_elsewhere():
    from scripts.issue1739_r2v2_run import sentinel_name

    # claim4 leg: plan §9 name issue-1739-claim4-<behavior>-<half>.json
    assert (
        sentinel_name(["fits-claim4"], [0, 1, 2], "sycophancy")
        == "issue-1739-claim4-sycophancy-s0-1-2.json"
    )
    assert sentinel_name(["fits-claim4"], [3, 4]) == "issue-1739-claim4-all-s3-4.json"
    # the two pod halves can never collide with each other...
    assert sentinel_name(["fits-claim4"], [0, 1, 2], "evil") != sentinel_name(
        ["fits-claim4"], [3, 4], "evil"
    )
    # ...nor with a real fits-leg sentinel
    assert "r2v2fits" not in sentinel_name(["fits-claim4"], [0], "evil")
    # every other leg keeps the legacy names byte-identically
    assert sentinel_name(["fits"], [0, 1, 2], "evil") == "issue-1739-r2v2fits-evil.json"
    assert sentinel_name(["pc"], [0]) == "issue-1739-r2v2fits-all.json"


def test_write_sentinel_atomic_and_drainable(tmp_path):
    from scripts.issue1739_r2v2_run import _write_sentinel

    path = _write_sentinel(tmp_path / "logs", "issue-1739-claim4-evil-s0.json", {"rc": 0})
    assert path.exists()
    assert json.loads(path.read_text()) == {"rc": 0}
    # no half-written tmp residue left behind to confuse the poller glob
    assert list((tmp_path / "logs").glob("*.tmp")) == []
