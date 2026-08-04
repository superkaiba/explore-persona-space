"""Equivalence pin for the rb-independent transfer-fit cache hoist (#1739).

``scripts/issue1739_rescore_ood.py`` (and the armfill sibling) hold the
rb-INDEPENDENT arm fits in a cache hoisted ABOVE the ``(variant, regime)``
group loop, keyed by ``(variant, rs_key)`` — so a fit computed under one
regime is reused for every other regime of the same variant. The hoist is
output-preserving iff the rb_indep half of :func:`arms.run_transfer_cell` is
a pure function of regime-free inputs. Three legs pin that:

1. two SEPARATE ``run_transfer_cell`` calls whose inputs differ ONLY in the
   regime direction ``rb`` — and are otherwise fresh array objects, mimicking
   the per-group recompute — return bit-identical rb_indep scores, while the
   rb_dep arms genuinely change (so the partition is load-bearing, not
   vacuous);
2. simulating the OLD per-group cache against the NEW hoisted
   ``(variant, rs_key)`` cache over a (variant x regime x row-set) grid gives
   bit-identical per-unit rb_indep score dicts;
3. the per-group whitening / linear-map refits are deterministic (identical
   bytes in -> identical fit out), closing the "each group recomputes ``wh``
   and ``mapfit``, maybe differently" hole — those recomputes feed the
   rb_indep call, so their determinism is part of the hoist's correctness.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from explore_persona_space.experiments.issue_1739 import arms  # noqa: E402
from explore_persona_space.experiments.issue_1739.fits import (  # noqa: E402
    BudgetCell,
    MapFit,
    apply_whitening,
    fit_linear_map,
    fit_whitening,
)

LY, D, N_TR, N_EV = 2, 5, 24, 9


def _mapfit_copy(m: MapFit) -> MapFit:
    """Fresh-object MapFit (same values) — mimics the per-group map refit."""
    return MapFit(
        w=m.w.copy(),
        x_mu=m.x_mu.copy(),
        x_sd=m.x_sd.copy(),
        y_mu=m.y_mu.copy(),
        diagnostics=dict(m.diagnostics),
        kind=m.kind,
    )


def _toy_tables(seed: int = 0):
    """One variant's train/eval tables + a linear mapfit (toy scale)."""
    rng = np.random.default_rng(seed)
    z_tr = rng.normal(size=(LY, N_TR, D))
    za_tr = z_tr + 0.3 * rng.normal(size=(LY, N_TR, D))
    dv_tr = rng.normal(size=N_TR)
    z_ev = rng.normal(size=(LY, N_EV, D))
    za_ev = rng.normal(size=(LY, N_EV, D))
    dv_ev = rng.normal(size=N_EV)
    mapfit = MapFit(
        w=np.stack([np.eye(D) for _ in range(LY)]),
        x_mu=np.zeros((LY, 1, D)),
        x_sd=np.ones((LY, 1, D)),
        y_mu=np.zeros((LY, 1, D)),
        diagnostics={},
        kind="linear",
    )
    return z_tr, za_tr, dv_tr, z_ev, za_ev, dv_ev, mapfit


def _cell(row_idx: np.ndarray) -> BudgetCell:
    n = len(row_idx)
    return BudgetCell(
        row_idx=np.asarray(row_idx, dtype=np.int64),
        fold_ids=np.arange(n) % 3,
        n_folds=3,
        budget_l=n,
        draw=0,
        seed=0,
        fold_scheme="toy",
    )


def _assert_scores_equal(a: dict, b: dict, *, ctx: str) -> None:
    assert sorted(a) == sorted(b), (ctx, sorted(a), sorted(b))
    for slug in a:
        assert np.array_equal(a[slug], b[slug], equal_nan=True), (
            f"{ctx}: {slug} scores differ — the rb_indep fit is NOT a pure "
            "function of regime-free inputs; the hoisted cache would change results"
        )


def test_rb_indep_bit_identical_across_regime_only_rb_change():
    """Fresh-object inputs + a different rb -> bit-identical rb_indep scores.

    This is exactly the reuse the hoisted cache performs: group (variant, e2)
    reuses the fit group (variant, e1) computed, where the two groups'
    recomputed inputs are value-identical fresh objects and only ``rb``
    (dead to rb_indep arms) differs. Includes arm 5 (the MLP — its init seed
    is the module DEFAULT_MLP_SEED, never ``cell.seed``).
    """
    z_tr, za_tr, dv_tr, z_ev, za_ev, dv_ev, mapfit = _toy_tables()
    rng = np.random.default_rng(99)
    rb_e1 = rng.normal(size=(LY, D))
    rb_e2 = rng.normal(size=(LY, D))
    assert not np.array_equal(rb_e1, rb_e2)
    rb_indep, rb_dep = arms.partition_transfer_roster(arms.TRANSFER_ARMS_WIDE)
    cell = _cell(np.arange(N_TR))

    data_e1 = arms.CellData(
        z_ctx=z_tr, z_ans=za_tr, dv=dv_tr, rb=rb_e1, mapfit=mapfit, layers=(0, 1)
    )
    # regime e2: SAME values, FRESH objects (the per-group recompute), new rb
    data_e2 = arms.CellData(
        z_ctx=z_tr.copy(),
        z_ans=za_tr.copy(),
        dv=dv_tr.copy(),
        rb=rb_e2,
        mapfit=_mapfit_copy(mapfit),
        layers=(0, 1),
    )

    kw = dict(za_ev=za_ev, device="cpu", ridge_folds=(0,))
    s1, sk1 = arms.run_transfer_cell(data_e1, cell, z_ev, dv_ev, arms=rb_indep, **kw)
    s2, sk2 = arms.run_transfer_cell(
        data_e2, _cell(np.arange(N_TR)), z_ev.copy(), dv_ev.copy(), arms=rb_indep, **kw
    )
    assert sk1 == sk2
    assert set(s1) == set(rb_indep) - set(sk1), (sorted(s1), sk1)
    _assert_scores_equal(s1, s2, ctx="rb_indep across regimes")

    # sanity: the rb change is not a no-op — every rb_dep PROJECTION arm moves
    # (a vacuous rb delta would make the equality above prove nothing)
    d1, dsk1 = arms.run_transfer_cell(data_e1, cell, z_ev, dv_ev, arms=rb_dep, **kw)
    d2, _ = arms.run_transfer_cell(data_e2, cell, z_ev, dv_ev, arms=rb_dep, **kw)
    moved = [
        slug for slug in d1 if slug in d2 and not np.array_equal(d1[slug], d2[slug], equal_nan=True)
    ]
    assert "arm1_ctx_e1" in moved and "arm11_oracle_proj" in moved, (moved, dsk1)


def test_hoisted_cache_matches_per_group_cache():
    """OLD per-group ``rs_key`` cache == NEW hoisted ``(variant, rs_key)`` cache.

    Reconstructs the rescore scripts' cache-key semantics over 2 variants x
    2 regimes x 2 realized row sets (cheap ridge-only rb_indep roster) and
    asserts every (group, unit)'s rb_indep score dict is bit-identical under
    both disciplines.
    """
    tables = {"context_end": _toy_tables(seed=1), "prefix_end": _toy_tables(seed=2)}
    rb_by_vr = {
        (v, r): np.random.default_rng([1739, vi, ri]).normal(size=(LY, D))
        for vi, v in enumerate(sorted(tables))
        for ri, r in enumerate(("e1", "e2"))
    }
    row_sets = {"full": np.arange(N_TR), "sub": np.arange(0, N_TR, 2)}
    roster = ["arm4_ridge_ctx", "arm12_oracle_reg"]

    def _run(hoisted: bool) -> dict:
        results: dict = {}
        cache_hoisted: dict = {}
        for variant, regime in sorted(rb_by_vr):
            cache = cache_hoisted if hoisted else {}
            z_tr, za_tr, dv_tr, z_ev, za_ev, dv_ev, mapfit = tables[variant]
            # per-GROUP fresh objects, as the scripts' per-group re-whitening produces
            z_tr, za_tr, z_ev, za_ev = (a.copy() for a in (z_tr, za_tr, z_ev, za_ev))
            data = arms.CellData(
                z_ctx=z_tr,
                z_ans=za_tr,
                dv=dv_tr.copy(),
                rb=rb_by_vr[(variant, regime)],
                mapfit=_mapfit_copy(mapfit),
                layers=(0, 1),
            )
            for unit, rows in sorted(row_sets.items()):
                cell = _cell(rows)
                rs_key = hashlib.sha1(cell.row_idx.tobytes()).hexdigest()
                key = (variant, rs_key) if hoisted else rs_key
                if key not in cache:
                    cache[key] = arms.run_transfer_cell(
                        data,
                        cell,
                        z_ev,
                        dv_ev,
                        za_ev=za_ev,
                        arms=roster,
                        device="cpu",
                        ridge_folds=(0,),
                    )
                results[(variant, regime, unit)] = cache[key]
        return results

    old, new = _run(hoisted=False), _run(hoisted=True)
    assert sorted(old) == sorted(new)
    for gk in old:
        (s_old, sk_old), (s_new, sk_new) = old[gk], new[gk]
        assert sk_old == sk_new, gk
        _assert_scores_equal(s_old, s_new, ctx=f"group-unit {gk}")


def test_per_group_whitening_and_map_refits_are_deterministic():
    """Identical U-pool bytes -> bit-identical whitening + linear-map refits.

    The scripts refit ``wh`` and ``mapfit`` inside EVERY group from
    regime-free inputs; the hoist reuses arrays whitened by ONE group's
    ``wh`` in later groups, so those refits must be reproducible."""
    rng = np.random.default_rng(3)
    u_x = rng.normal(size=(LY, 40, D))
    u_y = u_x + 0.1 * rng.normal(size=(LY, 40, D))

    wh_a = fit_whitening(u_x, device="cpu", seed=42)
    wh_b = fit_whitening(u_x.copy(), device="cpu", seed=42)
    for attr in ("mu", "w", "gamma"):
        assert np.array_equal(getattr(wh_a, attr), getattr(wh_b, attr)), attr

    zw_a = apply_whitening(u_x, wh_a)
    zw_b = apply_whitening(u_x.copy(), wh_b)
    assert np.array_equal(zw_a, zw_b)

    m_a = fit_linear_map(zw_a, apply_whitening(u_y, wh_a), device="cpu")
    m_b = fit_linear_map(zw_b, apply_whitening(u_y.copy(), wh_b), device="cpu")
    assert np.array_equal(m_a.w, m_b.w)
    assert np.array_equal(m_a.x_mu, m_b.x_mu)
    assert np.array_equal(m_a.x_sd, m_b.x_sd)
    assert np.array_equal(m_a.y_mu, m_b.y_mu)
