"""Arm-roster / gating-input regression pins for the #1739 eval-rung drivers.

Origin (2026-08-03): `scripts/issue1739_rescore_ood.py` shipped with TWO
independent defects, both instances of the same root cause — a fresh driver
reimplementing what canonical in-repo helpers already provide instead of
reusing them (the CLAUDE.md "Reuse existing in-repo tools/helpers" rule):

1. It declared its own 16-slug `_ALL16_NAMES` roster instead of resolving
   `arms.TRANSFER_ARMS_WIDE` via `arms.resolve_transfer_roster`. That roster
   included the arms the transfer leg DELIBERATELY excludes (9/14 L2-SP,
   10 stacked — which RAISES under `ridge_folds=(0,)` — and 15/16 text).
2. It built its `arms.CellData` with no `mapfit`, so the four map-consuming
   arms that ARE canonical here (the map family 6/7/8 and its shuffled-map
   null 13) were skipped with reason "no mapfit" — silently gutting the very
   map-vs-context comparison the round exists to make.

It also hand-rolled a local `_whiten_acts` closure whose
`z - wh.mu[None, None, :]` broadcast raised ValueError for every realistic
shape (`wh.mu` is `(Ly, d)`) and silently produced WRONG values in the
degenerate `n == Ly` case a tiny smoke would hit.

These tests pin the BEHAVIOR (the roster produces every arm) and the
STRUCTURE (the driver reuses the canonical helpers), so the next driver in
this family cannot regress the same way.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from explore_persona_space.experiments.issue_1739 import arms, fits

# The tree THIS test lives in — deliberately NOT task_workflow.repo_root(),
# which branch-guards to `main` and would read the pre-fix copy of the driver
# while the fix is still on the issue worktree branch.
TREE_ROOT = Path(__file__).resolve().parents[1]

# Arms that consume a fitted context->answer map. Kept in sync with
# `run_cell_multi`'s own `mp_arms` set; the test below asserts that sync.
MAP_CONSUMING_ARMS = frozenset(
    {
        "arm6_map_proj_e1",
        "arm7_map_ridge_pred",
        "arm8_map_ridge_true",
        "arm9_pretrain_ft",
        "arm10_stacked",
        "arm13_shuffled_map",
        "arm14_shuffled_pt",
        # fair-protocol round (2026-08-05, commit aa6e040543): MLP readout on
        # the MAPPED answer — consumes the fitted map exactly like arm 7.
        "arm19_map_mlp_pred",
        # fair-roster follow-up (2026-08-06): ridge on the SHUFFLED-weight
        # mapped answer — consumes the map's weight tensor (permuted), exactly
        # like arm 13; grouped in mp_arms for the "no mapfit" skip semantics.
        "arm20_shuffled_map_ridge",
    }
)

RESCORE_OOD = Path("scripts/issue1739_rescore_ood.py")


def _driver_source(rel: Path) -> str:
    return (TREE_ROOT / rel).read_text(encoding="utf-8")


def _toy_cell(*, mapfit: fits.MapFit | None, seed: int = 0):
    """A tiny synthetic transfer cell: (data, cell, z_ev, dv_ev, za_ev)."""
    rng = np.random.default_rng(seed)
    n_layers, n_train, n_eval, dim = 2, 40, 20, 6
    data = arms.CellData(
        z_ctx=rng.standard_normal((n_layers, n_train, dim)),
        z_ans=rng.standard_normal((n_layers, n_train, dim)),
        dv=rng.uniform(0, 100, n_train),
        rb=rng.standard_normal((n_layers, dim)),
        mapfit=mapfit,
        layers=tuple(range(n_layers)),
    )
    cell = fits.realize_budget_cell(np.arange(n_train) % 5, budget_l=n_train, draw=0, seed=0)
    return (
        data,
        cell,
        rng.standard_normal((n_layers, n_eval, dim)),
        rng.uniform(0, 100, n_eval),
        rng.standard_normal((n_layers, n_eval, dim)),
    )


def _toy_mapfit(*, n_layers: int = 2, dim: int = 6, seed: int = 0) -> fits.MapFit:
    rng = np.random.default_rng(seed)
    return fits.MapFit(
        w=rng.standard_normal((n_layers, dim, dim)) * 0.1,
        x_mu=np.zeros((n_layers, 1, dim)),
        x_sd=np.ones((n_layers, 1, dim)),
        y_mu=np.zeros((n_layers, 1, dim)),
        diagnostics={},
        kind="linear",
    )


def test_map_consuming_arm_set_matches_run_cell_multi():
    """The constant above must track `run_cell_multi`'s own `mp_arms` set.

    If arms.py adds/removes a map-consuming arm, this test fails loudly rather
    than letting the roster pins below silently under-check.
    """
    src = (TREE_ROOT / "src/explore_persona_space/experiments/issue_1739/arms.py").read_text(
        encoding="utf-8"
    )
    block = re.search(r"mp_arms\s*=\s*\{(.*?)\}", src, re.DOTALL)
    assert block, "could not locate the `mp_arms` set literal in arms.py"
    found = set(re.findall(r'"(arm\d+_[a-z0-9_]+)"', block.group(1)))
    assert found == set(MAP_CONSUMING_ARMS), (
        "MAP_CONSUMING_ARMS drifted from run_cell_multi's mp_arms set: "
        f"only-in-test={sorted(set(MAP_CONSUMING_ARMS) - found)} "
        f"only-in-arms.py={sorted(found - set(MAP_CONSUMING_ARMS))}"
    )


def test_canonical_wide_roster_produces_every_arm_with_a_mapfit():
    """THE regression pin: canonical roster + a mapfit ⇒ zero skipped arms.

    Guards the #1739 defect where the eval-rung re-score emitted skip reasons
    instead of values for the whole map family.
    """
    roster = arms.resolve_transfer_roster(None)
    data, cell, z_ev, dv_ev, za_ev = _toy_cell(mapfit=_toy_mapfit())
    scores, skipped = arms.run_transfer_cell(
        data, cell, z_ev, dv_ev, za_ev=za_ev, arms=roster, ridge_folds=(0,)
    )
    assert not skipped, f"canonical roster skipped arms: {skipped}"
    assert set(scores) == set(roster)
    # the map family + its null are the arms the round's headline needs
    assert {
        "arm6_map_proj_e1",
        "arm7_map_ridge_pred",
        "arm8_map_ridge_true",
        "arm13_shuffled_map",
    } <= set(scores)


def test_missing_mapfit_skips_the_map_family():
    """Documents the gate: no mapfit ⇒ map arms are SKIPPED, never zero-filled.

    This is correct fail-visible behavior in arms.py (a recorded skip reason,
    not a silent zero). The defect was the DRIVER never supplying a map — the
    structural pins below are what prevent that recurring.
    """
    roster = arms.resolve_transfer_roster(None)
    data, cell, z_ev, dv_ev, za_ev = _toy_cell(mapfit=None)
    scores, skipped = arms.run_transfer_cell(
        data, cell, z_ev, dv_ev, za_ev=za_ev, arms=roster, ridge_folds=(0,)
    )
    map_arms_in_roster = MAP_CONSUMING_ARMS & set(roster)
    assert map_arms_in_roster, "roster should contain map-consuming arms"
    for slug in map_arms_in_roster:
        assert slug in skipped and "no mapfit" in skipped[slug]
        assert slug not in scores, f"{slug} must be skipped, never zero-filled"


def test_rescore_ood_resolves_the_canonical_roster():
    """The driver must not re-declare a local arm roster."""
    src = _driver_source(RESCORE_OOD)
    assert "resolve_transfer_roster" in src, (
        "issue1739_rescore_ood.py must resolve the canonical transfer roster "
        "via arms.resolve_transfer_roster, not a locally-declared arm list"
    )
    # `_ALL16_NAMES` survives ONLY as a legacy export for the sibling
    # train-side driver (issue1739_holdout_rung.py). This driver must not use
    # it to build its OWN eval-rung roster — that list includes the arms the
    # transfer leg deliberately excludes (9/10/14/15/16).
    code_lines = [ln for ln in src.splitlines() if not ln.lstrip().startswith("#")]
    misuse = [
        ln.strip()
        for ln in code_lines
        if "_ALL16_NAMES" in ln and not ln.lstrip().startswith("_ALL16_NAMES")
    ]
    assert not misuse, (
        "issue1739_rescore_ood.py must not build its eval-rung roster from the "
        f"legacy _ALL16_NAMES list; use resolve_transfer_roster. Offending: {misuse}"
    )


def test_rescore_ood_threads_mapfit_into_celldata():
    """The driver must supply the gating input its roster requires."""
    src = _driver_source(RESCORE_OOD)
    assert re.search(r"\bmapfit\s*=\s*mapfit\b", src), (
        "issue1739_rescore_ood.py must pass mapfit= into arms.CellData; without "
        "it the map family (6/7/8) and its null (13) are skipped 'no mapfit'"
    )
    assert re.search(r"_fit_map\s*\(", src), (
        "the map must be refit in-process from the U pool via _fit_map (the "
        "same estimator the main run / pvsynth / wcrung legs use)"
    )


def test_rescore_ood_uses_canonical_whitening_helper():
    """No local whitening reimplementation.

    The retired local closure used `wh.mu[None, None, :]`, which cannot
    broadcast against (Ly, n, d) because `wh.mu` is (Ly, d).
    """
    src = _driver_source(RESCORE_OOD)
    assert "fits.apply_whitening" in src, (
        "use the canonical fits.apply_whitening (per-layer centering, chunked, "
        "bit-identical-pinned) instead of a local whitening closure"
    )
    # Check CODE only — the driver's own explanatory comment names the retired
    # broken expression on purpose, and must not trip this pin.
    code_lines = [ln for ln in src.splitlines() if not ln.lstrip().startswith("#")]
    offenders = [ln.strip() for ln in code_lines if "mu[None, None, :]" in ln]
    assert not offenders, (
        "wh.mu is (Ly, d); `mu[None, None, :]` raises ValueError against "
        f"(Ly, n, d) and silently mis-centers when n == Ly. Offending: {offenders}"
    )


def test_local_whitening_broadcast_is_actually_broken():
    """Pins WHY the local closure was wrong, so nobody 'restores' it.

    `wh.mu` is (Ly, d): the retired expression raises for n != Ly and returns
    values that disagree with the canonical helper in the n == Ly case.
    """
    n_layers, n_rows, dim = 3, 7, 5
    rng = np.random.default_rng(0)
    z = rng.standard_normal((n_layers, n_rows, dim))
    mu = rng.standard_normal((n_layers, dim))
    with pytest.raises(ValueError):
        _ = z - mu[None, None, :]

    # degenerate n == Ly: broadcasts, but disagrees with per-layer centering
    z_sq = rng.standard_normal((n_layers, n_layers, dim))
    canonical = np.stack([z_sq[i] - mu[i][None, :] for i in range(n_layers)])
    assert not np.allclose(z_sq - mu[None, None, :], canonical)
