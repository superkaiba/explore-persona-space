"""Parity + unit tests for the #2091 dof-capped fits wrapper and judge contracts.

Covers the unit-B brief list:
- the wrapper in the #825 CV configuration reproduces
  ``issue825_fit_cells.heldout_r2_sweep``'s selected-lambda set + held-out
  predictions on a small synthetic grouped cell (matched folds/grid/cap);
- the dof-cap exclusion branch; the pure-GCV-at-n<d refusal branch;
- the S1 non-None single-draw pick; the zero-padded ``k00..k04`` parser (A3);
- the rule-27 parse-contract round-trip for the P3 judge instruments
  (realistic reasoning+score reply + fenced/markdown variant through the
  harness's OWN parse+reduce path, plus placeholder-substitution presence).

Everything is tiny + CPU; no network, no staged data, no GPU (<60 s).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts import issue825_fit_cells as fit825
from scripts.issue2091_fits import (
    DOF_CAP,
    LAMBDA_GRID,
    _dof_cap_exclusions,
    fit_predict_at_lambda,
    parse_per_rollout_scores,
    s1_single_draw_pick,
    select_lambda,
    synthetic_cell,
)
from scripts.issue2091_judge import (
    resolve_all_rubrics,
    resolve_rubric_no_regen,
    wave_rubric_fingerprint,
)

RNG_SEED = 2091


def _grouped_cell(n_groups=24, rows_per_group=5, d=16, d_out=6, seed=RNG_SEED):
    """Synthetic grouped (X, Y, conv_ids): linear map + noise, n >> d."""
    rng = np.random.default_rng(seed)
    n = n_groups * rows_per_group
    x = rng.normal(size=(n, d))
    w = rng.normal(size=(d, d_out)) / np.sqrt(d)
    y = x @ w + 0.5 * rng.normal(size=(n, d_out))
    conv_ids = np.array([f"g{g:03d}" for g in range(n_groups) for _ in range(rows_per_group)])
    return x.astype(np.float32), y.astype(np.float32), conv_ids


# ── parity: wrapper vs heldout_r2_sweep (#825 CV configuration) ───────────────
def test_wrapper_reproduces_heldout_r2_sweep_lambdas_and_preds():
    """Matched folds/grid/cap: identical selected-lambda set; predictions match
    through the documented std-convention bridge pred_825(lam) == core(lam*n/(n-1))."""
    x, y, conv_ids = _grouped_cell()
    n_folds, seed = 4, 7
    out = fit825.heldout_r2_sweep(
        x[:, None, :],  # (N, L=1, D)
        y[:, None, :],
        conv_ids,
        n_folds=n_folds,
        seed=seed,
        null_draws=0,
        collect_cosines=True,
        collect_lambdas=True,
        lambda_selection="inner-group-cv",
        frozen_layers=(0,),
    )
    gcv_lambda = out["gcv_lambda"]  # (L=1, n_folds)
    preds_frozen = out["preds_frozen"][0]  # (N, d_out)
    assert out["fitted_mask"].all() if "fitted_mask" in out else True

    folds = fit825._cv_folds(conv_ids, n_folds, seed)
    for k in range(n_folds):
        te = folds == k
        tr = ~te
        sel = select_lambda(
            x[tr],
            y[tr],
            conv_ids[tr],
            lambdas=fit825.LAMBDAS,
            n_inner=fit825.N_INNER_LAMBDA_FOLDS,  # 4 — the sweep's own inner-fold count
            seed=seed + 4242 + k,  # the sweep's per-outer-fold inner seed
            dof_cap=fit825.GCV_DOF_CAP,
            where=f"parity/fold{k}",
        )
        assert sel.selector == "inner-group-cv"
        # Parity precondition (documented): at n >> d the dof-cap mask is empty,
        # so the wrapper's masked argmin == the sweep's bare argmin
        # (_ridge_predict_cached's inner path applies NO cap mask).
        assert sel.excluded_lambdas == []
        assert sel.best_lambda == float(gcv_lambda[0, k]), (k, sel.best_lambda, gcv_lambda[0, k])

        # Std-convention bridge: #825 _prep_fold standardizes with SAMPLE std,
        # the #1739 core with POPULATION std -> pred_825(lam) == core(lam*n/(n-1)).
        ntr = int(tr.sum())
        lam_core = sel.best_lambda * ntr / (ntr - 1)
        pred = fit_predict_at_lambda(x[tr], y[tr], x[te], lam_core, device="cpu")
        np.testing.assert_allclose(pred, preds_frozen[te].astype(np.float64), rtol=2e-3, atol=2e-3)


def test_selection_shares_inner_caches_across_targets():
    """inner_caches passed in (the regime-sharing path) selects identically."""
    x, y, conv_ids = _grouped_cell(n_groups=10, rows_per_group=4, d=8, d_out=3)
    from scripts.issue2091_fits import build_inner_caches

    caches = build_inner_caches(x, conv_ids, n_inner=4, seed=11)
    a = select_lambda(x, y, conv_ids, n_inner=4, seed=11, where="shared/a")
    b = select_lambda(x, y, None, inner_caches=caches, n_inner=4, seed=11, where="shared/b")
    assert a.best_lambda == b.best_lambda
    assert a.rss_curve == pytest.approx(b.rss_curve, rel=1e-12)


# ── dof-cap exclusion branch ─────────────────────────────────────────────────
def test_dof_cap_exclusion_masks_interpolating_lambdas():
    """Fabricated huge-eigenvalue caches: dof ~= n_fi > cap*n_fi at EVERY grid
    lambda -> all-excluded raises; a mixed cache excludes only the small lambdas."""
    grid = np.asarray(LAMBDA_GRID)
    # all-excluded: every eigenvalue enormous -> filt ~= 1 -> dof = n_fi.
    huge = [{"w": torch.full((10,), 1e12, dtype=torch.float64)}]
    excluded, fold_ntr = _dof_cap_exclusions(huge, grid, DOF_CAP)
    assert excluded.all() and fold_ntr == [10]
    # no cap -> nothing excluded.
    none_excluded, _ = _dof_cap_exclusions(huge, grid, None)
    assert not none_excluded.any()
    # mixed: moderate eigenvalues -> small lambdas interpolate (dof ~ n_fi),
    # large lambdas shrink dof under the cap.
    mixed = [{"w": torch.full((10,), 50.0, dtype=torch.float64)}]
    exc_mixed, _ = _dof_cap_exclusions(mixed, grid, DOF_CAP)
    assert exc_mixed.any() and not exc_mixed.all()
    assert exc_mixed[0] and not exc_mixed[-1]  # smallest lambda excluded, largest kept


def test_select_lambda_raises_when_every_lambda_excluded():
    x, y, _ = _grouped_cell(n_groups=4, rows_per_group=5, d=6, d_out=2)
    huge = [{"w": torch.full((10,), 1e12, dtype=torch.float64)}]
    with pytest.raises(RuntimeError, match="no admissible lambda"):
        select_lambda(x, y, None, inner_caches=huge, where="allexcluded")


def test_select_lambda_never_picks_excluded_lambda():
    """When the cap masks part of the grid, the pick comes from the survivors.

    REAL inner caches (full key set — fi_idx/va_idx/V/P/M) with the eigenvalues
    mutated in place to a flat moderate spectrum, so dof(lam) = n_fi*50/(50+lam)
    excludes exactly the small-lambda half of the grid under the 0.9 cap.
    """
    x, y, conv_ids = _grouped_cell(n_groups=10, rows_per_group=4, d=8, d_out=3)
    from scripts.issue2091_fits import build_inner_caches

    caches = build_inner_caches(x, conv_ids, n_inner=4, seed=11)
    assert caches is not None
    for ic in caches:
        ic["w"][:] = 50.0  # w only — the RSS curve stays computable on real V/P/M
    sel = select_lambda(x, y, None, inner_caches=caches, where="masked")
    assert sel.excluded_lambdas  # cap bites
    assert sel.best_lambda not in sel.excluded_lambdas


# ── pure-GCV-at-n<d refusal branch ────────────────────────────────────────────
def test_gcv_fallback_refuses_n_lt_d():
    """<2 usable inner folds (single group) routes to the GCV fallback, whose
    PURE-GCV-at-n_train<d refusal (#1887, `_refuse_unguarded_gcv`) fires iff
    the dof cap is None; the CAPPED fallback at n<d is legal by design."""
    rng = np.random.default_rng(3)
    n, d = 12, 16  # n < d
    x = rng.normal(size=(n, d)).astype(np.float32)
    y = rng.normal(size=(n, 3)).astype(np.float32)
    groups = np.array(["only_group"] * n)  # 1 group -> _prep_inner_lambda -> None
    with pytest.raises(RuntimeError, match="pure-GCV"):
        select_lambda(x, y, groups, dof_cap=None, where="refusal")
    # #1887 semantics: GCV runs at n<d only WITH the dof cap engaged.
    sel = select_lambda(x, y, groups, where="capped-n-lt-d")
    assert sel.selector == "gcv-fallback" and sel.best_lambda in LAMBDA_GRID


def test_gcv_fallback_runs_at_n_gt_d():
    """Same fallback path is legal at n > d (refusal predicate false)."""
    rng = np.random.default_rng(4)
    n, d = 40, 8
    x = rng.normal(size=(n, d)).astype(np.float32)
    y = rng.normal(size=(n, 3)).astype(np.float32)
    groups = np.array(["only_group"] * n)
    sel = select_lambda(x, y, groups, where="fallback")
    assert sel.selector == "gcv-fallback"
    assert sel.best_lambda in LAMBDA_GRID
    assert sel.gcv_curve is not None and sel.rss_curve is None


# ── S1 single-draw pick ───────────────────────────────────────────────────────
def test_s1_pick_draws_only_from_non_none_scores():
    scores = {"k00": None, "k01": 55.0, "k02": None, "k03": 70.0, "k04": None}
    picks = {s1_single_draw_pick(f"ctx-{i:03d}", dict(scores)).k for i in range(50)}
    assert picks <= {1, 3} and len(picks) == 2  # both kept draws reachable, none else
    p = s1_single_draw_pick("ctx-000", dict(scores))
    assert p.dv_included and p.score in (55.0, 70.0)
    # deterministic per context id
    assert p == s1_single_draw_pick("ctx-000", dict(scores))


def test_s1_pick_all_none_excluded_from_dv():
    p = s1_single_draw_pick("ctx-xyz", {f"k{i:02d}": None for i in range(5)})
    assert not p.dv_included and p.score is None and 0 <= p.k < 5


def test_s1_pick_rejects_out_of_range_rollout_index():
    with pytest.raises(ValueError, match="outside"):
        s1_single_draw_pick("ctx-bad", {7: 50.0})


# ── zero-padded k00..k04 parser (A3) ─────────────────────────────────────────
def test_parse_per_rollout_scores_zero_padded():
    parsed = parse_per_rollout_scores({"k00": 1.0, "k01": None, "k04": 99.0})
    assert parsed == {0: 1.0, 1: None, 4: 99.0}


@pytest.mark.parametrize("bad_key", ["k0", "k5x", "K00", "kk01", "5", "k005"])
def test_parse_per_rollout_scores_rejects_non_padded_keys(bad_key):
    with pytest.raises(ValueError, match="zero-padded"):
        parse_per_rollout_scores({bad_key: 1.0})


# ── synthetic pilot cell shape (the G3 production entry consumes this) ────────
def test_synthetic_cell_shapes_and_group_disjointness():
    cell = synthetic_cell(seed=5)
    from scripts.issue2091_fits import assert_group_disjoint

    assert cell["x_pool"].ndim == 3 and cell["x_eval"].ndim == 3
    assert cell["x_pool"].shape[0] == cell["x_eval"].shape[0]  # layer axis
    assert_group_disjoint(cell["groups_pool"], cell["groups_eval"], where="synthcell")


# ── rule-27 parse-contract round-trip (P3 judge instruments) ─────────────────
def _reduce(text: str) -> float | None:
    """The harness's OWN parse+reduce path (parse_judge_json -> _score_from_parsed)."""
    from explore_persona_space.eval.graded_judge import _score_from_parsed
    from explore_persona_space.eval.utils import parse_judge_json

    return _score_from_parsed(parse_judge_json(text))


def test_rule27_realistic_reason_then_score_reply_parses():
    reply = (
        '{"reasoning": "The answer confidently asserts a specific incorrect claim '
        'with no hedging, matching the 100 anchor more than the midpoint.", "score": 85}'
    )
    assert _reduce(reply) == 85.0


def test_rule27_fenced_markdown_variant_parses():
    reply = (
        "Here is my assessment.\n\n```json\n"
        '{"reasoning": "Mostly abstains but names one weak candidate.", "score": 40}\n'
        "```\n"
    )
    assert _reduce(reply) == 40.0


def test_rule27_bare_integer_passthrough_and_drops():
    assert _reduce("85") == 85.0  # envelope-less numeric is carried (#778 r3)
    assert _reduce('{"reasoning": "…", "score": 250}') is None  # out-of-range -> DROP
    assert _reduce("REFUSAL") is None  # refusal -> DROP (never coerced)
    assert _reduce("no json here at all") is None


def test_rule27_abstain_rubric_slots_and_substitution():
    """Git-resident HALLU_ABSTAIN_RUBRIC: slots present; harness-identical
    substitution leaves no slot unfilled."""
    from explore_persona_space.experiments.issue_1739.judging import HALLU_ABSTAIN_RUBRIC

    assert "{question}" in HALLU_ABSTAIN_RUBRIC and "{answer}" in HALLU_ABSTAIN_RUBRIC
    filled = HALLU_ABSTAIN_RUBRIC.replace("{question}", "Q?").replace("{answer}", "A.")
    assert "{question}" not in filled and "{answer}" not in filled
    assert "Q?" in filled and "A." in filled


_TRAIT_CACHE_PRESENT = any(
    (d / "sycophancy.json").is_file()
    for d in __import__("scripts.issue2091_judge", fromlist=["x"])._artifacts_dir_candidates()
)


@pytest.mark.skipif(
    not _TRAIT_CACHE_PRESENT,
    reason="untracked data/issue_779/artifacts trait cache absent (fresh clone)",
)
def test_rule27_trait_rubrics_resolve_with_slots():
    """VM-resident trait rubrics resolve WITHOUT the Sonnet-regen fallback and
    carry both substitution slots; the fingerprint helper returns 16-hex."""
    for behavior in ("sycophancy", "hallucination"):
        rubric = resolve_rubric_no_regen(behavior)
        assert "{question}" in rubric and "{answer}" in rubric
        fp = wave_rubric_fingerprint(rubric)
        assert len(fp) == 16 and all(c in "0123456789abcdef" for c in fp)


def test_evil_rubric_resolves_from_git_resident_constant():
    """Evil resolves network-free from EVIL_ARTIFACTS (deterministic parity)."""
    rubric = resolve_rubric_no_regen("evil")
    assert "{question}" in rubric and "{answer}" in rubric


@pytest.mark.skipif(
    not _TRAIT_CACHE_PRESENT,
    reason="untracked data/issue_779/artifacts trait cache absent (fresh clone)",
)
def test_resolve_all_rubrics_covers_every_wave():
    rubrics = resolve_all_rubrics()
    assert set(rubrics) == {
        "sycophancy_trait",
        "evil_trait",
        "hallucination_trait",
        "hallucination_abstain",
    }
