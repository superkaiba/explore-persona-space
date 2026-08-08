"""Shared paired-bootstrap CI helpers for #1739 evil-OOD-spread arms.

Task #1739 evil-ood-spread-round unit 5a (plan v16 §6).

Extracted from ``scripts/issue1739_score_new_rungs.py`` so ``rescore_ood.py``
(existing rungs) and ``score_new_rungs.py`` (new rungs) share ONE implementation
of the paired-CI + AUROC-CI + positive-count schema. A shared helper closes the
r1 gap where ``rescore_ood.py`` output rows carried only marginal ``ci_rho`` and
missed the paired-difference reads plan v16 §6 mandates.

Public API
----------
- ``paired_bootstrap_rho_delta(preds_arm, preds_comparator, dv, n_boot=500, seed=42)``
  Paired Spearman-rho delta CI, SHARED resample indices per draw. Returns
  ``(rho_arm, rho_comp, delta_ci_lo, delta_ci_hi)``.
- ``paired_bootstrap_auroc_delta(preds_arm, preds_comparator, dv_binary, n_boot=500, seed=42)``
  Paired AUROC delta CI, SHARED resample indices per draw. Returns
  ``(auroc_arm, auroc_comp, delta_ci_lo, delta_ci_hi)``.
- ``marginal_bootstrap_auroc(preds, dv_binary, n_boot=500, seed=42)``
  Marginal AUROC + 95% CI. Returns ``(auroc, ci_lo, ci_hi)``.
- ``positive_count(dv, threshold=AUROC_POS_THR)``
  Count of finite ``dv`` values ``>= threshold``.
- ``selection_inherited_paired_max(preds_by_arm, comparator_preds, dv, arm_family,
      n_boot=500, seed=42)``
  Selection-inherited (per-draw argmax) best-of-family paired-rho-delta CI.

Grounded on
-----------
- Plan v16 §6 "Paired-contrast CIs" (paired-difference reads).
- ``experiments.issue_1739.arms.bootstrap_rhos`` (shared idx across arms =>
  paired differences); we mirror the shared-idx contract for AUROC.
- ``.claude/rules/selection-symmetric-nulls.md`` (per-draw selection).
- ``.claude/rules/llm-judging.md`` rule 24 (drop-never-coerce; DV / preds
  arriving here are already drop-cleaned by the caller).

Numeric contract: all inputs are cast to ``float64``; a keep mask drops any
``(dv, preds)`` cell that is non-finite; CIs are 95% empirical quantiles
(alpha/2, 1-alpha/2). A draw whose bootstrap resample contains zero positives
or zero negatives yields NaN for that draw's AUROC (mirrors
``arms.auroc_rows``); CI quantiles are ``nanquantile`` to survive them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # numpy imported lazily so import is cheap
    import numpy as np  # noqa: F401


# 50/100 threshold for binarising a graded 0-100 judge DV, mirroring
# ``scripts/issue1739_rescore_ood.py::AUROC_POS_THR``.
AUROC_POS_THR: float = 50.0


# ---------------------------------------------------------------------------
# small internals
# ---------------------------------------------------------------------------


def _finite_pair_mask(a, b):
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.isfinite(a) & np.isfinite(b), a, b


def _finite_pair_bool_mask(a, dv_bin):
    """Finite mask for a float score vs a boolean-label pair."""
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    dv_bin = np.asarray(dv_bin, dtype=bool)
    return np.isfinite(a), a, dv_bin


def _quantile_ci(draws, alpha: float = 0.05) -> tuple[float, float]:
    """2.5/97.5 percentile CI (NaN-safe)."""
    import numpy as np

    draws = np.asarray(draws, dtype=np.float64)
    if draws.size == 0:
        return (float("nan"), float("nan"))
    lo = float(np.nanquantile(draws, alpha / 2.0))
    hi = float(np.nanquantile(draws, 1.0 - alpha / 2.0))
    return (lo, hi)


# ---------------------------------------------------------------------------
# positive_count
# ---------------------------------------------------------------------------


def positive_count(dv, threshold: float = AUROC_POS_THR) -> int:
    """Count of finite ``dv`` values ``>= threshold`` (default 50.0).

    Mirrors the ``labels_finite = np.isfinite(dv) & (dv >= AUROC_POS_THR)``
    convention used in ``issue1739_score_new_rungs.py::_score_rung``.
    """
    import numpy as np

    dv_arr = np.asarray(dv, dtype=np.float64)
    finite = np.isfinite(dv_arr)
    return int(((dv_arr >= threshold) & finite).sum())


# ---------------------------------------------------------------------------
# paired-bootstrap: Spearman rho delta
# ---------------------------------------------------------------------------


def paired_bootstrap_rho_delta(
    preds_arm,
    preds_comparator,
    dv,
    n_boot: int = 500,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """Paired-bootstrap Spearman-rho delta CI.

    The SAME resample indices are applied to both arms per draw
    (paired-difference-valid): correlated noise from the shared DV cancels.

    Parameters
    ----------
    preds_arm, preds_comparator : array-like of shape (n,)
        Arm and comparator per-context predictor scores.
    dv : array-like of shape (n,)
        Dependent variable (continuous or graded).
    n_boot : int
        Number of bootstrap draws (default 500 — plan v16 §6).
    seed : int
        Seed for the shared-idx RNG.

    Returns
    -------
    (rho_arm, rho_comparator, delta_ci_lo, delta_ci_hi)
        Point estimates of the two marginal rhos, plus the 95% empirical
        quantile CI on the paired difference ``rho_arm - rho_comparator``.
    """
    import numpy as np
    from scipy.stats import rankdata

    # keep only fully-finite triples
    mask_ab, arm, comp = _finite_pair_mask(preds_arm, preds_comparator)
    mask_ad, _, dv_arr = _finite_pair_mask(preds_arm, dv)
    keep = mask_ab & mask_ad
    arm = arm[keep]
    comp = comp[keep]
    dv_arr = dv_arr[keep]

    n = int(arm.shape[0])
    if n < 3:
        return (float("nan"), float("nan"), float("nan"), float("nan"))

    # point rhos via ranks
    dv_ranks = rankdata(dv_arr)
    rho_arm = float(np.corrcoef(rankdata(arm), dv_ranks)[0, 1])
    rho_comp = float(np.corrcoef(rankdata(comp), dv_ranks)[0, 1])

    rng = np.random.default_rng([1739, 6, int(seed), int(n)])
    idx = rng.integers(0, n, size=(n_boot, n))  # shared per draw

    rho_delta = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        i = idx[b]
        arm_b = arm[i]
        comp_b = comp[i]
        dv_b = dv_arr[i]
        # ranks recomputed on the resample (bit-safe, sd-stable)
        dv_r = rankdata(dv_b)
        with np.errstate(invalid="ignore"):
            r_arm = np.corrcoef(rankdata(arm_b), dv_r)[0, 1]
            r_comp = np.corrcoef(rankdata(comp_b), dv_r)[0, 1]
        rho_delta[b] = r_arm - r_comp

    lo, hi = _quantile_ci(rho_delta)
    return (rho_arm, rho_comp, lo, hi)


# ---------------------------------------------------------------------------
# paired-bootstrap: AUROC delta
# ---------------------------------------------------------------------------


def _auroc(scores, labels_bool) -> float:
    """Marginal AUROC (rank-formula; NaN on degenerate label vector).

    Mirrors ``experiments.issue_1739.arms.auroc_rows`` for a single row.
    """
    import numpy as np
    from scipy.stats import rankdata

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels_bool, dtype=bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def paired_bootstrap_auroc_delta(
    preds_arm,
    preds_comparator,
    dv_binary,
    n_boot: int = 500,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """Paired-bootstrap AUROC delta CI on a binary DV.

    ``dv_binary`` is a boolean label vector (typically ``dv >= AUROC_POS_THR``).
    Shared resample indices per draw: identical construction to
    :func:`paired_bootstrap_rho_delta`, AUROC substituted for Spearman rho.

    Returns
    -------
    (auroc_arm, auroc_comparator, delta_ci_lo, delta_ci_hi)
    """
    import numpy as np

    # keep only fully-finite triples (label vector is boolean, always finite)
    mask_ab, arm, comp = _finite_pair_mask(preds_arm, preds_comparator)
    labels = np.asarray(dv_binary, dtype=bool)
    keep = mask_ab
    arm = arm[keep]
    comp = comp[keep]
    labels = labels[keep]

    n = int(arm.shape[0])
    if n < 3 or int(labels.sum()) == 0 or int((~labels).sum()) == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))

    auroc_arm = _auroc(arm, labels)
    auroc_comp = _auroc(comp, labels)

    rng = np.random.default_rng([1739, 6, int(seed), int(n)])
    idx = rng.integers(0, n, size=(n_boot, n))

    auroc_delta = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        i = idx[b]
        arm_b = arm[i]
        comp_b = comp[i]
        lbl_b = labels[i]
        auroc_delta[b] = _auroc(arm_b, lbl_b) - _auroc(comp_b, lbl_b)

    lo, hi = _quantile_ci(auroc_delta)
    return (auroc_arm, auroc_comp, lo, hi)


# ---------------------------------------------------------------------------
# marginal AUROC + CI (per-arm bootstrap)
# ---------------------------------------------------------------------------


def marginal_bootstrap_auroc(
    preds,
    dv_binary,
    n_boot: int = 500,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Marginal AUROC + 95% CI for ONE arm.

    Uses a standard nonparametric bootstrap (resample rows, recompute AUROC).
    Stratification is unnecessary because ``_auroc`` already returns NaN for
    a degenerate resample and ``_quantile_ci`` is NaN-safe.

    Returns
    -------
    (auroc, ci_lo, ci_hi)
    """
    import numpy as np

    mask, scores, _ = _finite_pair_bool_mask(preds, dv_binary)
    labels = np.asarray(dv_binary, dtype=bool)
    keep = mask
    scores = scores[keep]
    labels = labels[keep]

    n = int(scores.shape[0])
    if n < 3 or int(labels.sum()) == 0 or int((~labels).sum()) == 0:
        return (float("nan"), float("nan"), float("nan"))

    auroc = _auroc(scores, labels)

    rng = np.random.default_rng([1739, 6, int(seed), int(n)])
    idx = rng.integers(0, n, size=(n_boot, n))

    auroc_draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        i = idx[b]
        auroc_draws[b] = _auroc(scores[i], labels[i])

    lo, hi = _quantile_ci(auroc_draws)
    return (auroc, lo, hi)


# ---------------------------------------------------------------------------
# selection-inherited best-of-family paired delta
# ---------------------------------------------------------------------------


def selection_inherited_paired_max(
    preds_by_arm: dict,
    comparator_preds,
    dv,
    arm_family: list,
    n_boot: int = 500,
    seed: int = 42,
) -> dict:
    """Selection-inherited best-of-family paired-rho-delta CI (per plan v16 §6).

    For each bootstrap draw, take the max rho over ``arm_family`` predictors
    against ``dv``, subtract the comparator's rho on the SAME resample, and
    quantile the selected-max distribution — the selection rides per draw
    (``.claude/rules/selection-symmetric-nulls.md``), so the CI width is
    honestly selection-inherited.

    Parameters
    ----------
    preds_by_arm : dict[str, array-like of shape (n,)]
        Per-arm predictor scores keyed by arm slug. Missing family arms are
        silently dropped from the max (never raise — an arm might not have
        been fit on this rung).
    comparator_preds : array-like of shape (n,)
        Comparator arm scores (typically ``arm16_surface_feat``).
    dv : array-like of shape (n,)
        Dependent variable.
    arm_family : list[str]
        Family arm slugs (e.g. the map family from plan v16 §3).
    n_boot, seed : as above.

    Returns
    -------
    dict with fields:
        best_arm_per_draw : list[str]
            Which arm won the max on each draw (length ``n_boot``); useful
            for a per-draw selection audit.
        delta_ci_lo, delta_ci_hi : float
            95% CI on the selected-max minus comparator rho.
        family_members : list[str]
            Family arms that had non-empty ``preds_by_arm`` entries.
    """
    import numpy as np
    from scipy.stats import rankdata

    members = [a for a in arm_family if a in preds_by_arm]
    if not members:
        return {
            "best_arm_per_draw": [],
            "delta_ci_lo": float("nan"),
            "delta_ci_hi": float("nan"),
            "family_members": [],
        }

    # stack family predictions as (F, n)
    family_stack = np.stack(
        [np.asarray(preds_by_arm[a], dtype=np.float64) for a in members], axis=0
    )
    comp = np.asarray(comparator_preds, dtype=np.float64)
    dv_arr = np.asarray(dv, dtype=np.float64)

    # keep rows where ALL family + comparator + dv are finite (paired mask)
    finite_family = np.all(np.isfinite(family_stack), axis=0)
    keep = finite_family & np.isfinite(comp) & np.isfinite(dv_arr)
    family_stack = family_stack[:, keep]
    comp = comp[keep]
    dv_arr = dv_arr[keep]

    n = int(dv_arr.shape[0])
    if n < 3:
        return {
            "best_arm_per_draw": [],
            "delta_ci_lo": float("nan"),
            "delta_ci_hi": float("nan"),
            "family_members": members,
        }

    rng = np.random.default_rng([1739, 6, int(seed), int(n)])
    idx = rng.integers(0, n, size=(n_boot, n))

    delta = np.empty(n_boot, dtype=np.float64)
    best_arm = [""] * n_boot
    for b in range(n_boot):
        i = idx[b]
        dv_r = rankdata(dv_arr[i])
        comp_r = rankdata(comp[i])
        rho_comp_b = float(np.corrcoef(comp_r, dv_r)[0, 1])
        rhos = np.empty(len(members), dtype=np.float64)
        for k, _ in enumerate(members):
            arm_r = rankdata(family_stack[k][i])
            with np.errstate(invalid="ignore"):
                rhos[k] = np.corrcoef(arm_r, dv_r)[0, 1]
        best_k = int(np.nanargmax(rhos))
        best_arm[b] = members[best_k]
        delta[b] = float(rhos[best_k]) - rho_comp_b

    lo, hi = _quantile_ci(delta)
    return {
        "best_arm_per_draw": best_arm,
        "delta_ci_lo": lo,
        "delta_ci_hi": hi,
        "family_members": members,
    }
