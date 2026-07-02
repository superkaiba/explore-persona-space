#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (Δ, ΔM, r_B, r̂, →, ρ, M⁺, M0, ※, ×, Ŵ, ‖·‖) in scientific
# docstrings + log messages.
"""Issue #667 — MARKER map-change (M0 vs M⁺) at all layers 1-27, floor-normalized.

The #722 map-change headline (``Δ_med = median_c |Δ(c)·r̂_B|`` / floor) was NOT
computed for the ``marker`` behavior because marker has NO ``r_B`` (it is a
programmatic token, not a contrastive-activation direction — ``r_b.pt`` /
``r_b_fact.pt`` carry no marker column, so ``issue722_fit_M._r_hat_for`` KeyErrors
for marker and the #667 fast/per-cell map-change paths list only em/sycophancy/fact
in ``MAP_CHANGE_BEHAVIORS``). This read fills that gap with TWO estimators, both
built on the EXACT #722 ridge + refit-floor machinery, no r_B required:

**Read 1 — UNPROJECTED ``‖ΔM(c)‖ / floor`` (behavior-agnostic).** The total
map-output change at fixed base input, floor-normalized. Numerator =
``median_c ‖delta_full(c)‖`` where ``delta_full = M⁺(c) − M0(c)`` is the SAME
(n_grid, 3584) refit-difference #722 computes at line ``delta_full = delta @
pca_basis`` in ``fit_cell`` (before it projects on r_B). The floor is the norm of
the SAME three refit-difference nulls (``floor_M0_refit`` / ``floor_Mplus_refit``
/ ``floor_shifted``), built through the IDENTICAL bootstrap+random-init refit
harness (``make_refit_pair``'s loop) but taking ``‖·‖`` of each refit-pair delta
instead of the r_B projection. This answers "does the marker map change at all"
without any behavior direction.

**Read 2 — W_U[※]-projected ``|ΔM(c)·Ŵ_U[※]| / floor`` (marker-specific).**
Substitutes the marker's read-out direction for r_B: ``Ŵ_U[※]`` = the
unit-normalized unembedding (lm_head) row for token id 83399, a 3584-dim
residual-space direction. Used EXACTLY where ``fit_cell`` uses ``r_hat`` — the
numerator ``median_c |delta_full·Ŵ_U[※]|`` is literally ``make_refit_pair`` /
``clustered_bootstrap_scalar`` with ``r_hat = Ŵ_U[※]``, so the floor here is the
verbatim #722 ``make_refit_pair`` (no local twin needed).

**CRITICAL VALIDITY CHECK (per layer, reported prominently).** The fitted map's
OUTPUT lives in the top-64 ``v0``-PC subspace (``pca_basis``). ``W_U[※]`` may lie
mostly OUTSIDE that subspace, which would make read 2 artifactually ~0 (a map that
cannot express any component of ``W_U[※]`` produces ``delta_full·Ŵ_U[※] ≈ 0`` by
construction, NOT by absence of change). So per layer we report
``frac_in_subspace = ‖pca_basis @ (pca_basisᵀ @ Ŵ_U[※])‖ / ‖Ŵ_U[※]‖`` — the
fraction of the marker direction captured by the 64 v0-PCs. When it is tiny
(< 0.1) read 2 is FLAGGED uninformative and read 1 is the load-bearing estimator.

Optional chain-ρ (skipped if E is unavailable): held-out Spearman of
``Ŵ_U[※]ᵀ M̂(c)`` (r_B := Ŵ_U[※]) vs the #537 marker leakage E
(``log P(※)`` per cell, from ``eval_results/issue_537/G_tensor/G_meta.json``),
under M0 vs M⁺, on the SAME ridge-LOCO path #722 uses.

This is EXPLORATORY, user-directed inline analysis. It REUSES the #722/#658 fit
machinery WITHOUT modifying it — ``issue722_fit_M.fit_cell`` (and thus the
em/fact/syco headline) is byte-for-byte untouched; every read here is computed
from the same primitives (``_ridge_fit_predict``, ``_refit_ridge_fn``,
``m0_at_cplus_ridge_full``, ``_pca_basis_v0``, ``make_refit_pair``,
``clustered_bootstrap_scalar``, ``clustered_bootstrap_spearman``) called directly.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

# DOTENV_LINT_EXEMPT: exploratory analysis script (no HF upload); shell exports
# cover the pod. Mirrors the sibling issue722_fit_M.py load-at-entry.
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
load_dotenv(str(PROJECT_ROOT / ".env"))

import issue658_fit_predictors as fit658  # noqa: E402
import issue722_fit_M as fitM  # noqa: E402
import issue722_load_activations as loadact  # noqa: E402
from issue722_bootstrap import (  # noqa: E402
    _resample_family_idx,
    clustered_bootstrap_scalar,
    floor_sd,
    make_refit_pair,
)

from explore_persona_space.analysis.issue667.gate_chain import (  # noqa: E402
    clustered_bootstrap_spearman,
)

logger = logging.getLogger("issue667.marker_mapchange")

HIDDEN = 3584
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MARKER_TEXT = " ※"  # leading space; single token id 83399 on Qwen-2.5-7B
MARKER_ID = 83399
DEFAULT_LAYERS = tuple(range(1, 28))  # 1..27 (brief); layer 0 = embeddings, excluded
N_REFIT_PAIRS = 100
N_SCALAR_BOOT = 1000
TARGET_DIM = 64  # top-v0 PCs, the shared #658 A35_MLP_TARGET_DIM output target
SUPPORT_SHIFT_PCTL = 90
FRAC_IN_SUBSPACE_UNINFORMATIVE = 0.1  # read-2 flagged uninformative below this


def load_wu_marker_direction() -> np.ndarray:
    """Unit ``Ŵ_U[※]``: the lm_head row for token 83399, unit-normalized (3584,).

    Loads ONLY the tokenizer (for the in-process marker-id assert, per
    ``.claude/rules/marker-leakage-measurement.md``) + the lm_head weight of the
    base model. Asserts ``tokenizer.encode(" ※", add_special_tokens=False) ==
    [83399]`` so a marker-token drift fails loud BEFORE any fit, then returns the
    unit-normalized unembedding row as a residual-space direction (the marker's
    read-out direction, the r_B substitute for reads 2 + chain-ρ).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    ids = tok.encode(MARKER_TEXT, add_special_tokens=False)
    assert ids == [MARKER_ID], (
        f"marker token drift: encode({MARKER_TEXT!r}) == {ids}, expected [{MARKER_ID}] "
        "(bare ※ id 63680 is the WRONG token — see marker-leakage-measurement.md)"
    )
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
    W_U = model.get_output_embeddings().weight  # (vocab, hidden), fp32
    assert W_U.shape[1] == HIDDEN, f"lm_head hidden {W_U.shape} != {HIDDEN}"
    row = W_U[MARKER_ID].detach().to(torch.float64).cpu().numpy()  # (3584,)
    norm = float(np.linalg.norm(row))
    if norm < 1e-9:
        raise RuntimeError(f"degenerate W_U[{MARKER_ID}] (norm {norm:.2e})")
    logger.info(
        "[phase=map_change] loaded Ŵ_U[※] (token %d) from %s lm_head; ‖row‖=%.4g",
        MARKER_ID,
        BASE_MODEL,
        norm,
    )
    # free the model weights promptly; only the (3584,) direction is retained.
    del model, W_U
    return row / norm


def _frac_in_subspace(pca_basis: np.ndarray, direction: np.ndarray) -> float:
    """‖pca_basisᵀ pca_basis d‖ / ‖d‖ — fraction of ``direction`` in the 64-PC subspace.

    ``pca_basis`` is (k, 3584) with orthonormal rows (right-singular vectors of the
    mean-centered v0 output). The projector onto that subspace is
    ``P = pca_basisᵀ pca_basis`` and the captured fraction is ``‖P d‖ / ‖d‖``.
    Computed as ``‖pca_basis @ (pca_basis @ d)‖`` — the coefficients
    ``coef = pca_basis @ d`` (k,) then back to 3584 via ``pca_basisᵀ @ coef``.
    Returns 0.0 for an empty basis or a zero direction (fully outside / undefined).
    """
    if pca_basis.shape[0] == 0:
        return 0.0
    d = np.asarray(direction, dtype=np.float64)
    dn = float(np.linalg.norm(d))
    if dn < 1e-12:
        return 0.0
    coef = pca_basis @ d  # (k,)
    proj = pca_basis.T @ coef  # (3584,) back-projection into the subspace
    return float(np.linalg.norm(proj) / dn)


def _pca_basis_v0_fast(V0: np.ndarray, dim: int) -> np.ndarray:
    """Top-`dim` v0-PC basis via a DUAL eigh on the (n,n) Gram — the #667 fast PCA.

    A verbatim copy of ``issue667_alllayer_analysis._pca_basis_v0_fast`` (kept
    local so this script does not import the heavy all-layer driver): returns the
    SAME top-`dim` right-singular subspace of the mean-centered ``V0`` (up to
    per-component sign, which CANCELS in the ridge projection ``Y @ pca.T`` then
    back ``@ pca``) as ``issue722_fit_M._pca_basis_v0``'s ``np.linalg.svd`` path,
    but computes it as an eigh on the small (n, n) Gram (n≪3584) — the
    vectorize-many-cell-fits win at n≪H, ~4× faster than the (n, 3584) SVD the
    ridge floor otherwise recomputes ~600× per layer. ``Vₖ = Uₖᵀ Vc / Sₖ`` where
    ``Vc Vcᵀ = U diag(S²) Uᵀ``. Preserves the gesdd→gesvd robustness contract
    (np.linalg.eigh fast path → scipy.linalg.eigh syevr fallback), keeps only
    numerically-positive eigenvalues (the SVD's rank truncation). Verified in
    #667 to reproduce np.svd's subspace to min|cos|=1.0 (singular values match to
    5.7e-14), and the marker read-2 numerator+floor were shown bit-identical to
    ``fit_cell`` under the np.svd PCA (rtol=1e-9), so the end-to-end |Δ·Ŵ_U| and
    ‖Δ‖ reads are unchanged by this substitution.
    """
    Vc = V0 - V0.mean(axis=0, keepdims=True)
    G = Vc @ Vc.T  # (n, n) dual Gram — SPD, n≪3584 is the win
    try:
        w, U = np.linalg.eigh(G)  # ascending eigenvalues; LAPACK syevd
    except np.linalg.LinAlgError:
        from scipy.linalg import eigh as _scipy_eigh

        logger.warning(
            "[phase=map_change] np.linalg.eigh (syevd) did not converge on a %s Gram "
            "(near-singular resample); retrying with scipy syevr",
            G.shape,
        )
        w, U = _scipy_eigh(G)
    order = np.argsort(w)[::-1]  # descending
    if order.size:
        pos = w[order] > 1e-12 * float(max(w[order][0], 1.0))
        order = order[pos]
    k = min(dim, order.size)
    order = order[:k]
    if k == 0:
        return np.zeros((0, V0.shape[1]), dtype=np.float64)
    S = np.sqrt(np.clip(w[order], 0.0, None))  # (k,)
    return (U[:, order].T @ Vc) / S[:, None]  # (k, 3584)


class _fast_pca_injected:
    """Swap ``issue722_fit_M._pca_basis_v0`` for the dual-eigh fast PCA in a `with`.

    ``fit_marker_layer`` resolves the headline PCA through ``fitM._pca_basis_v0``,
    and the refit floors resolve it through the reused ``fitM._refit_ridge_fn`` /
    ``fitM.m0_at_cplus_ridge_full`` closures (which ALSO call the module attribute
    ``fitM._pca_basis_v0``). Swapping that one module attribute redirects EVERY
    PCA in the ridge + floor path to the fast subspace WITHOUT editing the reused
    module — the same runtime-override pattern ``issue667_alllayer_analysis`` uses.
    Restored in ``__exit__`` so nothing outside the `with` sees the swap.
    """

    def __enter__(self):
        self._saved = fitM._pca_basis_v0
        fitM._pca_basis_v0 = _pca_basis_v0_fast
        return self

    def __exit__(self, *exc):
        fitM._pca_basis_v0 = self._saved
        return False


def _refit_ridge_fn(grid: np.ndarray):
    """Reuse #722's exact per-bootstrap ridge fit_fn (recomputes its own top-64 v0 PCs).

    Thin wrapper around ``issue722_fit_M._refit_ridge_fn`` so the norm-floor twin
    below drives the SAME refit as ``make_refit_pair`` does for read 2 — the two
    floors differ ONLY in how the refit-pair delta is reduced (``‖·‖`` vs
    ``|··Ŵ_U|``), never in the fit.
    """
    return fitM._refit_ridge_fn(grid)


def _refit_pair_norm(
    X: np.ndarray,
    Y: np.ndarray,
    fit_fn,
    eval_grid: np.ndarray,
    families,
    *,
    n_pairs: int = 100,
    seed: int = 0,
    skip_counter: dict | None = None,
) -> np.ndarray:
    """Read-1 floor: ``make_refit_pair`` with the r_B projection replaced by ``‖·‖``.

    A faithful copy of ``issue722_bootstrap.make_refit_pair``'s loop — same
    family-clustered double resample, same independent random-init refits, same
    gesdd→gesvd-guarded LinAlgError skip contract — with EXACTLY ONE line changed:
    the per-pair statistic is ``median_c ‖pred_a(grid) − pred_b(grid)‖`` (the total
    map-output change) instead of ``median_c |(pred_a − pred_b)·r̂_B|``. This is the
    "#722 floor machinery with the projection replaced by a vector norm" the read-1
    numerator needs. Returns the (surviving-pairs,) array; the caller takes its
    95th percentile as the read-1 floor. Read 2's floor uses the UNMODIFIED
    ``make_refit_pair`` (r_hat = Ŵ_U[※]), so only this norm variant is local.
    """
    n = X.shape[0]
    fams = np.asarray(list(families), dtype=object)
    assert fams.shape == (n,), (fams.shape, n)
    uniq = sorted({str(f) for f in fams})
    clustered = len(uniq) >= 2
    fam_to_idx = {f: np.where(fams.astype(str) == f)[0] for f in uniq}
    rng = np.random.default_rng(seed)
    survivors: list[float] = []
    n_skipped = 0
    for p in range(n_pairs):
        if clustered:
            idx_a = _resample_family_idx(fam_to_idx, uniq, rng)
            idx_b = _resample_family_idx(fam_to_idx, uniq, rng)
        else:
            idx_a = rng.integers(0, n, size=n)
            idx_b = rng.integers(0, n, size=n)
        rng_a = np.random.default_rng(rng.integers(0, 2**31 - 1))
        rng_b = np.random.default_rng(rng.integers(0, 2**31 - 1))
        try:
            pred_a = fit_fn(X[idx_a], Y[idx_a], rng_a)  # (n_grid, 3584)
            pred_b = fit_fn(X[idx_b], Y[idx_b], rng_b)
        except np.linalg.LinAlgError as e:
            n_skipped += 1
            logger.warning(
                "[phase=map_change] _refit_pair_norm: skipping pair %d/%d after "
                "LinAlgError (%s); %d skipped so far",
                p + 1,
                n_pairs,
                e,
                n_skipped,
            )
            continue
        delta = pred_a - pred_b  # (n_grid, 3584)
        norm = np.linalg.norm(delta, axis=1)  # (n_grid,) — ‖·‖ replaces |··r̂_B|
        survivors.append(float(np.median(norm)))
    if skip_counter is not None:
        skip_counter["n_attempted"] = n_pairs
        skip_counter["n_skipped"] = n_skipped
    if not survivors:
        raise np.linalg.LinAlgError(
            f"_refit_pair_norm: all {n_pairs} refit pairs failed with LinAlgError "
            "(resample geometry fully degenerate — cannot build a norm floor)"
        )
    return np.asarray(survivors, dtype=float)


def _load_marker_E(cell_keys: list[str]) -> np.ndarray:
    """#537 marker leakage E = ``g`` per cell (log P(※)), aligned to cell_keys.

    Reuses ``issue722_fit_M._load_E`` (reads ``eval_results/issue_537/
    G_tensor/G_meta.json`` per_cell ``g``). ``cell_keys`` are
    ``marker/{source}__{target}`` strings which key directly into G_meta's
    per_cell dict (the marker prefix). NaN where a cell has no E.
    """
    return fitM._load_E("marker", cell_keys)


def _chain_rho_one(pred64: np.ndarray, pca_basis: np.ndarray, r_hat: np.ndarray, E: np.ndarray):
    """Spearman(r_Bᵀ M̂(c), E) with r_B := Ŵ_U[※] — reuses #722's exact chain read."""
    return fitM._chain_rho_one(pred64, pca_basis, r_hat, E)


def fit_marker_layer(layer: int, cells: list, wu_marker: np.ndarray, *, with_chain: bool) -> dict:
    """Both floor-normalized reads (+ optional chain-ρ) for MARKER at one layer.

    Mirrors ``issue722_fit_M.fit_cell``'s ridge headline arithmetic exactly through
    the ``delta_full = delta @ pca_basis`` line, then branches into the two marker
    reads. The refit floors reuse the shared harness: read 2 = verbatim
    ``make_refit_pair`` (r_hat = Ŵ_U[※]); read 1 = ``_refit_pair_norm`` (‖·‖). Both
    pass ``families`` so the floor resample is family-clustered — the SAME unit as
    the ``clustered_bootstrap_scalar`` numerator CI.
    """
    stacks = loadact.stack_for_fit(cells)
    C0, Cplus = stacks["C0"], stacks["Cplus"]
    V0, Vplus = stacks["V0"], stacks["Vplus"]
    families = stacks["families"]
    cell_keys = stacks["cell_keys"]
    n = C0.shape[0]
    assert n >= 4, f"marker L{layer}: only {n} cells (<4) — cannot fit"

    pca_basis = fitM._pca_basis_v0(V0, TARGET_DIM)  # (k<=64, 3584), orthonormal rows
    V0_64 = V0 @ pca_basis.T
    Vplus_64 = Vplus @ pca_basis.T
    grid = loadact.common_c_grid(stacks)  # base c0 grid (n, 3584)

    # ---- The SHARED #722 delta_full (before any behavior projection) ----
    m0_grid = fitM._ridge_fit_predict(C0, V0_64, grid)  # (n_grid, 64)
    mplus_grid = fitM._ridge_fit_predict(Cplus, Vplus_64, grid)  # (n_grid, 64)
    delta = mplus_grid - m0_grid  # (n_grid, 64)
    delta_full = delta @ pca_basis  # (n_grid, 3584) — identical to fit_cell's delta_full

    # ---- Read 1 numerator: ‖ΔM(c)‖ (behavior-agnostic) ----
    unproj = np.linalg.norm(delta_full, axis=1)  # (n_grid,)
    unproj_ci = clustered_bootstrap_scalar(
        unproj, families, statistic="median", n_resamples=N_SCALAR_BOOT
    )
    unproj_med = unproj_ci["point"]

    # ---- Read 2 numerator: |ΔM(c)·Ŵ_U[※]| (marker-specific) ----
    wu_proj = np.abs(delta_full @ wu_marker)  # (n_grid,)
    wu_proj_ci = clustered_bootstrap_scalar(
        wu_proj, families, statistic="median", n_resamples=N_SCALAR_BOOT
    )
    wu_proj_med = wu_proj_ci["point"]

    # ---- CRITICAL: fraction of Ŵ_U[※] captured by the 64-PC output subspace ----
    frac = _frac_in_subspace(pca_basis, wu_marker)
    read2_informative = frac >= FRAC_IN_SUBSPACE_UNINFORMATIVE

    # ---- Three refit floors, both reductions, family-clustered (identical harness) ----
    # Read 2 floor: verbatim make_refit_pair with r_hat = Ŵ_U[※].
    sc2_m0: dict = {}
    sc2_mplus: dict = {}
    sc2_shift: dict = {}
    m0_at_cplus = fitM.m0_at_cplus_ridge_full(C0, V0, Cplus, pca_basis)
    fn = _refit_ridge_fn(grid)
    fl2_m0 = make_refit_pair(
        C0, V0, fn, grid, wu_marker, families, n_pairs=N_REFIT_PAIRS, skip_counter=sc2_m0
    )
    fl2_mplus = make_refit_pair(
        Cplus, Vplus, fn, grid, wu_marker, families, n_pairs=N_REFIT_PAIRS, skip_counter=sc2_mplus
    )
    fl2_shift = make_refit_pair(
        Cplus,
        m0_at_cplus,
        fn,
        grid,
        wu_marker,
        families,
        n_pairs=N_REFIT_PAIRS,
        skip_counter=sc2_shift,
    )
    # Read 1 floor: _refit_pair_norm (‖·‖) on the SAME three refit designs.
    sc1_m0: dict = {}
    sc1_mplus: dict = {}
    sc1_shift: dict = {}
    fl1_m0 = _refit_pair_norm(
        C0, V0, fn, grid, families, n_pairs=N_REFIT_PAIRS, skip_counter=sc1_m0
    )
    fl1_mplus = _refit_pair_norm(
        Cplus, Vplus, fn, grid, families, n_pairs=N_REFIT_PAIRS, skip_counter=sc1_mplus
    )
    fl1_shift = _refit_pair_norm(
        Cplus, m0_at_cplus, fn, grid, families, n_pairs=N_REFIT_PAIRS, skip_counter=sc1_shift
    )

    def _combined(fm0, fmp, fsh):
        p95 = (
            float(np.percentile(fm0, 95)),
            float(np.percentile(fmp, 95)),
            float(np.percentile(fsh, 95)),
        )
        sd = max(floor_sd(fm0), floor_sd(fmp), floor_sd(fsh))
        return p95, max(p95), sd

    unproj_floor_p95, unproj_floor_comb, unproj_floor_sd = _combined(fl1_m0, fl1_mplus, fl1_shift)
    wu_floor_p95, wu_floor_comb, wu_floor_sd = _combined(fl2_m0, fl2_mplus, fl2_shift)

    def _skips(*counters):
        na = sum(int(c.get("n_attempted", 0)) for c in counters)
        ns = sum(int(c.get("n_skipped", 0)) for c in counters)
        return {"n_attempted": na, "n_skipped": ns, "skip_frac": (ns / na) if na else 0.0}

    # ---- Support distance ‖cplus − c0‖ ----
    support = np.linalg.norm(Cplus - C0, axis=1)

    cell: dict = {
        "behavior": "marker",
        "layer": layer,
        "n_cells": n,
        "n_families": len({*families}),
        # Read 1 — unprojected (behavior-agnostic).
        "unproj_delta_med": unproj_med,
        "unproj_delta_med_ci": unproj_ci,
        "unproj_floor_p95": {
            "M0": unproj_floor_p95[0],
            "Mplus": unproj_floor_p95[1],
            "shifted": unproj_floor_p95[2],
            "combined": unproj_floor_comb,
        },
        "unproj_floor_sd_combined": unproj_floor_sd,
        "unproj_delta_over_floor": (
            None if unproj_floor_comb < 1e-12 else float(unproj_med / unproj_floor_comb)
        ),
        "unproj_delta_over_floor_sd": (
            None if unproj_floor_sd < 1e-12 else float(unproj_med / unproj_floor_sd)
        ),
        # Read 2 — W_U[※]-projected (marker-specific).
        "wu_proj_delta_med": wu_proj_med,
        "wu_proj_delta_med_ci": wu_proj_ci,
        "wu_proj_floor_p95": {
            "M0": wu_floor_p95[0],
            "Mplus": wu_floor_p95[1],
            "shifted": wu_floor_p95[2],
            "combined": wu_floor_comb,
        },
        "wu_proj_floor_sd_combined": wu_floor_sd,
        "wu_proj_delta_over_floor": (
            None if wu_floor_comb < 1e-12 else float(wu_proj_med / wu_floor_comb)
        ),
        "wu_proj_delta_over_floor_sd": (
            None if wu_floor_sd < 1e-12 else float(wu_proj_med / wu_floor_sd)
        ),
        # CRITICAL validity check.
        "wu_frac_in_subspace": frac,
        "wu_read2_informative": read2_informative,
        "pca_dim": int(pca_basis.shape[0]),
        "support_distance": {
            "mean": float(support.mean()),
            "p90": float(np.percentile(support, SUPPORT_SHIFT_PCTL)),
        },
        "refit_skip": {
            "read1": _skips(sc1_m0, sc1_mplus, sc1_shift),
            "read2": _skips(sc2_m0, sc2_mplus, sc2_shift),
        },
    }

    # ---- Optional chain-ρ (Ŵ_U[※]ᵀ M̂(c) vs #537 marker E), ridge LOCO ----
    if with_chain:
        E = _load_marker_E(cell_keys)
        keep = ~np.isnan(E)
        chain: dict = {"n_with_E": int(keep.sum())}
        if keep.sum() >= 4:
            Ek = E[keep]
            fam_k = [f for f, m in zip(families, keep, strict=True) if m]
            m0_loco = fit658._ridge_predict_loco(C0, V0_64, fit658.RIDGE_LAMBDAS)
            mplus_loco = fit658._ridge_predict_loco(Cplus, Vplus_64, fit658.RIDGE_LAMBDAS)
            rho_m0, chain_m0 = _chain_rho_one(m0_loco[keep], pca_basis, wu_marker, Ek)
            rho_mplus, chain_mplus = _chain_rho_one(mplus_loco[keep], pca_basis, wu_marker, Ek)
            chain["rho_M0_ridge"] = rho_m0
            chain["rho_Mplus_ridge"] = rho_mplus
            chain["rho_diff_ridge"] = (
                None if (rho_m0 is None or rho_mplus is None) else float(rho_mplus - rho_m0)
            )
            if rho_m0 is not None:
                chain["ci_M0_ridge"] = clustered_bootstrap_spearman(chain_m0, Ek, fam_k)
            if rho_mplus is not None:
                chain["ci_Mplus_ridge"] = clustered_bootstrap_spearman(chain_mplus, Ek, fam_k)
        cell["chain_rho"] = chain

    return cell


def main() -> int:
    global N_REFIT_PAIRS, TARGET_DIM
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Issue #667 marker map-change (M0 vs M⁺), all layers")
    ap.add_argument("--layers", nargs="+", type=int, default=list(DEFAULT_LAYERS))
    ap.add_argument(
        "--local-store-root",
        required=True,
        help="on-disk mirror root (…/issue_667_alllayer/analysis_tensors)",
    )
    ap.add_argument("--out", type=Path, default=Path("/workspace/i667_marker_mapchange.json"))
    ap.add_argument("--refit-pairs", type=int, default=N_REFIT_PAIRS)
    ap.add_argument("--target-dim", type=int, default=TARGET_DIM)
    ap.add_argument(
        "--no-chain-rho",
        action="store_true",
        help="skip the optional chain-ρ (r_B=Ŵ_U[※] vs #537 marker E)",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="1 layer, capped sources/targets, few refit pairs, small target-dim",
    )
    ap.add_argument("--max-sources", type=int, default=None)
    ap.add_argument("--max-targets-per-source", type=int, default=None)
    ap.add_argument(
        "--no-fast-pca",
        action="store_true",
        help="use the ORIGINAL np.svd PCA (slow ~hours at all layers) instead of the "
        "#667 dual-eigh fast PCA (verified subspace-identical). Default: fast PCA on.",
    )
    args = ap.parse_args()

    if args.smoke:
        args.layers = args.layers[:1]
        args.max_sources = args.max_sources or 6
        if args.max_targets_per_source is None:
            args.max_targets_per_source = 4
        args.refit_pairs = min(args.refit_pairs, 8)
        args.target_dim = min(args.target_dim, 4)
    N_REFIT_PAIRS = args.refit_pairs
    TARGET_DIM = args.target_dim

    # fit658.DEVICE resolves at import (EPM_FIT_DEVICE env if set, else auto —
    # cuda when available; #876), so no hand-patch is needed here.
    logger.info("[phase=map_change] device=%s", fit658.DEVICE)
    fit658._assert_ridge_exactness()  # #658 reduction-order gate
    logger.info("[phase=map_change] ridge exactness gate PASS")

    wu_marker = load_wu_marker_direction()  # (3584,) unit Ŵ_U[※] + marker-id assert

    layers = tuple(args.layers)
    logger.info("[phase=map_change] marker map-change over layers %s", list(layers))

    # Local-mirror cell load (NO HF tree walk — the #667 hang). Marker cells only.
    streamer = loadact._Streamer(local_root=args.local_store_root)
    try:
        layout = loadact.list_store_layout_local(args.local_store_root, ("marker",))
        cells_by = loadact.load_cells(
            behaviors=("marker",),
            layers=layers,
            streamer=streamer,
            strict_counts=False,  # subset-safe; a smoke or all-layer subset need not be 480
            max_sources=args.max_sources,
            max_targets_per_source=args.max_targets_per_source,
            layout=layout,
        )
    finally:
        streamer.cleanup()

    use_fast = not args.no_fast_pca
    logger.info(
        "[phase=map_change] PCA path: %s", "dual-eigh FAST" if use_fast else "np.svd (slow)"
    )

    per_layer: dict[str, dict] = {}
    # Per-layer checkpoint dir (checkpoint-per-phase: a crash keeps completed layers).
    ckpt_dir = args.out.parent / f"{args.out.stem}_cells"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pca_ctx = _fast_pca_injected() if use_fast else _nullctx()
    with pca_ctx:
        for layer in layers:
            cells = cells_by[("marker", layer)]
            logger.info("[phase=map_change] marker L%d (%d cells)", layer, len(cells))
            cell = fit_marker_layer(layer, cells, wu_marker, with_chain=not args.no_chain_rho)
            cell["pca_path"] = "dual_eigh_fast" if use_fast else "np_svd"
            per_layer[f"marker_L{layer}"] = cell
            (ckpt_dir / f"marker_L{layer}.json").write_text(
                json.dumps(cell, indent=2, default=float)
            )
            logger.info(
                "[phase=map_change]   L%d unproj Δ/floor=%s  wu_proj Δ/floor=%s  "
                "frac_in_sub=%.4g%s",
                layer,
                _fmt(cell["unproj_delta_over_floor"]),
                _fmt(cell["wu_proj_delta_over_floor"]),
                cell["wu_frac_in_subspace"],
                "" if cell["wu_read2_informative"] else "  [READ2 UNINFORMATIVE]",
            )

    out_obj = {
        "issue": 667,
        "read": "marker_map_change_M0_vs_Mplus_floor_normalized",
        "marker_text": MARKER_TEXT,
        "marker_token_id": MARKER_ID,
        "base_model": BASE_MODEL,
        "wu_marker_direction": "unit lm_head row 83399 (residual-space read-out dir)",
        "n_refit_pairs": N_REFIT_PAIRS,
        "target_dim": TARGET_DIM,
        "frac_in_subspace_uninformative_threshold": FRAC_IN_SUBSPACE_UNINFORMATIVE,
        "git_commit": _git_sha(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "per_layer": per_layer,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_obj, indent=2, default=float))
    logger.info("[phase=map_change] wrote %s", args.out)

    _print_table(per_layer)
    logger.info("[phase=done]")
    return 0


def _nullctx():
    """No-op context manager for the ``--no-fast-pca`` path (leaves np.svd PCA in place)."""
    import contextlib

    return contextlib.nullcontext()


def _fmt(x) -> str:
    return "None" if x is None else f"{x:.3f}"


def _git_sha() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _print_table(per_layer: dict) -> None:
    """Print a per-layer table of the two Δ/floor reads + the subspace fraction."""
    print("\n=== MARKER map-change (M0 vs M⁺), floor-normalized ===")
    print(
        f"{'layer':>5} {'unproj Δ/floor':>15} {'wu_proj Δ/floor':>16} "
        f"{'frac_in_subspace':>17} {'read2':>10}"
    )
    print("-" * 70)
    for key in sorted(per_layer, key=lambda k: int(k.split("_L")[1])):
        c = per_layer[key]
        lay = c["layer"]
        u = _fmt(c["unproj_delta_over_floor"])
        w = _fmt(c["wu_proj_delta_over_floor"])
        f = c["wu_frac_in_subspace"]
        r2 = "OK" if c["wu_read2_informative"] else "UNINFORM"
        star = " <--" if lay in (14, 18) else ""
        print(f"{lay:>5} {u:>15} {w:>16} {f:>17.4f} {r2:>10}{star}")


if __name__ == "__main__":
    raise SystemExit(main())
