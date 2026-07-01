"""Exp-5 pre-finetuning data screening (arXiv 2507.21509 ``section:data_iden``).

Dataset-level "projection difference" (all ASCII: L=layer, dot=inner product,
DeltaP=projection difference, v_hat=unit-normalized direction):

    DeltaP(dataset, trait) = mean_i [ a_L(x_i, y_i) - a_L(x_i, y'_i) ] dot v_hat_L

    a_L(x_i, y_i)  = response-avg activation of the TRAINING response y_i (layer L)
    a_L(x_i, y'_i) = base "natural" response projection, estimated by the paper's
                     LAST-PROMPT-TOKEN approximation (App ``appendix:efficient_estimation``
                     strategy 2) -- avoids running full base generation over the corpus
    v_hat_L        = unit-normalized r_B[trait][L]

The 24 datasets' DeltaP predicts their #778 post-ft trait shift (n=24 per trait).
The real ``v_hat`` is tested against the #778 4-null battery
(``src/explore_persona_space/analysis/null_battery.py``): norm-matched random,
shuffled-label permutation, cross-trait, PCA top-5.

REUSE STRATEGY (vectorized, no per-sample Python model loop):
this module does NOT do model forwards -- it consumes per-sample activation
tensors already captured by ``scripts/issue778_lib``'s
``capture_response_avg_all_layers`` / ``capture_last_prompt_token_all_layers``.
It reduces each dataset's per-sample (train-response-avg - last-prompt-token)
activation DIFFERENCES to a single mean-difference vector per (dataset, layer),
producing a ``(n_datasets, N_LAYERS, D)`` predictor tensor. Projecting that onto
a direction (``null_battery.project`` == the paper's ``a dot b / ||b||``) and
correlating with the trait-shift target IS the null battery's ``r_per_layer``
contract, so the random / permutation / cross-trait / PCA nulls all run UNMODIFIED
on this tensor (the permutation null re-derives the DIRECTION from the #778
extraction pos/neg pools; see ``run_null_battery_screening``).

The direction is FROZEN at layer 20 (``.claude/rules/selection-symmetric-nulls.md``
fixed-axis carve-out -- no max-over-layer in any Exp-816 headline), but we compute
+ persist all layers so the analyzer can recompute any honest per-layer band
post-hoc.
"""

from __future__ import annotations

import logging

import numpy as np

from explore_persona_space.analysis import null_battery

logger = logging.getLogger("issue816.screening")

# The paper's 1-indexed selected layer 20 == 0-indexed r_B/predictor index 19.
LAYER_20_IDX = 19


def build_projection_diff_predictor(
    train_resp_avg: np.ndarray,
    last_prompt_token: np.ndarray,
) -> np.ndarray:
    """Per-sample projection-difference activation, all layers.

    Returns ``diff_acts`` of shape ``(n_samples, N_LAYERS, D)`` where
    ``diff_acts[i] = train_resp_avg[i] - last_prompt_token[i]``. Projecting this
    onto ``v_hat`` (``null_battery.project``) gives the per-sample projection
    difference; averaging over samples then gives the dataset-level DeltaP.
    Keeping the per-sample tensor (rather than pre-averaging) lets the analyzer
    recompute any honest reduction post-hoc.

    Args:
        train_resp_avg: ``(n, N_LAYERS, D)`` response-avg activation of the
            TRAINING responses (``capture_response_avg_all_layers``).
        last_prompt_token: ``(n, N_LAYERS, D)`` last-prompt-token activation
            (``capture_last_prompt_token_all_layers``) -- the base "natural"
            projection approximation.
    """
    train_resp_avg = np.asarray(train_resp_avg, dtype=np.float64)
    last_prompt_token = np.asarray(last_prompt_token, dtype=np.float64)
    if train_resp_avg.shape != last_prompt_token.shape:
        raise ValueError(
            f"shape mismatch: train_resp_avg {train_resp_avg.shape} != "
            f"last_prompt_token {last_prompt_token.shape}"
        )
    if train_resp_avg.ndim != 3:
        raise ValueError(f"expected (n, N_LAYERS, D), got {train_resp_avg.shape}")
    return train_resp_avg - last_prompt_token


def dataset_projection_difference(
    diff_acts: np.ndarray,
    direction_per_layer: np.ndarray,
) -> np.ndarray:
    """DeltaP per layer for ONE dataset: mean over samples of (diff dot v_hat).

    Args:
        diff_acts: ``(n_samples, N_LAYERS, D)`` per-sample projection-difference
            activation (from ``build_projection_diff_predictor``).
        direction_per_layer: ``(N_LAYERS, D)`` direction (``r_B[trait]``; the
            projection is scale-invariant per ``null_battery.project`` which
            divides by ||direction||, so passing the RAW ``r_B`` gives the same
            DeltaP as passing the unit-normalized ``v_hat``).
    Returns:
        ``(N_LAYERS,)`` dataset-level DeltaP per layer.
    """
    diff_acts = np.asarray(diff_acts, dtype=np.float64)
    _n, L, D = diff_acts.shape
    direction_per_layer = np.asarray(direction_per_layer, dtype=np.float64)
    if direction_per_layer.shape != (L, D):
        raise ValueError(f"direction {direction_per_layer.shape} != (L,D)=({L},{D})")
    out = np.empty(L, dtype=np.float64)
    for layer in range(L):
        proj = null_battery.project(diff_acts[:, layer, :], direction_per_layer[layer])
        out[layer] = float(proj.mean())
    return out


def dataset_mean_diff_activation(diff_acts: np.ndarray) -> np.ndarray:
    """Mean projection-difference activation over samples: ``(N_LAYERS, D)``.

    This is the per-dataset predictor row the null battery consumes: stacking one
    per dataset yields ``predictor_acts`` of shape ``(n_datasets, N_LAYERS, D)``,
    and ``project(predictor_acts[:, L, :], v_hat[L]).mean``-per-row == the DeltaP
    vector. ``null_battery.r_per_layer(predictor_acts, direction, target)`` then
    computes the Pearson r between the per-dataset DeltaP and the trait-shift
    target at every layer -- EXACTLY Exp-5's regression, with EVERY null
    direction/draw inheriting the identical reduction (selection symmetry).
    """
    diff_acts = np.asarray(diff_acts, dtype=np.float64)
    if diff_acts.ndim != 3:
        raise ValueError(f"expected (n, N_LAYERS, D), got {diff_acts.shape}")
    return diff_acts.mean(axis=0)


def run_null_battery_screening(
    predictor_acts: np.ndarray,
    target: np.ndarray,
    rb_trait: np.ndarray,
    *,
    pool_acts_per_layer: dict[int, np.ndarray],
    extraction_pos_acts: np.ndarray,
    extraction_neg_acts: np.ndarray,
    other_rbs: dict[str, np.ndarray],
    pca_diff_acts: np.ndarray,
    layer_idx: int = LAYER_20_IDX,
    n_draws: int = null_battery.DEFAULT_N_DRAWS,
    seed: int = 0,
) -> dict:
    """Exp-5 real-vs-null-battery read at the FROZEN layer ``layer_idx`` (20, 0-idx 19).

    ``predictor_acts`` is the ``(n_datasets, N_LAYERS, D)`` per-dataset
    mean-projection-difference tensor; ``target`` the ``(n_datasets,)`` #778
    post-ft trait scores. The real |r| and every null's |r| are read at the SAME
    frozen layer (fixed-axis carve-out -- NO max-over-layer). The per-null-draw x
    per-layer |r| matrix from each null is persisted so the analyzer can recompute
    any honest band post-hoc.

    Args:
        rb_trait: ``(N_LAYERS, D)`` this trait's r_B (RAW; project is scale-invariant).
        pool_acts_per_layer: ``{layer: (n_pool, D)}`` activation pool for the
            covariance-realistic randnorm null (the #778 extraction pos+neg pool
            stacked, one entry per layer -- sampled activation covariance there).
        extraction_pos_acts / extraction_neg_acts: ``(n_pos, N_LAYERS, D)`` /
            ``(n_neg, N_LAYERS, D)`` #778 extraction response-avg activation pools
            (kept rollouts) -- the shuffled-label permutation null re-derives a
            diff-of-means DIRECTION from these (the same pools the real r_B was
            built from), NOT from the DeltaP predictor's train/base pairs.
        other_rbs: ``{other_trait: (N_LAYERS, D)}`` for the cross-trait null.
        pca_diff_acts: ``(n_pairs, N_LAYERS, D)`` (pos-neg) per-pair activation
            diffs for the PCA-top-5 null.
    Returns:
        A JSON-serializable dict: observed |r| at the frozen layer, each null's
        band + one-sided p at the frozen layer, and the per-draw x per-layer
        matrices (as nested lists) for post-hoc recompute.
    """
    predictor_acts = np.asarray(predictor_acts, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    rb_trait = np.asarray(rb_trait, dtype=np.float64)
    n = predictor_acts.shape[0]

    # Observed real-vector |r| per layer (frozen-layer read is out[layer_idx]).
    real_r_layers = null_battery.r_per_layer(predictor_acts, rb_trait, target)
    real_r_frozen = float(np.abs(real_r_layers[layer_idx]))

    rb_norm_per_layer = np.linalg.norm(rb_trait, axis=1)

    # --- Null 1: covariance-realistic norm-matched random ---
    randnorm_mat = null_battery.randnorm_null_draws(
        pool_acts_per_layer,
        rb_norm_per_layer,
        predictor_acts,
        target,
        n_draws=n_draws,
        seed=seed,
    )  # (n_draws, L) |r|
    # --- Null 2: shuffled-label permutation (re-derives the DIRECTION from the
    # #778 extraction pos/neg pools by shuffling their labels), then correlates
    # against the SAME ΔP predictor. Destroys the trait signal, keeps the pipeline.
    perm_mat = null_battery.perm_null_draws(
        extraction_pos_acts,
        extraction_neg_acts,
        predictor_acts,
        target,
        n_draws=n_draws,
        seed=seed,
    )  # (n_draws, L) |r|
    # --- Null 3: cross-trait (fixed directions) ---
    crosstrait_mat = null_battery.crosstrait_null(other_rbs, predictor_acts, target)
    # --- Null 4: PCA top-5 (fixed) ---
    pca_mat = null_battery.pca_topk_null(pca_diff_acts, predictor_acts, target)

    def _band_and_p(mat: np.ndarray) -> dict:
        col = np.asarray(mat, dtype=np.float64)[:, layer_idx]
        col = col[~np.isnan(col)]
        if col.size == 0:
            return {"p2_5": None, "p97_5": None, "one_sided_p": None, "n_draws": 0}
        lo, hi = np.percentile(col, [2.5, 97.5])
        one_sided_p = float(np.mean(col >= real_r_frozen))
        return {
            "p2_5": float(lo),
            "p97_5": float(hi),
            "one_sided_p": one_sided_p,
            "n_draws": int(col.size),
        }

    result = {
        "layer_idx_frozen": layer_idx,
        "layer_1indexed": layer_idx + 1,
        "n_datasets": int(n),
        "real_abs_r_frozen": real_r_frozen,
        "real_r_per_layer": [float(x) for x in real_r_layers],
        "nulls": {
            "randnorm": _band_and_p(randnorm_mat),
            "perm": _band_and_p(perm_mat),
            "crosstrait": _band_and_p(crosstrait_mat),
            "pca": _band_and_p(pca_mat),
        },
        # Per-draw x per-layer matrices (nested lists) for post-hoc honest bands.
        "matrices": {
            "randnorm": randnorm_mat.tolist(),
            "perm": perm_mat.tolist(),
            "crosstrait": crosstrait_mat.tolist(),
            "pca": pca_mat.tolist(),
        },
    }
    return result


def sample_level_separation(
    proj_ii: np.ndarray,
    proj_normal: np.ndarray,
) -> dict:
    """Per-sample projection separation (AUC) between a trait-inducing (II) dataset
    and its ``_normal`` control (the paper's Fig ``sample_wise`` histogram overlap).

    Args:
        proj_ii: ``(n_ii,)`` per-sample projections of the II dataset onto v̂[layer].
        proj_normal: ``(n_normal,)`` per-sample projections of the normal control.
    Returns:
        ``{"auc": ..., "n_ii": ..., "n_normal": ..., "mean_ii": ..., "mean_normal": ...}``
        AUC = P(a random II sample projects higher than a random normal sample),
        the Mann-Whitney-U-based rank-AUC (no sklearn dep).
    """
    a = np.asarray(proj_ii, dtype=np.float64)
    b = np.asarray(proj_normal, dtype=np.float64)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    n_a, n_b = a.size, b.size
    if n_a == 0 or n_b == 0:
        return {"auc": None, "n_ii": int(n_a), "n_normal": int(n_b)}
    # Rank-based AUC via Mann-Whitney U (ties count 0.5).
    all_vals = np.concatenate([a, b])
    order = np.argsort(all_vals, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, all_vals.size + 1)
    # Average ranks for ties.
    _assign_tie_ranks(all_vals, ranks)
    rank_sum_a = ranks[:n_a].sum()
    u_a = rank_sum_a - n_a * (n_a + 1) / 2.0
    auc = u_a / (n_a * n_b)
    return {
        "auc": float(auc),
        "n_ii": int(n_a),
        "n_normal": int(n_b),
        "mean_ii": float(a.mean()),
        "mean_normal": float(b.mean()),
    }


def _assign_tie_ranks(vals: np.ndarray, ranks: np.ndarray) -> None:
    """In-place average-rank assignment for tied values (Mann-Whitney convention)."""
    order = np.argsort(vals, kind="mergesort")
    sorted_vals = vals[order]
    i = 0
    n = vals.size
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
