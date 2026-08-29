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


def run_null_battery_screening(  # noqa: C901
    predictor_acts: np.ndarray,
    target: np.ndarray,
    rb_trait: np.ndarray,
    *,
    extraction_pos_acts: np.ndarray,
    extraction_neg_acts: np.ndarray,
    neutral_cov_per_layer: np.ndarray,
    other_rbs: dict[str, np.ndarray],
    pca_diff_acts: np.ndarray,
    layer_idx: int = LAYER_20_IDX,
    n_draws_stochastic: int = 200,
    n_draws_within_class: int = 50,
    seed: int = 42,
) -> dict:
    """Exp-5 honest 8-family null battery at the FROZEN layer ``layer_idx`` (20, 0-idx 19).

    Fixes the contaminated-null defect from the prior run:
    - OLD (contaminated): null draws from a shrunk covariance fit on the POOLED pos+neg
      pool, whose top PC ~ r_B (cos ~0.996) — the null was a near-copy of r_B.
    - NEW (honest): 8 families using SEPARATE pos/neg pools, isotropic draws, and
      the neutral covariance from v2 artifacts. See plan §4.C.

    Null families:
      (1) isotropic    — N(0, I*s2_l) normalized to ||r_B[l]||; s2 = mean eigenval of
                         neutral_cov, n=200. STOCHASTIC.
      (2) neutral_cov  — N(0, Sigma_neutral_l) renormed to ||r_B[l]||; n=200. STOCHASTIC.
      (3) within_pos   — random linear combo of pos residuals at l, project out r_B
                         BEFORE renorm; n=50. STOCHASTIC. Must-Fix A1.
      (4) within_neg   — same for neg residuals; n=50. STOCHASTIC.
      (5) rb_out_iso   — isotropic in the r_B-orthogonal subspace, renormed; n=100.
                         STOCHASTIC. (r_B projected out from isotropic draws.)
      (6) cross_trait  — fixed other-trait r_B directions; n=2. DESCRIPTIVE ONLY.
      (7) pca_top5     — fixed PCA top-5 of (pos-neg) diffs; n=5. DESCRIPTIVE ONLY.
      (8) contaminated — old pooled-cov randnorm (cos ~0.996 to r_B); n=200.
                         LABELED CONTAMINATED -- included as a reference/comparison
                         only; excluded from BH correction (plan §4.C Must-Fix S1).

    BH FDR correction is applied ONLY over stochastic families (1)-(5) with
    n_draws >= 50. Descriptive-only families (6-7) and the contaminated reference
    (8) are excluded.

    Must-Fix A1: within-class draws project out r_B BEFORE renorm; per-draw
    cos_to_rb is persisted in each draw's metadata.
    Must-Fix S1: conservative empirical-p formula p=(r+1)/(n+1); BH only over
    stochastic families.

    Args:
        predictor_acts: ``(n_datasets, N_LAYERS, D)`` per-dataset mean-projection-
            difference tensor (from ``dataset_mean_diff_activation``).
        target: ``(n_datasets,)`` #778 post-ft trait scores.
        rb_trait: ``(N_LAYERS, D)`` this trait's v2 r_B (RAW; project is scale-invariant).
        extraction_pos_acts: ``(n_pos, N_LAYERS, D)`` #778 extraction pos response-avg acts.
        extraction_neg_acts: ``(n_neg, N_LAYERS, D)`` #778 extraction neg response-avg acts.
        neutral_cov_per_layer: ``(N_LAYERS, D, D)`` full neutral covariance OR
            ``(N_LAYERS, D)`` diagonal form (from v2 HF artifact; used for families 1+2).
        other_rbs: ``{other_trait: (N_LAYERS, D)}`` for cross-trait null (family 6).
        pca_diff_acts: ``(n_pairs, N_LAYERS, D)`` (pos-neg) per-pair diffs for PCA null (family 7).
        layer_idx: frozen layer index (0-indexed; default 19 = layer 20 paper-1indexed).
        n_draws_stochastic: draws for families 1, 2, 8 (default 200).
        n_draws_within_class: draws for families 3, 4 and also family 5 (default 50;
            family 5 uses 100 draws by default via 2x this value).
        seed: base RNG seed; families draw in sequence from shared rng(seed).

    Returns:
        A JSON-serializable dict: observed |r| at the frozen layer, each null's
        band + conservative one-sided p at the frozen layer (stochastic families),
        BH-corrected q-values over stochastic families only, and per-draw x per-layer
        matrices (nested lists) for post-hoc recompute.
    """
    predictor_acts = np.asarray(predictor_acts, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    rb_trait = np.asarray(rb_trait, dtype=np.float64)
    neutral_cov_arr = np.asarray(neutral_cov_per_layer, dtype=np.float64)
    n_datasets = predictor_acts.shape[0]
    n_layers = predictor_acts.shape[1]
    D = predictor_acts.shape[2]
    assert rb_trait.shape == (n_layers, D), rb_trait.shape

    # Shared RNG — families draw in sequence so seed=42 is fully reproducible.
    rng = np.random.default_rng(seed)

    # Observed real-vector |r| per layer (frozen-layer read = out[layer_idx]).
    real_r_layers = null_battery.r_per_layer(predictor_acts, rb_trait, target)
    real_r_frozen = float(np.abs(real_r_layers[layer_idx]))

    rb_norm_per_layer = np.linalg.norm(rb_trait, axis=1)  # (L,)

    # --- Neutral cov helpers ---
    def _neutral_cov_at(layer: int) -> np.ndarray:
        """Return the (D, D) covariance at ``layer`` (expand diagonal form if needed)."""
        cov_layer = neutral_cov_arr[layer]
        if cov_layer.ndim == 1:
            # Diagonal form → full diagonal matrix
            return np.diag(cov_layer)
        return cov_layer

    def _sigma2_iso(layer: int) -> float:
        """Isotropic σ² = mean diagonal of neutral_cov at ``layer``."""
        cov_layer = neutral_cov_arr[layer]
        if cov_layer.ndim == 1:
            return float(cov_layer.mean())
        return float(np.diag(cov_layer).mean())

    # --- Family (1): isotropic N(0, I*s2_l) renormed ---
    logger.info("[null] family 1: isotropic (n=%d)", n_draws_stochastic)
    iso_mat = np.zeros((n_draws_stochastic, n_layers), dtype=np.float64)
    for draw in range(n_draws_stochastic):
        dirs = np.zeros((n_layers, D), dtype=np.float64)
        for layer in range(n_layers):
            sigma = float(np.sqrt(_sigma2_iso(layer)))
            z = rng.standard_normal(D) * sigma
            znorm = float(np.linalg.norm(z))
            target_norm = float(rb_norm_per_layer[layer])
            dirs[layer] = (z / znorm * target_norm) if znorm > 1e-12 else z
        iso_mat[draw] = np.abs(null_battery.r_per_layer(predictor_acts, dirs, target))

    # --- Family (2): neutral_cov N(0, Sigma_neutral_l) renormed ---
    # Precompute ONE Cholesky factor per layer (28 total) to avoid re-factorizing
    # inside the draw loop (~5,600 calls at n_draws=200). Diagonal fast path +
    # isotropic fallback when Cholesky fails.
    logger.info("[null] family 2: neutral_cov — precomputing %d Cholesky factors", n_layers)
    chol_factors: list = []  # length n_layers; entry is L (D,D), diag-std (D,), or None
    chol_types: list[str] = []  # "full", "diag", "iso_fallback"
    for layer in range(n_layers):
        cov_l = _neutral_cov_at(layer)
        if cov_l.ndim == 1:
            # Diagonal storage — fast path: treat as independent scales
            chol_factors.append(np.sqrt(np.maximum(cov_l, 0.0)))
            chol_types.append("diag")
        else:
            try:
                L_chol = np.linalg.cholesky(cov_l + 1e-8 * np.eye(D))
                chol_factors.append(L_chol)
                chol_types.append("full")
            except np.linalg.LinAlgError:
                # Fallback: isotropic using mean diagonal variance
                sigma = float(np.sqrt(_sigma2_iso(layer)))
                chol_factors.append(np.full(D, sigma))
                chol_types.append("iso_fallback")
    logger.info(
        "[null] family 2: Cholesky types — full=%d diag=%d iso_fallback=%d",
        chol_types.count("full"),
        chol_types.count("diag"),
        chol_types.count("iso_fallback"),
    )

    logger.info("[null] family 2: neutral_cov draw loop (n=%d)", n_draws_stochastic)
    ncov_mat = np.zeros((n_draws_stochastic, n_layers), dtype=np.float64)
    for draw in range(n_draws_stochastic):
        dirs = np.zeros((n_layers, D), dtype=np.float64)
        for layer in range(n_layers):
            factor = chol_factors[layer]
            ftype = chol_types[layer]
            z_unit = rng.standard_normal(D)
            z = factor @ z_unit if ftype == "full" else factor * z_unit
            znorm = float(np.linalg.norm(z))
            target_norm = float(rb_norm_per_layer[layer])
            dirs[layer] = (z / znorm * target_norm) if znorm > 1e-12 else z
        ncov_mat[draw] = np.abs(null_battery.r_per_layer(predictor_acts, dirs, target))

    # --- Family (3): within-class-pos (random combos of pos residuals, r_B projected out)
    # Must-Fix A1: project out r_B BEFORE renorm; persist cos_to_rb per draw.
    logger.info("[null] family 3: within_pos (n=%d)", n_draws_within_class)
    pos_acts = np.asarray(extraction_pos_acts, dtype=np.float64)  # (n_pos, L, D)
    neg_acts = np.asarray(extraction_neg_acts, dtype=np.float64)  # (n_neg, L, D)
    within_pos_mat = np.zeros((n_draws_within_class, n_layers), dtype=np.float64)
    within_pos_cos_to_rb = []  # per draw: list of cos(dir[layer_idx], rb[layer_idx])
    rb_unit_frozen = rb_trait[layer_idx] / (float(np.linalg.norm(rb_trait[layer_idx])) + 1e-12)
    for draw in range(n_draws_within_class):
        dirs = np.zeros((n_layers, D), dtype=np.float64)
        n_pos = pos_acts.shape[0]
        coeffs = rng.standard_normal(n_pos)
        for layer in range(n_layers):
            # Weighted combo of pos residuals
            v = coeffs @ pos_acts[:, layer, :]  # (D,)
            # Must-Fix A1: project out r_B at this layer BEFORE renorm
            rb_l = rb_trait[layer]
            rb_l_norm = float(np.linalg.norm(rb_l))
            if rb_l_norm > 1e-12:
                rb_l_unit = rb_l / rb_l_norm
                v = v - np.dot(v, rb_l_unit) * rb_l_unit
            vnorm = float(np.linalg.norm(v))
            target_norm = float(rb_norm_per_layer[layer])
            dirs[layer] = (v / vnorm * target_norm) if vnorm > 1e-12 else v
        within_pos_mat[draw] = np.abs(null_battery.r_per_layer(predictor_acts, dirs, target))
        # cos_to_rb at frozen layer (after projection-out, should be near-zero)
        d_frozen = dirs[layer_idx]
        d_frozen_norm = float(np.linalg.norm(d_frozen))
        cos_rb = (
            float(np.dot(d_frozen / d_frozen_norm, rb_unit_frozen))
            if d_frozen_norm > 1e-12
            else 0.0
        )
        within_pos_cos_to_rb.append(float(cos_rb))

    # --- Family (4): within-class-neg (random combos of neg residuals, r_B projected out)
    logger.info("[null] family 4: within_neg (n=%d)", n_draws_within_class)
    within_neg_mat = np.zeros((n_draws_within_class, n_layers), dtype=np.float64)
    within_neg_cos_to_rb = []
    for draw in range(n_draws_within_class):
        dirs = np.zeros((n_layers, D), dtype=np.float64)
        n_neg = neg_acts.shape[0]
        coeffs = rng.standard_normal(n_neg)
        for layer in range(n_layers):
            v = coeffs @ neg_acts[:, layer, :]  # (D,)
            # Must-Fix A1: project out r_B at this layer BEFORE renorm
            rb_l = rb_trait[layer]
            rb_l_norm = float(np.linalg.norm(rb_l))
            if rb_l_norm > 1e-12:
                rb_l_unit = rb_l / rb_l_norm
                v = v - np.dot(v, rb_l_unit) * rb_l_unit
            vnorm = float(np.linalg.norm(v))
            target_norm = float(rb_norm_per_layer[layer])
            dirs[layer] = (v / vnorm * target_norm) if vnorm > 1e-12 else v
        within_neg_mat[draw] = np.abs(null_battery.r_per_layer(predictor_acts, dirs, target))
        d_frozen = dirs[layer_idx]
        d_frozen_norm = float(np.linalg.norm(d_frozen))
        cos_rb = (
            float(np.dot(d_frozen / d_frozen_norm, rb_unit_frozen))
            if d_frozen_norm > 1e-12
            else 0.0
        )
        within_neg_cos_to_rb.append(float(cos_rb))

    # --- Family (5): r_B-projected-out isotropic (n = 2 * n_draws_within_class = 100)
    n_draws_rb_out = n_draws_within_class * 2
    logger.info("[null] family 5: rb_out_iso (n=%d)", n_draws_rb_out)
    rb_out_mat = np.zeros((n_draws_rb_out, n_layers), dtype=np.float64)
    for draw in range(n_draws_rb_out):
        dirs = np.zeros((n_layers, D), dtype=np.float64)
        for layer in range(n_layers):
            sigma = float(np.sqrt(_sigma2_iso(layer)))
            z = rng.standard_normal(D) * sigma
            # Project out r_B BEFORE renorm
            rb_l = rb_trait[layer]
            rb_l_norm = float(np.linalg.norm(rb_l))
            if rb_l_norm > 1e-12:
                rb_l_unit = rb_l / rb_l_norm
                z = z - np.dot(z, rb_l_unit) * rb_l_unit
            znorm = float(np.linalg.norm(z))
            target_norm = float(rb_norm_per_layer[layer])
            dirs[layer] = (z / znorm * target_norm) if znorm > 1e-12 else z
        rb_out_mat[draw] = np.abs(null_battery.r_per_layer(predictor_acts, dirs, target))

    # --- Family (6): cross-trait fixed directions (descriptive only, no p-value) ---
    logger.info("[null] family 6: cross_trait (n=%d)", len(other_rbs))
    crosstrait_mat = null_battery.crosstrait_null(other_rbs, predictor_acts, target)

    # --- Family (7): PCA top-5 (fixed, descriptive only) ---
    logger.info("[null] family 7: pca_top5")
    pca_mat = null_battery.pca_topk_null(pca_diff_acts, predictor_acts, target)

    # --- Family (8): contaminated reference (old pooled-cov randnorm, labeled as such) ---
    # This is the CONTAMINATED null from the prior run, included for comparison.
    # The pooled covariance top-PC ~ r_B (cos ~0.996) — it is NOT an honest null.
    # Excluded from BH correction. n=200.
    logger.info("[null] family 8: contaminated_pooled (reference only, n=%d)", n_draws_stochastic)
    pool_all = np.concatenate([pos_acts, neg_acts], axis=0)  # (n_pool, L, D)
    pool_per_layer = {layer: pool_all[:, layer, :] for layer in range(n_layers)}
    contaminated_mat = null_battery.randnorm_null_draws(
        pool_per_layer,
        rb_norm_per_layer,
        predictor_acts,
        target,
        n_draws=n_draws_stochastic,
        seed=seed,  # same seed for reproducibility with old run
    )

    # --- Conservative empirical-p (Must-Fix S1) ---
    # p = (r + 1) / (n_draws + 1) — applied only to stochastic families with n_draws >= 50.
    def _conservative_p(col: np.ndarray, observed: float) -> float:
        """Conservative one-sided p-value: (count_ge + 1) / (n + 1)."""
        col = col[~np.isnan(col)]
        n = int(col.size)
        if n == 0:
            return float("nan")
        r = int(np.sum(col >= observed))
        return float((r + 1) / (n + 1))

    def _band_and_p(mat: np.ndarray, stochastic: bool = True, label: str = "") -> dict:
        """Band + conservative one-sided p at the frozen layer."""
        col = np.asarray(mat, dtype=np.float64)[:, layer_idx]
        col = col[~np.isnan(col)]
        n = int(col.size)
        if n == 0:
            return {
                "p2_5": None,
                "p97_5": None,
                "one_sided_p": None,
                "n_draws": 0,
                "stochastic": stochastic,
            }
        lo, hi = np.percentile(col, [2.5, 97.5])
        one_sided_p = _conservative_p(col, real_r_frozen) if stochastic and n >= 50 else None
        return {
            "p2_5": float(lo),
            "p97_5": float(hi),
            "one_sided_p": one_sided_p,
            "n_draws": n,
            "stochastic": stochastic,
        }

    nulls = {
        "isotropic": _band_and_p(iso_mat, stochastic=True),
        "neutral_cov": _band_and_p(ncov_mat, stochastic=True),
        "within_pos": _band_and_p(within_pos_mat, stochastic=True),
        "within_neg": _band_and_p(within_neg_mat, stochastic=True),
        "rb_out_iso": _band_and_p(rb_out_mat, stochastic=True),
        "cross_trait": _band_and_p(crosstrait_mat, stochastic=False),
        "pca_top5": _band_and_p(pca_mat, stochastic=False),
        "contaminated_pooled": _band_and_p(contaminated_mat, stochastic=False),  # excluded from BH
    }

    # --- BH correction over stochastic families with n_draws >= 50 (Must-Fix S1) ---
    # Families (1)-(5); cross-trait (6), PCA (7), contaminated (8) excluded.
    stochastic_keys = ["isotropic", "neutral_cov", "within_pos", "within_neg", "rb_out_iso"]
    bh_pvals = []
    bh_keys_used = []
    for key in stochastic_keys:
        p = nulls[key]["one_sided_p"]
        if p is not None:
            bh_pvals.append(p)
            bh_keys_used.append(key)

    if bh_pvals:
        bh_qvals = null_battery.benjamini_hochberg(bh_pvals)
        for key, qval in zip(bh_keys_used, bh_qvals, strict=True):
            nulls[key]["bh_q"] = float(qval)
    for key in nulls:
        if "bh_q" not in nulls[key]:
            nulls[key]["bh_q"] = None

    # Emit bh_adjusted_stochastic as a top-level key for analysis_summary.json writer
    # (analysis.py:139 reads res.get("bh_adjusted_stochastic"))
    bh_adjusted_stochastic = {key: nulls[key].get("bh_q") for key in stochastic_keys}

    result = {
        "layer_idx_frozen": layer_idx,
        "layer_1indexed": layer_idx + 1,
        "n_datasets": int(n_datasets),
        "real_abs_r_frozen": real_r_frozen,
        "real_r_per_layer": [float(x) for x in real_r_layers],
        "seed": seed,
        "bh_correction_scope": "stochastic families (1-5) with n_draws>=50 only",
        "bh_keys_used": bh_keys_used,
        "bh_adjusted_stochastic": bh_adjusted_stochastic,
        "nulls": nulls,
        "within_class_metadata": {
            "within_pos_cos_to_rb_at_frozen_layer": within_pos_cos_to_rb,
            "within_neg_cos_to_rb_at_frozen_layer": within_neg_cos_to_rb,
        },
        # Per-draw x per-layer matrices (nested lists) for post-hoc recompute.
        "matrices": {
            "isotropic": iso_mat.tolist(),
            "neutral_cov": ncov_mat.tolist(),
            "within_pos": within_pos_mat.tolist(),
            "within_neg": within_neg_mat.tolist(),
            "rb_out_iso": rb_out_mat.tolist(),
            "cross_trait": crosstrait_mat.tolist(),
            "pca_top5": pca_mat.tolist(),
            "contaminated_pooled": contaminated_mat.tolist(),
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
