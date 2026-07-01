"""Issue #779 R3: context->answer map granularity (descriptive characterization).

(a) reconstruction R2/cosine (shared with R1's h diagnostic in fit_h.py), vs a
    shuffled-context null.
(b) per-relative-position predictability decay (Stage 2; computed from the pass-A
    10-rollout data — the expected-per-position, NOT pass B's single rollout).
(c) set-to-set linear CKA + linear predictability per layer-pair, with the
    Kornblith n-points >> dim reliability assertion (arXiv 1905.00414).

Stage 1 uses ONLY the (a) reconstruction-vs-shuffled-null cross-check (the
concerns-for-analyzer H1-direct disambiguator: a readout win WITHOUT a
reconstruction win above the shuffled-context null is a red flag). The full
R3(b)/(c) sweep is Stage 2.
"""

from __future__ import annotations

import numpy as np


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two activation sets (n, d_x) and (n, d_y).

    linear CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)  (Kornblith 1905.00414).
    Both matrices are column-mean-centered first. Asserts n >> max(d_x, d_y)
    (the reliability caveat: CKA/CCA-family invariants are unreliable when the
    number of points is below the representation dimension).
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    assert X.shape[0] == Y.shape[0], (X.shape, Y.shape)
    n = X.shape[0]
    dim = max(X.shape[1], Y.shape[1])
    assert n > dim, (
        f"linear_cka: n_points={n} <= dim={dim} — CKA unreliable below the "
        "representation dimension (Kornblith 1905.00414 n>>dim caveat). Pool more "
        "token vectors before computing."
    )
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    yx = float(np.linalg.norm(Yc.T @ Xc, ord="fro") ** 2)
    xx = float(np.linalg.norm(Xc.T @ Xc, ord="fro"))
    yy = float(np.linalg.norm(Yc.T @ Yc, ord="fro"))
    denom = xx * yy
    return yx / denom if denom > 1e-12 else float("nan")


def shuffled_context_null_r2(
    pred_fn, X_ctx: np.ndarray, Y_ans: np.ndarray, *, n_shuffle: int = 20, seed: int = 0
) -> dict:
    """Reconstruction R2 vs a shuffled-context null.

    ``pred_fn(X_ctx_perm) -> pred`` re-fits/re-applies the reconstruction under a
    row-permuted context (breaking the context->answer pairing). Returns the
    observed R2 and the null distribution mean/quantiles. A readout/reconstruction
    that does not beat this null is not carrying real context->answer structure.
    """
    from .fit_h import reconstruction_metrics

    rng = np.random.default_rng(seed)
    obs = reconstruction_metrics(pred_fn(X_ctx), Y_ans)["r2"]
    null_r2 = []
    n = X_ctx.shape[0]
    for _ in range(n_shuffle):
        perm = rng.permutation(n)
        r2 = reconstruction_metrics(pred_fn(X_ctx[perm]), Y_ans)["r2"]
        if np.isfinite(r2):
            null_r2.append(r2)
    return {
        "observed_r2": obs,
        "null_mean_r2": float(np.mean(null_r2)) if null_r2 else float("nan"),
        "null_p95_r2": float(np.quantile(null_r2, 0.95)) if null_r2 else float("nan"),
        "beats_null": bool(np.isfinite(obs) and null_r2 and obs > np.quantile(null_r2, 0.95)),
        "n_shuffle": len(null_r2),
    }


def position_bin_decay(
    per_token_answer: list[np.ndarray],
    context_vec: list[np.ndarray],
    *,
    n_bins: int = 10,
) -> dict:
    """R3(b) per-relative-position predictability decay (Stage 2 scaffold).

    ``per_token_answer[i]`` is the (n_resp_i, H) answer-token activation stack for
    context i (at a single layer); ``context_vec[i]`` is the (H,) pooled context.
    Bins each answer token by relative position j/T_a into ``n_bins`` bins and
    reports, per bin, the linear predictability (R2 of a ridge from context_vec to
    the bin-averaged answer activation) — the decay curve. Computed over the
    EXPECTED per-position activation (averaged within (context, bin)) across the
    10 rollouts upstream (the caller supplies the rollout-averaged stacks).

    Returns {"bin_r2": [...], "n_per_bin": [...]}. Stage-2 entry point; not run in
    Stage 1.
    """
    from .fit_h import ridge_fit_predict

    n_ctx = len(per_token_answer)
    assert n_ctx == len(context_vec), (n_ctx, len(context_vec))
    # For each context, average answer activations within each relative-position bin.
    bin_targets: list[list[np.ndarray]] = [[] for _ in range(n_bins)]
    bin_ctx: list[list[np.ndarray]] = [[] for _ in range(n_bins)]
    for i in range(n_ctx):
        acts = np.asarray(per_token_answer[i], dtype=np.float64)  # (T, H)
        t = acts.shape[0]
        if t == 0:
            continue
        rel = (np.arange(t) / t * n_bins).astype(int)
        rel = np.clip(rel, 0, n_bins - 1)
        for b in range(n_bins):
            mask = rel == b
            if mask.any():
                bin_targets[b].append(acts[mask].mean(0))
                bin_ctx[b].append(np.asarray(context_vec[i], dtype=np.float64))
    bin_r2, n_per_bin = [], []
    for b in range(n_bins):
        if len(bin_targets[b]) < 5:
            bin_r2.append(float("nan"))
            n_per_bin.append(len(bin_targets[b]))
            continue
        X = np.stack(bin_ctx[b])
        Y = np.stack(bin_targets[b])
        # simple in-sample R2 of a ridge fit (descriptive predictability).
        pred = ridge_fit_predict(X, Y, X)
        mu = Y.mean(0)
        ss_res = float(np.sum((Y - pred) ** 2))
        ss_tot = float(np.sum((Y - mu) ** 2))
        bin_r2.append(float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot)
        n_per_bin.append(len(bin_targets[b]))
    return {"bin_r2": bin_r2, "n_per_bin": n_per_bin, "n_bins": n_bins}
