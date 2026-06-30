"""Issue #744 — token-to-token residual-stream continuity primitives.

Pure, I/O-free, unit-testable numeric functions for the consecutive-token
residual-stream continuity / discontinuity characterization (plan §4.4):

* ``zscore_population``        — per-dim z-standardization under FIXED population
  mean/std (Timkey 2109.04404: standardize before any token-to-token similarity).
* ``rank_rogue_dims``          — rank cosine-dominating dims by a NON-degenerate
  statistic computed on the RAW (pre-z-score) residuals (plan-marker concern #3:
  "standardized variance" is 1 everywhere after z-scoring and is therefore a
  degenerate ranking — we rank by ``raw_variance`` / ``max_dominance`` /
  ``contribution_to_cosine`` / ``kurtosis``).
* ``rogue_dim_ablate``         — zero out the top-k offending dims.
* ``consec_cosine``            — per-layer cos(h_t, h_{t+1}) over consecutive
  positions (signed, NOT abs — these are token-to-token similarities, not the
  Barenholtz abs-cosine direction read).
* ``direction_preservation``   — Barenholtz 2606.05346 §2.3 abs-cosine between the
  k=3 OLS fitted trajectory direction and the actual displacement at +0/+1/+2/+3.
* ``extrap_error``             — Barenholtz §2.2 L2 trajectory-extrapolation error.
* ``random_baseline``          — empirical Qwen chance abs-cosine of random token
  pairs (the d=3584 analogue of Barenholtz's 0.029 at d=768), per flavor.
* ``WelfordDimStats``          — streaming per-dim mean/var sufficient statistics
  (fp32 sums + sum-of-squares; population standardization stats for the broader
  STREAM corpus without retaining the 200 GB raw dump, plan §4.3).

All functions operate on ``torch.Tensor`` activation stacks shaped ``(L, T, H)``
(per sequence: L=layers, T=tokens, H=hidden). Functions are deterministic given
the seed. NO file I/O, NO model loading — those live in the dump / analyze
scripts so this module stays trivially unit-testable.

Norm note (plan §6): direction preservation is abs-cosine of *directions*, hence
scale-invariant — standardization matters less there than for ``extrap_error``,
which is an L2 distance and so changes with the per-dim scale. Both the
standardized (primary) and raw (Barenholtz-exact) variants are computed by the
caller by passing the appropriate ``H`` flavor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

# Rogue-dim ranking statistics (plan-marker concern #3). "standardized_variance"
# is deliberately ABSENT — after per-dim z-scoring every dim has variance 1, so
# ranking by it is degenerate. Each statistic below is computed on the RAW
# (pre-z-score) residual population.
ROGUE_RANK_METRICS = ("raw_variance", "max_dominance", "contribution_to_cosine", "kurtosis")
DEFAULT_ROGUE_RANK_METRIC = "raw_variance"

_EPS = 1e-8


def zscore_population(H: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Per-dim z-standardize ``H`` under FIXED population ``mu`` / ``sigma``.

    ``H`` is ``(..., hidden)``; ``mu`` / ``sigma`` are ``(hidden,)`` population
    statistics (per layer, estimated over the full token population — Pass 1 in
    the dump rig). Standardizing under the population stats (not the
    per-sequence stats) keeps the read population-consistent across sequences.
    """
    assert mu.shape == sigma.shape == (H.shape[-1],), (mu.shape, sigma.shape, H.shape)
    return (H - mu) / (sigma + _EPS)


def rank_rogue_dims(
    H_raw: torch.Tensor,
    top_k: int = 3,
    metric: str = DEFAULT_ROGUE_RANK_METRIC,
) -> torch.Tensor:
    """Rank the top-``k`` cosine-dominating dims by a NON-degenerate statistic.

    ``H_raw`` is the RAW (pre-z-score) activation population for one layer,
    shaped ``(N, hidden)`` (N tokens). Returns a ``(top_k,)`` LongTensor of dim
    indices, highest-statistic first.

    Why not "standardized variance" (plan-marker concern #3): after per-dim
    z-scoring every dim has variance exactly 1, so that statistic is degenerate.
    The supported metrics, all computed on the RAW residuals:

    * ``raw_variance``           — per-dim variance of the un-standardized
      activations (Timkey's anisotropy lens: the rogue dims carry outsized raw
      variance).
    * ``max_dominance``          — per-dim ``max|h_d| / median|h_d|`` (a sink-like
      dominance ratio; the dims a single token can blow up).
    * ``contribution_to_cosine`` — mean per-dim contribution to the
      consecutive-pair dot product ``mean_t(h_t,d * h_{t+1},d)`` in magnitude
      (the dims that literally drive the cosine numerator).
    * ``kurtosis``               — per-dim excess kurtosis (heavy-tailed dims).
    """
    assert H_raw.dim() == 2, H_raw.shape
    assert metric in ROGUE_RANK_METRICS, (
        f"unknown rogue-rank metric {metric!r}; {ROGUE_RANK_METRICS}"
    )
    n, hidden = H_raw.shape
    k = min(top_k, hidden)
    Hf = H_raw.float()
    if metric == "raw_variance":
        stat = Hf.var(dim=0, unbiased=False)
    elif metric == "max_dominance":
        absH = Hf.abs()
        med = absH.median(dim=0).values + _EPS
        stat = absH.max(dim=0).values / med
    elif metric == "contribution_to_cosine":
        if n < 2:
            stat = Hf.abs().mean(dim=0)
        else:
            prod = Hf[:-1] * Hf[1:]  # (N-1, hidden) per-dim consecutive products
            stat = prod.mean(dim=0).abs()
    elif metric == "kurtosis":
        mu = Hf.mean(dim=0, keepdim=True)
        sd = Hf.std(dim=0, unbiased=False, keepdim=True) + _EPS
        z = (Hf - mu) / sd
        stat = (z**4).mean(dim=0) - 3.0  # excess kurtosis
    else:  # pragma: no cover - guarded by the assert above
        raise ValueError(metric)
    return torch.topk(stat, k).indices.sort().values


def rogue_dim_ablate(H_std: torch.Tensor, rogue_idx: torch.Tensor) -> torch.Tensor:
    """Zero out the ``rogue_idx`` dims of the (already standardized) ``H_std``.

    ``H_std`` is ``(..., hidden)``; ``rogue_idx`` is a ``(k,)`` LongTensor.
    Returns a clone with those dims set to 0.0 (so the cosine no longer sees the
    anisotropy-dominating dims — Timkey 2109.04404).
    """
    H_ab = H_std.clone()
    H_ab[..., rogue_idx] = 0.0
    return H_ab


def make_flavors_from_stats(
    H: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, rogue_idx: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Build the {raw, std, ablate} flavor triad for one ``(L, T, hidden)`` stack.

    ``mu`` / ``sigma`` are per-layer ``(L, hidden)`` population stats (each layer
    standardized by its OWN mu/sigma); ``rogue_idx`` is per-layer ``(L, k)``
    top-k dim indices for the ablate flavor. Returns ``{"raw", "std",
    "ablate"}`` each ``(L, T, hidden)``. Vectorized over the layer axis (no
    per-layer Python loop): the ablate flavor zeros each layer's own rogue dims
    via a scatter.
    """
    assert H.dim() == 3, H.shape
    L, T, hidden = H.shape
    assert mu.shape == sigma.shape == (L, hidden), (mu.shape, sigma.shape, H.shape)
    std = (H - mu.unsqueeze(1)) / (sigma.unsqueeze(1) + _EPS)  # (L, T, hidden)
    ablate = std.clone()
    # scatter 0.0 into each layer's own rogue dims, broadcast over the T axis.
    k = rogue_idx.shape[1]
    idx = rogue_idx.unsqueeze(1).expand(L, T, k)  # (L, T, k)
    ablate.scatter_(2, idx, 0.0)
    return {"raw": H, "std": std, "ablate": ablate}


def consec_cosine(H: torch.Tensor) -> torch.Tensor:
    """Per-layer signed cosine between consecutive token positions.

    ``H`` is ``(L, T, hidden)``; returns ``(L, T-1)`` with
    ``cos(h^L_t, h^L_{t+1})``. Signed (not abs) — this is the token-to-token
    *similarity* DV (plan §6 row 1), distinct from the abs-cosine direction
    read.
    """
    assert H.dim() == 3, H.shape
    a, b = H[:, :-1], H[:, 1:]
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def _ols_fit_directions(H: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized k-window OLS line fit per layer over all valid windows.

    For each layer L and each end-position ``t`` with a full k-window
    ``H[:, t-k:t]`` (positions x = 0..k-1), fit ``h(x) ~ intercept + slope*x`` by
    OLS. Returns ``(slopes, intercepts)``, each ``(L, W, hidden)`` where
    ``W = T - k`` is the number of valid windows (window ``w`` covers
    ``H[:, w:w+k]`` and ends just before position ``w+k``).

    Closed-form OLS for x = 0..k-1: slope = cov(x, h) / var(x),
    intercept = mean(h) - slope*mean(x). No Python loop over t (einsum reduce).
    """
    _L, T, _hidden = H.shape
    assert k <= T, (T, k)
    W = T - k
    # Sliding windows: windows[:, w, j, :] = H[:, w+j, :], j in 0..k-1
    windows = H.unfold(dimension=1, size=k, step=1)  # (L, T-k+1, hidden, k)
    windows = windows[:, :W]  # drop the last window so step +s reads stay in-range
    windows = windows.permute(0, 1, 3, 2).contiguous()  # (L, W, k, hidden)
    x = torch.arange(k, dtype=H.dtype, device=H.device)  # (k,)
    x_mean = x.mean()
    x_centered = x - x_mean  # (k,)
    var_x = (x_centered**2).sum()  # scalar
    h_mean = windows.mean(dim=2)  # (L, W, hidden)
    # cov(x, h) = sum_j (x_j - x_mean) * h_j  (the /k cancels with var_x's /k)
    cov = torch.einsum("j,lwjh->lwh", x_centered, windows)  # (L, W, hidden)
    slope = cov / (var_x + _EPS)  # (L, W, hidden)
    intercept = h_mean - slope * x_mean
    return slope, intercept


def direction_preservation(
    H: torch.Tensor, k: int = 3, steps: tuple[int, ...] = (0, 1, 2, 3)
) -> dict[int, torch.Tensor]:
    """Barenholtz §2.3 direction preservation: abs-cosine(fitted dir, actual disp).

    ``H`` is ``(L, T, hidden)``. For each window ending just before position
    ``p = w + k`` (so the fit uses ``H[:, w:p]``), the fitted trajectory
    direction is the unit OLS slope vector. For each step ``s``:

        actual_disp = H[:, p + s] - H[:, p + s - 1]
        dp[s] = |cos(fitted_dir, actual_disp)|        (abs cosine)

    Returns ``{s: (L,) per-layer mean abs-cosine over valid windows}``. A window
    is valid for step ``s`` iff ``p + s <= T - 1`` (the +s-step displacement
    exists). Returns NaN for a layer/step with zero valid windows (the caller
    drops NaNs at the bootstrap; never silently zero-fills).
    """
    assert H.dim() == 3, H.shape
    L, T, _hidden = H.shape
    slope, _intercept = _ols_fit_directions(H, k)  # (L, W, hidden), W = T - k
    fitted_dir = slope / (slope.norm(dim=-1, keepdim=True) + _EPS)  # (L, W, hidden)
    out: dict[int, torch.Tensor] = {}
    for s in steps:
        # window w (fit H[:, w:w+k]) ends just before p = w + k; +s read needs
        # p + s = w + k + s <= T - 1  =>  w <= T - 1 - k - s.
        max_w = T - 1 - k - s
        if max_w < 0:
            out[s] = torch.full((L,), float("nan"))
            continue
        n_valid = max_w + 1
        p = torch.arange(n_valid, device=H.device) + k  # absolute end positions
        cur = H[:, p + s]  # (L, n_valid, hidden)
        prev = H[:, p + s - 1]  # (L, n_valid, hidden)
        disp = cur - prev
        disp_dir = disp / (disp.norm(dim=-1, keepdim=True) + _EPS)
        dirs = fitted_dir[:, :n_valid]  # (L, n_valid, hidden)
        cos = (dirs * disp_dir).sum(dim=-1).abs()  # (L, n_valid) abs cosine
        out[s] = cos.mean(dim=1)  # (L,)
    return out


def extrap_error(H: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Barenholtz §2.2 L2 trajectory-extrapolation error, per layer.

    ``H`` is ``(L, T, hidden)``. Fit the k=3 OLS line to ``H[:, w:w+k]``
    (positions 0..k-1), extrapolate to position k -> ``h_hat = intercept +
    slope*k``, and compute ``||H[:, w+k] - h_hat||_2`` (the actual next state at
    absolute position ``w+k``). Returns ``(L,)`` per-layer mean L2 error over
    valid windows. NaN if no valid window.

    L2 is scale-dependent — compute on the SAME ``H`` flavor (standardized for
    the primary read, raw for the Barenholtz-exact read), per plan §6 norm note.
    """
    assert H.dim() == 3, H.shape
    L, T, _hidden = H.shape
    if k + 1 > T:
        return torch.full((L,), float("nan"))
    slope, intercept = _ols_fit_directions(H, k)  # (L, W, hidden), W = T - k
    W = slope.shape[1]  # = T - k; window w predicts position w + k, valid iff w+k <= T-1
    h_hat = intercept + slope * float(k)  # (L, W, hidden) extrapolated to x=k
    p = torch.arange(W, device=H.device) + k  # absolute predicted positions
    actual = H[:, p]  # (L, W, hidden)
    err = (actual - h_hat).norm(dim=-1)  # (L, W)
    return err.mean(dim=1)  # (L,)


def random_baseline(H: torch.Tensor, n_pairs: int, seed: int, chunk: int = 4096) -> torch.Tensor:
    """Empirical Qwen chance abs-cosine of random token pairs, per layer.

    ``H`` is ``(L, T, hidden)`` (one flavor — caller computes one baseline per
    flavor, plan-marker concern #2). Samples ``n_pairs`` random ``(i, j)``
    token-position pairs (i != j) per layer with a seeded RNG, returns the
    ``(L,)`` mean abs-cosine. This is the d=3584 analogue of Barenholtz's 0.029
    chance baseline at d=768 (expect ~0.013 = sqrt(2 / (pi d))).

    The pairs are gathered + cosine'd in ``chunk``-sized blocks so the peak
    intermediate stays ``(L, chunk, hidden)`` instead of ``(L, n_pairs,
    hidden)`` — at L=24/28, hidden~896/3584, n_pairs=100k the un-chunked gather
    materializes multi-GB fp32 tensors and is pathologically slow on CPU.
    """
    assert H.dim() == 3, H.shape
    L, T, _hidden = H.shape
    if T < 2:
        return torch.full((L,), float("nan"))
    g = torch.Generator(device="cpu").manual_seed(seed)
    i = torch.randint(0, T, (n_pairs,), generator=g)
    j = torch.randint(0, T, (n_pairs,), generator=g)
    same = i == j  # resample collisions to j+1 (mod T) so i != j
    j[same] = (j[same] + 1) % T
    acc = torch.zeros(L, dtype=torch.float64)
    done = 0
    for start in range(0, n_pairs, chunk):
        end = min(start + chunk, n_pairs)
        a = H[:, i[start:end]]  # (L, c, hidden)
        b = H[:, j[start:end]]
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).abs()  # (L, c)
        acc += cos.sum(dim=1).double()
        done += end - start
    return (acc / max(done, 1)).float()  # (L,)


def closed_form_random_abs_cosine(hidden: int) -> float:
    """Closed-form expected abs-cosine of two random unit vectors in R^hidden.

    ``E[|cos|] ~ sqrt(2 / (pi * d))`` for large d (the A8 sanity check against
    the empirical ``random_baseline``). For d=768 -> ~0.029 (Barenholtz);
    d=3584 -> ~0.0133.
    """
    return math.sqrt(2.0 / (math.pi * hidden))


@dataclass
class WelfordDimStats:
    """Streaming per-dim mean / variance over a token population, per layer.

    Accumulates fp32 sum + sum-of-squares + count so the population
    z-standardization mean/std for the STREAM (broader) corpus can be fixed in
    one pass without retaining the 200 GB raw dump (plan §4.3 Pass 1). Variance
    via the sum-of-squares identity in fp32 (A7: reproduces the full-batch
    numpy mean/var within fp32 precision).

    Per layer ``L`` and dim ``d``:
        mean_d = sum_d / count
        var_d  = max(0, sumsq_d / count - mean_d**2)   # population variance
    """

    n_layers: int
    hidden: int
    count: torch.Tensor = field(default=None)  # (L,) int64
    sums: torch.Tensor = field(default=None)  # (L, hidden) fp64
    sumsq: torch.Tensor = field(default=None)  # (L, hidden) fp64

    def __post_init__(self) -> None:
        if self.count is None:
            self.count = torch.zeros(self.n_layers, dtype=torch.int64)
        if self.sums is None:
            self.sums = torch.zeros(self.n_layers, self.hidden, dtype=torch.float64)
        if self.sumsq is None:
            self.sumsq = torch.zeros(self.n_layers, self.hidden, dtype=torch.float64)

    def update(self, H: torch.Tensor) -> None:
        """Fold one sequence's ``(L, T, hidden)`` activations into the stats."""
        assert H.shape[0] == self.n_layers and H.shape[2] == self.hidden, H.shape
        Hf = H.double()
        self.count += H.shape[1]
        self.sums += Hf.sum(dim=1)
        self.sumsq += (Hf**2).sum(dim=1)

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mu, sigma)`` each ``(L, hidden)`` fp32 population stats."""
        cnt = self.count.clamp(min=1).double().unsqueeze(-1)  # (L, 1)
        mu = self.sums / cnt
        var = (self.sumsq / cnt - mu**2).clamp(min=0.0)
        sigma = var.sqrt()
        return mu.float(), sigma.float()
