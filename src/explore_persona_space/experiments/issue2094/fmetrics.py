"""Issue #2094 — pure-math fraction-of-swap metric library (plan §4.4 / §6).

Torch-vectorized, CPU-friendly, no model/network/file dependencies. All
functions accept batched leading dims (``*B``) so the analysis unit can
compute ~42k cells in one call instead of a Python loop (vectorize-first
rule). Internally computed in float64 for stability; results returned as
float32 tensors (or float64 where noted).

Contents:

- **F_act** (:func:`f_act`) — signed projection ``F = (s*t)/||t||^2`` with
  ``s`` = patched-minus-floor answer-state shift and ``t`` =
  ceiling-minus-floor axis; the floor is estimated from DISJOINT HALVES of
  the K floor draws, with both half-assignments averaged (the #1415
  shared-baseline-inflation fix). The naive shared-baseline estimator is
  returned as a record-only companion.
- **F_beh** (:func:`f_beh`) — ``(Δ̄_patched - Δ̄_floor)/(Δ̄_ceiling -
  Δ̄_floor)`` over per-draw dual-rubric contrasts ``Δ = (judge_B -
  judge_A)/100``; near-zero denominators return NaN + an explicit flag
  (never silent coercion), the unnormalized contrast rides along as the
  low-separation diagnostic.
- **Transport apply** (:func:`apply_ridge_map`, :func:`bind_map_orientation`,
  :func:`transport_predicted_shift`) — standardized-ridge application
  ``ŷ = ymu + z@W`` with ``z = (x - xmu)/xsd`` for the banked #779/#1738
  bundles, plus the runtime ORIENTATION BIND (both ``z@W`` and ``z@W.T``
  applied to probe rows; the decision is recorded for
  ``map_parity.json``).
- **Homogeneity** (:func:`pairwise_shift_cosines`,
  :func:`disattenuated_cosines`, :func:`log_log_magnitude_fit`,
  :func:`unity_slope_reference`) — direction-stability cosines with
  split-half disattenuation and the ``||shift||`` vs alpha log-log read with
  its unity-slope reference anchored at the alpha=1 fixed point.

Split-half reliability helpers (:func:`axis_split_half_reliability`,
:func:`shift_split_half_reliability`) expose the half-split machinery so
the t-axis reliability is computable per pair (registered analyzer
companion). The pair-clustered B=10,000 bootstrap is deliberately NOT here
— unit E owns it (batched index-GEMM per the vectorize rule).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

# Required keys of a banked standardized-ridge map bundle (#779 ridge.pt /
# #1738 prefix_ridge.pt; realized keys mmap-verified in plan §10).
REQUIRED_MAP_KEYS: tuple[str, ...] = ("kind", "xmu", "xsd", "ymu", "W", "layer")

# ── small shared numerics ─────────────────────────────────────────────


def safe_cosine(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Cosine similarity along ``dim``; a zero-norm side yields NaN (flagged, never coerced).

    A zero realized shift is a legitimate outcome (#1415 precedent), so this
    returns NaN rather than raising; callers count NaNs explicitly.
    """
    assert a.shape == b.shape, (a.shape, b.shape)
    a64, b64 = a.double(), b.double()
    na = a64.norm(dim=dim)
    nb = b64.norm(dim=dim)
    dot = (a64 * b64).sum(dim=dim)
    denom = na * nb
    out = torch.where(denom > 0, dot / denom.clamp_min(torch.finfo(torch.float64).tiny), torch.nan)
    return out.float()


def signed_projection(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Signed projection ``(s*t)/||t||^2`` along the last dim, batched.

    A zero-norm ``t`` yields NaN (explicit degenerate flag lives with the
    caller via ``torch.isnan``); never silently coerced.
    """
    assert s.shape == t.shape, (s.shape, t.shape)
    s64, t64 = s.double(), t.double()
    tt = (t64 * t64).sum(dim=-1)
    st = (s64 * t64).sum(dim=-1)
    out = torch.where(tt > 0, st / tt.clamp_min(torch.finfo(torch.float64).tiny), torch.nan)
    return out.float()


def spearman_brown(r: torch.Tensor | float) -> torch.Tensor | float:
    """Spearman-Brown step-up ``2r/(1+r)`` for a half-half reliability.

    Exact for correlations; used as the standard approximation for
    half-based cosine reliabilities of mean vectors. ``r == -1`` is
    undefined and returns NaN.
    """
    if isinstance(r, torch.Tensor):
        return torch.where(r > -1.0, 2.0 * r / (1.0 + r), torch.nan)
    return 2.0 * r / (1.0 + r) if r > -1.0 else math.nan


# ── disjoint-half baseline machinery (the #1415 fix) ──────────────────


def half_split_indices(k: int) -> tuple[list[int], list[int]]:
    """Deterministic even/odd half split of ``range(k)`` (reproducible default)."""
    assert k >= 2, f"need >=2 draws to split into disjoint halves, got {k}"
    return list(range(0, k, 2)), list(range(1, k, 2))


def disjoint_half_means(draws: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Means of the deterministic even/odd halves of ``draws``: ``(*B, K, d)`` → two ``(*B, d)``."""
    k = draws.shape[-2]
    even, odd = half_split_indices(k)
    return draws[..., even, :].mean(dim=-2), draws[..., odd, :].mean(dim=-2)


def _random_half_masks(k: int, n_splits: int, seed: int) -> torch.Tensor:
    """(n_splits, k) boolean masks, each a balanced (⌈k/2⌉) random half; seeded."""
    gen = torch.Generator().manual_seed(seed)
    masks = torch.zeros(n_splits, k, dtype=torch.bool)
    for i in range(n_splits):
        perm = torch.randperm(k, generator=gen)
        masks[i, perm[: (k + 1) // 2]] = True
    return masks


@dataclass
class FActResult:
    """Batched F_act read (all tensor fields share the batch shape ``*B``).

    ``f_act`` is the mean of the two disjoint half-assignments (primary);
    ``f_act_shared`` is the naive shared-baseline estimator, RECORD-ONLY
    (known-inflated; kept for comparability per the #1415 primary vs
    record-only labeling rule). ``degenerate`` marks cells whose t-axis had
    zero norm under ANY assignment (their ``f_act`` is NaN).
    """

    f_act: torch.Tensor
    f_act_assignments: torch.Tensor  # (2, *B) — the two half-assignments
    f_act_shared: torch.Tensor
    s_norm: torch.Tensor  # ||v_patched - floor_full_mean|| (diagnostic)
    t_norm: torch.Tensor  # ||ceiling_full_mean - floor_full_mean|| (diagnostic)
    traversal_ratio: torch.Tensor  # ||s||/||t|| (full-mean based, Result 3 companion)
    degenerate: torch.Tensor  # bool mask


def f_act(
    v_patched: torch.Tensor,
    floor_draws: torch.Tensor,
    ceiling_draws: torch.Tensor,
) -> FActResult:
    """F_act = (s*t)/||t||² with disjoint-half floor estimation, both assignments averaged.

    Args:
        v_patched: ``(*B, d)`` patched answer state (span-mean V_a).
        floor_draws: ``(*B, K_f, d)`` or ``(K_f, d)`` unpatched-under-A anchor
            draws (K_f >= 2; broadcast over ``*B`` when unbatched).
        ceiling_draws: ``(*B, K_c, d)`` or ``(K_c, d)`` generate-under-B anchor
            draws (full mean used — the ceiling appears only in ``t`` so it
            shares no noise with ``s``; only the FLOOR is split, per plan §4.4).

    The shared-baseline inflation being removed: with a shared floor mean,
    ``E[s*t]`` picks up ``+tr(Σ_floor)/K_f`` even when ``s`` and ``t`` are
    truly unrelated (#1415, ~+0.08 measured there). Disjoint halves make the
    numerator unbiased; the halved-floor ``t`` is noisier (attenuation), so
    truth typically sits between the two reads — both are returned.
    """
    d = v_patched.shape[-1]
    if floor_draws.dim() == 2:
        floor_draws = floor_draws.expand(*v_patched.shape[:-1], *floor_draws.shape)
    if ceiling_draws.dim() == 2:
        ceiling_draws = ceiling_draws.expand(*v_patched.shape[:-1], *ceiling_draws.shape)
    assert floor_draws.shape[-1] == d and ceiling_draws.shape[-1] == d, (
        v_patched.shape,
        floor_draws.shape,
        ceiling_draws.shape,
    )
    assert floor_draws.shape[:-2] == v_patched.shape[:-1], (
        floor_draws.shape,
        v_patched.shape,
    )

    fl_h1, fl_h2 = disjoint_half_means(floor_draws.double())
    fl_full = floor_draws.double().mean(dim=-2)
    ceil_full = ceiling_draws.double().mean(dim=-2)
    vp = v_patched.double()

    per_assignment = []
    for fl_s, fl_t in ((fl_h1, fl_h2), (fl_h2, fl_h1)):
        s = vp - fl_s
        t = ceil_full - fl_t
        per_assignment.append(signed_projection(s, t))
    assignments = torch.stack(per_assignment)  # (2, *B)

    s_full = vp - fl_full
    t_full = ceil_full - fl_full
    shared = signed_projection(s_full, t_full)

    s_norm = s_full.norm(dim=-1).float()
    t_norm = t_full.norm(dim=-1).float()
    traversal = torch.where(t_norm > 0, s_norm / t_norm.clamp_min(1e-38), torch.nan)

    return FActResult(
        f_act=assignments.mean(dim=0),
        f_act_assignments=assignments,
        f_act_shared=shared,
        s_norm=s_norm,
        t_norm=t_norm,
        traversal_ratio=traversal,
        degenerate=torch.isnan(assignments).any(dim=0),
    )


@dataclass
class AxisReliability:
    """Split-half reliability of the t = ceiling - floor axis (per batch element).

    ``mean_cos`` is the mean cosine between the two half-estimates of the
    axis across ``n_splits`` partitions (half-based); ``spearman_brown`` is
    the step-up to full-K reliability. Disattenuation consumes the
    Spearman-Brown value.
    """

    mean_cos: torch.Tensor
    spearman_brown: torch.Tensor
    n_splits: int


def axis_split_half_reliability(
    floor_draws: torch.Tensor,
    ceiling_draws: torch.Tensor,
    *,
    n_splits: int = 20,
    seed: int = 0,
) -> AxisReliability:
    """Split-half reliability of the ceiling-floor axis (registered analyzer companion).

    Both draw stacks (``(*B, K, d)``) are split into random balanced halves
    (aligned across the batch — ONE partition per split applied everywhere,
    the design-aligned split rule); the axis is estimated per half and the
    cosine between half-estimates is averaged over ``n_splits`` seeded
    partitions. ``n_splits=1`` with any seed still randomizes; use
    :func:`disjoint_half_means` for the deterministic even/odd split.
    """
    assert floor_draws.shape[-1] == ceiling_draws.shape[-1], (
        floor_draws.shape,
        ceiling_draws.shape,
    )
    kf, kc = floor_draws.shape[-2], ceiling_draws.shape[-2]
    assert kf >= 2 and kc >= 2, (kf, kc)
    fmask = _random_half_masks(kf, n_splits, seed)
    cmask = _random_half_masks(kc, n_splits, seed + 1)
    fd, cd = floor_draws.double(), ceiling_draws.double()

    cos_per_split = []
    for i in range(n_splits):
        f1 = fd[..., fmask[i], :].mean(dim=-2)
        f2 = fd[..., ~fmask[i], :].mean(dim=-2)
        c1 = cd[..., cmask[i], :].mean(dim=-2)
        c2 = cd[..., ~cmask[i], :].mean(dim=-2)
        cos_per_split.append(safe_cosine(c1 - f1, c2 - f2))
    mean_cos = torch.stack(cos_per_split).double().mean(dim=0).float()
    return AxisReliability(
        mean_cos=mean_cos,
        spearman_brown=spearman_brown(mean_cos),
        n_splits=n_splits,
    )


def shift_split_half_reliability(
    v_patched: torch.Tensor,
    floor_draws: torch.Tensor,
    *,
    n_splits: int = 20,
    seed: int = 0,
) -> torch.Tensor:
    """Split-half reliability of the shift s = v_patched - floor (floor-noise contribution).

    ``v_patched`` is a deterministic greedy draw (a constant, not noise), so
    only the floor estimation noise is assessed: the shift is recomputed
    under each half's floor mean and the half-half cosine is averaged over
    ``n_splits`` seeded partitions. Returned raw (half-based); apply
    :func:`spearman_brown` for the full-K value.
    """
    if floor_draws.dim() == 2:
        floor_draws = floor_draws.expand(*v_patched.shape[:-1], *floor_draws.shape)
    k = floor_draws.shape[-2]
    masks = _random_half_masks(k, n_splits, seed)
    vp, fd = v_patched.double(), floor_draws.double()
    cos_per_split = []
    for i in range(n_splits):
        s1 = vp - fd[..., masks[i], :].mean(dim=-2)
        s2 = vp - fd[..., ~masks[i], :].mean(dim=-2)
        cos_per_split.append(safe_cosine(s1, s2))
    return torch.stack(cos_per_split).double().mean(dim=0).float()


# ── F_beh (behavioral fraction-of-swap) ───────────────────────────────


def delta_contrast(judge_b: torch.Tensor, judge_a: torch.Tensor) -> torch.Tensor:
    """Per-draw dual-rubric contrast Δ = (judge_B - judge_A)/100 ∈ [-1, 1].

    Judge scores must already be the KEPT graded values in [0, 100]
    (drop-never-coerce happens upstream in the judge pipeline); out-of-range
    values fail loud here rather than propagate.
    """
    assert judge_b.shape == judge_a.shape, (judge_b.shape, judge_a.shape)
    for name, t in (("judge_b", judge_b), ("judge_a", judge_a)):
        assert torch.isfinite(t.double()).all(), f"{name} carries non-finite scores"
        assert bool((t >= 0).all() and (t <= 100).all()), f"{name} outside [0, 100]"
    return ((judge_b.double() - judge_a.double()) / 100.0).float()


@dataclass
class FBehResult:
    """Batched F_beh read. NaN + ``degenerate_denominator`` on near-zero separation.

    ``contrast`` (Δ̄_patched - Δ̄_floor, unnormalized) is the registered
    low-separation-pair diagnostic companion; ``denominator`` (Δ̄_ceiling -
    Δ̄_floor) lets the analysis unit apply its own separation threshold.
    """

    f_beh: torch.Tensor
    contrast: torch.Tensor
    denominator: torch.Tensor
    degenerate_denominator: torch.Tensor  # bool: |denom| < min_denominator → f_beh is NaN
    negative_denominator: torch.Tensor  # bool: ceiling below floor (pathological pair)


def f_beh(
    delta_patched_mean: torch.Tensor,
    delta_floor_mean: torch.Tensor,
    delta_ceiling_mean: torch.Tensor,
    *,
    min_denominator: float = 1e-9,
) -> FBehResult:
    """F_beh = (Δ̄_patched - Δ̄_floor)/(Δ̄_ceiling - Δ̄_floor), batched.

    Inputs are per-cell MEANS of :func:`delta_contrast` values (the caller
    aggregates draws; coherence gating and drop accounting live upstream).
    ``|denominator| < min_denominator`` yields NaN with an explicit flag —
    never a silent coercion; the unnormalized ``contrast`` remains valid for
    those cells. ``min_denominator`` is a pure numeric guard; the analysis
    unit applies its own (registered) low-separation threshold on
    ``denominator``.
    """
    assert delta_patched_mean.shape == delta_floor_mean.shape == delta_ceiling_mean.shape, (
        delta_patched_mean.shape,
        delta_floor_mean.shape,
        delta_ceiling_mean.shape,
    )
    dp = delta_patched_mean.double()
    df = delta_floor_mean.double()
    dc = delta_ceiling_mean.double()
    contrast = dp - df
    denom = dc - df
    degenerate = denom.abs() < min_denominator
    fb = torch.where(~degenerate, contrast / torch.where(degenerate, torch.inf, denom), torch.nan)
    return FBehResult(
        f_beh=fb.float(),
        contrast=contrast.float(),
        denominator=denom.float(),
        degenerate_denominator=degenerate,
        negative_denominator=denom < 0,
    )


# ── banked-map transport apply + orientation bind ─────────────────────


def validate_map_bundle(bundle: Mapping) -> None:
    """Assert a loaded #779/#1738-style bundle carries the standardized-ridge schema."""
    missing = [k for k in REQUIRED_MAP_KEYS if k not in bundle]
    assert not missing, f"map bundle missing keys {missing}; has {sorted(bundle.keys())}"
    xmu, xsd, ymu, w = bundle["xmu"], bundle["xsd"], bundle["ymu"], bundle["W"]
    assert xmu.dim() == 1 and xsd.dim() == 1 and ymu.dim() == 1 and w.dim() == 2, (
        xmu.shape,
        xsd.shape,
        ymu.shape,
        w.shape,
    )
    assert xmu.shape == xsd.shape, (xmu.shape, xsd.shape)
    for name, t in (("xmu", xmu), ("xsd", xsd), ("ymu", ymu), ("W", w)):
        assert torch.isfinite(t.double()).all(), f"bundle[{name!r}] carries non-finite values"
    assert bool((xsd.double() > 0).all()), "bundle xsd must be strictly positive"


def apply_ridge_map(
    bundle: Mapping,
    x: torch.Tensor,
    *,
    orientation: str,
) -> torch.Tensor:
    """Standardized-ridge apply ŷ = ymu + z@W (or z@W.T), z = (x - xmu)/xsd.

    ``orientation`` is the runtime-bound decision from
    :func:`bind_map_orientation`: ``"zW"`` (the native #779 fit convention,
    ``((x-xmu)/xsd) @ W + ymu``) or ``"Wz"`` (``z @ W.T``, i.e. ``W @ z``
    for a single row). Computed float64, returned float32 (the
    issue1415_map_transport precedent).
    """
    assert orientation in ("zW", "Wz"), orientation
    validate_map_bundle(bundle)
    xmu = bundle["xmu"].double()
    xsd = bundle["xsd"].double()
    ymu = bundle["ymu"].double()
    w = bundle["W"].double()
    d_in = w.shape[0] if orientation == "zW" else w.shape[1]
    assert x.shape[-1] == xmu.shape[0] == d_in, (x.shape, xmu.shape, w.shape, orientation)
    z = (x.double() - xmu) / xsd
    dev = z @ w if orientation == "zW" else z @ w.T
    assert dev.shape[-1] == ymu.shape[0], (dev.shape, ymu.shape)
    return (ymu + dev).float()


@dataclass
class OrientationDecision:
    """Runtime orientation bind for a banked map (recorded into map_parity.json)."""

    orientation: str  # "zW" | "Wz"
    criterion: str  # "probe-residual" | "scale-match"
    margin: float  # winner advantage; >= the min_margin passed to the bind
    stats: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "orientation": self.orientation,
            "criterion": self.criterion,
            "margin": self.margin,
            "stats": dict(self.stats),
        }


def bind_map_orientation(
    bundle: Mapping,
    probe_x: torch.Tensor,
    probe_y: torch.Tensor | None = None,
    *,
    reference_scale: float | None = None,
    min_margin: float = 1.1,
) -> OrientationDecision:
    """Bind the W orientation at runtime by applying BOTH ``z@W`` and ``z@W.T`` to probe rows.

    Preferred evidence (``probe_y`` given, e.g. a small lineage reproduction
    sample): pick the orientation with the smaller mean squared prediction
    residual ``||(ymu + dev) - y||²``. Fallback (``reference_scale`` given —
    the RMS norm of ``y - ymu`` on the lineage): pick the orientation whose
    prediction-deviation RMS is closer to it in log space (the plan §12
    "output-space scale against ymu residuals" check).

    Fail-fast: an ambiguous bind (winner advantage < ``min_margin``) raises
    ValueError — fall back to the held-out reproduction check rather than
    guessing. A non-square W is orientation-determined by shape alone and is
    decided directly (criterion ``"shape"``).
    """
    validate_map_bundle(bundle)
    w = bundle["W"].double()
    xmu = bundle["xmu"].double()
    xsd = bundle["xsd"].double()
    ymu = bundle["ymu"].double()
    if probe_x.dim() == 1:
        probe_x = probe_x.unsqueeze(0)
    assert probe_x.dim() == 2, probe_x.shape

    if w.shape[0] != w.shape[1]:
        d_in = xmu.shape[0]
        assert d_in in w.shape, (w.shape, d_in)
        orientation = "zW" if w.shape[0] == d_in else "Wz"
        return OrientationDecision(
            orientation=orientation, criterion="shape", margin=math.inf, stats={}
        )

    assert probe_x.shape[-1] == xmu.shape[0], (probe_x.shape, xmu.shape)
    z = (probe_x.double() - xmu) / xsd
    dev = {"zW": z @ w, "Wz": z @ w.T}

    if probe_y is not None:
        if probe_y.dim() == 1:
            probe_y = probe_y.unsqueeze(0)
        assert probe_y.shape == (probe_x.shape[0], ymu.shape[0]), (probe_y.shape, probe_x.shape)
        resid = {
            o: float(((ymu + d) - probe_y.double()).pow(2).sum(dim=-1).mean().sqrt())
            for o, d in dev.items()
        }
        winner = min(resid, key=resid.__getitem__)
        loser = "Wz" if winner == "zW" else "zW"
        margin = resid[loser] / max(resid[winner], torch.finfo(torch.float64).tiny)
        decision = OrientationDecision(
            orientation=winner,
            criterion="probe-residual",
            margin=margin,
            stats={f"rms_residual_{o}": v for o, v in resid.items()},
        )
    elif reference_scale is not None:
        assert reference_scale > 0, reference_scale
        scale = {o: float(d.norm(dim=-1).pow(2).mean().sqrt()) for o, d in dev.items()}
        dist = {
            o: abs(math.log(max(s, 1e-300)) - math.log(reference_scale)) for o, s in scale.items()
        }
        winner = min(dist, key=dist.__getitem__)
        loser = "Wz" if winner == "zW" else "zW"
        margin = math.exp(dist[loser] - dist[winner])
        decision = OrientationDecision(
            orientation=winner,
            criterion="scale-match",
            margin=margin,
            stats={
                **{f"dev_rms_{o}": v for o, v in scale.items()},
                "reference_scale": reference_scale,
            },
        )
    else:
        raise ValueError("bind_map_orientation needs probe_y or reference_scale")

    if decision.margin < min_margin:
        raise ValueError(
            f"ambiguous map orientation (criterion={decision.criterion}, "
            f"margin={decision.margin:.4f} < {min_margin}); run the held-out "
            f"reproduction check instead of guessing. stats={decision.stats}"
        )
    return decision


def transport_predicted_shift(
    bundle: Mapping,
    v_s: torch.Tensor,
    delta: torch.Tensor,
    alpha: float,
    *,
    orientation: str,
) -> torch.Tensor:
    """Predicted answer-state shift under the banked map: f(V_s + alphaΔ) - f(V_s).

    Computed as two applies (robust to any future non-affine map); for the
    affine standardized ridge this equals ``((alphaΔ)/xsd) @ W`` exactly. Pair
    with :func:`safe_cosine` against the realized shift for the transport
    cosine (plan §4.4).
    """
    assert v_s.shape == delta.shape, (v_s.shape, delta.shape)
    f_edit = apply_ridge_map(bundle, v_s + alpha * delta, orientation=orientation)
    f_base = apply_ridge_map(bundle, v_s, orientation=orientation)
    return f_edit - f_base


# ── homogeneity / linearity reads (Result 1c) ─────────────────────────


def pairwise_shift_cosines(shifts: torch.Tensor) -> torch.Tensor:
    """(A, A) cosine matrix over dose-indexed shifts ``(A, d)`` (one matmul, no loops).

    A zero-norm shift row yields NaN in its row/column (flagged, never
    coerced).
    """
    assert shifts.dim() == 2, shifts.shape
    s = shifts.double()
    norms = s.norm(dim=-1, keepdim=True)
    unit = s / norms.clamp_min(torch.finfo(torch.float64).tiny)
    cos = unit @ unit.T
    zero = norms.squeeze(-1) == 0
    cos[zero, :] = torch.nan
    cos[:, zero] = torch.nan
    return cos.float()


def disattenuated_cosines(cos_matrix: torch.Tensor, reliabilities: torch.Tensor) -> torch.Tensor:
    """Disattenuate ``cos_obs / sqrt(r_i * r_j)``; non-positive reliabilities yield NaN.

    ``reliabilities`` are the per-dose split-half reliabilities of the shift
    estimates (Spearman-Brown-corrected values from
    :func:`shift_split_half_reliability` + :func:`spearman_brown`). The
    diagonal is not special-cased (it disattenuates to ``1/r_i``, a
    reliability read in itself). Values may exceed 1 under noisy
    reliabilities — report, don't clip.
    """
    a = cos_matrix.shape[0]
    assert cos_matrix.shape == (a, a), cos_matrix.shape
    assert reliabilities.shape == (a,), (reliabilities.shape, cos_matrix.shape)
    r = reliabilities.double()
    denom = (r.unsqueeze(0) * r.unsqueeze(1)).sqrt()
    valid = (r > 0).unsqueeze(0) & (r > 0).unsqueeze(1)
    out = torch.where(
        valid, cos_matrix.double() / denom.clamp_min(torch.finfo(torch.float64).tiny), torch.nan
    )
    return out.float()


def log_log_magnitude_fit(
    alphas: torch.Tensor, norms: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Least-squares slope + intercept of ``log||shift||`` vs ``log alpha``, batched.

    ``alphas``: ``(A,)`` strictly positive doses; ``norms``: ``(*B, A)``
    strictly positive shift norms (a non-positive norm fails loud — log
    undefined; exclude degenerate cells upstream). Returns ``(slope,
    intercept)`` each ``(*B,)``; slope 1 = homogeneous (linear) response.
    """
    assert alphas.dim() == 1 and alphas.shape[0] >= 2, alphas.shape
    assert norms.shape[-1] == alphas.shape[0], (norms.shape, alphas.shape)
    assert bool((alphas.double() > 0).all()), "alphas must be strictly positive"
    assert bool((norms.double() > 0).all()), "norms must be strictly positive (log undefined)"
    la = alphas.double().log()
    ln = norms.double().log()
    la_c = la - la.mean()
    var = (la_c * la_c).sum()
    assert float(var) > 0, "alphas are all identical — slope undefined"
    slope = (ln * la_c).sum(dim=-1) / var
    intercept = ln.mean(dim=-1) - slope * la.mean()
    return slope.float(), intercept.float()


def unity_slope_reference(alphas: torch.Tensor, norm_at_alpha1: torch.Tensor) -> torch.Tensor:
    """Unity-slope reference norms ``||shift(1)|| * alpha`` (alpha=1 fixed-point convention).

    ``norm_at_alpha1``: ``(*B,)`` the measured shift norm at alpha=1; returns
    ``(*B, A)`` reference norms for the log-log overlay.
    """
    assert alphas.dim() == 1, alphas.shape
    assert bool((alphas.double() > 0).all()), "alphas must be strictly positive"
    return (norm_at_alpha1.double().unsqueeze(-1) * alphas.double()).float()
