"""Issue #651 — off-pod analysis stack (CPU, runs on the VM).

Pure, testable functions that turn the per-cell shift tensors (produced by the
extraction sweep) into the Q1 / Q2 / variance-decomposition / seed-ceiling
answers (plan §6). Nothing here touches a GPU or HF; it operates on the
``.pt`` payloads written by ``analysis/activation_shift.py`` (schema_version 2).

The geometric objects (plan §6.2 / §14.1-2 — DO NOT conflate):

- **per-cell shift** ``delta_v_b(context)`` — the mean-over-questions slot
  (or mean_resp) residual shift for ONE (behavior, context, seed) cell, on the
  fixed 14-persona panel. A single (H=3584,) vector PER cell after we collapse
  the panel — but for the per-cell read we first build the cell's (H x 14)
  panel matrix and take its top-direction U1 (the cell's "write direction").
- **Q1 matrix** for behavior b: stack the per-context cell vectors into an
  (n_context x H) matrix → SVD → top-share + per-context cos-to-U1.
- **Q1 ceiling** = per-(behavior, context) cosine of the cell's read across
  seeds (n=2: 42 vs 1042). A single cosine per cell.
- **Q2 ceiling** = per-behavior U1 cross-seed cosine.
- Both ceilings are DIFFERENT geometric objects from #552's published
  0.975/0.982 (that is the within-cell per-persona mean_cos_to_U1
  CONCENTRATION). We compute the operative ceiling FRESH here (plan §14.1).

The per-cell "read vector" used for Q1/Q2 is the cell's panel-collapsed
direction. Two equally-defensible collapses exist; the plan's headline reads
the cell's TOP SINGULAR DIRECTION over the 14-persona panel (its dominant
write direction). We expose both ``cell_read="mean"`` (mean over panel columns)
and ``cell_read="u1"`` (top singular direction); ``u1`` is the headline.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Literal

import numpy as np

from explore_persona_space.analysis.svd_direction_constancy import (
    cosine,
    sign_flip_null,
    svd_summary,
)
from explore_persona_space.experiments.issue_651 import panel_column_order

logger = logging.getLogger(__name__)

CellRead = Literal["mean", "u1"]


def cell_panel_matrix(
    shifts: dict[str, dict],
    *,
    key: str = "delta_v",
    column_order: Sequence[str] | None = None,
) -> np.ndarray:
    """Assemble one cell's (H x N=14) panel matrix from its shifts dict.

    ``shifts`` is the ``payload["shifts"]`` from a cell's .pt file:
    ``{persona_name: {"delta_v": (H,) tensor, "delta_v_mean_resp": ...}}``.
    ``key`` selects the slot read (``delta_v``) or the mean-resp read
    (``delta_v_mean_resp``). Column order is pinned to the panel (default
    ``panel_column_order()``) so it never co-varies with the cell.
    """
    order = list(column_order) if column_order is not None else panel_column_order()
    cols = []
    for p in order:
        if p not in shifts:
            raise KeyError(f"cell shifts missing panel persona {p!r}")
        entry = shifts[p]
        if key not in entry:
            raise KeyError(f"persona {p!r}: missing key {key!r} (have {sorted(entry)})")
        v = entry[key]
        arr = np.asarray(v.detach().float().cpu().numpy() if hasattr(v, "detach") else v)
        cols.append(arr.astype(np.float32).ravel())
    M = np.stack(cols, axis=1)  # (H, N)
    assert M.ndim == 2 and M.shape[1] == len(order), M.shape
    return M


def cell_read_vector(
    shifts: dict[str, dict],
    *,
    key: str = "delta_v",
    cell_read: CellRead = "u1",
    column_order: Sequence[str] | None = None,
) -> np.ndarray:
    """One cell's single (H,) read direction (the unit of Q1/Q2 geometry).

    ``cell_read="u1"`` (headline): the cell's top left-singular direction over
    its 14-persona panel matrix (its dominant write direction).
    ``cell_read="mean"``: the mean over the 14 panel columns.
    Returned UNNORMALIZED (callers unit-norm for the dose-invariant Q2 DV).
    """
    M = cell_panel_matrix(shifts, key=key, column_order=column_order)
    if cell_read == "mean":
        return M.mean(axis=1).astype(np.float32)
    summ = svd_summary(M)
    # svd_summary already sign-orients U1 so the mean column projects >= 0.
    return summ["U1"].astype(np.float32)


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)


def q1_context_invariance(
    per_context_read: dict[str, np.ndarray],
    *,
    n_reps: int = 1000,
    seed: int = 0,
) -> dict:
    """Q1 for one behavior: is the write the same direction across contexts?

    ``per_context_read`` = {context_id: (H,) cell read vector} for ONE behavior
    at ONE seed (the contexts it was implanted under). Build the
    (n_context x H) matrix, SVD it (transposed to H x n_context so the columns
    are contexts), report top-share (norm-weighted) + per-context cos-to-U1 +
    the sign-flip null p95 (BINDING, plan §5 control ii).
    """
    contexts = sorted(per_context_read)
    if len(contexts) < 2:
        raise ValueError(f"need >=2 contexts for Q1 SVD, got {len(contexts)}")
    # Columns = contexts (matches svd_direction_constancy's M convention: H x N).
    M = np.stack(
        [np.asarray(per_context_read[c], dtype=np.float32).ravel() for c in contexts], axis=1
    )
    summ = svd_summary(M)
    null = sign_flip_null(M, n_reps=n_reps, seed=seed)
    # Unit-norm top-share: SVD of the column-unit-normalized matrix.
    M_unit = np.stack([_unit(M[:, i]) for i in range(M.shape[1])], axis=1)
    summ_unit = svd_summary(M_unit)
    return {
        "contexts": contexts,
        "n_contexts": len(contexts),
        "top_share_norm_weighted": float(summ["s_top1_frac"]),
        "top_share_unit_norm": float(summ_unit["s_top1_frac"]),
        "cos_to_U1": {c: float(v) for c, v in zip(contexts, summ["cos_to_U1"], strict=True)},
        "mean_cos_to_U1": float(np.mean(summ["cos_to_U1"])),
        "sign_flip_null_p95": float(null["p95"]),
        "sign_flip_null_p99": float(null["p99"]),
        "top_share_clears_null_p95": bool(summ["s_top1_frac"] > null["p95"]),
        "U1": summ["U1"].tolist(),
    }


def q1_verdict(
    q1: dict,
    seed_ceiling_median: float,
    *,
    ceiling_fraction: float = 0.85,
    context_coverage: float = 0.80,
) -> dict:
    """Q1 context-invariance verdict against the seed ceiling (plan §6.2).

    context-invariant iff (a) norm-weighted top-share > sign-flip null p95 AND
    (b) per-context cos-to-U1 >= ceiling_fraction x seed_ceiling_median for
    >= context_coverage of contexts. ``seed_ceiling_median`` is computed FRESH
    from this task's seed-42 + seed-1042 tensors (NEVER the 0.975/0.982 #552
    concentration — plan §14.1).
    """
    bar = ceiling_fraction * seed_ceiling_median
    cos = q1["cos_to_U1"]
    n_pass = sum(1 for v in cos.values() if v >= bar)
    frac_pass = n_pass / max(len(cos), 1)
    invariant = bool(q1["top_share_clears_null_p95"] and frac_pass >= context_coverage)
    return {
        "verdict": "context_invariant" if invariant else "context_specific",
        "context_invariant": invariant,
        "per_context_bar": float(bar),
        "frac_contexts_at_or_above_bar": float(frac_pass),
        "n_contexts_at_or_above_bar": int(n_pass),
        "seed_ceiling_median": float(seed_ceiling_median),
        "ceiling_fraction": ceiling_fraction,
        "context_coverage_required": context_coverage,
    }


def seed_ceiling_per_cell(
    read_seed_a: dict[str, np.ndarray],
    read_seed_b: dict[str, np.ndarray],
) -> dict:
    """Within-cell seed ceiling: |cos(read_seed42, read_seed1042)| per context.

    The benchmark every Q1/Q2 cosine is reported AS A FRACTION OF (#552 lesson,
    plan §14.1). A single cosine per (behavior, context) cell — n=2 seeds, so
    NO within-ceiling CI; the analyzer flags near-boundary ratios as
    noise-limited (plan §14.3). Returns per-context cosines + the behavior median.
    """
    shared = sorted(set(read_seed_a) & set(read_seed_b))
    if not shared:
        raise ValueError("no shared contexts between the two seeds for the ceiling")
    per_context = {c: abs(cosine(read_seed_a[c], read_seed_b[c])) for c in shared}
    vals = list(per_context.values())
    return {
        "per_context": per_context,
        "median": float(np.median(vals)),
        "mean": float(np.mean(vals)),
        "n_cells": len(vals),
        "contexts": shared,
    }


def q2_cross_behavior_matrix(
    behavior_u1: dict[str, np.ndarray],
    seed_ceilings: dict[str, float],
    *,
    n_reps: int = 1000,
    seed: int = 0,
) -> dict:
    """Q2: 4x4 (or kxk) cross-behavior dominant-direction cosine matrix.

    ``behavior_u1`` = {behavior: (H,) dominant direction U1} (one per behavior,
    the per-behavior cross-context U1 from q1["U1"]). ``seed_ceilings`` =
    {behavior: median seed ceiling}. The unit-norm |cos| is dose-invariant
    (plan §6.1 — the explicit reason this is the Q2 DV). Each off-diagonal is
    ALSO reported as a fraction of geomean(ceiling_b, ceiling_b') (plan §6.2).
    The cross-behavior null band is the sign-flip null on the pooled
    behavior-direction matrix.
    """
    behaviors = sorted(behavior_u1)
    k = len(behaviors)
    raw = np.eye(k, dtype=np.float64)
    ceil_frac = np.full((k, k), np.nan)
    unit = {b: _unit(behavior_u1[b]) for b in behaviors}
    for i, bi in enumerate(behaviors):
        for j, bj in enumerate(behaviors):
            if i == j:
                ceil_frac[i, j] = 1.0
                continue
            c = abs(cosine(unit[bi], unit[bj]))
            raw[i, j] = c
            denom = float(
                np.sqrt(
                    max(seed_ceilings.get(bi, np.nan), 0) * max(seed_ceilings.get(bj, np.nan), 0)
                )
            )
            ceil_frac[i, j] = (c / denom) if denom > 0 else np.nan
    # Cross-behavior null band: stack the behavior U1s into an (H x k) matrix
    # and sign-flip null its top-share — the structural floor for "do these
    # share a direction?" (plan §14.4: small off-diagonals interpreted against
    # this band, not a measured zero-info adapter).
    M_behaviors = np.stack([behavior_u1[b] for b in behaviors], axis=1)
    null = sign_flip_null(M_behaviors, n_reps=n_reps, seed=seed) if k >= 2 else None
    return {
        "behaviors": behaviors,
        "raw_cosine_matrix": raw.tolist(),
        "ceiling_normalized_matrix": ceil_frac.tolist(),
        "seed_ceilings": {b: float(seed_ceilings.get(b, float("nan"))) for b in behaviors},
        "cross_behavior_null_p95": float(null["p95"]) if null else None,
        "cross_behavior_null_p99": float(null["p99"]) if null else None,
    }


def q2_verdict(q2: dict, *, coincide_frac: float = 0.85, distinct_frac: float = 0.5) -> dict:
    """Q2 family verdict (plan §6.2/§3).

    H-coincide if any off-diagonal ceiling-fraction >= coincide_frac.
    H-distinct if all off-diagonal ceiling-fractions < distinct_frac.
    H-family-cluster otherwise (the {sycophancy, em} block is flagged for the
    analyzer to weigh against the data-shape confound — plan §14.6).
    """
    behaviors = q2["behaviors"]
    cf = np.asarray(q2["ceiling_normalized_matrix"], dtype=np.float64)
    off = [
        (behaviors[i], behaviors[j], cf[i, j])
        for i in range(len(behaviors))
        for j in range(len(behaviors))
        if i < j
    ]
    finite = [(a, b, v) for a, b, v in off if np.isfinite(v)]
    any_coincide = any(v >= coincide_frac for _, _, v in finite)
    all_distinct = bool(finite) and all(v < distinct_frac for _, _, v in finite)
    if any_coincide:
        verdict = "coincide"
    elif all_distinct:
        verdict = "distinct"
    else:
        verdict = "family_cluster_or_intermediate"
    return {
        "verdict": verdict,
        "off_diagonal_ceiling_fractions": [
            {"pair": [a, b], "ceiling_fraction": float(v)} for a, b, v in off
        ],
        "coincide_frac": coincide_frac,
        "distinct_frac": distinct_frac,
        "near_boundary": [  # plan §14.3 — n=2 ceiling, near-boundary is noise-limited
            {"pair": [a, b], "ceiling_fraction": float(v)}
            for a, b, v in finite
            if abs(v - distinct_frac) < 0.1 or abs(v - coincide_frac) < 0.1
        ],
    }


def variance_decomposition(
    cell_reads: dict[tuple[str, str], np.ndarray],
) -> dict:
    """Variance decomposition of the (behavior x context x H) shift tensor.

    ``cell_reads`` = {(behavior, context): (H,) read vector} at one seed.
    Frobenius-energy fractions: shared "any-implant" component (the global mean
    direction's projection), behavior-specific (per-behavior mean minus global),
    and residual context-specific. DOSE-SENSITIVE (operates on un-normed reads,
    plan §6.1) — reported alongside the dose table, NOT the headline.
    """
    keys = sorted(cell_reads)
    X = np.stack(
        [np.asarray(cell_reads[k], dtype=np.float64).ravel() for k in keys], axis=0
    )  # (n, H)
    total_energy = float(np.sum(X**2))
    if total_energy == 0:
        return {"total_energy": 0.0, "shared_frac": 0.0, "behavior_frac": 0.0, "context_frac": 0.0}
    global_mean = X.mean(axis=0)  # (H,)
    shared = np.tile(global_mean, (X.shape[0], 1))
    shared_energy = float(np.sum(shared**2))
    # Per-behavior mean (minus global) = behavior-specific component.
    behaviors = [k[0] for k in keys]
    beh_component = np.zeros_like(X)
    for b in set(behaviors):
        idx = [i for i, kb in enumerate(behaviors) if kb == b]
        beh_mean = X[idx].mean(axis=0)
        beh_component[idx] = beh_mean - global_mean
    behavior_energy = float(np.sum(beh_component**2))
    context_component = X - shared - beh_component
    context_energy = float(np.sum(context_component**2))
    return {
        "total_energy": total_energy,
        "shared_frac": shared_energy / total_energy,
        "behavior_frac": behavior_energy / total_energy,
        "context_frac": context_energy / total_energy,
        "n_cells": len(keys),
        "behaviors": sorted(set(behaviors)),
    }


def construct_bridge_cosine(
    u1_neutral: np.ndarray, u1_canonical: np.ndarray, *, bar: float = 0.5
) -> dict:
    """Construct-validity bridge for fact/sycophancy (plan §6.1).

    cos(U1_neutral_panel, U1_canonical_surface). >= bar licenses a
    "behavior-direction" claim; below bar the analyzer downgrades to
    "panel-direction" for that behavior.
    """
    c = abs(cosine(u1_neutral, u1_canonical))
    return {
        "cos_neutral_vs_canonical": float(c),
        "bar": bar,
        "label": "behavior-direction" if c >= bar else "panel-direction",
        "licenses_behavior_claim": bool(c >= bar),
    }
