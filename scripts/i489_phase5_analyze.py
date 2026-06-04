# ruff: noqa: RUF001, RUF002, RUF003
"""Issue #489 Phase 5 — H1/H2/H3/H4 hypothesis battery on the 552 off-diagonal cells.

Plan v5 §3 + §6.2. Round-2 fixes: B1 (raise on empty delta_g), B4 (genuinely-
paired H3 bootstrap), B6 (paired |ρ_cos|−|ρ_JS| dissociation in H1), M-a (dual-
side graded partial on kind_distinctness), M-b (three-outcome H2 verdict tree +
identifiability gate), M-c (diagonal-adjusted SURVIVES statistic), M-d (don't
mutate args.fracs), M-g (guard NaN/constant columns).

Inputs (read from disk; each Phase persists its own artifact per CLAUDE.md):
  - ``eval_results/issue_489/phase1/cosine_per_layer.json``      (predictor: cosine)
  - ``eval_results/issue_489/phase1/js_rb_pairs.json``           (predictor: JS RB)
  - ``eval_results/issue_489/phase1/kind_distinctness.json``     (covariate, M-a)
  - ``eval_results/issue_489/phase1/scaffold_overlap.json``      (covariate)
  - ``eval_results/issue_489/phase4/per_cell/G_*.json``          (DV per cell)

Statistics:
  - H1: length-partial Spearman ρ(cos_distance_L21, ΔG) on off-diagonal cells with
        dyadic cluster-bootstrap CI (resampling source-context AND target-context
        independently, 5000 boots). PASS = ρ ≤ -0.30 AND CI excludes 0 AND
        |ρ_cos| − |ρ_JS| ≥ 0.10 with paired-bootstrap CI excluding 0.
  - H2: H1 survives source-OR-target drop on STRONG_KIND_SET (240 cells) AND
        dual-side graded partial on kind_distinctness; three-outcome verdict:
        SURVIVES / NULL_MOST_DISTINCT_ARTIFACT / UNIDENTIFIABLE.
  - H3: |ρ_ICL_within| − |ρ_SP_within| ≥ 0.15 with genuinely-paired bootstrap at
        the (frac × cid) shared-unit level (72 shared LoRA-snapshots at seed 42);
        smoke-gate-7 ESS auto-fallback to independent two-sample at raw-ρ gap 0.55.
  - H4(a): partial Spearman ρ controlling length + scaffold_overlap_score on the
        256 cross-type cells; PASS = ρ ≤ -0.20 with CI excluding 0.
  - H4(b): cosine + overlap-controlled residual test — regress ΔG on (cos, length,
        scaffold_overlap), test matched-pair residuals vs nearest-(cos,overlap)
        neighbor mismatched residuals via paired bootstrap; PASS = CI excludes 0
        positive.

Output: ``eval_results/issue_489/phase5/analysis.json`` with H1/H2/H3/H4 verdicts,
CIs, and the diagonal-adjusted SURVIVES check.

CLI:
    uv run python scripts/i489_phase5_analyze.py
    uv run python scripts/i489_phase5_analyze.py --bootstrap-n 5000
    uv run python scripts/i489_phase5_analyze.py --smoke   # tiny inputs OK
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

from explore_persona_space.experiments.i489_contexts import (
    MATCHED_PAIRS,
    STRONG_KIND_SET,
    UNION_BY_CID,
    UNION_CONTEXTS,
    ICLContext,
    is_cross_type,
)

logger = logging.getLogger("i489.phase5")

PHASE1_DIR = Path("eval_results/issue_489/phase1")
PHASE4_DIR = Path("eval_results/issue_489/phase4/per_cell")
OUT_DIR = Path("eval_results/issue_489/phase5")
HEADLINE_LAYER = 21
ESS_FLOOR = 24
RAW_RHO_FALLBACK_BAR = 0.55  # M4 independent two-sample bar
IDENTIFIABILITY_PEARSON_BAR = 0.85  # M-b: kind_distinctness vs cosine_distance
IDENTIFIABILITY_NON_STRONG_FLOOR = 5  # M-b: ≥5 non-strong-either-side cells in high band


def _git_commit_hash() -> str:
    import subprocess

    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _load_cells(
    fracs_request: list[float], seed: int, allow_smoke: bool
) -> tuple[dict[float, list[dict]], list[float]]:
    """Return ({frac: [cell dict, ...]}, ordered list of present fracs).

    Round-2 fix M-d: does NOT mutate the caller's ``fracs_request``. Returns the
    list of fracs actually present on disk as a separate value; callers iterate
    that, never the request list.
    """
    out: dict[float, list[dict]] = {f: [] for f in fracs_request}
    extra_fracs: set[float] = set()
    for p in PHASE4_DIR.glob("G_*.json"):
        try:
            payload = json.loads(p.read_text())
        except Exception:
            continue
        f = float(payload.get("frac", -1.0))
        if f not in out:
            extra_fracs.add(f)
            out.setdefault(f, [])
        if payload.get("seed", seed) != seed:
            continue
        if "delta_g" not in payload and not allow_smoke:
            continue
        out[f].append(payload)
    present = sorted({f for f, cells in out.items() if cells})
    return out, present


def _safe_array(xs) -> np.ndarray:
    return np.asarray(xs, dtype=float)


def _is_constant_or_nan(arr: np.ndarray) -> bool:
    """Round-2 fix M-g: skip covariate columns that would crash pearsonr/lstsq."""
    if arr.size == 0:
        return True
    if not np.all(np.isfinite(arr)):
        return True
    return np.nanstd(arr) < 1e-12


def _spearman_partial(x, y, z=None):
    """Spearman ρ(x, y | z). If z is None, returns plain Spearman ρ.

    z can be a 1-D array or a 2-D array (each column a covariate). Round-2 fix
    M-g: each covariate column is checked for finite + non-constant; degenerate
    columns are dropped. Returns NaN if x or y is constant.
    """
    x = _safe_array(x)
    y = _safe_array(y)
    if _is_constant_or_nan(x) or _is_constant_or_nan(y):
        return float("nan")
    if z is None or (hasattr(z, "size") and z.size == 0):
        rho, _ = sp_stats.spearmanr(x, y)
        return float(rho)
    z = _safe_array(z)
    if z.ndim == 1:
        z = z[:, None]
    keep_cols = [j for j in range(z.shape[1]) if not _is_constant_or_nan(z[:, j])]
    if not keep_cols:
        rho, _ = sp_stats.spearmanr(x, y)
        return float(rho)
    z = z[:, keep_cols]
    rx = sp_stats.rankdata(x)
    ry = sp_stats.rankdata(y)
    rz = np.apply_along_axis(sp_stats.rankdata, 0, z)
    A = np.column_stack([np.ones(len(rx)), rz])
    bx, *_ = np.linalg.lstsq(A, rx, rcond=None)
    by, *_ = np.linalg.lstsq(A, ry, rcond=None)
    ex = rx - A @ bx
    ey = ry - A @ by
    if _is_constant_or_nan(ex) or _is_constant_or_nan(ey):
        return float("nan")
    rho, _ = sp_stats.pearsonr(ex, ey)
    return float(rho)


def _dyadic_cluster_bootstrap_rho(
    cells: list[dict],
    cos_dist_fn,
    overlap_fn,
    length_fn,
    n_boots: int,
    rng: np.random.Generator,
    extra_partial: bool = False,
) -> tuple[float, tuple[float, float]]:
    """Bootstrap ρ(cos_distance, delta_g | length [+ overlap]) over off-diagonal cells.

    Resample sources AND targets independently. Each boot: pick a random subset
    of source cids + target cids (with replacement at the cluster level), then
    take all cells whose (i, j) sit inside both pools.
    """
    if not cells:
        return float("nan"), (float("nan"), float("nan"))
    all_sources = sorted({c["T_i"] for c in cells})
    all_targets = sorted({c["T_j"] for c in cells})
    cell_index = {(c["T_i"], c["T_j"]): c for c in cells}

    def _build_panel(sources, targets):
        x: list[float] = []
        y: list[float] = []
        z: list[list[float]] = []
        for si in sources:
            for tj in targets:
                if si == tj:
                    continue
                c = cell_index.get((si, tj))
                if c is None:
                    continue
                x.append(cos_dist_fn(si, tj))
                y.append(c["delta_g"])
                row = [length_fn(c)]
                if extra_partial:
                    row.append(overlap_fn(si, tj))
                z.append(row)
        return np.array(x), np.array(y), np.array(z)

    x0, y0, z0 = _build_panel(all_sources, all_targets)
    rho0 = _spearman_partial(x0, y0, z0)
    boot_rhos: list[float] = []
    n_s, n_t = len(all_sources), len(all_targets)
    for _ in range(n_boots):
        idx_s = rng.integers(0, n_s, n_s)
        idx_t = rng.integers(0, n_t, n_t)
        srcs = [all_sources[i] for i in idx_s]
        tgts = [all_targets[i] for i in idx_t]
        xb, yb, zb = _build_panel(srcs, tgts)
        if len(xb) < 5:
            continue
        rho_b = _spearman_partial(xb, yb, zb)
        if rho_b == rho_b:
            boot_rhos.append(rho_b)
    if not boot_rhos:
        return rho0, (float("nan"), float("nan"))
    lo, hi = np.percentile(boot_rhos, [2.5, 97.5])
    return rho0, (float(lo), float(hi))


def _paired_diff_bootstrap_rho(
    cells: list[dict],
    fn_a,
    fn_b,
    length_fn,
    n_boots: int,
    rng: np.random.Generator,
) -> tuple[float, tuple[float, float]]:
    """Round-2 fix B6: paired |ρ_a| − |ρ_b| bootstrap on the same cells.

    Same dyadic cluster bootstrap as ``_dyadic_cluster_bootstrap_rho`` but
    inside each draw we recompute ρ for BOTH predictors on the resampled cells
    and form (|ρ_a| − |ρ_b|). Sources and targets are resampled jointly so
    cov(ρ_a, ρ_b) > 0 (recipe-level noise cancels in the difference).
    """
    if not cells:
        return float("nan"), (float("nan"), float("nan"))
    all_sources = sorted({c["T_i"] for c in cells})
    all_targets = sorted({c["T_j"] for c in cells})
    cell_index = {(c["T_i"], c["T_j"]): c for c in cells}

    def _build_panel(sources, targets, fn):
        x: list[float] = []
        y: list[float] = []
        z: list[list[float]] = []
        for si in sources:
            for tj in targets:
                if si == tj:
                    continue
                c = cell_index.get((si, tj))
                if c is None:
                    continue
                v = fn(si, tj)
                if v != v:  # NaN guard for missing predictor pair
                    continue
                x.append(v)
                y.append(c["delta_g"])
                z.append([length_fn(c)])
        return np.array(x), np.array(y), np.array(z)

    xa0, ya0, za0 = _build_panel(all_sources, all_targets, fn_a)
    xb0, yb0, zb0 = _build_panel(all_sources, all_targets, fn_b)
    rho_a0 = _spearman_partial(xa0, ya0, za0) if len(xa0) >= 5 else float("nan")
    rho_b0 = _spearman_partial(xb0, yb0, zb0) if len(xb0) >= 5 else float("nan")
    diff0 = (abs(rho_a0) - abs(rho_b0)) if (rho_a0 == rho_a0 and rho_b0 == rho_b0) else float("nan")
    boots: list[float] = []
    n_s, n_t = len(all_sources), len(all_targets)
    for _ in range(n_boots):
        idx_s = rng.integers(0, n_s, n_s)
        idx_t = rng.integers(0, n_t, n_t)
        srcs = [all_sources[i] for i in idx_s]
        tgts = [all_targets[i] for i in idx_t]
        xa, ya, za = _build_panel(srcs, tgts, fn_a)
        xb, yb, zb = _build_panel(srcs, tgts, fn_b)
        if len(xa) < 5 or len(xb) < 5:
            continue
        ra = _spearman_partial(xa, ya, za)
        rb = _spearman_partial(xb, yb, zb)
        if ra != ra or rb != rb:
            continue
        boots.append(abs(ra) - abs(rb))
    if not boots:
        return diff0, (float("nan"), float("nan"))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return diff0, (float(lo), float(hi))


def _h3_paired_bootstrap(  # noqa: C901 - paired + independent-fallback branches are deliberate (B4)
    icl_cells: list[dict],
    sp_cells: list[dict],
    cos_dist_fn,
    length_fn,
    n_boots: int,
    rng: np.random.Generator,
) -> dict:
    """Round-2 fix B4: genuinely-paired H3 bootstrap on (cid_i × frac) units.

    Each draw resamples the shared LoRA-snapshot units (cid_i, frac) WITH
    REPLACEMENT, then recomputes BOTH rho_icl and rho_sp on the cells whose
    source is in the resampled unit set. ICL cells use the resampled unit's
    cells against ICL targets; SP cells use the same unit's cells against SP
    targets. Forms CI on (|ρ_icl| − |ρ_sp|).

    Auto-fallback: if the effective sample size at the LoRA-snapshot level
    is < ``ESS_FLOOR`` (24), switch to independent two-sample with raw-ρ gap
    bar 0.55 per M4 documented alternative.

    Returns a dict with rho_icl, rho_sp, ci_icl, ci_sp, abs_diff, ci_diff,
    mechanic ('paired' or 'independent'), ess_lora_snapshots.
    """
    # Build the LoRA-snapshot unit set: (cid_i, frac) pairs that appear in BOTH
    # arms. Each cid_i contributes cells to the ICL-within sub-grid (when target
    # is ICL) AND to the SP-within sub-grid (when target is SP), so the SHARED
    # unit is just (cid_i, frac).
    icl_units = {(c["T_i"], c["frac"]) for c in icl_cells}
    sp_units = {(c["T_i"], c["frac"]) for c in sp_cells}
    shared_units = sorted(icl_units & sp_units)
    ess = len(shared_units)
    if ess == 0:
        return {
            "rho_icl": float("nan"),
            "rho_sp": float("nan"),
            "ci_icl": (float("nan"), float("nan")),
            "ci_sp": (float("nan"), float("nan")),
            "abs_diff": float("nan"),
            "ci_diff": (float("nan"), float("nan")),
            "mechanic": "none",
            "ess_lora_snapshots": 0,
            "note": "no shared (cid_i, frac) LoRA-snapshot units between arms",
        }

    icl_by_unit: dict[tuple[str, float], list[dict]] = {u: [] for u in shared_units}
    sp_by_unit: dict[tuple[str, float], list[dict]] = {u: [] for u in shared_units}
    for c in icl_cells:
        u = (c["T_i"], c["frac"])
        if u in icl_by_unit:
            icl_by_unit[u].append(c)
    for c in sp_cells:
        u = (c["T_i"], c["frac"])
        if u in sp_by_unit:
            sp_by_unit[u].append(c)

    def _rho_for_cells(cells: list[dict]) -> float:
        if len(cells) < 5:
            return float("nan")
        x = [cos_dist_fn(c["T_i"], c["T_j"]) for c in cells]
        y = [c["delta_g"] for c in cells]
        z = [[length_fn(c)] for c in cells]
        return _spearman_partial(x, y, np.array(z))

    rho_icl0 = _rho_for_cells(icl_cells)
    rho_sp0 = _rho_for_cells(sp_cells)

    def _bootstrap_ci(values: list[float]) -> tuple[float, float]:
        if not values:
            return (float("nan"), float("nan"))
        lo, hi = np.percentile(values, [2.5, 97.5])
        return (float(lo), float(hi))

    if ess < ESS_FLOOR:
        # M4 documented fallback: independent two-sample on full cell sets.
        boots_icl: list[float] = []
        boots_sp: list[float] = []
        boots_diff: list[float] = []
        # Source-cluster resampling independently on each arm.
        icl_sources = sorted({c["T_i"] for c in icl_cells})
        sp_sources = sorted({c["T_i"] for c in sp_cells})
        for _ in range(n_boots):
            isrc = rng.choice(icl_sources, size=len(icl_sources), replace=True)
            ssrc = rng.choice(sp_sources, size=len(sp_sources), replace=True)
            icl_sub = [c for c in icl_cells if c["T_i"] in set(isrc)]
            sp_sub = [c for c in sp_cells if c["T_i"] in set(ssrc)]
            ra = _rho_for_cells(icl_sub)
            rb = _rho_for_cells(sp_sub)
            if ra != ra or rb != rb:
                continue
            boots_icl.append(ra)
            boots_sp.append(rb)
            boots_diff.append(abs(ra) - abs(rb))
        return {
            "rho_icl": rho_icl0,
            "rho_sp": rho_sp0,
            "ci_icl": _bootstrap_ci(boots_icl),
            "ci_sp": _bootstrap_ci(boots_sp),
            "abs_diff": (
                abs(rho_icl0) - abs(rho_sp0)
                if (rho_icl0 == rho_icl0 and rho_sp0 == rho_sp0)
                else float("nan")
            ),
            "ci_diff": _bootstrap_ci(boots_diff),
            "mechanic": "independent_fallback",
            "ess_lora_snapshots": ess,
            "note": (
                f"ESS={ess} < {ESS_FLOOR} floor — fell back to M4 independent "
                f"two-sample (raw-ρ bar {RAW_RHO_FALLBACK_BAR})"
            ),
            "raw_rho_fallback_bar": RAW_RHO_FALLBACK_BAR,
        }

    # PAIRED MECHANIC: resample shared (cid_i, frac) units WITH REPLACEMENT.
    boots_icl: list[float] = []
    boots_sp: list[float] = []
    boots_diff: list[float] = []
    for _ in range(n_boots):
        idx = rng.integers(0, ess, ess)
        sampled_units = [shared_units[k] for k in idx]
        icl_sub: list[dict] = []
        sp_sub: list[dict] = []
        for u in sampled_units:
            icl_sub.extend(icl_by_unit[u])
            sp_sub.extend(sp_by_unit[u])
        ra = _rho_for_cells(icl_sub)
        rb = _rho_for_cells(sp_sub)
        if ra != ra or rb != rb:
            continue
        boots_icl.append(ra)
        boots_sp.append(rb)
        boots_diff.append(abs(ra) - abs(rb))
    return {
        "rho_icl": rho_icl0,
        "rho_sp": rho_sp0,
        "ci_icl": _bootstrap_ci(boots_icl),
        "ci_sp": _bootstrap_ci(boots_sp),
        "abs_diff": (
            abs(rho_icl0) - abs(rho_sp0)
            if (rho_icl0 == rho_icl0 and rho_sp0 == rho_sp0)
            else float("nan")
        ),
        "ci_diff": _bootstrap_ci(boots_diff),
        "mechanic": "paired",
        "ess_lora_snapshots": ess,
    }


def _cell_off_diagonal(cells: list[dict]) -> list[dict]:
    return [c for c in cells if c["T_i"] != c["T_j"]]


def _length_for(cell: dict) -> float:
    """log(mean prompt_len + mean R_len + 1). Falls back to a constant if
    Phase 4 didn't persist per-q lengths (smoke / legacy payloads)."""
    L = cell.get("prompt_lens_per_q")
    R = cell.get("R_lens_per_q_sample") or cell.get("R_lens_per_q")
    if isinstance(L, list) and isinstance(R, list) and L and R:
        L_mean = float(np.mean(L))
        # R can be list-of-list (per-q per-k) OR flat list.
        flat_R: list[float] = []
        for entry in R:
            if isinstance(entry, list):
                flat_R.extend(entry)
            else:
                flat_R.append(float(entry))
        R_mean = float(np.mean(flat_R)) if flat_R else 0.0
        return float(np.log(L_mean + R_mean + 1))
    return float(np.log(cell.get("n_q", 20) * 200 + 1))  # fallback


def _h2_three_outcome(
    h2_pass_after_drop: bool,
    h2_cells: list[dict],
    cos_dist_fn,
    length_fn,
    kind_distinctness: dict[str, float] | None,
) -> dict:
    """Round-2 fix M-b: three-outcome H2 verdict tree.

    Outcomes:
      SURVIVES — H1 survives the strong-kind drop AND the identifiability gate
        passes (kind_distinctness vs cosine_distance Pearson ≤ 0.85 AND ≥5
        non-strong-on-either-side cells in the high cosine_distance band).
      UNIDENTIFIABLE — strong-kind cells are heavily collinear with cosine
        distance (Pearson > 0.85) OR the high-band has < 5 non-strong cells;
        the body must narrate "can't separate distinctness from distance."
      NULL_MOST_DISTINCT_ARTIFACT — H1 does NOT survive the strong-kind drop
        AND identifiability passes (we COULD have rejected, but didn't).
    """
    if not h2_cells or kind_distinctness is None:
        return {
            "verdict": "INSUFFICIENT_DATA",
            "identifiability_pearson": float("nan"),
            "identifiability_pass": False,
            "non_strong_high_band_n": 0,
        }
    # Identifiability statistic: per-cell max(kind_dist_i, kind_dist_j) vs
    # cosine_distance over the FULL pre-drop panel (we want the structural
    # relationship between the covariate and the predictor).
    max_kd: list[float] = []
    cos_d: list[float] = []
    for c in h2_cells:
        kd_i = kind_distinctness.get(c["T_i"], float("nan"))
        kd_j = kind_distinctness.get(c["T_j"], float("nan"))
        if kd_i != kd_i or kd_j != kd_j:
            continue
        v = max(kd_i, kd_j)
        d = cos_dist_fn(c["T_i"], c["T_j"])
        if d != d:
            continue
        max_kd.append(v)
        cos_d.append(d)
    if (
        len(max_kd) < 5
        or _is_constant_or_nan(np.array(max_kd))
        or _is_constant_or_nan(np.array(cos_d))
    ):
        pearson_r = float("nan")
    else:
        pearson_r, _ = sp_stats.pearsonr(max_kd, cos_d)
        pearson_r = float(pearson_r)
    # High-band = upper quartile of cosine_distance over all cells; count
    # non-strong-on-either-side cells in that band.
    hi_thresh = float(np.percentile(cos_d, 75)) if cos_d else float("nan")
    non_strong_hi = 0
    for c in h2_cells:
        d = cos_dist_fn(c["T_i"], c["T_j"])
        if d != d or d < hi_thresh:
            continue
        if c["T_i"] not in STRONG_KIND_SET and c["T_j"] not in STRONG_KIND_SET:
            non_strong_hi += 1
    identifiable = (
        pearson_r == pearson_r
        and abs(pearson_r) <= IDENTIFIABILITY_PEARSON_BAR
        and non_strong_hi >= IDENTIFIABILITY_NON_STRONG_FLOOR
    )
    if not identifiable:
        verdict = "UNIDENTIFIABLE"
    elif h2_pass_after_drop:
        verdict = "SURVIVES"
    else:
        verdict = "NULL_MOST_DISTINCT_ARTIFACT"
    return {
        "verdict": verdict,
        "identifiability_pearson_max_kd_vs_cos": pearson_r,
        "identifiability_pearson_bar": IDENTIFIABILITY_PEARSON_BAR,
        "identifiability_pass": bool(identifiable),
        "non_strong_high_band_n": non_strong_hi,
        "non_strong_high_band_floor": IDENTIFIABILITY_NON_STRONG_FLOOR,
        "high_band_threshold_cos_dist": hi_thresh,
    }


def _h2_diagonal_adjusted(
    off_cells: list[dict],
    diag_by_cid: dict[str, dict],
    cos_dist_fn,
    length_fn,
) -> dict:
    """Round-2 fix M-c: SURVIVES-only diagonal-adjusted statistic.

    Two computations:
      1. Partial Spearman ρ(cos_dist, delta_g | length, emission_ii_source) —
         control for the source-diagonal emission rate so the test isolates
         transfer beyond raw implant strength.
      2. Normalized ΔG_ij / max(epsilon, emission_ii_source) sensitivity —
         report the ρ on the normalized variant as a robustness anchor.
    """
    if not off_cells:
        return {"available": False, "reason": "no off-diagonal cells"}
    x: list[float] = []
    y: list[float] = []
    y_norm: list[float] = []
    z_len: list[float] = []
    z_ii: list[float] = []
    for c in off_cells:
        d = cos_dist_fn(c["T_i"], c["T_j"])
        if d != d:
            continue
        diag = diag_by_cid.get(c["T_i"])
        if diag is None or "delta_g" not in diag:
            continue
        ii = float(diag["delta_g"])
        x.append(d)
        y.append(float(c["delta_g"]))
        y_norm.append(float(c["delta_g"]) / max(0.5, abs(ii)))
        z_len.append(length_fn(c))
        z_ii.append(ii)
    if len(x) < 10:
        return {"available": False, "reason": f"only {len(x)} cells with source-diagonal pair"}
    z = np.column_stack([z_len, z_ii])
    rho_partial = _spearman_partial(x, y, z)
    rho_norm = _spearman_partial(x, y_norm, np.array(z_len))
    return {
        "available": True,
        "n_cells": len(x),
        "rho_partial_on_emission_ii": rho_partial,
        "rho_normalized_offdiag_over_emission_ii": rho_norm,
    }


def main(argv: list[str] | None = None) -> int:  # noqa: C901 - H1/H2/H3/H4 battery
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fracs", nargs="+", type=float, default=[0.25, 0.50, 1.00])
    ap.add_argument("--bootstrap-n", type=int, default=5000)
    ap.add_argument("--bootstrap-rng-seed", type=int, default=123)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run with whatever cells are on disk (no minimum-cell assert). "
            "Used by the local CPU smoke run."
        ),
    )
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cos_path = PHASE1_DIR / "cosine_per_layer.json"
    js_path = PHASE1_DIR / "js_rb_pairs.json"
    overlap_path = PHASE1_DIR / "scaffold_overlap.json"
    kind_path = PHASE1_DIR / "kind_distinctness.json"
    if not cos_path.exists():
        raise FileNotFoundError(f"missing {cos_path}; run i489_phase1_predictors.py first")
    cos_payload = json.loads(cos_path.read_text())
    cos_sim = cos_payload["cos_sim_per_layer"][str(HEADLINE_LAYER)]

    def cos_dist(ci, cj):
        try:
            return 1.0 - cos_sim[ci][cj]
        except KeyError:
            return float("nan")

    js_pairs = None
    if js_path.exists():
        js_pairs = json.loads(js_path.read_text())["js_rb_pairs"]
    overlap = None
    if overlap_path.exists():
        overlap = json.loads(overlap_path.read_text())["scaffold_overlap_per_cell"]
    kind_distinctness: dict[str, float] | None = None
    if kind_path.exists():
        kind_distinctness = json.loads(kind_path.read_text())["kind_distinctness_score"]

    def overlap_score(ci, cj):
        if overlap is None:
            return float("nan")
        try:
            return float(overlap[ci][cj]["scaffold_overlap_score"])
        except KeyError:
            return float("nan")

    def js_dist(ci, cj):
        if js_pairs is None:
            return float("nan")
        try:
            return float(js_pairs[ci][cj])
        except KeyError:
            return float("nan")

    # Round-2 fix M-d: don't mutate args.fracs; iterate a snapshot copy.
    requested_fracs = list(args.fracs)
    cells_by_frac, present_fracs = _load_cells(requested_fracs, args.seed, allow_smoke=args.smoke)
    fracs_to_analyze = present_fracs if not args.smoke else (present_fracs or requested_fracs)
    rng = np.random.default_rng(args.bootstrap_rng_seed)

    # Round-2 fix B1: count cells carrying delta_g across ALL fracs. If zero,
    # RAISE — never silently ship an empty analysis as a PASS.
    total_off_cells_with_dg = 0
    for frac in fracs_to_analyze:
        for c in cells_by_frac.get(frac, []):
            if c["T_i"] != c["T_j"] and "delta_g" in c:
                total_off_cells_with_dg += 1
    if total_off_cells_with_dg == 0 and not args.smoke:
        raise RuntimeError(
            f"Phase 5: zero off-diagonal cells across {len(fracs_to_analyze)} frac(s) carry "
            f"'delta_g'. Phase 4 either crashed before PASS B or wrote the legacy "
            f"'phase4b_pending' payload. Refusing to silently produce an empty H1/H2/H3/H4 "
            f"battery — fix Phase 4 (see scripts/i489_phase4_eval_onpolicy.py) and re-run."
        )
    if total_off_cells_with_dg == 0 and args.smoke:
        logger.warning(
            "Smoke: zero cells with delta_g — analysis is a structural wiring check only."
        )

    h1_per_frac: dict[float, dict] = {}
    h2_per_frac: dict[float, dict] = {}
    h3_per_frac: dict[float, dict] = {}
    h4a_per_frac: dict[float, dict] = {}
    h4b_per_frac: dict[float, dict] = {}

    for frac in fracs_to_analyze:
        cells = cells_by_frac.get(frac, [])
        off = _cell_off_diagonal(cells)
        diag_by_cid = {c["T_i"]: c for c in cells if c["T_i"] == c["T_j"]}
        logger.info("Phase 5 frac=%.2f: %d off-diagonal cells loaded", frac, len(off))
        if not off:
            continue

        # --- H1: full panel cosine-vs-JS dissociation ----------------------
        rho_cos, ci_cos = _dyadic_cluster_bootstrap_rho(
            off, cos_dist, overlap_score, _length_for, args.bootstrap_n, rng
        )
        rho_js, ci_js = _dyadic_cluster_bootstrap_rho(
            off, js_dist, overlap_score, _length_for, args.bootstrap_n, rng
        )
        # Round-2 fix B6: paired |ρ_cos| − |ρ_JS| CI on the same cells.
        diff_paired, ci_diff_paired = _paired_diff_bootstrap_rho(
            off, cos_dist, js_dist, _length_for, args.bootstrap_n, rng
        )
        h1_cos_signed_pass = (
            rho_cos == rho_cos and rho_cos <= -0.30 and not (ci_cos[0] <= 0 <= ci_cos[1])
        )
        # B6: dissociation requires paired |Δ|≥0.10 AND paired CI excludes 0.
        h1_dissociation_pass = (
            diff_paired == diff_paired
            and diff_paired >= 0.10
            and not (ci_diff_paired[0] <= 0 <= ci_diff_paired[1])
        )
        h1_per_frac[frac] = {
            "rho_cos": rho_cos,
            "rho_cos_ci": ci_cos,
            "rho_js": rho_js,
            "rho_js_ci": ci_js,
            "abs_diff_cos_minus_js_paired": diff_paired,
            "ci_diff_paired": ci_diff_paired,
            "cos_signed_pass": bool(h1_cos_signed_pass),
            "cos_vs_js_dissociation_pass": bool(h1_dissociation_pass),
            "pass": bool(h1_cos_signed_pass and h1_dissociation_pass),
        }

        # --- H2: source-OR-target drop on STRONG_KIND_SET ------------------
        off_h2 = [
            c for c in off if c["T_i"] not in STRONG_KIND_SET and c["T_j"] not in STRONG_KIND_SET
        ]
        rho_cos_h2, ci_cos_h2 = _dyadic_cluster_bootstrap_rho(
            off_h2, cos_dist, overlap_score, _length_for, args.bootstrap_n, rng
        )
        h2_after_drop_pass = (
            rho_cos_h2 == rho_cos_h2 and rho_cos_h2 < 0 and not (ci_cos_h2[0] <= 0 <= ci_cos_h2[1])
        )
        # M-a: dual-side graded partial on kind_distinctness.
        dual_graded = float("nan")
        if kind_distinctness is not None and off_h2:
            x = [cos_dist(c["T_i"], c["T_j"]) for c in off_h2]
            y = [c["delta_g"] for c in off_h2]
            z = []
            for c in off_h2:
                kd_i = kind_distinctness.get(c["T_i"], float("nan"))
                kd_j = kind_distinctness.get(c["T_j"], float("nan"))
                z.append([_length_for(c), kd_i, kd_j, max(kd_i, kd_j)])
            dual_graded = _spearman_partial(x, y, np.array(z))
        # M-b: three-outcome verdict tree + identifiability gate.
        three_outcome = _h2_three_outcome(
            h2_after_drop_pass, off, cos_dist, _length_for, kind_distinctness
        )
        # M-c: diagonal-adjusted statistic (only meaningful for SURVIVES).
        diag_adjusted = _h2_diagonal_adjusted(off, diag_by_cid, cos_dist, _length_for)
        h2_per_frac[frac] = {
            "n_cells_after_strong_drop": len(off_h2),
            "rho_cos": rho_cos_h2,
            "rho_cos_ci": ci_cos_h2,
            "h1_survives_strong_kind_drop": bool(h2_after_drop_pass),
            "rho_cos_dual_side_kind_distinctness_partial": dual_graded,
            **three_outcome,
            "diagonal_adjusted": diag_adjusted,
        }

        # --- H3: within-arm ICL vs SP paired-bootstrap --------------------
        off_icl = [
            c
            for c in off
            if isinstance(UNION_BY_CID[c["T_i"]], ICLContext)
            and isinstance(UNION_BY_CID[c["T_j"]], ICLContext)
        ]
        off_sp = [
            c
            for c in off
            if not isinstance(UNION_BY_CID[c["T_i"]], ICLContext)
            and not isinstance(UNION_BY_CID[c["T_j"]], ICLContext)
        ]
        h3_paired = _h3_paired_bootstrap(
            off_icl, off_sp, cos_dist, _length_for, args.bootstrap_n, rng
        )
        # PASS: paired CI on |ρ_ICL| − |ρ_SP| excludes 0 positive, |Δ| ≥ 0.15,
        # AND |ρ_ICL| ≥ 0.30 (signed).
        diff = h3_paired["abs_diff"]
        ci_diff_h3 = h3_paired["ci_diff"]
        rho_icl_signed = h3_paired["rho_icl"]
        if h3_paired["mechanic"] == "independent_fallback":
            # Independent fallback uses raw-ρ gap bar 0.55.
            h3_pass = (
                diff == diff
                and diff >= RAW_RHO_FALLBACK_BAR
                and not (ci_diff_h3[0] <= 0 <= ci_diff_h3[1])
                and rho_icl_signed == rho_icl_signed
                and abs(rho_icl_signed) >= 0.30
            )
        else:
            h3_pass = (
                diff == diff
                and diff >= 0.15
                and not (ci_diff_h3[0] <= 0 <= ci_diff_h3[1])
                and rho_icl_signed == rho_icl_signed
                and abs(rho_icl_signed) >= 0.30
            )
        h3_per_frac[frac] = {
            **h3_paired,
            "n_icl_cells": len(off_icl),
            "n_sp_cells": len(off_sp),
            "pass": bool(h3_pass),
            "pass_bar": 0.15
            if h3_paired["mechanic"] != "independent_fallback"
            else RAW_RHO_FALLBACK_BAR,
        }

        # --- H4(a): cross-type dual-partial -------------------------------
        off_cross = [c for c in off if is_cross_type(c["T_i"], c["T_j"])]
        rho_cross, ci_cross = _dyadic_cluster_bootstrap_rho(
            off_cross,
            cos_dist,
            overlap_score,
            _length_for,
            args.bootstrap_n,
            rng,
            extra_partial=True,
        )
        h4a_per_frac[frac] = {
            "n_cells": len(off_cross),
            "rho_dual_partial": rho_cross,
            "ci": ci_cross,
            "pass": bool(
                rho_cross == rho_cross
                and rho_cross <= -0.20
                and not (ci_cross[0] <= 0 <= ci_cross[1])
            ),
        }

        # --- H4(b): cosine + overlap-controlled matched-pair residual ----
        if len(off_cross) >= max(len(MATCHED_PAIRS) * 2, 5):
            X_cos = np.array([cos_dist(c["T_i"], c["T_j"]) for c in off_cross])
            X_len = np.array([_length_for(c) for c in off_cross])
            X_ov = np.array([overlap_score(c["T_i"], c["T_j"]) for c in off_cross])
            Y = np.array([c["delta_g"] for c in off_cross])
            # M-g: drop columns that would crash the regression.
            cols = [
                (np.ones(len(X_cos)), "intercept"),
                (sp_stats.rankdata(X_cos), "cos"),
                (sp_stats.rankdata(X_len), "len"),
                (sp_stats.rankdata(X_ov), "overlap"),
            ]
            kept = [
                (v, name) for v, name in cols if name == "intercept" or not _is_constant_or_nan(v)
            ]
            rX = np.column_stack([v for v, _ in kept])
            rY = sp_stats.rankdata(Y)
            if _is_constant_or_nan(rY) or rX.shape[1] < 2:
                h4b_per_frac[frac] = {
                    "n_matched": 0,
                    "pass": False,
                    "note": "degenerate covariate panel — H4(b) regression skipped",
                }
            else:
                beta, *_ = np.linalg.lstsq(rX, rY, rcond=None)
                resid = rY - rX @ beta
                matched_idx: list[int] = []
                for i, c in enumerate(off_cross):
                    pair = (c["T_i"], c["T_j"])
                    rev = (c["T_j"], c["T_i"])
                    if pair in MATCHED_PAIRS or rev in MATCHED_PAIRS:
                        matched_idx.append(i)
                if not matched_idx:
                    h4b_per_frac[frac] = {
                        "n_matched": 0,
                        "pass": False,
                        "note": "no matched-pair cells found in off_cross",
                    }
                else:
                    # Nearest-(cos, overlap) neighbor in z-space.
                    cos_std = X_cos.std() + 1e-12
                    ov_std = X_ov.std() + 1e-12
                    cos_norm = (X_cos - X_cos.mean()) / cos_std
                    ov_norm = (X_ov - X_ov.mean()) / ov_std
                    mismatched_idx = [i for i in range(len(off_cross)) if i not in matched_idx]
                    diffs: list[float] = []
                    for mi in matched_idx:
                        best = None
                        best_d = float("inf")
                        for nj in mismatched_idx:
                            d = (cos_norm[mi] - cos_norm[nj]) ** 2 + (
                                ov_norm[mi] - ov_norm[nj]
                            ) ** 2
                            if d < best_d:
                                best_d = d
                                best = nj
                        if best is not None:
                            diffs.append(resid[mi] - resid[best])
                    if not diffs:
                        h4b_per_frac[frac] = {"n_matched": len(matched_idx), "pass": False}
                    else:
                        boots = []
                        for _ in range(args.bootstrap_n):
                            samp = rng.choice(diffs, size=len(diffs), replace=True)
                            boots.append(float(np.median(samp)))
                        lo, hi = np.percentile(boots, [2.5, 97.5])
                        med = float(np.median(diffs))
                        h4b_per_frac[frac] = {
                            "n_matched": len(matched_idx),
                            "median_resid_diff": med,
                            "ci": [float(lo), float(hi)],
                            "pass": bool(lo > 0),
                        }
        else:
            h4b_per_frac[frac] = {
                "n_matched": 0,
                "pass": False,
                "note": "insufficient cross-type cells",
            }

    payload = {
        "schema_version": "i489_phase5_v2",
        "git_commit": _git_commit_hash(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "seed": args.seed,
        "fracs_requested": requested_fracs,
        "fracs_present_on_disk": present_fracs,
        "bootstrap_n": args.bootstrap_n,
        "headline_layer": HEADLINE_LAYER,
        "single_seed_scope_caveat": ("v5: seed=42 only; no across-seed variance estimate."),
        "h1": h1_per_frac,
        "h2": h2_per_frac,
        "h3": h3_per_frac,
        "h4a": h4a_per_frac,
        "h4b": h4b_per_frac,
        "n_contexts_in_union": len(UNION_CONTEXTS),
        "total_off_cells_with_delta_g": total_off_cells_with_dg,
        "smoke": bool(args.smoke),
    }
    out_path = OUT_DIR / "analysis.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("Phase 5 wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
