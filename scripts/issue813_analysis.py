#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (Δ, M⁺, M0, →, ρ, ×, Ŵ, ‖·‖, ※) in scientific docstrings + log messages.
"""Issue #813 — DVs: Δ/floor + chain-ρ + substrate-swap null + pairwise-diff CIs.

Runs OFF-POD (VM CPU) — closed-form ridge + sampling stats over the reduced
per-(behavior, substrate) summaries the extraction wave wrote. NOT an iterative
fit (ridge-only, ``include_mlp=False``), so it belongs on the CPU/VM.

Per (behavior, substrate) at the FROZEN headline layer 14 (#651/#658; applied
IDENTICALLY to the observed statistic AND the substrate-swap null — the
selection-symmetric frozen-position route, `.claude/rules/selection-symmetric-nulls.md`):

- em / fact / sycophancy → ``issue722_fit_M.fit_cell(behavior, 14, cells, rb_main,
  rb_fact, include_mlp=False)`` (the RIDGE-only headline: ``Delta_med`` /
  ``floor_combined`` / ``Delta_over_floor_sd`` / chain-ρ / support_distance /
  n_with_E). fact/syco/em have an r_B; marker does NOT (fit_cell KeyErrors on it).
- marker → ``issue667_marker_mapchange.fit_marker_layer(14, cells, wu_marker,
  with_chain=)`` (read-1 unprojected ‖ΔM‖/floor + read-2 W_U[※]-projected
  |ΔM·Ŵ_U[※]|/floor + ``wu_frac_in_subspace`` — read-2 uninformative when < 0.1).

Then, per behavior:
- **Substrate-swap null (matched-n).** Within EACH substrate, resample the
  substrate's questions and re-split them into TWO pseudo-substrates of the SAME n
  per pseudo-arm (question-average per context → a pseudo-map pair), compute the
  SAME Δ/floor DV for each pseudo-substrate, take |Δ/floor(A) − Δ/floor(B)|. The
  null holds ΔM fixed (same adapter) and varies ONLY the question sample — its
  95th percentile is the behavior-specific threshold X a REAL substrate difference
  must clear (plan §3). Matched-n keeps em's low power conservative (not inflated).
- **Pairwise substrate-difference CIs (D1).** Family-clustered bootstrap CI on the
  SIGNED Δ/floor difference Δ/floor(A) − Δ/floor(B) for the three substrate pairs.
  Both substrates fit over the SAME shared 50-context battery (plan §4.2), so ONE
  family-clustered context resample refits BOTH arms per draw and the paired
  difference is recomputed each resample — the same refit machinery the observed
  read + the substrate-swap null use. The CI EXCLUDES 0 iff its whole interval is
  on one side of 0; that is the SECOND conjunct of the plan §3 verdict.
- **Verdict (D1 CONJUNCTION).** "substrate matters" (H0) iff BOTH conjuncts fire:
  (i) the max-vs-min Δ/floor difference exceeds the substrate-swap null p95 AND
  (ii) a DRIVING-pair pairwise CI (the pair whose difference IS max_diff) excludes
  0. "substrate-agnostic" (H1) iff BOTH fail (max within band AND all CIs include
  0). AMBIGUOUS (None) iff exactly one conjunct fires or a conjunct is undecidable.
  The reducer ``decide_substrate_matters`` is a pure function (unit-testable).

Reads the frozen headline-layer per-question rows (``per_question_L14.npz``) for
the null; reads the 28-layer reduced summary for the observed fit_cell/marker read
AND the pairwise-CI refit (both share the shared battery contexts per substrate).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue658_fit_predictors as fit658  # noqa: E402
import issue667_marker_mapchange as marker_mc  # noqa: E402
import issue722_fit_M as fitM  # noqa: E402
from issue722_bootstrap import floor_sd, make_refit_pair  # noqa: E402

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue813.analysis")

DATA_REPO = "superkaiba1/explore-persona-space-data"
EXPERIMENT_NAME = "issue813_mapchange_substrate"
BEHAVIORS = ("em", "fact", "sycophancy", "marker")
SUBSTRATES = ("generic", "elicit", "mix")
HEADLINE_LAYER = 14  # frozen (#651/#658); observed + null read at the SAME layer
HIDDEN = 3584
N_LAYERS = 28
TARGET_DIM = 64  # top-64 v0 PCs (NEVER 48)
N_NULL_RESAMPLES = 1000
NULL_SEED = 42
# Refit-pair count for the PER-PSEUDO-ARM floor inside the substrate-swap null (B2).
# The observed read uses 100 (issue722_fit_M.N_REFIT_PAIRS); the null refits a floor
# per pseudo-arm per resample (n_resamples × 2 arms × NULL_REFIT_PAIRS refits), so it
# uses fewer pairs to stay tractable — each pseudo-floor is a coarse-but-honest
# per-arm estimate, and the null's own resampling dominates the band width. Smoke
# clamps this via --null-refit-pairs.
NULL_REFIT_PAIRS = 40
# The plan §3 "substrate matters" decision rule (D1): a CONJUNCTION of the null-band
# gate AND a driving-pair pairwise-CI excluding 0 — NOT the single null-band gate the
# round-1/2 verdict shipped. One constant so the reducer + the summary metadata agree.
_DECISION_RULE = "conjunction: (max_diff > null_p95) AND (a driving-pair pairwise CI excludes 0)"


def _git_sha() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


# ── Observed Δ/floor read (reused fit machinery) ───────────────────────────────


def _cells_from_summary(behavior: str, substrate: str, layer: int, reduced_root: Path) -> list:
    """Build CellRecord list at ``layer`` from the reduced summary (import shared loader)."""
    import issue813_save_maps as savemaps813

    return savemaps813.load_reduced_cells(behavior, substrate, layer, reduced_root)


def observed_read(
    behavior: str,
    substrate: str,
    reduced_root: Path,
    rb_main: dict,
    rb_fact: dict | None,
    wu_marker: np.ndarray | None,
) -> dict:
    """The observed Δ/floor read at the frozen headline layer for one (behavior, substrate)."""
    cells = _cells_from_summary(behavior, substrate, HEADLINE_LAYER, reduced_root)
    if behavior == "marker":
        # marker has no r_B → the two-read marker path (unproj ‖ΔM‖ + W_U[※]-proj).
        cell = marker_mc.fit_marker_layer(
            HEADLINE_LAYER, cells, wu_marker, with_chain=(substrate != "generic")
        )
        return {
            "behavior": behavior,
            "substrate": substrate,
            "layer": HEADLINE_LAYER,
            "n_cells": cell["n_cells"],
            # read-1 (behavior-agnostic) is the marker's PRIMARY floor-normalized DV
            "delta_over_floor": cell["unproj_delta_over_floor"],
            "delta_over_floor_sd": cell["unproj_delta_over_floor_sd"],
            "delta_med": cell["unproj_delta_med"],
            "floor_combined": cell["unproj_floor_p95"]["combined"],
            # read-2 (W_U[※]-projected, marker-specific) + its subspace-capture gate
            "wu_delta_over_floor": cell["wu_proj_delta_over_floor"],
            "wu_frac_in_subspace": cell["wu_frac_in_subspace"],
            "wu_read2_informative": cell["wu_read2_informative"],
            "support_distance": cell["support_distance"],
            "chain_rho": cell.get("chain_rho"),
            "marker_two_read": True,
        }
    # em / fact / sycophancy → the ridge-only headline (Delta_med / floor / chain-ρ).
    cell = fitM.fit_cell(behavior, HEADLINE_LAYER, cells, rb_main, rb_fact, include_mlp=False)
    return {
        "behavior": behavior,
        "substrate": substrate,
        "layer": HEADLINE_LAYER,
        "n_cells": cell["n_cells"],
        "delta_over_floor": cell["Delta_over_floor_sd"],
        "delta_over_floor_sd": cell["Delta_over_floor_sd"],
        "delta_med": cell["Delta_med"],
        "floor_combined": cell["floor_combined"],
        "support_distance": cell["support_distance"],
        "chain_rho": cell["chain_rho"],
        "n_with_E": cell["chain_rho"].get("n_with_E"),
        "marker_two_read": False,
    }


# ── Pseudo-substrate Δ/floor read (headline-layer only, for the null) ──────────


def _pseudo_delta_over_floor(
    c0: np.ndarray,
    cplus: np.ndarray,
    v0: np.ndarray,
    vplus: np.ndarray,
    families: list[str],
    r_hat: np.ndarray | None,
    *,
    n_refit_pairs: int,
) -> tuple[float, float]:
    """(Δ_med, Δ/floor) for a headline-layer pseudo-map — B2: the REGISTERED DV space.

    Fits M0 = ridge(c0→V0_64) and M⁺ = ridge(cplus→Vplus_64) at THIS layer via the
    reused ``_ridge_fit_predict`` + ``_pca_basis_v0`` (top-64 shared V0 basis) and
    reduces the base-grid difference by the r_hat projection (em/fact/syco r_B) or by
    the vector norm (marker read-1) — EXACTLY ``fit_cell`` / ``fit_marker_layer``'s
    numerator. It ALSO refits a per-pseudo-arm FLOOR through the SAME shared harness
    (``make_refit_pair`` for the r_hat path / ``marker_mc._refit_pair_norm`` for the
    norm path, over the M0 / M⁺ / shifted refit designs, family-clustered) and returns
    the normalized DV in each behavior's own convention (em/fact/syco: ``Δ_med /
    floor_sd_combined`` matching ``Delta_over_floor_sd``; marker: ``Δ_med /
    floor_p95_combined`` matching ``unproj_delta_over_floor``), so the null band is
    built in the REGISTERED Δ/floor space, not raw Δ (concern
    i813-verdict-raw-delta-not-registered-floor). Returns ``(delta_med, delta_over_floor)``;
    ``delta_over_floor`` is NaN when the floor underflows (excluded by the caller).
    """
    pca_basis = fitM._pca_basis_v0(v0, TARGET_DIM)  # (k<=64, HIDDEN)
    v0_64 = fitM._to64(v0, pca_basis)
    vplus_64 = fitM._to64(vplus, pca_basis)
    m0_grid = fitM._ridge_fit_predict(c0, v0_64, c0)  # (n, 64)
    mplus_grid = fitM._ridge_fit_predict(cplus, vplus_64, c0)
    delta_full = (mplus_grid - m0_grid) @ pca_basis  # (n, HIDDEN)
    if r_hat is None:
        delta_med = float(np.median(np.linalg.norm(delta_full, axis=1)))
    else:
        delta_med = float(np.median(np.abs(delta_full @ r_hat)))

    # Per-pseudo-arm refit floor via the SHARED harness (same three refit designs the
    # observed read uses: M0, M⁺, shifted M0(cplus)). Grid = the base c0, matching the
    # numerator's eval grid. r_hat=None routes through the marker read-1 ‖·‖ variant.
    fit_fn = fitM._refit_ridge_fn(c0)  # returns preds at c0, back-projected to HIDDEN
    m0_at_cplus = fitM.m0_at_cplus_ridge_full(c0, v0, cplus, pca_basis)
    if r_hat is None:
        fl_m0 = marker_mc._refit_pair_norm(c0, v0, fit_fn, c0, families, n_pairs=n_refit_pairs)
        fl_mp = marker_mc._refit_pair_norm(
            cplus, vplus, fit_fn, c0, families, n_pairs=n_refit_pairs
        )
        fl_sh = marker_mc._refit_pair_norm(
            cplus, m0_at_cplus, fit_fn, c0, families, n_pairs=n_refit_pairs
        )
        # marker read-1 normalizes by the p95-COMBINED floor (unproj_delta_over_floor).
        floor = max(
            float(np.percentile(fl_m0, 95)),
            float(np.percentile(fl_mp, 95)),
            float(np.percentile(fl_sh, 95)),
        )
    else:
        fl_m0 = make_refit_pair(c0, v0, fit_fn, c0, r_hat, families, n_pairs=n_refit_pairs)
        fl_mp = make_refit_pair(cplus, vplus, fit_fn, c0, r_hat, families, n_pairs=n_refit_pairs)
        fl_sh = make_refit_pair(
            cplus, m0_at_cplus, fit_fn, c0, r_hat, families, n_pairs=n_refit_pairs
        )
        # em/fact/syco normalize by the SD-COMBINED floor (Delta_over_floor_sd).
        floor = max(floor_sd(fl_m0), floor_sd(fl_mp), floor_sd(fl_sh))
    dof = float("nan") if floor < 1e-12 else float(delta_med / floor)
    return delta_med, dof


def substrate_swap_null(
    behavior: str,
    substrate: str,
    reduced_root: Path,
    r_hat: np.ndarray | None,
    n_resamples: int,
    *,
    n_refit_pairs: int = NULL_REFIT_PAIRS,
) -> dict:
    """Matched-n substrate-swap null in the REGISTERED Δ/floor space (B2).

    Reads ``per_question_L{HEADLINE}.npz`` (flat headline-layer rows + per-row context
    index + per-context family). Per resample: draw the substrate's question indices
    with replacement, split them into two matched-n pseudo-substrate halves,
    question-average each half per context → a pseudo-map pair, compute BOTH the raw
    Δ_med AND the normalized Δ/floor (each pseudo-arm refits its own floor through the
    shared harness) for each half, and record ``|Δ(A) − Δ(B)|`` in BOTH spaces. The
    95th percentile of the Δ/floor diffs is X_reg — the REGISTERED threshold a real
    substrate difference in Δ/floor must clear (plan §3/§6/§6.5); the raw-Δ percentiles
    are kept for continuity/diagnostics. The full per-resample Δ/floor null array is
    persisted (``null_delta_over_floor_diffs``) so the analyzer can reconstruct the
    registered band post-hoc.

    Matched-n: both pseudo-arms use the SAME per-half question count, so em's small
    pool yields a WIDE (conservative) null, never an artificially tight one.
    """
    pq_path = reduced_root / behavior / substrate / f"per_question_L{HEADLINE_LAYER}.npz"
    if not pq_path.exists():
        raise FileNotFoundError(f"per-question headline rows missing: {pq_path}")
    d = np.load(pq_path, allow_pickle=True)
    c0 = np.asarray(d["c_C_base"], dtype=np.float64)  # (n_rows, HIDDEN)
    cp = np.asarray(d["c_C_trained"], dtype=np.float64)
    v0 = np.asarray(d["v_A_base"], dtype=np.float64)
    vp = np.asarray(d["v_A_trained"], dtype=np.float64)
    row_ctx = np.asarray(d["row_context_index"], dtype=np.int64)
    row_q = np.asarray(d["row_question_index"], dtype=np.int64)
    # families is full-length, indexed by ORIGINAL context index (savemaps writes it so).
    ctx_families = [str(x) for x in d["families"]]
    q_ids = sorted(set(row_q.tolist()))
    n_q = len(q_ids)
    empty = {
        "null_p95": None,
        "null_p975": None,
        "null_over_floor_p95": None,
        "null_over_floor_p975": None,
        "n_questions": n_q,
        "n_resamples_used": 0,
        "null_space": "delta_over_floor",
    }
    if n_q < 4:
        return {**empty, "note": "too few questions (<4) for a matched-n split"}

    # Map (context, question) → row index for fast per-half question-averaging.
    rc_index: dict[tuple[int, int], int] = {}
    for i in range(len(row_ctx)):
        rc_index[(int(row_ctx[i]), int(row_q[i]))] = i
    contexts = sorted(set(row_ctx.tolist()))

    def _pseudo_stack(
        q_subset: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Question-average the subset per context → (n_ctx_kept, HIDDEN) stacks + families."""
        rows_c0, rows_cp, rows_v0, rows_vp, fams = [], [], [], [], []
        for ctx in contexts:
            idxs = [rc_index[(ctx, q)] for q in q_subset if (ctx, q) in rc_index]
            if not idxs:
                continue
            rows_c0.append(c0[idxs].mean(0))
            rows_cp.append(cp[idxs].mean(0))
            rows_v0.append(v0[idxs].mean(0))
            rows_vp.append(vp[idxs].mean(0))
            fams.append(ctx_families[ctx])  # family of THIS battery context
        return (np.stack(rows_c0), np.stack(rows_cp), np.stack(rows_v0), np.stack(rows_vp), fams)

    rng = np.random.default_rng(NULL_SEED)
    half = n_q // 2
    diffs: list[float] = []  # raw Δ_med diffs (diagnostic, continuity)
    dof_diffs: list[float] = []  # Δ/floor diffs (the REGISTERED null space)
    for _ in range(n_resamples):
        drawn = rng.choice(q_ids, size=n_q, replace=True).tolist()
        # split the RESAMPLED question list into two matched-n halves.
        a_qs = drawn[:half]
        b_qs = drawn[half : 2 * half]
        try:
            sa = _pseudo_stack(a_qs)
            sb = _pseudo_stack(b_qs)
            if sa[0].shape[0] < 4 or sb[0].shape[0] < 4:
                continue  # a degenerate half (too few contexts covered) — skip
            da_med, da_dof = _pseudo_delta_over_floor(
                sa[0], sa[1], sa[2], sa[3], sa[4], r_hat, n_refit_pairs=n_refit_pairs
            )
            db_med, db_dof = _pseudo_delta_over_floor(
                sb[0], sb[1], sb[2], sb[3], sb[4], r_hat, n_refit_pairs=n_refit_pairs
            )
        except np.linalg.LinAlgError:
            continue  # degenerate resample geometry — skip (bootstrap noise)
        diffs.append(abs(da_med - db_med))
        if not (np.isnan(da_dof) or np.isnan(db_dof)):
            dof_diffs.append(abs(da_dof - db_dof))
    if not dof_diffs:
        return {**empty, "note": "all resamples degenerate or floor-underflowed"}
    raw = np.asarray(diffs, dtype=np.float64)
    dof = np.asarray(dof_diffs, dtype=np.float64)
    return {
        # REGISTERED Δ/floor null (the band the verdict + pairwise diff are judged against)
        "null_space": "delta_over_floor",
        "null_over_floor_p95": float(np.percentile(dof, 95)),
        "null_over_floor_p975": float(np.percentile(dof, 97.5)),
        "null_over_floor_median": float(np.median(dof)),
        # full per-resample Δ/floor null array (post-hoc band reconstruction)
        "null_delta_over_floor_diffs": dof.tolist(),
        "n_over_floor_resamples_used": len(dof_diffs),
        # raw Δ_med null (diagnostic / continuity only — NOT the registered band)
        "null_p95": float(np.percentile(raw, 95)) if raw.size else None,
        "null_p975": float(np.percentile(raw, 97.5)) if raw.size else None,
        "null_median": float(np.median(raw)) if raw.size else None,
        "n_questions": n_q,
        "n_resamples_used": len(diffs),
        "n_refit_pairs": n_refit_pairs,
    }


def _r_hat_for(
    behavior: str, rb_main: dict, rb_fact: dict | None, wu_marker: np.ndarray | None
) -> np.ndarray | None:
    """The read-out direction at the headline layer (None for marker read-1 = ‖ΔM‖)."""
    if behavior == "marker":
        return None  # marker null uses read-1 (‖ΔM‖), behavior-agnostic
    return fitM._r_hat_for(behavior, HEADLINE_LAYER, rb_main, rb_fact)


def _headline_stacks(
    behavior: str, substrate: str, reduced_root: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Per-context c_C/v_A stacks (base + trained) at the FROZEN headline layer + keys.

    Reads the reduced 28-layer summary (via the shared CellRecord loader) and slices
    the headline-layer plane into (n_ctx, HIDDEN) stacks — the same object the observed
    ``fit_cell`` / ``fit_marker_layer`` read consumes. Returns
    ``(c0, cplus, v0, vplus, families, context_ids)`` with ``families`` / ``context_ids``
    parallel to the rows (the clustered-bootstrap resampling unit + the pair-alignment
    key across substrates).
    """
    cells = _cells_from_summary(behavior, substrate, HEADLINE_LAYER, reduced_root)
    c0 = np.stack([c.c0 for c in cells]).astype(np.float64)
    cplus = np.stack([c.cplus for c in cells]).astype(np.float64)
    v0 = np.stack([c.v0 for c in cells]).astype(np.float64)
    vplus = np.stack([c.vplus for c in cells]).astype(np.float64)
    families = [str(c.family) for c in cells]
    context_ids = [str(c.source_cid) for c in cells]
    return c0, cplus, v0, vplus, families, context_ids


def pairwise_diff_ci(
    behavior: str,
    sub_a: str,
    sub_b: str,
    reduced_root: Path,
    r_hat: np.ndarray | None,
    *,
    n_resamples: int = N_NULL_RESAMPLES,
    n_refit_pairs: int = NULL_REFIT_PAIRS,
    seed: int = NULL_SEED,
) -> dict:
    """Family-clustered bootstrap CI on the PAIRED Δ/floor difference for one substrate pair.

    D1 (conjunct restore): plan §3 registers "substrate matters" as a CONJUNCTION —
    the max-vs-min substrate Δ/floor difference must (i) exceed the substrate-swap null's
    p95 AND (ii) have a pairwise-difference CI on the Δ/floor difference that EXCLUDES 0.
    The verdict previously gated on (i) alone; this restores (ii).

    Both substrates fit the map over the SAME shared 50-context battery (plan §4.2), so a
    single family-clustered resample of the battery contexts (the ~7-family cluster unit)
    applies IDENTICALLY to both substrate stacks — the paired difference is on the same
    contexts per draw. Per resample: draw whole battery FAMILIES with replacement, restrict
    BOTH substrates to the resampled contexts, refit each substrate's Δ/floor at the frozen
    headline layer via the SHARED refit harness (``_pseudo_delta_over_floor`` — the exact
    ``fit_cell`` / ``fit_marker_layer`` numerator + per-arm refit floor, in each behavior's
    own DV convention: em/fact/syco SD-combined floor, marker read-1 p95-combined floor),
    and record ``Δ/floor(A) − Δ/floor(B)``. The percentile CI on that signed difference is
    the pairwise CI; it EXCLUDES 0 iff ``ci_lo > 0`` or ``ci_hi < 0``.

    The battery contexts are shared, so both substrates key on the same ``context_ids`` —
    a resampled context maps to the same row in each stack via the per-substrate context
    index. Returns the pair record: point ``abs_diff`` (from the FULL-sample observed reads,
    the caller passes these separately for the verdict) plus ``ci_lo`` / ``ci_hi`` /
    ``ci_excludes_zero`` / ``n_families`` / ``n_resamples_used`` from the bootstrap.
    """
    c0a, cpa, v0a, vpa, fam_a, ctx_a = _headline_stacks(behavior, sub_a, reduced_root)
    c0b, cpb, v0b, vpb, _fam_b, ctx_b = _headline_stacks(behavior, sub_b, reduced_root)
    # Both substrates fit over the shared battery, so index each by its OWN context list
    # and resample on the INTERSECTION (a context present in both) — the paired difference
    # requires both arms to cover the drawn context. Families come from the shared battery
    # so they agree per context; use substrate A's family map (identical by construction —
    # substrate B's `_fam_b` would give the same per-context labels).
    a_ctx_to_row = {cid: i for i, cid in enumerate(ctx_a)}
    b_ctx_to_row = {cid: i for i, cid in enumerate(ctx_b)}
    shared_ctx = [cid for cid in ctx_a if cid in b_ctx_to_row]
    ctx_family = {cid: fam_a[a_ctx_to_row[cid]] for cid in shared_ctx}
    empty = {
        "pair": f"{sub_a}_vs_{sub_b}",
        "dv_space": "delta_over_floor",
        "ci_lo": None,
        "ci_hi": None,
        "ci_excludes_zero": None,
        "n_families": len({ctx_family[c] for c in shared_ctx}),
        "n_resamples_used": 0,
    }
    if len(shared_ctx) < 4:
        return {**empty, "note": "too few shared battery contexts (<4) for a paired CI"}

    uniq_fams = sorted({ctx_family[c] for c in shared_ctx})
    fam_to_ctx: dict[str, list[str]] = {f: [] for f in uniq_fams}
    for cid in shared_ctx:
        fam_to_ctx[ctx_family[cid]].append(cid)
    clustered = len(uniq_fams) >= 2
    rng = np.random.default_rng(seed)
    signed_diffs: list[float] = []

    def _stack_for(ctx_subset: list[str], side: str):
        """Assemble (c0, cplus, v0, vplus, families) for one substrate over ctx_subset."""
        rows = (
            [a_ctx_to_row[c] for c in ctx_subset]
            if side == "a"
            else [b_ctx_to_row[c] for c in ctx_subset]
        )
        fams = [ctx_family[c] for c in ctx_subset]
        if side == "a":
            return c0a[rows], cpa[rows], v0a[rows], vpa[rows], fams
        return c0b[rows], cpb[rows], v0b[rows], vpb[rows], fams

    for _ in range(n_resamples):
        if clustered:
            chosen_fams = rng.choice(uniq_fams, size=len(uniq_fams), replace=True)
            drawn_ctx = [c for f in chosen_fams for c in fam_to_ctx[str(f)]]
        else:
            drawn_ctx = list(rng.choice(shared_ctx, size=len(shared_ctx), replace=True))
        # Need enough distinct contexts on BOTH arms to fit (the refit needs >=4 rows).
        if len(set(drawn_ctx)) < 4:
            continue
        try:
            sa = _stack_for(drawn_ctx, "a")
            sb = _stack_for(drawn_ctx, "b")
            _, da_dof = _pseudo_delta_over_floor(
                sa[0], sa[1], sa[2], sa[3], sa[4], r_hat, n_refit_pairs=n_refit_pairs
            )
            _, db_dof = _pseudo_delta_over_floor(
                sb[0], sb[1], sb[2], sb[3], sb[4], r_hat, n_refit_pairs=n_refit_pairs
            )
        except np.linalg.LinAlgError:
            continue  # degenerate resample geometry — skip (bootstrap noise)
        if np.isnan(da_dof) or np.isnan(db_dof):
            continue  # a floor underflowed on this resample — excluded
        signed_diffs.append(float(da_dof - db_dof))
    if not signed_diffs:
        return {**empty, "note": "all resamples degenerate or floor-underflowed"}
    arr = np.asarray(signed_diffs, dtype=np.float64)
    ci_lo = float(np.percentile(arr, 2.5))
    ci_hi = float(np.percentile(arr, 97.5))
    return {
        "pair": f"{sub_a}_vs_{sub_b}",
        "dv_space": "delta_over_floor",
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        # the CI EXCLUDES 0 iff the whole interval is on one side of 0
        "ci_excludes_zero": bool(ci_lo > 0.0 or ci_hi < 0.0),
        "ci_median": float(np.median(arr)),
        "n_families": len(uniq_fams),
        "n_resamples_used": len(signed_diffs),
        "n_refit_pairs": n_refit_pairs,
    }


def pairwise_substrate_diff_cis(
    observed_by_sub: dict[str, dict],
    behavior: str,
    reduced_root: Path,
    r_hat: np.ndarray | None,
    *,
    n_resamples: int = N_NULL_RESAMPLES,
    n_refit_pairs: int = NULL_REFIT_PAIRS,
) -> list[dict]:
    """Pairwise Δ/floor difference + family-clustered bootstrap CI across substrate pairs (D1).

    Reports, for each substrate pair in the REGISTERED Δ/floor space (plan §3/§6/§6.5):
    the full-sample ``abs_diff`` = |Δ/floor(A) − Δ/floor(B)| (the observed point the null
    band gates), the SIGNED-difference clustered-bootstrap CI (``ci_lo`` / ``ci_hi`` from
    ``pairwise_diff_ci``), and ``ci_excludes_zero`` — the second conjunct of the plan §3
    "substrate matters" decision rule (H1 requires the CI to INCLUDE 0; H0 requires it to
    EXCLUDE 0). The raw Δ_med difference is carried alongside as a diagnostic only.

    The CI is a genuine bootstrap (not a delegated placeholder): both substrates fit over
    the SAME shared 50-context battery, so a single family-clustered context resample refits
    both arms per draw and the paired Δ/floor difference is recomputed each resample — the
    identical refit machinery the observed read + the substrate-swap null use.
    """
    out = []
    subs = [s for s in SUBSTRATES if s in observed_by_sub]
    for i in range(len(subs)):
        for j in range(i + 1, len(subs)):
            a, b = subs[i], subs[j]
            da = observed_by_sub[a].get("delta_over_floor")
            db = observed_by_sub[b].get("delta_over_floor")
            da_raw = observed_by_sub[a].get("delta_med")
            db_raw = observed_by_sub[b].get("delta_med")
            rec = {
                "pair": f"{a}_vs_{b}",
                "dv_space": "delta_over_floor",
                "delta_over_floor_a": da,
                "delta_over_floor_b": db,
                "abs_diff": (None if (da is None or db is None) else abs(da - db)),
                # raw Δ_med diagnostic (NOT the registered comparison)
                "delta_med_a": da_raw,
                "delta_med_b": db_raw,
                "abs_diff_delta_med": (
                    None if (da_raw is None or db_raw is None) else abs(da_raw - db_raw)
                ),
                "ci_lo": None,
                "ci_hi": None,
                "ci_excludes_zero": None,
            }
            # Only bootstrap the CI when both observed reads exist (a floor-underflowed
            # substrate has no Δ/floor point, so its pairwise diff is undefined).
            if da is not None and db is not None:
                ci = pairwise_diff_ci(
                    behavior,
                    a,
                    b,
                    reduced_root,
                    r_hat,
                    n_resamples=n_resamples,
                    n_refit_pairs=n_refit_pairs,
                )
                rec.update(
                    {
                        "ci_lo": ci.get("ci_lo"),
                        "ci_hi": ci.get("ci_hi"),
                        "ci_median": ci.get("ci_median"),
                        "ci_excludes_zero": ci.get("ci_excludes_zero"),
                        "ci_n_families": ci.get("n_families"),
                        "ci_n_resamples_used": ci.get("n_resamples_used"),
                        "ci_note": ci.get("note"),
                    }
                )
            out.append(rec)
    return out


def decide_substrate_matters(
    dofs: dict[str, float | None],
    null_by_sub: dict[str, dict],
    pairwise: list[dict],
) -> dict:
    """The plan §3 CONJUNCTION verdict — pure function (unit-testable, D1 regression).

    Plan §3 registers the decision rule as a CONJUNCTION, NOT the single null-band gate
    the shipped verdict used:

    - **H1 (substrate-agnostic):** per-behavior Δ/floor is INDISTINGUISHABLE across the
      substrates — the max-vs-min Δ/floor difference is WITHIN the substrate-swap null band
      AND every pairwise-difference CI INCLUDES 0.
    - **H0 (substrate matters):** Δ/floor DIFFERS beyond the noise band — the max-vs-min
      difference EXCEEDS the null's p95 AND at least one pairwise-difference CI EXCLUDES 0.

    So:
      substrate_matters = (max_diff > null_x) AND (some pairwise CI excludes 0)
    with the additional gate that the CI-excluding pair must be one that also DRIVES the
    max_diff (its members are the max-vs-min Δ/floor substrates) — a pairwise CI on a pair
    NOT involved in the max spread cannot, on its own, flip the max-vs-min verdict.

    Returns the verdict dict:
      - ``substrate_matters``: True (both conjuncts fire), False (both fail — max within band
        AND all CIs include 0), or None (AMBIGUOUS — exactly one conjunct fires, or an input
        is missing so a conjunct is undecidable).
    ``dofs`` maps substrate → observed Δ/floor (None where floor-underflowed);
    ``null_by_sub`` maps substrate → its substrate-swap-null dict (``null_over_floor_p95``);
    ``pairwise`` is the ``pairwise_substrate_diff_cis`` output.
    """
    valid = {s: v for s, v in dofs.items() if v is not None}
    if len(valid) < 2:
        return {
            "dv_space": "delta_over_floor",
            "decision_rule": _DECISION_RULE,
            "max_vs_min_delta_over_floor_diff": None,
            "null_x_over_floor_p95": None,
            "null_band_conjunct": None,
            "pairwise_ci_conjunct": None,
            "substrate_matters": None,
            "note": "fewer than 2 substrates with a valid Δ/floor point",
        }
    hi_sub = max(valid, key=lambda s: valid[s])
    lo_sub = min(valid, key=lambda s: valid[s])
    max_diff = valid[hi_sub] - valid[lo_sub]
    null_x = max(
        (null_by_sub[s].get("null_over_floor_p95") or 0.0) for s in valid if null_by_sub.get(s)
    )
    # Conjunct (i): the max-vs-min Δ/floor difference clears the substrate-swap null p95.
    null_band_conjunct = (max_diff > null_x) if null_x else None
    # Conjunct (ii): a pairwise CI that EXCLUDES 0 on a DRIVING pair (the {hi,lo} substrates
    # whose difference IS max_diff). A CI on a non-driving pair does not flip the max verdict.
    driving_pair_keys = {f"{hi_sub}_vs_{lo_sub}", f"{lo_sub}_vs_{hi_sub}"}
    driving_recs = [p for p in pairwise if p.get("pair") in driving_pair_keys]
    excludes = [
        p.get("ci_excludes_zero") for p in driving_recs if p.get("ci_excludes_zero") is not None
    ]
    # None when the driving pair's CI could not be computed; else True iff any excludes 0.
    pairwise_ci_conjunct = None if not excludes else any(excludes)
    # Combine the two conjuncts into the tri-state verdict.
    if null_band_conjunct is None or pairwise_ci_conjunct is None:
        matters: bool | None = None  # a conjunct is undecidable → AMBIGUOUS
    elif null_band_conjunct and pairwise_ci_conjunct:
        matters = True  # both fire → substrate matters (H0)
    elif (not null_band_conjunct) and (not pairwise_ci_conjunct):
        matters = False  # both fail → substrate-agnostic (H1)
    else:
        matters = None  # exactly one conjunct fires → AMBIGUOUS (neither H0 nor H1)
    return {
        "dv_space": "delta_over_floor",
        "decision_rule": _DECISION_RULE,
        "max_vs_min_delta_over_floor_diff": max_diff,
        "max_substrate": hi_sub,
        "min_substrate": lo_sub,
        "null_x_over_floor_p95": null_x,
        "null_band_conjunct": null_band_conjunct,
        "pairwise_ci_conjunct": pairwise_ci_conjunct,
        "driving_pair_ci_excludes_zero": excludes,
        "substrate_matters": matters,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fit658.DEVICE = fit658._resolve_device("cpu")  # closed-form ridge — CPU by design
    fit658._assert_ridge_exactness()
    logger.info("[phase=analysis] device=%s; ridge exactness gate PASS", fit658.DEVICE)

    ap = argparse.ArgumentParser(
        description="Issue #813 — DVs (Δ/floor + chain-ρ + substrate-swap null)"
    )
    ap.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS), choices=list(BEHAVIORS))
    ap.add_argument("--substrates", nargs="+", default=list(SUBSTRATES), choices=list(SUBSTRATES))
    ap.add_argument(
        "--reduced-root", type=Path, default=PROJECT_ROOT / "eval_results/issue_813/reduced"
    )
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "eval_results/issue_813")
    ap.add_argument("--n-null-resamples", type=int, default=N_NULL_RESAMPLES)
    ap.add_argument(
        "--null-refit-pairs",
        type=int,
        default=NULL_REFIT_PAIRS,
        help="per-pseudo-arm refit-floor pairs inside the Δ/floor null (smoke clamps this)",
    )
    args = ap.parse_args()

    # r_B artifacts (em/syco from #658 r_b.pt; fact from #667 r_b_fact.pt; marker W_U[※]).
    rb_main = fitM._load_rb_main() if any(b in ("em", "sycophancy") for b in args.behaviors) else {}
    rb_fact = fitM._load_rb_fact() if "fact" in args.behaviors else None
    wu_marker = marker_mc.load_wu_marker_direction() if "marker" in args.behaviors else None

    delta_dir = args.out_dir / "delta_floor"
    chain_dir = args.out_dir / "chain_rho"
    null_dir = args.out_dir / "substrate_swap_null"
    for d in (delta_dir, chain_dir, null_dir):
        d.mkdir(parents=True, exist_ok=True)

    per_behavior: dict[str, dict] = {}
    for behavior in args.behaviors:
        observed_by_sub: dict[str, dict] = {}
        null_by_sub: dict[str, dict] = {}
        r_hat = _r_hat_for(behavior, rb_main, rb_fact, wu_marker)
        for substrate in args.substrates:
            logger.info(
                "[phase=analysis] observed read %s/%s L%d", behavior, substrate, HEADLINE_LAYER
            )
            obs = observed_read(behavior, substrate, args.reduced_root, rb_main, rb_fact, wu_marker)
            observed_by_sub[substrate] = obs
            (delta_dir / f"{behavior}__{substrate}.json").write_text(
                json.dumps(obs, indent=2, default=float)
            )
            # chain-ρ (elicit+mix only; generic E≈0 → N/A per plan §3/§6)
            if substrate != "generic" and obs.get("chain_rho") is not None:
                (chain_dir / f"{behavior}__{substrate}.json").write_text(
                    json.dumps(
                        {
                            "behavior": behavior,
                            "substrate": substrate,
                            "chain_rho": obs["chain_rho"],
                        },
                        indent=2,
                        default=float,
                    )
                )
            # substrate-swap null (matched-n) at the frozen headline layer
            logger.info("[phase=analysis] substrate-swap null %s/%s", behavior, substrate)
            null = substrate_swap_null(
                behavior,
                substrate,
                args.reduced_root,
                r_hat,
                args.n_null_resamples,
                n_refit_pairs=args.null_refit_pairs,
            )
            null_by_sub[substrate] = null
            (null_dir / f"{behavior}__{substrate}.json").write_text(
                json.dumps(null, indent=2, default=float)
            )

        # Pairwise Δ/floor difference + family-clustered bootstrap CI (D1): both the
        # point diff (null-band gate) AND the signed-difference CI (the second conjunct
        # of the plan §3 decision rule — does the CI exclude 0?).
        pairwise = pairwise_substrate_diff_cis(
            observed_by_sub,
            behavior,
            args.reduced_root,
            r_hat,
            n_resamples=args.n_null_resamples,
            n_refit_pairs=args.null_refit_pairs,
        )
        per_behavior[behavior] = {
            "observed": observed_by_sub,
            "substrate_swap_null": null_by_sub,
            "pairwise_substrate_diff": pairwise,
        }
        # Verdict per behavior (D1 conjunction restore): plan §3 registers "substrate
        # matters" as (max-vs-min Δ/floor diff > substrate-swap null p95) AND (a
        # driving-pair pairwise-difference CI EXCLUDES 0). The shipped verdict gated on
        # the null-band conjunct ALONE (BLOCKER i813-pairwise-ci-conjunct-missing); the
        # pure `decide_substrate_matters` reducer now enforces BOTH conjuncts. Raw Δ_med
        # is a diagnostic only (NOT the registered comparison — B2).
        dofs = {s: observed_by_sub[s].get("delta_over_floor") for s in observed_by_sub}
        verdict = decide_substrate_matters(dofs, null_by_sub, pairwise)
        # raw Δ_med diagnostic (continuity only)
        raw_valid = {
            s: observed_by_sub[s].get("delta_med")
            for s in observed_by_sub
            if observed_by_sub[s].get("delta_med") is not None
        }
        verdict["max_vs_min_delta_med_diff"] = (
            (max(raw_valid.values()) - min(raw_valid.values())) if len(raw_valid) >= 2 else None
        )
        per_behavior[behavior]["verdict"] = verdict
        logger.info(
            "[phase=analysis] %s: max-min Δ/floor diff=%s vs null X(p95)=%s | "
            "null_band=%s pairwise_CI_excl0=%s → matters=%s",
            behavior,
            verdict.get("max_vs_min_delta_over_floor_diff"),
            verdict.get("null_x_over_floor_p95"),
            verdict.get("null_band_conjunct"),
            verdict.get("pairwise_ci_conjunct"),
            verdict.get("substrate_matters"),
        )

    summary = {
        "issue": 813,
        "read": "map_change_substrate_dependence_M0_vs_Mplus",
        "headline_layer": HEADLINE_LAYER,
        "target_dim": TARGET_DIM,
        # B2: verdict + null are BOTH in the registered Δ/floor space
        "verdict_dv_space": "delta_over_floor",
        # D1: the verdict is the plan §3 CONJUNCTION (null-band AND pairwise-CI),
        # not the single null-band gate the round-1/2 verdict shipped.
        "verdict_decision_rule": _DECISION_RULE,
        "n_null_resamples": args.n_null_resamples,
        "null_refit_pairs": args.null_refit_pairs,
        "null_seed": NULL_SEED,
        "git_commit": _git_sha(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "per_behavior": per_behavior,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    logger.info("[phase=analysis] wrote %s", args.out_dir / "summary.json")
    # NO [phase=done] here — analysis runs as a SUBPROCESS of issue813_dispatch.sh
    # (phase 4), whose stdout it inherits; the poller reserves [phase=done] for the
    # ONE terminal line the .sh emits AFTER the sentinel write (#545). A premature
    # [phase=done] here would false-signal completion before the sentinel exists.
    logger.info("[phase=analysis] analysis complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
