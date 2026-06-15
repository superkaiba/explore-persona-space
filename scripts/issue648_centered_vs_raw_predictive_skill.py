#!/usr/bin/env python3
"""Task #648 - head-to-head predictive SKILL of raw vs centered persona-distance
cosine, per recoverable bank from #536. CPU-only; reuses #536's join builders +
recoverable-bank gate.

PRIMARY DV  = paired (CV R2_centered - CV R2_raw) with TRAIN-FOLD-ONLY centering
              (the held-out unit never enters the mean it is centered against;
              applied SYMMETRICALLY to both recipes - raw has no fitted mean and
              is bit-identical to the bank-global raw predictor, so the moved
              variable stays raw-vs-centered).  SS_tot is pinned to the PER-FOLD
              TRAIN-mean baseline (MF4).
SECONDARY DV = in-sample (rho_centered - rho_raw), bank-global centering, labeled
              `transductive` in the output schema (matches #536's
              `raw_vs_centered_x_spearman` survival family).

Per-bank ONLY, never pooled across banks (#536 pin).  Banks with n_groups <= 5
(#66, #142) report their delta + CI but do NOT drive the H-verdict (MF3); a bank
whose two predictors both fail out-of-sample (both CV R2 < 0) is excluded with
`both_predictors_fail_oos` regardless of CI (MF4, precedence checked FIRST).

Single variable vs #536: the cosine centering recipe.  Cells, target, length
residualization method + covariate, CV partition, and the rho estimator are held
constant across the two recipes within each bank.

Provenance / signatures confirmed in plan §11 (carried as comments at each use):
  * compute_cosine_matrix(C, centering=...) - representation_shift.py:142-162:
        'none' is a no-op; 'global_mean' subtracts C.mean(dim=0, keepdim=True).
  * length_partial_spearman - issue536_recompute_driver.py:139-147:
        (rxy - rx*ry)/sqrt((1-rx^2)(1-ry^2)); denom < 1e-9 -> NaN.
  * GATE_MATRIX_TOL=1e-4, GATE_RHO_TOL=0.02 - issue536_recompute_driver.py:72-73.
  * affected_set.raw_line_regraded=[478,490,505,396,415,405],
    canonical_line_verified=[66,142,311,380] - eval_results/issue_536/audit_table.json.

This is a CPU-only re-analysis of cached artifacts; it trains nothing, generates
no new data, and (apart from the single already-mirrored #505 centroid fallback
inherited from #536's family_505) performs no HF/WandB/RunPod data flow.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.stats as sps

# REPO = the checkout that holds THIS script (the worktree when run from one).
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

# Reuse #536 verbatim - loaders, gates, length-partial, provenance helpers.
import torch  # noqa: E402  - centroid reads
from issue536_recompute_driver import (  # noqa: E402
    CORE11_142,
    GATE_RHO_TOL,
    SOURCES_142,
    _git_sha,
    _names_hash,
    _now,
    family_20bank,
    family_111bank,
    family_505,
    family_n24,
    length_partial_spearman,
    spearman,
)

log = logging.getLogger("i648.skill")

# Output dirs default to THIS checkout (the worktree) so results land on issue-648.
OUT_DIR = REPO / "eval_results" / "issue_648"
FIG_DIR = REPO / "figures" / "issue_648"

# 111-bank distance JSON: untracked-in-HEAD, restorable from git (plan §4.5).
BANK111_JSON_REL = "eval_results/single_token_100_persona/cosine_distance_matrix_layer20.json"
BANK111_RESTORE_SHA = "776c7c3b75"  # task #560 Phase B; immutable git object.

RNG_SEED = 20648  # §11
N_BOOT = 10_000  # §11 (>= published bootstrap convention)
VAR_FLOOR = 1e-8  # predictor/target-variance degeneracy floor (§6; float64 unit-scale)
LOWN_VERDICT_FLOOR = 5  # n_groups <= 5 -> contributes_to_h_verdict = False (MF3)
DEGEN_FOLD_FRAC = 0.25  # > this fraction of folds degenerate-skipped -> verdict-ineligible

# Tasks in the two recoverable lists that are NOT assembled as their own bank,
# with the documented reason (MF1 accounting - covered, not dropped):
#   #415 corroborates #396 on ONE shared n24 join + DV -> reported ONCE as the
#   "#396/#415" bank, #415 named here so the accounting gate counts it.
EXCLUDED_WITH_REASON = {
    415: "corroborating-duplicate-of-396-shared-join-and-DV",
}


# ──────────────────────────────────────────────────────────────────────────
# §4.5 prerequisite: restore the one untracked 111-bank distance JSON.
# ──────────────────────────────────────────────────────────────────────────
def ensure_bank111_json(data_root: Path) -> None:
    """Restore eval_results/single_token_100_persona/cosine_distance_matrix_layer20.json
    into `data_root` from git `776c7c3b75` if absent. Read-only restore of an
    artifact #536 itself consumed (the 1e-4 matrix gate in family_111bank is the
    runtime content-identity check). Never overwrites an existing file."""
    dst = data_root / BANK111_JSON_REL
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    log.info("[restore] 111-bank distance JSON from git %s -> %s", BANK111_RESTORE_SHA, dst)
    res = subprocess.run(
        ["git", "show", f"{BANK111_RESTORE_SHA}:{BANK111_JSON_REL}"],
        cwd=data_root,
        capture_output=True,
        text=True,
        env={**os.environ},
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"111-bank JSON restore FAILED (git show {BANK111_RESTORE_SHA}): {res.stderr.strip()}"
        )
    dst.write_text(res.stdout)


# ──────────────────────────────────────────────────────────────────────────
# §11-grounded recoverable-bank gate: read #536's affected_set, not a guess.
# ──────────────────────────────────────────────────────────────────────────
def recoverable_banks(data_root: Path) -> dict:
    """The #536 affected_set partition (exact strings, not a re-derivation)."""
    at = json.loads((data_root / "eval_results" / "issue_536" / "audit_table.json").read_text())
    return at["affected_set"]


def assert_recoverable_set_accounted(panels: list[BankPanel], aset: dict) -> None:
    """MF1 mechanical gate: every task in raw_line_regraded  union  canonical_line_verified
    is covered by an assembled bank OR EXCLUDED_WITH_REASON; else ABORT."""
    recoverable = set(aset["raw_line_regraded"]) | set(aset["canonical_line_verified"])
    covered: set[int] = set()
    for p in panels:
        covered |= set(p.source_task_ids)
    covered |= set(EXCLUDED_WITH_REASON)
    missing = recoverable - covered
    if missing:
        raise RuntimeError(
            f"recoverable_banks gate FAILED: tasks {sorted(missing)} are in #536's "
            f"recoverable set but neither assembled nor documented-excluded. Aborting."
        )


# ──────────────────────────────────────────────────────────────────────────
# BankPanel - carries the centroid tensor + per-cell index/reduction so the
# predictor can be RE-CENTERED on a train-fold subset inside CV.
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class BankPanel:
    bank_id: str  # plain-English label, e.g. "100-persona L20 (#66)"
    family: str  # #536 family slug
    source_task_ids: list[int]  # [66] / [142] / [396, 415] / ... for the MF1 gate
    C: np.ndarray  # centroid bank (n_personas x d) for this layer
    cell_reduce: str  # "pair" | "min_over_pos" | "two_pair_mean" | "mean_to_others"
    # | "single_ref" | "midpoint" - see cosine_predictor()
    cell_idx: list  # per-cell index payload (shape depends on cell_reduce)
    centering_bank_idx: np.ndarray  # rows of C that define the centering universe
    use_similarity: bool  # True -> predictor = cos (#66/#142/#396/#505); False -> 1 - cos
    y: np.ndarray  # per-cell continuous leakage target
    group: np.ndarray  # ORIGINAL CV group id per cell (LOGO unit)
    covar: np.ndarray  # per-cell partial covariate (length / s / log_tokens)
    has_covar: bool  # False -> no residualization (plain Spearman / OLS)
    cv_unit: str  # plain-English LOGO unit
    gate_max_dev: float  # the #536 join-gate deviation that PASSED at assembly
    # The in-sample rho method MUST match #536's per-bank published statistic so the
    # join-sanity check is a true gate AND the secondary Delta-rho is faithful:
    #   "plain"           -> plain Spearman (#66/#142/#405/#478/#490/#505; no covar)
    #   "length_partial"  -> (rxy - rx*ry)/sqrt(...) rank partial (#396, length covar)
    #   "rank_residual"   -> verify_380's rank-residualize-then-Spearman (#380, log_tokens)
    #   "value_residual"  -> verify_311's value-residualize-on-s-then-Spearman (#311)
    rho_method: str = "plain"
    # #311 midpoint is built on CENTERED-but-UN-normalised vectors (verify_311 does
    # NOT re-normalise before averaging A,B); all other reductions use unit rows.
    midpoint_unnormalised: bool = False
    join_sanity: dict = field(default_factory=dict)  # in-sample rho vs #536 published


# ──────────────────────────────────────────────────────────────────────────
# Predictor construction: center the FULL bank tensor on `centering_rows`, then
# apply the bank-specific per-cell reduction. raw (centering='none') ignores the
# subset (no mean) -> bit-identical to the bank-global raw predictor.
# ──────────────────────────────────────────────────────────────────────────
def _centered_rows(
    C: np.ndarray, centering_rows: np.ndarray, centering: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (centered, unit) rows of C after applying the centering recipe.

    centering='none'        -> no mean subtracted (centering_rows ignored).
    centering='global_mean' -> subtract C[centering_rows].mean(0) from ALL rows.
    Mirrors compute_cosine_matrix's subtract-mean-then-normalise contract
    (representation_shift.py:142-162) but over an arbitrary centering subset, so
    the train-fold-only primary DV can center on the train subset (MF5). The
    centered (un-normalised) rows are returned alongside the unit rows because
    #311's midpoint is built on centered-but-un-normalised vectors (verify_311)."""
    Ct = torch.as_tensor(C, dtype=torch.float64)
    if centering == "none":
        Cc = Ct
    elif centering == "global_mean":
        rows = torch.as_tensor(np.asarray(centering_rows), dtype=torch.long)
        mu = Ct[rows].mean(dim=0, keepdim=True)
        Cc = Ct - mu
    else:
        raise ValueError(f"unknown centering={centering!r}")
    unit = Cc / Cc.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return Cc, unit


def _centered_unit_rows(C: np.ndarray, centering_rows: np.ndarray, centering: str) -> torch.Tensor:
    return _centered_rows(C, centering_rows, centering)[1]


def cosine_predictor(panel: BankPanel, centering_rows: np.ndarray, centering: str) -> np.ndarray:
    """Per-cell predictor vector under `centering` with the centering mean taken
    over `centering_rows`. Distance (1 - cos) unless panel.use_similarity."""
    Cc, Cn = _centered_rows(panel.C, centering_rows, centering)

    def _sim(i: int, j: int) -> float:
        return float((Cn[i] * Cn[j]).sum().item())

    def _cos_vec(u: torch.Tensor, v: torch.Tensor) -> float:
        # F.cosine_similarity-equivalent for arbitrary (possibly un-normalised) vectors.
        return float((u @ v / (u.norm().clamp_min(1e-12) * v.norm().clamp_min(1e-12))).item())

    vals = np.empty(len(panel.cell_idx), dtype=np.float64)
    reduce = panel.cell_reduce
    for k, cell in enumerate(panel.cell_idx):
        if reduce in ("pair", "single_ref"):  # (#66/#142/#396/#505)  cell = (src, tgt)
            s = _sim(cell[0], cell[1])
            vals[k] = s if panel.use_similarity else 1.0 - s
        elif reduce == "min_over_pos":  # (#405, #478)  cell = (held, [positives])
            ds = [1.0 - _sim(cell[0], p) for p in cell[1]]
            vals[k] = float(min(ds))
        elif reduce == "two_pair_mean":  # (#490)  cell = (persona, A, B)
            dA = 1.0 - _sim(cell[0], cell[1])
            dB = 1.0 - _sim(cell[0], cell[2])
            vals[k] = 0.5 * (dA + dB)
        elif reduce == "mean_to_others":  # (#380)  cell = persona idx (scalar)
            i = int(cell)
            ds = [1.0 - _sim(i, j) for j in range(Cn.shape[0]) if j != i]
            vals[k] = float(np.mean(ds))
        elif reduce == "midpoint":  # (#311)  cell = (bystander, A, B) ; d to mid(A,B)
            if panel.midpoint_unnormalised:
                # verify_311: mid = 0.5*(cenA + cenB) on CENTERED un-normalised
                # vectors; cosine then re-normalises both sides at comparison.
                mid = 0.5 * (Cc[cell[1]] + Cc[cell[2]])
                s = _cos_vec(Cc[cell[0]], mid)
            else:
                mid = 0.5 * (Cn[cell[1]] + Cn[cell[2]])
                mid = mid / mid.norm().clamp_min(1e-12)
                s = float((Cn[cell[0]] * mid).sum().item())
            vals[k] = 1.0 - s
        else:
            raise ValueError(f"unknown cell_reduce={reduce!r}")
    return vals


def _train_centering_rows(panel: BankPanel, train_mask: np.ndarray) -> np.ndarray:
    """MF5 centering universe inside a fold = (panel.centering_bank_idx)  intersect
    (personas appearing in TRAIN cells). The held-out group's personas are
    excluded from the mean. A persona appears in a cell via its src/ref/held/
    bystander index AND any positive/A/B/mean-to-others companion."""
    train_personas: set[int] = set()
    reduce = panel.cell_reduce
    for cell in (panel.cell_idx[i] for i in np.where(train_mask)[0]):
        if reduce in ("pair", "single_ref"):
            train_personas.update((int(cell[0]), int(cell[1])))
        elif reduce == "min_over_pos":
            train_personas.add(int(cell[0]))
            train_personas.update(int(p) for p in cell[1])
        elif reduce in ("two_pair_mean", "midpoint"):
            train_personas.update((int(cell[0]), int(cell[1]), int(cell[2])))
        elif reduce == "mean_to_others":
            # the predictor reduces over the WHOLE bank, so the only persona the
            # cell is "about" is itself; train-fold centering excludes held-out
            # personas from the mean while the reduction still spans the bank.
            train_personas.add(int(cell))
    rows = np.intersect1d(panel.centering_bank_idx, np.array(sorted(train_personas), dtype=int))
    return rows


# ──────────────────────────────────────────────────────────────────────────
# Length-partial residualisation (TRAIN-fold OLS coefficients) - matches #536's
# (rxy - rx*ry)/sqrt(...) family for the in-sample read; OLS-residual form for
# the linear CV fit. Banks with no covariate skip residualisation entirely.
# ──────────────────────────────────────────────────────────────────────────
def _residualize(values: np.ndarray, covar: np.ndarray, coef: np.ndarray | None) -> np.ndarray:
    return values - np.polyval(coef, covar)


# ──────────────────────────────────────────────────────────────────────────
# NEW (v2 MF5): leave-one-group-out CV R^2 with TRAIN-FOLD-ONLY centering.
# Returns per-original-group held-out (y_true_frame, y_hat_frame) contributions
# (already in the per-fold-train-mean residual frame, MF4) + n_folds_skipped.
# ──────────────────────────────────────────────────────────────────────────
def _logo_group_contributions(panel: BankPanel, centering: str) -> tuple[dict, int]:
    """For EACH held-out original group g:
      1. TRAIN = cells whose ORIGINAL group != g (assert disjoint by group id).
      2. centering_rows = panel.centering_bank_idx  intersect  personas-present-in-TRAIN
         (MF5: predictor - incl. its centering mean - never sees its own test unit).
      3. build x_train, x_test from the train-fold centering.
      4. length-residualise x and y on the covariate (TRAIN-fold OLS coeffs);
         banks with no covariate skip residualisation.
      5. fit y_resid ~ x_resid (OLS) on TRAIN, apply to TEST.
      6. store (y_true - train_mean, y_hat - train_mean) - per-fold TRAIN-mean
         baseline for SS_tot (MF4).
    Degenerate folds (var(x_resid_train) or var(y_resid_train) < VAR_FLOOR, or
    < 3 train cells) are skipped + counted."""
    uniq = np.unique(panel.group)
    contrib: dict = {}
    n_skip = 0
    for g in uniq:
        tr = panel.group != g
        te = panel.group == g
        # MF2 disjointness assert (both recipes): held out by ORIGINAL group id.
        assert set(panel.group[te]).isdisjoint(set(panel.group[tr])), (
            f"LOGO fold leak: group {g} present in both train and test"
        )
        if te.sum() == 0 or tr.sum() < 3:
            n_skip += 1
            continue
        centering_rows = (
            _train_centering_rows(panel, tr)
            if centering == "global_mean"
            else (panel.centering_bank_idx)
        )
        if centering == "global_mean" and centering_rows.size < 2:
            n_skip += 1
            continue
        x = cosine_predictor(panel, centering_rows, centering)
        if panel.has_covar:
            bx = np.polyfit(panel.covar[tr], x[tr], 1)
            by = np.polyfit(panel.covar[tr], panel.y[tr], 1)
            xr_tr = _residualize(x[tr], panel.covar[tr], bx)
            yr_tr = _residualize(panel.y[tr], panel.covar[tr], by)
            xr_te = _residualize(x[te], panel.covar[te], bx)
            yr_te = _residualize(panel.y[te], panel.covar[te], by)
        else:
            xr_tr, yr_tr, xr_te, yr_te = x[tr], panel.y[tr], x[te], panel.y[te]
        if np.var(xr_tr) < VAR_FLOOR or np.var(yr_tr) < VAR_FLOOR:
            n_skip += 1
            continue
        m = np.polyfit(xr_tr, yr_tr, 1)
        train_mean = float(np.mean(yr_tr))
        y_hat = np.polyval(m, xr_te) - train_mean
        y_true = yr_te - train_mean
        contrib[int(g)] = (
            np.asarray(y_true, dtype=np.float64),
            np.asarray(y_hat, dtype=np.float64),
        )
    return contrib, n_skip


def _r2_from_contributions(items: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """Pool (y_true_frame, y_hat_frame) contributions (with multiplicity) into
    1 - SumSS_res / SumSS_tot. y_true is already in the per-fold-train-mean residual
    frame (MF4), so SS_tot = Sum y_true^2."""
    if not items:
        return float("nan")
    yt = np.concatenate([it[0] for it in items])
    yh = np.concatenate([it[1] for it in items])
    ss_res = float(np.sum((yt - yh) ** 2))
    ss_tot = float(np.sum(yt**2))
    return (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def logo_cv_r2(panel: BankPanel, centering: str) -> tuple[float, int, dict]:
    """Full-data out-of-sample R^2 (leave-one-original-group-out) + n_folds_skipped
    + the per-group contributions cache the bootstrap reuses."""
    contrib, n_skip = _logo_group_contributions(panel, centering)
    r2 = _r2_from_contributions(list(contrib.values()))
    return r2, n_skip, contrib


def _rank_residualize(values: np.ndarray, covar: np.ndarray) -> np.ndarray:
    """verify_380's rank-residualize: rank both, OLS-fit rank(value) ~ rank(covar)
    (slope + intercept), return the rank residual."""
    rv, rc = sps.rankdata(values), sps.rankdata(covar)
    slope, intercept = np.polyfit(rc, rv, 1)
    return rv - (slope * rc + intercept)


def _value_residualize(values: np.ndarray, covar: np.ndarray) -> np.ndarray:
    """verify_311's value-residualize: OLS-fit value ~ covar on the VALUES, return
    the value residual (then Spearman-correlate the residuals)."""
    return values - np.polyval(np.polyfit(covar, values, 1), covar)


def _insample_rho(panel: BankPanel, centering: str) -> float:
    """In-sample (transductive) rho, bank-global centering, using the SAME per-bank
    method #536 published (so the join-sanity gate is real and Delta-rho is faithful)."""
    x = cosine_predictor(panel, panel.centering_bank_idx, centering)
    return _rho_on_subset(panel, x, np.arange(x.size))


def _rho_on_subset(panel: BankPanel, x: np.ndarray, sel: np.ndarray) -> float:
    """In-sample rho on a (possibly multiplicity-weighted) cell subset, using the
    panel's per-bank rho method - shared by the point estimate AND the bootstrap so
    they compute the IDENTICAL statistic (the CI is then centred on the point)."""
    xs, ys = x[sel], panel.y[sel]
    method = panel.rho_method
    if method == "plain":
        return spearman(xs, ys)[0]
    cs = panel.covar[sel]
    if method == "length_partial":  # (rxy - rx*ry)/sqrt(...) rank partial (#396)
        return length_partial_spearman(xs, ys, cs)
    if method == "rank_residual":  # verify_380
        return spearman(_rank_residualize(xs, cs), _rank_residualize(ys, cs))[0]
    if method == "value_residual":  # verify_311 (covar = s)
        return spearman(_value_residualize(xs, cs), _value_residualize(ys, cs))[0]
    raise ValueError(f"unknown rho_method={method!r}")


def point_delta_r2(panel: BankPanel, contrib_raw: dict, contrib_cen: dict) -> float:
    """Full-panel LOGO DeltaR^2 over the FULL fold map (matches the bootstrap's
    per-resample statistic exactly - same estimator, so the CI is centred on it)."""
    r2c = _r2_from_contributions(list(contrib_cen.values()))
    r2r = _r2_from_contributions(list(contrib_raw.values()))
    if not (np.isfinite(r2c) and np.isfinite(r2r)):
        return float("nan")
    return r2c - r2r


def point_delta_rho(panel: BankPanel) -> float:
    """Full-panel in-sample Deltarho over the FULL sample (matches the bootstrap stat)."""
    rc = _insample_rho(panel, "global_mean")
    rr = _insample_rho(panel, "none")
    if not (np.isfinite(rc) and np.isfinite(rr)):
        return float("nan")
    return rc - rr


# ──────────────────────────────────────────────────────────────────────────
# NEW (v2 MF2): paired bootstrap in ORIGINAL-group space - no duplicate held-out
# leakage. Resample GROUPS with replacement; pool each DRAWN original group's
# cached held-out contribution (with multiplicity). A group drawn twice
# contributes its own held-out errors twice - never scored against a training
# copy of itself. The in-sample rho on the bootstrap sample re-pools the drawn
# groups' cells with multiplicity.
# ──────────────────────────────────────────────────────────────────────────
def paired_bootstrap_delta(panel: BankPanel, contrib_raw: dict, contrib_cen: dict) -> dict:
    rng = np.random.default_rng(RNG_SEED)
    uniq = np.unique(panel.group)
    # Precompute the bank-global predictors once for the in-sample rho bootstrap.
    x_cen = cosine_predictor(panel, panel.centering_bank_idx, "global_mean")
    x_raw = cosine_predictor(panel, panel.centering_bank_idx, "none")
    cells_of_group = {int(g): np.where(panel.group == g)[0] for g in uniq}

    d_r2, d_rho = [], []
    for _ in range(N_BOOT):
        gs = rng.choice(uniq, size=len(uniq), replace=True)  # multiset of ORIGINAL group ids
        gs_int = [int(g) for g in gs]
        # R^2 over the bootstrap: pool the cached held-out contribution of each
        # DRAWN group (with multiplicity). Only groups that produced a non-skipped
        # LOGO fold under BOTH recipes contribute (so the pairing stays valid).
        items_c, items_r = [], []
        for g in gs_int:
            if g in contrib_cen and g in contrib_raw:
                items_c.append(contrib_cen[g])
                items_r.append(contrib_raw[g])
        r2c = _r2_from_contributions(items_c)
        r2r = _r2_from_contributions(items_r)
        if np.isfinite(r2c) and np.isfinite(r2r):
            d_r2.append(r2c - r2r)
        # in-sample rho on the bootstrap sample (cells of drawn groups, w/ mult.),
        # using the panel's per-bank rho method (same statistic as the point estimate).
        sel = np.concatenate([cells_of_group[g] for g in gs_int])
        rc = _rho_on_subset(panel, x_cen, sel)
        rr = _rho_on_subset(panel, x_raw, sel)
        if np.isfinite(rc) and np.isfinite(rr):
            d_rho.append(rc - rr)

    def _ci(arr: list[float]) -> list[float]:
        if not arr:
            return [float("nan"), float("nan")]
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    return {
        "delta_r2_point": point_delta_r2(panel, contrib_raw, contrib_cen),
        "delta_r2_ci95": _ci(d_r2),
        "delta_rho_point": point_delta_rho(panel),
        "delta_rho_ci95": _ci(d_rho),
        "n_boot_kept_r2": len(d_r2),
        "n_boot_kept_rho": len(d_rho),
    }


# ──────────────────────────────────────────────────────────────────────────
# Per-bank ASSEMBLERS - each runs #536's join gate (verbatim family loader),
# then builds the BankPanel + a raw-in-sample-rho join-sanity check vs #536.
# ──────────────────────────────────────────────────────────────────────────
def _marker_pairs(base: Path, names: list[str], sources: list[str], target_pool: list[str]):
    """Shared (#66, #142) directed-pair builder: cell = (src, tgt), Y = leakage
    rate. Returns (cell_idx, y, group). group = source index."""
    idx = {n: i for i, n in enumerate(names)}
    cell_idx, y, group = [], [], []
    leak = {s: json.loads((base / s / "marker_eval.json").read_text()) for s in sources}
    for si, src in enumerate(sources):
        for tgt in target_pool:
            if tgt == src or tgt not in leak[src]:
                continue
            cell_idx.append((idx[src], idx[tgt]))
            y.append(float(leak[src][tgt]["rate"]))
            group.append(si)
    return cell_idx, np.asarray(y, dtype=np.float64), np.asarray(group, dtype=int)


def assemble_100p_marker(data_root: Path) -> BankPanel:
    """#66 - 100/111-persona L20 bank; cell = (source, target) over the FULL bank;
    Y = marker leakage rate. Sources = the 5 #66 sources. No length covariate
    (#66 reported plain pooled Spearman of CENTERED cosine SIMILARITY vs rate).
    Centering universe = the full 111 bank. Predictor = cosine SIMILARITY (matches
    verify_66's xs.append(cos_mc[...]) — NOT distance), so the in-sample rho sign
    matches #536's published pooled +0.6016."""
    fam = family_111bank(data_root)
    names = fam["names"]
    base = data_root / "eval_results" / "single_token_100_persona"
    sources = ["villain", "comedian", "assistant", "software_engineer", "kindergarten_teacher"]
    cell_idx, y, group = _marker_pairs(base, names, sources, names)
    C = (
        torch.load(
            base / "centroids" / "centroids_layer20.pt", map_location="cpu", weights_only=True
        )
        .to(torch.float32)
        .numpy()
        .astype(np.float64)
    )
    pub = json.loads((base / "cosine_leakage_correlation.json").read_text())["layer20"][
        "_aggregate"
    ]
    panel = BankPanel(
        bank_id="100-persona L20 (#66)",
        family="single_token_100p_L20",
        source_task_ids=[66],
        C=C,
        cell_reduce="pair",
        cell_idx=cell_idx,
        centering_bank_idx=np.arange(len(names)),
        use_similarity=True,  # verify_66 correlates cos (similarity), not distance
        y=y,
        group=group,
        covar=np.zeros_like(y),
        has_covar=False,
        cv_unit="leave-one-source-out",
        gate_max_dev=float(fam["gate"]["max_abs_dev"]),
        rho_method="plain",
    )
    # Join sanity: pooled centered SIM Spearman reproduces #66 published pooled rho.
    x_cen = cosine_predictor(panel, panel.centering_bank_idx, "global_mean")
    panel.join_sanity = {
        "centered_pooled_spearman": float(spearman(x_cen, y)[0]),
        "published_pooled_rho": float(pub["spearman_rho"]),
    }
    return panel


def assemble_core11_142(data_root: Path) -> BankPanel:
    """#142 - CORE-11 SUBSET of the 100-persona bank at L20; cell = (source, core11
    target); Y = marker leakage rate. Centering universe = the CORE-11 subset
    (gate-discovered by verify_142; full-111 centering gives the wrong rho). The
    BankPanel's C is the 11-row subset; cell indices index into that subset."""
    base = data_root / "eval_results" / "single_token_100_persona"
    cached = json.loads((base / "cosine_distance_matrix_layer20.json").read_text())
    names = cached["persona_names"]
    idx = {n: i for i, n in enumerate(names)}
    C_full = (
        torch.load(
            base / "centroids" / "centroids_layer20.pt", map_location="cpu", weights_only=True
        )
        .to(torch.float32)
        .numpy()
        .astype(np.float64)
    )
    sub_idx = [idx[n] for n in CORE11_142]
    C_sub = C_full[sub_idx]  # 11 x d - the core-11 subset bank
    i11 = {n: i for i, n in enumerate(CORE11_142)}
    leak = {s: json.loads((base / s / "marker_eval.json").read_text()) for s in SOURCES_142}
    cell_idx, y, group = [], [], []
    for si, src in enumerate(SOURCES_142):
        for tgt in CORE11_142:
            if tgt == src or tgt not in leak[src]:
                continue
            cell_idx.append((i11[src], i11[tgt]))
            y.append(float(leak[src][tgt]["rate"]))
            group.append(si)
    y = np.asarray(y, dtype=np.float64)
    group = np.asarray(group, dtype=int)
    panel = BankPanel(
        bank_id="core-11 subset L20 (#142)",
        family="single_token_100p_core11",
        source_task_ids=[142],
        C=C_sub,
        cell_reduce="pair",
        cell_idx=cell_idx,
        centering_bank_idx=np.arange(len(CORE11_142)),  # CORE-11 subset is the universe
        use_similarity=True,  # verify_142 correlates centered cos (similarity)
        y=y,
        group=group,
        covar=np.zeros_like(y),
        has_covar=False,
        cv_unit="leave-one-source-out",
        gate_max_dev=0.0,  # verify_142 gate is statistic-level (rho), checked in main sanity
        rho_method="plain",
    )
    # Join sanity: core-11-bank centered rho at L20 reproduces #142 published 0.567.
    x_cen = cosine_predictor(panel, panel.centering_bank_idx, "global_mean")
    panel.join_sanity = {
        "centered_core11_spearman_L20": float(spearman(x_cen, y)[0]),
        "published_rho_L20": 0.567,
    }
    return panel


def assemble_20bank_405(data_root: Path) -> BankPanel:
    """#405 - 20-bank L20; cell = (held_persona, K-subset, seed) CORE rows;
    predictor = min over positives of (1 - cos(held, positive)); Y = deltaLogP_mean.
    No length covariate (MixedLM design). group = held persona."""
    fam = family_20bank(data_root)
    names = fam["names"]
    idx = {n: i for i, n in enumerate(names)}
    C = fam_centroids_20bank(data_root)
    cell_idx, y, group = [], [], []
    held_to_g: dict[str, int] = {}
    with (
        data_root / "eval_results" / "issue_405" / "aggregate" / "per_cell_persona_tidy.csv"
    ).open() as f:
        for r in csv.DictReader(f):
            if r["track"] != "CORE":
                continue
            positives = list(ast.literal_eval(r["positives"]))
            held = r["held_persona"]
            cell_idx.append((idx[held], [idx[p] for p in positives]))
            y.append(float(r["deltaLogP_mean"]))
            g = held_to_g.setdefault(held, len(held_to_g))
            group.append(g)
    y = np.asarray(y, dtype=np.float64)
    group = np.asarray(group, dtype=int)
    assert len(y) == 336, f"#405 CORE rows = {len(y)}, expected 336"
    panel = BankPanel(
        bank_id="20-bank L20 (#405)",
        family="extraction_method_a_L20",
        source_task_ids=[405],
        C=C,
        cell_reduce="min_over_pos",
        cell_idx=cell_idx,
        centering_bank_idx=np.arange(len(names)),
        use_similarity=False,
        y=y,
        group=group,
        covar=np.zeros_like(y),
        has_covar=False,
        cv_unit="leave-one-held-persona-out",
        gate_max_dev=float(fam["gate"]["max_abs_dev"]),
    )
    x_raw = cosine_predictor(panel, panel.centering_bank_idx, "none")
    panel.join_sanity = {"raw_pooled_spearman": float(spearman(x_raw, y)[0])}
    return panel


def fam_centroids_20bank(data_root: Path) -> np.ndarray:
    base = data_root / "eval_results" / "extraction_method_comparison"
    bundle = torch.load(base / "centroids_method_a.pt", map_location="cpu", weights_only=True)
    return bundle["layer_20"].to(torch.float32).numpy().astype(np.float64)


def assemble_111bank_478(data_root: Path) -> BankPanel:
    """#478 - 111-bank L20; cell = (held_persona, K-subset, seed) from the
    i478 tidy snapshot; predictor = min over positives of (1 - cos(held, positive));
    Y = deltaLogP_mean. No length covariate. group = held persona."""
    import pandas as pd

    fam = family_111bank(data_root)
    names = fam["names"]
    idx = {n: i for i, n in enumerate(names)}
    C = (
        torch.load(
            data_root
            / "eval_results"
            / "single_token_100_persona"
            / "centroids"
            / "centroids_layer20.pt",
            map_location="cpu",
            weights_only=True,
        )
        .to(torch.float32)
        .numpy()
        .astype(np.float64)
    )
    snap = data_root / "eval_results" / "issue_536" / "inputs" / "i478_tidy_69b34b94.csv"
    df = pd.read_csv(snap)
    assert len(df) == 2800, f"#478 tidy rows = {len(df)}, expected 2800"
    cell_idx, y, group = [], [], []
    held_to_g: dict[str, int] = {}
    for _, r in df.iterrows():
        subs = str(r["positives"]).split(";")
        held = r["held_out_persona"]
        cell_idx.append((idx[held], [idx[s] for s in subs]))
        y.append(float(r["deltaLogP_mean"]))
        g = held_to_g.setdefault(held, len(held_to_g))
        group.append(g)
    y = np.asarray(y, dtype=np.float64)
    group = np.asarray(group, dtype=int)
    panel = BankPanel(
        bank_id="111-bank L20 (#478)",
        family="single_token_100p_L20",
        source_task_ids=[478],
        C=C,
        cell_reduce="min_over_pos",
        cell_idx=cell_idx,
        centering_bank_idx=np.arange(len(names)),
        use_similarity=False,
        y=y,
        group=group,
        covar=np.zeros_like(y),
        has_covar=False,
        cv_unit="leave-one-held-persona-out",
        gate_max_dev=float(fam["gate"]["max_abs_dev"]),
    )
    x_raw = cosine_predictor(panel, panel.centering_bank_idx, "none")
    panel.join_sanity = {"raw_pooled_spearman": float(spearman(x_raw, y)[0])}
    return panel


def assemble_111bank_490(data_root: Path) -> BankPanel:
    """#490 - 111-bank L20; cell = (persona, A, B, seed) on/off-axis rows;
    predictor = mean_d = 0.5*(d_A + d_B); Y = dose-matched gap
    (shared - 0.5*(pooled_A + pooled_B)). No length covariate. group = persona."""
    import pandas as pd

    fam = family_111bank(data_root)
    names = fam["names"]
    idx = {n: i for i, n in enumerate(names)}
    C = (
        torch.load(
            data_root
            / "eval_results"
            / "single_token_100_persona"
            / "centroids"
            / "centroids_layer20.pt",
            map_location="cpu",
            weights_only=True,
        )
        .to(torch.float32)
        .numpy()
        .astype(np.float64)
    )
    pl = pd.read_csv(data_root / "eval_results" / "issue_490" / "aggregate" / "persona_level.csv")
    pl = pl[pl["subpanel"].isin(["on_axis", "off_axis"])].copy()
    # Reconstruct the dose-matched gap per (pair, seed, persona) - same logic as
    # regrade_490._fit: y = shared_2D - 0.5*(pooled_2D_A + pooled_2D_B).
    piv: dict[tuple, dict] = {}
    for _, r in pl.iterrows():
        key = (r["pair_id"], int(r["seed"]), r["persona"])
        if key not in piv:
            piv[key] = {"persona": r["persona"], "A": r["A"], "B": r["B"], "conds": {}}
        piv[key]["conds"][r["condition"]] = float(r["deltaLogP_mean"])
    cell_idx, y, group = [], [], []
    persona_to_g: dict[str, int] = {}
    for rec in piv.values():
        c = rec["conds"]
        has = [k for k in c if k.startswith("shared_2D")] and all(
            any(k.startswith(p) for k in c) for p in ("pooled_2D_A", "pooled_2D_B")
        )
        if not has:
            continue
        shared = next(c[k] for k in c if k.startswith("shared_2D"))
        pA = next(c[k] for k in c if k.startswith("pooled_2D_A"))
        pB = next(c[k] for k in c if k.startswith("pooled_2D_B"))
        cell_idx.append((idx[rec["persona"]], idx[rec["A"]], idx[rec["B"]]))
        y.append(shared - 0.5 * (pA + pB))
        g = persona_to_g.setdefault(rec["persona"], len(persona_to_g))
        group.append(g)
    y = np.asarray(y, dtype=np.float64)
    group = np.asarray(group, dtype=int)
    panel = BankPanel(
        bank_id="111-bank L20 (#490)",
        family="single_token_100p_L20",
        source_task_ids=[490],
        C=C,
        cell_reduce="two_pair_mean",
        cell_idx=cell_idx,
        centering_bank_idx=np.arange(len(names)),
        use_similarity=False,
        y=y,
        group=group,
        covar=np.zeros_like(y),
        has_covar=False,
        cv_unit="leave-one-persona-out",
        gate_max_dev=float(fam["gate"]["max_abs_dev"]),
    )
    x_raw = cosine_predictor(panel, panel.centering_bank_idx, "none")
    panel.join_sanity = {"raw_pooled_spearman": float(spearman(x_raw, y)[0])}
    return panel


def assemble_n24_380(data_root: Path) -> BankPanel:
    """#380 - n24 L15; cell = persona; predictor = mean pairwise centered DISTANCE
    to all others; Y = source_rate; covariate = log_tokens. group = persona
    (LOO N=24). rho_method = verify_380's rank-residualize (rank both, OLS
    rank(value) ~ rank(log_tokens), Spearman of the rank residuals) — NOT the
    (rxy - rx*ry)/sqrt(...) form, which gives 0.0875 vs the published 0.1113."""
    fam = family_n24(data_root, layer=15)
    names = fam["names"]
    idx = {n: i for i, n in enumerate(names)}
    bundle = torch.load(
        data_root / "eval_results" / "issue_274" / "centroids" / "centroids_n24_layers0_27.pt",
        map_location="cpu",
        weights_only=False,
    )
    layer_dict = bundle[15]
    C = torch.stack([layer_dict[n].to(torch.float32) for n in names]).numpy().astype(np.float64)
    pub = json.loads(
        (
            data_root / "eval_results" / "issue_380" / "cosine_pairwise_n24" / "correlation.json"
        ).read_text()
    )
    rows = pub["rows"]
    cell_idx, y, covar, group = [], [], [], []
    for gi, r in enumerate(rows):
        cell_idx.append(idx[r["persona"]])
        y.append(float(r["source_rate"]))
        covar.append(float(r["log_tokens"]))
        group.append(gi)  # one persona per fold
    y = np.asarray(y, dtype=np.float64)
    covar = np.asarray(covar, dtype=np.float64)
    group = np.asarray(group, dtype=int)
    panel = BankPanel(
        bank_id="n24 L15 (#380)",
        family=fam["family"],
        source_task_ids=[380],
        C=C,
        cell_reduce="mean_to_others",
        cell_idx=cell_idx,
        centering_bank_idx=np.arange(len(names)),
        use_similarity=False,
        y=y,
        group=group,
        covar=covar,
        has_covar=True,
        cv_unit="leave-one-persona-out",
        gate_max_dev=0.0,  # verify_380 value gate is centered-distance specific; sanity below
        rho_method="rank_residual",
    )
    # Join sanity: rank-residual centered rho reproduces #380's published value.
    panel.join_sanity = {
        "centered_rank_residual_rho": float(_insample_rho(panel, "global_mean")),
        "published_rank_residual_rho": float(pub["length_partial_spearman"]["rho"]),
    }
    return panel


def assemble_n24_pred_396_415(data_root: Path) -> BankPanel:
    """#396/#415 - n24 L15; cell = source; predictor = 1 - cos(source,
    helpful_assistant) [headline surface = cos_to_assistant]; Y =
    logp_end_of_response_diagonal_mean; covariate = inherited-prompt length
    (length-partialled). group = source. #415 corroborates #396 (one bank)."""
    fam = family_n24(data_root, layer=15)
    names = fam["names"]
    idx = {n: i for i, n in enumerate(names)}
    bundle = torch.load(
        data_root / "eval_results" / "issue_274" / "centroids" / "centroids_n24_layers0_27.pt",
        map_location="cpu",
        weights_only=False,
    )
    layer_dict = bundle[15]
    C = torch.stack([layer_dict[n].to(torch.float32) for n in names]).numpy().astype(np.float64)
    preds = json.loads(
        (data_root / "eval_results" / "issue_415" / "base_model_predictors_v2.json").read_text()
    )
    summary_396 = json.loads(
        (data_root / "eval_results" / "issue_396" / "analysis_summary.json").read_text()
    )
    per_src = {row["source"]: row for row in summary_396["per_source_aggregation"]}
    sources = sorted(s for s in per_src if s in preds["predictor_1_cosine_to_assistant_L15"])
    assert set(sources) <= set(names), sorted(set(sources) - set(names))
    from analyze_length_rate_n48 import get_inherited_prompt

    ref_idx = idx["helpful_assistant"]
    headline = "logp_end_of_response_diagonal_mean"
    cell_idx, y, covar, group = [], [], [], []
    for gi, s in enumerate(sources):
        cell_idx.append((idx[s], ref_idx))
        y.append(float(per_src[s][headline]))
        covar.append(float(len(get_inherited_prompt(s))))
        group.append(gi)  # one source per fold
    y = np.asarray(y, dtype=np.float64)
    covar = np.asarray(covar, dtype=np.float64)
    group = np.asarray(group, dtype=int)
    panel = BankPanel(
        bank_id="n24-predictor L15 (#396/#415, headline surface)",
        family=fam["family"],
        source_task_ids=[396, 415],
        C=C,
        cell_reduce="single_ref",
        cell_idx=cell_idx,
        centering_bank_idx=np.arange(len(names)),
        use_similarity=True,  # verify_415 headline correlates cos-to-assistant (similarity)
        y=y,
        group=group,
        covar=covar,
        has_covar=True,
        cv_unit="leave-one-source-out",
        gate_max_dev=0.0,  # statistic-level sanity below (raw headline rho vs #536)
        rho_method="length_partial",  # (rxy - rx*ry)/sqrt(...) — verify_415's family
    )
    # Join sanity: raw cos-to-assistant length-partial headline rho reproduces
    # #536's published headline (+0.018 on logp_end_of_response).
    panel.join_sanity = {
        "raw_length_partial_headline_rho": float(_insample_rho(panel, "none")),
        "published_headline_rho_partial": 0.018,
    }
    return panel


def assemble_19bank_311(data_root: Path) -> BankPanel:
    """#311 - 19-bank L20; cell = bystander; predictor = d_mid = 1 - cos(bystander,
    midpoint(A, B)) on CENTERED vectors; Y = r_p_primary_per_persona; covariate =
    s(p) = 0.5*(cos(p,A) + cos(p,B)) [NOT length]. group = bystander (LOO N=17).

    Per the statistics-critic note: #311's partial covariate is `s` (a geometry
    covariate from the centered vectors), so the residualisation uses `s`, not
    length - matching verify_311's headline (value-residualise d_mid and r_p on
    s, then rank-correlate). For the join sanity check we reproduce the centered
    headline partial rho (-0.348)."""
    import torch.nn.functional as F

    base = data_root / "eval_results" / "issue_311"
    b = torch.load(base / "centroids_base.pt", map_location="cpu", weights_only=False)
    pair = json.loads((base / "pair_selection.json").read_text())
    a_name, b_name = pair["A"], pair["B"]
    an = json.loads((base / "analysis.json").read_text())
    bys = an["bystanders"]
    r_p = np.asarray(an["r_p_primary_per_persona"], dtype=np.float64)
    # C = the FULL 19-persona bank of RAW centroids, so global_mean centering over
    # the full bank reproduces the bundle's centroids_centered (verify_311 confirms
    # cos_mc == normalize(centroids_centered) @ .T at 1e-4). The midpoint reduction
    # is built on the centered-but-UN-normalised rows (midpoint_unnormalised=True),
    # matching verify_311's mid = 0.5*(cenA + cenB) before the cosine re-normalises.
    personas = list(b["personas"])
    raw20 = b["centroids_raw"][20]
    C = torch.stack([raw20[p].to(torch.float32) for p in personas]).numpy().astype(np.float64)
    pidx = {p: i for i, p in enumerate(personas)}
    cell_idx, y, group = [], [], []
    for gi, p in enumerate(bys):
        cell_idx.append((pidx[p], pidx[a_name], pidx[b_name]))
        y.append(float(r_p[gi]))
        group.append(gi)  # one bystander per fold
    y = np.asarray(y, dtype=np.float64)
    group = np.asarray(group, dtype=int)
    # covariate s(p) = 0.5*(cos(p,A) + cos(p,B)) on the CENTERED bundle vectors
    # (verify_311's `s`, computed from centroids_centered with F.cosine_similarity).
    cen20 = b["centroids_centered"][20]
    cenA = cen20[a_name].to(torch.float32)
    cenB = cen20[b_name].to(torch.float32)

    def _cos(u: torch.Tensor, v: torch.Tensor) -> float:
        return float(F.cosine_similarity(u.unsqueeze(0), v.unsqueeze(0)).item())

    s = np.array(
        [
            0.5 * (_cos(cen20[p].to(torch.float32), cenA) + _cos(cen20[p].to(torch.float32), cenB))
            for p in bys
        ],
        dtype=np.float64,
    )
    panel = BankPanel(
        bank_id="19-bank L20 (#311)",
        family="issue311_19bank_L20",
        source_task_ids=[311],
        C=C,
        cell_reduce="midpoint",
        cell_idx=cell_idx,
        centering_bank_idx=np.arange(len(personas)),  # full 19-bank centering universe
        use_similarity=False,
        y=y,
        group=group,
        covar=s,  # #311 covariate is `s`, NOT length (statistics-critic note)
        has_covar=True,
        cv_unit="leave-one-bystander-out",
        gate_max_dev=0.0,
        rho_method="value_residual",  # verify_311: residualise VALUES on s, then Spearman
        midpoint_unnormalised=True,  # verify_311 averages A,B BEFORE re-normalising
    )
    # Join sanity: centered d_mid value-residualised on s, rank-correlated with
    # r_p value-residualised on s - reproduces #311 published headline -0.348.
    panel.join_sanity = {
        "centered_partial_rho_given_s": float(_insample_rho(panel, "global_mean")),
        "published_headline_rho": -0.348,
    }
    return panel


def assemble_505(data_root: Path) -> BankPanel:
    """#505 - PV bank L21; cell = (b, j_i, seed); predictor = cos(b, j_i)
    (SIMILARITY, not distance - #505's regression is on cos, not 1-cos);
    Y = delta_leakage. No length covariate. group = bystander b."""
    fam = family_505(data_root, layer=21)
    names = fam["names"]
    idx = {n: i for i, n in enumerate(names)}
    # Re-load the centroid tensor (family_505 returns matrices, not C).
    local = data_root / "data" / "issue_505" / "centroids_pv" / "centroids_pv_L21.pt"
    if local.exists():
        path = local
    else:
        from huggingface_hub import hf_hub_download

        path = Path(
            hf_hub_download(
                "superkaiba1/explore-persona-space-data",
                "issue505_loo_contrastive/geometry/centroids_pv_L21.pt",
                repo_type="dataset",
            )
        )
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    C = bundle["centroids"].to(torch.float32).numpy().astype(np.float64)
    rows = json.loads(
        (
            data_root / "eval_results" / "issue_505" / "analysis" / "delta_leakage_per_seed.json"
        ).read_text()
    )["rows"]
    cell_idx, y, group = [], [], []
    b_to_g: dict[str, int] = {}
    for r in rows:
        cell_idx.append((idx[r["b"]], idx[r["j_i"]]))
        y.append(float(r["delta_leakage"]))
        g = b_to_g.setdefault(r["b"], len(b_to_g))
        group.append(g)  # bystander b
    y = np.asarray(y, dtype=np.float64)
    group = np.asarray(group, dtype=int)
    panel = BankPanel(
        bank_id="505 PV L21 (#505)",
        family="issue505_pv_L21",
        source_task_ids=[505],
        C=C,
        cell_reduce="pair",
        cell_idx=cell_idx,
        centering_bank_idx=np.arange(len(names)),
        use_similarity=True,  # #505 regresses on cos(b, j), NOT distance
        y=y,
        group=group,
        covar=np.zeros_like(y),
        has_covar=False,
        cv_unit="leave-one-bystander-out",
        gate_max_dev=float(fam["gate"]["max_abs_dev"]),
    )
    x_raw = cosine_predictor(panel, panel.centering_bank_idx, "none")
    panel.join_sanity = {"raw_pooled_spearman": float(spearman(x_raw, y)[0])}
    return panel


BANK_BUILDERS = [
    assemble_100p_marker,
    assemble_core11_142,
    assemble_20bank_405,
    assemble_111bank_478,
    assemble_111bank_490,
    assemble_n24_380,
    assemble_n24_pred_396_415,
    assemble_19bank_311,
    assemble_505,
]
# bank_id substring -> builder, for --only.
BUILDER_BY_KEY = {
    "66": assemble_100p_marker,
    "142": assemble_core11_142,
    "405": assemble_20bank_405,
    "478": assemble_111bank_478,
    "490": assemble_111bank_490,
    "380": assemble_n24_380,
    "396": assemble_n24_pred_396_415,
    "311": assemble_19bank_311,
    "505": assemble_505,
}


# ──────────────────────────────────────────────────────────────────────────
# Table writer (checkpoint-per-row, mirroring #536) + CSV mirror.
# ──────────────────────────────────────────────────────────────────────────
_TABLE_COLS = [
    "bank_id",
    "family",
    "source_task_ids",
    "cv_unit",
    "n_cells",
    "n_groups",
    "join_gate_max_dev",
    "cv_r2_raw",
    "cv_r2_centered",
    "delta_cv_r2",
    "boot_delta_r2_point",
    "boot_delta_r2_ci95",
    "boot_delta_rho_point",
    "boot_delta_rho_ci95",
    "boot_n_boot_kept_r2",
    "boot_n_boot_kept_rho",
    "rho_raw",
    "rho_centered",
    "delta_rho",
    "rho_regime",
    "cv_centering_regime",
    "ss_tot_definition",
    "n_folds_skipped_raw",
    "n_folds_skipped_centered",
    "delta_cv_r2_sign",
    "contributes_to_h_verdict",
    "exclusion_reason",
]


def write_table(out_json: Path, rows: list[dict], meta: dict) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {**meta, "rows": rows, "updated_at": _now()}
    tmp = out_json.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=float))
    tmp.replace(out_json)
    # CSV mirror (flatten list-valued cells to a JSON string).
    csv_path = out_json.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_TABLE_COLS)
        for r in rows:
            w.writerow(
                [
                    json.dumps(r[c]) if isinstance(r.get(c), (list, dict)) else r.get(c)
                    for c in _TABLE_COLS
                ]
            )
    log.info("[table] %d rows -> %s (+ .csv)", len(rows), out_json.name)


# ──────────────────────────────────────────────────────────────────────────
# Forest figure (hero).
# ──────────────────────────────────────────────────────────────────────────
def make_forest_figure(rows: list[dict], fig_dir: Path, meta: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from explore_persona_space.analysis.paper_plots import set_paper_style

        set_paper_style("generic")
    except Exception as e:  # paper style is cosmetic; never block the figure
        log.warning("paper_plots style unavailable (%s); using matplotlib defaults", e)

    fig_dir.mkdir(parents=True, exist_ok=True)

    def _forest(rows_in, key_point, key_ci, fname, xlabel):
        order = list(reversed(rows_in))  # top row = first bank
        ys = np.arange(len(order))
        fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(order) + 1.3))
        for yi, r in zip(ys, order, strict=True):
            pt = r[key_point]
            ci = r[key_ci]
            contributes = r["contributes_to_h_verdict"]
            determinate = (
                contributes
                and ci is not None
                and np.isfinite(ci[0])
                and np.isfinite(ci[1])
                and not (ci[0] <= 0.0 <= ci[1])
            )
            filled = bool(determinate)
            if pt is None or not np.isfinite(pt):
                continue
            lo, hi = ci if ci is not None else (pt, pt)
            ax.errorbar(
                pt,
                yi,
                xerr=[[max(0.0, pt - lo)], [max(0.0, hi - pt)]],
                fmt="o",
                color="#1f77b4",
                markerfacecolor=("#1f77b4" if filled else "white"),
                markeredgecolor="#1f77b4",
                capsize=3,
                markersize=7,
                lw=1.4,
            )
        ax.axvline(0.0, color="0.4", lw=1.0, ls="--")
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{r['bank_id']}  (n_groups={r['n_groups']})" for r in order])
        ax.set_xlabel(xlabel)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(fig_dir / f"{fname}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        log.info("[figure] %s.png", fname)

    _forest(
        rows,
        "boot_delta_r2_point",
        "boot_delta_r2_ci95",
        "forest_centered_vs_raw_delta_cvR2",
        r"$\Delta R^2 = $ CV $R^2_{centered} - $ CV $R^2_{raw}$ (paired-bootstrap 95% CI)",
    )
    _forest(
        rows,
        "boot_delta_rho_point",
        "boot_delta_rho_ci95",
        "forest_centered_vs_raw_delta_rho",
        r"$\Delta\rho = \rho_{centered} - \rho_{raw}$ (in-sample, paired-bootstrap 95% CI)",
    )
    meta_path = fig_dir / "meta.json"
    meta_path.write_text(
        json.dumps(
            {
                **meta,
                "figures": [
                    "forest_centered_vs_raw_delta_cvR2.png",
                    "forest_centered_vs_raw_delta_rho.png",
                ],
            },
            indent=2,
            default=float,
        )
    )


# ──────────────────────────────────────────────────────────────────────────
# Per-bank loop.
# ──────────────────────────────────────────────────────────────────────────
def process_panel(panel: BankPanel) -> dict:
    r2_raw, sk_raw, contrib_raw = logo_cv_r2(panel, "none")
    r2_cen, sk_cen, contrib_cen = logo_cv_r2(panel, "global_mean")
    rho_raw = _insample_rho(panel, "none")
    rho_cen = _insample_rho(panel, "global_mean")
    boot = paired_bootstrap_delta(panel, contrib_raw, contrib_cen)
    n_grp = int(np.unique(panel.group).size)
    n_skip_frac = max(sk_raw, sk_cen) / n_grp if n_grp else 1.0

    # MF4 both-negative precedence is checked FIRST, then MF3 low-N, then degenerate.
    both_fail = np.isfinite(r2_raw) and np.isfinite(r2_cen) and r2_raw < 0 and r2_cen < 0
    lown = n_grp <= LOWN_VERDICT_FLOOR
    degraded = n_skip_frac > DEGEN_FOLD_FRAC
    exclusion = (
        "both_predictors_fail_oos"
        if both_fail
        else "low_n_groups<=5"
        if lown
        else "degenerate_folds>25pct"
        if degraded
        else None
    )
    contributes = exclusion is None
    delta_r2 = (r2_cen - r2_raw) if (np.isfinite(r2_raw) and np.isfinite(r2_cen)) else None
    return {
        "bank_id": panel.bank_id,
        "family": panel.family,
        "source_task_ids": panel.source_task_ids,
        "cv_unit": panel.cv_unit,
        "n_cells": int(panel.y.size),
        "n_groups": n_grp,
        "join_gate_max_dev": panel.gate_max_dev,
        "join_sanity": panel.join_sanity,
        "cv_r2_raw": r2_raw,
        "cv_r2_centered": r2_cen,
        "delta_cv_r2": delta_r2,
        **{f"boot_{k}": v for k, v in boot.items()},
        "rho_raw": rho_raw,
        "rho_centered": rho_cen,
        "delta_rho": (rho_cen - rho_raw)
        if (np.isfinite(rho_raw) and np.isfinite(rho_cen))
        else None,
        "rho_regime": "transductive_in_sample_bank_global_centering",  # MF5 label
        "cv_centering_regime": "train_fold_only",  # MF5 label
        "ss_tot_definition": "per_fold_train_mean_baseline",  # MF4 pin
        "n_folds_skipped_raw": sk_raw,
        "n_folds_skipped_centered": sk_cen,
        "delta_cv_r2_sign": int(np.sign(delta_r2)) if (contributes and delta_r2 is not None) else 0,
        "contributes_to_h_verdict": contributes,  # MF3/MF4
        "exclusion_reason": exclusion,  # MF3/MF4
        "computed_at": _now(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Task #648 centered-vs-raw predictive-skill driver (see module docstring).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        default=REPO,
        help="Checkout holding the untracked input tensors (default: this checkout). "
        "Point this at the main repo root when running from a sparse worktree.",
    )
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR, help="eval_results/issue_648 dir.")
    ap.add_argument("--fig-dir", type=Path, default=FIG_DIR, help="figures/issue_648 dir.")
    ap.add_argument(
        "--only", default=None, help=f"Run a single bank; one of {sorted(BUILDER_BY_KEY)}."
    )
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=skill] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    data_root = args.data_root.resolve()
    ensure_bank111_json(data_root)  # §4.5 prerequisite restore (idempotent)

    aset = recoverable_banks(data_root)
    builders = [BUILDER_BY_KEY[args.only]] if args.only else list(BANK_BUILDERS)
    log.info("[assemble] %d bank(s)", len(builders))
    panels: list[BankPanel] = []
    for build in builders:
        panel = build(data_root)  # each raises if the #536 join gate fails
        log.info(
            "[assembled] %s n_cells=%d n_groups=%d join_sanity=%s",
            panel.bank_id,
            panel.y.size,
            np.unique(panel.group).size,
            panel.join_sanity,
        )
        panels.append(panel)

    # MF1 mechanical accounting gate (only meaningful on a full run).
    if not args.only:
        assert_recoverable_set_accounted(panels, aset)

    # Per-bank join SANITY check (statistics-critic note): the recomputed in-sample
    # rho must reproduce #536's published number (|drho| <= GATE_RHO_TOL=0.02), using
    # #536's EXACT per-bank statistic, before any skill is read. Each verifiable bank
    # stores a (recomputed_key, published_key) pair in join_sanity; banks whose join
    # gate is the matrix 1e-4 check inside the family loader (#405/#478/#490/#505)
    # carry only an informational raw_pooled_spearman, no published-rho compare.
    _SANITY_PAIRS = [
        ("centered_pooled_spearman", "published_pooled_rho"),  # #66
        ("centered_core11_spearman_L20", "published_rho_L20"),  # #142
        ("centered_rank_residual_rho", "published_rank_residual_rho"),  # #380
        ("raw_length_partial_headline_rho", "published_headline_rho_partial"),  # #396
        ("centered_partial_rho_given_s", "published_headline_rho"),  # #311
    ]
    for panel in panels:
        js = panel.join_sanity
        for got_key, want_key in _SANITY_PAIRS:
            if got_key in js and want_key in js:
                got, want = js[got_key], js[want_key]
                assert abs(got - want) <= GATE_RHO_TOL, (
                    f"{panel.bank_id} join sanity FAILED ({got_key}): recomputed "
                    f"{got:.4f} vs #536 published {want} (|drho| > {GATE_RHO_TOL})"
                )

    meta = {
        "schema_version": "i648_predictive_skill_v1",
        "generated_at": _now(),
        "git_commit": _git_sha(REPO),
        "data_root_commit": _git_sha(data_root),
        "bank111_restore_sha": BANK111_RESTORE_SHA,
        "rng_seed": RNG_SEED,
        "n_boot": N_BOOT,
        "var_floor": VAR_FLOOR,
        "lown_verdict_floor": LOWN_VERDICT_FLOOR,
        "degen_fold_frac": DEGEN_FOLD_FRAC,
        "pin": "per-bank ONLY, never pooled across banks (#536 pin)",
        "names_hash": _names_hash([p.bank_id for p in panels]),
    }

    out_json = args.out_dir / "per_bank_skill_table.json"
    rows: list[dict] = []
    for panel in panels:
        row = process_panel(panel)
        rows.append(row)
        write_table(out_json, rows, meta)  # checkpoint-per-row
        log.info(
            "[row] %s  dR2=%.4f CI=%s  contributes=%s reason=%s",
            panel.bank_id,
            (row["delta_cv_r2"] or float("nan")),
            row["boot_delta_r2_ci95"],
            row["contributes_to_h_verdict"],
            row["exclusion_reason"],
        )

    # MF3 contract assertion: no n_groups<=5 row may claim verdict eligibility.
    for r in rows:
        if r["n_groups"] <= LOWN_VERDICT_FLOOR and r["contributes_to_h_verdict"]:
            raise RuntimeError(
                f"CONTRACT VIOLATION: {r['bank_id']} n_groups<=5 but "
                f"contributes_to_h_verdict=True (MF3). Aborting."
            )

    make_forest_figure(rows, args.fig_dir, meta)
    log.info("[done] %d bank(s) -> %s", len(rows), out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
