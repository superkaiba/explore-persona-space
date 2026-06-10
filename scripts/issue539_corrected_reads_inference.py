"""Task #539 follow-up — uncertainty for the corrected (FE / partial / marginal) reads.

Free-analysis, CPU-only, additive. The committed #539 analysis
(``eval_results/issue_539/residual_per_cohort.json``) carries three families of
source-and-context-corrected point estimates that drive its re-attribution —
``rho_twoway`` (exact two-way source+bystander FE), ``rho_partial_source_dose``
(rank partial controlling the per-source mean emission), and
``source_marginal`` (n=16 or 13 source-level pairs) — but only the
RESIDUAL-read family shipped with bootstrap CIs + permutation p. This script
adds, for every corrected read in the committed output (7 slices x 2 primary
predictors):

1. ``rho_twoway``: 10,000-rep cell-level percentile bootstrap that RE-RUNS the
   exact two-way FE residualization (DV and geometry) per resample; 2,000-rep
   source-cluster and bystander-cluster bootstraps (drawn cluster copies
   relabeled as distinct groups so the FE re-estimation does not merge
   duplicates); and an FE-respecting permutation p (the geometry FE-residuals
   are permuted across cells against the fixed DV FE-residuals — ter
   Braak-style residual permutation, which preserves the FE null because both
   sides have already had the source and bystander structure projected out;
   10,000 reps, two-sided, add-one formula).
2. ``rho_partial_source_dose``: same bootstrap machinery, with the source-dose
   control RECOMPUTED inside every resample (propagating its estimation
   uncertainty); permutation p via the standard residual-permutation approach
   on ranks (Anderson & Legendre 1999 / Freedman-Lane): rank-residualize the
   predictor and the DV on the rank of the control ONCE, then permute the
   predictor's rank-residuals against the fixed DV rank-residuals (10,000
   reps, two-sided, add-one).
3. ``source_marginal``: 2,000-rep percentile bootstrap over the source-level
   (mean geometry, mean emission) pairs + a 10,000-rep MC permutation of the
   source labels (16! ~ 2e13 >> n_perm, so Monte-Carlo, two-sided, add-one) —
   with the small n reported alongside every number.

Degenerate-resample policy inherited from the parent script: dropped AND
counted, never silently averaged. Seed 42, deterministic. Output is ADDITIVE
(``corrected_reads_inference.json``); ``residual_per_cohort.json`` is not
touched. A consistency gate recomputes every covered point estimate from the
rebuilt #532 panel and aborts (exit 1) on any mismatch with the committed
analysis, after re-running the parent's own step-0 gate.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import issue539_residual_per_cohort as base
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import rankdata

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

# ── Covered reads ────────────────────────────────────────────────────────────

# (slice_name, dv_key, mask_name) — every suite in the committed analysis that
# carries the corrected reads. dvB_ordinary is the ordinary cohort re-read on
# the graded appended-slot log-prob DV (the exploratory +0.473 cosine lead).
SLICES: tuple[tuple[str, str, str], ...] = (
    ("ordinary_cross", "emit_rate", "ordinary_cross"),
    ("instructed_strip", "emit_rate", "instructed_strip"),
    ("dvB_ordinary", "extra_marker_logp", "ordinary_cross"),
    ("nonstylized_ordinary_cross", "emit_rate", "nonstylized_ordinary_cross"),
    ("nonstylized_instructed_strip", "emit_rate", "nonstylized_instructed_strip"),
    ("nonstylized_strict_ordinary_cross", "emit_rate", "nonstylized_strict_ordinary_cross"),
    ("class_letter_cross", "emit_rate", "class_letter_cross"),
)

SLICE_DISPLAY = {
    "ordinary_cross": "Ordinary cross-context",
    "instructed_strip": "Instructed strip",
    "dvB_ordinary": "Ordinary, log-prob DV",
    "nonstylized_ordinary_cross": "Ordinary, drop stylized sources",
    "nonstylized_instructed_strip": "Instructed, drop stylized sources",
    "nonstylized_strict_ordinary_cross": "Ordinary, drop stylized both sides",
    "class_letter_cross": "Different class letter",
}

CHECK_TOL = 1e-9  # recomputed point estimate vs committed analysis JSON


def _committed_suite(committed: dict, slice_name: str) -> dict:
    """Locate the committed cohort suite for a covered slice."""
    if slice_name in ("ordinary_cross", "instructed_strip"):
        return committed["cohorts"][slice_name]
    rob = committed["robustness"]
    if slice_name == "dvB_ordinary":
        return rob["dvB_ordinary"]
    if slice_name == "nonstylized_ordinary_cross":
        return rob["nonstylized"]["ordinary_cross"]
    if slice_name == "nonstylized_instructed_strip":
        return rob["nonstylized"]["instructed_strip"]
    if slice_name == "nonstylized_strict_ordinary_cross":
        return rob["nonstylized_strict"]["ordinary_cross"]
    if slice_name == "class_letter_cross":
        return rob["class_letter_cross"]
    raise KeyError(slice_name)


# ── Fast helpers (equivalence-asserted against the parent implementations) ───


def _twoway_resid_pair(
    x: np.ndarray, y: np.ndarray, sc: np.ndarray, bc: np.ndarray, n_s: int, n_b: int
) -> tuple[np.ndarray, np.ndarray]:
    """Exact two-way FE residuals of x AND y in one lstsq (two RHS columns).

    Same dummy-regression estimand as ``base._twoway_fe_residualize`` (whose
    fail-loud postcondition guards the observed point estimates); this variant
    takes pre-factorized integer codes so the resampling loops skip the
    per-call ``np.unique``. Groups absent from a resample leave all-zero
    columns; ``lstsq(rcond=None)`` returns the least-norm solution and the
    residual is still the unique projection residual. Equivalence to the
    parent implementation is asserted on the observed data per slice.
    """
    n = len(x)
    design = np.zeros((n, 1 + n_s + n_b), dtype=np.float64)
    design[:, 0] = 1.0
    rows = np.arange(n)
    design[rows, 1 + sc] = 1.0
    design[rows, 1 + n_s + bc] = 1.0
    rhs = np.column_stack([x, y])
    coef, *_ = np.linalg.lstsq(design, rhs, rcond=None)
    resid = rhs - design @ coef
    return resid[:, 0], resid[:, 1]


def _rank_ols_residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Closed-form simple-OLS residual (used on rank vectors; never degenerate
    on ranks of a non-constant control)."""
    xc = x - x.mean()
    denom = float(xc @ xc)
    if denom < 1e-12:
        return y - y.mean()
    slope = float(xc @ (y - y.mean())) / denom
    return y - (y.mean() + slope * xc)


def _fast_partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Rank partial Spearman (closed-form OLS on ranks).

    Same estimand as ``base._partial_spearman`` (which uses np.polyfit);
    equivalence asserted on the observed data per slice. Returns NaN when a
    rank-residual is constant (degenerate)."""
    if base._is_degenerate(z):
        return base._spearman_rho(x, y)
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    rx_res = _rank_ols_residual(rx, rz)
    ry_res = _rank_ols_residual(ry, rz)
    if float(np.std(rx_res)) < 1e-12 or float(np.std(ry_res)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx_res, ry_res)[0, 1])


def _group_mean_by_code(values: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Per-group mean broadcast back to rows (bincount; absent groups never indexed)."""
    sums = np.bincount(codes, weights=values, minlength=n_groups)
    counts = np.bincount(codes, minlength=n_groups).clip(min=1)
    return (sums / counts)[codes]


def _percentile_summary(rhos: list[float], n_boot: int, n_degenerate: int) -> dict:
    if not rhos:
        return {
            "boot_mean": float("nan"),
            "low": float("nan"),
            "high": float("nan"),
            "n_boot": n_boot,
            "n_degenerate_resamples": n_degenerate,
        }
    arr = np.asarray(rhos)
    return {
        "boot_mean": float(np.mean(arr)),
        "low": float(np.percentile(arr, 2.5)),
        "high": float(np.percentile(arr, 97.5)),
        "n_boot": n_boot,
        "n_degenerate_resamples": n_degenerate,
    }


# ── Two-way FE inference ─────────────────────────────────────────────────────


def _cell_boot_twoway(
    x: np.ndarray, y: np.ndarray, sc: np.ndarray, bc: np.ndarray, n_boot: int, seed: int
) -> dict:
    """Cell-level percentile bootstrap; exact FE residualization re-run per resample."""
    rng = np.random.default_rng(seed)
    n = len(x)
    n_s, n_b = int(sc.max()) + 1, int(bc.max()) + 1
    rhos: list[float] = []
    n_deg = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xt, yt = _twoway_resid_pair(x[idx], y[idx], sc[idx], bc[idx], n_s, n_b)
        if base._is_degenerate(xt) or base._is_degenerate(yt):
            n_deg += 1
            continue
        rhos.append(base._fast_spearman(xt, yt))
    return _percentile_summary(rhos, n_boot, n_deg)


def _cluster_boot_twoway(
    x: np.ndarray,
    y: np.ndarray,
    src: np.ndarray,
    byst: np.ndarray,
    cluster_on: str,
    n_boot: int,
    seed: int,
) -> dict:
    """Cluster percentile bootstrap on rho_twoway, re-residualizing per resample.

    Drawn cluster copies are RELABELED as distinct groups on the resampled
    axis (fresh integer code per copy), so the FE dummy regression treats two
    draws of the same source/bystander as separate clusters instead of
    silently merging them."""
    rng = np.random.default_rng(seed)
    labels = src if cluster_on == "source" else byst
    other = byst if cluster_on == "source" else src
    uniq = np.unique(labels)
    idx_of = {c: np.where(labels == c)[0] for c in uniq}
    _, other_codes_full = np.unique(other, return_inverse=True)
    n_other = int(other_codes_full.max()) + 1
    rhos: list[float] = []
    n_deg = 0
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_of[c] for c in chosen])
        copy_codes = np.repeat(np.arange(len(chosen)), [len(idx_of[c]) for c in chosen])
        oc = other_codes_full[idx]
        if cluster_on == "source":
            xt, yt = _twoway_resid_pair(x[idx], y[idx], copy_codes, oc, len(chosen), n_other)
        else:
            xt, yt = _twoway_resid_pair(x[idx], y[idx], oc, copy_codes, n_other, len(chosen))
        if base._is_degenerate(xt) or base._is_degenerate(yt):
            n_deg += 1
            continue
        rhos.append(base._fast_spearman(xt, yt))
    out = _percentile_summary(rhos, n_boot, n_deg)
    out["n_clusters"] = len(uniq)
    return out


# ── Partial (source-dose) inference ──────────────────────────────────────────


def _cell_boot_partial(
    x: np.ndarray, y: np.ndarray, sc: np.ndarray, n_boot: int, seed: int
) -> dict:
    """Cell-level percentile bootstrap on the rank partial; the source-dose
    control is recomputed within every resample."""
    rng = np.random.default_rng(seed)
    n = len(x)
    n_s = int(sc.max()) + 1
    rhos: list[float] = []
    n_deg = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb, scb = x[idx], y[idx], sc[idx]
        zb = _group_mean_by_code(yb, scb, n_s)
        r = _fast_partial_spearman(xb, yb, zb)
        if np.isnan(r):
            n_deg += 1
            continue
        rhos.append(r)
    return _percentile_summary(rhos, n_boot, n_deg)


def _cluster_boot_partial(
    x: np.ndarray,
    y: np.ndarray,
    src: np.ndarray,
    byst: np.ndarray,
    cluster_on: str,
    n_boot: int,
    seed: int,
) -> dict:
    """Cluster percentile bootstrap on the rank partial (dose recomputed per
    resample; source-cluster copies relabeled so each draw is its own dose group)."""
    rng = np.random.default_rng(seed)
    labels = src if cluster_on == "source" else byst
    uniq = np.unique(labels)
    idx_of = {c: np.where(labels == c)[0] for c in uniq}
    _, src_codes_full = np.unique(src, return_inverse=True)
    n_src = int(src_codes_full.max()) + 1
    rhos: list[float] = []
    n_deg = 0
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_of[c] for c in chosen])
        xb, yb = x[idx], y[idx]
        if cluster_on == "source":
            scb = np.repeat(np.arange(len(chosen)), [len(idx_of[c]) for c in chosen])
            zb = _group_mean_by_code(yb, scb, len(chosen))
        else:
            zb = _group_mean_by_code(yb, src_codes_full[idx], n_src)
        r = _fast_partial_spearman(xb, yb, zb)
        if np.isnan(r):
            n_deg += 1
            continue
        rhos.append(r)
    out = _percentile_summary(rhos, n_boot, n_deg)
    out["n_clusters"] = len(uniq)
    return out


def _partial_residual_permutation_p(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, n_perm: int, seed: int
) -> dict:
    """Residual-permutation p for the rank partial (Anderson & Legendre 1999).

    Rank-residualize the predictor and the DV on the rank of the control ONCE,
    then permute the predictor's rank-residuals across cells against the fixed
    DV rank-residuals; Pearson per permutation; two-sided, add-one formula.
    Falls back to the plain permutation test when the control is constant
    (the partial degenerates to the plain Spearman there)."""
    if base._is_degenerate(z):
        out = base._permutation_p(x, y, n_perm, seed)
        out["method"] = "plain permutation (constant control)"
        return out
    rng = np.random.default_rng(seed)
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    rx_res = _rank_ols_residual(rx, rz)
    ry_res = _rank_ols_residual(ry, rz)
    obs = float(np.corrcoef(rx_res, ry_res)[0, 1])
    perms = rng.permuted(np.tile(rx_res, (n_perm, 1)), axis=1)
    perms -= perms.mean(axis=1, keepdims=True)
    yc = ry_res - ry_res.mean()
    num = perms @ yc
    den = np.sqrt((perms * perms).sum(axis=1) * (yc @ yc))
    null = num / den
    count = int((np.abs(null) >= abs(obs)).sum())
    return {
        "p": float((1 + count) / (n_perm + 1)),
        "rho_obs": obs,
        "null_mean": float(np.mean(null)),
        "null_sd": float(np.std(null)),
        "n_perm": n_perm,
        "method": "rank residual permutation (Anderson & Legendre 1999): predictor "
        "rank-residuals permuted against fixed DV rank-residuals",
    }


# ── Per-slice computation ────────────────────────────────────────────────────


def compute_slice_inference(
    panel: dict,
    mask: np.ndarray,
    dv_key: str,
    committed_suite: dict,
    args: argparse.Namespace,
    checks: list[dict],
    slice_name: str,
) -> dict:
    """All new inference blocks for one covered slice."""
    y = panel[dv_key][mask].astype(np.float64)
    src = panel["source_cid"][mask]
    byst = panel["bystander_label"][mask]
    src_u, sc = np.unique(src, return_inverse=True)
    byst_u, bc = np.unique(byst, return_inverse=True)

    def check(name: str, got: float, want: float, tol: float = CHECK_TOL) -> None:
        checks.append(
            {
                "name": f"{slice_name}/{name}",
                "got": float(got),
                "want": float(want),
                "pass": bool(abs(got - want) <= tol),
            }
        )

    check("n", len(y), committed_suite["n"], 0)

    out: dict = {
        "n": len(y),
        "dv": dv_key,
        "n_sources": len(src_u),
        "n_bystanders": len(byst_u),
        "predictors": {},
    }
    for pk in base.PRIMARY_PKS:
        x = panel[pk][mask].astype(np.float64)
        blk_committed = committed_suite["predictors"][pk]

        # Observed two-way FE residuals via the parent's fail-loud estimator;
        # the fast pair variant used inside the loops must agree exactly.
        x_tw, _ = base._twoway_fe_residualize(x, src, byst)
        y_tw, _ = base._twoway_fe_residualize(y, src, byst)
        x_tw_f, y_tw_f = _twoway_resid_pair(x, y, sc, bc, len(src_u), len(byst_u))
        max_drift = max(float(np.max(np.abs(x_tw - x_tw_f))), float(np.max(np.abs(y_tw - y_tw_f))))
        assert max_drift < 1e-8, f"fast two-way residual drift {max_drift!r} on {slice_name}/{pk}"

        rho_twoway = base._spearman_rho(x_tw, y_tw)
        check(f"{pk}/rho_twoway", rho_twoway, blk_committed["rho_twoway"])

        # Partial: parent reference vs the fast closed-form used in the loops.
        source_dose = _group_mean_by_code(y, sc, len(src_u))
        rho_partial_ref = base._partial_spearman(x, y, source_dose)
        rho_partial = _fast_partial_spearman(x, y, source_dose)
        assert abs(rho_partial - rho_partial_ref) < 1e-9, (
            f"fast partial drift on {slice_name}/{pk}: {rho_partial!r} vs {rho_partial_ref!r}"
        )
        check(
            f"{pk}/rho_partial_source_dose", rho_partial, blk_committed["rho_partial_source_dose"]
        )

        # Source marginal (n = n_sources pairs of row means).
        mean_geom = np.array([float(x[sc == k].mean()) for k in range(len(src_u))])
        mean_emit = np.array([float(y[sc == k].mean()) for k in range(len(src_u))])
        rho_marginal = base._spearman_rho(mean_geom, mean_emit)
        sm_committed = committed_suite["source_marginal"][pk]
        check(f"{pk}/source_marginal_rho", rho_marginal, sm_committed["rho"])
        check(f"{pk}/source_marginal_n", len(src_u), sm_committed["n_sources"], 0)

        # FE-respecting permutation: permute the geometry FE-residuals against
        # the fixed DV FE-residuals (rho is symmetric in its arguments).
        p_perm_twoway = base._permutation_p(y_tw, x_tw, args.n_perm, args.seed)
        p_perm_twoway["method"] = (
            "FE residual permutation: geometry FE-residuals permuted across cells "
            "against fixed DV FE-residuals (both sides projected on source + "
            "bystander dummies first)"
        )

        out["predictors"][pk] = {
            "rho_twoway": {
                "estimate": float(rho_twoway),
                "ci95_cell_boot": _cell_boot_twoway(x, y, sc, bc, args.n_boot, args.seed),
                "ci95_cluster_source": _cluster_boot_twoway(
                    x, y, src, byst, "source", args.n_cluster_boot, args.seed
                ),
                "ci95_cluster_bystander": _cluster_boot_twoway(
                    x, y, src, byst, "bystander", args.n_cluster_boot, args.seed
                ),
                "p_perm_fe": p_perm_twoway,
            },
            "rho_partial_source_dose": {
                "estimate": float(rho_partial),
                "ci95_cell_boot": _cell_boot_partial(x, y, sc, args.n_boot, args.seed),
                "ci95_cluster_source": _cluster_boot_partial(
                    x, y, src, byst, "source", args.n_cluster_boot, args.seed
                ),
                "ci95_cluster_bystander": _cluster_boot_partial(
                    x, y, src, byst, "bystander", args.n_cluster_boot, args.seed
                ),
                "p_perm_residual": _partial_residual_permutation_p(
                    x, y, source_dose, args.n_perm, args.seed
                ),
            },
            "source_marginal": {
                "estimate": float(rho_marginal),
                "n_sources": len(src_u),
                "ci95_boot_sources": base._bootstrap_spearman_ci(
                    mean_geom, mean_emit, args.n_marginal_boot, args.seed
                ),
                "p_perm": {
                    **base._permutation_p(mean_geom, mean_emit, args.n_perm, args.seed),
                    "method": f"MC permutation of the {len(src_u)} source labels "
                    f"({len(src_u)}! >> n_perm)",
                },
            },
        }
    return out


# ── Figure ───────────────────────────────────────────────────────────────────


def make_figure(reads: dict, fig_dir: Path) -> None:
    """Forest plot: every corrected read with its new 95% CI (cell-level
    bootstrap for the two FE-corrected reads; bootstrap over sources for the
    source marginal)."""
    set_paper_style("blog")
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = paper_palette(2)
    pk_color = dict(zip(base.PRIMARY_PKS, colors, strict=True))
    rows = [(s, pk) for s, _, _ in SLICES for pk in base.PRIMARY_PKS]
    ypos = np.arange(len(rows))
    labels = [f"{SLICE_DISPLAY[s]} — {base.PK_DISPLAY[pk]}" for s, pk in rows]

    panels = [
        ("rho_twoway", "ci95_cell_boot", "p_perm_fe", "Two-way FE ρ"),  # noqa: RUF001
        (
            "rho_partial_source_dose",
            "ci95_cell_boot",
            "p_perm_residual",
            "Partial ρ (source dose)",  # noqa: RUF001
        ),
        ("source_marginal", "ci95_boot_sources", "p_perm", "Source-marginal ρ"),  # noqa: RUF001
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 7.0), sharey=True)
    for ax, (read_key, ci_key, _p_key, title) in zip(axes, panels, strict=True):
        for yi, (s, pk) in zip(ypos, rows, strict=True):
            blk = reads[s]["predictors"][pk][read_key]
            ci = blk[ci_key]
            c = pk_color[pk]
            ax.plot([ci["low"], ci["high"]], [yi, yi], color=c, lw=1.4)
            ax.plot(blk["estimate"], yi, "o", ms=4.5, color=c)
        ax.axvline(0.0, color="0.4", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Spearman ρ")  # noqa: RUF001
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Corrected reads with 95% bootstrap CIs (cell-level, FE re-run per resample; "
        "source marginal bootstrapped over sources)",
        fontsize=9,
    )
    savefig_paper(fig, "explore_corrected_reads_inference", dir=fig_dir)
    plt.close(fig)
    print(f"[figures] wrote explore_corrected_reads_inference to {fig_dir}")


# ── Main ─────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task #539 follow-up: bootstrap CIs + permutation p for the corrected reads.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--in-dir", type=Path, default=Path("eval_results/issue_532"))
    parser.add_argument(
        "--committed", type=Path, default=Path("eval_results/issue_539/residual_per_cohort.json")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_539"))
    parser.add_argument("--fig-dir", type=Path, default=Path("figures/issue_539"))
    parser.add_argument("--n-boot", type=int, default=10_000, dest="n_boot")
    parser.add_argument("--n-cluster-boot", type=int, default=2_000, dest="n_cluster_boot")
    parser.add_argument("--n-marginal-boot", type=int, default=2_000, dest="n_marginal_boot")
    parser.add_argument("--n-perm", type=int, default=10_000, dest="n_perm")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = datetime.now(UTC)

    panel = base.build_panel(args.in_dir)
    step0 = base.step0_consistency(panel, args.in_dir)  # sys.exit(1) on mismatch
    masks = base.cohort_masks(panel)
    committed = json.loads(args.committed.read_text())

    checks: list[dict] = []
    reads: dict = {}
    for slice_name, dv_key, mask_name in SLICES:
        print(f"[compute] {slice_name} ({dv_key}) ...")
        reads[slice_name] = compute_slice_inference(
            panel,
            masks[mask_name],
            dv_key,
            _committed_suite(committed, slice_name),
            args,
            checks,
            slice_name,
        )

    failed = [c for c in checks if not c["pass"]]
    if failed:
        print(
            "CONSISTENCY GATE FAILED — recomputed corrected reads diverge from the "
            "committed residual_per_cohort.json. NOT writing inference output.",
            file=sys.stderr,
        )
        for c in failed:
            print(f"  FAIL {c['name']}: got {c['got']!r}, want {c['want']!r}", file=sys.stderr)
        return 1
    print(f"[consistency] {len(checks)} recomputed point estimates match the committed analysis")

    results = {
        "metadata": {
            "task": 539,
            "followup": "corrected-reads inference (free-analysis, Step 9a-ter)",
            "git_commit": base._git_commit(),
            "timestamp_utc": t0.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "platform": platform.platform(),
            "seed": args.seed,
            "n_boot": args.n_boot,
            "n_cluster_boot": args.n_cluster_boot,
            "n_marginal_boot": args.n_marginal_boot,
            "n_perm": args.n_perm,
            "in_dir": str(args.in_dir),
            "committed_analysis": str(args.committed),
            "committed_analysis_git_commit": committed["metadata"]["git_commit"],
            "argv": sys.argv[1:],
            "predictors": list(base.PRIMARY_PKS),
            "n_covered_reads": len(SLICES) * len(base.PRIMARY_PKS) * 3,
            "methods": {
                "twoway_cell_boot": "percentile bootstrap over cells; the exact two-way "
                "(source + bystander) FE dummy-regression residualization is re-run on "
                "every resample for both the DV and the geometry; degenerate resamples "
                "dropped + counted",
                "twoway_cluster_boot": "cluster percentile bootstrap (source / bystander); "
                "drawn cluster copies relabeled as distinct groups before the per-resample "
                "FE re-estimation; degenerate resamples dropped + counted",
                "twoway_permutation": "FE residual permutation: geometry FE-residuals "
                "permuted across cells against fixed DV FE-residuals; two-sided, add-one "
                "p = (1 + #{|rho_perm| >= |rho_obs|}) / (n_perm + 1)",
                "partial_cell_boot": "percentile bootstrap over cells; the source-dose "
                "control (per-source mean DV) is recomputed inside every resample; "
                "degenerate resamples dropped + counted",
                "partial_permutation": "rank residual permutation (Anderson & Legendre "
                "1999 / Freedman-Lane): predictor and DV rank-residualized on the rank "
                "of the control once, predictor rank-residuals permuted; two-sided, add-one",
                "source_marginal": "n_sources source-level (row-mean geometry, row-mean "
                "DV) pairs; percentile bootstrap resamples the pairs; MC permutation of "
                "the source labels (n_sources! >> n_perm); two-sided, add-one",
            },
        },
        "step0_consistency": step0,
        "consistency_with_committed": checks,
        "reads": reads,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "corrected_reads_inference.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"[write] {out_path}")

    make_figure(reads, args.fig_dir)

    wall = (datetime.now(UTC) - t0).total_seconds()
    for slice_name in ("ordinary_cross", "dvB_ordinary"):
        for pk in base.PRIMARY_PKS:
            blk = reads[slice_name]["predictors"][pk]["rho_twoway"]
            ci = blk["ci95_cell_boot"]
            print(
                f"[headline] {slice_name:14s} {pk:8s} rho_twoway={blk['estimate']:+.3f} "
                f"[{ci['low']:+.3f}, {ci['high']:+.3f}] p_perm_fe={blk['p_perm_fe']['p']:.4g}"
            )
    print(f"[done] wall={wall:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
