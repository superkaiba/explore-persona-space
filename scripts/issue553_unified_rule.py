# ruff: noqa: RUF003
# Intentional Unicode (rho, ※, Δ, −, —) in scientific docstrings + labels.
"""Task #553 — Deliverable 2: unified leakage rule, re-fit per the critic convention.

DV = ``margin_trained`` (ABSOLUTE trained EOS margin at the corrected slot,
cell-level; the modeling DV per the metric-critic bundle — shift readouts are
DERIVED algebraically, never fit as standalone shift regressions). Joint fit

    margin_trained ~ alpha * prior_margin_own(B) + beta * cosine(S, B)

on the three registered cohorts (Ordinary cross-context cells n=240;
Instructed strip n=160; Pooled with cohort FE n=400) with variants:
(i) + interaction (the additivity test; standardized-product std-beta per plan
concern 13.10, judged within margin space AND, as the contrast, within
log-prob space — no cross-DV R² comparison), (ii) + source FE (labeled
``post-train forecast-where`` ONLY; LOBO-only in CV — the statistics-
reconciler round-1 REVISION excludes ``+srcFE`` from LOSO because a held-out
source has no estimable FE and min-norm zero-imputation would silently change
the reported number), (iii) the B1/C1 duplicate-dropped slice.

Coefficient inference: source- and bystander-cluster bootstraps (drawn copies
relabeled; standardization/FE re-estimated inside every resample), the WIDER
one-way CI as primary, plus the CGM (Cameron-Gelbach-Miller 2011) two-way
plug-in SE as a cross-check (non-PSD flagged, never silent — plan concern
13.6). Collinearity gate: per-cohort Pearson(prior, cosine); |r| > 0.6 adds a
tercile-bucket median read.

Cross-validation: LOBO (26-fold leave-one-bystander-out) and LOSO (16-fold
leave-one-source-out) out-of-fold R² per feature set, per cohort and
pooled-with-FE. The inline "~0.71 full panel" LOBO number is a pooled
cross-cohort read — REPRODUCED ONLY for the ``inline_vs_reviewed`` delta
report and marked forbidden-as-headline; the per-cohort + pooled-FE numbers
are the deliverable. The leave-one-group-out CV family precedent lives in
``issue493_extraction_metric_bakeoff.py::_loocv_r2`` (leave-one-context-out,
the i474 fig9 pattern) and ``issue532_followup_logp_slot.py::_cv_r2_loco``
(leave-one-class-out); the #502 scripts cite it in prose only — the LOBO/LOSO
axes here are therefore THIS plan's own convention (plan section 11
fact-checker note, resolved 2026-06-10).

Smoke = this exact script with reduced ``--n-cluster-boot`` (cell bootstrap
and permutation are not used here; coefficient inference is cluster-level).
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import issue553_panel as p553
import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

COHORTS = ("ordinary_cross", "instructed_strip", "pooled_cohort_fe")
COHORT_DISPLAY = {
    "ordinary_cross": "Ordinary cross-context cells",
    "instructed_strip": "Instructed strip",
    "pooled_cohort_fe": "Pooled with cohort FE",
}
DUP_SLICE_OF = {
    "ordinary_cross": "noB1C1_ordinary_cross",
    "instructed_strip": "noB1C1_instructed_strip",
    "pooled_cohort_fe": "noB1C1_pooled_cohort_fe",
}


def _zscore(v: np.ndarray) -> np.ndarray | None:
    sd = float(np.std(v))
    if sd < 1e-12:
        return None
    return (v - np.mean(v)) / sd


def _build_design(
    prior: np.ndarray,
    cos: np.ndarray,
    interaction: bool,
    cohort_flag: np.ndarray | None,
    src_codes: np.ndarray | None,
    n_src: int,
    standardized: bool,
) -> tuple[np.ndarray, list[str]] | None:
    """Design matrix for one fit. Returns None when standardization degenerates.

    Standardized variant: z(prior), z(cosine), and the STANDARDIZED PRODUCT of
    the z-columns (plan concern 13.10); raw variant: raw columns + raw product.
    Cohort FE enters as a 0/1 indicator column; source FE as one-hot dummies
    (min-norm lstsq; predictions stay unique because test rows lie in the
    training design's row space).
    """
    if standardized:
        zp, zc = _zscore(prior), _zscore(cos)
        if zp is None or zc is None:
            return None
        cols = [zp, zc]
        names = ["alpha_prior", "beta_cosine"]
        if interaction:
            prod = _zscore(zp * zc)
            if prod is None:
                return None
            cols.append(prod)
            names.append("gamma_interaction")
    else:
        cols = [prior, cos]
        names = ["alpha_prior", "beta_cosine"]
        if interaction:
            cols.append(prior * cos)
            names.append("gamma_interaction")
    design = [np.ones(len(prior)), *cols]
    if cohort_flag is not None:
        design.append(cohort_flag.astype(np.float64))
    if src_codes is not None:
        onehot = np.zeros((len(prior), n_src))
        onehot[np.arange(len(prior)), src_codes] = 1.0
        mat = np.column_stack(design)
        return np.hstack([mat, onehot]), names
    return np.column_stack(design), names


def _fit_coefs(
    y, prior, cos, interaction, cohort_flag, src_codes, n_src, standardized
) -> tuple[np.ndarray, list[str]] | None:
    built = _build_design(prior, cos, interaction, cohort_flag, src_codes, n_src, standardized)
    if built is None:
        return None
    design, names = built
    ydv = _zscore(y) if standardized else y
    if ydv is None:
        return None
    coef, *_ = np.linalg.lstsq(design, ydv, rcond=None)
    return coef[1 : 1 + len(names)], names


def fit_block(panel: dict, mask: np.ndarray, variant: dict, args) -> dict:
    """One (cohort x variant) joint fit with full coefficient inference."""
    y = panel["margin_trained"][mask]
    prior = panel["prior_margin_own"][mask]
    cos = panel["cosine"][mask]
    src = panel["source_cid"][mask]
    byst = panel["bystander_label"][mask]
    cohort_flag = panel["is_instructed"][mask].astype(np.float64) if variant["cohort_fe"] else None
    src_u, sc = np.unique(src, return_inverse=True)
    _, bc = np.unique(byst, return_inverse=True)
    interaction = variant["interaction"]
    use_src_fe = variant["src_fe"]

    raw = _fit_coefs(
        y, prior, cos, interaction, cohort_flag, sc if use_src_fe else None, len(src_u), False
    )
    std = _fit_coefs(
        y, prior, cos, interaction, cohort_flag, sc if use_src_fe else None, len(src_u), True
    )
    assert raw is not None and std is not None, "observed fit degenerate"
    raw_coefs, names = raw
    std_coefs, _ = std

    def stat_fn_factory(cluster_axis: str):
        def stat_fn(idx: np.ndarray, copy_codes: np.ndarray):
            yb, pb, cb = y[idx], prior[idx], cos[idx]
            cfb = cohort_flag[idx] if cohort_flag is not None else None
            if use_src_fe:
                if cluster_axis == "source":
                    scb, n_s = copy_codes, int(copy_codes.max()) + 1
                else:
                    scb, n_s = sc[idx], len(src_u)
            else:
                scb, n_s = None, 0
            r = _fit_coefs(yb, pb, cb, interaction, cfb, scb, n_s, False)
            s = _fit_coefs(yb, pb, cb, interaction, cfb, scb, n_s, True)
            if r is None or s is None:
                return None
            return np.concatenate([r[0], s[0]])

        return stat_fn

    cluster_cis: dict[str, dict] = {}
    for axis, labels in (("source", src), ("bystander", byst)):
        stats, n_boot, n_deg = p553.cluster_boot_stat(
            labels, stat_fn_factory(axis), args.n_cluster_boot, args.seed
        )
        arr = np.asarray(stats)
        per_coef = {}
        for i, nm in enumerate(names):
            per_coef[nm] = {
                "raw": {
                    "low": float(np.percentile(arr[:, i], 2.5)),
                    "high": float(np.percentile(arr[:, i], 97.5)),
                    "boot_mean": float(np.mean(arr[:, i])),
                },
                "std_beta": {
                    "low": float(np.percentile(arr[:, len(names) + i], 2.5)),
                    "high": float(np.percentile(arr[:, len(names) + i], 97.5)),
                    "boot_mean": float(np.mean(arr[:, len(names) + i])),
                },
            }
        cluster_cis[axis] = {
            "per_coef": per_coef,
            "n_clusters": len(np.unique(labels)),
            "n_boot": n_boot,
            "n_degenerate_resamples": n_deg,
        }

    # CGM two-way plug-in cross-check on the RAW design.
    built = _build_design(
        prior, cos, interaction, cohort_flag, sc if use_src_fe else None, len(src_u), False
    )
    design, _ = built
    _, resid, xtx_inv = p553.ols_fit(design, y)
    cgm = p553.cgm_twoway_se(design, resid, xtx_inv, sc, bc)
    cgm_per_coef = {
        nm: {"se": cgm["se"][1 + i], "non_psd_flag": cgm["non_psd_flag"]}
        for i, nm in enumerate(names)
    }

    out: dict = {"n": int(mask.sum()), "coefficients": {}}
    for i, nm in enumerate(names):
        wider = p553.wider_ci(
            {
                "source": cluster_cis["source"]["per_coef"][nm]["raw"],
                "bystander": cluster_cis["bystander"]["per_coef"][nm]["raw"],
            }
        )
        out["coefficients"][nm] = {
            "raw_estimate": float(raw_coefs[i]),
            "std_beta": float(std_coefs[i]),
            "ci95_cluster_source": cluster_cis["source"]["per_coef"][nm],
            "ci95_cluster_bystander": cluster_cis["bystander"]["per_coef"][nm],
            "primary_ci_raw": wider,
            "cgm_crosscheck": cgm_per_coef[nm],
        }
    out["cluster_boot_meta"] = {
        axis: {k: v for k, v in blk.items() if k != "per_coef"} for axis, blk in cluster_cis.items()
    }
    out["feature_label"] = (
        "post-train forecast-where (+srcFE conditions on the trained model)"
        if use_src_fe
        else "pre-training forecast (prior_margin_own + cosine are computable before training)"
    )
    return out


def _cv_r2(
    y: np.ndarray,
    prior: np.ndarray,
    cos: np.ndarray,
    fold_labels: np.ndarray,
    interaction: bool,
    cohort_flag: np.ndarray | None,
    src_codes: np.ndarray | None,
    n_src: int,
) -> dict:
    """Leave-one-group-out CV R² (out-of-fold predictions pooled, raw design)."""
    pred = np.full(len(y), np.nan)
    n_skipped = 0
    for g in np.unique(fold_labels):
        test = fold_labels == g
        train = ~test
        built = _build_design(
            prior[train],
            cos[train],
            interaction,
            cohort_flag[train] if cohort_flag is not None else None,
            src_codes[train] if src_codes is not None else None,
            n_src,
            False,
        )
        if built is None:
            n_skipped += 1
            continue
        design_tr, _ = built
        coef, *_ = np.linalg.lstsq(design_tr, y[train], rcond=None)
        built_te = _build_design(
            prior[test],
            cos[test],
            interaction,
            cohort_flag[test] if cohort_flag is not None else None,
            src_codes[test] if src_codes is not None else None,
            n_src,
            False,
        )
        design_te, _ = built_te
        pred[test] = design_te @ coef
    ok = ~np.isnan(pred)
    ss_res = float(np.sum((y[ok] - pred[ok]) ** 2))
    ss_tot = float(np.sum((y[ok] - y[ok].mean()) ** 2))
    return {
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "n_folds": len(np.unique(fold_labels)),
        "n_folds_skipped_degenerate": n_skipped,
        "n_predicted": int(ok.sum()),
    }


def cv_suite(panel: dict, mask: np.ndarray, cohort_fe: bool) -> dict:
    """LOBO + LOSO R² for the registered feature sets on one slice."""
    y = panel["margin_trained"][mask]
    prior = panel["prior_margin_own"][mask]
    cos = panel["cosine"][mask]
    src = panel["source_cid"][mask]
    byst = panel["bystander_label"][mask]
    cohort_flag = panel["is_instructed"][mask].astype(np.float64) if cohort_fe else None
    src_u, sc = np.unique(src, return_inverse=True)
    feature_sets = {
        "prior_plus_cosine": dict(interaction=False, src=False),
        "prior_plus_cosine_plus_interaction": dict(interaction=True, src=False),
        "prior_plus_cosine_plus_srcFE": dict(interaction=False, src=True),
    }
    out: dict = {}
    for name, fs in feature_sets.items():
        blk: dict = {"label": "post-train forecast-where" if fs["src"] else "pre-training forecast"}
        blk["lobo"] = _cv_r2(
            y,
            prior,
            cos,
            byst,
            fs["interaction"],
            cohort_flag,
            sc if fs["src"] else None,
            len(src_u),
        )
        if fs["src"]:
            blk["loso"] = {
                "excluded": True,
                "reason": "a held-out source has no estimable source FE from training folds; "
                "min-norm zero-imputation would change the reported number (statistics-"
                "reconciler round-1 REVISION) — +srcFE is LOBO-only",
            }
        else:
            blk["loso"] = _cv_r2(
                y, prior, cos, src, fs["interaction"], cohort_flag, None, len(src_u)
            )
        out[name] = blk
    return out


def make_figure(fits: dict, fig_dir: Path) -> None:
    """Coefficient forest: (alpha, beta, interaction) x cohort x slice."""
    set_paper_style("blog")
    colors = paper_palette(3)
    coef_color = dict(zip(("alpha_prior", "beta_cosine", "gamma_interaction"), colors, strict=True))
    rows = []
    for cohort in COHORTS:
        for slice_name in ("full", "noB1C1"):
            key = f"{cohort}/{slice_name}/with_interaction"
            if key not in fits:
                continue
            for nm in ("alpha_prior", "beta_cosine", "gamma_interaction"):
                rows.append((cohort, slice_name, nm))
    fig, ax = plt.subplots(figsize=(8.5, 0.32 * len(rows) + 1.5))
    for yi, (cohort, slice_name, nm) in enumerate(rows):
        blk = fits[f"{cohort}/{slice_name}/with_interaction"]["coefficients"][nm]
        ci = blk["primary_ci_raw"]
        lo, hi = ci["low"], ci["high"]
        ax.plot([lo, hi], [yi, yi], color=coef_color[nm], lw=1.5)
        ax.plot(blk["raw_estimate"], yi, "o", ms=4.5, color=coef_color[nm])
    ax.axvline(0.0, color="0.4", lw=0.8)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([f"{COHORT_DISPLAY[c]} ({s}) — {nm}" for c, s, nm in rows], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Raw coefficient (wider-of one-way cluster-bootstrap 95% CI)")
    ax.set_title(
        "Unified rule joint fits: margin_trained ~ alpha*prior_margin_own + beta*cosine "
        "(+ interaction)",
        fontsize=9,
    )
    fig.tight_layout()
    savefig_paper(fig, "unified_rule_coefficient_forest", dir=fig_dir)
    plt.close(fig)
    print(f"[figures] wrote unified_rule_coefficient_forest to {fig_dir}")


def main(argv: list[str] | None = None) -> int:
    args = p553.common_parser(
        "Task #553 D2: unified leakage rule re-fit (EOS-margin DV, joint fits + CV)."
    ).parse_args(argv)
    t0 = datetime.now(UTC)

    panel = p553.build_margin_panel(args.i532_dir)
    step0 = p553.step0_i532(panel, args.i532_dir)
    masks = p553.cohort_masks_553(panel)

    # Collinearity gate (plan section 3.3.4).
    collinearity: dict = {}
    for cohort in COHORTS:
        m = masks[cohort]
        r = float(np.corrcoef(panel["prior_margin_own"][m], panel["cosine"][m])[0, 1])
        blk = {"pearson_prior_vs_cosine": r, "gate_triggered": bool(abs(r) > 0.6)}
        if abs(r) > 0.6:
            prior, cos, y = (
                panel["prior_margin_own"][m],
                panel["cosine"][m],
                panel["margin_trained"][m],
            )
            tp = np.digitize(prior, np.quantile(prior, [1 / 3, 2 / 3]))
            tc = np.digitize(cos, np.quantile(cos, [1 / 3, 2 / 3]))
            table = [
                [
                    (
                        float(np.median(y[(tp == i) & (tc == j)]))
                        if ((tp == i) & (tc == j)).sum()
                        else None
                    )
                    for j in range(3)
                ]
                for i in range(3)
            ]
            blk["tercile_median_margin_trained"] = {
                "rows": "prior_margin_own terciles (low/mid/high)",
                "cols": "cosine terciles (low/mid/high)",
                "table": table,
            }
        collinearity[cohort] = blk

    variants = {
        "base": dict(interaction=False, src_fe=False),
        "with_interaction": dict(interaction=True, src_fe=False),
        "with_srcFE": dict(interaction=False, src_fe=True),
    }
    fits: dict = {}
    for cohort in COHORTS:
        for slice_name, mask_name in (("full", cohort), ("noB1C1", DUP_SLICE_OF[cohort])):
            for vname, v in variants.items():
                key = f"{cohort}/{slice_name}/{vname}"
                print(f"[fit] {key} ...")
                vv = dict(v, cohort_fe=(cohort == "pooled_cohort_fe"))
                fits[key] = fit_block(panel, masks[mask_name], vv, args)

    # Log-prob-space contrast for the interaction (within-space judgement only).
    logp_contrast: dict = {}
    for cohort in COHORTS:
        m = masks[cohort]
        sub = {
            "margin_trained": panel["trained_logp"],
            "prior_margin_own": panel["prior_logp_own"],
            "cosine": panel["cosine"],
            "source_cid": panel["source_cid"],
            "bystander_label": panel["bystander_label"],
            "is_instructed": panel["is_instructed"],
        }
        vv = dict(interaction=True, src_fe=False, cohort_fe=(cohort == "pooled_cohort_fe"))
        logp_contrast[cohort] = fit_block(sub, m, vv, args)
        logp_contrast[cohort]["dv"] = "trained_logp (log-prob space; prior = prior_logp_own)"
        logp_contrast[cohort]["note"] = (
            "within-space interaction read ONLY — no cross-DV R² comparison (forbidden move)"
        )

    # Cross-validation suites.
    cv: dict = {}
    for cohort in COHORTS:
        for slice_name, mask_name in (("full", cohort), ("noB1C1", DUP_SLICE_OF[cohort])):
            print(f"[cv] {cohort}/{slice_name} ...")
            cv[f"{cohort}/{slice_name}"] = cv_suite(
                panel, masks[mask_name], cohort_fe=(cohort == "pooled_cohort_fe")
            )
    # Inline reproduction target: pooled WITHOUT cohort FE (forbidden as headline).
    pooled_no_fe = cv_suite(panel, masks["pooled_cohort_fe"], cohort_fe=False)
    pooled_no_fe["_note"] = (
        "pooled cross-cohort WITHOUT cohort FE — computed ONLY to reproduce the inline LOBO "
        "0.71/0.89 numbers for the inline_vs_reviewed delta; forbidden as a headline read"
    )
    cv["pooled_NO_cohort_fe_inline_reproduction_only"] = pooled_no_fe

    # Derived shift readout (never fit): Dmargin-implied R² from the absolute fit.
    derived: dict = {}
    for cohort in COHORTS:
        m = masks[cohort]
        y = panel["margin_trained"][m]
        built = _build_design(
            panel["prior_margin_own"][m],
            panel["cosine"][m],
            False,
            panel["is_instructed"][m].astype(np.float64) if cohort == "pooled_cohort_fe" else None,
            None,
            0,
            False,
        )
        design, _ = built
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        yhat = design @ coef
        dmargin = panel["dmargin"][m]
        dmargin_pred = yhat - panel["margin_base_matched"][m]
        ss_res = float(np.sum((dmargin - dmargin_pred) ** 2))
        ss_tot = float(np.sum((dmargin - dmargin.mean()) ** 2))
        derived[cohort] = {
            "dmargin_implied_r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "note": "Dmargin readout DERIVED as fitted(margin_trained) - observed "
            "margin_base_matched; coefficients carry over algebraically — never fit as a "
            "standalone shift regression (forbidden move)",
        }

    alpha_ord = fits["ordinary_cross/full/base"]["coefficients"]["alpha_prior"]
    inline_vs_reviewed = [
        p553.ivr_entry(
            "LOBO-CV R² full panel (pooled, no cohort FE)",
            0.71,
            cv["pooled_NO_cohort_fe_inline_reproduction_only"]["prior_plus_cosine"]["lobo"]["r2"],
            True,
            "pooled cross-cohort read now forbidden as headline; replaced by per-cohort + "
            "pooled-with-FE LOBO/LOSO (see cv block)",
        ),
        p553.ivr_entry(
            "LOBO-CV R² ordinary-cross",
            0.37,
            cv["ordinary_cross/full"]["prior_plus_cosine"]["lobo"]["r2"],
            False,
            "same cohort, reviewed fold convention",
        ),
        p553.ivr_entry(
            "LOBO-CV R² + srcFE (pooled, no cohort FE)",
            0.89,
            cv["pooled_NO_cohort_fe_inline_reproduction_only"]["prior_plus_cosine_plus_srcFE"][
                "lobo"
            ]["r2"],
            True,
            "post-train forecast-where label mandatory; pooled read reproduced for delta only",
        ),
        p553.ivr_entry(
            "alpha bystander-cluster CI on ordinary-cross",
            [-0.92, 1.37],
            [
                alpha_ord["ci95_cluster_bystander"]["raw"]["low"],
                alpha_ord["ci95_cluster_bystander"]["raw"]["high"],
            ],
            True,
            "inline clustered on bystander only; reviewed convention adds source clustering + "
            "wider-of primary + CGM cross-check",
        ),
    ]

    results = {
        "metadata": p553.result_metadata(args, "issue553_unified_rule.py"),
        "step0_i532": step0,
        "collinearity_gate": collinearity,
        "fits": fits,
        "logp_space_interaction_contrast": logp_contrast,
        "cv": cv,
        "derived_shift_readout": derived,
        "inline_vs_reviewed": inline_vs_reviewed,
    }
    p553.write_json(args.out_dir / "unified_rule.json", results)
    make_figure(fits, args.fig_dir)

    b = fits["ordinary_cross/full/with_interaction"]["coefficients"]
    print(
        f"[headline] ordinary-cross: alpha={b['alpha_prior']['raw_estimate']:+.3f} "
        f"beta={b['beta_cosine']['raw_estimate']:+.3f} "
        f"interaction std-beta={b['gamma_interaction']['std_beta']:+.3f} "
        f"[{b['gamma_interaction']['primary_ci_raw']['low']:+.3f}, "
        f"{b['gamma_interaction']['primary_ci_raw']['high']:+.3f}] (raw CI)"
    )
    print(
        f"[headline] LOBO R² "
        f"ordinary={cv['ordinary_cross/full']['prior_plus_cosine']['lobo']['r2']:.3f} "
        f"instructed={cv['instructed_strip/full']['prior_plus_cosine']['lobo']['r2']:.3f} "
        f"pooled-FE={cv['pooled_cohort_fe/full']['prior_plus_cosine']['lobo']['r2']:.3f}"
    )
    print(f"[done] wall={(datetime.now(UTC) - t0).total_seconds():.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
