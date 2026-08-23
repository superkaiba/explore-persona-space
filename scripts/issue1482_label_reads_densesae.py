#!/usr/bin/env python
"""Issue #1482: R^2-dependent LABEL reads against the REAL dense->SAE target.

The four DEFERRED_READS manifest items (`issue1482_discrete_predictors.py`)
still outstanding after the c9d7c365b0 continuous-read repoint, run per arm
(ridge / MLP) via one `--r2-npy` swap:

  1. raw AUROC + EXCESS-OVER-STRATIFIED-NULL per binary one-vs-rest label,
     SORTED BY EXCESS — the committed `label_reads` machinery verbatim
     (labeled-vs-unlabeled R^2, activity-decile-stratified label-shuffle
     null centred on its OWN mean, not on 0.5).
  2. the UNIFIED AUROC-at-depth sweep: label = top-k vs bottom-k by R^2,
     score = THE PREDICTOR, one construction across continuous covariates
     AND binary one-vs-rest codings; per-k recomputed stratified null band;
     Delta_k kept in JSON ONLY, with the identity AUROC = 0.5 + Delta_k/2
     noted so it is never read as independent evidence.
  3. abstraction: per-level medians + n + bootstrap CIs (PRIMARY),
     Kruskal-Wallis omnibus eta^2 vs the activity-stratified null
     (COMPARABLE across axes), ordinal Spearman with its TIE CEILING and
     attainment fraction (SECONDARY, a monotone-trend read).
  4. per-axis omnibus (Kruskal-Wallis eta^2 vs the stratified null)
     alongside the one-vs-rest AUROCs.

Universe: the committed `fullwidth_matrix.npz` label universe (114,980)
intersected with finite rows of the real target — features scored on the new
target but OUTSIDE the label universe cannot enter a label read (no labels)
and their count is reported, never silently absorbed. Every permutation null
is a GEMM against a fixed rank vector (the `_mw_from_ranks` construction);
nothing loops per draw.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402


# Derived from __file__, NOT task_workflow.repo_root(): that resolver branch-guards to the
# MAIN checkout (it refuses sparse/shallow checkouts). In a default sparse worktree (no
# eval_results/ cones) a re-run fails LOUD (FileNotFoundError) rather than silently reading
# main's copies. #2183; precedent: scripts/issue1482_densesae_fullwidth.py.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue1482_predictor_battery as PB  # noqa: E402
import issue1482_predictor_battery_fullwidth as FW  # noqa: E402

DICT_SIZE = 131_072
SEED = 1482
MATRIX = "eval_results/issue_1482/predictor_battery/fullwidth_matrix.npz"
OUT_DIR = "eval_results/issue_1482/predictor_battery"
FIG_DIR = "figures/issue_1482/predictor_battery"

# Resolved levels per judged axis for the omnibus (unlabeled + unresolved +
# unclear excluded — the omnibus asks "do the RESOLVED classes differ", the
# same row policy as the one-vs-rest codings). functional_role stays retired.
OMNIBUS_AXES = {
    "abstraction": ("token_surface", "lexical_semantic", "abstract_contextual"),
    "content_type": ("syntax", "topic", "operation", "entity", "task_format"),
    "speaker_property": ("none", "language", "register_style", "identity_disposition"),
    "interpretable": ("yes", "no"),
}
ABSTRACTION_ORD = FW.ABSTRACTION_ORD


def _log(msg: str) -> None:
    print(f"[label-densesae] {msg}", flush=True)


# ── bundle ────────────────────────────────────────────────────────────────────


def build_bundle(r2_npy: Path) -> tuple[dict, dict]:
    """Label-universe bundle with the target swapped to `r2_npy` (full width)."""
    with np.load(PROJECT_ROOT / MATRIX, allow_pickle=True) as z:
        feat_ids = np.asarray(z["feat_ids"], dtype=np.int64)
        cov = {k: np.asarray(z[k], dtype=np.float64) for k in FW.CONTINUOUS}
        labels = {
            k[len("label__") :]: np.asarray(z[k]).astype(str)
            for k in z.files
            if k.startswith("label__")
        }
    r2_full = np.asarray(np.load(r2_npy), dtype=np.float64)
    if r2_full.shape != (DICT_SIZE,):
        raise AssertionError(f"R^2 must be ({DICT_SIZE},), got {r2_full.shape}")
    r2_u = r2_full[feat_ids]
    ok = np.isfinite(r2_u)
    stats = {
        "label_universe": int(len(feat_ids)),
        "n_used": int(ok.sum()),
        "scored_full_width": int(np.isfinite(r2_full).sum()),
        "scored_outside_label_universe": int(
            np.isfinite(r2_full).sum() - np.isfinite(r2_full[feat_ids]).sum()
        ),
        "note": (
            "features scored on the new target but outside the 114,980 label universe have no "
            "judged labels and cannot enter a label read"
        ),
    }
    bundle = {
        "feat_ids": feat_ids[ok],
        "r2": r2_u[ok],
        "cov": {k: v[ok] for k, v in cov.items()},
        "labels": {ax: v[ok] for ax, v in labels.items()},
    }
    _log(
        f"bundle: {stats['n_used']:,} rows (label universe {stats['label_universe']:,}; "
        f"{stats['scored_outside_label_universe']:,} scored features outside it)"
    )
    return bundle, stats


# ── item 2: unified AUROC-at-depth ───────────────────────────────────────────


def unified_predictors(bundle: dict) -> dict[str, dict]:
    """Score set for the unified sweep: continuous covariates + binary codings."""
    out: dict[str, dict] = {}
    for key, (label, _lx) in FW.CONTINUOUS.items():
        out[key] = {"values": bundle["cov"][key], "kind": "continuous", "label": label}
    for coding, (axis, pos_fn, drop) in FW.BINARY_AXES.items():
        raw = bundle["labels"][axis]
        keep = np.array([s not in drop and s != "unlabeled" for s in raw])
        v = np.full(len(raw), np.nan)
        v[keep] = np.array([1.0 if pos_fn(s) else 0.0 for s in raw[keep]])
        out[coding] = {"values": v, "kind": "binary", "label": f"{axis}: {coding}"}
    return out


def unified_auroc_depth(bundle: dict, n_perm: int, rng, k_grid, chunk: int = 100) -> dict:
    """label = top-k vs bottom-k by R^2, score = the predictor; per-k null band.

    ONE stratified label-permutation matrix per (k, chunk), shared across every
    predictor (per-predictor finite masks slice it), so the null is never
    re-drawn per predictor. Delta_k = 2*(AUROC-0.5) is JSON-only.
    """
    r2, act = bundle["r2"], bundle["cov"]["activity"]
    n = len(r2)
    order = np.argsort(-r2)
    strata_full = FW._decile_of(act)
    preds = unified_predictors(bundle)
    ks = [int(k) for k in k_grid if 2 * k <= n]
    per_pred: dict[str, list] = {name: [] for name in preds}

    for k in ks:
        sel = np.concatenate([order[:k], order[n - k :]])
        lab = np.concatenate([np.ones(k, dtype=np.int8), np.zeros(k, dtype=np.int8)])
        strata = strata_full[sel]
        # ranks + observed AUROC per predictor (1-based ranks => exact Mann-Whitney)
        pre = {}
        for name, p in preds.items():
            v = p["values"][sel]
            m = np.isfinite(v)
            if m.sum() < 10 or lab[m].sum() in (0, m.sum()):
                pre[name] = None
                continue
            ranks = PB._rank(v[m]) + 1.0
            pre[name] = (m, ranks, float(FW._auroc_from_ranks(ranks, lab[m])))
        nulls: dict[str, list] = {name: [] for name in preds}
        done = 0
        while done < n_perm:
            b = min(chunk, n_perm - done)
            P = FW._perm_chunk_sorted(lab, strata, b, rng)
            for name, entry in pre.items():
                if entry is None:
                    continue
                m, ranks, _ = entry
                nulls[name].append(FW._mw_from_ranks(ranks, P[m], int(m.sum())))
            done += b
        for name, entry in pre.items():
            if entry is None:
                per_pred[name].append({"k": k, "auroc": None})
                continue
            m, _, obs = entry
            ng = np.concatenate(nulls[name])
            ngf = ng[np.isfinite(ng)]
            per_pred[name].append(
                {
                    "k": k,
                    "n_finite": int(m.sum()),
                    "auroc": obs,
                    "delta_k": 2 * (obs - 0.5),
                    "perm_band_2p5": float(np.percentile(ngf, 2.5)) if len(ngf) else None,
                    "perm_band_97p5": float(np.percentile(ngf, 97.5)) if len(ngf) else None,
                    "null_mean": float(np.mean(ngf)) if len(ngf) else None,
                }
            )
        _log(f"depth k={k}: {sum(1 for e in pre.values() if e)} predictors scored")
    return {
        "construction": (
            "UNIFIED: label = top-k vs bottom-k features by R^2, score = the predictor "
            "(continuous covariates and binary one-vs-rest codings under ONE construction); "
            "null = activity-decile-stratified permutation of the top/bottom label within the "
            "2k subset, recomputed per k"
        ),
        "delta_note": (
            "delta_k = 2*(AUROC - 0.5) is kept in JSON only; AUROC = 0.5 + delta_k/2 is an "
            "IDENTITY, so delta is never independent evidence"
        ),
        "k_grid": ks,
        "predictors": {
            name: {"kind": preds[name]["kind"], "label": preds[name]["label"], "depth": rows}
            for name, rows in per_pred.items()
        },
    }


# ── items 3+4: Kruskal-Wallis omnibus + abstraction reads ────────────────────


def _kw_eta2_of(ranks1: np.ndarray, lab_mat: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Kruskal-Wallis eta^2 per column of a (n, draws) integer level matrix.

    With the R^2 ranks FIXED (1-based average ranks), each draw's per-group
    rank sums are one GEMM per level. eta^2 = (H - k + 1) / (n - k).
    """
    n = len(ranks1)
    h = np.zeros(lab_mat.shape[1])
    for lv in levels:
        m = (lab_mat == lv).astype(np.float64)
        ng = m.sum(axis=0)
        rg = ranks1 @ m
        with np.errstate(invalid="ignore", divide="ignore"):
            h += np.where(ng > 0, rg**2 / np.maximum(ng, 1), 0.0)
    h = 12.0 / (n * (n + 1)) * h - 3.0 * (n + 1)
    k = len(levels)
    return (h - k + 1) / (n - k)


def kw_omnibus(
    r2: np.ndarray, lab_int: np.ndarray, strata: np.ndarray, n_perm: int, rng, chunk: int = 200
) -> dict:
    levels = np.unique(lab_int)
    ranks1 = PB._rank(r2) + 1.0
    obs = float(_kw_eta2_of(ranks1, lab_int[:, None].astype(np.float64), levels)[0])
    null: list[np.ndarray] = []
    done = 0
    while done < n_perm:
        b = min(chunk, n_perm - done)
        p = FW._perm_chunk_sorted(lab_int, strata, b, rng)
        null.append(_kw_eta2_of(ranks1, p, levels))
        done += b
    ng = np.concatenate(null)
    return {
        "n": int(len(r2)),
        "n_levels": int(len(levels)),
        "eta2": obs,
        "eta2_null_mean": float(np.mean(ng)),
        "eta2_null_band": [float(np.percentile(ng, 2.5)), float(np.percentile(ng, 97.5))],
        "p_perm_one_sided": float(((ng >= obs).sum() + 1) / (len(ng) + 1)),
        "tie_note": "R^2 is continuous (ties negligible); no tie correction applied to H",
    }


def _median_ci(x: np.ndarray, n_boot: int, rng) -> list[float]:
    draws = np.median(
        x[rng.integers(0, len(x), size=(n_boot, len(x)))],
        axis=1,
    )
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def abstraction_read(bundle: dict, n_perm: int, n_boot: int, rng) -> dict:
    """Per-level medians (PRIMARY), KW omnibus vs stratified null (COMPARABLE),
    ordinal Spearman + tie ceiling + attainment (SECONDARY monotone trend)."""
    raw = bundle["labels"]["abstraction"]
    keep = np.array([s in ABSTRACTION_ORD for s in raw])
    r2 = bundle["r2"][keep]
    act = bundle["cov"]["activity"][keep]
    ordv = np.array([ABSTRACTION_ORD[s] for s in raw[keep]])
    strata = FW._decile_of(act)

    per_level = []
    for name, o in sorted(ABSTRACTION_ORD.items(), key=lambda kv: kv[1]):
        x = r2[ordv == o]
        per_level.append(
            {
                "level": name,
                "ordinal": o,
                "n": int(len(x)),
                "median_r2": float(np.median(x)),
                "median_ci95": _median_ci(x, n_boot, rng),
            }
        )

    omni = kw_omnibus(r2, ordv.astype(np.int64), strata, n_perm, rng)

    rho = PB._spearman(ordv, r2)
    order = np.argsort(ordv, kind="stable")
    ideal = np.empty(len(r2))
    ideal[order] = np.arange(len(r2), dtype=np.float64)
    rho_ceiling = PB._spearman(ordv, ideal)
    return {
        "per_level": per_level,
        "omnibus": omni,
        "ordinal_spearman": {
            "rho": float(rho),
            "tie_ceiling": float(rho_ceiling),
            "attainment_fraction": float(rho / rho_ceiling) if rho_ceiling else None,
            "note": (
                "the ceiling is the max |rho| attainable given the 3-level tie structure of the "
                "ordinal variable against a continuous target; attainment = rho / ceiling. "
                "SECONDARY monotone-trend read — the per-level medians are primary."
            ),
        },
    }


def per_axis_omnibus(bundle: dict, n_perm: int, rng) -> dict:
    out = {}
    for axis, levels in OMNIBUS_AXES.items():
        raw = bundle["labels"][axis]
        keep = np.isin(raw, levels)
        r2 = bundle["r2"][keep]
        lab = np.searchsorted(np.array(sorted(levels)), raw[keep]).astype(np.int64)
        strata = FW._decile_of(bundle["cov"]["activity"][keep])
        out[axis] = kw_omnibus(r2, lab, strata, n_perm, rng)
        out[axis]["levels"] = sorted(levels)
        out[axis]["row_policy"] = "resolved levels only (unlabeled/unresolved/unclear excluded)"
        _log(
            f"omnibus {axis}: eta2={out[axis]['eta2']:.5f} "
            f"null=[{out[axis]['eta2_null_band'][0]:.5f}, {out[axis]['eta2_null_band'][1]:.5f}] "
            f"p={out[axis]['p_perm_one_sided']:.4f} n={out[axis]['n']:,}"
        )
    return out


# ── figures ──────────────────────────────────────────────────────────────────


def fig_excess_forest(lr: dict, stats: dict, fig_dir: Path, suffix: str, note: str) -> str:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    rows = sorted(lr.items(), key=lambda kv: kv[1]["auroc"] - kv[1]["auroc_perm_null_mean"])
    y = np.arange(len(rows), dtype=float)
    fig, ax = plt.subplots(figsize=(9.6, 0.42 * len(rows) + 2.9))
    for i, (name, d) in enumerate(rows):
        lo, hi = d["auroc_perm_band"]
        ax.plot([lo, hi], [i, i], color="#BBBBBB", lw=5.0, alpha=0.85, zorder=1)
        ax.plot(d["auroc_perm_null_mean"], i, "|", ms=11, color="#666666", mew=1.6, zorder=2)
        ax.plot(d["auroc"], i, "o", ms=6, color=paper_palette_role("primary"), zorder=3)
        ax.text(
            max(hi, d["auroc"]) + 0.006,
            i,
            f"excess {d['auroc'] - d['auroc_perm_null_mean']:+.3f}  p={d['auroc_perm_p']:.3f}"
            f"  prev={d['marginal_prevalence']:.3f}",
            va="center",
            fontsize=6.6,
            color="#5A5A5A",
        )
    ax.axvline(0.5, color="#999999", lw=0.8, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels([name for name, _ in rows], fontsize=8.2)
    ax.set_xlabel("AUROC (labeled vs rest, by $R^2$)", fontsize=9)
    ax.set_xlim(0.35, 0.78)
    fig.suptitle("Label AUROCs sorted by excess over the stratified null", fontsize=12.5, y=0.98)
    fig.text(
        0.5,
        0.928,
        f"{stats['n_used']:,} features (label universe)  |  {note}",
        ha="center",
        fontsize=7.0,
        color="#5A5A5A",
    )
    fig.text(
        0.5,
        0.012,
        "band = 2.5..97.5 pct of the activity-decile-stratified label-shuffle null; tick = null "
        "mean (NOT 0.5 — stratification preserves the activity-label association); dot = "
        "observed. p is two-sided against the null's own centre.",
        ha="center",
        fontsize=6.8,
        color="#5A5A5A",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    stem = "label_auroc_excess" + suffix
    savefig_paper(fig, stem, dir=fig_dir)
    plt.close(fig)
    return stem


def fig_unified_depth(ud: dict, stats: dict, fig_dir: Path, suffix: str, note: str) -> str:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    names = [n for n, p in ud["predictors"].items() if any(r.get("auroc") for r in p["depth"])]
    ncol = 6
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(2.55 * ncol, 2.15 * nrow + 1.5), sharex=True, sharey=True
    )
    axes = np.atleast_2d(axes)
    for j, name in enumerate(names):
        ax = axes[j // ncol, j % ncol]
        rows = [r for r in ud["predictors"][name]["depth"] if r.get("auroc") is not None]
        ks = [r["k"] for r in rows]
        ax.fill_between(
            ks,
            [r["perm_band_2p5"] for r in rows],
            [r["perm_band_97p5"] for r in rows],
            color="#CCCCCC",
            alpha=0.6,
            lw=0,
        )
        ax.plot(
            ks,
            [r["auroc"] for r in rows],
            "-o",
            ms=2.6,
            lw=1.2,
            color=paper_palette_role(
                "primary" if ud["predictors"][name]["kind"] == "continuous" else "control"
            ),
        )
        ax.axhline(0.5, color="#999999", lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_title(name, fontsize=7.4)
        ax.tick_params(labelsize=6.4)
    for j in range(len(names), nrow * ncol):
        axes[j // ncol, j % ncol].set_visible(False)
    fig.suptitle(
        "Unified AUROC-at-depth: top-k vs bottom-k by $R^2$, scored by each predictor",
        fontsize=12.0,
        y=0.995,
    )
    fig.text(
        0.5,
        0.955,
        f"{stats['n_used']:,} features  |  {note}",
        ha="center",
        fontsize=7.0,
        color="#5A5A5A",
    )
    fig.text(
        0.5,
        0.008,
        "grey band = per-k activity-stratified permutation null (2.5..97.5 pct). Blue = "
        "continuous predictor, green = binary one-vs-rest label as the score. "
        "Metric is AUROC-at-depth, never classification accuracy.",
        ha="center",
        fontsize=6.8,
        color="#5A5A5A",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    stem = "unified_auroc_depth" + suffix
    savefig_paper(fig, stem, dir=fig_dir)
    plt.close(fig)
    return stem


def fig_abstraction(
    ab: dict, omni: dict, stats: dict, fig_dir: Path, suffix: str, note: str
) -> str:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.4))
    lv = ab["per_level"]
    x = np.arange(len(lv))
    med = [d["median_r2"] for d in lv]
    lo = [max(0.0, d["median_r2"] - d["median_ci95"][0]) for d in lv]
    hi = [max(0.0, d["median_ci95"][1] - d["median_r2"]) for d in lv]
    ax.errorbar(
        x, med, yerr=[lo, hi], fmt="o", ms=6, capsize=4, color=paper_palette_role("primary")
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d['level']}\nn={d['n']:,}" for d in lv], fontsize=7.6)
    ax.set_ylabel("median per-feature $R^2$", fontsize=9)
    o = ab["ordinal_spearman"]
    ax.set_title(
        f"abstraction: per-level medians (PRIMARY)\nordinal rho {o['rho']:+.3f} of ceiling "
        f"{o['tie_ceiling']:+.3f} (attainment {o['attainment_fraction']:.2f})",
        fontsize=8.8,
        loc="left",
    )

    axes_names = list(omni)
    y = np.arange(len(axes_names), dtype=float)
    for i, axn in enumerate(axes_names):
        d = omni[axn]
        b = d["eta2_null_band"]
        ax2.plot(b, [i, i], color="#BBBBBB", lw=5.0, alpha=0.85, zorder=1)
        ax2.plot(d["eta2"], i, "o", ms=6, color=paper_palette_role("primary"), zorder=3)
        ax2.text(
            max(b[1], d["eta2"]) + 0.0005,
            i,
            f"p={d['p_perm_one_sided']:.3f} n={d['n']:,}",
            va="center",
            fontsize=6.8,
            color="#5A5A5A",
        )
    ax2.set_yticks(y)
    ax2.set_yticklabels(axes_names, fontsize=8.2)
    ax2.set_xlabel(r"Kruskal-Wallis $\eta^2$ (resolved levels)", fontsize=9)
    ax2.set_title("per-axis omnibus vs stratified null", fontsize=8.8, loc="left")

    fig.suptitle("Judged-axis omnibus reads", fontsize=12.5, y=0.99)
    fig.text(
        0.5,
        0.925,
        f"{stats['n_used']:,}-feature label universe  |  {note}",
        ha="center",
        fontsize=7.0,
        color="#5A5A5A",
    )
    fig.text(
        0.5,
        0.012,
        "grey band = activity-decile-stratified level-shuffle null (2.5..97.5 pct); one-sided p "
        "against that null. Median CIs are 95% bootstrap. Omnibus answers 'do the resolved "
        "classes differ at all', the one-vs-rest AUROCs answer 'which class'.",
        ha="center",
        fontsize=6.8,
        color="#5A5A5A",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.915))
    stem = "abstraction_omnibus" + suffix
    savefig_paper(fig, stem, dir=fig_dir)
    plt.close(fig)
    return stem


# ── entrypoint ───────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="#1482 label reads vs the real dense->SAE target")
    ap.add_argument("--r2-npy", type=Path, required=True)
    ap.add_argument("--r2-label", required=True)
    ap.add_argument("--stem-suffix", required=True, help="e.g. _densesae_ridge")
    ap.add_argument("--target-note", required=True)
    ap.add_argument("--n-perm", type=int, default=FW.N_PERM)
    ap.add_argument("--n-boot", type=int, default=FW.N_BOOT)
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / OUT_DIR)
    ap.add_argument("--fig-dir", type=Path, default=PROJECT_ROOT / FIG_DIR)
    ap.add_argument("--figs-only", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="n_perm/n_boot=50, first 4 depth ks")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    suffix, note = args.stem_suffix, args.target_note
    k_grid = FW.K_GRID[:4] if args.smoke else FW.K_GRID
    if args.smoke:
        args.n_perm, args.n_boot = 50, 50
        _log("SMOKE MODE — n_perm/n_boot=50, truncated depth grid; point dirs at scratch")

    out_json = args.out_dir / f"fullwidth_label_reads{suffix}.json"
    t0 = time.time()
    if args.figs_only:
        import matplotlib

        matplotlib.use("Agg")
        doc = json.loads(out_json.read_text())
        note = doc["design"].get("target_note", note)
        stems = [
            fig_excess_forest(doc["label_reads"], doc["universe"], args.fig_dir, suffix, note),
            fig_unified_depth(
                doc["unified_auroc_depth"], doc["universe"], args.fig_dir, suffix, note
            ),
            fig_abstraction(
                doc["abstraction"],
                doc["per_axis_omnibus"],
                doc["universe"],
                args.fig_dir,
                suffix,
                note,
            ),
        ]
        _log(f"figures (from JSON): {', '.join(stems)}  ({time.time() - t0:.0f}s)")
        return

    rng = np.random.default_rng(SEED)
    bundle, stats = build_bundle(args.r2_npy)

    lr = FW.label_reads(bundle, args.n_perm, args.n_boot, rng)
    for name, d in lr.items():
        d["excess_over_null"] = d["auroc"] - d["auroc_perm_null_mean"]
    order = sorted(lr, key=lambda n: -lr[n]["excess_over_null"])
    _log(
        "excess-over-null order: "
        + ", ".join(f"{n} {lr[n]['excess_over_null']:+.3f}" for n in order)
    )

    ud = unified_auroc_depth(bundle, args.n_perm, rng, k_grid)
    ab = abstraction_read(bundle, args.n_perm, args.n_boot, rng)
    omni = per_axis_omnibus(bundle, args.n_perm, rng)

    doc = {
        "design": {
            "scope": "FULL DICTIONARY label reads vs the REAL dense->SAE target",
            "r2_source": str(
                args.r2_npy.resolve().relative_to(PROJECT_ROOT)
                if args.r2_npy.resolve().is_relative_to(PROJECT_ROOT)
                else args.r2_npy.resolve()
            ),
            "r2_label": args.r2_label,
            "target_note": note,
            "n_perm": args.n_perm,
            "n_boot": args.n_boot,
            "smoke": bool(args.smoke),
            "auroc_definition": (
                "P(R^2 of labeled > R^2 of rest) + 0.5 P(tie) (Mann-Whitney), "
                "group-conditional bootstrap CI, activity-decile-stratified label-shuffle "
                "null centred on its own mean"
            ),
            "retired_axes": list(FW.RETIRED_AXES),
            "seed": SEED,
        },
        "universe": stats,
        "label_reads": lr,
        "label_reads_sorted_by_excess": order,
        "unified_auroc_depth": ud,
        "abstraction": ab,
        "per_axis_omnibus": omni,
        "metadata": PB._metadata(),
    }
    out_json.write_text(json.dumps(doc, indent=1))
    _log(f"reads -> {out_json}  ({time.time() - t0:.0f}s)")

    import matplotlib

    matplotlib.use("Agg")
    stems = [
        fig_excess_forest(lr, stats, args.fig_dir, suffix, note),
        fig_unified_depth(ud, stats, args.fig_dir, suffix, note),
        fig_abstraction(ab, omni, stats, args.fig_dir, suffix, note),
    ]
    _log(f"figures: {', '.join(stems)}  (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
