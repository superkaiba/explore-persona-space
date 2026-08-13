"""Issue #2054 user-gaps extension: digests + figures for the three landed grids.

Companion-artifact round (no interpretation): builds per-grid digest JSONs and
paper-plots figures from the per-unit JSONs staged off the HF data repo
(``superkaiba1/explore-persona-space-data``):

  1. Gap A  ``issue2054_lattice/ctx2ctx_full/percell/``       (112 units)
  2. Gap B  ``issue2054_lattice/pool_specialize/aggregate.json`` (+5 spot files)
  3. Rungs  ``issue2054_lattice/pool_rungs/percell_rungs/``    (112 units)

Outputs:
  eval_results/issue_2054/{ctx2ctx_full,pool_specialize,pool_rungs}/digest.json
  figures/issue_2054/user_gaps_extension/*.{png,pdf,meta.json}

All fits are held-out K=5 conversation-grouped folds over teacher-forced
layer-19 captures (on-policy + inserted banked activations). Pair-class
denominators are derived from the staged data, never from constants.

Usage:
  uv run python scripts/issue2054_userfigs_gaps.py --staged /tmp/issue2054_agg
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

ASSISTANT_IDENTITY = "conversation_paired_stories_assistant"
RUNGS = ["m0", "ctx_offset", "ans_offset", "m1", "scale", "rot", "rot_scale"]
RUNG_LABEL = {
    "m0": "pooled map",
    "ctx_offset": "+ context offset",
    "ans_offset": "+ answer offset",
    "m1": "+ per-cell bias",
    "scale": "+ scale",
    "rot": "+ rotation",
    "rot_scale": "+ rotation & scale",
    "identity_cell": "identity + bias",
}
IDENTITY_LABEL = {
    "char_dana": "Dana",
    "char_helios": "Helios",
    "char_vex": "Vex",
    "char_wren": "Wren",
    "char_dana_op": "Dana (chat)",
    "char_helios_op": "Helios (chat)",
    "char_vex_op": "Vex (chat)",
    "char_wren_op": "Wren (chat)",
    "char_dana_op_base": "Dana (chat)",
    "char_helios_op_base": "Helios (chat)",
    "char_vex_op_base": "Vex (chat)",
    "char_wren_op_base": "Wren (chat)",
    ASSISTANT_IDENTITY: "assistant",
}
COND_LABEL = {"inserted": "inserted", "on_policy": "on-policy", "cell_c": "on-policy"}
FRAMING_LABEL = {
    "attrib_quoted": "quote",
    "bare_label": "label",
    "chat": "chat",
    "bare_text": "text",
}
MODEL_LABEL = {"qwen2.5-7b": "base", "qwen2.5-7b-instruct": "instruct"}
ARM_LABEL = {"context": "context arm", "prefix": "prefix arm"}
ARM_COLOR = {}  # filled at style time


def parse_cell(key: str) -> dict:
    ident, cond, framing, model = key.split("__")
    return {"identity": ident, "condition": cond, "framing": framing, "model": model}


def cell_label(key: str) -> str:
    c = parse_cell(key)
    return (
        f"{IDENTITY_LABEL[c['identity']]} · {COND_LABEL[c['condition']]} · "
        f"{FRAMING_LABEL[c['framing']]} · {MODEL_LABEL[c['model']]}"
    )


def med_iqr(vals) -> dict:
    v = np.asarray(sorted(float(x) for x in vals))
    if len(v) == 0:
        return {"n": 0, "median": None, "q25": None, "q75": None}
    return {
        "n": int(len(v)),
        "median": float(np.median(v)),
        "q25": float(np.percentile(v, 25)),
        "q75": float(np.percentile(v, 75)),
        "min": float(v.min()),
        "max": float(v.max()),
    }


def pair_class(src_key: str, tgt_key: str, family: str) -> str:
    if family == "framing_pairs":
        return "assistant: framing to framing"
    a = parse_cell(src_key)["identity"] == ASSISTANT_IDENTITY
    b = parse_cell(tgt_key)["identity"] == ASSISTANT_IDENTITY
    if a and not b:
        return "assistant to character"
    if b and not a:
        return "character to assistant"
    return "character to character"


PROVENANCE = {
    "substrate": ("on-policy + inserted banked activations, teacher-forced capture, layer 19"),
    "fits": "held-out K=5 conversation-grouped folds (shared production fold map)",
    "arms": "context (prefix + user query) and prefix (everything before the query)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Grid 1: ctx2ctx_full


def build_ctx2ctx(staged: Path) -> dict:
    files = sorted(glob.glob(str(staged / "ctx2ctx_full/percell/*.json")))
    units, pairs = [], []
    fold_meta = None
    for f in files:
        d = json.load(open(f))
        if fold_meta is None:
            fold_meta = d["metadata"]["fold_map"]
        arm = d["arm"]
        src = d["source_cell"]
        fitted, n_skip, n_degxy = [], 0, 0
        for p in d["pairs"]:
            cls = pair_class(src, p["target_cell"], p["family"])
            row = {
                "source_cell": src,
                "target_cell": p["target_cell"],
                "arm": arm,
                "family": p["family"],
                "pair_class": cls,
                "n_join": p["n_join"],
                "n_folds_skipped_degenerate": p.get("n_folds_skipped_degenerate", 0),
            }
            if p["pooled"] is None:
                row["status"] = "skipped_constant_vector"
                n_skip += 1
            elif p["pooled"].get("degenerate_identical_xy"):
                row["status"] = "degenerate_identical_xy"
                n_degxy += 1
            else:
                row["status"] = "fitted"
                row.update(
                    fitted_r2=p["pooled"]["fitted_r2"],
                    identity_bias_r2=p["pooled"]["identity_bias_r2"],
                    delta_r2_fitted_minus_identity=p["pooled"]["delta_r2_fitted_minus_identity"],
                    clears_identity_baseline=p["pooled"]["clears_identity_baseline"],
                    null_mean=p["pooled"]["null_mean"],
                    null_p_value_fitted=p["pooled"]["null_p_value_fitted"],
                )
                fitted.append(p["pooled"]["fitted_r2"])
            pairs.append(row)
        units.append(
            {
                "source_cell": src,
                "arm": arm,
                "n_pairs": len(d["pairs"]),
                "n_pairs_fitted": len(fitted),
                "n_pairs_skipped_constant": n_skip,
                "n_pairs_degenerate_identical_xy": n_degxy,
                "median_fitted_r2": float(np.median(fitted)) if fitted else None,
            }
        )

    fitted_rows = [r for r in pairs if r["status"] == "fitted"]
    by_arm = {}
    for arm in ("context", "prefix"):
        rows = [r for r in fitted_rows if r["arm"] == arm]
        by_arm[arm] = {
            "fitted_r2": med_iqr([r["fitted_r2"] for r in rows]),
            "identity_bias_r2": med_iqr([r["identity_bias_r2"] for r in rows]),
            "clears_identity_fraction": (
                sum(r["clears_identity_baseline"] for r in rows) / len(rows) if rows else None
            ),
            "n_pairs_fitted": len(rows),
        }
    by_class = {}
    for cls in sorted({r["pair_class"] for r in pairs}):
        by_class[cls] = {}
        for arm in ("context", "prefix"):
            rows = [r for r in fitted_rows if r["pair_class"] == cls and r["arm"] == arm]
            n_all = sum(1 for r in pairs if r["pair_class"] == cls and r["arm"] == arm)
            by_class[cls][arm] = {
                "n_pairs_total": n_all,
                "n_pairs_fitted": len(rows),
                "fitted_r2": med_iqr([r["fitted_r2"] for r in rows]),
                "clears_identity_fraction": (
                    sum(r["clears_identity_baseline"] for r in rows) / len(rows) if rows else None
                ),
            }
    census = {
        "n_unit_files": len(files),
        "pairs_total": len(pairs),
        "pairs_fitted": len(fitted_rows),
        "pairs_skipped_constant_vector": sum(
            1 for r in pairs if r["status"] == "skipped_constant_vector"
        ),
        "pairs_degenerate_identical_xy": sum(
            1 for r in pairs if r["status"] == "degenerate_identical_xy"
        ),
        "units_with_any_skip": sum(1 for u in units if u["n_pairs_skipped_constant"] > 0),
        "units_fully_skipped": sum(1 for u in units if u["n_pairs_fitted"] == 0),
    }
    return {
        "metadata": {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_prefix": "issue2054_lattice/ctx2ctx_full/percell/",
            "script": "scripts/issue2054_userfigs_gaps.py",
            "fold_map": fold_meta,
            "provenance": PROVENANCE,
            "note": (
                "Pair-class denominators derived from the staged per-unit JSONs. "
                "Statistics pool the per-pair fold-mean held-out R^2 (each pair's "
                "'pooled' block = mean over its scored folds)."
            ),
        },
        "units": units,
        "pairs": pairs,
        "summary": {
            "by_arm": by_arm,
            "by_pair_class": by_class,
            "degeneracy_census": census,
        },
    }


def fig_ctx2ctx(digest: dict, figdir: str) -> None:
    fitted = [r for r in digest["pairs"] if r["status"] == "fitted"]
    classes = [
        "assistant: framing to framing",
        "character to character",
        "character to assistant",
        "assistant to character",
    ]
    class_short = {
        "assistant: framing to framing": "framing pairs\n(assistant)",
        "character to character": "character to\ncharacter",
        "character to assistant": "character to\nassistant",
        "assistant to character": "assistant to\ncharacter",
    }
    # summary strip
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    rng = np.random.default_rng(42)
    for ci, cls in enumerate(classes):
        for ai, arm in enumerate(("context", "prefix")):
            vals = [r["fitted_r2"] for r in fitted if r["pair_class"] == cls and r["arm"] == arm]
            if not vals:
                continue
            x0 = ci + (-0.18 if ai == 0 else 0.18)
            xs = x0 + rng.uniform(-0.09, 0.09, len(vals))
            ax.scatter(
                xs,
                vals,
                s=16,
                alpha=0.55,
                color=ARM_COLOR[arm],
                label=ARM_LABEL[arm] if ci == 0 else None,
                linewidths=0,
            )
            ax.plot(
                [x0 - 0.13, x0 + 0.13],
                [np.median(vals)] * 2,
                color=ARM_COLOR[arm],
                lw=2.4,
                solid_capstyle="butt",
            )
    ax.axhline(0.0, color="0.55", lw=0.8, ls=":")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([class_short[c] for c in classes])
    ax.set_ylabel("held-out R² of cell-to-cell context map")
    ax.legend(loc="upper right")
    set_title_subtitle(
        ax,
        "Cross-cell context maps by pair class",
        "each point is one fitted source-to-target cell pair; bars mark medians",
    )
    savefig_paper(fig, f"{figdir}/ctx2ctx_pair_r2_by_class", dir="figures/")
    plt.close(fig)

    # low-level: fitted vs identity+bias
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    xlim = (-2.2, 1.05)
    n_off = 0
    for arm in ("context", "prefix"):
        rows = [r for r in fitted if r["arm"] == arm]
        x = np.array([r["identity_bias_r2"] for r in rows])
        y = np.array([r["fitted_r2"] for r in rows])
        n_off += int((x < xlim[0]).sum())
        ax.scatter(
            x,
            y,
            s=14,
            alpha=0.55,
            color=ARM_COLOR[arm],
            label=ARM_LABEL[arm],
            linewidths=0,
        )
    lo, hi = -2.2, 1.05
    ax.plot([lo, hi], [lo, hi], color="0.4", lw=0.9, ls="--")
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.35, 1.0)
    # label extremes: the best fitted pair (top right) + the largest gain over identity
    best_fit = max(fitted, key=lambda r: r["fitted_r2"])
    ax.text(
        best_fit["identity_bias_r2"] - 0.03,
        best_fit["fitted_r2"] + 0.03,
        f"{cell_label(best_fit['source_cell'])} → "
        f"{IDENTITY_LABEL[parse_cell(best_fit['target_cell'])['identity']]}",
        fontsize=6.5,
        ha="right",
    )
    in_view = [r for r in fitted if r["identity_bias_r2"] >= -1.2 and r["arm"] == "context"]
    best_gain = max(in_view, key=lambda r: r["delta_r2_fitted_minus_identity"])
    ax.text(
        best_gain["identity_bias_r2"],
        best_gain["fitted_r2"] + 0.035,
        f"{cell_label(best_gain['source_cell'])} → "
        f"{IDENTITY_LABEL[parse_cell(best_gain['target_cell'])['identity']]}",
        fontsize=6.5,
        ha="center",
    )
    ax.set_xlabel("identity + per-pair bias baseline R²")
    ax.set_ylabel("fitted ridge map R²")
    ax.legend(loc="upper left")
    set_title_subtitle(
        ax,
        "Fitted cross-cell map vs identity baseline, per pair",
        "points above the dashed diagonal clear the identity + bias baseline",
    )
    savefig_paper(fig, f"{figdir}/ctx2ctx_fitted_vs_identity", dir="figures/")
    plt.close(fig)
    digest["summary"]["figure_notes"] = {
        "fitted_vs_identity_points_left_of_xlim": n_off,
        "xlim": list(xlim),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Grid 2: pool_specialize


def build_pool_specialize(staged: Path) -> dict:
    agg = json.load(open(staged / "pool_specialize/aggregate.json"))
    per_cell = []
    summary_by_arm = {}
    for arm in ("context", "prefix"):
        a = agg["arms"][arm]
        cells = a["per_cell"]
        n_usable = 0
        fr_m2 = []
        n_spec_95 = 0
        for cell, e in sorted(cells.items()):
            ceil = e["ceiling"]
            frac = e["fraction_of_ceiling"]
            rec = {
                "cell": cell,
                "arm": arm,
                "ceiling_r2": ceil["ceiling_r2"],
                "ceiling_usable": bool(ceil["usable"]),
                "banked_null_r2_pooled_p95": ceil["banked_null_r2_pooled_p95"],
                "r2_pooled_m0": e["r2_point_fixed_mean"]["m0"],
                "r2_m1_bias": e["r2_point_fixed_mean"]["m1"],
                "r2_specialized_m2_k8": e["r2_point_fixed_mean"]["m2_k8"],
                "r2_specialized_m2_k32": e["r2_point_fixed_mean"]["m2_k32"],
                "r2_specialized_m2_k128": e["r2_point_fixed_mean"]["m2_k128"],
                "r2_identity_cell": e["r2_point_fixed_mean"]["identity_cell"],
                "fraction_of_ceiling": (
                    None if frac is None else {k: frac[k] for k in ("m0", "m1", "m2_k128")}
                ),
            }
            per_cell.append(rec)
            if ceil["usable"]:
                n_usable += 1
                if frac is not None:
                    fr_m2.append(frac["m2_k128"])
                    if frac["m2_k128"] >= 0.95:
                        n_spec_95 += 1
        summary_by_arm[arm] = {
            "n_cells": a["aggregate"]["n_cells"],
            "n_ceiling_usable": n_usable,
            "cross_cell_mean_ci_r2": a["aggregate"]["r2_cell_mean_ci"],
            "increment_m1_minus_m0_mean_ci": a["aggregate"]["increment_m1_minus_m0"],
            "increments_m2_minus_m1_mean_ci": a["aggregate"]["increments_m2_minus_m1"],
            "fraction_of_ceiling_m2_k128": med_iqr(fr_m2),
            "n_cells_specialized_ge_95pct_ceiling": n_spec_95,
            "note_denominator": (
                f"fraction-of-ceiling statistics over the {n_usable} cells with a "
                "usable banked ceiling (ceiling below max(0.01, banked null p95) "
                "yields a null fraction by construction)"
            ),
        }

    # spot-check: staged percell files vs aggregate
    spot = []
    for f in sorted(glob.glob(str(staged / "pool_specialize/percell/*.json"))):
        d = json.load(open(f))
        cell, arm = d["cell"], d["arm"]
        agg_e = agg["arms"][arm]["per_cell"][cell]["r2_point_fixed_mean"]
        pc = d["pooled"]["r2_mean_over_folds"]
        keys = ["m0", "m1", "m2_k8", "m2_k32", "m2_k128", "identity_cell"]
        diffs = {k: abs(agg_e[k] - pc[k]) for k in keys}
        spot.append(
            {
                "unit": os.path.basename(f),
                "cell": cell,
                "arm": arm,
                "aggregate_r2_point_fixed_mean": {k: agg_e[k] for k in keys},
                "percell_r2_mean_over_folds": {k: pc[k] for k in keys},
                "max_abs_diff": max(diffs.values()),
                "max_abs_diff_key": max(diffs, key=diffs.get),
            }
        )
    return {
        "metadata": {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_prefix": "issue2054_lattice/pool_specialize/",
            "script": "scripts/issue2054_userfigs_gaps.py",
            "aggregate_metadata": {
                k: agg["metadata"].get(k)
                for k in (
                    "git_commit",
                    "script_version",
                    "utc",
                    "bootstrap_draws",
                    "bootstrap_sstot_convention",
                    "fold_map_sha256",
                )
            },
            "provenance": PROVENANCE,
            "spot_check_note": (
                "5 staged per-cell files (seed-42 sample) compared against the "
                "aggregate; aggregate r2_point_fixed_mean uses the fixed "
                "full-scored-mean SS_tot convention while per-cell "
                "r2_mean_over_folds is fold-local, so small differences are "
                "expected — the check verifies consistency, not identity."
            ),
        },
        "per_cell": per_cell,
        "summary_by_arm": summary_by_arm,
        "spot_check": spot,
    }


def fig_pool_specialize(digest: dict, figdir: str) -> None:
    rows = digest["per_cell"]
    # summary scatter: R^2 vs banked ceiling
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for arm in ("context", "prefix"):
        r = [x for x in rows if x["arm"] == arm]
        ceil = np.array([x["ceiling_r2"] for x in r])
        m0 = np.array([x["r2_pooled_m0"] for x in r])
        m2 = np.array([x["r2_specialized_m2_k128"] for x in r])
        ax.scatter(
            ceil,
            m0,
            s=22,
            facecolors="none",
            edgecolors=ARM_COLOR[arm],
            linewidths=1.1,
            label=f"pooled map, {ARM_LABEL[arm]}",
        )
        ax.scatter(
            ceil,
            m2,
            s=22,
            color=ARM_COLOR[arm],
            linewidths=0,
            alpha=0.75,
            label=f"specialized (rank 128), {ARM_LABEL[arm]}",
        )
    lim = (-0.08, 0.62)
    ax.plot(lim, lim, color="0.4", lw=0.9, ls="--")
    ax.set_xlim(*lim)
    ax.set_ylim(-0.28, 0.62)
    top = max(rows, key=lambda x: x["ceiling_r2"])
    ax.text(
        top["ceiling_r2"] - 0.01,
        top["r2_specialized_m2_k128"] + 0.025,
        cell_label(top["cell"]),
        fontsize=6.5,
        ha="right",
    )
    ax.set_xlabel("banked within-cell fit R² (ceiling)")
    ax.set_ylabel("held-out R² of pooled / specialized map")
    ax.legend(loc="upper left", fontsize=7.5)
    set_title_subtitle(
        ax,
        "Pooled map recovers most of each cell's own ceiling",
        "one point per cell and arm; dashed line marks parity with the ceiling",
    )
    savefig_paper(fig, f"{figdir}/pool_specialize_ceiling_scatter", dir="figures/")
    plt.close(fig)

    # low-level per-cell dot plot, both arms, rows sorted by context ceiling
    ctx = {x["cell"]: x for x in rows if x["arm"] == "context"}
    pre = {x["cell"]: x for x in rows if x["arm"] == "prefix"}
    order = sorted(ctx, key=lambda c: ctx[c]["ceiling_r2"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 12.0), sharey=True)
    ys = np.arange(len(order))
    for ax, side, label in (
        (axes[0], ctx, "context arm"),
        (axes[1], pre, "prefix arm"),
    ):
        for yi, c in zip(ys, order):
            e = side[c]
            ax.plot(
                [e["ceiling_r2"]] * 2,
                [yi - 0.34, yi + 0.34],
                color="0.45",
                lw=1.3,
                solid_capstyle="butt",
            )
        ax.scatter(
            [side[c]["r2_pooled_m0"] for c in order],
            ys,
            s=20,
            facecolors="none",
            edgecolors=ARM_COLOR["context" if side is ctx else "prefix"],
            linewidths=1.1,
            label="pooled map",
        )
        ax.scatter(
            [side[c]["r2_specialized_m2_k128"] for c in order],
            ys,
            s=20,
            color=ARM_COLOR["context" if side is ctx else "prefix"],
            linewidths=0,
            alpha=0.85,
            label="specialized (rank 128)",
        )
        ax.axvline(0.0, color="0.6", lw=0.7, ls=":")
        ax.set_xlabel("held-out R²")
        ax.set_title(label, loc="left", fontsize=10)
        ax.legend(loc="lower right", fontsize=7.5)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([cell_label(c) for c in order], fontsize=6.0)
    axes[0].set_ylim(-1, len(order))
    fig.suptitle(
        "Per-cell pooled vs specialized map, with each cell's banked ceiling (grey bar)",
        x=0.02,
        ha="left",
        fontsize=11,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    savefig_paper(fig, f"{figdir}/pool_specialize_per_cell", dir="figures/")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Grid 3: pool_rungs


def build_pool_rungs(staged: Path, poolspec_digest: dict) -> dict:
    ceilings = {
        (x["cell"], x["arm"]): (x["ceiling_r2"], x["ceiling_usable"])
        for x in poolspec_digest["per_cell"]
    }
    files = sorted(glob.glob(str(staged / "pool_rungs/percell_rungs/*.json")))
    units = []
    fold_meta = None
    n_folds_skipped_const_y = 0
    for f in files:
        d = json.load(open(f))
        if fold_meta is None:
            fold_meta = d["metadata"].get("fold_map")
        folds = [x for x in d["folds"] if not x.get("skipped")]
        n_folds_skipped_const_y += len(d["folds"]) - len(folds)
        cx = [x for x in folds if x.get("degenerate_gain_rot") == "constant_x"]
        cpp = [x for x in folds if x.get("degenerate_gain_rot") == "constant_pooled_prediction"]
        r2 = {
            r: float(np.mean([x["metrics"][r]["r2"] for x in folds]))
            for r in RUNGS + ["identity_cell"]
        }
        knn10 = {
            m: float(
                np.mean(
                    [
                        x["knn"][m]["euclidean"]["acc_at_k"]["10"]
                        for x in folds
                        if m in x.get("knn", {})
                    ]
                )
            )
            for m in ("m0", "m1", "rot_scale")
        }
        key = (d["cell"], d["arm"])
        ceiling_r2, usable = ceilings.get(key, (None, False))
        frac = {r: r2[r] / ceiling_r2 for r in RUNGS} if usable and ceiling_r2 else None
        units.append(
            {
                "cell": d["cell"],
                "arm": d["arm"],
                "n_join": d["n_join"],
                "n_folds_scored": len(folds),
                "n_folds_constant_x": len(cx),
                "n_folds_constant_pooled_prediction": len(cpp),
                "gain_rot_degenerate": bool(cx or cpp),
                "degeneracy_reason": (
                    "constant_x" if cx else ("constant_pooled_prediction" if cpp else None)
                ),
                "r2_mean_over_folds": r2,
                "knn_acc_at_10_euclidean": knn10,
                "ceiling_r2": ceiling_r2,
                "ceiling_usable": usable,
                "fraction_of_ceiling": frac,
            }
        )

    summary = {}
    for arm in ("context", "prefix"):
        rows = [u for u in units if u["arm"] == arm]
        usable_rows = [u for u in rows if u["ceiling_usable"]]
        per_rung = {}
        for r in RUNGS + ["identity_cell"]:
            vals = [u["r2_mean_over_folds"][r] for u in rows]
            entry = {"r2": med_iqr(vals)}
            if r != "identity_cell":
                for thr in (0.90, 0.95):
                    entry[f"n_cells_within_{int(thr * 100)}pct_of_ceiling"] = sum(
                        1
                        for u in usable_rows
                        if u["r2_mean_over_folds"][r] >= thr * u["ceiling_r2"]
                    )
                entry["ceiling_denominator_n_cells_usable"] = len(usable_rows)
            per_rung[r] = entry
        summary[arm] = {
            "n_units": len(rows),
            "per_rung": per_rung,
            "degeneracy_census": {
                "units_gain_rot_degenerate": sum(1 for u in rows if u["gain_rot_degenerate"]),
                "degenerate_units": [
                    {"cell": u["cell"], "reason": u["degeneracy_reason"]}
                    for u in rows
                    if u["gain_rot_degenerate"]
                ],
            },
        }
    return {
        "metadata": {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_prefix": "issue2054_lattice/pool_rungs/percell_rungs/",
            "script": "scripts/issue2054_userfigs_gaps.py",
            "fold_map": fold_meta,
            "provenance": PROVENANCE,
            "rungs": RUNGS,
            "ceiling_source": (
                "banked within-cell ceilings joined from the pool_specialize "
                "aggregate (per-cell ceiling.ceiling_r2 + usable flag); "
                "degenerate gain/rot folds substitute the per-cell-bias rung "
                "predictions, named in degeneracy_census"
            ),
            "n_folds_skipped_constant_y": n_folds_skipped_const_y,
        },
        "units": units,
        "summary": summary,
    }


def fig_pool_rungs(digest: dict, figdir: str) -> None:
    units = digest["units"]
    # summary box/strip per rung by arm
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    rng = np.random.default_rng(42)
    for ri, r in enumerate(RUNGS):
        for ai, arm in enumerate(("context", "prefix")):
            vals = np.array([u["r2_mean_over_folds"][r] for u in units if u["arm"] == arm])
            x0 = ri + (-0.19 if ai == 0 else 0.19)
            bp = ax.boxplot(
                vals,
                positions=[x0],
                widths=0.3,
                showfliers=False,
                patch_artist=True,
                medianprops={"color": "0.15", "lw": 1.4},
                boxprops={
                    "facecolor": ARM_COLOR[arm],
                    "alpha": 0.35,
                    "lw": 0.8,
                    "edgecolor": ARM_COLOR[arm],
                },
                whiskerprops={"color": ARM_COLOR[arm], "lw": 0.9},
                capprops={"color": ARM_COLOR[arm], "lw": 0.9},
            )
            xs = x0 + rng.uniform(-0.07, 0.07, len(vals))
            ax.scatter(
                xs,
                vals,
                s=7,
                color=ARM_COLOR[arm],
                alpha=0.45,
                linewidths=0,
                label=ARM_LABEL[arm] if ri == 0 else None,
            )
    ax.axhline(0.0, color="0.55", lw=0.8, ls=":")
    ax.set_xticks(range(len(RUNGS)))
    ax.set_xticklabels([RUNG_LABEL[r] for r in RUNGS], fontsize=7.5)
    ax.set_ylabel("held-out R², mean over folds")
    ax.legend(loc="upper left")
    set_title_subtitle(
        ax,
        "Transfer-tier rungs on the pooled context-to-answer map",
        "one point per cell; boxes span the interquartile range across 56 cells",
    )
    savefig_paper(fig, f"{figdir}/pool_rungs_ladder", dir="figures/")
    plt.close(fig)

    # low-level per-unit profiles
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
    for ax, arm in zip(axes, ("context", "prefix")):
        rows = [u for u in units if u["arm"] == arm]
        xs = np.arange(len(RUNGS))
        for u in rows:
            ys = [u["r2_mean_over_folds"][r] for r in RUNGS]
            ax.plot(
                xs,
                ys,
                color="0.55" if u["gain_rot_degenerate"] else ARM_COLOR[arm],
                alpha=0.55 if u["gain_rot_degenerate"] else 0.28,
                lw=1.4 if u["gain_rot_degenerate"] else 0.8,
                ls="--" if u["gain_rot_degenerate"] else "-",
            )
        med = [np.median([u["r2_mean_over_folds"][r] for u in rows]) for r in RUNGS]
        ax.plot(xs, med, color=ARM_COLOR[arm], lw=2.6)
        top = max(rows, key=lambda u: u["r2_mean_over_folds"]["rot_scale"])
        bot = min(rows, key=lambda u: u["r2_mean_over_folds"]["rot_scale"])
        for u, dy, va in ((top, 0.015, "bottom"), (bot, -0.015, "top")):
            ax.text(
                xs[-1] + 0.06,
                u["r2_mean_over_folds"]["rot_scale"] + dy,
                cell_label(u["cell"]),
                fontsize=6.0,
                va=va,
                ha="left",
            )
        ax.axhline(0.0, color="0.55", lw=0.8, ls=":")
        ax.set_xticks(xs)
        ax.set_xticklabels([RUNG_LABEL[r] for r in RUNGS], fontsize=6.8, rotation=15)
        ax.set_title(ARM_LABEL[arm], loc="left", fontsize=10)
        ax.set_xlim(-0.3, len(RUNGS) + 1.6)
    axes[0].set_ylabel("held-out R², mean over folds")
    fig.suptitle(
        "Per-cell rung profiles (thick line = median; dashed grey = "
        "degenerate gain/rotation cells)",
        x=0.02,
        ha="left",
        fontsize=11,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig_paper(fig, f"{figdir}/pool_rungs_per_unit", dir="figures/")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staged", type=Path, default=Path("/tmp/issue2054_agg"))
    args = ap.parse_args()
    staged = args.staged / "issue2054_lattice"

    os.chdir(_REPO)
    set_paper_style("blog")
    pal = paper_palette_blog(4)
    ARM_COLOR["context"] = pal[0]
    ARM_COLOR["prefix"] = pal[1]
    figdir = "issue_2054/user_gaps_extension"
    (_REPO / "figures" / figdir).mkdir(parents=True, exist_ok=True)

    out_root = _REPO / "eval_results/issue_2054"

    ctx = build_ctx2ctx(staged)
    fig_ctx2ctx(ctx, figdir)
    p = out_root / "ctx2ctx_full/digest.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(ctx, indent=1))
    print(f"wrote {p}")

    ps = build_pool_specialize(staged)
    fig_pool_specialize(ps, figdir)
    p = out_root / "pool_specialize/digest.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(ps, indent=1))
    print(f"wrote {p}")

    pr = build_pool_rungs(staged, ps)
    fig_pool_rungs(pr, figdir)
    p = out_root / "pool_rungs/digest.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(pr, indent=1))
    print(f"wrote {p}")

    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
