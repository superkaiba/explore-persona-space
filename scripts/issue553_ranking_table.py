# ruff: noqa: RUF002, RUF003
# Intentional Unicode (rho, ※, Δ, −, —) in scientific docstrings + labels.
"""Task #553 — Deliverable 5: within-source context-ranking table with CIs.

Per-source Spearman across bystanders of each ranker vs the trained EOS-margin
LEVEL (primary DV) and vs the on-policy in-R emission rate (secondary DV, from
the parent ``per_cell/loc_ep1`` panel via #539's ``build_panel`` + its own
step-0 gate). Rankers: base matched-slot margin (``post-train forecast-where``
— NOT pre-training-available), own-response prior margin, cosine, and the
``z(prior_own) + z(cosine)`` stack (z-scored over the union 416-cell panel,
matching the parent's combined-predictor convention).

Slices: all 25 bystanders (self excluded) and 15 ordinary-only. The 16
per-source rho are reported as a distribution (median + IQR + per-source
values) with a 2,000-rep source-level bootstrap CI on the median; B1/C1
duplicate-dropped medians at n=15/n=14. Degenerate per-source reads (constant
DV — common for emission with exact zeros) are reported + dropped WITH COUNT,
never silently averaged (plan concern 13.9).

Top-ranker comparisons (statistics-reconciler round-1 REVISION): judged on the
PAIRED per-source difference bootstrap — the per-ranker rho are paired on the
same sources, so the demotion read is the 2,000-rep source-pair bootstrap CI
for the per-source difference (top ranker minus each runner-up), NOT marginal
median-CI overlap (biased toward preserving the incumbent).

Smoke = this exact script with reduced ``--n-marginal-boot``.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import issue539_residual_per_cohort as i539
import issue553_panel as p553
import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

RANKERS = ("margin_base_matched", "prior_margin_own", "cosine", "z_prior_plus_z_cosine")
RANKER_LABEL = {
    "margin_base_matched": "post-train forecast-where",
    "prior_margin_own": "pre-training forecast",
    "cosine": "pre-training forecast",
    "z_prior_plus_z_cosine": "pre-training forecast",
}
INLINE = {
    # (all-25, ordinary-15) vs trained margin level (task body finding 3).
    "margin_base_matched": (0.75, 0.71),
    "prior_margin_own": (0.85, 0.37),
    "cosine": (-0.23, 0.51),
    "z_prior_plus_z_cosine": (0.83, 0.54),
}
DUP_DROPS = {"n16_full": (), "n15_dropC1": ("C1",), "n14_dropB1C1": ("B1", "C1")}


def _zscore(v: np.ndarray) -> np.ndarray:
    return (v - np.mean(v)) / (np.std(v) + 1e-12)


def _per_source_rhos(
    panel: dict, ranker_vals: np.ndarray, dv_vals: np.ndarray, mask: np.ndarray
) -> tuple[dict[str, float], int]:
    """Per-source Spearman across the masked bystanders; NaN reads counted."""
    out: dict[str, float] = {}
    n_degenerate = 0
    for s in panel["_sources"]:
        m = mask & (panel["source_cid"] == s)
        rho = i539._spearman_rho(ranker_vals[m], dv_vals[m])
        if np.isnan(rho) or float(np.std(dv_vals[m])) < 1e-12:
            n_degenerate += 1
            out[s] = float("nan")
            continue
        out[s] = float(rho)
    return out, n_degenerate


def _median_block(rhos: dict[str, float], args) -> dict:
    vals = np.array([v for v in rhos.values() if not np.isnan(v)])
    n_nan = sum(1 for v in rhos.values() if np.isnan(v))
    rng = np.random.default_rng(args.seed)
    med_boot = [
        float(np.median(vals[rng.integers(0, len(vals), size=len(vals))]))
        for _ in range(args.n_marginal_boot)
    ]
    out = {
        "median": float(np.median(vals)),
        "iqr": [float(np.percentile(vals, 25)), float(np.percentile(vals, 75))],
        "n_sources_used": len(vals),
        "n_degenerate_dropped": n_nan,
        "median_ci95_source_boot": {
            "low": float(np.percentile(med_boot, 2.5)),
            "high": float(np.percentile(med_boot, 97.5)),
            "n_boot": args.n_marginal_boot,
        },
    }
    for name, drop in DUP_DROPS.items():
        if not drop:
            continue
        kept = np.array([v for s, v in rhos.items() if s not in drop and not np.isnan(v)])
        out[f"median_{name}"] = float(np.median(kept)) if len(kept) else float("nan")
    return out


def _paired_difference_block(rho_a: dict[str, float], rho_b: dict[str, float], args) -> dict:
    """Source-pair bootstrap CI for the per-source rho difference (a − b)."""
    pairs = [(rho_a[s], rho_b[s]) for s in rho_a if not (np.isnan(rho_a[s]) or np.isnan(rho_b[s]))]
    n_dropped = len(rho_a) - len(pairs)
    diffs = np.array([a - b for a, b in pairs])
    rng = np.random.default_rng(args.seed)
    med_boot, mean_boot = [], []
    for _ in range(args.n_marginal_boot):
        idx = rng.integers(0, len(diffs), size=len(diffs))
        med_boot.append(float(np.median(diffs[idx])))
        mean_boot.append(float(np.mean(diffs[idx])))
    return {
        "n_paired_sources": len(pairs),
        "n_pairs_dropped_nan": n_dropped,
        "median_difference": float(np.median(diffs)),
        "mean_difference": float(np.mean(diffs)),
        "median_diff_ci95": {
            "low": float(np.percentile(med_boot, 2.5)),
            "high": float(np.percentile(med_boot, 97.5)),
        },
        "mean_diff_ci95": {
            "low": float(np.percentile(mean_boot, 2.5)),
            "high": float(np.percentile(mean_boot, 97.5)),
        },
        "n_boot": args.n_marginal_boot,
        "method": "paired per-source difference, 2,000-rep source-pair bootstrap "
        "(statistics-reconciler round-1 REVISION; replaces marginal median-CI overlap)",
    }


def make_figure(
    table: dict,
    fig_dir: Path,
    dv_name: str = "margin_trained",
    dv_label: str = "trained EOS margin",
    fig_name: str = "ranking_table_per_source_rho",
    exploratory: bool = False,
) -> None:
    """Per-source rho strip per ranker x slice, pre/post-train visually split.

    The trained-EOS-margin DV is the hero; the secondary emission-rate DV
    renders into the exploratory dump (plan section 4) with the same layout.
    """
    set_paper_style("blog")
    colors = paper_palette(2)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, slice_name in zip(axes, ("all_25", "ordinary_15"), strict=True):
        for xi, ranker in enumerate(RANKERS):
            blk = table[dv_name][slice_name][ranker]
            vals = [v for v in blk["per_source_rho"].values() if not np.isnan(v)]
            color = colors[1] if RANKER_LABEL[ranker] == "post-train forecast-where" else colors[0]
            jitter = (np.random.default_rng(0).random(len(vals)) - 0.5) * 0.18
            ax.plot(np.full(len(vals), xi) + jitter, vals, "o", ms=3.5, alpha=0.6, color=color)
            med = blk["summary"]["median"]
            ax.plot([xi - 0.22, xi + 0.22], [med, med], color=color, lw=2.2)
        ax.axhline(0.0, color="0.4", lw=0.8)
        ax.set_xticks(range(len(RANKERS)))
        ax.set_xticklabels(
            ["base matched-slot\nmargin", "own-response\nprior", "cosine", "z(prior)+z(cos)"],
            fontsize=8,
        )
        ax.set_title(
            f"{'All 25 bystanders' if slice_name == 'all_25' else '15 ordinary bystanders'}",
            fontsize=9,
        )
    axes[0].set_ylabel(f"Per-source Spearman rho vs {dv_label}")
    fig.suptitle(
        "Within-source context ranking (16 source dots + median; orange = post-train "
        "forecast-where, blue = pre-training forecast)",
        fontsize=9,
    )
    fig.tight_layout()
    savefig_paper(fig, fig_name, dir=fig_dir)
    plt.close(fig)
    tag = "figures:exploratory" if exploratory else "figures"
    print(f"[{tag}] wrote {fig_name} to {fig_dir}")


def main(argv: list[str] | None = None) -> int:
    args = p553.common_parser(
        "Task #553 D5: within-source context-ranking table with CIs."
    ).parse_args(argv)
    t0 = datetime.now(UTC)

    panel = p553.build_margin_panel(args.i532_dir)
    step0 = p553.step0_i532(panel, args.i532_dir)
    masks = p553.cohort_masks_553(panel)

    # Secondary DV: parent emission panel via #539's loader + ITS step-0 gate.
    panel539 = i539.build_panel(args.i532_dir)
    step0_539 = i539.step0_consistency(panel539, args.i532_dir)  # sys.exit(1) on fail
    emit_map = {
        (s, b): float(e)
        for s, b, e in zip(
            panel539["source_cid"], panel539["bystander_label"], panel539["emit_rate"], strict=True
        )
    }
    emit_rate = np.array(
        [
            emit_map[(s, b)]
            for s, b in zip(panel["source_cid"], panel["bystander_label"], strict=True)
        ]
    )

    rankers = {
        "margin_base_matched": panel["margin_base_matched"],
        "prior_margin_own": panel["prior_margin_own"],
        "cosine": panel["cosine"],
        # z over the union 416-cell panel (parent combined-predictor convention).
        "z_prior_plus_z_cosine": _zscore(panel["prior_margin_own"]) + _zscore(panel["cosine"]),
    }
    dvs = {"margin_trained": panel["margin_trained"], "emission_rate": emit_rate}
    slices = {
        "all_25": ~panel["is_self"],
        "ordinary_15": masks["ordinary_cross"],
    }

    table: dict = {}
    for dv_name, dv_vals in dvs.items():
        table[dv_name] = {}
        for slice_name, mask in slices.items():
            blk: dict = {}
            for ranker, vals in rankers.items():
                rhos, _ = _per_source_rhos(panel, vals, dv_vals, mask)
                blk[ranker] = {
                    "label": RANKER_LABEL[ranker],
                    "per_source_rho": rhos,
                    "summary": _median_block(rhos, args),
                }
            # Paired top-ranker comparisons.
            medians = {r: blk[r]["summary"]["median"] for r in RANKERS}
            top = max(medians, key=lambda r: medians[r])
            comparisons = {}
            for other in RANKERS:
                if other == top:
                    continue
                comparisons[f"{top}_minus_{other}"] = _paired_difference_block(
                    blk[top]["per_source_rho"], blk[other]["per_source_rho"], args
                )
            table[dv_name][slice_name] = blk | {
                "top_ranker_by_median": top,
                "paired_differences_vs_top": comparisons,
            }

    inline_vs_reviewed = []
    for ranker, (inline_all, inline_ord) in INLINE.items():
        inline_vs_reviewed.append(
            p553.ivr_entry(
                f"median per-source rho, {ranker}, all 25",
                inline_all,
                table["margin_trained"]["all_25"][ranker]["summary"]["median"],
                False,
                "reviewed adds source-bootstrap median CI + dup-dropped medians + NaN counts",
            )
        )
        inline_vs_reviewed.append(
            p553.ivr_entry(
                f"median per-source rho, {ranker}, ordinary 15",
                inline_ord,
                table["margin_trained"]["ordinary_15"][ranker]["summary"]["median"],
                False,
                "same",
            )
        )

    results = {
        "metadata": p553.result_metadata(args, "issue553_ranking_table.py"),
        "step0_i532": step0,
        "step0_i539_emission_panel": step0_539,
        "ranking_table": table,
        "open_question_note": "the matched-slot base margin is NOT pre-training-available; "
        "closing that gap (a pre-training-available analogue) stays an open methods question "
        "either way (plan section 9.3)",
        "inline_vs_reviewed": inline_vs_reviewed,
    }
    p553.write_json(args.out_dir / "ranking_table.json", results)
    make_figure(table, args.fig_dir)
    # Exploratory dump: same strip layout for the secondary emission-rate DV.
    make_figure(
        table,
        args.fig_dir,
        dv_name="emission_rate",
        dv_label="on-policy in-R emission rate (secondary DV)",
        fig_name="ranking_table_per_source_rho_emission",
        exploratory=True,
    )

    for slice_name in ("all_25", "ordinary_15"):
        blk = table["margin_trained"][slice_name]
        meds = ", ".join(f"{r}={blk[r]['summary']['median']:+.3f}" for r in RANKERS)
        print(f"[headline] {slice_name} medians vs margin level: {meds}")
        print(f"[headline] {slice_name} top ranker: {blk['top_ranker_by_median']}")
    print(f"[done] wall={(datetime.now(UTC) - t0).total_seconds():.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
