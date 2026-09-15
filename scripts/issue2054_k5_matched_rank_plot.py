"""Plot character-shift spectra and report query dependence with consistent denominators."""

# ruff: noqa: E402
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from explore_persona_space.analysis.c2a_plot_style import (
    ROLES,
    c2a_figure,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
)
from scripts import issue2054_k5_loso_calibration as base

LABELS = {2: "HELIOS", 3: "Wren", 4: "Dana", 5: "Vex"}
STYLES = {base.MODELS[0]: ROLES["base_model"], base.MODELS[1]: ROLES["post_trained"]}


def spread(values):
    """Unweighted pair summaries; min/max are pair ranges, not confidence intervals."""
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "min": float(a.min()), "max": float(a.max())}


def summarize(data):
    """Summarize exactly six unordered character pairs for every model and summary."""
    groups = []
    for model in base.MODELS:
        for arm in ("context", "answer"):
            rows = [r for r in data["pairs"] if r["model"] == model and r["arm"] == arm]
            if len(rows) != 6:
                raise RuntimeError("expected all six character pairs")
            result = {
                "model": model,
                "arm": arm,
                "n_pairs": 6,
                "spectra": {},
                "constancy": {},
                "heldout": {},
            }
            for mode in ("raw", "centered"):
                result["spectra"][mode] = {
                    k: spread([r["spectra"][mode][k] for r in rows])
                    for k in (
                        "top1",
                        "top10",
                        "r50",
                        "r90",
                        "r95",
                        "participation_ratio",
                        "stable_rank",
                    )
                }
                result["heldout"][mode] = {
                    str(k): spread([r["heldout"][mode]["fraction_at_rank"][str(k)] for r in rows])
                    for k in (1, 10, 100, 300)
                }
            for k in (
                "constant_vector_fraction",
                "variable_amplitude_mean_direction_fraction",
                "fraction_of_constant_residual_perpendicular_to_mean",
                "shift_norm_cv",
            ):
                result["constancy"][k] = spread([r["query_constancy"][k] for r in rows])
            result["constancy"]["median_cosine"] = spread(
                [r["query_constancy"]["cosine_quantiles_10_50_90"][1] for r in rows]
            )
            groups.append(result)
    return groups


def save(fig, name, title, subject, args):
    """Use the shared paper style and bind each plot to its exact input result."""
    saved = save_c2a_figure(
        fig, args.figures / name, title=title, subject=subject, creator=Path(__file__).name
    )
    base.atomic_json(
        args.figures / f"{name}.meta.json",
        {
            "render": saved["record"],
            "results_sha256": base.sha(args.out / "results.json"),
            "script_sha256": base.sha(__file__),
            "subject": subject,
            "color_meaning": {m: STYLES[m].color for m in base.MODELS},
        },
    )
    plt.close(fig)


def plot_spectra(data, args):
    """Show full-cohort energy spectra and held-out projection without mixing models."""
    cache = {}
    for r in data["pairs"]:
        path = (
            args.out
            / "pairs"
            / f"{r['model']}__{r['source_index']}_{r['target_index']}__{r['arm']}.npz"
        )
        if base.sha(path) != r["array_sha256"]:
            raise RuntimeError("plot input array hash mismatch")
        with np.load(path, allow_pickle=False) as z:
            cache[(r["model"], r["arm"], r["source_index"], r["target_index"])] = {
                k: z[k]
                for k in (
                    "raw_energy",
                    "centered_energy",
                    "raw_heldout_fraction",
                    "centered_heldout_fraction",
                )
            }
    fig, _ = c2a_figure("full", aspect=1.04)
    axes = fig.subplots(2, 2)
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.16, top=0.88, wspace=0.22, hspace=0.61)
    for row, mode in enumerate(("raw", "centered")):
        for col, arm in enumerate(("context", "answer")):
            ax = axes[row, col]
            for model in base.MODELS:
                arrays = [v for (m, a, _, _), v in cache.items() if m == model and a == arm]
                n = min(len(z[f"{mode}_energy"]) for z in arrays)
                curves = np.stack(
                    [np.cumsum(z[f"{mode}_energy"])[:n] / z[f"{mode}_energy"].sum() for z in arrays]
                )
                x = np.arange(1, n + 1)
                style = STYLES[model]
                ax.fill_between(
                    x,
                    100 * curves.min(0),
                    100 * curves.max(0),
                    color=style.color,
                    alpha=0.12,
                    linewidth=0,
                )
                ax.plot(
                    x,
                    100 * curves.mean(0),
                    color=style.color,
                    linewidth=2,
                    marker=style.marker,
                    markersize=5,
                    markevery=[k for k in (0, 9, 99, 999) if k < n],
                )
                n_test = min(len(z[f"{mode}_heldout_fraction"]) for z in arrays)
                heldout = np.stack([z[f"{mode}_heldout_fraction"][:n_test] for z in arrays])
                ax.plot(
                    np.arange(1, n_test),
                    100 * heldout.mean(0)[1:],
                    color=style.color,
                    linestyle="--",
                    linewidth=1.7,
                )
            ax.set_xscale("log")
            ax.set_xlim(1, 1450)
            ax.set_xticks(
                [1, 3, 10, 30, 100, 300, 1000], ["1", "3", "10", "30", "100", "300", "1000"]
            )
            ax.set_ylim(0, 101)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.set_xlabel("Number of directions")
            if col == 0:
                ax.set_ylabel("Squared shift energy captured (%)")
            title = "Raw shifts" if mode == "raw" else "After subtracting mean shift"
            panel_header(
                ax, "ABCD"[2 * row + col], "Contexts" if arm == "context" else "Answers", title
            )
    handles = [
        Line2D([], [], color=STYLES[m].color, marker=STYLES[m].marker, linewidth=2, label=label)
        for m, label in zip(base.MODELS, ("Base", "Instruct"), strict=True)
    ]
    handles += [
        Line2D([], [], color="0.25", label="All-query spectrum", linewidth=2),
        Line2D([], [], color="0.25", linestyle="--", label="Held-out projection", linewidth=1.7),
    ]
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.035), ncol=2, frameon=False
    )
    save(
        fig,
        "character_shift_rank",
        "Rank of character-to-character shifts",
        "Squared singular-value energy; raw versus mean-centered differences. Lines are unweighted means across six matched-query character pairs; bands are pair min/max, not confidence intervals. Solid: descriptive full-cohort spectrum. Dashed: projection onto subspaces learned on the other four conversation folds. Held-out projection uses observed target differences, not source-only predictions. Each row normalizes by its corresponding raw or centered energy.",
        args,
    )


def plot_constancy(groups, args):
    """Compare fixed shift with an oracle signed amplitude along the same mean line."""
    fig, _ = c2a_figure("full", aspect=0.51)
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.26, top=0.74, wspace=0.23)
    metrics = ("constant_vector_fraction", "variable_amplitude_mean_direction_fraction")
    for j, arm in enumerate(("context", "answer")):
        ax = axes[j]
        for i, model in enumerate(base.MODELS):
            r = next(g for g in groups if g["model"] == model and g["arm"] == arm)
            for k, metric in enumerate(metrics):
                s = r["constancy"][metric]
                x = i + (k - 0.5) * 0.32
                ax.bar(
                    x,
                    100 * s["mean"],
                    width=0.29,
                    color=STYLES[model].color,
                    alpha=1 if k == 0 else 0.55,
                    hatch=None if k == 0 else "///",
                    edgecolor=STYLES[model].color,
                )
                ax.errorbar(
                    x,
                    100 * s["mean"],
                    yerr=np.array([[100 * (s["mean"] - s["min"])], [100 * (s["max"] - s["mean"])]]),
                    fmt="none",
                    color="0.25",
                    capsize=4,
                    linewidth=1,
                )
        ax.set_xticks([0, 1], ["Base", "Instruct"])
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel("Total squared shift explained (%)")
        panel_header(
            ax,
            "AB"[j],
            "Contexts" if arm == "context" else "Answers",
            "Fixed versus varying strength",
        )
    fig.legend(
        handles=[
            Patch(facecolor="0.5", label="One constant vector"),
            Patch(
                facecolor="0.75", hatch="///", label="Same line, query-dependent signed strength"
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=1,
        frameon=False,
    )
    save(
        fig,
        "character_shift_constancy",
        "Are shifts the same for each query?",
        "Held-out fraction of total squared character shift explained, relative to zero shift. Fixed vector is the other-fold training mean. The comparison projects each observed test difference onto that mean's line with a freely fitted signed coefficient; this is an oracle coverage diagnostic, not a source-only prediction. Bars are unweighted means across six character pairs; whiskers are pair ranges, not confidence intervals.",
        args,
    )


def write_report(data, groups, args):
    """Persist exact tables, metric definitions and limits alongside the plots."""
    lines = [
        "# Character-shift rank and query dependence",
        "",
        data["method"],
        "",
        "For each character pair, D[q,:] is the difference for the same full query. If every query receives exactly one fixed vector b, raw D has rank one and centered D has rank zero. Rank one alone allows a query-dependent signed amplitude. We therefore measure both spectral dimensionality and deviation from the training-mean vector.",
        "",
        "## Query constancy",
        "",
        "The Fixed vector and Same line percentages use total squared raw shift as denominator. Entries are unweighted means across the six character pairs. Fixed b is estimated on four global conversation folds and evaluated on the fifth. The variable-strength column is an oracle projection of the observed held-out difference onto the line spanned by b. Its signed coefficients use the target, so it measures geometric coverage rather than prediction. Perpendicular residual instead uses total fixed-vector residual energy as denominator: it is the fraction of fixed-vector error that remains even after allowing signed strength to vary.",
        "",
        "| Model | Representation | Fixed vector | Same line, varying signed strength | Perpendicular share of fixed-vector residual | Mean of pair-median cosines to b |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for g in groups:
        c = g["constancy"]
        lines.append(
            f"| {g['model']} | {g['arm']} | {100 * c['constant_vector_fraction']['mean']:.2f}% | {100 * c['variable_amplitude_mean_direction_fraction']['mean']:.2f}% | {100 * c['fraction_of_constant_residual_perpendicular_to_mean']['mean']:.2f}% | {c['median_cosine']['mean']:.3f} |"
        )
    lines += [
        "",
        "## Spectral rank",
        "",
        "Spectral energy is sigma_i squared. r90 is the smallest number of directions capturing 90% of the observed squared energy, not algebraic or noise-corrected population rank. Stable rank = sum(energy)/max(energy); participation ratio = sum(energy)^2/sum(energy^2). Full spectra are descriptive on all retained queries. Held-out projection curves learn their bases on the other four folds and use the observed held-out difference to measure coverage; no unseen-persona or prompt-family claim. Ranges below are the min/max across six pairs.",
        "",
        "| Model | Representation | Spectrum | Top direction (mean) | Top 10 (mean) | r50 range | r90 range | r95 range | Participation ratio range | Held-out top 10 (mean) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for g in groups:
        for mode in ("raw", "centered"):
            s = g["spectra"][mode]
            ranges = " | ".join(
                f"{s[k]['min']:.0f}–{s[k]['max']:.0f}"
                for k in ("r50", "r90", "r95", "participation_ratio")
            )
            lines.append(
                f"| {g['model']} | {g['arm']} | {mode} | {100 * s['top1']['mean']:.2f}% | {100 * s['top10']['mean']:.2f}% | {ranges} | {100 * g['heldout'][mode]['10']['mean']:.2f}% |"
            )
    lines += [
        "",
        "## Rank of mean shifts between characters",
        "",
        "Four character means are estimated on exactly the same all-character query intersection. Their centered contrast matrix has rank at most three by construction. The six pairwise mean differences have the same normalized spectrum; this identity is numerically verified. A small mean-contrast rank does not establish that each query has the same shift.",
        "",
        "| Model | Representation | Common queries | First direction | First two directions | r95 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in data["mean_shifts"]:
        path = args.out / "means" / f"{r['model']}__{r['arm']}.npz"
        if base.sha(path) != r["array_sha256"]:
            raise RuntimeError("mean-shift array changed")
        with np.load(path, allow_pickle=False) as z:
            e = z["centroid_energy"]
        lines.append(
            f"| {r['model']} | {r['arm']} | {r['n_common_queries']} | {100 * e[0] / e.sum():.2f}% | {100 * e[:2].sum() / e.sum():.2f}% | {r['r95']} |"
        )
    lines += [
        "",
        "## Per-pair values",
        "",
        "| Model | Representation | Pair | Queries | Fixed vector (%) | Varying strength (%) | Raw r90 | Centered r90 | Centered held-out top 10 (%) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in data["pairs"]:
        c = r["query_constancy"]
        lines.append(
            f"| {r['model']} | {r['arm']} | {LABELS[r['source_index']]} ↔ {LABELS[r['target_index']]} | {r['n']} | {100 * c['constant_vector_fraction']:.2f} | {100 * c['variable_amplitude_mean_direction_fraction']:.2f} | {r['spectra']['raw']['r90']} | {r['spectra']['centered']['r90']} | {100 * r['heldout']['centered']['fraction_at_rank']['10']:.2f} |"
        )
    lines += [
        "",
        "## Scope, limitations and reproduction",
        "",
        *[f"- {s}" for s in data["limitations"]],
        "",
        "All 24 planned pair/representation/model panels, 120 fold evaluations and four mean-shift panels completed. The constant-vector per-query errors and aggregate fraction match the prior strict matched-query analysis. All input banks, parent cohort arrays, output arrays and source code are fingerprinted. Exact-constant synthetic controls return raw rank one and centered rank zero; direct-SVD tests validate the Gram calculation for both n<d and n>d. Tiny eigenvalues below the recorded Gram roundoff tolerance are excluded from projection bases; this is numerical resolution, not denoising.",
        "",
        "Reproduce: `issue2054_k5_matched_run.py --rank --out eval_results/issue_2054/k5_matched_rank --inputs eval_results/issue_2054/k5_matched_offsets_strict/inputs.json`, then `issue2054_k5_matched_rank_plot.py --out eval_results/issue_2054/k5_matched_rank --figures figures/issue_2054/k5_matched_rank`. Use a fresh output directory. Eight BLAS threads per worker, two checkpoint workers; no new generation or downloads. Parent result SHA-256: "
        + data["parent_results_sha256"]
        + ".",
        "",
    ]
    (args.out / "README.md").write_text("\n".join(lines))


def main():
    """Render reviewed completed spectra and exact summaries."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--figures", type=Path, required=True)
    args = p.parse_args()
    data = json.loads((args.out / "results.json").read_text())
    if data["status"] != "complete" or len(data["pairs"]) != 24 or len(data["mean_shifts"]) != 4:
        raise RuntimeError("incomplete rank analysis")
    groups = summarize(data)
    base.atomic_json(args.out / "summary.json", groups)
    set_c2a_style()
    args.figures.mkdir(parents=True, exist_ok=True)
    plot_spectra(data, args)
    plot_constancy(groups, args)
    write_report(data, groups, args)


if __name__ == "__main__":
    main()
