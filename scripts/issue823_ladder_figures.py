"""Issue #823 follow-up `inconsistent-origin-persona-ladder` — P-Analysis figures + aggregation.

Reads THIS round's P-Fit artifacts (fetched from HF
``issue823_inconsistent_origin_ladder/logs/fits`` into
``eval_results/issue_823/inconsistent_origin_ladder/``) and produces:

- ``figures/issue_823/ladder_fig1_r2_vs_k.{png,pdf}``          (HERO: pooled OOF R^2 vs k)
- ``figures/issue_823/ladder_fig2_retrieval_identity.{png,pdf}`` (kNN acc@1 + identity-bias R^2)
- ``figures/issue_823/ladder_fig3_percontext_ecdf.{png,pdf}``  (per-context R^2 ECDFs, per-unit view)
- ``figures/issue_823/ladder_fig4_mixture_floor.{png,pdf}``    (observed vs implied mixture penalty)
- ``eval_results/issue_823/inconsistent_origin_ladder/ladder_analysis_summary.json``

P2 (single-split n-ladder / per-persona control) is WITHHELD under the solver-parity
contingency (G2 FAIL on cuda AND cpu-float64); it is carried as explicit
``N/A — not tested (solver-parity contingency)`` entries in the summary JSON, never as
zeros or silent omissions. This script deliberately does NOT extend
``scripts/issue823_figures.py`` (that reads the PARENT artifact ``ridge_r2_by_arm.json``
with a different schema); it follows the same conventions (load_dotenv first,
``set_paper_style`` + ``savefig_paper``, plain-English labels).

Run from the issue-823 worktree root:
    uv run python scripts/issue823_ladder_figures.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

# Project dotenv wrapper: .env load + the shared-VM thread caps (#847) — called
# BEFORE numpy freezes the BLAS pools.
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "eval_results" / "issue_823" / "inconsistent_origin_ladder"
FIG_DIR = REPO_ROOT / "figures" / "issue_823"

K_ARMS = ["k1", "k2", "k4", "k8", "k16"]
K_VALUES = [1, 2, 4, 8, 16]
# Pre-registered read-out layers (parent/#779); L19 is the P2-protocol layer,
# reported as diagnostic only (P2 itself is withheld).
READOUT_LAYERS = [14, 26, 17]
ALL_LAYERS = [14, 17, 19, 26]
LAYER_TITLE = {
    14: "layer 14 (evil read-out)",
    26: "layer 26 (sycophancy read-out)",
    17: "layer 17 (hallucination read-out)",
    19: "layer 19 (single-split protocol layer)",
}
ARM_LABEL = {
    "own": "own answer (regenerated)",
    "plain": "external answer (plain style)",
}
P2_NA = "N/A — not tested (solver-parity contingency)"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_inputs(results_dir: Path) -> dict:
    out = {}
    for name in [
        "ladder_r2_p1.json",
        "ladder_baselines.json",
        "ladder_manipulation_checks.json",
        "ladder_singlesplit_p2.json",
    ]:
        with open(results_dir / name) as f:
            out[name] = json.load(f)
    out["percontext_ladder.npz"] = np.load(
        results_dir / "percontext_ladder.npz", allow_pickle=False
    )
    return out


def fold_mean_baselines(baselines_p1: dict, arm: str, layer: int) -> dict:
    """Aggregate the per-fold baseline records for one (arm, layer) cell."""
    folds = [baselines_p1[f"{arm}:L{layer}:fold{i}"] for i in range(5)]
    agg: dict = {
        "identity_bias_r2_fold_mean": float(np.mean([f["identity_bias_r2"] for f in folds])),
        "n_pool_mean": float(np.mean([f["n_pool"] for f in folds])),
        "small_cell": any(f["small_cell"] for f in folds),
    }
    for metric in ["knn_euclidean", "knn_cosine"]:
        agg[metric] = {
            "acc_at_k": {
                k: float(np.mean([f[metric]["acc_at_k"][k] for f in folds]))
                for k in ["1", "5", "10"]
            },
            "chance_at_k": {
                k: float(np.mean([f[metric]["chance_at_k"][k] for f in folds]))
                for k in ["1", "5", "10"]
            },
            "mrr_fold_mean": float(np.mean([f[metric]["mrr"] for f in folds])),
        }
    return agg


def fig1_hero(p1: dict, out_dir: Path) -> None:
    """Pooled OOF R^2 vs k, one panel per read-out layer, own/plain anchors.

    Broken-axis rendering: the anchor arms' pooled R^2 sit at ~-9..-11 while
    the k-ladder sits at ~0.28..0.52, so a shared linear axis crushes the
    ladder; top row = ladder (true values), bottom row = anchors (true values).
    """
    pooled = p1["pooled_r2"]
    ci = p1["bootstrap"]["per_cell_ci"]
    c_ladder, c_own, c_plain = paper_palette(3)
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(10.5, 4.6),
        sharex="col",
        height_ratios=[3, 1],
    )
    for col, layer in enumerate(READOUT_LAYERS):
        ax = axes[0][col]
        ys = [pooled[f"{a}:L{layer}"] for a in K_ARMS]
        lo = [ys[i] - ci[f"{a}:L{layer}"]["ci_low"] for i, a in enumerate(K_ARMS)]
        hi = [ci[f"{a}:L{layer}"]["ci_high"] - ys[i] for i, a in enumerate(K_ARMS)]
        ax.errorbar(
            K_VALUES,
            ys,
            yerr=[lo, hi],
            marker="o",
            color=c_ladder,
            label="persona mixture (k mixed personas)",
            capsize=3,
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks(K_VALUES)
        ax.set_xticklabels([str(k) for k in K_VALUES])
        ax.set_ylim(0.0, 0.6)
        ax.set_title(LAYER_TITLE[layer])
        ax.spines["bottom"].set_linestyle((0, (3, 3)))
        bx = axes[1][col]
        bx.axhline(pooled[f"own:L{layer}"], color=c_own, ls="--", label="own answer (regenerated)")
        bx.axhline(
            pooled[f"plain:L{layer}"], color=c_plain, ls=":", label="external answer (plain style)"
        )
        bx.set_xscale("log", base=2)
        bx.set_xticks(K_VALUES)
        bx.set_xticklabels([str(k) for k in K_VALUES])
        bx.set_ylim(-12.0, -7.5)
        bx.spines["top"].set_linestyle((0, (3, 3)))
        bx.set_xlabel("number of mixed personas k")
    axes[0][0].set_ylabel("pooled held-out R$^2$")
    axes[1][0].set_ylabel("anchor arms\n(same axis units)")
    axes[0][0].legend(fontsize=8, loc="lower left")
    axes[1][0].legend(fontsize=7, loc="upper left")
    savefig_paper(fig, "ladder_fig1_r2_vs_k", dir=out_dir)
    plt.close(fig)


def fig2_retrieval_identity(p1: dict, base_agg: dict, out_dir: Path) -> None:
    """Top: kNN retrieval acc@1 vs k (cosine + euclidean). Bottom: identity-bias R^2 vs k."""
    c_ladder, c_own, c_plain = paper_palette(3)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), sharex=True)
    for col, layer in enumerate(READOUT_LAYERS):
        ax = axes[0][col]
        for metric, ls in [("knn_cosine", "-"), ("knn_euclidean", "--")]:
            ys = [base_agg[f"{a}:L{layer}"][metric]["acc_at_k"]["1"] for a in K_ARMS]
            name = "cosine" if metric == "knn_cosine" else "euclidean"
            ax.plot(
                K_VALUES, ys, marker="o", ls=ls, color=c_ladder, label=f"retrieval acc@1 ({name})"
            )
        ax.plot(
            K_VALUES,
            [base_agg[f"own:L{layer}"]["knn_cosine"]["acc_at_k"]["1"]] * 5,
            ls="--",
            color=c_own,
            label="own answer (cosine)",
        )
        ax.plot(
            K_VALUES,
            [base_agg[f"plain:L{layer}"]["knn_cosine"]["acc_at_k"]["1"]] * 5,
            ls=":",
            color=c_plain,
            label="external answer, plain (cosine)",
        )
        chance = base_agg[f"k1:L{layer}"]["knn_cosine"]["chance_at_k"]["1"]
        ax.axhline(chance, color="grey", lw=0.8, label=f"chance = {chance:.4f}")
        ax.set_xscale("log", base=2)
        ax.set_xticks(K_VALUES)
        ax.set_xticklabels([str(k) for k in K_VALUES])
        ax.set_title(LAYER_TITLE[layer])
        ax = axes[1][col]
        ys = [p1["cells"][f"{a}:L{layer}"]["identity_bias_pooled_r2"] for a in K_ARMS]
        ax.plot(K_VALUES, ys, marker="s", color=c_ladder, label="identity + learned bias")
        ax.axhline(
            p1["cells"][f"own:L{layer}"]["identity_bias_pooled_r2"],
            color=c_own,
            ls="--",
            label="own answer",
        )
        ax.axhline(
            p1["cells"][f"plain:L{layer}"]["identity_bias_pooled_r2"],
            color=c_plain,
            ls=":",
            label="external answer (plain)",
        )
        ax.axhline(0.0, color="grey", lw=0.8)
        ax.set_xlabel("number of mixed personas k")
    axes[0][0].set_ylabel("retrieval acc@1")
    axes[1][0].set_ylabel("identity+bias baseline\npooled held-out R$^2$")
    axes[0][0].legend(fontsize=7, loc="upper right")
    axes[1][0].legend(fontsize=7, loc="lower left")
    savefig_paper(fig, "ladder_fig2_retrieval_identity", dir=out_dir)
    plt.close(fig)


def fig3_percontext_ecdf(npz, out_dir: Path) -> None:
    """Per-unit companion: ECDF of per-context R^2 per arm at each read-out layer."""
    arm_names = [str(a) for a in npz["arm_names"]]
    ss_res, ss_tot = npz["p1_ss_res"], npz["p1_ss_tot"]
    cmap = matplotlib.colormaps["viridis"]
    k_colors = {a: cmap(i / max(len(K_ARMS) - 1, 1)) for i, a in enumerate(K_ARMS)}
    _, c_own, c_plain = paper_palette(3)
    arm_color = {**k_colors, "own": c_own, "plain": c_plain}
    arm_disp = {
        **{a: f"{a[1:]} mixed persona{'s' if a != 'k1' else ''}" for a in K_ARMS},
        "own": "own answer (regenerated)",
        "plain": "external answer (plain style)",
    }
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for ax, layer in zip(axes, READOUT_LAYERS):
        for arm in K_ARMS + ["own", "plain"]:
            ai = arm_names.index(arm)
            r2 = 1.0 - ss_res[ai, layer, :] / ss_tot[ai, layer, :]
            r2 = np.clip(np.sort(r2), -2.0, 1.0)
            ecdf = np.arange(1, r2.size + 1) / r2.size
            ax.plot(r2, ecdf, color=arm_color[arm], lw=1.2, label=arm_disp[arm])
        ax.set_title(LAYER_TITLE[layer])
        ax.set_xlabel("per-context R$^2$ (clipped at $-2$)")
        ax.set_xlim(-2.0, 1.0)
    axes[0].set_ylabel("fraction of contexts $\\leq$ x")
    axes[0].legend(fontsize=7, loc="upper left")
    savefig_paper(fig, "ladder_fig3_percontext_ecdf", dir=out_dir)
    plt.close(fig)


def fig5_identity_percontext_ecdf(npz, out_dir: Path) -> None:
    """Per-unit companion for the identity+bias baseline read (round-8 revision, B4).

    ECDF of per-context identity+learned-bias R^2 (out-of-fold v-hat = x + b
    predictions; 1 - ss_res_i / ss_tot_i from the committed per-context sums
    of squares) per arm at each read-out layer. Clip at -6 for display: the
    layer-26 lower tail reaches ~-25 (1st percentile) while every median sits
    above -4. Retrieval has NO per-context companion — per-context ranks were
    not persisted (fold-level acc@k / median rank / MRR only); the body carries
    the explicit per-unit exemption for that read.
    """
    arm_names = [str(a) for a in npz["arm_names"]]
    ss_res, ss_tot = npz["p1_identity_ss_res"], npz["p1_identity_ss_tot"]
    cmap = matplotlib.colormaps["viridis"]
    k_colors = {a: cmap(i / max(len(K_ARMS) - 1, 1)) for i, a in enumerate(K_ARMS)}
    _, c_own, c_plain = paper_palette(3)
    arm_color = {**k_colors, "own": c_own, "plain": c_plain}
    arm_disp = {
        **{a: f"{a[1:]} mixed persona{'s' if a != 'k1' else ''}" for a in K_ARMS},
        "own": "own answer (regenerated)",
        "plain": "external answer (plain style)",
    }
    clip_lo = -6.0
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for ax, layer in zip(axes, READOUT_LAYERS):
        for arm in K_ARMS + ["own", "plain"]:
            ai = arm_names.index(arm)
            r2 = 1.0 - ss_res[ai, layer, :] / ss_tot[ai, layer, :]
            r2 = np.clip(np.sort(r2), clip_lo, 1.0)
            ecdf = np.arange(1, r2.size + 1) / r2.size
            ax.plot(r2, ecdf, color=arm_color[arm], lw=1.2, label=arm_disp[arm])
        ax.set_title(LAYER_TITLE[layer])
        ax.set_xlabel("identity+bias R$^2$ (clipped at $-6$)")
        ax.set_xlim(clip_lo, 1.0)
    axes[0].set_ylabel("fraction of contexts $\\leq$ x")
    axes[0].legend(fontsize=7, loc="upper left")
    savefig_paper(fig, "ladder_fig5_identity_percontext_ecdf", dir=out_dir)
    plt.close(fig)


def fig4_mixture_floor(p1: dict, out_dir: Path) -> None:
    """Observed R^2 drop vs the implied mechanical mixture penalty; fixed-denominator re-read."""
    mf = p1["mixture_floor"]
    pooled = p1["pooled_r2"]
    pal = paper_palette(5)
    c_obs, c_imp, c_raw, c_fix = pal[0], pal[3], pal[0], pal[4]
    ks = [2, 4, 8, 16]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), sharex=True)
    for col, layer in enumerate(READOUT_LAYERS):
        ax = axes[0][col]
        obs = [mf["implied_mixture_penalty"][f"k{k}:L{layer}"]["observed_delta_vs_k1"] for k in ks]
        imp = [mf["implied_mixture_penalty"][f"k{k}:L{layer}"]["implied_r2_penalty"] for k in ks]
        ax.plot(ks, obs, marker="o", color=c_obs, label="observed R$^2$ drop vs k=1")
        ax.plot(ks, imp, marker="^", color=c_imp, label="implied mechanical mixture penalty")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_title(LAYER_TITLE[layer])
        ax = axes[1][col]
        raw = [pooled[f"{a}:L{layer}"] for a in K_ARMS]
        fixed = [mf["fixed_reference_denominator_r2"][f"{a}:L{layer}"] for a in K_ARMS]
        ax.plot(K_VALUES, raw, marker="o", color=c_raw, label="pooled R$^2$ (own denominator)")
        ax.plot(
            K_VALUES,
            fixed,
            marker="d",
            ls="--",
            color=c_fix,
            label="pooled R$^2$ (fixed k=1 denominator)",
        )
        ax.set_xticks(K_VALUES)
        ax.set_xticklabels([str(k) for k in K_VALUES])
        ax.set_xlabel("number of mixed personas k")
    axes[0][0].set_ylabel("R$^2$ drop vs k=1")
    axes[1][0].set_ylabel("pooled held-out R$^2$")
    axes[0][0].legend(fontsize=7, loc="upper left")
    axes[1][0].legend(fontsize=7, loc="lower left")
    savefig_paper(fig, "ladder_fig4_mixture_floor", dir=out_dir)
    plt.close(fig)


def build_summary(inputs: dict, results_dir: Path) -> dict:
    p1 = inputs["ladder_r2_p1.json"]
    baselines = inputs["ladder_baselines.json"]
    manip = inputs["ladder_manipulation_checks.json"]
    p2 = inputs["ladder_singlesplit_p2.json"]

    base_agg = {
        f"{arm}:L{layer}": fold_mean_baselines(baselines["p1"], arm, layer)
        for arm in K_ARMS + ["own", "plain"]
        for layer in ALL_LAYERS
    }
    cells = {}
    for arm in K_ARMS + ["own", "plain"]:
        for layer in ALL_LAYERS:
            key = f"{arm}:L{layer}"
            cell = p1["cells"][key]
            cells[key] = {
                "pooled_r2": cell["pooled_r2"],
                "fold_mean_r2": cell["fold_mean_r2"],
                "bootstrap_ci_95": p1["bootstrap"]["per_cell_ci"].get(key),
                "identity_bias_pooled_r2": cell["identity_bias_pooled_r2"],
                "estimator_degenerate": cell["estimator_degenerate"],
                "n_train_per_fold": sorted({f["n_train"] for f in cell["folds"]}),
                "baselines": base_agg[key],
                "layer_role": (
                    "read-out headline"
                    if layer in READOUT_LAYERS
                    else "P2-protocol layer (diagnostic only; P2 withheld)"
                ),
            }
    n_train = sorted({f["n_train"] for c in p1["cells"].values() for f in c["folds"]})
    g2 = p1["gates"]["g2"]
    summary = {
        "metadata": {
            "script": "scripts/issue823_ladder_figures.py",
            "task": 823,
            "followup_label": "inconsistent-origin-persona-ladder",
            **as_metadata_dict(git_provenance(), phase="panalysis"),
            "numpy_version": np.__version__,
            "source_artifacts": {
                name: sha256_file(results_dir / name)
                for name in [
                    "ladder_r2_p1.json",
                    "ladder_baselines.json",
                    "ladder_manipulation_checks.json",
                    "ladder_singlesplit_p2.json",
                    "percontext_ladder.npz",
                ]
            },
            "fits_metadata": p1["metadata"],
        },
        "headline": {
            "delta_mean": p1["delta_mean"],
            "ci_low_delta_mean": p1["ci_low_delta_mean"],
            "ci_high_delta_mean": p1["ci_high_delta_mean"],
            "delta_per_layer": p1["delta_per_layer"],
            "spearman_k_r2_descriptive": p1["spearman_k_r2_descriptive"],
            "lattice": p1["lattice"],
            "lattice_thresholds": p1["lattice_thresholds"],
        },
        "cells": cells,
        "mixture_floor": p1["mixture_floor"],
        "conditioning_and_lambda_diagnostics": {
            "primary_estimator": p1["primary_estimator"],
            "solver_mode": p1["metadata"]["solver_mode"],
            "n_train_per_fold": n_train,
            "d": 3584,
            "n_over_d_ratio": [round(n / 3584, 4) for n in n_train],
            "per_fold_lambda_dof_persisted": (
                "NULL in the persisted artifact — the canonical-contingency solver did not "
                "persist per-fold selected lambda/dof into ladder_r2_p1.json cells; the only "
                "persisted lambda evidence is the G2 parity slices below (lambda_gram == "
                "lambda_canonical == 0.01 on all 6 probed (layer, fold) slices)"
            ),
            "sensitivity_dof_capped": p1["sensitivity_dof_capped"],
            "dof_cap_bindable": p1["dof_cap_bindable"],
            "g2_parity_gate": {
                "pass": g2["pass"],
                "contingency_engaged": g2["contingency_engaged"],
                "tolerances": g2["tolerances"],
                "slices": g2["slices"],
            },
            "g1_reproduce_gate_pass": p1["gates"]["g1"]["pass"],
            "conditioning_caveat": (
                "n_train ~= 3703 vs d = 3584 (ratio ~1.033, +119 above well-posedness); the "
                "G2 solver-parity gate FAILED on cuda AND cpu-float64 (delta_r2 up to 3.6e-4 "
                "vs 1e-4 tolerance), so R^2 reads are conditioning-sensitive; the canonical "
                "solver (G1 PASS vs banked parent values) produced all reported fits"
            ),
        },
        "drop_accounting": {
            k: v for k, v in p1["drop_accounting"].items() if k != "new_dropped_ids_union"
        },
        "manipulation_checks": {
            "distinct": manip["distinct"],
            "m1_pass": manip["m1_tfidf"]["m1_pass"],
            "m1_mean_within_context_cross_persona_tfidf_cosine": manip["m1_tfidf"][
                "mean_within_context_cross_persona_tfidf_cosine"
            ],
            "m1_bar": manip["m1_tfidf"]["bar"],
            "m2_pass": manip["m2_paired_separation"]["m2_pass"],
            "m2_n_personas_passing": manip["m2_paired_separation"]["n_personas_passing"],
            "m3_gate": manip["m3_accounting"]["capture_cap_hit"]["gate"],
            "m3_worst_cap_hit_fraction_realized": max(
                p["cap_hit_fraction_realized"]
                for p in manip["m3_accounting"]["capture_cap_hit"]["per_persona"].values()
            ),
        },
        "p2_single_split": {
            "status": p2["status"],
            "reason": p2["reason"],
            "n_ladder": P2_NA,
            "per_persona_control": P2_NA,
            "g_mix": P2_NA,
            "minimum_context_question": (
                "UNANSWERED this round: the n-ladder (minimum contexts for a well-posed "
                "mapping) is withheld under the solver-parity contingency; additionally the "
                "d-boundary rung was UNREALIZABLE at the realized mask (realized train 3336 "
                "< d = 3584 after drops)"
            ),
            "d_boundary": p2["split"]["d_boundary"],
            "realized_split": p2["split"]["realized"],
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=FIG_DIR)
    parser.add_argument(
        "--only-fig5",
        action="store_true",
        help=(
            "Render ONLY the fig5 identity per-context ECDF companion (round-8 "
            "revision). Skips figs 1-4 AND the summary-JSON rewrite so the "
            "already-pinned sidecars/summary are not overwritten."
        ),
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    set_paper_style()
    if args.only_fig5:
        npz = np.load(args.results_dir / "percontext_ladder.npz", allow_pickle=False)
        fig5_identity_percontext_ecdf(npz, args.out_dir)
        print(f"wrote {args.out_dir / 'ladder_fig5_identity_percontext_ecdf'}.png")
        return

    inputs = load_inputs(args.results_dir)
    p1 = inputs["ladder_r2_p1.json"]
    summary = build_summary(inputs, args.results_dir)

    fig1_hero(p1, args.out_dir)
    fig2_retrieval_identity(
        p1, {k: v["baselines"] for k, v in summary["cells"].items()}, args.out_dir
    )
    fig3_percontext_ecdf(inputs["percontext_ladder.npz"], args.out_dir)
    fig4_mixture_floor(p1, args.out_dir)

    out_json = args.results_dir / "ladder_analysis_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_json}")
    for stem in [
        "ladder_fig1_r2_vs_k",
        "ladder_fig2_retrieval_identity",
        "ladder_fig3_percontext_ecdf",
        "ladder_fig4_mixture_floor",
    ]:
        print(f"wrote {args.out_dir / stem}.png")


if __name__ == "__main__":
    main()
