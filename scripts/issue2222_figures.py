"""P5 figures for issue #2222 — predictor-arm comparison + exploratory dump.

NEW unit-3 file (plan v5 §6 "Figures to produce"). Renders EVERY figure from
the persisted P3/P4 artifacts only — no recomputation of statistics:

- ``eval_results/issue_2222/predictor_correlations.json`` (records +
  hypothesis_tests + arm_order + dataset_values)
- ``eval_results/issue_2222/{map_quality,auc_misaligned2_vs_normal,tuned_map,
  form_b_regression,form_a_probe}.json``
- ``data/issue_2222/nulls/{boot_flat,perm}_<trait>_<arm>.npz`` — the persisted
  bootstrap-r and permutation-|r| matrices (plan §6 selection-symmetric-nulls
  block registers them for exactly this post-hoc read).

Figures (hero first, then the over-produced exploratory dump):

1. hero_arm_r_comparison — per-trait grouped bars of Pearson r at the
   pre-registered steering layer, arms in ``arm_order``, 95% bootstrap-CI
   whiskers from the persisted flat boot matrices, published Qwen r values as
   reference marks (sycophancy panel foregrounded).
2. scatter_per_dataset_<trait> — the per-unit companion: per-arm panels of the
   24 dataset-level predictor values vs post-finetuning trait score, points
   marked by dataset version and labeled by family (raw data alongside the
   aggregate hero).
3. layer_sweep_<trait> — per-arm r-per-layer curves with the per-layer
   permutation p97.5 |r| null band + steering-layer marker.
4. auc_by_arm — sample-level AUC (misaligned II vs normal) per arm x trait.
4b. roc_by_arm — sample-level ROC curves per arm (plan §6 item 3): rank-
    cumulative rendering of the per-sample scores PERSISTED by stage_aggregate
    (``data/issue_2222/nulls/roc_scores_<trait>.npz``); AUC legend values come
    from the persisted auc json — no statistic recomputed.
5. map_quality_r2 (+ _full raw-scale twin) — frozen-map held-out R^2 per layer
   vs identity / identity+bias baselines and the exploratory tuned map.
6. map_quality_knn — kNN retrieval acc@1 per stored layer x prediction kind,
   per-LOFO-fold reads with the chance floor.
7. ci_flat_vs_clustered — H1/H2 delta-r with flat vs family-clustered and
   frozen vs selection-inherited 95% CIs.
8. form_a_probe_r2 / form_a_probe_summary — exploratory Form-A probe panels.
9. form_b_cosine — exploratory Form-B coefficient cosine to r_B per fold.
10. tuned_vs_frozen_<trait-grid> — frozen mapped-arm r per layer vs the
    exploratory tuned map.

Conventions: paper-plots SKILL (one color = one meaning across every figure —
the arm palette is pinned module-level; no text-overlay annotations beyond the
sanctioned correlation r/p labels; plain-English labels only). Errorbar
offsets are clamped element-wise to >= 0 (gotchas.md xerr/yerr rule; a
quantile CI can invert around the point estimate at tiny n —
tests/test_issue2222_figures.py routes an inverted CI through the real hero
function).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# load_dotenv BEFORE any heavy import (numpy/matplotlib below) so the #847
# shared-VM thread caps bind in-process (tests/test_shared_vm_thread_caps.py):
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

_SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPTS_DIR.parent

# --- Pinned encodings (one color = one meaning across EVERY figure) -------------

_BASE_ARMS = ("raw", "exact_dp", "prompt_dp", "mapped_ctx", "mapped_pfx", "id_bias")
_PALETTE = paper_palette(8)
ARM_COLORS = {arm: _PALETTE[i] for i, arm in enumerate(_BASE_ARMS)}
ARM_COLORS["mapped_tuned"] = _PALETTE[6]
# Map-quality series reuse the arm color whose assumption they quantify
# (identity == the prompt-token arm's implicit identity map; identity+bias ==
# the id_bias arm), so color meaning stays constant across figures.
MAPQ_COLORS = {
    "mapped_ctx": ARM_COLORS["mapped_ctx"],
    "mapped_pfx": ARM_COLORS["mapped_pfx"],
    "identity": ARM_COLORS["prompt_dp"],
    "identity_plus_bias": ARM_COLORS["id_bias"],
    "tuned": ARM_COLORS["mapped_tuned"],
}
ARM_LABELS = {
    "raw": "Raw projection",
    "exact_dp": "Exact ΔP (base generation)",
    "prompt_dp": "Prompt-token ΔP",
    "mapped_ctx": "Mapped ΔP (context)",
    "mapped_pfx": "Mapped ΔP (prefix)",
    "id_bias": "Identity + learned bias ΔP",
    "mapped_tuned": "Tuned-map ΔP (exploratory)",
}
MAPQ_LABELS = {
    "mapped_ctx": "Frozen map (context input)",
    "mapped_pfx": "Frozen map (prefix input)",
    "identity": "Identity (prompt-token assumption)",
    "identity_plus_bias": "Identity + learned bias",
    "tuned": "Tuned map (exploratory)",
}
TRAIT_LABELS = {"evil": "Evil", "sycophancy": "Sycophancy", "hallucination": "Hallucination"}
# Sycophancy foregrounded (plan §6: the headline gap lives there).
_TRAIT_PRIORITY = ("sycophancy", "evil", "hallucination")
VERSION_MARKERS = {"normal": "o", "misaligned_1": "s", "misaligned_2": "^"}
VERSION_LABELS = {
    "normal": "Normal version",
    "misaligned_1": "Misaligned I version",
    "misaligned_2": "Misaligned II version",
}
NEUTRAL = "#666666"
# CI-scheme grey ramp (fig_ci_schemes) — deliberately DISJOINT from ARM_COLORS
# so no scheme reads as an arm (round-2 C7); grey = statistical annotation.
_SCHEME_GREYS = ("#000000", "#4d4d4d", "#8c8c8c", "#c9c9c9")


def trait_label(trait: str) -> str:
    return TRAIT_LABELS.get(trait, trait.replace("_", " ").capitalize())


def ordered_traits(traits: list[str]) -> list[str]:
    front = [t for t in _TRAIT_PRIORITY if t in traits]
    return front + [t for t in traits if t not in front]


def arm_label(arm: str) -> str:
    return ARM_LABELS.get(arm, arm.replace("_", " "))


def arm_color(arm: str) -> str:
    return ARM_COLORS.get(arm, NEUTRAL)


def split_dataset_id(ds: str) -> tuple[str, str]:
    """``{family}_{version}`` -> (family, version); suffix-safe for the fixed
    version set (families themselves contain underscores)."""
    for ver in ("misaligned_1", "misaligned_2", "normal"):
        if ds.endswith(f"_{ver}"):
            return ds[: -(len(ver) + 1)], ver
    raise ValueError(f"unrecognized dataset id {ds!r}")


def ci_offsets(value: float, lo: float, hi: float, *, label: str = "") -> tuple[float, float]:
    """Non-negative errorbar offsets (gotchas.md xerr/yerr rule): a quantile CI
    can genuinely invert around the point estimate at tiny n (#1335/#547).

    A firing clamp is LOGGED (the gotcha calls it "a diagnostic worth logging")
    — a large inversion warrants checking the point/CI computation upstream.
    """
    if lo > value or hi < value:
        tag = f" ({label})" if label else ""
        print(
            f"[p5_figures] ci_offsets clamp fired{tag}: value={value:.6g} ci=[{lo:.6g}, {hi:.6g}]",
            flush=True,
        )
    return (max(0.0, value - lo), max(0.0, hi - value))


# --- Artifact loading (fail loud; no recomputation) ------------------------------


def load_json(path: Path, hint: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — {hint}")
    return json.loads(path.read_text())


def steer_records(corr: dict) -> dict[tuple[str, str], dict]:
    return {(r["trait"], r["arm"]): r for r in corr["records"] if r.get("layer_regime") == "steer"}


def sweep_records(corr: dict) -> dict[tuple[str, str], dict]:
    return {(r["trait"], r["arm"]): r for r in corr["records"] if r.get("layer_regime") == "sweep"}


def load_boot_flat(nulls_dir: Path, trait: str, arm: str) -> np.ndarray:
    """(B, L) bootstrap Pearson-r matrix persisted by P3 stage_aggregate."""
    path = nulls_dir / f"boot_flat_{trait}_{arm}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run issue2222_reduce.py --stage aggregate first "
            "(hero CI whiskers read the persisted flat bootstrap matrices)"
        )
    with np.load(path) as z:
        return np.asarray(z["r"], dtype=np.float64)


def load_perm(nulls_dir: Path, trait: str, arm: str) -> np.ndarray:
    """(n_perms, L) permutation |r| matrix persisted by P3 stage_aggregate."""
    path = nulls_dir / f"perm_{trait}_{arm}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run issue2222_reduce.py --stage aggregate")
    with np.load(path) as z:
        return np.asarray(z["abs_r"], dtype=np.float64)


# --- 1. Hero ---------------------------------------------------------------------


def fig_hero(corr: dict, nulls_dir: Path, fig_dir: Path) -> list[str]:
    """Per-trait grouped bars of steering-layer r with bootstrap-CI whiskers +
    published reference marks (plan §6 hero)."""
    arm_order = corr["arm_order"]
    recs = steer_records(corr)
    traits = ordered_traits(sorted({t for t, _ in recs}))
    fig, axes = plt.subplots(1, len(traits), figsize=(4.2 * len(traits), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    any_published = False
    for ax, trait in zip(axes, traits):
        xs = np.arange(len(arm_order))
        for xi, arm in enumerate(arm_order):
            rec = recs.get((trait, arm))
            if rec is None:
                continue
            r = rec["r"]
            steer = rec["layer"]
            boot = load_boot_flat(nulls_dir, trait, arm)[:, steer]
            lo, hi = np.nanquantile(boot, 0.025), np.nanquantile(boot, 0.975)
            err_lo, err_hi = ci_offsets(r, lo, hi, label=f"hero {trait}/{arm}")
            ax.bar(xi, r, color=arm_color(arm), width=0.72)
            ax.errorbar(
                xi, r, yerr=[[err_lo], [err_hi]], fmt="none", ecolor="black", capsize=3, lw=1.0
            )
            pub = rec.get("published_r")
            if pub is not None:
                ax.scatter(xi, pub, marker="_", s=340, color="black", zorder=5, linewidths=1.6)
                any_published = True
        ax.set_xticks(xs)
        ax.set_xticklabels([arm_label(a) for a in arm_order])
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)
        ax.set_title(trait_label(trait))
        ax.axhline(0.0, color="#999999", lw=0.8)
    axes[0].set_ylabel("Pearson r (predictor vs trait score)")
    if any_published:
        axes[-1].legend(
            handles=[
                Line2D(
                    [],
                    [],
                    color="black",
                    marker="_",
                    markersize=14,
                    linestyle="none",
                    label="Published r (Persona Vectors, Qwen)",
                )
            ],
            loc="lower right",
            fontsize=7,
        )
    fig.suptitle("Data-screening predictor comparison at the pre-registered steering layer")
    savefig_paper(fig, "hero_arm_r_comparison", dir=fig_dir)
    plt.close(fig)
    return ["hero_arm_r_comparison"]


# --- 2. Per-dataset scatter companion ---------------------------------------------


def fig_scatters(corr: dict, fig_dir: Path) -> list[str]:
    """Per-unit companion: dataset-level predictor value vs trait score, one
    panel per arm, points labeled by family + marked by version."""
    dv = corr.get("dataset_values")
    if not dv:
        raise KeyError(
            "predictor_correlations.json carries no dataset_values block — re-run "
            "issue2222_reduce.py --stage aggregate (unit-3 reduce adds it)"
        )
    fam_of = corr.get("family_of_dataset") or {}
    arm_order = corr["arm_order"]
    recs = steer_records(corr)
    stems = []
    for trait in ordered_traits(sorted(dv)):
        block = dv[trait]
        datasets = sorted(block["y_trait_score"])
        y = np.array([block["y_trait_score"][ds] for ds in datasets])
        n_cols = 3
        n_rows = int(np.ceil(len(arm_order) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.4 * n_rows), sharey=True)
        axes = np.atleast_1d(axes).ravel()
        for ax in axes[len(arm_order) :]:
            ax.set_visible(False)
        for ax, arm in zip(axes, arm_order):
            vals = np.array([block["arms"][arm].get(ds, np.nan) for ds in datasets])
            for ds, xv, yv in zip(datasets, vals, y):
                family, version = split_dataset_id(ds)
                family = fam_of.get(ds, family)
                ax.scatter(
                    xv,
                    yv,
                    marker=VERSION_MARKERS.get(version, "o"),
                    color=arm_color(arm),
                    s=28,
                    alpha=0.85,
                    zorder=3,
                )
                ax.annotate(
                    family.replace("_", " "),
                    (xv, yv),
                    textcoords="offset points",
                    xytext=(3, 3),
                    fontsize=5.5,
                    color="#444444",
                )
            rec = recs.get((trait, arm))
            # Correlation stat goes in the panel-title line (sanctioned statistical
            # label, paper-plots SKILL §3.8 carve-out) — never on the canvas, where
            # it collides with top-edge points.
            stat = ""
            if rec is not None:
                p = rec.get("perm_p_fixed_layer")
                stat = f"\nr={rec['r']:.3f}" + (f", perm p={p:.3g}" if p is not None else "")
            ax.set_title(arm_label(arm) + stat, fontsize=8)
            ax.set_xlabel("Dataset-level predictor value (steering layer)", fontsize=8)
        for ri in range(n_rows):
            axes[ri * n_cols].set_ylabel("Post-fine-tuning trait score (0-100)", fontsize=8)
        handles = [
            Line2D(
                [],
                [],
                marker=m,
                linestyle="none",
                color=NEUTRAL,
                label=VERSION_LABELS[v],
                markersize=6,
            )
            for v, m in VERSION_MARKERS.items()
        ]
        # Figure-level legend at the right, suptitle left-aligned so the two
        # never collide (blog register favors left-aligned titles anyway).
        fig.legend(handles=handles, loc="upper right", ncol=3, fontsize=6, frameon=False)
        fig.suptitle(
            f"{trait_label(trait)} — per-dataset predictor values vs trait score "
            f"(n={len(datasets)} datasets)",
            x=0.01,
            ha="left",
        )
        stem = f"scatter_per_dataset_{trait}"
        savefig_paper(fig, stem, dir=fig_dir)
        plt.close(fig)
        stems.append(stem)
    return stems


# --- 3. Layer sweeps with null bands ----------------------------------------------


def fig_layer_sweeps(corr: dict, nulls_dir: Path, fig_dir: Path) -> list[str]:
    arm_order = corr["arm_order"]
    sweeps = sweep_records(corr)
    steers = steer_records(corr)
    stems = []
    for trait in ordered_traits(sorted({t for t, _ in sweeps})):
        n_cols = 3
        n_rows = int(np.ceil(len(arm_order) / n_cols))
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(4.0 * n_cols, 3.0 * n_rows), sharey=True, sharex=True
        )
        axes = np.atleast_1d(axes).ravel()
        for ax in axes[len(arm_order) :]:
            ax.set_visible(False)
        for ax, arm in zip(axes, arm_order):
            rec = sweeps.get((trait, arm))
            if rec is None:
                continue
            r_layers = np.asarray(rec["r_per_layer"], dtype=np.float64)
            layers = np.arange(len(r_layers))
            perm = load_perm(nulls_dir, trait, arm)
            # POINTWISE band: per-layer 97.5th pct of the permutation |r| at each layer.
            band = np.nanquantile(perm, 0.975, axis=0)
            ax.fill_between(
                layers,
                -band,
                band,
                color="#bbbbbb",
                alpha=0.45,
                lw=0,
                label="Per-layer 97.5% permutation band",
            )
            # SELECTION-SYMMETRIC threshold: 97.5th pct of the per-draw max-over-layers
            # |r| (matches nulls/summary.json p975_max_selected — interp-critic r1 req 2).
            p975_max = float(np.nanquantile(np.nanmax(perm, axis=1), 0.975))
            ax.axhline(
                p975_max,
                color="#666666",
                lw=1.0,
                ls=":",
                label="Selection-symmetric (max-selected) p97.5",
            )
            ax.axhline(-p975_max, color="#666666", lw=1.0, ls=":")
            ax.plot(layers, r_layers, color=arm_color(arm), lw=1.6, label="Observed r")
            steer_rec = steers.get((trait, arm))
            if steer_rec is not None:
                ax.axvline(
                    steer_rec["layer"], color="#333333", lw=0.9, ls="--", label="Steering layer"
                )
            ax.axhline(0.0, color="#999999", lw=0.6)
            ax.set_title(arm_label(arm), fontsize=9)
        axes[0].legend(fontsize=6, loc="lower left")
        for ri in range(n_rows):
            axes[ri * n_cols].set_ylabel("Pearson r", fontsize=8)
        for ci in range(n_cols):
            axes[(n_rows - 1) * n_cols + ci].set_xlabel("Layer index", fontsize=8)
        fig.suptitle(
            f"{trait_label(trait)} — r per layer; per-layer permutation band "
            "+ max-selected p97.5 line"
        )
        stem = f"layer_sweep_{trait}"
        savefig_paper(fig, stem, dir=fig_dir)
        plt.close(fig)
        stems.append(stem)
    return stems


# --- 4. AUC summary ----------------------------------------------------------------


def fig_auc(aucj: dict, arm_order: list[str], fig_dir: Path) -> list[str]:
    recs = aucj.get("records", [])
    if not recs:
        raise ValueError("auc_misaligned2_vs_normal.json carries no records")
    traits = ordered_traits(sorted({r["trait"] for r in recs}))
    by_key = {(r["trait"], r["arm"]): r for r in recs}
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    n_arms = len(arm_order)
    width = 0.8 / max(n_arms, 1)
    for ai, arm in enumerate(arm_order):
        xs, ys = [], []
        for ti, trait in enumerate(traits):
            rec = by_key.get((trait, arm))
            if rec is None:
                continue
            xs.append(ti + (ai - (n_arms - 1) / 2) * width)
            ys.append(rec["auc"])
        ax.scatter(xs, ys, color=arm_color(arm), label=arm_label(arm), s=34, zorder=3)
    ax.axhline(0.5, color="#999999", lw=0.9, ls="--")
    ax.set_xticks(np.arange(len(traits)))
    ax.set_xticklabels([trait_label(t) for t in traits])
    ax.set_ylabel("AUC (misaligned II vs normal samples, steering layer)")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Sample-level separability per predictor arm (dashed line = chance)")
    savefig_paper(fig, "auc_by_arm", dir=fig_dir)
    plt.close(fig)
    return ["auc_by_arm"]


# --- 4b. ROC curves (plan §6 figure item 3) -----------------------------------------


def fig_roc(corr: dict, aucj: dict, nulls_dir: Path, fig_dir: Path) -> list[str]:
    """Sample-level ROC curves per arm (plan §6 "sample-level ROC curves per arm").

    Rank-cumulative rendering of the per-sample steering-layer scores PERSISTED
    by P3 ``stage_aggregate`` (``roc_scores_<trait>.npz``); the AUC values in
    the legend come from the persisted auc json — no statistic recomputed here.
    """
    recs = {(r["trait"], r["arm"]): r for r in aucj.get("records", [])}
    arm_order = corr["arm_order"]
    traits = ordered_traits(sorted({t for t, _ in recs}))
    if not traits:
        raise ValueError("auc_misaligned2_vs_normal.json carries no records for the ROC read")
    fig, axes = plt.subplots(1, len(traits), figsize=(4.0 * len(traits), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, trait in zip(axes, traits):
        path = nulls_dir / f"roc_scores_{trait}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing — run issue2222_reduce.py --stage aggregate (round-2 "
                "reduce persists the per-sample ROC scores)"
            )
        with np.load(path) as z:
            labels = np.asarray(z["labels"], dtype=bool)
            n_pos, n_neg = int(labels.sum()), int((~labels).sum())
            for arm in arm_order:
                key = f"score_{arm}"
                if key not in z:
                    continue
                scores = np.asarray(z[key], dtype=np.float64)
                order = np.argsort(-scores, kind="stable")
                l_sorted = labels[order]
                s_sorted = scores[order]
                # One point per DISTINCT score threshold (ties grouped), plus
                # the (0, 0) origin — the standard threshold sweep.
                cut = np.flatnonzero(np.diff(s_sorted) != 0)
                cut = np.concatenate([cut, [len(s_sorted) - 1]])
                tpr = np.concatenate([[0.0], np.cumsum(l_sorted)[cut] / max(1, n_pos)])
                fpr = np.concatenate([[0.0], np.cumsum(~l_sorted)[cut] / max(1, n_neg)])
                rec = recs.get((trait, arm))
                lab = arm_label(arm) + (f" (AUC {rec['auc']:.3f})" if rec else "")
                ax.plot(fpr, tpr, color=arm_color(arm), lw=1.4, label=lab)
        ax.plot([0, 1], [0, 1], color="#999999", lw=0.9, ls="--")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{trait_label(trait)} (n_pos={n_pos}, n_neg={n_neg})", fontsize=9)
        ax.set_xlabel("False-positive rate", fontsize=8)
        ax.legend(fontsize=5.5, loc="lower right")
    axes[0].set_ylabel("True-positive rate")
    fig.suptitle(
        "Sample-level ROC per predictor arm (misaligned II vs normal, steering layer; "
        "dashed line = chance)"
    )
    savefig_paper(fig, "roc_by_arm", dir=fig_dir)
    plt.close(fig)
    return ["roc_by_arm"]


# --- 5-6. Map quality ---------------------------------------------------------------


def fig_map_quality(mq: dict, tuned: dict | None, fig_dir: Path) -> list[str]:
    stems = []
    r2 = mq["r2_per_layer"]
    for stem, clip in (("map_quality_r2", True), ("map_quality_r2_full", False)):
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        for kind in ("mapped_ctx", "mapped_pfx", "identity", "identity_plus_bias"):
            vals = np.asarray(r2[kind], dtype=np.float64)
            ax.plot(
                np.arange(len(vals)),
                vals,
                color=MAPQ_COLORS[kind],
                lw=1.6,
                label=MAPQ_LABELS[kind],
            )
        if tuned is not None and tuned.get("heldout_r2_per_layer") is not None:
            vals = np.asarray(tuned["heldout_r2_per_layer"], dtype=np.float64)
            ax.plot(
                np.arange(len(vals)),
                vals,
                color=MAPQ_COLORS["tuned"],
                lw=1.4,
                ls="--",
                label=MAPQ_LABELS["tuned"],
            )
        ax.axhline(0.0, color="#999999", lw=0.7)
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Held-out R² of the answer-representation prediction")
        if clip:
            ax.set_ylim(-1.0, 1.0)
            ax.set_title(
                "Map quality per layer (y clipped to [-1, 1]; frozen context, frozen prefix,\n"
                "and plain Identity all sit below the clip — see the raw-scale twin)"
            )
        else:
            ax.set_title("Map quality per layer (raw scale)")
        ax.legend(fontsize=7)
        savefig_paper(fig, stem, dir=fig_dir)
        plt.close(fig)
        stems.append(stem)

    knn = mq.get("knn_retrieval", {})
    if knn:
        layers = sorted({key.split("/", 1)[0] for key in knn})
        metric_marker = {"euclidean": "o", "cosine": "D"}
        fig, axes = plt.subplots(1, len(layers), figsize=(4.4 * len(layers), 4.0), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, layer_key in zip(axes, layers):
            arms = sorted(k.split("/", 1)[1] for k in knn if k.startswith(layer_key + "/"))
            chances = []
            for xi, arm in enumerate(arms):
                reads = knn[f"{layer_key}/{arm}"]
                for metric, marker in metric_marker.items():
                    accs = [
                        rd["acc_at_k"]["1"] if "1" in rd["acc_at_k"] else rd["acc_at_k"][1]
                        for rd in reads
                        if rd.get("metric") == metric
                    ]
                    if not accs:
                        continue
                    jit = (np.arange(len(accs)) - (len(accs) - 1) / 2) * 0.03
                    ax.scatter(
                        xi + jit + (-0.15 if metric == "euclidean" else 0.15),
                        accs,
                        marker=marker,
                        color=arm_color(arm) if arm in ARM_COLORS else NEUTRAL,
                        s=22,
                        alpha=0.8,
                    )
                chances.extend(
                    rd["chance_at_k"]["1"] if "1" in rd["chance_at_k"] else rd["chance_at_k"][1]
                    for rd in reads
                )
            if chances:
                ax.axhline(float(np.mean(chances)), color="#999999", lw=0.9, ls="--")
            ax.set_xticks(np.arange(len(arms)))
            ax.set_xticklabels([arm_label(a) for a in arms])
            plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=7)
            ax.set_title(layer_key.replace("layer", "Layer "), fontsize=9)
        axes[0].set_ylabel("kNN retrieval acc@1 (per LOFO fold)")
        axes[-1].legend(
            handles=[
                Line2D([], [], marker=m, linestyle="none", color=NEUTRAL, label=lb, markersize=6)
                for lb, m in (("Euclidean", "o"), ("Cosine", "D"))
            ]
            + [Line2D([], [], color="#999999", ls="--", label="Chance (1 / pool size)")],
            fontsize=6,
            loc="upper right",
        )
        fig.suptitle("Retrieval read per stored layer and prediction kind")
        savefig_paper(fig, "map_quality_knn", dir=fig_dir)
        plt.close(fig)
        stems.append("map_quality_knn")
    return stems


# --- 7. Flat vs clustered CI comparison ----------------------------------------------


def fig_ci_schemes(corr: dict, fig_dir: Path) -> list[str]:
    ht = corr["hypothesis_tests"]
    rows: list[tuple[str, dict]] = []
    h1 = ht.get("H1_sycophancy_gap")
    if h1:
        rows.append(("H1 sycophancy: mapped (context) − prompt-token", h1))
    for trait, rec in sorted((ht.get("H2_equivalence") or {}).items()):
        rows.append((f"H2 {trait_label(trait).lower()}: mapped (context) − exact", rec))
    if not rows:
        raise ValueError("hypothesis_tests carries neither H1 nor H2 records")
    schemes = [
        ("ci95_frozen_flat", "Frozen layer, flat bootstrap", "o", -0.27),
        ("ci95_frozen_clustered", "Frozen layer, family-clustered", "s", -0.09),
        ("ci95_selection_inherited_flat", "Selection-inherited, flat", "^", 0.09),
        ("ci95_selection_inherited_clustered", "Selection-inherited, family-clustered", "D", 0.27),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 1.2 + 0.9 * len(rows)))
    # Round-2 C7: scheme colors must be DISJOINT from the pinned ARM palette
    # (_PALETTE[0..6] all encode arms — one color = one meaning). A sequential
    # grey ramp reads as the statistical-annotation register (like the hero's
    # black CI whiskers); the four markers already disambiguate schemes.
    scheme_colors = {k: _SCHEME_GREYS[i] for i, (k, _, _, _) in enumerate(schemes)}
    for ri, (label, rec) in enumerate(rows):
        delta = rec["delta_r"]
        for key, _slabel, marker, off in schemes:
            ci = rec.get(key)
            if ci is None:
                continue
            lo, hi = float(ci[0]), float(ci[1])
            err_lo, err_hi = ci_offsets(delta, lo, hi, label=f"ci_schemes {label}/{key}")
            ax.errorbar(
                delta,
                ri + off,
                xerr=[[err_lo], [err_hi]],
                fmt=marker,
                color=scheme_colors[key],
                capsize=3,
                markersize=5,
            )
    ax.axvline(0.0, color="#999999", lw=0.9, ls="--")
    # Registered H2 equivalence margin (r_a >= r_b - 0.10 <=> delta_r >= -0.10):
    # drawn only across the H2 rows (interp-critic r1 req 2 / plot-prose defect).
    h2_rows = [ri for ri, (label, _) in enumerate(rows) if label.startswith("H2")]
    if h2_rows:
        ax.plot(
            [-0.10, -0.10],
            [min(h2_rows) - 0.4, max(h2_rows) + 0.4],
            color="#333333",
            lw=1.0,
            ls=":",
            zorder=1,
        )
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([label for label, _ in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Δr at the steering layer (95% CI)")
    ax.legend(
        handles=[
            Line2D(
                [], [], marker=m, linestyle="none", color=scheme_colors[k], label=sl, markersize=6
            )
            for k, sl, m, _ in schemes
        ]
        + [
            Line2D(
                [],
                [],
                color="#333333",
                ls=":",
                label="H2 equivalence margin (Δr = −0.10, H2 rows only)",
            )
        ],
        fontsize=6,
        loc="best",
    )
    ax.set_title("Hypothesis-test CIs under the four registered schemes")
    savefig_paper(fig, "ci_flat_vs_clustered", dir=fig_dir)
    plt.close(fig)
    return ["ci_flat_vs_clustered"]


# --- 8. Form A (exploratory) ----------------------------------------------------------


def fig_form_a(fa: dict, fig_dir: Path) -> list[str]:
    stems = []
    r2 = fa.get("heldout_r2_per_layer") or {}
    traits = ordered_traits(sorted(r2))
    if traits:
        fig, axes = plt.subplots(1, len(traits), figsize=(4.0 * len(traits), 3.4), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, trait in zip(axes, traits):
            vals = np.asarray([np.nan if v is None else v for v in r2[trait]], dtype=np.float64)
            layers = np.arange(len(vals))
            ok = ~np.isnan(vals)
            ax.plot(layers[ok], vals[ok], color=NEUTRAL, lw=1.6, marker="o", markersize=3)
            ax.axhline(0.0, color="#999999", lw=0.7)
            ax.set_title(trait_label(trait), fontsize=9)
            ax.set_xlabel("Layer index", fontsize=8)
        axes[0].set_ylabel("Held-out R² (probe: activation → judge score)")
        fig.suptitle("Form-A probe fit quality per layer (exploratory)")
        savefig_paper(fig, "form_a_probe_r2", dir=fig_dir)
        plt.close(fig)
        stems.append("form_a_probe_r2")

    recs = [r for r in fa.get("records", []) if r.get("layer_regime") == "steer"]
    val = fa.get("graded_vs_rate_validation") or {}
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    ax = axes[0]
    if recs:
        traits_r = ordered_traits(sorted({r["trait"] for r in recs}))
        arms = sorted({r["arm"].split("/", 1)[1] for r in recs})
        n_arms = len(arms)
        width = 0.8 / max(n_arms, 1)
        for ai, arm in enumerate(arms):
            xs, ys = [], []
            for ti, trait in enumerate(traits_r):
                rec = next(
                    (r for r in recs if r["trait"] == trait and r["arm"].endswith("/" + arm)),
                    None,
                )
                if rec is None:
                    continue
                xs.append(ti + (ai - (n_arms - 1) / 2) * width)
                ys.append(rec["r"])
            ax.scatter(xs, ys, color=arm_color(arm), label=arm_label(arm), s=30, zorder=3)
        ax.set_xticks(np.arange(len(traits_r)))
        ax.set_xticklabels([trait_label(t) for t in traits_r], fontsize=8)
        ax.axhline(0.0, color="#999999", lw=0.7)
        ax.set_ylabel("Pearson r (probe difference grid vs trait score)")
        ax.legend(fontsize=6)
        ax.set_title("Form-A probe screening grid at the steering layer", fontsize=9)
    ax = axes[1]
    if val:
        traits_v = ordered_traits(sorted(val))
        xs = np.arange(len(traits_v))
        ax.scatter(
            xs - 0.1,
            [val[t]["spearman_all"] for t in traits_v],
            color=NEUTRAL,
            label="All items",
            s=34,
        )
        ax.scatter(
            xs + 0.1,
            [val[t]["spearman_heldout_20pct"] for t in traits_v],
            color="black",
            marker="^",
            label="Held-out 20%",
            s=34,
        )
        ax.set_xticks(xs)
        ax.set_xticklabels([trait_label(t) for t in traits_v], fontsize=8)
        ax.axhline(0.0, color="#999999", lw=0.7)
        ax.set_ylabel("Spearman ρ (graded score vs rate > 50)")
        ax.legend(fontsize=7)
        ax.set_title("Graded-vs-rate validation", fontsize=9)
    fig.suptitle("Form-A probe summary (exploratory)")
    savefig_paper(fig, "form_a_probe_summary", dir=fig_dir)
    plt.close(fig)
    stems.append("form_a_probe_summary")
    return stems


# --- 9. Form B (exploratory) -----------------------------------------------------------


def fig_form_b(fb: dict, fig_dir: Path) -> list[str]:
    recs = fb.get("records", [])
    if not recs:
        raise ValueError("form_b_regression.json carries no records")
    traits = ordered_traits(sorted({r["trait"] for r in recs}))
    by_trait = {r["trait"]: r for r in recs}
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for ti, trait in enumerate(traits):
        rec = by_trait[trait]
        folds = np.asarray(rec["cosine_to_rb_by_fold"], dtype=np.float64)
        jit = (np.arange(len(folds)) - (len(folds) - 1) / 2) * 0.03
        ax.scatter(ti + jit, folds, color=NEUTRAL, s=20, alpha=0.7, label=None)
        ax.scatter(
            [ti],
            [rec["cosine_to_rb_mean"]],
            color="black",
            marker="D",
            s=48,
            zorder=4,
        )
    ax.axhline(0.0, color="#999999", lw=0.8)
    ax.set_xticks(np.arange(len(traits)))
    ax.set_xticklabels([trait_label(t) for t in traits])
    ax.set_ylabel("Cosine(regression coefficient, persona direction)")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(
        "Form-B dataset-level regression coefficient vs persona direction\n"
        "(per LOFO fold, diamond = mean; exploratory estimator-degenerate control)"
    )
    savefig_paper(fig, "form_b_cosine", dir=fig_dir)
    plt.close(fig)
    return ["form_b_cosine"]


# --- 10. Tuned vs frozen map arm ---------------------------------------------------------


def fig_tuned_vs_frozen(corr: dict, tuned: dict, fig_dir: Path) -> list[str]:
    sweeps = sweep_records(corr)
    steers = steer_records(corr)
    tuned_by_trait = {r["trait"]: r for r in tuned.get("records", [])}
    traits = ordered_traits(sorted(tuned_by_trait))
    if not traits:
        raise ValueError("tuned_map.json carries no records")
    fig, axes = plt.subplots(1, len(traits), figsize=(4.2 * len(traits), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, trait in zip(axes, traits):
        frozen = sweeps.get((trait, "mapped_ctx"))
        if frozen is not None:
            vals = np.asarray(frozen["r_per_layer"], dtype=np.float64)
            ax.plot(
                np.arange(len(vals)),
                vals,
                color=ARM_COLORS["mapped_ctx"],
                lw=1.6,
                label=ARM_LABELS["mapped_ctx"],
            )
        vals_t = np.asarray(tuned_by_trait[trait]["r_per_layer"], dtype=np.float64)
        ax.plot(
            np.arange(len(vals_t)),
            vals_t,
            color=ARM_COLORS["mapped_tuned"],
            lw=1.4,
            ls="--",
            label=ARM_LABELS["mapped_tuned"],
        )
        steer_rec = steers.get((trait, "mapped_ctx"))
        if steer_rec is not None:
            ax.axvline(steer_rec["layer"], color="#333333", lw=0.9, ls=":")
        ax.axhline(0.0, color="#999999", lw=0.6)
        ax.set_title(trait_label(trait), fontsize=9)
        ax.set_xlabel("Layer index", fontsize=8)
    axes[0].set_ylabel("Pearson r (predictor vs trait score)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Frozen vs tuned mapped arm, r per layer (tuned arm is exploratory)")
    savefig_paper(fig, "tuned_vs_frozen_map", dir=fig_dir)
    plt.close(fig)
    return ["tuned_vs_frozen_map"]


def fig_basegen_map(basegen: dict, fig_dir: Path) -> list[str]:
    """Follow-up round: base-generation-target map vs the exact-ΔP companion, r per layer.

    One panel per trait; solid = the ridge map fit directly to the base-generation
    response-average (context-end input), thin dashed grey = the same fit from the
    constant prefix-end input (degenerate), dashed = exact ΔP recomputed on the SAME
    k=250 row subset the maps consumed. Dotted vertical = the paper's steering layer;
    the dot marks the layer the LOFO folds selected for the context-input arm.
    """
    ctx_by_trait = {r["trait"]: r for r in basegen["arms"]["mapped_base_ctx"]["records"]}
    pfx_by_trait = {r["trait"]: r for r in basegen["arms"]["mapped_base_pfx"]["records"]}
    exact_by_trait = {r["trait"]: r for r in basegen["exact_dp_k250_companion"]["records"]}
    traits = ordered_traits(sorted(ctx_by_trait))
    if not traits:
        raise ValueError("basegen_map.json carries no mapped_base_ctx records")
    base_ctx_color = _PALETTE[7]
    fig, axes = plt.subplots(1, len(traits), figsize=(4.2 * len(traits), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, trait in zip(axes, traits):
        rec = ctx_by_trait[trait]
        vals = np.asarray(rec["r_per_layer"], dtype=np.float64)
        ax.plot(
            np.arange(len(vals)),
            vals,
            color=base_ctx_color,
            lw=1.6,
            label="Base-target map ΔP (context input)",
        )
        pfx = pfx_by_trait.get(trait)
        if pfx is not None:
            vals_p = np.asarray(pfx["r_per_layer"], dtype=np.float64)
            ax.plot(
                np.arange(len(vals_p)),
                vals_p,
                color=NEUTRAL,
                lw=1.0,
                ls="--",
                label="Base-target map ΔP (prefix input, degenerate)",
            )
        ex = exact_by_trait.get(trait)
        if ex is not None:
            vals_e = np.asarray(ex["r_per_layer"], dtype=np.float64)
            ax.plot(
                np.arange(len(vals_e)),
                vals_e,
                color=ARM_COLORS["exact_dp"],
                lw=1.4,
                ls="--",
                label="Exact ΔP, same row subset",
            )
        sel_by_fold = rec["sweep"]["selected_layer_by_fold"]
        sel_layers = [int(v) for v in sel_by_fold.values()]
        sel_mode = max(set(sel_layers), key=sel_layers.count)
        ax.scatter(
            [sel_mode],
            [vals[sel_mode]],
            color=base_ctx_color,
            s=28,
            zorder=5,
            edgecolors="white",
            linewidths=0.8,
            label="Layer the LOFO folds select (context input)",
        )
        ax.axvline(rec["steer_layer"], color="#333333", lw=0.9, ls=":")
        ax.axhline(0.0, color="#999999", lw=0.6)
        ax.set_title(trait_label(trait), fontsize=9)
        ax.set_xlabel("Layer index", fontsize=8)
    axes[0].set_ylabel("Pearson r (predictor vs trait score)")
    axes[0].legend(fontsize=6.5)
    fig.suptitle("Base-generation-target map vs exact ΔP, r per layer (n = 24 datasets)")
    savefig_paper(fig, "basegen_map_layer_sweep", dir=fig_dir)
    plt.close(fig)
    return ["basegen_map_layer_sweep"]


# --- CLI ------------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-root", default=str(REPO_ROOT / "eval_results" / "issue_2222"))
    ap.add_argument(
        "--data-root",
        default=str(REPO_ROOT / "data" / "issue_2222"),
        help="P3 data root (persisted nulls/boot matrices live under <data-root>/nulls)",
    )
    ap.add_argument("--fig-dir", default=str(REPO_ROOT / "figures" / "issue_2222"))
    ap.add_argument(
        "--skip-missing-exploratory",
        action="store_true",
        help="log-and-skip exploratory panels whose input JSON is absent (smoke / "
        "pre-P4 renders only; the production P5 run requires every input)",
    )
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="args-attribute completeness check (all imports are top-level), then exit 0",
    )
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("[import-check] issue2222_figures OK")
        return 0
    set_paper_style("blog")
    out_root = Path(args.out_root)
    nulls_dir = Path(args.data_root) / "nulls"
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    corr = load_json(
        out_root / "predictor_correlations.json", "run issue2222_reduce.py --stage aggregate"
    )
    mq = load_json(out_root / "map_quality.json", "run issue2222_reduce.py --stage aggregate")
    aucj = load_json(
        out_root / "auc_misaligned2_vs_normal.json", "run issue2222_reduce.py --stage aggregate"
    )
    # nulls summary is read for existence parity with the plan §6 outputs
    # (bands themselves come from the persisted per-arm matrices).
    load_json(out_root / "nulls" / "summary.json", "run issue2222_reduce.py --stage aggregate")

    def _optional(name: str, hint: str) -> dict | None:
        path = out_root / name
        if not path.exists():
            if args.skip_missing_exploratory:
                print(f"[p5_figures] SKIP {name} (absent; --skip-missing-exploratory)")
                return None
            raise FileNotFoundError(f"{path} missing — {hint}")
        return json.loads(path.read_text())

    tuned = _optional("tuned_map.json", "run issue2222_reduce.py --stage tuned_map")
    fb = _optional("form_b_regression.json", "run issue2222_reduce.py --stage form_b")
    fa = _optional("form_a_probe.json", "run issue2222_judge.py --stage probe (P4)")
    basegen = _optional(
        "followup_free_analysis/basegen_map.json",
        "run issue2222_followup_basegen_map.py (9a-ter follow-up round)",
    )

    written: list[str] = []
    written += fig_hero(corr, nulls_dir, fig_dir)
    written += fig_scatters(corr, fig_dir)
    written += fig_layer_sweeps(corr, nulls_dir, fig_dir)
    written += fig_auc(aucj, corr["arm_order"], fig_dir)
    written += fig_roc(corr, aucj, nulls_dir, fig_dir)
    written += fig_map_quality(mq, tuned, fig_dir)
    written += fig_ci_schemes(corr, fig_dir)
    if fa is not None:
        written += fig_form_a(fa, fig_dir)
    if fb is not None:
        written += fig_form_b(fb, fig_dir)
    if tuned is not None:
        written += fig_tuned_vs_frozen(corr, tuned, fig_dir)
    if basegen is not None:
        written += fig_basegen_map(basegen, fig_dir)
    for stem in written:
        print(f"[p5_figures] wrote {fig_dir / stem}.png")
    print(f"[p5_figures] done n_figures={len(written)} fig_dir={fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
