"""Issue #952 Phase 2 (VM, CPU): hero figures + exploratory dump (plan §6 Figures).

Heroes:
  hero1_position_r2      — per-position test R² curves (F16 / deciles / L16 on
                           one axis, four arm curves, bootstrap bands, turn-end
                           template slots shaded)
  hero2_matched_prefix   — MATCHED H2 contrast bars (G_matched(0) vs
                           G_matched(t), paired n annotated per the plan) +
                           descriptive remainder-R² closure curves + a per-t
                           survivor-attrition strip
  hero3_divergence       — divergent-vs-control paired drop bars per category x
                           arm + the per-pair raw scatter (low-level data plot)

Exploratory dump (over-produced, plan §6): slot x layer validation heatmap;
surprisal curves; per-context R² ECDFs; turn-end split-out bars; H3
length-stratified sweep; pooled-prefix secondary curves; per-slot
layer-selection sensitivity.

All figures ride the paper-plots conventions (``set_paper_style`` +
``savefig_paper`` — commit-pinned metadata + per-point sidecars). Missing
inputs hard-fail in production; ``--smoke`` tolerates absent OPTIONAL families
(logged loudly), never absent hero-1 inputs.

Usage:
  uv run python scripts/issue952_figures.py \
    --eval-dir eval_results/issue_952 --stats eval_results/issue_952/stats_summary.json \
    --out-dir figures/issue_952 [--tensors-dir data/issue_952/analysis_tensors] [--smoke]
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)
from explore_persona_space.experiments.issue_952.run_952 import (  # noqa: E402
    ARMS,
    D10_SLOTS,
    F16_SLOTS,
    MATCHED_T2,
    PREFIX_TS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("issue952.figures")

ARM_LABEL = {
    "own": "Own answer (regenerated)",
    "ext_plain": "External plain (Claude)",
    "ext_style": "External distinct-style (Claude)",
    "mismatch": "Mismatched (shuffled pairing)",
}
CATEGORY_LABEL = {
    "china_politics": "Geo-political",
    "model_identity": "Model identity",
    "refusal_boundary": "Compliance boundary",
    "style_format": "Style / formatting",
}
# One position axis: F16 (1..16), deciles (5%..95%), L16 aligned from the end
# (-16..-1; -1 = trailing newline, -2 = end-of-turn token — template slots).
L16_ORDER = [f"l16_m{k}" for k in range(16, 0, -1)]
POSITION_AXIS: list[tuple[str, str]] = (
    [(s, f"+{t}") for t, s in enumerate(F16_SLOTS, start=1)]
    + [(s, f"{int(s.split('_p')[1])}%") for s in D10_SLOTS]
    + [(s, f"-{int(s.split('_m')[1])}") for s in L16_ORDER]
)


def _load_json(path: pathlib.Path, *, optional: bool, smoke: bool) -> dict | None:
    if path.exists():
        return json.loads(path.read_text())
    if optional or smoke:
        logger.warning(
            "[figures] missing input tolerated (%s): %s", "smoke" if smoke else "optional", path
        )
        return None
    raise FileNotFoundError(path)


def hero1_position_r2(stats: dict, out_dir: pathlib.Path) -> None:
    """Hero 1: per-position test pooled R² per arm with bootstrap bands."""
    cells = stats["cells"]
    fig, ax = plt.subplots(figsize=(9.2, 4.4), layout="constrained")
    colors = dict(zip(ARMS, paper_palette(4), strict=True))
    xs = np.arange(len(POSITION_AXIS))
    plotted = False
    for arm in ARMS:
        ys, los, his = [], [], []
        for slot, _lab in POSITION_AXIS:
            c = cells.get(f"A|{slot}|{arm}", {})
            ys.append(c.get("observed") if c.get("observed") is not None else np.nan)
            ci = c.get("ci95") or [np.nan, np.nan]
            los.append(ci[0])
            his.append(ci[1])
        ys = np.asarray(ys, dtype=float)
        if not np.isfinite(ys).any():
            continue
        plotted = True
        ax.plot(xs, ys, marker="o", ms=2.5, lw=1.4, color=colors[arm], label=ARM_LABEL[arm])
        ax.fill_between(
            xs, np.asarray(los, float), np.asarray(his, float), color=colors[arm], alpha=0.15, lw=0
        )
    if not plotted:
        raise RuntimeError("hero1: no position cells present in stats_summary")
    # Section boundaries + template-slot shading.
    n_f16, n_d10 = len(F16_SLOTS), len(D10_SLOTS)
    for x in (n_f16 - 0.5, n_f16 + n_d10 - 0.5):
        ax.axvline(x, color="#B0B0B0", lw=0.8, ls=":")
    ax.axvspan(len(POSITION_AXIS) - 2.5, len(POSITION_AXIS) - 0.5, color="#D9D9D9", alpha=0.5)
    tick_every = [0, 7, 14, 17, 21, 24, 27, 33, 38, 41]
    ax.set_xticks([x for x in tick_every if x < len(POSITION_AXIS)])
    ax.set_xticklabels(
        [POSITION_AXIS[x][1] for x in tick_every if x < len(POSITION_AXIS)], fontsize=8
    )
    ax.set_xlabel(
        "Answer position (first 16 tokens | relative deciles | last 16 tokens; "
        "shaded = turn-end template tokens)"
    )
    ax.set_ylabel("Held-out test pooled R²")
    ax.legend(frameon=False, fontsize=8)
    set_title_subtitle(
        ax,
        "Where the context predicts the coming answer",
        f"Ridge map from the last context token, frozen layer/λ; n(test) per slot varies — "
        f"universe span ≥ 32 in all arms (l* = {stats['h2_matched'].get('l_star')})",
    )
    savefig_paper(fig, "hero1_position_r2", dir=out_dir)
    plt.close(fig)


def hero2_matched_prefix(stats: dict, closure: dict, out_dir: pathlib.Path) -> None:
    """Hero 2: matched H2 contrast bars + closure curves + attrition strip."""
    matched = stats["h2_matched"]["contrasts"]
    l_star = stats["h2_matched"].get("l_star")
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.5, 6.8),
        layout="constrained",
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    ax_bars, ax_closure, ax_attr, ax_attr2 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # (a) MATCHED contrast bars (decision panel): G_matched(0) vs G_matched(t2).
    ext_arms = ("ext_plain", "ext_style")
    bar_w = 0.35
    palette = paper_palette(4)
    xs, labels = [], []
    x = 0.0
    plotted_bars = False
    paired_n = stats.get("matched_paired_n", {})
    for t2 in MATCHED_T2:
        rec = matched.get(f"t{t2}_L{l_star}")
        if not rec:
            continue
        for ei, ext in enumerate(ext_arms):
            r = rec.get(ext)
            if not r:
                continue
            g0, gt = r["G_matched_0"], r["G_matched_t"]
            ax_bars.bar(x - bar_w / 2, g0, bar_w, color=palette[2 * ei], alpha=0.9)
            ax_bars.bar(x + bar_w / 2, gt, bar_w, color=palette[2 * ei + 1], alpha=0.9)
            n_rec = paired_n.get(f"t{t2}") or {}
            short = "plain" if ext == "ext_plain" else "style"
            labels.append(f"0 vs {t2} ({short}), n={n_rec.get('test', '?')}")
            xs.append(x)
            x += 1.0
            plotted_bars = True
    if plotted_bars:
        ax_bars.set_xticks(xs)
        ax_bars.set_xticklabels(labels, fontsize=6, rotation=35, ha="right")
        ax_bars.axhline(0, color="#444444", lw=0.8)
        ax_bars.set_ylabel(
            "Own-answer R² minus external-answer R²\n(matched survivors, identical target)"
        )
        set_title_subtitle(
            ax_bars,
            "Does absorbing the answer prefix close the own-advantage?",
            f"Left bar: R² gap with no prefix absorbed (t=0); right bar: gap after t prefix\n"
            f"tokens absorbed (layer {l_star}, common survivors, identical target)",
        )
    else:
        ax_bars.text(0.5, 0.5, "no matched cells (smoke)", ha="center", va="center")

    # (b) Descriptive closure curves: remainder-mean test R² vs prefix t per arm.
    colors = dict(zip(ARMS, paper_palette(4), strict=True))
    cells = closure.get("cells", {})
    for arm in ARMS:
        ts, ys = [], []
        for t in PREFIX_TS:
            rec = cells.get(f"L{l_star}|{arm}|t{t}|rem_mean")
            if rec:
                ts.append(t)
                ys.append(rec["test_pooled_r2"])
        if ts:
            ax_closure.plot(
                ts, ys, marker="o", ms=3, lw=1.4, color=colors[arm], label=ARM_LABEL[arm]
            )
    ax_closure.set_xscale("log", base=2)
    ax_closure.set_xticks(list(PREFIX_TS))
    ax_closure.set_xticklabels([str(t) for t in PREFIX_TS], fontsize=8)
    ax_closure.set_xlabel("Prefix length t (tokens absorbed)")
    ax_closure.set_ylabel("Remainder-mean test R² (descriptive)")
    ax_closure.legend(frameon=False, fontsize=6, loc="upper right")
    set_title_subtitle(
        ax_closure,
        "Prediction of the remainder as the prefix is absorbed",
        "Descriptive view; survivor\npopulations vary per t",
    )

    # (c)+(d) Attrition strip: n surviving + fraction excluded per (arm, t).
    attr = stats.get("attrition", {})
    for arm in ARMS:
        ts, ns, fr = [], [], []
        for t in PREFIX_TS:
            rec = attr.get(f"{arm}|t{t}")
            if rec:
                ts.append(t)
                ns.append(rec["n_test"])
                fr.append(rec["frac_excluded"])
        if ts:
            ax_attr.plot(ts, ns, marker="o", ms=2.5, lw=1.2, color=colors[arm])
            ax_attr2.plot(ts, fr, marker="o", ms=2.5, lw=1.2, color=colors[arm])
    for ax, ylab in ((ax_attr, "n surviving (test)"), (ax_attr2, "fraction excluded")):
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(PREFIX_TS))
        ax.set_xticklabels([str(t) for t in PREFIX_TS], fontsize=7)
        ax.set_xlabel("Prefix length t")
        ax.set_ylabel(ylab, fontsize=8)
    savefig_paper(fig, "hero2_matched_prefix", dir=out_dir)
    plt.close(fig)


def hero3_divergence(stats: dict, out_dir: pathlib.Path, smoke: bool) -> None:
    """Hero 3: paired drop bars per category x arm + per-pair raw scatter."""
    h3 = stats.get("h3", {})
    rows = h3.get("pair_rows")
    if not rows:
        if smoke:
            logger.warning("[figures] hero3 skipped — no H3 pair rows (smoke)")
            return
        raise RuntimeError("hero3: no H3 pair rows in stats_summary")
    fig, (ax_bar, ax_sc) = plt.subplots(1, 2, figsize=(10.5, 4.2), layout="constrained")
    cats = sorted({r["category"] for r in rows})
    colors = {"own": paper_palette(2)[0], "ext_plain": paper_palette(2)[1]}
    bar_w = 0.35
    for ci, cat in enumerate(cats):
        sub = [r for r in rows if r["category"] == cat]
        for ai, arm in enumerate(("own", "ext_plain")):
            vals = np.asarray([r[f"drop_{arm}"] for r in sub], dtype=float)
            mean = float(np.nanmean(vals))
            sem = float(np.nanstd(vals) / max(np.sqrt(len(vals)), 1.0))
            ax_bar.bar(
                ci + (ai - 0.5) * bar_w,
                mean,
                bar_w,
                yerr=max(sem, 0.0),
                color=colors[arm],
                alpha=0.9,
                label=ARM_LABEL[arm] if ci == 0 else None,
                capsize=2,
            )
        ax_bar.text(ci, ax_bar.get_ylim()[0], f"n={len(sub)}", ha="center", va="bottom", fontsize=7)
    ax_bar.set_xticks(range(len(cats)))
    ax_bar.set_xticklabels(
        [CATEGORY_LABEL.get(c, c) for c in cats], fontsize=6.5, rotation=10, ha="right"
    )
    ax_bar.axhline(0, color="#444444", lw=0.8)
    ax_bar.set_ylabel("Per-context R² drop (control - divergent)")
    ax_bar.legend(frameon=False, fontsize=8)
    set_title_subtitle(
        ax_bar,
        "Where the two models disagree, whose answers stop being predictable?",
        "Paired entity-swapped control minus divergent query, per category (mean ± s.e.m.)",
    )
    own = np.asarray([r["drop_own"] for r in rows], dtype=float)
    ext = np.asarray([r["drop_ext_plain"] for r in rows], dtype=float)
    cat_colors = dict(zip(cats, paper_palette(max(2, len(cats))), strict=False))
    for cat in cats:
        m = np.asarray([r["category"] == cat for r in rows])
        ax_sc.scatter(
            own[m],
            ext[m],
            s=14,
            alpha=0.75,
            color=cat_colors[cat],
            label=CATEGORY_LABEL.get(cat, cat),
        )
    lim = np.nanmax(np.abs(np.concatenate([own, ext]))) * 1.1 + 1e-6
    ax_sc.plot([-lim, lim], [-lim, lim], color="#888888", lw=0.8, ls="--")
    ax_sc.set_xlim(-lim, lim)
    ax_sc.set_ylim(-lim, lim)
    ax_sc.set_xlabel("Drop, own answer (regenerated)")
    ax_sc.set_ylabel("Drop, external plain (Claude)")
    ax_sc.legend(frameon=False, fontsize=7)
    set_title_subtitle(ax_sc, "Per-pair raw drops", "One point per kept entity-swapped pair")
    savefig_paper(fig, "hero3_divergence", dir=out_dir)
    plt.close(fig)


def exploratory_dump(  # noqa: C901 — one guarded block per exploratory panel
    stats: dict,
    closure: dict | None,
    valmat: dict | None,
    tensors_dir: pathlib.Path | None,
    out_dir: pathlib.Path,
) -> None:
    """The plan §6 exploratory panels (each optional; produced when inputs exist)."""
    # Slot x layer validation heatmap (position slots, max over λ, mean over arms).
    if valmat is not None:
        layers = valmat["layers"]
        groups = valmat["groups_A"]
        arr = np.asarray(valmat["val_pooled_A"], dtype=float)  # (L_layers, n_lam, G)
        best = np.nanmax(arr, axis=1)  # (L_layers, G)
        slot_names = [s for s, _l in (g.split("|") for g in groups)]
        pos_slots = [s for s in dict.fromkeys(slot_names) if not s.startswith("rem_")]
        mat = np.full((len(pos_slots), len(layers)), np.nan)
        for si, slot in enumerate(pos_slots):
            cols = [gi for gi, g in enumerate(groups) if g.split("|")[0] == slot]
            mat[si] = np.nanmean(best[:, cols], axis=1)
        fig, ax = plt.subplots(figsize=(6.5, max(4.0, 0.16 * len(pos_slots))), layout="constrained")
        im = ax.imshow(mat, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([str(la) for la in layers], fontsize=8)
        ax.set_yticks(range(len(pos_slots)))
        ax.set_yticklabels(pos_slots, fontsize=5)
        ax.set_xlabel("Layer")
        fig.colorbar(im, ax=ax, label="Validation pooled R² (best λ, arm-averaged)")
        set_title_subtitle(ax, "Validation R² by position slot and layer", None)
        savefig_paper(fig, "exp_validation_heatmap", dir=out_dir)
        plt.close(fig)

        # Per-slot layer-selection sensitivity: argmax layer per position slot.
        arg = np.asarray(layers)[np.nanargmax(np.where(np.isfinite(mat), mat, -np.inf), axis=1)]
        fig, ax = plt.subplots(figsize=(6.0, 3.2), layout="constrained")
        vals, counts = np.unique(arg, return_counts=True)
        ax.bar([str(v) for v in vals], counts, color=paper_palette(1)[0])
        ax.set_xlabel("Per-slot best validation layer")
        ax.set_ylabel("Number of position slots")
        set_title_subtitle(
            ax, "Layer-selection sensitivity", "Diagnostic — headline uses one family layer"
        )
        savefig_paper(fig, "exp_layer_selection_sensitivity", dir=out_dir)
        plt.close(fig)

    # Surprisal companion — ONE two-panel figure (curves + per-slot scatter),
    # merged from the former exp_surprisal_curves / exp_r2_vs_surprisal_scatter
    # pair (clean-result round-2: one figure unit per result).
    if tensors_dir is not None:
        colors = dict(zip(ARMS, paper_palette(4), strict=True))
        fig, (ax_curves, ax_scatter) = plt.subplots(1, 2, figsize=(11.5, 3.8), layout="constrained")
        any_arm = False
        for arm in ARMS:
            p = tensors_dir / f"surprisal_{arm}.npz"
            if not p.exists():
                continue
            d = np.load(p)
            flat, offs = d["flat"], d["offsets"]
            max_t = 128
            sums = np.zeros(max_t)
            cnts = np.zeros(max_t)
            for i in range(len(offs) - 1):
                seg = flat[offs[i] : offs[i + 1]][:max_t]
                sums[: len(seg)] += seg
                cnts[: len(seg)] += 1
            ok = cnts > 0
            ax_curves.plot(
                np.arange(1, max_t + 1)[ok],
                (sums / np.maximum(cnts, 1))[ok],
                lw=1.3,
                color=colors[arm],
                label=ARM_LABEL[arm],
            )
            any_arm = True
        ax_curves.set_xlabel("Answer position t")
        ax_curves.set_ylabel("Teacher-forced -log P(token)")
        ax_curves.legend(frameon=False, fontsize=7)
        set_title_subtitle(
            ax_curves, "Token-level surprise per arm", "In-context adaptation companion"
        )

        # Per-position R²-vs-surprisal scatter (plan §6 exploratory; round-2 Minor 3c).
        cells_scatter = stats.get("cells", {})
        scatter_ts = [*range(1, 17), 32, 64, 128]
        any_pt = False
        for arm in ARMS:
            p = tensors_dir / f"surprisal_{arm}.npz"
            if not p.exists():
                continue
            d = np.load(p)
            flat, offs = d["flat"], d["offsets"]
            xs, ys = [], []
            for t in scatter_ts:
                slot = f"f16_t{t}" if t <= 16 else f"z_t{t}"
                obs = cells_scatter.get(f"A|{slot}|{arm}", {}).get("observed")
                if obs is None:
                    continue
                vals = [
                    float(flat[offs[i] + t - 1])
                    for i in range(len(offs) - 1)
                    if offs[i + 1] - offs[i] >= t
                ]
                if not vals:
                    continue
                xs.append(float(np.mean(vals)))
                ys.append(float(obs))
            if xs:
                any_pt = True
                ax_scatter.scatter(
                    xs, ys, s=14, color=colors[arm], label=ARM_LABEL[arm], alpha=0.85
                )
        ax_scatter.set_xlabel("Mean teacher-forced -log P at position t (all captured contexts)")
        ax_scatter.set_ylabel("Test pooled R² at position t")
        ax_scatter.legend(frameon=False, fontsize=7)
        set_title_subtitle(
            ax_scatter,
            "Predictability vs token-level surprise per position",
            "Each point = one answer position t (1-16, 32, 64, 128)",
        )
        if any_arm or any_pt:
            savefig_paper(fig, "exp_surprisal_combined", dir=out_dir)
        plt.close(fig)

    # Per-cluster R² spread (TF-idf k-means OOD diagnostic; plan §6 item (b)).
    km = stats.get("kmeans_ood_diagnostic", {})
    km_clusters = km.get("per_cluster")
    if km_clusters:
        colors = dict(zip(ARMS, paper_palette(4), strict=True))
        fig, ax = plt.subplots(figsize=(6.5, 3.6), layout="constrained")
        width = 0.2
        xs_c = np.arange(len(km_clusters))
        plotted = False
        for ai, arm in enumerate(ARMS):
            ys = np.asarray(
                [
                    c[arm]["mean_r2"] if c[arm]["mean_r2"] is not None else np.nan
                    for c in km_clusters
                ],
                dtype=float,
            )
            if np.isfinite(ys).any():
                plotted = True
                ax.bar(
                    xs_c + (ai - 1.5) * width, ys, width, color=colors[arm], label=ARM_LABEL[arm]
                )
        if plotted:
            ax.set_xticks(xs_c)
            ax.set_xticklabels([f"c{c['cluster']}\nn={c['n']}" for c in km_clusters], fontsize=7)
            ax.set_xlabel("TF-idf k-means cluster (test contexts)")
            ax.set_ylabel("Mean per-context R² (position slots)")
            ax.legend(frameon=False, fontsize=7)
            set_title_subtitle(
                ax, "Per-cluster predictability spread", "OOD diagnostic — headline unaffected"
            )
            savefig_paper(fig, "exp_kmeans_cluster_spread", dir=out_dir)
        plt.close(fig)

    # Per-context R² ECDF per arm (F16-pooled per context, from the cells table
    # we cannot rebuild per-context here — use pair-level scatter for bank; the
    # LMSYS per-context ECDF needs the npz, plotted only when provided).
    if tensors_dir is not None and (tensors_dir / "per_context_stats.npz").exists():
        npz = dict(np.load(tensors_dir / "per_context_stats.npz", allow_pickle=False))
        if "A_test_ssres" in npz:
            groups = [g for g in npz["A_group_names"].tolist()]
            colors = dict(zip(ARMS, paper_palette(4), strict=True))
            fig, ax = plt.subplots(figsize=(6.0, 3.6), layout="constrained")
            for arm in ARMS:
                cols = [
                    gi
                    for gi, g in enumerate(groups)
                    if g.endswith(f"|{arm}") and g.split("|")[0] in set(F16_SLOTS)
                ]
                if not cols:
                    continue
                ssr = npz["A_test_ssres"][:, cols].astype(float)
                sst = npz["A_test_sstot"][:, cols].astype(float)
                fin = np.isfinite(ssr) & np.isfinite(sst)
                num = np.where(fin, ssr, 0.0).sum(axis=1)
                den = np.where(fin, sst, 0.0).sum(axis=1)
                r2 = np.where(den > 1e-12, 1.0 - num / den, np.nan)
                r2 = r2[np.isfinite(r2)]
                if len(r2) == 0:
                    continue
                xs = np.sort(r2)
                ax.step(
                    xs, np.arange(1, len(xs) + 1) / len(xs), color=colors[arm], label=ARM_LABEL[arm]
                )
            ax.set_xlabel("Per-context R² (first-16-token slots pooled)")
            ax.set_ylabel("ECDF")
            ax.legend(frameon=False, fontsize=7)
            set_title_subtitle(
                ax,
                "Per-context predictability distribution",
                "Raw per-unit view behind the pooled R²",
            )
            savefig_paper(fig, "exp_percontext_ecdf", dir=out_dir)
            plt.close(fig)

        # Per-context paired own-minus-plain gap at t=0 vs t=16 (matched H2
        # subset, layer 20) — the low-level per-unit view behind the hero-2
        # matched-contrast decision bars (clean-result round-2, Lens 11).
        if "M16_L20_cleg_own_ssres" in npz:

            def _m16_r2(leg: str, arm: str) -> np.ndarray:
                ssr = npz[f"M16_L20_{leg}_{arm}_ssres"].astype(float)
                sst = npz[f"M16_L20_{leg}_{arm}_sstot"].astype(float)
                return np.where(sst > 1e-12, 1.0 - ssr / sst, np.nan)

            fig, ax = plt.subplots(figsize=(6.0, 3.6), layout="constrained")
            leg_labels = {
                "cleg": "No prefix absorbed (t = 0)",
                "zleg": "After 16 prefix tokens (t = 16)",
            }
            pal2 = paper_palette(2)
            plotted_gap = False
            for li, leg in enumerate(("cleg", "zleg")):
                gap = _m16_r2(leg, "own") - _m16_r2(leg, "ext_plain")
                gap = gap[np.isfinite(gap)]
                if len(gap) == 0:
                    continue
                xs = np.sort(gap)
                ax.step(
                    xs,
                    np.arange(1, len(xs) + 1) / len(xs),
                    color=pal2[li],
                    lw=1.4,
                    label=f"{leg_labels[leg]}, n={len(gap)}",
                )
                plotted_gap = True
            if plotted_gap:
                ax.axvline(0, color="#444444", lw=0.8)
                ax.set_xlabel("Per-context R² gap: own answer minus external plain")
                ax.set_ylabel("ECDF")
                ax.legend(frameon=False, fontsize=7, loc="lower right")
                set_title_subtitle(
                    ax,
                    "Per-context gaps behind the matched prefix-closure bars",
                    "Matched survivors, identical remainder target, layer 20",
                )
                savefig_paper(fig, "exp_prefix_gap_percontext", dir=out_dir)
            plt.close(fig)

        # Per-context R² ECDF behind the mismatched t=128 recovery read (own vs
        # mismatched arms, per-layer prefix battery, layers 20 + 17) — the
        # low-level per-unit view behind the 50%-recovery aggregate (Lens 11,
        # clean-result round 4). Micro-pooling these arrays reproduces the
        # stats-battery cells (L17: own 0.367 / mismatched 0.351; L20:
        # own 0.425 / mismatched 0.405).
        if "P_own_t128_L20_ssres" in npz:
            fig, ax = plt.subplots(figsize=(6.0, 3.6), layout="constrained")
            pal2 = paper_palette(2)
            arm_color = {"own": pal2[0], "mismatch": pal2[1]}
            layer_style = {20: "-", 17: "--"}
            plotted_rec = False
            for layer in (20, 17):
                for arm in ("own", "mismatch"):
                    ssr = npz[f"P_{arm}_t128_L{layer}_ssres"].astype(float)
                    sst = npz[f"P_{arm}_t128_L{layer}_sstot"].astype(float)
                    fin = np.isfinite(ssr) & np.isfinite(sst) & (sst > 1e-12)
                    r2 = 1.0 - ssr[fin] / sst[fin]
                    if len(r2) == 0:
                        continue
                    xs = np.sort(r2)
                    ax.step(
                        xs,
                        np.arange(1, len(xs) + 1) / len(xs),
                        color=arm_color[arm],
                        ls=layer_style[layer],
                        lw=1.4,
                        label=f"{ARM_LABEL[arm]}, layer {layer}, n={len(r2)}",
                    )
                    plotted_rec = True
            if plotted_rec:
                ax.axvline(0, color="#444444", lw=0.8)
                ax.set_xlabel("Per-context remainder R² with 128 own prefix tokens absorbed")
                ax.set_ylabel("ECDF")
                ax.legend(frameon=False, fontsize=7, loc="upper left")
                set_title_subtitle(
                    ax,
                    "Per-context predictability behind the t = 128 recovery read",
                    "Own vs mismatched answer arms; test contexts with 144-token spans",
                )
                savefig_paper(fig, "exp_t128_recovery_percontext", dir=out_dir)
            plt.close(fig)

    # Turn-end split-out bars (template vs content L16 slots).
    cells = stats["cells"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6), layout="constrained")
    colors = dict(zip(ARMS, paper_palette(4), strict=True))
    groups3 = ["last-16 content tokens", "end-of-turn token", "trailing newline"]
    slot_of = {
        "last-16 content tokens": [f"l16_m{k}" for k in range(3, 17)],
        "end-of-turn token": ["l16_m2"],
        "trailing newline": ["l16_m1"],
    }
    width = 0.2
    plotted = False
    for ai, arm in enumerate(ARMS):
        ys = []
        for g in groups3:
            vals = [
                cells.get(f"A|{s}|{arm}", {}).get("observed")
                for s in slot_of[g]
                if cells.get(f"A|{s}|{arm}", {}).get("observed") is not None
            ]
            ys.append(float(np.mean(vals)) if vals else np.nan)
        if np.isfinite(ys).any():
            plotted = True
            ax.bar(
                np.arange(3) + (ai - 1.5) * width,
                ys,
                width,
                color=colors[arm],
                label=ARM_LABEL[arm],
            )
    if plotted:
        ax.set_xticks(range(3))
        ax.set_xticklabels(groups3, fontsize=8)
        ax.set_ylabel("Test pooled R²")
        ax.legend(frameon=False, fontsize=7)
        set_title_subtitle(
            ax,
            "Fixed template tokens are mechanically predictable",
            "Why the H1 late window excludes the two turn-end slots",
        )
        savefig_paper(fig, "exp_turnend_splitout", dir=out_dir)
    plt.close(fig)

    # H3 length-stratified sweep.
    h3 = stats.get("h3", {})
    strata = h3.get("length_stratified")
    if strata:
        fig, ax = plt.subplots(figsize=(5.4, 3.4), layout="constrained")
        labels, means, lo, hi = [], [], [], []
        base = h3.get("headline_mean_drop_diff", {})
        if base.get("mean") is not None:
            labels.append("all pairs")
            means.append(base["mean"])
            lo.append(base["mean_ci95"][0])
            hi.append(base["mean_ci95"][1])
        for k, rec in strata.items():
            if rec.get("mean") is None:
                continue
            labels.append(k.replace("abs_len_diff_le_", "|Δlen| ≤ "))
            means.append(rec["mean"])
            lo.append(rec["mean_ci95"][0])
            hi.append(rec["mean_ci95"][1])
        if labels:
            xs = np.arange(len(labels))
            yerr = np.vstack(
                [np.maximum(0.0, np.asarray(means) - lo), np.maximum(0.0, np.asarray(hi) - means)]
            )
            ax.errorbar(xs, means, yerr=yerr, fmt="o", capsize=3, color=paper_palette(1)[0])
            ax.axhline(0, color="#444444", lw=0.8)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel("External - own drop difference")
            set_title_subtitle(ax, "H3 under length stratification", "Answer-length confound check")
            savefig_paper(fig, "exp_h3_length_strata", dir=out_dir)
        plt.close(fig)

    # Pooled-prefix secondary curves.
    if closure is not None:
        sec = closure.get("pooled_prefix_secondary", {})
        colors = dict(zip(ARMS, paper_palette(4), strict=True))
        fig, ax = plt.subplots(figsize=(6.0, 3.6), layout="constrained")
        any_curve = False
        for arm in ARMS:
            ts, ys = [], []
            for t in PREFIX_TS:
                rec = sec.get(f"{arm}|t{t}|rem_mean")
                if isinstance(rec, dict):
                    ts.append(t)
                    ys.append(rec["test_pooled_r2"])
            if ts:
                any_curve = True
                ax.plot(ts, ys, marker="o", ms=3, lw=1.3, color=colors[arm], label=ARM_LABEL[arm])
        if any_curve:
            ax.set_xscale("log", base=2)
            ax.set_xticks(list(PREFIX_TS))
            ax.set_xticklabels([str(t) for t in PREFIX_TS], fontsize=8)
            ax.set_xlabel("Prefix length t")
            ax.set_ylabel("Remainder-mean test R²")
            ax.legend(frameon=False, fontsize=7)
            set_title_subtitle(
                ax, "Pooled-prefix predictor (secondary)", "Mean over context+prefix positions ≤ t"
            )
            savefig_paper(fig, "exp_pooled_prefix", dir=out_dir)
        plt.close(fig)


def _parent_marker(ax, x: float, y: float, color) -> None:
    """Open-circle marker distinguishing carried PARENT rows from follow-up reads."""
    ax.plot([x], [y], marker="o", ms=6, mfc="white", mec=color, mew=1.4, ls="none")


def crosslayer_decision_profile(stats: dict, out_dir: pathlib.Path) -> None:
    """Cross-layer decision figure: per-layer H1 contrast + H2 ΔG with CIs,
    margins shaded; carried L20/L17 rows plotted as open markers labeled PARENT
    (plan §3 figures; analyzer guidance: parent rows never re-derived)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2), layout="constrained")
    pal = paper_palette(4)
    added = [int(la) for la in stats["added_layers"]]

    # (a) H1 contrast (F16-vs-L16 own-plain gap contrast) per layer.
    ax1.axhspan(-0.03, 0.03, color="#D9EAD3", alpha=0.6, lw=0)
    xs, ys, ylo, yhi = [], [], [], []
    for la in added:
        rec = stats["h1_by_layer"][str(la)]["ext_plain"]
        if rec.get("h1_contrast") is None:
            continue
        xs.append(la)
        ys.append(rec["h1_contrast"])
        ci = rec.get("ci95") or [np.nan, np.nan]
        ylo.append(ci[0])
        yhi.append(ci[1])
    if xs:
        yerr = np.vstack(
            [
                np.maximum(0.0, np.asarray(ys) - np.asarray(ylo)),
                np.maximum(0.0, np.asarray(yhi) - np.asarray(ys)),
            ]
        )
        ax1.errorbar(xs, ys, yerr=yerr, fmt="o", ms=4, capsize=3, color=pal[0], label="This round")
    carried_h1 = (stats.get("carried_parent_rows") or {}).get("h1") or {}
    l20 = (carried_h1.get("L20") or {}).get("ext_plain") if carried_h1.get("L20") else None
    if l20 and l20.get("h1_contrast") is not None:
        ci = l20.get("ci95") or [np.nan, np.nan]
        ax1.errorbar(
            [20],
            [l20["h1_contrast"]],
            yerr=[[max(0.0, l20["h1_contrast"] - ci[0])], [max(0.0, ci[1] - l20["h1_contrast"])]],
            fmt="none",
            capsize=3,
            color=pal[1],
        )
        _parent_marker(ax1, 20, l20["h1_contrast"], pal[1])
        ax1.plot([], [], marker="o", mfc="white", mec=pal[1], ls="none", label="Parent (L20)")
    ax1.axhline(0, color="#444444", lw=0.8)
    ax1.set_xlabel("Read-out layer")
    ax1.set_ylabel("H1 contrast: first-16 gap minus last-16 gap\n(own vs external plain)")
    ax1.legend(frameon=False, fontsize=7)
    set_title_subtitle(
        ax1,
        "Does the early-vs-late equivalence hold across layers?",
        "Shaded band = registered ±0.03 equivalence margin; 95% bootstrap CIs",
    )

    # (b) H2 matched ΔG(0→16) per layer.
    ax2.axhspan(0.0, 0.02, color="#F4CCCC", alpha=0.5, lw=0)
    xs2, ys2, ylo2, yhi2 = [], [], [], []
    for la in added:
        rec = stats["h2_by_layer"].get(str(la)) or {}
        r = rec.get("ext_plain")
        if not r:
            continue
        xs2.append(la)
        ys2.append(r["delta_G"])
        ylo2.append(r["ci95"][0])
        yhi2.append(r["ci95"][1])
    if xs2:
        yerr2 = np.vstack(
            [
                np.maximum(0.0, np.asarray(ys2) - np.asarray(ylo2)),
                np.maximum(0.0, np.asarray(yhi2) - np.asarray(ys2)),
            ]
        )
        ax2.errorbar(
            xs2, ys2, yerr=yerr2, fmt="o", ms=4, capsize=3, color=pal[0], label="This round"
        )
    carried_h2 = (stats.get("carried_parent_rows") or {}).get("h2") or {}
    for li, la in enumerate((20, 17)):
        rec = (carried_h2.get(f"L{la}") or {}).get("ext_plain")
        if not rec:
            continue
        ci = rec.get("ci95") or [np.nan, np.nan]
        ax2.errorbar(
            [la],
            [rec["delta_G"]],
            yerr=[[max(0.0, rec["delta_G"] - ci[0])], [max(0.0, ci[1] - rec["delta_G"])]],
            fmt="none",
            capsize=3,
            color=pal[1],
        )
        _parent_marker(ax2, la, rec["delta_G"], pal[1])
        if li == 0:
            ax2.plot(
                [], [], marker="o", mfc="white", mec=pal[1], ls="none", label="Parent (L20, L17)"
            )
    ax2.axhline(0, color="#444444", lw=0.8)
    ax2.axhline(0.02, color="#B45F5F", lw=0.8, ls=":")
    ax2.set_xlabel("Read-out layer")
    ax2.set_ylabel("Matched ΔG(0→16): own-advantage closed\nby absorbing 16 prefix tokens")
    ax2.legend(frameon=False, fontsize=7)
    set_title_subtitle(
        ax2,
        "Does the prefix closure replicate across layers?",
        "Shaded band = below the registered 0.02 margin; matched survivors, identical target",
    )
    savefig_paper(fig, "crosslayer_decision_profile", dir=out_dir)
    plt.close(fig)


def crosslayer_h3_descriptive(stats: dict, out_dir: pathlib.Path) -> None:
    """Descriptive per-layer H3 panel: pooled paired drop difference with
    bootstrap CIs per layer; carried parent L20 row as an open marker."""
    fig, ax = plt.subplots(figsize=(6.0, 3.8), layout="constrained")
    pal = paper_palette(4)
    added = [int(la) for la in stats["added_layers"]]
    xs, ys, ylo, yhi = [], [], [], []
    for la in added:
        rec = stats["h3_by_layer_descriptive"].get(str(la)) or {}
        head = rec.get("headline_mean_drop_diff")
        if not head:
            continue
        xs.append(la)
        ys.append(head["mean"])
        ylo.append(head["mean_ci95"][0])
        yhi.append(head["mean_ci95"][1])
    if xs:
        yerr = np.vstack(
            [
                np.maximum(0.0, np.asarray(ys) - np.asarray(ylo)),
                np.maximum(0.0, np.asarray(yhi) - np.asarray(ys)),
            ]
        )
        ax.errorbar(xs, ys, yerr=yerr, fmt="o", ms=4, capsize=3, color=pal[0], label="This round")
    parent_h3 = (stats.get("carried_parent_rows") or {}).get("h3_L20_headline") or {}
    head = parent_h3.get("headline_mean_drop_diff")
    if head and head.get("mean") is not None:
        ci = head.get("mean_ci95") or [np.nan, np.nan]
        ax.errorbar(
            [20],
            [head["mean"]],
            yerr=[[max(0.0, head["mean"] - ci[0])], [max(0.0, ci[1] - head["mean"])]],
            fmt="none",
            capsize=3,
            color=pal[1],
        )
        _parent_marker(ax, 20, head["mean"], pal[1])
        ax.plot([], [], marker="o", mfc="white", mec=pal[1], ls="none", label="Parent (L20)")
    ax.axhline(0, color="#444444", lw=0.8)
    ax.set_xlabel("Read-out layer")
    ax.set_ylabel("External - own paired drop difference\n(divergence bank, mean)")
    ax.legend(frameon=False, fontsize=7)
    set_title_subtitle(
        ax,
        "Divergence-bank drop difference by layer",
        "Descriptive rider — no decision, no correction (plan §2)",
    )
    savefig_paper(fig, "crosslayer_h3_descriptive", dir=out_dir)
    plt.close(fig)


def crosslayer_prefix_gap_ecdf(
    stats: dict, npz: dict[str, np.ndarray], out_dir: pathlib.Path
) -> None:
    """Per-context own-minus-plain R² gap ECDF per layer (t2 = 16 matched subset)
    — the low-level per-unit companion behind the decision profile."""
    layers = sorted(
        {int(la) for la in stats["added_layers"]} | {int(stats["calibration_layer"]), 17}
    )
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 3.8), layout="constrained")
    pal = paper_palette(max(2, len(layers)))
    color_of = dict(zip(layers, pal, strict=False))

    def _gap(leg: str, layer: int) -> np.ndarray | None:
        try:
            r2 = {}
            for arm in ("own", "ext_plain"):
                ssr = npz[f"M16_L{layer}_{leg}_{arm}_ssres"].astype(float)
                sst = npz[f"M16_L{layer}_{leg}_{arm}_sstot"].astype(float)
                r2[arm] = np.where(sst > 1e-12, 1.0 - ssr / sst, np.nan)
        except KeyError:
            return None
        gap = r2["own"] - r2["ext_plain"]
        gap = gap[np.isfinite(gap)]
        return gap if len(gap) else None

    plotted = False
    for ax, leg, lab in ((ax0, "cleg", "no prefix absorbed (t = 0)"), (ax1, "zleg", "t = 16")):
        for layer in layers:
            gap = _gap(leg, layer)
            if gap is None:
                continue
            xs = np.sort(gap)
            ax.step(
                xs,
                np.arange(1, len(xs) + 1) / len(xs),
                color=color_of[layer],
                lw=1.3,
                label=f"layer {layer}, n={len(gap)}",
            )
            plotted = True
        ax.axvline(0, color="#444444", lw=0.8)
        ax.set_xlabel(f"Per-context R² gap, own minus external plain ({lab})")
        ax.set_ylabel("ECDF")
        ax.legend(frameon=False, fontsize=7, loc="lower right")
    set_title_subtitle(
        ax0,
        "Per-context gaps behind the cross-layer decision cells",
        "Matched survivors, identical remainder target",
    )
    if plotted:
        savefig_paper(fig, "crosslayer_prefix_gap_ecdf", dir=out_dir)
    plt.close(fig)


def cross_layer_figures(
    stats: dict, npz_path: pathlib.Path | None, out_dir: pathlib.Path, smoke: bool
) -> None:
    """--cross-layer driver: decision profile + descriptive H3 + per-unit ECDFs."""
    crosslayer_decision_profile(stats, out_dir)
    crosslayer_h3_descriptive(stats, out_dir)
    if npz_path is not None and npz_path.exists():
        npz = dict(np.load(npz_path, allow_pickle=False))
        crosslayer_prefix_gap_ecdf(stats, npz, out_dir)
    elif smoke:
        logger.warning("[figures] cross-layer npz absent (%s) — ECDF panel skipped", npz_path)
    else:
        raise FileNotFoundError(f"cross-layer npz required for the per-unit ECDF: {npz_path}")
    logger.info("[figures] wrote cross-layer figures to %s", out_dir)


def main() -> None:
    """Figure driver: heroes (required) + exploratory dump (input-gated)."""
    ap = argparse.ArgumentParser(description="Issue #952 figures (VM, CPU)")
    ap.add_argument("--eval-dir", type=str, required=True)
    ap.add_argument("--stats", type=str, required=True, help="stats_summary.json path")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--tensors-dir", type=str, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--cross-layer",
        action="store_true",
        help="follow-up mode: --stats points at stats_cross_layer.json; writes "
        "crosslayer_*.png only (parent hero/exploratory figures untouched)",
    )
    args = ap.parse_args()

    set_paper_style("neurips")
    eval_dir = pathlib.Path(args.eval_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = json.loads(pathlib.Path(args.stats).read_text())
    tensors_dir = pathlib.Path(args.tensors_dir) if args.tensors_dir else None

    if args.cross_layer:
        npz_path = (
            tensors_dir / "per_context_stats_cross_layer.npz" if tensors_dir is not None else None
        )
        cross_layer_figures(stats, npz_path, out_dir, args.smoke)
        return

    closure = _load_json(eval_dir / "prefix_closure_by_arm.json", optional=False, smoke=args.smoke)
    valmat = _load_json(
        eval_dir / "validation_selection_matrix.json", optional=True, smoke=args.smoke
    )

    hero1_position_r2(stats, out_dir)
    if closure is not None:
        hero2_matched_prefix(stats, closure, out_dir)
    hero3_divergence(stats, out_dir, args.smoke)
    exploratory_dump(stats, closure, valmat, tensors_dir, out_dir)
    logger.info("[figures] wrote figures to %s", out_dir)


if __name__ == "__main__":
    main()
