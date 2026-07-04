# ruff: noqa: RUF001, RUF003
"""Issue #813 follow-up figures — per-example vs question-averaged map (frozen L14).

Reads eval_results/issue_813/per_example_vs_averaged/*.json (the
issue813_per_example_maps.py driver outputs) and emits, under figures/issue_813/:

1. hero_pe_vs_avg_transfer_<arm>   — paired own-vs-transfer R² per cell, both task
                                     directions, with the question-half refit-twin
                                     ceiling band (the same-function noise ceiling).
2. per_context_delta_r2_<arm>      — low-level per-unit companion: per-context paired
                                     ΔR² points (avg task), context-labeled.
3. overlap_cka_vs_k_<arm>          — top-k subspace overlap + CKA vs the twin bands
                                     (averaged-grain AND per-example-grain twins) and
                                     the analytic random floor.
4. r2_vs_k_attenuation_<arm>       — R²-vs-k averaging curves with the analytic
                                     reliability-attenuation overlay.
5. dv4_query_specific              — within-context incremental R² bars vs shuffle nulls.
6. dv6_pe_vs_avg_scatter           — per-example vs committed averaged Δ/floor
                                     (12 labeled points) + dv6_em_null_hist.

Uses the paper-plots conventions (set_paper_style("blog") + savefig_paper); labels
are plain English end to end (no opaque condition codes).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RES = ROOT / "eval_results/issue_813/per_example_vs_averaged"
DEFAULT_OUT = ROOT / "figures/issue_813"

BEHAVIORS = ["em", "fact", "sycophancy", "marker"]
SUBSTRATES = ["generic", "elicit", "mix"]
BEH_LABEL = {
    "em": "emergent misalignment",
    "fact": "fact",
    "sycophancy": "sycophancy",
    "marker": "marker",
}
SUB_LABEL = {"generic": "generic UltraChat", "elicit": "behavior-eliciting", "mix": "mixed-pool"}
TICK_LABEL = {
    "generic": "generic\nUltraChat",
    "elicit": "behavior-\neliciting",
    "mix": "mixed\npool",
}
ARM_LABEL = {"base": "base model", "trained": "finetuned model"}
L = 14


def load_cells(res_dir: Path) -> dict[tuple[str, str], dict]:
    cells = {}
    missing = []
    for beh in BEHAVIORS:
        for sub in SUBSTRATES:
            p = res_dir / f"transfer_L{L}_{beh}__{sub}.json"
            if p.exists():
                cells[(beh, sub)] = json.loads(p.read_text())
            else:
                missing.append(f"{beh}/{sub}")
    if missing:
        print(f"NOTE: {len(missing)} cell(s) absent (plotting available only): {missing}")
    if not cells:
        raise SystemExit(f"no transfer_L{L}_*.json found under {res_dir}")
    return cells


def _reads(cell: dict, arm: str) -> dict:
    return cell["dv1"]["loco"][arm]["reads"]["shared"]


def fig_hero(cells: dict, arm: str, out: Path) -> None:
    colors = paper_palette_blog(3)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.6), constrained_layout=True)
    tasks = [
        (
            "averaged task",
            "m_own_avg",
            "hpe_to_avg",
            "averaged map (own fit)",
            "per-example map (transferred)",
        ),
        (
            "per-question task",
            "hpe_own_pq",
            "m_to_pq",
            "per-example map (own fit)",
            "averaged map (transferred)",
        ),
    ]
    for row, (tname, own_key, xfer_key, _own_lab, _xfer_lab) in enumerate(tasks):
        for col, beh in enumerate(BEHAVIORS):
            ax = axes[row, col]
            for i, sub in enumerate(SUBSTRATES):
                c = cells.get((beh, sub))
                if c is None:
                    continue
                r = _reads(c, arm)
                own = r[own_key]["r2_pooled"]
                xfer = r[xfer_key]["r2_pooled"]
                ax.plot([i, i], [own, xfer], color="0.6", lw=1.2, zorder=1)
                ax.plot([i], [own], "o", color=colors[i], ms=8, zorder=3)
                ax.plot([i], [xfer], "s", color=colors[i], ms=7, mfc="white", zorder=3)
                if row == 0:  # twin ceiling band on the averaged task (own − twin gap)
                    tw = c["twins"]["per_arm"][arm]
                    if tw["gap_p95"] is not None:
                        ax.fill_between(
                            [i - 0.28, i + 0.28],
                            own - tw["gap_p95"],
                            own,
                            color=colors[i],
                            alpha=0.18,
                            zorder=0,
                        )
            ax.set_xticks(range(3))
            ax.set_xticklabels([TICK_LABEL[s] for s in SUBSTRATES], fontsize=8)
            ax.axhline(0, color="0.3", lw=0.8)
            if row == 0:
                ax.set_title(BEH_LABEL[beh])
            if col == 0:
                ax.set_ylabel(f"{tname}\nheld-out R² (LOCO, L{L})")
    own_h = plt.Line2D([], [], marker="o", ls="", color="0.3", label="own-grain fit")
    xf_h = plt.Line2D(
        [], [], marker="s", ls="", mfc="white", color="0.3", label="cross-grain transfer"
    )
    band = plt.Rectangle(
        (0, 0), 1, 1, color="0.6", alpha=0.25, label="question-half refit-twin ceiling (p95 gap)"
    )
    fig.legend(handles=[own_h, xf_h, band], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(
        f"Does the per-example map compute the same function as the question-averaged map? "
        f"({ARM_LABEL[arm]}, layer {L})",
        y=1.03,
    )
    savefig_paper(fig, f"hero_pe_vs_avg_transfer_{arm}", dir=out)
    plt.close(fig)


def fig_per_context(cells: dict, arm: str, out: Path) -> None:
    colors = paper_palette_blog(3)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), constrained_layout=True)
    for col, beh in enumerate(BEHAVIORS):
        ax = axes[col]
        for i, sub in enumerate(SUBSTRATES):
            c = cells.get((beh, sub))
            if c is None:
                continue
            r = _reads(c, arm)
            own = np.asarray(r["m_own_avg"]["per_fold_r2"], dtype=float)
            xfer = np.asarray(r["hpe_to_avg"]["per_fold_r2"], dtype=float)
            gaps = own - xfer
            xs = np.full(len(gaps), i, dtype=float) + (np.arange(len(gaps)) - len(gaps) / 2) * (
                0.5 / max(len(gaps), 1)
            )
            ax.scatter(xs, gaps, s=12, color=colors[i], alpha=0.75)
            # label extreme contexts (top-2 |gap|) with their battery context id
            ids = c.get("ctx_ids", [str(j) for j in range(len(gaps))])
            for j in np.argsort(-np.abs(gaps))[:2]:
                ax.annotate(str(ids[j])[:14], (xs[j], gaps[j]), fontsize=6, ha="left", va="bottom")
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set_xticks(range(3))
        ax.set_xticklabels([TICK_LABEL[s] for s in SUBSTRATES], fontsize=8)
        ax.set_title(BEH_LABEL[beh])
        if col == 0:
            ax.set_ylabel("per-context ΔR²\n(averaged-map own − per-example transferred)")
    fig.suptitle(
        f"Per-context paired transfer gaps on the averaged task ({ARM_LABEL[arm]}, layer {L}); "
        "labels mark the two largest-gap contexts",
        y=1.06,
    )
    savefig_paper(fig, f"per_context_delta_r2_{arm}", dir=out)
    plt.close(fig)


def fig_overlap(cells: dict, arm: str, out: Path) -> None:
    colors = paper_palette_blog(3)
    ks = [5, 10, 20]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.4), constrained_layout=True)
    for col, beh in enumerate(BEHAVIORS):
        for row, side in enumerate(("overlap_input", "overlap_output")):
            ax = axes[row, col]
            for i, sub in enumerate(SUBSTRATES):
                c = cells.get((beh, sub))
                if c is None:
                    continue
                blk = c["dv2"]["per_arm"][arm]
                obs = [blk["observed_pe_vs_avg"][side][str(k)] for k in ks]
                ax.plot(ks, obs, "o-", color=colors[i], label=SUB_LABEL[sub] if col == 0 else None)
                tw = blk["twin_avg"][f"{side.replace('overlap_', 'overlap_')}_full"]
                lo = [tw[str(k)].get("p5") for k in ks]
                hi = [tw[str(k)].get("p95") for k in ks]
                if all(v is not None for v in lo + hi):
                    ax.fill_between(ks, lo, hi, color=colors[i], alpha=0.15)
                floor = [blk["random_floor"][side.split("_")[1]][str(k)] for k in ks]
                ax.plot(ks, floor, ":", color="0.4", lw=1.0)
            ax.set_xticks(ks)
            ax.set_ylim(0, 1.02)
            if row == 0:
                ax.set_title(BEH_LABEL[beh])
            if col == 0:
                ax.set_ylabel(
                    ("input-side" if row == 0 else "output-side")
                    + "\ntop-k subspace overlap\n(mean sq. canonical cosine)"
                )
            ax.set_xlabel("k")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], ls=":", color="0.4", label="random-subspace floor"))
    labels.append("random-subspace floor")
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(
        f"Per-example vs averaged map: weight-space agreement, shaded = averaged-grain "
        f"question-half twin band ({ARM_LABEL[arm]}, layer {L})",
        y=1.03,
    )
    savefig_paper(fig, f"overlap_vs_k_{arm}", dir=out)
    plt.close(fig)
    # CKA companion (single row): observed vs both twin ceilings
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), constrained_layout=True)
    for col, beh in enumerate(BEHAVIORS):
        ax = axes[col]
        for i, sub in enumerate(SUBSTRATES):
            c = cells.get((beh, sub))
            if c is None:
                continue
            blk = c["dv2"]["per_arm"][arm]
            ax.plot([i], [blk["observed_pe_vs_avg"]["cka"]], "o", color=colors[i], ms=9)
            for twk, mk in (("twin_avg", "_"), ("twin_pe", "x")):
                tw = blk[twk]["cka"]
                if tw.get("n", 0) > 0:
                    ax.plot([i - 0.15, i + 0.15], [tw["p5"]] * 2, color=colors[i], lw=1.4)
                    ax.plot([i], [tw["mean"]], marker=mk, color=colors[i], ms=7, mfc="none")
        ax.set_xticks(range(3))
        ax.set_xticklabels([TICK_LABEL[s] for s in SUBSTRATES], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(BEH_LABEL[beh])
        if col == 0:
            ax.set_ylabel("linear CKA between maps")
    fig.suptitle(
        f"CKA: dot = per-example vs averaged (observed); bar = twin p5 ceiling; "
        f"_ = averaged twin mean, x = per-example twin mean ({ARM_LABEL[arm]})",
        y=1.06,
    )
    savefig_paper(fig, f"cka_{arm}", dir=out)
    plt.close(fig)


def fig_r2_vs_k(cells: dict, arm: str, out: Path) -> None:
    colors = paper_palette_blog(3)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), constrained_layout=True)
    for col, beh in enumerate(BEHAVIORS):
        ax = axes[col]
        for i, sub in enumerate(SUBSTRATES):
            c = cells.get((beh, sub))
            if c is None:
                continue
            blk = c["dv3"]["per_arm"][arm]
            ks = [int(k) for k in c["dv3"]["k_grid"]]
            mean = [blk["per_k"][str(k)]["r2_mean"] for k in ks]
            lo = [blk["per_k"][str(k)]["r2_p2_5"] for k in ks]
            hi = [blk["per_k"][str(k)]["r2_p97_5"] for k in ks]
            kv = [k for k, m in zip(ks, mean, strict=True) if m is not None]
            mv = [m for m in mean if m is not None]
            ax.plot(kv, mv, "o-", color=colors[i], label=SUB_LABEL[sub] if col == 0 else None)
            if all(v is not None for v in lo + hi):
                ax.fill_between(
                    kv,
                    [v for v in lo if v is not None],
                    [v for v in hi if v is not None],
                    color=colors[i],
                    alpha=0.15,
                )
            # analytic reliability-attenuation overlay, anchored at the largest k
            att = blk["attenuation"]
            rel = {
                int(k): att["output_64d"]["reliability_signal_weighted"][k]
                * att["input_raw"]["reliability_signal_weighted"][k]
                for k in att["output_64d"]["reliability_signal_weighted"]
            }
            if kv and mv[-1] is not None and rel.get(kv[-1], 0) > 0:
                scale = mv[-1] / rel[kv[-1]]
                ax.plot(kv, [scale * rel[k] for k in kv], "--", color=colors[i], lw=1.0, alpha=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("questions averaged per context (k)")
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set_title(BEH_LABEL[beh])
        if col == 0:
            ax.set_ylabel("held-out R² of the k-average map")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(
        plt.Line2D([], [], ls="--", color="0.4", label="analytic reliability attenuation (scaled)")
    )
    labels.append("analytic reliability attenuation (scaled)")
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle(
        f"Averaging more questions mostly buys denoising ({ARM_LABEL[arm]}, layer {L})", y=1.06
    )
    savefig_paper(fig, f"r2_vs_k_attenuation_{arm}", dir=out)
    plt.close(fig)


def fig_dv4(cells: dict, out: Path) -> None:
    colors = paper_palette_blog(3)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), constrained_layout=True)
    width = 0.34
    for col, beh in enumerate(BEHAVIORS):
        ax = axes[col]
        for i, sub in enumerate(SUBSTRATES):
            c = cells.get((beh, sub))
            if c is None:
                continue
            for a, arm in enumerate(("base", "trained")):
                blk = c["dv4"][arm]
                x = i + (a - 0.5) * width
                ax.bar(
                    x,
                    blk["r2_within_observed"],
                    width=width * 0.92,
                    color=colors[i],
                    alpha=1.0 if arm == "trained" else 0.45,
                )
                ax.plot(
                    [x - width / 2, x + width / 2],
                    [blk["null_p95"]] * 2,
                    color="0.15",
                    lw=1.6,
                )
        ax.set_xticks(range(3))
        ax.set_xticklabels([TICK_LABEL[s] for s in SUBSTRATES], fontsize=8)
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set_title(BEH_LABEL[beh])
        if col == 0:
            ax.set_ylabel("within-context incremental R²\n(query-specific signal)")
    fig.suptitle(
        "Query-specific signal the averaged map is blind to: bars = observed (light = base, "
        "dark = finetuned); black line = within-context shuffle null p95",
        y=1.05,
    )
    savefig_paper(fig, "dv4_query_specific", dir=out)
    plt.close(fig)


def fig_dv6(cells: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 6.2), constrained_layout=True)
    beh_colors = dict(zip(BEHAVIORS, paper_palette_blog(4), strict=True))
    xs, ys = [], []
    for (beh, sub), c in cells.items():
        dv6 = c["dv6"]
        x = dv6.get("committed_averaged", {}).get("avg_delta_over_floor")
        y = dv6.get("delta_pe_over_floor")
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
        ax.plot([x], [y], "o", color=beh_colors[beh], ms=9)
        ax.annotate(
            f"{BEH_LABEL[beh]}, {SUB_LABEL[sub]}", (x, y), fontsize=7, ha="left", va="bottom"
        )
    if xs:
        lim = [min(xs + ys) * 0.6 + 1e-3, max(xs + ys) * 1.6]
        ax.plot(lim, lim, "--", color="0.5", lw=1.0)
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("committed question-averaged map change (Δ/floor, layer 14)")
    ax.set_ylabel("per-example map change (Δ/floor, layer 14)")
    ax.set_title("Pre/post-finetuning map change: per-example vs averaged grain (12 cells)")
    savefig_paper(fig, "dv6_pe_vs_avg_scatter", dir=out)
    plt.close(fig)
    # em null histogram (raw Δ_pe diffs, per substrate) + committed parent p95 note
    em_cells = {s: cells.get(("em", s)) for s in SUBSTRATES}
    if any(c is not None and "em_pe_null" in c.get("dv6", {}) for c in em_cells.values()):
        colors = paper_palette_blog(3)
        fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
        for i, sub in enumerate(SUBSTRATES):
            c = em_cells.get(sub)
            if c is None or "em_pe_null" not in c["dv6"]:
                continue
            null = c["dv6"]["em_pe_null"]
            draws = np.asarray(null["null_draws_raw"], dtype=float)
            fl = c["dv6"]["floor_combined"]
            if draws.size and fl and fl > 0:
                ax.hist(
                    draws / fl,
                    bins=40,
                    density=True,
                    alpha=0.45,
                    color=colors[i],
                    label=f"null, {SUB_LABEL[sub]} questions (÷ observed floor)",
                )
        dofs = {
            s: em_cells[s]["dv6"]["delta_pe_over_floor"]
            for s in SUBSTRATES
            if em_cells.get(s) is not None and em_cells[s]["dv6"].get("delta_pe_over_floor")
        }
        if len(dofs) >= 2:
            spread = max(dofs.values()) - min(dofs.values())
            ax.axvline(
                spread, color="#c1272d", lw=2.2, label="observed per-example max − min spread"
            )
        p95s = [
            em_cells[s]["dv6"]
            .get("committed_averaged", {})
            .get("parent_null_over_floor_p95_1000draw")
            for s in SUBSTRATES
            if em_cells.get(s) is not None
        ]
        p95s = [p for p in p95s if p is not None]
        if p95s:
            ax.axvline(
                max(p95s),
                color="0.25",
                ls="--",
                lw=1.6,
                label="parent averaged-grain null p95 (1000 draws, widest substrate)",
            )
        ax.set_xlabel("difference in map-change size between question halves (Δ/floor units)")
        ax.set_ylabel("null density")
        ax.set_title("emergent misalignment: per-example question-resampling null (200 draws)")
        ax.legend(fontsize=8)
        savefig_paper(fig, "dv6_em_null_hist", dir=out)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Issue #813 per-example-vs-averaged figures")
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RES)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--arms", nargs="+", default=["trained", "base"], choices=["base", "trained"])
    args = ap.parse_args()
    set_paper_style("blog")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cells = load_cells(args.results_dir)
    for arm in args.arms:
        fig_hero(cells, arm, args.out_dir)
        fig_per_context(cells, arm, args.out_dir)
        fig_overlap(cells, arm, args.out_dir)
        fig_r2_vs_k(cells, arm, args.out_dir)
    fig_dv4(cells, args.out_dir)
    fig_dv6(cells, args.out_dir)
    print(f"figures written to {args.out_dir}")


if __name__ == "__main__":
    main()
