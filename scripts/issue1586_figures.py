"""#1586 paper-quality figures (analyzer, p11 — plan §6 figure list).

Hero 1: paired Δnorm forest (per behavior × regime × seed + seed-pooled con
CI, dose-mismatch coded). Hero 2: paired leakage ΔM forest. Exploratory dump:
per-cell ‖μ‖ bars, per-context μ-norm dumbbells, own-vs-shared-text, marker
FT dose-cliff ladders, content dose ladders, per-context panel rates,
margin-vs-rate validation scatter, marker three-space + transfer fractions,
pooled shape-DV forest.

All figures via paper_plots.set_paper_style("blog") + savefig_paper
(PNG + PDF + .meta.json sidecars) into figures/issue_1586/.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

GEO = Path("eval_results/issue_1586/geometry")
HF = Path("data/issue_1586/hf_dl/p11_json/issue1586_methodgen")
FIGDIR = Path("figures/issue_1586")
BEH_LAYER = {"syc": 14, "imp": 14, "cas": 14, "mk": 25}
BEH_NAME = {"syc": "sycophancy", "imp": "impolite", "cas": "casual style", "mk": "marker"}
SOURCE_CTX = "persona_software_engineer"


def ctx_short(c: str) -> str:
    """Plain-English read-context label (slugs stay out of rendered figures)."""
    if c.startswith("icl_prefix"):
        return "ICL prefix"
    return CTX_SHORT.get(c, c.replace("_", " "))


CTX_SHORT = {
    "persona_software_engineer": "source persona",
    "default": "default assistant",
    "wildchat_prefix_real545": "WildChat prefix",
    "icl_prefix_sycophancy": "ICL prefix",
    "icl_prefix_impoliteness": "ICL prefix",
    "icl_prefix_writing_style": "ICL prefix",
    "icl_prefix_marker": "ICL prefix",
    "neg_sp_police": "police persona",
    "neg_sp_ph4": "maritime-medic persona",
}

J = lambda p: json.loads(Path(p).read_text())  # noqa: E731


def dose_labels() -> dict[str, dict]:
    """Per pair: dose-match label from Tier-2 FT rate vs LoRA anchor (content,
    gap <= 0.10 + both in band) / selected ΔG vs anchor (marker, <= 1.5 nat)."""
    out = {}
    for beh in ("syc", "imp", "cas", "mk"):
        for reg in ("con", "po"):
            for s in ("s42", "s137"):
                cell = f"{beh}-pers-ft-{reg}-{s}"
                sel = J(HF / "selection" / cell / "selection.json")
                if beh == "mk":
                    gap = abs(sel["metric"] - sel["anchor"])
                    matched = sel["in_band"] and gap <= 1.5
                    out[f"{beh}-{reg}-{s}"] = {
                        "ft_dose": sel["metric"],
                        "anchor": sel["anchor"],
                        "gap": gap,
                        "matched": bool(matched),
                        "fallback": sel.get("fallback"),
                    }
                else:
                    t2 = J(HF / "selection" / cell / "tier2.json")["tier2_rate"]
                    if "anchor" not in sel:
                        # reused #1112 checkpoint — anchor from the #1481
                        # verdict manifest (the paired LoRA arm's rate)
                        man = J(
                            "/home/thomasjiralerspong/explore-persona-space/"
                            "eval_results/issue_1481/analysis/verdict_manifest.json"
                        )
                        arm = {"con": "con-lr1e5", "po": "po-lr1e5"}[reg]
                        sel = dict(sel)
                        sel["anchor"] = man["content"][beh]["pers"]["arms"][
                            f"{beh}-pers-{arm}-{s}"
                        ]["selection"]["rate"]
                    gap = abs(t2 - sel["anchor"])
                    matched = (0.60 <= t2 <= 0.85) and gap <= 0.10
                    out[f"{beh}-{reg}-{s}"] = {
                        "ft_dose": t2,
                        "anchor": sel["anchor"],
                        "gap": gap,
                        "matched": bool(matched),
                        "fallback": sel.get("fallback"),
                    }
    return out


def fig_hero_dnorm(doses: dict) -> None:
    per = {}
    for b in BEH_LAYER:
        d = J(GEO / f"_beh_{b}_own_norm2000.json")["diffs"]
        for name, e in d.items():
            key = name.replace("__ft_vs_lora", "").replace(f"{b}-pers-ft-", f"{b}-")
            per[key] = e["reads"][f"response/L{BEH_LAYER[b]}"]
    pooled = J(GEO / "pooled_lattice.json")["norm"]
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(9.5, 8.4))
    cols = paper_palette(4)
    y = 0
    yticks, ylabels = [], []
    for bi, b in enumerate(("syc", "imp", "cas", "mk")):
        for reg in ("con", "po"):
            for s in ("s42", "s137"):
                r = per[f"{b}-{reg}-{s}"]
                lab = doses[f"{b}-{reg}-{s}"]
                marker = "o" if lab["matched"] else "s"
                face = cols[bi] if lab["matched"] else "white"
                ax.errorbar(
                    r["point"],
                    y,
                    xerr=[[r["point"] - r["ci_low"]], [r["ci_high"] - r["point"]]],
                    fmt=marker,
                    color=cols[bi],
                    markerfacecolor=face,
                    markeredgecolor=cols[bi],
                    markeredgewidth=1.4,
                    markersize=6,
                    capsize=2,
                    lw=1.2,
                )
                yticks.append(y)
                ylabels.append(f"{BEH_NAME[b]} {reg} {s}")
                y += 1
            if reg == "con":
                pr = pooled[f"own/{b}/con/response/L{BEH_LAYER[b]}"]
                ax.errorbar(
                    pr["point"],
                    y,
                    xerr=[[pr["point"] - pr["ci_low"]], [pr["ci_high"] - pr["point"]]],
                    fmt="D",
                    color=cols[bi],
                    markersize=8,
                    capsize=3,
                    lw=2.0,
                )
                yticks.append(y)
                ylabels.append(f"{BEH_NAME[b]} con POOLED")
                y += 1
        y += 0.6
    ax.axvline(0, color="0.4", lw=0.8, ls="--")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(
        "Mean activation-shift norm difference, full fine-tune minus LoRA\n"
        "(response tokens, registered layer; 95% bootstrap CI)"
    )
    ax.set_title(
        "Full fine-tuning shifts activations farther for sycophancy and casual style,\n"
        "less far for impolite (con) and the under-dosed marker — filled = dose-matched pair",
        pad=18,
        fontsize=11,
    )
    savefig_paper(fig, "hero_dnorm_forest", dir=FIGDIR)
    plt.close(fig)


def fig_hero_leakage(doses: dict) -> None:
    lat = J(Path("eval_results/issue_1586/panel/leakage_lattice.json"))
    set_paper_style("blog")
    fig, axes = plt.subplots(
        1, 2, figsize=(12.5, 6.2), layout="none", gridspec_kw={"width_ratios": [3, 1.4]}
    )
    ax = axes[0]
    cols = paper_palette(4)
    y = 0
    yticks, ylabels = [], []
    for bi, b in enumerate(("syc", "imp", "cas")):
        for reg in ("con", "po"):
            rec = lat["content"][f"{b}/{reg}"]
            for s in ("s42", "s137"):
                r = rec[s]
                lab = doses[f"{b}-{reg}-{s}"]
                marker = "o" if lab["matched"] else "s"
                face = cols[bi] if lab["matched"] else "white"
                ax.errorbar(
                    r["delta"],
                    y,
                    xerr=[[r["delta"] - r["ci_low"]], [r["ci_high"] - r["delta"]]],
                    fmt=marker,
                    color=cols[bi],
                    markerfacecolor=face,
                    markeredgecolor=cols[bi],
                    markeredgewidth=1.4,
                    markersize=6,
                    capsize=2,
                    lw=1.2,
                )
                yticks.append(y)
                ylabels.append(f"{BEH_NAME[b]} {reg} {s}")
                y += 1
            if reg == "con":
                p = rec["pooled"]
                ax.errorbar(
                    p["delta"],
                    y,
                    xerr=[[p["delta"] - p["ci_low"]], [p["ci_high"] - p["delta"]]],
                    fmt="D",
                    color=cols[bi],
                    markersize=8,
                    capsize=3,
                    lw=2.0,
                )
                yticks.append(y)
                ylabels.append(f"{BEH_NAME[b]} con POOLED")
                y += 1
        y += 0.6
    ax.axvline(0, color="0.4", lw=0.8, ls="--")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(
        "Pooled non-source judged-rate difference,\nfull fine-tune minus LoRA (95% bootstrap CI)"
    )
    ax2 = axes[1]
    y = 0
    yticks2, ylabels2 = [], []
    for reg in ("con", "po"):
        rec = lat["marker"][reg]
        for s in ("s42", "s137"):
            d = rec["margin"]["per_seed"][s]
            lo, hi = rec["margin"]["per_seed_ci"][s]
            ax2.errorbar(
                d,
                y,
                xerr=[[d - lo], [hi - d]],
                fmt="s",
                color=cols[3],
                markerfacecolor="white",
                markeredgecolor=cols[3],
                markeredgewidth=1.4,
                markersize=6,
                capsize=2,
                lw=1.2,
            )
            yticks2.append(y)
            ylabels2.append(f"marker {reg} {s}")
            y += 1
        p, (lo, hi) = rec["margin"]["pooled_delta"], rec["margin"]["pooled_ci"]
        ax2.errorbar(
            p,
            y,
            xerr=[[p - lo], [hi - p]],
            fmt="D",
            color=cols[3],
            markerfacecolor="white",
            markeredgecolor=cols[3],
            markeredgewidth=1.6,
            markersize=8,
            capsize=3,
            lw=2.0,
        )
        yticks2.append(y)
        ylabels2.append(f"marker {reg} POOLED")
        y += 1.6
    ax2.axvline(0, color="0.4", lw=0.8, ls="--")
    ax2.set_yticks(yticks2)
    ax2.yaxis.tick_right()
    ax2.set_yticklabels(ylabels2, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("Pooled non-source EOS-margin\ndifference (nats)")
    fig.text(
        0.5,
        0.985,
        "Full fine-tuning leaks impolite and casual style to non-source contexts more than "
        "LoRA;\nsycophancy shows no matched-dose difference; every marker pair is dose-mismatched",
        ha="center",
        va="top",
        fontsize=11,
    )
    fig.subplots_adjust(top=0.86, right=0.86, bottom=0.16, wspace=0.08)
    savefig_paper(fig, "hero_leakage_forest", dir=FIGDIR)
    plt.close(fig)


def fig_percell_munorm() -> None:
    rec = J(GEO / "geometry_per_cell.json")["records"]
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.4), layout="none", sharey=False)
    for bi, b in enumerate(("syc", "imp", "cas", "mk")):
        ax = axes[bi]
        cells, vals, errs = [], [], []
        for reg in ("con", "po"):
            for s in ("s42", "s137"):
                for m in ("lora", "ft"):
                    k = f"{b}-pers-{m}-{reg}-{s}/selected/response/L{BEH_LAYER[b]}"
                    r = rec[k]
                    ci = r["boot_ci"]
                    ci = eval(ci) if isinstance(ci, str) else ci  # legacy str dict
                    cells.append(f"{m}-{reg}-{s.replace('s', '')}")
                    vals.append(r["mu_norm"])
                    errs.append(np.nan)
        xs = np.arange(len(cells))
        colors = ["#888888" if c.startswith("lora") else paper_palette(4)[bi] for c in cells]
        ax.bar(xs, vals, color=colors)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.08, f"{v:.1f}", ha="center", fontsize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels(cells, rotation=90, fontsize=7)
        ax.set_title(f"{BEH_NAME[b]} (L{BEH_LAYER[b]})", fontsize=10)
        if bi == 0:
            ax.set_ylabel("Mean activation-shift norm vs base")
    fig.text(
        0.5,
        0.99,
        "Per-cell mean-shift norms behind the paired differences (grey = LoRA, color = full FT)",
        ha="center",
        va="top",
        fontsize=11,
    )
    fig.subplots_adjust(top=0.82)
    savefig_paper(fig, "percell_munorm_bars", dir=FIGDIR)
    plt.close(fig)


def fig_perctx_dumbbells() -> None:
    """Low-level per-unit view: per-context mean-shift norm (6 contexts),
    FT vs LoRA, con regime, both seeds — from the raw capture stores."""
    import sys

    sys.path.insert(0, "scripts")
    from issue1586_geometry import CAPTURE_ARMS  # noqa: F401

    from explore_persona_space.experiments.issue_1112 import geometry as geo

    tree = GEO / "_work" / "own" / "tree"
    set_paper_style("blog")
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.2), layout="none", sharex=False)
    for bi, b in enumerate(("syc", "imp", "cas", "mk")):
        base = geo.load_store(tree / f"base_{b}" / "selected" / "pooled.pt")
        keys = geo._row_keys(base)
        ctxs = sorted({c for c, _ in keys})
        L = BEH_LAYER[b]
        for si, s in enumerate(("s42", "s137")):
            ax = axes[si][bi]
            for mi, m in enumerate(("lora", "ft")):
                st = geo.load_store(tree / f"{b}-pers-{m}-con-{s}" / "selected" / "pooled.pt")
                cloud = geo.delta_cloud(st, base, "response", L)
                vals = []
                for ctx in ctxs:
                    rows = [i for i, (c, _) in enumerate(keys) if c == ctx]
                    vals.append(float(np.linalg.norm(cloud[rows].mean(axis=0))))
                xs = np.arange(len(ctxs)) + (mi - 0.5) * 0.18
                ax.scatter(
                    xs,
                    vals,
                    s=28,
                    color="#888888" if m == "lora" else paper_palette(4)[bi],
                    label=("LoRA" if m == "lora" else "full FT") if bi == 0 else None,
                )
                for x, v in zip(xs, vals):
                    ax.text(x, v + 0.12, f"{v:.1f}", ha="center", fontsize=6)
            ax.set_xticks(np.arange(len(ctxs)))
            ax.set_xticklabels(
                [ctx_short(c)[:14] for c in ctxs], rotation=45, fontsize=6.5, ha="right"
            )
            if bi == 0:
                ax.set_ylabel(f"seed {s[1:]}\nper-context shift norm")
            if si == 0:
                ax.set_title(f"{BEH_NAME[b]} (L{L})", fontsize=10)
    axes[0][0].legend(fontsize=8)
    fig.text(
        0.5,
        0.99,
        "Per-context mean-shift norms behind each con-regime pair (grey = LoRA, color = full FT)",
        ha="center",
        va="top",
        fontsize=11,
    )
    fig.subplots_adjust(top=0.88)
    savefig_paper(fig, "perctx_shiftnorm_points", dir=FIGDIR)
    plt.close(fig)


def fig_own_vs_tf() -> None:
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    cols = paper_palette(4)
    for bi, b in enumerate(("syc", "imp", "cas", "mk")):
        own = J(GEO / f"_beh_{b}_own_norm2000.json")["diffs"]
        tf = J(GEO / "tf_shared" / f"_beh_{b}_tf_norm2000.json")["diffs"]
        L = BEH_LAYER[b]
        for name in own:
            o = own[name]["reads"][f"response/L{L}"]
            t = tf[name]["reads"][f"response/L{L}"]
            ax.scatter(
                o["point"],
                t["point"],
                color=cols[bi],
                s=34,
                label=BEH_NAME[b] if name.endswith("con-s42__ft_vs_lora") else None,
            )
            short = name.replace("__ft_vs_lora", "").replace(f"{b}-pers-ft-", "")
            ax.text(o["point"] + 0.06, t["point"], short, fontsize=6.5)
    lim = max(abs(v) for v in ax.get_xlim() + ax.get_ylim())
    ax.plot([-lim, lim], [-lim, lim], color="0.7", lw=0.8, ls=":")
    ax.axhline(0, color="0.4", lw=0.7, ls="--")
    ax.axvline(0, color="0.4", lw=0.7, ls="--")
    ax.set_xlabel("Own-generation Δnorm (full FT − LoRA)")
    ax.set_ylabel("Shared-text (teacher-forced) Δnorm")
    ax.set_title(
        "The method contrast survives the shared-text control:\nsame sign in every pair",
        pad=14,
        fontsize=11,
    )
    ax.legend(fontsize=8)
    savefig_paper(fig, "own_vs_sharedtext_dnorm", dir=FIGDIR)
    plt.close(fig)


def fig_mk_ladders() -> None:
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), layout="none")
    ax = axes[0]
    cols = paper_palette(4)
    for i, (reg, s) in enumerate([(r, s) for r in ("con", "po") for s in ("s42", "s137")]):
        cell = f"mk-pers-ft-{reg}-{s}"
        lad = J(HF / "selection" / cell / "ladder.json")["reads_by_step"]
        steps = sorted(int(k) for k in lad)
        dg = [lad[str(k)]["delta_logp_mean"] for k in steps]
        ax.plot(steps, dg, "-o", color=cols[i], ms=4, label=f"{reg} {s}")
        sel = J(HF / "selection" / cell / "selection.json")
        ax.scatter([sel["step"]], [sel["metric"]], marker="*", s=170, color=cols[i], zorder=5)
        ax.axhline(sel["anchor"], color=cols[i], lw=0.7, ls=":")
    ax.axhspan(5, 12, color="0.92", zorder=0)
    ax.set_xlabel("Full fine-tune training step")
    ax.set_ylabel("Marker install (log-prob shift vs base, nats)")
    ax.set_title(
        "Install jumps 2→9+ nats in one step;\nstars = selected rung, dotted = LoRA anchor",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax2 = axes[1]
    for i, (reg, s) in enumerate([(r, s) for r in ("con", "po") for s in ("s42", "s137")]):
        cell = f"mk-pers-ft-{reg}-{s}"
        lad = J(HF / "selection" / cell / "ladder.json")["reads_by_step"]
        steps = sorted(int(k) for k in lad)
        em = [lad[str(k)]["gen_emission_rate"] for k in steps]
        ax2.plot(steps, em, "-o", color=cols[i], ms=4, label=f"{reg} {s}")
    ax2.set_xlabel("Full fine-tune training step")
    ax2.set_ylabel("Greedy source emission rate (gate: must be 0)")
    ax2.set_title(
        "The same step turns on greedy emission —\nno rung is both in-window and gate-clean",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.99,
        "Marker full fine-tuning cannot be dose-matched: a one-step install cliff",
        ha="center",
        va="top",
        fontsize=12,
    )
    fig.subplots_adjust(top=0.84)
    savefig_paper(fig, "mk_ft_dose_cliff", dir=FIGDIR)
    plt.close(fig)


def fig_content_ladders() -> None:
    set_paper_style("blog")
    fig, axes = plt.subplots(3, 4, figsize=(14, 9.5), layout="none", sharey=True)
    cells = [
        f"{b}-pers-ft-{reg}-{s}"
        for b in ("syc", "imp", "cas")
        for reg in ("con", "po")
        for s in ("s42", "s137")
    ]
    cells = [c for c in cells if c != "syc-pers-ft-con-s42"]  # reused #1112 ckpt — no fresh ladder
    for i, cell in enumerate(cells):
        ax = axes[i // 4][i % 4]
        sel = J(HF / "selection" / cell / "selection.json")
        steps = sorted(int(k) for k in sel["reads_by_step"])
        rates = [sel["reads_by_step"][str(k)]["rate"] for k in steps]
        ax.plot(steps, rates, "-o", ms=3.5, color=paper_palette(4)[0])
        ax.axhspan(0.60, 0.85, color="0.92", zorder=0)
        ax.axhline(sel["anchor"], color="crimson", lw=0.8, ls=":")
        ax.scatter([sel["step"]], [sel["metric"]], marker="*", s=150, color="crimson", zorder=5)
        ax.set_title(cell.replace("-pers-ft", ""), fontsize=9)
        ax.set_ylim(0, 1)
    for ax in axes[-1]:
        ax.set_xlabel("training step")
    for row in axes:
        row[0].set_ylabel("Tier-1 judged rate")
    axes[2][3].axis("off")
    fig.text(
        0.5,
        0.99,
        "Content full fine-tune dose ladders (band shaded, dotted = LoRA anchor, star = selected)",
        ha="center",
        va="top",
        fontsize=11,
    )
    fig.subplots_adjust(top=0.92, hspace=0.45)
    savefig_paper(fig, "content_dose_ladders", dir=FIGDIR)
    plt.close(fig)


def fig_panel_rates(doses: dict) -> None:
    set_paper_style("blog")
    fig, axes = plt.subplots(3, 4, figsize=(14, 9.8), layout="none", sharey=True)
    pairs = [
        (b, reg, s) for b in ("syc", "imp", "cas") for reg in ("con", "po") for s in ("s42", "s137")
    ]
    for i, (b, reg, s) in enumerate(pairs):
        ax = axes[i // 4][i % 4]
        ft = J(HF / "panel" / f"{b}-pers-ft-{reg}-{s}" / "panel_summary.json")
        lo = J(HF / "panel" / f"{b}-pers-lora-{reg}-{s}" / "panel_summary.json")
        ctxs = [SOURCE_CTX] + sorted(c for c in ft["rates_by_context"] if c != SOURCE_CTX)
        xs = np.arange(len(ctxs))
        ax.bar(
            xs - 0.18,
            [lo["rates_by_context"][c] for c in ctxs],
            width=0.34,
            color="#888888",
            label="LoRA",
        )
        ax.bar(
            xs + 0.18,
            [ft["rates_by_context"][c] for c in ctxs],
            width=0.34,
            color=paper_palette(4)[0],
            label="full FT",
        )
        ax.set_xticks(xs)
        ax.set_xticklabels([ctx_short(c)[:14] for c in ctxs], rotation=45, fontsize=6.5, ha="right")
        lab = doses[f"{b}-{reg}-{s}"]
        tag = "" if lab["matched"] else "  [dose-mismatched]"
        ax.set_title(f"{BEH_NAME[b]} {reg} {s}{tag}", fontsize=9)
        ax.set_ylim(0, 1)
    axes[0][0].legend(fontsize=8)
    for row in axes:
        row[0].set_ylabel("judged rate")
    fig.text(
        0.5,
        0.995,
        "Six-context behavior rates per pair (first bar group = source context)",
        ha="center",
        va="top",
        fontsize=11,
    )
    fig.subplots_adjust(top=0.93, hspace=0.55)
    savefig_paper(fig, "panel_context_rates", dir=FIGDIR)
    plt.close(fig)


def fig_margin_vs_rate() -> None:
    from scipy.stats import spearmanr

    set_paper_style("blog")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), layout="none")
    for bi, b in enumerate(("syc", "imp", "cas")):
        ax = axes[bi]
        xs, ys, labs = [], [], []
        for reg in ("con", "po"):
            for s in ("s42", "s137"):
                for m in ("ft", "lora"):
                    arm = f"{b}-pers-{m}-{reg}-{s}"
                    xs.append(J(HF / "panel" / arm / "panel_summary.json")["source_rate"])
                    ys.append(J(HF / "margin" / arm / "margin.json")["margin_delta"])
                    labs.append(f"{m}-{reg}-{s.replace('s', '')}")
        rho, p = spearmanr(xs, ys)
        ax.scatter(xs, ys, s=34, color=paper_palette(4)[bi])
        for x, yv, la in zip(xs, ys, labs):
            ax.text(x + 0.004, yv, la, fontsize=6)
        ax.set_title(f"{BEH_NAME[b]}: rho = {rho:+.2f} (p = {p:.2f}, n = 8)", fontsize=10)
        ax.set_xlabel("Source judged rate (panel read)")
        if bi == 0:
            ax.set_ylabel("Teacher-forced fixed-pool margin shift")
    fig.text(
        0.5,
        0.99,
        "Continuous-companion validation: the margin tracks the rate for sycophancy only",
        ha="center",
        va="top",
        fontsize=11,
    )
    fig.subplots_adjust(top=0.82)
    savefig_paper(fig, "margin_vs_rate_validation", dir=FIGDIR)
    plt.close(fig)


def fig_mk_threespace() -> None:
    set_paper_style("blog")
    arms = [
        f"mk-pers-{m}-{reg}-{s}"
        for reg in ("con", "po")
        for m in ("lora", "ft")
        for s in ("s42", "s137")
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), layout="none", sharey=True)
    data = {a: J(HF / "marker_panel" / a / "slot_reads.json") for a in arms}
    ctxs = [SOURCE_CTX] + sorted(c for c in data[arms[0]]["by_context"] if c != SOURCE_CTX)
    metrics = [
        ("delta_logp_mean", "log-prob shift (nats)"),
        ("delta_margin_mean", "EOS-margin shift (nats)"),
        ("emission_rate", "greedy emission rate"),
    ]
    for mi, (mk, mlabel) in enumerate(metrics):
        ax = axes[mi]
        for ai, a in enumerate(arms):
            vals = [data[a]["by_context"][c][mk] for c in ctxs]
            color = "#888888" if "-lora-" in a else paper_palette(4)[3]
            ls = "-" if "-con-" in a else "--"
            ax.plot(
                np.arange(len(ctxs)),
                vals,
                ls,
                color=color,
                lw=1.1,
                alpha=0.85,
                label=a.replace("mk-pers-", "") if mi == 0 else None,
            )
        ax.set_xticks(np.arange(len(ctxs)))
        ax.set_xticklabels([ctx_short(c)[:14] for c in ctxs], rotation=45, fontsize=6.5, ha="right")
        ax.set_title(mlabel, fontsize=10)
    axes[0].legend(fontsize=6.5)
    axes[0].set_ylabel("trained − base at the marker slot")
    fig.text(
        0.5,
        0.99,
        "Marker three-space panel reads per context (grey = LoRA, color = full FT; "
        "solid = con, dashed = po)",
        ha="center",
        va="top",
        fontsize=11,
    )
    fig.subplots_adjust(top=0.84)
    savefig_paper(fig, "mk_threespace_contexts", dir=FIGDIR)
    plt.close(fig)


def fig_mk_transfer() -> None:
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    arms = [
        f"mk-pers-{m}-{reg}-{s}"
        for reg in ("con", "po")
        for m in ("lora", "ft")
        for s in ("s42", "s137")
    ]
    fracs, cols_, labs = [], [], []
    for a in arms:
        d = J(HF / "marker_panel" / a / "slot_reads.json")
        src = d["by_context"][SOURCE_CTX]["delta_margin_mean"]
        fracs.append(d["pooled_nonsource_delta_margin"] / src)
        cols_.append("#888888" if "-lora-" in a else paper_palette(4)[3])
        labs.append(a.replace("mk-pers-", ""))
    xs = np.arange(len(arms))
    ax.bar(xs, fracs, color=cols_)
    for x, v in zip(xs, fracs):
        ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labs, rotation=45, fontsize=8, ha="right")
    ax.set_ylabel("Non-source / source EOS-margin shift")
    ax.set_title(
        "Install-normalized marker transfer fraction (EOS-margin space;\n"
        "full-FT arms are dose-mismatched — descriptive only)",
        pad=12,
        fontsize=10,
    )
    savefig_paper(fig, "mk_transfer_fraction", dir=FIGDIR)
    plt.close(fig)


def fig_shape_forest() -> None:
    pooled = J(GEO / "pooled_lattice.json")["shape"]
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.2), layout="none", sharey=True)
    cols = paper_palette(4)
    dvs = [
        ("rank_k_at_90", "rank containing 90% of shift variance"),
        ("pr_lambda", "participation ratio"),
        ("top_share_lambda", "top-eigenvalue share"),
    ]
    for di, (dv, dlabel) in enumerate(dvs):
        ax = axes[di]
        y = 0
        yt, yl = [], []
        for bi, b in enumerate(("syc", "imp", "cas", "mk")):
            for reg in ("con", "po"):
                r = pooled[f"own/{b}/{reg}/response/L{BEH_LAYER[b]}"][dv]
                ax.errorbar(
                    r["point"],
                    y,
                    xerr=[
                        [max(0.0, r["point"] - r["ci_low"])],
                        [max(0.0, r["ci_high"] - r["point"])],
                    ],
                    fmt="o",
                    color=cols[bi],
                    markersize=6,
                    capsize=2,
                    lw=1.2,
                )
                yt.append(y)
                yl.append(f"{BEH_NAME[b]} {reg}")
                y += 1
            y += 0.5
        ax.axvline(0, color="0.4", lw=0.8, ls="--")
        if di == 0:
            ax.set_yticks(yt)
            ax.set_yticklabels(yl, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel(f"Δ {dlabel}\n(full FT − LoRA, pooled)")
    fig.text(
        0.5,
        0.99,
        "Shift-cloud shape differences at the selected rungs: null except impolite-con "
        "and casual-po concentration",
        ha="center",
        va="top",
        fontsize=11,
    )
    fig.subplots_adjust(top=0.88)
    savefig_paper(fig, "shape_dv_forest", dir=FIGDIR)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """Input/output-root plumb (plan v7 §4.C item 3): ``--geo-root`` /
    ``--hf-root`` / ``--fig-dir`` / ``--lattice`` override the module roots so
    the analyzer can render against the FU trees. A FU invocation MUST pass a
    non-default ``--fig-dir`` (e.g. figures/issue_1586/fu_caveatfix) — the
    default dir holds the EXECUTED run's committed figures + dose_labels.json,
    which this script rewrites (smoke/committed-artifact clobber rule).
    Defaults are byte-identical to the executed run's behavior."""
    global GEO, HF, FIGDIR
    import argparse

    ap = argparse.ArgumentParser(description="#1586 paper-quality figures (p11)")
    ap.add_argument("--geo-root", type=Path, default=GEO)
    ap.add_argument("--hf-root", type=Path, default=HF)
    ap.add_argument("--fig-dir", type=Path, default=FIGDIR)
    ap.add_argument(
        "--lattice",
        type=Path,
        default=Path("eval_results/issue_1586/panel/leakage_lattice.json"),
    )
    args = ap.parse_args(argv)
    GEO, HF, FIGDIR = Path(args.geo_root), Path(args.hf_root), Path(args.fig_dir)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    doses = dose_labels()
    (FIGDIR / "dose_labels.json").write_text(json.dumps(doses, indent=1))
    fig_hero_dnorm(doses)
    fig_percell_munorm()
    fig_perctx_dumbbells()
    fig_own_vs_tf()
    fig_mk_ladders()
    fig_content_ladders()
    fig_margin_vs_rate()
    fig_mk_threespace()
    fig_mk_transfer()
    fig_shape_forest()
    if Path(args.lattice).exists():
        fig_hero_leakage(doses)
        fig_panel_rates(doses)
    else:
        print("leakage_lattice.json missing — hero 2 + panel rates SKIPPED")
    print("figures written to", FIGDIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
