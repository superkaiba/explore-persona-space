"""#1586 fu caveat-fix round figures (analyzer-authored; issue1586_figures.py
refuses --fu by design).

Five figures for the round `caveat-fix-marker-dosematch-impolite-lr-deconfound`,
all read from the round's committed eval JSONs (never hand-typed numbers):

1. fu_mk_dose_ladder_2e6      — marker full-FT install ladders at lr 2e-6 vs the
                                executed 5e-6 ladders; window, anchors, selections.
2. fu_mk_matched_dnorm_forest — marker paired Δnorm at matched install (own +
                                shared-text), per pair + seed-pooled.
3. fu_imp_lr_deconfound_forest— impolite response-arm paired Δnorm: factory-LR
                                anchors (executed) vs matched-LR anchors (this
                                round), own + shared-text.
4. fu_imp_leakage             — impolite leakage: paired pooled non-source rate
                                deltas (executed vs matched-LR) + per-context rates.
5. fu_mk_leakage_forest       — marker leakage at matched install: pooled
                                non-source EOS-margin + log-prob deltas.

Style: paper_plots "blog", colorblind-safe, no text overlays/arrows.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

WT = Path(__file__).resolve().parents[1]
R = WT / "eval_results/issue_1586/caveat-fix-marker-dosematch-impolite-lr-deconfound"
MAIN = Path("/home/thomasjiralerspong/explore-persona-space")
FIGDIR = WT / "figures/issue_1586"
LADDERS = Path("/tmp/i1586fu/mk_ladders_combined.json")
PANEL_SUMMARIES = Path("/tmp/i1586fu/issue1586_methodgen/fu_caveatfix/panel")

set_paper_style("blog")
C = paper_palette(6)


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def fig_mk_dose_ladder() -> None:
    d = _load(LADDERS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    short = {
        "mk-pers-ft2e6-con-s42": "contrastive s42",
        "mk-pers-ft2e6-con-s137": "contrastive s137",
        "mk-pers-ft2e6-po-s42": "positive-only s42",
        "mk-pers-ft2e6-po-s137": "positive-only s137",
    }
    map5 = {
        "mk-pers-ft2e6-con-s42": "mk-pers-ft-con-s42",
        "mk-pers-ft2e6-con-s137": "mk-pers-ft-con-s137",
        "mk-pers-ft2e6-po-s42": "mk-pers-ft-po-s42",
        "mk-pers-ft2e6-po-s137": "mk-pers-ft-po-s137",
    }
    ax = axes[0]
    ax.axhspan(d["window"][0], d["window"][1], color="0.92", zorder=0)
    for i, (cell, lab) in enumerate(short.items()):
        steps = sorted(int(k) for k in d["lr2e6"][cell])
        vals = [d["lr2e6"][cell][str(s)][0] for s in steps]
        ax.plot(steps, vals, marker="o", ms=3.5, color=C[i], label=f"{lab} (2e-6)")
        sel = d["selected"][cell]
        ax.plot([sel], [d["lr2e6"][cell][str(sel)][0]], marker="*", ms=15, color=C[i], zorder=5)
        s5 = sorted(int(k) for k in d["lr5e6"][map5[cell]])
        v5 = [d["lr5e6"][map5[cell]][str(s)][0] for s in s5]
        ax.plot(s5, v5, ls="--", lw=1.0, alpha=0.55, color=C[i])
        ax.axhline(d["anchors_committed"][cell], ls=":", lw=0.8, color=C[i], alpha=0.7)
    ax.set_xlabel("training step (effective batch 64)")
    ax.set_ylabel("marker install, log-prob shift vs base (nats)")
    ax.legend(fontsize=8, loc="lower right")
    ax = axes[1]
    for i, (cell, lab) in enumerate(short.items()):
        steps = sorted(int(k) for k in d["lr2e6"][cell])
        ax.plot(
            steps,
            [d["lr2e6"][cell][str(s)][1] for s in steps],
            marker="o",
            ms=3.5,
            color=C[i],
        )
        s5 = sorted(int(k) for k in d["lr5e6"][map5[cell]])
        ax.plot(
            s5,
            [d["lr5e6"][map5[cell]][str(s)][1] for s in s5],
            ls="--",
            lw=1.0,
            alpha=0.55,
            color=C[i],
        )
    ax.set_xlabel("training step")
    ax.set_ylabel("greedy source emission rate")
    ax.set_ylim(-0.02, 0.65)
    fig.text(
        0.5,
        1.005,
        "Marker full fine-tune install per step: learning rate 2e-6 (solid, this round)"
        " vs 5e-6 (dashed, executed run); stars = selected rungs",
        ha="center",
        fontsize=11,
    )
    savefig_paper(fig, "fu_mk_dose_ladder_2e6", dir=FIGDIR)
    plt.close(fig)


def _forest(ax, rows, xlabel):
    """rows: list of (label, point, lo, hi, color, marker, filled)."""
    y = np.arange(len(rows))[::-1]
    for yi, (lab, pt, lo, hi, col, mk, filled) in zip(y, rows):
        if lo is not None:
            ax.plot([lo, hi], [yi, yi], color=col, lw=1.6)
        ax.plot(
            [pt],
            [yi],
            marker=mk,
            ms=8,
            color=col,
            mfc=col if filled else "white",
            mew=1.5,
        )
    ax.axvline(0, color="0.4", lw=0.9, ls="-")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel(xlabel)


def fig_mk_matched_forest() -> None:
    own = _load(R / "geometry/_beh_mk_own_norm2000.json")["diffs"]
    tf = _load(R / "geometry/tf_shared/_beh_mk_tf_norm2000.json")["diffs"]
    pooled = _load(R / "geometry/pooled_lattice_fu.json")["norm"]

    def rd(src, pair):
        r = src[pair]["reads"]["response/L25"]
        return r["point"], r["ci_low"], r["ci_high"]

    rows = []
    for regime, col in (("con", C[0]), ("po", C[1])):
        for seed in ("s42", "s137"):
            pair = f"mk-pers-ft2e6-{regime}-{seed}__ft_vs_lora"
            lab = f"{'contrastive' if regime == 'con' else 'positive-only'} {seed}"
            mk = "s" if (regime, seed) == (("con", "s137")) else "o"
            filled = (regime, seed) != ("con", "s137")
            rows.append((lab, *rd(own, pair), col, mk, filled))
            p2, l2, h2 = rd(tf, pair)
            rows.append((f"{lab} (shared text)", p2, l2, h2, col, "D", False))
        pr = pooled[f"own/mk/{regime}/response/L25"]
        rows.append(
            (
                f"{'contrastive' if regime == 'con' else 'positive-only'} pooled",
                pr["point"],
                pr["ci_low"],
                pr["ci_high"],
                col,
                "o",
                True,
            )
        )
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    _forest(
        ax,
        rows,
        "paired shift-norm difference, full FT minus LoRA\n(layer 25, response arm)",
    )
    ax.set_title(
        "Marker method contrast at matched install (learning rate 2e-6 full fine-tune)",
        pad=14,
        fontsize=11,
    )
    savefig_paper(fig, "fu_mk_matched_dnorm_forest", dir=FIGDIR)
    plt.close(fig)


def fig_imp_deconfound_forest() -> None:
    exec_pairs = _load(MAIN / "eval_results/issue_1586/geometry/_beh_imp_own_norm2000.json")[
        "diffs"
    ]
    exec_pooled = _load(MAIN / "eval_results/issue_1586/geometry/pooled_lattice.json")["norm"]
    own = _load(R / "geometry/_beh_imp_own_norm2000.json")["diffs"]
    tf = _load(R / "geometry/tf_shared/_beh_imp_tf_norm2000.json")["diffs"]
    pooled = _load(R / "geometry/pooled_lattice_fu.json")["norm"]

    def rd(src, pair, key="response/L14"):
        r = src[pair]["reads"][key]
        return r["point"], r["ci_low"], r["ci_high"]

    rows = []
    for seed in ("s42", "s137"):
        p, lo, hi = rd(exec_pairs, f"imp-pers-ft-con-{seed}__ft_vs_lora")
        rows.append((f"factory-rate anchors {seed}", p, lo, hi, C[3], "o", True))
    ep = exec_pooled["own/imp/con/response/L14"]
    rows.append(
        ("factory-rate anchors pooled", ep["point"], ep["ci_low"], ep["ci_high"], C[3], "o", True)
    )
    for seed in ("s42", "s137"):
        p, lo, hi = rd(own, f"imp-pers-lora5e6-con-{seed}__ft_vs_lora")
        rows.append((f"matched-rate anchors {seed}", p, lo, hi, C[0], "o", True))
    np_ = pooled["own/imp/con/response/L14"]
    rows.append(
        (
            "matched-rate anchors pooled",
            np_["point"],
            np_["ci_low"],
            np_["ci_high"],
            C[0],
            "o",
            True,
        )
    )
    for seed in ("s42", "s137"):
        p, lo, hi = rd(tf, f"imp-pers-lora5e6-con-{seed}__ft_vs_lora")
        rows.append((f"matched-rate, shared text {seed}", p, lo, hi, C[0], "D", False))
    tp = pooled["tf/imp/con/response/L14"]
    rows.append(
        (
            "matched-rate, shared text pooled",
            tp["point"],
            tp["ci_low"],
            tp["ci_high"],
            C[0],
            "D",
            False,
        )
    )
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    _forest(
        ax,
        rows,
        "paired shift-norm difference, full FT minus LoRA\n(layer 14, response arm)",
    )
    ax.set_title(
        "Impolite contrastive reversal: factory learning-rate anchors (3e-5, executed)\n"
        "vs anchors retrained at the full-fine-tune rate (5e-6, dose-matched)",
        pad=14,
        fontsize=11,
    )
    savefig_paper(fig, "fu_imp_lr_deconfound_forest", dir=FIGDIR)
    plt.close(fig)


def fig_imp_leakage() -> None:
    lat = _load(R / "panel/leakage_lattice.json")["content"]["imp/con"]
    exec_lat = _load(MAIN / "eval_results/issue_1586/panel/leakage_lattice.json")["content"][
        "imp/con"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), gridspec_kw={"width_ratios": [1, 1.5]})
    rows = []
    for seed in ("s42", "s137"):
        e = exec_lat[seed]
        rows.append(
            (f"factory-rate anchors {seed}", e["delta"], e["ci_low"], e["ci_high"], C[3], "o", True)
        )
    ep = exec_lat["pooled"]
    rows.append(
        ("factory-rate anchors pooled", ep["delta"], ep["ci_low"], ep["ci_high"], C[3], "o", True)
    )
    for seed in ("s42", "s137"):
        e = lat[seed]
        rows.append(
            (f"matched-rate anchors {seed}", e["delta"], e["ci_low"], e["ci_high"], C[0], "o", True)
        )
    pp = lat["pooled"]
    rows.append(
        ("matched-rate anchors pooled", pp["delta"], pp["ci_low"], pp["ci_high"], C[0], "o", True)
    )
    _forest(axes[0], rows, "pooled non-source judged-rate difference\n(full fine-tune minus LoRA)")
    ax = axes[1]
    cells = [
        ("imp-pers-lora5e6-con-s42", "LoRA 5e-6 s42", C[0]),
        ("imp-pers-lora5e6-con-s137", "LoRA 5e-6 s137", C[2]),
        ("imp-pers-ft-con-s42", "full FT s42", C[3]),
        ("imp-pers-ft-con-s137", "full FT s137", C[5]),
    ]
    ctx_order = [
        "persona_software_engineer",
        "default",
        "wildchat_prefix_real545",
        "icl_prefix_impolite",
        "neg_sp_police",
        "neg_sp_ph4",
    ]
    ctx_labels = [
        "source persona",
        "default assistant",
        "WildChat prefix",
        "in-context examples",
        "police officer",
        "maritime medic",
    ]
    x = np.arange(len(ctx_order))
    w = 0.2
    for j, (cell, lab, col) in enumerate(cells):
        s = _load(PANEL_SUMMARIES / cell / "panel_summary.json")["rates_by_context"]
        ax.bar(x + (j - 1.5) * w, [s[c] for c in ctx_order], width=w, color=col, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(ctx_labels, rotation=25, ha="right", fontsize=8.5)
    ax.set_ylabel("judged impolite rate")
    ax.legend(fontsize=8)
    fig.suptitle(
        "Impolite leakage at matched learning rate and matched install (fresh reads, one pod)",
        fontsize=11,
        y=1.02,
    )
    savefig_paper(fig, "fu_imp_leakage", dir=FIGDIR)
    plt.close(fig)


def fig_mk_leakage_forest() -> None:
    lat = _load(R / "panel/leakage_lattice.json")["marker"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, space, xlabel in (
        (axes[0], "margin", "pooled non-source end-of-turn-margin shift difference (logits)"),
        (axes[1], "logp", "pooled non-source log-prob shift difference (nats)"),
    ):
        rows = []
        for regime, col in (("con", C[0]), ("po", C[1])):
            m = lat[regime][space]
            for seed in ("s42", "s137"):
                lo, hi = m["per_seed_ci"][seed]
                mk = "s" if (regime, seed) == ("con", "s137") else "o"
                filled = (regime, seed) != ("con", "s137")
                rows.append(
                    (
                        f"{'contrastive' if regime == 'con' else 'positive-only'} {seed}",
                        m["per_seed"][seed],
                        lo,
                        hi,
                        col,
                        mk,
                        filled,
                    )
                )
            rows.append(
                (
                    f"{'contrastive' if regime == 'con' else 'positive-only'} pooled",
                    m["pooled_delta"],
                    m["pooled_ci"][0],
                    m["pooled_ci"][1],
                    col,
                    "o",
                    True,
                )
            )
        _forest(ax, rows, xlabel)
    axes[1].tick_params(labelleft=False)
    fig.suptitle(
        "Marker leakage at matched install, full fine-tune minus LoRA (both sides fresh)",
        fontsize=11,
        y=1.02,
    )
    savefig_paper(fig, "fu_mk_leakage_forest", dir=FIGDIR)
    plt.close(fig)


def fig_imp_percell_norms() -> None:
    """Low-level per-cell response-arm shift norms behind the deconfound forest."""
    exec_cells = {
        "imp-pers-lora-con-s42": None,
        "imp-pers-lora-con-s137": None,
        "imp-pers-ft-con-s42": None,
        "imp-pers-ft-con-s137": None,
    }
    d = _load(MAIN / "eval_results/issue_1586/geometry/geometry_per_cell.json")["records"]
    for c in exec_cells:
        exec_cells[c] = d[f"{c}/selected/response/L14"]["mu_norm"]
    fu = _load(R / "geometry/geometry_per_cell.json")["records"]
    bars = [
        ("LoRA 3e-5 s42\n(executed)", exec_cells["imp-pers-lora-con-s42"], C[3]),
        ("LoRA 3e-5 s137\n(executed)", exec_cells["imp-pers-lora-con-s137"], C[3]),
        (
            "LoRA 5e-6 s42\n(this round)",
            fu["imp-pers-lora5e6-con-s42/selected/response/L14"]["mu_norm"],
            C[0],
        ),
        (
            "LoRA 5e-6 s137\n(this round)",
            fu["imp-pers-lora5e6-con-s137/selected/response/L14"]["mu_norm"],
            C[0],
        ),
        ("full FT s42\n(executed)", exec_cells["imp-pers-ft-con-s42"], C[5]),
        ("full FT s137\n(executed)", exec_cells["imp-pers-ft-con-s137"], C[5]),
        (
            "full FT s42\n(fresh re-capture)",
            fu["imp-pers-ft-con-s42/selected/response/L14"]["mu_norm"],
            C[2],
        ),
        (
            "full FT s137\n(fresh re-capture)",
            fu["imp-pers-ft-con-s137/selected/response/L14"]["mu_norm"],
            C[2],
        ),
    ]
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    x = np.arange(len(bars))
    ax.bar(x, [b[1] for b in bars], color=[b[2] for b in bars])
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars], fontsize=8)
    ax.set_ylabel("mean-shift norm vs base (layer 14, response arm)")
    ax.set_title(
        "Impolite contrastive per-cell shift norms: both LoRA recipes and both full-FT reads",
        pad=12,
        fontsize=11,
    )
    savefig_paper(fig, "fu_imp_percell_norms", dir=FIGDIR)
    plt.close(fig)


if __name__ == "__main__":
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig_mk_dose_ladder()
    fig_mk_matched_forest()
    fig_imp_deconfound_forest()
    fig_imp_leakage()
    fig_mk_leakage_forest()
    fig_imp_percell_norms()
    print("all figures written to", FIGDIR)
