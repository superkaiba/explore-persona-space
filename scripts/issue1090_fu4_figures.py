"""Figures for issue #1090 fu4/fu5 (round-parametrized via ``--round``).

``--round fu4`` (default; plan v6 §6) reads the fu4 aggregate
(eval_results/issue_1090/fu4-extended-dose-lr/fu4_ladders.json) and writes:

- HERO ``fu4_dose_lr_grid``: 3 panels (one per cell), x = optimizer step
  (5..75), y = primary rate (structural for formatting, judged for impolite),
  one line per lr rung, the registered 0.60-0.85 band shaded, the reused fu3
  base rate dashed; companion tf-margin markers at the selected rungs.
- EXPLORATORY dump under figures/issue_1090/fu4/: per-run margin-by-context
  curves, degeneracy-flag table, Tier-1 (max rung) vs Tier-2 agreement scatter.

``--round fu5`` (plan v7 §6, ``finish-impolite-bare-and-formatting-rank``)
reads eval_results/issue_1090/finish-impolite-bare-and-formatting-rank/
fu5_ladders.json and writes:

- HERO ``fu5_unlock_grid``: 2 panels — (A) impolite bare-context dose ladder,
  one line per lr, the fu4 persona/WildChat unlock rates as reference ticks;
  (B) formatting structural rate vs step, one line per LoRA rank (the rank-32
  line is the REUSED fu4 curve) — band shaded + reused base rates dashed.
- EXPLORATORY dump under figures/issue_1090/fu5/: Tier-2 verdict reads
  (raw + CJK-zeroed bounds), margin-at-selected, degeneracy flags, and the
  list-affordable vs prose-natural eval-split bars (plan D2 item 6).

Every figure uses plain-English condition names (no bare run ids in prose).
Runs VM-side after ``issue1090_fu4.py [--round fu5] --phase judge-aggregate``;
``--ladders`` overrides the input path (smoke figures read the smoke mirror,
never the committed deliverables path).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# Shared-VM thread caps (#847): apply env caps BEFORE matplotlib/numpy import.
load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LADDERS = ROOT / "eval_results" / "issue_1090" / "fu4-extended-dose-lr" / "fu4_ladders.json"
DEFAULT_TEXT_AUDIT = (
    ROOT / "eval_results" / "issue_1090" / "fu4-extended-dose-lr" / "fu4_text_audit.json"
)
FU5_DELIVERABLES = ROOT / "eval_results" / "issue_1090" / "finish-impolite-bare-and-formatting-rank"
FU5_LADDERS = FU5_DELIVERABLES / "fu5_ladders.json"
FU5_TEXT_AUDIT = FU5_DELIVERABLES / "fu5_text_audit.json"
FU7_DELIVERABLES = ROOT / "eval_results" / "issue_1090" / "sycophancy-lr-install-and-remeasure"
FU7_LADDERS = FU7_DELIVERABLES / "fu7_ladders.json"
FU7_TEXT_AUDIT = FU7_DELIVERABLES / "fu7_text_audit.json"
FU6_AGGREGATES = (
    ROOT
    / "eval_results"
    / "issue_1090"
    / "sycophancy-pv-vector-dv-rubric-reanchor"
    / "fu6_aggregates.json"
)
FU2_LADDER_BY_CELL = {
    "syc-c3": ROOT
    / "eval_results"
    / "issue_1090"
    / "fu2-dose-extension"
    / "c3-sycophancy-claude"
    / "fu2_ladder.json",
    "syc-c5": ROOT
    / "eval_results"
    / "issue_1090"
    / "fu2-dose-extension"
    / "c5-sycophancy-qwen"
    / "fu2_ladder.json",
}
FU7_CELL_LABEL = {
    "syc-c3": "sycophancy, Claude-data (c3)",
    "syc-c5": "sycophancy, Qwen-data (c5)",
}
FU7_CELL_ORDER = ("syc-c3", "syc-c5")
FU7_PROJ_LAYER = 22  # fu6 h2_headline.selected_layer (frozen; plan §11)

CELL_LABEL = {
    "fmt-pers": "list formatting, persona-trained (control)",
    "imp-pers": "impolite, persona-trained",
    "imp-conv": "impolite, WildChat-trained",
}
CELL_ORDER = ("fmt-pers", "imp-pers", "imp-conv")
LR_LABEL = {1e-05: "lr 1e-5 (usual)", 3e-05: "lr 3e-5", 0.0001: "lr 1e-4"}

# fu5 (plan v7): Arm A varies lr on the bare context; Arm B varies LoRA rank.
FU5_ARM_LABEL = {
    "imp-bare-lr1e5": "impolite bare, lr 1e-5 (usual)",
    "imp-bare-lr3e5": "impolite bare, lr 3e-5",
    "imp-bare-lr1e4": "impolite bare, lr 1e-4",
    "reused_fu4_r32": "formatting, rank 32 (reused fu4)",
    "fmt-pers-r128": "formatting, rank 128 (4x)",
    "fmt-pers-r256": "formatting, rank 256 (8x)",
}
FU5_ARM_ORDER = tuple(FU5_ARM_LABEL)
RANK_LABEL = {32: "rank 32 (reused fu4)", 128: "rank 128 (4x)", 256: "rank 256 (8x)"}
FU4_IMPOLITE_REF_LABEL = {
    "imp-pers": "fu4 persona unlock",
    "imp-conv": "fu4 WildChat unlock",
}


def _lr_colors() -> dict[float, str]:
    pal = paper_palette_blog(3)
    return dict(zip(sorted(LR_LABEL), pal, strict=True))


def _rank_colors() -> dict[int, str]:
    pal = paper_palette_blog(3)
    return dict(zip(sorted(RANK_LABEL), pal, strict=True))


def _fu5_arm_color(run: dict) -> str:
    """Line/marker color per fu5 arm: lr-keyed for the impolite lr sweep,
    rank-keyed for the formatting rank ladder."""
    if run.get("cell_key") == "fmt-pers":
        return _rank_colors()[int(run["lora_r"])]
    return _lr_colors()[run["lr"]]


def fig_dose_lr_grid(agg: dict, out_prefix: str, out_dir: str) -> None:
    """HERO: per-cell dose ladders, one line per lr, band + base overlays."""
    colors = _lr_colors()
    band = agg.get("band", [0.60, 0.85])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
    for ax, cell_key in zip(axes, CELL_ORDER, strict=True):
        runs = [r for r in agg["runs"].values() if r["cell_key"] == cell_key]
        if not runs:
            ax.set_title(f"{CELL_LABEL[cell_key]}\nN/A — not run")
            continue
        ax.axhspan(band[0], band[1], color="0.9", zorder=0)
        base = (runs[0].get("base_tier2") or {}).get("rate")
        if base is not None:
            ax.axhline(base, ls="--", color="0.4", lw=1.2, label="base rate (reused fu3 read)")
        for r in sorted(runs, key=lambda x: x["lr"]):
            rates = r.get("rates_by_step") or {}
            if not rates:
                continue
            steps = sorted(int(s) for s in rates)
            ys = [rates[str(s)] for s in steps]
            ax.plot(
                steps,
                ys,
                marker="o",
                ms=3.5,
                lw=1.6,
                color=colors[r["lr"]],
                label=LR_LABEL[r["lr"]] + (" — DIVERGED" if r["status"] == "diverged" else ""),
            )
            sel = r.get("selection")
            t2 = r.get("tier2_trained")
            if sel is not None and t2 is not None:
                ax.scatter(
                    [sel["step"]],
                    [t2["rate"]],
                    marker="D",
                    s=42,
                    color=colors[r["lr"]],
                    edgecolor="black",
                    lw=0.6,
                    zorder=5,
                )
        ax.set_title(CELL_LABEL[cell_key])
        ax.set_xlabel("optimizer step (5 steps = 1 epoch)")
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("primary rate (structural / judged)")
    handles, labels = axes[1].get_legend_handles_labels()
    if not labels:
        handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    savefig_paper(fig, f"{out_prefix}/fu4_dose_lr_grid", dir=out_dir)
    plt.close(fig)


def fig_margin_at_selected(agg: dict, out_prefix: str, out_dir: str) -> None:
    """Exploratory: per-run tf-margin (trained vs base means) at the selected
    rung — the install-without-expression companion read."""
    rows = []
    for r in agg["runs"].values():
        m = r.get("margin") or {}
        if m.get("status") != "computed":
            continue
        rows.append((r["cell_key"], r["lr"], m.get("margin_base"), m.get("margin_trained")))
    if not rows:
        return
    colors = _lr_colors()
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    xticks, xlabels = [], []
    x = 0
    for cell_key in CELL_ORDER:
        cell_rows = [r for r in rows if r[0] == cell_key]
        for _ck, lr, mb, mt in sorted(cell_rows, key=lambda t: t[1]):
            if mb is not None:
                ax.scatter([x], [mb], marker="s", color="0.5", s=30)
            if mt is not None:
                ax.scatter([x], [mt], marker="o", color=colors[lr], s=40)
            xticks.append(x)
            cell_short = CELL_LABEL[_ck].replace(" (control)", "").replace(", ", ",\n")
            xlabels.append(f"{cell_short}\n{LR_LABEL[lr].split()[1]}")
            x += 1
        x += 1
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.set_xticks(xticks, xlabels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("tf-margin (mean over eval contexts)")
    ax.set_title("fixed-pool margin at the selected rung (circles trained, squares base)")
    fig.tight_layout()
    savefig_paper(fig, f"{out_prefix}/fu4_margin_at_selected", dir=out_dir)
    plt.close(fig)


def fig_degeneracy_flags(agg: dict, out_prefix: str, out_dir: str) -> None:
    """Exploratory: per-run count of degeneracy-flagged rungs (flag-only guard)."""
    labels, counts = [], []
    for r in sorted(agg["runs"].values(), key=lambda x: (x["cell_key"], x["lr"])):
        flags = [d for d in (r.get("degeneracy_by_step") or {}).values() if d.get("degenerate")]
        labels.append(f"{r['cell_key']} {LR_LABEL[r['lr']].split()[1]}")
        counts.append(len(flags))
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.bar(range(len(labels)), counts, color=paper_palette_blog(1)[0])
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("rungs flagged degenerate")
    ax.set_title("degenerate-output guard flags per run (length/4-gram; flag-only)")
    fig.tight_layout()
    savefig_paper(fig, f"{out_prefix}/fu4_degeneracy_flags", dir=out_dir)
    plt.close(fig)


def fig_tier2_verdict(agg: dict, out_prefix: str, out_dir: str, text_audit: dict | None) -> None:
    """Per-arm Tier-2 confirmatory rate (Wilson 95% CI) with the registered
    0.60 band floor + 0.30 cut, reused base rates, and — for judged impolite
    arms — the worst-case judge-drop bound (every dropped completion scored
    non-impolite: k/200) plus, when ``text_audit`` (fu4_text_audit.json) is
    given, the CJK-intrusion-zeroed bound ((k - cjk_firing)/200, x markers)."""
    colors = _lr_colors()
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    xticks, xlabels = [], []
    x = 0
    for cell_key in CELL_ORDER:
        cell_runs = sorted(
            (r for r in agg["runs"].values() if r["cell_key"] == cell_key),
            key=lambda r: r["lr"],
        )
        for r in cell_runs:
            t2 = r.get("tier2_trained") or {}
            rate, lo_hi = t2.get("rate"), t2.get("wilson95") or (None, None)
            if rate is None:
                continue
            ax.errorbar(
                [x],
                [rate],
                yerr=[[rate - lo_hi[0]], [lo_hi[1] - rate]],
                fmt="o",
                color=colors[r["lr"]],
                capsize=3,
                ms=6,
            )
            ax.text(x + 0.12, rate, f"{rate:.2f}", fontsize=7, va="center")
            if t2.get("mode") == "judged" and t2.get("n", 200) < 200:
                wc = t2["k"] / 200.0
                ax.scatter(
                    [x],
                    [wc],
                    marker="v",
                    facecolors="none",
                    edgecolors=colors[r["lr"]],
                    s=45,
                    linewidths=1.2,
                )
            arm_audit = (text_audit or {}).get("arms", {}).get(r["run_id"])
            if t2.get("mode") == "judged" and arm_audit is not None:
                ax.scatter(
                    [x],
                    [arm_audit["cjk_zeroed_rate"]],
                    marker="x",
                    color=colors[r["lr"]],
                    s=45,
                    linewidths=1.4,
                )
            base = (r.get("base_tier2") or {}).get("rate")
            if base is not None:
                ax.scatter([x], [base], marker="s", color="0.5", s=28, zorder=1)
            cell_short = CELL_LABEL[cell_key].replace(" (control)", "").replace(", ", ",\n")
            xlabels.append(f"{cell_short}\n{LR_LABEL[r['lr']].split()[1]}")
            xticks.append(x)
            x += 1
        x += 1
    ax.axhline(0.60, color="0.4", lw=0.9, ls="--")
    ax.axhline(0.30, color="0.7", lw=0.9, ls=":")
    ax.set_xticks(xticks, xlabels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Tier-2 confirmatory rate")
    ax.set_ylim(-0.03, 1.0)
    ax.set_title(
        "Tier-2 verdict reads (circles trained + Wilson 95% CI; squares base;\n"
        "open triangles worst-case judge-drop bound; x CJK-intrusion-zeroed bound;\n"
        "dashes 0.60 band floor, dots 0.30 cut)"
    )
    fig.tight_layout()
    savefig_paper(fig, f"{out_prefix}/fu4_tier2_verdict", dir=out_dir)
    plt.close(fig)


# ── fu5 figures (plan v7 §6) ─────────────────────────────────────────────────


def _fu5_arms(agg: dict) -> list[dict]:
    """fu5 aggregate entries in the registered arm order (missing arms skipped;
    the reused r32 entry is a first-class arm)."""
    return [agg["runs"][rid] for rid in FU5_ARM_ORDER if rid in agg.get("runs", {})]


def fig_fu5_unlock_grid(agg: dict, fu4_agg: dict | None, out_prefix: str, out_dir: str) -> None:
    """HERO: (A) impolite bare-context dose ladder, one line per lr, fu4
    persona/WildChat unlock rates as right-edge reference ticks; (B) formatting
    structural rate vs step, one line per LoRA rank (r32 = the reused fu4
    curve). Band shaded; reused base rates dashed; Tier-2 confirmatory reads
    at the selected rungs as black-edged diamonds."""
    band = agg.get("band", [0.60, 0.85])
    panels = (
        ("imp-bare", "impolite, bare-context-trained (judged rate)"),
        ("fmt-pers", "list formatting, persona-trained (structural rate, rank ladder)"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, (cell_key, title) in zip(axes, panels, strict=True):
        runs = [r for r in _fu5_arms(agg) if r.get("cell_key") == cell_key]
        if not runs:
            ax.set_title(f"{title}\nN/A — not run")
            continue
        ax.axhspan(band[0], band[1], color="0.9", zorder=0)
        base = next(
            ((r.get("base_tier2") or {}).get("rate") for r in runs if r.get("base_tier2")),
            None,
        )
        if base is not None:
            ax.axhline(base, ls="--", color="0.4", lw=1.2, label="base rate (reused fu3 read)")
        for r in runs:
            rates = r.get("rates_by_step") or {}
            if not rates:
                continue
            steps = sorted(int(s) for s in rates)
            ys = [rates[str(s)] for s in steps]
            label = FU5_ARM_LABEL[r["run_id"]].split(", ", 1)[1]
            ax.plot(
                steps,
                ys,
                marker="o",
                ms=3.5,
                lw=1.6,
                color=_fu5_arm_color(r),
                label=label + (" — DIVERGED" if r.get("status") == "diverged" else ""),
            )
            sel, t2 = r.get("selection"), r.get("tier2_trained")
            if sel is not None and t2 is not None:
                ax.scatter(
                    [sel["step"]],
                    [t2["rate"]],
                    marker="D",
                    s=42,
                    color=_fu5_arm_color(r),
                    edgecolor="black",
                    lw=0.6,
                    zorder=5,
                )
        if cell_key == "imp-bare" and fu4_agg is not None:
            for ref_cell, ref_label in FU4_IMPOLITE_REF_LABEL.items():
                best = max(
                    (
                        (r.get("tier2_trained") or {}).get("rate")
                        for r in fu4_agg["runs"].values()
                        if r.get("cell_key") == ref_cell and r.get("tier2_trained")
                    ),
                    default=None,
                )
                if best is None:
                    continue
                ax.plot([73, 77], [best, best], color="0.25", lw=1.4, clip_on=False)
                ax.annotate(
                    f"{ref_label} {best:.2f}",
                    xy=(77, best),
                    fontsize=6.5,
                    va="center",
                    ha="left",
                    color="0.25",
                    annotation_clip=False,
                )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("optimizer step (5 steps = 1 epoch)")
        ax.set_ylim(-0.03, 1.03)
        ax.legend(fontsize=7, loc="upper left", frameon=False)
    axes[0].set_ylabel("primary rate")
    fig.tight_layout()
    savefig_paper(fig, f"{out_prefix}/fu5_unlock_grid", dir=out_dir)
    plt.close(fig)


def fig_fu5_tier2_verdict(
    agg: dict, out_prefix: str, out_dir: str, text_audit: dict | None
) -> None:
    """Exploratory: per-arm Tier-2 confirmatory rate (Wilson 95% CI) with the
    0.60 band floor + 0.30 cut, reused base squares, worst-case judge-drop
    bounds (open triangles), the CJK-intrusion-zeroed bound (x markers, from
    fu5_text_audit.json — now ALL six arms) and, for the formatting rank arms,
    the CJK-intrusion-EXCLUDED recount (open diamonds; interp-critique fu5 r1
    request 1 — the r256 excluded recount crosses the 0.30 cut)."""
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    xticks, xlabels = [], []
    x = 0
    prev_cell = None
    for r in _fu5_arms(agg):
        t2 = r.get("tier2_trained") or {}
        rate, lo_hi = t2.get("rate"), t2.get("wilson95") or (None, None)
        if rate is None:
            continue
        if prev_cell is not None and r["cell_key"] != prev_cell:
            x += 1  # gap between the two arms' groups
        prev_cell = r["cell_key"]
        color = _fu5_arm_color(r)
        ax.errorbar(
            [x],
            [rate],
            yerr=[[rate - lo_hi[0]], [lo_hi[1] - rate]],
            fmt="o",
            color=color,
            capsize=3,
            ms=6,
        )
        ax.text(x + 0.12, rate, f"{rate:.2f}", fontsize=7, va="center")
        if t2.get("mode") == "judged" and t2.get("n", 200) < 200:
            ax.scatter(
                [x],
                [t2["k"] / 200.0],
                marker="v",
                facecolors="none",
                edgecolors=color,
                s=45,
                linewidths=1.2,
            )
        arm_audit = (text_audit or {}).get("arms", {}).get(r["run_id"])
        if arm_audit is not None and "cjk_zeroed_rate" in arm_audit:
            ax.scatter(
                [x],
                [arm_audit["cjk_zeroed_rate"]],
                marker="x",
                color=color,
                s=45,
                linewidths=1.4,
            )
        if arm_audit is not None and "cjk_excluded_rate" in arm_audit:
            ax.scatter(
                [x],
                [arm_audit["cjk_excluded_rate"]],
                marker="D",
                facecolors="none",
                edgecolors=color,
                s=40,
                linewidths=1.2,
            )
        base = (r.get("base_tier2") or {}).get("rate")
        if base is not None:
            ax.scatter([x], [base], marker="s", color="0.5", s=28, zorder=1)
        xlabels.append(FU5_ARM_LABEL[r["run_id"]].replace(", ", ",\n"))
        xticks.append(x)
        x += 1
    ax.axhline(0.60, color="0.4", lw=0.9, ls="--")
    ax.axhline(0.30, color="0.7", lw=0.9, ls=":")
    ax.set_xticks(xticks, xlabels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Tier-2 confirmatory rate")
    ax.set_ylim(-0.03, 1.0)
    ax.set_title(
        "Tier-2 verdict reads (circles trained + Wilson 95% CI; squares base;\n"
        "open triangles worst-case judge-drop bound; x CJK-intrusion-zeroed bound;\n"
        "open diamonds CJK-intrusion-excluded recount (formatting arms);\n"
        "dashes 0.60 band floor, dots 0.30 cut)"
    )
    fig.tight_layout()
    savefig_paper(fig, f"{out_prefix}/fu5_tier2_verdict", dir=out_dir)
    plt.close(fig)


FU5_FMT_AUDIT_ARMS = ("reused_fu4_r32", "fmt-pers-r128", "fmt-pers-r256", "fu3-base-formatting")
FU5_FMT_AUDIT_LABEL = {
    "reused_fu4_r32": "rank 32 (reused)",
    "fmt-pers-r128": "rank 128",
    "fmt-pers-r256": "rank 256",
    "fu3-base-formatting": "base (reused fu3)",
}


def fig_fu5_fmt_intrusion(text_audit: dict, out_prefix: str, out_dir: str) -> None:
    """Formatting-arm CJK-intrusion audit (fu5 interp-critique r1 request 1).

    Panel A: per-arm raw structural Tier-2 rate (circle), CJK-zeroed bound (x)
    and CJK-excluded recount (open diamond + Wilson 95% CI) against the 0.30
    cut. Panel B: the per-arm counts behind the aggregate — English-question
    completions split intruded-and-firing / intruded-non-firing / non-intruded.
    """
    arms = {a: text_audit["arms"][a] for a in FU5_FMT_AUDIT_ARMS if a in text_audit["arms"]}
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(9.2, 4.0), width_ratios=[1.15, 1.0])
    for x, a in enumerate(arms.values()):
        color = "0.45" if a["kind"].startswith("fu3-base") else "#2d7f5e"
        raw = a["ladders_rate"]
        ax.scatter([x], [raw], marker="o", color=color, s=42, zorder=3)
        ax.text(x + 0.1, raw, f"{raw:.2f}", fontsize=7, va="bottom")
        ax.scatter([x], [a["cjk_zeroed_rate"]], marker="x", color=color, s=45, linewidths=1.4)
        ax.text(x + 0.1, a["cjk_zeroed_rate"], f"{a['cjk_zeroed_rate']:.2f}", fontsize=7, va="top")
        exc, (lo, hi) = a["cjk_excluded_rate"], a["cjk_excluded_wilson95"]
        ax.errorbar([x], [exc], yerr=[[exc - lo], [hi - exc]], fmt="none", ecolor=color, capsize=3)
        ax.scatter(
            [x], [exc], marker="D", facecolors="none", edgecolors=color, s=42, linewidths=1.3
        )
        ax.text(x - 0.12, exc, f"{exc:.3f}", fontsize=7, va="center", ha="right")
    ax.axhline(0.30, color="0.6", lw=0.9, ls=":")
    ax.set_xticks(range(len(arms)), [FU5_FMT_AUDIT_LABEL[a] for a in arms], fontsize=8)
    ax.set_ylabel("Structural Tier-2 rate")
    ax.set_ylim(0.0, 0.45)
    ax.set_title(
        "Rate under the three intrusion dispositions\n(circle raw; x zeroed;\n"
        "diamond excluded + Wilson 95% CI; dots 0.30 cut)",
        fontsize=9,
    )
    xs = range(len(arms))
    firing = [a["n_cjk_firing"] for a in arms.values()]
    nonfiring = [a["n_cjk"] - a["n_cjk_firing"] for a in arms.values()]
    clean = [a["n_english_rows"] - a["n_cjk"] for a in arms.values()]
    axb.bar(xs, firing, color="#b0413e", label="intruded, predicate fires")
    axb.bar(xs, nonfiring, bottom=firing, color="#e0a458", label="intruded, no fire")
    axb.bar(
        xs,
        clean,
        bottom=[f + nf for f, nf in zip(firing, nonfiring, strict=True)],
        color="0.85",
        label="not intruded",
    )
    for x, a in enumerate(arms.values()):
        axb.text(x, a["n_cjk"] + 6, str(a["n_cjk"]), ha="center", fontsize=7)
    axb.set_xticks(range(len(arms)), [FU5_FMT_AUDIT_LABEL[a] for a in arms], fontsize=8)
    axb.set_ylabel("English-question completions (of 290)")
    axb.set_ylim(0, 420)
    axb.set_title("Per-arm intrusion counts (bar label: intruded rows)")
    axb.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    savefig_paper(fig, f"{out_prefix}/fu5_fmt_intrusion", dir=out_dir)
    plt.close(fig)


def fig_fu5_margin_at_selected(agg: dict, out_prefix: str, out_dir: str) -> None:
    """Exploratory: per-arm tf-margin (trained circles vs base squares) at the
    selected rung — the representational-install companion read."""
    rows = []
    for r in _fu5_arms(agg):
        m = r.get("margin") or {}
        if m.get("status") != "computed":
            continue
        rows.append((r, m.get("margin_base"), m.get("margin_trained")))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    xticks, xlabels = [], []
    for x, (r, mb, mt) in enumerate(rows):
        if mb is not None:
            ax.scatter([x], [mb], marker="s", color="0.5", s=30)
        if mt is not None:
            ax.scatter([x], [mt], marker="o", color=_fu5_arm_color(r), s=40)
        xticks.append(x)
        xlabels.append(FU5_ARM_LABEL[r["run_id"]].replace(", ", ",\n"))
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.set_xticks(xticks, xlabels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("tf-margin (mean over eval contexts)")
    ax.set_title("fixed-pool margin at the selected rung (circles trained, squares base)")
    fig.tight_layout()
    savefig_paper(fig, f"{out_prefix}/fu5_margin_at_selected", dir=out_dir)
    plt.close(fig)


def fig_fu5_degeneracy_flags(agg: dict, out_prefix: str, out_dir: str) -> None:
    """Exploratory: per-arm count of degeneracy-flagged rungs (flag-only guard;
    extra force on the r256 + lr 1e-4 arms per plan §8)."""
    labels, counts = [], []
    for r in _fu5_arms(agg):
        if "degeneracy_by_step" not in r:
            continue
        flags = [d for d in (r.get("degeneracy_by_step") or {}).values() if d.get("degenerate")]
        labels.append(FU5_ARM_LABEL[r["run_id"]])
        counts.append(len(flags))
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.bar(range(len(labels)), counts, color=paper_palette_blog(1)[0])
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("rungs flagged degenerate")
    ax.set_title("degenerate-output guard flags per run (length/4-gram; flag-only)")
    fig.tight_layout()
    savefig_paper(fig, f"{out_prefix}/fu5_degeneracy_flags", dir=out_dir)
    plt.close(fig)


def fig_fu5_eval_split(agg: dict, out_prefix: str, out_dir: str) -> None:
    """Exploratory (plan D2 item 6): Tier-2 structural rate per eval-question
    split (list-affordable vs prose-natural) for every formatting arm + the
    fu3 base arm — the 'installing but the eval can't show it' diagnostic."""
    diag = agg.get("eval_split_diagnostic")
    if not diag:
        return
    per_arm = diag.get("per_arm") or {}
    arm_ids = [a for a in (*FU5_ARM_ORDER, "fu3_base") if a in per_arm]
    rows = [(a, per_arm[a]) for a in arm_ids if "list_affordable" in per_arm[a]]
    if not rows:
        return
    split_colors = dict(
        zip(("list_affordable", "prose_natural"), paper_palette_blog(2), strict=True)
    )
    split_label = {"list_affordable": "list-affordable", "prose_natural": "prose-natural"}
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    width = 0.38
    for i, (_arm, rec) in enumerate(rows):
        for j, split in enumerate(("list_affordable", "prose_natural")):
            s = rec.get(split) or {}
            rate, ci = s.get("rate"), s.get("wilson95")
            if rate is None:
                continue
            xpos = i + (j - 0.5) * width
            yerr = [[rate - ci[0]], [ci[1] - rate]] if ci else None
            ax.bar(
                xpos,
                rate,
                width=width,
                color=split_colors[split],
                yerr=yerr,
                capsize=3,
                label=split_label[split] if i == 0 else None,
            )
            ax.text(xpos, rate + 0.02, f"{rate:.2f}", fontsize=6.5, ha="center")
    ax.set_xticks(
        range(len(rows)),
        [
            FU5_ARM_LABEL.get(a, "base model (reused fu3 read)").replace(", ", ",\n")
            for a, _ in rows
        ],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.set_ylabel("Tier-2 structural rate")
    ax.set_ylim(0, 1.0)
    ax.set_title(
        "structural rate by eval-question split (Wilson 95% CI;\n"
        "Sonnet 3-draw-majority classification of the 30 WildChat slices)"
    )
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    savefig_paper(fig, f"{out_prefix}/fu5_eval_split", dir=out_dir)
    plt.close(fig)


# ── fu7 figures (plan v13 §6) ────────────────────────────────────────────────


def _fu7_cell_runs(agg: dict, cell_key: str) -> list[dict]:
    return sorted(
        (r for r in agg["runs"].values() if r.get("cell_key") == cell_key),
        key=lambda r: r["lr"],
    )


def fig_fu7_unlock_grid(agg: dict, out_prefix: str, out_dir: str) -> None:
    """HERO (plan §6): 2 panels (C3, C5) — Tier-1 legacy rate vs optimizer
    step per lr arm, 0.60-0.85 band shaded, fu2's committed 30-step ladder
    overlaid dashed (the dose-only reference), reused base rate dashed,
    selected rungs marked, Tier-2 confirmatory reads as filled points with
    Wilson bars."""
    colors = _lr_colors()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    for ax, cell_key in zip(axes, FU7_CELL_ORDER, strict=True):
        ax.axhspan(0.60, 0.85, color="0.92", zorder=0)
        fu2_path = FU2_LADDER_BY_CELL[cell_key]
        if fu2_path.exists():
            fu2 = json.loads(fu2_path.read_text())
            steps = sorted(int(s) for s in fu2["rates_by_step"])
            ax.plot(
                steps,
                [fu2["rates_by_step"][str(s)] for s in steps],
                ls="--",
                color="0.55",
                lw=1.1,
                label="fu2 dose-only ladder (epochs 6)",
            )
        for r in _fu7_cell_runs(agg, cell_key):
            rbs = r.get("rates_by_step") or {}
            if not rbs:
                continue
            steps = sorted(int(s) for s in rbs)
            ax.plot(
                steps,
                [rbs[str(s)] for s in steps],
                marker="o",
                ms=3,
                color=colors[r["lr"]],
                label=LR_LABEL[r["lr"]],
            )
            sel = r.get("selection") or {}
            if sel.get("step") is not None:
                ax.scatter(
                    [sel["step"]],
                    [sel["rate"]],
                    marker="D",
                    s=55,
                    facecolors="none",
                    edgecolors=colors[r["lr"]],
                    zorder=5,
                )
            t2 = r.get("tier2_trained") or {}
            if t2.get("rate") is not None and sel.get("step") is not None:
                lo_hi = t2.get("wilson95") or (t2["rate"], t2["rate"])
                ax.errorbar(
                    [sel["step"]],
                    [t2["rate"]],
                    yerr=[
                        [max(0.0, t2["rate"] - lo_hi[0])],
                        [max(0.0, lo_hi[1] - t2["rate"])],
                    ],
                    fmt="s",
                    ms=6,
                    color=colors[r["lr"]],
                    capsize=3,
                    zorder=6,
                )
            base = (r.get("base_tier2") or {}).get("rate")
            if base is not None:
                ax.axhline(base, color="0.75", lw=0.8, ls=":")
        ax.set_title(FU7_CELL_LABEL[cell_key])
        ax.set_xlabel("optimizer step")
        ax.set_ylim(-0.03, 1.0)
    axes[0].set_ylabel("Tier-1 legacy judged rate")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(
        "fu7 lr sweep: does the impolite lr-unlock transfer to sycophancy?\n"
        "(band 0.60-0.85 shaded; diamonds selected rungs; squares Tier-2 confirm "
        "+ Wilson 95%; dashed grey fu2 dose-only; dotted grey reused base)",
        fontsize=9,
    )
    fig.tight_layout()
    savefig_paper(fig, f"{out_prefix}/fu7_lr_unlock_grid", dir=out_dir)
    plt.close(fig)


def fig_fu7_dual_rubric(agg: dict, out_prefix: str, out_dir: str, text_audit: dict | None) -> None:
    """Paired legacy-vs-paper Tier-2 bars per arm (the H3 instrument offset) +
    raw-vs-CJK-zeroed markers + the reused base rates under both instruments."""
    colors = _lr_colors()
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    xticks, xlabels = [], []
    x = 0
    for cell_key in FU7_CELL_ORDER:
        for r in _fu7_cell_runs(agg, cell_key):
            t2 = r.get("tier2_trained") or {}
            pv = r.get("tier2_trained_pv") or {}
            if t2.get("rate") is None:
                continue
            ax.bar([x - 0.2], [t2["rate"]], width=0.38, color=colors[r["lr"]], alpha=0.95)
            if pv.get("rate") is not None:
                ax.bar(
                    [x + 0.2],
                    [pv["rate"]],
                    width=0.38,
                    color=colors[r["lr"]],
                    alpha=0.45,
                    hatch="//",
                )
            arm_audit = (text_audit or {}).get("arms", {}).get(r["run_id"])
            if arm_audit is not None and arm_audit.get("cjk_zeroed_rate") is not None:
                ax.scatter(
                    [x - 0.2],
                    [arm_audit["cjk_zeroed_rate"]],
                    marker="x",
                    color="k",
                    s=40,
                    zorder=5,
                )
            base = (r.get("base_tier2") or {}).get("rate")
            if base is not None:
                ax.scatter([x - 0.2], [base], marker="s", color="0.4", s=22, zorder=4)
            xticks.append(x)
            xlabels.append(f"{cell_key}\n{LR_LABEL[r['lr']].split()[1]}")
            x += 1
        x += 1
    ax.axhline(0.60, color="0.4", lw=0.9, ls="--")
    ax.set_xticks(xticks, xlabels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Tier-2 rate at the selected rung")
    ax.set_ylim(0, 1.0)
    ax.set_title(
        "paired dual-rubric Tier-2 reads (solid legacy / hatched paper rubric;\n"
        "x CJK-zeroed bound; squares reused legacy base; dashes 0.60 band floor)"
    )
    fig.tight_layout()
    savefig_paper(fig, f"{out_prefix}/fu7_tier2_dual_rubric", dir=out_dir)
    plt.close(fig)


def fig_fu7_rubric_offset_scatter(agg: dict, out_prefix: str, out_dir: str) -> None:
    """Paper-rubric vs legacy Tier-2 rate per arm, with the fu6 ~-0.15
    rank-preserving offset reference (H3; fu6 paired_old_vs_new rho=0.95)."""
    colors = _lr_colors()
    pts = []
    for cell_key in FU7_CELL_ORDER:
        for r in _fu7_cell_runs(agg, cell_key):
            t2 = (r.get("tier2_trained") or {}).get("rate")
            pv = (r.get("tier2_trained_pv") or {}).get("rate")
            if t2 is not None and pv is not None:
                pts.append((t2, pv, r))
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for t2, pv, r in pts:
        ax.scatter(
            [t2],
            [pv],
            color=colors[r["lr"]],
            marker="o" if r["cell_key"] == "syc-c3" else "^",
            s=48,
        )
        ax.annotate(r["run_id"], (t2, pv), fontsize=6, xytext=(3, 3), textcoords="offset points")
    lo = 0.0
    hi = 1.0
    ax.plot([lo, hi], [lo, hi], color="0.8", lw=0.8)
    ax.plot([lo, hi], [lo - 0.15, hi - 0.15], color="0.5", lw=0.9, ls="--")
    ax.set_xlabel("legacy Tier-2 rate")
    ax.set_ylabel("paper-rubric Tier-2 rate")
    ax.set_title(
        "rubric re-anchor offset (dashed: fu6's ~-0.15 rank-preserving offset,\n"
        "paired rho=0.95; circles C3, triangles C5)",
        fontsize=9,
    )
    fig.tight_layout()
    savefig_paper(fig, f"{out_prefix}/fu7_rubric_offset_scatter", dir=out_dir)
    plt.close(fig)


def fig_fu7_panel_leakage(agg: dict, out_prefix: str, out_dir: str) -> None:
    """Panel judged reads per (best organism x context x rubric), base rates
    marked (raw reads; the install delta is named beside each — no
    cross-condition leakage headline, plan §6)."""
    panel = ((agg.get("remeasure") or {}).get("panel")) or {}
    if not panel:
        return
    fig, axes = plt.subplots(1, len(panel), figsize=(6.0 * len(panel), 4.2), squeeze=False)
    for ax, (cell_key, rec) in zip(axes[0], sorted(panel.items()), strict=True):
        ctxs = sorted((rec.get("contexts") or {}).items())
        xs = range(len(ctxs))
        for i, (_ctx_id, reads) in enumerate(ctxs):
            leg = (reads.get("legacy") or {}).get("rate")
            pv = (reads.get("pv") or {}).get("rate")
            if leg is not None:
                ax.bar([i - 0.2], [leg], width=0.38, color="C0")
            if pv is not None:
                ax.bar([i + 0.2], [pv], width=0.38, color="C0", alpha=0.45, hatch="//")
            lb = (reads.get("legacy_base") or {}).get("rate")
            pb = (reads.get("pv_base") or {}).get("rate")
            if lb is not None:
                ax.scatter([i - 0.2], [lb], marker="s", color="0.3", s=24, zorder=4)
            if pb is not None:
                ax.scatter([i + 0.2], [pb], marker="s", color="0.3", s=24, zorder=4)
        ax.set_xticks(list(xs), [c for c, _ in ctxs], rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{cell_key} (best arm {rec.get('run_id')})", fontsize=9)
        ax.set_ylabel("panel judged rate (n=100)")
    fig.suptitle(
        "fu7 panel leakage reads (solid legacy / hatched paper rubric; squares = "
        "reused committed base per context)",
        fontsize=9,
    )
    fig.tight_layout()
    savefig_paper(fig, f"{out_prefix}/fu7_panel_leakage", dir=out_dir)
    plt.close(fig)


def fig_fu7_projection_scatter(
    agg: dict, fu6_agg: dict | None, out_prefix: str, out_dir: str
) -> None:
    """r_B-projection (frozen layer 22) vs paper-rubric judged delta, prefix +
    context arms; fu6's 62 validation cells greyed underneath. DIAGNOSTIC
    (fu6 verdict `Contradicted`) — no verdict rests on this read."""
    proj = ((agg.get("remeasure") or {}).get("projection")) or {}
    cells = proj.get("cells") or []
    if not cells:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))
    for ax, arm in zip(axes, ("prefix", "context"), strict=True):
        if fu6_agg is not None:
            fu6_cells = ((fu6_agg.get("h2_arms") or {}).get(arm) or {}).get("cells") or []
            ax.scatter(
                [c["proj_selected_layer"] for c in fu6_cells],
                [c["delta"] for c in fu6_cells],
                color="0.8",
                s=18,
                label="fu6 validation cells (n=62)",
            )
        pts = [c for c in cells if c.get("pv_delta") is not None]
        key = f"proj_{arm}_layer{FU7_PROJ_LAYER}"
        ax.scatter(
            [c[key] for c in pts],
            [c["pv_delta"] for c in pts],
            color="C1",
            s=42,
            label="fu7 cells",
        )
        sp = (proj.get("spearman") or {}).get(arm) or {}
        rho = sp.get("rho")
        ax.set_title(
            f"{arm} arm — Spearman rho={rho:.2f} (n={sp.get('n_cells')})"
            if rho is not None
            else f"{arm} arm",
            fontsize=9,
        )
        ax.set_xlabel(f"trained-base shift . r_B unit (layer {FU7_PROJ_LAYER})")
        ax.set_ylabel("paper-rubric judged delta")
        ax.legend(fontsize=7)
    fig.suptitle(
        "fu6 r_B projection DIAGNOSTIC (failed fu6 validation: Contradicted, "
        "rho*=-0.456 non-specific — no verdict rests on this read)",
        fontsize=9,
    )
    fig.tight_layout()
    savefig_paper(fig, f"{out_prefix}/fu7_projection_scatter", dir=out_dir)
    plt.close(fig)


def fig_fu7_margin_at_selected(agg: dict, out_prefix: str, out_dir: str) -> None:
    """Exploratory: per-arm tf-margin (trained vs base) at the selected rung."""
    colors = _lr_colors()
    rows = []
    for cell_key in FU7_CELL_ORDER:
        for r in _fu7_cell_runs(agg, cell_key):
            m = r.get("margin") or {}
            if m.get("status") != "computed":
                continue
            rows.append((cell_key, r["lr"], m.get("margin_base"), m.get("margin_trained")))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    xticks, xlabels = [], []
    for x, (ck, lr, mb, mt) in enumerate(rows):
        if mb is not None:
            ax.scatter([x], [mb], marker="s", color="0.5", s=30)
        if mt is not None:
            ax.scatter([x], [mt], marker="o", color=colors[lr], s=40)
        xticks.append(x)
        xlabels.append(f"{ck}\n{LR_LABEL[lr].split()[1]}")
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.set_xticks(xticks, xlabels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("tf-margin (mean over eval contexts)")
    ax.set_title("fixed-pool margin at the selected rung (circles trained, squares base)")
    fig.tight_layout()
    savefig_paper(fig, f"{out_prefix}/fu7_margin_at_selected", dir=out_dir)
    plt.close(fig)


def fig_fu7_degeneracy_flags(agg: dict, out_prefix: str, out_dir: str) -> None:
    """Exploratory: per-run degeneracy-flagged rung counts (flag-only guard)."""
    labels, counts = [], []
    for cell_key in FU7_CELL_ORDER:
        for r in _fu7_cell_runs(agg, cell_key):
            flags = [d for d in (r.get("degeneracy_by_step") or {}).values() if d.get("degenerate")]
            labels.append(f"{cell_key} {LR_LABEL[r['lr']].split()[1]}")
            counts.append(len(flags))
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    ax.bar(range(len(labels)), counts, color=paper_palette_blog(1)[0])
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("rungs flagged degenerate")
    ax.set_title("degenerate-output guard flags per run (length/4-gram; flag-only)")
    fig.tight_layout()
    savefig_paper(fig, f"{out_prefix}/fu7_degeneracy_flags", dir=out_dir)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="#1090 fu4/fu5/fu7 figures")
    ap.add_argument("--round", choices=("fu4", "fu5", "fu7"), default="fu4")
    ap.add_argument("--ladders", default=None, help="default: the round's committed aggregate")
    ap.add_argument("--text-audit", default=None, help="default: the round's committed audit")
    ap.add_argument(
        "--fu4-ladders", default=str(DEFAULT_LADDERS), help="fu5 panel-A reference ticks"
    )
    ap.add_argument("--out-prefix", default=None, help="default: issue_1090/<round>")
    ap.add_argument("--out-dir", default="figures/")
    args = ap.parse_args()
    set_paper_style()
    fu5 = args.round == "fu5"
    fu7 = args.round == "fu7"
    default_ladders = FU7_LADDERS if fu7 else (FU5_LADDERS if fu5 else DEFAULT_LADDERS)
    default_audit = FU7_TEXT_AUDIT if fu7 else (FU5_TEXT_AUDIT if fu5 else DEFAULT_TEXT_AUDIT)
    ladders = Path(args.ladders or default_ladders)
    audit_path = Path(args.text_audit or default_audit)
    out_prefix = args.out_prefix or f"issue_1090/{args.round}"
    agg = json.loads(ladders.read_text())
    text_audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
    if fu7:
        fu6_agg = json.loads(FU6_AGGREGATES.read_text()) if FU6_AGGREGATES.exists() else None
        fig_fu7_unlock_grid(agg, out_prefix, args.out_dir)
        fig_fu7_dual_rubric(agg, out_prefix, args.out_dir, text_audit)
        fig_fu7_rubric_offset_scatter(agg, out_prefix, args.out_dir)
        fig_fu7_panel_leakage(agg, out_prefix, args.out_dir)
        fig_fu7_projection_scatter(agg, fu6_agg, out_prefix, args.out_dir)
        fig_fu7_margin_at_selected(agg, out_prefix, args.out_dir)
        fig_fu7_degeneracy_flags(agg, out_prefix, args.out_dir)
        return
    if fu5:
        fu4_path = Path(args.fu4_ladders)
        fu4_agg = json.loads(fu4_path.read_text()) if fu4_path.exists() else None
        fig_fu5_unlock_grid(agg, fu4_agg, out_prefix, args.out_dir)
        fig_fu5_tier2_verdict(agg, out_prefix, args.out_dir, text_audit)
        if text_audit is not None and "fmt-pers-r256" in text_audit.get("arms", {}):
            fig_fu5_fmt_intrusion(text_audit, out_prefix, args.out_dir)
        fig_fu5_margin_at_selected(agg, out_prefix, args.out_dir)
        fig_fu5_degeneracy_flags(agg, out_prefix, args.out_dir)
        fig_fu5_eval_split(agg, out_prefix, args.out_dir)
        return
    fig_dose_lr_grid(agg, out_prefix, args.out_dir)
    fig_margin_at_selected(agg, out_prefix, args.out_dir)
    fig_degeneracy_flags(agg, out_prefix, args.out_dir)
    fig_tier2_verdict(agg, out_prefix, args.out_dir, text_audit)


if __name__ == "__main__":
    main()
