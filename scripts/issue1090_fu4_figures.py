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
        "Rate under the three intrusion dispositions\n(circle raw; x zeroed;\ndiamond excluded + Wilson 95% CI; dots 0.30 cut)",
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


def main() -> None:
    ap = argparse.ArgumentParser(description="#1090 fu4/fu5 figures")
    ap.add_argument("--round", choices=("fu4", "fu5"), default="fu4")
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
    ladders = Path(args.ladders or (FU5_LADDERS if fu5 else DEFAULT_LADDERS))
    audit_path = Path(args.text_audit or (FU5_TEXT_AUDIT if fu5 else DEFAULT_TEXT_AUDIT))
    out_prefix = args.out_prefix or f"issue_1090/{args.round}"
    agg = json.loads(ladders.read_text())
    text_audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
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
