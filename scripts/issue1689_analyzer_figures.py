"""Issue #1689 analyzer round-1 figures (paper-plots conventions).

Reads eval_results/issue_1689/analyzer/pair_digest.csv (from
issue1689_analyzer_digest.py) + the percell held-out R^2 JSONs and renders:

1. fig1_rung_heatmap_<model>.png    — hero: validity-annotated rung-reached
                                      heatmaps (prefix + context panels).
2. fig2_ceilings_per_cell.png       — per-cell within-cell held-out R^2 (L19),
                                      both arms, degenerate user cells marked.
3. fig3_class_rung_stack.png        — informative-pair rung distribution by
                                      pair class per model x arm.
4. fig4_crossed_vs_marginal.png     — Q5 item 4: rung-9 recovery into the
                                      assistant-chat target, identity-only vs
                                      framing-only vs crossed, labeled points.
5. fig5_side_localization.png       — Q5 item 5: context-side-only (rung 7) vs
                                      answer-side-only (rung 8) recovery.
6. fig6_provenance_ladder.png       — per-rung recovery curves, user-provenance
                                      pairs (prefix arm).
7. fig7_assistant_framing_ladder.png— per-rung recovery curves, assistant
                                      framing pairs (context arm).

Usage: uv run python scripts/issue1689_analyzer_figures.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

# CRITICAL: load_dotenv() BEFORE importing matplotlib / numpy — shared-VM
# thread caps (#847) freeze at first BLAS import.
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[1]
DIGEST = REPO / "eval_results/issue_1689/analyzer/pair_digest.csv"
PERCELL = REPO / "eval_results/issue_1689/percell"
FIGDIR = REPO / "figures/issue_1689"

IDENTITY_ORDER = [
    "assistant",
    "user_lmsys",
    "user_haiku",
    "user_onpolicy",
    "helios",
    "wren",
    "dana",
]
FRAMING_ORDER = ["chat", "naturalistic", "story"]
CELL_ORDER = [f"{i}_{f}" for i in IDENTITY_ORDER for f in FRAMING_ORDER]
SHORT = {
    "assistant": "asst",
    "user_lmsys": "user-LMSYS",
    "user_haiku": "user-haiku",
    "user_onpolicy": "user-onpol",
    "helios": "HELIOS",
    "wren": "Wren",
    "dana": "Dana",
    "chat": "chat",
    "naturalistic": "plain",
    "story": "story",
}
NO_RECONCILE = 10


def short_cell(slug: str) -> str:
    for ident in sorted(IDENTITY_ORDER, key=len, reverse=True):
        if slug.startswith(ident + "_"):
            return f"{SHORT[ident]}/{SHORT[slug[len(ident) + 1 :]]}"
    return slug


def load_digest() -> list[dict]:
    rows = list(csv.DictReader(DIGEST.open()))
    for r in rows:
        for k in list(r):
            if k in {
                "model",
                "arm",
                "pair",
                "src",
                "tgt",
                "src_identity",
                "src_framing",
                "tgt_identity",
                "tgt_framing",
                "cls",
            }:
                continue
            r[k] = float(r[k]) if r[k] not in ("", "nan") else math.nan
    return rows


def fig1_heatmaps(rows: list[dict]) -> None:

    cmap = plt.get_cmap("viridis", 9)
    for model in ("base", "instruct"):
        fig, axes = plt.subplots(1, 2, figsize=(15.5, 8.2))
        for ax, arm in zip(axes, ("prefix", "context")):
            grid = np.full((21, 21), np.nan)
            flags = {}
            for r in rows:
                if r["model"] != model or r["arm"] != arm:
                    continue
                i = CELL_ORDER.index(r["src"])
                j = CELL_ORDER.index(r["tgt"])
                if r["arm_invalid"]:
                    flags[(i, j)] = "inv"
                elif r["degenerate_ceiling"]:
                    flags[(i, j)] = "deg"
                else:
                    grid[i, j] = min(r["rung090"], 10)
            # base layer: informative rungs
            masked = np.ma.masked_invalid(np.where(grid == NO_RECONCILE, np.nan, grid))
            ax.imshow(masked, cmap=cmap, vmin=1, vmax=9, aspect="equal")
            # overlays
            for (i, j), kind in flags.items():
                color = "#f0f0f0" if kind == "inv" else "#c8c8c8"
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=color))
                ax.text(
                    j,
                    i,
                    "x" if kind == "inv" else "d",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="#666666",
                )
            for r in rows:
                if r["model"] != model or r["arm"] != arm:
                    continue
                if r["arm_invalid"] or r["degenerate_ceiling"]:
                    continue
                i = CELL_ORDER.index(r["src"])
                j = CELL_ORDER.index(r["tgt"])
                v = int(r["rung090"])
                if v == NO_RECONCILE:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="#8b1a1a"))
                    ax.text(j, i, "–", ha="center", va="center", fontsize=7, color="white")
                else:
                    ax.text(
                        j,
                        i,
                        str(v),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if v <= 5 else "black",
                    )
            labels = [short_cell(c) for c in CELL_ORDER]
            ax.set_xticks(range(21), labels, rotation=90, fontsize=7)
            ax.set_yticks(range(21), labels, fontsize=7)
            ax.set_xlabel("target condition (map evaluated here)")
            ax.set_ylabel("source condition (map fit here)")
            ax.set_title(f"{arm} arm", fontsize=12)
        fig.suptitle(
            f"Weakest reconciling ladder rung at the 0.90 bar — Qwen2.5-7B{'-Instruct' if model == 'instruct' else ''} (L19)\n"
            "1=direct .. 9=full A·M·B; dark red = no rung reconciles; d = target ceiling <= 0 (rung trivial); x = arm invalid (user-cell X==Y)",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        savefig_paper(fig, f"fig1_rung_heatmap_{model}", dir=FIGDIR)
        plt.close(fig)


def fig2_ceilings() -> None:
    import json as _json

    cells = {}
    for f in sorted(PERCELL.glob("heldout_r2_*.json")):
        name = f.stem[len("heldout_r2_") :]
        if "Instruct" in name:
            model, cell = "instruct", name.split("Instruct_")[1]
        else:
            model, cell = "base", name.split("7B_")[1]
        d = _json.loads(f.read_text())
        li = d["layers"].index(19)
        cells[(model, cell)] = {
            "prefix": d["prefix"]["held_out_r2_per_layer"][li],
            "context": d["context"]["held_out_r2_per_layer"][li],
            "idbias_prefix": d["prefix"]["identity_bias_r2_headline"],
            "idbias_context": d["context"]["identity_bias_r2_headline"],
            "n": d["n_rows"],
        }
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 9.0), sharex=True)
    x = np.arange(len(CELL_ORDER))
    pal = paper_palette(2)
    for ax, model in zip(axes, ("base", "instruct")):
        for k, (arm, off) in enumerate([("prefix", -0.2), ("context", 0.2)]):
            vals = [cells[(model, c)][arm] for c in CELL_ORDER]
            degen = [cells[(model, c)][f"idbias_{arm}"] >= 0.999 for c in CELL_ORDER]
            bars = ax.bar(x + off, vals, width=0.38, color=pal[k], label=f"{arm} arm")
            for xi, (v, dg) in enumerate(zip(vals, degen)):
                if dg:
                    bars[xi].set_hatch("///")
                    bars[xi].set_edgecolor("#8b1a1a")
                ax.text(
                    xi + off, max(v, 0) + 0.015, f"{v:.2f}", ha="center", fontsize=6.2, rotation=90
                )
        ax.axhline(0, color="#444444", lw=0.8)
        ax.set_ylabel("within-cell held-out R$^2$ (L19)")
        ax.set_title(f"Qwen2.5-7B{'-Instruct' if model == 'instruct' else ''}", fontsize=12)
        ax.set_ylim(-0.08, 1.12)
        if model == "base":
            ax.legend(loc="upper left")
    axes[1].set_xticks(x, [short_cell(c) for c in CELL_ORDER], rotation=90, fontsize=8)
    fig.suptitle(
        "Per-cell map ceilings: within-cell held-out R$^2$, both arms\n"
        "hatched red = degenerate arm (identity+bias baseline R$^2$ = 1.0: X equals Y by construction)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    savefig_paper(fig, "fig2_ceilings_per_cell", dir=FIGDIR)
    plt.close(fig)


def fig3_class_stack(rows: list[dict]) -> None:
    classes = ["framing", "user-framing", "identity", "identity-vs-user", "provenance", "crossed"]
    rung_bins = [
        (1, "1 direct"),
        (2, "2-3 offsets"),
        (4, "4-5 bias/scale"),
        (6, "6 rotation"),
        (7, "7-8 one-side"),
        (9, "9 full"),
        (10, "none"),
    ]

    def bini(v: float) -> int:
        v = int(v)
        if v == 1:
            return 0
        if v in (2, 3):
            return 1
        if v in (4, 5):
            return 2
        if v == 6:
            return 3
        if v in (7, 8):
            return 4
        if v == 9:
            return 5
        return 6

    # Color semantics match fig1: viridis 1->9 for rung bins, dark red for "none".
    vir = plt.get_cmap("viridis")
    pal = [vir(v) for v in (0.0, 0.25, 0.42, 0.58, 0.75, 0.99)] + ["#8b1a1a"]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4), sharey=True)
    for ax, (model, arm) in zip(
        axes.flat,
        [("base", "prefix"), ("base", "context"), ("instruct", "prefix"), ("instruct", "context")],
    ):
        counts = np.zeros((len(classes), 7))
        for r in rows:
            if r["model"] != model or r["arm"] != arm:
                continue
            if r["arm_invalid"] or r["degenerate_ceiling"]:
                continue
            counts[classes.index(r["cls"]), bini(r["rung090"])] += 1
        bottom = np.zeros(len(classes))
        for b, (_, lab) in enumerate(rung_bins):
            ax.bar(range(len(classes)), counts[:, b], bottom=bottom, color=pal[b], label=lab)
            bottom += counts[:, b]
        for ci in range(len(classes)):
            ax.text(ci, bottom[ci] + 0.3, str(int(bottom[ci])), ha="center", fontsize=8)
        ax.set_xticks(
            range(len(classes)),
            ["framing", "user\nframing", "identity", "asst-vs\nuser", "prove-\nnance", "crossed"],
            fontsize=8,
        )
        ax.set_title(f"{model} · {arm} arm", fontsize=11)
    axes[0, 0].set_ylabel("informative ordered pairs")
    axes[1, 0].set_ylabel("informative ordered pairs")
    axes[0, 1].legend(fontsize=8, title="weakest reconciling rung", loc="upper right")
    fig.suptitle(
        "Rung-reached (0.90 bar) by pair class — informative pairs only\n"
        "(construct-invalid arms and degenerate-ceiling targets excluded)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    savefig_paper(fig, "fig3_class_rung_stack", dir=FIGDIR)
    plt.close(fig)


def fig4_crossed_vs_marginal(rows: list[dict]) -> None:
    groups = ["identity-only", "framing-only", "crossed"]
    pal = paper_palette(3)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), sharey=True)
    for ax, arm in zip(axes, ("context", "prefix")):
        for gi, g in enumerate(groups):
            xs, ys, labs = [], [], []
            for r in rows:
                if r["model"] != "instruct" or r["arm"] != arm or r["tgt"] != "assistant_chat":
                    continue
                if r["arm_invalid"] or r["degenerate_ceiling"]:
                    continue
                if r["src_identity"].startswith("user_"):
                    continue  # user-source pairs measure a different answer construct (u2-end)
                gmap = {
                    "identity": "identity-only",
                    "framing": "framing-only",
                    "crossed": "crossed",
                }
                if gmap.get(r["cls"]) != g:
                    continue
                xs.append(gi + (len(xs) - 1) * 0.06)
                ys.append(r["rec_9"])
                labs.append(short_cell(r["src"]))
            ax.scatter(xs, ys, s=46, color=pal[gi], zorder=3)
            for x, y, lab in zip(xs, ys, labs):
                ax.text(x + 0.05, y, lab, fontsize=7, va="center")
            if ys:
                ax.hlines(np.median(ys), gi - 0.22, gi + 0.22, color=pal[gi], lw=2.4, zorder=4)
        ax.set_xticks(range(3), ["identity\nchanges", "framing\nchanges", "both\nchange"])
        ax.set_title(f"{arm} arm", fontsize=11)
        ax.set_ylim(0, 1.45)
        ax.axhline(1.0, color="#888888", lw=0.8, ls="--")
    axes[0].set_ylabel("rung-9 (full A·M·B) recovery fraction of target ceiling")
    fig.suptitle(
        "Transfers into the assistant-chat cell (instruct, L19): what changing each axis costs\n"
        "points = source cells (medians as thick ticks); dashed line = full recovery",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    savefig_paper(fig, "fig4_crossed_vs_marginal", dir=FIGDIR)
    plt.close(fig)


def fig5_side_localization(rows: list[dict]) -> None:
    cls_style = {
        "framing": ("o", 0),
        "identity": ("s", 1),
        "provenance": ("^", 2),
        "identity-vs-user": ("D", 3),
        "user-framing": ("v", 4),
        "crossed": ("P", 5),
    }
    pal = paper_palette(6)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8), sharex=True, sharey=True)
    for ax, model in zip(axes, ("base", "instruct")):
        for cls, (marker, ci) in cls_style.items():
            xs, ys = [], []
            for r in rows:
                if r["model"] != model or r["arm_invalid"] or r["degenerate_ceiling"]:
                    continue
                if r["cls"] != cls or r["ceiling"] < 0.05:
                    continue
                xs.append(np.clip(r["rec_7"], -1.5, 1.5))
                ys.append(np.clip(r["rec_8"], -1.5, 1.5))
            ax.scatter(
                xs,
                ys,
                marker=marker,
                s=42,
                color=pal[ci],
                label=f"{cls} (n={len(xs)})",
                alpha=0.85,
                zorder=3,
            )
        ax.plot([-1.5, 1.5], [-1.5, 1.5], color="#999999", lw=0.8, ls="--")
        ax.axhline(0, color="#cccccc", lw=0.7)
        ax.axvline(0, color="#cccccc", lw=0.7)
        ax.set_xlabel("context-side-only recovery (rung 7, A-only)")
        ax.set_title(f"{model}", fontsize=11)
    axes[0].set_ylabel("answer-side-only recovery (rung 8, B-only)")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "Side localization, pairs with target ceiling >= 0.05 (both arms pooled; values clipped to [-1.5, 1.5])\n"
        "above the diagonal = answer-side correction recovers more than context-side",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    savefig_paper(fig, "fig5_side_localization", dir=FIGDIR)
    plt.close(fig)


def _rung_curves(ax, rows_sel: list[dict], pal) -> None:
    for k, r in enumerate(sorted(rows_sel, key=lambda q: q["pair"])):
        recs = [r[f"rec_{i}"] for i in range(1, 10)]
        ax.plot(
            range(1, 10),
            np.clip(recs, -1.0, 1.6),
            marker="o",
            ms=3.5,
            color=pal[k % len(pal)],
            lw=1.4,
            label=f"{short_cell(r['src'])} to {short_cell(r['tgt'])} (n={int(r['n_common'])})",
        )
    ax.axhline(0.9, color="#888888", lw=0.8, ls="--")
    ax.axhline(0.0, color="#cccccc", lw=0.7)
    ax.set_xticks(
        range(1, 10),
        [
            "1\ndirect",
            "2\nΔx",
            "3\nΔy",
            "4\nbias",
            "5\nscale",
            "6\nrotate",
            "7\nA-only",
            "8\nB-only",
            "9\nA·M·B",
        ],
        fontsize=8,
    )


def fig6_provenance(rows: list[dict]) -> None:
    pal = paper_palette(4)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6), sharey=True)
    for ax, model in zip(axes, ("base", "instruct")):
        sel = [
            r
            for r in rows
            if r["model"] == model
            and r["arm"] == "prefix"
            and r["cls"] == "provenance"
            and not r["arm_invalid"]
            and not r["degenerate_ceiling"]
            and {"user_lmsys", "user_haiku"} == {r["src_identity"], r["tgt_identity"]}
        ]
        _rung_curves(ax, sel, pal)
        ax.set_title(f"{model}", fontsize=11)
        ax.legend(fontsize=7, loc="lower right")
        ax.set_ylim(-1.05, 1.3)
    axes[0].set_ylabel("recovery fraction of target ceiling")
    fig.suptitle(
        "User-turn provenance: real-LMSYS vs haiku-simulated u2 (prefix arm; shared u1+a1 prefix)\n"
        "dashed line = 0.90 reconciliation bar; haiku-vs-on-policy pairs omitted (stores duplicated, contrast void)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    savefig_paper(fig, "fig6_provenance_ladder", dir=FIGDIR)
    plt.close(fig)


def fig7_assistant_framing(rows: list[dict]) -> None:
    pal = paper_palette(6)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6), sharey=True)
    for ax, model in zip(axes, ("base", "instruct")):
        sel = [
            r
            for r in rows
            if r["model"] == model
            and r["arm"] == "context"
            and r["cls"] == "framing"
            and r["src_identity"] == "assistant"
        ]
        _rung_curves(ax, sel, pal)
        ax.set_title(f"{model}", fontsize=11)
        ax.legend(fontsize=7, loc="lower right")
        ax.set_ylim(-1.05, 1.6)
    axes[0].set_ylabel("recovery fraction of target ceiling")
    fig.suptitle(
        "Assistant framing pairs, all 9 rungs (context arm): base reaches the 0.90 bar at no rung;"
        " instruct only at rung 9 (3 of 6 pairs)\n"
        "into-story transfers exceed 1.0 only because the story-cell ceiling is near zero",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    savefig_paper(fig, "fig7_assistant_framing_ladder", dir=FIGDIR)
    plt.close(fig)


def main() -> int:
    set_paper_style("blog")
    FIGDIR.mkdir(parents=True, exist_ok=True)
    rows = load_digest()
    fig1_heatmaps(rows)
    fig2_ceilings()
    fig3_class_stack(rows)
    fig4_crossed_vs_marginal(rows)
    fig5_side_localization(rows)
    fig6_provenance(rows)
    fig7_assistant_framing(rows)
    print("figures written to", FIGDIR)
    return 0


if __name__ == "__main__":
    import os
    import sys

    rc = main()
    # C-extension interpreter-shutdown-race workaround; see the corresponding
    # block in scripts/issue1689_gen_corpus.py for the full rationale +
    # gotchas.md § PyGILState_Release SIGABRT pointer (matplotlib/torch atexit
    # teardown safely skipped; figures are written via savefig before return).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc if isinstance(rc, int) else 0)


def fig8_provenance_duplication(rows: list[dict]) -> None:
    """Forward vs reverse rung R^2 for provenance pairs: haiku-vs-onpolicy pairs
    sit exactly on the diagonal (identical stored data); LMSYS pairs do not."""
    idx = {(r["model"], r["arm"], r["pair"]): r for r in rows}
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    pal = paper_palette(3)
    fams = [
        ("user_haiku", "user_onpolicy", "haiku vs on-policy", 0, "o"),
        ("user_lmsys", "user_haiku", "LMSYS vs haiku", 1, "s"),
        ("user_lmsys", "user_onpolicy", "LMSYS vs on-policy", 2, "^"),
    ]
    for a, b, lab, ci, mk in fams:
        xs, ys = [], []
        for model in ("base", "instruct"):
            for fr in ("chat", "story", "naturalistic"):
                k1 = (model, "prefix", f"{a}_{fr}__{b}_{fr}")
                k2 = (model, "prefix", f"{b}_{fr}__{a}_{fr}")
                if k1 in idx and k2 in idx:
                    for i in range(1, 10):
                        xs.append(np.clip(idx[k1][f"r2_{i}"], -1.2, 1.2))
                        ys.append(np.clip(idx[k2][f"r2_{i}"], -1.2, 1.2))
        ax.scatter(
            xs,
            ys,
            s=30,
            marker=mk,
            color=pal[ci],
            alpha=0.75,
            label=f"{lab} ({len(xs)} rung values)",
            zorder=3,
        )
    ax.plot([-1.2, 1.2], [-1.2, 1.2], color="#999999", lw=0.8, ls="--")
    ax.set_xlabel("rung R$^2$, forward direction (A to B)")
    ax.set_ylabel("rung R$^2$, reverse direction (B to A)")
    ax.set_title(
        "User-provenance pairs, prefix arm: forward vs reverse rung R$^2$\n"
        "haiku-vs-on-policy pairs are identical in both directions (stored data duplicated)",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper left")
    savefig_paper(fig, "fig8_provenance_duplication", dir=FIGDIR)
    plt.close(fig)
