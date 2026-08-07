"""Analyzer-round figures for issue #2054 (6k-row scaffold-and-splice lattice).

Reads the committed per-cell fit JSONs (data/issue_2054/fits/), the 816-rung
ladder census (data/issue_2054/ladder/), the authorship x presentation 2x2
(eval_results/issue_2054/fits/authorship_presentation_2x2.json), and the
analyzer companion digests under /tmp (length parity + cap-hit refits).

Writes blog-style figures to the MAIN checkout's figures/issue_2054/
(figures-only commit; this script lives on the issue-2054 branch).

Color discipline (one color = one meaning across every figure):
  instruct = PAL[0], base = PAL[1]; inserted = PAL[4], on-policy = PAL[5];
  ladder classes use PAL[2,3,6,7].
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE numpy/matplotlib so the shared-VM thread caps bind (#847)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

WT = Path("/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-2054")
MAIN_FIG = "/home/thomasjiralerspong/explore-persona-space/figures/"
PAL = paper_palette_blog(8)
C_INSTRUCT, C_BASE = PAL[0], PAL[1]
C_INSERTED, C_ONPOLICY = PAL[4], PAL[5]
CLASS_COLORS = {
    "cross_framing": PAL[2],
    "cross_character": PAL[3],
    "cross_model": PAL[6],
    "twobytwo": PAL[7],
}
FORM_LABEL = {
    "chat": "chat template",
    "bare_text": "bare text",
    "attrib_quoted": "story, attributed quote",
    "bare_label": "story, bare label",
}
COND_LABEL = {
    "inserted": "inserted answer",
    "on_policy": "on-policy answer",
    "cell_c": "transposed (story answer in chat)",
}
MODEL_LABEL = {"qwen2.5-7b": "base", "qwen2.5-7b-instruct": "instruct"}
RUNG_KEYS = [
    "1_direct",
    "2_ctx_offset",
    "3_ans_offset",
    "4_bias_refit",
    "5_global_scale",
    "6_rotation",
    "7_ctx_reparam",
    "8_ans_reparam",
    "9_full_AMB",
]
RUNG_LABEL = [
    "direct",
    "context\noffset",
    "answer\noffset",
    "bias\nrefit",
    "global\nscale",
    "rotation",
    "context\nre-map",
    "answer\nre-map",
    "full\nrefit",
]


def parse_cell(c: str) -> dict:
    p = c.split("__")
    return dict(variant=p[0], cond=p[1], form=p[2], model=p[3])


def load_fits() -> list[dict]:
    rows = []
    for f in sorted((WT / "data/issue_2054/fits").glob("*.json")):
        if "pilot_gate" in f.name:
            continue
        d = json.load(open(f))
        rows.append(d)
    return rows


def load_ladder() -> list[dict]:
    return json.load(open("/tmp/issue2054_ladder_rows.json"))


def fig_hero_boundary_vs_prose(ladder: list[dict]) -> None:
    """Per-target mean direct-transfer R^2: prose swapped (same boundary) vs
    boundary swapped (same prose)."""
    ctx = [r for r in ladder if r["arm"] == "context"]
    wb, cs = defaultdict(list), defaultdict(list)
    for r in ctx:
        S, T = parse_cell(r["src"]), parse_cell(r["tgt"])
        diffs = [k for k in ("variant", "cond", "form", "model") if S[k] != T[k]]
        if diffs == ["form"]:
            wb[r["tgt"]].append(r["rungs"]["1_direct"])
        elif diffs == ["variant"]:
            cs[r["tgt"]].append(r["rungs"]["1_direct"])
    targets = sorted(set(wb) & set(cs))
    set_paper_style("blog")
    fig, ax = plt.subplots()
    for t in targets:
        P = parse_cell(t)
        x = st.mean(cs[t])
        y = st.mean(wb[t])
        col = C_INSERTED if P["cond"] == "inserted" else C_ONPOLICY
        mk = "o" if P["model"].endswith("instruct") else "s"
        ax.scatter([x], [y], color=col, marker=mk, s=42, zorder=3)
    lim_lo, lim_hi = -0.35, 0.40
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="#999999", lw=1.0, ls="--", zorder=1)
    ax.axhline(0.0, color="#cccccc", lw=0.8, zorder=1)
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("direct transfer from a different story, same answer boundary (held-out R²)")
    ax.set_ylabel("direct transfer across answer boundary,\nsame story prose (held-out R²)")
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=C_INSERTED, marker="o", ls="", label="inserted answer"),
        Line2D([], [], color=C_ONPOLICY, marker="o", ls="", label="on-policy answer"),
        Line2D([], [], color="#666666", marker="o", ls="", label="instruct (circle)"),
        Line2D([], [], color="#666666", marker="s", ls="", label="base (square)"),
        Line2D([], [], color="#999999", ls="--", label="equal transfer"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    set_title_subtitle(
        ax,
        "Swapping the answer boundary breaks map transfer;\nswapping the story prose does not",
        "32 story target cells; each point averages the direct-transfer rungs into that cell",
    )
    savefig_paper(fig, "issue_2054/hero_boundary_vs_prose", dir=MAIN_FIG)
    plt.close(fig)


def fig_boundary_vs_prose_pairs(ladder: list[dict]) -> None:
    """Raw sibling: every ordered pair's direct-transfer R^2, by single-axis class."""
    ctx = [r for r in ladder if r["arm"] == "context"]
    # Label precision (round-2 critique): 20 of the 56 boundary-swap pairs are
    # assistant form swaps that cross story<->non-story renders, so the class is
    # not "same prose" throughout; 24 of the 48 model-swap pairs are on-policy,
    # where the answer text differs by construction.
    buckets = {
        "boundary swapped": [],
        "prose swapped\n(same boundary)": [],
        "model swapped": [],
    }
    for r in ctx:
        S, T = parse_cell(r["src"]), parse_cell(r["tgt"])
        diffs = [k for k in ("variant", "cond", "form", "model") if S[k] != T[k]]
        if diffs == ["form"]:
            buckets["boundary swapped"].append(r["rungs"]["1_direct"])
        elif diffs == ["variant"]:
            buckets["prose swapped\n(same boundary)"].append(r["rungs"]["1_direct"])
        elif diffs == ["model"]:
            buckets["model swapped"].append(r["rungs"]["1_direct"])
    set_paper_style("blog")
    fig, ax = plt.subplots()
    rng = np.random.default_rng(42)
    cols = [PAL[2], PAL[3], PAL[6]]
    for i, (name, vals) in enumerate(buckets.items()):
        x = i + rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(x, vals, s=14, alpha=0.55, color=cols[i])
        med = st.median(vals)
        ax.hlines(med, i - 0.22, i + 0.22, color="#333333", lw=2.0, zorder=4)
    ax.axhline(0.0, color="#cccccc", lw=0.8)
    ax.set_xticks(range(len(buckets)))
    ax.set_xticklabels(list(buckets.keys()))
    ax.set_ylabel("direct-transfer held-out R²")
    set_title_subtitle(
        ax,
        "Per-pair direct transfer by which single axis was swapped",
        "every ordered source→target pair differing on exactly one axis; black bar = median",
    )
    savefig_paper(fig, "issue_2054/boundary_vs_prose_pairs", dir=MAIN_FIG)
    plt.close(fig)


def fig_ceilings(fits: list[dict]) -> None:
    """Within-cell ceilings across the lattice, context arm; prefix arm as gray x."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    order = [
        ("chat", "inserted"),
        ("chat", "on_policy"),
        ("bare_text", "inserted"),
        ("bare_text", "on_policy"),
        ("attrib_quoted", "inserted"),
        ("attrib_quoted", "on_policy"),
        ("bare_label", "inserted"),
        ("bare_label", "on_policy"),
        ("chat", "cell_c"),
    ]
    xticklabels = []
    rng = np.random.default_rng(42)
    for i, (form, cond) in enumerate(order):
        for d in fits:
            if d["form"] != form or d["condition"] != cond:
                continue
            ctx = d["arm_reports"]["context"]["pooled"]["r2_ambient_mean"]
            pfx = d["arm_reports"]["prefix"]["pooled"]["r2_ambient_mean"]
            col = C_INSTRUCT if d["model"].endswith("instruct") else C_BASE
            is_assistant = d["variant"].startswith("conversation")
            x = i + float(rng.uniform(-0.13, 0.13))
            ax.scatter(
                [x],
                [ctx],
                s=(52 if is_assistant else 20),
                color=col,
                marker=("D" if is_assistant else "o"),
                zorder=3,
                edgecolors="white",
                linewidths=0.5,
            )
            ax.scatter([x], [pfx], s=14, color="#aaaaaa", marker="x", linewidths=1.2, zorder=2)
        lab = {
            "chat": "chat",
            "bare_text": "bare\ntext",
            "attrib_quoted": "story\nattrib.",
            "bare_label": "story\nlabel",
        }[form]
        clab = {"inserted": "ins", "on_policy": "on-pol", "cell_c": "transposed"}[cond]
        xticklabels.append(f"{lab}\n{clab}")
    ax.axhline(0.0, color="#cccccc", lw=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(xticklabels, fontsize=8)
    ax.set_ylabel("within-cell held-out R² (ambient basis)")
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=C_INSTRUCT, marker="o", ls="", label="instruct"),
        Line2D([], [], color=C_BASE, marker="o", ls="", label="base"),
        Line2D([], [], color="#666666", marker="D", ls="", label="assistant cell (diamond)"),
        Line2D([], [], color="#666666", marker="o", ls="", label="character cell (circle)"),
        Line2D([], [], color="#aaaaaa", marker="x", ls="", label="prefix arm"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    set_title_subtitle(
        ax,
        "Every lattice cell fits a real context→answer map,\nand the answer-boundary form sets its ceiling",
        "56 cells; shuffled-answer null p95 ≈ −0.03 everywhere; identity+bias ≤ −0.48 everywhere",
    )
    savefig_paper(fig, "issue_2054/ceilings_lattice", dir=MAIN_FIG)
    plt.close(fig)


def fig_ceilings_folds(fits: list[dict]) -> None:
    """Raw sibling: per-fold ambient R^2 for all 56 cells (context arm)."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    cells = sorted(fits, key=lambda d: d["arm_reports"]["context"]["pooled"]["r2_ambient_mean"])
    for i, d in enumerate(cells):
        pf = d["arm_reports"]["context"]["per_fold"]
        col = C_INSTRUCT if d["model"].endswith("instruct") else C_BASE
        ax.scatter([i] * len(pf), [x["r2_ambient"] for x in pf], s=8, alpha=0.7, color=col)
    ax.set_xlabel("lattice cell (sorted by mean ceiling)")
    ax.set_ylabel("per-fold held-out R² (ambient)")
    from matplotlib.lines import Line2D

    ax.legend(
        handles=[
            Line2D([], [], color=C_INSTRUCT, marker="o", ls="", label="instruct"),
            Line2D([], [], color=C_BASE, marker="o", ls="", label="base"),
        ],
        loc="upper left",
        fontsize=8,
    )
    set_title_subtitle(
        ax,
        "Per-fold ceilings behind the lattice summary",
        "5 shared conversation-grouped folds per cell, 56 cells",
    )
    savefig_paper(fig, "issue_2054/ceilings_lattice_folds", dir=MAIN_FIG)
    plt.close(fig)


def fig_twobytwo(fits: list[dict]) -> None:
    """2x2 terms (attributed-quote story form, byte-matched c/d) + raw cell ceilings."""
    d2 = json.load(open(WT / "eval_results/issue_2054/fits/authorship_presentation_2x2.json"))
    recs = [
        r for r in d2["records"] if r["arm"] == "context" and r["story_form"] == "attrib_quoted"
    ]
    recs.sort(key=lambda r: (r["model"], r["character"]))
    labels = [f"{r['character'][5:].capitalize()}\n{MODEL_LABEL[r['model']]}" for r in recs]
    terms = ["authorship_c_minus_a", "presentation_b_minus_a", "interaction"]
    tlabel = [
        "authorship (story-authored − chat-authored)",
        "presentation (story − chat template)",
        "interaction",
    ]
    tcol = [PAL[3], PAL[2], PAL[7]]
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for j, term in enumerate(terms):
        xs = np.arange(len(recs)) + (j - 1) * 0.22
        pts = [r["terms"][term]["point"] for r in recs]
        los = [r["terms"][term]["ci95"][0] for r in recs]
        his = [r["terms"][term]["ci95"][1] for r in recs]
        yerr = [np.array(pts) - np.array(los), np.array(his) - np.array(pts)]
        ax.errorbar(
            xs,
            pts,
            yerr=yerr,
            fmt="o",
            ms=5,
            color=tcol[j],
            label=tlabel[j],
            capsize=2,
            markeredgewidth=1.0,
            elinewidth=1.2,
        )
    ax.axhline(0.0, color="#cccccc", lw=0.8)
    ax.set_xticks(range(len(recs)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("ceiling difference (held-out R²)")
    ax.legend(loc="lower right", fontsize=8)
    set_title_subtitle(
        ax,
        "The inserted-vs-on-policy gap does not decompose additively:\nboth main terms are negative and the interaction is positive",
        "8 character × model pairs, attributed-quote story form",
    )
    savefig_paper(fig, "issue_2054/twobytwo_terms", dir=MAIN_FIG)
    plt.close(fig)

    # raw sibling: the four cell ceilings per pair
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    letters = ["a", "b", "c", "d"]
    llabel = {
        "a": "chat-authored, chat-presented (a)",
        "b": "chat-authored, story-presented (b)",
        "c": "story-authored, chat-presented (c)",
        "d": "story-authored, story-presented (d)",
    }
    lcol = {"a": PAL[0], "b": PAL[2], "c": PAL[6], "d": PAL[5]}
    for j, let in enumerate(letters):
        xs = np.arange(len(recs)) + (j - 1.5) * 0.17
        pts = [r["ceilings"][let]["fold_mean"] for r in recs]
        per_fold = [list(r["ceilings"][let]["per_fold"].values()) for r in recs]
        for xi, pfv in zip(xs, per_fold):
            ax.scatter([xi] * len(pfv), pfv, s=7, alpha=0.5, color=lcol[let])
        ax.scatter(
            xs,
            pts,
            s=34,
            color=lcol[let],
            label=llabel[let],
            zorder=4,
            edgecolors="white",
            linewidths=0.5,
        )
    ax.set_xticks(range(len(recs)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("within-cell held-out R² (ambient)")
    ax.legend(loc="upper right", fontsize=7)
    set_title_subtitle(
        ax,
        "The four 2×2 cell ceilings behind the decomposition",
        "large dot = 5-fold mean, small dots = per-fold values",
    )
    savefig_paper(fig, "issue_2054/twobytwo_cells", dir=MAIN_FIG)
    plt.close(fig)


def fig_ladder(ladder: list[dict]) -> None:
    """Median ratio-to-target-ceiling per rung, by pair class (context arm)."""
    ctx = [r for r in ladder if r["arm"] == "context"]

    def fname_class(r):
        S, T = parse_cell(r["src"]), parse_cell(r["tgt"])
        diffs = [k for k in ("variant", "cond", "form", "model") if S[k] != T[k]]
        if diffs == ["form"]:
            return "cross_framing"
        if diffs == ["model"]:
            return "cross_model"
        if diffs == ["variant"]:
            return "cross_character"
        return "twobytwo"

    # Label precision (round-2 critique): see fig_boundary_vs_prose_pairs note.
    CLS_LABEL = {
        "cross_framing": "boundary swapped",
        "cross_character": "prose swapped (same boundary)",
        "cross_model": "model swapped",
        "twobytwo": "authorship/presentation swapped (2×2 edges)",
    }
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for cls in ["cross_model", "cross_character", "cross_framing", "twobytwo"]:
        sub = [r for r in ctx if fname_class(r) == cls]
        med = [st.median(r["ratios"][k] for r in sub) for k in RUNG_KEYS]
        q1 = [np.percentile([r["ratios"][k] for r in sub], 25) for k in RUNG_KEYS]
        q3 = [np.percentile([r["ratios"][k] for r in sub], 75) for k in RUNG_KEYS]
        x = np.arange(9)
        ax.plot(
            x,
            med,
            marker="o",
            ms=4,
            color=CLASS_COLORS[cls],
            label=f"{CLS_LABEL[cls]} (n={len(sub)})",
        )
        ax.fill_between(x, q1, q3, color=CLASS_COLORS[cls], alpha=0.12, linewidth=0)
    ax.axhline(1.0, color="#cccccc", lw=0.8, ls=":")
    ax.axhline(0.0, color="#cccccc", lw=0.8)
    ax.set_xticks(range(9))
    ax.set_xticklabels(RUNG_LABEL, fontsize=8)
    ax.set_ylim(-1.6, 1.15)
    ax.set_xlabel("adaptation allowed on the source map (9-rung ladder)")
    ax.set_ylabel("transfer R² / target's own ceiling")
    ax.legend(loc="lower right", fontsize=8)
    set_title_subtitle(
        ax,
        "What it takes to align maps across cells:\na context-side linear re-map recovers boundary swaps, answer-side does not",
        "median with interquartile band over ordered pairs; context arm",
    )
    savefig_paper(fig, "issue_2054/ladder_recovery", dir=MAIN_FIG)
    plt.close(fig)


def fig_ks_pairs() -> None:
    """Per-pair answer-length KS D (inserted vs on-policy, matched conversations).

    Round 2: the low-level view behind the re-scoped parity claim — all 16
    character-cell pairs breach the 0.30 bound; two assistant pairs pass.
    """
    par = json.load(
        open(WT / "eval_results/issue_2054/analyzer_companions/answer_length_parity.json")
    )
    pairs = par["pairs"]
    groups = {
        "character pairs (n=16)": [
            (k, v["ks_matched"]) for k, v in pairs.items() if k.startswith("char_")
        ],
        "assistant pairs (n=8)": [
            (k, v["ks_matched"]) for k, v in pairs.items() if not k.startswith("char_")
        ],
    }
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    rng = np.random.default_rng(42)
    cols = [PAL[3], PAL[0]]
    for i, (name, vals) in enumerate(groups.items()):
        x = i + rng.uniform(-0.10, 0.10, size=len(vals))
        ax.scatter(x, [v for _, v in vals], s=26, alpha=0.8, color=cols[i])
        if i == 1:  # label the two passing assistant pairs + the maximum
            for xi, (k, v) in zip(x, vals):
                short = {
                    "conversation_paired_stories_assistant__bare_text__qwen2.5-7b": "bare text, base",
                    "conversation_paired_stories_assistant__chat__qwen2.5-7b-instruct": "chat, instruct",
                    "conversation_paired_stories_assistant__bare_text__qwen2.5-7b-instruct": "bare text, instruct",
                }.get(k)
                if short:
                    ax.text(xi + 0.03, v, short, fontsize=7, va="center")
    ax.axhline(0.30, color="#888888", lw=1.0, ls="--")
    ax.text(1.42, 0.305, "parity bound (D = 0.30)", fontsize=7.5, color="#666666")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(list(groups.keys()))
    ax.set_ylabel("answer-length KS D, inserted vs on-policy")
    ax.set_ylim(0, 0.65)
    set_title_subtitle(
        ax,
        "The answer-length parity bound fails in every character cell",
        "one dot per (identity, boundary form, model) pair; matched conversations (n ≈ 7,985–7,989)",
    )
    savefig_paper(fig, "issue_2054/ks_parity_pairs", dir=MAIN_FIG)
    plt.close(fig)


def fig_length_hist() -> None:
    """Answer-length distributions (tokens): inserted pool vs on-policy cells."""
    par = json.load(open("/tmp/issue2054_length_parity.json"))
    set_paper_style("blog")
    fig, ax = plt.subplots()
    groups = [
        (
            "inserted answer pool (all cells)",
            "conversation_paired_stories_assistant__inserted",
            PAL[4],
        ),
        (
            "on-policy, story attributed quote (instruct)",
            "conversation_paired_stories_assistant__on_policy__attrib_quoted__qwen2.5-7b-instruct",
            PAL[3],
        ),
        (
            "on-policy, chat (base)",
            "conversation_paired_stories_assistant__on_policy__chat__qwen2.5-7b",
            PAL[1],
        ),
        (
            "on-policy, bare text (instruct)",
            "conversation_paired_stories_assistant__on_policy__bare_text__qwen2.5-7b-instruct",
            PAL[5],
        ),
    ]
    xs = np.arange(len(groups))
    for i, (lab, key, col) in enumerate(groups):
        v = par["cells"][key]
        ax.errorbar(
            [i],
            [v["median"]],
            yerr=[[v["median"] - v["p10"]], [v["p90"] - v["median"]]],
            fmt="o",
            ms=7,
            color=col,
            capsize=4,
            elinewidth=1.4,
            markeredgewidth=1.2,
        )
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [
            "inserted pool",
            "on-policy story\nattrib. (instruct)",
            "on-policy chat\n(base)",
            "on-policy bare\ntext (instruct)",
        ],
        fontsize=8,
    )
    ax.set_ylabel("answer length (tokens, log scale)")
    set_title_subtitle(
        ax,
        "Answer lengths are far from parity across compared cells",
        "dot = median, whiskers = p10–p90; assistant cells shown; inserted pool is shared across framings",
    )
    savefig_paper(fig, "issue_2054/answer_length_parity", dir=MAIN_FIG)
    plt.close(fig)


def fig_caphit(fits: list[dict]) -> None:
    """Cap-hit censoring + equalize-down companion refits.

    Round 2: also plots the language-drift-excluded chat-base refit
    (/tmp/issue2054_drift_refit.json) beside the cap-hit bars when present.
    """
    ref = json.load(open("/tmp/issue2054_companion_refits.json"))
    try:
        ref.update(json.load(open("/tmp/issue2054_drift_refit.json")))
    except FileNotFoundError:
        pass
    committed = {}
    for d in fits:
        committed[
            d["cell"]
            if "cell" in d
            else f"{d['variant']}__{d['condition']}__{d['form']}__{d['model']}"
        ] = (
            d["arm_reports"]["context"]["pooled"]["r2_ambient_mean"],
            [x["r2_ambient"] for x in d["arm_reports"]["context"]["per_fold"]],
        )
    bt_full = committed[
        "conversation_paired_stories_assistant__on_policy__bare_text__qwen2.5-7b-instruct"
    ]
    ch_full = committed["conversation_paired_stories_assistant__on_policy__chat__qwen2.5-7b"]
    a_full = committed["conversation_paired_stories_assistant__inserted__chat__qwen2.5-7b-instruct"]

    def pooled(tag):
        return ref[tag]["pooled"]["r2_ambient_mean"]

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = [
        ("bare text\nall rows", bt_full[0], PAL[5]),
        ("bare text\ncapped\nremoved", pooled("bare_text_instr_excl_caphit"), PAL[2]),
        ("bare text\nrandom\nremoved", pooled("bare_text_instr_random_matched_n"), PAL[6]),
        ("chat base\nall rows", ch_full[0], PAL[5]),
        ("chat base\ncapped\nremoved", pooled("chat_base_excl_caphit"), PAL[2]),
        ("chat base\nrandom\nremoved", pooled("chat_base_random_matched_n"), PAL[6]),
    ]
    if "chat_base_excl_drift" in ref:
        bars += [
            ("chat base\ndrifted\nremoved", pooled("chat_base_excl_drift"), PAL[3]),
            ("chat base\nrandom\n(drift n)", pooled("chat_base_random_matched_n_drift"), PAL[6]),
        ]
    bars += [
        ("chat ins.\nall rows", a_full[0], PAL[4]),
        ("chat ins.\nsubsampled", pooled("cell_a_instr_sub8000"), PAL[6]),
    ]
    xs = np.arange(len(bars))
    ax.bar(xs, [b[1] for b in bars], color=[b[2] for b in bars], width=0.62)
    # per-fold points where available (index-robust: keyed by bar label)
    label_folds = {
        "bare text\nall rows": bt_full[1],
        "chat base\nall rows": ch_full[1],
        "chat ins.\nall rows": a_full[1],
    }
    label_tags = {
        "bare text\ncapped\nremoved": "bare_text_instr_excl_caphit",
        "bare text\nrandom\nremoved": "bare_text_instr_random_matched_n",
        "chat base\ncapped\nremoved": "chat_base_excl_caphit",
        "chat base\nrandom\nremoved": "chat_base_random_matched_n",
        "chat base\ndrifted\nremoved": "chat_base_excl_drift",
        "chat base\nrandom\n(drift n)": "chat_base_random_matched_n_drift",
        "chat ins.\nsubsampled": "cell_a_instr_sub8000",
    }
    for xi, (label, _, _) in enumerate(bars):
        pts = label_folds.get(label) or (
            ref[label_tags[label]].get("per_fold_r2") if label in label_tags else None
        )
        if pts:
            ax.scatter([xi] * len(pts), pts, s=10, color="#333333", zorder=4, alpha=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([b[0] for b in bars], fontsize=8)
    ax.set_ylabel("within-cell held-out R² (ambient, context arm)")
    set_title_subtitle(
        ax,
        "Removing runaway cap-hit generations, not the smaller n,\nmoves the flagged on-policy ceilings",
        "black dots = per-fold values; committed full-cell fits vs analyzer companion refits, shared fold map",
    )
    savefig_paper(fig, "issue_2054/caphit_censoring_refits", dir=MAIN_FIG)
    plt.close(fig)


def main() -> None:
    fits = load_fits()
    ladder = load_ladder()
    fig_hero_boundary_vs_prose(ladder)
    fig_boundary_vs_prose_pairs(ladder)
    fig_ceilings(fits)
    fig_ceilings_folds(fits)
    fig_twobytwo(fits)
    fig_ladder(ladder)
    fig_ks_pairs()
    fig_length_hist()
    try:
        fig_caphit(fits)
    except FileNotFoundError:
        print("companion refits not ready; skipping caphit figure")
    print("FIGS_DONE")


if __name__ == "__main__":
    main()
