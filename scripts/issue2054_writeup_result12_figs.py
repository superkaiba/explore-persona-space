#!/usr/bin/env python
"""Writeup Result 1 + 2 figures over #2054's 9-rung transfer-ladder outputs.

Render-only: everything here reads already-computed ladder rung JSONs (0 GPU-h,
no refits). Data-loading follows scripts/issue2054_framing_character_transfer_figs.py;
regenerate the merged row file with scripts/issue2054_fetch_ladder_rows.py first.

Six files under figures/issue_2054/writeup_result12/ (each raw R^2 + a
recovery-fraction twin):

1. ``result1_framing_rungs`` — the assistant's CHAT-template map re-used on the
   same assistant's other framings (bare text / story attributed-quote / story
   bare-label), solid = inserted (verbatim answers), dashed = on-policy.
   The framing-5 indirect ("full prose") render has NO chat-source ladder pairs
   by design (no inserted arm; on-policy cross-framing pairs excluded), so it
   cannot appear here.
2. ``result2a_story_to_characters`` — assistant-IN-STORY map -> each story
   character at the SAME bare-label boundary (persona changes, framing fixed).
3. ``result2b_chat_to_characters`` — assistant CHAT map -> each story character
   (framing AND persona change at once). The ladder enumerates chat sources
   only through the 2x2 INSERTED chat anchor, so the source condition is
   always inserted; the target condition still varies (solid/dashed).

The x-axis shows a user-selected SUBSET of the ladder — rungs 1 (direct),
6 (rotation), 7 (context re-map), 8 (answer re-map), evenly spaced with their
original rung numbers kept in the tick labels (rungs 2-5 and 9 dropped on user
request 2026-08-13; the full 9-rung renders live in the git history) — plus a
visually separated "paired fit" column at the right: the cross-render DIRECT
fit (source-render context -> target-render answer, ridge fit ON pairs), the
pair-informed predictability bound no ladder rung sees (rung maps never touch
cross-render pairs). Figure 1 reads it from the banked cross_render_fit.json
(assistant 4-way-intersection rows); Figure 2 from
cross_render_fit_characters.json (per-pair rows = the ladder's own row sets;
produced by `issue2054_cross_render_fit.py --figure2`). Recovery twins
normalize the paired fit by the same banked ladder per-target ceiling as the
rungs. Point fill follows the same well-posedness convention (hollow when the
paired fit's own n_train < d).

Reference lines (RAW-R^2 figures only — the recovery twins normalize per
target, so a single horizontal line has no meaning there):
- dotted, target color: the TARGET cell's own within-cell held-out R^2 (the
  ceiling a transfer into that target is trying to reach);
- dash-dot, target color: the TARGET cell's identity + bias baseline (predict
  the context vector + train-fold mean offset) — the floor a transfer must
  beat; banked per cell in the fits digest (``ctx.idb``);
- gray: the SOURCE map's own within-cell held-out R^2 (``ctx.r2`` of the
  source cell). Figures 1/2a use provenance-matched sources, so there are two
  gray lines per panel (solid = inserted source, dashed = on-policy source);
  figure 2b's source is the single inserted chat anchor (one solid line).

Marker fill encodes the body's well-posedness convention: rungs 7-8 refit a
d x d map on the target train fold, so at pairs whose equalized intersection
sits below the 4,480-conversation floor (0.8 x n < d = 3,584) those rungs are
descriptive-only and drawn HOLLOW; well-posed rungs are filled. Rungs 1-6
never refit at d x d and are always filled. Explained here + in the writeup
prose, deliberately NOT on the canvas (figures stay axes+ticks+legend only).

Usage:
  uv run python scripts/issue2054_fetch_ladder_rows.py   # stages + merges rows
  uv run python scripts/issue2054_writeup_result12_figs.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
LADDER_ROWS = Path("/tmp/issue2054_ladder_rows_merged.json")
FITS_DIGEST = Path("/tmp/issue2054_fits_digest.json")
FITS_DIGEST_BLOB = "eval_results/issue_2054/analyzer_companions/fits_digest.json"
P1345_JUDGE = REPO / "eval_results/issue_1345/judge_legs/judge_legs_summary.json"
CROSS_FIT = REPO / "eval_results/issue_2054/analyzer_companions/cross_render_fit.json"
CROSS_FIT_CHARS = (
    REPO / "eval_results/issue_2054/analyzer_companions/cross_render_fit_characters.json"
)
OUT_DIR = REPO / "figures/issue_2054/writeup_result12"

ASSIST = "conversation_paired_stories_assistant"
INSTRUCT = "qwen2.5-7b-instruct"
BASE = "qwen2.5-7b"

# d of the ambient basis. Rungs 7-8 refit a d x d map on the target train fold
# (n_train = 0.8 x intersection), so an intersection below 4,480 conversations
# puts those rungs in the under-determined, descriptive-only regime.
D_AMBIENT = 3584

# User-selected rung subset (original ladder numbering kept in the labels).
PLOT_RUNGS = ["1_direct", "6_rotation", "7_ctx_reparam", "8_ans_reparam"]
PLOT_RUNG_LABELS = ["1 direct\ntransfer", "6 rotation", "7 context\nre-map", "8 answer\nre-map"]
REPARAM_IDX = [PLOT_RUNGS.index(k) for k in ("7_ctx_reparam", "8_ans_reparam")]

# Cross-render DIRECT paired fit (source-render context -> target-render answer,
# fit on pairs — the pair-informed predictability bound no ladder rung sees).
# Rendered as one extra x-position past a thin separator; points offset left
# (inserted) / right (on-policy) of the column center so the two provenance
# points of one line don't overprint.
PAIRED_X = 4.7
SEP_X = 4.0
PAIRED_DX = {"inserted": -0.14, "on_policy": +0.14}

# One color = one meaning, matched to the committed #2054 figure family.
RENDER_STYLE = {
    "bare_text": ("bare text", "#1F4E9F", "o"),
    "attrib_quoted": ("story (attributed quote)", "#3FA577", "s"),
    "bare_label": ("story (bare label)", "#E08220", "^"),
}
# Maximally-distinct categorical colors (Wong colorblind-safe set), chosen to
# avoid Figure 1's framing hues (deep blue / green / orange) where possible —
# user feedback 2026-08-13 replaced the earlier AI-likeness blue ramp, whose
# lines were too close; AI-likeness ordering still rides the legend labels.
CHAR_COLOR = {
    "helios": "#000000",
    "wren": "#56B4E9",
    "vex": "#D55E00",
    "dana": "#CC79A7",
}
CHAR_DISPLAY = {"helios": "HELIOS", "wren": "Wren", "vex": "Vex", "dana": "Dana"}
CHARACTERS = ["helios", "wren", "vex", "dana"]

COND_LS = {"inserted": "-", "on_policy": "--"}
COND_ALPHA = {"inserted": 0.85, "on_policy": 0.45}
PANELS = [(BASE, "Qwen2.5-7B (base)"), (INSTRUCT, "Qwen2.5-7B-Instruct")]
C_SOURCE = "#7A7A7A"  # gray — source map's own R^2 reference


def _rows() -> list[dict]:
    if not LADDER_ROWS.exists():
        raise SystemExit(
            f"{LADDER_ROWS} missing — run `uv run python scripts/issue2054_fetch_ladder_rows.py`"
        )
    rows = json.loads(LADDER_ROWS.read_text())
    assert rows, "merged ladder row file is empty"
    return rows


def _fits() -> dict[str, dict]:
    """Per-cell banked fit stats keyed by cell: own within-cell held-out R^2
    (``r2`` — the source-line / ceiling value) and the identity + bias
    baseline (``idb`` — the floor). Reads the staged digest, falling back to
    the committed blob on branch ``issue-2054`` (never hardcoded values)."""
    if FITS_DIGEST.exists():
        text = FITS_DIGEST.read_text()
    else:
        text = subprocess.run(
            ["git", "show", f"origin/issue-2054:{FITS_DIGEST_BLOB}"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    d = json.loads(text)
    out = {r["cell"]: {"r2": r["ctx"]["r2"], "idb": r["ctx"]["idb"]} for r in d["rows"]}
    assert out, "fits digest is empty"
    return out


def _ai_likeness() -> dict[str, float]:
    """Judge-scored AI-likeness of each character's OWN on-policy answers.

    Source: #1345 judge_legs_summary.json, leg ``ai_likeness``, instruct
    on-policy cells ``char_<name>_op`` (base-model ``_op_base`` scores order
    identically; reported in prose, not the legend).
    """
    d = json.loads(P1345_JUDGE.read_text())
    out = {}
    for c in d["legs"]["ai_likeness"]["cells"]:
        name = c["cell"]
        if name.startswith("char_") and name.endswith("_op"):
            out[name[len("char_") : -len("_op")]] = c["pooled"]["mean"]
    assert set(out) >= set(CHARACTERS), sorted(out)
    return out


def _pair(rows: list[dict], src: str, tgt: str) -> dict:
    m = [r for r in rows if r["src"] == src and r["tgt"] == tgt and r["arm"] == "context"]
    assert len(m) == 1, (src, tgt, len(m))
    return m[0]


def _paired_fig1() -> dict[tuple[str, str, str], dict]:
    """Banked cross-render direct fits for Figure 1, keyed (condition, model,
    target_form). Chat source; computed on the assistant grid's 4-way form
    intersection (n=11,901 inserted / 8,000 on-policy — a different, larger row
    set than the ladder pairs; stated in the writeup prose)."""
    d = json.loads(CROSS_FIT.read_text())
    out = {
        (c["condition"], c["model"], c["target_form"]): c
        for c in d["cells"]
        if not c["is_identity"]
    }
    assert len(out) == 12, len(out)
    return out


def _paired_fig2() -> dict[tuple[str, str, str], dict]:
    """Cross-render direct fits onto the character cells, keyed
    (group, model, character) with group = '2a__<cond>' / '2b__<target cond>'.
    Computed on per-PAIR conv intersections (the ladder's own row convention)."""
    d = json.loads(CROSS_FIT_CHARS.read_text())
    out = {(c["group"], c["model"], c["character"]): c for c in d["cells"]}
    assert len(out) == 32, len(out)
    return out


def _draw_paired(ax, y: float, color: str, marker: str, cond: str, well_posed: bool) -> None:
    """One paired-fit point in the separated right-hand column (line identity via
    color+marker; provenance via the left/right offset; fill = well-posedness of
    the paired fit itself, same hollow convention as rungs 7-8)."""
    x = PAIRED_X + PAIRED_DX[cond]
    if well_posed:
        ax.scatter(
            [x],
            [y],
            marker=marker,
            s=24,
            facecolors=color,
            edgecolors=color,
            linewidths=0.5,
            zorder=4,
        )
    else:
        ax.scatter(
            [x],
            [y],
            marker=marker,
            s=30,
            facecolors="white",
            edgecolors=color,
            linewidths=1.3,
            zorder=4,
        )


def _series(pair: dict, recovery: bool) -> list[float]:
    ys = [pair["rungs"][k] for k in PLOT_RUNGS]
    if recovery:
        assert pair["ceiling"], (pair["src"], pair["tgt"])
        ys = [y / pair["ceiling"] for y in ys]
    return ys


def _hollow_mask(pair: dict) -> list[bool]:
    """True at rungs whose fit is descriptive-only (n_train < d) for this pair."""
    under = 0.8 * pair["n"] < D_AMBIENT
    return [under and i in REPARAM_IDX for i in range(len(PLOT_RUNGS))]


def _draw_pair(ax, pair: dict, color: str, ls: str, marker: str, recovery: bool) -> None:
    xs = list(range(len(PLOT_RUNGS)))
    ys = _series(pair, recovery)
    hollow = _hollow_mask(pair)
    ax.plot(xs, ys, color=color, ls=ls, lw=1.7, zorder=3)
    filled = [i for i in xs if not hollow[i]]
    ax.scatter(
        [xs[i] for i in filled],
        [ys[i] for i in filled],
        marker=marker,
        s=24,
        facecolors=color,
        edgecolors=color,
        linewidths=0.5,
        zorder=4,
    )
    open_ = [i for i in xs if hollow[i]]
    if open_:
        ax.scatter(
            [xs[i] for i in open_],
            [ys[i] for i in open_],
            marker=marker,
            s=30,
            facecolors="white",
            edgecolors=color,
            linewidths=1.3,
            zorder=4,
        )


def _draw_target_refs(ax, ceiling: float, idb: float, color: str, cond: str) -> None:
    """Per-target reference lines on RAW figures: dotted ceiling (the target
    cell's own R^2) and dash-dot identity+bias floor."""
    a = COND_ALPHA[cond]
    ax.axhline(ceiling, color=color, ls=":", lw=1.0, alpha=a, zorder=2)
    ax.axhline(idb, color=color, ls="-.", lw=1.0, alpha=a, zorder=2)


def _style_axis(ax, title: str, ylab: bool, recovery: bool, paired: bool = True) -> None:
    ax.set_title(title)
    if paired:
        ax.set_xticks(list(range(len(PLOT_RUNGS))) + [PAIRED_X])
        ax.set_xticklabels(PLOT_RUNG_LABELS + ["paired\nfit"], fontsize=8)
        ax.axvline(SEP_X, color="#BBBBBB", lw=0.8, zorder=1)
        ax.set_xlim(-0.4, PAIRED_X + 0.55)
    else:
        ax.set_xticks(range(len(PLOT_RUNGS)))
        ax.set_xticklabels(PLOT_RUNG_LABELS, fontsize=8)
        ax.set_xlim(-0.4, len(PLOT_RUNGS) - 0.6)
    ax.set_xlabel("transfer rung (more adaptation allowed →)")
    if ylab:
        ax.set_ylabel(
            "fraction of target ceiling recovered"
            if recovery
            else "held-out $R^2$ (context arm, layer 19)"
        )
    ax.axhline(0.0, color="#222222", lw=0.8, zorder=1)
    if recovery:
        ax.axhline(1.0, color="#888888", ls=":", lw=1.2, zorder=1)
    ax.grid(axis="y", alpha=0.25, lw=0.6)


def _set_ylim(axes, lo_data: float, hi_data: float) -> None:
    """Shared y-limits across panels from the tracked data + reference-line
    extremes (axhlines are excluded from matplotlib autoscale), clamped so an
    extreme point cannot crush the readable range."""
    lo = max(lo_data - 0.07, -2.0)
    hi = min(hi_data + 0.07, 1.15)
    for ax in axes:
        ax.set_ylim(lo, hi)


def _finish(fig, handles, labels, stem: str) -> None:
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(4, len(labels)),
        frameon=False,
        fontsize=8.0,
    )
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    savefig_paper(fig, stem, dir=OUT_DIR)
    plt.close(fig)
    print(f"[fig] {OUT_DIR / stem}.png")


def _ref_handles(labels_so_far: list, handles_so_far: list, two_source_lines: bool) -> None:
    """Legend entries for the raw-figure reference lines (shared shape)."""
    handles_so_far.append(Line2D([], [], color="#888888", ls=":", lw=1.2))
    labels_so_far.append("target cell's own $R^2$ (ceiling)")
    handles_so_far.append(Line2D([], [], color="#888888", ls="-.", lw=1.2))
    labels_so_far.append("identity + bias floor (per target)")
    handles_so_far.append(Line2D([], [], color=C_SOURCE, ls="-", lw=2.0))
    labels_so_far.append(
        "source map's own $R^2$ (solid/dashed = provenance)"
        if two_source_lines
        else "source map's own $R^2$"
    )


# --------------------------------------------------------------------------- #
# Result 1 — assistant chat map re-used on the assistant's other framings
# --------------------------------------------------------------------------- #
def fig_result1(
    rows: list[dict], fits: dict[str, dict], paired: dict[tuple, dict], recovery: bool
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.9), sharey=True)
    vals: list[float] = []
    for ax, (model, title) in zip(axes, PANELS):
        for cond, ls in COND_LS.items():
            src = f"{ASSIST}__{cond}__chat__{model}"
            for form, (_, color, marker) in RENDER_STYLE.items():
                p = _pair(rows, src, f"{ASSIST}__{cond}__{form}__{model}")
                _draw_pair(ax, p, color, ls, marker, recovery)
                vals += _series(p, recovery)
                pf = paired[(cond, model, form)]
                y_pf = pf["cross_render_r2"] / p["ceiling"] if recovery else pf["cross_render_r2"]
                _draw_paired(ax, y_pf, color, marker, cond, pf["well_posed_ambient"])
                vals.append(y_pf)
                if not recovery:
                    tgt = f"{ASSIST}__{cond}__{form}__{model}"
                    _draw_target_refs(ax, p["ceiling"], fits[tgt]["idb"], color, cond)
                    vals += [p["ceiling"], fits[tgt]["idb"]]
            if not recovery:
                # the SOURCE map's own within-cell R^2 (provenance-matched source)
                ax.axhline(fits[src]["r2"], color=C_SOURCE, ls=ls, lw=2.0, alpha=0.9, zorder=2)
                vals.append(fits[src]["r2"])
        _style_axis(ax, title, ylab=ax is axes[0], recovery=recovery)
    _set_ylim(axes, min(vals), max(vals))

    handles = [
        Line2D([], [], color=c, marker=m, ls="-", lw=1.7, ms=5) for _, c, m in RENDER_STYLE.values()
    ]
    labels = [f"chat → {lab}" for lab, _, _ in RENDER_STYLE.values()]
    handles += [
        Line2D([], [], color="#333333", ls="-", lw=1.7),
        Line2D([], [], color="#333333", ls="--", lw=1.7),
    ]
    labels += ["inserted (verbatim answers)", "on-policy"]
    if not recovery:
        _ref_handles(labels, handles, two_source_lines=True)
    fig.suptitle("Assistant chat-template map re-used on the assistant's other framings")
    _finish(fig, handles, labels, f"result1_framing_rungs{'_recovery' if recovery else ''}")


# --------------------------------------------------------------------------- #
# Result 2 — character transfer (user excluded), bare-label story boundary
# --------------------------------------------------------------------------- #
def fig_result2(
    rows: list[dict],
    fits: dict[str, dict],
    paired: dict[tuple, dict],
    chat_source: bool,
    recovery: bool,
) -> None:
    """chat_source=False: assistant-in-story -> character (persona only).
    chat_source=True: assistant-chat -> character (framing + persona; the
    ladder's chat anchor is INSERTED-only, so the source condition is fixed)."""
    ail = _ai_likeness()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.9), sharey=True)
    vals: list[float] = []
    grp_fig = "2b" if chat_source else "2a"
    for ax, (model, title) in zip(axes, PANELS):
        drawn_sources: set[str] = set()
        for cond, ls in COND_LS.items():
            src = (
                f"{ASSIST}__inserted__chat__{model}"
                if chat_source
                else f"{ASSIST}__{cond}__bare_label__{model}"
            )
            for ch in CHARACTERS:
                tgt = f"char_{ch}__{cond}__bare_label__{model}"
                p = _pair(rows, src, tgt)
                _draw_pair(ax, p, CHAR_COLOR[ch], ls, "o", recovery)
                vals += _series(p, recovery)
                pf = paired[(f"{grp_fig}__{cond}", model, f"char_{ch}")]
                y_pf = pf["cross_render_r2"] / p["ceiling"] if recovery else pf["cross_render_r2"]
                _draw_paired(ax, y_pf, CHAR_COLOR[ch], "o", cond, pf["well_posed_ambient"])
                vals.append(y_pf)
                if not recovery:
                    _draw_target_refs(ax, p["ceiling"], fits[tgt]["idb"], CHAR_COLOR[ch], cond)
                    vals += [p["ceiling"], fits[tgt]["idb"]]
            if not recovery and src not in drawn_sources:
                # figure 2a: provenance-matched sources (two gray lines);
                # figure 2b: the single inserted chat anchor (one solid line).
                src_ls = "-" if chat_source else ls
                ax.axhline(fits[src]["r2"], color=C_SOURCE, ls=src_ls, lw=2.0, alpha=0.9, zorder=2)
                vals.append(fits[src]["r2"])
                drawn_sources.add(src)
        _style_axis(ax, title, ylab=ax is axes[0], recovery=recovery)
    _set_ylim(axes, min(vals), max(vals))

    handles = [
        Line2D([], [], color=CHAR_COLOR[ch], marker="o", ls="-", lw=1.7, ms=5) for ch in CHARACTERS
    ]
    labels = [f"{CHAR_DISPLAY[ch]} (AI-likeness {ail[ch]:.0f})" for ch in CHARACTERS]
    handles += [
        Line2D([], [], color="#333333", ls="-", lw=1.7),
        Line2D([], [], color="#333333", ls="--", lw=1.7),
    ]
    labels += ["inserted (verbatim answers)", "on-policy"]
    if not recovery:
        _ref_handles(labels, handles, two_source_lines=not chat_source)
    fig.suptitle(
        "Assistant chat map re-used on each story character (bare-label boundary)"
        if chat_source
        else "Assistant-in-story map re-used on each story character (bare-label boundary)"
    )
    stem = "result2b_chat_to_characters" if chat_source else "result2a_story_to_characters"
    _finish(fig, handles, labels, f"{stem}{'_recovery' if recovery else ''}")


# --------------------------------------------------------------------------- #
# wr* writeup-doc variants (fixed filenames — an assembler embeds these paths).
# Same banked data re-sliced to the writeup's literal plot descriptions: the 4
# tiers + dotted own-map reference lines ONLY (no identity+bias floors, no
# paired-fit column, y tight to the data band). wr3/wr6 (pooled-map vs own-map)
# render once eval_results/issue_2054/specialization_ladder/pooled_tier_ladder.json
# lands — see fig_wr36 gating in main().
# --------------------------------------------------------------------------- #

# Short character descriptions for the wr4/wr5 legends — quoted (lightly
# shortened) from the actual data-generation configs: the story-generation
# prompt (scripts/issue1345_gen_stories_paired.py) interpolates each
# character's description verbatim from scripts/issue1310_common.py::PERSONAS.
WR_CHAR_DESC = {
    "helios": "HELIOS — a calm, precise artificial intelligence",
    "wren": "Wren — a warm, endlessly helpful assistant",
    "dana": "Dana — an ordinary everyday person",
    "vex": "Vex — a theatrical, scheming villain",
}


def _wr_ref_handles(labels: list, handles: list) -> None:
    handles.append(Line2D([], [], color="#888888", ls=":", lw=1.2))
    labels.append("target's own-map $R^2$ (dotted)")
    handles.append(Line2D([], [], color=C_SOURCE, ls=":", lw=1.8))
    labels.append("source's own-map $R^2$ (gray)")


def fig_wr1(rows: list[dict], fits: dict[str, dict]) -> None:
    """wr1: chat map -> the assistant's other framings; tiers + dotted own-map
    lines only, y tight to the data band."""
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharey=True)
    vals: list[float] = []
    for ax, (model, title) in zip(axes, PANELS):
        for cond, ls in COND_LS.items():
            src = f"{ASSIST}__{cond}__chat__{model}"
            for form, (_, color, marker) in RENDER_STYLE.items():
                p = _pair(rows, src, f"{ASSIST}__{cond}__{form}__{model}")
                _draw_pair(ax, p, color, ls, marker, recovery=False)
                vals += _series(p, recovery=False)
                ax.axhline(
                    p["ceiling"], color=color, ls=":", lw=1.0, alpha=COND_ALPHA[cond], zorder=2
                )
                vals.append(p["ceiling"])
            ax.axhline(
                fits[src]["r2"], color=C_SOURCE, ls=":", lw=1.8, alpha=COND_ALPHA[cond], zorder=2
            )
            vals.append(fits[src]["r2"])
        _style_axis(ax, title, ylab=ax is axes[0], recovery=False, paired=False)
    _set_ylim(axes, min(vals), max(vals))

    handles = [
        Line2D([], [], color=c, marker=m, ls="-", lw=1.7, ms=5) for _, c, m in RENDER_STYLE.values()
    ]
    labels = [f"chat → {lab}" for lab, _, _ in RENDER_STYLE.values()]
    handles += [
        Line2D([], [], color="#333333", ls="-", lw=1.7),
        Line2D([], [], color="#333333", ls="--", lw=1.7),
    ]
    labels += ["inserted (verbatim answers)", "on-policy"]
    _wr_ref_handles(labels, handles)
    fig.suptitle("Assistant chat-template map re-used on the assistant's other framings")
    _finish(fig, handles, labels, "wr1_framing_tiers")


def fig_wr2(rows: list[dict]) -> None:
    """wr2: ONE panel — per (framing x provenance) the instruct-minus-base gap
    in transferred R^2 at each tier (positive = instruct transfers better)."""
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.4))
    vals: list[float] = []
    for cond, ls in COND_LS.items():
        for form, (_, color, marker) in RENDER_STYLE.items():
            ys = []
            for model in (INSTRUCT, BASE):
                src = f"{ASSIST}__{cond}__chat__{model}"
                p = _pair(rows, src, f"{ASSIST}__{cond}__{form}__{model}")
                ys.append(_series(p, recovery=False))
            delta = [yi - yb for yi, yb in zip(*ys)]
            xs = list(range(len(PLOT_RUNGS)))
            ax.plot(xs, delta, color=color, ls=ls, lw=1.7, marker=marker, ms=5, zorder=3)
            vals += delta
    _style_axis(
        ax, "Instruct − base gap in chat-map transfer", ylab=False, recovery=False, paired=False
    )
    ax.set_ylabel("held-out $R^2$, instruct − base\n(positive = instruct transfers better)")
    _set_ylim([ax], min(vals), max(vals))

    handles = [
        Line2D([], [], color=c, marker=m, ls="-", lw=1.7, ms=5) for _, c, m in RENDER_STYLE.values()
    ]
    labels = [f"chat → {lab}" for lab, _, _ in RENDER_STYLE.values()]
    handles += [
        Line2D([], [], color="#333333", ls="-", lw=1.7),
        Line2D([], [], color="#333333", ls="--", lw=1.7),
    ]
    labels += ["inserted (verbatim answers)", "on-policy"]
    _finish(fig, handles, labels, "wr2_framing_tiers_delta")


def fig_wr45(rows: list[dict], fits: dict[str, dict], chat_source: bool) -> None:
    """wr4 (story source) / wr5 (chat source): character transfer, tiers +
    dotted own-map lines only; legend labels describe each character from the
    data-generation configs; hollow = descriptive-only kept."""
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharey=True)
    vals: list[float] = []
    for ax, (model, title) in zip(axes, PANELS):
        drawn_sources: set[str] = set()
        for cond, ls in COND_LS.items():
            src = (
                f"{ASSIST}__inserted__chat__{model}"
                if chat_source
                else f"{ASSIST}__{cond}__bare_label__{model}"
            )
            for ch in CHARACTERS:
                tgt = f"char_{ch}__{cond}__bare_label__{model}"
                p = _pair(rows, src, tgt)
                _draw_pair(ax, p, CHAR_COLOR[ch], ls, "o", recovery=False)
                vals += _series(p, recovery=False)
                ax.axhline(
                    p["ceiling"],
                    color=CHAR_COLOR[ch],
                    ls=":",
                    lw=1.0,
                    alpha=COND_ALPHA[cond],
                    zorder=2,
                )
                vals.append(p["ceiling"])
            if src not in drawn_sources:
                src_ls_alpha = 0.85 if chat_source else COND_ALPHA[cond]
                ax.axhline(
                    fits[src]["r2"], color=C_SOURCE, ls=":", lw=1.8, alpha=src_ls_alpha, zorder=2
                )
                vals.append(fits[src]["r2"])
                drawn_sources.add(src)
        _style_axis(ax, title, ylab=ax is axes[0], recovery=False, paired=False)
    _set_ylim(axes, min(vals), max(vals))

    handles = [
        Line2D([], [], color=CHAR_COLOR[ch], marker="o", ls="-", lw=1.7, ms=5) for ch in CHARACTERS
    ]
    labels = [WR_CHAR_DESC[ch] for ch in CHARACTERS]
    handles += [
        Line2D([], [], color="#333333", ls="-", lw=1.7),
        Line2D([], [], color="#333333", ls="--", lw=1.7),
    ]
    labels += ["inserted (verbatim answers)", "on-policy"]
    _wr_ref_handles(labels, handles)
    fig.suptitle(
        "Assistant chat map re-used on each story character (bare-label boundary)"
        if chat_source
        else "Assistant-in-story map re-used on each story character (bare-label boundary)"
    )
    _finish(
        fig,
        handles,
        labels,
        "wr5_chat_to_characters_desc" if chat_source else "wr4_story_to_characters_desc",
    )


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--set",
        choices=["full", "wr", "all"],
        default="all",
        help="full = the 6 floors+paired-fit variants; wr = the writeup-doc wr* files",
    )
    args = ap.parse_args()

    set_paper_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _rows()
    fits = _fits()
    if args.set in ("wr", "all"):
        fig_wr1(rows, fits)
        fig_wr2(rows)
        fig_wr45(rows, fits, chat_source=False)
        fig_wr45(rows, fits, chat_source=True)
    if args.set in ("full", "all"):
        paired1 = _paired_fig1()
        paired2 = _paired_fig2()
        for recovery in (False, True):
            fig_result1(rows, fits, paired1, recovery)
            fig_result2(rows, fits, paired2, chat_source=False, recovery=recovery)
            fig_result2(rows, fits, paired2, chat_source=True, recovery=recovery)


if __name__ == "__main__":
    main()
