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

Marker fill encodes the body's well-posedness convention: rungs 7-9 refit a
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
P1345_JUDGE = REPO / "eval_results/issue_1345/judge_legs/judge_legs_summary.json"
OUT_DIR = REPO / "figures/issue_2054/writeup_result12"

ASSIST = "conversation_paired_stories_assistant"
INSTRUCT = "qwen2.5-7b-instruct"
BASE = "qwen2.5-7b"

# d of the ambient basis. Rungs 7-9 refit a d x d map on the target train fold
# (n_train = 0.8 x intersection), so an intersection below 4,480 conversations
# puts those rungs in the under-determined, descriptive-only regime.
D_AMBIENT = 3584

RUNGS = [
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
RUNG_LABELS = [
    "1 direct\ntransfer",
    "2 context\noffset",
    "3 answer\noffset",
    "4 bias\nrefit",
    "5 global\nscale",
    "6 rotation",
    "7 context\nre-map",
    "8 answer\nre-map",
    "9 full\nrefit",
]
REPARAM_IDX = [RUNGS.index(k) for k in ("7_ctx_reparam", "8_ans_reparam", "9_full_AMB")]

# One color = one meaning, matched to the committed #2054 figure family.
RENDER_STYLE = {
    "bare_text": ("bare text", "#1F4E9F", "o"),
    "attrib_quoted": ("story (attributed quote)", "#3FA577", "s"),
    "bare_label": ("story (bare label)", "#E08220", "^"),
}
# Sequential ramp keyed to AI-likeness: darkest = most AI-like (same as the
# committed assistant_to_character figures).
CHAR_COLOR = {
    "helios": "#08306B",
    "wren": "#2171B5",
    "vex": "#6BAED6",
    "dana": "#BDD7E7",
}
CHAR_DISPLAY = {"helios": "HELIOS", "wren": "Wren", "vex": "Vex", "dana": "Dana"}
CHARACTERS = ["helios", "wren", "vex", "dana"]

COND_LS = {"inserted": "-", "on_policy": "--"}
PANELS = [(BASE, "Qwen2.5-7B (base)"), (INSTRUCT, "Qwen2.5-7B-Instruct")]


def _rows() -> list[dict]:
    if not LADDER_ROWS.exists():
        raise SystemExit(
            f"{LADDER_ROWS} missing — run `uv run python scripts/issue2054_fetch_ladder_rows.py`"
        )
    rows = json.loads(LADDER_ROWS.read_text())
    assert rows, "merged ladder row file is empty"
    return rows


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


def _series(pair: dict, recovery: bool) -> list[float]:
    ys = [pair["rungs"][k] for k in RUNGS]
    if recovery:
        assert pair["ceiling"], (pair["src"], pair["tgt"])
        ys = [y / pair["ceiling"] for y in ys]
    return ys


def _hollow_mask(pair: dict) -> list[bool]:
    """True at rungs whose fit is descriptive-only (n_train < d) for this pair."""
    under = 0.8 * pair["n"] < D_AMBIENT
    return [under and i in REPARAM_IDX for i in range(len(RUNGS))]


def _draw_pair(ax, pair: dict, color: str, ls: str, marker: str, recovery: bool) -> None:
    xs = list(range(len(RUNGS)))
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


def _style_axis(ax, title: str, ylab: bool, recovery: bool) -> None:
    ax.set_title(title)
    ax.set_xticks(range(len(RUNGS)))
    ax.set_xticklabels(RUNG_LABELS, fontsize=7)
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


def _clamp_ylim(axes, lo_floor: float, hi_cap: float) -> None:
    """Shared y-limits across panels, clamped so one extreme early rung cannot
    squash the readable range (off-scale points run off the axis; stated in
    the writeup prose, not on the canvas)."""
    lo = min(ax.get_ylim()[0] for ax in axes)
    hi = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(max(lo, lo_floor), min(hi, hi_cap))


def _finish(fig, axes, handles, labels, stem: str) -> None:
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(4, len(labels)),
        frameon=False,
        fontsize=8.0,
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    savefig_paper(fig, stem, dir=OUT_DIR)
    plt.close(fig)
    print(f"[fig] {OUT_DIR / stem}.png")


# --------------------------------------------------------------------------- #
# Result 1 — assistant chat map re-used on the assistant's other framings
# --------------------------------------------------------------------------- #
def fig_result1(rows: list[dict], recovery: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.9), sharey=True)
    for ax, (model, title) in zip(axes, PANELS):
        for cond, ls in COND_LS.items():
            src = f"{ASSIST}__{cond}__chat__{model}"
            for form, (_, color, marker) in RENDER_STYLE.items():
                p = _pair(rows, src, f"{ASSIST}__{cond}__{form}__{model}")
                _draw_pair(ax, p, color, ls, marker, recovery)
                if not recovery:
                    # the target cell's own within-cell R^2 = this transfer's ceiling
                    ax.axhline(
                        p["ceiling"],
                        color=color,
                        ls=":",
                        lw=1.0,
                        alpha=0.85 if cond == "inserted" else 0.45,
                        zorder=2,
                    )
        _style_axis(ax, title, ylab=ax is axes[0], recovery=recovery)
    _clamp_ylim(axes, lo_floor=-2.0, hi_cap=1.15)

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
        handles.append(Line2D([], [], color="#888888", ls=":", lw=1.2))
        labels.append("target cell's own $R^2$ (ceiling)")
    fig.suptitle("Assistant chat-template map re-used on the assistant's other framings")
    _finish(fig, axes, handles, labels, f"result1_framing_rungs{'_recovery' if recovery else ''}")


# --------------------------------------------------------------------------- #
# Result 2 — character transfer (user excluded), bare-label story boundary
# --------------------------------------------------------------------------- #
def fig_result2(rows: list[dict], chat_source: bool, recovery: bool) -> None:
    """chat_source=False: assistant-in-story -> character (persona only).
    chat_source=True: assistant-chat -> character (framing + persona; the
    ladder's chat anchor is INSERTED-only, so the source condition is fixed)."""
    ail = _ai_likeness()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.9), sharey=True)
    for ax, (model, title) in zip(axes, PANELS):
        for cond, ls in COND_LS.items():
            src = (
                f"{ASSIST}__inserted__chat__{model}"
                if chat_source
                else f"{ASSIST}__{cond}__bare_label__{model}"
            )
            for ch in CHARACTERS:
                p = _pair(rows, src, f"char_{ch}__{cond}__bare_label__{model}")
                _draw_pair(ax, p, CHAR_COLOR[ch], ls, "o", recovery)
                if not recovery:
                    ax.axhline(
                        p["ceiling"],
                        color=CHAR_COLOR[ch],
                        ls=":",
                        lw=1.0,
                        alpha=0.85 if cond == "inserted" else 0.45,
                        zorder=2,
                    )
        _style_axis(ax, title, ylab=ax is axes[0], recovery=recovery)
    _clamp_ylim(axes, lo_floor=-2.0, hi_cap=1.15)

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
        handles.append(Line2D([], [], color="#888888", ls=":", lw=1.2))
        labels.append("target cell's own $R^2$ (ceiling)")
    fig.suptitle(
        "Assistant chat map re-used on each story character (bare-label boundary)"
        if chat_source
        else "Assistant-in-story map re-used on each story character (bare-label boundary)"
    )
    stem = "result2b_chat_to_characters" if chat_source else "result2a_story_to_characters"
    _finish(fig, axes, handles, labels, f"{stem}{'_recovery' if recovery else ''}")


def main() -> None:
    set_paper_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _rows()
    for recovery in (False, True):
        fig_result1(rows, recovery)
        fig_result2(rows, chat_source=False, recovery=recovery)
        fig_result2(rows, chat_source=True, recovery=recovery)


if __name__ == "__main__":
    main()
