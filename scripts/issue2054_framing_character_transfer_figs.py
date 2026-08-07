#!/usr/bin/env python
"""User-requested #2054 figures: framing transfer + assistant-vs-character transfer.

Two figures, both read-only over already-committed / already-harvested fit
artifacts (0 GPU-h, no refits):

1. ``framing_transfer_tiers`` — held-out R^2 at each of the 9 transfer-ladder
   rungs for the ASSISTANT chat-template map re-used on the same assistant's
   OTHER framings: bare text, story with an attributed-quote answer boundary,
   story with a bare-label answer boundary. Base and Instruct panels. All from
   the #2054 6k-row lattice, ``inserted`` answer provenance (the only
   provenance for which the assistant cross-framing rungs were enumerated).

2. ``framing_transfer_provenance`` — the provenance contrast the #2054 lattice
   does not carry: chat map -> assistant-in-story, inserted vs on-policy answer
   text, from the earlier #1345 5k-context character ladder (smaller n; plotted
   separately, never pooled with figure 1).

3. ``assistant_to_character_transfer`` — held-out R^2 at each rung for the
   assistant chat map re-used on each STORY CHARACTER's map, one line per
   character labelled with that character's judge-scored AI-likeness. Two
   controls on every panel: (a) the same chat map re-used on the ASSISTANT in
   story (framing change, persona held) and (b) the median character->character
   transfer (persona change, framing held).

Sources (all read-only):
  /tmp/issue2054_ladder_rows.json                              (816 pair x rung rows)
  /tmp/issue2054_fits_digest.json                              (per-cell ceilings + nulls)
  eval_results/issue_1345/story_char_ladder_fill/ladders.json  (provenance ladder)
  eval_results/issue_1345/judge_legs/judge_legs_summary.json   (AI-likeness)

Usage:
  uv run python scripts/issue2054_framing_character_transfer_figs.py
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
LADDER_ROWS = Path("/tmp/issue2054_ladder_rows.json")
FITS_DIGEST = Path("/tmp/issue2054_fits_digest.json")
P1345_LADDERS = REPO / "eval_results/issue_1345/story_char_ladder_fill/ladders.json"
P1345_JUDGE = REPO / "eval_results/issue_1345/judge_legs/judge_legs_summary.json"
OUT_DIR = REPO / "figures/issue_2054/framing_character_transfer"

ASSIST = "conversation_paired_stories_assistant"
INSTRUCT = "qwen2.5-7b-instruct"
BASE = "qwen2.5-7b"

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
    "7 context\nreparam",
    "8 answer\nreparam",
    "9 full\nA·M·B",
]

# One color = one meaning across BOTH figures in this set.
# Figure 1 + 2: target framing.  Figure 3: characters ride a separate ramp.
C_BARE_TEXT = "#1F4E9F"  # deep blue   — bare text
C_STORY_AQ = "#3FA577"  # forest green — story, attributed-quote boundary
C_STORY_BL = "#E08220"  # warm orange  — story, bare-label boundary
C_INSERTED = "#5A6975"  # slate        — inserted answer text
C_ONPOLICY = "#C0413B"  # warm red     — on-policy answer text
C_CTRL_SAME = "#3FA577"  # green  — control: assistant in story (persona held)
C_CTRL_CHAR = "#8C8C8C"  # grey   — control: character -> character

# Judge-scored AI-likeness (0-100, claude-sonnet-4-5, k=5 draws, n=~300 items
# per cell) of each character's OWN on-policy answers.  Source: #1345
# judge_legs_summary.json, leg `ai_likeness`, cells char_<name>_op.
CHARACTERS = ["helios", "wren", "vex", "dana"]
# Sequential ramp keyed to AI-likeness: darkest = most AI-like.
CHAR_COLOR = {
    "helios": "#08306B",
    "wren": "#2171B5",
    "vex": "#6BAED6",
    "dana": "#BDD7E7",
}


def _rows() -> list[dict]:
    return json.loads(LADDER_ROWS.read_text())


def _cell_nulls() -> dict[str, float]:
    d = json.loads(FITS_DIGEST.read_text())
    return {r["cell"]: r["ctx"]["null_p95"] for r in d["rows"]}


def _ai_likeness() -> dict[str, float]:
    d = json.loads(P1345_JUDGE.read_text())
    out = {}
    for c in d["legs"]["ai_likeness"]["cells"]:
        name = c["cell"]
        if name.endswith("_op"):  # instruct, on-policy answers
            out[name[len("char_") : -len("_op")]] = c["pooled"]["mean"]
    return out


def _pair(rows: list[dict], src: str, tgt: str, arm: str = "context") -> dict | None:
    m = [r for r in rows if r["src"] == src and r["tgt"] == tgt and r["arm"] == arm]
    assert len(m) <= 1, (src, tgt, arm, len(m))
    return m[0] if m else None


def _series(pair: dict) -> list[float]:
    return [pair["rungs"][k] for k in RUNGS]


def _style_rung_axis(ax, ylab: bool) -> None:
    ax.set_xticks(range(len(RUNGS)))
    ax.set_xticklabels(RUNG_LABELS, fontsize=7)
    # Rung 1 re-uses the source map verbatim; rung 9 refits a full A·M·B around
    # it. So the free parameters GROW left to right.
    ax.set_xlabel("transfer tier (more correction allowed →)")
    if ylab:
        ax.set_ylabel("held-out $R^2$ (context arm, layer 19)")
    ax.axhline(0.0, color="#222222", lw=0.8, zorder=1)
    ax.grid(axis="y", alpha=0.25, lw=0.6)


# --------------------------------------------------------------------------- #
# Figure 1 — assistant chat map re-used on the assistant's other framings
# --------------------------------------------------------------------------- #
def fig_framing_tiers() -> None:
    rows, nulls = _rows(), _cell_nulls()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), sharey=True)

    targets = [
        ("bare_text", "bare text", C_BARE_TEXT, "o", "-"),
        ("attrib_quoted", "in story (attributed-quote boundary)", C_STORY_AQ, "s", "-"),
        ("bare_label", "in story (bare-label boundary)", C_STORY_BL, "^", "-"),
    ]
    panels = [(axes[0], INSTRUCT, "Qwen2.5-7B-Instruct"), (axes[1], BASE, "Qwen2.5-7B (base)")]

    n_rows_seen = set()
    for ax, model, title in panels:
        src = f"{ASSIST}__inserted__chat__{model}"
        null_hi = max(nulls.get(f"{ASSIST}__inserted__{tf}__{model}", 0.0) for tf, *_ in targets)
        ax.axhspan(-abs(null_hi), abs(null_hi), color="#BBBBBB", alpha=0.35, lw=0, zorder=0)
        for tf, lab, color, marker, ls in targets:
            tgt = f"{ASSIST}__inserted__{tf}__{model}"
            p = _pair(rows, src, tgt)
            assert p is not None, (src, tgt)
            n_rows_seen.add(p["n"])
            ax.plot(
                range(len(RUNGS)),
                _series(p),
                color=color,
                marker=marker,
                ls=ls,
                ms=5,
                lw=1.8,
                label=f"chat → {lab}",
                zorder=3,
            )
            # the target map's OWN within-cell held-out R^2 = the ceiling this
            # transfer is trying to reach.
            ax.axhline(p["ceiling"], color=color, ls=":", lw=1.3, alpha=0.85, zorder=2)
        ax.set_title(title)
        _style_rung_axis(ax, ylab=ax is axes[0])

    assert len(n_rows_seen) == 1, n_rows_seen
    n = n_rows_seen.pop()
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [
        Line2D([], [], color="#444444", ls=":", lw=1.3),
        Line2D([], [], color="#BBBBBB", lw=6, alpha=0.5),
    ]
    labels += ["target map's own within-cell $R^2$ (ceiling)", "shuffle null (95th pct)"]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    fig.suptitle(
        "Re-using the assistant's chat-template context→answer map under a changed framing",
        fontsize=12.5,
    )
    fig.text(
        0.5,
        0.895,
        f"#2054 lattice · inserted answer text · n={n:,} paired rows · read = boundary before the "
        "answer begins → mean over answer tokens",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.88))
    savefig_paper(fig, "framing_transfer_tiers", dir=OUT_DIR)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 2 — the provenance contrast (separate run, smaller n, NEVER pooled)
# --------------------------------------------------------------------------- #
def fig_framing_provenance() -> None:
    d = json.loads(P1345_LADDERS.read_text())
    fig, ax = plt.subplots(figsize=(6.6, 4.6))

    legs = [
        ("chat<->story_inserted", "chat->story_inserted", "inserted answer text", C_INSERTED, "s"),
        (
            "chat<->story_onpolicy",
            "chat->story_onpolicy",
            "on-policy answer text",
            C_ONPOLICY,
            "o",
        ),
    ]
    ns = []
    for key, direction, lab, color, marker in legs:
        L = d["ladders"][key]
        v = L["ambient"][direction]
        ns.append(L["n_matched"])
        ax.plot(
            range(len(RUNGS)),
            [v["r2"][k][0] for k in RUNGS],
            color=color,
            marker=marker,
            ms=5,
            lw=1.8,
            label=f"chat → in story, {lab} (n={L['n_matched']:,})",
            zorder=3,
        )
        ax.axhline(v["ceiling_r2"][0], color=color, ls=":", lw=1.3, alpha=0.85, zorder=2)
        ax.plot(
            range(len(RUNGS)),
            [v["null_r2"][k][0] for k in RUNGS],
            color=color,
            ls="--",
            lw=1.0,
            alpha=0.5,
            zorder=2,
        )

    _style_rung_axis(ax, ylab=True)
    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Line2D([], [], color="#444444", ls=":", lw=1.3),
        Line2D([], [], color="#444444", ls="--", lw=1.0, alpha=0.6),
    ]
    labels += ["target map's own $R^2$ (ceiling)", "matched-capacity null"]
    ax.legend(handles, labels, fontsize=7.2, frameon=False, loc="center right")
    ax.set_title(
        "Answer provenance: chat map → assistant in story",
        fontsize=11.5,
        pad=40,
    )
    ax.text(
        0.5,
        1.015,
        f"#1345 5k-context ladder · Qwen2.5-7B-Instruct · layer 19 · n={min(ns):,}–{max(ns):,} "
        "— NOT pooled with the #2054 lattice (n=11,901)",
        transform=ax.transAxes,
        ha="center",
        fontsize=7.2,
        color="#555555",
    )
    ax.text(
        0.5,
        1.055,
        "the inserted leg's swing to $R^2$ = −1.4 at rung 1 is small-$n$ estimator instability, "
        "not a signal read",
        transform=ax.transAxes,
        ha="center",
        fontsize=7.2,
        color="#555555",
    )
    fig.tight_layout()
    savefig_paper(fig, "framing_transfer_provenance", dir=OUT_DIR)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 3 — assistant map -> each story character, ranked by AI-likeness
# --------------------------------------------------------------------------- #
def fig_assistant_to_character(form: str = "attrib_quoted") -> None:
    rows = _rows()
    ail = _ai_likeness()
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), sharey="row", sharex=True)

    panels = [
        (axes[0][0], INSTRUCT, "inserted", "Qwen2.5-7B-Instruct · inserted answer text"),
        (axes[0][1], INSTRUCT, "on_policy", "Qwen2.5-7B-Instruct · on-policy answer text"),
        (axes[1][0], BASE, "inserted", "Qwen2.5-7B (base) · inserted answer text"),
        (axes[1][1], BASE, "on_policy", "Qwen2.5-7B (base) · on-policy answer text"),
    ]

    for ax, model, cond, title in panels:
        src = f"{ASSIST}__inserted__chat__{model}"

        # control (a): same chat map -> the ASSISTANT in story. Framing changes,
        # persona does not. Only the `inserted` assistant-in-story cell has a
        # ladder pair, so this control is the inserted one on both columns.
        ctrl = _pair(rows, src, f"{ASSIST}__inserted__{form}__{model}")
        if ctrl is not None:
            # MATCHED-TARGET DISCLOSURE: only the `inserted` assistant-in-story
            # cell has a ladder pair, so on the on-policy column this control is
            # scored against a DIFFERENT target provenance than the character
            # lines beside it. Say so in the label rather than imply a match.
            ax.plot(
                range(len(RUNGS)),
                _series(ctrl),
                color=C_CTRL_SAME,
                ls="--",
                lw=2.0,
                marker="D",
                ms=4,
                label="control: → the ASSISTANT in story, persona held (inserted target)",
                zorder=4,
            )
            if cond != "inserted":
                ax.text(
                    0.02,
                    0.97,
                    "green control has an INSERTED target —\nprovenance differs from the character lines",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=6.8,
                    color=C_CTRL_SAME,
                )

        # control (b): median character -> character transfer at the same
        # framing + provenance. Persona changes, framing does not.
        cc = {k: [] for k in RUNGS}
        for a in CHARACTERS:
            for b in CHARACTERS:
                if a == b:
                    continue
                p = _pair(
                    rows, f"char_{a}__{cond}__{form}__{model}", f"char_{b}__{cond}__{form}__{model}"
                )
                if p is None:
                    continue
                for k in RUNGS:
                    cc[k].append(p["rungs"][k])
        n_cc = len(cc[RUNGS[0]])
        if n_cc:
            ax.plot(
                range(len(RUNGS)),
                [st.median(cc[k]) for k in RUNGS],
                color=C_CTRL_CHAR,
                ls="-.",
                lw=2.0,
                label=f"control: character → character, median of {n_cc} pairs",
                zorder=4,
            )

        panel_ns = []
        for ch in CHARACTERS:
            p = _pair(rows, src, f"char_{ch}__{cond}__{form}__{model}")
            assert p is not None, (src, ch, cond, model)
            panel_ns.append(p["n"])
            ax.plot(
                range(len(RUNGS)),
                _series(p),
                color=CHAR_COLOR[ch],
                marker="o",
                ms=5,
                lw=1.8,
                # n is per-PANEL, so it goes in the panel title, not the shared
                # legend (the legend is built from one panel's handles).
                label=f"→ {ch.capitalize()} (AI-likeness {ail[ch]:.0f})",
                zorder=3,
            )

        ax.set_title(
            f"{title}\nn={min(panel_ns):,}–{max(panel_ns):,} paired rows per character",
            fontsize=10.0,
        )
        _style_rung_axis(ax, ylab=ax in (axes[0][0], axes[1][0]))

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8.2)
    fig.suptitle(
        "Is the assistant a privileged persona? Assistant chat map re-used on each story character",
        fontsize=12.5,
    )
    form_lab = {"attrib_quoted": "attributed quote", "bare_label": "bare label"}[form]
    fig.text(
        0.5,
        0.952,
        f"#2054 lattice · story answer boundary = {form_lab} · AI-likeness = judge-scored "
        "(claude-sonnet-4-5, k=5, n≈300) on each character's OWN on-policy answers · "
        "user turn excluded · y-axis shared within a row only",
        ha="center",
        fontsize=8.2,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.945))
    savefig_paper(fig, f"assistant_to_character_transfer_{form}", dir=OUT_DIR)
    plt.close(fig)


def main() -> None:
    set_paper_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_framing_tiers()
    fig_framing_provenance()
    fig_assistant_to_character("attrib_quoted")
    fig_assistant_to_character("bare_label")
    print(f"wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
