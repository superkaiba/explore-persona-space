#!/usr/bin/env python
"""User-requested #2054 figures: framing transfer + assistant-vs-character transfer.

Two figures, both read-only over already-computed ladder rung JSONs (0 GPU-h,
no refits here):

1. ``framing_transfer_tiers`` — held-out R^2 at each of the 9 transfer-ladder
   rungs for the ASSISTANT chat-template map re-used on the same assistant's
   OTHER framings: bare text, story with an attributed-quote answer boundary,
   story with a bare-label answer boundary. 2x2 panels: answer provenance
   (inserted / on-policy) x model (Instruct / base).

2. ``assistant_to_character_transfer_<form>`` — held-out R^2 at each rung for
   the assistant IN STORY map re-used on each STORY CHARACTER's map at the
   SAME boundary form, condition and model, one line per character labelled
   with that character's judge-scored AI-likeness. Framing is held fixed
   across the transfer, so the only thing that changes is the persona.

Both figures are built from ONE lattice (#2054), so every panel is directly
comparable; the earlier ``framing_transfer_provenance`` figure (a #1345
5k-context stand-in for the on-policy arm, which this lattice did not yet
carry) is SUPERSEDED by figure 1's on-policy row and has been removed.

Figure 1 additionally carries the CROSS-RENDER fit as a dash-dot reference line
per target: a fresh d x d ridge fit directly on (chat context, target-render
answer) pairs of the same conversation. It is NOT a tenth rung — every rung
freezes the chat operator and feeds it target contexts, while this refits with
full d^2 capacity — so it upper-bounds how much the chat context DETERMINES the
target answer and can never support a shared-operator claim.

Sources (all read-only):
  /tmp/issue2054_ladder_rows_merged.json                       (892 pair x arm rows)
  /tmp/issue2054_fits_digest.json                              (per-cell ceilings + nulls)
  eval_results/issue_1345/judge_legs/judge_legs_summary.json   (AI-likeness)
  eval_results/issue_2054/analyzer_companions/cross_render_fit.json  (cross-render)

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
LADDER_ROWS = Path("/tmp/issue2054_ladder_rows_merged.json")
FITS_DIGEST = Path("/tmp/issue2054_fits_digest.json")
P1345_JUDGE = REPO / "eval_results/issue_1345/judge_legs/judge_legs_summary.json"
XRENDER = REPO / "eval_results/issue_2054/analyzer_companions/cross_render_fit.json"
OUT_DIR = REPO / "figures/issue_2054/framing_character_transfer"

ASSIST = "conversation_paired_stories_assistant"
INSTRUCT = "qwen2.5-7b-instruct"
BASE = "qwen2.5-7b"

# d of the ambient basis; a fit whose n_train falls below it is a
# regularization-limit read, not an ambient-basis one.
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
    "7 context\nreparam",
    "8 answer\nreparam",
    "9 full\nA·M·B",
]
# Rungs 7-9 refit a d x d map on the TARGET train fold; they are the only
# rungs whose well-posedness depends on n_train vs d.
REPARAM_RUNGS = {"7_ctx_reparam", "8_ans_reparam", "9_full_AMB"}
REPARAM_FIRST_IDX = RUNGS.index("7_ctx_reparam")

# One color = one meaning across BOTH figures.
C_BARE_TEXT = "#1F4E9F"  # deep blue    — bare text
C_STORY_AQ = "#3FA577"  # forest green — story, attributed-quote boundary
C_STORY_BL = "#E08220"  # warm orange  — story, bare-label boundary
C_CTRL_CHAR = "#8C8C8C"  # grey        — control: character -> character

# Judge-scored AI-likeness (0-100, claude-sonnet-4-5, k=5 draws, n~=300 items
# per cell) of each character's OWN on-policy answers. Source: #1345
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


def _cell_ceilings() -> dict[str, float]:
    """Each cell's OWN within-cell held-out R^2 (context arm).

    For a TARGET cell this is the ceiling a transfer is trying to reach (already
    carried per-pair as ``ceiling``); for the SOURCE cell it is how well that
    map does on its own data — the reference the transfer is measured against.
    """
    d = json.loads(FITS_DIGEST.read_text())
    return {r["cell"]: r["ctx"]["r2"] for r in d["rows"]}


def _cross_render() -> dict[tuple[str, str, str], dict]:
    """Cross-render fits keyed (condition, model, target_form).

    NOT a ladder rung: every rung freezes the source operator and feeds it TARGET
    contexts, whereas this fits a fresh d x d ridge directly on cross-render
    PAIRS (chat context, target-render answer of the SAME conversation). It has
    full d^2 capacity, so it is a PREDICTABILITY upper bound and can never
    support a shared-operator claim — see the figure caption.

    Raises if the grid is incomplete: a partial file would silently drop lines
    from panels rather than fail.
    """
    d = json.loads(XRENDER.read_text())
    out = {(c["condition"], c["model"], c["target_form"]): c for c in d["cells"]}
    assert len(out) == len(d["cells"]), "duplicate (condition, model, target_form) in cross-render"
    return out


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


def _style_rung_axis(ax, ylab: bool, xlab: bool) -> None:
    ax.set_xticks(range(len(RUNGS)))
    ax.set_xticklabels(RUNG_LABELS if xlab else [""] * len(RUNGS), fontsize=7)
    # Rung 1 re-uses the source map verbatim; rung 9 refits a full A·M·B around
    # it. So the free parameters GROW left to right.
    if xlab:
        ax.set_xlabel("transfer tier (more correction allowed →)")
    if ylab:
        ax.set_ylabel("held-out $R^2$ (context arm, layer 19)")
    ax.axhline(0.0, color="#222222", lw=0.8, zorder=1)
    ax.grid(axis="y", alpha=0.25, lw=0.6)


def _mark_reparam_underdetermined(ax, n_train: int) -> None:
    """Shade rungs 7-9 when the target train fold is smaller than the ambient
    dimension: those rungs refit a d x d map, so below d they are
    regularization-limit reads (dof-capped GCV ridge), not ambient fits."""
    ax.axvspan(
        REPARAM_FIRST_IDX - 0.5, len(RUNGS) - 0.5, color="#B03A2E", alpha=0.07, lw=0, zorder=0
    )
    ax.text(
        len(RUNGS) - 0.6,
        ax.get_ylim()[1],
        f"rungs 7–9: $n_{{train}}$≈{n_train:,} < $d$={D_AMBIENT:,}\nregularization-limit",
        ha="right",
        va="top",
        fontsize=6.4,
        color="#B03A2E",
    )


# --------------------------------------------------------------------------- #
# Figure 1 — assistant chat map re-used on the assistant's other framings
# --------------------------------------------------------------------------- #
def fig_framing_tiers() -> None:
    rows, nulls, xr = _rows(), _cell_nulls(), _cross_render()
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.4), sharey=True, sharex=True)

    targets = [
        ("bare_text", "bare text", C_BARE_TEXT, "o"),
        ("attrib_quoted", "in story (attributed-quote boundary)", C_STORY_AQ, "s"),
        ("bare_label", "in story (bare-label boundary)", C_STORY_BL, "^"),
    ]
    panels = [
        (axes[0][0], INSTRUCT, "inserted", "Qwen2.5-7B-Instruct · INSERTED (controlled arm)"),
        (axes[0][1], BASE, "inserted", "Qwen2.5-7B base · INSERTED (controlled arm)"),
        (axes[1][0], INSTRUCT, "on_policy", "Qwen2.5-7B-Instruct · ON-POLICY (joint arm)"),
        (axes[1][1], BASE, "on_policy", "Qwen2.5-7B base · ON-POLICY (joint arm)"),
    ]

    for ax, model, cond, title in panels:
        src = f"{ASSIST}__{cond}__chat__{model}"
        null_hi = max(nulls.get(f"{ASSIST}__{cond}__{tf}__{model}", 0.0) for tf, *_ in targets)
        ax.axhspan(-abs(null_hi), abs(null_hi), color="#BBBBBB", alpha=0.35, lw=0, zorder=0)
        ns, xr_ns = [], []
        for tf, lab, color, marker in targets:
            p = _pair(rows, src, f"{ASSIST}__{cond}__{tf}__{model}")
            assert p is not None, (src, tf)
            ns.append(p["n"])
            ax.plot(
                range(len(RUNGS)),
                _series(p),
                color=color,
                marker=marker,
                ms=5,
                lw=1.8,
                label=f"chat → {lab}",
                zorder=3,
            )
            # the target map's OWN within-cell held-out R^2 = the ceiling this
            # transfer is trying to reach.
            ax.axhline(p["ceiling"], color=color, ls=":", lw=1.3, alpha=0.85, zorder=2)
            # Cross-render fit: NOT a rung — a fresh d x d ridge fit directly on
            # (chat context, target-render answer) pairs. Full d^2 capacity, so
            # it upper-bounds how much of the target answer the CHAT context
            # determines at all, independent of any shared-operator question.
            c = xr[(cond, model, tf)]
            xr_ns.append(c["n_intersection"])
            ax.axhline(c["cross_render_r2"], color=color, ls="-.", lw=1.4, alpha=0.95, zorder=2)
        n_str = f"{min(ns):,}" if min(ns) == max(ns) else f"{min(ns):,}–{max(ns):,}"
        assert min(xr_ns) == max(xr_ns), (cond, model, xr_ns)  # one shared 4-way row set
        ax.set_title(
            f"{title}\nn={n_str} paired rows · cross-render n={xr_ns[0]:,} (4-way shared)",
            fontsize=9.5,
        )
        _style_rung_axis(
            ax, ylab=ax in (axes[0][0], axes[1][0]), xlab=ax in (axes[1][0], axes[1][1])
        )

    handles, labels = axes[0][0].get_legend_handles_labels()
    handles += [
        Line2D([], [], color="#444444", ls=":", lw=1.3),
        Line2D([], [], color="#444444", ls="-.", lw=1.4),
        Line2D([], [], color="#BBBBBB", lw=6, alpha=0.5),
    ]
    labels += [
        "target map's own within-cell $R^2$ (ceiling)",
        "cross-render fit: chat context → target answer (not a tier)",
        "shuffle null (95th pct)",
    ]
    # Bottom strip, stacked bottom-up: explanation text, then the legend above
    # it, then the axes (tight_layout rect). Keep these three in sync.
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.088),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    fig.text(
        0.5,
        0.008,
        "CONTROLLED arm (top row): the answer text is held fixed across framings, so a delta is "
        "attributable to framing.\nJOINT arm (bottom row): the answer is regenerated in each framing, "
        "so a delta mixes what is said with how it is encoded — it is NOT a framing effect.\n"
        "The dash-dot cross-render line is NOT a tier: every tier re-uses the FROZEN chat operator, "
        "while this refits a fresh $d\\times d$ ridge on (chat context, target answer) pairs.\n"
        "With full $d^2$ capacity it upper-bounds how much the chat context DETERMINES the target "
        "answer — it cannot show the two framings share an operator.",
        ha="center",
        va="bottom",
        fontsize=7.6,
        color="#555555",
    )
    fig.suptitle(
        "Re-using the assistant's chat-template context→answer map under a changed framing",
        fontsize=12.5,
    )
    fig.text(
        0.5,
        0.945,
        "#2054 lattice · read = boundary before the assistant starts speaking → mean over answer tokens · "
        f"every panel $n_{{train}}$ > $d$={D_AMBIENT:,} (ambient basis)",
        ha="center",
        fontsize=8.2,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.155, 1, 0.94))
    savefig_paper(fig, "framing_transfer_tiers", dir=OUT_DIR)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 2 — assistant IN STORY map -> each story character, ranked by AI-likeness
# --------------------------------------------------------------------------- #
def fig_assistant_to_character(form: str = "attrib_quoted") -> None:
    rows = _rows()
    ail = _ai_likeness()
    ceil = _cell_ceilings()
    # One color = one meaning, keyed to figure 1: green = story attributed
    # quote, orange = story bare label. The SOURCE map is the assistant at that
    # same boundary, so the reference line is drawn in the boundary's color.
    src_color = {"attrib_quoted": C_STORY_AQ, "bare_label": C_STORY_BL}[form]
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.8), sharey="row", sharex=True)

    panels = [
        (axes[0][0], INSTRUCT, "inserted", "Qwen2.5-7B-Instruct · inserted answer text"),
        (axes[0][1], BASE, "inserted", "Qwen2.5-7B (base) · inserted answer text"),
        (axes[1][0], INSTRUCT, "on_policy", "Qwen2.5-7B-Instruct · on-policy answer text"),
        (axes[1][1], BASE, "on_policy", "Qwen2.5-7B (base) · on-policy answer text"),
    ]

    for ax, model, cond, title in panels:
        # SOURCE is the assistant IN STORY at the SAME boundary form as every
        # target, so framing is held fixed and only the persona changes.
        src = f"{ASSIST}__{cond}__{form}__{model}"

        # control: median character -> character transfer at the same framing +
        # provenance. Persona changes, framing does not — same contrast as the
        # character lines, but with no assistant at either end.
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
        for ch in sorted(CHARACTERS, key=lambda c: -ail[c]):
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
                label=f"→ {ch.capitalize()} (AI-likeness {ail[ch]:.0f})",
                zorder=3,
            )

        # The SOURCE map's own within-cell held-out R^2: how well the
        # assistant-in-story map predicts its OWN answers. Every character line
        # below it is what that same map loses to the persona swap alone.
        src_r2 = ceil.get(src)
        assert src_r2 is not None, src
        ax.axhline(src_r2, color=src_color, ls="--", lw=1.6, alpha=0.9, zorder=2)
        ax.text(
            0.04,
            src_r2,
            f"assistant source map on its OWN data: {src_r2:.3f}",
            fontsize=6.8,
            color=src_color,
            va="bottom",
            ha="left",
        )

        n_train_lo = int(0.8 * min(panel_ns))
        ax.set_title(
            f"{title}\nn={min(panel_ns):,}–{max(panel_ns):,} paired rows per character",
            fontsize=9.5,
        )
        _style_rung_axis(
            ax, ylab=ax in (axes[0][0], axes[1][0]), xlab=ax in (axes[1][0], axes[1][1])
        )
        if n_train_lo < D_AMBIENT:
            _mark_reparam_underdetermined(ax, n_train_lo)

    handles, labels = axes[0][0].get_legend_handles_labels()
    handles += [Line2D([], [], color=src_color, ls="--", lw=1.6)]
    labels += ["assistant source map's own within-cell $R^2$"]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8.2)
    fig.suptitle(
        "Is the assistant a privileged persona? Assistant-in-story map re-used on each story character",
        fontsize=12.5,
    )
    form_lab = {"attrib_quoted": "attributed quote", "bare_label": "bare label"}[form]
    fig.text(
        0.5,
        0.945,
        f"#2054 lattice · story answer boundary = {form_lab}, HELD FIXED across every transfer "
        "(only the persona changes) · AI-likeness = judge-scored (claude-sonnet-4-5, k=5, n≈300) on "
        "each character's OWN on-policy answers · user turn excluded · y-axis shared within a row only",
        ha="center",
        fontsize=7.8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.935))
    savefig_paper(fig, f"assistant_to_character_transfer_{form}", dir=OUT_DIR)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 3 — assistant CHAT-template map -> each story character
# --------------------------------------------------------------------------- #
def fig_chat_to_character(form: str = "attrib_quoted") -> None:
    """Both axes changed at once: framing (chat -> story) AND persona.

    Figure 1 changes framing only (assistant -> assistant); figure 2 changes
    persona only (assistant-in-story -> character, framing held fixed). This
    one composes them, so the grey control here is figure 2's persona-only
    transfer at the SAME form and condition — the gap between the two is what
    the framing change costs on top of the persona swap.

    The source is ALWAYS assistant x chat x INSERTED: the ladder enumerates
    these pairs only through its 2x2 chat anchor, so an on-policy chat source
    has no such pair by construction (target condition still varies).
    """
    rows = _rows()
    ail = _ai_likeness()
    ceil = _cell_ceilings()
    tgt_color = {"attrib_quoted": C_STORY_AQ, "bare_label": C_STORY_BL}[form]
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.8), sharey="row", sharex=True)

    panels = [
        (axes[0][0], INSTRUCT, "inserted", "Qwen2.5-7B-Instruct · inserted answer text"),
        (axes[0][1], BASE, "inserted", "Qwen2.5-7B (base) · inserted answer text"),
        (axes[1][0], INSTRUCT, "on_policy", "Qwen2.5-7B-Instruct · on-policy answer text"),
        (axes[1][1], BASE, "on_policy", "Qwen2.5-7B (base) · on-policy answer text"),
    ]

    for ax, model, cond, title in panels:
        # The 2x2 chat anchor is the INSERTED chat cell for both target conditions.
        src = f"{ASSIST}__inserted__chat__{model}"

        # control: the SAME targets reached from the assistant IN STORY at this
        # form + condition (figure 2's persona-only transfer), median of 4.
        story_src = f"{ASSIST}__{cond}__{form}__{model}"
        cc = {k: [] for k in RUNGS}
        for ch in CHARACTERS:
            p = _pair(rows, story_src, f"char_{ch}__{cond}__{form}__{model}")
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
                label=f"control: assistant IN STORY → character, median of {n_cc}",
                zorder=4,
            )

        panel_ns, tgt_ceils = [], []
        for ch in sorted(CHARACTERS, key=lambda c: -ail[c]):
            p = _pair(rows, src, f"char_{ch}__{cond}__{form}__{model}")
            assert p is not None, (src, ch, cond, model)
            panel_ns.append(p["n"])
            tgt_ceils.append(p["ceiling"])
            ax.plot(
                range(len(RUNGS)),
                _series(p),
                color=CHAR_COLOR[ch],
                marker="o",
                ms=5,
                lw=1.8,
                label=f"→ {ch.capitalize()} (AI-likeness {ail[ch]:.0f})",
                zorder=3,
            )

        # Band spanning the four target cells' OWN within-cell R^2 — the range of
        # ceilings these transfers are trying to reach (per-character dotted lines
        # would be four more series for no extra information).
        ax.axhspan(min(tgt_ceils), max(tgt_ceils), color=tgt_color, alpha=0.16, lw=0, zorder=1)

        src_r2 = ceil.get(src)
        assert src_r2 is not None, src
        ax.axhline(src_r2, color="#6A3D9A", ls="--", lw=1.6, alpha=0.9, zorder=2)
        ax.text(
            0.04,
            src_r2,
            f"chat source map on its OWN data: {src_r2:.3f}",
            fontsize=6.8,
            color="#6A3D9A",
            va="bottom",
            ha="left",
        )

        n_train_lo = int(0.8 * min(panel_ns))
        ax.set_title(
            f"{title}\nn={min(panel_ns):,}–{max(panel_ns):,} paired rows per character",
            fontsize=9.5,
        )
        _style_rung_axis(
            ax, ylab=ax in (axes[0][0], axes[1][0]), xlab=ax in (axes[1][0], axes[1][1])
        )
        if n_train_lo < D_AMBIENT:
            _mark_reparam_underdetermined(ax, n_train_lo)

    handles, labels = axes[0][0].get_legend_handles_labels()
    handles += [
        Line2D([], [], color="#6A3D9A", ls="--", lw=1.6),
        Line2D([], [], color=tgt_color, lw=6, alpha=0.3),
    ]
    labels += [
        "chat source map's own within-cell $R^2$",
        "range of the 4 target cells' own $R^2$ (ceilings)",
    ]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8.2)
    fig.suptitle(
        "Changing framing AND persona at once: assistant chat-template map re-used on each story character",
        fontsize=12.5,
    )
    form_lab = {"attrib_quoted": "attributed quote", "bare_label": "bare label"}[form]
    fig.text(
        0.5,
        0.945,
        f"#2054 lattice · source = assistant × chat × INSERTED (the only chat anchor the ladder enumerates) · "
        f"story answer boundary = {form_lab} · AI-likeness = judge-scored (claude-sonnet-4-5, k=5, n≈300) on "
        "each character's OWN on-policy answers · user turn excluded · y-axis shared within a row only",
        ha="center",
        fontsize=7.6,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.935))
    savefig_paper(fig, f"chat_to_character_transfer_{form}", dir=OUT_DIR)
    plt.close(fig)


def main() -> None:
    set_paper_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_framing_tiers()
    fig_assistant_to_character("attrib_quoted")
    fig_assistant_to_character("bare_label")
    fig_chat_to_character("attrib_quoted")
    fig_chat_to_character("bare_label")
    print(f"wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
