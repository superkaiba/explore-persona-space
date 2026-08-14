#!/usr/bin/env python
"""Writeup-v2 figures for #2054 (user spec revision of 2026-08-14).

Render-only over banked artifacts (0 GPU-h, no refits) — supersedes NONE of the
earlier figure scripts (`issue2054_writeup_result12_figs.py` renders the
2026-08-13 spec; the user changed the figure specifications, so these render
fresh under `figures/issue_2054/writeup_v2/`).

Inputs
------
- /tmp/issue2054_ladder_rows_merged.json          (scripts/issue2054_fetch_ladder_rows.py)
- eval_results/issue_2054/specialization_ladder/pooled_tier_ladder.json
- data/issue_2054/ladderstage/issue2054_lattice/pool_rungs/percell_rungs/*__context.json
- eval_results/issue_2054/specialization_ladder/loco_pooled.json           (speaker LOCO)
- eval_results/issue_2054/specialization_ladder/loco_framing_pooled.json   (framing LOCO)

Transfer-tier vocabulary (user's 5 tiers -> ladder rungs):
  Direct          = 1_direct
  Bias only       = 4_bias_refit        (bias refit on the target train fold)
  Rotation + bias = 6_rotation          (Procrustes on centered clouds + train means)
  Context re-map  = 7_ctx_reparam
  Answer re-map   = 8_ans_reparam
Rungs 7/8 refit a d x d map on the target train fold: at pairs whose equalized
intersection has 0.8 x n < d = 3,584 they are descriptive-only and drawn HOLLOW
(explained in the writeup prose, never on the canvas).

Pooled-map tiers (figures 3/7):
  Pooled direct   = pooled_tier_ladder r2.pooled_direct  (banked pool_rungs m0)
  + per-cell bias = pool_rungs percell m1 (fold-mean)
  Rotation + bias = pooled_tier_ladder r2.rotation_bias
  Context re-map  = pooled_tier_ladder r2.ctx_remap
  Answer re-map   = pooled_tier_ladder r2.ans_remap
  Own map         = pooled_tier_ladder r2.own_map (the cell's banked ceiling)

Usage:
  uv run python scripts/issue2054_fetch_ladder_rows.py   # if /tmp rows are absent
  uv run python scripts/issue2054_writeup_v2_figs.py [--skip-loco-framing]
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parent.parent
ROWS_PATH = Path("/tmp/issue2054_ladder_rows_merged.json")
PTL_PATH = REPO / "eval_results/issue_2054/specialization_ladder/pooled_tier_ladder.json"
POOL_RUNGS_DIR = REPO / "data/issue_2054/ladderstage/issue2054_lattice/pool_rungs/percell_rungs"
LOCO_SPEAKER_PATH = REPO / "eval_results/issue_2054/specialization_ladder/loco_pooled.json"
LOCO_FRAMING_PATH = REPO / "eval_results/issue_2054/specialization_ladder/loco_framing_pooled.json"
NULLS_PATH = REPO / "eval_results/issue_2054/specialization_ladder/remap_pair_nulls.json"
OUT_DIR = REPO / "figures/issue_2054/writeup_v2"

ASSIST = "conversation_paired_stories_assistant"
INSTRUCT = "qwen2.5-7b-instruct"
BASE = "qwen2.5-7b"
D_AMBIENT = 3584
WELL_POSED_FLOOR = 4480  # 0.8 x 4480 = 3584 = d

TIERS = ["1_direct", "4_bias_refit", "6_rotation", "7_ctx_reparam", "8_ans_reparam"]
TIER_LABELS = ["Direct", "Bias\nonly", "Rotation\n+ bias", "Context\nre-map", "Answer\nre-map"]
REFIT_TIERS = {"7_ctx_reparam", "8_ans_reparam"}

POOL_TIERS = ["pooled_direct", "bias", "rotation_bias", "ctx_remap", "ans_remap"]
POOL_TIER_LABELS = [
    "Pooled\ndirect",
    "+ per-cell\nbias",
    "Rotation\n+ bias",
    "Context\nre-map",
    "Answer\nre-map",
]

FRAMING_COLOR = {
    "chat": "#444444",
    "bare_text": "#1f77b4",
    "bare_label": "#ff7f0e",
    "attrib_quoted": "#2ca02c",
}
FRAMING_LABEL = {
    "chat": "chat template",
    "bare_text": "bare text",
    "bare_label": "story, bare label",
    "attrib_quoted": "story, attributed quote",
}
CHAR_COLOR = {
    "wren": "#1f77b4",
    "helios": "#9467bd",
    "dana": "#8c564b",
    "vex": "#d62728",
    "assistant": "#444444",
}
CHAR_LABEL = {
    "wren": "Wren",
    "helios": "HELIOS",
    "dana": "Dana",
    "vex": "Vex",
    "assistant": "Assistant",
}


def parse_cell(key: str) -> dict:
    """`<identity>__<condition>__<framing>__<model>` -> dict (cell_c aware)."""
    ident, cond, framing, model = key.split("__")
    return {"identity": ident, "condition": cond, "framing": framing, "model": model}


def load_rows() -> list[dict]:
    rows = json.loads(ROWS_PATH.read_text())
    for r in rows:
        r["s"] = parse_cell(r["src"])
        r["t"] = parse_cell(r["tgt"])
    return [r for r in rows if r["arm"] == "context"]


def find_row(rows: list[dict], **kw) -> dict | None:
    """Unique row whose src/tgt fields match `s_*` / `t_*` kwargs."""
    hits = []
    for r in rows:
        ok = True
        for k, v in kw.items():
            side, field = k.split("_", 1)
            if r[side][field] != v:
                ok = False
                break
        if ok:
            hits.append(r)
    if not hits:
        return None
    assert len(hits) == 1, f"non-unique row for {kw}: {[(h['src'], h['tgt']) for h in hits]}"
    return hits[0]


def load_nulls() -> dict[tuple[str, str], dict]:
    """(src, tgt) -> shuffled-pair null summary; empty when not yet computed."""
    if not NULLS_PATH.exists():
        return {}
    payload = json.loads(NULLS_PATH.read_text())
    return {(u["src"], u["tgt"]): u["mean"] for u in payload["units"]}


def _draw_null_bands(ax, null_means: list[dict]) -> bool:
    """Shade the p95 range of the shuffled-pair nulls at the two re-map tiers.

    One band per tier position (x = 3 ctx re-map, x = 4 ans re-map) spanning
    min..max of the panel's per-pair null p95 values. Returns True if drawn.
    """
    if not null_means:
        return False
    for xi, key in ((3, "null_7_p95"), (4, "null_8_p95")):
        vals = [m[key] for m in null_means]
        lo, hi = min(vals), max(vals)
        if hi - lo < 0.008:  # keep a visible sliver when the nulls coincide
            pad = (0.008 - (hi - lo)) / 2
            lo, hi = lo - pad, hi + pad
        ax.fill_between([xi - 0.3, xi + 0.3], lo, hi, color="#8a8a8a", alpha=0.30, zorder=1)
    return True


NULL_BAND_HANDLE = Patch(
    facecolor="#8a8a8a", alpha=0.30, label="shuffled-pair null (p95 range, re-map tiers)"
)


def tier_values(row: dict) -> tuple[list[float], list[bool]]:
    """(values per TIERS, filled-marker flags — hollow when a refit rung is under-determined)."""
    vals = [row["rungs"][t] for t in TIERS]
    filled = [not (t in REFIT_TIERS and row["n"] < WELL_POSED_FLOOR) for t in TIERS]
    return vals, filled


def _plot_tier_line(ax, vals, filled, color, label=None, ls="-"):
    x = range(len(vals))
    ax.plot(x, vals, ls, color=color, label=label, lw=1.8, zorder=3)
    for xi, (v, f) in enumerate(zip(vals, filled)):
        ax.plot(
            xi,
            v,
            "o",
            color=color,
            markerfacecolor=color if f else "white",
            markeredgecolor=color,
            markersize=5.5,
            zorder=4,
        )


def _tier_axes(ax, labels, title, ylabel=None, zero_line=False):
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel)
    if zero_line:
        ax.axhline(0.0, color="#999999", lw=0.8, ls=":")


# ---------------------------------------------------------------- ladder figs


def fig_tier_grid(
    rows: list[dict],
    stem: str,
    pair_fn,
    series: list[str],
    color_map: dict,
    label_map: dict,
    row_specs: list[tuple[str, str]],
    src_ceiling_fn=None,
    nulls: dict | None = None,
):
    """2x2 grid: rows = row_specs (condition views), cols = instruct | instruct - base.

    `pair_fn(cond_key, model, series_key) -> row or None` supplies the ladder row.
    Absolute panels carry dotted target-ceiling lines (series color) and, when
    `src_ceiling_fn(cond_key, model)` is given, a gray dotted source-map ceiling.
    `nulls` (optional) adds gray shuffled-pair null p95 bands at the re-map tiers.
    """
    drew_nulls = False
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), sharex=True)
    for ri, (cond_key, cond_title) in enumerate(row_specs):
        ax_abs, ax_diff = axes[ri]
        panel_nulls: list[dict] = []
        for sk in series:
            r_i = pair_fn(cond_key, INSTRUCT, sk)
            r_b = pair_fn(cond_key, BASE, sk)
            if r_i is None:
                continue
            vals_i, filled_i = tier_values(r_i)
            _plot_tier_line(ax_abs, vals_i, filled_i, color_map[sk], label=label_map[sk])
            ax_abs.axhline(r_i["ceiling"], color=color_map[sk], lw=1.0, ls=":", alpha=0.8)
            if nulls and (r_i["src"], r_i["tgt"]) in nulls:
                panel_nulls.append(nulls[(r_i["src"], r_i["tgt"])])
            if r_b is not None:
                vals_b, filled_b = tier_values(r_b)
                diff = [a - b for a, b in zip(vals_i, vals_b)]
                both = [fa and fb for fa, fb in zip(filled_i, filled_b)]
                _plot_tier_line(ax_diff, diff, both, color_map[sk])
        drew_nulls = _draw_null_bands(ax_abs, panel_nulls) or drew_nulls
        if src_ceiling_fn is not None:
            sc = src_ceiling_fn(cond_key, INSTRUCT)
            if sc is not None:
                ax_abs.axhline(sc, color="#888888", lw=1.2, ls="--", alpha=0.9)
        _tier_axes(
            ax_abs,
            TIER_LABELS,
            f"{cond_title} — instruct",
            ylabel="held-out $R^2$ (context arm)",
        )
        _tier_axes(
            ax_diff,
            TIER_LABELS,
            f"{cond_title} — instruct − base",
            ylabel="$\\Delta R^2$ (instruct − base)",
            zero_line=True,
        )
    handles = [
        Line2D([0], [0], color=color_map[sk], marker="o", lw=1.8, label=label_map[sk])
        for sk in series
    ]
    handles.append(
        Line2D([0], [0], color="#333", lw=1.0, ls=":", label="target's own-map $R^2$ (dotted)")
    )
    if src_ceiling_fn is not None:
        handles.append(
            Line2D([0], [0], color="#888888", lw=1.2, ls="--", label="source's own-map $R^2$")
        )
    if drew_nulls:
        handles.append(NULL_BAND_HANDLE)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    savefig_paper(fig, stem, dir=OUT_DIR)
    plt.close(fig)


def fig_base_abs(
    rows: list[dict],
    stem: str,
    pair_fn,
    series: list[str],
    color_map: dict,
    label_map: dict,
    row_specs: list[tuple[str, str]],
    src_ceiling_fn=None,
    nulls: dict | None = None,
):
    """1x2: base-model absolute R^2 per tier (the other term of the difference)."""
    drew_nulls = False
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), sharex=True)
    for ci, (cond_key, cond_title) in enumerate(row_specs):
        ax = axes[ci]
        panel_nulls: list[dict] = []
        for sk in series:
            r_b = pair_fn(cond_key, BASE, sk)
            if r_b is None:
                continue
            vals, filled = tier_values(r_b)
            _plot_tier_line(ax, vals, filled, color_map[sk], label=label_map[sk])
            ax.axhline(r_b["ceiling"], color=color_map[sk], lw=1.0, ls=":", alpha=0.8)
            if nulls and (r_b["src"], r_b["tgt"]) in nulls:
                panel_nulls.append(nulls[(r_b["src"], r_b["tgt"])])
        drew_nulls = _draw_null_bands(ax, panel_nulls) or drew_nulls
        if src_ceiling_fn is not None:
            sc = src_ceiling_fn(cond_key, BASE)
            if sc is not None:
                ax.axhline(sc, color="#888888", lw=1.2, ls="--", alpha=0.9)
        _tier_axes(
            ax,
            TIER_LABELS,
            f"{cond_title} — base",
            ylabel="held-out $R^2$ (context arm)" if ci == 0 else None,
        )
    handles = [
        Line2D([0], [0], color=color_map[sk], marker="o", lw=1.8, label=label_map[sk])
        for sk in series
    ]
    handles.append(
        Line2D([0], [0], color="#333", lw=1.0, ls=":", label="target's own-map $R^2$ (dotted)")
    )
    if drew_nulls:
        handles.append(NULL_BAND_HANDLE)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    savefig_paper(fig, stem, dir=OUT_DIR)
    plt.close(fig)


# --------------------------------------------------------------- pooled figs


def load_pooled_units() -> dict[str, dict]:
    """cell key -> pooled-tier unit (context arm), with the m1 bias tier joined in."""
    ptl = json.loads(PTL_PATH.read_text())
    units = {u["cell"]: dict(u) for u in ptl["units"] if u["arm"] == "context"}
    for key, u in units.items():
        pr_path = POOL_RUNGS_DIR / f"{key}__context.json"
        pr = json.loads(pr_path.read_text())
        m1 = [f["metrics"]["m1"]["r2"] for f in pr["folds"]]
        u["r2"] = dict(u["r2"])
        u["r2"]["bias"] = sum(m1) / len(m1)
    return units


def fig_pooled_grid(
    units: dict[str, dict],
    stem: str,
    cell_fn,
    series: list[str],
    color_map: dict,
    label_map: dict,
):
    """2x2 grid (rows: on-policy / inserted; cols: instruct / base) of pooled-map tiers."""
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), sharex=True)
    for ri, (cond, cond_title) in enumerate([("on_policy", "on-policy"), ("inserted", "inserted")]):
        for ci, (model, m_title) in enumerate([(INSTRUCT, "instruct"), (BASE, "base")]):
            ax = axes[ri][ci]
            for sk in series:
                key = cell_fn(cond, model, sk)
                u = units.get(key)
                if u is None:
                    continue
                vals = [u["r2"][t] for t in POOL_TIERS]
                filled = [True] * len(vals)
                _plot_tier_line(ax, vals, filled, color_map[sk], label=label_map[sk])
                ax.axhline(u["r2"]["own_map"], color=color_map[sk], lw=1.0, ls=":", alpha=0.8)
            _tier_axes(
                ax,
                POOL_TIER_LABELS,
                f"{cond_title} — {m_title}",
                ylabel="held-out $R^2$ (context arm)" if ci == 0 else None,
            )
    handles = [
        Line2D([0], [0], color=color_map[sk], marker="o", lw=1.8, label=label_map[sk])
        for sk in series
    ]
    handles.append(Line2D([0], [0], color="#333", lw=1.0, ls=":", label="own-map $R^2$ (dotted)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    savefig_paper(fig, stem, dir=OUT_DIR)
    plt.close(fig)


# ----------------------------------------------------------------- LOCO figs


def fig_loco(
    loco_path: Path,
    units: dict[str, dict],
    stem: str,
    group_field: str,
    groups: list[str],
    group_labels: list[str],
    cell_filter,
    xlabel: str,
):
    """2x2 grid (rows: on-policy / inserted; cols: instruct / base).

    x = held-out group; markers per cell: LOCO (fit on everything else, applied
    frozen), matched full-pool, own-map ceiling.
    """
    loco = json.loads(loco_path.read_text())
    per = [u for u in loco["per_unit"] if u["arm"] == "context" and cell_filter(u)]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), sharex=True, sharey=True)
    series = [
        ("loco_r2", "#d62728", "fit on all settings but this one (held out), applied frozen"),
        ("full_pool_r2", "#1f77b4", "fit on all settings (incl. this one)"),
        ("ceiling", "#444444", "own-map $R^2$ (fit only on this setting)"),
    ]
    for ri, (cond, cond_title) in enumerate([("on_policy", "on-policy"), ("inserted", "inserted")]):
        for ci, (model_tok, m_title) in enumerate([("instruct", "instruct"), ("base", "base")]):
            ax = axes[ri][ci]
            for gi, g in enumerate(groups):
                cells = [
                    u
                    for u in per
                    if u.get(group_field) == g
                    and u["condition"] == cond
                    and (
                        u["model"].endswith("-instruct")
                        if model_tok == "instruct"
                        else not u["model"].endswith("-instruct")
                    )
                ]
                for u in cells:
                    ceil = units.get(u["cell"], {}).get("r2", {}).get("own_map")
                    for off, (field, color, _lbl) in zip((-0.18, 0.0, 0.18), series):
                        val = u["mean"][field] if field != "ceiling" else ceil
                        if val is None:
                            continue
                        ax.plot(gi + off, val, "o", color=color, markersize=6, zorder=3)
            ax.axhline(0.0, color="#999999", lw=0.8, ls=":")
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels(group_labels, fontsize=8)
            ax.set_title(f"{cond_title} — {m_title}", fontsize=10)
            if ci == 0:
                ax.set_ylabel("held-out $R^2$ (context arm)")
            if ri == 1:
                ax.set_xlabel(xlabel)
    handles = [Line2D([0], [0], color=c, marker="o", lw=0, label=lbl) for _f, c, lbl in series]
    fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    savefig_paper(fig, stem, dir=OUT_DIR)
    plt.close(fig)


# ----------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.replace("%", "%%"))
    ap.add_argument("--skip-loco-framing", action="store_true")
    args = ap.parse_args()

    set_paper_style("blog")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    units = load_pooled_units()
    nulls = load_nulls()

    story_forms = ["bare_text", "bare_label", "attrib_quoted"]

    # -- Result 1: assistant chat -> assistant other framings -----------------
    def r1_pair(cond, model, framing):
        return find_row(
            rows,
            s_identity=ASSIST,
            s_condition=cond,
            s_framing="chat",
            s_model=model,
            t_identity=ASSIST,
            t_condition=cond,
            t_framing=framing,
            t_model=model,
        )

    def r1_src_ceiling(cond, model):
        u = units.get(f"{ASSIST}__{cond}__chat__{model}")
        return None if u is None else u["r2"]["own_map"]

    row_specs = [("on_policy", "on-policy"), ("inserted", "inserted")]
    fig_tier_grid(
        rows,
        "r1_framing_tiers",
        r1_pair,
        story_forms,
        FRAMING_COLOR,
        FRAMING_LABEL,
        row_specs,
        src_ceiling_fn=r1_src_ceiling,
        nulls=nulls,
    )
    fig_base_abs(
        rows,
        "r1_framing_tiers_base",
        r1_pair,
        story_forms,
        FRAMING_COLOR,
        FRAMING_LABEL,
        row_specs,
        src_ceiling_fn=r1_src_ceiling,
        nulls=nulls,
    )

    def r1_pool_cell(cond, model, framing):
        return f"{ASSIST}__{cond}__{framing}__{model}"

    fig_pooled_grid(
        units,
        "r1_pooled_vs_own",
        r1_pool_cell,
        ["chat", "bare_text", "bare_label", "attrib_quoted"],
        FRAMING_COLOR,
        FRAMING_LABEL,
    )

    if not args.skip_loco_framing and LOCO_FRAMING_PATH.exists():
        fig_loco(
            LOCO_FRAMING_PATH,
            units,
            "r1_loco_framings",
            "group",
            ["chat", "bare_text", "bare_label", "attrib_quoted"],
            [
                FRAMING_LABEL[f].replace(", ", ",\n")
                for f in ["chat", "bare_text", "bare_label", "attrib_quoted"]
            ],
            cell_filter=lambda u: u["identity"] == ASSIST,
            xlabel="held-out framing (assistant cells shown)",
        )

    # -- Result 2: assistant story (bare label) -> characters -----------------
    chars = ["wren", "helios", "dana", "vex"]

    def r2_pair(cond, model, ch):
        return find_row(
            rows,
            s_identity=ASSIST,
            s_condition=cond,
            s_framing="bare_label",
            s_model=model,
            t_identity=f"char_{ch}",
            t_condition=cond,
            t_framing="bare_label",
            t_model=model,
        )

    def r2_src_ceiling(cond, model):
        u = units.get(f"{ASSIST}__{cond}__bare_label__{model}")
        return None if u is None else u["r2"]["own_map"]

    fig_tier_grid(
        rows,
        "r2_character_tiers",
        r2_pair,
        chars,
        CHAR_COLOR,
        CHAR_LABEL,
        row_specs,
        src_ceiling_fn=r2_src_ceiling,
        nulls=nulls,
    )
    fig_base_abs(
        rows,
        "r2_character_tiers_base",
        r2_pair,
        chars,
        CHAR_COLOR,
        CHAR_LABEL,
        row_specs,
        src_ceiling_fn=r2_src_ceiling,
        nulls=nulls,
    )

    def r2_pool_cell(cond, model, ch):
        return f"char_{ch}__{cond}__bare_label__{model}"

    fig_pooled_grid(
        units,
        "r2_pooled_vs_own_characters",
        r2_pool_cell,
        chars,
        CHAR_COLOR,
        CHAR_LABEL,
    )

    fig_loco(
        LOCO_SPEAKER_PATH,
        units,
        "r2_loco_characters",
        "speaker",
        ["assistant", "wren", "helios", "dana", "vex"],
        [CHAR_LABEL[c] for c in ["assistant", "wren", "helios", "dana", "vex"]],
        cell_filter=lambda u: u["framing"] == "bare_label",
        xlabel="held-out character (bare-label story cells shown)",
    )

    # -- Result 3: assistant chat (inserted source) -> characters in story ----
    def r3_pair(tcond, model, ch):
        return find_row(
            rows,
            s_identity=ASSIST,
            s_condition="inserted",
            s_framing="chat",
            s_model=model,
            t_identity=f"char_{ch}",
            t_condition=tcond,
            t_framing="bare_label",
            t_model=model,
        )

    def r3_src_ceiling(tcond, model):
        u = units.get(f"{ASSIST}__inserted__chat__{model}")
        return None if u is None else u["r2"]["own_map"]

    fig_tier_grid(
        rows,
        "r3_chat_to_characters",
        r3_pair,
        chars,
        CHAR_COLOR,
        CHAR_LABEL,
        [("on_policy", "on-policy targets"), ("inserted", "inserted targets")],
        src_ceiling_fn=r3_src_ceiling,
        nulls=nulls,
    )

    print(f"[writeup-v2-figs] wrote figures under {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
